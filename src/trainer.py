import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import WeightedRandomSampler
from src.dataset import VariantDataset, make_loader
from src.utils import compute_metrics, print_metrics, save_predictions


@torch.no_grad()
def run_inference(model, loader, device, fp16):
    """추론 및 평가를 위한 루프"""
    model.eval()
    all_labels, all_probs, all_tissues = [], [], []

    for batch in tqdm(loader, desc="  Inference", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        with torch.cuda.amp.autocast(enabled=fp16):
            logits = model(ids, mask)

        probs = torch.sigmoid(logits).float().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(batch["label"].numpy().tolist())
        all_tissues.extend(batch["tissue_id"].numpy().tolist())

    return np.array(all_labels), np.array(all_probs), np.array(all_tissues)


def run_train(cfg, train_df, val_df, test_df, model, device, seed, resume=False):
    """학습 루프 (체크포인트 저장 및 재개 기능 포함)"""
    out_dir = Path(cfg.OUTPUT_DIR) / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- splits 폴더 및 인덱스 저장 ---
    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    # 훈련 시에 사용한 검증/테스트 데이터의 원본 인덱스를 저장
    if not (split_dir / "val_orig_idx.npy").exists():
        np.save(str(split_dir / "val_orig_idx.npy"), val_df["_orig_idx"].values)
        np.save(str(split_dir / "test_orig_idx.npy"), test_df["_orig_idx"].values)
        print(f"  💾 {seed}번 시드 데이터 인덱스(splits) 저장 완료")

    # 1. 데이터 로더 준비
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True)
    train_ds = VariantDataset(train_df, tokenizer, cfg.MAX_LENGTH)
    val_ds = VariantDataset(val_df, tokenizer, cfg.MAX_LENGTH)

    # 클래스 불균형 해소를 위한 Sampler
    labels = train_df["label"].values
    counts = np.bincount(labels)
    weights = torch.tensor(1.0 / counts[labels], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = make_loader(train_ds, cfg.BATCH_SIZE, sampler=sampler, num_workers=cfg.NUM_WORKERS)
    val_loader = make_loader(val_ds, cfg.BATCH_SIZE * 2, num_workers=cfg.NUM_WORKERS)

    # 2. 옵티마이저 및 스케줄러 설정
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": cfg.LR_BACKBONE},
        {"params": model.classifier.parameters(), "lr": cfg.LR_HEAD},
    ], weight_decay=cfg.WEIGHT_DECAY)

    total_steps = (len(train_loader) // cfg.GRAD_ACCUM) * cfg.NUM_EPOCHS
    warmup_steps = int(total_steps * cfg.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 3. 손실 함수 및 Mixed Precision 설정
    pos_w = torch.tensor((train_df["label"] == 0).sum() / (train_df["label"] == 1).sum()).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.FP16)

    # 4. 학습 상태 변수 초기화
    start_epoch = 1
    best_macro = 0.0
    patience_cnt = 0
    history = []

    latest_ckpt_path = out_dir / "latest_checkpoint.pt"

    # --- [RESUME] 중단된 지점부터 재개 로직 ---
    # python main.py --mode train --resume
    if resume and latest_ckpt_path.exists():
        print(f"\n  🔄 중단된 학습을 발견했습니다. 불러오는 중: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)

        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        scaler.load_state_dict(checkpoint['scaler_state'])

        start_epoch = checkpoint['epoch'] + 1
        best_macro = checkpoint['best_macro']
        patience_cnt = checkpoint['patience_cnt']
        history = checkpoint.get('history', [])

        print(f"  ✅ {checkpoint['epoch']} 에폭부터 학습을 재개합니다. (최고 점수: {best_macro:.4f})")
    # ----------------------------------------

    # 5. 메인 학습 루프
    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"  Epoch {epoch:02d}/{cfg.NUM_EPOCHS}")
        for step, batch in pbar:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl = batch["label"].to(device)

            with torch.cuda.amp.autocast(enabled=cfg.FP16):
                loss = crit(model(ids, mask), lbl) / cfg.GRAD_ACCUM

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * cfg.GRAD_ACCUM

            if (step + 1) % cfg.GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{epoch_loss / (step + 1):.4f}"})

        # 검증(Validation) 단계
        val_lbl, val_prob, val_tid = run_inference(model, val_loader, device, cfg.FP16)
        val_m = compute_metrics(val_lbl, val_prob, val_tid, cfg)
        macro = val_m["macro_auprc"]
        avg_l = epoch_loss / len(train_loader)

        print(f"\n  Epoch {epoch:02d} | Loss={avg_l:.4f} | Val Macro-AUPRC={macro:.4f}")
        print_metrics(val_m, prefix="  ")

        # 결과 기록
        history.append({"epoch": epoch, "loss": avg_l, "val_macro_auprc": macro})

        # --- [CHECKPOINT] 매 에폭 종료 시 상태 저장 ---
        checkpoint_data = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
            'best_macro': best_macro,
            'patience_cnt': patience_cnt,
            'history': history
        }
        torch.save(checkpoint_data, latest_ckpt_path)
        # -------------------------------------------

        # Best 모델 업데이트 및 Early Stopping
        if macro > best_macro:
            best_macro = macro
            patience_cnt = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_metrics": val_m
            }, out_dir / "best_model.pt")
            print(f"  ⭐ Best 모델 저장됨 (AUPRC: {best_macro:.4f})")
        else:
            patience_cnt += 1
            print(f"  patience: {patience_cnt}/{cfg.PATIENCE}")
            if patience_cnt >= cfg.PATIENCE:
                print(f"\n  Early stopping @ epoch {epoch}")
                break

    # 최종 결과 리포트 저장
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)


def run_eval(cfg, eval_df, model, device, seed, split="val"):
    """저장된 베스트 모델을 로드하여 평가만 수행하는 함수"""
    out_dir = Path(cfg.OUTPUT_DIR) / f"seed_{seed}"
    ckpt_path = out_dir / "best_model.pt"

    if not ckpt_path.exists():
        print(f"  ❌ 에러: 저장된 모델이 없습니다. ({ckpt_path})")
        print("     먼저 train 모드로 학습을 완료해 주세요.")
        return None

    # 모델 가중치 로드
    print(f"  🔄 베스트 모델 로드 중: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)

    # 토크나이저 및 로더 준비
    from src.dataset import VariantDataset, make_loader
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True)
    eval_ds = VariantDataset(eval_df, tokenizer, cfg.MAX_LENGTH)
    eval_loader = make_loader(eval_ds, cfg.BATCH_SIZE * 2, num_workers=cfg.NUM_WORKERS)

    # 추론 및 메트릭 계산
    labels, probs, tissues = run_inference(model, eval_loader, device, cfg.FP16)
    metrics = compute_metrics(labels, probs, tissues, cfg)

    print(f"\n[{split.upper()} 결과 - Seed {seed}]")
    print_metrics(metrics, prefix="  ")

    # 예측 결과 저장
    save_predictions(labels, probs, tissues, out_dir / f"{split}_predictions.csv")
    return metrics