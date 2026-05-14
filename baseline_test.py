"""
=======================================================================
Stage 1 Baseline: DNABERT-2 Only
Tissue-Specific DNA Variant Pathogenicity Prediction

실행 모드:
  train : 데이터 분할 + 학습 + 모델/split 인덱스 저장
  val   : 저장된 모델 + split 인덱스로 Validation 평가
  test  : 저장된 모델 + split 인덱스로 Test 평가
  all   : train → val → test 순차 실행

Usage:
  python baseline_dnabert2.py --mode train --seed 42
  python baseline_dnabert2.py --mode val   --seed 42
  python baseline_dnabert2.py --mode test  --seed 42
  python baseline_dnabert2.py --mode all   --seeds 42 123 456
=======================================================================
"""

import os, sys, json, random, warnings, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer, AutoModel,
    get_linear_schedule_with_warmup,
)
from transformers.models.bert.configuration_bert import BertConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    precision_score, recall_score, f1_score,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
# 1. Config
# ══════════════════════════════════════════════════════════════════════
class Config:
    DATA_DIR    = "sequences_extracted"
    BRAIN_FILE  = "pathogenic_brain_with_seq.tsv"
    HEART_FILE  = "pathogenic_heart_with_seq.tsv"
    LIVER_FILE  = "pathogenic_liver_with_seq.tsv"
    BENIGN_FILE = "benign_with_seq.tsv"

    # tissue_id 매핑 — verify_tissue_ids() 로 확인 후 수정
    TISSUE_MAP = {0: "liver", 1: "heart", 2: "brain"}
    BENIGN_TID = 3

    BENIGN_RATIO   = 3
    BENIGN_CHUNK   = 100_000
    BENIGN_APPROX  = 1_281_188
    BENIGN_SEED    = 42       # benign 샘플링 고정 seed (재현성)
    VAL_RATIO      = 0.10
    TEST_RATIO     = 0.10

    MODEL_NAME  = "zhihan1996/DNABERT-2-117M"
    MAX_LENGTH  = 256
    HIDDEN_DIM  = 768
    DROPOUT     = 0.1

    BATCH_SIZE     = 8
    GRAD_ACCUM     = 2
    LR_BACKBONE    = 1e-5
    LR_HEAD        = 1e-4
    WEIGHT_DECAY   = 0.01
    MAX_GRAD_NORM  = 1.0
    NUM_EPOCHS     = 30
    PATIENCE       = 5
    WARMUP_RATIO   = 0.05
    FP16           = True

    SEEDS       = [42, 123, 456]
    OUTPUT_DIR  = "outputs/stage1_baseline"
    NUM_WORKERS = 4


# ══════════════════════════════════════════════════════════════════════
# 2. Utilities
# ══════════════════════════════════════════════════════════════════════
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device(fp16: bool):
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda"), fp16
    print("  GPU 없음 → CPU 모드 (FP16 비활성화)")
    return torch.device("cpu"), False


# ══════════════════════════════════════════════════════════════════════
# 3. Data Loading
# ══════════════════════════════════════════════════════════════════════
def verify_tissue_ids(cfg: Config):
    print("\n─── Tissue ID 검증 ─────────────────────────────────────")
    for fname in [cfg.BRAIN_FILE, cfg.HEART_FILE, cfg.LIVER_FILE]:
        fpath = Path(cfg.DATA_DIR) / fname
        if fpath.exists():
            df = pd.read_csv(fpath, sep="\t", nrows=5)
            print(f"  {fname}: tissue_id={df['tissue_id'].unique().tolist()}")
    fpath = Path(cfg.DATA_DIR) / cfg.BENIGN_FILE
    if fpath.exists():
        df = pd.read_csv(fpath, sep="\t", nrows=5)
        print(f"  {cfg.BENIGN_FILE}: tissue_id={df['tissue_id'].unique().tolist()}")
    print(f"\n  현재 Config: TISSUE_MAP={cfg.TISSUE_MAP}, BENIGN_TID={cfg.BENIGN_TID}")
    print("  불일치 시 Config를 수정하세요.\n")


def load_pathogenic(cfg: Config) -> pd.DataFrame:
    dfs = []
    for tname, fname in [("brain", cfg.BRAIN_FILE),
                          ("heart", cfg.HEART_FILE),
                          ("liver", cfg.LIVER_FILE)]:
        df = pd.read_csv(Path(cfg.DATA_DIR) / fname, sep="\t")
        print(f"  Pathogenic {tname:5s}: {len(df):>8,} rows")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_benign_sampled(cfg: Config, n_target: int) -> pd.DataFrame:
    fpath     = Path(cfg.DATA_DIR) / cfg.BENIGN_FILE
    rng       = np.random.RandomState(cfg.BENIGN_SEED)   # 고정 seed
    keep_prob = min(1.0, (n_target * 1.1) / cfg.BENIGN_APPROX)
    chunks    = []
    print(f"  Benign 로드 중 (target={n_target:,}, seed={cfg.BENIGN_SEED})...")
    for i, chunk in enumerate(
        pd.read_csv(fpath, sep="\t", chunksize=cfg.BENIGN_CHUNK)
    ):
        keep_n  = max(1, int(len(chunk) * keep_prob))
        sampled = chunk.sample(n=keep_n, random_state=rng.randint(0, 2**31))
        chunks.append(sampled)
        if (i + 1) % 5 == 0:
            print(f"    청크 {i+1} 완료...")
    benign = pd.concat(chunks, ignore_index=True)
    if len(benign) > n_target:
        benign = benign.sample(n=n_target, random_state=cfg.BENIGN_SEED)
    print(f"  Benign 완료: {len(benign):,} rows")
    return benign


def prepare_data(cfg: Config) -> pd.DataFrame:
    print("\n─── 데이터 로드 ────────────────────────────────────────")
    patho    = load_pathogenic(cfg)
    n_benign = len(patho) * cfg.BENIGN_RATIO
    benign   = load_benign_sampled(cfg, n_benign)
    df       = pd.concat([patho, benign], ignore_index=True)
    df       = df.dropna(subset=["sequence", "label"])
    df["label"]     = df["label"].astype(int)
    df["tissue_id"] = df["tissue_id"].astype(int)
    df["sequence"]  = df["sequence"].str.upper().str.strip()
    seq_len = df["sequence"].str.len()
    df = df[(seq_len >= 600) & (seq_len <= 1100)].reset_index(drop=True)
    # ── 재현성을 위한 원본 인덱스 열 추가 ──
    df["_orig_idx"] = np.arange(len(df))
    print(f"\n  총 샘플: {len(df):,}")
    print(f"  Label 분포:  {dict(df['label'].value_counts().sort_index())}")
    print(f"  Tissue 분포: {dict(df['tissue_id'].value_counts().sort_index())}")
    return df


def stratified_split(df: pd.DataFrame, cfg: Config, seed: int):
    strat = df["tissue_id"].astype(str) + "_" + df["label"].astype(str)
    if strat.value_counts().min() < 2:
        strat = df["label"].astype(str)

    train_val, test = train_test_split(
        df, test_size=cfg.TEST_RATIO, stratify=strat, random_state=seed)

    strat_tv = train_val["tissue_id"].astype(str) + "_" + train_val["label"].astype(str)
    if strat_tv.value_counts().min() < 2:
        strat_tv = train_val["label"].astype(str)

    val_adj = cfg.VAL_RATIO / (1.0 - cfg.TEST_RATIO)
    train, val = train_test_split(
        train_val, test_size=val_adj, stratify=strat_tv, random_state=seed)

    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════
# 4. Dataset & DataLoader
# ══════════════════════════════════════════════════════════════════════
class VariantDataset(Dataset):
    def __init__(self, df, tokenizer, max_length: int):
        self.sequences  = df["sequence"].tolist()
        self.labels     = df["label"].tolist()
        self.tissue_ids = df["tissue_id"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.sequences[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx],     dtype=torch.float),
            "tissue_id":      torch.tensor(self.tissue_ids[idx], dtype=torch.long),
        }


def make_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    labels  = df["label"].values
    counts  = np.bincount(labels)
    weights = torch.tensor(1.0 / counts[labels], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def _make_loader(ds, batch_size, shuffle=False, sampler=None, num_workers=4):
    use_pin = torch.cuda.is_available()
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      sampler=sampler, num_workers=num_workers,
                      pin_memory=use_pin)


# ══════════════════════════════════════════════════════════════════════
# 5. Model
# ══════════════════════════════════════════════════════════════════════
class DNABERT2Baseline(nn.Module):
    """
    Stage 1: DNABERT-2 → Masked Mean Pooling → MLP
    Stage 2 확장: pooled [B,768] + epi_feat → cat → MLP
    Stage 3 확장: Cross-Attention Fusion + Tissue Embedding
    """
    def __init__(self, model_name: str, hidden: int = 768, dropout: float = 0.1):
        super().__init__()
        config = BertConfig.from_pretrained(model_name) #
        self.encoder = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, config=config)#, low_cpu_mem_usage=False)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 256), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),    nn.GELU(),
            nn.Linear(64, 1),
        )

    def _masked_mean_pool(self, hidden_states, attention_mask):
        mask   = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        count  = mask.sum(dim=1).clamp(min=1e-9)
        return summed / count   # [B, 768]

    def forward(self, input_ids, attention_mask):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._masked_mean_pool(out[0], attention_mask)
        return self.classifier(pooled).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════
# 6. Evaluation
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_inference(model, loader, device, fp16):
    model.eval()
    all_labels, all_probs, all_tissues = [], [], []
    for batch in tqdm(loader, desc="  Inference", leave=False):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.cuda.amp.autocast(enabled=fp16):
            logits = model(ids, mask)
        probs = torch.sigmoid(logits).float().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(batch["label"].numpy().tolist())
        all_tissues.extend(batch["tissue_id"].numpy().tolist())
    return (np.array(all_labels,  dtype=np.float32),
            np.array(all_probs,   dtype=np.float32),
            np.array(all_tissues, dtype=np.int32))


def compute_metrics(labels, probs, tissues, cfg: Config) -> dict:
    metrics, auprc_list = {}, []
    benign_msk = (tissues == cfg.BENIGN_TID)
    for tid, tname in cfg.TISSUE_MAP.items():
        msk = (tissues == tid) | benign_msk
        if msk.sum() == 0 or labels[msk].sum() == 0:
            print(f"  Warning: {tname} 평가 데이터 없음")
            continue
        l, p  = labels[msk], probs[msk]
        preds = (p >= 0.5).astype(int)
        auprc = average_precision_score(l, p)
        metrics[tname] = {
            "auprc":        float(round(auprc, 4)),
            "precision":    float(round(precision_score(l, preds, zero_division=0), 4)),
            "recall":       float(round(recall_score(l, preds, zero_division=0), 4)),
            "f1":           float(round(f1_score(l, preds, zero_division=0), 4)),
            "n_pathogenic": int(l.sum()),
            "n_benign":     int((l == 0).sum()),
        }
        auprc_list.append(auprc)
    metrics["macro_auprc"] = float(round(np.mean(auprc_list), 4)) if auprc_list else 0.0
    return metrics


def print_metrics(metrics: dict, prefix: str = ""):
    print(f"{prefix}Macro-AUPRC: {metrics['macro_auprc']:.4f}")
    for k, v in metrics.items():
        if k == "macro_auprc": continue
        print(f"{prefix}  {k:6s}  AUPRC={v['auprc']:.4f} | "
              f"P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}  "
              f"(pos={v['n_pathogenic']}, neg={v['n_benign']})")


def _save_predictions(labels, probs, tissues, path: Path):
    pd.DataFrame({
        "label": labels.tolist(), "prob": probs.tolist(),
        "tissue_id": tissues.tolist(),
    }).to_csv(path, index=False)


# ══════════════════════════════════════════════════════════════════════
# 7-A. TRAIN MODE
# ══════════════════════════════════════════════════════════════════════
def run_train(cfg: Config, df: pd.DataFrame, seed: int):
    """
    학습 모드:
      1) Stratified split + split 인덱스 저장 (val/test 모드 재현용)
      2) DNABERT-2 fine-tuning (val AUPRC 기반 early stopping)
      3) Best 모델 + val metrics 저장
    Returns: (history DataFrame, best val metrics dict)
    """
    set_seed(seed)
    out_dir   = Path(cfg.OUTPUT_DIR) / f"seed_{seed}"
    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    device, use_fp16 = get_device(cfg.FP16)
    print(f"\n{'═'*62}")
    print(f"  [TRAIN] Seed={seed} | FP16={use_fp16}")
    print('═'*62)

    # ── 1. Split ─────────────────────────────────────────────
    train_df, val_df, test_df = stratified_split(df, cfg, seed)

    # _orig_idx 저장 → val / test 모드에서 동일 split 재현
    np.save(str(split_dir / "val_orig_idx.npy"),  val_df["_orig_idx"].values)
    np.save(str(split_dir / "test_orig_idx.npy"), test_df["_orig_idx"].values)
    split_meta = {
        "seed": seed, "benign_seed": cfg.BENIGN_SEED,
        "train_size": len(train_df), "val_size": len(val_df),
        "test_size": len(test_df),   "n_total": len(df),
    }
    with open(split_dir / "meta.json", "w") as f:
        json.dump(split_meta, f, indent=2)
    print(f"  Split 저장: {split_dir}/")
    print(f"  Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}")

    # ── 2. Tokenizer & DataLoaders ───────────────────────────
    print(f"\n  Tokenizer 로드: {cfg.MODEL_NAME}")
    tokenizer  = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True)
    train_ds   = VariantDataset(train_df, tokenizer, cfg.MAX_LENGTH)
    val_ds     = VariantDataset(val_df,   tokenizer, cfg.MAX_LENGTH)
    sampler    = make_weighted_sampler(train_df)
    train_loader = _make_loader(train_ds, cfg.BATCH_SIZE, sampler=sampler,
                                num_workers=cfg.NUM_WORKERS)
    val_loader   = _make_loader(val_ds, cfg.BATCH_SIZE * 2,
                                num_workers=cfg.NUM_WORKERS)

    # ── 3. Model ─────────────────────────────────────────────
    print(f"  모델 로드: {cfg.MODEL_NAME}")
    model  = DNABERT2Baseline(cfg.MODEL_NAME, cfg.HIDDEN_DIM, cfg.DROPOUT).to(device)
    n_p    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  학습 파라미터: {n_p/1e6:.1f}M")

    # ── 4. Optimizer (차등 학습률) ───────────────────────────
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(),    "lr": cfg.LR_BACKBONE},
        {"params": model.classifier.parameters(), "lr": cfg.LR_HEAD},
    ], weight_decay=cfg.WEIGHT_DECAY)
    total_steps  = (len(train_loader) // cfg.GRAD_ACCUM) * cfg.NUM_EPOCHS
    warmup_steps = int(total_steps * cfg.WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── 5. Loss ──────────────────────────────────────────────
    n_pos = int((train_df["label"] == 1).sum())
    n_neg = int((train_df["label"] == 0).sum())
    pos_w = torch.tensor(n_neg / n_pos, dtype=torch.float).to(device)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    print(f"  pos_weight: {pos_w.item():.2f}  (neg={n_neg:,} / pos={n_pos:,})")

    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # ── 6. Training Loop ─────────────────────────────────────
    best_macro, patience_cnt, history = 0.0, 0, []

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"  Epoch {epoch:02d}/{cfg.NUM_EPOCHS}")
        for step, batch in pbar:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl  = batch["label"].to(device)
            with torch.cuda.amp.autocast(enabled=use_fp16):
                loss = crit(model(ids, mask), lbl) / cfg.GRAD_ACCUM
            scaler.scale(loss).backward()
            epoch_loss += loss.item() * cfg.GRAD_ACCUM
            if (step + 1) % cfg.GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                scaler.step(optimizer); scaler.update()
                scheduler.step(); optimizer.zero_grad()
            pbar.set_postfix({"loss": f"{epoch_loss/(step+1):.4f}"})

        val_lbl, val_prob, val_tid = run_inference(model, val_loader, device, use_fp16)
        val_m  = compute_metrics(val_lbl, val_prob, val_tid, cfg)
        macro  = val_m["macro_auprc"]
        avg_l  = epoch_loss / len(train_loader)
        print(f"\n  Epoch {epoch:02d} | Loss={avg_l:.4f} | Val Macro-AUPRC={macro:.4f}")
        print_metrics(val_m, prefix="  ")

        row = {"epoch": epoch, "loss": avg_l, "val_macro_auprc": macro}
        for t, m in val_m.items():
            if t != "macro_auprc": row[f"val_{t}_auprc"] = m["auprc"]
        history.append(row)

        if macro > best_macro:
            best_macro, patience_cnt = macro, 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_metrics": val_m,
                        "config": {"model_name": cfg.MODEL_NAME,
                                   "max_length": cfg.MAX_LENGTH}},
                       out_dir / "best_model.pt")
            print(f"  ✅ Best 모델 저장 (Macro-AUPRC={best_macro:.4f})")
        else:
            patience_cnt += 1
            print(f"  patience: {patience_cnt}/{cfg.PATIENCE}")
            if patience_cnt >= cfg.PATIENCE:
                print(f"\n  Early stopping @ epoch {epoch}")
                break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "training_history.csv", index=False)

    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    with open(out_dir / "val_metrics.json", "w") as f:
        json.dump(ckpt["val_metrics"], f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ 학습 완료: {out_dir}/")
    print(f"     → val 평가:  python baseline_dnabert2.py --mode val  --single_seed {seed}")
    print(f"     → test 평가: python baseline_dnabert2.py --mode test --single_seed {seed}")
    return hist_df, ckpt["val_metrics"]


# ══════════════════════════════════════════════════════════════════════
# 7-B. VAL / TEST MODE
# ══════════════════════════════════════════════════════════════════════
def run_eval(cfg: Config, df: pd.DataFrame, seed: int, split: str = "test"):
    """
    평가 모드 (val 또는 test):
      1) 저장된 split 인덱스로 해당 subset 복원
      2) Best 모델 로드
      3) 추론 → 메트릭 계산 → 저장
    split: 'val' | 'test'
    Returns: metrics dict
    """
    out_dir   = Path(cfg.OUTPUT_DIR) / f"seed_{seed}"
    split_dir = out_dir / "splits"
    device, use_fp16 = get_device(cfg.FP16)

    print(f"\n{'═'*62}")
    print(f"  [{split.upper()}] Seed={seed} | FP16={use_fp16}")
    print('═'*62)

    # ── 1. Split 인덱스 로드 ────────────────────────────────
    idx_path = split_dir / f"{split}_orig_idx.npy"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"\n  ❌ {idx_path} 없음.\n"
            f"     먼저 train 모드를 실행하세요:\n"
            f"     python baseline_dnabert2.py --mode train --single_seed {seed}")

    orig_idx = np.load(str(idx_path))
    eval_df  = df.iloc[orig_idx].reset_index(drop=True)
    print(f"  {split} 샘플: {len(eval_df):,}")
    print(f"  Label 분포:  {dict(eval_df['label'].value_counts().sort_index())}")
    print(f"  Tissue 분포: {dict(eval_df['tissue_id'].value_counts().sort_index())}")

    # ── 2. 모델 로드 ─────────────────────────────────────────
    ckpt_path = out_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"\n  ❌ {ckpt_path} 없음.\n"
            f"     먼저 train 모드를 실행하세요.")

    ckpt  = torch.load(str(ckpt_path), map_location=device)
    model = DNABERT2Baseline(cfg.MODEL_NAME, cfg.HIDDEN_DIM, cfg.DROPOUT)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    print(f"  모델 로드 완료 (Best epoch={ckpt['epoch']})")

    # ── 3. Tokenizer & DataLoader ────────────────────────────
    tokenizer   = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True)
    eval_ds     = VariantDataset(eval_df, tokenizer, cfg.MAX_LENGTH)
    eval_loader = _make_loader(eval_ds, cfg.BATCH_SIZE * 2,
                               num_workers=cfg.NUM_WORKERS)

    # ── 4. 추론 & 메트릭 ─────────────────────────────────────
    labels, probs, tissues = run_inference(model, eval_loader, device, use_fp16)
    metrics = compute_metrics(labels, probs, tissues, cfg)

    print(f"\n  {split.upper()} 결과:")
    print_metrics(metrics, prefix="  ")

    # ── 5. 저장 ──────────────────────────────────────────────
    with open(out_dir / f"{split}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    _save_predictions(labels, probs, tissues,
                      out_dir / f"{split}_predictions.csv")
    print(f"\n  결과 저장: {out_dir}/{split}_metrics.json")
    return metrics


# ══════════════════════════════════════════════════════════════════════
# 8. Main
# ══════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="DNABERT-2 Stage 1 Baseline")
    p.add_argument("--mode", choices=["train", "val", "test", "all"],
                   default="all",
                   help="실행 모드 (train/val/test/all)")
    p.add_argument("--data_dir",     default=Config.DATA_DIR)
    p.add_argument("--output_dir",   default=Config.OUTPUT_DIR)
    p.add_argument("--seeds",        nargs="+", type=int, default=Config.SEEDS)
    p.add_argument("--single_seed",  type=int,  default=None,
                   help="단일 seed 지정 (예: --single_seed 42)")
    p.add_argument("--batch_size",   type=int,  default=Config.BATCH_SIZE)
    p.add_argument("--num_epochs",   type=int,  default=Config.NUM_EPOCHS)
    p.add_argument("--benign_ratio", type=int,  default=Config.BENIGN_RATIO)
    p.add_argument("--no_fp16",      action="store_true")
    p.add_argument("--verify_only",  action="store_true",
                   help="tissue_id 확인만 하고 종료")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config()
    cfg.DATA_DIR     = args.data_dir
    cfg.OUTPUT_DIR   = args.output_dir
    cfg.SEEDS        = [args.single_seed] if args.single_seed else args.seeds
    cfg.BATCH_SIZE   = args.batch_size
    cfg.NUM_EPOCHS   = args.num_epochs
    cfg.BENIGN_RATIO = args.benign_ratio
    cfg.FP16         = not args.no_fp16
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    verify_tissue_ids(cfg)
    if args.verify_only:
        return

    # 데이터는 모든 모드에서 공통 로드 (_orig_idx 포함)
    df = prepare_data(cfg)

    all_results = {}
    for seed in cfg.SEEDS:
        results = {}

        if args.mode in ("train", "all"):
            hist, val_m = run_train(cfg, df, seed)
            results["val"] = val_m

        if args.mode in ("val",):
            results["val"] = run_eval(cfg, df, seed, split="val")

        if args.mode in ("test", "all"):
            results["test"] = run_eval(cfg, df, seed, split="test")

        all_results[seed] = results

    # ── 최종 요약 (test 또는 all 모드일 때) ─────────────────
    if args.mode in ("test", "all") and all_results:
        print("\n" + "═"*62)
        print(" 최종 결과  (mean ± std across seeds)")
        print("═"*62)
        summary = {}
        for tname in cfg.TISSUE_MAP.values():
            vals = [r["test"][tname]["auprc"]
                    for r in all_results.values()
                    if "test" in r and tname in r["test"]]
            if vals:
                summary[tname] = {"mean": round(float(np.mean(vals)), 4),
                                  "std":  round(float(np.std(vals)),  4)}
                print(f"  {tname:6s}  AUPRC = {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        macro_vals = [r["test"]["macro_auprc"]
                      for r in all_results.values() if "test" in r]
        if macro_vals:
            summary["macro"] = {"mean": round(float(np.mean(macro_vals)), 4),
                                "std":  round(float(np.std(macro_vals)),  4)}
            print(f"  macro   AUPRC = {np.mean(macro_vals):.4f} ± {np.std(macro_vals):.4f}")

        final_path = Path(cfg.OUTPUT_DIR) / "final_summary.json"
        with open(final_path, "w") as f:
            json.dump({"mode": args.mode,
                       "per_seed": {str(k): v for k, v in all_results.items()},
                       "summary": summary}, f, indent=2, ensure_ascii=False)
        print(f"\n  전체 결과 저장: {final_path}")


if __name__ == "__main__":
    main()
