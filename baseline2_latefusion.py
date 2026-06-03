"""
=======================================================================
Stage 2 Baseline: DNABERT-2 + Late Fusion (Epigenomic Concatenation)

baseline3_crossattn.py 와 다른 점: 모델 구조만 다름
  Cross-Attention → Late Fusion (CLS + GlobalAvgPool + TissueEmb concat)

baseline3 와 동일하게 맞춘 것:
  - stratified_split (tissue+label 층화 분할)
  - splits/ 저장 (val_orig_idx.npy, test_orig_idx.npy, meta.json)
  - _orig_idx 기반 epi 배열 인덱싱
  - pos_weight = n_neg / n_pos (동적 계산)
  - compute_metrics: benign을 label==0 기준으로 구분
  - 출력 파일 구조 동일
    best_model.pt / val_metrics.json / test_metrics.json /
    test_predictions.csv / training_history.csv / splits/
=======================================================================
"""
import os, json, random, warnings, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from transformers.models.bert.configuration_bert import BertConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ── Config ────────────────────────────────────────────────────────────
class Config:
    EPI_DIR = "epigenomic_signals"
    PATHO_FILES  = [("signals_liver_zscore.npz", 0),
                    ("signals_heart_zscore.npz",  1),
                    ("signals_brain_zscore.npz",  2)]
    BENIGN_FILES = [("signals_benign_liver_zscore.npz", 0),
                    ("signals_benign_heart_zscore.npz", 1),
                    ("signals_benign_brain_zscore.npz", 2)]

    EPI_N_CHANNELS = 2
    EPI_USE_LEN    = 1024
    TISSUE_MAP     = {0: "liver", 1: "heart", 2: "brain"}
    TISSUE_STR_MAP = {"liver": 0, "heart": 1, "brain": 2}
    N_TISSUES      = 3
    BENIGN_RATIO   = 3
    BENIGN_SEED    = 42
    VAL_RATIO      = 0.10
    TEST_RATIO     = 0.10

    MODEL_NAME     = "zhihan1996/DNABERT-2-117M"
    MAX_LENGTH     = 256
    HIDDEN_DIM     = 768
    EPI_HIDDEN     = 128
    CNN_KERNEL     = 7
    TISSUE_DIM     = 768       # baseline3 TISSUE_EMB_DIM 과 동일
    FUSION_HIDDEN  = 512

    DROPOUT        = 0.1
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
    SEEDS          = [42, 123, 456]
    OUTPUT_DIR     = "outputs/basic_baseline2"
    NUM_WORKERS    = 4


# ── Utilities ─────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(fp16):
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda"), fp16
    print("  GPU 없음 → CPU (FP16 비활성화)")
    return torch.device("cpu"), False


# ── 데이터 로드 ───────────────────────────────────────────────────────
def _load_npz(fpath, cfg):
    data = np.load(str(fpath), allow_pickle=True)
    sequences = data["sequence"].tolist()
    labels    = data["label"].astype(np.int64)
    tid_raw   = data["tissue_id"]
    if tid_raw.dtype.kind in ("U", "S", "O"):
        tissue_ids = np.array([cfg.TISSUE_STR_MAP[str(t)] for t in tid_raw], dtype=np.int64)
    else:
        tissue_ids = tid_raw.astype(np.int64)
    h3k27ac = data["h3k27ac"][:, :cfg.EPI_USE_LEN].astype(np.float32)
    dnase   = data["dnase"][:,   :cfg.EPI_USE_LEN].astype(np.float32)
    return dict(sequences=sequences, labels=labels, tissue_ids=tissue_ids,
                h3k27ac=h3k27ac, dnase=dnase)

def load_all_data(cfg):
    epi_dir = Path(cfg.EPI_DIR)
    all_seqs, all_labels, all_tids, all_h3k, all_dnase = [], [], [], [], []

    print("\n─── Pathogenic npz 로드 ───────────────────────────────")
    n_patho = {}
    for fname, tid in cfg.PATHO_FILES:
        d = _load_npz(epi_dir / fname, cfg)
        n = len(d["sequences"]); n_patho[tid] = n
        print(f"  {fname}: {n:,} rows (tid={tid}, label=1)")
        all_seqs.extend(d["sequences"]); all_labels.append(d["labels"])
        all_tids.append(d["tissue_ids"]); all_h3k.append(d["h3k27ac"])
        all_dnase.append(d["dnase"])

    print(f"\n─── Benign npz 로드 (ratio={cfg.BENIGN_RATIO}) ──")
    rng = np.random.RandomState(cfg.BENIGN_SEED)
    for fname, tid in cfg.BENIGN_FILES:
        d = _load_npz(epi_dir / fname, cfg)
        n_total  = len(d["sequences"])
        n_sample = min(n_patho[tid] * cfg.BENIGN_RATIO, n_total)
        chosen   = rng.choice(n_total, size=n_sample, replace=False)
        print(f"  {fname}: {n_total:,} → {n_sample:,} (tid={tid}, label=0)")
        all_seqs.extend([d["sequences"][i] for i in chosen])
        all_labels.append(np.zeros(n_sample, dtype=np.int64))
        all_tids.append(np.full(n_sample, tid, dtype=np.int64))
        all_h3k.append(d["h3k27ac"][chosen]); all_dnase.append(d["dnase"][chosen])

    labels_arr = np.concatenate(all_labels); tids_arr = np.concatenate(all_tids)
    h3k_arr    = np.concatenate(all_h3k);   dnase_arr = np.concatenate(all_dnase)

    df = pd.DataFrame({"sequence": all_seqs, "label": labels_arr, "tissue_id": tids_arr})
    keep = (df["sequence"].str.len() >= 600) & (df["sequence"].str.len() <= 1100)
    keep_idx  = np.where(keep)[0]
    df        = df[keep].reset_index(drop=True)
    h3k_arr   = h3k_arr[keep_idx]; dnase_arr = dnase_arr[keep_idx]
    df["_orig_idx"] = np.arange(len(df))

    print(f"\n  총 샘플: {len(df):,} | label: {dict(df['label'].value_counts().sort_index())}")
    for tid, tname in cfg.TISSUE_MAP.items():
        sub = df[df["tissue_id"]==tid]
        print(f"  {tname:5s}: pos={int((sub['label']==1).sum()):,} neg={int((sub['label']==0).sum()):,}")
    return df, h3k_arr, dnase_arr


# ── Data Split ────────────────────────────────────────────────────────
def stratified_split(df, cfg, seed):
    strat = df["tissue_id"].astype(str) + "_" + df["label"].astype(str)
    if strat.value_counts().min() < 2:
        strat = df["label"].astype(str)
    train_val, test = train_test_split(df, test_size=cfg.TEST_RATIO,
                                       stratify=strat, random_state=seed)
    strat_tv = (train_val["tissue_id"].astype(str) + "_" + train_val["label"].astype(str))
    if strat_tv.value_counts().min() < 2:
        strat_tv = train_val["label"].astype(str)
    train, val = train_test_split(train_val, test_size=cfg.VAL_RATIO/(1-cfg.TEST_RATIO),
                                  stratify=strat_tv, random_state=seed)
    return (train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True))


# ── Dataset ───────────────────────────────────────────────────────────
class VariantEpiDataset(Dataset):
    def __init__(self, df, tokenizer, h3k_arr, dnase_arr, cfg):
        self.sequences  = df["sequence"].tolist()
        self.labels     = df["label"].tolist()
        self.tissue_ids = df["tissue_id"].tolist()
        self.orig_idxs  = df["_orig_idx"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = cfg.MAX_LENGTH
        self.h3k_arr    = h3k_arr
        self.dnase_arr  = dnase_arr

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.sequences[idx], max_length=self.max_length,
                             padding="max_length", truncation=True, return_tensors="pt")
        oi  = self.orig_idxs[idx]
        epi = torch.from_numpy(np.stack([self.h3k_arr[oi], self.dnase_arr[oi]], axis=0)).float()
        return {"input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "epi_signal":     epi,
                "label":          torch.tensor(self.labels[idx],     dtype=torch.float),
                "tissue_id":      torch.tensor(self.tissue_ids[idx], dtype=torch.long)}

def make_weighted_sampler(df):
    labels  = df["label"].values; counts = np.bincount(labels)
    weights = torch.tensor(1.0 / counts[labels], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def _make_loader(ds, batch_size, shuffle=False, sampler=None, num_workers=4):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


# ── Model — Late Fusion ───────────────────────────────────────────────
class EpigenomicEncoder(nn.Module):
    def __init__(self, in_channels=2, hidden=128, kernel=7):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel, padding=kernel//2),
            nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel, padding=kernel//2),
            nn.BatchNorm1d(hidden), nn.GELU(),
        )
    def forward(self, x):
        return self.cnn(x).mean(dim=2)   # GlobalAvgPool → [B, epi_hidden]

class LateFusionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        bert_cfg     = BertConfig.from_pretrained(cfg.MODEL_NAME)
        self.dna_enc = AutoModel.from_pretrained(cfg.MODEL_NAME,
                           trust_remote_code=True, config=bert_cfg)
        self.epi_enc     = EpigenomicEncoder(cfg.EPI_N_CHANNELS, cfg.EPI_HIDDEN, cfg.CNN_KERNEL)
        self.tissue_emb  = nn.Embedding(cfg.N_TISSUES, cfg.TISSUE_DIM)
        self.tissue_norm = nn.LayerNorm(cfg.TISSUE_DIM)

        fusion_in = cfg.HIDDEN_DIM + cfg.EPI_HIDDEN + cfg.TISSUE_DIM
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_in, cfg.FUSION_HIDDEN),
            nn.LayerNorm(cfg.FUSION_HIDDEN), nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.FUSION_HIDDEN, 256), nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, input_ids, attention_mask, epi_signal, tissue_id):
        dna_out = self.dna_enc(input_ids=input_ids, attention_mask=attention_mask)
        hidden  = dna_out[0] if isinstance(dna_out, tuple) else dna_out.last_hidden_state
        cls_vec = hidden[:, 0, :]                                          # [B, 768]
        epi_vec = self.epi_enc(epi_signal)                                 # [B, epi_hidden]
        tis_vec = self.tissue_norm(self.tissue_emb(tissue_id))             # [B, tissue_dim]
        fused   = self.fusion_proj(torch.cat([cls_vec, epi_vec, tis_vec], dim=1))
        return self.classifier(fused).squeeze(-1)


# ── Metrics ───────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, device, fp16):
    model.eval()
    all_labels, all_probs, all_tissues = [], [], []
    for batch in tqdm(loader, desc="  Inference", leave=False):
        ids  = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
        epi  = batch["epi_signal"].to(device); tids = batch["tissue_id"].to(device)
        with torch.cuda.amp.autocast(enabled=fp16):
            logits = model(ids, mask, epi, tids)
        probs = torch.sigmoid(logits).float().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(batch["label"].numpy().tolist())
        all_tissues.extend(batch["tissue_id"].numpy().tolist())
    return (np.array(all_labels, dtype=np.float32),
            np.array(all_probs,  dtype=np.float32),
            np.array(all_tissues,dtype=np.int32))

def compute_metrics(labels, probs, tissues, cfg):
    metrics, auprc_list = {}, []
    benign_msk = (labels == 0)   # label 기준으로 benign 구분 (baseline3 동일)
    for tid, tname in cfg.TISSUE_MAP.items():
        msk = ((tissues == tid) & (labels == 1)) | benign_msk
        if msk.sum() == 0 or labels[msk].sum() == 0:
            print(f"  Warning: {tname} 평가 데이터 없음"); continue
        l, p  = labels[msk], probs[msk]; preds = (p >= 0.5).astype(int)
        auprc = average_precision_score(l, p)
        metrics[tname] = {
            "auprc":        float(round(auprc, 4)),
            "precision":    float(round(precision_score(l, preds, zero_division=0), 4)),
            "recall":       float(round(recall_score(l, preds, zero_division=0), 4)),
            "f1":           float(round(f1_score(l, preds, zero_division=0), 4)),
            "n_pathogenic": int(l.sum()), "n_benign": int((l==0).sum()),
        }
        auprc_list.append(auprc)
    metrics["macro_auprc"] = float(round(np.mean(auprc_list), 4)) if auprc_list else 0.0
    return metrics

def print_metrics(metrics, prefix=""):
    print(f"{prefix}Macro-AUPRC: {metrics['macro_auprc']:.4f}")
    for k, v in metrics.items():
        if k == "macro_auprc": continue
        print(f"{prefix}  {k:6s} AUPRC={v['auprc']:.4f} | "
              f"P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} "
              f"(pos={v['n_pathogenic']}, neg={v['n_benign']})")

def _save_predictions(labels, probs, tissues, path):
    pd.DataFrame({"label": labels.tolist(), "prob": probs.tolist(),
                  "tissue_id": tissues.tolist()}).to_csv(path, index=False)


# ── Train ─────────────────────────────────────────────────────────────
def run_train(cfg, df, h3k_arr, dnase_arr, seed):
    set_seed(seed)
    out_dir   = Path(cfg.OUTPUT_DIR) / f"seed_{seed}"
    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    device, use_fp16 = get_device(cfg.FP16)

    print(f"\n{'='*62}")
    print(f"  [TRAIN] Stage2 LateFusion | Seed={seed} | FP16={use_fp16}")
    print('='*62)

    train_df, val_df, test_df = stratified_split(df, cfg, seed)
    np.save(str(split_dir / "val_orig_idx.npy"),  val_df["_orig_idx"].values)
    np.save(str(split_dir / "test_orig_idx.npy"), test_df["_orig_idx"].values)
    with open(split_dir / "meta.json", "w") as f:
        json.dump({"seed": seed, "benign_seed": cfg.BENIGN_SEED,
                   "train": len(train_df), "val": len(val_df),
                   "test": len(test_df), "total": len(df)}, f, indent=2)

    n_pos = int((train_df["label"]==1).sum()); n_neg = int((train_df["label"]==0).sum())
    print(f"  Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}")
    print(f"  Train pos={n_pos:,} neg={n_neg:,}")

    tokenizer    = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True)
    train_ds     = VariantEpiDataset(train_df, tokenizer, h3k_arr, dnase_arr, cfg)
    val_ds       = VariantEpiDataset(val_df,   tokenizer, h3k_arr, dnase_arr, cfg)
    train_loader = _make_loader(train_ds, cfg.BATCH_SIZE,
                                sampler=make_weighted_sampler(train_df),
                                num_workers=cfg.NUM_WORKERS)
    val_loader   = _make_loader(val_ds, cfg.BATCH_SIZE*2, num_workers=cfg.NUM_WORKERS)

    model = LateFusionModel(cfg).to(device)
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  학습 파라미터: {n_p/1e6:.1f}M")

    optimizer = torch.optim.AdamW([
        {"params": model.dna_enc.parameters(),    "lr": cfg.LR_BACKBONE},
        {"params": model.epi_enc.parameters(),    "lr": cfg.LR_HEAD},
        {"params": model.tissue_emb.parameters(), "lr": cfg.LR_HEAD},
        {"params": model.tissue_norm.parameters(),"lr": cfg.LR_HEAD},
        {"params": model.fusion_proj.parameters(),"lr": cfg.LR_HEAD},
        {"params": model.classifier.parameters(), "lr": cfg.LR_HEAD},
    ], weight_decay=cfg.WEIGHT_DECAY)

    total_steps  = (len(train_loader) // cfg.GRAD_ACCUM) * cfg.NUM_EPOCHS
    warmup_steps = int(total_steps * cfg.WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    pos_w        = torch.tensor(n_neg / n_pos, dtype=torch.float).to(device)
    crit         = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    scaler       = torch.cuda.amp.GradScaler(enabled=use_fp16)
    print(f"  pos_weight={pos_w.item():.2f}")

    best_macro, patience_cnt, history = 0.0, 0, []
    for epoch in range(1, cfg.NUM_EPOCHS+1):
        model.train(); epoch_loss = 0.0; optimizer.zero_grad()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"  Epoch {epoch:02d}/{cfg.NUM_EPOCHS}")
        for step, batch in pbar:
            ids  = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
            epi  = batch["epi_signal"].to(device); tids = batch["tissue_id"].to(device)
            lbl  = batch["label"].to(device)
            with torch.cuda.amp.autocast(enabled=use_fp16):
                loss = crit(model(ids, mask, epi, tids), lbl) / cfg.GRAD_ACCUM
            scaler.scale(loss).backward()
            epoch_loss += loss.item() * cfg.GRAD_ACCUM
            if (step+1) % cfg.GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                scaler.step(optimizer); scaler.update()
                scheduler.step(); optimizer.zero_grad()
            pbar.set_postfix({"loss": f"{epoch_loss/(step+1):.4f}"})

        val_lbl, val_prob, val_tid = run_inference(model, val_loader, device, use_fp16)
        val_m = compute_metrics(val_lbl, val_prob, val_tid, cfg)
        macro = val_m["macro_auprc"]; avg_l = epoch_loss / len(train_loader)
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
                        "config": {"model_name": cfg.MODEL_NAME, "max_length": cfg.MAX_LENGTH,
                                   "epi_hidden": cfg.EPI_HIDDEN, "fusion_hidden": cfg.FUSION_HIDDEN,
                                   "fusion_type": "late_fusion"}},
                       out_dir / "best_model.pt")
            print(f"  ✅ Best 모델 저장 (Macro-AUPRC={best_macro:.4f})")
        else:
            patience_cnt += 1
            print(f"  patience: {patience_cnt}/{cfg.PATIENCE}")
            if patience_cnt >= cfg.PATIENCE:
                print(f"\n  Early stopping @ epoch {epoch}"); break

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    with open(out_dir / "val_metrics.json", "w") as f:
        json.dump(ckpt["val_metrics"], f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ 학습 완료: {out_dir}/")
    return pd.DataFrame(history), ckpt["val_metrics"]


# ── Eval ──────────────────────────────────────────────────────────────
def run_eval(cfg, df, h3k_arr, dnase_arr, seed, split="test"):
    out_dir   = Path(cfg.OUTPUT_DIR) / f"seed_{seed}"
    split_dir = out_dir / "splits"
    device, use_fp16 = get_device(cfg.FP16)
    print(f"\n{'='*62}\n  [{split.upper()}] Stage2 LateFusion | Seed={seed}\n{'='*62}")

    orig_idx = np.load(str(split_dir / f"{split}_orig_idx.npy"))
    eval_df  = df.iloc[orig_idx].reset_index(drop=True)
    print(f"  {split} 샘플: {len(eval_df):,}  "
          f"(pos={int((eval_df['label']==1).sum()):,}, neg={int((eval_df['label']==0).sum()):,})")

    ckpt  = torch.load(str(out_dir / "best_model.pt"), map_location=device)
    model = LateFusionModel(cfg)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    print(f"  모델 로드 완료 (Best epoch={ckpt['epoch']})")

    tokenizer   = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True)
    eval_ds     = VariantEpiDataset(eval_df, tokenizer, h3k_arr, dnase_arr, cfg)
    eval_loader = _make_loader(eval_ds, cfg.BATCH_SIZE*2, num_workers=cfg.NUM_WORKERS)

    labels, probs, tissues = run_inference(model, eval_loader, device, use_fp16)
    metrics = compute_metrics(labels, probs, tissues, cfg)
    print(f"\n  {split.upper()} 결과:"); print_metrics(metrics, prefix="  ")

    with open(out_dir / f"{split}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    _save_predictions(labels, probs, tissues, out_dir / f"{split}_predictions.csv")
    print(f"  결과 저장: {out_dir}/{split}_metrics.json")
    return metrics


# ── Main ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Stage 2: DNABERT-2 + Late Fusion")
    p.add_argument("--mode",          choices=["train","val","test","all"], default="all")
    p.add_argument("--epi_dir",       default=Config.EPI_DIR)
    p.add_argument("--output_dir",    default=Config.OUTPUT_DIR)
    p.add_argument("--seeds",         nargs="+", type=int, default=Config.SEEDS)
    p.add_argument("--single_seed",   type=int,  default=None)
    p.add_argument("--batch_size",    type=int,  default=Config.BATCH_SIZE)
    p.add_argument("--num_epochs",    type=int,  default=Config.NUM_EPOCHS)
    p.add_argument("--benign_ratio",  type=int,  default=Config.BENIGN_RATIO)
    p.add_argument("--epi_hidden",    type=int,  default=Config.EPI_HIDDEN)
    p.add_argument("--fusion_hidden", type=int,  default=Config.FUSION_HIDDEN)
    p.add_argument("--lr_backbone",   type=float,default=Config.LR_BACKBONE)
    p.add_argument("--lr_head",       type=float,default=Config.LR_HEAD)
    p.add_argument("--no_fp16",       action="store_true")
    p.add_argument("--verify_only",   action="store_true")
    return p.parse_args()

def main():
    args = parse_args(); cfg = Config()
    cfg.EPI_DIR      = args.epi_dir;      cfg.OUTPUT_DIR   = args.output_dir
    cfg.SEEDS        = [args.single_seed] if args.single_seed else args.seeds
    cfg.BATCH_SIZE   = args.batch_size;   cfg.NUM_EPOCHS   = args.num_epochs
    cfg.BENIGN_RATIO = args.benign_ratio; cfg.EPI_HIDDEN   = args.epi_hidden
    cfg.FUSION_HIDDEN= args.fusion_hidden;cfg.LR_BACKBONE  = args.lr_backbone
    cfg.LR_HEAD      = args.lr_head;      cfg.FP16         = not args.no_fp16

    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    set_seed(cfg.BENIGN_SEED)
    df, h3k_arr, dnase_arr = load_all_data(cfg)

    if args.verify_only:
        print("\n  verify_only 완료."); return

    all_results = {}
    for seed in cfg.SEEDS:
        results = {}
        if args.mode in ("train","all"):
            _, val_m = run_train(cfg, df, h3k_arr, dnase_arr, seed)
            results["val"] = val_m
        if args.mode == "val":
            results["val"]  = run_eval(cfg, df, h3k_arr, dnase_arr, seed, "val")
        if args.mode in ("test","all"):
            results["test"] = run_eval(cfg, df, h3k_arr, dnase_arr, seed, "test")
        all_results[seed] = results

    if args.mode in ("test","all") and all_results:
        print("\n" + "="*62 + "\n  최종 결과 (mean ± std)\n" + "="*62)
        summary = {}
        for tname in cfg.TISSUE_MAP.values():
            vals = [r["test"][tname]["auprc"] for r in all_results.values()
                    if "test" in r and tname in r["test"]]
            if vals:
                summary[tname] = {"mean": round(float(np.mean(vals)),4),
                                  "std":  round(float(np.std(vals)), 4)}
                print(f"  {tname:6s} AUPRC = {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        macro_vals = [r["test"]["macro_auprc"] for r in all_results.values() if "test" in r]
        if macro_vals:
            summary["macro"] = {"mean": round(float(np.mean(macro_vals)),4),
                                 "std":  round(float(np.std(macro_vals)), 4)}
            print(f"  macro  AUPRC = {np.mean(macro_vals):.4f} ± {np.std(macro_vals):.4f}")
        with open(Path(cfg.OUTPUT_DIR) / "final_summary.json", "w") as f:
            json.dump({"mode": args.mode,
                       "per_seed": {str(k): v for k, v in all_results.items()},
                       "summary": summary}, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
