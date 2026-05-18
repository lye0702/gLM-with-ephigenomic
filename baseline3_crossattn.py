"""
=======================================================================
Stage 3 Baseline: DNABERT-2 + Epigenomic Cross-Attention Fusion

■ 핵심 수정사항 (버그 수정)
  - benign tissue_id를 원래 조직(0/1/2)으로 유지 (BENIGN_TID=3 제거)
  - benign은 label=0으로만 구분
  - compute_metrics: benign_msk를 tissue_id가 아닌 label==0 기준으로 변경
  - _load_npz: tissue_id 문자열→정수 변환 처리 유지

■ 데이터 구조
  signals_liver_zscore.npz        → pathogenic (label=1, tissue_id=0)
  signals_heart_zscore.npz        → pathogenic (label=1, tissue_id=1)
  signals_brain_zscore.npz        → pathogenic (label=1, tissue_id=2)
  signals_benign_liver_zscore.npz → benign     (label=0, tissue_id=0)
  signals_benign_heart_zscore.npz → benign     (label=0, tissue_id=1)
  signals_benign_brain_zscore.npz → benign     (label=0, tissue_id=2)

■ 실행 예시
  python baseline3_crossattn.py --verify_only --epi_dir epigenomic_signals
  python baseline3_crossattn.py --mode all --single_seed 42 --epi_dir epigenomic_signals
  python baseline3_crossattn.py --mode all --seeds 42 123 456 --epi_dir epigenomic_signals
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
    # ── npz 파일 경로 설정 ────────────────────────────────────────────
    EPI_DIR = "epigenomic_signals"

    EPI_LIVER_FILE  = "signals_liver_zscore.npz"
    EPI_HEART_FILE  = "signals_heart_zscore.npz"
    EPI_BRAIN_FILE  = "signals_brain_zscore.npz"

    EPI_LIVER_BENIGN_FILE = "signals_benign_liver_zscore.npz"
    EPI_HEART_BENIGN_FILE = "signals_benign_heart_zscore.npz"
    EPI_BRAIN_BENIGN_FILE = "signals_benign_brain_zscore.npz"

    # ── 신호 차원 설정 ────────────────────────────────────────────────
    EPI_N_CHANNELS = 2
    EPI_USE_LEN    = 1024   # 앞 1024열 사용 (1025번째 제외)

    # ── tissue 설정 ───────────────────────────────────────────────────
    # benign도 동일한 tissue_id(0/1/2) 사용, label=0으로 구분
    TISSUE_MAP     = {0: "liver", 1: "heart", 2: "brain"}
    TISSUE_STR_MAP = {"liver": 0, "heart": 1, "brain": 2}
    N_TISSUES      = 3

    # ── benign 샘플링 ─────────────────────────────────────────────────
    BENIGN_RATIO = 3   # 조직별 pathogenic 수 × BENIGN_RATIO
    BENIGN_SEED  = 42

    # ── 데이터 분할 ───────────────────────────────────────────────────
    VAL_RATIO  = 0.10
    TEST_RATIO = 0.10

    # ── DNABERT-2 ─────────────────────────────────────────────────────
    MODEL_NAME = "zhihan1996/DNABERT-2-117M"
    MAX_LENGTH = 256
    HIDDEN_DIM = 768

    # ── Epigenomic Encoder ────────────────────────────────────────────
    EPI_HIDDEN = 128
    CNN_KERNEL = 7

    # ── Cross-Attention ───────────────────────────────────────────────
    ATTN_HEADS   = 8
    ATTN_DROPOUT = 0.1

    # ── Tissue Embedding ──────────────────────────────────────────────
    TISSUE_EMB_DIM = 768

    # ── 학습 설정 ─────────────────────────────────────────────────────
    DROPOUT       = 0.1
    BATCH_SIZE    = 8
    GRAD_ACCUM    = 2
    LR_BACKBONE   = 1e-5
    LR_HEAD       = 1e-4
    WEIGHT_DECAY  = 0.01
    MAX_GRAD_NORM = 1.0
    NUM_EPOCHS    = 30
    PATIENCE      = 5
    WARMUP_RATIO  = 0.05
    FP16          = True

    SEEDS       = [42, 123, 456]
    OUTPUT_DIR  = "outputs/stage3_crossattn"
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
    torch.backends.cudnn.benchmark = False


def get_device(fp16: bool):
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda"), fp16
    print("  GPU 없음 → CPU 모드 (FP16 비활성화)")
    return torch.device("cpu"), False


# ══════════════════════════════════════════════════════════════════════
# 3. 데이터 로드
# ══════════════════════════════════════════════════════════════════════

def _load_npz(fpath: Path, cfg: Config) -> dict:
    """
    단일 npz 로드.
    tissue_id가 문자열('liver')이어도 정수(0)로 자동 변환.
    """
    data = np.load(str(fpath), allow_pickle=True)

    sequences = data["sequence"].tolist()
    labels    = data["label"].astype(np.int64)

    # tissue_id: 문자열/정수 모두 처리
    tid_raw = data["tissue_id"]
    if tid_raw.dtype.kind in ("U", "S", "O"):
        tissue_ids = np.array(
            [cfg.TISSUE_STR_MAP[str(t)] for t in tid_raw], dtype=np.int64)
    else:
        tissue_ids = tid_raw.astype(np.int64)

    h3k27ac = data["h3k27ac"][:, :cfg.EPI_USE_LEN].astype(np.float32)
    dnase   = data["dnase"][:, :cfg.EPI_USE_LEN].astype(np.float32)

    return {
        "sequences":  sequences,
        "labels":     labels,
        "tissue_ids": tissue_ids,
        "h3k27ac":    h3k27ac,
        "dnase":      dnase,
    }


def load_all_data(cfg: Config) -> tuple:
    """
    pathogenic 3개 + benign 3개 npz 로드.
    benign은 조직별로 pathogenic 수 × BENIGN_RATIO 만큼 샘플링.
    tissue_id: pathogenic/benign 동일(0/1/2), label(0/1)로 구분.

    반환:
        df        : pd.DataFrame (sequence, label, tissue_id, _orig_idx)
        h3k_arr   : np.ndarray (N_total, EPI_USE_LEN)
        dnase_arr : np.ndarray (N_total, EPI_USE_LEN)
    """
    epi_dir = Path(cfg.EPI_DIR)

    patho_files = [
        (epi_dir / cfg.EPI_LIVER_FILE,        0),
        (epi_dir / cfg.EPI_HEART_FILE,        1),
        (epi_dir / cfg.EPI_BRAIN_FILE,        2),
    ]
    benign_files = [
        (epi_dir / cfg.EPI_LIVER_BENIGN_FILE, 0),
        (epi_dir / cfg.EPI_HEART_BENIGN_FILE, 1),
        (epi_dir / cfg.EPI_BRAIN_BENIGN_FILE, 2),
    ]

    all_seqs, all_labels, all_tids = [], [], []
    all_h3k,  all_dnase = [], []

    # ── Pathogenic 로드 ───────────────────────────────────────────────
    print("\n─── Pathogenic npz 로드 ───────────────────────────────")
    n_patho_per_tissue = {}
    for fpath, tid in patho_files:
        if not fpath.exists():
            raise FileNotFoundError(f"파일 없음: {fpath}")
        d = _load_npz(fpath, cfg)
        n = len(d["sequences"])
        n_patho_per_tissue[tid] = n
        print(f"  {fpath.name}: {n:,} rows  (tissue_id={tid}, label=1)")
        all_seqs.extend(d["sequences"])
        all_labels.append(d["labels"])
        all_tids.append(d["tissue_ids"])
        all_h3k.append(d["h3k27ac"])
        all_dnase.append(d["dnase"])

    # ── Benign 로드 & 조직별 샘플링 ──────────────────────────────────
    print(f"\n─── Benign npz 로드 (BENIGN_RATIO={cfg.BENIGN_RATIO}) ──")
    rng = np.random.RandomState(cfg.BENIGN_SEED)

    for fpath, tid in benign_files:
        if not fpath.exists():
            raise FileNotFoundError(f"파일 없음: {fpath}")
        d       = _load_npz(fpath, cfg)
        n_total  = len(d["sequences"])
        n_target = n_patho_per_tissue[tid] * cfg.BENIGN_RATIO
        n_sample = min(n_target, n_total)
        chosen   = rng.choice(n_total, size=n_sample, replace=False)

        print(f"  {fpath.name}: {n_total:,} → {n_sample:,} 샘플링 "
              f"(tissue_id={tid}, label=0)")

        all_seqs.extend([d["sequences"][i] for i in chosen])
        all_labels.append(np.zeros(n_sample, dtype=np.int64))       # label=0
        all_tids.append(np.full(n_sample, tid, dtype=np.int64))     # tissue_id 유지
        all_h3k.append(d["h3k27ac"][chosen])
        all_dnase.append(d["dnase"][chosen])

    # ── 전체 합치기 ───────────────────────────────────────────────────
    labels_arr = np.concatenate(all_labels,  axis=0)
    tids_arr   = np.concatenate(all_tids,    axis=0)
    h3k_arr    = np.concatenate(all_h3k,     axis=0)
    dnase_arr  = np.concatenate(all_dnase,   axis=0)

    df = pd.DataFrame({
        "sequence":  all_seqs,
        "label":     labels_arr,
        "tissue_id": tids_arr,
    })

    # 서열 길이 필터 (600~1100bp)
    seq_len_s = df["sequence"].str.len()
    keep_mask = (seq_len_s >= 600) & (seq_len_s <= 1100)
    keep_idx  = np.where(keep_mask)[0]
    df        = df[keep_mask].reset_index(drop=True)
    h3k_arr   = h3k_arr[keep_idx]
    dnase_arr = dnase_arr[keep_idx]
    df["_orig_idx"] = np.arange(len(df))

    print(f"\n  총 샘플:    {len(df):,}")
    print(f"  Label  분포: {dict(df['label'].value_counts().sort_index())}")
    print(f"  Tissue 분포: {dict(df['tissue_id'].value_counts().sort_index())}")
    print(f"  조직별 pos/neg:")
    for tid, tname in cfg.TISSUE_MAP.items():
        sub = df[df["tissue_id"] == tid]
        pos = int((sub["label"] == 1).sum())
        neg = int((sub["label"] == 0).sum())
        print(f"    {tname:5s}: pos={pos:,}  neg={neg:,}")

    return df, h3k_arr, dnase_arr


# ══════════════════════════════════════════════════════════════════════
# 4. Data Split
# ══════════════════════════════════════════════════════════════════════

def stratified_split(df, cfg, seed):
    strat = df["tissue_id"].astype(str) + "_" + df["label"].astype(str)
    if strat.value_counts().min() < 2:
        strat = df["label"].astype(str)

    train_val, test = train_test_split(
        df, test_size=cfg.TEST_RATIO, stratify=strat, random_state=seed)

    strat_tv = (train_val["tissue_id"].astype(str) + "_" +
                train_val["label"].astype(str))
    if strat_tv.value_counts().min() < 2:
        strat_tv = train_val["label"].astype(str)
    val_adj = cfg.VAL_RATIO / (1.0 - cfg.TEST_RATIO)

    train, val = train_test_split(
        train_val, test_size=val_adj, stratify=strat_tv, random_state=seed)

    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════
# 5. Dataset & DataLoader
# ══════════════════════════════════════════════════════════════════════

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

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.sequences[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        oi  = self.orig_idxs[idx]
        epi = torch.from_numpy(
            np.stack([self.h3k_arr[oi], self.dnase_arr[oi]], axis=0)
        ).float()   # (2, 1024)

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "epi_signal":     epi,
            "label":          torch.tensor(self.labels[idx],     dtype=torch.float),
            "tissue_id":      torch.tensor(self.tissue_ids[idx], dtype=torch.long),
        }


def make_weighted_sampler(df):
    labels  = df["label"].values
    counts  = np.bincount(labels)
    weights = torch.tensor(1.0 / counts[labels], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def _make_loader(ds, batch_size, shuffle=False, sampler=None, num_workers=4):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      sampler=sampler, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())


# ══════════════════════════════════════════════════════════════════════
# 6. Model
# ══════════════════════════════════════════════════════════════════════

class EpigenomicEncoder(nn.Module):
    """1D CNN: (B, 2, 1024) → (B, MAX_LENGTH, epi_hidden)"""

    def __init__(self, in_channels=2, hidden=128, kernel=7, target_len=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel, padding=kernel // 2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(target_len)

    def forward(self, x):
        out = self.cnn(x)           # (B, hidden, 1024)
        out = self.pool(out)        # (B, hidden, target_len)
        return out.permute(0, 2, 1) # (B, target_len, hidden)


class CrossAttentionFusion(nn.Module):
    """DNA(Query) × Epi(Key, Value) Cross-Attention + residual + LayerNorm"""

    def __init__(self, dna_dim=768, epi_dim=128, n_heads=8, dropout=0.1):
        super().__init__()
        assert dna_dim % n_heads == 0
        self.n_heads  = n_heads
        self.dna_dim  = dna_dim
        self.head_dim = dna_dim // n_heads

        self.q_proj   = nn.Linear(dna_dim, dna_dim)
        self.k_proj   = nn.Linear(epi_dim, dna_dim)
        self.v_proj   = nn.Linear(epi_dim, dna_dim)
        self.out_proj = nn.Linear(dna_dim, dna_dim)
        self.drop     = nn.Dropout(dropout)
        self.norm     = nn.LayerNorm(dna_dim)

    def forward(self, dna, epi):
        B, L, _ = dna.shape

        def split_heads(x):
            return x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(self.q_proj(dna))
        K = split_heads(self.k_proj(epi))
        V = split_heads(self.v_proj(epi))

        attn = self.drop(F.softmax(
            torch.matmul(Q, K.transpose(-2, -1)) * (self.head_dim ** -0.5),
            dim=-1))
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, self.dna_dim)
        return self.norm(dna + self.out_proj(out))


class DNABERT2CrossAttn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        bert_cfg     = BertConfig.from_pretrained(cfg.MODEL_NAME)
        self.dna_enc = AutoModel.from_pretrained(
            cfg.MODEL_NAME, trust_remote_code=True, config=bert_cfg)
        self.epi_enc = EpigenomicEncoder(
            in_channels=cfg.EPI_N_CHANNELS,
            hidden=cfg.EPI_HIDDEN,
            kernel=cfg.CNN_KERNEL,
            target_len=cfg.MAX_LENGTH,
        )
        self.fusion = CrossAttentionFusion(
            dna_dim=cfg.HIDDEN_DIM,
            epi_dim=cfg.EPI_HIDDEN,
            n_heads=cfg.ATTN_HEADS,
            dropout=cfg.ATTN_DROPOUT,
        )
        self.tissue_emb  = nn.Embedding(cfg.N_TISSUES, cfg.TISSUE_EMB_DIM)
        self.tissue_norm = nn.LayerNorm(cfg.TISSUE_EMB_DIM)
        self.classifier  = nn.Sequential(
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.HIDDEN_DIM, 256), nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def _masked_mean_pool(self, h, mask):
        m = mask.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask, epi_signal, tissue_id):
        dna_repr = self.dna_enc(input_ids=input_ids,
                                attention_mask=attention_mask)[0]   # (B, L, 768)
        epi_repr = self.epi_enc(epi_signal)                         # (B, L, epi_hidden)
        fused    = self.fusion(dna_repr, epi_repr)                  # (B, L, 768)
        t        = self.tissue_norm(self.tissue_emb(tissue_id))     # (B, 768)
        fused    = fused + t.unsqueeze(1)                           # (B, L, 768)
        pooled   = self._masked_mean_pool(fused, attention_mask)    # (B, 768)
        return self.classifier(pooled).squeeze(-1)                  # (B,)


# ══════════════════════════════════════════════════════════════════════
# 7. Metrics & Evaluation
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(model, loader, device, fp16):
    model.eval()
    all_labels, all_probs, all_tissues = [], [], []

    for batch in tqdm(loader, desc="  Inference", leave=False):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        epi  = batch["epi_signal"].to(device)
        tids = batch["tissue_id"].to(device)

        with torch.cuda.amp.autocast(enabled=fp16):
            logits = model(ids, mask, epi, tids)
        probs = torch.sigmoid(logits).float().cpu().numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(batch["label"].numpy().tolist())
        all_tissues.extend(batch["tissue_id"].numpy().tolist())

    return (np.array(all_labels,  dtype=np.float32),
            np.array(all_probs,   dtype=np.float32),
            np.array(all_tissues, dtype=np.int32))


def compute_metrics(labels, probs, tissues, cfg) -> dict:
    """
    조직별 평가:
      - 해당 조직의 pathogenic(label=1)
      - 전체 benign(label=0, tissue_id 무관)
    → benign은 label 기준으로 구분 (tissue_id 기준 X)
    """
    metrics, auprc_list = {}, []
    benign_msk = (labels == 0)   # ← label 기준으로 benign 구분

    for tid, tname in cfg.TISSUE_MAP.items():
        patho_msk = (tissues == tid) & (labels == 1)
        msk       = patho_msk | benign_msk

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


def print_metrics(metrics, prefix=""):
    print(f"{prefix}Macro-AUPRC: {metrics['macro_auprc']:.4f}")
    for k, v in metrics.items():
        if k == "macro_auprc":
            continue
        print(f"{prefix}  {k:6s} AUPRC={v['auprc']:.4f} | "
              f"P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} "
              f"(pos={v['n_pathogenic']}, neg={v['n_benign']})")


def _save_predictions(labels, probs, tissues, path):
    pd.DataFrame({
        "label":     labels.tolist(),
        "prob":      probs.tolist(),
        "tissue_id": tissues.tolist(),
    }).to_csv(path, index=False)


# ══════════════════════════════════════════════════════════════════════
# 8-A. TRAIN
# ══════════════════════════════════════════════════════════════════════

def run_train(cfg, df, h3k_arr, dnase_arr, seed):
    set_seed(seed)
    out_dir   = Path(cfg.OUTPUT_DIR) / f"seed_{seed}"
    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    device, use_fp16 = get_device(cfg.FP16)

    print(f"\n{'═'*62}")
    print(f"  [TRAIN] Stage3 CrossAttn | Seed={seed} | FP16={use_fp16}")
    print('═'*62)

    # ── Split ─────────────────────────────────────────────────────────
    train_df, val_df, test_df = stratified_split(df, cfg, seed)

    np.save(str(split_dir / "val_orig_idx.npy"),  val_df["_orig_idx"].values)
    np.save(str(split_dir / "test_orig_idx.npy"), test_df["_orig_idx"].values)
    with open(split_dir / "meta.json", "w") as f:
        json.dump({"seed": seed, "benign_seed": cfg.BENIGN_SEED,
                   "train": len(train_df), "val": len(val_df),
                   "test":  len(test_df),  "total": len(df)}, f, indent=2)

    n_pos = int((train_df["label"] == 1).sum())
    n_neg = int((train_df["label"] == 0).sum())
    print(f"  Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}")
    print(f"  Train pos(patho)={n_pos:,} | neg(benign)={n_neg:,}")

    # ── Tokenizer & DataLoaders ───────────────────────────────────────
    print(f"\n  Tokenizer 로드: {cfg.MODEL_NAME}")
    tokenizer    = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True)
    train_ds     = VariantEpiDataset(train_df, tokenizer, h3k_arr, dnase_arr, cfg)
    val_ds       = VariantEpiDataset(val_df,   tokenizer, h3k_arr, dnase_arr, cfg)
    sampler      = make_weighted_sampler(train_df)
    train_loader = _make_loader(train_ds, cfg.BATCH_SIZE, sampler=sampler,
                                num_workers=cfg.NUM_WORKERS)
    val_loader   = _make_loader(val_ds, cfg.BATCH_SIZE * 2,
                                num_workers=cfg.NUM_WORKERS)

    # ── Model ─────────────────────────────────────────────────────────
    print("  모델 초기화: DNABERT2CrossAttn")
    model = DNABERT2CrossAttn(cfg).to(device)
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  학습 파라미터: {n_p/1e6:.1f}M")

    # ── Optimizer (차등 학습률) ───────────────────────────────────────
    optimizer = torch.optim.AdamW([
        {"params": model.dna_enc.parameters(),     "lr": cfg.LR_BACKBONE},
        {"params": model.epi_enc.parameters(),     "lr": cfg.LR_HEAD},
        {"params": model.fusion.parameters(),      "lr": cfg.LR_HEAD},
        {"params": model.tissue_emb.parameters(),  "lr": cfg.LR_HEAD},
        {"params": model.tissue_norm.parameters(), "lr": cfg.LR_HEAD},
        {"params": model.classifier.parameters(),  "lr": cfg.LR_HEAD},
    ], weight_decay=cfg.WEIGHT_DECAY)

    total_steps  = (len(train_loader) // cfg.GRAD_ACCUM) * cfg.NUM_EPOCHS
    warmup_steps = int(total_steps * cfg.WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Loss ──────────────────────────────────────────────────────────
    pos_w = torch.tensor(n_neg / n_pos, dtype=torch.float).to(device)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    print(f"  pos_weight={pos_w.item():.2f}  (neg={n_neg:,} / pos={n_pos:,})")

    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # ── Training Loop ─────────────────────────────────────────────────
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
            epi  = batch["epi_signal"].to(device)
            tids = batch["tissue_id"].to(device)
            lbl  = batch["label"].to(device)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                loss = crit(model(ids, mask, epi, tids), lbl) / cfg.GRAD_ACCUM

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * cfg.GRAD_ACCUM

            if (step + 1) % cfg.GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{epoch_loss/(step+1):.4f}"})

        val_lbl, val_prob, val_tid = run_inference(model, val_loader, device, use_fp16)
        val_m  = compute_metrics(val_lbl, val_prob, val_tid, cfg)
        macro  = val_m["macro_auprc"]
        avg_l  = epoch_loss / len(train_loader)

        print(f"\n  Epoch {epoch:02d} | Loss={avg_l:.4f} | Val Macro-AUPRC={macro:.4f}")
        print_metrics(val_m, prefix="  ")

        row = {"epoch": epoch, "loss": avg_l, "val_macro_auprc": macro}
        for t, m in val_m.items():
            if t != "macro_auprc":
                row[f"val_{t}_auprc"] = m["auprc"]
        history.append(row)

        if macro > best_macro:
            best_macro, patience_cnt = macro, 0
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_metrics": val_m,
                "config": {"model_name": cfg.MODEL_NAME,
                           "max_length": cfg.MAX_LENGTH,
                           "epi_hidden": cfg.EPI_HIDDEN,
                           "attn_heads": cfg.ATTN_HEADS}
            }, out_dir / "best_model.pt")
            print(f"  ✅ Best 모델 저장 (Macro-AUPRC={best_macro:.4f})")
        else:
            patience_cnt += 1
            print(f"  patience: {patience_cnt}/{cfg.PATIENCE}")
            if patience_cnt >= cfg.PATIENCE:
                print(f"\n  Early stopping @ epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    with open(out_dir / "val_metrics.json", "w") as f:
        json.dump(ckpt["val_metrics"], f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ 학습 완료: {out_dir}/")
    return pd.DataFrame(history), ckpt["val_metrics"]


# ══════════════════════════════════════════════════════════════════════
# 8-B. EVAL (val / test)
# ══════════════════════════════════════════════════════════════════════

def run_eval(cfg, df, h3k_arr, dnase_arr, seed, split="test"):
    out_dir   = Path(cfg.OUTPUT_DIR) / f"seed_{seed}"
    split_dir = out_dir / "splits"
    device, use_fp16 = get_device(cfg.FP16)

    print(f"\n{'═'*62}")
    print(f"  [{split.upper()}] Stage3 | Seed={seed}")
    print('═'*62)

    idx_path = split_dir / f"{split}_orig_idx.npy"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"  ❌ {idx_path} 없음. 먼저 train을 실행하세요.")
    orig_idx = np.load(str(idx_path))
    eval_df  = df.iloc[orig_idx].reset_index(drop=True)
    n_pos    = int((eval_df["label"] == 1).sum())
    n_neg    = int((eval_df["label"] == 0).sum())
    print(f"  {split} 샘플: {len(eval_df):,}  (pos={n_pos:,}, neg={n_neg:,})")

    ckpt_path = out_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"  ❌ {ckpt_path} 없음.")
    ckpt  = torch.load(str(ckpt_path), map_location=device)
    model = DNABERT2CrossAttn(cfg)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    print(f"  모델 로드 완료 (Best epoch={ckpt['epoch']})")

    tokenizer   = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True)
    eval_ds     = VariantEpiDataset(eval_df, tokenizer, h3k_arr, dnase_arr, cfg)
    eval_loader = _make_loader(eval_ds, cfg.BATCH_SIZE * 2,
                               num_workers=cfg.NUM_WORKERS)

    labels, probs, tissues = run_inference(model, eval_loader, device, use_fp16)
    metrics = compute_metrics(labels, probs, tissues, cfg)

    print(f"\n  {split.upper()} 결과:")
    print_metrics(metrics, prefix="  ")

    with open(out_dir / f"{split}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    _save_predictions(labels, probs, tissues,
                      out_dir / f"{split}_predictions.csv")
    print(f"  결과 저장: {out_dir}/{split}_metrics.json")
    return metrics


# ══════════════════════════════════════════════════════════════════════
# 9. Main
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 3: DNABERT-2 + Epigenomic Cross-Attention")
    p.add_argument("--mode", choices=["train", "val", "test", "all"], default="all")
    p.add_argument("--epi_dir",      default=Config.EPI_DIR)
    p.add_argument("--output_dir",   default=Config.OUTPUT_DIR)
    p.add_argument("--seeds",        nargs="+", type=int, default=Config.SEEDS)
    p.add_argument("--single_seed",  type=int,  default=None)
    p.add_argument("--batch_size",   type=int,  default=Config.BATCH_SIZE)
    p.add_argument("--num_epochs",   type=int,  default=Config.NUM_EPOCHS)
    p.add_argument("--benign_ratio", type=int,  default=Config.BENIGN_RATIO)
    p.add_argument("--epi_hidden",   type=int,  default=Config.EPI_HIDDEN)
    p.add_argument("--attn_heads",   type=int,  default=Config.ATTN_HEADS)
    p.add_argument("--no_fp16",      action="store_true")
    p.add_argument("--verify_only",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config()

    cfg.EPI_DIR      = args.epi_dir
    cfg.OUTPUT_DIR   = args.output_dir
    cfg.SEEDS        = [args.single_seed] if args.single_seed else args.seeds
    cfg.BATCH_SIZE   = args.batch_size
    cfg.NUM_EPOCHS   = args.num_epochs
    cfg.BENIGN_RATIO = args.benign_ratio
    cfg.EPI_HIDDEN   = args.epi_hidden
    cfg.ATTN_HEADS   = args.attn_heads
    cfg.FP16         = not args.no_fp16

    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    set_seed(cfg.BENIGN_SEED)
    df, h3k_arr, dnase_arr = load_all_data(cfg)

    if args.verify_only:
        print("\n  verify_only 완료. 종료합니다.")
        return

    all_results = {}
    for seed in cfg.SEEDS:
        results = {}
        if args.mode in ("train", "all"):
            _, val_m = run_train(cfg, df, h3k_arr, dnase_arr, seed)
            results["val"] = val_m
        if args.mode == "val":
            results["val"] = run_eval(cfg, df, h3k_arr, dnase_arr, seed, "val")
        if args.mode in ("test", "all"):
            results["test"] = run_eval(cfg, df, h3k_arr, dnase_arr, seed, "test")
        all_results[seed] = results

    if args.mode in ("test", "all") and all_results:
        print("\n" + "═"*62)
        print("  최종 결과 (mean ± std across seeds)")
        print("═"*62)
        summary = {}
        for tname in cfg.TISSUE_MAP.values():
            vals = [r["test"][tname]["auprc"]
                    for r in all_results.values()
                    if "test" in r and tname in r["test"]]
            if vals:
                summary[tname] = {"mean": round(float(np.mean(vals)), 4),
                                  "std":  round(float(np.std(vals)),  4)}
                print(f"  {tname:6s} AUPRC = {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        macro_vals = [r["test"]["macro_auprc"]
                      for r in all_results.values() if "test" in r]
        if macro_vals:
            summary["macro"] = {"mean": round(float(np.mean(macro_vals)), 4),
                                 "std":  round(float(np.std(macro_vals)),  4)}
            print(f"  macro  AUPRC = {np.mean(macro_vals):.4f} ± {np.std(macro_vals):.4f}")

        final_path = Path(cfg.OUTPUT_DIR) / "final_summary.json"
        with open(final_path, "w") as f:
            json.dump({"mode": args.mode,
                       "per_seed": {str(k): v for k, v in all_results.items()},
                       "summary":  summary},
                      f, indent=2, ensure_ascii=False)
        print(f"\n  전체 결과 저장: {final_path}")


if __name__ == "__main__":
    main()
