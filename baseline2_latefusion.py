"""
=======================================================================
Stage 2 Baseline: DNABERT-2 + Late Fusion (Epigenomic Concatenation)

■ baseline3_crossattn.py 와 다른 점 (단 한 곳)
  Cross-Attention Fusion  →  Late Fusion (concat + Linear projection)
  - DNA CLS 토큰 [768] + EpiEncoder 글로벌 평균풀링 [epi_hidden] + Tissue Emb [epi_hidden]
  - concat → Linear → LayerNorm → MLP → 병원성 확률

■ 데이터 구조 (baseline3 와 동일)
  signals_liver_zscore.npz / heart / brain
  signals_liver_benign_zscore.npz / heart / brain
  keys: chrom, pos, ref, alt, label, tissue_id,
        sequence, h3k27ac(N,1025), dnase(N,1025)

■ 실행 예시
  python baseline2_latefusion.py --verify_only
  python baseline2_latefusion.py --mode all --no_fp16
  python baseline2_latefusion.py --mode train --epi_dir /path/to/npz --output_dir ./out_latefusion
=======================================================================
"""

import os, json, random, warnings, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer, AutoModel,
    get_linear_schedule_with_warmup,
)
from transformers.models.bert.configuration_bert import BertConfig
from sklearn.metrics import (
    average_precision_score,
    precision_score, recall_score, f1_score,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
# 1. 설정 (baseline3 와 동일값 유지)
# ══════════════════════════════════════════════════════════════════════
class CFG:
    MODEL_NAME   = "zhihan1996/DNABERT-2-117M"
    MAX_LENGTH   = 256          # DNABERT-2 토큰 최대 길이
    EPI_HIDDEN   = 128          # Epigenomic Encoder 출력 차원
    CNN_KERNEL   = 7
    N_TISSUES    = 3            # liver=0, heart=1, brain=2
    TISSUE_DIM   = 128          # Tissue Embedding 차원 (EPI_HIDDEN 과 동일)

    # Late Fusion 전용 파라미터
    # concat 후 입력 차원 = 768(CLS) + EPI_HIDDEN + TISSUE_DIM
    # → fusion_in = 768 + 128 + 128 = 1024
    FUSION_HIDDEN = 512         # Fusion Linear 중간 차원

    BATCH_SIZE   = 16
    GRAD_ACCUM   = 2
    EPOCHS       = 30
    PATIENCE     = 5
    LR_BACKBONE  = 1e-5
    LR_HEAD      = 1e-4
    WEIGHT_DECAY = 1e-2
    WARMUP_RATIO = 0.1
    BENIGN_RATIO = 3            # benign : pathogenic 비율
    SEEDS        = [42, 123, 456]

    EPI_DIR      = Path("/workspace/gLM-with-ephigenomic/epigenomic_signals")
    OUTPUT_DIR   = Path("./outputs/basic_baseline2")

    TISSUE_MAP   = {"liver": 0, "heart": 1, "brain": 2}
    SEQ_MIN      = 600
    SEQ_MAX      = 1100

    # npz 파일명 패턴 (baseline3 와 동일)
    PATHO_FILES  = [
        "signals_liver_zscore.npz",
        "signals_heart_zscore.npz",
        "signals_brain_zscore.npz",
    ]
    BENIGN_FILES = [
        "signals_benign_liver_zscore.npz",
        "signals_benign_heart_zscore.npz",
        "signals_benign_brain_zscore.npz",
    ]


cfg = CFG()


# ══════════════════════════════════════════════════════════════════════
# 2. 데이터셋 (baseline3 와 완전 동일)
# ══════════════════════════════════════════════════════════════════════
TISSUE_MAP = {"liver": 0, "heart": 1, "brain": 2}

def _load_npz(path: Path, benign_ratio: int = None, rng=None):
    """npz 한 파일 로드 → list of dict"""
    d = np.load(path, allow_pickle=True)
    seqs     = d["sequence"]
    labels   = d["label"].astype(int)
    tids_raw = d["tissue_id"]
    h3k      = d["h3k27ac"][:, :1024].astype(np.float32)
    dnase    = d["dnase"][:, :1024].astype(np.float32)

    # tissue_id: str → int 변환
    def _tid(x):
        if isinstance(x, (int, np.integer)):
            return int(x)
        s = str(x).strip().lower()
        return TISSUE_MAP.get(s, 0)

    rows = []
    for i in range(len(seqs)):
        seq = str(seqs[i])
        if not (cfg.SEQ_MIN <= len(seq) <= cfg.SEQ_MAX):
            continue
        rows.append({
            "sequence":  seq,
            "label":     int(labels[i]),
            "tissue_id": _tid(tids_raw[i]),
            "h3k27ac":   h3k[i],
            "dnase":     dnase[i],
        })
    return rows


def load_all_data(epi_dir: Path, benign_ratio: int, seed: int):
    rng = np.random.default_rng(seed)

    patho_rows = []
    for fn in cfg.PATHO_FILES:
        fp = epi_dir / fn
        if fp.exists():
            patho_rows += _load_npz(fp)
        else:
            print(f"  [WARN] 파일 없음: {fp}")

    benign_rows = []
    for fn in cfg.BENIGN_FILES:
        fp = epi_dir / fn
        if fp.exists():
            benign_rows += _load_npz(fp)
        else:
            print(f"  [WARN] 파일 없음: {fp}")

    # benign 다운샘플링
    n_target = min(len(patho_rows) * benign_ratio, len(benign_rows))
    idx = rng.choice(len(benign_rows), n_target, replace=False)
    benign_rows = [benign_rows[i] for i in idx]

    all_rows = patho_rows + benign_rows
    rng.shuffle(all_rows)

    print(f"  Pathogenic: {len(patho_rows):,} | Benign: {len(benign_rows):,} | Total: {len(all_rows):,}")
    return all_rows


def chrom_split(rows, val_chroms=None, test_chroms=None):
    """염색체 기반 train/val/test 분할 — npz에 chrom 없으면 비율 분할로 fallback"""
    # npz에 chrom 키 없는 경우가 많으므로 비율 fallback
    n = len(rows)
    idx = list(range(n))
    random.shuffle(idx)
    t1 = int(n * 0.7)
    t2 = int(n * 0.85)
    train = [rows[i] for i in idx[:t1]]
    val   = [rows[i] for i in idx[t1:t2]]
    test  = [rows[i] for i in idx[t2:]]
    return train, val, test


class VariantDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.rows       = rows
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r   = self.rows[idx]
        enc = self.tokenizer(
            r["sequence"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        epi = np.stack([r["h3k27ac"], r["dnase"]], axis=0)  # (2, 1024)
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "epi":            torch.tensor(epi, dtype=torch.float32),
            "tissue_id":      torch.tensor(r["tissue_id"], dtype=torch.long),
            "label":          torch.tensor(r["label"], dtype=torch.float32),
        }


# ══════════════════════════════════════════════════════════════════════
# 3. 모델 — Late Fusion 버전
#    baseline3 와 달라지는 부분:
#      EpiEncoder → GlobalAvgPool → [B, epi_hidden]
#      Tissue Embedding → [B, tissue_dim]
#      DNA CLS token → [B, 768]
#      세 벡터를 concat → Linear(fusion_in, fusion_hidden) → LayerNorm → MLP
# ══════════════════════════════════════════════════════════════════════
class EpiEncoder(nn.Module):
    """
    baseline3 와 동일한 1D CNN 구조.
    Late Fusion 에서는 출력에 GlobalAvgPool 적용해 [B, epi_hidden] 벡터로 압축.
    """
    def __init__(self, epi_hidden: int, kernel: int):
        super().__init__()
        self.conv1 = nn.Conv1d(2, epi_hidden, kernel, padding=kernel // 2)
        self.norm1 = nn.LayerNorm(epi_hidden)
        self.conv2 = nn.Conv1d(epi_hidden, epi_hidden, kernel, padding=kernel // 2)
        self.norm2 = nn.LayerNorm(epi_hidden)

    def forward(self, epi):
        # epi: [B, 2, 1024]
        x = F.gelu(self.conv1(epi))                # [B, epi_hidden, 1024]
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x = F.gelu(self.conv2(x))                  # [B, epi_hidden, 1024]
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = x.mean(dim=2)                           # GlobalAvgPool → [B, epi_hidden]
        return x


class LateFusionModel(nn.Module):
    """
    DNABERT-2 (CLS 토큰) + EpiEncoder (GlobalAvgPool) + Tissue Embedding
    → Concatenation → Linear → MLP → 병원성 확률
    """
    def __init__(self, model_name: str, epi_hidden: int, n_tissues: int,
                 tissue_dim: int, fusion_hidden: int, kernel: int):
        super().__init__()

        # ── DNA Encoder ──────────────────────────────────────────────
        bert_cfg  = BertConfig.from_pretrained(model_name, trust_remote_code=True)
        self.dna_enc = AutoModel.from_pretrained(
            model_name,
            config=bert_cfg,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )
        dna_dim = self.dna_enc.config.hidden_size  # 768

        # ── Epigenomic Encoder ────────────────────────────────────────
        self.epi_enc = EpiEncoder(epi_hidden, kernel)

        # ── Tissue Embedding ──────────────────────────────────────────
        self.tissue_emb  = nn.Embedding(n_tissues, tissue_dim)
        self.tissue_norm = nn.LayerNorm(tissue_dim)

        # ── Late Fusion: concat → projection ─────────────────────────
        fusion_in = dna_dim + epi_hidden + tissue_dim   # 768 + 128 + 128 = 1024
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
        )

        # ── MLP Classifier ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden // 2, 1),
        )

    def forward(self, input_ids, attention_mask, epi, tissue_id):
        # ── DNA: CLS 토큰 ─────────────────────────────────────────────
        dna_out = self.dna_enc(input_ids=input_ids,
                               attention_mask=attention_mask)
        hidden = dna_out[0] if isinstance(dna_out, tuple) else dna_out.last_hidden_state
        cls_vec = hidden[:, 0, :]  # [B, 768]

        # ── Epigenomic: GlobalAvgPool ─────────────────────────────────
        epi_vec = self.epi_enc(epi)                   # [B, epi_hidden]

        # ── Tissue Embedding ──────────────────────────────────────────
        tis_vec = self.tissue_norm(self.tissue_emb(tissue_id))  # [B, tissue_dim]

        # ── Late Fusion ───────────────────────────────────────────────
        fused = torch.cat([cls_vec, epi_vec, tis_vec], dim=1)   # [B, fusion_in]
        fused = self.fusion_proj(fused)                          # [B, fusion_hidden]

        logit = self.classifier(fused).squeeze(-1)               # [B]
        return logit


# ══════════════════════════════════════════════════════════════════════
# 4. 학습 유틸 (baseline3 와 동일)
# ══════════════════════════════════════════════════════════════════════
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_sampler(rows):
    labels = np.array([r["label"] for r in rows])
    counts = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.float32),
                                 num_samples=len(rows), replacement=True)


def compute_metrics(labels, probs, tissue_ids):
    labels    = np.array(labels)
    probs     = np.array(probs)
    tissue_ids = np.array(tissue_ids)

    per_tissue = {}
    for tid, tname in enumerate(["liver", "heart", "brain"]):
        mask = tissue_ids == tid
        if mask.sum() < 2 or labels[mask].sum() == 0:
            per_tissue[tname] = float("nan")
            continue
        try:
            per_tissue[tname] = average_precision_score(labels[mask], probs[mask])
        except Exception:
            per_tissue[tname] = float("nan")

    valid = [v for v in per_tissue.values() if not np.isnan(v)]
    macro_auprc = float(np.mean(valid)) if valid else float("nan")

    preds = (probs >= 0.5).astype(int)
    return {
        "macro_auprc": macro_auprc,
        "per_tissue":  per_tissue,
        "f1":          f1_score(labels, preds, zero_division=0),
        "precision":   precision_score(labels, preds, zero_division=0),
        "recall":      recall_score(labels, preds, zero_division=0),
    }


def get_optimizer(model, lr_backbone, lr_head, weight_decay):
    backbone_ids = set(id(p) for p in model.dna_enc.parameters())
    backbone_params = [p for p in model.parameters() if id(p) in backbone_ids]
    head_params     = [p for p in model.parameters() if id(p) not in backbone_ids]
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params,     "lr": lr_head},
    ], weight_decay=weight_decay)


# ══════════════════════════════════════════════════════════════════════
# 5. 학습 / 평가 루프 (baseline3 와 동일)
# ══════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, scheduler, scaler,
                    device, grad_accum, use_fp16):
    model.train()
    pos_w = torch.tensor([cfg.BENIGN_RATIO], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    total_loss, steps = 0.0, 0
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(loader, desc="  train", leave=False)):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        epi       = batch["epi"].to(device)
        tissue_id = batch["tissue_id"].to(device)
        labels    = batch["label"].to(device)

        if use_fp16:
            with torch.cuda.amp.autocast():
                logit = model(input_ids, attn_mask, epi, tissue_id)
                loss  = criterion(logit, labels) / grad_accum
            scaler.scale(loss).backward()
        else:
            logit = model(input_ids, attn_mask, epi, tissue_id)
            loss  = criterion(logit, labels) / grad_accum
            loss.backward()

        if (i + 1) % grad_accum == 0:
            if use_fp16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        steps += 1

    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate(model, loader, device, use_fp16):
    model.eval()
    all_labels, all_probs, all_tids = [], [], []

    for batch in tqdm(loader, desc="  eval ", leave=False):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        epi       = batch["epi"].to(device)
        tissue_id = batch["tissue_id"].to(device)

        if use_fp16:
            with torch.cuda.amp.autocast():
                logit = model(input_ids, attn_mask, epi, tissue_id)
        else:
            logit = model(input_ids, attn_mask, epi, tissue_id)

        prob = torch.sigmoid(logit).cpu().float().numpy()
        all_labels.extend(batch["label"].numpy().tolist())
        all_probs.extend(prob.tolist())
        all_tids.extend(batch["tissue_id"].numpy().tolist())

    return compute_metrics(all_labels, all_probs, all_tids)


# ══════════════════════════════════════════════════════════════════════
# 6. 메인 학습 루프
# ══════════════════════════════════════════════════════════════════════
def run_single_seed(seed: int, args):
    set_seed(seed)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = not args.no_fp16 and device.type == "cuda"
    out_dir  = Path(args.output_dir) / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Seed={seed} | device={device} | fp16={use_fp16}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    # ── 데이터 ────────────────────────────────────────────────────────
    all_rows = load_all_data(Path(args.epi_dir), args.benign_ratio, seed)
    train_rows, val_rows, test_rows = chrom_split(all_rows)
    print(f"  Split  train={len(train_rows):,} val={len(val_rows):,} test={len(test_rows):,}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME, trust_remote_code=True
    )

    train_ds = VariantDataset(train_rows, tokenizer, cfg.MAX_LENGTH)
    val_ds   = VariantDataset(val_rows,   tokenizer, cfg.MAX_LENGTH)
    test_ds  = VariantDataset(test_rows,  tokenizer, cfg.MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=make_sampler(train_rows),
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── 모델 ──────────────────────────────────────────────────────────
    model = LateFusionModel(
        model_name   = cfg.MODEL_NAME,
        epi_hidden   = args.epi_hidden,
        n_tissues    = cfg.N_TISSUES,
        tissue_dim   = cfg.TISSUE_DIM,
        fusion_hidden= args.fusion_hidden,
        kernel       = cfg.CNN_KERNEL,
    ).to(device)

    total_steps   = (len(train_loader) // cfg.GRAD_ACCUM) * args.epochs
    warmup_steps  = int(total_steps * cfg.WARMUP_RATIO)
    optimizer     = get_optimizer(model, args.lr_backbone, args.lr_head, cfg.WEIGHT_DECAY)
    scheduler     = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler        = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # ── 학습 ──────────────────────────────────────────────────────────
    best_auprc   = -1.0
    patience_cnt = 0
    history      = []

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                                  scaler, device, cfg.GRAD_ACCUM, use_fp16)
        val_m   = evaluate(model, val_loader, device, use_fp16)

        print(f"  Epoch {epoch:02d} | loss={tr_loss:.4f} "
              f"| val AUPRC={val_m['macro_auprc']:.4f} "
              f"(liver={val_m['per_tissue']['liver']:.3f} "
              f"heart={val_m['per_tissue']['heart']:.3f} "
              f"brain={val_m['per_tissue']['brain']:.3f}) "
              f"| F1={val_m['f1']:.4f}")

        history.append({"epoch": epoch, "tr_loss": tr_loss, **val_m})

        if val_m["macro_auprc"] > best_auprc:
            best_auprc = val_m["macro_auprc"]
            patience_cnt = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_metrics": val_m,
                "config": {
                    "model_name":   cfg.MODEL_NAME,
                    "epi_hidden":   args.epi_hidden,
                    "tissue_dim":   cfg.TISSUE_DIM,
                    "fusion_hidden":args.fusion_hidden,
                    "fusion_type":  "late_fusion",
                },
            }, out_dir / "best_model.pt")
            print(f"  ✓ best saved (AUPRC={best_auprc:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    # ── 테스트 ────────────────────────────────────────────────────────
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_m = evaluate(model, test_loader, device, use_fp16)

    print(f"\n  [Test] AUPRC={test_m['macro_auprc']:.4f} "
          f"F1={test_m['f1']:.4f} "
          f"Prec={test_m['precision']:.4f} "
          f"Rec={test_m['recall']:.4f}")
    print(f"         liver={test_m['per_tissue']['liver']:.4f} "
          f"heart={test_m['per_tissue']['heart']:.4f} "
          f"brain={test_m['per_tissue']['brain']:.4f}")

    # ── 저장 ──────────────────────────────────────────────────────────
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_m, f, indent=2)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return test_m


# ══════════════════════════════════════════════════════════════════════
# 7. CLI
# ══════════════════════════════════════════════════════════════════════
def verify_data(epi_dir: Path):
    """데이터 파일 존재 여부 및 배열 shape 확인"""
    print("\n[Verify] npz 파일 확인")
    all_ok = True
    for fn in cfg.PATHO_FILES + cfg.BENIGN_FILES:
        fp = epi_dir / fn
        if not fp.exists():
            print(f"  ✗ 없음: {fp}")
            all_ok = False
            continue
        d = np.load(fp, allow_pickle=True)
        keys = list(d.keys())
        n    = len(d["sequence"])
        h3k_shape = d["h3k27ac"].shape
        dns_shape = d["dnase"].shape
        print(f"  ✓ {fn}: n={n:,} h3k27ac={h3k_shape} dnase={dns_shape}")
        if not all(k in keys for k in ["sequence","label","tissue_id","h3k27ac","dnase"]):
            print(f"    [WARN] 누락 키: {set(['sequence','label','tissue_id','h3k27ac','dnase'])-set(keys)}")
    if all_ok:
        print("  → 모든 파일 확인 완료\n")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Baseline2: Late Fusion")

    parser.add_argument("--mode",         default="all",  choices=["train","eval","all"])
    parser.add_argument("--epi_dir",      default=str(cfg.EPI_DIR))
    parser.add_argument("--output_dir",   default=str(cfg.OUTPUT_DIR))
    parser.add_argument("--epochs",       type=int,   default=cfg.EPOCHS)
    parser.add_argument("--batch_size",   type=int,   default=cfg.BATCH_SIZE)
    parser.add_argument("--epi_hidden",    type=int,   default=cfg.EPI_HIDDEN)
    parser.add_argument("--fusion_hidden", type=int,   default=cfg.FUSION_HIDDEN)
    parser.add_argument("--lr_backbone",  type=float, default=cfg.LR_BACKBONE)
    parser.add_argument("--lr_head",      type=float, default=cfg.LR_HEAD)
    parser.add_argument("--benign_ratio", type=int,   default=cfg.BENIGN_RATIO)
    parser.add_argument("--seeds",        type=int,   nargs="+", default=cfg.SEEDS)
    parser.add_argument("--single_seed",  type=int,   default=None)
    parser.add_argument("--no_fp16",      action="store_true")
    parser.add_argument("--verify_only",  action="store_true")

    args = parser.parse_args()

    if args.verify_only:
        verify_data(Path(args.epi_dir))
        return

    seeds = [args.single_seed] if args.single_seed is not None else args.seeds

    all_results = []
    for seed in seeds:
        result = run_single_seed(seed, args)
        all_results.append(result)

    # 멀티 시드 요약
    if len(all_results) > 1:
        auprcs = [r["macro_auprc"] for r in all_results if not np.isnan(r["macro_auprc"])]
        print(f"\n{'='*60}")
        print(f"  [Summary] Late Fusion | Seeds={seeds}")
        print(f"  Macro AUPRC: {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}")
        print(f"{'='*60}")

        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "summary.json", "w") as f:
            json.dump({
                "seeds": seeds,
                "mean_auprc": float(np.mean(auprcs)),
                "std_auprc":  float(np.std(auprcs)),
                "all_results": all_results,
            }, f, indent=2)


if __name__ == "__main__":
    main()
