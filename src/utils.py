import random, json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

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

def compute_metrics(labels, probs, tissues, cfg) -> dict:
    metrics, auprc_list = {}, []
    benign_msk = (tissues == cfg.BENIGN_TID)
    for tid, tname in cfg.TISSUE_MAP.items():
        msk = (tissues == tid) | benign_msk
        if msk.sum() == 0 or labels[msk].sum() == 0:
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

def save_predictions(labels, probs, tissues, path):
    pd.DataFrame({
        "label": labels.tolist(), "prob": probs.tolist(),
        "tissue_id": tissues.tolist(),
    }).to_csv(path, index=False)