import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split


class VariantDataset(Dataset):
    def __init__(self, df, tokenizer, max_length: int):
        self.sequences = df["sequence"].tolist()
        self.labels = df["label"].tolist()
        self.tissue_ids = df["tissue_id"].tolist()
        self.tokenizer = tokenizer
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
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
            "tissue_id": torch.tensor(self.tissue_ids[idx], dtype=torch.long),
        }


def load_pathogenic(cfg) -> pd.DataFrame:
    dfs = []
    for tname, fname in [("brain", cfg.BRAIN_FILE),
                         ("heart", cfg.HEART_FILE),
                         ("liver", cfg.LIVER_FILE)]:
        fpath = Path(cfg.DATA_DIR) / fname
        if fpath.exists():
            df = pd.read_csv(fpath, sep="\t")
            print(f"  Pathogenic {tname:5s}: {len(df):>8,} rows")
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_benign_sampled(cfg, n_target: int) -> pd.DataFrame:
    fpath = Path(cfg.DATA_DIR) / cfg.BENIGN_FILE
    rng = np.random.RandomState(cfg.BENIGN_SEED)
    keep_prob = min(1.0, (n_target * 1.1) / cfg.BENIGN_APPROX)
    chunks = []
    print(f"  Benign 로드 중 (target={n_target:,})...")
    for chunk in pd.read_csv(fpath, sep="\t", chunksize=cfg.BENIGN_CHUNK):
        keep_n = max(1, int(len(chunk) * keep_prob))
        sampled = chunk.sample(n=keep_n, random_state=rng.randint(0, 2 ** 31))
        chunks.append(sampled)
    benign = pd.concat(chunks, ignore_index=True)
    if len(benign) > n_target:
        benign = benign.sample(n=n_target, random_state=cfg.BENIGN_SEED)
    print(f"  Benign 완료: {len(benign):,} rows")
    return benign


def prepare_data(cfg):
    print("\n─── 데이터 로드 ────────────────────────────────────────")
    patho = load_pathogenic(cfg)
    n_benign = len(patho) * cfg.BENIGN_RATIO
    benign = load_benign_sampled(cfg, n_benign)
    df = pd.concat([patho, benign], ignore_index=True)
    df = df.dropna(subset=["sequence", "label"])
    df["label"] = df["label"].astype(int)
    df["tissue_id"] = df["tissue_id"].astype(int)
    df["sequence"] = df["sequence"].str.upper().str.strip()
    seq_len = df["sequence"].str.len()
    df = df[(seq_len >= 600) & (seq_len <= 1100)].reset_index(drop=True)
    df["_orig_idx"] = np.arange(len(df))
    return df


def stratified_split(df, cfg, seed):
    strat = df["tissue_id"].astype(str) + "_" + df["label"].astype(str)
    if strat.value_counts().min() < 2:
        strat = df["label"].astype(str)

    train_val, test = train_test_split(df, test_size=cfg.TEST_RATIO, stratify=strat, random_state=seed)

    strat_tv = train_val["tissue_id"].astype(str) + "_" + train_val["label"].astype(str)
    if strat_tv.value_counts().min() < 2:
        strat_tv = train_val["label"].astype(str)

    val_adj = cfg.VAL_RATIO / (1.0 - cfg.TEST_RATIO)
    train, val = train_test_split(train_val, test_size=val_adj, stratify=strat_tv, random_state=seed)

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def make_loader(ds, batch_size, sampler=None, num_workers=4):
    return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())