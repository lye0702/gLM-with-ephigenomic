import argparse, json, torch
import numpy as np
from pathlib import Path
from src.config import Config
from src.utils import set_seed, get_device, print_metrics, compute_metrics, save_predictions
from src.dataset import prepare_data, stratified_split, VariantDataset, make_loader
from src.model import DNABERT2Baseline
from src.trainer import run_train, run_inference
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "val", "test", "all"], default="all")
    p.add_argument("--single_seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    set_seed(cfg.BENIGN_SEED)
    device, use_fp16 = get_device(cfg.FP16)

    df = prepare_data(cfg)
    seeds = [args.single_seed] if args.single_seed else cfg.SEEDS

    for seed in seeds:
        set_seed(seed)
        train_df, val_df, test_df = stratified_split(df, cfg, seed)
        model = DNABERT2Baseline(cfg.MODEL_NAME, cfg.HIDDEN_DIM, cfg.DROPOUT).to(device)

        if args.mode in ("train", "all"):
            run_train(cfg, train_df, val_df, model, device, seed)

        if args.mode in ("test", "all"):
            ckpt = torch.load(Path(cfg.OUTPUT_DIR) / f"seed_{seed}" / "best_model.pt")
            model.load_state_dict(ckpt["model_state"])
            tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True)
            test_loader = make_loader(VariantDataset(test_df, tokenizer, cfg.MAX_LENGTH), cfg.BATCH_SIZE * 2,
                                      num_workers=cfg.NUM_WORKERS)
            l, p, t = run_inference(model, test_loader, device, cfg.FP16)
            metrics = compute_metrics(l, p, t, cfg)
            print(f"\n[TEST Seed {seed}] Result:")
            print_metrics(metrics, "  ")
            save_predictions(l, p, t, Path(cfg.OUTPUT_DIR) / f"seed_{seed}" / "test_predictions.csv")


if __name__ == "__main__":
    main()