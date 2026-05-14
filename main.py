import argparse
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

from src.config import Config
from src.utils import set_seed, get_device, print_metrics, compute_metrics, save_predictions
from src.dataset import prepare_data, stratified_split, VariantDataset, make_loader
from src.model import DNABERT2Baseline
from src.trainer import run_train, run_eval


def parse_args():
    p = argparse.ArgumentParser(description="DNABERT-2 Stage 1 Training & Evaluation")
    p.add_argument("--mode", choices=["train", "val", "test", "all"], default="all",
                   help="실행 모드 설정 (train, val, test, all)")
    p.add_argument("--single_seed", type=int, default=None,
                   help="특정 시드 하나만 실행하고 싶을 때 번호 입력")
    p.add_argument("--resume", action="store_true",
                   help="중단된 학습(latest_checkpoint.pt)이 있다면 이어서 시작")
    return p.parse_args()


def main():
    # 1. 초기 설정 및 환경 준비
    args = parse_args()
    cfg = Config()

    # 데이터 로딩을 위한 시드 고정 (Benign 샘플링 일관성 유지)
    set_seed(cfg.BENIGN_SEED)
    device, use_fp16 = get_device(cfg.FP16)

    # 2. 전체 데이터 로드 (원본 인덱스 _orig_idx 포함)
    df = prepare_data(cfg)

    # 실행할 시드 목록 결정
    seeds = [args.single_seed] if args.single_seed else cfg.SEEDS

    # 3. 시드별 루프 시작
    for seed in seeds:
        print(f"\n" + "═" * 60)
        print(f"  🚀 [SEED {seed}] 프로세스 시작")
        print("═" * 60)

        # 모델 및 데이터 분할을 위한 시드 재설정
        set_seed(seed)

        # 데이터 분할 (Train/Val/Test)
        train_df, val_df, test_df = stratified_split(df, cfg, seed)

        # 모델 초기화
        model = DNABERT2Baseline(cfg.MODEL_NAME, cfg.HIDDEN_DIM, cfg.DROPOUT).to(device)

        # [TRAIN / ALL 모드]
        if args.mode in ("train", "all"):
            print(f"\n[학습 단계]")
            # run_train에 test_df를 함께 전달하여 splits 인덱스를 저장하도록 함
            run_train(cfg, train_df, val_df, test_df, model, device, seed, resume=args.resume)

        # [VAL 모드]
        if args.mode == "val":
            print(f"\n[검증 단계]")
            # 저장된 모델을 불러와서 검증 데이터로 성적 산출
            run_eval(cfg, val_df, model, device, seed, split="val")

        # [TEST / ALL 모드]
        if args.mode in ("test", "all"):
            print(f"\n[최종 테스트 단계]")
            # 저장된 모델을 불러와서 테스트 데이터로 성적 산출
            run_eval(cfg, test_df, model, device, seed, split="test")

    print("\n" + "═" * 60)
    print("  ✅ 모든 시드 및 모드 실행 완료")
    print("═" * 60)


if __name__ == "__main__":
    main()