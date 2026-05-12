import torch
from pathlib import Path

class Config:
    # 데이터 경로
    DATA_DIR    = "sequences_extracted"
    BRAIN_FILE  = "pathogenic_brain_with_seq.tsv"
    HEART_FILE  = "pathogenic_heart_with_seq.tsv"
    LIVER_FILE  = "pathogenic_liver_with_seq.tsv"
    BENIGN_FILE = "benign_with_seq.tsv"

    # 조직 ID 매핑
    TISSUE_MAP = {0: "liver", 1: "heart", 2: "brain"}
    BENIGN_TID = 3

    # 데이터 샘플링 및 분할
    BENIGN_RATIO   = 3
    BENIGN_CHUNK   = 100_000
    BENIGN_APPROX  = 1_281_188
    BENIGN_SEED    = 42
    VAL_RATIO      = 0.10
    TEST_RATIO     = 0.10

    # 모델 설정
    MODEL_NAME  = "zhihan1996/DNABERT-2-117M"
    MAX_LENGTH  = 256  # 분석 결과에 따른 최적화값
    HIDDEN_DIM  = 768
    DROPOUT     = 0.1

    # 학습 하이퍼파라미터
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

    # 실행 설정
    SEEDS       = [42, 123, 456]
    OUTPUT_DIR  = "outputs/stage1_baseline"
    NUM_WORKERS = 4