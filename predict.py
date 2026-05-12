import torch
from transformers import AutoTokenizer
from src.config import Config
from src.model import DNABERT2Baseline


def predict_pathogenicity(sequence, model_path, device='cuda'):
    cfg = Config()

    # 1. 모델 및 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True)
    model = DNABERT2Baseline(cfg.MODEL_NAME, cfg.HIDDEN_DIM, cfg.DROPOUT)

    # 저장된 가중치 불러오기
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()

    # 2. 서열 전처리 및 토큰화
    sequence = sequence.upper().strip()
    inputs = tokenizer(
        sequence,
        max_length=cfg.MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # 3. 예측 (Inference)
    with torch.no_grad():
        ids = inputs["input_ids"].to(device)
        mask = inputs["attention_mask"].to(device)
        logits = model(ids, mask)
        probability = torch.sigmoid(logits).item()  # 0 ~ 1 사이 확률로 변환

    return probability


if __name__ == "__main__":
    # 테스트하고 싶은 모델 경로
    MODEL_PATH = "outputs/stage1_baseline/seed_42/best_model.pt"

    # 테스트하고 싶은 실제 DNA 서열 (예시: ClinVar 등에서 가져온 서열)
    my_sequence = "GGCCACTGCCATCTTTCTTGCGGGCGGGGGCGGTGCGAACGGGCGCGACCTCACGGAGGGGACGCCGGCGCCACCATCTCTCCTCCGGGCGGAAGCGGTCGCGGGGCCGCTCCGAGGTTGACCAATGACAAGGGTGCCCGAGGCCACGTGACGGCCGCCGATTGGCCGCCGGCCTCCGAGCGCCCCGGGGCTCGGCGTCTGCGGAAGGCCCCGGCGCGCTCCCAGGAGCGCCGTGCGCACGCGCACCGCCCCGAGCCGGCGGCGCCTGCGCACTCGCGAGTCCGGCCTGGGCCGCCGGCCCGGCGCGGGCGCCATGAAGCTGCTGCGGCGGGCGTGGCGGCGGCGGGCGGCGCTAGGCCTGGGCACGCTGGCGCTGTGCGGGGCGGCGCTGCTCTACCTGGCGCGCTGCGCGGCCGAGCCCGGGGACCCCAGGGCGATGTCGGGCCGCAGCCCGCCTCCCCCCGCGCCCGCGCGCGCCGCCGCCTTCCTGGCAGTGCTGGTGGCCAGCGCGCTCCGCGCCGCCGAGCGCCGCAGCGTGATCCGCAGCACGTGGCTTGCGCGGCGCGGGGCCCCGGGCGACGTGTGGGCGCGCTTTGCCGTGGGCACGGCCGGCCTGGGCGCCGAGGAGCGGCGCGCCCTGGAGCGGGAGCAGGCGCGGCACGGGGACCTGCTGCTGCTGCCCGCGCTGCGCGACGCCTACGAAAACCTCACGGCCAAGGTGCTGGCCATGCTGGCCTGGCTGGACGAGCACGTGGCCTTCGAGTTCGTGCTCAAGGCGGACGACGACTCCTTCGCGCGGCTGGACGCGCTGCTGGCCGAGCTGCGCGCCCGCGAGCCCGCGCGCCGCCGCCGCCTCTACTGGGGCTTCTTCTCGGGCCGCGGCCGCGTCAAGCCGGGGGGGCGCTGGCGCGAGGCCGCCTGGCAACTCTGCGACTACTACCTGCCCTACGCGCTGGGCGGCGGCTACGTGCTCTCGGCCGACCTGGTGCACTACCTGCGCCTCAGCCGCGACTACCTGCGCGCCT"  # 여기에 실제 서열을 입력하세요 (약 1000bp 권장)

    # 실행 환경 설정 (GPU가 없으면 'cpu')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    prob = predict_pathogenicity(my_sequence, MODEL_PATH, device=dev)

    print(f"\n[분석 결과]")
    print(f"입력 서열: {my_sequence[:50]}... (총 {len(my_sequence)}bp)")
    print(f"병원성 확률: {prob:.4f}")

    if prob >= 0.5:
        print("결과: ⚠️ 병원성(Pathogenic) 가능성이 높습니다.")
    else:
        print("결과: ✅ 양성(Benign) 가능성이 높습니다.")