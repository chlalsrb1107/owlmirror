# AST 모델 체크포인트

`best.pt`, `last.pt`는 각각 345MB라 GitHub 파일 크기 제한(100MB)을 넘어 이 저장소에는 포함하지 않았습니다 (`.gitignore` 처리됨).

## 복원 방법

1. `outputs_ast.zip`을 원본 출처(학습 환경/Colab/Drive)에서 다시 받는다.
2. 압축을 풀어 `best.pt`(필수), `last.pt`(선택)를 이 폴더에 둔다.
3. `03_Audio_Classification/code/realtime_classify.py`는 기본적으로 `../model_outputs/outputs_ast/best.pt`를 찾는다. 다른 경로에 두려면 `--model-path`로 지정.

## 이 폴더에 있는 것 (버전관리 됨)

- `summary.json` — 검증 정확도(91.0%), 샘플 수
- `train_log.csv` — epoch별 train/val loss·accuracy
- `result.csv` — 검증셋 837개 샘플의 실제/예측 클래스 및 confidence
- `embeddings.csv`는 7MB로 크지 않지만 재현에 필수는 아니라 제외함 — 필요하면 `outputs_ast.zip`에서 복원
