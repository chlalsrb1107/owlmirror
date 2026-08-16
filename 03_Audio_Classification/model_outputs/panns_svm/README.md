# PANNs Cnn14 + SVM 모델 체크포인트 (2026-08-17 — 현재 채택 모델)

`best.pt`, `embeddings.pt`, `embeddings_v3.pt`, `audioset_outputs.pt`, `audioset_outputs_views.pt`는
GitHub 파일 크기 제한(100MB, `embeddings_v3.pt`가 약 95MB로 근접)과 저장소 용량 문제로 이 저장소에는
포함하지 않았습니다 (`.gitignore`의 `*.pt` 처리됨). `Cnn14_16k_mAP=0.438.pth`(PANNs 공식 사전학습
가중치, 약 340MB)도 마찬가지로 미포함.

이 모델은 팀원이 학습해 프로젝트 루트에 업로드한 것을 정리한 것이다. 문서(README/현재_상태_요약 등)에
한때 "AST"로 기록돼 있었으나, 실제 체크포인트 텐서를 열어 확인한 결과 **AST가 아니라 PANNs Cnn14
임베딩 + SVM 분류기** 조합이었다 — 관련 정정 내역은 `00_Overview/현재_상태_요약.md`의 "2026-08-17 추가
정정" 참고.

## 복원 방법

1. `best.pt`(필수) — 재학습 담당 팀원에게 원본을 받아 이 폴더에 둔다.
2. `Cnn14_16k_mAP=0.438.pth`(필수, PANNs 공식 배포본) — `realtime_classify.py`가 최초 실행 시
   Zenodo에서 자동 다운로드를 시도한다 (`https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1`,
   약 340MB, 인터넷 필요). 수동으로 받아 이 폴더에 둬도 된다.
3. `embeddings.pt`, `embeddings_v3.pt`, `audioset_outputs.pt`, `audioset_outputs_views.pt`는
   **실시간 추론에는 불필요** — 학습/threshold 계산 과정의 중간 산출물로 보임 (아래 "이 폴더에 있는 것" 참고).
   재현 실험이 필요할 때만 복원.
4. `03_Audio_Classification/code/realtime_classify.py`는 기본적으로 이 폴더의 `best.pt`,
   `Cnn14_16k_mAP=0.438.pth`를 찾는다. 다른 경로에 두려면 `--svm-path`, `--panns-checkpoint`로 지정.

## 이 폴더에 있는 것 (모두 gitignore됨, 로컬 전용)

| 파일 | 내용 |
|---|---|
| `best.pt` | `{classifier, classes, panns_checkpoint, C, semantic_thresholds, val_accuracy, val_macro_recall, val_confusion_matrix, test_accuracy, test_macro_recall, test_confusion_matrix}` 딕셔너리. `classifier`는 sklearn `Pipeline(StandardScaler, SVC(C=3.0, class_weight='balanced'))`, 입력은 PANNs Cnn14 2048차원 임베딩 1개 |
| `Cnn14_16k_mAP=0.438.pth` | PANNs 공식 Cnn14(16kHz 입력 변형) 사전학습 가중치. `best.pt`의 `panns_checkpoint` 필드(`pretrained\Cnn14_16k_mAP=0.438.pth`, Windows 경로 — 학습이 Windows 환경에서 진행됐음을 시사)가 가리키는 파일과 동일 |
| `embeddings.pt`, `embeddings_v3.pt` | train/val/test 스플릿별 PANNs 임베딩. shape이 각각 `(N, 2, 2048)`, `(N, 3, 2048)` — 샘플당 2개/3개 view(멀티크롭으로 추정)의 임베딩을 담고 있음. `best.pt`의 SVM은 `n_features_in_=2048`이라 실제 학습에는 view 중 하나 또는 평균 등으로 축약된 단일 2048차원이 쓰였을 것으로 보이나, 축약 방식은 이 파일만으로는 알 수 없음 |
| `audioset_outputs.pt`, `audioset_outputs_views.pt` | PANNs의 원래 AudioSet 527클래스 sigmoid 출력(val/test, `audioset_outputs.pt`는 단일 view, `_views`는 2-view). `best.pt`의 `semantic_thresholds`(클래스 인덱스 0,2,3,4에 대한 값, 1=children_playing은 없음) 계산에 쓰인 것으로 추정 |

## ⚠️ 알려진 제한사항 / 재현 안 된 부분

- **`semantic_thresholds`를 `realtime_classify.py`가 적용하지 않음**: 값(예: car_horn=0.0040,
  engine_idling=0.0522, siren=0.1197, motorcycle=0.0116)의 스케일로 볼 때 PANNs의 AudioSet
  527클래스 sigmoid 출력 중 의미상 관련된 클래스(예: siren → AudioSet에 "Siren"/"Police car
  (siren)"/"Ambulance (siren)"/"Fire engine, fire truck (siren)" 등 후보가 5개나 있음)를 게이팅하는
  값으로 추정되지만, **어느 인덱스에 매핑되는지 정하는 학습/threshold 계산 스크립트가 저장소에도
  전달받은 산출물에도 없어 정확한 재현이 불가능함**. 학습 담당 팀원에게 해당 스크립트를 요청해
  확보되면 이 문서와 `realtime_classify.py`를 갱신할 것.
- **SVC가 `probability=False`로 학습됨**: 확률(%)이 아니라 `decision_function` 마진 점수만 얻을 수
  있음 — `realtime_classify.py` 콘솔 출력이 `confidence %` 대신 `score`로 바뀐 이유.
- **평가지표(val 98.0%/test 89.5%)가 단일 임베딩 기준인지 멀티뷰 평균 기준인지 불명확**:
  `embeddings.pt`/`embeddings_v3.pt`의 존재로 볼 때 학습 시 멀티뷰를 실험했을 가능성이 있음.
  `realtime_classify.py`는 마이크 캡처 윈도우 1개당 PANNs 순전파 1회(단일 임베딩)만 수행 —
  멀티뷰 평균을 썼다면 실측 정확도가 체크포인트 기록치보다 낮게 나올 수 있음.

`car_horn/children_playing/engine_idling/siren/motorcycle` 클래스와 최종 타겟(`car_horn`/
`emergency_siren`/`motorcycle_exhaust`/`background`)의 대응 관계는 `00_Overview/현재_상태_요약.md`
"음향 분류 모델 현재 상태" 절에 정리됨 (`siren`↔`emergency_siren`, `motorcycle`↔`motorcycle_exhaust`,
`children_playing`/`engine_idling`은 배경음 역할).

## 관련 노트

- `00_Overview/현재_상태_요약.md` — 음향 분류 모델 현재 상태
- `03_Audio_Classification/모델_아키텍처.md` — 아키텍처 상세
- `03_Audio_Classification/code/realtime_classify.py` — 실제 구현 코드
