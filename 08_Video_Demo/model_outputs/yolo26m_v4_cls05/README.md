# 구급차·오토바이 시각 Detection 모델 (YOLO, 2026-08-25 — 9/8 데모 채택)

팀원이 학습해 저장소 루트에 올린 걸 `08_Video_Demo/model_outputs/`로 정리한 것. YOLO(`yolo26m` 베이스,
`args.yaml` 기준) 커스텀 학습, 대상 데이터셋은 `ambulance_motorcycle_yolo_v4`(팀원 로컬 경로,
이 저장소에는 없음).

## 클래스

`confusion_matrix.png` 기준 2클래스: **Motorcycle**, **Ambulance** (background는 클래스가 아니라
YOLO confusion matrix가 오탐/미탐을 표시하는 행/열).

## 학습 결과 (`results.csv`, 180 epoch 완료)

| 지표 | 값 |
|---|---|
| precision | 0.907 |
| recall | 0.845 |
| mAP50 | 0.919 |
| mAP50-95 | 0.755 |

## 복원 방법

`weights/best.pt`, `weights/last.pt`는 `.gitignore`의 `*.pt` 처리로 이 저장소에는 없다(각 43MB,
100MB 제한 자체는 안 넘지만 기존 정책과 통일). 팀원에게 원본을 받아 `weights/` 폴더에 두면
`08_Video_Demo/code/live_demo.py`의 `load_detector()`가 그대로 찾는다.

## 사용

```
pip install ultralytics
```

`live_demo.py`가 `ultralytics.YOLO(best.pt)`로 로드해 사이렌/오토바이 소리 감지 시 선택된 카메라
프레임에서 추론한다. 단독으로 테스트하려면:

```python
from ultralytics import YOLO
model = YOLO("weights/best.pt")
results = model.predict("test.jpg")
```

## 미확인

- `args.yaml`의 `model: yolo26m.pt` — Ultralytics 표준 배포판에 없는 이름이면 학습 시 쓴 것과
  동일한 `ultralytics` 버전/커스텀 소스가 필요할 수 있음. 로드 실패하면 팀원에게 학습 환경(패키지
  버전) 확인 요청할 것.
- 실제 차량 카메라 화각·거리에서 정확도 검증 안 됨 (위 지표는 팀원 학습 데이터셋 기준 val 결과)
