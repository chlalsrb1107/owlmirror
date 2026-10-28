# yolo_detection — 구급차·오토바이 시각 Detection 모델 학습 결과

`08_Video_Demo/code/live_demo.py`가 쓰는 YOLO 모델의 **학습 산출물**이다.
사이렌·오토바이가 감지되면 해당 방향 카메라 영상에서 대상을 찾는 데 사용한다.

## 가중치는 여기 없다

`best.pt` / `last.pt`(각 44MB)는 `.gitignore`의 `*.pt` 규칙으로 제외돼 있다.
GitHub 100MB 제한과 저장소 비대화를 피하기 위한 것이며, **팀원끼리 별도로 공유**한다.

실행하려면 가중치를 아래 경로에 두어야 한다:

```
08_Video_Demo/model_outputs/yolo26m_v4_cls05/weights/best.pt
```

없으면 `live_demo.py`는 Detection만 끄고 나머지(오디오·LiDAR·화면)는 정상 동작한다.

## 클래스

`confusion_matrix.png` 기준 **Ambulance / Motorcycle** 2종. val mAP50 0.919.

## 들어있는 것

| 파일 | 내용 |
|---|---|
| `results.csv`, `results.png` | 학습 곡선(loss·mAP 추이) |
| `confusion_matrix*.png` | 혼동행렬 — 클래스명 확인용 |
| `Box*_curve.png` | Precision·Recall·F1·PR 곡선 |
| `train_batch*.jpg`, `val_batch*.jpg` | 학습·검증 배치 샘플 (라벨 vs 예측) |
| `args.yaml` | 학습 하이퍼파라미터 |

## 오검출 차단 장치

2026-09-02 실내 테스트에서 실험실 책상이 "Motorcycle 48%"로 잡힌 적이 있다. 이를 계기로
`live_demo.py`에 두 겹의 거름망을 넣었다.

1. **신뢰도 임계값** `DETECTION_MIN_CONF = 0.55` — 확신이 낮은 박스는 쓰지 않는다
2. **소리↔라벨 대조** `DETECTION_EXPECT` — 사이렌이 들렸으면 Ambulance만, 오토바이
   배기음이면 Motorcycle만 받는다. 소리와 어긋나는 검출은 버린다

둘째가 특히 중요하다. 이 시스템에서 YOLO는 "화면에 무엇이 있나"가 아니라 "그 소리를 낸 것이
어디 있나"를 찾는 역할이므로, 소리와 맞지 않는 박스는 애초에 쓸 이유가 없다.
