# 올빼미러 — AI 기반 실시간 외부 음향 인식 및 운전자 시각화 보조 시스템

> ReSpeaker 4 Mic Array v2.0 + Jetson Orin Nano(오디오 전담) + 카메라 4대·LiDAR VLP-16(노트북, ROS2)로 사이렌·경적·오토바이 배기음을 감지하고 방향·거리를 계산해 BEV 지도로 보여주는 한이음 공모전 프로젝트.

---

## ⚠️ 새 세션에서는 이 순서로 읽을 것

1. **`00_Overview/현재_상태_요약.md`** — 확정 하드웨어, 아키텍처, 모델 진행상황 등 항상 최신 기준. **가장 먼저 읽을 것.**
2. 이 README — 노트 구조 파악용
3. 필요한 개별 노트 — 단, 아래 "구버전 표기 남아있음" 문서는 세부 사양이 `현재_상태_요약.md`와 다를 수 있으니 그 기준으로 교차 확인

---

## 확정 아키텍처 요약

```
[Jetson Orin Nano]                              [노트북 (ASUS TUF A16, Ubuntu, ROS2)]
  ReSpeaker 4 Mic Array v2.0                        카메라 4대 (전/후/좌/우 90°)
  → 소리 분류 (PANNs Cnn14 임베딩+SVM, 5클래스)        LiDAR VLP-16
  → 방향(DoA) 추정                                    ↓
  ROS 미사용, 소켓/JSON으로 전송 ──이더넷 직결──▶     ROS2 브릿지 노드가 수신 → 토픽 발행
                                                     → YOLO 경광등 검출 + LiDAR 거리매칭
                                                     → BEV 지도 UI로 표시
```

자세한 내용: `00_Overview/현재_상태_요약.md`

---

## 프로젝트 노트 구조

```
올빼미러/
├── 00_Overview/
│   ├── 현재_상태_요약.md         ← ★ 항상 최신 기준, 새 세션은 여기부터
│   ├── 프로젝트_개요.md          ← 전체 목적·구성 요약
│   ├── 파이프라인_다이어그램.md  ← 레이어별 데이터 흐름 + 타이밍
│   ├── 중간평가_전략.md          ← 중간평가/중간보고 결과 및 이후 전략
│   ├── 2026-07-05_회의록.md, 2026-07-08_풍절음_실험_계획.md ← 회의록/실험계획
│   └── 보고자료/                 ← 중간보고 PPT, 중간평가 PDF/HTML 보고서
│
├── 01_Hardware/
│   ├── ReSpeaker_설정.md         ← 장치 연결, 드라이버, 차량 설치
│   ├── Jetson_Orin_Nano_환경.md  ← JetPack, PyTorch, TensorRT 설정
│   ├── 카메라_디스플레이_연결.md ← 카메라 4대 + LiDAR VLP-16 장착, ROS2 노드 구성
│   └── code/probe_depth_range.py ← (기록용) 폐기된 Depth 카메라 검증 스크립트
│
├── 02_Data_Collection/
│   └── 데이터셋_수집_계획.md     ← 공개 데이터셋, 오토바이 배기음 직접 녹음 계획
│
├── 03_Audio_Classification/
│   ├── 전처리_파이프라인.md      ← 오디오 전처리, 실시간 스트리밍
│   ├── 모델_아키텍처.md          ← PANNs Cnn14 임베딩 + SVM, 5클래스 분류
│   ├── 실시간_추론_데모.md       ← 실시간 추론 파이프라인 검증 기록
│   ├── model_outputs/            ← 학습 결과(정확도, 로그) — 가중치(.pt/.pth)는 용량 문제로 미포함
│   └── code/realtime_classify.py 등 ← ReSpeaker 캡처 → PANNs 임베딩 → SVM 추론 실행 코드
│
├── 04_Sound_Localization/
│   ├── TDoA_원리.md              ← 기하학적 원리, 음속 보정
│   ├── GCC_PHAT_구현.md          ← TDoA 추정 코드, 성능 특성
│   ├── 빔포밍_알고리즘.md        ← SRP-PHAT 빔포밍
│   └── code/gcc_phat.py, validate_doa.py ← GCC-PHAT DoA 구현·검증 코드
│
├── 05_Camera_Tracking/
│   └── DoA_카메라_매핑.md        ← 소리 방향 → 고정 카메라 4대 중 선택 로직
│
├── 06_Display_Integration/
│   ├── ui_state_spec.md          ← BEV UI 상태·알림 사양 (주 화면 설계)
│   ├── bev_mockup.html           ← 인터랙티브 BEV 목업 (브라우저로 열람)
│   └── 영상_파이프라인.md        ← 카메라 영상을 BEV UI에 보조 화면(PIP)으로 통합하는 방식
│
├── 07_System_Integration/
│   └── 전체_시스템_통합.md       ← Jetson/노트북 분리 구조, ROS2 노드·토픽 구성
│
└── 08_Video_Demo/                ← ⚠️ 최종 구현과 거의 동일(카메라 4대+LiDAR), 방향추정만 펌웨어 DoA로 대체한 9/8 한이음 영상 제출 전용 구현 (2026-08-31 갱신)
    ├── README.md                 ← 카메라 4대 + LiDAR 거리매칭 + 펌웨어 DoA 구성, 파이프라인, 미완료 항목
    ├── code/doa_camera_select.py, lidar_distance_match.py, live_demo.py ← 펌웨어 DoA→카메라 매핑, LiDAR 거리매칭(신규·미검증), 분류+DoA+Detection+LiDAR 연동 오케스트레이터
    └── model_outputs/yolo26m_v4_cls05/ ← 구급차·오토바이 시각 Detection 모델(YOLO, 팀원 학습, 가중치 미포함)
```

---

## 시작 순서

1. `00_Overview/현재_상태_요약.md` — 현재 확정 사항 파악
2. `01_Hardware/ReSpeaker_설정.md`, `Jetson_Orin_Nano_환경.md` — Jetson 개발 환경 구성
3. `02_Data_Collection/데이터셋_수집_계획.md` — 학습 데이터 준비
4. `03_Audio_Classification/모델_아키텍처.md` — PANNs Cnn14+SVM 모델 구조·학습
5. `04_Sound_Localization/GCC_PHAT_구현.md` — 방향 추정 구현
6. `01_Hardware/카메라_디스플레이_연결.md` — 카메라 4대 + LiDAR 노트북 측 구성
7. `06_Display_Integration/ui_state_spec.md`, `bev_mockup.html` — BEV UI 사양·목업
8. `07_System_Integration/전체_시스템_통합.md` — Jetson↔노트북 통합 실행
