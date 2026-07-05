# ReSpeaker 차량 위험음 감지 및 추적 시스템

> ReSpeaker 6-Mic Array + Jetson Orin Nano 기반 실시간 차량 위험음 감지 및 카메라 추적 시스템

---

## 프로젝트 노트 구조

```
ReSpeaker-Vehicle-Alert-System/
├── 00_Overview/
│   ├── 프로젝트_개요          ← 전체 목적·구성 요약
│   ├── 파이프라인_다이어그램  ← 레이어별 데이터 흐름 + 타이밍
│   └── 중간평가_전략          ← 7/14 마감 타임라인 + 제출 체크리스트
│
├── 01_Hardware/
│   ├── ReSpeaker_설정         ← 장치 연결, 드라이버, 차량 설치
│   ├── Jetson_Orin_Nano_환경  ← JetPack, PyTorch, TRT 설정
│   └── 카메라_디스플레이_연결 ← 카메라 선택, GStreamer, HDMI 출력
│
├── 02_Data_Collection/
│   └── 데이터셋_수집_계획     ← 공개 데이터셋, 현장 수집, 증강
│
├── 03_Audio_Classification/
│   ├── 전처리_파이프라인      ← Mel-Spectrogram, 실시간 스트리밍
│   └── 모델_아키텍처          ← MobileNet-V3, 학습, TRT 변환
│
├── 04_Sound_Localization/
│   ├── TDoA_원리              ← 기하학적 원리, 음속 보정
│   ├── GCC_PHAT_구현          ← TDoA 추정 코드, 성능 특성
│   └── 빔포밍_알고리즘        ← SRP-PHAT, 칼만 필터 추적
│
├── 05_Camera_Tracking/
│   └── DoA_카메라_매핑        ← 어안 ROI 매핑, VISCA PTZ 제어
│
├── 06_Display_Integration/
│   └── 영상_파이프라인        ← GStreamer, HUD 렌더링, HDMI 출력
│
└── 07_System_Integration/
    └── 전체_시스템_통합       ← 멀티스레드 메인 루프, 부팅 자동화
```

---

## 시작 순서

1. [[01_Hardware/Jetson_Orin_Nano_환경]] — 개발 환경 구성
2. [[01_Hardware/ReSpeaker_설정]] — 마이크 어레이 연결
3. [[02_Data_Collection/데이터셋_수집_계획]] — 학습 데이터 준비
4. [[03_Audio_Classification/모델_아키텍처]] — 모델 학습 및 TRT 변환
5. [[04_Sound_Localization/GCC_PHAT_구현]] — 위치 추정 구현
6. [[07_System_Integration/전체_시스템_통합]] — 통합 실행
