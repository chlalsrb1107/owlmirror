# 08_Video_Demo — 9/8 한이음 영상 제출용 축소 구현

> ⚠️ 이 폴더는 **9/8 16:00 마감 영상(1분 30초 이내)** 전용 구현이다. 카메라 4대+LiDAR 기준 최종 아키텍처(00~07 폴더)와는 별개이며, 10월 말 최종 시연은 원래 계획대로 진행한다. 배경: `00_Overview/2026-08-25_9.8_영상제출_촬영_계획.md`.

## 이 데모의 축소 구성 (최종 구현과 다른 점)

| 항목 | 최종 구현 (00~07) | 9/8 데모 (이 폴더) |
|---|---|---|
| 카메라 | 4대 (전/후/좌/우) | **3대 (좌/우/후방만, 전방 없음)** |
| 거리·차량 확정 | LiDAR | **없음** (BEV 확정 마커 미사용) |
| 방향추정(DoA) | ReSpeaker raw ch1~4 + 자체 GCC-PHAT(`04_Sound_Localization/code/gcc_phat.py`, 미보정) | **ReSpeaker 온보드 펌웨어 DoA** (USB Tuning 인터페이스) |
| 기본 화면 | BEV 지도 | **후방 카메라 기본 표시**, 좌/우 감지 시 전환, 전방 감지 시 "전방 확인" 텍스트 |

## 파이프라인

```
ReSpeaker 마이크
  ├─ ch0 (AEC) → PANNs Cnn14 임베딩 → SVM 분류 (03_Audio_Classification/code/realtime_classify.py, 기존)
  └─ 펌웨어 DoA (USB Tuning) → 방향(0~359°) (doa_camera_select.py, 신규)

분류 결과가 car_horn/siren/motorcycle 중 하나면
  → DoA로 좌/우/후방 카메라 선택 (전방이면 카메라 대신 "전방 확인" 텍스트)
  → 그 중 siren/motorcycle이면 해당 카메라 영상에서 구급차·오토바이 시각 Detection 실행
     (yolo26m_v4_cls05, model_outputs/ 참고)
```

## 화면 UI (2026-08-25 확정 — BEV 아님)

기존 `06_Display_Integration/bev_mockup.html`은 BEV 지도가 주화면이고 카메라가 작은 PIP였는데,
9/8 데모는 **카메라 영상이 주화면**이고 BEV는 뺀다. 라이다가 없어 거리·다중 차량을 채울 내용이
없는 채로 원형 지도를 띄우면 오히려 어색해서, 방향 정보는 상단 알림 배너(아이콘+텍스트+방향)로만
전달한다. 같은 이유로 **모든 감지는 "주의" 단계까지만** 표시 — 거리를 모르는 채로 "경고"(빨강)를
쓰면 없는 정확도를 있는 척하는 셈이라 `ui_state_spec.md`의 정직성 원칙과 어긋난다.

목업: `camera_ui_mockup.html` (브라우저로 열람, 시나리오 6개: 평상시/경적/사이렌/오토바이 접근/
오토바이 골목(시야 밖)/전방 감지(카메라 없음)).

## 파일

- `camera_ui_mockup.html` — 9/8 데모 주화면(카메라+알림배너) 인터랙티브 목업
- `code/doa_camera_select.py` — 펌웨어 DoA 읽기(`Tuning` 클래스) + 방향→카메라 매핑(`select_camera`). `--selftest`로 하드웨어 없이 매핑 로직 검증 가능, `--live`는 실제 ReSpeaker 필요.
- `code/live_demo.py` — 실제 라이브 데모 앱. 좌/우/후방 카메라 3대를 시작할 때 전부 미리 열어두고(전환 지연 없음), 오디오 분류+DoA는 백그라운드 스레드, 카메라 표시는 메인 스레드(OpenCV 창)로 분리해서 소리 감지 시 방향에 맞는 카메라+배너(+사이렌/오토바이는 Detection 박스)로 3초간 전환했다가 후방으로 자동 복귀한다(`ui_state_spec.md`의 "3초 유지" 원칙). 레이아웃은 `camera_ui_mockup.html`을 따름. **카메라 3대는 일반 UVC 웹캠이 아니라 Daheng Imaging MER2-240-159U3C(USB3 Vision 산업용 카메라)**라 `cv2.VideoCapture`가 아니라 Daheng 공식 SDK `gxipy`로 프레임을 읽는다. **실제 시리얼번호는 자리표시값**(아래 참고). 한글 텍스트는 `cv2.putText`가 못 그려서 Pillow+한글 폰트로 그림 — `FONT_PATH`가 실제 노트북 폰트 경로를 가리키는지 확인 필요(우분투는 `sudo apt install fonts-nanum`).
- `model_outputs/yolo26m_v4_cls05/` — 팀원이 학습한 구급차(Ambulance)/오토바이(Motorcycle) YOLO 모델(2026-08-25 저장소 루트에 업로드된 것 정리). 가중치(`.pt`)는 gitignore 처리, 상세는 그 폴더의 README 참고.
- `calibration/cam1_calib`~`cam4_calib/` — 카메라 4대 각각의 내부 파라미터(camera_matrix, distortion_coefficients) 캘리브레이션 결과(2026-08-25 저장소 루트에 업로드된 것 정리). **방향 매핑 확정: cam1=좌, cam2=우, cam3=후방** (`live_demo.py`의 `CAMERA_CALIB_ID`). cam4는 10월 최종(4대) 구성의 전방용으로 남겨둠 — 설치 시 각 방향에 해당 번호로 캘리브레이션된 물리 카메라를 붙이면 됨.

## 아직 안 된 것 (팀 확인 필요)

- [ ] `DOAANGLE` 레지스터 id(=21)가 실제 이 보드에서 맞는지 `--live`로 검증 (ReSpeaker 공식 참고값이나 실측 안 됨)
- [ ] `MOUNT_OFFSET_DEG` — 차량에 마이크 장착 후 정면(0°) 기준 보정
- [x] ~~카메라 3대 임시 마운트 방법~~ → 구매·규격·마운트·캘리브레이션 완료 (2026-08-25). 리허설에서 케이블 길이·진동만 재확인 (`00_Overview/2026-08-25_9.8_영상제출_촬영_계획.md` 참고)
- [ ] `live_demo.py`의 `CAMERA_SERIAL`(시리얼번호, 현재 빈 값) — 설치 시 `CAMERA_CALIB_ID`에 맞는 물리 카메라(예: left=cam1 캘리브레이션 개체)의 실제 시리얼번호로 채울 것
- [ ] `gxipy`(Daheng Galaxy SDK) 설치 및 카메라 인식 확인 — 노트북(우분투)에 벤더 드라이버 설치 필요할 수 있음
- [ ] 한글 폰트 설치 확인(`fonts-nanum` 등) 및 `live_demo.py`의 `FONT_PATH`를 실제 경로로 수정
- [ ] `yolo26m_v4_cls05`의 베이스 모델(`yolo26m.pt`)이 표준 `ultralytics` 배포판에 있는 이름인지 확인 — 없으면 학습 팀원의 패키지 버전 확인 필요
