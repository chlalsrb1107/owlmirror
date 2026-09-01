# 08_Video_Demo — 9/8 한이음 영상 제출용 구현

> ⚠️ (2026-08-31 갱신) 카메라 4대+LiDAR가 촬영일까지 장착·데이터 수신은 가능해져, 8/25에 정했던
> "카메라 3대·라이다 없음" 축소 구성을 되돌리고 최종 아키텍처(00~07 폴더)와 거의 같은 구성으로
> 이 폴더를 재구성했다. 남은 차이는 방향추정(아래 표)뿐이다. 단, **LiDAR 거리 매칭·4카메라 연동
> 소프트웨어는 8/31 기준 전혀 검증되지 않았다** — 배경·구현 체크리스트는
> `00_Overview/2026-08-25_9.8_영상제출_촬영_계획.md`(2026-08-31 갱신)와
> `00_Overview/현재_상태_요약.md`.

## 최종 구현과의 남은 차이

| 항목 | 최종 구현 (00~07) | 9/8 데모 (이 폴더) |
|---|---|---|
| 카메라 | 4대 (전/후/좌/우) | 4대, 동일 |
| 거리·차량 확정 | LiDAR | LiDAR, 동일 — **단 연동 코드가 미검증** |
| 방향추정(DoA) | ReSpeaker raw ch1~4 + 자체 GCC-PHAT(`04_Sound_Localization/code/gcc_phat.py`, 미보정) | **ReSpeaker 온보드 펌웨어 DoA** (USB Tuning 인터페이스) — 보정 불확실성 리스크 회피 목적으로 8/25에 결정, 유지 |
| 기본 화면 | BEV 지도 | BEV(좌)+카메라(우) 반반 — **(2026-09-02) 구현 완료**, 차이 없음 |

## 파이프라인

```
ReSpeaker 마이크                              VLP-16 LiDAR
  ├─ ch0 (AEC) → PANNs Cnn14 임베딩 → SVM 분류    └─ UDP 스트림 → 최신 포인트클라우드
  │  (03_Audio_Classification/code/realtime_classify.py, 기존)   (lidar_distance_match.py, 신규)
  └─ 펌웨어 DoA (USB Tuning) → 방향(0~359°) (doa_camera_select.py)

분류 결과가 car_horn/siren/motorcycle 중 하나면
  → DoA로 전/좌/우/후방 카메라 선택
  → siren/motorcycle이면 해당 카메라 영상에서 구급차·오토바이 시각 Detection 실행
     (yolo26m_v4_cls05, model_outputs/ 참고)
  → siren/motorcycle이면 추가로 그 방향의 LiDAR 거리 매칭 시도(경적은 대상 아님)
     → 매칭 성공 시 "경고"(빨강·거리 표시), 실패 시 "주의"(노랑, 8/25 버전과 동일 폴백)
```

## 화면 UI (2026-09-01 갱신 — 목업만 BEV|카메라 반반으로 복귀)

**목업·설계 문서 범위에서 BEV를 다시 넣는다.** 화면을 세로로 반 나눠 왼쪽은 BEV 원형 지도(LiDAR
포인트 클러스터로 확정/미확정 표현), 오른쪽은 소리 방향에 맞춰 전환되는 카메라 영상을 상시 함께
띄운다 — 8/25~8/31에 "8일 안에 실시간 BEV 렌더링까지 검증하기엔 리스크가 크다"며 카메라 주화면 +
상단 배너로 축소했던 것을 목업에서는 되돌린 것. 확정된 대상(LiDAR 매칭 성공)은 여전히 **"경고"
(빨강, 거리 표시)까지 정직하게 올리고**, 미확정이면 "주의"(노랑)까지만.

**(2026-09-02) `code/live_demo.py`가 이 반반 레이아웃으로 재구현됐다.** 9/1까지는 "목업만 바꾸고
코드는 9/3 go/no-go 이후에 정한다"고 미뤄뒀으나, 최종 결과물에 BEV가 반드시 들어가야 한다는
판단으로 앞당겨 구현했다. 그리기는 `code/bev_render.py`(신규)가 전담하고 `live_demo.py`는 상태만
넘긴다. LiDAR가 죽거나 `LIDAR_AVAILABLE=False`여도 BEV는 계속 그려진다 — 링·자차·부채꼴은 오디오
방향만으로 성립하므로 폴백 경로는 그대로 살아 있고, 거리·클러스터만 빠지고 좌하단에
"LiDAR 미연결 — 방향만 표시"가 뜬다.

목업: `camera_ui_mockup.html` (브라우저로 열람, 시나리오 10개 — 평상시/경적/사이렌 미확정·확정/
오토바이 접근·근접(경고)·사각지대(경고)/오토바이 골목(시야 밖)/전방 카메라/동시 감지).

## 파일

- `camera_ui_mockup.html` — 9/8 데모 UI 목업, **(2026-09-01) BEV(좌)|카메라(우) 반반 레이아웃으로 갱신** — BEV는 LiDAR 포인트 클러스터(확정)·성긴 부채꼴(미확정) 표현 포함, 주의/경고 색 분기 포함. 인터랙티브. **(2026-09-02) `live_demo.py`/`bev_render.py`에 구현 완료** — 목업이 기준 디자인 역할로 남는다
- `code/doa_camera_select.py` — 펌웨어 DoA 읽기(`Tuning` 클래스) + 방향→카메라 매핑(`select_camera`, 전/좌/후/우 4방향). `--selftest`로 하드웨어 없이 매핑 로직 검증 가능, `--live`는 실제 ReSpeaker 필요.
- `code/lidar_distance_match.py` — VLP-16 UDP 스트림을 배경 스레드로 읽어(`LidarScanner`) DoA 방향의 가장 가까운 물체까지 거리를 매칭(`match_distance`). 반경 6m 이내(사각지대)는 거리 대신 "사각지대"만 반환. `--selftest`로 가짜 포인트로 매칭 로직 검증 가능, `--live --theta <deg>`는 실제 VLP-16 필요. **실물로 전혀 검증 안 됨** — 파일 상단 "미검증" 목록 참고.
- `code/alert_policy.py` — **(2026-09-02 신규)** 감지 하나를 화면 상태(주의/경고)로 바꾸는 클래스별 규칙. 경적은 "가장 큰 것 선택 + 같은 방향 반복 시 경고", 사이렌은 "항상 경고", 오토바이는 "라이다 → Detection → 배기음" 순으로 근거를 찾고 소리만으로도 사각지대 경고를 낸다. `--selftest`로 규칙 22개를 하드웨어 없이 검증한다.
- `code/bev_render.py` — **(2026-09-02 신규)** 화면 합성 전담(1600x900, BEV 좌 | 카메라 우). LiDAR 리턴을 그대로 점으로 그리고, 확정 대상은 색 클러스터·미확정은 부채꼴·6m 이내는 사각 구역 점등으로 구분한다. `--selftest`로 하드웨어 없이 시나리오 8종 PNG를 뽑아 레이아웃을 확인할 수 있다. ⚠️ 목업 SVG는 시계+ 각도, 시스템 `theta`는 반시계+ 라 `_screen_angle()`에서 뒤집어 맞춘다 — 목업 코드를 옮길 때 이걸 빠뜨리면 BEV가 좌우 반전된다.
- `code/live_demo.py` — 실제 라이브 데모 앱. 전/좌/우/후방 카메라 4대를 시작할 때 전부 미리 열어두고(전환 지연 없음), 오디오 분류+DoA는 백그라운드 스레드, LiDAR는 자체 배경 스레드(`LidarScanner`), 카메라 표시는 메인 스레드(OpenCV 창)로 분리해서 소리 감지 시 방향에 맞는 카메라+배너(+사이렌/오토바이는 Detection 박스, +LiDAR 거리 매칭 성공 시 "경고" 빨강 테두리)로 3초간 전환했다가 후방으로 자동 복귀한다(`ui_state_spec.md`의 "3초 유지" 원칙). LiDAR 연동이 시작 시 실패하면 `LIDAR_AVAILABLE=False`로 자동 폴백해 모든 감지를 8/25 버전과 동일하게 "주의"로만 표시한다. 레이아웃은 `camera_ui_mockup.html`을 따름. **카메라 4대는 일반 UVC 웹캠이 아니라 Daheng Imaging MER2-240-159U3C(USB3 Vision 산업용 카메라)**라 `cv2.VideoCapture`가 아니라 Daheng 공식 SDK `gxipy`로 프레임을 읽는다. **실제 시리얼번호는 자리표시값**(아래 참고). 한글 텍스트는 `cv2.putText`가 못 그려서 Pillow+한글 폰트로 그림 — `FONT_PATH`가 실제 노트북 폰트 경로를 가리키는지 확인 필요(우분투는 `sudo apt install fonts-nanum`).
- `model_outputs/yolo26m_v4_cls05/` — 팀원이 학습한 구급차(Ambulance)/오토바이(Motorcycle) YOLO 모델(2026-08-25 저장소 루트에 업로드된 것 정리). 가중치(`.pt`)는 gitignore 처리, 상세는 그 폴더의 README 참고.
- `calibration/cam1_calib`~`cam4_calib/` — 카메라 4대 각각의 내부 파라미터(camera_matrix, distortion_coefficients) 캘리브레이션 결과(2026-08-25 저장소 루트에 업로드된 것 정리). **방향 매핑 확정: cam1=좌, cam2=우, cam3=후방, cam4=전방** (`live_demo.py`의 `CAMERA_CALIB_ID`) — 설치 시 각 방향에 해당 번호로 캘리브레이션된 물리 카메라를 붙이면 됨.

## 아직 안 된 것 (팀 확인 필요, 촬영 전 최우선순위 순)

- [ ] **`code/lidar_distance_match.py` 전체가 미검증** — VLP-16 UDP 포트/패킷 포맷, 좌표계→차량 정면 오프셋(`MOUNT_OFFSET_DEG`), 최소 포인트 수·노이즈 필터링 전부 실물 없이 작성함. 실물 장비로 `--live --theta <deg>`부터 확인
- [ ] `code/live_demo.py`의 4번째 카메라(전방) 연동 확인 — 기존 좌/우/후방 3대는 이미 마운트·캘리브레이션 완료, 전방(cam4)만 신규로 물리 장착·시리얼번호 확인 필요
- [ ] `DOAANGLE` 레지스터 id(=21)가 실제 이 보드에서 맞는지 `--live`로 검증 (ReSpeaker 공식 참고값이나 실측 안 됨)
- [ ] `MOUNT_OFFSET_DEG` — 차량에 마이크 장착 후 정면(0°) 기준 보정 (라이다용 오프셋은 별도 값, `lidar_distance_match.py`의 동명 상수 참고)
- [x] ~~카메라 3대 임시 마운트 방법~~ → 좌/우/후방 구매·규격·마운트·캘리브레이션 완료 (2026-08-25)
- [ ] `live_demo.py`의 `CAMERA_SERIAL`(시리얼번호, 현재 빈 값) — 설치 시 `CAMERA_CALIB_ID`에 맞는 물리 카메라(예: left=cam1 캘리브레이션 개체)의 실제 시리얼번호로 채울 것
- [ ] `gxipy`(Daheng Galaxy SDK), `velodyne-decoder` 설치 및 카메라·라이다 인식 확인 — 노트북(우분투)에 벤더 드라이버 설치 필요할 수 있음
- [ ] 한글 폰트 설치 확인(`fonts-nanum` 등) 및 `live_demo.py`의 `FONT_PATH`를 실제 경로로 수정
- [ ] `yolo26m_v4_cls05`의 베이스 모델(`yolo26m.pt`)이 표준 `ultralytics` 배포판에 있는 이름인지 확인 — 없으면 학습 팀원의 패키지 버전 확인 필요
- [ ] **폴백 리허설**: LiDAR 연동이 촬영 전날까지도 불안정하면 `LIDAR_AVAILABLE=False`로 강제 고정해 8/25 버전(모든 감지 "주의" 캡)으로 촬영할 수 있는지 최소 한 번은 미리 확인해 둘 것
