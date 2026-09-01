# 올빼미러 — 소리로 위험을 보는 운전 보조 시스템

> **주 사용자는 청각장애인 운전자다.** 사이렌·경적·오토바이 배기음을 마이크 어레이로 듣고,
> 방향을 계산해 해당 방향 카메라를 자동으로 띄우고, LiDAR로 거리를 확정해 화면에 보여준다.
> 소리로 보완할 수 없으므로 **화면이 유일한 전달 경로**이고, 그 전제가 모든 설계의 근거다.
>
> 한이음 공모전 프로젝트 · ReSpeaker 4 Mic Array + Jetson Orin Nano + 카메라 4대 + VLP-16

---

## ⚠️ 새 세션에서는 이 순서로 읽을 것

1. **[`00_Overview/현재_상태_요약.md`](00_Overview/현재_상태_요약.md)** — 확정 사항·실측치·미해결 문제. **항상 최신 기준.**
2. 이 README — 구조와 실행 방법
3. 개별 노트 — 단, `현재_상태_요약.md` 하단의 "알려진 노트 불일치" 표를 먼저 확인할 것

---

## 지금 어디까지 됐나 (2026-09-02 실측)

전체 파이프라인이 실장비로 한 번에 동작한다 — 젯슨(마이크) + 카메라 4대 + LiDAR + 화면.

| 항목 | 실측 결과 |
|---|---|
| 사이렌 검출률 | **83%** · 주기 1.22s |
| 방향 정밀도 (σ) | **5~7°** (정지·실내, 소리 고정 시) |
| 카메라 4대 동시 | 124 fps 합계 (305 MB/s) |
| LiDAR 수신 | 753.6 pps · 유실 0.1% |
| 젯슨→노트북 UDP | 300+ 패킷 유실 0 |
| 젯슨 전송 주기 | **1.024초** — CUDA 적용 후 추론이 수음 시간 안에 끝난다 (CPU 때 1.220초) |
| 화면 렌더 | 6~8 ms/프레임 (1920×1080) |

**미해결 (2026-09-02 기준)**

- **오토바이 검출률 13%** — 음량은 충분했다(최고 −8.7 dB). 레벨이 아니라 모델이 이 배기음을 분류하지 못한다
- **`score`/`margin`을 신뢰도로 쓸 수 없다** — sklearn SVC 다중클래스 OvR 값이 "이긴 1:1 대결 수(0~4)+소수점" 구조라, 확신 있는 예측이면 클래스와 무관하게 늘 4.2가 나온다. 오경보 차단은 음량(`--min-db`)으로 우회 중
- **`MOUNT_OFFSET_DEG` 미실측** — σ 5~7°는 **정밀도**이고 정확도는 아직 모른다. 차량 장착 후 각도를 아는 위치에서 재야 한다
- **차량 미장착** — 카메라 시리얼→방향 배정이 캘리브레이션 ID로 유도한 값이라 물리 검증 필요

---

## 구조

```
┌─ 젯슨 Orin Nano ─────────────┐        ┌─ 노트북 (ASUS TUF A14, Ubuntu 22.04) ──┐
│  ReSpeaker 4 Mic Array v2.0  │        │  카메라 4대 (전/좌/후/우, 90° 간격)      │
│    → PANNs Cnn14 + SVM 분류  │        │  LiDAR Velodyne VLP-16                 │
│    → 펌웨어 DoA (방향)        │        │                                        │
│    → 음량(rms) · 방향 신뢰도  │        │  UDP 수신 → 카메라 선택 → YOLO 검출     │
│                              │  UDP   │    → LiDAR 거리 매칭 → 화면 합성        │
│  JSON 한 줄만 전송 ──────────┼───────▶│                                        │
│  (영상·점군은 오가지 않는다)   │ 이더넷 │  BEV 지도 | 카메라 영상 반반 화면        │
└──────────────────────────────┘  직결  └────────────────────────────────────────┘
     192.168.10.2                             192.168.10.1
```

**젯슨→노트북 인터페이스** (초당 약 0.8회, `--interval 0` 기준)

```json
{"seq": 1234, "t": 1751250042.318, "class": "siren", "conf": 0.91,
 "score": 4.27, "margin": 1.06, "theta": 194.0, "sigma": 7.4,
 "rms_db": -22.4, "theta_ok": true, "raw": "siren"}
```

| 필드 | 설명 |
|---|---|
| `class` | `car_horn` / `siren` / `motorcycle` / `none` |
| `theta` | 차량 좌표계 방위각 — **전방 0°, 반시계 +** |
| `sigma` | 방향 불확실성(도). 수음 구간 안 DoA 8회 읽기의 **실측 흩어짐**. BEV 부채꼴 폭이 된다 |
| `rms_db` | 음량(dBFS). 오경보 차단·근접 판정에 사용 |
| `theta_ok` | 방향을 믿을 수 있는가. `false`면 노트북이 카메라를 돌리지 않고 방향 표현을 그리지 않는다 |
| `raw` | 진단용 — `class`가 `none`일 때 실제로 1위였던 클래스 |

> ⚠️ **각도 규약이 두 개다.** 시스템 `theta`는 반시계 +인데 UI 목업 SVG는 시계 +다.
> `bev_render._screen_angle()`에서 한 번만 뒤집어 맞춘다 — 빠뜨리면 BEV가 통째로 좌우 반전된다.

---

## 알림 규칙

소리마다 운전자가 알아야 할 것이 달라서 클래스별로 규칙을 나눴다.
구현·검증: [`08_Video_Demo/code/alert_policy.py`](08_Video_Demo/code/alert_policy.py) (`--selftest`)

| 소리 | 규칙 |
|---|---|
| **경적** | 동시 발생 시 **가장 큰 소리** 채택(1.5초 래치), 4초 유지. 같은 방향 ±40°에서 12초 안에 2회 → 경고 승격 |
| **사이렌** | **항상 경고** — 긴급차량은 위치 확정 여부와 무관하게 운전자가 알아야 한다(양보 의무) |
| **오토바이** | Detection 성공 → 경고·위치 추적 / 배기음 커짐 → 경고·근접 / 그 외 → 주의·사각지대 위험 |

**공통** — 음량이 `LOUD_NEAR_DB`(−26 dBFS)를 넘으면 "바로 옆"으로 보고 사각지대 판정.
LiDAR를 쓰는 사이렌·오토바이는 6m 이내 관측도 사각지대로 처리한다.

**우선순위** (동시 감지 시 배너는 1위만, BEV 지도에는 전부)

```
오토바이 사각지대 > 오토바이 경고 > 사이렌 > 경적 반복 > 경적 1회 > 오토바이 주의
```

---

## 화면 설계 — 청각장애인 운전자 기준

상세 원칙: [`06_Display_Integration/ui_state_spec.md`](06_Display_Integration/ui_state_spec.md) §1-1

1. **소리로 보완할 수 있다고 가정하지 않는다.** 경고음은 도달하지 않는다. 화면에 없는 정보는 전달되지 않은 정보다
2. **정보마다 그림과 글자를 함께 준다.** 종류는 아이콘+이름, 방향은 회전 화살표+"좌측". 중복이 아니라 이중 경로
3. **주변시야에 걸리게 한다.** 경고 단계에서 테두리와 자차 방향 표시가 거리에 반비례하는 주기(0.3~1.2초)로 깜빡인다
4. **읽는 시간을 줄인다.** 방향은 글자보다 화살표가, 위험도는 글자보다 색이 빠르다

화면은 **왼쪽 BEV 지도 | 오른쪽 카메라 영상** 반반이다. BEV는 마커가 아니라 **LiDAR 실제 리턴 포인트**로 그린다 — 매칭 확정은 색 클러스터, 미확정은 부채꼴, 6m 이내는 사각 구역 점등. 자차 둘레에는 감지 방향으로 호가 켜진다(양산차 사각지대 경고등과 같은 방식).

---

## 실행

### 노트북

```bash
pip install ultralytics gxipy opencv-python Pillow velodyne-decoder

# 운전자용 외장 디스플레이에 전체화면
python3 08_Video_Demo/code/live_demo.py --fullscreen --monitor HDMI-1-0

# 창 모드 / 야간 팔레트 / 헤드리스 점검
python3 08_Video_Demo/code/live_demo.py --theme night
python3 08_Video_Demo/code/live_demo.py --snapshot ./out --frames 20
```

`--monitor`를 주면 그 모니터 **해상도 그대로 렌더링**한다. 다른 해상도의 차량 디스플레이도 대응된다.

### 젯슨

```bash
python3 -u jetson_audio_sender.py --host 192.168.10.1 \
        --svm-path ~/owlmirror/03_Audio_Classification/model_outputs/panns_svm/best.pt
```

> ⚠️ **`-u`를 붙일 것.** SSH 파이프로는 stdout이 블록 버퍼링돼 출력이 안 보인다(멈춘 것처럼 보인다).
> ⚠️ **종료는 Ctrl+C.** SSH 창을 그냥 닫으면 젯슨 쪽 파이썬이 살아남아 마이크를 계속 점유하고,
> 다음 실행이 "마이크 못 찾음"으로 죽는다. 그때는 `pkill -f "[j]etson_audio_sender\.py"`.

### 하드웨어 없이 확인

```bash
python3 08_Video_Demo/code/bev_render.py    --selftest --outdir /tmp/bev   # 화면 10종 PNG
python3 08_Video_Demo/code/alert_policy.py  --selftest                     # 알림 규칙
python3 08_Video_Demo/code/lidar_distance_match.py --selftest              # 거리 매칭
python3 08_Video_Demo/code/doa_camera_select.py    --selftest              # 방향→카메라
python3 08_Video_Demo/code/jetson_audio_sender.py  --host 127.0.0.1 --simulate
```

---

## 저장소 구조

```
올빼미러/
├── 00_Overview/                  ← ★ 현재_상태_요약.md 부터 읽을 것
│   ├── 현재_상태_요약.md          확정 사항·실측치·미해결 문제 (항상 최신)
│   ├── 2026-08-25_9.8_영상제출_촬영_계획.md
│   ├── 프로젝트_개요.md · 파이프라인_다이어그램.md · 중간평가_전략.md
│   └── 보고자료/                 중간보고 PPT·PDF
│
├── 01_Hardware/                  ReSpeaker·젯슨 환경, 카메라 4대+LiDAR 장착
├── 02_Data_Collection/           데이터셋 수집 계획
├── 03_Audio_Classification/      PANNs Cnn14+SVM 5클래스, 전처리, 실시간 추론
├── 04_Sound_Localization/        TDoA 원리, GCC-PHAT, 빔포밍 (자체 구현 — 현재 미사용)
├── 05_Camera_Tracking/           DoA→카메라 매핑
├── 06_Display_Integration/       ui_state_spec.md (화면 사양), bev_mockup.html
├── 07_System_Integration/        젯슨/노트북 분리 구조, ROS2 노드 설계
│
└── 08_Video_Demo/                ← 실제 동작하는 구현이 전부 여기 있다
    ├── camera_ui_mockup.html     UI 기준 디자인 (시나리오 10종, 인터랙티브)
    ├── calibration/cam1~4_calib/ 카메라 내부 파라미터
    ├── model_outputs/            YOLO 구급차·오토바이 검출 모델 (가중치는 gitignore)
    └── code/
        ├── jetson_audio_sender.py   [젯슨] 수음→분류→DoA→UDP 송신
        ├── audio_receiver.py        [노트북] UDP 수신, 유실·시계 스큐·링크 단절 추적
        ├── alert_policy.py          클래스별 알림 규칙 (주의/경고 판정)
        ├── bev_render.py            화면 합성 — BEV|카메라 반반, 주간/야간 팔레트
        ├── live_demo.py             오케스트레이터 (카메라·LiDAR·수신·화면)
        ├── lidar_distance_match.py  VLP-16 수신, 지면 제거, 거리 클러스터링
        ├── doa_camera_select.py     펌웨어 DoA 읽기 + 방향→카메라 매핑
        └── preflight_view.py        카메라 4대+LiDAR 사전 점검 뷰어
```

---

## 하드웨어

| | |
|---|---|
| 젯슨 | Jetson Orin Nano Super Developer Kit (8GB) — **오디오 전담** |
| 마이크 | ReSpeaker 4 Mic Array v2.0 (6채널, ch0=AEC) · 방풍 커버 필요 |
| 카메라 | **Daheng MER2-240-159U3C ×4** (USB3 Vision 산업용, 2048×1200) + C-mount 2.8mm |
| LiDAR | Velodyne VLP-16 — 루프랙 중앙 최상단. **반경 6m는 물리적 사각지대** |
| 노트북 | ASUS TUF Gaming A14 (FA401UM) · Ubuntu 22.04 · ROS2 Humble |
| 네트워크 | 이더넷 직결. 젯슨 `192.168.10.2` / 노트북 `192.168.10.1` (USB 랜 어댑터) |

> ⚠️ 카메라는 일반 UVC 웹캠이 아니라 산업용이라 `cv2.VideoCapture`로 열리지 않는다.
> Daheng 공식 SDK `gxipy`가 필요하다.

---

## 알려진 함정

실장비 연동에서 실제로 겪은 것들. 같은 데서 두 번 막히지 않도록 적어둔다.

| 증상 | 원인 |
|---|---|
| 젯슨 실행이 "멈춘" 것처럼 보임 | SSH 파이프 stdout 버퍼링. `python3 -u` |
| "마이크 못 찾음" | 이전 실행이 살아서 점유 중. `arecord -l`에서 `Subdevices: 0/1` |
| 핑은 되는데 패킷이 안 옴 | 젯슨·노트북 IP가 겹침. 루프백이 답한 것 |
| 모든 방향이 6.34m로 나옴 | LiDAR 지면 미제거. 1.7m/tan(15°) = 노면까지 거리 |
| 카메라가 4.5fps | 자동노출이 노출시간만 223ms까지 올림. 50km/h에서 3.1m 모션블러 |
| 화면 위아래 흰 띠 | OpenCV Qt 백엔드 기본 툴바. `WINDOW_GUI_NORMAL` |
| BEV 좌우 반전 | 각도 규약(시계 vs 반시계) 변환 누락 |
