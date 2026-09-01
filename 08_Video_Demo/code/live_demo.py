"""
live_demo.py — 9/8 한이음 영상 제출용 라이브 데모 앱 (2026-08-31 갱신: 카메라 4대+LiDAR 복귀).

기본은 후방 카메라를 화면에 띄워두고, 소리(경적/사이렌/오토바이)가 감지되면 방향에 맞는 카메라로
바꾸면서 상단 배너(+사이렌/오토바이는 Detection 박스)를 3초간 띄웠다가 자동으로 후방으로 복귀한다.
전/좌/우/후방 카메라 4대는 시작할 때 전부 미리 열어(stream_on) 둬서, 전환 시 첫 프레임을 기다리는
지연이 없다. 화면 레이아웃은 08_Video_Demo/camera_ui_mockup.html의 디자인을 따른다.

배경: 00_Overview/2026-08-25_9.8_영상제출_촬영_계획.md (2026-08-31 갱신) — 카메라 4대+LiDAR가
촬영일까지 장착·데이터 수신 가능해져, 8/25에 정했던 "카메라 3대·라이다 없음·모든 감지 주의 캡"
축소 구성을 되돌린다. 전방은 더 이상 "전방 확인" 텍스트가 아니라 실제 카메라로 표시하고,
LiDAR로 방향별 거리를 매칭해 확정된 대상은 ui_state_spec.md 기준대로 "경고"(빨강)까지 올린다.
**(2026-09-01) LiDAR 연동(lidar_distance_match.py)은 노트북에서 실제 VLP-16으로 검증 완료**
   (해당 파일 상단 참고 — decode API 오류·지면 반사 오탐 등 치명 버그 수정됨). 그래도 촬영 당일
   케이블링 문제 등에 대비해 LIDAR_AVAILABLE로 계속 가드하며, 연동 실패 시 8/25 버전과 동일하게
   모든 감지를 "주의"로 캡하는 폴백으로 자동 전환한다.
⚠️ 단, 이 audio_worker는 pyaudio로 로컬 ReSpeaker를 직접 잡는 단일 기기 구조다.
   jetson_audio_sender.py(젯슨/노트북 분리, 실장비 UDP 링크 검증 완료 — 아래 실행 섹션 참고)의
   수신 측(노트북에서 UDP로 분류 결과를 받는 코드)은 아직 이 파일에 통합돼 있지 않다 —
   9/8 데모를 분리 아키텍처로 찍을지, 이 파일처럼 노트북에 마이크까지 물려 단일 기기로 찍을지
   팀 결정 필요(00_Overview/현재_상태_요약.md 참고).

구조: 오디오 캡처+분류+DoA는 백그라운드 스레드(audio_worker), 카메라 표시는 메인 스레드(video_loop,
OpenCV 창은 메인 스레드에서 돌려야 안전)로 분리해 영상 프레임레이트가 오디오 추론 주기(1~2초)에
발목 잡히지 않게 했다. 둘 사이는 SharedState(락으로 보호)로만 통신한다. LiDAR 스캐너는
lidar_distance_match.LidarScanner가 자체 배경 스레드로 최신 스캔을 갱신한다.

⚠️ 카메라 4대는 일반 UVC 웹캠이 아니라 **Daheng Imaging MER2-240-159U3C (USB3 Vision 산업용
   area-scan 카메라)**다. OpenCV의 cv2.VideoCapture로는 열리지 않아 Daheng 공식 파이썬 SDK
   `gxipy`(Galaxy SDK 설치 시 포함, https://www.get-cameras.com/showdownloadcenter 등에서
   드라이버+SDK 배포)로 접근해야 한다.
**(2026-09-01 실장비 점검 완료)** CAMERA_SERIAL은 실제 4대 동시 열기·프레임 획득까지 확인한
값으로 채움(front=FHH26070137/left=FHH26070138/right=FHH26070139/rear=FHH26070140). 단 방향
배정은 캘리브레이션 ID와 sync_cam.py 순서로 "유도"한 값이라, 장착 후 preflight_view.py로 각
방향에서 손을 흔들어 해당 칸이 반응하는지 물리 검증 필수 — 배선 순서가 어긋나도 코드는 알아채지
못한다. 카메라 1대(front, FHH26070137)는 TB5 독이 아니라 노트북 본체에 직결돼야 하고, 나머지
2대(right/rear)는 독 내부 5Gbps 구간을 라이다 이더넷과 공유하므로 프레임레이트를 올리면 라이다
패킷이 유실될 수 있다.
⚠️ Detection 모델(yolo26m_v4_cls05/weights/best.pt)은 gitignore 처리된 파일이라 로컬에 직접
   있어야 동작한다. 클래스는 confusion_matrix.png 기준 Motorcycle/Ambulance 2종.
⚠️ CAMERA_CALIB_ID로 방향별 담당 캠 번호(front=4/left=1/right=2/rear=3)를 미리 확정해뒀다 —
   설치 시 이 번호로 캘리브레이션된 물리 카메라를 해당 방향에 붙일 것. 08_Video_Demo/calibration/ 참고.
⚠️ cv2.putText는 한글을 그리지 못해 Pillow로 한글 폰트를 얹어 그린다(put_text_kr). FONT_PATH
   하드코딩(나눔고딕)이 미설치 우분투에서 첫 프레임에 OSError로 죽던 문제를 2026-09-01에
   확인·수정 — resolve_font_path()가 나눔고딕/Noto CJK 후보 경로를 자동 탐색해 없으면 fc-list로
   찾는다(Noto CJK는 보통 sudo 없이도 이미 깔려 있어 그걸로 동작 확인됨).

실행:
    pip install ultralytics gxipy opencv-python Pillow velodyne-decoder
    python3 live_demo.py   (표시 창에서 q 누르면 종료)
"""

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "03_Audio_Classification" / "code"))

from doa_camera_select import Tuning, find_device, select_camera  # noqa: E402
import lidar_distance_match as lidar  # noqa: E402
import realtime_classify as clf  # noqa: E402

# 경적 포함 3클래스는 카메라 전환까지, 사이렌/오토바이 2개만 시각 Detection까지 실행
CAMERA_TRIGGER_CLASSES = {"car_horn", "siren", "motorcycle"}
DETECTION_CLASSES = {"siren", "motorcycle"}
LIDAR_MATCH_CLASSES = {"siren", "motorcycle"}  # 경적은 거리 추정 대상이 아님 (현재_상태_요약.md 참고)
CLASS_LABEL_KO = {"car_horn": "경적", "siren": "사이렌", "motorcycle": "오토바이"}
CAMERA_LABEL_KO = {"front": "전방 카메라", "left": "좌측 카메라", "right": "우측 카메라", "rear": "후방 카메라(기본)"}
LOC_LABEL_KO = {"front": "전방", "left": "좌측", "right": "우측", "rear": "후방"}
# BGR (OpenCV) — camera_ui_mockup.html의 종류색(teal/red/violet)과 맞춤
KIND_COLOR_BGR = {"car_horn": (199, 194, 0), "siren": (48, 59, 255), "motorcycle": (255, 140, 185)}
# ui_state_spec.md §1 상태 색(BGR) — 이제 "경고"가 실제로 쓰인다 (라이다 거리 확정 시)
LEVEL_COLOR_BGR = {"주의": (0, 212, 255), "경고": (48, 59, 255)}
HOLD_SEC = 3.0  # 마지막 감지 이후 배너/카메라 전환을 유지하는 시간 (ui_state_spec.md "3초 유지" 원칙)

# ui_state_spec.md §5 오토바이 거리 임계값
MOTORCYCLE_NEAR_M = 15.0   # 이 안쪽이면 "경고"
MOTORCYCLE_WATCH_M = 25.0  # 15~25m는 "주의"(접근 관찰), 그 밖/미확정은 주의로 캡(안전 폴백)

# 한글 폰트 후보 — 우선순위순. 첫 프레임에서 OSError로 죽는 걸 막기 위해 존재하는 첫 경로를 씀.
# 전부 없으면 fc-list로 "CJK"가 들어간 폰트를 찾는다(Noto CJK는 sudo 없이도 이미 깔려 있는 경우가 많음).
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
]
_font_cache = {}
_font_path_cache = None

DETECTOR_WEIGHTS = REPO_ROOT / "08_Video_Demo" / "model_outputs" / "yolo26m_v4_cls05" / "weights" / "best.pt"

# 2026-09-01 노트북 실장비 점검에서 4대 동시 열기·프레임 획득까지 확인한 실제 시리얼번호.
# ⚠️ 방향(front/left/right/rear) 배정은 캘리브레이션 ID·sync_cam.py 순서로 "유도"한 값 —
# 장착 후 preflight_view.py로 물리 검증 필수(어긋나면 여기를 실측대로 재배정).
CAMERA_SERIAL = {"front": "FHH26070137", "left": "FHH26070138",
                  "right": "FHH26070139", "rear": "FHH26070140"}
CAMERA_INDEX_FALLBACK = {"front": 4, "left": 1, "right": 2, "rear": 3}  # gxipy는 index가 1부터 시작
_camera_handles = {}
_device_manager = None  # 전역으로 유지 — 지역변수로 두면 GC가 gx_close_lib()를 호출해 카메라가 죽는다

# 2026-09-01 실측: 실내 8ms+게인16dB도 어두웠음(밝기 27~95) — 야외 주간엔 8ms로 충분한 것으로
# 관측됐으나 촬영 현장에서 최종값 확정 필요. 자동노출이 게인 없이 노출만 올려(최대 223ms 관측)
# 50km/h에서 3.1m 모션블러가 나던 문제 수정: 노출은 고정하고 게인만 자동으로 보정한다.
EXPOSURE_TIME_US = 8000  # 8ms, gxipy는 마이크로초 단위
FRAME_RATE_HZ = 30.0

CALIBRATION_DIR = REPO_ROOT / "08_Video_Demo" / "calibration"
# 방향별 담당 캠 번호를 미리 확정 — 설치 시 이 번호에 맞는 물리 카메라를 해당 방향에 붙이면 됨.
CAMERA_CALIB_ID = {"front": 4, "left": 1, "right": 2, "rear": 3}
_calib_cache = {}

# LiDAR 연동은 2026-09-01 실장비 점검에서 검증됨(아래 참고) — 그래도 촬영 당일 케이블링 문제 등에
# 대비해, main()에서 스캐너 시작에 실패하면 이 값을 False로 내려 이후 모든 감지가 8/25 버전과
# 동일하게 "주의"로만 표시되도록 자동 폴백한다.
LIDAR_AVAILABLE = True
_lidar_scanner = None
# 줄자 실측값으로 교체할 것 — lidar_distance_match.py의 지면 필터 기준값(임시 거치대 기준 1.7m).
LIDAR_MOUNT_HEIGHT_M = 1.7


def priority_rank(class_name: str, level: str, blind: bool) -> int:
    """숫자가 작을수록 우선순위 높음. ui_state_spec.md §2 7단계 표를 이 데모의 3클래스에 맞춘 것."""
    if class_name == "motorcycle" and blind:
        return 1  # 오토바이 · 사각지대 진입
    if class_name == "motorcycle" and level == "경고":
        return 2  # 오토바이 · 15m 이내 근접
    if class_name == "siren" and level == "경고":
        return 3  # 사이렌 · 차량 확정
    if class_name == "siren":
        return 4  # 사이렌 · 위치 미확정
    if class_name == "car_horn":
        return 5  # 경적
    return 6  # 오토바이 · 15~25m 접근 (또는 미확정)


def load_detector():
    from ultralytics import YOLO

    return YOLO(str(DETECTOR_WEIGHTS))


def open_daheng_camera(camera: str):
    import gxipy as gx

    global _device_manager
    # 2026-09-01 발견: DeviceManager를 지역변수로 두면 함수 리턴 시 GC가 돌면서 gx_close_lib()가
    # 호출돼 카메라 4대가 열자마자 무효화됐다(NotInitApi -13). 프로세스 전역으로 유지해야 한다.
    if _device_manager is None:
        _device_manager = gx.DeviceManager()
    _device_manager.update_device_list()
    serial = CAMERA_SERIAL[camera]
    cam = (_device_manager.open_device_by_sn(serial) if serial
           else _device_manager.open_device_by_index(CAMERA_INDEX_FALLBACK[camera]))
    cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
    # 2026-09-01 발견: 자동노출이 게인 없이 노출시간만 올려(최대 223ms 관측, 4.5fps) 50km/h에서
    # 3.1m 모션블러가 났다 — 노출은 고정하고 게인만 자동으로 보정, 프레임레이트도 상한을 건다.
    cam.ExposureAuto.set(gx.GxAutoEntry.OFF)
    cam.ExposureTime.set(EXPOSURE_TIME_US)
    cam.GainAuto.set(gx.GxAutoEntry.CONTINUOUS)
    cam.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)
    cam.AcquisitionFrameRate.set(FRAME_RATE_HZ)
    cam.BalanceWhiteAuto.set(gx.GxAutoEntry.CONTINUOUS)
    cam.stream_on()
    return cam


def open_all_cameras():
    """전/좌/우/후방을 전부 미리 열어(stream_on) 둔다 — 전환 시 첫 프레임 지연이 없도록."""
    for camera in ("front", "left", "right", "rear"):
        print(f"[*] {camera} 카메라 여는 중...")
        _camera_handles[camera] = open_daheng_camera(camera)


def close_all_cameras():
    for cam in _camera_handles.values():
        cam.stream_off()
        cam.close_device()


def load_calibration(camera: str):
    """camera_matrix, dist_coeffs 튜플. CAMERA_CALIB_ID[camera]가 None이면 None 반환."""
    cam_id = CAMERA_CALIB_ID.get(camera)
    if cam_id is None:
        return None
    if camera not in _calib_cache:
        calib_path = CALIBRATION_DIR / f"cam{cam_id}_calib" / "results" / "calibration.json"
        with open(calib_path) as f:
            data = json.load(f)
        camera_matrix = np.array(data["camera_matrix"])
        dist_coeffs = np.array(data["distortion_coefficients"])
        _calib_cache[camera] = (camera_matrix, dist_coeffs)
    return _calib_cache[camera]


def read_camera_frame(camera: str):
    """open_all_cameras()로 이미 열려 있는 카메라에서 프레임 1장을 읽어 왜곡보정까지 적용."""
    raw_image = _camera_handles[camera].data_stream[0].get_image()
    if raw_image is None:
        return None
    frame = raw_image.convert("RGB").get_numpy_array()

    calib = load_calibration(camera)
    if calib is not None:
        import cv2

        camera_matrix, dist_coeffs = calib
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    return frame


def run_detection(detector, camera: str):
    """siren/motorcycle 감지 시 해당 카메라 프레임에서 구급차/오토바이를 찾는다. 없으면 None."""
    frame = read_camera_frame(camera)
    if frame is None:
        return None
    result = detector.predict(frame, verbose=False)[0]
    if len(result.boxes) == 0:
        return None
    box = max(result.boxes, key=lambda b: float(b.conf))
    return {"label": result.names[int(box.cls)], "conf": float(box.conf)}


class SharedState:
    """오디오 스레드 <-> 영상 스레드 사이에서 공유하는 표시 상태. 락으로 보호.

    동시에 여러 종류의 소리가 감지될 수 있으므로, 클래스별로 "현재 유효한 감지"를 각자의
    hold 시각과 함께 self.active에 들고 있는다. 화면에 그릴 때(snapshot)만 그중 우선순위
    1위를 메인 카메라/배너로, 나머지는 작은 아이콘+방향으로 뽑아낸다 — ui_state_spec.md
    §2 "지도는 전부, 알림은 하나" 원칙을 지도 없는 이 데모에서 아이콘으로 구현한 것.
    """

    def __init__(self):
        self.lock = threading.Lock()
        # class_name -> {"camera": str, "loc": str, "detection": dict|None,
        #                "level": "주의"|"경고", "distance": float|None, "blind": bool, "until": float}
        self.active = {}

    def trigger(self, camera: str, class_name: str, loc: str, detection,
                level: str, distance, blind: bool):
        now = time.time()
        with self.lock:
            self.active[class_name] = {
                "camera": camera, "loc": loc, "detection": detection,
                "level": level, "distance": distance, "blind": blind, "until": now + HOLD_SEC,
            }

    def snapshot(self):
        now = time.time()
        with self.lock:
            for name in [n for n, v in self.active.items() if now > v["until"]]:
                del self.active[name]

            if not self.active:
                return "rear", None, None, []

            top_name = min(self.active, key=lambda n: priority_rank(n, self.active[n]["level"], self.active[n]["blind"]))
            top = self.active[top_name]
            secondary = [{"class_name": n, "loc": v["loc"]}
                         for n, v in self.active.items() if n != top_name]

            banner = {"class_name": top_name, "loc": top["loc"], "level": top["level"],
                      "distance": top["distance"], "blind": top["blind"]}
            return top["camera"], banner, top["detection"], secondary


def audio_worker(state: SharedState, panns_model, svm, class_names, tuning, detector, stream):
    seconds, interval = 1.0, 2.0
    n_samples = int(seconds * clf.SR)
    while True:
        frames = []
        collected = 0
        while collected < n_samples:
            raw = stream.read(clf.CHUNK, exception_on_overflow=False)
            chunk = np.frombuffer(raw, dtype=np.int16).reshape(-1, clf.CHANNELS)
            frames.append(chunk[:, clf.MIC_CHANNEL_INDEX])
            collected += chunk.shape[0]

        mono = np.concatenate(frames)[:n_samples].astype(np.float32) / 32768.0
        results = clf.classify(panns_model, svm, class_names, mono)
        top_class, top_score = results[0]
        print(f"[{time.strftime('%H:%M:%S')}] class={top_class} score={top_score:+.3f}")

        if top_class in CAMERA_TRIGGER_CLASSES:
            doa = tuning.direction
            camera = select_camera(doa)
            detection = run_detection(detector, camera) if top_class in DETECTION_CLASSES else None
            level, distance, blind = classify_level(top_class, doa)
            print(f"    -> doa={doa}deg: {camera} 카메라로 전환, level={level}, "
                  f"distance={distance}, blind={blind}, detection={detection}")
            state.trigger(camera, top_class, LOC_LABEL_KO[camera], detection, level, distance, blind)

        time.sleep(interval)


def classify_level(class_name: str, doa_deg: float):
    """클래스+DoA로 (level, distance_m|None, blind)를 정한다.

    경적은 거리 추정 대상이 아니라 항상 "주의". 사이렌/오토바이는 LIDAR_AVAILABLE이고 매칭이
    성공하면 ui_state_spec.md §5/§7 기준으로 "경고"까지 올리고, 매칭 실패(장비 문제 포함)면
    8/25 축소 버전과 동일하게 "주의"로 캡한다 — 없는 정확도를 있는 척하지 않기 위한 안전 폴백.
    """
    if class_name not in LIDAR_MATCH_CLASSES or not LIDAR_AVAILABLE:
        return "주의", None, False

    if not _lidar_scanner.healthy():
        # 스캐너 배경 스레드가 죽어 있으면 latest_points()는 마지막 스캔에서 멈춰 있다 —
        # 조용히 오래된 값을 쓰지 않고 "관측 안 됨"과 동일하게 폴백한다.
        print(f"[!] LiDAR 스캐너 비정상: {_lidar_scanner.status()}")
        return "주의", None, False

    points = _lidar_scanner.latest_points()
    match = lidar.match_distance(points, doa_deg)
    if match is None:
        return "주의", None, False  # 관측 안 됨(위치 미확정) — 안전 폴백
    if match["blind"]:
        return "경고", None, True  # 반경 6m 사각지대 — 거리 숫자 없이 경고

    distance = match["distance_m"]
    if class_name == "siren":
        return "경고", distance, False  # 차량 확정
    # motorcycle
    if distance < MOTORCYCLE_NEAR_M:
        return "경고", distance, False
    if distance < MOTORCYCLE_WATCH_M:
        return "주의", distance, False
    return "주의", distance, False  # 멀리 있음 — 이미 감지는 됐으니 안내는 유지, 상향은 안 함


def resolve_font_path() -> str:
    """한글 폰트 경로를 찾는다 — 고정 경로 하나만 가정하면 미설치 환경에서 첫 프레임에 OSError로
    죽는다(2026-09-01 발견). 후보 목록을 먼저 보고, 없으면 fc-list로 CJK 폰트를 탐색한다."""
    global _font_path_cache
    if _font_path_cache is not None:
        return _font_path_cache

    for path in FONT_CANDIDATES:
        if Path(path).exists():
            _font_path_cache = path
            return path

    import subprocess

    try:
        out = subprocess.run(["fc-list"], capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            if "CJK" in line or "Nanum" in line:
                _font_path_cache = line.split(":")[0].strip()
                return _font_path_cache
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    raise RuntimeError(
        "한글 폰트를 찾지 못했습니다. `sudo apt install fonts-noto-cjk` 또는 "
        "`fonts-nanum` 설치 후 다시 실행하세요."
    )


def _font(size: int):
    from PIL import ImageFont

    if size not in _font_cache:
        _font_cache[size] = ImageFont.truetype(resolve_font_path(), size)
    return _font_cache[size]


def put_text_kr(frame_bgr, text: str, org, size: int, color_bgr):
    """cv2.putText는 한글을 못 그려서 Pillow로 얹어 그린다."""
    import cv2
    from PIL import Image, ImageDraw

    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(img).text(org, text, font=_font(size),
                              fill=(color_bgr[2], color_bgr[1], color_bgr[0]))
    frame_bgr[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_secondary_icons(frame_bgr, secondary):
    """메인 배너를 차지하지 못한(우선순위가 낮은) 동시 감지들을 우측 상단에 작은 아이콘+방향으로 표시."""
    import cv2

    w = frame_bgr.shape[1]
    y = 24
    for item in secondary:
        col = KIND_COLOR_BGR[item["class_name"]]
        cx, cy = w - 46, y + 20
        cv2.circle(frame_bgr, (cx, cy), 18, col, -1)
        label = f"{CLASS_LABEL_KO[item['class_name']]} · {item['loc']}"
        put_text_kr(frame_bgr, label, (w - 300, y + 6), 22, (230, 230, 230))
        y += 46


def draw_overlay(frame_bgr, camera: str, banner, detection, secondary):
    import cv2

    h, w = frame_bgr.shape[:2]
    put_text_kr(frame_bgr, CAMERA_LABEL_KO[camera], (24, h - 48), 26, (200, 200, 200))
    draw_secondary_icons(frame_bgr, secondary)

    if banner is not None:
        col = KIND_COLOR_BGR[banner["class_name"]]
        level_col = LEVEL_COLOR_BGR[banner["level"]]

        if banner["level"] == "경고":  # ui_state_spec.md §5 "테두리 빛" — 정적 근사(애니메이션 생략)
            cv2.rectangle(frame_bgr, (6, 6), (w - 6, h - 6), level_col, 10)

        cv2.rectangle(frame_bgr, (24, 24), (24 + 320, 24 + 90), col, -1)
        put_text_kr(frame_bgr, CLASS_LABEL_KO[banner["class_name"]], (40, 42), 34, (10, 10, 10))
        if banner["blind"]:
            loc_text = f"{banner['level']} → {banner['loc']} · 사각지대"
        elif banner["distance"] is not None:
            loc_text = f"{banner['level']} → {banner['loc']} · {banner['distance']:.0f}m"
        else:
            loc_text = f"{banner['level']} → {banner['loc']}"
        put_text_kr(frame_bgr, loc_text, (24 + 332, 46), 26, level_col)

        if detection is not None:
            x0, y0, x1, y1 = w // 3, h // 4, w * 2 // 3, h * 3 // 4
            cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), col, 3)
            tag = f"{detection['label']} {round(detection['conf'] * 100)}%"
            cv2.rectangle(frame_bgr, (x0, y0 - 36), (x0 + 220, y0), col, -1)
            cv2.putText(frame_bgr, tag, (x0 + 10, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (10, 10, 10), 2, cv2.LINE_AA)


def video_loop(state: SharedState):
    import cv2

    print("[*] 화면 표시 시작 — 창에서 q 누르면 종료")
    while True:
        camera, banner, detection, secondary = state.snapshot()
        frame = read_camera_frame(camera)
        if frame is None:
            time.sleep(0.05)
            continue

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        draw_overlay(frame_bgr, camera, banner, detection, secondary)
        cv2.imshow("올빼미러 - 9/8 데모", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def main():
    print("[*] 모델 로딩 중...")
    clf.ensure_panns_checkpoint(clf.DEFAULT_PANNS_CKPT_PATH)
    panns_model = clf.load_panns_model(clf.DEFAULT_PANNS_CKPT_PATH)
    svm, class_names, ckpt = clf.load_svm(clf.DEFAULT_SVM_PATH)
    print(f"[*] 클래스: {class_names}")

    print("[*] Detection 모델(구급차/오토바이) 로딩 중...")
    detector = load_detector()
    # 2026-09-01 발견: YOLO 첫 추론에 워밍업 비용이 있어(이후 46~86ms) 실제 첫 사이렌/오토바이
    # 감지 시 배너가 3.3초 지연됐다 — 기동 시 더미 프레임으로 미리 흡수해 둔다.
    detector.predict(np.zeros((1200, 2048, 3), dtype=np.uint8), verbose=False)

    open_all_cameras()

    global LIDAR_AVAILABLE, _lidar_scanner
    try:
        _lidar_scanner = lidar.LidarScanner(mount_height_m=LIDAR_MOUNT_HEIGHT_M)
        _lidar_scanner.start()
        print("[*] LiDAR 스캐너 시작")
    except Exception as e:  # noqa: BLE001 — 실물 미검증 연동이라 실패해도 데모 전체를 죽이지 않음
        print(f"[!] LiDAR 연동 실패({e}) — 8/25 버전과 동일하게 모든 감지를 '주의'로 표시합니다.")
        LIDAR_AVAILABLE = False

    import pyaudio

    p = pyaudio.PyAudio()
    device_index = clf.get_respeaker_index(p)
    if device_index is None:
        print("[!] ReSpeaker 마이크를 찾을 수 없습니다.")
        close_all_cameras()
        return
    stream = p.open(format=pyaudio.paInt16, channels=clf.CHANNELS, rate=clf.SR,
                     input=True, input_device_index=device_index,
                     frames_per_buffer=clf.CHUNK)

    tuning = Tuning(find_device())
    state = SharedState()

    worker = threading.Thread(
        target=audio_worker,
        args=(state, panns_model, svm, class_names, tuning, detector, stream),
        daemon=True,
    )
    worker.start()

    try:
        video_loop(state)  # 메인 스레드 — OpenCV 창은 메인 스레드에서 돌려야 안전
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        close_all_cameras()
        if _lidar_scanner is not None:
            _lidar_scanner.stop()


if __name__ == "__main__":
    main()
