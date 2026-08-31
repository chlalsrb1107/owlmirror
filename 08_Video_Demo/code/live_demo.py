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
⚠️ 이 LiDAR 연동(lidar_distance_match.py)은 실물로 검증된 적이 없다 — LIDAR_AVAILABLE로
   가드해서, 연동 실패 시 8/25 버전과 동일하게 모든 감지를 "주의"로 캡하는 폴백으로 자동 전환한다.

⚠️ (2026-09-01) 젯슨/노트북 분리 구조로 전환됨. 이 파일은 이제 **노트북에서만** 돌아간다.
   오디오 수음·분류(PANNs+SVM)·DoA는 전부 젯슨(jetson_audio_sender.py)이 담당하고, 이 앱은
   UDP로 그 결과(class/theta)만 받는다 — 따라서 노트북에는 torch/pyaudio/pyusb가 필요 없고
   ReSpeaker를 꽂을 필요도 없다. 이전 단일 머신 버전의 audio_worker는 detection_worker로
   대체됐다(수음·분류가 빠지고 네트워크 수신만 남음).

구조: 젯슨 패킷 수신+카메라 선택+Detection+LiDAR 매칭은 백그라운드 스레드(detection_worker),
카메라 표시는 메인 스레드(video_loop, OpenCV 창은 메인 스레드에서 돌려야 안전)로 분리해 영상
프레임레이트가 Detection/LiDAR 처리에 발목 잡히지 않게 했다. 둘 사이는 SharedState(락으로 보호)
로만 통신한다. AudioDetectionReceiver와 lidar_distance_match.LidarScanner는 각자 자체 배경
스레드로 최신 상태를 갱신한다.

⚠️ 링크가 끊기면 화면이 거짓말하지 않도록: 젯슨에서 패킷이 끊기면 마지막 감지는 HOLD_SEC 후
   자동으로 사라지고, 화면 좌하단에 "젯슨 연결 끊김"이 표시된다.

⚠️ 카메라 4대는 일반 UVC 웹캠이 아니라 **Daheng Imaging MER2-240-159U3C (USB3 Vision 산업용
   area-scan 카메라)**다. OpenCV의 cv2.VideoCapture로는 열리지 않아 Daheng 공식 파이썬 SDK
   `gxipy`(Galaxy SDK 설치 시 포함, https://www.get-cameras.com/showdownloadcenter 등에서
   드라이버+SDK 배포)로 접근해야 한다.
⚠️ 카메라 4대(전/좌/우/후방)의 실제 시리얼번호(CAMERA_SERIAL)는 이 저장소에 카메라가 없어
   자리표시값(빈 문자열)이다. 설치 후 `gx.DeviceManager().update_device_list()`로 확인한
   시리얼번호로 채울 것 (00_Overview/2026-08-25_9.8_영상제출_촬영_계획.md "사전 점검" 참고).
   시리얼번호 대신 index로 열 수도 있으나 USB 재연결 시 순서가 바뀔 수 있어 시리얼 고정을 권장.
⚠️ Detection 모델(yolo26m_v4_cls05/weights/best.pt)은 gitignore 처리된 파일이라 로컬에 직접
   있어야 동작한다. 클래스는 confusion_matrix.png 기준 Motorcycle/Ambulance 2종.
⚠️ CAMERA_CALIB_ID로 방향별 담당 캠 번호(front=4/left=1/right=2/rear=3)를 미리 확정해뒀다 —
   설치 시 이 번호로 캘리브레이션된 물리 카메라를 해당 방향에 붙일 것. 08_Video_Demo/calibration/ 참고.
⚠️ cv2.putText는 한글을 그리지 못해 Pillow로 한글 폰트를 얹어 그린다(put_text_kr). FONT_PATH가
   실제 노트북에 있는 한글 폰트 경로를 가리키는지 확인할 것 (우분투는 보통
   `sudo apt install fonts-nanum` 후 `/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf`).

실행 (노트북):
    pip install ultralytics gxipy opencv-python Pillow velodyne-decoder
    python3 live_demo.py                 # 기본 포트(9870)에서 젯슨 패킷 대기
    python3 live_demo.py --port 9870     # 포트 지정

    # 젯슨 쪽 (별도 기기에서):
    #   python3 jetson_audio_sender.py --host <노트북IP>
    # 장비 없이 화면만 검증하려면 노트북에서 직접:
    #   python3 jetson_audio_sender.py --host 127.0.0.1 --simulate
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 오디오 분류(realtime_classify)와 DoA 직접 읽기(Tuning/find_device)는 젯슨으로 넘어가
# 여기서 import하지 않는다 — 노트북에 torch/pyaudio/pyusb를 설치할 필요가 없어진다.
from audio_receiver import AudioDetectionReceiver, DEFAULT_PORT  # noqa: E402
from doa_camera_select import select_camera  # noqa: E402
import lidar_distance_match as lidar  # noqa: E402

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
# 창 이름은 ASCII로 — 한글로 두면 Qt 제목표시줄에 "???? - 9/8 ??"로 깨져 나온다.
WINDOW_NAME = "OwlMirror Demo (9/8)"
DISPLAY_WIDTH = 1280  # 표시용 가로 크기. 센서 원본 2048은 화면에 너무 크고 그리기도 무겁다.

HOLD_SEC = 3.0  # 마지막 감지 이후 배너/카메라 전환을 유지하는 시간 (ui_state_spec.md "3초 유지" 원칙)
# 젯슨 패킷 폴링 주기. 젯슨 전송 주기(기본 2초, 스펙 0.25초)보다 충분히 짧게 잡아 전환을 늦추지 않는다.
POLL_INTERVAL = 0.05

# ui_state_spec.md §5 오토바이 거리 임계값
MOTORCYCLE_NEAR_M = 15.0   # 이 안쪽이면 "경고"
MOTORCYCLE_WATCH_M = 25.0  # 15~25m는 "주의"(접근 관찰), 그 밖/미확정은 주의로 캡(안전 폴백)

# 한글 폰트 후보 — 앞에서부터 실제로 존재하는 첫 번째를 쓴다.
# 특정 경로 하나만 박아두면(예전엔 나눔고딕 고정) 그 패키지가 없는 기기에서 첫 프레임에
# 바로 OSError로 죽는다. 우분투 기본 설치에 Noto CJK가 들어있어 sudo 없이도 동작한다.
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
]
_font_cache = {}
_font_path_cache = None

DETECTOR_WEIGHTS = REPO_ROOT / "08_Video_Demo" / "model_outputs" / "yolo26m_v4_cls05" / "weights" / "best.pt"

# (2026-09-01) 실장비 4대 열거로 시리얼 확보 — 전부 MER2-240-159U3C, 2048x1200 BayerGB8.
# 아래 방향 배정은 CAMERA_CALIB_ID(front=cam4/left=cam1/right=cam2/rear=cam3)와
# daheng_ws_1/sync_cam.py의 CAM1~CAM4 순서를 결합해 **유도**한 값이다.
# ⚠️ 아직 차량에 장착하지 않았으므로 물리적으로 검증된 배정이 아니다. 루프랙에 올린 뒤
#    각 방향에서 손을 흔들어 화면이 맞게 뜨는지 반드시 확인하고, 어긋나면 여기를 고칠 것.
#    (배선 순서와 소프트웨어 순서가 어긋나도 코드는 절대 알아채지 못한다 — sync_cam.py 주석 참고)
CAMERA_SERIAL = {
    "front": "FHH26070137",  # = CAM4
    "left":  "FHH26070138",  # = CAM1
    "right": "FHH26070139",  # = CAM2
    "rear":  "FHH26070140",  # = CAM3 (기본 표시 카메라)
}
CAMERA_INDEX_FALLBACK = {"front": 4, "left": 1, "right": 2, "rear": 3}  # gxipy는 index가 1부터 시작
_camera_handles = {}
# ⚠️ DeviceManager를 살려둬야 한다. gxipy는 DeviceManager 인스턴스 수를 세다가 0이 되면
#    __del__에서 gx_close_lib()를 호출하고, 그 순간 이미 열려 있던 카메라 핸들이 전부
#    무효화된다(이후 get_image가 NotInitApi -13으로 실패). 지역변수로 만들면 함수를
#    빠져나가는 순간 GC돼서 카메라가 조용히 죽는다 — 2026-09-01 실장비에서 실제로 겪음.
_device_manager = None

CALIBRATION_DIR = REPO_ROOT / "08_Video_Demo" / "calibration"
# 방향별 담당 캠 번호를 미리 확정 — 설치 시 이 번호에 맞는 물리 카메라를 해당 방향에 붙이면 됨.
CAMERA_CALIB_ID = {"front": 4, "left": 1, "right": 2, "rear": 3}

# 노출 상한(us) — 주행 중 모션블러를 지배하는 값. 50km/h(=13.9m/s)에서 한 프레임 흐림:
#   8ms → 11cm(허용) / 20ms → 28cm / 223ms → 3.1m(형체 불명, 상한 없을 때의 실측값)
MAX_EXPOSURE_US = 8000.0
MAX_GAIN_DB = 16.0  # MER2-240-159U3C 최대치

# 프레임레이트 상한(Hz). 노출 상한을 걸면 센서가 122fps까지 올라가는데, 4대면
# 4 x 122 x 2.46MB = 약 1,200 MB/s로 USB 대역폭을 넘는다. 더 나쁜 건 카메라 2대가
# 라이다 이더넷(r8152)과 같은 5Gbps 구간(~400MB/s)을 공유한다는 점 — 포화되면
# 라이다 UDP 패킷이 유실되고 거리 매칭이 통째로 망가진다.
# 30fps면 4대 합쳐 약 295MB/s이고, 데모 표시용으로 충분하다.
MAX_FRAMERATE_HZ = 30.0
_calib_cache = {}

# LiDAR 연동은 실물로 검증된 적이 없어, main()에서 스캐너 시작에 실패하면 이 값을 False로
# 내려 이후 모든 감지가 8/25 버전과 동일하게 "주의"로만 표시되도록 자동 폴백한다.
LIDAR_AVAILABLE = True
_lidar_scanner = None
# 노면에서 라이다 원점까지의 높이(m). 지면 제거 기준이라 이 값이 틀리면 노면이 안 걸러지거나
# (너무 크면) 실제 차량까지 잘려나간다. ⚠️ 루프랙 장착 후 줄자로 실측해 교체할 것.
LIDAR_MOUNT_HEIGHT_M = lidar.MOUNT_HEIGHT_M


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
    """YOLO 가중치가 없으면 None을 돌려준다 (Detection만 빼고 나머지는 그대로 동작).

    가중치(best.pt)는 gitignore 대상이라 저장소에 없고 팀원에게 따로 받아야 한다. 없다고
    데모 전체를 못 띄우면 카메라·라이다·젯슨 링크 점검조차 막히므로, 여기서 죽이지 않는다.
    """
    if not DETECTOR_WEIGHTS.exists():
        print(f"[!] Detection 가중치 없음({DETECTOR_WEIGHTS}) — 경광등/오토바이 검출을 끄고 진행합니다.")
        return None

    from ultralytics import YOLO

    detector = YOLO(str(DETECTOR_WEIGHTS))
    # 첫 추론은 커널 컴파일/메모리 할당 때문에 유독 느리다(실측 3.3초, 이후 46~86ms).
    # 그대로 두면 데모에서 "첫 사이렌"만 배너가 3초 늦게 뜬다 — 여기서 미리 태워 없앤다.
    print("[*] Detection 워밍업 중...")
    t0 = time.time()
    detector.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    print(f"[*] Detection 워밍업 완료 ({time.time() - t0:.1f}s)")
    return detector


def get_device_manager():
    """프로세스 전체에서 DeviceManager 하나만 만들어 계속 살려둔다 (위 주석 참고)."""
    global _device_manager
    if _device_manager is None:
        import gxipy as gx

        _device_manager = gx.DeviceManager()
        _device_manager.update_device_list()
    return _device_manager


def configure_exposure(cam, gx):
    """자동노출을 쓰되 노출시간에 상한을 걸고, 부족한 밝기는 게인으로 채우게 한다.

    ⚠️ 상한을 안 걸면 어두울 때 자동노출이 노출시간만 끝까지 밀어올리고 게인은 0dB로 둔다
       (2026-09-01 실측: 실내에서 노출 223ms / 게인 0dB / 4.5fps). 주행 중이면 한 프레임에
       50km/h 기준 3.1m가 뭉개져 형체를 알아볼 수 없고, YOLO 경광등 검출도 불가능해진다.
       daheng_ws_1/sync_cam.py가 고정 8ms를 쓰는 것과 같은 이유다.

    sync_cam.py처럼 완전 고정이 아니라 상한만 두는 이유: 저 파일은 하드웨어 트리거 동기
    수집용이라 프레임마다 지연이 일정해야 하지만, 이 데모는 실시간 표시용이고 주행 중
    조도 변화(터널, 그늘, 역광)가 커서 자동 적응이 필요하다.
    """
    cam.ExposureAuto.set(gx.GxAutoEntry.CONTINUOUS)
    cam.AutoExposureTimeMax.set(MAX_EXPOSURE_US)
    cam.GainAuto.set(gx.GxAutoEntry.CONTINUOUS)
    cam.AutoGainMax.set(MAX_GAIN_DB)
    cam.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)
    cam.AcquisitionFrameRate.set(MAX_FRAMERATE_HZ)


def open_daheng_camera(camera: str):
    import gxipy as gx

    device_manager = get_device_manager()
    serial = CAMERA_SERIAL[camera]
    cam = (device_manager.open_device_by_sn(serial) if serial
           else device_manager.open_device_by_index(CAMERA_INDEX_FALLBACK[camera]))
    cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
    configure_exposure(cam, gx)
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
    if detector is None:
        return None
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


def detection_worker(state: SharedState, receiver: AudioDetectionReceiver, detector):
    """젯슨에서 온 감지 패킷을 받아 카메라 선택 → Detection → LiDAR 매칭까지 처리한다.

    이전 단일 머신 버전의 audio_worker를 대체한다. 수음·분류·DoA 추정이 젯슨으로 넘어갔으므로
    이 스레드는 네트워크 수신만 기다리고, 무거운 작업(YOLO, LiDAR 매칭)은 그대로 여기서 한다.
    """
    was_connected = False
    while True:
        link = receiver.link()
        if link["connected"] != was_connected:
            was_connected = link["connected"]
            print("[*] 젯슨 연결됨" if was_connected
                  else "[!] 젯슨 연결 끊김 — 감지가 들어오지 않습니다")

        packet = receiver.get_new()
        if packet is None:
            time.sleep(POLL_INTERVAL)
            continue

        top_class = packet["class"]
        if top_class not in CAMERA_TRIGGER_CLASSES:
            continue  # "none"이거나 배경음 클래스 — 화면은 HOLD_SEC 후 알아서 풀린다

        # ⚠️ 젯슨이 이미 마운트 오프셋을 적용해 차량 좌표계로 보냈다.
        #    여기서 오프셋을 다시 적용하면 이중 적용이 되므로 반드시 0.0으로 호출할 것.
        doa = packet["theta"]
        camera = select_camera(doa, mount_offset_deg=0.0)
        detection = run_detection(detector, camera) if top_class in DETECTION_CLASSES else None
        level, distance, blind = classify_level(top_class, doa)

        latency_ms = (time.time() - packet["t"]) * 1000
        print(f"[{time.strftime('%H:%M:%S')}] class={top_class} "
              f"score={packet.get('score', 0.0):+.3f} theta={doa:.1f} "
              f"(수음~표시 지연 {latency_ms:.0f}ms)")
        print(f"    -> {camera} 카메라로 전환, level={level}, "
              f"distance={distance}, blind={blind}, detection={detection}")
        state.trigger(camera, top_class, LOC_LABEL_KO[camera], detection, level, distance, blind)


def classify_level(class_name: str, doa_deg: float):
    """클래스+DoA로 (level, distance_m|None, blind)를 정한다.

    경적은 거리 추정 대상이 아니라 항상 "주의". 사이렌/오토바이는 LIDAR_AVAILABLE이고 매칭이
    성공하면 ui_state_spec.md §5/§7 기준으로 "경고"까지 올리고, 매칭 실패(장비 문제 포함)면
    8/25 축소 버전과 동일하게 "주의"로 캡한다 — 없는 정확도를 있는 척하지 않기 위한 안전 폴백.
    """
    if class_name not in LIDAR_MATCH_CLASSES or not LIDAR_AVAILABLE:
        return "주의", None, False

    # start()가 성공했다는 것만으로는 수신 스레드가 살아있다는 보장이 안 된다 —
    # 스레드가 죽으면 latest_points()가 계속 옛 스캔을 돌려줘서 거리가 멈춘 채로 표시된다.
    if not _lidar_scanner.healthy():
        return "주의", None, False

    points = _lidar_scanner.latest_points()
    match = lidar.match_distance(points, doa_deg, mount_height_m=LIDAR_MOUNT_HEIGHT_M)
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


def resolve_font_path():
    """존재하는 첫 한글 폰트 경로. 하나도 없으면 명확한 안내와 함께 예외."""
    global _font_path_cache
    if _font_path_cache is None:
        for candidate in FONT_CANDIDATES:
            if Path(candidate).exists():
                _font_path_cache = candidate
                break
        else:
            raise FileNotFoundError(
                "한글 폰트를 찾을 수 없습니다. `sudo apt install fonts-nanum` 후 다시 실행하거나 "
                "FONT_CANDIDATES에 실제 경로를 추가하세요.")
    return _font_path_cache


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


def draw_overlay(frame_bgr, camera: str, banner, detection, secondary, link=None):
    import cv2

    h, w = frame_bgr.shape[:2]
    put_text_kr(frame_bgr, CAMERA_LABEL_KO[camera], (24, h - 48), 26, (200, 200, 200))
    draw_secondary_icons(frame_bgr, secondary)

    # 링크가 끊겼는데 화면이 평소와 똑같아 보이면 "소리가 없다"로 오해하게 된다 — 명시적으로 알린다.
    if link is not None and not link["connected"]:
        put_text_kr(frame_bgr, "젯슨 연결 끊김 — 오디오 감지 없음", (24, h - 84), 26, (60, 60, 255))

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


def video_loop(state: SharedState, receiver: AudioDetectionReceiver = None):
    import cv2

    # 창을 명시적으로 만들고 크기를 지정한다. 안 하면 OpenCV/Qt가 744x332 같은 엉뚱한 크기로
    # 띄워서, 다른 창 뒤에 묻히면 사용자는 "아무것도 안 뜬다"고 느끼게 된다.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, int(DISPLAY_WIDTH * 1200 / 2048))
    print(f"[*] 화면 표시 시작 — '{WINDOW_NAME}' 창에서 q 누르면 종료")
    print("[*] 창이 안 보이면 다른 창에 가려진 것입니다 (Alt+Tab으로 전환)")
    while True:
        camera, banner, detection, secondary = state.snapshot()
        frame = read_camera_frame(camera)
        if frame is None:
            time.sleep(0.05)
            continue

        link = receiver.link() if receiver is not None else None
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # 표시용으로 먼저 줄인 뒤 오버레이를 그린다 (반대로 하면 글자까지 축소돼 안 읽힌다).
        # 2048x1200을 그대로 띄우면 화면을 넘기도 하고 그리는 비용도 크다.
        if frame_bgr.shape[1] != DISPLAY_WIDTH:
            h = int(frame_bgr.shape[0] * DISPLAY_WIDTH / frame_bgr.shape[1])
            frame_bgr = cv2.resize(frame_bgr, (DISPLAY_WIDTH, h))
        draw_overlay(frame_bgr, camera, banner, detection, secondary, link)
        cv2.imshow(WINDOW_NAME, frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="올빼미러 9/8 데모 (노트북) — 젯슨 오디오 감지를 UDP로 받아 화면에 표시")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="젯슨 패킷을 받을 UDP 포트")
    parser.add_argument("--bind", default="0.0.0.0",
                        help="바인드 주소. 이더넷 직결만 받으려면 해당 NIC의 고정 IP를 지정")
    return parser.parse_args()


def main():
    args = parse_args()

    print("[*] Detection 모델(구급차/오토바이) 로딩 중...")
    detector = load_detector()

    open_all_cameras()

    global LIDAR_AVAILABLE, _lidar_scanner
    try:
        _lidar_scanner = lidar.LidarScanner()
        _lidar_scanner.start()
        print("[*] LiDAR 스캐너 시작")
    except Exception as e:  # noqa: BLE001 — 실물 미검증 연동이라 실패해도 데모 전체를 죽이지 않음
        print(f"[!] LiDAR 연동 실패({e}) — 8/25 버전과 동일하게 모든 감지를 '주의'로 표시합니다.")
        LIDAR_AVAILABLE = False

    receiver = AudioDetectionReceiver(port=args.port, bind_host=args.bind).start()
    print(f"[*] 젯슨 패킷 수신 대기: {args.bind}:{args.port} (UDP)")

    state = SharedState()
    worker = threading.Thread(
        target=detection_worker, args=(state, receiver, detector), daemon=True)
    worker.start()

    try:
        video_loop(state, receiver)  # 메인 스레드 — OpenCV 창은 메인 스레드에서 돌려야 안전
    except KeyboardInterrupt:
        pass
    finally:
        link = receiver.link()
        print(f"[*] 젯슨 패킷 수신 {link['received']}개, 유실 추정 {link['dropped']}개, "
              f"불량 {link['malformed']}개")
        receiver.stop()
        close_all_cameras()
        if _lidar_scanner is not None:
            _lidar_scanner.stop()


if __name__ == "__main__":
    main()
