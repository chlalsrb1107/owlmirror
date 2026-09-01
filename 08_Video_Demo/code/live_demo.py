"""
live_demo.py — 9/8 한이음 영상 제출용 라이브 데모 앱 (2026-09-02 갱신: BEV|카메라 반반 화면).

화면을 세로로 반 나눠 **왼쪽은 BEV 원형 지도, 오른쪽은 카메라 영상**을 상시 함께 띄운다
(08_Video_Demo/camera_ui_mockup.html 목업 그대로). 카메라는 기본이 후방이고, 소리(경적/사이렌/
오토바이)가 감지되면 방향에 맞는 카메라로 바꾸면서 상단 배너를 3초간 띄웠다가 자동 복귀한다.
BEV에는 LiDAR가 실제로 들고 있는 포인트를 그대로 그린다 — 매칭 확정 대상은 색 클러스터,
미확정은 성긴 부채꼴, 6m 이내는 사각 구역 점등. 그리기는 전부 bev_render.py가 맡는다.
전/좌/우/후방 카메라 4대는 시작할 때 전부 미리 열어(stream_on) 둬서 전환 지연이 없다.

⚠️ (2026-09-02) 이 반반 레이아웃 전환은 09-01까지 "목업만 바꾸고 코드는 9/3 go/no-go 이후에
   정한다"고 미뤄뒀던 것을, 최종 결과물에 BEV가 반드시 들어가야 한다는 판단으로 앞당겨 구현한
   것이다. LiDAR가 죽거나 LIDAR_AVAILABLE=False여도 BEV 자체는 계속 그려진다(링·자차·부채꼴은
   오디오 방향만으로 성립) — 폴백 경로는 그대로 살아 있고, 다만 거리·클러스터가 빠지고
   좌하단에 "LiDAR 미연결"이 표시된다.

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
⚠️ cv2.putText는 한글을 그리지 못해 Pillow로 얹어 그린다 — 폰트 탐색·텍스트 합성은
   bev_render.py(FONT_CANDIDATES, TextLayer)로 옮겼다. 한글이 깨지면 그쪽을 볼 것
   (우분투는 보통 `sudo apt install fonts-nanum`, 기본 Noto CJK로도 동작).

실행 (노트북):
    pip install ultralytics gxipy opencv-python Pillow velodyne-decoder
    python3 live_demo.py                 # 기본 포트(9870)에서 젯슨 패킷 대기
    python3 live_demo.py --port 9870     # 포트 지정

    # 젯슨 쪽 (별도 기기에서):
    #   python3 jetson_audio_sender.py --host <노트북IP>
    # 장비 없이 화면만 검증하려면 노트북에서 직접:
    #   python3 jetson_audio_sender.py --host 127.0.0.1 --simulate
    # 카메라·라이다도 없이 BEV 레이아웃만 확인하려면:
    #   python3 bev_render.py --selftest --outdir /tmp/bev
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
import bev_render  # noqa: E402
import alert_policy  # noqa: E402
from alert_policy import AlertPolicy, priority_rank  # noqa: E402

# 경적 포함 3클래스는 카메라 전환까지, 사이렌/오토바이 2개만 시각 Detection까지 실행
CAMERA_TRIGGER_CLASSES = {"car_horn", "siren", "motorcycle"}
DETECTION_CLASSES = {"siren", "motorcycle"}
LIDAR_MATCH_CLASSES = {"siren", "motorcycle"}  # 경적은 거리 추정 대상이 아님 (현재_상태_요약.md 참고)

# 소리 클래스별로 **기대하는** Detection 라벨. 사이렌인데 Motorcycle이 잡히면 그건 오검출이지
# 확인이 아니다 — 2026-09-02 실물 테스트에서 사이렌 감지 중 실험실 책상이 "Motorcycle 48%"로
# 잡혀 화면 전체에 박스가 그려졌다. 클래스가 어긋나면 버린다.
DETECTION_EXPECT = {"siren": "Ambulance", "motorcycle": "Motorcycle"}
# YOLO 기본 신뢰도 임계값(0.25)은 데모용으로 너무 낮다. 오검출 박스가 영상에 그대로 남는 쪽이
# 놓치는 것보다 나쁘다 — 화면은 "확실할 때만" 대상을 지목해야 한다(ui_state_spec.md 원칙).
# ⚠️ 실차 영상으로 재조정 필요. 너무 높이면 실제 구급차도 놓친다.
DETECTION_MIN_CONF = 0.55
# 라벨·색·폰트는 화면을 그리는 쪽(bev_render)에 모아뒀다 — 두 군데에 두면 목업과 어긋난다.
CLASS_LABEL_KO = bev_render.CLASS_LABEL_KO
LOC_LABEL_KO = bev_render.LOC_LABEL_KO
# 창 이름은 ASCII로 — 한글로 두면 Qt 제목표시줄에 "???? - 9/8 ??"로 깨져 나온다.
WINDOW_NAME = "OwlMirror Demo (9/8)"

HOLD_SEC = 3.0  # 마지막 감지 이후 배너/카메라 전환을 유지하는 시간 (ui_state_spec.md "3초 유지" 원칙)
# 경적만 더 길게 유지한다 — 사이렌·배기음과 달리 한 번 빵 하고 끝나는 소리라, 3초로는
# 운전자가 화면을 보기 전에 배너가 사라진다. (alert_policy.HORN_HOLD_SEC과 같은 값)
HOLD_SEC_BY_CLASS = {"car_horn": alert_policy.HORN_HOLD_SEC}
# 젯슨 패킷 폴링 주기. 젯슨 전송 주기(기본 2초, 스펙 0.25초)보다 충분히 짧게 잡아 전환을 늦추지 않는다.
POLL_INTERVAL = 0.05


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


def run_detection(detector, camera: str, class_name: str):
    """siren/motorcycle 감지 시 해당 카메라 프레임에서 구급차/오토바이를 찾는다. 없으면 None.

    box는 **원본 프레임 좌표**(x0,y0,x1,y1)로 돌려준다 — 화면에 넣을 때 bev_render가 표시
    배율만큼 줄여서 그린다. 이전 버전은 좌표를 버리고 화면 1/3~2/3에 고정 사각형을 그렸는데,
    그러면 "Ambulance 92%"가 구급차가 아닌 엉뚱한 자리에 붙는다 — 영상에 그대로 남는 거짓말이다.
    """
    if detector is None:
        return None
    frame = read_camera_frame(camera)
    if frame is None:
        return None
    result = detector.predict(frame, verbose=False)[0]
    if len(result.boxes) == 0:
        return None

    expected = DETECTION_EXPECT.get(class_name)
    best = None
    for b in result.boxes:
        conf = float(b.conf)
        label = result.names[int(b.cls)]
        if conf < DETECTION_MIN_CONF:
            continue
        if expected is not None and label != expected:
            continue  # 사이렌인데 Motorcycle 같은 경우 — 확인이 아니라 오검출이다
        if best is None or conf > float(best.conf):
            best = b
    if best is None:
        return None

    xyxy = [float(v) for v in best.xyxy[0].tolist()]
    return {"label": result.names[int(best.cls)], "conf": float(best.conf), "box": xyxy}


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
                level: str, distance, blind: bool, theta: float, sigma: float, detail: str):
        now = time.time()
        hold = HOLD_SEC_BY_CLASS.get(class_name, HOLD_SEC)
        with self.lock:
            self.active[class_name] = {
                "class_name": class_name, "camera": camera, "loc": loc, "detection": detection,
                "level": level, "distance": distance, "blind": blind, "detail": detail,
                "theta": theta, "sigma": sigma, "until": now + hold,
            }

    def snapshot(self):
        """(표시할 카메라, 우선순위 순 대상 목록, 1위의 detection)을 돌려준다.

        BEV가 들어오면서 반환 형태가 바뀌었다. 이전에는 1위만 배너로 쓰고 나머지는 이름·방향만
        아이콘으로 넘겼지만, 이제 **모든 대상을 좌표까지 온전히** 넘긴다 — ui_state_spec.md §2
        "지도는 전부, 알림은 하나"를 지도 없이 아이콘으로 흉내내던 것을 진짜 지도로 대체한 것.
        """
        now = time.time()
        with self.lock:
            for name in [n for n, v in self.active.items() if now > v["until"]]:
                del self.active[name]

            if not self.active:
                return "rear", [], None

            targets = sorted(self.active.values(),
                             key=lambda v: priority_rank(v["class_name"], v["level"], v["blind"]))
            targets = [dict(t) for t in targets]  # 락 밖에서 안전하게 쓰도록 복사
            return targets[0]["camera"], targets, targets[0]["detection"]


def detection_worker(state: SharedState, receiver: AudioDetectionReceiver, detector,
                     policy: AlertPolicy):
    """젯슨에서 온 감지 패킷을 받아 카메라 선택 → Detection → LiDAR 매칭까지 처리한다.

    이전 단일 머신 버전의 audio_worker를 대체한다. 수음·분류·DoA 추정이 젯슨으로 넘어갔으므로
    이 스레드는 네트워크 수신만 기다리고, 무거운 작업(YOLO, LiDAR 매칭)은 그대로 여기서 한다.
    """
    was_connected = False
    last_camera = "rear"   # 방향을 못 믿을 때 되돌아갈 기준 화면
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
            continue  # "none"이거나 배경음 클래스 — 화면은 유지시간이 지나면 알아서 풀린다

        # ⚠️ 젯슨이 이미 마운트 오프셋을 적용해 차량 좌표계로 보냈다.
        #    여기서 오프셋을 다시 적용하면 이중 적용이 되므로 반드시 0.0으로 호출할 것.
        doa = packet["theta"]
        rms_db = float(packet.get("rms_db", -60.0))
        score = float(packet.get("score", 0.0))

        # 경적은 배너를 차지할 자격부터 따진다 — 유지 중인 더 큰 경적을 밀어내지 못하면
        # 아예 무시한다(래치). 카메라 전환·Detection도 돌리지 않아 헛일을 줄인다.
        if top_class == "car_horn" and not policy.horn_accepts(doa, rms_db):
            continue

        # ⚠️ 젯슨이 방향을 못 믿겠다고 표시하면(theta_ok=False) 그 방향으로 카메라를
        #    돌리지 않는다. 소리가 끊긴 구간에서 DoA가 튀면 화면이 엉뚱한 쪽으로
        #    홱홱 돌아가는데, 그건 알려주지 않느니만 못하다 — 특히 청각장애인 운전자는
        #    화면을 되짚어 확인할 다른 경로가 없다.
        theta_ok = bool(packet.get("theta_ok", True))
        if theta_ok:
            camera = select_camera(doa, mount_offset_deg=0.0)
            last_camera = camera
        else:
            camera = last_camera

        detection = (run_detection(detector, camera, top_class)
                     if top_class in DETECTION_CLASSES else None)
        level, distance, blind, detail = decide_alert(policy, top_class, doa, detection,
                                                      rms_db, score, theta_ok)
        # 방향을 모를 때 "위치 미확정"까지 덧붙이면 배너에 같은 말이 두 번 나온다
        # ("경고 · 방향 미확정 · 위치 미확정"). 방향 쪽 문구만 남긴다.
        if not theta_ok and detail == "위치 미확정":
            detail = None

        latency_ms = (time.time() - packet["t"]) * 1000
        print(f"[{time.strftime('%H:%M:%S')}] class={top_class} "
              f"score={score:+.3f} theta={doa:.1f}{'' if theta_ok else '(불신)'} "
              f"rms={rms_db:.1f}dB "
              f"(수음~표시 지연 {latency_ms:.0f}ms)")
        print(f"    -> {camera} 카메라로 전환, level={level} ({detail}), "
              f"distance={distance}, blind={blind}, detection={detection}")
        state.trigger(camera, top_class,
                      LOC_LABEL_KO[camera] if theta_ok else "방향 미확정",
                      detection, level, distance, blind,
                      theta=doa if theta_ok else None,
                      sigma=float(packet.get("sigma", 12.0)), detail=detail)


def lidar_match(doa_deg: float):
    """그 방향의 LiDAR 매칭 결과. 라이다를 쓸 수 없으면 None (= 위치 미확정과 같은 취급)."""
    if not LIDAR_AVAILABLE or _lidar_scanner is None:
        return None
    # start()가 성공했다는 것만으로는 수신 스레드가 살아있다는 보장이 안 된다 —
    # 스레드가 죽으면 latest_points()가 계속 옛 스캔을 돌려줘서 거리가 멈춘 채로 표시된다.
    if not _lidar_scanner.healthy():
        return None
    return lidar.match_distance(_lidar_scanner.latest_points(), doa_deg,
                                mount_height_m=LIDAR_MOUNT_HEIGHT_M)


def decide_alert(policy: AlertPolicy, class_name: str, doa_deg: float, detection,
                 rms_db: float, score: float, theta_ok: bool = True):
    """(level, distance, blind, detail)을 정한다. 규칙 자체는 alert_policy.py에 있다.

    detail은 배너 오른쪽에 붙는 짧은 문구다 — 왜 이 단계가 됐는지를 운전자가 읽을 수 있게
    하려는 것. "경고 → 좌측 · 사각지대 추정"처럼 근거를 같이 보여준다.

    ⚠️ 라이다를 못 쓰면 match가 None이 되는데, 클래스마다 그 뜻이 다르다. 사이렌은 원래
       거리와 무관하게 경고라 영향이 없고, 오토바이는 Detection·배기음 경로로 넘어가며,
       경적은 애초에 라이다를 쓰지 않는다. 그래서 예전처럼 "라이다 없으면 전부 주의로 캡"
       하지 않는다 — 그 캡은 모든 판정이 거리에 매여 있던 시절의 폴백이었다.
    """
    if class_name == "car_horn":
        level, repeats, near = policy.horn_level(doa_deg, rms_db)
        if near:
            detail = "근접"          # 소리만으로도 바로 옆이라고 판단
        elif level == "경고":
            detail = f"반복 {repeats}회"
        else:
            detail = "방향 안내"
        return level, None, near, detail

    # ⚠️ 방향을 못 믿으면 라이다도 조회하지 않는다. 그 각도로 점군을 뒤지면 엉뚱한
    #    방향의 물체까지 거리를 재서, 틀린 방향에 그럴듯한 숫자를 붙이게 된다.
    match = (lidar_match(doa_deg)
             if theta_ok and class_name in LIDAR_MATCH_CLASSES else None)

    if class_name == "siren":
        level, distance, blind = policy.siren_level(match)
        if blind:
            detail = "사각지대"
        elif distance is not None:
            detail = "차량 확정"
        else:
            detail = "위치 미확정"
        return level, distance, blind, detail

    # motorcycle — 잡았나(경고·추적) / 가까워지나(경고·근접) / 모르나(주의) 셋뿐이다
    return policy.motorcycle_level(match, detection is not None, rms_db)


def monitor_geometry(name: str):
    """xrandr에서 해당 모니터의 (x, y, width, height)를 찾는다. 못 찾으면 None.

    확장 배치에서는 각 모니터가 가상 화면의 다른 좌표를 차지한다(예: 노트북 +0+0,
    외장 +1920+0). 전체화면은 **창이 올라가 있는 모니터**를 기준으로 잡히므로,
    먼저 창을 그 좌표로 옮긴 다음 전체화면을 켜야 원하는 화면에 뜬다.
    """
    import re
    import subprocess

    try:
        out = subprocess.run(["xrandr", "--listmonitors"],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001 — xrandr가 없거나 X가 아니면 그냥 포기
        return None
    for line in out.splitlines():
        if name in line:
            m = re.search(r"(\d+)/\d+x(\d+)/\d+\+(\d+)\+(\d+)", line)
            if m:
                w, h, x, y = (int(m.group(i)) for i in (1, 2, 3, 4))
                return x, y, w, h
    return None


def video_loop(state: SharedState, receiver: AudioDetectionReceiver = None,
               snapshot_dir=None, frames=0, fullscreen=False, monitor=None):
    """메인 스레드 표시 루프 — 왼쪽 BEV | 오른쪽 카메라 반반 화면을 매 프레임 합성한다.

    (2026-09-02) camera_ui_mockup.html의 반반 레이아웃을 실제 화면으로 구현. 이전에는 카메라
    프레임 위에 배너·아이콘만 얹었고 BEV는 목업에만 있었다. 그리기 자체는 bev_render.render()가
    전담하고 여기서는 최신 상태(감지 목록 / 카메라 프레임 / LiDAR 포인트)를 모아 넘기기만 한다.
    """
    import cv2

    # snapshot_dir이 있으면 창을 띄우지 않고 프레임을 파일로 남긴다. 화면 없는 환경(SSH,
    # 헤드리스)에서 파이프라인 전체를 점검하거나, 영상 편집용 스틸을 뽑을 때 쓴다.
    saving = snapshot_dir is not None
    if saving:
        snapshot_dir = Path(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        print(f"[*] 스냅샷 모드 — {frames}장을 {snapshot_dir}에 저장하고 종료합니다")
    else:
        # 창을 명시적으로 만들고 크기를 지정한다. 안 하면 OpenCV/Qt가 744x332 같은 엉뚱한
        # 크기로 띄워서, 다른 창 뒤에 묻히면 사용자는 "아무것도 안 뜬다"고 느끼게 된다.
        # ⚠️ WINDOW_GUI_NORMAL을 반드시 붙인다. OpenCV의 Qt 백엔드는 기본이
        #    WINDOW_GUI_EXPANDED라 툴바와 상태바를 얹는데, 밝은 색이라 차량 화면에서
        #    위아래 흰 띠로 보인다(2026-09-02 실물에서 확인).
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        if monitor:
            geo = monitor_geometry(monitor)
            if geo is None:
                print(f"[!] 모니터 '{monitor}'를 찾지 못했습니다 — 현재 화면에 표시합니다.")
                print("    사용 가능한 이름은 `xrandr --listmonitors`로 확인하세요.")
            else:
                x, y, w, h = geo
                # 표시 해상도 그대로 그린다 — 확대/축소가 없어야 여백도 흐림도 없다.
                bev_render.configure_canvas(w, h)
                cv2.moveWindow(WINDOW_NAME, x, y)
                cv2.waitKey(1)  # 창 이동이 반영된 뒤에 전체화면을 켜야 한다
                print(f"[*] {monitor} 모니터({x},{y}) {w}x{h}로 창 이동 · 캔버스도 동일 해상도")
        if fullscreen:
            # 차량 디스플레이용 — 제목표시줄·작업표시줄 없이 화면을 꽉 채운다.
            # 캔버스가 16:9(1600x900)라 1920x1080 화면에서는 여백 없이 정확히 맞는다.
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print(f"[*] 전체화면 표시 시작 — q 누르면 종료")
        else:
            cv2.resizeWindow(WINDOW_NAME, bev_render.CANVAS_W, bev_render.CANVAS_H)
            print(f"[*] 화면 표시 시작 — '{WINDOW_NAME}' 창에서 q 누르면 종료")
            print("[*] 창이 안 보이면 다른 창에 가려진 것입니다 (Alt+Tab으로 전환)")
    saved = 0
    while True:
        camera, targets, detection = state.snapshot()
        frame = read_camera_frame(camera)
        if frame is None:
            time.sleep(0.05)
            continue

        # LiDAR는 살아있을 때만 포인트를 넘긴다. 죽은 스캐너가 들고 있는 옛 스캔을 그대로
        # 그리면 지도가 몇 초 전 세상을 보여주면서도 멀쩡해 보인다.
        lidar_ok = LIDAR_AVAILABLE and _lidar_scanner is not None and _lidar_scanner.healthy()
        points = _lidar_scanner.latest_points() if lidar_ok else None

        canvas = bev_render.render(
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), camera, targets, points,
            lidar_ok=lidar_ok,
            link=receiver.link() if receiver is not None else None,
            detection=detection)
        if saving:
            label = targets[0]["class_name"] if targets else "idle"
            path = snapshot_dir / f"{saved:03d}_{label}.png"
            cv2.imwrite(str(path), canvas)
            print(f"    저장 {path.name}  (감지 {len(targets)}건)")
            saved += 1
            if saved >= frames:
                break
            time.sleep(0.8)
            continue

        cv2.imshow(WINDOW_NAME, canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    if not saving:
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="올빼미러 9/8 데모 (노트북) — 젯슨 오디오 감지를 UDP로 받아 화면에 표시")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="젯슨 패킷을 받을 UDP 포트")
    parser.add_argument("--bind", default="0.0.0.0",
                        help="바인드 주소. 이더넷 직결만 받으려면 해당 NIC의 고정 IP를 지정")
    parser.add_argument("--snapshot", default=None, metavar="DIR",
                        help="창을 띄우지 않고 합성 프레임을 DIR에 PNG로 저장 (헤드리스 점검용)")
    parser.add_argument("--frames", type=int, default=8,
                        help="--snapshot일 때 저장할 장수")
    parser.add_argument("--fullscreen", action="store_true",
                        help="전체화면으로 표시 (차량 디스플레이용). 종료는 q")
    parser.add_argument("--theme", default="day", choices=sorted(bev_render.THEMES),
                        help="화면 팔레트. night=야간 계기판(기본), day=주간 주행용 밝은 배경. "
                             "햇빛 아래에서는 night가 거의 안 보인다")
    parser.add_argument("--monitor", default=None, metavar="NAME",
                        help="표시할 모니터 이름(예: HDMI-1-0). 확장 배치에서 운전자용 "
                             "외장 디스플레이를 지정할 때 사용. `xrandr --listmonitors` 참고")
    return parser.parse_args()


def main():
    args = parse_args()

    bev_render.set_theme(args.theme)
    print(f"[*] 화면 팔레트: {args.theme}")

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
        target=detection_worker, args=(state, receiver, detector, AlertPolicy()), daemon=True)
    worker.start()

    try:
        # 메인 스레드 — OpenCV 창은 메인 스레드에서 돌려야 안전
        video_loop(state, receiver, snapshot_dir=args.snapshot, frames=args.frames,
                   fullscreen=args.fullscreen, monitor=args.monitor)
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
