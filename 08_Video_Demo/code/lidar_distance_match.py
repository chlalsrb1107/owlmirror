"""
lidar_distance_match.py — 소리가 들린 방향을 LiDAR(VLP-16)로 들여다보고 그 음원까지의
거리를 확정한다. 차량 실장착(2026-10-15, 업체 시공) 기준으로 전면 개편한 버전.

파이프라인 (음원 방향 → 거리):
  1. VLP-16이 1회전(~100ms)마다 뿜는 포인트를 UDP(기본 2368)로 계속 수신해 최신 스캔을
     배경 스레드가 갱신한다. 스캔마다 번호(scan_id)를 매긴다 — 같은 스캔을 두 번 세지 않으려고.
  2. 젯슨이 보낸 방향(theta)과 **그 방향의 불확실성(sigma)**으로 각도 창을 만든다.
     창 폭 = clamp(sigma_k x sigma, angle_margin_min, angle_margin_max).
     고정 ±25°를 쓰던 이전 버전과 달라진 핵심 — 방향이 또렷하면 창을 좁혀 엉뚱한 물체를
     덜 집고, 흐릿하면 넓혀 놓치지 않는다.
  3. 창 안에서 지면·천장을 제거하고 **거리축 + 방위축 2단 클러스터링**으로 물체 후보를 만든다.
  4. 후보를 점수로 고른다 — 방향 일치를 가장 무겁게, 가까움·점 개수·거리 퍼짐(벽/연석 배제)을
     보조로. 사각지대(6m) 안에 후보가 있으면 거리 대신 "사각지대"로 보고한다.
  5. `TargetTracker`가 **스캔을 넘어가며 같은 물체를 추적**해 거리를 안정화하고
     접근 속도(closing_mps)와 충돌 예상 시간(ttc_s)을 낸다. 한두 스캔 놓쳐도 잠깐
     coast(마지막 값 유지)해서 화면의 숫자가 깜빡 사라지지 않게 한다.

장착값(높이·요 오프셋)은 소스에 박아두지 않고 `08_Video_Demo/config/lidar_mount.json`에서
읽는다 (`lidar_calibrate.py`가 현장에서 실측해 써주는 파일). 파일이 있으면 import 시
자동 적용되므로 live_demo·bev_render·calibrate가 같은 값을 쓴다.

⚠️ 아직 실차에서 검증되지 않은 부분 (2026-10-15 현재):
  - `mount_offset_deg`, `mount_height_m`가 실측값으로 채워지기 전까지는 기본값(0°, 1.7m)이고,
    이 둘이 틀리면 거리 자체가 아니라 **방향 배정과 지면 제거**가 먼저 깨진다.
    → 현장에서 `python3 lidar_calibrate.py --height --bearing --expect 0 --write` 먼저 실행.
  - 후보 점수 가중치(w_*), 방위 클러스터 간격, 접근 속도 임계값은 모두 책상 추정치다.
  - 자차가 움직이면 정지 구조물(가드레일·주차차량)도 "접근"으로 보인다. 차속(CAN/GPS)이
    없으므로 closing_mps는 **상대 속도**이고, 그 한계를 화면 문구에서도 숨기지 않는다.

의존성:
    pip install velodyne-decoder numpy

실행:
    python3 lidar_distance_match.py --selftest                 # 하드웨어 없이 로직 검증
    python3 lidar_distance_match.py --live --theta 194 --sigma 7
    python3 lidar_distance_match.py --live --theta 0 --track    # 추적·접근속도까지 출력
"""

import argparse
import json
import math
import socket
import threading
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

VLP16_UDP_PORT = 2368
VLP16_DATA_PACKET_BYTES = 1206  # 데이터 패킷 크기. 위치 패킷 등 다른 크기는 디코더에 넣지 않는다

# ---- 하위 호환 모듈 상수 -------------------------------------------------------
# bev_render.py가 import 시점에 직접 참조하는 값들이라 이름을 유지한다.
# 실제 동작에 쓰이는 값은 아래 LidarConfig(=CONFIG)이고, 이 상수들은 그 기본값이다.
BLIND_RADIUS_M = 6.0         # ui_state_spec.md §5 "사각 구역" — 이 반경 이내는 거리 신뢰 안 함
MIN_POINTS = 3               # 각도 창 안에 이보다 적으면 "관측 안 됨"
MOUNT_HEIGHT_M = 1.7         # 노면 → 라이다 원점 (실측값은 설정 파일로 들어온다)
MIN_OBJECT_HEIGHT_M = 0.3    # 노면에서 이만큼 위에 있어야 물체로 인정
MAX_OBJECT_HEIGHT_M = 1.5    # 센서보다 이만큼 위는 육교·표지판·터널 천장으로 보고 제외
CLUSTER_GAP_M = 1.0          # 거리축에서 이보다 벌어지면 다른 물체
MIN_CLUSTER_POINTS = 5       # 이보다 작은 덩어리는 노이즈로 본다
DEFAULT_ANGLE_MARGIN_DEG = 25.0   # sigma를 모를 때 쓰는 창 폭
MOUNT_OFFSET_DEG = 0.0       # 라이다 요 오프셋 (마이크용과 별개)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "lidar_mount.json"


@dataclass
class LidarConfig:
    """장착·매칭 파라미터 한 묶음. 현장 실측값은 JSON으로 들어온다.

    장착값(mount_*)만 실측 대상이고 나머지는 알고리즘 파라미터다 — 섞여 있지만 한 파일로
    두는 쪽이 현장에서 고치기 쉽다(차 안에서 여러 파일을 여는 것보다).
    """

    # --- 실측 대상 (lidar_calibrate.py가 채운다) ---
    mount_offset_deg: float = MOUNT_OFFSET_DEG   # 라이다 0° → 차량 전방 0° 보정
    mount_height_m: float = MOUNT_HEIGHT_M       # 노면 → 라이다 원점 높이

    # --- 지면/천장 제거 ---
    min_object_height_m: float = MIN_OBJECT_HEIGHT_M
    max_object_height_m: float = MAX_OBJECT_HEIGHT_M

    # --- 유효 거리 ---
    blind_radius_m: float = BLIND_RADIUS_M
    range_max_m: float = 60.0    # 이보다 먼 것은 화면에도 안 들어오고 판정에도 안 쓴다
    range_min_m: float = 0.5     # 무반사(거리 0) 점과 차체 자체 반사를 버린다

    # --- 각도 창 (방향 → 창 폭) ---
    sigma_k: float = 2.0             # 창 폭 = sigma_k x sigma (양쪽 합이 아니라 ±폭)
    angle_margin_min_deg: float = 8.0
    angle_margin_max_deg: float = 30.0

    # --- 클러스터링 ---
    cluster_gap_m: float = CLUSTER_GAP_M
    cluster_azimuth_gap_deg: float = 2.5   # 방위축에서 이보다 벌어지면 다른 물체
    min_cluster_points: int = MIN_CLUSTER_POINTS

    # --- 후보 점수 (낮을수록 좋음) ---
    w_bearing: float = 1.0       # 방향 일치 — 가장 무겁다. 우리가 찾는 건 "그 소리를 낸 것"
    w_distance: float = 0.45     # 가까운 쪽 우선
    w_spread: float = 0.35       # 거리 퍼짐이 큰 것(벽·가드레일·연석)에 벌점
    w_points: float = 0.25       # 점이 많은 쪽에 가점
    wall_range_span_m: float = 8.0   # 이 이상 거리로 퍼진 덩어리는 면(벽)으로 본다

    # --- 추적 ---
    track_confirm_hits: int = 2       # 서로 다른 스캔에서 이만큼 잡히면 "확정"
    track_coast_sec: float = 0.6      # 놓쳐도 이만큼은 마지막 값을 유지한다
    track_drop_sec: float = 2.0       # 이만큼 못 보면 추적을 버린다
    track_gate_m: float = 4.0         # 예측 거리에서 이만큼 벗어나면 다른 물체로 본다
    track_follow_max_deg: float = 30.0  # 추적 중 조회 방위가 원 요청에서 이 이상 벗어나지 않게 묶는다
    track_ema_alpha: float = 0.45     # 거리 평활화 (1.0이면 평활화 없음)
    closing_window_sec: float = 2.0   # 접근 속도를 재는 과거 구간
    closing_min_span_sec: float = 0.5  # 이보다 짧은 구간으로는 속도를 내지 않는다
    closing_min_samples: int = 4
    # 표시 지연 보정(m). 평활화(EMA)와 스캔 나이 때문에 화면 거리가 실제보다 **멀게** 나온다 —
    # 운전자가 여유 있다고 느끼는 쪽이라 위험한 방향의 오차다. 접근 속도를 알면 그만큼 되돌릴 수
    # 있고, 과보정을 막기 위해 이 값까지만 보정한다. 0이면 보정하지 않는다.
    lag_compensation_max_m: float = 2.0

    @classmethod
    def from_dict(cls, data: dict) -> "LidarConfig":
        """모르는 키와 "_"로 시작하는 주석 키는 조용히 무시한다 (JSON에 주석을 못 쓰므로)."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)


CONFIG = LidarConfig()


def config() -> LidarConfig:
    return CONFIG


def set_config(cfg: LidarConfig):
    """모듈 전역 설정을 바꾼다. bev_render처럼 모듈 함수를 직접 쓰는 쪽까지 같이 따라오게."""
    global CONFIG
    CONFIG = cfg
    return CONFIG


def load_config(path=None, quiet: bool = True) -> LidarConfig:
    """설정 파일을 읽어 전역에 적용하고 돌려준다. 없으면 기본값 그대로."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        if not quiet:
            print(f"[!] LiDAR 설정 파일이 없습니다({path}) — 기본값(오프셋 0°, 높이 "
                  f"{MOUNT_HEIGHT_M}m)으로 진행합니다. lidar_calibrate.py로 실측하세요.")
        return CONFIG
    with open(path) as f:
        data = json.load(f)
    cfg = set_config(LidarConfig.from_dict(data))
    if not quiet:
        print(f"[*] LiDAR 설정 적용: {path}  오프셋={cfg.mount_offset_deg:+.1f}° "
              f"높이={cfg.mount_height_m:.2f}m")
    return cfg


def save_config(cfg: LidarConfig, path=None) -> Path:
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"_note": "lidar_calibrate.py가 쓰는 파일. 장착을 다시 하면 재측정할 것.",
            "_measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            **cfg.to_dict()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


# 설정 파일이 있으면 import 시점에 자동 적용한다 — live_demo / bev_render / calibrate가
# 서로 다른 장착값을 쓰는 사고를 막기 위한 것. 깨진 파일 하나가 전체를 못 띄우게 만들지는 않는다.
try:
    load_config()
except Exception as _e:  # noqa: BLE001
    print(f"[!] LiDAR 설정 파일을 읽지 못했습니다({_e}) — 기본값으로 진행합니다.")


def _wrap180(deg):
    return (np.asarray(deg, dtype=np.float64) + 180.0) % 360.0 - 180.0


def angle_window_deg(sigma_deg=None, cfg: LidarConfig = None) -> float:
    """방향 불확실성(sigma)에서 각도 창의 반폭(±deg)을 만든다.

    젯슨이 보내는 sigma는 수음 구간 안 DoA 8회 읽기의 실측 흩어짐이다(README 인터페이스 표).
    그 값이 곧 "이 방향을 어디까지 믿을 수 있나"라서, 창 폭의 근거로 쓰기에 가장 적절하다.
    sigma가 없으면(구버전 패킷) 보수적으로 최대 폭을 쓴다.
    """
    cfg = cfg or CONFIG
    if sigma_deg is None or not np.isfinite(sigma_deg) or sigma_deg <= 0:
        return float(cfg.angle_margin_max_deg)
    return float(min(cfg.angle_margin_max_deg,
                     max(cfg.angle_margin_min_deg, cfg.sigma_k * float(sigma_deg))))


class LidarScanner:
    """VLP-16 UDP 스트림을 배경 스레드로 계속 읽어 최신 스캔만 들고 있는다.

    포인트는 차량 좌표계 기준 (x, y, z, range_m, azimuth_deg) 5열 numpy 배열.
    스캔마다 scan_id가 1 올라간다 — 추적기가 같은 스캔을 독립 관측으로 두 번 세지 않으려고.
    """

    def __init__(self, mount_offset_deg: float = None, port: int = VLP16_UDP_PORT):
        # None이면 전역 설정(CONFIG)을 따른다 — 캘리브레이션 결과가 바로 반영되도록.
        self._offset_override = mount_offset_deg
        self.port = port
        self._lock = threading.Lock()
        self._latest = np.empty((0, 5), dtype=np.float32)
        self._thread = None
        self._stop = threading.Event()
        # 수신 스레드가 조용히 죽으면 "라이다는 붙었는데 아무것도 안 잡힘"으로 보인다.
        # 실패 사유를 남겨 호출측(live_demo.py)이 폴백을 실제로 탈 수 있게 한다.
        self._error = None
        self._scans = 0
        self._last_scan_at = 0.0

    @property
    def mount_offset_deg(self) -> float:
        return (CONFIG.mount_offset_deg if self._offset_override is None
                else self._offset_override)

    def set_mount_offset(self, offset_deg: float):
        """현장 캘리브레이션 중에 오프셋을 바꿔가며 확인할 때 쓴다(재시작 불필요)."""
        self._offset_override = float(offset_deg)

    def start(self, port: int = None):
        import velodyne_decoder as vd  # noqa: F401  (여기서 import해 --selftest는 의존성 없이 동작)

        if port is not None:
            self.port = port
        self._thread = threading.Thread(target=self._run, args=(self.port,), daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _fail(self, message: str):
        with self._lock:
            self._error = message
        print(f"[!] {message}")

    def _run(self, port: int):
        import velodyne_decoder as vd

        sock = None
        try:
            # ⚠️ model은 문자열("VLP-16")이 아니라 Model enum이어야 한다 (velodyne_decoder 3.x).
            decoder = vd.StreamDecoder(vd.Config(model=vd.Model.VLP16))
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", port))
            sock.settimeout(0.5)
        except Exception as e:  # noqa: BLE001
            self._fail(f"LiDAR 수신 초기화 실패: {e}")
            if sock is not None:
                sock.close()
            return

        try:
            while not self._stop.is_set():
                try:
                    data, _ = sock.recvfrom(2048)
                except socket.timeout:
                    continue
                except OSError as e:
                    self._fail(f"LiDAR 소켓 오류: {e}")
                    break

                if len(data) != VLP16_DATA_PACKET_BYTES:
                    continue  # 위치 패킷 등은 디코더에 넣지 않는다

                try:
                    # ⚠️ 메서드는 decode(stamp, data). decode_packet은 존재하지 않으며 인자 순서도 반대.
                    result = decoder.decode(time.time(), data)
                except Exception as e:  # noqa: BLE001
                    self._fail(f"LiDAR 디코딩 실패: {e}")
                    break

                # 한 바퀴(약 76패킷)가 차야 스캔 1개가 나온다 — 대부분의 호출은 None을 돌려준다.
                if result is None:
                    continue
                _, points = result
                points = np.asarray(points)
                if points.size == 0:
                    continue

                # velodyne_decoder 3.x 컬럼: x, y, z, intensity, time, azimuth_col, ring, return_type
                x, y, z = points[:, 0], points[:, 1], points[:, 2]
                rng = np.sqrt(x**2 + y**2)  # BEV 지도용 수평거리 (3D 직선거리 아님)
                # 무반사(거리 0)와 차체 자체 반사를 버린다. 안 버리면 원점 부근에 점이 뭉쳐
                # "0m에 물체"라는 가짜 클러스터가 생긴다.
                keep = rng >= CONFIG.range_min_m
                if not np.any(keep):
                    continue
                x, y, z, rng = x[keep], y[keep], z[keep], rng[keep]
                azimuth = _wrap180(np.degrees(np.arctan2(y, x)) - self.mount_offset_deg)
                with self._lock:
                    self._latest = np.column_stack([x, y, z, rng, azimuth]).astype(np.float32)
                    self._scans += 1
                    self._last_scan_at = time.time()
        finally:
            sock.close()

    def latest_points(self):
        with self._lock:
            return self._latest.copy()

    def snapshot(self):
        """(포인트 배열, scan_id) — 추적기에 넘길 때 쓴다. 같은 scan_id면 같은 관측이다."""
        with self._lock:
            return self._latest.copy(), self._scans

    def healthy(self, max_age_sec: float = 1.0) -> bool:
        """최근 max_age_sec 안에 스캔이 갱신됐고 오류가 없으면 True.

        live_demo.py는 매 감지마다 이걸 확인해야 한다 — start()가 성공했다는 것만으로는
        수신 스레드가 살아있다는 보장이 안 된다.
        """
        with self._lock:
            if self._error is not None:
                return False
            return self._last_scan_at > 0 and (time.time() - self._last_scan_at) <= max_age_sec

    def status(self):
        with self._lock:
            age = time.time() - self._last_scan_at if self._last_scan_at else float("inf")
            return {"error": self._error, "scans": self._scans, "age": age,
                    "points": self._latest.shape[0]}


def filter_ground(points: np.ndarray, mount_height_m: float = None,
                  cfg: LidarConfig = None) -> np.ndarray:
    """노면과 머리 위 구조물을 빼고 "도로 위 물체" 높이대의 점만 남긴다.

    z는 라이다 원점 기준이므로 노면은 z = -mount_height_m 부근에 있다.

    VLP-16은 수직 FOV가 ±15°다. 1.7m 루프랙에 올리면 최하단 빔이 1.7/tan(15°) ≈ 6.3m 앞
    노면에 닿는다 — 하필 사각지대 반경 6m 바로 바깥이다. 그래서 지면을 안 걸러내면 어느
    방향을 조회하든 "6.3m에 노면"이 최근접으로 잡혀, 30m 앞 구급차 대신 매번 아스팔트까지의
    거리를 보고하게 된다. (2026-09-01 실측: 필터 없을 때 모든 방향에서 6.000m 고정 반환)
    """
    cfg = cfg or CONFIG
    if points.shape[0] == 0:
        return points
    h = cfg.mount_height_m if mount_height_m is None else mount_height_m
    z = points[:, 2]
    above_ground = z > (-h + cfg.min_object_height_m)
    below_canopy = z < cfg.max_object_height_m
    return points[above_ground & below_canopy]


def nearest_cluster(ranges: np.ndarray, gap_m: float = None, min_points: int = None):
    """거리값들을 덩어리로 묶어 가장 가까운 "충분히 큰" 덩어리를 돌려준다.

    반환: (대표거리, 점 개수) 또는 None
    대표거리는 최솟값이 아니라 중앙값 — 노이즈 한 점이 거리를 끌어당기지 못하게.

    (2026-10-15) 거리축만 보는 이 함수는 cluster_candidates()로 대체됐다. 외부에서 쓰던
    코드가 있을 수 있어 남겨둔다.
    """
    gap_m = CONFIG.cluster_gap_m if gap_m is None else gap_m
    min_points = CONFIG.min_cluster_points if min_points is None else min_points
    if ranges.size == 0:
        return None
    ordered = np.sort(ranges)
    splits = np.where(np.diff(ordered) > gap_m)[0] + 1
    for group in np.split(ordered, splits):   # np.split은 가까운 순서를 유지한다
        if group.size >= min_points:
            return float(np.median(group)), int(group.size)
    return None


def cluster_candidates(window: np.ndarray, theta_deg: float,
                       cfg: LidarConfig = None) -> list:
    """각도 창 안의 점(지면 제거 완료)을 물체 후보 목록으로 바꾼다. 가까운 순 정렬.

    거리축으로 한 번 끊고, 각 덩어리를 **방위축으로 다시** 끊는다. 거리축만 끊던 이전
    버전은 같은 거리대에 있는 창 왼쪽 끝 물체와 오른쪽 끝 물체를 한 덩어리로 합쳐서,
    그 중앙값(= 둘 사이 허공)을 거리로 보고할 수 있었다.

    후보마다 방향 오차·폭·거리 퍼짐까지 같이 낸다 — 점수로 고를 때와, 화면에 라이다가
    실제로 본 방위를 그릴 때 쓴다.
    """
    cfg = cfg or CONFIG
    out = []
    if window is None or window.shape[0] == 0:
        return out

    rel = _wrap180(window[:, 4] - theta_deg)
    order = np.argsort(window[:, 3])
    rngs, rels, zs = window[order, 3], rel[order], window[order, 2]

    for g_r, g_rel, g_z in zip(*(np.split(a, np.where(np.diff(rngs) > cfg.cluster_gap_m)[0] + 1)
                                 for a in (rngs, rels, zs))):
        if g_r.size == 0:
            continue
        a_order = np.argsort(g_rel)
        a_r, a_rel, a_z = g_r[a_order], g_rel[a_order], g_z[a_order]
        splits = np.where(np.diff(a_rel) > cfg.cluster_azimuth_gap_deg)[0] + 1
        for c_r, c_rel, c_z in zip(*(np.split(a, splits) for a in (a_r, a_rel, a_z))):
            if c_r.size < cfg.min_cluster_points:
                continue
            distance = float(np.median(c_r))
            bearing_err = float(np.median(c_rel))
            az_span = float(c_rel.max() - c_rel.min())
            out.append({
                "distance_m": distance,
                "azimuth_deg": float(_wrap180(theta_deg + bearing_err)),
                "bearing_error_deg": bearing_err,
                "points": int(c_r.size),
                "range_span_m": float(c_r.max() - c_r.min()),
                # 현(弦) 길이 — 차량 폭(~1.8m)과 비교해 "물체인가 면인가"를 가늠하는 데 쓴다
                "width_m": 2.0 * distance * math.sin(math.radians(max(az_span, 0.0)) / 2.0),
                "height_span_m": float(c_z.max() - c_z.min()),
            })
    out.sort(key=lambda c: c["distance_m"])
    return out


def score_candidate(cand: dict, window_deg: float, cfg: LidarConfig = None) -> float:
    """후보 점수. **낮을수록 좋다.**

    우리가 찾는 것은 "가장 가까운 물체"가 아니라 "그 소리를 낸 물체"다. 그래서 방향 일치에
    가장 큰 가중치를 주고, 가까움은 보조로만 쓴다 — 창 가장자리에 걸친 가드레일(가깝지만
    방향이 어긋남)보다 창 중앙의 구급차(멀지만 방향이 맞음)를 고르게 하려는 것.
    거리로 길게 퍼진 덩어리(벽·연석·가드레일)에는 벌점을 준다.
    """
    cfg = cfg or CONFIG
    ang = abs(cand["bearing_error_deg"]) / max(window_deg, 1e-6)
    far = min(cand["distance_m"] / max(cfg.range_max_m, 1e-6), 1.0)
    bulk = min(cand["points"] / float(max(4 * cfg.min_cluster_points, 1)), 1.0)
    spread = min(cand["range_span_m"] / max(cfg.wall_range_span_m, 1e-6), 1.0)
    return (cfg.w_bearing * ang + cfg.w_distance * far
            + cfg.w_spread * spread - cfg.w_points * bulk)


def match_distance(points: np.ndarray, theta_deg: float,
                   angle_margin_deg: float = None,
                   blind_radius_m: float = None,
                   mount_height_m: float = None,
                   sigma_deg: float = None,
                   cfg: LidarConfig = None):
    """방향(theta_deg, 차량 좌표계 0=전방)에서 들린 음원까지의 거리를 찾는다.

    순서: 각도 창(sigma 기반) → 지면/천장 제거 → 거리·방위 클러스터링 → 후보 점수 → 선택.

    반환:
        {"distance_m": float, "blind": False, ...} — 사각지대 밖에서 물체 확정
        {"distance_m": None,  "blind": True,  ...} — 사각지대(6m 이내) 물체
        None — 그 방향에 유효한 물체가 없음(위치 미확정)

    dict에는 거리 외에 라이다가 **실제로 본 방위**(azimuth_deg), 음향 방향과의 차이
    (bearing_error_deg), 쓰인 창 폭(window_deg), 후보 수(candidates)가 함께 담긴다 —
    화면은 소리 방향이 아니라 라이다 방위에 물체를 그려야 하고, 두 방향의 차이는
    마이크 오프셋 실측(lidar_calibrate.py --audio-offset)의 재료가 된다.

    사각지대 우선 원칙: 창 안 가장 가까운 유효 후보가 6m 이내면, 6m 밖에 더 좋은 후보가
    있어도 blind로 보고한다 — 가까운 쪽이 더 위급하다(ui_state_spec.md).
    """
    cfg = cfg or CONFIG
    if points is None or points.shape[0] == 0:
        return None

    window_deg = (angle_window_deg(sigma_deg, cfg) if angle_margin_deg is None
                  else float(angle_margin_deg))
    blind_radius = cfg.blind_radius_m if blind_radius_m is None else blind_radius_m

    rel = _wrap180(points[:, 4] - theta_deg)
    candidates_pts = points[np.abs(rel) <= window_deg]
    if candidates_pts.shape[0] < MIN_POINTS:
        return None

    candidates_pts = filter_ground(candidates_pts, mount_height_m, cfg)
    if candidates_pts.shape[0] < MIN_POINTS:
        return None

    cands = [c for c in cluster_candidates(candidates_pts, theta_deg, cfg)
             if c["distance_m"] <= cfg.range_max_m]
    if not cands:
        return None

    nearest = cands[0]
    if nearest["distance_m"] < blind_radius:
        return {"distance_m": None, "blind": True, "points": nearest["points"],
                "azimuth_deg": nearest["azimuth_deg"],
                "bearing_error_deg": nearest["bearing_error_deg"],
                "width_m": nearest["width_m"], "range_span_m": nearest["range_span_m"],
                "window_deg": window_deg, "candidates": len(cands)}

    best = min(cands, key=lambda c: score_candidate(c, window_deg, cfg))
    return {"distance_m": best["distance_m"], "blind": False, "points": best["points"],
            "azimuth_deg": best["azimuth_deg"],
            "bearing_error_deg": best["bearing_error_deg"],
            "width_m": best["width_m"], "range_span_m": best["range_span_m"],
            "window_deg": window_deg, "candidates": len(cands)}


class TargetTracker:
    """소리 클래스별로 매칭된 물체를 스캔을 넘어가며 추적한다.

    단발 매칭만 쓰면 (1) 거리가 스캔마다 튀고, (2) 한 스캔 놓치면 화면의 숫자가 깜빡 사라지고,
    (3) 접근 중인지 멀어지는지를 알 수 없다. 추적기는 그 셋을 메운다.

    - 같은 scan_id는 한 번만 센다 — 표시 루프가 30Hz로 물어도 라이다는 10Hz라서,
      그러지 않으면 같은 관측 하나로 "확정"이 돼버린다.
    - 예측 거리에서 track_gate_m 이상 벗어난 매칭은 다른 물체로 보고 추적을 새로 시작한다.
    - 놓친 구간은 track_coast_sec까지 마지막 값을 stale=True로 유지한다.
    - closing_mps는 최근 구간 거리의 1차 회귀 기울기(부호 반전)다. **상대 속도**이고,
      자차 속도를 모르므로 정지 구조물도 접근으로 보일 수 있다.

    live_demo는 감지 스레드와 표시 루프 양쪽에서 호출하므로 내부에 락을 둔다.
    """

    def __init__(self, cfg: LidarConfig = None):
        self._cfg = cfg
        self._lock = threading.Lock()
        self._tracks = {}

    @property
    def cfg(self) -> LidarConfig:
        return self._cfg or CONFIG

    def reset(self):
        with self._lock:
            self._tracks.clear()

    def drop(self, class_name: str):
        with self._lock:
            self._tracks.pop(class_name, None)

    def get(self, class_name: str):
        with self._lock:
            tr = self._tracks.get(class_name)
            return None if tr is None else self._as_result(tr, time.time())

    def update(self, class_name: str, theta_deg: float, points: np.ndarray,
               sigma_deg: float = None, scan_id: int = None, now: float = None,
               window_deg: float = None, follow: bool = True):
        """새 스캔으로 추적을 갱신하고 결과 dict(또는 None)를 돌려준다.

        결과는 match_distance()의 dict에 추적 정보를 더한 형태라, alert_policy가 보던
        "distance_m / blind" 계약을 그대로 만족한다.

        window_deg: 각도 창을 직접 지정한다(없으면 sigma에서 만든다). **카메라가 YOLO로
            대상을 지목한 경우**가 이 경로다 — 박스 폭에서 나온 2~6° 창은 소리 방향에서
            나오는 8~30° 창보다 훨씬 좁아서, 옆 차선 차량이 아니라 지목된 그 물체를 잰다.
        follow: 추적 중이면 요청 방위 대신 **직전에 라이다가 본 방위**에서 이어 찾는다.
            좁은 창을 쓸 때 꼭 필요하다 — 카메라 검출은 젯슨 패킷 주기(약 1초)로만
            갱신되는데 대상은 그 사이에도 움직이므로, 고정 방위로 좁게 조회하면 놓친다.
            원 요청에서 track_follow_max_deg 이상은 벗어나지 않게 묶어, 추적이 엉뚱한
            물체로 흘러가는 것을 막는다.
        """
        cfg = self.cfg
        now = time.time() if now is None else now

        with self._lock:
            tr = self._tracks.get(class_name)
            if tr is not None and now - tr["last_at"] > cfg.track_drop_sec:
                tr = None
                self._tracks.pop(class_name, None)

            # 같은 스캔을 다시 물어본 경우 — 독립 관측이 아니므로 세지 않고 현재 상태만 준다.
            if tr is not None and scan_id is not None and scan_id == tr.get("scan_id"):
                return self._as_result(tr, now)

            # 추적 중이면 마지막으로 본 방위에서 이어 찾는다 (위 follow 설명 참고)
            query_deg = float(theta_deg)
            if follow and tr is not None and not tr["blind"]:
                last_az = tr["match"].get("azimuth_deg")
                if last_az is not None:
                    drift = float(_wrap180(last_az - theta_deg))
                    drift = max(-cfg.track_follow_max_deg,
                                min(cfg.track_follow_max_deg, drift))
                    query_deg = float(_wrap180(theta_deg + drift))

            match = match_distance(points, query_deg, angle_margin_deg=window_deg,
                                   sigma_deg=sigma_deg, cfg=cfg)
            if match is not None:
                # 방향 오차는 **원 요청 방향**(소리 방향 또는 카메라 방위) 기준으로 다시 쓴다.
                # follow로 조회 방위가 옮겨가도, 화면과 마이크 보정이 봐야 할 기준은
                # "우리가 물어본 방향과 라이다가 본 방향의 차이"다.
                match["query_deg"] = query_deg
                match["bearing_error_deg"] = float(_wrap180(match["azimuth_deg"] - theta_deg))

            if match is None:
                if tr is not None and now - tr["last_at"] <= cfg.track_coast_sec:
                    return self._as_result(tr, now)   # 잠깐 놓친 것 — 마지막 값 유지
                self._tracks.pop(class_name, None)
                return None

            if tr is not None:
                tr = self._advance(tr, match, theta_deg, scan_id, now, cfg)
            else:
                tr = self._new_track(class_name, match, theta_deg, scan_id, now)
            self._tracks[class_name] = tr
            return self._as_result(tr, now)

    # ---- 내부 -----------------------------------------------------------------
    @staticmethod
    def _new_track(class_name, match, theta_deg, scan_id, now):
        return {"class_name": class_name, "match": match, "theta_deg": theta_deg,
                "distance_m": match["distance_m"], "blind": match["blind"],
                "samples": ([] if match["distance_m"] is None
                            else [(now, match["distance_m"])]),
                "hits": 1, "first_at": now, "last_at": now, "scan_id": scan_id}

    def _advance(self, tr, match, theta_deg, scan_id, now, cfg):
        dt = max(now - tr["last_at"], 1e-3)
        closing = self._closing(tr, cfg) or 0.0
        same = True
        if match["distance_m"] is not None and tr["distance_m"] is not None:
            predicted = tr["distance_m"] - closing * dt
            same = abs(match["distance_m"] - predicted) <= cfg.track_gate_m
        elif match["blind"] != tr["blind"]:
            # 6m 경계를 넘나든 것은 같은 물체일 수 있으나 거리 이력은 못 이어 붙인다
            same = True
            tr = dict(tr, samples=[])

        if not same:
            return self._new_track(tr["class_name"], match, theta_deg, scan_id, now)

        distance = match["distance_m"]
        if distance is not None and tr["distance_m"] is not None:
            a = cfg.track_ema_alpha
            distance = a * distance + (1.0 - a) * tr["distance_m"]

        samples = list(tr["samples"])
        if distance is not None:
            samples.append((now, distance))
            samples = [(t, d) for t, d in samples if now - t <= cfg.closing_window_sec]

        return {**tr, "match": match, "theta_deg": theta_deg, "distance_m": distance,
                "blind": match["blind"], "samples": samples,
                "hits": tr["hits"] + 1, "last_at": now, "scan_id": scan_id}

    def _closing(self, tr, cfg=None):
        """접근 속도(m/s, 양수 = 가까워짐). 표본이 모자라면 None."""
        cfg = cfg or self.cfg
        samples = tr.get("samples") or []
        if len(samples) < cfg.closing_min_samples:
            return None
        t = np.array([s[0] for s in samples], dtype=np.float64)
        d = np.array([s[1] for s in samples], dtype=np.float64)
        if t[-1] - t[0] < cfg.closing_min_span_sec:
            return None
        slope = float(np.polyfit(t - t[0], d, 1)[0])
        return -slope

    def _measure_dt(self, tr) -> float:
        """관측 간격(초) 추정 — 보정량 계산에 쓴다. 표본이 없으면 VLP-16 기본 10Hz."""
        samples = tr.get("samples") or []
        if len(samples) < 2:
            return 0.1
        gaps = np.diff([t for t, _ in samples])
        return float(np.median(gaps)) if gaps.size else 0.1

    def _as_result(self, tr, now):
        cfg = self.cfg
        closing = self._closing(tr, cfg)
        ttc = None
        if closing is not None and closing > 0.5 and tr["distance_m"]:
            ttc = tr["distance_m"] / closing

        # ---- 표시 지연 보정 ----------------------------------------------------
        # 보이는 거리가 실제보다 **멀다**. 두 가지가 겹친다.
        #   · 측정 나이 : 마지막 스캔 이후 흐른 시간만큼 대상이 더 다가왔다
        #   · 평활화    : EMA의 정상상태 지연 = (1-a)/a x 관측 간격
        # 접근 속도를 알면 그만큼 되돌릴 수 있다. 표본이 모자라 closing이 없으면 보정하지
        # 않고(모르면 손대지 않는다), 과보정을 막으려 상한을 둔다.
        # ⚠️ 보정값은 **표시용**이고 samples에는 넣지 않는다 — 넣으면 보정이 다음 기울기에
        #    먹혀들어 스스로를 증폭시킨다.
        distance = tr["distance_m"]
        lag_sec = 0.0
        if (distance is not None and closing is not None and not tr["blind"]
                and cfg.lag_compensation_max_m > 0):
            a = max(cfg.track_ema_alpha, 1e-6)
            lag_sec = (now - tr["last_at"]) + ((1.0 - a) / a) * self._measure_dt(tr)
            shift = max(-cfg.lag_compensation_max_m,
                        min(cfg.lag_compensation_max_m, closing * lag_sec))
            distance = max(0.0, distance - shift)
            if closing > 0.5:
                ttc = distance / closing

        out = dict(tr["match"])
        out.update({
            "distance_m": distance, "distance_raw_m": tr["distance_m"],
            "lag_sec": lag_sec, "blind": tr["blind"],
            "closing_mps": closing, "ttc_s": ttc,
            "hits": tr["hits"], "confirmed": tr["hits"] >= cfg.track_confirm_hits,
            "age_sec": now - tr["first_at"],
            "stale": (now - tr["last_at"]) > 1e-3,
            "last_age_sec": now - tr["last_at"],
        })
        return out


# ---- 테스트용 장면 생성 -------------------------------------------------------------
def _make_fake_points(theta_deg: float, distance_m: float, n: int = 20) -> np.ndarray:
    """--selftest용 가짜 포인트: theta_deg 방향, distance_m 거리에 물체 하나를 흩뿌린다."""
    rng_gen = np.random.default_rng(3)
    jitter = rng_gen.uniform(-1.5, 1.5, n)
    az = theta_deg + jitter
    rng = distance_m + rng_gen.uniform(-0.2, 0.2, n)
    x = rng * np.cos(np.radians(az))
    y = rng * np.sin(np.radians(az))
    z = np.zeros(n)
    return np.column_stack([x, y, z, rng, _wrap180(az)]).astype(np.float32)


def _make_road_scene(mount_height_m: float = 1.7, car_range_m: float = 20.0,
                     car_azimuth_deg: float = 0.0) -> np.ndarray:
    """VLP-16 실제 기하로 "평평한 노면 + 차량 1대" 장면을 합성한다 (회귀 테스트용).

    빔 앙각은 -15~+15도 2도 간격 16개. 하향 빔은 노면에 r = h/tan|angle| 에서 닿는다 —
    1.7m 장착이면 최하단 빔이 6.34m로, 사각지대 반경 6m 바로 바깥이다. 지면 제거가 없으면
    이 노면 반사가 항상 최근접으로 잡혀 차량 대신 아스팔트 거리를 보고하게 된다.
    """
    pts = []
    for elev in range(-15, 16, 2):
        if elev >= 0:
            continue  # 상향 빔은 평지에서 아무것도 맞히지 않는다
        r = mount_height_m / np.tan(np.radians(abs(elev)))
        if r > 100:
            continue
        for az in np.arange(-180, 180, 0.4):
            x = r * np.cos(np.radians(az))
            y = r * np.sin(np.radians(az))
            pts.append([x, y, -mount_height_m, r, az])

    pts.extend(_vehicle_points(car_range_m, car_azimuth_deg, mount_height_m))
    return np.array(pts, dtype=np.float32)


def _vehicle_points(range_m: float, azimuth_deg: float, mount_height_m: float = 1.7,
                    width_m: float = 1.8):
    """거리 range_m, 방위 azimuth_deg에 폭 width_m 차량 1대의 리턴을 만든다."""
    half_az = np.degrees(np.arctan((width_m / 2.0) / range_m))
    pts = []
    for az in np.arange(azimuth_deg - half_az, azimuth_deg + half_az, 0.2):
        for z in np.arange(-mount_height_m + 0.2, -mount_height_m + 1.5, 0.12):
            x = range_m * np.cos(np.radians(az))
            y = range_m * np.sin(np.radians(az))
            pts.append([x, y, z, range_m, float(_wrap180(az))])
    return pts


def _make_guardrail_scene(mount_height_m: float = 1.7) -> np.ndarray:
    """창 가장자리에 가드레일(가깝지만 방향 어긋남) + 창 중앙에 차량(멀지만 방향 맞음).

    "가장 가까운 것"을 고르던 이전 규칙이면 가드레일을 집는다. 소리를 낸 쪽은 차량이다.
    """
    pts = []
    for r in np.arange(8.0, 18.0, 0.25):           # 거리로 길게 퍼진 면 = 가드레일
        for z in np.arange(-mount_height_m + 0.35, -mount_height_m + 0.8, 0.12):
            for az in np.arange(20.0, 24.0, 0.4):
                x = r * np.cos(np.radians(az))
                y = r * np.sin(np.radians(az))
                pts.append([x, y, z, r, az])
    pts.extend(_vehicle_points(26.0, 1.0, mount_height_m))
    return np.array(pts, dtype=np.float32)


# ---- 셀프테스트 -------------------------------------------------------------------
def run_road_scene_test() -> bool:
    """지면 제거 회귀 테스트 — 이게 깨지면 모든 방향이 노면 거리(6.34m)로 돌아간다."""
    scene = _make_road_scene()
    ok = True

    got = match_distance(scene, 0.0, mount_height_m=1.7)
    hit = got is not None and not got["blind"] and abs(got["distance_m"] - 20.0) < 1.0
    ok &= hit
    print(f"  {'노면+20m차량, 차량 방향':30s} expected=20.0m  got={_fmt(got)}  [{_pf(hit)}]")

    for theta in (90.0, 180.0):
        got = match_distance(scene, theta, mount_height_m=1.7)
        # 차량이 없는 방향에는 노면만 있으므로 반드시 "없음"이어야 한다
        good = got is None
        ok &= good
        print(f"  {f'노면만 ({theta:.0f}도)':30s} expected=None    got={_fmt(got)}  [{_pf(good)}]")
    return ok


def run_window_test() -> bool:
    """sigma로 각도 창이 좁아지고 넓어지는지, 창 폭이 매칭 결과를 가르는지."""
    ok = True
    cases = [(3.0, CONFIG.angle_margin_min_deg), (7.0, 14.0), (40.0, CONFIG.angle_margin_max_deg),
             (None, CONFIG.angle_margin_max_deg)]
    for sigma, want in cases:
        got = angle_window_deg(sigma)
        good = abs(got - want) < 1e-6
        ok &= good
        print(f"  {f'창 폭(sigma={sigma})':30s} expected=±{want:.1f}도  got=±{got:.1f}도  [{_pf(good)}]")

    # 방향이 또렷하면(sigma=3 → ±8°) 18° 떨어진 물체는 집지 않는다.
    scene = _make_fake_points(18.0, 24.0, n=40)
    tight = match_distance(scene, 0.0, sigma_deg=3.0)
    loose = match_distance(scene, 0.0, sigma_deg=15.0)
    good = tight is None and loose is not None
    ok &= good
    print(f"  {'좁은 창은 18도 밖 물체 제외':30s} tight={_fmt(tight)} loose={_fmt(loose)}  [{_pf(good)}]")
    return ok


def run_candidate_test() -> bool:
    """방위축 분리와 후보 점수 — "가까운 것"이 아니라 "방향이 맞는 것"을 고르는지."""
    ok = True

    # 같은 거리대, 창의 양 끝에 물체 둘. 거리축만 끊으면 한 덩어리가 된다.
    two = np.concatenate([_make_fake_points(-20.0, 15.0, n=40),
                          _make_fake_points(20.0, 15.2, n=40)])
    cands = cluster_candidates(filter_ground(two), 0.0)
    good = len(cands) == 2
    ok &= good
    print(f"  {'같은 거리·다른 방위 → 2개 후보':30s} got={len(cands)}개  [{_pf(good)}]")

    # 가드레일(8m, 방향 +22°) vs 차량(26m, 방향 +1°) — 방향이 맞는 차량을 골라야 한다
    got = match_distance(_make_guardrail_scene(), 0.0, sigma_deg=15.0, mount_height_m=1.7)
    good = got is not None and not got["blind"] and abs(got["distance_m"] - 26.0) < 2.0
    ok &= good
    print(f"  {'가드레일보다 방향 맞는 차량':30s} expected=26m  got={_fmt(got)}  [{_pf(good)}]")

    # 좁은 창(카메라 지목)이 **실제로** 기여하는가 — 대조군 없이는 "어차피 맞았을 일"을
    # 검증했다고 착각하기 쉽다. 같은 장면에서 두 경로를 200번씩 돌려 비교한다.
    #   배치: 대상(사이렌) -15°/25m, 미끼(주차 차량) -24°/12m
    #   소리 경로는 theta가 σ=6으로 흔들려 미끼가 창 중앙에 오는 날이 있다 → 때때로 틀린다
    #   카메라 경로는 박스 방위(±3.4°)로 고정 조회 → 항상 대상
    scene2 = np.concatenate([
        _make_road_scene(mount_height_m=1.7, car_range_m=25.0, car_azimuth_deg=-15.0),
        np.array(_vehicle_points(12.0, -24.0, 1.7), dtype=np.float32)])
    gen = np.random.default_rng(0)
    audio_hit = cam_hit = 0
    for _ in range(200):
        theta = -15.0 + gen.normal(0.0, 6.0)
        m_a = match_distance(scene2, theta, sigma_deg=6.0, mount_height_m=1.7)
        m_c = match_distance(scene2, -15.0, angle_margin_deg=3.4, mount_height_m=1.7)
        audio_hit += bool(m_a and not m_a["blind"]
                          and abs(_wrap180(m_a["azimuth_deg"] + 15.0)) < 5.0)
        cam_hit += bool(m_c and not m_c["blind"]
                        and abs(_wrap180(m_c["azimuth_deg"] + 15.0)) < 5.0)
    good = cam_hit == 200 and audio_hit < 200
    ok &= good
    print(f"  {'카메라 지목이 미끼를 걸러낸다':30s} "
          f"카메라 {cam_hit}/200, 소리 {audio_hit}/200  [{_pf(good)}]")

    # 라이다가 본 방위를 같이 돌려주는지 (화면이 소리 방향 대신 이걸 쓴다)
    got = match_distance(_make_fake_points(14.0, 22.0, n=40), 0.0, sigma_deg=15.0)
    good = got is not None and abs(got["azimuth_deg"] - 14.0) < 3.0
    ok &= good
    print(f"  {'라이다 실측 방위 반환':30s} expected=14도  "
          f"got={None if got is None else round(got['azimuth_deg'], 1)}도  [{_pf(good)}]")
    return ok


def run_tracker_test() -> bool:
    """추적기 — 확정 조건, 같은 스캔 중복 차단, 접근 속도, coast."""
    ok = True
    tracker = TargetTracker()
    t0 = 1000.0

    first = tracker.update("siren", 0.0, _make_fake_points(0.0, 40.0, n=40),
                           sigma_deg=6.0, scan_id=1, now=t0)
    good = first is not None and not first["confirmed"] and first["hits"] == 1
    ok &= good
    print(f"  {'첫 스캔은 미확정':30s} hits={None if first is None else first['hits']}  [{_pf(good)}]")

    again = tracker.update("siren", 0.0, _make_fake_points(0.0, 40.0, n=40),
                           sigma_deg=6.0, scan_id=1, now=t0 + 0.03)
    good = again is not None and again["hits"] == 1   # 같은 scan_id는 안 센다
    ok &= good
    print(f"  {'같은 스캔 재조회는 hits 유지':30s} hits={again['hits']}  [{_pf(good)}]")

    # 10Hz로 1.5초 동안 40m → 25m (약 10m/s 접근)
    res = None
    for i in range(2, 17):
        t = t0 + 0.1 * (i - 1)
        dist = 40.0 - 1.0 * (i - 1)
        res = tracker.update("siren", 0.0, _make_fake_points(0.0, dist, n=40),
                             sigma_deg=6.0, scan_id=i, now=t)
    good = res is not None and res["confirmed"] and res["closing_mps"] is not None \
        and 6.0 < res["closing_mps"] < 13.0
    ok &= good
    print(f"  {'접근 속도 추정(약 10m/s)':30s} "
          f"got={None if res is None else (None if res['closing_mps'] is None else round(res['closing_mps'], 1))}m/s"
          f"  [{_pf(good)}]")

    good = res is not None and res["ttc_s"] is not None and 1.0 < res["ttc_s"] < 6.0
    ok &= good
    print(f"  {'TTC 산출':30s} "
          f"got={None if res is None or res['ttc_s'] is None else round(res['ttc_s'], 1)}s  [{_pf(good)}]")

    # 한 스캔 놓침 → coast로 마지막 거리 유지
    coast = tracker.update("siren", 0.0, np.empty((0, 5), dtype=np.float32),
                           sigma_deg=6.0, scan_id=18, now=t0 + 1.7)
    good = coast is not None and coast["stale"] and coast["distance_m"] is not None
    ok &= good
    print(f"  {'놓친 스캔은 coast(유지)':30s} got={_fmt(coast)}  [{_pf(good)}]")

    # 오래 놓치면 버린다
    dropped = tracker.update("siren", 0.0, np.empty((0, 5), dtype=np.float32),
                             sigma_deg=6.0, scan_id=19, now=t0 + 9.0)
    good = dropped is None
    ok &= good
    print(f"  {'오래 놓치면 추적 폐기':30s} got={_fmt(dropped)}  [{_pf(good)}]")

    # 카메라가 지목한 좁은 창(±2.5°)으로도, 대상이 옆으로 흘러가면 follow가 따라붙는다.
    # (YOLO 박스는 1초에 한 번만 갱신되는데 대상은 그 사이에도 움직인다)
    res_follow = res_fixed = None
    tr_follow, tr_fixed = TargetTracker(), TargetTracker()
    for i in range(1, 13):
        az = 1.5 * i                      # 매 스캔 1.5°씩 옆으로 — 12스캔이면 18°
        scene = _make_fake_points(az, 30.0 - 0.5 * i, n=40)
        res_follow = tr_follow.update("siren", 0.0, scene, sigma_deg=6.0, scan_id=i,
                                      now=t0 + 0.1 * i, window_deg=2.5)
        res_fixed = tr_fixed.update("siren", 0.0, scene, sigma_deg=6.0, scan_id=i,
                                    now=t0 + 0.1 * i, window_deg=2.5, follow=False)
    good = res_follow is not None and res_follow["confirmed"]
    ok &= good
    print(f"  {'좁은 창 + follow로 추적 유지':30s} got={_fmt(res_follow)}  [{_pf(good)}]")
    good = res_fixed is None
    ok &= good
    print(f"  {'follow 없으면 놓친다(대조군)':30s} got={_fmt(res_fixed)}  [{_pf(good)}]")

    # 표시 지연 보정 — 10m/s로 다가오는 대상을 EMA로 평활화하면 화면이 실제보다 멀게 나온다.
    # 보정 후 잔차가 보정 전보다 작아야 한다(그리고 부호가 뒤집히지 않아야 한다).
    tr_lag = TargetTracker()
    truth = None
    res_lag = None
    for i in range(1, 21):
        truth = 60.0 - 1.0 * i          # 0.1초마다 1m씩 = 10m/s
        res_lag = tr_lag.update("siren", 0.0, _make_fake_points(0.0, truth, n=40),
                                sigma_deg=6.0, scan_id=i, now=t0 + 0.1 * i)
    err_raw = res_lag["distance_raw_m"] - truth
    err_shown = res_lag["distance_m"] - truth
    good = abs(err_shown) < abs(err_raw) and err_shown > -1.0
    ok &= good
    print(f"  {'지연 보정이 오차를 줄인다':30s} "
          f"보정전={err_raw:+.2f}m 보정후={err_shown:+.2f}m  [{_pf(good)}]")

    # 멀어지는 물체는 closing이 음수
    tracker2 = TargetTracker()
    away = None
    for i in range(1, 17):
        away = tracker2.update("motorcycle", 180.0,
                               _make_fake_points(180.0, 15.0 + 0.5 * i, n=40),
                               sigma_deg=6.0, scan_id=i, now=t0 + 0.1 * i)
    good = away is not None and away["closing_mps"] is not None and away["closing_mps"] < 0
    ok &= good
    print(f"  {'멀어지면 closing 음수':30s} "
          f"got={None if away is None or away['closing_mps'] is None else round(away['closing_mps'], 1)}m/s"
          f"  [{_pf(good)}]")
    return ok


def run_config_test() -> bool:
    """설정값이 실제로 동작을 바꾸는지 — 장착 높이가 틀리면 지면이 안 걸러진다."""
    ok = True
    saved = config()
    try:
        # 실제 장착 높이 2.1m인 차에서 설정값만 바꿔본다.
        scene = _make_road_scene(mount_height_m=2.1, car_range_m=22.0)

        set_config(LidarConfig(mount_height_m=2.1))
        got_right = match_distance(scene, 0.0)
        good = (got_right is not None and not got_right["blind"]
                and abs(got_right["distance_m"] - 22.0) < 1.5)
        ok &= good
        print(f"  {'높이 실측값이면 차량 확정':30s} expected=22m  got={_fmt(got_right)}  [{_pf(good)}]")

        # 높이를 **크게** 잡으면 지면 제거 기준선이 노면 밑으로 내려가 노면이 물체로 남는다.
        # 그러면 2.1/tan(15°) ≈ 7.8m 노면 리턴이 최근접이 돼 차량 대신 아스팔트를 보고한다.
        set_config(LidarConfig(mount_height_m=2.6))
        got_high = match_distance(scene, 0.0)
        noisy = got_high is not None and (got_high["blind"] or got_high["distance_m"] < 15.0)
        ok &= noisy
        print(f"  {'높이 과대 → 노면이 섞인다':30s} got={_fmt(got_high)}  [{_pf(noisy)}]")

        # 반대로 **작게** 잡으면 지면은 걸러지지만 물체 하단이 같이 잘려 점이 줄어든다
        # (너무 작으면 물체 자체를 잃는다).
        set_config(LidarConfig(mount_height_m=1.5))
        got_low = match_distance(scene, 0.0)
        thin = (got_low is None or got_low.get("points", 0) < got_right.get("points", 0))
        ok &= thin
        print(f"  {'높이 과소 → 물체 점이 깎인다':30s} "
              f"got={_fmt(got_low)} (실측값일 때 {got_right.get('points')}점)  [{_pf(thin)}]")
    finally:
        set_config(saved)
    return ok


def _fmt(match):
    if match is None:
        return "None"
    if match.get("blind"):
        return f"blind(pts={match.get('points')})"
    extra = ""
    if match.get("closing_mps") is not None:
        extra = f", {match['closing_mps']:+.1f}m/s"
    return f"{match['distance_m']:.1f}m(pts={match.get('points')}{extra})"


def _pf(ok):
    return "PASS" if ok else "FAIL"


def run_selftest():
    cases = [
        ("정상 매칭(32m)", _make_fake_points(194.0, 32.0, n=40), 194.0,
         {"distance_m": 32.0, "blind": False}),
        ("사각지대(3m)", _make_fake_points(90.0, 3.0, n=40), 90.0,
         {"distance_m": None, "blind": True}),
        ("관측 안 됨(빈 스캔)", np.empty((0, 5), dtype=np.float32), 0.0, None),
        ("각도 밖(방향 다름)", _make_fake_points(0.0, 20.0, n=40), 180.0, None),
    ]
    all_ok = True
    print("\n[1] 단발 매칭")
    for name, points, theta, expected in cases:
        got = match_distance(points, theta)
        if expected is None:
            ok = got is None
        elif got is None:
            ok = False
        else:
            dist_ok = (expected["distance_m"] is None) == (got["distance_m"] is None) and (
                expected["distance_m"] is None or abs(got["distance_m"] - expected["distance_m"]) < 1.0
            )
            ok = dist_ok and expected["blind"] == got["blind"]
        all_ok &= ok
        print(f"  {name:30s} expected={expected}  got={_fmt(got)}  [{_pf(ok)}]")

    print("\n[2] 지면 제거 (노면 장면)")
    all_ok &= run_road_scene_test()
    print("\n[3] 방향 불확실성 → 각도 창")
    all_ok &= run_window_test()
    print("\n[4] 후보 분리·선택")
    all_ok &= run_candidate_test()
    print("\n[5] 추적·접근 속도")
    all_ok &= run_tracker_test()
    print("\n[6] 장착 설정 반영")
    all_ok &= run_config_test()
    print(f"\n[selftest] {'ALL PASS' if all_ok else 'FAIL 있음'}")
    return all_ok


def run_live(theta_deg: float, sigma_deg: float, interval_sec: float, port: int,
             track: bool = False):
    scanner = LidarScanner(port=port)
    scanner.start()
    cfg = config()
    tracker = TargetTracker() if track else None
    window = angle_window_deg(sigma_deg, cfg)
    print(f"[*] LiDAR 거리 매칭 — theta={theta_deg}° sigma={sigma_deg}° → 창 ±{window:.1f}°, "
          f"오프셋={cfg.mount_offset_deg:+.1f}° 높이={cfg.mount_height_m:.2f}m"
          f"{' · 추적 모드' if track else ''}. 종료: Ctrl+C\n")
    try:
        while True:
            points, scan_id = scanner.snapshot()
            if tracker is not None:
                result = tracker.update("siren", theta_deg, points,
                                        sigma_deg=sigma_deg, scan_id=scan_id)
                tail = ""
                if result is not None:
                    tail = (f" hits={result['hits']} 확정={result['confirmed']}"
                            f" 방위={result.get('azimuth_deg', float('nan')):.1f}°")
            else:
                result = match_distance(points, theta_deg, sigma_deg=sigma_deg)
                tail = "" if result is None else f" 방위={result['azimuth_deg']:.1f}°"
            st = scanner.status()
            print(f"[{time.strftime('%H:%M:%S')}] pts={points.shape[0]:6d} scan={scan_id:6d} "
                  f"age={st['age']:.2f}s  match={_fmt(result)}{tail}")
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("\n[*] 종료합니다.")
    finally:
        scanner.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="하드웨어 없이 매칭·추적 로직 검증")
    parser.add_argument("--live", action="store_true", help="실제 VLP-16으로 방향별 거리 실시간 출력")
    parser.add_argument("--theta", type=float, default=0.0, help="--live에서 조회할 방향(도, 전방=0)")
    parser.add_argument("--sigma", type=float, default=7.0,
                        help="방향 불확실성(도). 각도 창 폭이 여기서 나온다 (젯슨 패킷의 sigma와 같은 값)")
    parser.add_argument("--track", action="store_true", help="추적 모드 — 접근 속도·확정 여부까지 출력")
    parser.add_argument("--port", type=int, default=VLP16_UDP_PORT, help="VLP-16 UDP 포트")
    parser.add_argument("--interval", type=float, default=0.25, help="폴링 주기(초)")
    parser.add_argument("--config", default=None, help="장착 설정 JSON 경로 (기본: 08_Video_Demo/config/lidar_mount.json)")
    parser.add_argument("--mount-offset", type=float, default=None, help="라이다 요 오프셋(도) 임시 override")
    parser.add_argument("--mount-height", type=float, default=None, help="노면→라이다 높이(m) 임시 override")
    args = parser.parse_args()

    if args.config:
        load_config(args.config, quiet=False)
    cfg = config()
    if args.mount_offset is not None:
        cfg.mount_offset_deg = args.mount_offset
    if args.mount_height is not None:
        cfg.mount_height_m = args.mount_height

    if args.selftest:
        raise SystemExit(0 if run_selftest() else 1)
    if args.live:
        run_live(args.theta, args.sigma, args.interval, args.port, track=args.track)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
