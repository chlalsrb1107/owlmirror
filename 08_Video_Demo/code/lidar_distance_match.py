"""
lidar_distance_match.py — Velodyne VLP-16 포인트클라우드에서 DoA 방향에 해당하는
가장 가까운 물체까지의 거리를 추정한다 (9/8 영상 데모용).

배경: 00_Overview/2026-08-25_9.8_영상제출_촬영_계획.md (2026-08-31 갱신)
카메라 4대+LiDAR가 촬영일까지 장착·데이터 수신 가능해져, 8/25에 정했던 "라이다 없이
카메라만" 축소 구성을 되돌리고 최종 아키텍처(카메라+LiDAR 거리 매칭)로 복귀한다.
07_System_Integration/전체_시스템_통합.md의 matching_node(ROS2, velodyne_pointcloud 패키지
사용 예정)를 9/8 데모용으로 단일 프로세스·의존성 최소화 버전으로 압축한 것이 이 파일이다.

원리 (ui_state_spec.md §4/§5/§7 압축):
  1. VLP-16이 1회전(~100ms)마다 뿜는 포인트를 UDP(기본 2368/udp)로 계속 수신해
     최신 스캔(포인트 배열)을 배경 스레드가 갱신
  2. DoA 방향(theta_deg) ± angle_margin_deg 범위 안의 포인트만 추려서
  3. 반경 BLIND_RADIUS_M(6m) 이내(물리적 사각지대, ui_state_spec.md §5 "사각 구역")는 제외하고
  4. 남은 포인트 중 가장 가까운 것의 거리를 그 방향 후보의 거리로 반환

**(2026-09-01 실장비 점검 완료)** 노트북에서 카메라 4대+VLP-16+젯슨 직결로 검증. 아래 두 가지
치명 버그가 있어 지면 제거 전까지 모든 방향이 항상 6.34m(지면 반사)를 반환했었다 — 수정 완료:
  - `decoder.decode_packet(data, ts)`는 존재하지 않는 메서드였음 → `decoder.decode(ts, data)`
    (velodyne-decoder 3.x API, 인자 순서도 stamp가 먼저임)
  - `vd.Config(model="VLP-16")`은 문자열을 받지 않음 → `vd.Config(model=vd.Model.VLP16)`(enum)
  - 지면 미제거: 1.7m 장착 시 최하단 빔(-15°)이 1.7/tan15°≈6.34m 지점 노면에 닿아 그 점이 항상
    "가장 가까운 점"으로 잡혔음 → mount_height_m 기준 z 지면 필터 추가
  - 단일/소수 노이즈 점이 물체로 오인되는 것을 막기 위해 거리축 클러스터링(MIN_CLUSTER_POINTS,
    CLUSTER_GAP_M) 추가 — 실측: 28,817 pt/scan · 16링 · 754 pkt/s · 9.9Hz(101ms, 문서상
    "한 바퀴 100ms"와 일치) · 360° 균등 커버 · 거리 0.50~12.12m
  - 배경 스레드가 조용히 죽어 있어도 겉으론 "관측 안 됨"과 구분이 안 됐음 → healthy()/status()로
    스레드 생존 여부와 마지막 예외를 노출

⚠️ 여전히 미검증:
  - VLP-16 좌표계 → 차량 좌표계(전방 0°) 변환 오프셋(MOUNT_OFFSET_DEG) — 장착 후 실측 보정 필요
  - MIN_CLUSTER_POINTS(5)/CLUSTER_GAP_M(1.0) 기본값 — 촬영일 현장에서 원거리 대상(점이 적음)
    기준으로 조정 필요
  - 카메라-라이다 외부 캘리브레이션은 관측 부족(5개, 보통 15~30개 필요)으로 사용 불가 수준
    (RPY 편차 175° = 사실상 뒤집힘) — 단 이 모듈은 각도만 쓰고 카메라 외부파라미터가 필요
    없으므로 9/8 데모에는 영향 없음. 카메라 영상에 라이다를 겹쳐 그리거나 YOLO와 융합할 때만
    필요 → 재촬영(관측 20개+)은 9/8 이후로 미룰 것

실패 시 폴백 원칙: 이 모듈이 import 실패/장비 연결 실패해도 live_demo.py 전체가 죽지 않고
"모든 감지는 주의 단계까지만"(8/25 축소 버전과 동일한 동작)으로 자동 하향되도록 만들 것
— live_demo.py의 lidar_available 플래그 참고.

의존성:
    pip install velodyne-decoder numpy
    (velodyne-decoder가 VLP-16 공장 캘리브레이션 룩업 테이블을 내장하고 있어
    별도 .yaml 캘리브레이션 파일 없이 채널별 각도/거리를 바로 디코드할 수 있음)

실행:
    python3 lidar_distance_match.py --selftest             # 하드웨어 없이 매칭 로직만 검증
    python3 lidar_distance_match.py --live --theta 0 --mount-height 1.7   # 실제 장비로 해당 방향 거리 실시간 출력
"""

import argparse
import socket
import threading
import time

import numpy as np

VLP16_UDP_PORT = 2368
BLIND_RADIUS_M = 6.0        # ui_state_spec.md §5 "사각 구역" — 이 반경 이내는 거리 신뢰 안 함
MIN_POINTS = 3               # 이보다 적으면 "관측 안 됨"(거리 없음)으로 처리
DEFAULT_ANGLE_MARGIN_DEG = 25.0  # 펌웨어 DoA 오차 범위 추정치(미검증) — 넓혀야 할 수도 있음

# 라이다 장착 후 실측 보정 필요: 라이다가 출력하는 원점 좌표축 중 어느 방향이
# 차량 정면(0°)에 해당하는지 오프셋으로 보정한다. doa_camera_select.py의
# MOUNT_OFFSET_DEG(마이크용)와는 별개의 값이다 — 두 센서가 물리적으로 다른 위치에 있으므로.
MOUNT_OFFSET_DEG = 0.0

# 라이다 장착 높이(지면 필터 기준값) — 2026-09-01 노트북 검증 시 임시 거치대 기준 1.7m.
# 실차 루프랙 고정 후 줄자로 재실측해 교체할 것 (live_demo.py의 LIDAR_MOUNT_HEIGHT_M도 함께).
DEFAULT_MOUNT_HEIGHT_M = 1.7
GROUND_MARGIN_M = 0.3  # 지면으로 간주할 z 여유폭 — 장착 높이 대비 이 안쪽 점은 노면으로 버림

# 거리축 클러스터링: 각도창 안 후보를 거리순으로 정렬해 CLUSTER_GAP_M 이내끼리 묶고,
# MIN_CLUSTER_POINTS 미만인 클러스터(노이즈/단일 반사)는 버린다. 가장 가까운 유효 클러스터의
# 최솟값을 거리로 반환 — 실측 조정 필요(원거리 대상은 점이 적어 놓칠 수 있음).
MIN_CLUSTER_POINTS = 5
CLUSTER_GAP_M = 1.0


class LidarScanner:
    """VLP-16 UDP 스트림을 배경 스레드로 계속 읽어 최신 스캔(포인트 배열)만 들고 있는다.

    포인트는 차량 좌표계 기준 (x, y, z, range_m, azimuth_deg) 5열 numpy 배열. 지면 반사로
    보이는 점(z가 -mount_height_m 근방 이하)은 이 단계에서 이미 제거해 내보낸다.
    """

    def __init__(self, mount_offset_deg: float = MOUNT_OFFSET_DEG,
                 mount_height_m: float = DEFAULT_MOUNT_HEIGHT_M):
        self.mount_offset_deg = mount_offset_deg
        self.mount_height_m = mount_height_m
        self._lock = threading.Lock()
        self._latest = np.empty((0, 5), dtype=np.float32)
        self._thread = None
        self._stop = threading.Event()
        self._running = False
        self._last_error = None

    def start(self, port: int = VLP16_UDP_PORT):
        import velodyne_decoder as vd  # noqa: F401  (여기서 import해 --selftest는 의존성 없이도 동작)

        self._thread = threading.Thread(target=self._run, args=(port,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def healthy(self) -> bool:
        """배경 스레드가 살아서 정상 수신 중인지. False면 latest_points()가 오래된/빈 값일 수 있음."""
        return self._running and self._thread is not None and self._thread.is_alive()

    def status(self) -> str:
        if self.healthy():
            return "running"
        if self._last_error is not None:
            return f"dead: {self._last_error}"
        return "not started"

    def _run(self, port: int):
        import velodyne_decoder as vd

        try:
            config = vd.Config(model=vd.Model.VLP16)
            decoder = vd.StreamDecoder(config)
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("", port))
            sock.settimeout(0.5)
            self._running = True

            while not self._stop.is_set():
                try:
                    data, _ = sock.recvfrom(2048)
                except socket.timeout:
                    continue
                result = decoder.decode(time.time(), data)
                if result is None:
                    continue
                _, points = result  # points: Nx(x,y,z,intensity,...) sensor frame
                if points is None or len(points) == 0:
                    continue
                x, y, z = points[:, 0], points[:, 1], points[:, 2]
                x, y, z = filter_ground(x, y, z, self.mount_height_m)
                rng = np.sqrt(x**2 + y**2)
                azimuth = (np.degrees(np.arctan2(y, x)) - self.mount_offset_deg + 180) % 360 - 180
                with self._lock:
                    self._latest = np.column_stack([x, y, z, rng, azimuth]).astype(np.float32)
        except Exception as e:  # noqa: BLE001 — 원인을 status()로 노출하기 위해 여기서 잡음
            self._last_error = repr(e)
        finally:
            self._running = False

    def latest_points(self):
        with self._lock:
            return self._latest.copy()


def filter_ground(x: np.ndarray, y: np.ndarray, z: np.ndarray, mount_height_m: float,
                   ground_margin_m: float = GROUND_MARGIN_M):
    """지면 반사로 보이는 점(z가 -mount_height_m 근방 이하)을 제거한다.

    독립 함수로 뺀 이유: 실물 소켓/스레드 없이도 회귀 테스트(run_selftest)에서
    "1.7m 장착 시 -15° 빔이 6.34m 노면에 닿는" 시나리오를 재현·검증하기 위함.
    """
    above_ground = z > -(mount_height_m - ground_margin_m)
    return x[above_ground], y[above_ground], z[above_ground]


def _nearest_valid_cluster(ranges: np.ndarray, min_cluster_points: int, cluster_gap_m: float):
    """거리순 정렬 후 cluster_gap_m 이내끼리 묶어, min_cluster_points 이상인 가장 가까운
    클러스터의 최솟값을 반환한다. 유효 클러스터가 없으면 None — 단일/소수 노이즈 반사를
    "물체"로 오인하지 않기 위함."""
    if ranges.shape[0] == 0:
        return None
    sorted_ranges = np.sort(ranges)
    cluster = [sorted_ranges[0]]
    for r in sorted_ranges[1:]:
        if r - cluster[-1] <= cluster_gap_m:
            cluster.append(r)
            continue
        if len(cluster) >= min_cluster_points:
            return float(cluster[0])
        cluster = [r]
    if len(cluster) >= min_cluster_points:
        return float(cluster[0])
    return None


def match_distance(points: np.ndarray, theta_deg: float,
                    angle_margin_deg: float = DEFAULT_ANGLE_MARGIN_DEG,
                    blind_radius_m: float = BLIND_RADIUS_M,
                    min_cluster_points: int = MIN_CLUSTER_POINTS,
                    cluster_gap_m: float = CLUSTER_GAP_M):
    """방향(theta_deg, 차량 좌표계 0=전방) 근처에서 가장 가까운 물체까지의 거리를 찾는다.

    지면 필터를 거친 점들 중에서도 단일/소수 노이즈 반사가 섞일 수 있어, 거리축으로
    클러스터링해 min_cluster_points 이상 뭉친 것만 "물체"로 인정한다(_nearest_valid_cluster).

    반환:
        {"distance_m": float, "blind": False}  — 사각지대 밖에서 물체 확정
        {"distance_m": None, "blind": True}    — 사각지대(6m 이내)에만 점이 있어 거리 신뢰 불가
        None                                    — 그 방향에 관측된 점이 (충분히) 없음
    """
    if points.shape[0] == 0:
        return None

    rel = (points[:, 4] - theta_deg + 180) % 360 - 180
    in_window = np.abs(rel) <= angle_margin_deg
    candidates = points[in_window]
    if candidates.shape[0] < MIN_POINTS:
        return None

    ranges = candidates[:, 3]
    near_blind = ranges < blind_radius_m
    outside = ranges[~near_blind]

    nearest = _nearest_valid_cluster(outside, min_cluster_points, cluster_gap_m)
    if nearest is not None:
        return {"distance_m": nearest, "blind": False}
    if near_blind.sum() >= MIN_POINTS:
        return {"distance_m": None, "blind": True}
    return None


def _make_fake_points(theta_deg: float, distance_m: float, n: int = 20) -> np.ndarray:
    """--selftest용 가짜 포인트: theta_deg 방향, distance_m 거리에 물체 하나를 흩뿌린다."""
    jitter = np.random.uniform(-3, 3, n)
    az = theta_deg + jitter
    rng = distance_m + np.random.uniform(-0.2, 0.2, n)
    x = rng * np.cos(np.radians(az))
    y = rng * np.sin(np.radians(az))
    z = np.zeros(n)
    return np.column_stack([x, y, z, rng, az]).astype(np.float32)


def _make_ground_points(mount_height_m: float, n: int = 40) -> np.ndarray:
    """1.7m 장착 시 -15° 최하단 빔이 지면에 닿는 6.34m 링(2026-09-01 실측 재현용) 흉내."""
    ring_range = mount_height_m / np.tan(np.radians(15))
    az = np.random.uniform(0, 360, n)
    rng = ring_range + np.random.uniform(-0.1, 0.1, n)
    x = rng * np.cos(np.radians(az))
    y = rng * np.sin(np.radians(az))
    z = np.full(n, -mount_height_m)  # 센서 기준 지면은 -mount_height_m
    return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)


def run_selftest():
    cases = [
        ("정상 매칭(32m)", _make_fake_points(194.0, 32.0), 194.0, {"distance_m": 32.0, "blind": False}),
        ("사각지대(3m)", _make_fake_points(90.0, 3.0), 90.0, {"distance_m": None, "blind": True}),
        ("관측 안 됨(빈 스캔)", np.empty((0, 5), dtype=np.float32), 0.0, None),
        ("각도 밖(방향 다름)", _make_fake_points(0.0, 20.0), 180.0, None),
        ("노이즈 산발(클러스터 미달)", _make_fake_points(45.0, 10.0, n=3), 45.0, None),
    ]
    all_ok = True
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
        print(f"  {name:20s} expected={expected}  got={got}  [{'PASS' if ok else 'FAIL'}]")

    # 회귀 테스트: 2026-09-01 실장비 점검에서 발견된 "지면이 항상 6.34m 최근접으로 잡히는" 버그.
    # filter_ground가 이 지면 링을 걸러내는지 직접 확인 (스캐너 스레드/소켓 없이).
    gx, gy, gz = _make_ground_points(mount_height_m=1.7)
    fx, _, _ = filter_ground(gx, gy, gz, mount_height_m=1.7)
    ground_ok = fx.shape[0] == 0
    all_ok &= ground_ok
    print(f"  {'지면 필터(1.7m 장착)':20s} expected=0pt(전부 제거)  got={fx.shape[0]}pt  "
          f"[{'PASS' if ground_ok else 'FAIL'}]")

    print(f"\n[selftest] {'ALL PASS' if all_ok else 'FAIL 있음'}")
    return all_ok


def run_live(theta_deg: float, angle_margin_deg: float, interval_sec: float, mount_height_m: float):
    scanner = LidarScanner(mount_height_m=mount_height_m)
    scanner.start()
    print(f"[*] LiDAR 실시간 거리 매칭 시작 (theta={theta_deg}deg, margin=±{angle_margin_deg}deg, "
          f"mount_height={mount_height_m}m). 종료: Ctrl+C\n")
    try:
        while True:
            if not scanner.healthy():
                print(f"[!] 스캐너 비정상: {scanner.status()}")
            points = scanner.latest_points()
            result = match_distance(points, theta_deg, angle_margin_deg)
            print(f"[{time.strftime('%H:%M:%S')}] points={points.shape[0]:5d}  match={result}")
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("\n[*] 종료합니다.")
    finally:
        scanner.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="하드웨어 없이 매칭 로직만 검증")
    parser.add_argument("--live", action="store_true", help="실제 VLP-16으로 방향별 거리 실시간 출력")
    parser.add_argument("--theta", type=float, default=0.0, help="--live에서 조회할 방향(도, 전방=0)")
    parser.add_argument("--margin", type=float, default=DEFAULT_ANGLE_MARGIN_DEG, help="각도 여유창(도)")
    parser.add_argument("--interval", type=float, default=0.25, help="폴링 주기(초)")
    parser.add_argument("--mount-height", type=float, default=DEFAULT_MOUNT_HEIGHT_M,
                         help="라이다 장착 높이(m) — 지면 필터 기준값, 줄자 실측값으로 지정")
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
    elif args.live:
        run_live(args.theta, args.margin, args.interval, args.mount_height)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
