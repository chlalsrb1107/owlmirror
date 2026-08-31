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

⚠️ 미검증 (실물 장비로 확인 전까지 아래 모두 가정치):
  - VLP-16 좌표계(센서 원점, 라이다는 루프랙 중앙 최상단) → 차량 좌표계(전방 0°) 변환 오프셋
    (MOUNT_OFFSET_DEG — doa_camera_select.py의 마이크 보정값과 별개로 라이다도 자체 요(yaw)
    보정이 필요할 수 있음)
  - 포인트가 하나도 안 잡히는 "관측 안 됨"과 "그 방향에 물체 없음"을 가르는 최소 포인트 수
    (MIN_POINTS) — 현재는 임의로 3개
  - 노이즈·지면 반사 필터링이 전혀 없음(단순 최근접 포인트). 실차에서는 지면 다중 반사가
    "가짜 근접 물체"로 잡힐 수 있어 최소한의 높이(z) 필터가 필요할 가능성이 높음
  - UDP 포트/패킷 포맷은 Velodyne VLP-16 공식 스펙(2368/UDP, 1206바이트 패킷) 기준이며
    이 저장소에는 실물 장비가 없어 검증 못 함 — 아래 --live로 실제 장비 연결 후 확인 필수
  - 연구실(멘토 측)이 LiDAR 연결/동기화를 담당하기로 했던 최종 아키텍처와 달리, 이 데모는
    팀이 직접 라이다 데이터를 받아 처리해야 함 — 시계 동기화(laptop 단일 프로세스라 불필요)
    문제는 없지만 케이블링·전원은 직접 확인 필요

실패 시 폴백 원칙: 이 모듈이 import 실패/장비 연결 실패해도 live_demo.py 전체가 죽지 않고
"모든 감지는 주의 단계까지만"(8/25 축소 버전과 동일한 동작)으로 자동 하향되도록 만들 것
— live_demo.py의 lidar_available 플래그 참고.

의존성:
    pip install velodyne-decoder numpy
    (velodyne-decoder가 VLP-16 공장 캘리브레이션 룩업 테이블을 내장하고 있어
    별도 .yaml 캘리브레이션 파일 없이 채널별 각도/거리를 바로 디코드할 수 있음)

실행:
    python3 lidar_distance_match.py --selftest             # 하드웨어 없이 매칭 로직만 검증
    python3 lidar_distance_match.py --live --theta 194      # 실제 장비로 해당 방향 거리 실시간 출력
"""

import argparse
import socket
import threading
import time

import numpy as np

VLP16_UDP_PORT = 2368
VLP16_DATA_PACKET_BYTES = 1206  # 데이터 패킷 크기. 위치 패킷 등 다른 크기는 디코더에 넣지 않는다
BLIND_RADIUS_M = 6.0        # ui_state_spec.md §5 "사각 구역" — 이 반경 이내는 거리 신뢰 안 함
MIN_POINTS = 3               # 각도 창 안에 이보다 적으면 "관측 안 됨"으로 처리

# ---- 지면 제거 (2026-09-01 추가) ----------------------------------------------
# VLP-16은 수직 FOV가 ±15°다. 1.7m 루프랙에 올리면 최하단 빔이 1.7/tan(15°) ≈ 6.3m 앞
# 노면에 닿는다 — 하필 사각지대 반경 6m 바로 바깥이다. 그래서 지면을 안 걸러내면 어느
# 방향을 조회하든 "6.3m에 노면"이 최근접으로 잡혀, 30m 앞 구급차 대신 매번 아스팔트까지의
# 거리를 보고하게 된다. (실측: 필터 없을 때 모든 방향에서 6.000m 고정 반환)
MOUNT_HEIGHT_M = 1.7        # ⚠️ 차량 장착 후 줄자로 실측한 값으로 교체할 것 (노면→라이다 원점)
MIN_OBJECT_HEIGHT_M = 0.3   # 노면에서 이만큼 위에 있어야 물체로 인정
MAX_OBJECT_HEIGHT_M = 1.5   # 센서보다 이만큼 위는 육교·표지판·터널 천장으로 보고 제외

# ---- 거리축 클러스터링 (2026-09-01 추가) --------------------------------------
# 창 안의 최근접 점 하나만 보면 노이즈 한 점에 거리가 끌려간다. ui_state_spec.md가 말하는
# "촘촘한 점 클러스터 = 매칭 확정 대상"을 실제로 구현하기 위해, 거리축으로 이어진 점
# 덩어리를 찾아 충분히 큰 덩어리만 물체로 인정하고 그 중앙값을 거리로 쓴다.
CLUSTER_GAP_M = 1.0         # 정렬된 거리에서 이보다 벌어지면 다른 물체로 분리
MIN_CLUSTER_POINTS = 5      # ⚠️ 실차 시험 후 조정 필요. 멀수록 점이 줄어 놓칠 수 있다
DEFAULT_ANGLE_MARGIN_DEG = 25.0  # 펌웨어 DoA 오차 범위 추정치(미검증) — 넓혀야 할 수도 있음

# 라이다 장착 후 실측 보정 필요: 라이다가 출력하는 원점 좌표축 중 어느 방향이
# 차량 정면(0°)에 해당하는지 오프셋으로 보정한다. doa_camera_select.py의
# MOUNT_OFFSET_DEG(마이크용)와는 별개의 값이다 — 두 센서가 물리적으로 다른 위치에 있으므로.
MOUNT_OFFSET_DEG = 0.0


class LidarScanner:
    """VLP-16 UDP 스트림을 배경 스레드로 계속 읽어 최신 스캔(포인트 배열)만 들고 있는다.

    포인트는 차량 좌표계 기준 (x, y, z, range_m, azimuth_deg) 5열 numpy 배열.
    """

    def __init__(self, mount_offset_deg: float = MOUNT_OFFSET_DEG):
        self.mount_offset_deg = mount_offset_deg
        self._lock = threading.Lock()
        self._latest = np.empty((0, 5), dtype=np.float32)
        self._thread = None
        self._stop = threading.Event()
        # 수신 스레드가 조용히 죽으면 "라이다는 붙었는데 아무것도 안 잡힘"으로 보인다.
        # 실패 사유를 남겨 호출측(live_demo.py)이 폴백을 실제로 탈 수 있게 한다.
        self._error = None
        self._scans = 0
        self._last_scan_at = 0.0

    def start(self, port: int = VLP16_UDP_PORT):
        import velodyne_decoder as vd  # noqa: F401  (여기서 import해 --selftest는 의존성 없이도 동작)

        self._thread = threading.Thread(target=self._run, args=(port,), daemon=True)
        self._thread.start()

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

        try:
            # ⚠️ model은 문자열("VLP-16")이 아니라 Model enum이어야 한다 (velodyne_decoder 3.x).
            decoder = vd.StreamDecoder(vd.Config(model=vd.Model.VLP16))
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", port))
            sock.settimeout(0.5)
        except Exception as e:  # noqa: BLE001
            self._fail(f"LiDAR 수신 초기화 실패: {e}")
            return

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
            azimuth = (np.degrees(np.arctan2(y, x)) - self.mount_offset_deg + 180) % 360 - 180
            with self._lock:
                self._latest = np.column_stack([x, y, z, rng, azimuth]).astype(np.float32)
                self._scans += 1
                self._last_scan_at = time.time()

        sock.close()

    def latest_points(self):
        with self._lock:
            return self._latest.copy()

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


def filter_ground(points: np.ndarray, mount_height_m: float = MOUNT_HEIGHT_M) -> np.ndarray:
    """노면과 머리 위 구조물을 빼고 "도로 위 물체" 높이대의 점만 남긴다.

    z는 라이다 원점 기준이므로 노면은 z = -mount_height_m 부근에 있다.
    """
    if points.shape[0] == 0:
        return points
    z = points[:, 2]
    above_ground = z > (-mount_height_m + MIN_OBJECT_HEIGHT_M)
    below_canopy = z < MAX_OBJECT_HEIGHT_M
    return points[above_ground & below_canopy]


def nearest_cluster(ranges: np.ndarray,
                    gap_m: float = CLUSTER_GAP_M,
                    min_points: int = MIN_CLUSTER_POINTS):
    """거리값들을 덩어리로 묶어 가장 가까운 "충분히 큰" 덩어리를 돌려준다.

    반환: (대표거리, 점 개수) 또는 None
    대표거리는 최솟값이 아니라 중앙값 — 노이즈 한 점이 거리를 끌어당기지 못하게.
    """
    if ranges.size == 0:
        return None
    ordered = np.sort(ranges)
    splits = np.where(np.diff(ordered) > gap_m)[0] + 1
    for group in np.split(ordered, splits):   # np.split은 가까운 순서를 유지한다
        if group.size >= min_points:
            return float(np.median(group)), int(group.size)
    return None


def match_distance(points: np.ndarray, theta_deg: float,
                    angle_margin_deg: float = DEFAULT_ANGLE_MARGIN_DEG,
                    blind_radius_m: float = BLIND_RADIUS_M,
                    mount_height_m: float = MOUNT_HEIGHT_M):
    """방향(theta_deg, 차량 좌표계 0=전방) 근처에서 가장 가까운 "물체"까지의 거리를 찾는다.

    순서: 각도 창 → 지면/천장 제거 → 거리축 클러스터링 → 가장 가까운 유효 클러스터.

    반환:
        {"distance_m": float, "blind": False, "points": int} — 사각지대 밖에서 물체 확정
        {"distance_m": None,  "blind": True,  "points": int} — 사각지대(6m 이내) 물체
        None — 그 방향에 유효한 물체가 없음(위치 미확정)

    ⚠️ 2026-09-01 변경: 이전에는 "6m 밖 점들의 최솟값"을 그대로 돌려줬는데, 지면이
       섞여 들어와 실제로는 항상 6.0m 근처가 나왔다. 이제 가장 가까운 **유효 클러스터**를
       기준으로 판정하므로, 사각지대 안에 물체가 있으면 (6m 밖에 다른 것이 있더라도)
       그쪽을 우선해 blind로 보고한다 — 가까운 쪽이 더 위급하다는 ui_state_spec.md 원칙.
    """
    if points.shape[0] == 0:
        return None

    rel = (points[:, 4] - theta_deg + 180) % 360 - 180
    candidates = points[np.abs(rel) <= angle_margin_deg]
    if candidates.shape[0] < MIN_POINTS:
        return None

    candidates = filter_ground(candidates, mount_height_m)
    if candidates.shape[0] < MIN_POINTS:
        return None

    found = nearest_cluster(candidates[:, 3])
    if found is None:
        return None

    distance, n_points = found
    if distance < blind_radius_m:
        return {"distance_m": None, "blind": True, "points": n_points}
    return {"distance_m": distance, "blind": False, "points": n_points}


def _make_fake_points(theta_deg: float, distance_m: float, n: int = 20) -> np.ndarray:
    """--selftest용 가짜 포인트: theta_deg 방향, distance_m 거리에 물체 하나를 흩뿌린다."""
    jitter = np.random.uniform(-3, 3, n)
    az = theta_deg + jitter
    rng = distance_m + np.random.uniform(-0.2, 0.2, n)
    x = rng * np.cos(np.radians(az))
    y = rng * np.sin(np.radians(az))
    z = np.zeros(n)
    return np.column_stack([x, y, z, rng, az]).astype(np.float32)


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

    half_az = np.degrees(np.arctan(0.9 / car_range_m))  # 차량 폭 1.8m
    for az in np.arange(car_azimuth_deg - half_az, car_azimuth_deg + half_az, 0.2):
        for z in np.arange(-mount_height_m + 0.2, -mount_height_m + 1.5, 0.12):
            x = car_range_m * np.cos(np.radians(az))
            y = car_range_m * np.sin(np.radians(az))
            pts.append([x, y, z, car_range_m, az])
    return np.array(pts, dtype=np.float32)


def run_road_scene_test() -> bool:
    """지면 제거 회귀 테스트 — 이게 깨지면 모든 방향이 노면 거리(6.34m)로 돌아간다."""
    scene = _make_road_scene()
    ok = True

    got = match_distance(scene, 0.0, mount_height_m=1.7)
    hit = got is not None and not got["blind"] and abs(got["distance_m"] - 20.0) < 1.0
    ok &= hit
    print(f"  {'노면+20m차량, 차량 방향':26s} expected=20.0m  got={got}  [{'PASS' if hit else 'FAIL'}]")

    for theta in (90.0, 180.0):
        got = match_distance(scene, theta, mount_height_m=1.7)
        # 차량이 없는 방향에는 노면만 있으므로 반드시 "없음"이어야 한다
        good = got is None
        ok &= good
        print(f"  {f'노면만 ({theta:.0f}도)':26s} expected=None    got={got}  [{'PASS' if good else 'FAIL'}]")
    return ok


def run_selftest():
    cases = [
        ("정상 매칭(32m)", _make_fake_points(194.0, 32.0), 194.0, {"distance_m": 32.0, "blind": False}),
        ("사각지대(3m)", _make_fake_points(90.0, 3.0), 90.0, {"distance_m": None, "blind": True}),
        ("관측 안 됨(빈 스캔)", np.empty((0, 5), dtype=np.float32), 0.0, None),
        ("각도 밖(방향 다름)", _make_fake_points(0.0, 20.0), 180.0, None),
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
        print(f"  {name:26s} expected={expected}  got={got}  [{'PASS' if ok else 'FAIL'}]")
    all_ok &= run_road_scene_test()
    print(f"\n[selftest] {'ALL PASS' if all_ok else 'FAIL 있음'}")
    return all_ok


def run_live(theta_deg: float, angle_margin_deg: float, interval_sec: float,
             mount_height_m: float = MOUNT_HEIGHT_M):
    scanner = LidarScanner()
    scanner.start()
    print(f"[*] LiDAR 실시간 거리 매칭 시작 (theta={theta_deg}deg, margin=±{angle_margin_deg}deg). 종료: Ctrl+C\n")
    try:
        while True:
            points = scanner.latest_points()
            result = match_distance(points, theta_deg, angle_margin_deg,
                                    mount_height_m=mount_height_m)
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
    parser.add_argument("--mount-height", type=float, default=MOUNT_HEIGHT_M,
                        help="노면에서 라이다 원점까지 높이(m). 지면 제거 기준 — 실측값을 넣을 것")
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
    elif args.live:
        run_live(args.theta, args.margin, args.interval, args.mount_height)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
