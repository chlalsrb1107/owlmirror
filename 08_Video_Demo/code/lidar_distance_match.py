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
BLIND_RADIUS_M = 6.0        # ui_state_spec.md §5 "사각 구역" — 이 반경 이내는 거리 신뢰 안 함
MIN_POINTS = 3               # 이보다 적으면 "관측 안 됨"(거리 없음)으로 처리
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

    def start(self, port: int = VLP16_UDP_PORT):
        import velodyne_decoder as vd  # noqa: F401  (여기서 import해 --selftest는 의존성 없이도 동작)

        self._thread = threading.Thread(target=self._run, args=(port,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self, port: int):
        import velodyne_decoder as vd

        config = vd.Config(model="VLP-16")
        decoder = vd.StreamDecoder(config)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("", port))
        sock.settimeout(0.5)

        while not self._stop.is_set():
            try:
                data, _ = sock.recvfrom(2048)
            except socket.timeout:
                continue
            result = decoder.decode_packet(data, time.time())
            if result is None:
                continue
            _, points = result  # points: Nx(x,y,z,intensity,...) sensor frame
            if points is None or len(points) == 0:
                continue
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            rng = np.sqrt(x**2 + y**2)
            azimuth = (np.degrees(np.arctan2(y, x)) - self.mount_offset_deg + 180) % 360 - 180
            with self._lock:
                self._latest = np.column_stack([x, y, z, rng, azimuth]).astype(np.float32)

    def latest_points(self):
        with self._lock:
            return self._latest.copy()


def match_distance(points: np.ndarray, theta_deg: float,
                    angle_margin_deg: float = DEFAULT_ANGLE_MARGIN_DEG,
                    blind_radius_m: float = BLIND_RADIUS_M):
    """방향(theta_deg, 차량 좌표계 0=전방) 근처에서 가장 가까운 물체까지의 거리를 찾는다.

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

    if outside.shape[0] >= MIN_POINTS:
        return {"distance_m": float(outside.min()), "blind": False}
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
        print(f"  {name:20s} expected={expected}  got={got}  [{'PASS' if ok else 'FAIL'}]")
    print(f"\n[selftest] {'ALL PASS' if all_ok else 'FAIL 있음'}")
    return all_ok


def run_live(theta_deg: float, angle_margin_deg: float, interval_sec: float):
    scanner = LidarScanner()
    scanner.start()
    print(f"[*] LiDAR 실시간 거리 매칭 시작 (theta={theta_deg}deg, margin=±{angle_margin_deg}deg). 종료: Ctrl+C\n")
    try:
        while True:
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
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
    elif args.live:
        run_live(args.theta, args.margin, args.interval)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
