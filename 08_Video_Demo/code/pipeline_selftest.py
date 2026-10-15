"""
pipeline_selftest.py — 하드웨어 없이 **파이프라인 전체**를 돌려 검증한다 (2026-10-15 신규).

지금까지의 셀프테스트는 모듈별이었다(`lidar_distance_match --selftest`, `alert_policy
--selftest`, `bev_render --selftest`). 각 모듈이 맞아도 **연결이 틀리면** 화면은 조용히
거짓말을 한다 — sigma를 안 넘겨 창이 항상 ±30°로 열려 있거나, 거리를 갱신하는 코드가
표시 루프에 안 걸려 있거나, 라이다가 본 방위가 화면까지 전달되지 않는 식으로.
이 파일은 **live_demo.py의 실제 함수들**(detection_worker / refresh_lidar_targets /
decide_alert / SharedState)을 그대로 돌리고 가짜 센서만 꽂아서 그 연결을 확인한다.

꽂는 가짜 센서 2개
  · FakeScanner  — 평평한 노면 + 지정 방향에서 다가오는 차량 1대를 10Hz로 합성
                   (VLP-16 실제 빔 기하. lidar_distance_match._make_road_scene 사용)
  · FakeReceiver — 젯슨 패킷을 약 1초 주기로 흘려보낸다(방향에 sigma만큼 노이즈 포함)

확인하는 것
  1) 소리 방향(노이즈 있음)으로 조회해도 라이다가 그 차량을 잡아 거리를 확정한다
  2) 거리가 젯슨 패킷 주기(1초)가 아니라 **라이다 주기(10Hz)로 갱신**된다
  3) 다가오는 물체의 접근 속도가 추정되고, 그 근거로 단계가 올라간다
  4) 방향 배정(카메라)과 BEV 렌더가 끝까지 예외 없이 돈다
  5) 라이다를 죽이면 거리·클러스터만 빠지고 파이프라인은 계속 돈다(폴백)

실행 (실시간으로 약 15초 걸린다 — 추적·접근 속도는 시간이 흘러야 나온다):
    python3 pipeline_selftest.py
    python3 pipeline_selftest.py --outdir /tmp/pipeline   # 합성 화면 PNG까지 저장
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lidar_distance_match as lidar  # noqa: E402
import live_demo  # noqa: E402
from alert_policy import AlertPolicy  # noqa: E402

SCAN_HZ = 10.0          # VLP-16 실제 회전수(600rpm)
PACKET_SEC = 1.02       # 젯슨 전송 주기 실측치(현재_상태_요약.md: CUDA 적용 후 1.024초)
MOUNT_HEIGHT_M = 1.85   # 가짜 차량의 장착 높이 — 설정값과 일치시켜 테스트한다


class FakeScanner:
    """노면 + 다가오는 차량 1대를 10Hz로 합성하는 가짜 VLP-16.

    LidarScanner와 같은 인터페이스(snapshot/latest_points/healthy/stop)만 맞춘다.
    """

    def __init__(self, bearing_deg: float, start_m: float, closing_mps: float,
                 height_m: float = MOUNT_HEIGHT_M):
        self.bearing_deg = bearing_deg
        self.start_m = start_m
        self.closing_mps = closing_mps
        self.height_m = height_m
        self.t0 = time.time()
        self._cache = (None, -1)
        self.dead = False

    def truth_distance(self, now=None):
        now = time.time() if now is None else now
        return max(7.0, self.start_m - self.closing_mps * (now - self.t0))

    def snapshot(self):
        if self.dead:
            return np.empty((0, 5), dtype=np.float32), 0
        now = time.time()
        scan_id = int((now - self.t0) * SCAN_HZ)
        if scan_id != self._cache[1]:
            # 스캔 1개 = 그 시각의 노면 + 차량. 매 스캔 새로 만든다(차량이 다가오므로).
            pts = lidar._make_road_scene(mount_height_m=self.height_m,
                                         car_range_m=self.truth_distance(now),
                                         car_azimuth_deg=self.bearing_deg)
            self._cache = (pts, scan_id)
        return self._cache[0].copy(), self._cache[1]

    def latest_points(self):
        return self.snapshot()[0]

    def healthy(self, max_age_sec: float = 1.0):
        return not self.dead

    def status(self):
        return {"error": None, "scans": self._cache[1], "age": 0.0,
                "points": 0 if self._cache[0] is None else self._cache[0].shape[0]}

    def stop(self):
        self.dead = True


class FakeReceiver:
    """젯슨 패킷을 주기적으로 흘려보내는 가짜 수신기 (AudioDetectionReceiver 인터페이스)."""

    def __init__(self, class_name: str, bearing_deg: float, sigma_deg: float = 6.0,
                 rms_db: float = -38.0, period_sec: float = PACKET_SEC):
        self.class_name = class_name
        self.bearing_deg = bearing_deg
        self.sigma_deg = sigma_deg
        self.rms_db = rms_db
        self.period = period_sec
        self.seq = 0
        self._next = time.time()
        self._rng = np.random.default_rng(11)
        self.sent = []

    def get_new(self):
        now = time.time()
        if now < self._next:
            return None
        self._next = now + self.period
        self.seq += 1
        # 소리 방향은 sigma만큼 흔들린다 — 라이다 조회가 이 노이즈를 견디는지가 핵심이다
        theta = float(self.bearing_deg + self._rng.normal(0.0, self.sigma_deg))
        packet = {"seq": self.seq, "t": now, "class": self.class_name, "conf": 0.9,
                  "score": 4.2, "margin": 1.0, "theta": theta % 360.0,
                  "sigma": self.sigma_deg, "rms_db": self.rms_db, "theta_ok": True,
                  "raw": self.class_name}
        self.sent.append(packet)
        return packet

    def link(self):
        return {"connected": True, "received": self.seq, "dropped": 0, "malformed": 0}

    def stop(self):
        pass


def run_case(class_name: str, bearing_deg: float, start_m: float, closing_mps: float,
             duration_sec: float, outdir: Path = None, label: str = ""):
    """가짜 센서를 꽂고 live_demo의 실제 경로를 duration_sec 동안 돌린다."""
    lidar.set_config(lidar.LidarConfig(mount_height_m=MOUNT_HEIGHT_M, mount_offset_deg=0.0))
    scanner = FakeScanner(bearing_deg, start_m, closing_mps)
    receiver = FakeReceiver(class_name, bearing_deg)

    live_demo.LIDAR_AVAILABLE = True
    live_demo._lidar_scanner = scanner
    live_demo._lidar_tracker = lidar.TargetTracker()

    state = live_demo.SharedState()
    stop = threading.Event()
    worker = threading.Thread(
        target=_worker_until, args=(state, receiver, AlertPolicy(), stop), daemon=True)
    worker.start()

    samples = []          # (t, 화면에 보이는 거리, 진짜 거리)
    frames, t0 = 0, time.time()
    camera_frame = np.full((1200, 2048, 3), 40, dtype=np.uint8)
    while time.time() - t0 < duration_sec:
        points, lidar_ok = live_demo.refresh_lidar_targets(state)
        camera, targets, detection = state.snapshot()
        if targets:
            samples.append((time.time() - t0, targets[0].get("distance"),
                            scanner.truth_distance()))
        if outdir is not None and frames % 20 == 0:
            import bev_render
            img = bev_render.render(camera_frame, camera, targets, points,
                                    lidar_ok=lidar_ok, link=receiver.link(),
                                    detection=detection)
            import cv2
            path = outdir / f"{label}_{frames:04d}.png"
            cv2.imwrite(str(path), img)
        frames += 1
        time.sleep(1.0 / 30.0)   # 표시 루프와 같은 30Hz
    stop.set()
    worker.join(timeout=2.0)

    camera, targets, _ = state.snapshot()
    return {"camera": camera, "targets": targets, "samples": samples,
            "frames": frames, "packets": receiver.seq, "scanner": scanner}


def _worker_until(state, receiver, policy, stop):
    """detection_worker는 무한 루프라, 테스트에서는 종료 조건을 준 복제본을 쓴다.

    ⚠️ 여기서 복제하는 것은 **루프 제어뿐**이고 판정 경로(decide_alert / state.trigger)는
       live_demo의 실제 함수를 그대로 호출한다. 그 둘이 이 테스트의 검증 대상이다.
    """
    last_camera = "rear"
    while not stop.is_set():
        packet = receiver.get_new()
        if packet is None:
            time.sleep(live_demo.POLL_INTERVAL)
            continue
        top_class = packet["class"]
        if top_class not in live_demo.CAMERA_TRIGGER_CLASSES:
            continue
        doa = packet["theta"]
        rms_db = float(packet["rms_db"])
        sigma = float(packet["sigma"])
        camera = live_demo.select_camera(doa, mount_offset_deg=0.0)
        last_camera = camera
        level, distance, blind, detail = live_demo.decide_alert(
            policy, top_class, doa, None, rms_db, float(packet["score"]), True, sigma)
        track = live_demo._lidar_tracker.get(top_class)
        state.trigger(camera, top_class, live_demo.LOC_LABEL_KO[last_camera],
                      None, level, distance, blind, theta=doa, sigma=sigma,
                      detail=detail, match=track)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default=None,
                        help="합성 화면을 PNG로 저장할 디렉터리 (없으면 저장 안 함)")
    parser.add_argument("--duration", type=float, default=5.0, help="케이스별 실행 시간(초)")
    args = parser.parse_args()

    outdir = None
    if args.outdir:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

    ok = True

    def check(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  {name:38s} {detail}  [{'PASS' if good else 'FAIL'}]")

    # ── 1) 좌측 전방에서 다가오는 사이렌 ──────────────────────────────────────
    print(f"\n[1] 사이렌 — 좌측 전방 70°, 50m에서 6m/s로 접근 ({args.duration:.0f}초)")
    r = run_case("siren", 70.0, 50.0, 6.0, args.duration, outdir, "siren")
    t = r["targets"][0] if r["targets"] else None
    truth = r["scanner"].truth_distance()
    check("감지가 유지된다", t is not None, f"대상 {len(r['targets'])}건")
    if t is not None:
        err = None if t["distance"] is None else abs(t["distance"] - truth)
        check("거리가 실제와 맞는다", err is not None and err < 2.5,
              f"화면={None if t['distance'] is None else round(t['distance'], 1)}m "
              f"실제={truth:.1f}m")
        check("라이다 방위를 화면에 전달", t.get("theta_lidar") is not None
              and abs(lidar._wrap180(t["theta_lidar"] - 70.0)) < 4.0,
              f"방위={None if t.get('theta_lidar') is None else round(t['theta_lidar'], 1)}°")
        check("접근 속도 추정(약 6m/s)",
              t.get("closing") is not None and 3.0 < t["closing"] < 9.5,
              f"{None if t.get('closing') is None else round(t['closing'], 1)}m/s")
        check("사이렌은 경고", t["level"] == "경고", f"level={t['level']} ({t['detail']})")
        check("좌측 카메라로 전환", r["camera"] == "left", f"camera={r['camera']}")

        # 거리가 패킷 주기가 아니라 라이다 주기로 갱신되는가 —
        # 서로 다른 거리값의 개수를 패킷 수와 비교한다.
        shown = [d for _, d, _ in r["samples"] if d is not None]
        distinct = len(set(round(d, 2) for d in shown))
        check("거리가 라이다 주기로 갱신된다", distinct > r["packets"] + 2,
              f"서로 다른 거리값 {distinct}개 / 젯슨 패킷 {r['packets']}개")

        # 추적이 "멀어지는 것으로" 뒤집히지 않았는지 — 화면 거리가 단조 감소에 가까운가
        if len(shown) > 10:
            drops = sum(1 for a, b in zip(shown, shown[1:]) if b <= a + 0.3)
            check("거리가 줄어드는 방향으로 추적", drops > 0.8 * (len(shown) - 1),
                  f"{drops}/{len(shown)-1} 구간")

    # ── 2) 뒤에서 다가오는 오토바이 — 라이다 근거로 단계가 올라가는가 ─────────
    print(f"\n[2] 오토바이 — 후방 175°, 45m에서 7m/s로 접근 (Detection 없음·조용함, {args.duration:.0f}초)")
    r2 = run_case("motorcycle", 175.0, 45.0, 7.0, args.duration, outdir, "moto")
    t2 = r2["targets"][0] if r2["targets"] else None
    check("감지가 유지된다", t2 is not None)
    if t2 is not None:
        check("Detection 없이도 접근으로 경고", t2["level"] == "경고",
              f"level={t2['level']} ({t2['detail']})")
        check("후방 카메라", r2["camera"] == "rear", f"camera={r2['camera']}")
        check("거리 표시", t2["distance"] is not None,
              f"{None if t2['distance'] is None else round(t2['distance'], 1)}m")

    # ── 3) 라이다가 죽으면 ───────────────────────────────────────────────────
    print("\n[3] 라이다 폴백 — 스캐너를 죽이고 같은 소리를 넣는다")
    lidar.set_config(lidar.LidarConfig(mount_height_m=MOUNT_HEIGHT_M))
    dead = FakeScanner(0.0, 20.0, 0.0)
    dead.stop()
    live_demo._lidar_scanner = dead
    live_demo._lidar_tracker = lidar.TargetTracker()
    state = live_demo.SharedState()
    receiver = FakeReceiver("siren", 0.0)
    stop = threading.Event()
    th = threading.Thread(target=_worker_until,
                          args=(state, receiver, AlertPolicy(), stop), daemon=True)
    th.start()
    t0 = time.time()
    points = "unset"
    while time.time() - t0 < 2.5:
        points, lidar_ok = live_demo.refresh_lidar_targets(state)
        time.sleep(1.0 / 30.0)
    stop.set(); th.join(timeout=2.0)
    _, targets3, _ = state.snapshot()
    t3 = targets3[0] if targets3 else None
    check("라이다 없이도 감지는 뜬다", t3 is not None)
    if t3 is not None:
        check("거리는 비고 사이렌은 경고 유지",
              t3["distance"] is None and t3["level"] == "경고",
              f"distance={t3['distance']} level={t3['level']} ({t3['detail']})")
    check("포인트를 넘기지 않는다(옛 스캔 금지)", points is None, f"points={points}")

    if outdir is not None:
        print(f"\n[*] 합성 화면 PNG: {outdir}")
    print(f"\n[pipeline_selftest] {'ALL PASS' if ok else 'FAIL 있음'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
