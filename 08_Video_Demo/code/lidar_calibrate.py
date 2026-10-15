"""
lidar_calibrate.py — 차량에 올라간 LiDAR의 장착값을 현장에서 실측해
`08_Video_Demo/config/lidar_mount.json`에 써주는 도구 (2026-10-15 신규).

왜 필요한가: 거리 매칭은 두 개의 장착값에 전적으로 매여 있다.
  · `mount_height_m`  — 지면 제거 기준선. 틀리면 노면이 "6~8m 앞 물체"로 남아 어느 방향을
                        조회하든 아스팔트 거리를 보고한다(2026-09-01에 실제로 겪은 증상).
  · `mount_offset_deg`— 라이다 0°가 차량 전방이 아닐 때의 요(yaw) 보정. 틀리면 거리는
                        맞아도 **엉뚱한 방향의 물체** 거리를 소리의 방향에 붙인다.
이 둘은 줄자와 "각도를 아는 위치에 놓은 표적" 없이는 알 수 없어서, 그동안 코드에 추정치
(1.7m, 0°)로 박혀 있었다. 업체 시공으로 구성물이 차에 고정된 이제는 실측할 수 있다.

현장 절차 (자세한 체크리스트: 01_Hardware/차량장착_라이다_캘리브레이션_절차.md)
  0) 평평한 포장면에 차를 세우고 주변 10m를 비운다. 라이다 전원·이더넷 확인.
       python3 lidar_calibrate.py --sectors            # 점이 들어오는지, 방향 감각 확인
  1) 높이: 노면 리턴으로 추정 → 줄자 실측값과 비교(5cm 안이면 합격)
       python3 lidar_calibrate.py --height
  2) 요 오프셋: 차량 정면 정중앙 10m에 사람/콘을 세우고
       python3 lidar_calibrate.py --bearing --expect 0
     (좌측 90°에서 한 번 더 재서 두 값이 비슷한지 교차 확인하면 더 좋다)
  3) 둘 다 만족하면 설정 파일로 저장
       python3 lidar_calibrate.py --height --bearing --expect 0 --write
  4) 마이크 오프셋(젯슨 쪽 값)은 라이다를 기준으로 역산할 수 있다 —
     젯슨을 돌린 상태에서 소리를 내며
       python3 lidar_calibrate.py --audio-offset --samples 20

⚠️ 이 도구는 실물 라이다가 붙은 차에서만 의미가 있다. 저장소에는 장비가 없어 **단위
   로직만**(--selftest) 검증했다. 현장에서 처음 돌릴 때는 --write 없이 숫자부터 확인할 것.

실행:
    python3 lidar_calibrate.py --selftest     # 하드웨어 없이 추정 로직만 검증
    python3 lidar_calibrate.py --sectors      # 30° 구역별 최근접 물체 (방향 감각 확인)
    python3 lidar_calibrate.py --monitor      # 표적을 들고 움직이며 실시간 확인
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lidar_distance_match as lidar  # noqa: E402

DEFAULT_SCANS = 20          # 이 개수의 **서로 다른** 스캔을 모아 중앙값을 쓴다 (10Hz면 약 2초)
HEIGHT_BAND_M = (4.0, 25.0)  # 높이 추정에 쓸 수평거리 구간. 너무 가까우면 차체, 멀면 점이 희박
GROUND_BIN_M = 0.05        # 노면 z 히스토그램 해상도
BEARING_SEARCH_DEG = 45.0  # 표적을 찾을 각도 범위(기대 방향 ±)
BEARING_BAND_M = (3.0, 30.0)
AUDIO_MATCH_DEG = 40.0     # 마이크 오프셋 역산 시 라이다 후보를 찾을 각도 범위


# ---- 스캔 수집 ----------------------------------------------------------------
def collect_scans(scanner, n_scans: int = DEFAULT_SCANS, timeout_sec: float = 15.0,
                  quiet: bool = False):
    """서로 다른 scan_id의 스캔 n개를 모아 리스트로 돌려준다."""
    scans, seen, t0 = [], set(), time.time()
    while len(scans) < n_scans and time.time() - t0 < timeout_sec:
        points, scan_id = scanner.snapshot()
        if scan_id in seen or points.shape[0] == 0:
            time.sleep(0.02)
            continue
        seen.add(scan_id)
        scans.append(points)
        if not quiet:
            print(f"\r    스캔 {len(scans)}/{n_scans} 수집 ({points.shape[0]:,}점)", end="")
    if not quiet:
        print()
    if not scans:
        raise RuntimeError("스캔을 한 개도 받지 못했습니다 — 라이다 전원·이더넷·포트(2368)를 확인하세요. "
                           "`sudo tcpdump -i <iface> udp port 2368`로 패킷부터 확인할 것.")
    return scans


# ---- 1) 장착 높이 --------------------------------------------------------------
def estimate_height_one(points: np.ndarray, band=HEIGHT_BAND_M):
    """한 스캔에서 노면 z를 찾아 장착 높이를 추정한다.

    평평한 노면은 센서 z = -h 평면에 몰려 있다. 그 아래로는 리턴이 없으니, 수평거리
    band 안의 점을 5cm 빈으로 세어 **가장 아래쪽 밀집 빈**을 노면으로 본다
    (가장 큰 빈을 쓰면 벽·차량 측면이 더 많을 때 그쪽으로 끌려간다).
    """
    rng, z = points[:, 3], points[:, 2]
    sel = z[(rng >= band[0]) & (rng <= band[1])]
    if sel.size < 200:
        return None
    lo, hi = float(sel.min()), float(sel.max())
    if hi - lo < GROUND_BIN_M:
        return -float(np.median(sel))
    bins = np.arange(lo, hi + GROUND_BIN_M, GROUND_BIN_M)
    counts, edges = np.histogram(sel, bins=bins)
    if counts.max() < 50:
        return None
    # 최다 빈의 20% 이상인 빈들 중 가장 아래쪽을 노면으로 채택
    dense = np.where(counts >= max(50, 0.2 * counts.max()))[0]
    if dense.size == 0:
        return None
    k = int(dense[0])
    ground = sel[(sel >= edges[k] - GROUND_BIN_M) & (sel <= edges[k + 1] + GROUND_BIN_M)]
    if ground.size < 50:
        return None
    return -float(np.median(ground))


def estimate_height(scans, band=HEIGHT_BAND_M):
    vals = [v for v in (estimate_height_one(p, band) for p in scans) if v is not None]
    if not vals:
        return None
    arr = np.array(vals)
    return {"height_m": float(np.median(arr)), "spread_m": float(arr.max() - arr.min()),
            "scans_used": len(vals), "scans_total": len(scans)}


# ---- 2) 요(yaw) 오프셋 ---------------------------------------------------------
def measure_bearing_one(points: np.ndarray, center_deg: float, cfg,
                        search_deg=BEARING_SEARCH_DEG, band=BEARING_BAND_M):
    """center_deg ± search_deg 안에서 표적(가장 가까운 유효 클러스터)의 방위를 돌려준다."""
    kept = lidar.filter_ground(points, cfg.mount_height_m, cfg)
    if kept.shape[0] == 0:
        return None
    rel = lidar._wrap180(kept[:, 4] - center_deg)
    window = kept[np.abs(rel) <= search_deg]
    if window.shape[0] == 0:
        return None
    cands = [c for c in lidar.cluster_candidates(window, center_deg, cfg)
             if band[0] <= c["distance_m"] <= band[1]]
    return cands[0] if cands else None


def estimate_offset(scans, expect_deg: float, cfg, search_deg=BEARING_SEARCH_DEG,
                    band=BEARING_BAND_M):
    """표적을 차량 좌표계 expect_deg에 놓았을 때의 요 오프셋을 추정한다.

    스캐너를 오프셋 0으로 돌렸으므로 azimuth 컬럼은 **라이다 원시 방위**다.
        offset = 측정된 원시 방위 - 기대 방위
    이 값을 설정에 넣으면 LidarScanner가 azimuth에서 빼서 차량 좌표계로 맞춘다.
    """
    center = expect_deg + cfg.mount_offset_deg   # 현재 설정을 출발점으로 삼아 탐색
    found = [c for c in (measure_bearing_one(p, center, cfg, search_deg, band) for p in scans)
             if c is not None]
    if not found:
        return None
    az = np.array([c["azimuth_deg"] for c in found])
    dist = np.array([c["distance_m"] for c in found])
    pts = np.array([c["points"] for c in found])
    # 각도 중앙값은 랩어라운드에 주의 — center 기준 상대값으로 모아 계산한다
    rel = lidar._wrap180(az - center)
    raw_az = float(lidar._wrap180(center + np.median(rel)))
    return {"raw_azimuth_deg": raw_az,
            "offset_deg": float(lidar._wrap180(raw_az - expect_deg)),
            "distance_m": float(np.median(dist)),
            "spread_deg": float(rel.max() - rel.min()),
            "points_median": int(np.median(pts)),
            "scans_used": len(found), "scans_total": len(scans)}


# ---- 3) 구역별 최근접 (방향 감각 확인) ------------------------------------------
def sector_report(scans, cfg, step_deg: float = 30.0):
    """30° 구역마다 최근접 유효 클러스터 거리를 뽑는다.

    "앞이 비었고 왼쪽에 벽이 있다"가 화면·숫자와 맞는지 눈으로 확인하는 용도다 —
    오프셋이 90°씩 틀어져 있으면 여기서 바로 드러난다.
    """
    merged = np.concatenate(scans, axis=0) if len(scans) > 1 else scans[0]
    kept = lidar.filter_ground(merged, cfg.mount_height_m, cfg)
    rows = []
    for center in np.arange(-180.0, 180.0, step_deg):
        rel = lidar._wrap180(kept[:, 4] - center)
        window = kept[np.abs(rel) <= step_deg / 2.0]
        cands = lidar.cluster_candidates(window, float(center), cfg)
        rows.append({"center_deg": float(center),
                     "distance_m": cands[0]["distance_m"] if cands else None,
                     "points": cands[0]["points"] if cands else 0,
                     "window_points": int(window.shape[0])})
    return rows


def print_sector_report(rows):
    print(f"  {'구역(차량 기준)':>16s} {'최근접':>9s} {'점':>7s} {'창 안 점':>9s}")
    for r in rows:
        d = "—" if r["distance_m"] is None else f"{r['distance_m']:6.1f}m"
        name = _sector_name(r["center_deg"])
        print(f"  {name:>16s} {d:>9s} {r['points']:7d} {r['window_points']:9d}")


def _sector_name(center_deg: float) -> str:
    base = {0: "전방", 90: "좌측", 180: "후방", -90: "우측"}
    for deg, label in base.items():
        if abs(lidar._wrap180(center_deg - deg)) < 1e-6:
            return f"{label} {center_deg:+.0f}°"
    return f"{center_deg:+.0f}°"


# ---- 4) 마이크 오프셋 역산 -----------------------------------------------------
def estimate_audio_offset(scanner, receiver, cfg, samples: int = 20,
                          timeout_sec: float = 180.0, match_deg: float = AUDIO_MATCH_DEG):
    """젯슨이 보낸 theta와 라이다가 본 방위를 비교해 마이크 오프셋 보정량을 낸다.

    젯슨은 theta = (raw_doa - mic_offset) % 360으로 보낸다. 라이다가 같은 물체를
    azimuth에서 봤다면 잔차 delta = theta - azimuth 이고, 마이크 오프셋을
    **현재 값 + median(delta)** 로 바꾸면 두 센서가 맞는다.

    ⚠️ 라이다 오프셋·높이를 먼저 맞춘 뒤에 할 것 — 기준이 틀어진 상태로 재면 그 오차가
       그대로 마이크 오프셋으로 옮겨간다. 그리고 소리를 낸 물체가 라이다가 집은 그 물체인지는
       사람이 봐야 한다(주변에 다른 차가 없을 때 재는 것이 가장 깨끗하다).
    """
    deltas, rows, t0 = [], [], time.time()
    print(f"[*] 젯슨 감지를 기다립니다 — 표적(사이렌/경적 스피커) 앞에서 소리를 내세요. "
          f"목표 {samples}건, 최대 {timeout_sec:.0f}초")
    while len(deltas) < samples and time.time() - t0 < timeout_sec:
        packet = receiver.get_new()
        if packet is None:
            time.sleep(0.05)
            continue
        if packet.get("class") not in ("siren", "car_horn", "motorcycle"):
            continue
        if not packet.get("theta_ok", True):
            print("    [skip] 젯슨이 방향을 불신(theta_ok=false)")
            continue
        theta = float(packet["theta"])
        points, _ = scanner.snapshot()
        cand = measure_bearing_one(points, theta, cfg, match_deg, BEARING_BAND_M)
        if cand is None:
            print(f"    [skip] theta={theta:6.1f}° — 라이다가 그 방향에서 물체를 못 찾음")
            continue
        delta = float(lidar._wrap180(theta - cand["azimuth_deg"]))
        deltas.append(delta)
        rows.append({"theta": theta, "lidar_az": cand["azimuth_deg"],
                     "distance_m": cand["distance_m"], "delta": delta})
        print(f"    {len(deltas):2d}/{samples}  소리 theta={theta:6.1f}°  "
              f"라이다 방위={cand['azimuth_deg']:6.1f}°  거리={cand['distance_m']:5.1f}m  "
              f"차이={delta:+6.1f}°")
    if not deltas:
        return None
    arr = np.array(deltas)
    return {"delta_median_deg": float(np.median(arr)),
            "delta_std_deg": float(np.std(arr)),
            "samples": len(deltas), "rows": rows}


# ---- 실시간 확인 --------------------------------------------------------------
def run_monitor(scanner, cfg, center_deg: float, interval_sec: float = 0.3):
    """표적을 들고 돌아다니며 라이다가 보는 방위·거리를 실시간으로 확인한다."""
    print(f"[*] 실시간 확인 — 기준 방향 {center_deg:+.0f}° ±{BEARING_SEARCH_DEG:.0f}°. 종료: Ctrl+C\n")
    try:
        while True:
            points, scan_id = scanner.snapshot()
            cand = measure_bearing_one(points, center_deg, cfg) if points.shape[0] else None
            if cand is None:
                print(f"\r[{time.strftime('%H:%M:%S')}] scan={scan_id:6d} 표적 없음              ",
                      end="")
            else:
                print(f"\r[{time.strftime('%H:%M:%S')}] scan={scan_id:6d} "
                      f"방위={cand['azimuth_deg']:+7.1f}° 거리={cand['distance_m']:6.2f}m "
                      f"폭={cand['width_m']:4.2f}m 점={cand['points']:4d}   ", end="")
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("\n[*] 종료합니다.")


# ---- 셀프테스트 ---------------------------------------------------------------
def _scene_with_mount(height_m: float, offset_deg: float, target_bearing_deg: float,
                      target_range_m: float = 10.0):
    """장착 높이·요 오프셋이 주어진 차량에서 라이다가 실제로 뱉을 **원시** 스캔을 합성한다.

    원시 방위 = 차량 방위 + offset. (LidarScanner가 offset을 빼서 차량 좌표계로 만든다.)
    """
    scene = lidar._make_road_scene(mount_height_m=height_m, car_range_m=target_range_m,
                                   car_azimuth_deg=target_bearing_deg + offset_deg)
    return scene


def run_selftest():
    ok = True

    def check(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  {name:34s} {detail}  [{'PASS' if good else 'FAIL'}]")

    print("\n[1] 장착 높이 추정 (노면 리턴으로 역산)")
    for truth in (1.55, 1.70, 2.05):
        scans = [_scene_with_mount(truth, 0.0, 0.0) for _ in range(3)]
        got = estimate_height(scans)
        err = None if got is None else abs(got["height_m"] - truth)
        check(f"실제 {truth:.2f}m",
              err is not None and err <= 0.06,
              f"추정={None if got is None else round(got['height_m'], 3)}m "
              f"오차={None if err is None else round(err, 3)}m")

    print("\n[2] 요 오프셋 추정 (표적을 아는 방향에 둔 경우)")
    for truth_offset, expect in ((0.0, 0.0), (12.0, 0.0), (-25.0, 0.0), (8.0, 90.0)):
        cfg = lidar.LidarConfig(mount_height_m=1.7)
        scans = [_scene_with_mount(1.7, truth_offset, expect) for _ in range(3)]
        got = estimate_offset(scans, expect, cfg)
        err = None if got is None else abs(lidar._wrap180(got["offset_deg"] - truth_offset))
        check(f"실제 {truth_offset:+.0f}° (표적 {expect:+.0f}°)",
              err is not None and err <= 2.0,
              f"추정={None if got is None else round(got['offset_deg'], 1)}° "
              f"오차={None if err is None else round(float(err), 2)}°")

    print("\n[3] 추정한 오프셋을 넣으면 거리 매칭이 맞아떨어지는가")
    truth_offset, truth_h = 18.0, 1.85
    raw = _scene_with_mount(truth_h, truth_offset, 0.0, target_range_m=14.0)
    cfg = lidar.LidarConfig(mount_height_m=truth_h)
    got_off = estimate_offset([raw], 0.0, cfg)
    corrected = raw.copy()
    corrected[:, 4] = lidar._wrap180(raw[:, 4] - got_off["offset_deg"])
    cfg2 = lidar.LidarConfig(mount_height_m=truth_h,
                             mount_offset_deg=got_off["offset_deg"])
    match = lidar.match_distance(corrected, 0.0, sigma_deg=7.0, cfg=cfg2)
    check("보정 후 전방 14m 차량 확정",
          match is not None and not match["blind"] and abs(match["distance_m"] - 14.0) < 1.0,
          f"got={lidar._fmt(match)}")
    # 보정하지 않으면 18° 틀어진 채라 전방 창(±14°)에서 못 찾는다
    match_raw = lidar.match_distance(raw, 0.0, sigma_deg=7.0,
                                     cfg=lidar.LidarConfig(mount_height_m=truth_h))
    check("보정 전에는 전방에서 못 찾는다", match_raw is None, f"got={lidar._fmt(match_raw)}")

    print("\n[4] 구역 리포트")
    rows = sector_report([_scene_with_mount(1.7, 0.0, 90.0, target_range_m=12.0)],
                         lidar.LidarConfig(mount_height_m=1.7))
    left = [r for r in rows if abs(r["center_deg"] - 90.0) < 1e-6][0]
    front = [r for r in rows if abs(r["center_deg"]) < 1e-6][0]
    check("좌측 90°에만 물체가 보인다",
          left["distance_m"] is not None and abs(left["distance_m"] - 12.0) < 1.0
          and front["distance_m"] is None,
          f"좌측={None if left['distance_m'] is None else round(left['distance_m'], 1)}m "
          f"전방={front['distance_m']}")

    print(f"\n[selftest] {'ALL PASS' if ok else 'FAIL 있음'}")
    return ok


# ---- 실행 ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true", help="하드웨어 없이 추정 로직만 검증")
    p.add_argument("--sectors", action="store_true", help="30° 구역별 최근접 물체 출력 (방향 감각 확인)")
    p.add_argument("--height", action="store_true", help="노면 리턴으로 장착 높이 추정")
    p.add_argument("--bearing", action="store_true", help="표적 방위로 요 오프셋 추정 (--expect 필요)")
    p.add_argument("--expect", type=float, default=None,
                   help="표적을 놓은 차량 좌표계 방향(도, 전방=0, 반시계+). 예: 정면 0, 좌측 90")
    p.add_argument("--search", type=float, default=BEARING_SEARCH_DEG,
                   help="표적을 찾을 각도 범위(기대 방향 ±도)")
    p.add_argument("--monitor", action="store_true", help="표적을 들고 움직이며 실시간 확인")
    p.add_argument("--audio-offset", action="store_true",
                   help="젯슨 theta와 라이다 방위를 비교해 마이크 오프셋 보정량 역산")
    p.add_argument("--samples", type=int, default=20, help="--audio-offset에서 모을 감지 건수")
    p.add_argument("--audio-port", type=int, default=9870, help="젯슨 패킷 UDP 포트")
    p.add_argument("--scans", type=int, default=DEFAULT_SCANS, help="추정에 쓸 스캔 개수")
    p.add_argument("--port", type=int, default=lidar.VLP16_UDP_PORT, help="VLP-16 UDP 포트")
    p.add_argument("--config", default=None, help="장착 설정 JSON 경로")
    p.add_argument("--write", action="store_true",
                   help="추정 결과를 설정 파일에 저장 (확인 후에만 쓸 것)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.selftest:
        raise SystemExit(0 if run_selftest() else 1)

    if args.bearing and args.expect is None:
        raise SystemExit("--bearing에는 --expect(표적을 놓은 방향)가 필요합니다. 예: --bearing --expect 0")
    if not any((args.sectors, args.height, args.bearing, args.monitor, args.audio_offset)):
        raise SystemExit("할 일을 지정하세요: --sectors / --height / --bearing --expect 0 / "
                         "--monitor / --audio-offset  (자세한 절차는 파일 상단 주석 참고)")

    cfg = lidar.load_config(args.config, quiet=False)

    # 오프셋 추정은 **원시 방위**가 필요하므로 스캐너 보정을 0으로 고정한다.
    # 그 외 모드는 설정값을 그대로 적용해 "지금 설정으로 무엇이 보이는가"를 본다.
    raw_frame = args.bearing or args.monitor
    scanner = lidar.LidarScanner(mount_offset_deg=0.0 if raw_frame else None, port=args.port)
    scanner.start()
    if raw_frame:
        print("[*] 스캐너를 원시 방위(오프셋 0°)로 돌립니다 — 측정값이 라이다 자체 좌표계입니다.")

    measured = {}
    try:
        if args.monitor:
            center = (args.expect or 0.0) + cfg.mount_offset_deg
            run_monitor(scanner, cfg, center)
            return

        print(f"[*] 스캔 수집 중 (목표 {args.scans}개)...")
        scans = collect_scans(scanner, args.scans)

        if args.sectors:
            print("\n=== 30° 구역별 최근접 물체 "
                  f"(오프셋 {cfg.mount_offset_deg:+.1f}° 적용, 높이 {cfg.mount_height_m:.2f}m 기준) ===")
            print_sector_report(sector_report(scans, cfg))
            print("\n  → 비워둔 방향이 '—'로, 벽·건물 쪽이 실제 거리로 나오는지 확인하세요.\n"
                  "    90° 단위로 틀어져 보이면 --bearing으로 오프셋부터 재세요.")

        if args.height:
            print("\n=== 장착 높이 추정 ===")
            got = estimate_height(scans)
            if got is None:
                print("  [!] 노면 리턴을 찾지 못했습니다 — 평평한 포장면에서, 차 주변 "
                      f"{HEIGHT_BAND_M[0]:.0f}~{HEIGHT_BAND_M[1]:.0f}m를 비우고 다시 재세요.")
            else:
                measured["mount_height_m"] = round(got["height_m"], 3)
                print(f"  추정 높이 : {got['height_m']:.3f} m   (스캔 {got['scans_used']}/{got['scans_total']}개 "
                      f"사용, 스캔 간 편차 {got['spread_m']*100:.1f} cm)")
                print(f"  현재 설정 : {cfg.mount_height_m:.3f} m")
                print("  → 줄자 실측값(노면→라이다 원점)과 5cm 안에서 맞는지 반드시 대조할 것. "
                      "안 맞으면 노면이 기울었거나 band 설정을 손봐야 합니다.")

        if args.bearing:
            print(f"\n=== 요(yaw) 오프셋 추정 — 표적 기대 방향 {args.expect:+.1f}° ===")
            got = estimate_offset(scans, args.expect, cfg, args.search)
            if got is None:
                print(f"  [!] 기대 방향 ±{args.search:.0f}° / "
                      f"{BEARING_BAND_M[0]:.0f}~{BEARING_BAND_M[1]:.0f}m 안에서 표적을 못 찾았습니다.\n"
                      "      표적이 사람 1명이면 10m 안쪽에 두고, 주변에 다른 물체가 없게 하세요. "
                      "높이 설정이 틀어도 표적이 통째로 걸러질 수 있습니다.")
            else:
                measured["mount_offset_deg"] = round(got["offset_deg"], 2)
                print(f"  표적 원시 방위 : {got['raw_azimuth_deg']:+.2f}°  (거리 {got['distance_m']:.2f}m, "
                      f"점 {got['points_median']}개)")
                print(f"  추정 오프셋    : {got['offset_deg']:+.2f}°   (현재 설정 {cfg.mount_offset_deg:+.2f}°)")
                print(f"  스캔 간 흩어짐 : {got['spread_deg']:.2f}°  "
                      f"(스캔 {got['scans_used']}/{got['scans_total']}개 사용)")
                print("  → 흩어짐이 2° 넘으면 표적이 흔들렸거나 다른 물체를 집은 것입니다. 다시 재세요.\n"
                      "    가능하면 좌측 90°에서도 재서 두 추정이 1~2° 안에서 일치하는지 교차 확인할 것.")

        if args.audio_offset:
            print("\n=== 마이크 오프셋 역산 (라이다 기준) ===")
            if abs(cfg.mount_offset_deg) < 1e-9 and "mount_offset_deg" not in measured:
                print("  [!] 라이다 오프셋이 아직 0°입니다 — 라이다를 먼저 보정하지 않으면 "
                      "그 오차가 마이크 값으로 옮겨갑니다.")
            from audio_receiver import AudioDetectionReceiver

            receiver = AudioDetectionReceiver(port=args.audio_port).start()
            try:
                got = estimate_audio_offset(scanner, receiver, cfg, args.samples)
            finally:
                receiver.stop()
            if got is None:
                print("  [!] 비교할 감지를 못 모았습니다 — 젯슨이 돌고 있는지, theta_ok가 true인지 확인.")
            else:
                print(f"\n  측정 {got['samples']}건  차이 중앙값 {got['delta_median_deg']:+.2f}° "
                      f"(표준편차 {got['delta_std_deg']:.2f}°)")
                print(f"  → 젯슨 jetson_audio_sender.py의 MOUNT_OFFSET_DEG를 "
                      f"**현재 값 + {got['delta_median_deg']:+.2f}°**로 바꾸세요 "
                      f"(또는 --mount-offset으로 전달).")
                print("    표준편차가 10°를 넘으면 소리를 낸 물체와 라이다가 집은 물체가 "
                      "다를 수 있습니다 — 주변을 비우고 다시 재세요.")

        if args.write:
            if not measured:
                print("\n[!] 저장할 실측값이 없습니다 (--height / --bearing으로 측정한 값만 저장합니다).")
            else:
                for key, value in measured.items():
                    setattr(cfg, key, value)
                path = lidar.save_config(cfg, args.config)
                print(f"\n[*] 설정 저장: {path}")
                print(f"    mount_offset_deg={cfg.mount_offset_deg:+.2f}  "
                      f"mount_height_m={cfg.mount_height_m:.3f}")
                print("    live_demo.py를 다시 띄우면 이 값이 적용됩니다.")
        elif measured:
            print("\n[i] 숫자가 납득되면 같은 명령에 --write를 붙여 설정 파일에 저장하세요.")
    finally:
        scanner.stop()


if __name__ == "__main__":
    main()
