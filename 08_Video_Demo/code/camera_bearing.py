"""
camera_bearing.py — YOLO가 잡은 박스를 **차량 좌표계 방위각**으로 환산한다 (2026-10-15 신규).

이 파일이 최종 파이프라인의 마지막 연결 고리다.

    소리 → 방향(DoA) → 그 방향 카메라 → YOLO가 대상을 지목 → **박스 → 방위각(여기)**
    → 그 방위에서 라이다가 거리 측정 → 화면에 거리 표시

왜 필요한가: 지금까지는 라이다에게 **소리 방향**(σ 5~7°, 주행 중에는 더)을 물어봤다.
그 창 안에 차가 두 대 있으면 어느 쪽이 사이렌을 울리는 차인지 라이다는 모른다.
카메라가 경광등을 보고 대상을 지목하면 방위를 **1° 수준**으로 알 수 있고, 그 좁은 창으로
라이다를 조회하면 "소리를 낸 바로 그 물체"의 거리가 나온다. 즉 카메라는 *무엇*을,
라이다는 *얼마나 먼지*를 담당하고, 소리는 *어디를 볼지*를 정한다.

기하 (카메라는 수평으로 정렬돼 있고 좌우 반전이 없다고 가정)
    화면 안 각도 : a = atan((u - cx) / fx)      (오른쪽이 +)
    차량 방위각  : theta = (카메라 yaw + trim) - a
    차량 좌표계는 전방 0°, **반시계 +**인데 화면 오른쪽은 시계 방향이라 부호가 뒤집힌다.
    (이 부호를 빠뜨리면 좌우가 통째로 반대가 된다 — bev_render._screen_angle과 같은 함정)

⚠️ 화각(fx) 불일치 — 현장에서 가장 먼저 확인할 것
    `calibration/cam*_calib/results/calibration.json`의 fx ≈ 2069 px / 폭 2048 px로 환산하면
    수평화각이 **약 52.7°**다. 그런데 하드웨어 문서에는 **2.8mm 렌즈**(이 센서에서 약 103°)로
    적혀 있다. 둘 다 맞을 수는 없다. 박스→방위 환산이 이 값에 직접 매여 있어서, 틀리면
    거리는 맞는데 **엉뚱한 물체**의 거리를 집게 된다.
      · 52.7°가 맞다면 카메라 4대(90° 간격)는 방위를 **37°씩 비워둔다** — 그 구역의 대상은
        카메라로 지목할 수 없고 소리 방향 경로로만 처리된다
      · 103°가 맞다면 시야가 겹치므로 공백은 없다
    확인: 차량 정면 정중앙 10m에 표적을 두고 `python3 camera_bearing.py --report`로
    예상 화각을 본 뒤, 표적이 화면 가운데에 오는지 / 화면 가장자리가 실제로 몇 도인지
    대조한다. 실제 화각이 다르면 `08_Video_Demo/config/camera_mount.json`의
    `hfov_deg`로 덮어쓸 수 있다(캘리브레이션 파일은 건드리지 않는다).

실행:
    python3 camera_bearing.py --report     # 카메라별 화각·커버리지·공백 구역
    python3 camera_bearing.py --selftest   # 환산 로직 검증 (하드웨어 불필요)
"""

import argparse
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_DIR = REPO_ROOT / "08_Video_Demo" / "calibration"
CONFIG_PATH = REPO_ROOT / "08_Video_Demo" / "config" / "camera_mount.json"

# 방향별 담당 캠 번호 — 설치 시 이 번호로 캘리브레이션된 물리 카메라를 해당 방향에 붙인다.
# (live_demo.CAMERA_CALIB_ID가 여기를 import해 쓴다. 두 군데 두면 반드시 어긋난다)
CAMERA_CALIB_ID = {"front": 4, "left": 1, "right": 2, "rear": 3}

# 카메라가 바라보는 방향 (차량 좌표계, 전방 0°, 반시계 +)
CAMERA_YAW_DEG = {"front": 0.0, "left": 90.0, "rear": 180.0, "right": -90.0}

# 장착 미세 보정(도). 업체 시공이라 90° 정배치를 기대하지만, 1~2°만 틀어져도 먼 대상에서는
# 어긋난다(30m에서 2° = 1.05m). ⚠️ 현장에서 아는 방향의 표적으로 확인해 채울 것 —
# config/camera_mount.json의 yaw_trim_deg로 넣으면 코드를 안 고쳐도 된다.
CAMERA_YAW_TRIM_DEG = {"front": 0.0, "left": 0.0, "right": 0.0, "rear": 0.0}

# 화각 override(도). 비어 있으면 calibration.json의 fx로 계산한다. 위 ⚠️ 참고.
CAMERA_HFOV_OVERRIDE_DEG = {}

_calib_cache = {}
_config_loaded = False


def load_config(path=None, quiet: bool = True):
    """config/camera_mount.json에서 yaw_trim_deg / hfov_deg를 읽어 적용한다."""
    global _config_loaded
    path = Path(path) if path is not None else CONFIG_PATH
    _config_loaded = True
    if not path.exists():
        if not quiet:
            print(f"[i] 카메라 장착 설정 없음({path}) — 90° 정배치·캘리브레이션 fx를 그대로 씁니다.")
        return False
    with open(path) as f:
        data = json.load(f)
    for camera, value in (data.get("yaw_trim_deg") or {}).items():
        if camera in CAMERA_YAW_TRIM_DEG:
            CAMERA_YAW_TRIM_DEG[camera] = float(value)
    for camera, value in (data.get("hfov_deg") or {}).items():
        CAMERA_HFOV_OVERRIDE_DEG[camera] = float(value)
    if not quiet:
        print(f"[*] 카메라 장착 설정 적용: {path}  trim={CAMERA_YAW_TRIM_DEG} "
              f"hfov_override={CAMERA_HFOV_OVERRIDE_DEG or '없음'}")
    return True


def load_calibration(camera: str):
    """calibration.json 원본 dict. 파일이 없으면 None (Detection 기반 거리만 꺼진다)."""
    if camera in _calib_cache:
        return _calib_cache[camera]
    cam_id = CAMERA_CALIB_ID.get(camera)
    if cam_id is None:
        _calib_cache[camera] = None
        return None
    path = CALIBRATION_DIR / f"cam{cam_id}_calib" / "results" / "calibration.json"
    if not path.exists():
        _calib_cache[camera] = None
        return None
    with open(path) as f:
        _calib_cache[camera] = json.load(f)
    return _calib_cache[camera]


def intrinsics(camera: str):
    """(fx, cx, width) — 박스→방위 환산에 필요한 최소값. 없으면 None.

    hfov override가 있으면 그 화각에 맞는 fx를 역산해 쓴다(렌즈가 캘리브레이션 때와
    다른 경우를 코드 수정 없이 넘기기 위한 장치).
    """
    if not _config_loaded:
        load_config()
    calib = load_calibration(camera)
    if calib is None:
        return None
    width = float(calib["image_size"][0])
    cx = float(calib["camera_matrix"][0][2])
    fx = float(calib["camera_matrix"][0][0])
    hfov = CAMERA_HFOV_OVERRIDE_DEG.get(camera)
    if hfov:
        fx = (width / 2.0) / math.tan(math.radians(hfov) / 2.0)
        cx = width / 2.0
    return fx, cx, width


def hfov_deg(camera: str):
    """수평 화각(도). 모르면 None."""
    got = intrinsics(camera)
    if got is None:
        return None
    fx, cx, width = got
    # cx가 중앙에서 벗어나 있으면 좌우 화각이 다르다 — 둘을 합쳐 전체 화각으로 본다
    return math.degrees(math.atan(cx / fx) + math.atan((width - cx) / fx))


def camera_yaw_deg(camera: str) -> float:
    if not _config_loaded:
        load_config()
    return CAMERA_YAW_DEG[camera] + CAMERA_YAW_TRIM_DEG.get(camera, 0.0)


def bearing_from_pixel(camera: str, u_px: float):
    """화면 가로 좌표 → 차량 좌표계 방위각(도, 전방 0°, 반시계 +). 모르면 None."""
    got = intrinsics(camera)
    if got is None:
        return None
    fx, cx, _ = got
    angle_in_image = math.degrees(math.atan((float(u_px) - cx) / fx))
    return _wrap180(camera_yaw_deg(camera) - angle_in_image)


def bearing_from_box(camera: str, box, min_half_deg: float = 1.0):
    """YOLO 박스(원본 프레임 좌표 x0,y0,x1,y1) → 방위 정보. 환산 불가면 None.

    반환: {"bearing_deg": 중심 방위, "half_width_deg": 박스 절반 폭(각도),
           "edges_deg": (왼쪽 끝, 오른쪽 끝), "camera": 카메라 이름}

    half_width_deg가 그대로 라이다 각도 창이 된다 — 박스가 좁으면(멀리 있으면) 창도 좁아져
    옆 차선 차량을 집지 않는다.
    """
    if box is None:
        return None
    got = intrinsics(camera)
    if got is None:
        return None
    x0, _y0, x1, _y1 = (float(v) for v in box)
    if x1 <= x0:
        return None
    center = bearing_from_pixel(camera, (x0 + x1) / 2.0)
    left_edge = bearing_from_pixel(camera, x0)      # 화면 왼쪽 = 방위각이 더 큰 쪽(반시계 +)
    right_edge = bearing_from_pixel(camera, x1)
    half = abs(_wrap180(left_edge - right_edge)) / 2.0
    return {"bearing_deg": center, "half_width_deg": max(half, min_half_deg),
            "edges_deg": (left_edge, right_edge), "camera": camera}


def _wrap180(deg: float) -> float:
    return (float(deg) + 180.0) % 360.0 - 180.0


def coverage_report():
    """카메라별 화각과 4대 합친 커버리지. 공백 구역이 있으면 같이 돌려준다."""
    rows, covered = [], []
    for camera in ("front", "left", "rear", "right"):
        fov = hfov_deg(camera)
        got = intrinsics(camera)
        rows.append({"camera": camera, "calib_id": CAMERA_CALIB_ID[camera],
                     "yaw_deg": camera_yaw_deg(camera), "hfov_deg": fov,
                     "fx": None if got is None else got[0],
                     "override": camera in CAMERA_HFOV_OVERRIDE_DEG})
        if fov is not None:
            covered.append((camera_yaw_deg(camera), fov))
    gaps = []
    if len(covered) == 4:
        order = sorted(covered, key=lambda c: c[0])
        for (yaw_a, fov_a), (yaw_b, fov_b) in zip(order, order[1:] + [(order[0][0] + 360.0,
                                                                      order[0][1])]):
            gap = (yaw_b - fov_b / 2.0) - (yaw_a + fov_a / 2.0)
            if gap > 0.5:
                gaps.append({"from_deg": yaw_a + fov_a / 2.0, "to_deg": yaw_b - fov_b / 2.0,
                             "width_deg": gap})
    return rows, gaps


def print_report():
    rows, gaps = coverage_report()
    print("\n=== 카메라 화각·방위 커버리지 ===")
    print(f"  {'방향':>6s} {'캠':>4s} {'바라보는 방위':>12s} {'fx(px)':>9s} {'수평화각':>9s} {'담당 구역':>20s}")
    for r in rows:
        if r["hfov_deg"] is None:
            print(f"  {r['camera']:>6s} {r['calib_id']:>4d} {'—':>12s} {'캘리브레이션 없음':>9s}")
            continue
        lo = r["yaw_deg"] - r["hfov_deg"] / 2.0
        hi = r["yaw_deg"] + r["hfov_deg"] / 2.0
        mark = " (override)" if r["override"] else ""
        print(f"  {r['camera']:>6s} {r['calib_id']:>4d} {r['yaw_deg']:>11.1f}° "
              f"{r['fx']:>9.1f} {r['hfov_deg']:>8.1f}° {f'{lo:+.0f}° ~ {hi:+.0f}°':>20s}{mark}")
    if gaps:
        total = sum(g["width_deg"] for g in gaps)
        print(f"\n  ⚠️ 카메라가 못 보는 구역이 {len(gaps)}곳, 합계 {total:.0f}°입니다:")
        for g in gaps:
            print(f"     {g['from_deg']:+.0f}° ~ {g['to_deg']:+.0f}°  ({g['width_deg']:.0f}°)")
        print("     이 방위의 대상은 YOLO로 지목할 수 없어 **소리 방향 + 라이다**만으로 처리됩니다")
        print("     (거리는 나오지만 '그 물체가 소리를 낸 물체'라는 보장은 약해집니다).")
        print("     렌즈가 실제로는 더 넓다면(2.8mm = 약 103°) config/camera_mount.json의")
        print("     hfov_deg로 덮어쓰세요 — 화각이 틀리면 방위 환산 자체가 틀립니다.")
    else:
        print("\n  카메라 4대가 360°를 빈틈없이 덮습니다.")
    return rows, gaps


# ---- 셀프테스트 ---------------------------------------------------------------
def run_selftest():
    ok = True

    def check(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  {name:38s} {detail}  [{'PASS' if good else 'FAIL'}]")

    got = intrinsics("front")
    if got is None:
        print("  [!] calibration.json이 없어 환산 테스트를 건너뜁니다")
        return False
    fx, cx, width = got

    print("\n[1] 화면 좌표 → 방위각 (전방 카메라)")
    check("중앙 픽셀은 카메라 방위(0°)",
          abs(bearing_from_pixel("front", cx)) < 1e-6,
          f"{bearing_from_pixel('front', cx):+.2f}°")
    right = bearing_from_pixel("front", width - 1)
    check("화면 오른쪽은 음(시계) 방향", right < 0, f"{right:+.1f}°")
    left = bearing_from_pixel("front", 0)
    check("화면 왼쪽은 양(반시계) 방향", left > 0, f"{left:+.1f}°")
    check("좌우 끝 폭이 화각과 일치",
          abs(abs(left - right) - hfov_deg("front")) < 0.2,
          f"끝-끝 {abs(left - right):.1f}° vs 화각 {hfov_deg('front'):.1f}°")

    print("\n[2] 카메라별 기준 방위 (90° 정배치)")
    for camera, want in (("front", 0.0), ("left", 90.0), ("rear", 180.0), ("right", -90.0)):
        got_b = bearing_from_pixel(camera, intrinsics(camera)[1])
        check(f"{camera} 중앙 = {want:+.0f}°",
              abs(_wrap180(got_b - want)) < 1e-6, f"{got_b:+.1f}°")

    print("\n[3] 후방 카메라의 좌우 — 뒤를 보면 화면 오른쪽이 차량 왼쪽")
    rear_right = bearing_from_pixel("rear", width - 1)
    check("후방 화면 오른쪽은 좌후방(90°~180°)", 90.0 < rear_right < 180.0, f"{rear_right:+.1f}°")

    print("\n[4] 박스 → 방위·폭")
    box = (cx - 100, 300, cx + 100, 700)      # 중앙에 걸친 박스
    info = bearing_from_box("front", box)
    check("중앙 박스의 방위는 0°", abs(info["bearing_deg"]) < 1e-6,
          f"{info['bearing_deg']:+.2f}°")
    expect_half = math.degrees(math.atan(100.0 / fx))
    check("박스 절반 폭이 각도와 일치",
          abs(info["half_width_deg"] - expect_half) < 0.05,
          f"{info['half_width_deg']:.2f}° (기대 {expect_half:.2f}°)")

    far = bearing_from_box("front", (cx - 15, 500, cx + 15, 560))   # 멀리 있는 작은 박스
    check("먼 대상의 창이 더 좁다", far["half_width_deg"] < info["half_width_deg"],
          f"{far['half_width_deg']:.2f}° < {info['half_width_deg']:.2f}°")

    print("\n[5] 장착 trim 반영")
    CAMERA_YAW_TRIM_DEG["front"] = 3.0
    try:
        check("trim이 방위에 더해진다",
              abs(bearing_from_pixel("front", cx) - 3.0) < 1e-6,
              f"{bearing_from_pixel('front', cx):+.2f}°")
    finally:
        CAMERA_YAW_TRIM_DEG["front"] = 0.0

    print_report()
    print(f"\n[selftest] {'ALL PASS' if ok else 'FAIL 있음'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="환산 로직 검증")
    parser.add_argument("--report", action="store_true", help="카메라별 화각·커버리지 출력")
    parser.add_argument("--config", default=None, help="카메라 장착 설정 JSON 경로")
    args = parser.parse_args()
    load_config(args.config, quiet=False)
    if args.selftest:
        raise SystemExit(0 if run_selftest() else 1)
    if args.report:
        print_report()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
