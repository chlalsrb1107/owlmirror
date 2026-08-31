"""
preflight_view.py — [노트북] 카메라 4대 + VLP-16이 동시에 살아있는지 한 화면으로 확인한다.

숫자 로그만으로는 "다 연결됐다"를 확신하기 어려워서, 왼쪽에 카메라 4분할, 오른쪽에 라이다 BEV를
같이 띄운다. 촬영 전 사전 점검(그리고 차량 장착 후 방향 배정 검증)에 쓰는 용도다.

⚠️ 방향 배정 검증에 이 뷰어를 쓸 것: 루프랙에 올린 뒤 각 방향에서 손을 흔들어, 화면의
   FRONT/LEFT/RIGHT/REAR 칸이 실제 방향과 맞는지 확인한다. 어긋나면 live_demo.py의
   CAMERA_SERIAL을 고친다 — 배선과 소프트웨어 순서가 어긋나도 코드는 알아채지 못한다.

라벨을 영문으로 둔 이유: cv2.putText는 한글을 못 그리고, 한글 폰트(fonts-nanum)가 아직 설치되지
않은 환경에서도 이 점검 도구만은 무조건 떠야 하기 때문이다.

실행:
    python3 preflight_view.py                  # 라이브 창 (q 종료)
    python3 preflight_view.py --snapshot a.png # 1장만 저장하고 종료 (창 없이)
    python3 preflight_view.py --no-lidar       # 카메라만
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
# gxipy는 시스템에 설치돼 있지 않고 daheng_ws_1에 vendoring 돼 있다
_VENDOR = Path(os.path.expanduser("~/Documents/daheng_ws_1/vendor"))
if _VENDOR.is_dir():
    sys.path.insert(0, str(_VENDOR))

from live_demo import CAMERA_SERIAL, configure_exposure  # noqa: E402  (설정을 live_demo와 공유)

CELL_W, CELL_H = 512, 300      # 카메라 한 칸 크기
BEV_SIZE = 600                 # BEV 패널 한 변
BEV_RANGE_M = 20.0             # BEV에 그릴 최대 반경
BLIND_RADIUS_M = 6.0           # ui_state_spec.md §5 사각지대
ORDER = ["front", "left", "right", "rear"]


_DEVICE_MANAGER = None  # ⚠️ 전역으로 살려둘 것 — live_demo.py의 같은 주석 참고
                        #    (GC되면 gx_close_lib()가 불려 열린 카메라가 전부 죽는다)


def open_cameras():
    import gxipy as gx

    global _DEVICE_MANAGER
    _DEVICE_MANAGER = gx.DeviceManager()
    dm = _DEVICE_MANAGER
    count, _ = dm.update_device_list()
    cams = {}
    for name in ORDER:
        sn = CAMERA_SERIAL.get(name, "")
        try:
            cam = dm.open_device_by_sn(sn)
            cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
            configure_exposure(cam, gx)   # 노출 상한 — live_demo.py와 동일 설정
            cam.BalanceWhiteAuto.set(gx.GxAutoEntry.CONTINUOUS)
            cam.stream_on()
            cams[name] = cam
            print(f"  [OK]   {name:5s} sn={sn}")
        except Exception as e:  # noqa: BLE001
            cams[name] = None
            print(f"  [FAIL] {name:5s} sn={sn}  {type(e).__name__}: {e}")
    print(f"  검출된 장치 수: {count}")
    return cams


LAST_GRAB_ERROR = {}


def grab(cam, name="?", attempts=3, timeout_ms=1000):
    """프레임 1장. 실패 사유는 LAST_GRAB_ERROR에 남긴다 — 조용히 None을 돌려주면
    화면에는 NO SIGNAL만 뜨고 왜인지 알 수 없다."""
    if cam is None:
        LAST_GRAB_ERROR[name] = "장치가 열리지 않음"
        return None
    reason = "?"
    for _ in range(attempts):
        try:
            raw = cam.data_stream[0].get_image(timeout=timeout_ms)
            if raw is None:
                reason = "get_image=None"
                continue
            rgb = raw.convert("RGB")
            if rgb is None:
                reason = f"convert=None(status={raw.get_status()})"
                continue
            arr = rgb.get_numpy_array()
            if arr is None:
                reason = "numpy=None"
                continue
            LAST_GRAB_ERROR.pop(name, None)
            return arr
        except Exception as e:  # noqa: BLE001
            reason = f"{type(e).__name__}: {e}"
    LAST_GRAB_ERROR[name] = reason
    return None


def camera_grid(cams, cv2):
    """2x2 격자. 프레임이 없으면 그 칸을 회색 NO SIGNAL로 채운다."""
    cells = []
    for name in ORDER:
        frame = grab(cams.get(name), name)
        if frame is None:
            cell = np.full((CELL_H, CELL_W, 3), 40, np.uint8)
            cv2.putText(cell, "NO SIGNAL", (CELL_W // 2 - 90, CELL_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 220), 2, cv2.LINE_AA)
            cv2.putText(cell, LAST_GRAB_ERROR.get(name, "?")[:56], (12, CELL_H - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 220), 1, cv2.LINE_AA)
            ok = False
        else:
            cell = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (CELL_W, CELL_H))
            ok = True
        colour = (120, 220, 120) if ok else (60, 60, 220)
        cv2.rectangle(cell, (0, 0), (CELL_W - 1, CELL_H - 1), colour, 2)
        cv2.rectangle(cell, (0, 0), (CELL_W, 30), (0, 0, 0), -1)
        label = f"{name.upper():5s} {CAMERA_SERIAL.get(name, '?')}"
        cv2.putText(cell, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
        if ok:
            cv2.putText(cell, f"mean {cell.mean():.0f}", (CELL_W - 110, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        cells.append(cell)
    return np.vstack([np.hstack(cells[:2]), np.hstack(cells[2:])])


def bev_panel(points, cv2, scanner_status=None):
    """라이다 포인트를 위에서 내려다본 그림으로. 차량은 중앙, 위쪽이 전방(+x)."""
    img = np.full((BEV_SIZE, BEV_SIZE, 3), 18, np.uint8)
    cx = cy = BEV_SIZE // 2
    ppm = (BEV_SIZE / 2) / BEV_RANGE_M  # pixel per metre

    for r in (5, 10, 15, 20):  # 거리 링
        cv2.circle(img, (cx, cy), int(r * ppm), (55, 55, 55), 1)
        cv2.putText(img, f"{r}m", (cx + int(r * ppm) - 26, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (110, 110, 110), 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), int(BLIND_RADIUS_M * ppm), (0, 90, 130), 1)  # 사각지대 경계
    cv2.line(img, (cx, 0), (cx, BEV_SIZE), (45, 45, 45), 1)
    cv2.line(img, (0, cy), (BEV_SIZE, cy), (45, 45, 45), 1)

    if points is not None and len(points):
        x, y, rng = points[:, 0], points[:, 1], points[:, 3]
        keep = rng <= BEV_RANGE_M
        # 화면 좌표: +x(전방)가 위, +y(좌측)가 왼쪽
        px = (cx - y[keep] * ppm).astype(np.int32)
        py = (cy - x[keep] * ppm).astype(np.int32)
        inb = (px >= 0) & (px < BEV_SIZE) & (py >= 0) & (py < BEV_SIZE)
        px, py, pr = px[inb], py[inb], rng[keep][inb]
        # 사각지대 안은 어둡게, 밖은 거리에 따라 밝게 — 거리 신뢰도 차이를 눈으로 구분
        for j in range(len(px)):
            c = (90, 90, 90) if pr[j] < BLIND_RADIUS_M else (90, 230, 120)
            img[py[j], px[j]] = c
    cv2.circle(img, (cx, cy), 5, (255, 255, 255), -1)  # 차량 위치
    cv2.putText(img, "FRONT", (cx - 26, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    n = 0 if points is None else len(points)
    colour = (120, 220, 120) if n else (60, 60, 220)
    cv2.rectangle(img, (0, BEV_SIZE - 30), (BEV_SIZE, BEV_SIZE), (0, 0, 0), -1)
    txt = f"LiDAR {n:,} pts" if n else "LiDAR NO DATA"
    if scanner_status and scanner_status.get("error"):
        txt = f"LiDAR ERROR: {scanner_status['error'][:44]}"
    cv2.putText(img, txt, (10, BEV_SIZE - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", metavar="PNG", help="1장만 저장하고 종료 (창 없이)")
    ap.add_argument("--no-lidar", action="store_true")
    ap.add_argument("--warmup", type=float, default=1.5, help="자동노출/첫 스캔 안정화 대기(초)")
    args = ap.parse_args()

    import cv2

    print("[*] 카메라 여는 중...")
    cams = open_cameras()

    scanner = None
    if not args.no_lidar:
        import lidar_distance_match as lidar
        scanner = lidar.LidarScanner()
        scanner.start()
        print("[*] LiDAR 스캐너 시작")

    print(f"[*] 안정화 대기 {args.warmup}s...")
    t0 = time.time()
    while time.time() - t0 < args.warmup:
        for name in ORDER:
            grab(cams.get(name), name, attempts=1)  # 자동노출이 수렴하도록 버리는 프레임
        time.sleep(0.05)

    def compose():
        grid = camera_grid(cams, cv2)
        if scanner is None:
            return grid
        bev = bev_panel(scanner.latest_points(), cv2, scanner.status())
        pad = np.full((grid.shape[0], BEV_SIZE, 3), 18, np.uint8)
        pad[:BEV_SIZE] = bev
        return np.hstack([grid, pad])

    try:
        if args.snapshot:
            cv2.imwrite(args.snapshot, compose())
            print(f"[*] 저장: {args.snapshot}")
        else:
            print("[*] 라이브 표시 — 창에서 q 누르면 종료")
            while True:
                cv2.imshow("olbbaemireo preflight - cameras + LiDAR", compose())
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()
    finally:
        for cam in cams.values():
            if cam is not None:
                try:
                    cam.stream_off(); cam.close_device()
                except Exception:  # noqa: BLE001
                    pass
        if scanner is not None:
            scanner.stop()


if __name__ == "__main__":
    main()
