"""
doa_camera_select.py — ReSpeaker 4 Mic Array v2.0 온보드(펌웨어) DoA를 읽어
9/8 영상 제출용 3카메라(좌/우/후방) 데모의 방향→카메라 매핑을 계산한다.

배경: 00_Overview/2026-08-25_9.8_영상제출_촬영_계획.md
최종 구현(카메라 4대+LiDAR, 05_Camera_Tracking/DoA_카메라_매핑.md의 select_camera)과 달리,
이 데모는 방향추정을 04_Sound_Localization/code/gcc_phat.py(자체 구현, 미보정)가 아니라
ReSpeaker 펌웨어 온보드 DoA(USB Tuning 인터페이스)로 대체한다.

좌표계: 차량 전방(북)=0°, 반시계 +. 동=우측 카메라, 서=좌측 카메라, 남=후방 카메라(기본 표시).
북(전방)은 담당 카메라가 없어 "전방 확인" 텍스트로 대체한다.

하드웨어 요구:
    pip install pyusb
    (Ubuntu) sudo 없이 접근하려면 udev 규칙 필요:
        SUBSYSTEM=="usb", ATTRS{idVendor}=="2886", ATTRS{idProduct}=="0018", MODE="0666"

⚠️ DOAANGLE 레지스터 id(=21)는 ReSpeaker 공식 usb_4_mic_array/tuning.py 참고값이며, 이 리포에는
   실물 장비가 없어 검증하지 못했다. --live로 실제 장비 연결 후 각도가 정상 출력되는지 반드시 확인할 것.

실행:
    python3 doa_camera_select.py --selftest   # 하드웨어 없이 매핑 로직만 검증
    python3 doa_camera_select.py --live       # 실제 장비로 DoA + 선택된 카메라 실시간 출력
"""

import argparse
import struct
import time

VENDOR_ID = 0x2886
PRODUCT_ID = 0x0018
DOAANGLE_ID = 21  # ReSpeaker 공식 PARAMETERS 테이블 참고값, 실물로 검증 필요
CTRL_TIMEOUT_MS = 100000

# 차량 장착 후 실측 보정 필요: ReSpeaker가 출력하는 raw 각도 중 어느 값이
# 차량 정면(0°)에 해당하는지 오프셋으로 보정한다. --live로 정면에서 소리를 내며 확인할 것.
MOUNT_OFFSET_DEG = 0.0

# 동/서/남 90°씩 + 북(전방, 카메라 없음) 90°
CAMERA_RANGES = {
    "front_no_camera": (-45, 45),   # 북
    "left": (45, 135),              # 서
    "rear": (135, 225),             # 남 (기본 표시)
    "right": (-135, -45),           # 동
}


class Tuning:
    """ReSpeaker 4 Mic Array v2.0 USB Tuning 인터페이스에서 DOAANGLE만 읽는 최소 구현."""

    def __init__(self, dev):
        self.dev = dev

    @property
    def direction(self) -> int:
        import usb.util

        cmd = 0x80 | 0x40  # read + int type
        response = self.dev.ctrl_transfer(
            usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0, cmd, DOAANGLE_ID, 8, CTRL_TIMEOUT_MS,
        )
        value, _ = struct.unpack(b"ii", response.tobytes())
        return value


def find_device():
    import usb.core

    dev = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
    if dev is None:
        raise RuntimeError(
            "ReSpeaker 4 Mic Array v2.0을 찾을 수 없습니다. USB 연결 및 udev 규칙을 확인하세요."
        )
    return dev


def select_camera(doa_deg: float, mount_offset_deg: float = MOUNT_OFFSET_DEG) -> str:
    """DoA 각도(0~359)를 받아 담당 카메라("left"/"right"/"rear") 또는 "front_no_camera"를 반환."""
    rel = (doa_deg - mount_offset_deg + 180) % 360 - 180  # -180~180 정규화
    for name, (lo, hi) in CAMERA_RANGES.items():
        if lo <= rel < hi:
            return name
    return "rear"  # 경계값 예외 시 기본 표시 카메라로


def run_selftest():
    cases = [(0, "front_no_camera"), (90, "left"), (180, "rear"), (270, "right"), (-90, "right"), (44, "front_no_camera"), (46, "left")]
    all_ok = True
    for deg, expected in cases:
        got = select_camera(deg)
        ok = got == expected
        all_ok &= ok
        print(f"  doa={deg:6.1f}  expected={expected:16s}  got={got:16s}  [{'PASS' if ok else 'FAIL'}]")
    print(f"\n[selftest] {'ALL PASS' if all_ok else 'FAIL 있음'}")
    return all_ok


def run_live(interval_sec: float):
    dev = find_device()
    tuning = Tuning(dev)
    print("[*] ReSpeaker 온보드 DoA 실시간 읽기 시작. 종료: Ctrl+C\n")
    try:
        while True:
            doa = tuning.direction
            camera = select_camera(doa)
            print(f"[{time.strftime('%H:%M:%S')}] doa={doa:3d}deg -> camera={camera}")
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("\n[*] 종료합니다.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="하드웨어 없이 방향->카메라 매핑 로직만 검증")
    parser.add_argument("--live", action="store_true", help="실제 ReSpeaker로 DoA + 선택된 카메라 실시간 출력")
    parser.add_argument("--interval", type=float, default=0.25, help="폴링 주기(초), 기본 인터페이스 스펙(초당4회)과 동일")
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
    elif args.live:
        run_live(args.interval)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
