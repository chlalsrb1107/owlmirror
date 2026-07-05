"""
probe_depth_range.py
Orbbec Gemini 215의 실제 유효 depth 측정거리를 실측하기 위한 진단 스크립트.

배경: 제품 스펙상 "유효 측정거리 0.15~0.70m"로 알려져 있으나, 이 카메라는
depth work mode(Close_Up Precision Mode / Extended Distance Mode)에 따라
실제 유효거리가 크게 달라진다. 기본값은 Close_Up Precision Mode라 아무 설정도
하지 않으면 약 0.33m에서 끊긴다 — "0.70m"를 쓰려면 Extended Distance Mode로
명시적으로 전환해야 한다 (2026-07-05 실측 확인, 00_Overview/현재_상태_요약.md 참고).

실행:
    python3 probe_depth_range.py --mode extended --seconds 20
    python3 probe_depth_range.py --mode closeup

측정 방법: 카메라 정면 축을 따라 손(또는 평평한 물체)을 가까이에서 멀리까지
똑바로 이동시키면서 중앙 영역의 유효 depth 픽셀 비율과 min/max/mean 거리를 출력한다.
"""

import argparse
import time

import numpy as np
from pyorbbecsdk import Context, Pipeline, Config, OBSensorType


def switch_mode(dev, keyword: str):
    modes = dev.get_depth_work_mode_list()
    target = None
    for i in range(modes.get_count()):
        if keyword.lower() in modes[i].name.lower():
            target = modes[i]
            break
    if target is None:
        available = [modes[i].name for i in range(modes.get_count())]
        raise ValueError(f"'{keyword}'에 맞는 depth work mode를 찾을 수 없음. 사용 가능: {available}")
    dev.set_depth_work_mode(target.name)
    time.sleep(2)  # 모드 전환 후 안정화 대기
    print(f"[*] depth work mode -> {dev.get_depth_work_mode()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["closeup", "extended"], default="extended",
                        help="closeup=Close_Up Precision Mode(기본, ~0.15-0.33m), extended=Extended Distance Mode(~0.20-0.72m)")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--crop", type=float, default=0.4,
                        help="중앙 크롭 비율 (0.4 = 중앙 40%%x40%% 영역만 측정)")
    args = parser.parse_args()

    ctx = Context()
    dev = ctx.query_devices().get_device_by_index(0)
    switch_mode(dev, "Close_Up" if args.mode == "closeup" else "Extended")

    pipeline = Pipeline(dev)
    config = Config()
    profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)
    pipeline.start(config)

    t0 = time.time()
    last_print = 0.0
    lo, hi = (1 - args.crop) / 2, (1 + args.crop) / 2
    print(f"[*] {args.seconds}초간 측정 시작 — 손을 카메라 축 방향으로 가까이~멀리 이동시켜보세요")
    try:
        while time.time() - t0 < args.seconds:
            frames = pipeline.wait_for_frames(2000)
            if frames is None:
                continue
            depth = frames.get_depth_frame()
            if depth is None:
                continue
            w, h = depth.get_width(), depth.get_height()
            scale = depth.get_depth_scale()
            data = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(h, w).astype(np.float32) * scale
            region = data[int(h*lo):int(h*hi), int(w*lo):int(w*hi)]
            valid = region[region > 0]

            t = time.time() - t0
            if t - last_print < 0.2:
                continue
            last_print = t
            if valid.size > 0:
                print(f"t={t:5.1f}s  valid={100*valid.size/region.size:5.1f}%  "
                      f"min={valid.min():.0f}mm max={valid.max():.0f}mm mean={valid.mean():.0f}mm")
            else:
                print(f"t={t:5.1f}s  no valid depth")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
