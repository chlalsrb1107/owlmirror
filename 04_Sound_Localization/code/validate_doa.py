"""
validate_doa.py — GCC-PHAT DoA 추정이 ReSpeaker v2.0 raw 채널(ch1~4)에서
실제로 동작하는지 검증하는 스크립트. (2026-07-08 실차 풍절음 실험 계획의
"사전 점검" 항목: 04_Sound_Localization 참고)

3가지 모드:
    1. --selftest   합성 신호로 알고리즘 자체 정합성만 검증. 마이크 불필요,
                     지금 바로 (Jetson이 아니어도) 돌려서 코드가 맞는지 확인 가능.
    2. --wav FILE    녹음된 6채널 wav 파일로 오프라인 검증
                     (7/8 실험계획 파일명 규칙 예: speed0_pos-dash_deadcat-off_01.wav)
    3. 인자 없음     ReSpeaker 실시간 캡처 -> 콘솔에 추정 각도 실시간 출력.
                     스피커를 알려진 각도(0/90/180/270 등)에 두고 돌리면서
                     출력값이 그 방향으로 안정적으로 수렴하는지 확인할 것.

실행 예:
    python3 validate_doa.py --selftest
    python3 validate_doa.py --wav ../../02_Data_Collection/recordings/test.wav
    python3 validate_doa.py --log doa_log.csv --window 0.5
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from gcc_phat import (
    MIC_ANGLES_DEG,
    MIC_RADIUS_M,
    estimate_doa,
    fractional_delay,
    speed_of_sound,
)

SR = 16000
CHANNELS = 6
RAW_MIC_CHANNEL_INDICES = [1, 2, 3, 4]  # ch0=AEC, ch5=재생 참조는 제외


# ---------------------------------------------------------------------------
# 1) self-test: 마이크 없이 알고리즘 자체를 합성 신호로 검증
# ---------------------------------------------------------------------------

def _circular_error_deg(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def run_selftest(radius_m: float, angles_deg, c: float, verbose: bool = True) -> bool:
    rng = np.random.default_rng(0)
    n_samples = 8192
    base = rng.standard_normal(n_samples)

    from gcc_phat import mic_positions
    positions = mic_positions(radius_m, angles_deg)

    test_angles = list(range(0, 360, 45))
    all_ok = True

    print("[selftest] 잡음 없는 조건 (허용 오차 2°)")
    for true_deg in test_angles:
        u = np.array([np.cos(np.radians(true_deg)), np.sin(np.radians(true_deg))])
        delay_samples = [-(pos @ u) / c * SR for pos in positions]
        channels = np.stack([fractional_delay(base, d) for d in delay_samples])

        est_deg, _, _ = estimate_doa(channels, SR, c=c, radius_m=radius_m, angles_deg=angles_deg)
        err = _circular_error_deg(est_deg, true_deg)
        ok = err < 2.0
        all_ok &= ok
        if verbose:
            status = "PASS" if ok else "FAIL"
            print(f"  true={true_deg:5.1f}  est={est_deg:6.2f}  err={err:5.2f}   [{status}]")

    print("[selftest] SNR ~20dB 잡음 조건 (허용 오차 10°)")
    for true_deg in test_angles:
        u = np.array([np.cos(np.radians(true_deg)), np.sin(np.radians(true_deg))])
        delay_samples = [-(pos @ u) / c * SR for pos in positions]
        noise_scale = 0.1
        channels = np.stack([
            fractional_delay(base, d) + rng.standard_normal(n_samples) * noise_scale
            for d in delay_samples
        ])

        est_deg, _, _ = estimate_doa(channels, SR, c=c, radius_m=radius_m, angles_deg=angles_deg)
        err = _circular_error_deg(est_deg, true_deg)
        ok = err < 10.0
        all_ok &= ok
        if verbose:
            status = "PASS" if ok else "FAIL"
            print(f"  true={true_deg:5.1f}  est={est_deg:6.2f}  err={err:5.2f}   [{status}]")

    print(f"\n[selftest] 결과: {'ALL PASS' if all_ok else 'FAIL 있음 — gcc_phat.py 로직 점검 필요'}")
    return all_ok


# ---------------------------------------------------------------------------
# 2) 녹음 파일(wav) 오프라인 검증
# ---------------------------------------------------------------------------

def run_wav(path: Path, window_sec: float, radius_m: float, angles_deg, c: float,
            log_path: Path = None):
    from scipy.io import wavfile

    fs, data = wavfile.read(path)
    if data.ndim != 2 or data.shape[1] != CHANNELS:
        print(f"[!] 예상과 다른 채널 수: shape={data.shape} (6채널 녹음이어야 함)")
        sys.exit(1)

    data = data.astype(np.float32) / 32768.0
    n_window = int(window_sec * fs)
    n_total = data.shape[0]

    logger = _open_logger(log_path)
    print(f"[*] {path.name} 재생 검증 시작 ({fs}Hz, {n_total/fs:.1f}s, 창 {window_sec}s)")

    t = 0.0
    for start in range(0, n_total - n_window, n_window):
        window = data[start:start + n_window]
        channels = window[:, RAW_MIC_CHANNEL_INDICES].T
        est_deg, taus, _ = estimate_doa(channels, fs, c=c, radius_m=radius_m, angles_deg=angles_deg)
        _report(t, est_deg, taus, logger)
        t += window_sec

    _close_logger(logger)


# ---------------------------------------------------------------------------
# 3) ReSpeaker 실시간 캡처 검증
# ---------------------------------------------------------------------------

def run_live(window_sec: float, radius_m: float, angles_deg, c: float, log_path: Path = None):
    import pyaudio

    def get_respeaker_index(p):
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if "ReSpeaker" in info.get("name", "") and info.get("maxInputChannels", 0) > 0:
                return i
        return None

    p = pyaudio.PyAudio()
    device_index = get_respeaker_index(p)
    if device_index is None:
        print("[!] ReSpeaker 마이크를 찾을 수 없습니다. USB 연결을 확인하세요.")
        p.terminate()
        sys.exit(1)

    dev_info = p.get_device_info_by_index(device_index)
    print(f"[*] ReSpeaker 인식됨: index={device_index}, name={dev_info['name']}")

    stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=SR,
                     input=True, input_device_index=device_index,
                     frames_per_buffer=1024)

    n_samples = int(window_sec * SR)
    logger = _open_logger(log_path)
    print(f"[*] 실시간 DoA 검증 시작 (창 {window_sec}s). 스피커를 알려진 각도에 두고 비교할 것. 종료: Ctrl+C\n")

    t0 = time.time()
    try:
        while True:
            frames = []
            collected = 0
            while collected < n_samples:
                raw = stream.read(1024, exception_on_overflow=False)
                chunk = np.frombuffer(raw, dtype=np.int16).reshape(-1, CHANNELS)
                frames.append(chunk[:, RAW_MIC_CHANNEL_INDICES])
                collected += chunk.shape[0]

            window = np.concatenate(frames)[:n_samples].astype(np.float32) / 32768.0
            channels = window.T

            est_deg, taus, _ = estimate_doa(channels, SR, c=c, radius_m=radius_m, angles_deg=angles_deg)
            _report(time.time() - t0, est_deg, taus, logger)
    except KeyboardInterrupt:
        print("\n[*] 종료합니다.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        _close_logger(logger)


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------

def _open_logger(log_path: Path):
    if log_path is None:
        return None
    is_new = not log_path.exists()
    f = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if is_new:
        writer.writerow(["t", "theta_deg", "tau_01", "tau_02", "tau_03", "tau_12", "tau_13", "tau_23"])
    return f, writer


def _report(t: float, est_deg: float, taus: dict, logger):
    tau_str = " ".join(f"{k}={v*1e6:6.1f}us" for k, v in taus.items())
    print(f"[t={t:6.2f}s] theta={est_deg:6.1f}deg   {tau_str}")
    if logger is not None:
        f, writer = logger
        writer.writerow([f"{t:.3f}", f"{est_deg:.2f}"] + [f"{v:.9f}" for v in taus.values()])
        f.flush()


def _close_logger(logger):
    if logger is not None:
        logger[0].close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="합성 신호로 알고리즘 자체 검증 (마이크 불필요)")
    parser.add_argument("--wav", type=Path, default=None, help="6채널 wav 파일로 오프라인 검증")
    parser.add_argument("--window", type=float, default=0.5, help="추정 1회당 사용하는 오디오 길이(초)")
    parser.add_argument("--radius", type=float, default=MIC_RADIUS_M, help="마이크 배치 반경(m), 미검증 자리표시값")
    parser.add_argument("--angles", type=str, default=None,
                         help="쉼표구분 4개 각도(도), 채널1~4 순서. 기본값은 gcc_phat.MIC_ANGLES_DEG")
    parser.add_argument("--temp", type=float, default=20.0, help="기온(섭씨), 음속 보정용")
    parser.add_argument("--log", type=Path, default=None, help="CSV로 결과 기록")
    args = parser.parse_args()

    angles_deg = MIC_ANGLES_DEG if args.angles is None else [float(x) for x in args.angles.split(",")]
    c = speed_of_sound(args.temp)

    if args.selftest:
        ok = run_selftest(args.radius, angles_deg, c)
        sys.exit(0 if ok else 1)
    elif args.wav is not None:
        run_wav(args.wav, args.window, args.radius, angles_deg, c, args.log)
    else:
        run_live(args.window, args.radius, angles_deg, c, args.log)


if __name__ == "__main__":
    main()
