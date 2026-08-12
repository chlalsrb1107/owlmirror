"""
gcc_phat.py — GCC-PHAT 기반 TDoA/DoA 추정 (ReSpeaker v2.0 raw 채널 ch1~4 전용)

⚠️ ReSpeaker v2.0은 6채널로 노출되지만 실제 마이크는 ch1~4뿐이다.
   ch0은 보드에서 AEC 처리된 단일 채널, ch5는 재생 참조(에코 제거용 루프백)라
   물리적 마이크 위치를 갖지 않는다 -> TDoA 계산에 쓰면 안 됨.
   (01_Hardware/ReSpeaker_설정.md 참고)

⚠️ MIC_RADIUS_M / MIC_ANGLES_DEG는 아직 실측되지 않은 자리표시값이다.
   validate_doa.py로 알려진 각도에서 스피커를 재생해보고 실측/교정할 것.
"""

import numpy as np


def speed_of_sound(temp_celsius: float = 20.0) -> float:
    return 331.3 * (1 + temp_celsius / 273.15) ** 0.5


SPEED_OF_SOUND_20C = speed_of_sound(20.0)

# raw mic 채널(ch1~4)이 원형으로 배치되어 있다고 가정.
# TODO(실측 필요): 반경(m)과 채널별 실제 배치각(차량 전방=0°, 반시계 +, ui_state_spec.md의
# theta 좌표계와 통일)을 캘리브레이션할 것. 아래 값은 검증 전 자리표시값.
MIC_RADIUS_M = 0.032
MIC_ANGLES_DEG = [0.0, 90.0, 180.0, 270.0]  # ch1, ch2, ch3, ch4 순서 가정 (미검증)

MIC_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def mic_positions(radius_m: float = MIC_RADIUS_M, angles_deg=None) -> np.ndarray:
    angles_deg = MIC_ANGLES_DEG if angles_deg is None else angles_deg
    angles_rad = np.radians(angles_deg)
    return np.stack([radius_m * np.cos(angles_rad), radius_m * np.sin(angles_rad)], axis=1)


def gcc_phat(sig: np.ndarray, refsig: np.ndarray, fs: int, max_tau: float = None, interp: int = 16):
    """
    두 신호 간 TDoA 추정 (GCC-PHAT).

    Returns:
        tau: sig가 refsig 대비 지연된 시간(초). 양수면 sig가 더 늦게 도착(= refsig가 먼저 도착).
        cc:  상호상관 함수 (디버깅/시각화용)
    """
    n = sig.shape[0] + refsig.shape[0]
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    denom = np.abs(R)
    denom[denom < 1e-15] = 1e-15
    cc = np.fft.irfft(R / denom, n=interp * n)

    max_shift = int(interp * n / 2)
    if max_tau is not None:
        max_shift = max(1, min(int(interp * fs * max_tau), max_shift))

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)
    return tau, cc


def estimate_doa(channels: np.ndarray, fs: int, c: float = SPEED_OF_SOUND_20C,
                  radius_m: float = MIC_RADIUS_M, angles_deg=None, n_grid: int = 360,
                  max_tau: float = None):
    """
    raw mic 4채널(ch1~4) 프레임에서 쌍별 GCC-PHAT TDoA를 구하고,
    격자탐색(0~359°)으로 가장 잘 맞는 방위각을 추정한다.

    Args:
        channels: shape (4, N) — ch1~4 raw mic 샘플 (float, 정규화 권장)
    Returns:
        best_theta_deg, taus: dict[(i,j)] -> tau(초), error_map: shape (n_grid,)
    """
    assert channels.shape[0] == 4, "raw mic 4채널(ch1~4)만 입력해야 함 (ch0/ch5 제외)"

    positions = mic_positions(radius_m, angles_deg)

    taus = {}
    for i, j in MIC_PAIRS:
        # 쌍마다 실제 거리 기준으로 탐색창을 잡는다 (대각선 쌍은 인접 쌍보다 최대지연이 커서
        # radius 하나로 전체 창을 고정하면 대각선 쌍의 진짜 피크가 창 밖으로 잘릴 수 있음)
        pair_max_tau = max_tau
        if pair_max_tau is None:
            pair_dist = float(np.linalg.norm(positions[j] - positions[i]))
            pair_max_tau = pair_dist / c * 1.2  # 이론상 최대 지연보다 약간 여유
        tau, _ = gcc_phat(channels[i], channels[j], fs, max_tau=pair_max_tau)
        taus[(i, j)] = tau

    thetas = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    error_map = np.zeros(n_grid)
    for k, theta in enumerate(thetas):
        u = np.array([np.cos(theta), np.sin(theta)])
        err = 0.0
        for i, j in MIC_PAIRS:
            predicted = np.dot(positions[j] - positions[i], u) / c
            err += (taus[(i, j)] - predicted) ** 2
        error_map[k] = err

    best_idx = int(np.argmin(error_map))
    best_theta_deg = float(np.degrees(thetas[best_idx]))
    return best_theta_deg, taus, error_map


def fractional_delay(x: np.ndarray, delay_samples: float) -> np.ndarray:
    """x를 delay_samples(소수 가능)만큼 지연시킨 신호를 반환 (주파수영역 위상시프트)."""
    n = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n)  # cycles/sample
    phase = np.exp(-1j * 2 * np.pi * freqs * delay_samples)
    return np.fft.irfft(X * phase, n=n)
