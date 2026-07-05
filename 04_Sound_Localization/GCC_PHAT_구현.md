# GCC-PHAT 구현

## 개념

**Generalized Cross-Correlation with Phase Transform (GCC-PHAT)**은 두 신호의 교차 상관을 주파수 도메인에서 계산하고, 진폭을 정규화하여 반향(reverb)과 노이즈에 강인한 TDoA 추정을 제공한다.

```
GCC-PHAT(τ) = IFFT[ X_i(f) * X_j*(f) / |X_i(f) * X_j*(f)| ]

τ_ij = argmax_τ GCC-PHAT(τ)
```

---

## 구현 코드

```python
import numpy as np
from scipy.fft import fft, ifft

def gcc_phat(sig_i: np.ndarray, sig_j: np.ndarray,
             sr: int = 16000, max_tau: float = None) -> tuple[float, np.ndarray]:
    """
    두 신호 간 GCC-PHAT TDoA 추정.

    Returns:
        tau: 추정된 시간 지연 (초)
        cc:  전체 상관 함수
    """
    n = sig_i.shape[0] + sig_j.shape[0]
    # FFT
    SIG_I = fft(sig_i, n=n)
    SIG_J = fft(sig_j, n=n)
    # 교차 스펙트럼
    R = SIG_I * np.conj(SIG_J)
    # PHAT 가중치 (진폭 정규화)
    R /= (np.abs(R) + 1e-8)
    # 역FFT → 상관 함수
    cc = np.real(ifft(R))
    # 최대 지연 범위 제한
    if max_tau is None:
        max_tau = 0.043 / 343.0  # 마이크 반지름 / 음속
    max_shift = int(np.ceil(max_tau * sr))
    # 양쪽 끝 추출 (음수/양수 지연)
    cc = np.concatenate([cc[-max_shift:], cc[:max_shift+1]])
    tau = (np.argmax(cc) - max_shift) / sr
    return tau, cc


def compute_all_tdoas(multichannel: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    6채널 신호에서 15쌍의 TDoA 행렬 계산.

    multichannel: (6, N) ndarray
    Returns: (6, 6) TDoA 행렬 (초 단위)
    """
    n_mics = multichannel.shape[0]
    tdoa_matrix = np.zeros((n_mics, n_mics))
    for i in range(n_mics):
        for j in range(i + 1, n_mics):
            tau, _ = gcc_phat(multichannel[i], multichannel[j], sr)
            tdoa_matrix[i, j] = tau
            tdoa_matrix[j, i] = -tau
    return tdoa_matrix
```

---

## TDoA → 방위각 변환 (최소제곱법)

```python
def tdoa_to_doa(tdoa_matrix: np.ndarray,
                mic_positions: np.ndarray,
                c: float = 343.0) -> float:
    """
    TDoA 행렬과 마이크 위치로 방위각(DoA) 추정.
    원거리 음원 가정 (평면파).

    Returns: 방위각 (도, 0~360)
    """
    n_mics = mic_positions.shape[0]
    A = []
    b = []
    for i in range(n_mics):
        for j in range(i + 1, n_mics):
            d = mic_positions[i] - mic_positions[j]
            A.append(d)
            b.append(c * tdoa_matrix[i, j])
    A = np.array(A)
    b = np.array(b)
    # 단위 방향 벡터 추정 (최소제곱)
    u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    u = u / (np.linalg.norm(u) + 1e-8)
    angle_rad = np.arctan2(u[1], u[0])
    angle_deg = np.degrees(angle_rad) % 360
    return angle_deg
```

---

## 성능 특성

| 조건 | 예상 오차 |
|---|---|
| 직접음 (무반향) | < 3° |
| 실내 반향 중간 | 5~10° |
| 차량 내 장착 (진동 있음) | 8~15° |
| SNR 5dB 이하 | > 15° (신뢰도 낮음) |

신뢰도 낮은 추정은 이전 DoA 값으로 대체하는 **칼만 필터** 적용 권장.
