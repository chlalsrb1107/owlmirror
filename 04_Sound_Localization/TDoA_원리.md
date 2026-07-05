# TDoA (Time Difference of Arrival) 원리

## 개념

음원에서 방출된 소리가 **마이크 i**와 **마이크 j**에 도달하는 시간 차이를 측정하여 음원의 방향을 추정하는 기술.

```
음원 S
  │
  ├──(d_i)──→ 마이크 i (도달 시각 t_i)
  └──(d_j)──→ 마이크 j (도달 시각 t_j)

TDoA_ij = t_i - t_j = (d_i - d_j) / c
```

- `c` = 음속 ≈ 343 m/s (20°C, 차량 내부 보정 필요)
- 마이크 쌍 수 = C(6,2) = **15쌍** (6-Mic 어레이 기준)

---

## 기하학적 관계

2D 평면에서 마이크 i, j 위치와 TDoA로 **쌍곡선**이 정의되고, 여러 쌍의 교점이 음원 위치가 된다.

```
TDoA_ij = (1/c) * (||S - p_i|| - ||S - p_j||)
```

원거리 음원(파 프론트가 평면파) 가정 시 **방위각 θ**만으로 단순화:

```
TDoA_ij ≈ (d_ij * cos(θ - φ_ij)) / c

d_ij: 마이크 i-j 간 거리
φ_ij: 마이크 쌍의 기준 각도
```

---

## 음속 보정

| 온도 (°C) | 음속 (m/s) |
|---|---|
| 0 | 331.3 |
| 20 | 343.2 |
| 40 | 354.7 |

```python
def speed_of_sound(temp_celsius: float) -> float:
    return 331.3 * (1 + temp_celsius / 273.15) ** 0.5
```

차량 실내 온도 센서(OBD-II) 또는 고정값 20°C 사용.

---

## 마이크 어레이 배치 (6-Mic 원형)

```
        Mic 0 (0°)
    Mic 5     Mic 1
  (300°)       (60°)

  Mic 4     Mic 2
  (240°)     (120°)
      Mic 3 (180°)

반지름 r = 4.3cm (ReSpeaker v2.1 기준)
```

마이크 좌표:
```python
import numpy as np

r = 0.043  # meters
n_mics = 6
angles = np.linspace(0, 2 * np.pi, n_mics, endpoint=False)
mic_positions = np.array([[r * np.cos(a), r * np.sin(a)] for a in angles])
```

---

## 관련 노트

- [[04_Sound_Localization/GCC_PHAT_구현]] — TDoA 추정 알고리즘
- [[04_Sound_Localization/빔포밍_알고리즘]] — DoA 추정 심화
