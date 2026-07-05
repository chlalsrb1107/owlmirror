# DoA → 카메라 매핑

## 좌표계 정의

```
차량 전방 = 0° (또는 360°)
우측      = 90°
후방      = 180°
좌측      = 270°

카메라 Pan 범위: -180° ~ +180° (또는 0° ~ 360°)
```

마이크 어레이와 카메라의 기준 방향(전방)을 물리적으로 정렬하거나, 오프셋 보정값을 설정한다.

---

## 어안 카메라 ROI 매핑

카메라를 고정하고 소프트웨어로 ROI를 이동시키는 방식.

```python
import numpy as np
import cv2

class FisheyeTracker:
    def __init__(self, frame_w=1920, frame_h=1080,
                 roi_w=640, roi_h=480, camera_fov=160):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.roi_w = roi_w
        self.roi_h = roi_h
        self.px_per_deg = frame_w / camera_fov
        # 전방(0°)이 프레임 중앙에 오도록 오프셋 설정
        self.forward_px = frame_w // 2

    def doa_to_roi(self, doa_deg: float) -> tuple[int, int, int, int]:
        """
        DoA 각도로부터 ROI (x1, y1, x2, y2) 계산.
        전방 0° 기준, 우측 양수, 좌측 음수.
        """
        # 전방 기준 상대 각도 (-180 ~ +180)
        rel_deg = (doa_deg + 180) % 360 - 180
        center_x = int(self.forward_px + rel_deg * self.px_per_deg)
        center_x = np.clip(center_x, self.roi_w // 2,
                           self.frame_w - self.roi_w // 2)
        x1 = center_x - self.roi_w // 2
        y1 = (self.frame_h - self.roi_h) // 2
        return x1, y1, x1 + self.roi_w, y1 + self.roi_h

    def get_roi_frame(self, full_frame: np.ndarray,
                      doa_deg: float) -> np.ndarray:
        x1, y1, x2, y2 = self.doa_to_roi(doa_deg)
        return full_frame[y1:y2, x1:x2].copy()
```

---

## PTZ 카메라 Pan/Tilt 제어 (VISCA 프로토콜)

Sony VISCA 프로토콜은 PTZ 카메라 제어에 널리 사용된다.

```python
import serial

class VISCAController:
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600):
        self.ser = serial.Serial(port, baudrate, timeout=1)

    def _send(self, cmd: bytes):
        self.ser.write(cmd)

    def pan_tilt_absolute(self, pan_deg: float, tilt_deg: float,
                          pan_speed: int = 10, tilt_speed: int = 10):
        # VISCA 각도 단위 변환 (제조사별 상이)
        pan_pos = int(pan_deg * 182.044)   # 예: 0x0000 ~ 0xFFFF
        tilt_pos = int(tilt_deg * 182.044)

        pan_h = (pan_pos >> 8) & 0xFF
        pan_l = pan_pos & 0xFF
        tilt_h = (tilt_pos >> 8) & 0xFF
        tilt_l = tilt_pos & 0xFF

        cmd = bytes([
            0x81, 0x01, 0x06, 0x02,
            pan_speed, tilt_speed,
            (pan_h >> 4) & 0x0F, pan_h & 0x0F,
            (pan_l >> 4) & 0x0F, pan_l & 0x0F,
            (tilt_h >> 4) & 0x0F, tilt_h & 0x0F,
            (tilt_l >> 4) & 0x0F, tilt_l & 0x0F,
            0xFF
        ])
        self._send(cmd)

    def close(self):
        self.ser.close()
```

---

## 추적 전략

1. **즉시 반응**: DoA가 현재 뷰에서 ±30° 이상 벗어나면 즉시 카메라 이동
2. **부드러운 추적**: 칼만 필터 DoA가 점진적으로 변하면 속도 제한을 두고 이동
3. **음원 소멸 후 복귀**: 경보 종료 3초 후 전방(0°)으로 복귀
