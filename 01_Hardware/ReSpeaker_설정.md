# ReSpeaker Mic Array v2.0 설정

> ⚠️ **2026-07-05 정정**: 이 문서는 원래 v3.0(XVF-3800) 기준으로 작성되었으나, 실측 결과 확정 하드웨어는 **ReSpeaker 4 Mic Array v2.0 (UAC1.0)** 이다. 아래 "하드웨어 DoA 읽기" 절의 XVF-3800/HID 코드는 v3.0 전용이라 v2.0에는 그대로 적용 불가 — 별도 재작성 필요 (Seeed 공식 `tuning.py`, VID:PID `2886:0018` 기준).

## 하드웨어 사양

| 항목 | 사양 |
|---|---|
| 모델 | ReSpeaker 4 Mic Array v2.0 |
| USB ID | `2886:0018` (UAC1.0) |
| 마이크 수 | 4개 (원형 배치) |
| 인터페이스 | USB 2.0 (USB Audio Class 1.0) |
| 샘플링 레이트 | 16kHz 고정 |
| 채널 | **실측 6ch** — ch0: AEC 처리된 단일 채널, ch1~4: raw mic 1~4, ch5: playback reference |
| 내장 알고리즘 | AEC, NS, AGC, 빔포밍, DoA (온보드 DSP) |
| DoA 출력 방식 | USB HID (v3.0과 프로토콜 다름, 별도 확인 필요) |
| 동작 전압 | 5V USB |

---

## Jetson Orin Nano Super 연결

```bash
# 장치 확인
lsusb | grep -i seeed
# → Bus 001 Device 006: ID 2886:0018 Seeed Technology Co., Ltd. ReSpeaker 4 Mic Array (UAC1.0)

# ALSA 장치 확인
arecord -l
# "ArrayUAC10" 카드로 잡힘 (카드 번호는 시스템마다 다를 수 있음, 예: 카드 2)

# 6채널 녹음 테스트 (16kHz, 16bit) — 4ch가 아니라 6ch임에 주의
arecord -D hw:2,0 -r 16000 -c 6 -f S16_LE -d 5 test.wav

# Python(PyAudio)에서 장치 자동탐색 + 6ch 캡처 예시:
# 03_Audio_Classification/code/realtime_classify.py 참고 (MIC_CHANNEL_INDEX=0 이 처리된 채널)
```

---

## Python SDK 설치

```bash
pip3 install pyusb click numpy
# XMOS XVF-3800용 udev 규칙 추가 (권한 문제 방지)
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="2886", MODE="0666"' | \
    sudo tee /etc/udev/rules.d/99-respeaker.rules
sudo udevadm control --reload-rules
```

---

## 하드웨어 DoA 읽기

XVF-3800 내장 DSP가 실시간으로 DoA를 계산하여 HID로 출력한다.

```python
import usb.core
import usb.util
import time

VENDOR_ID  = 0x2886
PRODUCT_ID = 0x0019  # v3.0

class ReSpeakerV3:
    def __init__(self):
        self.dev = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
        if self.dev is None:
            raise RuntimeError("ReSpeaker v3.0 not found")
        self.dev.set_configuration()

    def get_doa(self) -> int:
        """하드웨어 DoA 읽기. 반환값: 0~359 (도)"""
        # XVF-3800 HID 명령 (Seeed 공식 SDK 참고)
        cmd = [0, 0, 0x08, 0, 0, 0, 0, 0]
        self.dev.ctrl_transfer(0xC0, 0, 0, 0x80, cmd)
        data = self.dev.ctrl_transfer(0xC0, 0, 0, 0x80, 8)
        doa = int.from_bytes(data[0:2], 'little')
        return doa

    def is_voice_active(self) -> bool:
        """VAD (Voice Activity Detection) 상태"""
        data = self.dev.ctrl_transfer(0xC0, 0, 0, 0x80, 8)
        return bool(data[2] & 0x01)

if __name__ == "__main__":
    mic = ReSpeakerV3()
    while True:
        doa = mic.get_doa()
        vad = mic.is_voice_active()
        print(f"DoA: {doa:3d}°  VAD: {vad}")
        time.sleep(0.1)
```

> v3.0은 v2.1과 Product ID가 다르므로 기존 `tuning.py` 그대로 사용 불가. Seeed 공식 GitHub에서 v3.0 SDK 확인 필요.

---

## 주요 파라미터 설정 (XVF-3800)

```python
# 빔포밍 모드 설정
# 0: Omni, 1: Cardioid 빔포밍, 2: Super-cardioid
mic.dev.ctrl_transfer(0x40, 0, 0, 0x00, [0, 0, 0x01])

# AEC 활성화 (에코 제거)
mic.dev.ctrl_transfer(0x40, 0, 0, 0x10, [1])

# 잡음 억제 강도 설정 (0~3)
mic.dev.ctrl_transfer(0x40, 0, 0, 0x20, [2])
```

---

## 차량 환경 DoA 정확도 현실

| 환경 | 예상 DoA 오차 |
|---|---|
| 정적 실내 (테스트 환경) | ~5° |
| 도심 주행 (저속) | 10~15° |
| 고속도로 주행 | 15~25° (바람 노이즈 심각) |
| 비/우천 | 20°+ (신뢰 불가) |

**이 때문에 Depth 카메라 병행이 필수적이다.** 오디오는 방향의 초기 단서를 제공하고, 카메라가 실물을 확인하는 구조로 설계한다.

→ [[01_Hardware/카메라_디스플레이_연결]] 참고

---

## 차량 장착 고려사항

- **위치**: 차량 루프 또는 대시보드 중앙 (전방 180° 커버)
- **방진**: 마이크 하단에 방진 고무 패드 부착 (엔진 진동 차단)
- **방풍**: 차량 내부 장착 시 바람 노이즈 크게 감소
- **케이블**: USB-C 연장 케이블 사용 시 USB 3.0 이상 권장 (오디오 데이터 손실 방지)
