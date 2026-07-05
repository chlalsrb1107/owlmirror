2# Jetson Orin Nano 개발 환경

## 사양 요약

| 항목 | Jetson Orin Nano **Super** Developer Kit |
|---|---|
| CPU | 6-core Arm Cortex-A78AE |
| GPU | 1024-core NVIDIA Ampere, 32 Tensor Cores |
| AI 성능 | **67 TOPS** (Super: 일반 대비 약 1.5배 향상) |
| 메모리 | 8GB LPDDR5 (CPU/GPU 공유) |
| 저장 | NVMe SSD 권장 (MicroSD는 I/O 병목) |
| AI 가속 | DLA 2-core, TensorRT 10.x |
| 카메라 | MIPI CSI-2 (2x 2-lane) |
| OS | JetPack 6.x (Ubuntu 22.04) |

> **Super** 버전은 클럭 향상으로 동일 모델 대비 약 50% 추론 속도 향상. FP16 TensorRT 모델 실행에 유리.

---

## JetPack 설치 후 기본 설정

```bash
# 파워 모드 최대 성능으로 설정
sudo nvpmodel -m 0
sudo jetson_clocks

# CUDA, cuDNN, TensorRT 버전 확인
nvcc --version
python3 -c "import tensorrt; print(tensorrt.__version__)"

# PyTorch (Jetson 전용 빌드)
# NVIDIA 공식 NGC 또는 Jetson Zoo에서 wheel 다운로드
pip3 install torch torchvision torchaudio --index-url <jetson-pytorch-url>
```

---

## 오디오 처리 라이브러리

```bash
pip3 install sounddevice numpy scipy librosa
pip3 install pyusb  # ReSpeaker 제어

# GStreamer (영상 파이프라인)
sudo apt-get install gstreamer1.0-tools gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad python3-gst-1.0
```

---

## AI 추론 스택

```
훈련 (PC / 클라우드)          추론 (Jetson)
─────────────────────         ──────────────────────
PyTorch (.pth)           →    ONNX Export
                         →    TensorRT .engine 변환
                         →    Python TRT Runtime
```

### ONNX → TensorRT 변환

```bash
# trtexec 사용
trtexec --onnx=sound_classifier.onnx \
        --saveEngine=sound_classifier.engine \
        --fp16 \
        --workspace=512
```

---

## 리소스 모니터링

```bash
# Jetson 전용 모니터
sudo pip3 install jetson-stats
jtop
```

---

## 연결 구성도

```
[ReSpeaker USB]  ──USB──▶  [Jetson Orin Nano]  ──HDMI──▶  [차량 디스플레이]
[PTZ 카메라]     ──USB/RTSP─▶  (GStreamer)
[서보 컨트롤러]  ──UART/PWM─▶  (Pan/Tilt 제어)
```
