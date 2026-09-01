"""
jetson_audio_sender.py — [젯슨에서 실행] ReSpeaker 오디오 분류 + DoA를 노트북으로 UDP 전송.

00_Overview/현재_상태_요약.md "시스템 아키텍처 — Jetson/노트북 분리"의 젯슨 쪽 구현이다.
젯슨은 오디오만 전담하고(수음 → PANNs Cnn14+SVM 분류 → 펌웨어 DoA), 아래 JSON 한 줄만
노트북으로 보낸다. 영상·점군은 이 링크로 오가지 않는다.

    {"seq": 1234, "t": 1751250042.318, "class": "siren",
     "conf": 0.91, "score": 1.83, "theta": 194.0, "sigma": 12.0}

| 필드    | 설명                                                              |
|---------|-------------------------------------------------------------------|
| `seq`   | 0부터 증가하는 시퀀스 번호 — 수신측 유실 검출용 (스펙 확장)        |
| `t`     | 분석 구간의 **중심 시각**(구간 시작 시각 아님), 젯슨 시계 기준     |
| `class` | 모델 클래스명 그대로 (car_horn/siren/motorcycle/...) 또는 "none"   |
| `conf`  | 0~1 유사 신뢰도 — ⚠️ 보정된 확률이 아님(아래 참고)                 |
| `score` | SVM decision_function 원시 마진 — 임계값 판단은 이 값으로 할 것    |
| `theta` | **차량 좌표계** 방위각(전방 0°, 반시계 +) — 마운트 오프셋 적용 완료 |
| `sigma` | 방향 오차 표준편차(도) — 노트북 BEV 부채꼴 폭(2×sigma)에 사용      |

⚠️ conf에 대하여: SVM이 `probability=False`로 학습돼 진짜 확률을 낼 수 없다
   (03_Audio_Classification/model_outputs/panns_svm/README.md 참고). conf는 마진에
   softmax를 씌운 **유사** 신뢰도이므로 표시용으로만 쓰고, 임계값 판단은 score로 할 것.

⚠️ theta에 대하여: 이 스크립트가 MOUNT_OFFSET_DEG를 이미 적용해 차량 좌표계로 보낸다.
   따라서 노트북 쪽 select_camera()는 반드시 mount_offset_deg=0.0으로 호출해야 한다
   (이중 적용 방지). 오프셋 실측은 --live로 차량 정면에서 소리를 내며 맞출 것.

⚠️ 전송 주기: 인터페이스 스펙은 초당 4회(250ms)지만, 젯슨 GPU가 인식되지 않아 CPU로만
   추론하면 PANNs Cnn14 1회에 수 초가 걸릴 수 있다(현재_상태_요약.md의 AST 실측 3초→3.9초).
   --interval을 스펙대로 0.25로 낮춰도 추론이 못 따라가면 실제 주기는 그만큼 벌어지며,
   이 스크립트는 종료 시 **실측 달성 주기**를 출력하니 그 값으로 스펙을 맞출지 판단할 것.

실행:
    # 젯슨(실장비): 노트북 IP로 전송
    python3 jetson_audio_sender.py --host 192.168.10.2

    # 노트북(장비 없이): 가짜 감지를 쏴서 수신측/화면을 먼저 검증
    python3 jetson_audio_sender.py --host 127.0.0.1 --simulate
"""

import argparse
import json
import math
import random
import socket
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "03_Audio_Classification" / "code"))

DEFAULT_PORT = 9870

# 노트북이 카메라 전환/경보에 쓰는 클래스. 그 외(children_playing/engine_idling)는 배경음이라
# "none"으로 눌러서 보낸다 — 링크에 불필요한 트래픽을 안 흘리기 위함.
ALERT_CLASSES = {"car_horn", "siren", "motorcycle"}

# 차량 장착 후 실측 보정 필요: ReSpeaker raw 각도 중 차량 정면(0°)에 해당하는 값.
MOUNT_OFFSET_DEG = 0.0

# 방향 오차 표준편차(도). 현재_상태_요약.md "위치추정(방향)"의 ±10~20°를 반영한 기본값.
# GCC-PHAT 교차검증으로 실측되면 그 값으로 교체할 것.
DEFAULT_SIGMA_DEG = 15.0


def softmax_conf(scores):
    """SVM 마진 리스트 → 최고 클래스의 유사 신뢰도(0~1). 보정된 확률이 아님."""
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    return max(exps) / sum(exps)


def to_vehicle_frame(raw_doa_deg: float, mount_offset_deg: float) -> float:
    """ReSpeaker raw 각도 → 차량 좌표계(전방 0°, 반시계 +, 0~360 정규화)."""
    return (raw_doa_deg - mount_offset_deg) % 360.0


class Sender:
    def __init__(self, host: str, port: int):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0

    def send(self, class_name: str, conf: float, score: float,
             theta: float, sigma: float, t_center: float):
        packet = {
            "seq": self.seq,
            "t": round(t_center, 3),
            "class": class_name,
            "conf": round(conf, 3),
            "score": round(score, 3),
            "theta": round(theta, 1),
            "sigma": round(sigma, 1),
        }
        self.sock.sendto(json.dumps(packet).encode("utf-8"), self.addr)
        self.seq += 1
        return packet

    def close(self):
        self.sock.close()


def run_simulate(sender: Sender, interval: float, verbose: bool):
    """장비 없이 노트북 수신측을 검증하기 위한 가짜 송신.

    실제 주행처럼 대부분은 none이고 가끔 한 종류가 여러 프레임 연속으로 잡히게 만든다
    (노트북의 HOLD_SEC 유지/해제 동작을 제대로 타보기 위함).
    """
    print(f"[simulate] {sender.addr}로 가짜 감지 전송. 종료: Ctrl+C\n")
    burst_class, burst_left, theta = None, 0, 0.0
    try:
        while True:
            if burst_left <= 0:
                if random.random() < 0.25:  # 25% 확률로 새 감지 버스트 시작
                    burst_class = random.choice(sorted(ALERT_CLASSES))
                    burst_left = random.randint(2, 5)
                    theta = random.uniform(0, 360)
                else:
                    burst_class, burst_left = None, 0

            if burst_class is None:
                pkt = sender.send("none", 0.0, -1.0, 0.0, DEFAULT_SIGMA_DEG, time.time())
            else:
                theta = (theta + random.uniform(-8, 8)) % 360  # 대상이 조금씩 움직이는 효과
                score = random.uniform(0.4, 2.2)
                pkt = sender.send(burst_class, softmax_conf([score, 0.0, -0.5, -0.8, -1.0]),
                                  score, theta, DEFAULT_SIGMA_DEG, time.time())
                burst_left -= 1

            if verbose or pkt["class"] != "none":
                print(f"  seq={pkt['seq']:5d} class={pkt['class']:11s} "
                      f"score={pkt['score']:+6.2f} theta={pkt['theta']:6.1f}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[simulate] 종료합니다.")


def run_live(sender: Sender, seconds: float, interval: float, min_score: float,
             mount_offset: float, sigma: float, verbose: bool,
             svm_path=None, panns_path=None):
    """실제 ReSpeaker에서 수음 → 분류 → DoA → 전송.

    svm_path/panns_path를 주면 저장소 구조 밖에 있는 가중치도 쓸 수 있다 — 젯슨에는 보통
    저장소 전체가 아니라 파일 몇 개만 올라가므로, 경로를 고정하면 찾지 못한다.
    """
    import numpy as np
    import pyaudio

    import realtime_classify as clf
    from doa_camera_select import Tuning, find_device

    svm_path = Path(svm_path) if svm_path else clf.DEFAULT_SVM_PATH
    panns_path = Path(panns_path) if panns_path else clf.DEFAULT_PANNS_CKPT_PATH
    print(f"[*] 모델 로딩 중...\n    SVM  : {svm_path}\n    PANNs: {panns_path}")
    if not svm_path.exists():
        print(f"[!] SVM 체크포인트가 없습니다: {svm_path}")
        print("    --svm-path 로 실제 위치를 지정하세요.")
        return
    clf.ensure_panns_checkpoint(panns_path)
    panns_model = clf.load_panns_model(panns_path)
    svm, class_names, _ = clf.load_svm(svm_path)
    print(f"[*] 클래스: {class_names}")

    p = pyaudio.PyAudio()
    device_index = clf.get_respeaker_index(p)
    if device_index is None:
        print("[!] ReSpeaker 마이크를 찾을 수 없습니다. USB 연결을 확인하세요.")
        p.terminate()
        return
    stream = p.open(format=pyaudio.paInt16, channels=clf.CHANNELS, rate=clf.SR,
                    input=True, input_device_index=device_index,
                    frames_per_buffer=clf.CHUNK)

    tuning = Tuning(find_device())
    n_samples = int(seconds * clf.SR)
    print(f"[*] {sender.addr}로 전송 시작 (목표 주기 {interval:.2f}s). 종료: Ctrl+C\n")

    cycles, t_first = 0, time.perf_counter()
    try:
        while True:
            t_capture_start = time.time()
            frames, collected = [], 0
            while collected < n_samples:
                raw = stream.read(clf.CHUNK, exception_on_overflow=False)
                chunk = np.frombuffer(raw, dtype=np.int16).reshape(-1, clf.CHANNELS)
                frames.append(chunk[:, clf.MIC_CHANNEL_INDEX])
                collected += chunk.shape[0]
            # 스펙상 t는 구간의 '중심' 시각 — 수음 시작 + 구간길이/2
            t_center = t_capture_start + seconds / 2.0

            mono = np.concatenate(frames)[:n_samples].astype(np.float32) / 32768.0
            results = clf.classify(panns_model, svm, class_names, mono)
            top_class, top_score = results[0]
            conf = softmax_conf([s for _, s in results])

            # DoA는 분류가 끝난 직후 읽는다 — 소리가 아직 나고 있을 때의 방향에 가장 가깝다.
            theta = to_vehicle_frame(tuning.direction, mount_offset)

            if top_class in ALERT_CLASSES and top_score >= min_score:
                pkt = sender.send(top_class, conf, top_score, theta, sigma, t_center)
            else:
                # 배경음이거나 마진 미달 — 방향은 의미 없으므로 0으로 보낸다
                pkt = sender.send("none", conf, top_score, 0.0, sigma, t_center)

            cycles += 1
            if verbose or pkt["class"] != "none":
                print(f"  seq={pkt['seq']:5d} class={pkt['class']:11s} "
                      f"score={pkt['score']:+6.2f} theta={pkt['theta']:6.1f}")

            time.sleep(interval)
    except KeyboardInterrupt:
        elapsed = time.perf_counter() - t_first
        print("\n[*] 종료합니다.")
        if cycles:
            actual = elapsed / cycles
            print(f"[*] 실측 달성 주기: {actual:.2f}s/회 ({1/actual:.2f} Hz), "
                  f"목표 {interval:.2f}s/회")
            if actual > interval * 1.5:
                print("[!] 추론이 목표 주기를 못 따라갑니다 — 젯슨 GPU 인식/TensorRT 변환 필요.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True, help="노트북 IP (이더넷 직결 시 고정 IP)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seconds", type=float, default=1.0, help="분석 구간 길이(초)")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="전송 주기(초). 스펙은 0.25지만 CPU 추론 속도에 따라 조정")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="이 SVM 마진 미만이면 none으로 보냄 (기본 0.0 = 기존 동작 유지)")
    parser.add_argument("--mount-offset", type=float, default=MOUNT_OFFSET_DEG,
                        help="ReSpeaker raw 각도 중 차량 정면에 해당하는 각도(도)")
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA_DEG,
                        help="방향 오차 표준편차(도)")
    parser.add_argument("--simulate", action="store_true",
                        help="장비 없이 가짜 감지 전송 (노트북 수신측 검증용)")
    parser.add_argument("--svm-path", default=None,
                        help="SVM 체크포인트(best.pt) 경로. 생략하면 저장소 기본 위치")
    parser.add_argument("--panns-path", default=None,
                        help="PANNs Cnn14 사전학습 가중치(.pth) 경로. 없으면 자동 다운로드")
    parser.add_argument("--verbose", action="store_true", help="none 패킷도 전부 출력")
    args = parser.parse_args()

    sender = Sender(args.host, args.port)
    try:
        if args.simulate:
            run_simulate(sender, args.interval, args.verbose)
        else:
            run_live(sender, args.seconds, args.interval, args.min_score,
                     args.mount_offset, args.sigma, args.verbose,
                     args.svm_path, args.panns_path)
    finally:
        sender.close()


if __name__ == "__main__":
    main()
