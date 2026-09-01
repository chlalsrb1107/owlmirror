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
| `rms_db`| 분석 구간 ch0의 RMS 음량(dBFS, 무음 -90 ~ 최대 0) — 아래 참고  |

⚠️ conf에 대하여: SVM이 `probability=False`로 학습돼 진짜 확률을 낼 수 없다
   (03_Audio_Classification/model_outputs/panns_svm/README.md 참고). conf는 마진에
   softmax를 씌운 **유사** 신뢰도이므로 표시용으로만 쓰고, 임계값 판단은 score로 할 것.

⚠️ rms_db에 대하여: 노트북의 알림 규칙 두 가지가 이 값을 쓴다 — (1) 경적이 여러 방향에서
   동시에 잡히면 **가장 큰 것**을 배너로 올리고, (2) 오토바이가 라이다·카메라 어느 쪽으로도
   확정되지 않았는데 배기음이 크고 또렷하면 "사각지대 근접"으로 보고 경고까지 올린다
   (alert_policy.py). ⚠️ 절대 음압(dB SPL)이 아니라 마이크 입력 기준 상대값이라, 차량 장착
   후 실제 주행 소음에서 임계값(alert_policy.MOTO_LOUD_DB 등)을 반드시 다시 맞춰야 한다.

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

# 방향 오차 표준편차(도) — BEV 부채꼴 폭(2×sigma)에 쓰인다.
# 2026-09-02 실측: 정지·실내에서 사이렌을 한 자리에 두면 DoA가 σ≈5~7°로 뭉친다
# (theta 274~289 구간). 문서 추정치 ±10~20°보다 좋다.
# ⚠️ 다만 이건 **정밀도(흩어짐)**이지 정확도가 아니다 — 274°가 실제 방향과 맞는지는
#    각도를 아는 위치에서 재봐야 하고(MOUNT_OFFSET_DEG 실측과 같은 작업), 주행 중에는
#    풍절음·반사로 더 나빠진다. 그래서 실측치보다 보수적으로 잡아 둔다.
DEFAULT_SIGMA_DEG = 10.0

# ---- DoA 신뢰도 (2026-09-02 추가) ---------------------------------------------
# ReSpeaker 펌웨어 DoA는 강한 방향성 소리가 없으면 직전 값이나 잡음을 그대로 뱉는다.
# 실측에서 사이렌이 끊긴 구간에 287·284·289로 안정적이던 값이 222·79로 튀었다.
# 한 번만 읽으면 그 튄 값이 그대로 화면 방향이 되므로, **수음 구간 안에서 여러 번 읽어**
# 원형 평균을 쓰고 흩어진 정도로 신뢰 여부를 판정한다.
DOA_SAMPLES_PER_WINDOW = 8   # 수음 1구간당 DoA 읽기 횟수 (USB 제어전송 1회당 ~1ms)
DOA_MAX_SPREAD_DEG = 35.0    # 원형 표준편차가 이보다 크면 방향을 믿지 않는다
DOA_MIN_DB = -42.0           # 이보다 조용하면 방향성 자체가 없다고 본다

# ---- 오경보 차단 (2026-09-02 추가) --------------------------------------------
# SVM 마진(score)이나 1위-2위 격차(margin)로는 오경보를 거를 수 없다 — 다중클래스 OvR
# 값이 클래스와 무관하게 늘 4.2 근처라 변별력이 없기 때문(실측 확인).
# 대신 **음량**은 물리적으로 확실한 근거다: 실제로 들리지 않는 사이렌은 없다.
# 실측 근거 — 진짜 감지: siren -17~-33dB, motorcycle -26~-35dB.
#             오경보: 아무 소리 없는 -44dB 구간에서 경적·사이렌이 각 1회 잡힘.
# 그 사이인 -40dB에 선을 그으면 오경보는 막고 진짜 감지는 남는다.
# ⚠️ 주행 중에는 배경 소음이 통째로 올라가므로 이 값도 실차에서 다시 맞춰야 한다.
MIN_ALERT_DB = -40.0


def circular_stats(angles_deg):
    """각도 목록의 (원형 평균, 원형 표준편차). 0/360 경계를 올바르게 처리한다.

    산술 평균을 쓰면 350°와 10°의 평균이 180°(정반대!)가 된다. 단위벡터로 더한 뒤
    각도를 되찾아야 한다. 표준편차는 벡터 합의 길이 R로 구한다 — R이 1에 가까우면
    읽은 값들이 한 방향에 모였다는 뜻이고, 0에 가까우면 사방으로 흩어졌다는 뜻이다.
    """
    if not angles_deg:
        return 0.0, 180.0
    xs = sum(math.cos(math.radians(a)) for a in angles_deg)
    ys = sum(math.sin(math.radians(a)) for a in angles_deg)
    n = len(angles_deg)
    mean = math.degrees(math.atan2(ys, xs)) % 360.0
    r = math.hypot(xs, ys) / n
    if r <= 1e-9:
        return mean, 180.0
    sd = math.degrees(math.sqrt(max(0.0, -2.0 * math.log(min(1.0, r)))))
    return mean, sd


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
             theta: float, sigma: float, t_center: float, rms_db: float = -60.0,
             margin: float = 0.0, theta_ok: bool = True, raw_class: str = ""):
        packet = {
            "seq": self.seq,
            "t": round(t_center, 3),
            "class": class_name,
            "conf": round(conf, 3),
            "score": round(score, 3),
            "theta": round(theta, 1),
            "sigma": round(sigma, 1),
            "rms_db": round(rms_db, 1),
            "margin": round(margin, 3),
            "theta_ok": bool(theta_ok),
            # 진단용: class가 "none"일 때 실제로 1위였던 클래스. 오토바이 배기음이
            # engine_idling으로 빨려가는지 같은 혼동을 보려면 이 값이 있어야 한다.
            "raw": raw_class,
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
                pkt = sender.send("none", 0.0, -1.0, 0.0, DEFAULT_SIGMA_DEG,
                                  time.time(), -72.0, 0.0, False, "engine_idling")
            else:
                theta = (theta + random.uniform(-8, 8)) % 360  # 대상이 조금씩 움직이는 효과
                score = random.uniform(0.4, 2.2)
                # 음량도 흩뿌려야 노트북의 "가장 큰 경적", "배기음 크면 경고" 규칙을 실제로 타본다
                rms_db = random.uniform(-42.0, -14.0)
                pkt = sender.send(burst_class, softmax_conf([score, 0.0, -0.5, -0.8, -1.0]),
                                  score, theta, DEFAULT_SIGMA_DEG, time.time(),
                                  rms_db, 1.0, True, burst_class)
                burst_left -= 1

            if verbose or pkt["class"] != "none":
                print(f"  seq={pkt['seq']:5d} class={pkt['class']:11s} "
                      f"score={pkt['score']:+6.2f} theta={pkt['theta']:6.1f} "
                      f"rms={pkt['rms_db']:6.1f}dB sigma={pkt['sigma']:4.1f}"
                      f" raw={pkt['raw']}"
                      f"{'' if pkt['theta_ok'] else '  [방향 불신]'}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[simulate] 종료합니다.")


def run_live(sender: Sender, seconds: float, interval: float, min_score: float,
             mount_offset: float, sigma: float, verbose: bool,
             svm_path=None, panns_path=None, min_db: float = MIN_ALERT_DB):
    """실제 ReSpeaker에서 수음 → 분류 → DoA → 전송.

    svm_path/panns_path를 주면 저장소 구조 밖에 있는 가중치도 쓸 수 있다 — 젯슨에는 보통
    저장소 전체가 아니라 파일 몇 개만 올라가므로, 경로를 고정하면 찾지 못한다.
    """
    import numpy as np

    # ⚠️ 의존성을 **모델 로딩 전에** 전부 확인한다. 예전에는 pyusb가 없어도 PANNs 340MB를
    #    다 읽고 오디오 스트림까지 연 뒤에야 ImportError로 죽었다 — 30초를 버리고 나서야
    #    "pip install pyusb 하세요"를 알게 되는 셈이라, 촬영 현장에서 치명적이다.
    missing = []
    for mod, pkg in (("pyaudio", "pyaudio"), ("usb.core", "pyusb"), ("torch", "torch")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[!] 젯슨에 다음 패키지가 없습니다: {', '.join(missing)}")
        print(f"    pip3 install {' '.join(missing)}")
        print("    (pyusb는 ReSpeaker 펌웨어 DoA를 읽는 데 필요합니다. 설치 후에도 "
              "Access denied가 나면 udev 규칙을 추가해야 합니다 — README 참고)")
        return

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
        # ⚠️ "못 찾음"의 원인이 둘인데 증상이 같다 — (1) USB에 아예 없거나,
        #    (2) 앞선 실행이 안 죽고 장치를 잡고 있거나. (2)일 때 ALSA는 장치를 계속
        #    보여주지만 `Subdevices: 0/1`이 되고, PortAudio는 프로브에 실패해 목록에서
        #    빼버린다. 2026-09-02에 실제로 (2)로 한참 헤맸다 — 그래서 구분해 알려준다.
        # 이름은 잡히는데 입력 채널이 0이면 = 다른 프로세스가 점유 중.
        # ALSA는 계속 장치를 보여주지만 PortAudio가 프로브에 실패해 채널 0으로 보고한다.
        busy = [p.get_device_info_by_index(i)
                for i in range(p.get_device_count())
                if "ReSpeaker" in p.get_device_info_by_index(i).get("name", "")]
        if busy:
            print("[!] ReSpeaker가 장치 목록에는 있으나 입력 채널이 0으로 보고됩니다 "
                  "— 다른 프로세스가 이미 마이크를 잡고 있습니다.")
            print(f"    ({busy[0].get('name')} · maxInputChannels="
                  f"{busy[0].get('maxInputChannels')})")
            print("    해결:  pkill -f jetson_audio_sender.py   후 다시 실행하세요.")
            print("    ⚠️ SSH 창을 그냥 닫으면 젯슨 쪽 파이썬이 살아남습니다. "
                  "끝낼 때는 Ctrl+C를 쓰세요.")
        else:
            print("[!] ReSpeaker가 입력 장치 목록에 없습니다 — USB에 올라오지 않았습니다.")
            print("    확인:  lsusb | grep 2886   /   케이블·포트를 바꿔 다시 연결")
            seen = [p.get_device_info_by_index(i).get("name", "")
                    for i in range(p.get_device_count())]
            print(f"    PortAudio가 본 장치: {seen if seen else '(없음)'}")
        p.terminate()
        return
    stream = p.open(format=pyaudio.paInt16, channels=clf.CHANNELS, rate=clf.SR,
                    input=True, input_device_index=device_index,
                    frames_per_buffer=clf.CHUNK)

    try:
        tuning = Tuning(find_device())
        # ⚠️ pyusb는 생성자에서 장치를 열지 않는다 — 첫 ctrl_transfer 때 연다. 그래서 권한
        #    문제는 Tuning() 이 아니라 한참 뒤 루프 안 tuning.direction에서 터진다.
        #    여기서 한 번 읽어 그 실패를 앞으로 당겨 온다(값도 첫 방위각으로 바로 확인 가능).
        probe = tuning.direction
        print(f"[*] ReSpeaker 펌웨어 DoA 정상 — 현재 방위각 {probe}deg")
    except Exception as e:  # noqa: BLE001 — USB 권한/장치 문제를 원인과 함께 알려준다
        print(f"[!] ReSpeaker DoA 인터페이스를 열 수 없습니다: {e}")
        print("    USB 권한 문제라면 젯슨에서 아래를 실행하고 ReSpeaker를 다시 꽂으세요:")
        print("      echo 'SUBSYSTEM==\"usb\", ATTRS{idVendor}==\"2886\", MODE=\"0666\"' \\")
        print("        | sudo tee /etc/udev/rules.d/99-respeaker.rules")
        print("      sudo udevadm control --reload-rules && sudo udevadm trigger")
        stream.stop_stream(); stream.close(); p.terminate()
        return
    n_samples = int(seconds * clf.SR)
    print(f"[*] {sender.addr}로 전송 시작 (목표 주기 {interval:.2f}s). 종료: Ctrl+C\n")

    cycles, t_first = 0, time.perf_counter()
    try:
        while True:
            t_capture_start = time.time()
            frames, collected, doa_samples = [], 0, []
            chunks_per_doa = max(1, int(n_samples / clf.CHUNK / DOA_SAMPLES_PER_WINDOW))
            n_chunk = 0
            while collected < n_samples:
                raw = stream.read(clf.CHUNK, exception_on_overflow=False)
                chunk = np.frombuffer(raw, dtype=np.int16).reshape(-1, clf.CHANNELS)
                frames.append(chunk[:, clf.MIC_CHANNEL_INDEX])
                collected += chunk.shape[0]
                # 수음하는 동안 DoA를 같이 읽는다 — 분류하는 그 소리의 방향이 된다.
                if n_chunk % chunks_per_doa == 0:
                    try:
                        doa_samples.append(float(tuning.direction))
                    except Exception:  # noqa: BLE001 — 한 번 실패해도 나머지로 계속한다
                        pass
                n_chunk += 1
            # 스펙상 t는 구간의 '중심' 시각 — 수음 시작 + 구간길이/2
            t_center = t_capture_start + seconds / 2.0

            # 수음 구간 안에서 모은 DoA로 방향과 그 불확실성을 함께 구한다.
            raw_theta, spread = circular_stats(doa_samples)
            theta = to_vehicle_frame(raw_theta, mount_offset)
            # sigma를 고정값이 아니라 **실측 흩어짐**으로 보낸다 — 노트북 BEV 부채꼴이
            # 불확실할 때 저절로 넓어지고 확실할 때 좁아진다. 정직한 표현이 된다.
            sigma_out = max(sigma, min(spread, 60.0))

            mono = np.concatenate(frames)[:n_samples].astype(np.float32) / 32768.0
            # dBFS. 완전 무음이면 log(0)이 되므로 바닥을 -90dB로 깐다.
            rms_db = 20.0 * math.log10(max(float(np.sqrt(np.mean(mono ** 2))), 10 ** (-90 / 20)))
            results = clf.classify(panns_model, svm, class_names, mono)
            top_class, top_score = results[0]
            conf = softmax_conf([s for _, s in results])
            # ⚠️ top_score는 신뢰도가 아니다. sklearn SVC의 다중클래스 OvR 값은
            #    "이긴 1:1 대결 수(0~4) + 소수점 보정(<1/3)" 구조라, 확신 있는 예측이면
            #    클래스와 무관하게 늘 4.2 근처가 나온다(2026-09-02 실측: siren +4.27,
            #    none +4.23, motorcycle +4.23 — 범위가 완전히 겹침).
            #    변별력은 1위와 2위의 **격차**에만 남아 있으므로 그것을 따로 보낸다.
            margin = top_score - results[1][1] if len(results) > 1 else 0.0

            # 방향을 믿을 수 있는가 — 읽은 값들이 모였고(spread), 소리가 있었나(rms).
            # 둘 중 하나라도 못 미치면 theta_ok=False로 보내고, 노트북은 그 방향으로
            # 카메라를 돌리지 않는다. "잘못된 지목은 알려주지 않는 것보다 위험하다".
            theta_ok = spread <= DOA_MAX_SPREAD_DEG and rms_db >= DOA_MIN_DB

            # 너무 조용하면 무엇이 잡혔든 내보내지 않는다. 들리지 않는 소리를 알릴 수는 없다.
            loud_enough = rms_db >= min_db

            if top_class in ALERT_CLASSES and top_score >= min_score and loud_enough:
                pkt = sender.send(top_class, conf, top_score, theta, sigma_out, t_center,
                                  rms_db, margin, theta_ok, top_class)
            else:
                # 배경음이거나 마진 미달 — 방향은 의미 없으므로 0으로 보낸다
                pkt = sender.send("none", conf, top_score, 0.0, sigma_out, t_center,
                                  rms_db, margin, False, top_class)

            cycles += 1
            if verbose or pkt["class"] != "none":
                print(f"  seq={pkt['seq']:5d} class={pkt['class']:11s} "
                      f"score={pkt['score']:+6.2f} theta={pkt['theta']:6.1f} "
                      f"rms={pkt['rms_db']:6.1f}dB sigma={pkt['sigma']:4.1f}"
                      f" raw={pkt['raw']}"
                      f"{'' if pkt['theta_ok'] else '  [방향 불신]'}")

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
    # ⚠️ 2026-09-02 실측으로 확정한 값. 0.5초로 줄이면 주기는 0.55초로 빨라지지만 사이렌
    #    검출률이 83% -> 35%로 무너진다. 사이렌은 주파수가 오르내리는 소리라 한 주기가
    #    1~2초인데 0.5초 창에는 특징이 반밖에 안 담긴다. 초당 감지 횟수는 0.68 vs 0.64로
    #    사실상 같아서, 검출률이 높은 1.0초가 명백히 낫다(배너가 깜빡이지 않는다).
    parser.add_argument("--seconds", type=float, default=1.0, help="분석 구간 길이(초)")
    # 기본값이 2.0이던 시절 실측 주기가 3.19초였다(캡처 1.0 + 추론 0.19 + sleep 2.0).
    # 추론이 느린 게 아니라 이 sleep이 원인이었다 — 0으로 두면 1.22초까지 당겨진다.
    parser.add_argument("--interval", type=float, default=0.0,
                        help="전송 주기(초). 스펙은 0.25지만 CPU 추론 속도에 따라 조정")
    parser.add_argument("--min-db", type=float, default=MIN_ALERT_DB,
                        help="이 음량(dBFS) 아래로는 감지를 내보내지 않는다 — 조용할 때 나오는 "
                             "오경보 차단용. 주행 중에는 배경 소음이 올라가므로 재조정 필요")
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
                     args.svm_path, args.panns_path, args.min_db)
    finally:
        sender.close()


if __name__ == "__main__":
    main()
