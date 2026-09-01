"""
realtime_classify.py
ReSpeaker 4 Mic Array -> PANNs Cnn14(16kHz) 임베딩 -> SVM 실시간 소리 분류 데모

실행:
    python3 realtime_classify.py
    python3 realtime_classify.py --seconds 4 --interval 2.0
    python3 realtime_classify.py --svm-path /path/to/best.pt --panns-checkpoint /path/to/Cnn14_16k_mAP=0.438.pth

사용 모델 (2026-08-17부로 적용 — 팀원이 학습해 전달한 5클래스 모델. 한때 "AST"로 문서화된 적이 있었으나
실제 체크포인트를 열어 확인한 결과 AST가 아니라 아래 구성임, 00_Overview/현재_상태_요약.md 참고):
    03_Audio_Classification/model_outputs/panns_svm/best.pt (저장소에는 용량 문제로 미포함, model_outputs/panns_svm/README.md 참고)
    - 특징 추출: PANNs Cnn14 (16kHz 사전학습, Zenodo 배포본 Cnn14_16k_mAP=0.438.pth) -> 2048차원 임베딩
      * 로컬에 없으면 최초 실행 시 Zenodo에서 자동 다운로드 시도 (약 340MB, 인터넷 필요)
    - 분류: 임베딩 위에 학습한 sklearn SVM (StandardScaler + SVC, C=3.0)
    - 클래스(5개): car_horn / children_playing / engine_idling / siren / motorcycle
      (siren<->emergency_siren, motorcycle<->motorcycle_exhaust 대응, children_playing/engine_idling은 배경음 역할)
    - val 정확도 98.0%, test 정확도 89.5% (체크포인트에 기록된 값, model_outputs/panns_svm/README.md 참고)

⚠️ 알려진 제한사항 (README.md에 상세):
    - SVC가 probability=False로 학습되어 확률(%) 대신 decision_function 점수(마진)를 출력함
    - 체크포인트의 semantic_thresholds(클래스별 AudioSet 게이팅 임계값)는 정확한 적용 방식(어떤 AudioSet
      클래스 인덱스에 매핑되는지)을 재현할 학습 스크립트가 없어 이 스크립트에서는 적용하지 않음 — SVM
      원시 예측만 사용
"""

import argparse
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pyaudio
import torch

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = REPO_ROOT / "03_Audio_Classification" / "model_outputs" / "panns_svm"
DEFAULT_SVM_PATH = MODEL_DIR / "best.pt"
DEFAULT_PANNS_CKPT_PATH = MODEL_DIR / "Cnn14_16k_mAP=0.438.pth"
PANNS_CKPT_URL = "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1"
PANNS_CKPT_MIN_SIZE = 3.5e8  # 정상 파일은 약 358MB, 다운로드 중단 시 훨씬 작음

SR = 16000            # ReSpeaker 네이티브 샘플레이트 & PANNs Cnn14 16kHz 변형 입력 샘플레이트
CHANNELS = 6          # 실측: ReSpeaker 4 Mic Array가 6채널로 노출됨 (0=AEC 처리 채널)
MIC_CHANNEL_INDEX = 0 # 0번 = 보드에서 처리된(AEC) 단일 채널 -> 분류에 가장 적합
CHUNK = 1024


def get_respeaker_index(p: pyaudio.PyAudio):
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if "ReSpeaker" in info.get("name", "") and info.get("maxInputChannels", 0) > 0:
            return i
    return None


def ensure_panns_checkpoint(ckpt_path: Path):
    if ckpt_path.exists() and ckpt_path.stat().st_size >= PANNS_CKPT_MIN_SIZE:
        return
    print(f"[*] PANNs Cnn14(16kHz) 사전학습 가중치가 없어 다운로드합니다 (~340MB): {PANNS_CKPT_URL}")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(PANNS_CKPT_URL, ckpt_path)


def resolve_device(device: str = "auto") -> str:
    """"auto"면 CUDA가 쓸 수 있을 때 cuda, 아니면 cpu."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_panns_model(ckpt_path: Path, device: str = "auto"):
    """PANNs Cnn14를 로드하고 **지정 장치로 옮긴다**.

    ⚠️ 2026-09-02 발견: 이 함수가 모델을 GPU로 옮기지 않아, CUDA가 멀쩡히 있는 젯슨에서도
       추론이 CPU에서 돌고 있었다. `torch.cuda.is_available()`이 True인 것과 모델이 GPU에
       올라가는 것은 별개다 — `map_location="cpu"`로 읽은 뒤 `.to()`를 하지 않으면 CPU에
       남는다. 실측: CPU 약 200~278ms vs CUDA 78ms(3.6배).
    """
    from panns_inference.models import Cnn14

    model = Cnn14(sample_rate=16000, window_size=512, hop_size=160,
                  mel_bins=64, fmin=50, fmax=8000, classes_num=527)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dev = resolve_device(device)
    model.to(dev)
    if dev == "cuda":
        # 첫 CUDA 호출은 커널 컴파일·메모리 할당 때문에 유독 느리다. 여기서 미리 태워
        # 없애지 않으면 "첫 감지만 유독 늦는" 현상이 된다(YOLO 워밍업과 같은 이유).
        with torch.no_grad():
            model(torch.zeros(1, SR, device=dev))
        torch.cuda.synchronize()
    print(f"[*] PANNs 장치: {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
    return model


def load_svm(svm_path: Path):
    if not svm_path.exists():
        raise FileNotFoundError(
            f"SVM 체크포인트를 찾을 수 없습니다: {svm_path}\n"
            "03_Audio_Classification/model_outputs/panns_svm/README.md 참고해 복원하거나 --svm-path로 위치를 지정하세요."
        )
    ckpt = torch.load(svm_path, map_location="cpu", weights_only=False)
    return ckpt["classifier"], ckpt["classes"], ckpt


def classify(panns_model, clf, class_names, audio_f32: np.ndarray):
    # 입력 텐서를 모델과 같은 장치로 보낸다. 모델만 GPU에 있고 입력이 CPU면 런타임 에러다.
    device = next(panns_model.parameters()).device
    audio_t = torch.from_numpy(audio_f32).unsqueeze(0).to(device)
    with torch.no_grad():
        # SVM(sklearn)은 CPU numpy만 받으므로 임베딩은 다시 내려받는다
        embedding = panns_model(audio_t)["embedding"].cpu().numpy()
    scores = clf.decision_function(embedding)[0]
    order = np.argsort(scores)[::-1]
    return [(class_names[i], float(scores[i])) for i in order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=1.0,
                        help="분류 1회당 사용하는 오디오 길이 (초)")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="분류 반복 주기 (초)")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--svm-path", type=Path, default=DEFAULT_SVM_PATH,
                        help="PANNs 임베딩 위 SVM 체크포인트(best.pt) 경로")
    parser.add_argument("--panns-checkpoint", type=Path, default=DEFAULT_PANNS_CKPT_PATH,
                        help="PANNs Cnn14(16kHz) 사전학습 가중치 경로 (없으면 자동 다운로드 시도)")
    args = parser.parse_args()

    print("[*] PANNs Cnn14 가중치 확인 중...")
    ensure_panns_checkpoint(args.panns_checkpoint)

    print("[*] 모델 로딩 중...")
    panns_model = load_panns_model(args.panns_checkpoint)
    clf, class_names, ckpt = load_svm(args.svm_path)
    print(f"[*] 모델 로드 완료 (SVM val_accuracy={ckpt.get('val_accuracy', float('nan'))*100:.1f}%, "
          f"test_accuracy={ckpt.get('test_accuracy', float('nan'))*100:.1f}%)")
    print(f"[*] 클래스: {class_names}")

    p = pyaudio.PyAudio()
    device_index = get_respeaker_index(p)
    if device_index is None:
        print("[!] ReSpeaker 마이크를 찾을 수 없습니다. USB 연결을 확인하세요.")
        p.terminate()
        return

    dev_info = p.get_device_info_by_index(device_index)
    print(f"[*] ReSpeaker 인식됨: index={device_index}, name={dev_info['name']}")

    stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=SR,
                     input=True, input_device_index=device_index,
                     frames_per_buffer=CHUNK)

    n_samples = int(args.seconds * SR)
    print(f"[*] 실시간 분류 시작 ({args.seconds}s 윈도우 / {args.interval}s 주기). 종료: Ctrl+C\n")

    try:
        while True:
            frames = []
            collected = 0
            t_capture_start = time.time()
            while collected < n_samples:
                raw = stream.read(CHUNK, exception_on_overflow=False)
                chunk = np.frombuffer(raw, dtype=np.int16).reshape(-1, CHANNELS)
                frames.append(chunk[:, MIC_CHANNEL_INDEX])
                collected += chunk.shape[0]
            capture_sec = time.time() - t_capture_start

            mono = np.concatenate(frames)[:n_samples].astype(np.float32) / 32768.0

            t_infer_start = time.time()
            results = classify(panns_model, clf, class_names, mono)
            infer_sec = time.time() - t_infer_start

            top_label, top_score = results[0]
            print(f"[{time.strftime('%H:%M:%S')}] 녹음 {capture_sec:.2f}s / 추론 {infer_sec:.2f}s")
            for label, score in results[:args.topk]:
                marker = ">>" if label == top_label else "  "
                print(f"  {marker} {label:20s} score={score:+.3f}")
            print()

            elapsed = time.time() - t_capture_start
            sleep_time = args.interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[*] 종료합니다.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    main()
