"""
realtime_classify.py
ReSpeaker 4 Mic Array -> AST(Audio Spectrogram Transformer) 실시간 소리 분류 데모

실행:
    python3 realtime_classify.py
    python3 realtime_classify.py --seconds 3 --interval 1.5
    python3 realtime_classify.py --model-path /path/to/best.pt

사용 모델: 03_Audio_Classification/model_outputs/outputs_ast/best.pt (저장소에는 용량 문제로 미포함,
  outputs_ast.zip에서 복원해 이 경로에 두거나 --model-path로 위치를 지정할 것)
  - MIT/ast-finetuned-audioset-10-10-0.4593 기반, UrbanSound8K 10클래스로 파인튜닝
  - 검증 정확도 91.0%
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pyaudio
import torch
from transformers import ASTConfig, ASTForAudioClassification, ASTFeatureExtractor

warnings.filterwarnings("ignore", message="At least one mel filter")

DEFAULT_CKPT_PATH = Path(__file__).resolve().parent.parent / "model_outputs" / "outputs_ast" / "best.pt"
SR = 16000            # ReSpeaker 네이티브 샘플레이트 & 모델 학습 샘플레이트
CHANNELS = 6          # 실측: ReSpeaker 4 Mic Array가 6채널로 노출됨 (0=AEC 처리 채널)
MIC_CHANNEL_INDEX = 0 # 0번 = 보드에서 처리된(AEC) 단일 채널 -> 분류에 가장 적합
CHUNK = 1024


def get_respeaker_index(p: pyaudio.PyAudio):
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if "ReSpeaker" in info.get("name", "") and info.get("maxInputChannels", 0) > 0:
            return i
    return None


def load_model(ckpt_path: Path):
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"모델 체크포인트를 찾을 수 없습니다: {ckpt_path}\n"
            "outputs_ast.zip을 복원해 이 경로에 두거나 --model-path로 위치를 지정하세요."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    class_names = ckpt["class_names"]
    model_name = ckpt["model_name"]

    config = ASTConfig.from_pretrained(model_name)
    config.num_labels = len(class_names)
    config.id2label = {i: c for i, c in enumerate(class_names)}
    config.label2id = {c: i for i, c in enumerate(class_names)}

    model = ASTForAudioClassification(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
    return model, feature_extractor, class_names, ckpt


def classify(model, feature_extractor, class_names, audio_f32: np.ndarray):
    inputs = feature_extractor(audio_f32, sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    order = torch.argsort(probs, descending=True)
    return [(class_names[i], probs[i].item()) for i in order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=1.0,
                        help="분류 1회당 사용하는 오디오 길이 (초). 학습 시 사용한 길이(4.0s)와 맞추는 것을 권장")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="분류 반복 주기 (초)")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_CKPT_PATH,
                        help="AST 체크포인트(best.pt) 경로")
    args = parser.parse_args()

    print("[*] 모델 로딩 중...")
    model, feature_extractor, class_names, ckpt = load_model(args.model_path)
    print(f"[*] 모델 로드 완료: {ckpt['model_name']}  (검증 정확도 {ckpt['best_acc']*100:.1f}%)")
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
            results = classify(model, feature_extractor, class_names, mono)
            infer_sec = time.time() - t_infer_start

            top_label, top_conf = results[0]
            print(f"[{time.strftime('%H:%M:%S')}] 녹음 {capture_sec:.2f}s / 추론 {infer_sec:.2f}s")
            for label, conf in results[:args.topk]:
                bar = "#" * int(conf * 30)
                marker = ">>" if label == top_label else "  "
                print(f"  {marker} {label:20s} {conf*100:5.1f}% {bar}")
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
