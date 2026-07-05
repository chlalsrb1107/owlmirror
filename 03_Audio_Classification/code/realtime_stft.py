import pyaudio
import numpy as np
import torch
import torch.nn.functional as F

# ============== [System Configuration] ==============
RATE_HW = 16000         # 하드웨어 고정 샘플링 레이트
RATE_MODEL = 8000       # AI 모델 기준 샘플링 레이트
CHANNELS = 4            # 마이크 채널 수
CHUNK = 2048            # 버퍼 크기 (0.128초 분량)
TARGET_SIZE = 224       # 모델 입력 이미지(텐서) 크기 (224x224)
# ====================================================

def get_respeaker_index(p):
    """ReSpeaker 마이크 Index 자동 탐색"""
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if 'ReSpeaker' in dev_info.get('name', '') and dev_info.get('maxInputChannels', 0) > 0:
            return i
    return None

def main():
    p = pyaudio.PyAudio()
    target_index = get_respeaker_index(p)
   
    if target_index is None:
        print("[!] ReSpeaker 마이크를 찾을 수 없습니다.")
        p.terminate()
        return

    stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE_HW,
                    input=True, input_device_index=target_index, frames_per_buffer=CHUNK)

    print("[*] 실시간 스펙트로그램 텐서 변환 가동 시작 (종료: Ctrl+C)")

    # STFT용 Window 객체 미리 생성 (연산 속도 최적화)
    # n_fft=1024 기준
    window = torch.hann_window(1024)

    try:
        while True:
            # 1. 스트림에서 바이너리 데이터 읽기
            raw_data = stream.read(CHUNK, exception_on_overflow=False)
            audio_np = np.frombuffer(raw_data, dtype=np.int16)

            # 2. 4채널 디인터리빙 및 CH0 분리
            channels_data = audio_np.reshape(-1, CHANNELS).T
            ch0_data = channels_data[0]

            # 3. 실시간 다운샘플링 (16000Hz -> 8000Hz)
            # 넘파이 배열 슬라이싱([::2])을 통해 절반의 데이터만 추출하여 8kHz로 변환
            ch0_8k = ch0_data[::2].astype(np.float32)

            # 4. PyTorch 텐서 변환 및 STFT 수행
            tensor_audio = torch.from_numpy(ch0_8k)
            stft_result = torch.stft(tensor_audio, n_fft=1024, hop_length=512,
                                     window=window, return_complex=True)

            # 5. 진폭(Magnitude)을 데시벨(DB) 스케일로 변환
            magnitude = torch.abs(stft_result)
            db_spectrogram = 20 * torch.log10(magnitude + 1e-5)

            # 6. CNN 입력을 위한 224x224 리사이즈
            # [Freq, Time] 2D 구조를 [1(Batch), 1(Channel), Freq, Time] 4D 구조로 확장
            db_tensor = db_spectrogram.unsqueeze(0).unsqueeze(0)
            final_input = F.interpolate(db_tensor, size=(TARGET_SIZE, TARGET_SIZE),
                                        mode='bilinear', align_corners=False)

            print(f"[+] 실시간 텐서 생성 완료 | 최종 형태: {final_input.shape}")
           
            # TODO: 다음 단계에서 여기에 CNN 추론 코드가 들어갑니다.
            # output = model(final_input.to('cuda'))

    except KeyboardInterrupt:
        print("\n[*] 시스템을 안전하게 종료합니다.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()