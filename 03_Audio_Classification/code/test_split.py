import pyaudio
import wave
import numpy as np

# ============== [System Configuration] ==============
RATE = 16000            # 샘플링 레이트 (8kHz)
CHANNELS = 4            # 입력 마이크 채널 수
CHUNK = 1024            # 버퍼 크기
RECORD_SECONDS = 5      # 녹음 지속 시간
OUTPUT_FILENAME = "ch1_isolated.wav"
# ====================================================

def get_respeaker_index(p):
    """시스템 장치 목록을 스캔하여 ReSpeaker 마이크의 Index를 자동으로 찾습니다."""
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        name = dev_info.get('name', '')
        channels = dev_info.get('maxInputChannels', 0)
       
        # 이름에 'ReSpeaker'가 포함되어 있고 입력 채널이 있는 장치 탐색
        if 'ReSpeaker' in name and channels > 0:
            return i
    return None

def main():
    p = pyaudio.PyAudio()

    # 1. 마이크 인덱스 자동 탐색
    target_index = get_respeaker_index(p)
   
    if target_index is None:
        print("[!] ReSpeaker 마이크를 찾을 수 없습니다. USB 연결을 확인하세요.")
        p.terminate()
        return

    print(f"[*] ReSpeaker 마이크 자동 인식 성공 (Index: {target_index})")
    print(f"[*] 오디오 캡처를 시작합니다. ({RECORD_SECONDS}초간 말씀해 주세요...)")

    try:
        # 2. 오디오 스트림 열기
        stream = p.open(format=pyaudio.paInt16,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=target_index,
                        frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"[!] 오디오 스트림을 열 수 없습니다: {e}")
        p.terminate()
        return

    frames = []

    # 3. 5초간 오디오 데이터 수집
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    print("[*] 오디오 캡처 완료. 채널 분리 작업을 시작합니다.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # 4. 바이트 데이터 병합 및 Numpy 배열 변환
    raw_bytes = b''.join(frames)
    audio_np = np.frombuffer(raw_bytes, dtype=np.int16)

    # 5. 4채널 디인터리빙 (De-interleaving)
    channels_data = audio_np.reshape(-1, CHANNELS).T

    # 6. 첫 번째 마이크(CH0) 데이터 추출
    ch1_data = channels_data[0]

    # 7. 분리된 데이터를 WAV 파일로 저장
    with wave.open(OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(RATE)
        wf.writeframes(ch1_data.tobytes())

    print(f"[*] 1번 채널 오디오가 성공적으로 저장되었습니다: {OUTPUT_FILENAME}")

if __name__ == "__main__":
    main()