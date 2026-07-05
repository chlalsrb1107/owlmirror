import pyaudio

def main():
    p = pyaudio.PyAudio()
    print("[*] 시스템의 오디오 입력 장치 스캔을 시작합니다...")

    # 시스템에 연결된 모든 오디오 장치를 순회
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
       
        # 입력(녹음) 채널이 1개 이상인 장치만 필터링하여 출력
        if dev_info.get('maxInputChannels') > 0:
            print(f"Index {i}: {dev_info.get('name')} (Channels: {dev_info.get('maxInputChannels')})")

    p.terminate()
    print("[*] 스캔이 완료되었습니다. ReSpeaker 또는 Seeed 장치의 Index 번호를 확인하세요.")

if __name__ == "__main__":
    main()