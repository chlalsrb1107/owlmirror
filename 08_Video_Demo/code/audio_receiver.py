"""
audio_receiver.py — [노트북에서 실행] 젯슨이 UDP로 보내는 오디오 감지 JSON 수신.

00_Overview/현재_상태_요약.md "Jetson→노트북 인터페이스"의 노트북 쪽 수신부.
jetson_audio_sender.py가 보내는 패킷을 배경 스레드로 계속 받아 **최신 것 하나만** 들고 있는다.
UDP라 유실·순서역전이 가능하므로 seq로 유실을 세고, 오래된 패킷은 버린다.

설계 원칙 — 링크가 죽었을 때 화면이 거짓말하지 않게 하는 것이 이 모듈의 핵심 책임이다.
젯슨이 죽거나 케이블이 빠지면 마지막 감지가 화면에 영원히 남아선 안 된다. `stale_after`초 동안
패킷이 없으면 link()가 connected=False를 돌려주고, 호출측(live_demo.py)이 이를 화면에 표시한다.

⚠️ 시계 동기화: `t`는 젯슨 시계 기준이라 두 기기 시계가 안 맞으면 지연 계산이 틀어진다
   (현재_상태_요약.md "점별 시각 매칭" 참고). 이 모듈은 스큐가 크면 경고를 한 번 출력하지만
   보정하지는 않는다 — 양쪽에 chrony/NTP를 걸어 해결할 것.

⚠️ theta는 젯슨이 이미 마운트 오프셋을 적용해 **차량 좌표계**로 보낸다.
   따라서 select_camera()는 mount_offset_deg=0.0으로 호출해야 한다 (이중 적용 방지).

단독 실행하면 수신되는 패킷을 그대로 찍어보는 모니터로 동작한다:
    python3 audio_receiver.py            # 기본 포트에서 수신 대기
    python3 audio_receiver.py --verbose  # none 패킷까지 전부 출력
"""

import argparse
import json
import socket
import threading
import time

DEFAULT_PORT = 9870
DEFAULT_STALE_AFTER = 5.0   # 이 시간 동안 패킷이 없으면 링크 끊긴 것으로 간주(초)
MAX_PACKET_AGE = 2.0        # 이보다 오래된 패킷은 버림(초) — 밀린 패킷이 화면을 되감지 않도록
CLOCK_SKEW_WARN = 1.0       # 이 이상 시계가 어긋나면 경고(초)
RECV_BUFSIZE = 4096

REQUIRED_FIELDS = ("t", "class", "theta")


class AudioDetectionReceiver:
    """젯슨 UDP 패킷을 배경 스레드로 수신해 최신 감지 하나를 보관한다."""

    def __init__(self, port: int = DEFAULT_PORT, bind_host: str = "0.0.0.0",
                 stale_after: float = DEFAULT_STALE_AFTER):
        self.port = port
        self.bind_host = bind_host
        self.stale_after = stale_after

        self._lock = threading.Lock()
        self._latest = None          # 아직 소비되지 않은 최신 패킷
        self._last_recv = 0.0        # 마지막 수신 시각(노트북 시계)
        self._received = 0
        self._dropped = 0            # seq 공백으로 추정한 유실 수
        self._malformed = 0
        self._next_seq = None
        self._skew = 0.0
        self._skew_warned = False

        self._sock = None
        self._thread = None
        self._stop = threading.Event()

    # ---------- 수명주기 ----------

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.bind_host, self.port))
        self._sock.settimeout(0.5)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._sock is not None:
            self._sock.close()

    # ---------- 수신 루프 ----------

    def _loop(self):
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(RECV_BUFSIZE)
            except socket.timeout:
                continue
            except OSError:
                break  # stop()으로 소켓이 닫힘

            packet = self._parse(data)
            if packet is None:
                continue

            now = time.time()
            with self._lock:
                self._received += 1
                self._last_recv = now
                self._track_seq(packet)
                self._track_skew(packet, now)

                # 밀려서 도착한 오래된 패킷은 버린다 (시계가 맞을 때만 의미 있는 판단)
                age = now - packet["t"]
                if abs(self._skew) < CLOCK_SKEW_WARN and age > MAX_PACKET_AGE:
                    continue

                self._latest = packet

    def _parse(self, data: bytes):
        try:
            packet = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            with self._lock:
                self._malformed += 1
            return None
        if not isinstance(packet, dict) or any(f not in packet for f in REQUIRED_FIELDS):
            with self._lock:
                self._malformed += 1
            return None
        return packet

    def _track_seq(self, packet):
        """호출측에서 이미 _lock을 잡고 있다고 가정."""
        seq = packet.get("seq")
        if seq is None:
            return
        if self._next_seq is not None and seq > self._next_seq:
            self._dropped += seq - self._next_seq
        if self._next_seq is None or seq >= self._next_seq:
            self._next_seq = seq + 1

    def _track_skew(self, packet, now):
        """호출측에서 이미 _lock을 잡고 있다고 가정."""
        self._skew = now - packet["t"]
        if not self._skew_warned and abs(self._skew) > CLOCK_SKEW_WARN:
            self._skew_warned = True
            print(f"[!] 젯슨과 시계가 {self._skew:+.2f}초 어긋나 있습니다 — "
                  f"양쪽에 chrony/NTP를 설정하세요. (지연 계산이 부정확해집니다)")

    # ---------- 조회 ----------

    def get_new(self):
        """아직 안 읽은 최신 감지 패킷을 반환하고 비운다. 없으면 None.

        일부러 큐가 아니라 '최신 하나'만 준다 — 화면은 항상 가장 최근 상황만 보여주면 되고,
        밀린 패킷을 순서대로 재생하면 오히려 과거 상태가 뒤늦게 뜬다.
        """
        with self._lock:
            packet, self._latest = self._latest, None
            return packet

    def link(self):
        """링크 상태. connected=False면 화면에 '젯슨 연결 끊김'을 띄울 것."""
        with self._lock:
            age = time.time() - self._last_recv if self._last_recv else float("inf")
            return {
                "connected": age <= self.stale_after,
                "age": age,
                "received": self._received,
                "dropped": self._dropped,
                "malformed": self._malformed,
                "skew": self._skew,
            }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--verbose", action="store_true", help="none 패킷도 출력")
    args = parser.parse_args()

    receiver = AudioDetectionReceiver(port=args.port).start()
    print(f"[*] UDP {args.port} 포트에서 젯슨 패킷 수신 대기. 종료: Ctrl+C\n")
    was_connected = False
    try:
        while True:
            packet = receiver.get_new()
            link = receiver.link()

            if link["connected"] != was_connected:
                was_connected = link["connected"]
                print("[*] 젯슨 연결됨" if was_connected else "[!] 젯슨 연결 끊김")

            if packet is not None and (args.verbose or packet["class"] != "none"):
                latency = time.time() - packet["t"]
                print(f"  seq={packet.get('seq', -1):5d} class={packet['class']:11s} "
                      f"score={packet.get('score', 0):+6.2f} theta={packet['theta']:6.1f} "
                      f"지연={latency*1000:6.0f}ms")
            time.sleep(0.05)
    except KeyboardInterrupt:
        link = receiver.link()
        print(f"\n[*] 종료. 수신 {link['received']}개, 유실 추정 {link['dropped']}개, "
              f"불량 {link['malformed']}개")
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
