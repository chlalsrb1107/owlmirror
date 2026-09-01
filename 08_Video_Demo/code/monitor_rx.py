"""
monitor_rx.py — [노트북] 젯슨 패킷 실시간 모니터. 촬영 현장에서 임계값을 맞추는 도구.

화면(live_demo.py)을 띄우기 전이나 별도 터미널에서 돌려, 소리가 실제로 어느 세기로
들어오는지 눈으로 보면서 스피커 볼륨·거리·`--min-db`를 조정한다.

왜 필요한가 — 임계값이 전부 실내 정지 기준으로 맞춰져 있다. 주행 중에는 엔진·노면·풍절음이
바닥을 올리기 때문에 그대로 쓰면 오경보가 쏟아지거나(바닥이 임계값을 넘음) 아무것도 안
잡힌다(소리가 묻힘). 종료 시 **관측된 소음 바닥을 근거로 `--min-db` 권장값을 출력**한다.

실행:
    python3 monitor_rx.py                 # 30초 관측 후 요약
    python3 monitor_rx.py --seconds 120   # 주행하며 길게
    python3 monitor_rx.py --quiet         # 막대 없이 요약만
"""

import argparse
import json
import socket
import time

DEFAULT_PORT = 9870
DB_MIN, DB_MAX = -60.0, 0.0     # 막대 눈금 범위
BAR_W = 46

# live_demo/alert_policy가 쓰는 기준선 — 막대에 같이 표시해 감각을 맞춘다
GATE_DB = -40.0    # jetson_audio_sender.MIN_ALERT_DB — 이 아래는 아예 안 나감
NEAR_DB = -26.0    # alert_policy.LOUD_NEAR_DB — 사각지대(근접) 판정


def bar(db: float) -> str:
    n = max(0, min(BAR_W, int((db - DB_MIN) * BAR_W / (DB_MAX - DB_MIN))))
    gate = max(0, min(BAR_W - 1, int((GATE_DB - DB_MIN) * BAR_W / (DB_MAX - DB_MIN))))
    near = max(0, min(BAR_W - 1, int((NEAR_DB - DB_MIN) * BAR_W / (DB_MAX - DB_MIN))))
    cells = ["#" if i < n else "." for i in range(BAR_W)]
    for pos, mark in ((gate, "|"), (near, "!")):     # 기준선은 막대 위에 겹쳐 표시
        if cells[pos] == ".":
            cells[pos] = mark
    return "".join(cells)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--seconds", type=float, default=30.0, help="관측 시간")
    ap.add_argument("--quiet", action="store_true", help="막대 없이 요약만")
    args = ap.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("", args.port))
    s.settimeout(2.0)

    print(f"[*] {args.port} 포트에서 {args.seconds:.0f}초 관측. 막대의 `|`={GATE_DB:.0f}dB(차단선) "
          f"`!`={NEAR_DB:.0f}dB(근접)\n")

    t_end = time.time() + args.seconds
    n = 0
    cls = {}
    quiet_rms, alert_rms, gaps = [], [], []
    prev_t = prev_seq = None
    lost = 0

    try:
        while time.time() < t_end:
            try:
                data, _ = s.recvfrom(2048)
            except socket.timeout:
                continue
            try:
                p = json.loads(data.decode("utf-8"))
            except Exception:  # noqa: BLE001
                continue

            n += 1
            now = time.time()
            if prev_t is not None and now - prev_t < 5:
                gaps.append(now - prev_t)
            if prev_seq is not None and p.get("seq", 0) > prev_seq + 1:
                lost += p["seq"] - prev_seq - 1
            prev_t, prev_seq = now, p.get("seq", 0)

            c = p.get("class", "?")
            cls[c] = cls.get(c, 0) + 1
            db = float(p.get("rms_db", -99))
            (alert_rms if c != "none" else quiet_rms).append(db)

            if not args.quiet:
                flag = "  <<< 감지" if c != "none" else ""
                ok = "" if p.get("theta_ok", True) else "  [방향불신]"
                print(f"  {db:6.1f}dB |{bar(db)}| {c:11s} θ{p.get('theta', 0):6.1f}{flag}{ok}")
    except KeyboardInterrupt:
        print("\n[*] 중단됨")
    finally:
        s.close()

    print("\n" + "=" * 62)
    if not n:
        print("  패킷 없음 — 젯슨에서 jetson_audio_sender.py가 도는지 확인하세요.")
        return

    cyc = sum(gaps) / len(gaps) if gaps else 0.0
    det = sum(v for k, v in cls.items() if k != "none")
    print(f"  패킷 {n}개 · 유실 {lost}개 · 주기 {cyc:.2f}s ({1 / cyc if cyc else 0:.2f} Hz)")
    print(f"  분류: {cls}")
    print(f"  감지율 {det / n * 100:.0f}%  ({det}/{n})")

    if quiet_rms:
        q = sorted(quiet_rms)
        floor = q[int(len(q) * 0.9)]     # 소음 바닥의 90퍼센타일 = 사실상의 최고 소음
        print(f"\n  소음 바닥(none일 때): 중앙값 {q[len(q) // 2]:.1f}dB · "
              f"90퍼센타일 {floor:.1f}dB · 최대 {q[-1]:.1f}dB")
        # 소음 바닥보다 3dB 위에 선을 그으면 바닥은 막고 진짜 소리는 통과한다
        rec = round(floor + 3.0)
        print(f"  → 권장 --min-db {rec:.0f}   (소음 90퍼센타일 +3dB)")
        if alert_rms:
            lo = min(alert_rms)
            print(f"  실제 감지된 소리의 최저 음량: {lo:.1f}dB")
            if rec >= lo:
                print(f"  ⚠️ 권장값이 감지 최저({lo:.1f}dB)보다 높다 — 이 값을 쓰면 약한 감지가 잘린다.")
                print(f"     소음을 줄이거나(창문·풍절음), 음원을 마이크에 더 가까이 둘 것.")
    if alert_rms:
        print(f"\n  감지 시 음량: {min(alert_rms):.1f} ~ {max(alert_rms):.1f}dB "
              f"(근접 판정선 {NEAR_DB:.0f}dB)")
    print("=" * 62)


if __name__ == "__main__":
    main()
