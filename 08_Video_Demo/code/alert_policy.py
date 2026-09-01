"""
alert_policy.py — 감지 하나를 화면 상태(주의/경고)로 바꾸는 규칙. 클래스별로 규칙이 다르다.

(2026-09-02) 팀 논의로 클래스별 규칙을 다시 정했다. 이전에는 세 클래스가 모두 "라이다로
거리가 확정되면 경고, 아니면 주의"라는 한 가지 규칙을 공유했는데, 소리마다 운전자가 알아야 할
것이 달라서 아래처럼 갈랐다.

┌ 경적 (car_horn) ─────────────────────────────────────────────────────────────┐
│ 거리는 재지 않는다(라이다 대상 아님). 대신 두 가지를 본다.                     │
│  · 여러 방향에서 동시에 잡히면 **가장 큰 소리**를 배너로 올리고 유지한다.       │
│    유지 중이라도 더 큰 경적이 오면 즉시 바뀌고, LATCH_SEC이 지난 뒤에는          │
│    더 작은 경적이라도 새 사건으로 보고 넘겨준다.                                │
│  · **같은 방향에서 반복**되면 주의 → 경고. 한 번은 지나가는 신호일 수 있지만     │
│    두 번째부터는 나를 향한 것으로 본다.                                         │
└──────────────────────────────────────────────────────────────────────────────┘
┌ 사이렌 (siren) ──────────────────────────────────────────────────────────────┐
│ **항상 경고.** 위치가 확정되든 아니든 긴급차량은 운전자가 위치를 알고 있어야     │
│ 한다는 판단(양보 의무). 확정되면 거리까지 붙고, 미확정이면 방향만 붙는다.        │
└──────────────────────────────────────────────────────────────────────────────┘
┌ 오토바이 (motorcycle) ───────────────────────────────────────────────────────┐
│ 상태가 셋뿐이다. 기준은 "눈으로 확인했나" 하나, 그리고 "가까워지고 있나" 하나.   │
│  · Detection 성공        → 경고. 대상을 잡았으니 위치를 추적해 보여준다          │
│                            (라이다 거리가 있으면 거리까지)                      │
│  · 배기음이 커짐         → 경고. 안 보여도 가까워지고 있다는 뜻                  │
│  · Detection 실패·조용   → 주의. "사각지대 위험"으로 방향만 알린다               │
│                                                                              │
│ 오토바이는 벽 뒤·근접 사각이 가장 위험한데 하필 라이다·카메라가 둘 다 못 보는   │
│ 구간이다. 그래서 안 보여도 소리가 커지면 경고를 낼 수 있어야 하고, 그게          │
│ 이 프로젝트의 차별점(사각지대 경고)이다.                                        │
│                                                                              │
│ (2026-09-02 2차 정리) 처음에는 라이다 사각지대 실측과 소리 기반 추정을 따로      │
│ 뒀는데, 운전자에게는 "확인됐나 / 가까워지나" 둘뿐이라 구분이 무의미해서 합쳤다.  │
│ 라이다는 이제 단계를 정하지 않고 **거리 숫자만 제공**한다.                       │
└──────────────────────────────────────────────────────────────────────────────┘

⚠️ 임계값은 전부 **미보정 추정치**다. rms_db는 절대 음압이 아니라 마이크 입력 기준 상대값이라
   차량·마운트·주행 소음에 따라 통째로 밀린다. 차량 장착 후 실제 주행에서 오토바이 배기음과
   배경 소음의 dB를 찍어보고 MOTO_LOUD_DB부터 다시 맞출 것 (그 전에는 오경보/미경보 어느
   쪽으로도 치우칠 수 있다).

단독 실행:
    python3 alert_policy.py --selftest    # 하드웨어 없이 규칙만 검증
"""

import argparse
import time

# ---- 경적 -----------------------------------------------------------------------
HORN_LATCH_SEC = 1.5        # 이 안에 들어온 경적은 "같은 사건"으로 보고 더 큰 것만 채택
HORN_HOLD_SEC = 4.0         # 경적 배너 유지 시간. 짧은 소리라 다른 클래스(3.0s)보다 길게 잡았다
HORN_REPEAT_WINDOW_SEC = 12.0   # 이 시간 안에
HORN_REPEAT_ANGLE_DEG = 40.0    # 이 각도 안에서
HORN_REPEAT_COUNT = 2           # 이 횟수 이상 울리면 경고로 승격

# ---- 근접 판정 (세 클래스 공통, 2026-09-02 통일) --------------------------------
# 소리가 아주 크면 음원이 바로 옆에 있다는 뜻이고, 그 거리대(반경 6m)는 하필 라이다가
# 못 보는 구간이다. 클래스마다 다르게 둘 이유가 없어 하나로 합쳤다.
#   · 라이다를 쓰는 클래스(사이렌·오토바이): 6m 이내 관측 → 사각지대
#   · 세 클래스 전부: 음량이 이 값을 넘으면 → 사각지대 근접으로 간주
# ⚠️ 미보정 추정치. rms_db는 절대 음압이 아니라 마이크 입력 기준 상대값이라 차량·마운트·
#    주행 소음에서 반드시 다시 맞춰야 한다.
LOUD_NEAR_DB = -26.0
MOTO_LOUD_DB = LOUD_NEAR_DB  # 이름 호환용 (기존 참조가 있을 수 있어 남겨둔다)


def _angle_close(a: float, b: float, margin_deg: float) -> bool:
    return abs((a - b + 180.0) % 360.0 - 180.0) <= margin_deg


class AlertPolicy:
    """클래스별 상태 판정. 경적의 래치·반복은 상태를 들고 있어야 해서 클래스로 만들었다.

    detection_worker 스레드에서만 호출되므로 별도 락은 두지 않는다.
    """

    def __init__(self):
        self._horn_theta = None      # 현재 배너를 차지한 경적의 방향
        self._horn_db = None
        self._horn_at = 0.0          # 그 경적을 채택한 시각
        self._horn_history = []      # [(시각, 방향)] — 반복 판정용

    # ---- 공통 ---------------------------------------------------------------
    @staticmethod
    def is_near(rms_db: float) -> bool:
        """음량만으로 "바로 옆"을 판정. 라이다·카메라가 못 보는 근접 구간을 소리로 메운다."""
        return rms_db >= LOUD_NEAR_DB

    # ---- 경적 ---------------------------------------------------------------
    def horn_accepts(self, theta: float, rms_db: float, now: float = None) -> bool:
        """이 경적이 배너를 차지해야 하는가.

        · 유지 중인 경적이 없거나 유지 시간이 끝났으면 → 채택
        · LATCH_SEC 안이면 → **더 큰 소리일 때만** 채택 (동시다발 중 가장 큰 것 고르기)
        · LATCH_SEC이 지났으면 → 새 사건으로 보고 크기와 무관하게 채택
        """
        now = time.time() if now is None else now
        if self._horn_theta is None or now - self._horn_at > HORN_HOLD_SEC:
            return True
        if now - self._horn_at <= HORN_LATCH_SEC:
            return rms_db > (self._horn_db if self._horn_db is not None else -999.0)
        return True

    def horn_level(self, theta: float, rms_db: float, now: float = None):
        """경적을 채택하고 (level, repeats, near)를 돌려준다.

        horn_accepts()가 True일 때만 호출할 것. near는 음량 기준 근접(사각지대) 여부다.
        """
        now = time.time() if now is None else now
        self._horn_theta, self._horn_db, self._horn_at = theta, rms_db, now

        self._horn_history = [(t, a) for t, a in self._horn_history
                              if now - t <= HORN_REPEAT_WINDOW_SEC]
        self._horn_history.append((now, theta))
        repeats = sum(1 for _, a in self._horn_history
                      if _angle_close(a, theta, HORN_REPEAT_ANGLE_DEG))
        near = AlertPolicy.is_near(rms_db)
        # 같은 방향에서 두 번째면 나를 향한 신호로 보고, 아주 가까우면 그것만으로도 경고.
        level = "경고" if (repeats >= HORN_REPEAT_COUNT or near) else "주의"
        return level, repeats, near

    # ---- 사이렌 -------------------------------------------------------------
    @staticmethod
    def siren_level(match, rms_db: float = -99.0):
        """긴급차량은 위치 확정 여부와 무관하게 항상 경고. 확정되면 거리가 붙을 뿐이다.

        라이다가 6m 이내에서 잡거나 소리가 아주 크면 사각지대(blind)로 표시한다.
        """
        if match is not None and match["blind"]:
            return "경고", None, True
        if match is not None:
            return "경고", match["distance_m"], False
        return "경고", None, AlertPolicy.is_near(rms_db)

    # ---- 오토바이 -----------------------------------------------------------
    @staticmethod
    def motorcycle_level(match, detected: bool, rms_db: float, score: float = 0.0):
        """(level, distance, blind, reason)을 돌려준다. reason은 화면 문구를 고르는 데 쓴다.

        상태는 셋뿐이다 — 잡았거나(경고·위치 추적), 가까워지거나(경고·근접), 모르거나(주의).
        라이다는 단계를 정하지 않고 잡힌 대상의 **거리 숫자만** 보탠다.
        """
        if detected:
            # 카메라로 확인됐으면 실체가 있는 것 — 경고까지 올리고 위치를 추적해 보여준다.
            # 라이다가 사각지대로 보고했으면 그 사실도 함께 넘긴다(거리는 못 준다).
            if match is not None and match["blind"]:
                return "경고", None, True, "사각지대"
            distance = match["distance_m"] if match else None
            return "경고", distance, False, "위치 추적"
        if match is not None and match["blind"]:
            # 눈으로는 못 봤지만 라이다가 6m 이내에서 뭔가 잡았다 — 가장 위험한 구간이다
            return "경고", None, True, "사각지대"
        if AlertPolicy.is_near(rms_db):
            # 안 보이는데 소리가 커지고 있다 = 사각지대에서 접근 중.
            # 이 프로젝트의 차별점이 성립하는 경로다.
            return "경고", None, True, "근접"
        return "주의", None, False, "사각지대 위험"


def priority_rank(class_name: str, level: str, blind: bool) -> int:
    """숫자가 작을수록 우선순위 높음. 동시 감지 시 배너/메인 카메라를 누가 차지할지 정한다.

    (2026-09-02) 클래스별 규칙 변경에 맞춰 재정렬. 큰 원칙은 그대로다 —
    물리적 충돌 임박(오토바이 근접·사각) > 법적 의무(사이렌) > 신호(경적).
    사이렌이 이제 항상 경고라, 사이렌 안에서는 위치 확정 여부로 순위를 가른다.
    """
    if class_name == "motorcycle" and blind:
        return 1   # 오토바이 · 사각지대(라이다 실측 또는 소리 근접) — 충돌 최임박
    if class_name == "motorcycle" and level == "경고":
        return 2   # 오토바이 · 영상으로 확인·추적 중
    if class_name == "siren":
        return 3   # 사이렌 — 항상 경고, 응급차량 양보 의무
    if class_name == "car_horn" and level == "경고":
        return 4   # 경적 · 같은 방향 반복 = 나를 향한 신호
    if class_name == "car_horn":
        return 5   # 경적 · 1회
    return 6       # 오토바이 · 주의(접근 관찰 / 방향 미확정)


# ---- 셀프테스트 -------------------------------------------------------------------
def run_selftest():
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}\n"
              f"          got={got}\n          want={want}" if not good
              else f"  [PASS] {name}")

    # ── 경적: 동시다발 중 가장 큰 것 ───────────────────────────────────────────
    p, t = AlertPolicy(), 1000.0
    check("경적 첫 감지는 무조건 채택", p.horn_accepts(90, -30, t), True)
    p.horn_level(90, -30, t)
    check("래치 안, 더 작은 경적은 거절", p.horn_accepts(200, -35, t + 0.5), False)
    check("래치 안, 더 큰 경적은 채택", p.horn_accepts(200, -20, t + 0.5), True)
    check("래치 후, 작은 경적도 새 사건으로 채택", p.horn_accepts(200, -35, t + 2.0), True)
    check("유지 시간이 끝나면 당연히 채택", p.horn_accepts(200, -50, t + 9.0), True)

    # ── 경적: 같은 방향 반복 → 경고 (멀리서 울린 경우, -40dB) ──────────────────
    p, t = AlertPolicy(), 2000.0
    check("경적 1회는 주의", p.horn_level(90, -40, t)[0], "주의")
    check("같은 방향 2회는 경고", p.horn_level(100, -40, t + 5)[0], "경고")
    p2 = AlertPolicy()
    p2.horn_level(90, -40, t)
    check("다른 방향이면 다시 주의", p2.horn_level(200, -40, t + 5)[0], "주의")
    p3 = AlertPolicy()
    p3.horn_level(90, -40, t)
    check("시간이 지나면 반복으로 안 침", p3.horn_level(90, -40, t + 30)[0], "주의")

    # ── 근접 판정은 세 클래스 공통 ─────────────────────────────────────────────
    p4 = AlertPolicy()
    lv, _, near = p4.horn_level(90, -18, 3000.0)
    check("경적도 아주 크면 1회로 경고+근접", (lv, near), ("경고", True))
    check("사이렌: 라이다 없고 소리 크면 사각지대",
          AlertPolicy.siren_level(None, -18.0), ("경고", None, True))
    check("사이렌: 라이다 없고 조용하면 사각지대 아님",
          AlertPolicy.siren_level(None, -45.0), ("경고", None, False))
    check("사이렌: 라이다 6m 이내면 사각지대",
          AlertPolicy.siren_level({"distance_m": None, "blind": True}, -45.0),
          ("경고", None, True))

    # ── 사이렌: 언제나 경고 ────────────────────────────────────────────────────
    check("사이렌 미확정도 경고", AlertPolicy.siren_level(None, -45.0), ("경고", None, False))
    check("사이렌 확정은 경고+거리",
          AlertPolicy.siren_level({"distance_m": 32.0, "blind": False}, -45.0),
          ("경고", 32.0, False))

    # ── 오토바이 (상태 3개) ────────────────────────────────────────────────────
    m = AlertPolicy.motorcycle_level
    check("Detection 실패 + 조용 → 주의",
          m(None, False, -45.0), ("주의", None, False, "사각지대 위험"))
    check("Detection 실패 + 소리 커짐 → 경고(근접)",
          m(None, False, -20.0), ("경고", None, True, "근접"))
    check("Detection 성공 → 경고(위치 추적)",
          m(None, True, -45.0), ("경고", None, False, "위치 추적"))
    check("Detection 성공 + 라이다 거리 → 거리까지 표시",
          m({"distance_m": 18.0, "blind": False}, True, -45.0),
          ("경고", 18.0, False, "위치 추적"))

    check("라이다에 잡혀도 Detection 실패면 단계는 안 올라감",
          m({"distance_m": 12.0, "blind": False}, False, -45.0),
          ("주의", None, False, "사각지대 위험"))
    check("오토바이: 라이다 6m 이내면 Detection 없어도 경고",
          m({"distance_m": None, "blind": True}, False, -45.0),
          ("경고", None, True, "사각지대"))
    check("오토바이: Detection + 라이다 사각지대",
          m({"distance_m": None, "blind": True}, True, -45.0),
          ("경고", None, True, "사각지대"))

    # ── 우선순위 ──────────────────────────────────────────────────────────────
    order = [("motorcycle", "경고", True), ("motorcycle", "경고", False), ("siren", "경고", False),
             ("car_horn", "경고", False), ("car_horn", "주의", False), ("motorcycle", "주의", False)]
    ranks = [priority_rank(*c) for c in order]
    check("우선순위가 오름차순", ranks, sorted(ranks))
    check("사이렌이 경적보다 위",
          priority_rank("siren", "경고", False) < priority_rank("car_horn", "경고", False), True)
    check("오토바이 근접이 사이렌보다 위",
          priority_rank("motorcycle", "경고", True) < priority_rank("siren", "경고", False), True)

    print(f"\n[selftest] {'ALL PASS' if ok else 'FAIL 있음'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="클래스별 알림 상태 규칙")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        raise SystemExit(0 if run_selftest() else 1)
    parser.print_help()


if __name__ == "__main__":
    main()
