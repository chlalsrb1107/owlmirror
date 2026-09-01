"""
bev_render.py — 9/8 데모 화면 합성: 왼쪽 BEV 지도 | 오른쪽 카메라 영상 (반반).

08_Video_Demo/camera_ui_mockup.html(2026-09-01 목업)의 레이아웃을 실제 화면으로 옮긴 것이다.
live_demo.py가 프레임마다 render()를 불러 최종 1600x900 BGR 캔버스를 받는다.

BEV는 단순화된 마커가 아니라 **LiDAR가 실제로 들고 있는 포인트**로 그린다
(ui_state_spec.md / camera_ui_mockup.html의 "성긴 점=미확정, 점 클러스터=확정" 원칙):

  - 지면 제거를 통과한 모든 리턴  → 흐린 회색 점 (배경 지형지물)
  - 매칭 확정(match_distance 성공) → 그 방향·거리대의 점을 클래스 색으로 다시 칠하고
                                     후광 + 글리프 + 거리 라벨
  - 미확정                        → 부채꼴(방향 ± sigma)만, 안쪽 점은 옅게
  - 사각지대(6m 이내)             → 점 위치 대신 사각 구역 자체를 붉게 점등

⚠️ 각도 규약이 두 개라 반드시 한 번 뒤집어야 한다.
   시스템 theta = **전방 0°, 반시계 +** (jetson_audio_sender.py가 보내는 값,
   doa_camera_select.py의 select_camera(90)=="left"로 확인됨).
   목업 SVG = 시계 + (0=전, 90=우).
   여기서는 _screen_angle()에서 한 번만 부호를 뒤집는다 — 빠뜨리면 BEV가 통째로
   좌우 반전돼 그려진다(왼쪽 사이렌이 오른쪽에 뜬다).

⚠️ LiDAR가 없거나 죽어도 BEV는 계속 그린다 — 링·자차·부채꼴은 오디오 방향만으로 성립하므로,
   LIDAR_AVAILABLE=False 폴백(2026-08-25 축소 버전 동등)에서도 화면 구성은 유지된다.
   대신 좌하단에 "LiDAR 미연결 — 방향만 표시"를 명시해 거리 없음을 숨기지 않는다.

단독 실행:
    python3 bev_render.py --selftest [--outdir /tmp/bev]
        하드웨어 없이 가짜 포인트/감지로 시나리오별 PNG를 렌더링해 레이아웃을 눈으로 확인한다.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

import lidar_distance_match as lidar

# ---- 캔버스 --------------------------------------------------------------------
# 목업(1920x1080)의 비율을 그대로 900p로 줄인 값. 촬영·화면녹화(OBS) 대상 해상도이기도 하다.
CANVAS_W, CANVAS_H = 1600, 900
BANNER_H = 125
SPLIT_X = CANVAS_W // 2          # 좌: BEV / 우: 카메라

# ---- BEV 좌표계 ----------------------------------------------------------------
CX, CY = 400, 512                # 자차 위치(=LiDAR 원점)
RMAX = 300                       # RANGE_MAX_M에 해당하는 픽셀 반지름
RANGE_MAX_M = 40.0               # 지도 바깥 테두리까지의 거리
RING_M = (10.0, 20.0, 40.0)
BLIND_M = lidar.BLIND_RADIUS_M   # 6.0 — 물리적 사각지대

# ---- 색 (BGR) ------------------------------------------------------------------
PAGE_BG = (10, 8, 8)             # #08080A
CAM_BG = (13, 11, 10)            # 카메라 레터박스 여백
EDGE = (39, 35, 35)              # #232327
TXT = (245, 245, 245)
DIM = (130, 122, 122)            # #7A7A82
TEAL = (199, 194, 0)             # #00C2C7
YELLOW = (0, 212, 255)           # #FFD400
RED = (48, 59, 255)              # #FF3B30
VIOLET = (255, 140, 185)         # #B98CFF
RING = (40, 33, 30)              # #1E2128
RING_TXT = (77, 67, 62)          # #3E434D
BLIND_FILL = (16, 13, 12)        # #101216 계열. sqrt 스케일이라 6m가 반지름의 38%를 차지해서,
                                 # 밝게 칠하면 지도를 삼킨다 — 페이지 배경에 가깝게 낮춰 둔다.
AMBIENT_PT = (54, 46, 42)        # #2A2E36

KIND_COLOR = {"car_horn": TEAL, "siren": RED, "motorcycle": VIOLET}
LEVEL_COLOR = {"주의": YELLOW, "경고": RED}
CLASS_LABEL_KO = {"car_horn": "경적", "siren": "사이렌", "motorcycle": "오토바이"}
CAMERA_LABEL_KO = {"front": "전방 카메라", "left": "좌측 카메라",
                   "right": "우측 카메라", "rear": "후방 카메라(기본)"}
LOC_LABEL_KO = {"front": "전방", "left": "좌측", "right": "우측", "rear": "후방"}
GLYPH_KIND = {"car_horn": "horn", "siren": "siren", "motorcycle": "moto"}

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # 최후 폴백(한글 깨짐, 죽지는 않음)
]

_font_path_cache = None
_font_cache = {}
_background_cache = None
_scratch = None


# ---- 폰트 / 텍스트 --------------------------------------------------------------
def resolve_font_path():
    """존재하는 첫 한글 폰트 경로. 하나도 없으면 명확한 안내와 함께 예외."""
    global _font_path_cache
    if _font_path_cache is None:
        for candidate in FONT_CANDIDATES:
            if Path(candidate).exists():
                _font_path_cache = candidate
                break
        else:
            raise FileNotFoundError(
                "한글 폰트를 찾을 수 없습니다. `sudo apt install fonts-nanum` 후 다시 실행하거나 "
                "FONT_CANDIDATES에 실제 경로를 추가하세요.")
    return _font_path_cache


def _font(size: int):
    from PIL import ImageFont

    if size not in _font_cache:
        _font_cache[size] = ImageFont.truetype(resolve_font_path(), size)
    return _font_cache[size]


class TextLayer:
    """한글 텍스트를 모았다가 **한 번에** 그린다. 같은 내용이면 이전 프레임 결과를 재사용한다.

    cv2.putText가 한글을 못 그려 Pillow를 써야 하는데, 호출마다 프레임 전체를 BGR->PIL->BGR로
    왕복시키면 1600x900 기준 14ms가 든다(실측). BEV가 들어오면서 라벨이 10개를 넘어, 그 방식은
    표시 루프 혼자 45fps 상한을 만들어 카메라 30Hz + 왜곡보정과 겹치면 프레임을 떨어뜨린다.

    그래서 두 단계로 줄였다:
      1. 왕복을 프레임당 1회로 묶고,
      2. 글자를 **투명 RGBA 레이어에 한 번만** 그려 (픽셀 좌표, 색, 알파)로 캐시한 뒤
         이후 프레임은 그 희소 픽셀만 덮어쓴다 — 내용이 바뀔 때만 다시 그린다.
    화면 글자는 감지가 바뀔 때나 바뀌므로 대부분의 프레임이 캐시 적중이다(14ms -> 0.5ms).
    ⚠️ 그래서 매 프레임 값이 달라지는 문구는 캐시를 무력화한다 — 포인트 수를 100단위로
       반올림해 표시하는 이유가 이것이다.
    """

    def __init__(self):
        self._items = []
        self._key = None
        self._cache = None

    def add(self, text, xy, size, color_bgr, anchor="lt"):
        self._items.append((text, (float(xy[0]), float(xy[1])), size,
                            tuple(int(c) for c in color_bgr), anchor))

    def flush(self, img_bgr):
        items, self._items = self._items, []
        if not items:
            return
        key = tuple(items)
        if key != self._key:
            self._key, self._cache = key, self._bake(items)
        ys, xs, rgb, alpha = self._cache
        if ys.size == 0:
            return
        base = img_bgr[ys, xs].astype(np.float32)
        img_bgr[ys, xs] = (base * (1.0 - alpha) + rgb * alpha).astype(np.uint8)

    @staticmethod
    def _bake(items):
        from PIL import Image, ImageDraw

        layer = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for text, xy, size, color, anchor in items:
            draw.text(xy, text, font=_font(size), anchor=anchor,
                      fill=(color[2], color[1], color[0], 255))
        arr = np.asarray(layer)
        ys, xs = np.nonzero(arr[:, :, 3])
        rgb = arr[ys, xs, :3][:, ::-1].astype(np.float32)          # RGB -> BGR
        alpha = (arr[ys, xs, 3].astype(np.float32) / 255.0)[:, None]
        return ys, xs, rgb, alpha


_TEXT = TextLayer()


# ---- 좌표 변환 ------------------------------------------------------------------
def _screen_angle(theta_ccw_deg):
    """시스템 theta(전방 0°, 반시계 +) → 화면 각도(전방 0°, 시계 +). 위 ⚠️ 참고."""
    return -np.asarray(theta_ccw_deg, dtype=np.float64)


def radius_px(dist_m):
    """거리 → 픽셀 반지름. sqrt 스케일이라 가까운 쪽이 넓게 잡힌다(근접 위험이 더 중요)."""
    d = np.clip(np.asarray(dist_m, dtype=np.float64), 0.0, RANGE_MAX_M)
    return np.sqrt(d / RANGE_MAX_M) * RMAX


def to_px(theta_ccw_deg, dist_m):
    """(theta, 거리) → (x, y) 픽셀. 스칼라/배열 모두 지원."""
    a = np.radians(_screen_angle(theta_ccw_deg) - 90.0)
    r = radius_px(dist_m)
    return CX + np.cos(a) * r, CY + np.sin(a) * r


def _cv_arc_angles(theta_ccw_deg, spread_deg):
    """cv2.ellipse용 (start, end). cv2는 +x축에서 시계방향(y가 아래) 각도를 쓴다."""
    center = float(_screen_angle(theta_ccw_deg)) - 90.0
    return center - spread_deg / 2.0, center + spread_deg / 2.0


# ---- 글리프 (목업 SVG를 cv2 프리미티브로 옮긴 것) --------------------------------
def _glyph(img, kind, cx, cy, scale, color, thickness=2):
    def p(x, y):
        return (int(round(cx + x * scale)), int(round(cy + y * scale)))

    t = max(1, int(round(thickness * scale * 1.6)))
    if kind == "horn":
        cv2.fillPoly(img, [np.array([p(-34, -16), p(-34, 16), p(-16, 24), p(-16, -24)])], color)
        for i, rr in enumerate((16, 30, 44)):
            cv2.ellipse(img, p(-6, 0), (int(rr * scale), int(rr * scale)), 0, -46, 46,
                        color, max(1, t - i), cv2.LINE_AA)
    elif kind == "siren":
        cv2.ellipse(img, p(0, 10), (int(24 * scale), int(24 * scale)), 0, 180, 360, color, -1,
                    cv2.LINE_AA)
        cv2.rectangle(img, p(-32, 10), p(32, 23), color, -1)
        for a, b in (((-40, -22), (-30, -12)), ((40, -22), (30, -12)), ((0, -42), (0, -30))):
            cv2.line(img, p(*a), p(*b), color, t, cv2.LINE_AA)
    elif kind == "moto":
        # ⚠️ 얇은 원 두 개 + 가는 프레임으로 그리면 작은 크기에서 자전거로 읽힌다.
        #    바퀴를 두껍게, 가운데를 **채운 덩어리**(엔진+탱크)로 그려야 오토바이가 된다.
        wheel = max(2, int(round(t * 1.5)))
        cv2.circle(img, p(-26, 16), int(13 * scale), color, wheel, cv2.LINE_AA)
        cv2.circle(img, p(26, 16), int(13 * scale), color, wheel, cv2.LINE_AA)
        cv2.fillPoly(img, [np.array([p(-24, 2), p(-8, -4), p(4, -10), p(14, -10),
                                     p(17, -1), p(12, 9), p(-15, 9)])], color)
        cv2.line(img, p(26, 16), p(17, -3), color, wheel, cv2.LINE_AA)      # 앞 포크
        cv2.line(img, p(14, -11), p(27, -17), color, wheel, cv2.LINE_AA)    # 핸들바
        cv2.line(img, p(-26, 16), p(-13, 9), color, t, cv2.LINE_AA)         # 배기구


# ---- 정적 배경 (한 번만 그려서 재사용) -------------------------------------------
def _dashed_circle(img, center, radius, color, dash_deg=5, gap_deg=9):
    a = 0.0
    while a < 360.0:
        cv2.ellipse(img, center, (radius, radius), 0, a, min(a + dash_deg, 360.0),
                    color, 1, cv2.LINE_AA)
        a += dash_deg + gap_deg


def build_background():
    """링·거리 라벨·방위 글자·분할선처럼 매 프레임 바뀌지 않는 것들을 미리 그려 캐시."""
    global _background_cache
    if _background_cache is not None:
        return _background_cache.copy()

    img = np.empty((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    img[:] = PAGE_BG
    img[BANNER_H:, SPLIT_X:] = CAM_BG
    cv2.line(img, (SPLIT_X, BANNER_H), (SPLIT_X, CANVAS_H), EDGE, 2)

    text = TextLayer()
    for m in RING_M:
        r = int(radius_px(m))
        if m == RING_M[-1]:
            _dashed_circle(img, (CX, CY), r, RING)
        else:
            cv2.circle(img, (CX, CY), r, RING, 1, cv2.LINE_AA)
        text.add(f"{m:.0f}m", (CX + 6, CY - r + 3), 13, RING_TXT)

    # 사각지대 원반은 링보다 위에 올려 안쪽 링 선이 비치지 않게 한다.
    cv2.circle(img, (CX, CY), int(radius_px(BLIND_M)), BLIND_FILL, -1, cv2.LINE_AA)
    _dashed_circle(img, (CX, CY), int(radius_px(BLIND_M)), (44, 37, 34), dash_deg=4, gap_deg=7)

    # 방위 글자는 **시스템 theta 기준**으로 배치한다(반시계 +): 우측은 theta=-90.
    for label, theta in (("전", 0.0), ("우", -90.0), ("후", 180.0), ("좌", 90.0)):
        x, y = to_px(theta, RANGE_MAX_M)
        x += (x - CX) * 0.12
        y += (y - CY) * 0.12
        text.add(label, (float(x), float(y)), 14, RING_TXT, anchor="mm")

    text.flush(img)
    _background_cache = img
    return img.copy()


# ---- BEV 동적 요소 ---------------------------------------------------------------
def _blend_shape(img, draw_fn, color, alpha):
    """반투명 도형: BEV 절반만 복사해 그린 뒤 합성한다 (전체 캔버스 왕복 회피).

    반투명 요소는 전부 BEV 안에만 있으므로 오른쪽 카메라 절반까지 복사할 이유가 없다.
    draw_fn에 넘어가는 이미지는 좌표계가 캔버스와 같도록 원본의 뷰(슬라이스)를 쓴다.
    """
    global _scratch
    if _scratch is None or _scratch.shape != img.shape:
        _scratch = np.empty_like(img)
    roi = img[BANNER_H:, :SPLIT_X]
    _scratch[BANNER_H:, :SPLIT_X] = roi     # BEV 절반만 복사
    draw_fn(_scratch)
    cv2.addWeighted(_scratch[BANNER_H:, :SPLIT_X], alpha, roi, 1.0 - alpha, 0, dst=roi)


def draw_lidar_points(img, points, color=AMBIENT_PT, size=1):
    """포인트 배열(N x 5: x,y,z,range,azimuth)을 BEV에 흩뿌린다. 벡터화 — 수만 점도 1ms 미만."""
    if points is None or points.shape[0] == 0:
        return
    rng, azi = points[:, 3], points[:, 4]
    keep = (rng > 0.2) & (rng <= RANGE_MAX_M)
    if not np.any(keep):
        return
    x, y = to_px(azi[keep], rng[keep])
    px = np.rint(x).astype(np.int32)
    py = np.rint(y).astype(np.int32)
    inside = (px >= size) & (px < SPLIT_X - size) & (py >= BANNER_H + size) & (py < CANVAS_H - size)
    px, py = px[inside], py[inside]
    for dy in range(-size + 1, size + 1):
        for dx in range(-size + 1, size + 1):
            img[py + dy, px + dx] = color


def _select_cluster_points(points, theta_deg, distance_m, angle_margin_deg):
    """확정된 대상에 해당하는 포인트만 골라낸다 (방향 창 + 거리 창 + 지면 제거)."""
    if points is None or points.shape[0] == 0:
        return None
    rel = (points[:, 4] - theta_deg + 180) % 360 - 180
    sel = points[np.abs(rel) <= angle_margin_deg]
    if sel.shape[0] == 0:
        return None
    sel = lidar.filter_ground(sel)
    if sel.shape[0] == 0:
        return None
    near = np.abs(sel[:, 3] - distance_m) <= lidar.CLUSTER_GAP_M
    return sel[near] if np.any(near) else None


def draw_wedge(img, theta_deg, spread_deg, color, points=None):
    """미확정 신호: 방향 ± sigma 부채꼴. 안쪽 포인트는 옅게 색을 입혀 '신호는 있음'을 보인다."""
    start, end = _cv_arc_angles(theta_deg, spread_deg)
    _blend_shape(img, lambda o: cv2.ellipse(
        o, (CX, CY), (int(RMAX * 1.02), int(RMAX * 1.02)), 0, start, end, color, -1, cv2.LINE_AA),
        color, 0.09)
    cv2.ellipse(img, (CX, CY), (int(RMAX * 1.02), int(RMAX * 1.02)), 0, start, end,
                color, 1, cv2.LINE_AA)
    if points is not None and points.shape[0]:
        rel = (points[:, 4] - theta_deg + 180) % 360 - 180
        inner = points[np.abs(rel) <= spread_deg / 2.0]
        draw_lidar_points(img, inner, tuple(int(c * 0.55) for c in color), size=1)


def draw_cluster(img, theta_deg, distance_m, color, cluster_points=None, pulse=False):
    """확정 대상: 실제 리턴 점을 클래스 색으로 다시 칠하고 후광을 얹는다."""
    cx, cy = to_px(theta_deg, distance_m)
    cx, cy = int(round(float(cx))), int(round(float(cy)))
    _blend_shape(img, lambda o: cv2.circle(o, (cx, cy), 40, color, -1, cv2.LINE_AA), color, 0.14)
    if pulse:
        cv2.circle(img, (cx, cy), 56, color, 1, cv2.LINE_AA)
    if cluster_points is not None and cluster_points.shape[0]:
        draw_lidar_points(img, cluster_points, color, size=2)
    else:
        cv2.circle(img, (cx, cy), 4, color, -1, cv2.LINE_AA)
    return cx, cy


def draw_blind(img, theta_deg, spread_deg=90.0):
    """사각지대(6m 이내): 점 위치 대신 구역 자체를 점등 — 없는 정확도를 있는 척하지 않는다."""
    start, end = _cv_arc_angles(theta_deg, spread_deg)
    rb = int(radius_px(BLIND_M))
    _blend_shape(img, lambda o: cv2.ellipse(o, (CX, CY), (rb, rb), 0, start, end, RED, -1,
                                            cv2.LINE_AA), RED, 0.38)
    cv2.ellipse(img, (CX, CY), (rb, rb), 0, start, end, RED, 1, cv2.LINE_AA)


def draw_ego(img):
    cv2.rectangle(img, (CX - 13, CY - 22), (CX + 13, CY + 22), TEAL, 2, cv2.LINE_AA)
    cv2.polylines(img, [np.array([[CX - 7, CY - 5], [CX, CY - 15], [CX + 7, CY - 5]])],
                  False, TEAL, 2, cv2.LINE_AA)


# ---- 카메라 절반 -----------------------------------------------------------------
CAMERA_ORDER = ("front", "left", "rear", "right")
CAMERA_SHORT_KO = {"front": "전", "left": "좌", "rear": "후", "right": "우"}


def draw_camera_selector(img, active, level, text, x0, y0, width):
    """어느 카메라가 잡혀 있는지 4칸 스트립으로 표시.

    카메라(2048x1200)와 표시 영역(800x775)의 비율이 달라 위아래로 레터박스 여백이 생기는데,
    화각을 잘라내면서까지 채우는 것보다(주행 중 대상이 가장자리로 지나간다) 그 여백을
    "지금 어느 방향 카메라인지"에 쓰는 편이 낫다 — 영상에서 카메라 자동 전환이 눈에 보인다.
    """
    cell_w, cell_h, gap = 62, 30, 6
    total = len(CAMERA_ORDER) * cell_w + (len(CAMERA_ORDER) - 1) * gap
    sx = x0 + (width - total) // 2
    for i, name in enumerate(CAMERA_ORDER):
        cx0 = sx + i * (cell_w + gap)
        on = name == active
        color = LEVEL_COLOR.get(level, TEAL) if on else EDGE
        cv2.rectangle(img, (cx0, y0), (cx0 + cell_w, y0 + cell_h), color, -1 if on else 1)
        text.add(CAMERA_SHORT_KO[name], (cx0 + cell_w // 2, y0 + cell_h // 2), 17,
                 (12, 12, 12) if on else DIM, anchor="mm")


def draw_camera_pane(img, frame_bgr, detection, text, active_camera=None, level=None):
    """카메라 프레임을 오른쪽 절반에 레터박스로 넣고, Detection 박스를 실제 좌표에 그린다.

    반환: 영상이 실제로 놓인 (x0, y0, w, h) — 없으면 None.
    """
    pane_w = CANVAS_W - SPLIT_X
    pane_h = CANVAS_H - BANNER_H
    if frame_bgr is None:
        text.add("카메라 프레임 없음", (SPLIT_X + pane_w // 2, BANNER_H + pane_h // 2),
                 22, DIM, anchor="mm")
        return None

    fh, fw = frame_bgr.shape[:2]
    scale = min(pane_w / fw, pane_h / fh)
    vw, vh = int(fw * scale), int(fh * scale)
    x0 = SPLIT_X + (pane_w - vw) // 2
    y0 = BANNER_H + (pane_h - vh) // 2
    img[y0:y0 + vh, x0:x0 + vw] = cv2.resize(frame_bgr, (vw, vh), interpolation=cv2.INTER_AREA)

    if detection is not None:
        col = KIND_COLOR.get(detection.get("kind", "siren"), RED)
        box = detection.get("box")
        if box is not None:                       # YOLO가 준 실제 bbox (원본 프레임 좌표)
            bx0, by0, bx1, by1 = (int(v * scale) for v in box)
            bx0, bx1 = x0 + bx0, x0 + bx1
            by0, by1 = y0 + by0, y0 + by1
        else:                                     # 좌표를 못 받은 경우엔 박스를 그리지 않는다
            bx0 = by0 = bx1 = by1 = None
        if bx0 is not None:
            cv2.rectangle(img, (bx0, by0), (bx1, by1), col, 2, cv2.LINE_AA)
            tag = f"{detection['label']} {round(detection['conf'] * 100)}%"
            cv2.rectangle(img, (bx0, max(BANNER_H, by0 - 26)), (bx0 + 150, by0), col, -1)
            cv2.putText(img, tag, (bx0 + 6, by0 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (10, 10, 10), 1, cv2.LINE_AA)

    if active_camera is not None and (y0 - BANNER_H) >= 46:
        draw_camera_selector(img, active_camera, level, text,
                             SPLIT_X, BANNER_H + (y0 - BANNER_H - 30) // 2, pane_w)
    return x0, y0, vw, vh


# ---- 배너 -------------------------------------------------------------------------
def draw_banner(img, banner, text):
    cv2.rectangle(img, (0, 0), (CANVAS_W, BANNER_H), (0, 0, 0), -1)
    cv2.line(img, (0, BANNER_H), (CANVAS_W, BANNER_H), EDGE, 1)
    if banner is None:
        text.add("올빼미러 · 감지 없음", (24, 46), 22, DIM)
        return

    col = KIND_COLOR[banner["class_name"]]
    level_col = LEVEL_COLOR[banner["level"]]
    cv2.rectangle(img, (20, 24), (206, 100), col, -1)
    text.add(CLASS_LABEL_KO[banner["class_name"]], (113, 62), 30, (12, 12, 12), anchor="mm")

    # detail은 왜 이 단계가 됐는지를 알려주는 근거 문구(alert_policy가 정한다).
    # 거리가 있으면 거리를 같이 붙인다 — "경고 → 좌측 · 차량 확정 · 32m"
    parts = [banner["level"], "→", banner["loc"]]
    line = " ".join(parts)
    if banner.get("detail"):
        line += " · " + banner["detail"]
    if banner["distance"] is not None:
        line += f" · {banner['distance']:.0f}m"
    text.add(line, (226, 62), 26, level_col, anchor="lm")


# ---- 최종 합성 ---------------------------------------------------------------------
def render(camera_frame_bgr, camera_name, targets, points=None,
           lidar_ok=True, link=None, detection=None):
    """한 프레임을 합성해 1600x900 BGR 캔버스로 돌려준다.

    targets: 우선순위 순 리스트. 각 항목은
        {"class_name", "theta", "level", "distance"|None, "blind", "sigma"}
        0번이 배너/메인 카메라를 차지하는 1위이고, 나머지도 **BEV에는 전부 그린다**
        (ui_state_spec.md §2 "지도는 전부, 알림은 하나").
    points: LidarScanner.latest_points() 결과 (N x 5) 또는 None.
    """
    img = build_background()
    text = _TEXT  # 모듈 전역 — 프레임 사이에 살아 있어야 글자 캐시가 적중한다

    ground_filtered = None
    if points is not None and points.shape[0]:
        ground_filtered = lidar.filter_ground(points)
        draw_lidar_points(img, ground_filtered, AMBIENT_PT, size=1)

    for i, tgt in enumerate(targets):
        col = LEVEL_COLOR.get(tgt["level"], KIND_COLOR[tgt["class_name"]])
        kind = GLYPH_KIND[tgt["class_name"]]
        theta = tgt["theta"]
        if tgt["blind"]:
            draw_blind(img, theta)
            gx, gy = to_px(theta, BLIND_M * 0.75)
            _glyph(img, kind, float(gx), float(gy) - 16, 0.52, RED)
            text.add(tgt.get("detail") or "사각지대", (float(gx), float(gy) + 26), 18,
                     RED, anchor="mm")
        elif tgt["distance"] is not None:
            cluster = _select_cluster_points(ground_filtered, theta, tgt["distance"],
                                             lidar.DEFAULT_ANGLE_MARGIN_DEG)
            cx, cy = draw_cluster(img, theta, tgt["distance"], col, cluster,
                                  pulse=(tgt["level"] == "경고"))
            _glyph(img, kind, cx, cy - 38, 0.52, col)
            left = cx < CX
            text.add(f"{tgt['distance']:.0f}m", (cx - 44 if left else cx + 44, cy + 4),
                     20, col, anchor="rm" if left else "lm")
        else:
            draw_wedge(img, theta, max(2.0 * tgt.get("sigma", 12.0), 16.0), col, ground_filtered)
            gx, gy = to_px(theta, 30.0)
            _glyph(img, kind, float(gx), float(gy), 0.58, col)
            if i == 0:
                text.add(tgt.get("detail") or "위치 미확정",
                         (float(gx), float(gy) + 40), 18, col, anchor="mm")

    draw_ego(img)

    # LiDAR 상태를 BEV 좌하단에 항상 명시 — 점이 안 보일 때 "조용한 도로"인지 "센서 없음"인지
    # 구분이 안 되면 화면이 거짓말을 하게 된다.
    if not lidar_ok:
        text.add("LiDAR 미연결 — 방향만 표시", (20, CANVAS_H - 30), 17, YELLOW)
    else:
        n = 0 if ground_filtered is None else int(round(ground_filtered.shape[0], -2))
        text.add(f"LiDAR ~{n:,} pts", (20, CANVAS_H - 30), 17, RING_TXT)

    draw_camera_pane(img, camera_frame_bgr, detection, text, camera_name,
                     targets[0]["level"] if targets else None)
    text.add(CAMERA_LABEL_KO.get(camera_name, camera_name),
             (SPLIT_X + 20, CANVAS_H - 30), 20, DIM)
    if link is not None and not link.get("connected", True):
        text.add("젯슨 연결 끊김 — 오디오 감지 없음", (SPLIT_X + 20, BANNER_H + 16), 20, RED)

    banner = None
    if targets:
        t0 = targets[0]
        banner = {"class_name": t0["class_name"], "loc": t0["loc"], "level": t0["level"],
                  "distance": t0["distance"], "blind": t0["blind"],
                  "detail": t0.get("detail")}
    draw_banner(img, banner, text)

    # 경고 테두리(ui_state_spec.md §5 "테두리 빛") — 정적 근사, 애니메이션은 생략
    if banner is not None and banner["level"] == "경고":
        cv2.rectangle(img, (3, 3), (CANVAS_W - 3, CANVAS_H - 3), RED, 6)

    text.flush(img)
    return img


# ---- 셀프테스트 ---------------------------------------------------------------------
def _fake_scene(theta_deg=None, distance_m=None, n_object=45):
    """가짜 스캔: 배경 리턴(건물·연석) + 지정 방향/거리의 물체 하나. 컬럼은 x,y,z,range,azimuth."""
    rng = np.random.default_rng(7)
    azi = rng.uniform(-180, 180, 900)
    dist = rng.uniform(8, 38, 900)
    z = rng.uniform(-0.2, 1.0, 900)
    if theta_deg is not None and distance_m is not None:
        azi = np.concatenate([azi, rng.normal(theta_deg, 3.0, n_object)])
        dist = np.concatenate([dist, rng.normal(distance_m, 0.35, n_object)])
        z = np.concatenate([z, rng.uniform(0.0, 0.9, n_object)])
    a = np.radians(azi)
    return np.column_stack([dist * np.cos(a), dist * np.sin(a), z, dist, azi]).astype(np.float32)


def run_selftest(outdir: Path):
    """하드웨어 없이 시나리오별 PNG를 만든다 — 레이아웃/각도 규약을 눈으로 확인하기 위한 것."""
    outdir.mkdir(parents=True, exist_ok=True)
    cam = np.full((1200, 2048, 3), 26, dtype=np.uint8)
    cv2.putText(cam, "CAMERA", (700, 640), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (70, 70, 70), 8)

    def tgt(cls, theta, level, dist=None, blind=False, sigma=12.0, loc="좌측", detail=None):
        return {"class_name": cls, "theta": theta, "level": level, "distance": dist,
                "blind": blind, "sigma": sigma, "loc": loc, "detail": detail}

    amb = {"label": "Ambulance", "conf": 0.92, "box": (760, 380, 1290, 860)}
    moto_det = {"label": "Motorcycle", "conf": 0.87, "box": (900, 500, 1330, 880)}

    # alert_policy.py의 클래스별 규칙(2026-09-02)을 그대로 반영한 시나리오
    scenes = [
        ("01_idle", "rear", [], _fake_scene(), True, None),
        ("02_horn_once", "right",
         [tgt("car_horn", -78, "주의", loc="우측", detail="방향 안내")],
         _fake_scene(), True, None),
        ("03_horn_repeat", "right",
         [tgt("car_horn", -82, "경고", loc="우측", detail="반복 2회")],
         _fake_scene(), True, None),
        ("04_siren_unmatched", "left",
         [tgt("siren", 94, "경고", detail="위치 미확정")], _fake_scene(), True, amb),
        ("05_siren_confirmed", "left",
         [tgt("siren", 94, "경고", 32.0, detail="차량 확정")], _fake_scene(94, 32.0), True, amb),
        ("06_moto_unknown", "left",
         [tgt("motorcycle", 104, "주의", sigma=16.0, detail="사각지대 위험")],
         _fake_scene(), True, None),
        ("07_moto_loud", "left",
         [tgt("motorcycle", 104, "경고", blind=True, detail="근접")],
         _fake_scene(), True, None),
        ("08_moto_tracked", "right",
         [tgt("motorcycle", -100, "경고", 18.0, loc="우측", detail="위치 추적")],
         _fake_scene(-100, 18.0), True, moto_det),
        ("09_multi", "left",
         [tgt("siren", 94, "경고", 28.0, detail="차량 확정"),
          tgt("motorcycle", -104, "주의", loc="우측", detail="사각지대 위험")],
         _fake_scene(94, 28.0), True, amb),
        ("10_lidar_down", "left",
         [tgt("siren", 94, "경고", detail="위치 미확정")], None, False, amb),
    ]

    for name, camera, targets, points, ok, det in scenes:
        img = render(cam, camera, targets, points, lidar_ok=ok, detection=det,
                     link={"connected": True})
        path = outdir / f"bev_{name}.png"
        cv2.imwrite(str(path), img)
        print(f"  {path}")
    print(f"\n[selftest] {len(scenes)}개 렌더링 완료 — 이미지 뷰어로 열어 확인하세요.")
    print("[selftest] 특히 확인할 것: 좌측 감지(theta=+94)가 화면 **왼쪽**에 그려지는지 "
          "(각도 규약 뒤집힘 검증)")
    return True


def main():
    parser = argparse.ArgumentParser(description="BEV|카메라 반반 화면 렌더러")
    parser.add_argument("--selftest", action="store_true", help="가짜 데이터로 시나리오 PNG 생성")
    parser.add_argument("--outdir", default="./bev_selftest", help="PNG 출력 폴더")
    args = parser.parse_args()
    if args.selftest:
        run_selftest(Path(args.outdir))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
