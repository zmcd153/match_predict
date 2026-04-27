import os
import cv2
import math
import json
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

IMAGE_PATH = "match_frame.jpg"
OUTPUT_PATH = "match_position_predict_v2.png"
DEBUG_MASK_PATH = "debug_team_masks.png"

ATTACKING_TEAM = "blue"   # 取值："red" 或 "blue"
ATTACK_DIRECTION = "left" # 取值："left" 或 "right"

# 可选的时序平滑缓存（若你按视频逐帧运行该脚本会有帮助）
USE_HISTORY_SMOOTHING = False
HISTORY_JSON_PATH = "player_history_cache.json"


@dataclass
class Detection:
    team: str
    point: Tuple[float, float]      # 球员在场地上的锚点：bbox 底部中心
    center: Tuple[float, float]     # bbox 的视觉中心
    bbox: Tuple[int, int, int, int] # 边界框：(x, y, w, h)
    area: int
    confidence: float
    noise_score: float


# ---------------------------
# 通用辅助函数
# ---------------------------

def clip_pt(x, y, w, h):
    x = int(max(0, min(w - 1, round(x))))
    y = int(max(0, min(h - 1, round(y))))
    return (x, y)


def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.zeros_like(v, dtype=np.float32)
    return v / n


def smoothstep(x, lo=0.0, hi=1.0):
    if hi <= lo:
        return 0.0
    t = max(0.0, min(1.0, (x - lo) / (hi - lo)))
    return t * t * (3.0 - 2.0 * t)


def nearest_point(p, others):
    if not others:
        return None
    arr = np.asarray(others, dtype=np.float32)
    p0 = np.asarray(p, dtype=np.float32)
    dist = ((arr - p0) ** 2).sum(axis=1)
    return tuple(arr[np.argmin(dist)])


def k_nearest_points(p, others, k=3):
    if not others:
        return []
    arr = np.asarray(others, dtype=np.float32)
    p0 = np.asarray(p, dtype=np.float32)
    dist = ((arr - p0) ** 2).sum(axis=1)
    idx = np.argsort(dist)[:k]
    return [tuple(arr[i]) for i in idx]


def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_len=8):
    dist = int(math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
    if dist <= 0:
        return
    for i in range(0, dist, dash_len * 2):
        r1 = i / max(dist, 1)
        r2 = min((i + dash_len) / max(dist, 1), 1.0)
        x1 = int(pt1[0] + (pt2[0] - pt1[0]) * r1)
        y1 = int(pt1[1] + (pt2[1] - pt1[1]) * r1)
        x2 = int(pt1[0] + (pt2[0] - pt1[0]) * r2)
        y2 = int(pt1[1] + (pt2[1] - pt1[1]) * r2)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)


# ---------------------------
# 球场分割
# ---------------------------

def extract_pitch_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    # 转播画面中常见的草坪绿色范围。
    green1 = cv2.inRange(hsv, (25, 20, 25), (95, 255, 255))

    # 抑制顶部转播信息条，以及上下明显非球场区域。
    roi = np.zeros((h, w), dtype=np.uint8)
    y0 = int(h * 0.14)
    y1 = int(h * 0.93)
    roi[y0:y1, :] = 255
    mask = cv2.bitwise_and(green1, roi)

    # 形态学处理：填补球场空洞并去除小噪声。
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels <= 1:
        return mask

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    pitch_mask = (labels == largest).astype(np.uint8) * 255
    pitch_mask = cv2.dilate(pitch_mask, np.ones((7, 7), np.uint8), iterations=1)
    return pitch_mask


# ---------------------------
# 红蓝两队分割与去噪
# ---------------------------

def build_team_masks(img, pitch_mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 红色通常会分布在两个色相区间。
    red1 = cv2.inRange(hsv, np.array([0, 65, 45]), np.array([14, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([168, 65, 45]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red1, red2)

    # 蓝色范围。对饱和度/亮度稍微设得更严格，以减少深蓝观众席/阴影噪声。
    blue_mask = cv2.inRange(hsv, np.array([96, 70, 40]), np.array([138, 255, 255]))

    # 仅保留球场区域内的像素。
    red_mask = cv2.bitwise_and(red_mask, pitch_mask)
    blue_mask = cv2.bitwise_and(blue_mask, pitch_mask)

    # 按队伍分别做形态学处理。
    # 红队额外做一点开运算，抑制 LED 广告牌或球袜碎片。
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # 蓝队在阴影下容易断裂，因此闭运算稍强一些。
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, np.ones((7, 5), np.uint8))

    return red_mask, blue_mask


def component_team_confidence(team, comp_mask, bbox, hsv_roi, img_h, img_w):
    x, y, w, h = bbox
    ys, xs = np.where(comp_mask > 0)
    area = int(comp_mask.sum() // 255)
    if area <= 0 or w <= 0 or h <= 0:
        return 0.0, 1.0

    fill = area / max(w * h, 1)
    aspect = h / max(w, 1)
    bottom_y = y + h

    pixels = hsv_roi[ys, xs]
    if len(pixels) == 0:
        return 0.0, 1.0

    hue = float(np.mean(pixels[:, 0]))
    sat = float(np.mean(pixels[:, 1]))
    val = float(np.mean(pixels[:, 2]))

    # 基础几何置信度。
    conf_geo = 0.35 * smoothstep(area, 18, 170) + 0.35 * smoothstep(aspect, 1.15, 3.5) + 0.30 * smoothstep(fill, 0.18, 0.78)

    # 位置先验：球员通常不会出现在最顶部或最底部边缘。
    pos_penalty = 0.0
    if y < int(0.20 * img_h):
        pos_penalty += 0.15
    if bottom_y > int(0.96 * img_h):
        pos_penalty += 0.10
    if x < 5 or (x + w) > img_w - 5:
        pos_penalty += 0.08

    # 队伍颜色置信度。
    if team == "red":
        red_hue_good = (0 <= hue <= 12) or (168 <= hue <= 180)
        conf_color = 0.55 if red_hue_good else 0.20
        conf_color += 0.20 * smoothstep(sat, 70, 180)
        conf_color += 0.10 * smoothstep(val, 40, 180)

        # 排除偏橙/偏黄的串色。
        if 12 < hue < 25:
            pos_penalty += 0.18

        # 上部区域里又宽又矮的连通域通常是广告牌或转播覆盖层。
        if w > 18 and h < 16:
            pos_penalty += 0.20
    else:
        blue_hue_good = 96 <= hue <= 138
        conf_color = 0.55 if blue_hue_good else 0.18
        conf_color += 0.20 * smoothstep(sat, 85, 190)
        conf_color += 0.10 * smoothstep(val, 35, 165)

        # 低饱和的深色块通常是阴影噪声，而不是蓝色球衣。
        if sat < 85 and val < 80:
            pos_penalty += 0.22

        if w > 20 and h < 18:
            pos_penalty += 0.18

    conf = np.clip(0.55 * conf_geo + 0.45 * conf_color - pos_penalty, 0.0, 1.0)
    noise = float(np.clip(1.0 - conf + max(0.0, 0.28 - fill), 0.0, 1.0))
    return float(conf), noise


def extract_team_detections(img, team_mask, team_name):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_h, img_w = img.shape[:2]
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(team_mask, 8)
    detections: List[Detection] = []

    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 12 or area > 380:
            continue
        if w < 3 or h < 7 or w > 28 or h > 48:
            continue

        comp = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        hsv_roi = hsv[y:y + h, x:x + w]
        conf, noise = component_team_confidence(team_name, comp, (x, y, w, h), hsv_roi, img_h, img_w)
        if conf < 0.40:
            continue

        cx, cy = centroids[i]

        # 用框底部中心作为更接近球员落脚点的锚点，而不是简单用连通域质心。
        foot_x = float(x + w / 2.0)
        foot_y = float(y + h * 0.95)

        detections.append(
            Detection(
                team=team_name,
                point=(foot_x, foot_y),
                center=(float(cx), float(cy)),
                bbox=(int(x), int(y), int(w), int(h)),
                area=int(area),
                confidence=float(conf),
                noise_score=float(noise),
            )
        )

    return detections


def merge_close_detections(detections: List[Detection], min_dist=16.0):
    if not detections:
        return []

    dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    used = [False] * len(dets)
    merged = []

    for i, d in enumerate(dets):
        if used[i]:
            continue
        cluster = [d]
        used[i] = True
        p0 = np.array(d.point, dtype=np.float32)

        for j in range(i + 1, len(dets)):
            if used[j]:
                continue
            p1 = np.array(dets[j].point, dtype=np.float32)
            if np.linalg.norm(p1 - p0) < min_dist:
                cluster.append(dets[j])
                used[j] = True

        ws = np.array([max(c.confidence, 1e-3) for c in cluster], dtype=np.float32)
        pts = np.array([c.point for c in cluster], dtype=np.float32)
        ctr = np.array([c.center for c in cluster], dtype=np.float32)
        bbox = cluster[int(np.argmax(ws))].bbox
        merged.append(
            Detection(
                team=d.team,
                point=tuple((pts * ws[:, None]).sum(axis=0) / ws.sum()),
                center=tuple((ctr * ws[:, None]).sum(axis=0) / ws.sum()),
                bbox=bbox,
                area=int(sum(c.area for c in cluster)),
                confidence=float(np.max(ws)),
                noise_score=float(np.mean([c.noise_score for c in cluster])),
            )
        )

    return merged


def temporal_smooth_points(team_name, detections, max_match_dist=28.0, alpha=0.72):
    if not USE_HISTORY_SMOOTHING:
        return detections

    if os.path.exists(HISTORY_JSON_PATH):
        try:
            with open(HISTORY_JSON_PATH, "r", encoding="utf-8") as f:
                hist = json.load(f)
        except Exception:
            hist = {}
    else:
        hist = {}

    prev = hist.get(team_name, [])
    if not prev:
        hist[team_name] = [{"x": d.point[0], "y": d.point[1]} for d in detections]
        with open(HISTORY_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)
        return detections

    prev_pts = np.array([[p["x"], p["y"]] for p in prev], dtype=np.float32)
    new_dets = []
    used_prev = set()

    for d in detections:
        p = np.array(d.point, dtype=np.float32)
        dist = np.linalg.norm(prev_pts - p[None, :], axis=1)
        j = int(np.argmin(dist))
        if dist[j] < max_match_dist and j not in used_prev:
            smoothed = alpha * p + (1.0 - alpha) * prev_pts[j]
            used_prev.add(j)
            d = Detection(
                team=d.team,
                point=(float(smoothed[0]), float(smoothed[1])),
                center=d.center,
                bbox=d.bbox,
                area=d.area,
                confidence=d.confidence,
                noise_score=max(0.0, d.noise_score * 0.9),
            )
        new_dets.append(d)

    hist[team_name] = [{"x": d.point[0], "y": d.point[1]} for d in new_dets]
    with open(HISTORY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)
    return new_dets


# ---------------------------
# 参考 FootBots / TranSPORTmer 的预测器
# ---------------------------

def estimate_pseudo_ball(attackers, defenders, w, h, attack_direction):
    if not attackers:
        return np.array([w * 0.5, h * 0.55], dtype=np.float32)

    pts = np.array(attackers, dtype=np.float32)
    if attack_direction == "left":
        # 选择在左侧更靠前的进攻球员。
        idx = np.argmin(pts[:, 0] + 0.18 * np.abs(pts[:, 1] - h * 0.55))
    else:
        idx = np.argmax(pts[:, 0] - 0.18 * np.abs(pts[:, 1] - h * 0.55))

    carrier = pts[idx]
    team_centroid = pts.mean(axis=0)
    return 0.7 * carrier + 0.3 * team_centroid


def local_team_spacing_vector(p, teammates):
    neigh = [q for q in k_nearest_points(p, teammates, 3) if np.linalg.norm(np.array(q) - np.array(p)) > 1.0]
    if not neigh:
        return np.zeros(2, dtype=np.float32)
    p0 = np.array(p, dtype=np.float32)
    v = np.zeros(2, dtype=np.float32)
    for q in neigh:
        q0 = np.array(q, dtype=np.float32)
        d = p0 - q0
        n = np.linalg.norm(d)
        if n < 1e-4:
            continue
        v += d / (n * n)
    return unit(v)


def opponent_pressure_vector(p, opponents):
    neigh = [q for q in k_nearest_points(p, opponents, 3)]
    if not neigh:
        return np.zeros(2, dtype=np.float32)
    p0 = np.array(p, dtype=np.float32)
    v = np.zeros(2, dtype=np.float32)
    for q in neigh:
        q0 = np.array(q, dtype=np.float32)
        d = p0 - q0
        n = np.linalg.norm(d)
        if n < 1e-4:
            continue
        v += d / (n * n)
    return unit(v)


def defensive_line_x(defenders):
    if not defenders:
        return None
    xs = sorted([p[0] for p in defenders])
    return float(np.median(xs))


def build_attack_velocity(p, attackers, defenders, pseudo_ball, w, h, attack_direction):
    p0 = np.array(p, dtype=np.float32)

    goal = np.array([10.0, h * 0.55], dtype=np.float32) if attack_direction == "left" else np.array([w - 10.0, h * 0.55], dtype=np.float32)
    direct_goal = unit(goal - p0)
    central_bias = unit(np.array([0.0, h * 0.55], dtype=np.float32) - p0)
    spacing = local_team_spacing_vector(p, attackers)
    pressure_away = opponent_pressure_vector(p, defenders)
    ball_link = unit(np.array(pseudo_ball, dtype=np.float32) - p0)

    # 对条件信息 / 社会交互进行近似聚合。
    v = (
        0.46 * direct_goal +
        0.16 * central_bias +
        0.14 * spacing +
        0.14 * pressure_away +
        0.10 * ball_link
    )

    # 越位线 / 防线感知的近似项。
    dline = defensive_line_x(defenders)
    if dline is not None:
        if attack_direction == "left":
            penetration = max(0.0, dline - p0[0])
        else:
            penetration = max(0.0, p0[0] - dline)
        if penetration > 0:
            v += 0.12 * central_bias

    return unit(v)


def build_defend_velocity(p, attackers, defenders, pseudo_ball, w, h, attack_direction):
    p0 = np.array(p, dtype=np.float32)
    own_goal = np.array([8.0, h * 0.56], dtype=np.float32) if attack_direction == "left" else np.array([w - 8.0, h * 0.56], dtype=np.float32)

    near_att = nearest_point(p, attackers)
    near_att = np.array(near_att, dtype=np.float32) if near_att is not None else np.array(pseudo_ball, dtype=np.float32)

    mark = unit(near_att - p0)
    goal_cover = unit(own_goal - p0)
    ball_compact = unit(np.array(pseudo_ball, dtype=np.float32) - p0)

    # 保持防线在水平方向上的紧凑性。
    dpts = np.array(defenders, dtype=np.float32) if defenders else np.array([[p0[0], p0[1]]], dtype=np.float32)
    line_center = dpts.mean(axis=0)
    line_keep = unit(np.array([line_center[0], p0[1]], dtype=np.float32) - p0)

    v = 0.34 * mark + 0.30 * goal_cover + 0.22 * ball_compact + 0.14 * line_keep
    return unit(v)


def make_path(p, v, base_step, uncertainty, w, h):
    p1 = np.array(p, dtype=np.float32)
    # 不确定性越高，可置信的预测长度越短。
    s = base_step * (1.0 - 0.35 * uncertainty)
    p2 = p1 + v * (s * 0.55)
    p3 = p1 + v * s
    return [clip_pt(p1[0], p1[1], w, h), clip_pt(p2[0], p2[1], w, h), clip_pt(p3[0], p3[1], w, h)]


def uncertainty_from_detection(det: Detection, teammates, opponents, w):
    # 参考 TranSPORTmer 的思路：对噪声区域和被遮挡区域周围建模不确定性。
    p = np.array(det.point, dtype=np.float32)
    crowding = 0.0
    opp = k_nearest_points(det.point, opponents, 2)
    for q in opp:
        d = np.linalg.norm(np.array(q, dtype=np.float32) - p)
        crowding += max(0.0, 1.0 - d / max(w * 0.10, 1.0))
    crowding = min(1.0, crowding)
    return float(np.clip(0.55 * det.noise_score + 0.25 * (1.0 - det.confidence) + 0.20 * crowding, 0.0, 1.0))


# ---------------------------
# 可视化
# ---------------------------

def draw_detection(canvas, det: Detection, color_good, color_box):
    x, y, w, h = det.bbox
    px, py = int(det.point[0]), int(det.point[1])
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color_box, 1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (px, py), 7, color_good, 2, lineType=cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"{det.confidence:.2f}",
        (x, max(10, y - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        color_box,
        1,
        lineType=cv2.LINE_AA,
    )


def draw_uncertainty_fan(canvas, origin, dest, uncertainty, color):
    ox, oy = origin
    dx, dy = dest
    v = np.array([dx - ox, dy - oy], dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-4:
        return
    v = v / n
    perp = np.array([-v[1], v[0]], dtype=np.float32)
    spread = int(4 + 14 * uncertainty)
    p1 = (int(dx + perp[0] * spread), int(dy + perp[1] * spread))
    p2 = (int(dx - perp[0] * spread), int(dy - perp[1] * spread))
    overlay = canvas.copy()
    cv2.fillConvexPoly(overlay, np.array([origin, p1, p2], dtype=np.int32), color)
    cv2.addWeighted(overlay, 0.14, canvas, 0.86, 0, canvas)


def visualize(img, red_dets, blue_dets, attacking_team, attack_direction):
    canvas = img.copy()
    h, w = img.shape[:2]

    for d in red_dets:
        draw_detection(canvas, d, (70, 255, 70), (90, 255, 120))
    for d in blue_dets:
        draw_detection(canvas, d, (0, 255, 255), (255, 220, 80))

    if attacking_team == "blue":
        attackers = blue_dets
        defenders = red_dets
        atk_color = (80, 80, 255)
        def_color = (255, 170, 80)
    else:
        attackers = red_dets
        defenders = blue_dets
        atk_color = (80, 80, 255)
        def_color = (255, 170, 80)

    attacker_pts = [d.point for d in attackers]
    defender_pts = [d.point for d in defenders]
    pseudo_ball = estimate_pseudo_ball(attacker_pts, defender_pts, w, h, attack_direction)
    bx, by = clip_pt(pseudo_ball[0], pseudo_ball[1], w, h)
    cv2.circle(canvas, (bx, by), 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (bx, by), 10, (40, 40, 40), 1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, "pseudo-ball", (bx + 8, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1, lineType=cv2.LINE_AA)

    for d in attackers:
        u = uncertainty_from_detection(d, attacker_pts, defender_pts, w)
        v = build_attack_velocity(d.point, attacker_pts, defender_pts, pseudo_ball, w, h, attack_direction)
        progress = (w - d.point[0]) / max(w, 1) if attack_direction == "left" else d.point[0] / max(w, 1)
        base_step = 14 + 15 * float(np.clip(progress, 0.0, 1.0))
        pts = make_path(d.point, v, base_step, u, w, h)
        draw_uncertainty_fan(canvas, pts[0], pts[-1], u, atk_color)
        for a, b in zip(pts[:-1], pts[1:]):
            draw_dashed_line(canvas, a, b, atk_color, thickness=2, dash_len=6)
        cv2.arrowedLine(canvas, pts[-2], pts[-1], atk_color, 2, tipLength=0.25)

    for d in defenders:
        u = uncertainty_from_detection(d, defender_pts, attacker_pts, w)
        v = build_defend_velocity(d.point, attacker_pts, defender_pts, pseudo_ball, w, h, attack_direction)
        base_step = 12.5
        pts = make_path(d.point, v, base_step, u, w, h)
        draw_uncertainty_fan(canvas, pts[0], pts[-1], u, def_color)
        for a, b in zip(pts[:-1], pts[1:]):
            draw_dashed_line(canvas, a, b, def_color, thickness=2, dash_len=6)
        cv2.arrowedLine(canvas, pts[-2], pts[-1], def_color, 2, tipLength=0.25)

    red_noise = np.mean([d.noise_score for d in red_dets]) if red_dets else 0.0
    blue_noise = np.mean([d.noise_score for d in blue_dets]) if blue_dets else 0.0

    cv2.rectangle(canvas, (8, 8), (390, 116), (18, 18, 18), -1)
    cv2.putText(canvas, "Circle: player anchor (bottom-center of bbox)", (16, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (230, 230, 230), 1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, "White dot: pseudo-ball / conditioned center", (16, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (230, 230, 230), 1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, "Red path: attack  Orange path: defend", (16, 63),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (230, 230, 230), 1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, f"Detected red={len(red_dets)}  blue={len(blue_dets)}", (16, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, f"Noise score red={red_noise:.2f}  blue={blue_noise:.2f}", (16, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, lineType=cv2.LINE_AA)

    return canvas


def build_debug_mask_view(img, pitch_mask, red_mask, blue_mask):
    h, w = img.shape[:2]
    out = np.zeros((h, w * 3, 3), dtype=np.uint8)
    out[:, :w] = img
    out[:, w:2 * w, 1] = pitch_mask
    out[:, 2 * w:3 * w, 2] = red_mask
    out[:, 2 * w:3 * w, 0] = blue_mask
    cv2.putText(out, "original", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(out, "pitch_mask", (w + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(out, "team_masks (R/B)", (2 * w + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    return out


# ---------------------------
# 主流程
# ---------------------------

def main():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"找不到图片: {IMAGE_PATH}")

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise RuntimeError("图片读取失败")

    pitch_mask = extract_pitch_mask(img)
    red_mask, blue_mask = build_team_masks(img, pitch_mask)

    red_dets = extract_team_detections(img, red_mask, "red")
    blue_dets = extract_team_detections(img, blue_mask, "blue")

    red_dets = merge_close_detections(red_dets, min_dist=16.0)
    blue_dets = merge_close_detections(blue_dets, min_dist=16.0)

    red_dets = temporal_smooth_points("red", red_dets)
    blue_dets = temporal_smooth_points("blue", blue_dets)

    print("识别结果：")
    print("red players:", len(red_dets), "avg conf:", round(np.mean([d.confidence for d in red_dets]) if red_dets else 0.0, 3))
    print("blue players:", len(blue_dets), "avg conf:", round(np.mean([d.confidence for d in blue_dets]) if blue_dets else 0.0, 3))
    print("red noise:", round(np.mean([d.noise_score for d in red_dets]) if red_dets else 0.0, 3))
    print("blue noise:", round(np.mean([d.noise_score for d in blue_dets]) if blue_dets else 0.0, 3))

    result = visualize(img, red_dets, blue_dets, ATTACKING_TEAM, ATTACK_DIRECTION)
    debug = build_debug_mask_view(img, pitch_mask, red_mask, blue_mask)

    ok1 = cv2.imwrite(OUTPUT_PATH, result)
    ok2 = cv2.imwrite(DEBUG_MASK_PATH, debug)
    if not (ok1 and ok2):
        raise RuntimeError("图片保存失败")

    print("结果已保存:", OUTPUT_PATH)
    print("调试图已保存:", DEBUG_MASK_PATH)


if __name__ == "__main__":
    main()
