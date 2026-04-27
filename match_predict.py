import os
import cv2
import math
import numpy as np

IMAGE_PATH = "match_frame.jpg"
OUTPUT_PATH = "match_position_predict.png"

# 这张图里示例设为蓝队进攻、向左推进
ATTACKING_TEAM = "blue"
ATTACK_DIRECTION = "left"


def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_len=8):
    dist = int(math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
    if dist <= 0:
        return

    for i in range(0, dist, dash_len * 2):
        r1 = i / dist
        r2 = min((i + dash_len) / dist, 1.0)

        x1 = int(pt1[0] + (pt2[0] - pt1[0]) * r1)
        y1 = int(pt1[1] + (pt2[1] - pt1[1]) * r1)
        x2 = int(pt1[0] + (pt2[0] - pt1[0]) * r2)
        y2 = int(pt1[1] + (pt2[1] - pt1[1]) * r2)

        cv2.line(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)


def clip_pt(x, y, w, h):
    x = int(max(0, min(w - 1, round(x))))
    y = int(max(0, min(h - 1, round(y))))
    return (x, y)


def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.array([0.0, 0.0], dtype=np.float32)
    return v / n


def nearest_point(p, others):
    if not others:
        return None

    arr = np.array(others, dtype=np.float32)
    p0 = np.array(p, dtype=np.float32)
    dist = ((arr - p0) ** 2).sum(axis=1)
    return tuple(arr[np.argmin(dist)])


def extract_pitch_mask(img):
    """
    提取球场草坪区域
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    green = cv2.inRange(hsv, (25, 30, 30), (95, 255, 255))

    # 限制到比赛主要区域，排除顶部记分牌和底部黑边
    roi = np.zeros_like(green)
    roi[45:270, :] = 255
    field = cv2.bitwise_and(green, roi)

    kernel = np.ones((5, 5), np.uint8)
    field = cv2.morphologyEx(field, cv2.MORPH_OPEN, kernel)
    field = cv2.morphologyEx(field, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(field, 8)
    if num_labels <= 1:
        return field

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    pitch_mask = (labels == largest).astype(np.uint8) * 255
    pitch_mask = cv2.dilate(pitch_mask, np.ones((9, 9), np.uint8), iterations=1)
    return pitch_mask


def extract_players_by_color(img, pitch_mask):
    """
    按球衣颜色提取红队和蓝队球员
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 红色
    red1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([12, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_and(red1 | red2, pitch_mask)

    # 蓝色
    blue_mask = cv2.bitwise_and(
        cv2.inRange(hsv, np.array([95, 60, 40]), np.array([140, 255, 255])),
        pitch_mask
    )

    def extract_points(mask, img_h):
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        points = []

        for i in range(1, num):
            x, y, w, h, area = stats[i]

            # 过滤噪声和广告牌误检
            if not (18 <= area <= 240 and 3 <= w <= 22 and 8 <= h <= 35):
                continue

            ratio = h / max(w, 1)
            if ratio < 1.15:
                continue

            # 排除太靠上和太靠下的区域
            if y < 75 or y > img_h - 30:
                continue

            if x < 10:
                continue

            cx, cy = centroids[i]
            points.append((float(cx), float(cy)))

        return points

    red_points = extract_points(red_mask, img.shape[0])
    blue_points = extract_points(blue_mask, img.shape[0])

    return red_points, blue_points


def dedup_points(points, min_dist=14):
    """
    去掉太近的重复点
    """
    kept = []
    for p in sorted(points, key=lambda z: (z[0], z[1])):
        if not kept:
            kept.append(p)
            continue

        ok = True
        for q in kept:
            if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < min_dist ** 2:
                ok = False
                break

        if ok:
            kept.append(p)

    return kept


def attack_path(p, w, h, attack_direction):
    """
    进攻队跑位：
    向球门方向推进，同时边路略微内收
    """
    p0 = np.array(p, dtype=np.float32)

    if attack_direction == "left":
        base = np.array([-1.0, 0.0], dtype=np.float32)
        center_bias = np.array([-0.15, (h * 0.55 - p0[1]) / max(h, 1)], dtype=np.float32)
        progress = (w - p0[0]) / max(w, 1)
    else:
        base = np.array([1.0, 0.0], dtype=np.float32)
        center_bias = np.array([0.15, (h * 0.55 - p0[1]) / max(h, 1)], dtype=np.float32)
        progress = p0[0] / max(w, 1)

    v = unit(base + 0.8 * center_bias)
    step = 16 + 18 * progress

    p1 = p0
    p2 = p1 + v * (step * 0.55)
    p3 = p1 + v * step
    return [p1, p2, p3]


def defend_path(p, attackers, w, h, attack_direction):
    """
    防守队跑位：
    向最近进攻球员和球门方向收缩
    """
    p0 = np.array(p, dtype=np.float32)

    if attack_direction == "left":
        goal = np.array([8.0, h * 0.56], dtype=np.float32)
    else:
        goal = np.array([w - 8.0, h * 0.56], dtype=np.float32)

    near = nearest_point(p, attackers)

    if near is None:
        target = goal
    else:
        near = np.array(near, dtype=np.float32)
        target = 0.65 * near + 0.35 * goal

    v = unit(target - p0)
    step = 14

    p1 = p0
    p2 = p1 + v * (step * 0.55)
    p3 = p1 + v * step
    return [p1, p2, p3]


def visualize(img, red_points, blue_points, attacking_team, attack_direction):
    canvas = img.copy()
    h, w = img.shape[:2]

    # 站位圈
    for p in red_points:
        x, y = clip_pt(p[0], p[1], w, h)
        cv2.circle(canvas, (x, y), 10, (60, 255, 60), 2, lineType=cv2.LINE_AA)

    for p in blue_points:
        x, y = clip_pt(p[0], p[1], w, h)
        cv2.circle(canvas, (x, y), 10, (0, 255, 255), 2, lineType=cv2.LINE_AA)

    if attacking_team == "blue":
        attackers = blue_points
        defenders = red_points
        atk_color = (80, 80, 255)
        def_color = (255, 170, 80)
    else:
        attackers = red_points
        defenders = blue_points
        atk_color = (80, 80, 255)
        def_color = (255, 170, 80)

    # 进攻轨迹
    for p in attackers:
        pts = attack_path(p, w, h, attack_direction)
        pts = [clip_pt(x, y, w, h) for x, y in pts]

        for a, b in zip(pts[:-1], pts[1:]):
            draw_dashed_line(canvas, a, b, atk_color, thickness=2, dash_len=6)
        cv2.arrowedLine(canvas, pts[-2], pts[-1], atk_color, 2, tipLength=0.25)

    # 防守轨迹
    for p in defenders:
        pts = defend_path(p, attackers, w, h, attack_direction)
        pts = [clip_pt(x, y, w, h) for x, y in pts]

        for a, b in zip(pts[:-1], pts[1:]):
            draw_dashed_line(canvas, a, b, def_color, thickness=2, dash_len=6)
        cv2.arrowedLine(canvas, pts[-2], pts[-1], def_color, 2, tipLength=0.25)

    # 图例
    cv2.rectangle(canvas, (8, 8), (290, 92), (20, 20, 20), -1)
    cv2.putText(canvas, "Circle: detected standing position", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, "Orange: defensive shift", (16, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, def_color, 1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, "Red: attacking move", (16, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, atk_color, 1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, f"Detected red={len(red_points)}  blue={len(blue_points)}", (16, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, lineType=cv2.LINE_AA)

    return canvas


def main():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"找不到图片: {IMAGE_PATH}")

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise RuntimeError("图片读取失败")

    pitch_mask = extract_pitch_mask(img)
    red_points, blue_points = extract_players_by_color(img, pitch_mask)

    red_points = dedup_points(red_points, 14)
    blue_points = dedup_points(blue_points, 14)

    print("识别结果：")
    print("red players:", len(red_points))
    print("blue players:", len(blue_points))

    result = visualize(img, red_points, blue_points, ATTACKING_TEAM, ATTACK_DIRECTION)

    ok = cv2.imwrite(OUTPUT_PATH, result)
    if not ok:
        raise RuntimeError("图片保存失败")

    print("结果已保存:", OUTPUT_PATH)


if __name__ == "__main__":
    main()