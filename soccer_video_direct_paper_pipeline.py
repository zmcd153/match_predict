import os
import math
from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------------------------
# 运行参数
# ---------------------------
INPUT_VIDEO_PATH = "match_video.mp4"
OUTPUT_VIDEO_PATH = "match_video_predict_paper_style.mp4"
ATTACK_DIRECTION = "right"   # 取值: "left" 或 "right"
ATTACKING_TEAM_MODE = "auto"  # 取值: "auto" 或 "manual"
MANUAL_ATTACKING_TEAM = 0      # 仅当 ATTACKING_TEAM_MODE="manual" 时生效，取值 0 或 1

PREDICT_HORIZON_STEPS = 8      # 每帧可视化的未来步数
HISTORY_SIZE = 10              # 轨迹历史长度
WARMUP_FRAMES = 18             # 前若干帧用于让背景建模稳定
MIN_TRACK_HITS = 3             # 轨迹至少命中这么多次才参与预测
MAX_MISSING_FRAMES = 10        # 轨迹允许连续丢失的最大帧数
MAX_MATCH_DISTANCE = 42.0      # 跟踪匹配的最大距离阈值（像素）

# 显示颜色仅用于绘图，不代表球衣真实颜色
TEAM_DRAW_COLORS = [
    (255, 180, 60),
    (80, 220, 255),
]
UNKNOWN_DRAW_COLOR = (180, 180, 180)


@dataclass
class Detection:
    """单帧检测结果。"""

    point: Tuple[float, float]                # 球员脚下锚点（bbox 底部中心）
    center: Tuple[float, float]               # 目标几何中心
    bbox: Tuple[int, int, int, int]           # x, y, w, h
    jersey_feat: np.ndarray                   # 球衣颜色特征
    confidence: float                         # 检测置信度（启发式）
    team_id: Optional[int] = None             # 自动分队结果，0/1/None


@dataclass
class Track:
    """跨帧轨迹。"""

    track_id: int
    point: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    jersey_feat: np.ndarray
    confidence: float
    team_id: Optional[int] = None
    history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=HISTORY_SIZE))
    age: int = 1
    hits: int = 1
    missing: int = 0

    def __post_init__(self):
        if len(self.history) == 0:
            self.history.append(self.point)

    @property
    def confirmed(self) -> bool:
        """是否是较稳定轨迹。"""
        return self.hits >= MIN_TRACK_HITS and self.missing <= 1

    def velocity(self) -> np.ndarray:
        """根据历史点估计当前速度。"""
        if len(self.history) < 2:
            return np.zeros(2, dtype=np.float32)
        p1 = np.array(self.history[-1], dtype=np.float32)
        p0 = np.array(self.history[-2], dtype=np.float32)
        return p1 - p0

    def smooth_velocity(self) -> np.ndarray:
        """使用最近几步做平滑速度，减少抖动。"""
        if len(self.history) < 2:
            return np.zeros(2, dtype=np.float32)
        pts = np.array(self.history, dtype=np.float32)
        diffs = pts[1:] - pts[:-1]
        if len(diffs) == 0:
            return np.zeros(2, dtype=np.float32)
        weights = np.linspace(0.6, 1.0, len(diffs), dtype=np.float32)
        v = (diffs * weights[:, None]).sum(axis=0) / max(weights.sum(), 1e-6)
        return v.astype(np.float32)


class OnlineTeamClustering:
    """
    自动分队器。

    做法：
    1) 从每个球员候选框的上半身区域抽取 HSV 颜色特征。
    2) 对当前帧候选做 2 类聚类。
    3) 与上一帧的队伍颜色中心做匹配，尽量保持 Team0 / Team1 不乱跳。
    """

    def __init__(self):
        self.team_centers: Optional[np.ndarray] = None

    @staticmethod
    def _feature_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def assign(self, detections: List[Detection]) -> List[Detection]:
        """对当前帧检测结果自动分成两队。"""
        if len(detections) < 4:
            return detections

        feats = np.array([d.jersey_feat for d in detections], dtype=np.float32)
        if len(feats) < 2:
            return detections

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
        compactness, labels, centers = cv2.kmeans(
            feats,
            2,
            None,
            criteria,
            3,
            cv2.KMEANS_PP_CENTERS,
        )
        labels = labels.reshape(-1)

        if self.team_centers is not None and len(self.team_centers) == 2:
            cost = np.zeros((2, 2), dtype=np.float32)
            for i in range(2):
                for j in range(2):
                    cost[i, j] = self._feature_distance(centers[i], self.team_centers[j])
            row_ind, col_ind = linear_sum_assignment(cost)
            mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
            mapped_labels = np.array([mapping[int(x)] for x in labels], dtype=np.int32)
            new_centers = np.zeros_like(centers)
            for k_src, k_dst in mapping.items():
                new_centers[k_dst] = centers[k_src]
            centers = new_centers
            labels = mapped_labels

        for det, team_id in zip(detections, labels):
            det.team_id = int(team_id)

        if self.team_centers is None:
            self.team_centers = centers.copy()
        else:
            self.team_centers = 0.85 * self.team_centers + 0.15 * centers

        return detections


class VideoSoccerTracker:
    """
    从比赛视频中检测球员、跟踪球员并自动分队。

    1) 先在球场区域中找运动目标。
    2) 再从目标框上半身提取球衣颜色。
    3) 用颜色聚类自动分两队。
    4) 用匈牙利匹配把当前检测与历史轨迹关联起来。
    """

    def __init__(self):
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(history=600, varThreshold=24, detectShadows=False)
        self.team_cluster = OnlineTeamClustering()
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1

    @staticmethod
    def clip_point(x: float, y: float, w: int, h: int) -> Tuple[int, int]:
        x = int(max(0, min(w - 1, round(x))))
        y = int(max(0, min(h - 1, round(y))))
        return x, y

    @staticmethod
    def unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        if n < 1e-6:
            return np.zeros_like(v, dtype=np.float32)
        return (v / n).astype(np.float32)

    def extract_pitch_mask(self, frame: np.ndarray) -> np.ndarray:
        """提取草坪区域，排除顶部记分牌、底部边框和看台。"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]

        green = cv2.inRange(hsv, (25, 20, 20), (95, 255, 255))
        roi = np.zeros((h, w), dtype=np.uint8)
        roi[int(h * 0.12):int(h * 0.95), :] = 255
        mask = cv2.bitwise_and(green, roi)

        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if num_labels <= 1:
            return mask
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        pitch = (labels == largest).astype(np.uint8) * 255
        pitch = cv2.dilate(pitch, np.ones((7, 7), np.uint8), iterations=1)
        return pitch

    def extract_player_candidates(self, frame: np.ndarray, pitch_mask: np.ndarray, frame_index: int) -> List[Detection]:
        """
        从视频帧中提取球员候选。

        """
        h, w = frame.shape[:2]

        learning_rate = 0.02 if frame_index < WARMUP_FRAMES else 0.001
        fg = self.bg_sub.apply(frame, learningRate=learning_rate)
        fg = cv2.threshold(fg, 180, 255, cv2.THRESH_BINARY)[1]
        fg = cv2.bitwise_and(fg, pitch_mask)

        # 形态学处理，尽量保留球员而抑制草坪纹理与压缩噪声
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_open)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close)
        fg = cv2.dilate(fg, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Detection] = []

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh

            # 按分辨率自适应设置面积阈值
            min_area = max(24, int(0.00008 * h * w))
            max_area = max(min_area + 1, int(0.006 * h * w))
            if not (min_area <= area <= max_area):
                continue

            if bw < 4 or bh < 9:
                continue

            aspect = bh / max(bw, 1)
            if aspect < 1.05 or aspect > 6.2:
                continue

            if y < int(0.16 * h) or (y + bh) > int(0.98 * h):
                continue

            # 计算目标在框内的占比，过滤广告条和大片拖影
            comp = np.zeros((bh, bw), dtype=np.uint8)
            cv2.drawContours(comp, [cnt - np.array([[x, y]])], -1, 255, -1)
            fill_ratio = float((comp > 0).sum()) / max(area, 1)
            if fill_ratio < 0.12 or fill_ratio > 0.92:
                continue

            feat = self.extract_jersey_feature(frame, (x, y, bw, bh), pitch_mask)
            if feat is None:
                continue

            # 底部中心更接近球员脚下位置，便于轨迹建模
            foot_x = float(x + bw * 0.5)
            foot_y = float(y + bh * 0.95)
            center_x = float(x + bw * 0.5)
            center_y = float(y + bh * 0.5)

            conf = self.compute_detection_confidence(bw, bh, fill_ratio, feat, frame.shape[:2])
            if conf < 0.33:
                continue

            detections.append(
                Detection(
                    point=(foot_x, foot_y),
                    center=(center_x, center_y),
                    bbox=(x, y, bw, bh),
                    jersey_feat=feat,
                    confidence=conf,
                )
            )

        detections = self.team_cluster.assign(detections)
        return detections

    def extract_jersey_feature(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], pitch_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        提取球衣颜色特征。

        只使用目标框的上半身区域，并尽量排除草坪绿色。
        特征采用：
        - hue 的圆周编码（sin / cos），避免色相 0/180 跳变问题
        - 饱和度均值
        - 明度均值
        """
        x, y, w, h = bbox
        img_h, img_w = frame.shape[:2]

        x0 = max(0, x + int(w * 0.18))
        x1 = min(img_w, x + int(w * 0.82))
        y0 = max(0, y + int(h * 0.10))
        y1 = min(img_h, y + int(h * 0.58))
        if x1 <= x0 or y1 <= y0:
            return None

        roi = frame[y0:y1, x0:x1]
        roi_pitch = pitch_mask[y0:y1, x0:x1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 排除绿色草坪像素和过暗像素
        non_green = ~((hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] >= 35))
        valid = (roi_pitch > 0) & non_green & (hsv[:, :, 2] >= 25)
        ys, xs = np.where(valid)
        if len(xs) < 12:
            return None

        pix = hsv[ys, xs].astype(np.float32)
        hue_deg = pix[:, 0] * (2.0 * math.pi / 180.0)
        hue_sin = float(np.mean(np.sin(hue_deg)))
        hue_cos = float(np.mean(np.cos(hue_deg)))
        sat = float(np.mean(pix[:, 1]) / 255.0)
        val = float(np.mean(pix[:, 2]) / 255.0)
        feat = np.array([hue_sin, hue_cos, sat, val], dtype=np.float32)
        return feat

    @staticmethod
    def compute_detection_confidence(bw: int, bh: int, fill_ratio: float, feat: np.ndarray, frame_hw: Tuple[int, int]) -> float:
        """用几何与颜色质量构造启发式置信度。"""
        h, w = frame_hw
        area = bw * bh
        area_score = min(1.0, area / max(0.0006 * h * w, 1.0))
        aspect = bh / max(bw, 1)
        aspect_score = 1.0 - min(abs(aspect - 2.3) / 2.3, 1.0)
        fill_score = 1.0 - min(abs(fill_ratio - 0.42) / 0.42, 1.0)
        color_score = float(np.clip(0.45 * feat[2] + 0.25 * feat[3] + 0.30, 0.0, 1.0))
        conf = 0.25 * area_score + 0.25 * aspect_score + 0.20 * fill_score + 0.30 * color_score
        return float(np.clip(conf, 0.0, 1.0))

    @staticmethod
    def jersey_distance(a: np.ndarray, b: np.ndarray) -> float:
        """颜色特征距离。"""
        return float(np.linalg.norm(a - b))

    def update_tracks(self, detections: List[Detection]):
        """使用匈牙利匹配更新轨迹。"""
        track_ids = list(self.tracks.keys())
        if len(track_ids) == 0:
            for det in detections:
                self.start_new_track(det)
            return

        track_list = [self.tracks[k] for k in track_ids]
        num_tracks = len(track_list)
        num_dets = len(detections)

        cost = np.full((num_tracks, num_dets), 1e6, dtype=np.float32)

        for i, tr in enumerate(track_list):
            pred = np.array(tr.point, dtype=np.float32) + tr.smooth_velocity()
            for j, det in enumerate(detections):
                dp = np.linalg.norm(np.array(det.point, dtype=np.float32) - pred)
                if dp > MAX_MATCH_DISTANCE:
                    continue
                dc = self.jersey_distance(tr.jersey_feat, det.jersey_feat)
                team_penalty = 0.0
                if tr.team_id is not None and det.team_id is not None and tr.team_id != det.team_id:
                    team_penalty = 0.45
                elif tr.team_id is None or det.team_id is None:
                    team_penalty = 0.08

                miss_penalty = 0.05 * tr.missing
                cost[i, j] = dp / MAX_MATCH_DISTANCE + 0.35 * dc + team_penalty + miss_penalty

        row_ind, col_ind = linear_sum_assignment(cost)
        assigned_tracks = set()
        assigned_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] > 1.35:
                continue
            tr = track_list[r]
            det = detections[c]
            self.apply_detection_to_track(tr, det)
            assigned_tracks.add(tr.track_id)
            assigned_dets.add(c)

        for tr in track_list:
            if tr.track_id not in assigned_tracks:
                tr.age += 1
                tr.missing += 1
                v = tr.smooth_velocity()
                p = np.array(tr.point, dtype=np.float32) + 0.75 * v
                tr.point = (float(p[0]), float(p[1]))
                tr.history.append(tr.point)

        for j, det in enumerate(detections):
            if j not in assigned_dets:
                self.start_new_track(det)

        # 删除长期丢失轨迹
        to_delete = [tid for tid, tr in self.tracks.items() if tr.missing > MAX_MISSING_FRAMES]
        for tid in to_delete:
            del self.tracks[tid]

    def start_new_track(self, det: Detection):
        """创建新轨迹。"""
        tr = Track(
            track_id=self.next_track_id,
            point=det.point,
            bbox=det.bbox,
            jersey_feat=det.jersey_feat.copy(),
            confidence=det.confidence,
            team_id=det.team_id,
        )
        self.tracks[tr.track_id] = tr
        self.next_track_id += 1

    def apply_detection_to_track(self, tr: Track, det: Detection):
        """用当前检测更新轨迹。"""
        alpha = 0.72
        px = alpha * det.point[0] + (1.0 - alpha) * tr.point[0]
        py = alpha * det.point[1] + (1.0 - alpha) * tr.point[1]
        tr.point = (float(px), float(py))
        tr.bbox = det.bbox
        tr.jersey_feat = 0.82 * tr.jersey_feat + 0.18 * det.jersey_feat
        tr.confidence = 0.65 * tr.confidence + 0.35 * det.confidence
        if det.team_id is not None:
            tr.team_id = det.team_id
        tr.history.append(tr.point)
        tr.age += 1
        tr.hits += 1
        tr.missing = 0

    def process_frame(self, frame: np.ndarray, frame_index: int) -> Tuple[np.ndarray, List[Track], np.ndarray]:
        """处理单帧，返回球场掩码、当前轨迹列表和可视化底图。"""
        pitch_mask = self.extract_pitch_mask(frame)
        detections = self.extract_player_candidates(frame, pitch_mask, frame_index)
        self.update_tracks(detections)
        tracks = list(self.tracks.values())
        return pitch_mask, tracks, frame.copy()


class PaperInspiredOnlinePredictor:
    """
    参考 FootBots / TranSPORTmer 思路的在线预测器。

    - 多主体社会交互：队友间距、对手压力、队形压缩
    - 条件预测：用伪球位置作为条件量
    - 不确定性：跟踪缺失、历史不足、局部拥挤时缩短预测长度
    """

    def __init__(self, attack_direction: str = "right"):
        self.attack_direction = attack_direction

    @staticmethod
    def unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        if n < 1e-6:
            return np.zeros_like(v, dtype=np.float32)
        return (v / n).astype(np.float32)

    @staticmethod
    def k_nearest(point: Tuple[float, float], others: List[Tuple[float, float]], k: int = 3) -> List[Tuple[float, float]]:
        if not others:
            return []
        p = np.array(point, dtype=np.float32)
        arr = np.array(others, dtype=np.float32)
        dist = np.linalg.norm(arr - p[None, :], axis=1)
        idx = np.argsort(dist)[:k]
        return [tuple(arr[i]) for i in idx]

    def choose_attacking_team(self, confirmed_tracks: List[Track], frame_w: int) -> int:
        """自动估计当前更像进攻方的是哪一队。"""
        team_groups = {0: [], 1: []}
        for tr in confirmed_tracks:
            if tr.team_id in team_groups:
                team_groups[tr.team_id].append(tr)

        if ATTACKING_TEAM_MODE == "manual":
            return int(MANUAL_ATTACKING_TEAM)

        scores = {0: -1e9, 1: -1e9}
        for team_id, group in team_groups.items():
            if len(group) == 0:
                continue
            pts = np.array([tr.point for tr in group], dtype=np.float32)
            vs = np.array([tr.smooth_velocity() for tr in group], dtype=np.float32)
            mean_x = float(np.mean(pts[:, 0]))
            mean_vx = float(np.mean(vs[:, 0]))
            if self.attack_direction == "right":
                scores[team_id] = mean_x / max(frame_w, 1) + 0.65 * mean_vx
            else:
                scores[team_id] = (1.0 - mean_x / max(frame_w, 1)) - 0.65 * mean_vx
        return 0 if scores[0] >= scores[1] else 1

    def estimate_pseudo_ball(self, attackers: List[Track], frame_h: int, frame_w: int) -> np.ndarray:
        """没有显式足球检测时，用进攻队最前沿球员近似持球中心。"""
        if not attackers:
            return np.array([frame_w * 0.5, frame_h * 0.55], dtype=np.float32)

        pts = np.array([tr.point for tr in attackers], dtype=np.float32)
        ys = np.abs(pts[:, 1] - frame_h * 0.55)
        if self.attack_direction == "right":
            score = pts[:, 0] - 0.25 * ys
            idx = int(np.argmax(score))
        else:
            score = -pts[:, 0] - 0.25 * ys
            idx = int(np.argmax(score))
        carrier = pts[idx]
        centroid = np.mean(pts, axis=0)
        return (0.72 * carrier + 0.28 * centroid).astype(np.float32)

    def local_spacing_vector(self, point: Tuple[float, float], teammates: List[Tuple[float, float]]) -> np.ndarray:
        """鼓励队友之间保持合理间距。"""
        p = np.array(point, dtype=np.float32)
        neigh = [q for q in self.k_nearest(point, teammates, 3) if np.linalg.norm(np.array(q, dtype=np.float32) - p) > 1.0]
        if not neigh:
            return np.zeros(2, dtype=np.float32)
        v = np.zeros(2, dtype=np.float32)
        for q in neigh:
            q = np.array(q, dtype=np.float32)
            d = p - q
            n = np.linalg.norm(d)
            if n < 1e-4:
                continue
            v += d / (n * n)
        return self.unit(v)

    def opponent_pressure_vector(self, point: Tuple[float, float], opponents: List[Tuple[float, float]]) -> np.ndarray:
        """远离对手逼抢的方向。"""
        p = np.array(point, dtype=np.float32)
        neigh = self.k_nearest(point, opponents, 3)
        if not neigh:
            return np.zeros(2, dtype=np.float32)
        v = np.zeros(2, dtype=np.float32)
        for q in neigh:
            q = np.array(q, dtype=np.float32)
            d = p - q
            n = np.linalg.norm(d)
            if n < 1e-4:
                continue
            v += d / (n * n)
        return self.unit(v)

    def defensive_line_x(self, defenders: List[Track]) -> Optional[float]:
        """估计防线的大致横向位置。"""
        if not defenders:
            return None
        xs = sorted([tr.point[0] for tr in defenders])
        return float(np.median(xs))

    def estimate_uncertainty(self, tr: Track, opponents: List[Track], frame_w: int) -> float:
        """根据缺失、历史长度和拥挤程度估计不确定性。"""
        base = 0.0
        base += 0.28 * min(tr.missing / max(MAX_MISSING_FRAMES, 1), 1.0)
        base += 0.22 * max(0.0, 1.0 - min(len(tr.history) / max(HISTORY_SIZE, 1), 1.0))
        base += 0.18 * (1.0 - tr.confidence)

        p = np.array(tr.point, dtype=np.float32)
        opp_pts = [op.point for op in opponents]
        crowd = 0.0
        for q in self.k_nearest(tr.point, opp_pts, 2):
            d = np.linalg.norm(np.array(q, dtype=np.float32) - p)
            crowd += max(0.0, 1.0 - d / max(frame_w * 0.10, 1.0))
        crowd = min(1.0, crowd)
        base += 0.32 * crowd
        return float(np.clip(base, 0.0, 1.0))

    def predict_path(
        self,
        tr: Track,
        teammates: List[Track],
        opponents: List[Track],
        pseudo_ball: np.ndarray,
        attacking_team: int,
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[List[Tuple[int, int]], float]:
        """给单个球员预测短时未来轨迹。"""
        h, w = frame_shape[:2]
        p0 = np.array(tr.point, dtype=np.float32)
        v_hist = tr.smooth_velocity()
        inertia = self.unit(v_hist)
        center_bias = self.unit(np.array([0.0, h * 0.55], dtype=np.float32) - p0)
        spacing = self.local_spacing_vector(tr.point, [tm.point for tm in teammates if tm.track_id != tr.track_id])
        pressure_away = self.opponent_pressure_vector(tr.point, [op.point for op in opponents])
        ball_link = self.unit(pseudo_ball - p0)

        if self.attack_direction == "right":
            goal = np.array([w - 10.0, h * 0.55], dtype=np.float32)
            own_goal = np.array([10.0, h * 0.55], dtype=np.float32)
            forward = np.array([1.0, 0.0], dtype=np.float32)
        else:
            goal = np.array([10.0, h * 0.55], dtype=np.float32)
            own_goal = np.array([w - 10.0, h * 0.55], dtype=np.float32)
            forward = np.array([-1.0, 0.0], dtype=np.float32)

        is_attacker = (tr.team_id == attacking_team)
        uncertainty = self.estimate_uncertainty(tr, opponents, w)

        if is_attacker:
            # 参考 FootBots 的“条件预测 + 社会交互”思想
            direct_goal = self.unit(goal - p0)
            v = (
                0.34 * inertia +
                0.24 * direct_goal +
                0.12 * center_bias +
                0.12 * spacing +
                0.10 * pressure_away +
                0.08 * ball_link
            )

            dline = self.defensive_line_x(opponents)
            if dline is not None:
                if self.attack_direction == "right":
                    near_line = max(0.0, p0[0] - dline)
                else:
                    near_line = max(0.0, dline - p0[0])
                if near_line > 0:
                    v += 0.08 * center_bias

            step = max(7.0, np.linalg.norm(v_hist) * 1.35 + 7.5)
        else:
            # 参考 TranSPORTmer 的“统一建模思路”，让防守动作也依赖同一组条件量
            near_att = None
            if len(opponents) > 0:
                opp_pts = [op.point for op in opponents]
                near_att = self.k_nearest(tr.point, opp_pts, 1)
                near_att = np.array(near_att[0], dtype=np.float32) if len(near_att) > 0 else pseudo_ball
            else:
                near_att = pseudo_ball

            mark = self.unit(near_att - p0)
            goal_cover = self.unit(own_goal - p0)
            ball_compact = self.unit(pseudo_ball - p0)
            teammates_pts = np.array([tm.point for tm in teammates], dtype=np.float32) if teammates else np.array([[p0[0], p0[1]]], dtype=np.float32)
            line_center = teammates_pts.mean(axis=0)
            line_keep = self.unit(np.array([line_center[0], p0[1]], dtype=np.float32) - p0)
            v = (
                0.26 * inertia +
                0.24 * mark +
                0.24 * goal_cover +
                0.16 * ball_compact +
                0.10 * line_keep
            )
            step = max(6.0, np.linalg.norm(v_hist) * 1.10 + 6.5)

        v = self.unit(v)
        step = step * (1.0 - 0.38 * uncertainty)

        pts = [p0.copy()]
        cur = p0.copy()
        for k in range(PREDICT_HORIZON_STEPS):
            cur = cur + v * step * (0.72 + 0.06 * k)
            cur[0] = np.clip(cur[0], 0.0, w - 1.0)
            cur[1] = np.clip(cur[1], 0.0, h - 1.0)
            pts.append(cur.copy())

        pts_int = [(int(round(p[0])), int(round(p[1]))) for p in pts]
        return pts_int, uncertainty


def draw_dashed_line(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int], thickness: int = 2, dash_len: int = 8):
    """绘制虚线。"""
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


def draw_uncertainty_fan(img: np.ndarray, start: Tuple[int, int], end: Tuple[int, int], uncertainty: float, color: Tuple[int, int, int]):
    """在预测终点附近画出不确定性扇形。"""
    p0 = np.array(start, dtype=np.float32)
    p1 = np.array(end, dtype=np.float32)
    v = p1 - p0
    n = np.linalg.norm(v)
    if n < 1e-6:
        return
    v = v / n
    perp = np.array([-v[1], v[0]], dtype=np.float32)
    span = 8.0 + 24.0 * uncertainty
    a = p1 + perp * span
    b = p1 - perp * span
    cv2.line(img, tuple(a.astype(int)), tuple(p1.astype(int)), color, 1, lineType=cv2.LINE_AA)
    cv2.line(img, tuple(b.astype(int)), tuple(p1.astype(int)), color, 1, lineType=cv2.LINE_AA)
    cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, 1, lineType=cv2.LINE_AA)


def render_frame(frame: np.ndarray, tracks: List[Track], predictor: PaperInspiredOnlinePredictor) -> np.ndarray:
    """绘制检测、分队、轨迹与预测结果。"""
    canvas = frame.copy()
    h, w = frame.shape[:2]

    confirmed_tracks = [tr for tr in tracks if tr.confirmed and tr.team_id in (0, 1)]
    attacking_team = predictor.choose_attacking_team(confirmed_tracks, w)
    attackers = [tr for tr in confirmed_tracks if tr.team_id == attacking_team]
    defenders = [tr for tr in confirmed_tracks if tr.team_id != attacking_team]
    pseudo_ball = predictor.estimate_pseudo_ball(attackers, h, w)

    # 先画轨迹历史
    for tr in tracks:
        color = TEAM_DRAW_COLORS[tr.team_id] if tr.team_id in (0, 1) else UNKNOWN_DRAW_COLOR
        pts = list(tr.history)
        for i in range(1, len(pts)):
            p0 = (int(round(pts[i - 1][0])), int(round(pts[i - 1][1])))
            p1 = (int(round(pts[i][0])), int(round(pts[i][1])))
            cv2.line(canvas, p0, p1, color, 2, lineType=cv2.LINE_AA)

    # 再画球员框、轨迹 ID 和未来预测
    for tr in tracks:
        color = TEAM_DRAW_COLORS[tr.team_id] if tr.team_id in (0, 1) else UNKNOWN_DRAW_COLOR
        x, y, bw, bh = tr.bbox
        px, py = int(round(tr.point[0])), int(round(tr.point[1]))
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 6, color, 2, lineType=cv2.LINE_AA)
        cv2.putText(canvas, f"ID{tr.track_id}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, lineType=cv2.LINE_AA)

        if tr.confirmed and tr.team_id in (0, 1):
            teammates = [tm for tm in confirmed_tracks if tm.team_id == tr.team_id]
            opponents = [op for op in confirmed_tracks if op.team_id != tr.team_id]
            path, uncertainty = predictor.predict_path(tr, teammates, opponents, pseudo_ball, attacking_team, frame.shape)
            for a, b in zip(path[:-1], path[1:]):
                draw_dashed_line(canvas, a, b, color, thickness=2, dash_len=6)
            if len(path) >= 2:
                cv2.arrowedLine(canvas, path[-2], path[-1], color, 2, tipLength=0.24)
                draw_uncertainty_fan(canvas, path[-2], path[-1], uncertainty, color)

    # 画伪球位置与信息面板
    pbx, pby = int(round(pseudo_ball[0])), int(round(pseudo_ball[1]))
    cv2.circle(canvas, (pbx, pby), 5, (0, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (pbx, pby), 10, (0, 255, 255), 1, lineType=cv2.LINE_AA)

    panel_h = 108
    cv2.rectangle(canvas, (8, 8), (390, 8 + panel_h), (20, 20, 20), -1)
    cv2.putText(canvas, "视频直跑版：自动分队 + 在线多智能体轨迹预测", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 235, 235), 1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, f"Team0 显示色 / Team1 显示色: 固定绘图色（非球衣真实颜色）", (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, f"自动判定进攻方: Team{attacking_team}    进攻方向: {ATTACK_DIRECTION}", (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, f"稳定轨迹数: {len(confirmed_tracks)}    总轨迹数: {len(tracks)}", (16, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, "黄色圆点: 伪球位置    虚线箭头: 未来短时轨迹", (16, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, lineType=cv2.LINE_AA)

    return canvas


def main():
    """主函数。"""
    if not os.path.exists(INPUT_VIDEO_PATH):
        raise FileNotFoundError(f"找不到输入视频: {INPUT_VIDEO_PATH}")

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("视频打开失败")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("输出视频创建失败")

    tracker = VideoSoccerTracker()
    predictor = PaperInspiredOnlinePredictor(attack_direction=ATTACK_DIRECTION)

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        _, tracks, _ = tracker.process_frame(frame, frame_index)
        vis = render_frame(frame, tracks, predictor)
        writer.write(vis)
        frame_index += 1

    cap.release()
    writer.release()
    print("处理完成，输出文件:", OUTPUT_VIDEO_PATH)


if __name__ == "__main__":
    main()
