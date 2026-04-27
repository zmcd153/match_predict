import os
import math
import argparse
from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as e:  # pragma: no cover
    raise RuntimeError("本脚本需要安装 PyTorch。请先安装 torch 后再运行。") from e


# ============================================================
# 参考 FootBots / TranSPORTmer 思路的“视频直跑”版本（中文注释）
#
# 设计目标：
# 1) 直接输入比赛视频，前端自动完成球场区域筛选、球员候选检测、跟踪、自动分队。
# 2) 后端提供一个真正的 PyTorch 多智能体轨迹预测器。
# 3) 若你提供训练好的 checkpoint，则使用 Transformer 做未来轨迹预测。
# 4) 若暂时没有 checkpoint，前端仍然可以跑通视频；预测会退回到速度基线，方便先联调。
#
# 与论文思路的对应：
# - FootBots：借鉴“多主体交互 + 条件预测 + set attention + temporal attention”。
# - TranSPORTmer：借鉴“mask 驱动的统一轨迹理解 + 不确定性 + 额外状态 token”。
#
# 重要说明：
# - 这个文件不是论文原始官方实现，而是“工程可跑版”的参考实现。
# - 真正的深度学习部分在 SocialTrajectoryTransformerCN 与 DeepPredictorWrapper。
# - 视频前端检测/跟踪为了降低依赖，默认使用 OpenCV；如果你本机装了 ultralytics，可选 YOLO 检测。
# ============================================================


# ---------------------------
# 数据结构
# ---------------------------

@dataclass
class Detection:
    """单帧检测结果。"""

    point: Tuple[float, float]                # 球员脚下锚点（bbox 底边中心）
    center: Tuple[float, float]               # 目标几何中心
    bbox: Tuple[int, int, int, int]           # x, y, w, h
    jersey_feat: np.ndarray                   # 球衣颜色特征（用于自动分队）
    confidence: float                         # 检测置信度
    team_id: Optional[int] = None             # 自动分队结果，0 / 1 / None


@dataclass
class Track:
    """跨帧轨迹。"""

    track_id: int
    point: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    jersey_feat: np.ndarray
    confidence: float
    team_id: Optional[int] = None
    history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=12))
    age: int = 1
    hits: int = 1
    missing: int = 0

    def __post_init__(self):
        if len(self.history) == 0:
            self.history.append(self.point)

    @property
    def confirmed(self) -> bool:
        """是否达到可用于预测的稳定轨迹标准。"""
        return self.hits >= 3 and self.missing <= 1

    def smooth_velocity(self) -> np.ndarray:
        """使用最近几步历史估计平滑速度。"""
        if len(self.history) < 2:
            return np.zeros(2, dtype=np.float32)
        pts = np.asarray(self.history, dtype=np.float32)
        diffs = pts[1:] - pts[:-1]
        if len(diffs) == 0:
            return np.zeros(2, dtype=np.float32)
        weights = np.linspace(0.6, 1.0, len(diffs), dtype=np.float32)
        vel = (diffs * weights[:, None]).sum(axis=0) / max(weights.sum(), 1e-6)
        return vel.astype(np.float32)


# ---------------------------
# 通用工具函数
# ---------------------------


def clip_pt(x: float, y: float, w: int, h: int) -> Tuple[int, int]:
    """把点裁剪到图像边界内。"""
    x = int(max(0, min(w - 1, round(x))))
    y = int(max(0, min(h - 1, round(y))))
    return x, y



def unit(v: np.ndarray) -> np.ndarray:
    """向量单位化。"""
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)



def smoothstep(x: float, lo: float, hi: float) -> float:
    """平滑区间映射。"""
    if hi <= lo:
        return 0.0
    t = max(0.0, min(1.0, (x - lo) / (hi - lo)))
    return t * t * (3.0 - 2.0 * t)



def greedy_match(cost_matrix: np.ndarray, max_cost: float) -> List[Tuple[int, int]]:
    """
    纯 NumPy 贪心匹配，避免依赖 scipy。

    参数：
        cost_matrix: [M, N]，值越小越好
        max_cost: 超过该代价的匹配会被拒绝
    返回：
        [(row_idx, col_idx), ...]
    """
    if cost_matrix.size == 0:
        return []
    pairs: List[Tuple[int, int, float]] = []
    m, n = cost_matrix.shape
    for i in range(m):
        for j in range(n):
            c = float(cost_matrix[i, j])
            if c <= max_cost:
                pairs.append((i, j, c))
    pairs.sort(key=lambda z: z[2])

    used_rows = set()
    used_cols = set()
    matched: List[Tuple[int, int]] = []
    for i, j, _ in pairs:
        if i in used_rows or j in used_cols:
            continue
        used_rows.add(i)
        used_cols.add(j)
        matched.append((i, j))
    return matched



def draw_dashed_line(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int], thickness: int = 2, dash_len: int = 8) -> None:
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


# ---------------------------
# 球场分割与检测前端
# ---------------------------


class PitchSegmenter:
    """提取球场草坪区域。"""

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        green = cv2.inRange(hsv, (25, 20, 25), (95, 255, 255))

        roi = np.zeros((h, w), dtype=np.uint8)
        roi[int(h * 0.12):int(h * 0.96), :] = 255
        mask = cv2.bitwise_and(green, roi)

        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if num_labels <= 1:
            return mask
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        pitch_mask = (labels == largest).astype(np.uint8) * 255
        pitch_mask = cv2.dilate(pitch_mask, np.ones((7, 7), np.uint8), iterations=1)
        return pitch_mask


class OpenCVPlayerDetector:
    """
    纯 OpenCV 球员候选检测器。

    思路：
    1) 只在球场区域内工作。
    2) 用背景减除提取运动前景。
    3) 结合非绿色约束与形状筛选，保留更像球员的候选框。
    """

    def __init__(self):
        self.bgsub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=32, detectShadows=False)

    @staticmethod
    def _jersey_feature(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """从上半身区域抽取球衣颜色特征。"""
        x, y, w, h = bbox
        torso_y0 = int(y + h * 0.18)
        torso_y1 = int(y + h * 0.58)
        torso_x0 = int(x + w * 0.18)
        torso_x1 = int(x + w * 0.82)
        roi = frame[max(0, torso_y0):max(0, torso_y1), max(0, torso_x0):max(0, torso_x1)]
        if roi.size == 0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        feat = np.median(hsv.reshape(-1, 3), axis=0).astype(np.float32)
        feat[0] /= 180.0
        feat[1] /= 255.0
        feat[2] /= 255.0
        return feat

    def detect(self, frame: np.ndarray, pitch_mask: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        fg = self.bgsub.apply(frame)

        non_green = cv2.bitwise_not(cv2.inRange(hsv, (25, 25, 25), (95, 255, 255)))
        motion = cv2.bitwise_and(fg, pitch_mask)
        motion = cv2.bitwise_and(motion, non_green)

        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, k1)
        motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, k2)

        num, labels, stats, centroids = cv2.connectedComponentsWithStats(motion, 8)
        detections: List[Detection] = []
        for i in range(1, num):
            x, y, bw, bh, area = stats[i]
            if area < 18 or area > 650:
                continue
            if bw < 4 or bh < 10 or bw > 36 or bh > 70:
                continue
            aspect = bh / max(bw, 1)
            if aspect < 1.10:
                continue
            if y < int(0.12 * h) or (y + bh) > int(0.98 * h):
                continue

            comp_mask = (labels[y:y + bh, x:x + bw] == i).astype(np.uint8) * 255
            fill_ratio = float(comp_mask.sum() / 255.0) / max(bw * bh, 1)
            conf = 0.40 * smoothstep(area, 18, 220) + 0.30 * smoothstep(aspect, 1.1, 3.2) + 0.30 * smoothstep(fill_ratio, 0.18, 0.85)
            if conf < 0.42:
                continue

            feat = self._jersey_feature(frame, (x, y, bw, bh))
            cx, cy = centroids[i]
            foot_x = float(x + bw / 2.0)
            foot_y = float(y + bh * 0.95)
            detections.append(
                Detection(
                    point=(foot_x, foot_y),
                    center=(float(cx), float(cy)),
                    bbox=(int(x), int(y), int(bw), int(bh)),
                    jersey_feat=feat,
                    confidence=float(conf),
                )
            )
        return detections


class YOLOPlayerDetector:
    """
    可选的 YOLO 球员检测器。

    说明：
    - 只有在本机安装了 ultralytics 时才可用。
    - 使用 person 类别，再限制到球场区域。
    """

    def __init__(self, model_name: str = "yolov8n.pt"):
        from ultralytics import YOLO

        self.model = YOLO(model_name)

    @staticmethod
    def _jersey_feature(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = bbox
        torso_y0 = int(y + h * 0.18)
        torso_y1 = int(y + h * 0.58)
        torso_x0 = int(x + w * 0.18)
        torso_x1 = int(x + w * 0.82)
        roi = frame[max(0, torso_y0):max(0, torso_y1), max(0, torso_x0):max(0, torso_x1)]
        if roi.size == 0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        feat = np.median(hsv.reshape(-1, 3), axis=0).astype(np.float32)
        feat[0] /= 180.0
        feat[1] /= 255.0
        feat[2] /= 255.0
        return feat

    def detect(self, frame: np.ndarray, pitch_mask: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        results = self.model.predict(frame, verbose=False, classes=[0])
        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf = float(box.conf.item())
                if conf < 0.20:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                px = int(x1 + bw / 2)
                py = int(y1 + bh * 0.95)
                px = max(0, min(w - 1, px))
                py = max(0, min(h - 1, py))
                if pitch_mask[py, px] == 0:
                    continue
                feat = self._jersey_feature(frame, (x1, y1, bw, bh))
                detections.append(
                    Detection(
                        point=(float(px), float(py)),
                        center=(float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)),
                        bbox=(x1, y1, bw, bh),
                        jersey_feat=feat,
                        confidence=conf,
                    )
                )
        return detections


# ---------------------------
# 自动分队与跟踪
# ---------------------------


class OnlineTeamClustering:
    """依据球衣主色自动分成两队，并尽量保持队伍编号跨帧稳定。"""

    def __init__(self):
        self.team_centers: Optional[np.ndarray] = None

    @staticmethod
    def _kmeans2(feats: np.ndarray, num_iters: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """纯 NumPy 的二类聚类。"""
        n = len(feats)
        if n < 2:
            labels = np.zeros((n,), dtype=np.int64)
            centers = feats[:1].copy() if n > 0 else np.zeros((1, feats.shape[1]), dtype=np.float32)
            return labels, centers

        idx0 = 0
        dists = np.linalg.norm(feats - feats[idx0:idx0+1], axis=1)
        idx1 = int(np.argmax(dists))
        centers = np.stack([feats[idx0], feats[idx1]], axis=0).astype(np.float32)

        labels = np.zeros((n,), dtype=np.int64)
        for _ in range(num_iters):
            dist0 = np.linalg.norm(feats - centers[0:1], axis=1)
            dist1 = np.linalg.norm(feats - centers[1:2], axis=1)
            labels = (dist1 < dist0).astype(np.int64)
            for k in range(2):
                sel = feats[labels == k]
                if len(sel) > 0:
                    centers[k] = sel.mean(axis=0)
        return labels, centers

    def assign(self, detections: List[Detection]) -> List[Detection]:
        if len(detections) < 4:
            return detections
        feats = np.asarray([d.jersey_feat for d in detections], dtype=np.float32)
        labels, centers = self._kmeans2(feats)

        if self.team_centers is not None and len(self.team_centers) == 2:
            cost = np.linalg.norm(centers[:, None, :] - self.team_centers[None, :, :], axis=2)
            pairs = greedy_match(cost, max_cost=10.0)
            remap = {new_idx: old_idx for new_idx, old_idx in pairs}
            labels = np.asarray([remap.get(int(lb), int(lb)) for lb in labels], dtype=np.int64)
            new_centers = np.zeros_like(centers)
            for new_idx in range(2):
                old_idx = remap.get(new_idx, new_idx)
                new_centers[old_idx] = centers[new_idx]
            centers = new_centers

        self.team_centers = centers.copy()
        for det, lb in zip(detections, labels.tolist()):
            det.team_id = int(lb)
        return detections


class SimpleTracker:
    """简单的基于距离与颜色的多目标跟踪器。"""

    def __init__(self, max_missing: int = 10, max_match_distance: float = 42.0):
        self.max_missing = max_missing
        self.max_match_distance = max_match_distance
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    def _cost(self, track: Track, det: Detection) -> float:
        p0 = np.asarray(track.point, dtype=np.float32)
        p1 = np.asarray(det.point, dtype=np.float32)
        dist = float(np.linalg.norm(p1 - p0))
        color_dist = float(np.linalg.norm(track.jersey_feat - det.jersey_feat)) * 28.0
        team_penalty = 0.0
        if track.team_id is not None and det.team_id is not None and track.team_id != det.team_id:
            team_penalty = 20.0
        return dist + color_dist + team_penalty

    def update(self, detections: List[Detection]) -> List[Track]:
        track_ids = list(self.tracks.keys())
        tracks = [self.tracks[tid] for tid in track_ids]

        if tracks and detections:
            cost = np.zeros((len(tracks), len(detections)), dtype=np.float32)
            for i, tr in enumerate(tracks):
                for j, det in enumerate(detections):
                    cost[i, j] = self._cost(tr, det)
            pairs = greedy_match(cost, max_cost=self.max_match_distance)
        else:
            pairs = []

        matched_track_idx = {i for i, _ in pairs}
        matched_det_idx = {j for _, j in pairs}

        for i, j in pairs:
            tr = tracks[i]
            det = detections[j]
            tr.point = det.point
            tr.bbox = det.bbox
            tr.jersey_feat = 0.75 * tr.jersey_feat + 0.25 * det.jersey_feat
            tr.confidence = 0.65 * tr.confidence + 0.35 * det.confidence
            tr.team_id = det.team_id if det.team_id is not None else tr.team_id
            tr.history.append(det.point)
            tr.age += 1
            tr.hits += 1
            tr.missing = 0

        for i, tr in enumerate(tracks):
            if i in matched_track_idx:
                continue
            tr.age += 1
            tr.missing += 1

        for j, det in enumerate(detections):
            if j in matched_det_idx:
                continue
            tr = Track(
                track_id=self.next_id,
                point=det.point,
                bbox=det.bbox,
                jersey_feat=det.jersey_feat.copy(),
                confidence=det.confidence,
                team_id=det.team_id,
            )
            self.tracks[self.next_id] = tr
            self.next_id += 1

        dead_ids = [tid for tid, tr in self.tracks.items() if tr.missing > self.max_missing]
        for tid in dead_ids:
            del self.tracks[tid]

        return list(self.tracks.values())


# ---------------------------
# 深度学习轨迹预测器（真正的 PyTorch 部分）
# ---------------------------


class MLP(nn.Module):
    """基础前馈网络。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinusoidalPositionEncoding(nn.Module):
    """时间位置编码。"""

    def __init__(self, dim: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, t: int, device: torch.device) -> torch.Tensor:
        return self.pe[:t].to(device)


class SetAttentionBlock(nn.Module):
    """对同一时间步的所有主体做集合注意力。"""

    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class TemporalAttentionBlock(nn.Module):
    """对同一主体的时间序列做注意力。"""

    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class SocialTrajectoryTransformerCN(nn.Module):
    """
    参考 FootBots / TranSPORTmer 思路的多智能体 Transformer。

    输入：
        coords: [1, T, N, 2]，归一化到 0~1 的历史位置
        observed_mask: [1, T, N]
        team_ids: [1, N]
    输出：
        future_coords: [1, H, N, 2]
        future_logvar: [1, H, N, 2]
        state_logits: [1, T, S]
    """

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, ff_dim: int = 256, num_spatial_layers: int = 2, num_temporal_layers: int = 2, num_states: int = 5, future_steps: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.future_steps = future_steps
        self.time_pe = SinusoidalPositionEncoding(hidden_dim)
        self.input_proj = MLP(6, hidden_dim, hidden_dim, dropout=dropout)
        self.team_embed = nn.Embedding(4, hidden_dim)
        self.cls_agent = nn.Parameter(torch.randn(1, 1, 1, hidden_dim) * 0.02)
        self.spatial_blocks = nn.ModuleList([SetAttentionBlock(hidden_dim, num_heads, ff_dim, dropout) for _ in range(num_spatial_layers)])
        self.temporal_blocks = nn.ModuleList([TemporalAttentionBlock(hidden_dim, num_heads, ff_dim, dropout) for _ in range(num_temporal_layers)])
        self.future_decoder = MLP(hidden_dim, ff_dim, future_steps * 4, dropout=dropout)
        self.state_head = MLP(hidden_dim, ff_dim, num_states, dropout=dropout)

    @staticmethod
    def _build_features(coords: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        vel = torch.zeros_like(coords)
        vel[:, 1:] = coords[:, 1:] - coords[:, :-1]
        valid_pair = observed_mask[:, 1:] * observed_mask[:, :-1]
        vel[:, 1:] = vel[:, 1:] * valid_pair.unsqueeze(-1)
        speed = torch.norm(vel, dim=-1, keepdim=True)
        obs = observed_mask.unsqueeze(-1)
        feat = torch.cat([coords, vel, speed, obs], dim=-1)
        return feat

    def forward(self, coords: torch.Tensor, observed_mask: torch.Tensor, team_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, n, _ = coords.shape
        feat = self._build_features(coords, observed_mask)
        x = self.input_proj(feat)
        x = x + self.team_embed(team_ids).unsqueeze(1)
        x = x + self.time_pe(t, coords.device).unsqueeze(0).unsqueeze(2)

        cls = self.cls_agent.expand(b, t, 1, self.hidden_dim)
        x = torch.cat([cls, x], dim=2)

        # 先在每个时间步上，对所有主体做集合注意力。
        for blk in self.spatial_blocks:
            frames = []
            for ti in range(t):
                frames.append(blk(x[:, ti]))
            x = torch.stack(frames, dim=1)

        # 再对每个主体做时间注意力。
        for blk in self.temporal_blocks:
            agents = []
            for ni in range(n + 1):
                agents.append(blk(x[:, :, ni, :]))
            x = torch.stack(agents, dim=2)

        cls_tokens = x[:, :, 0, :]
        agent_tokens = x[:, :, 1:, :]
        last_tokens = agent_tokens[:, -1]  # [B, N, C]
        out = self.future_decoder(last_tokens).view(b, n, self.future_steps, 4).permute(0, 2, 1, 3)
        future_delta = out[..., :2]
        future_logvar = out[..., 2:].clamp(-4.0, 3.0)

        last_pos = coords[:, -1:, :, :]
        future_coords = []
        prev = last_pos
        for hi in range(self.future_steps):
            step = prev + future_delta[:, hi:hi+1]
            future_coords.append(step)
            prev = step
        future_coords = torch.cat(future_coords, dim=1)
        state_logits = self.state_head(cls_tokens)
        return future_coords, future_logvar, state_logits


class DeepPredictorWrapper:
    """
    深度预测包装器。

    - 有 checkpoint：真正调用 Transformer。
    - 无 checkpoint：退回到速度基线，方便先跑通视频流程。
    """

    def __init__(self, future_steps: int = 8, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        self.future_steps = future_steps
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = SocialTrajectoryTransformerCN(future_steps=future_steps).to(self.device)
        self.model.eval()
        self.has_checkpoint = False

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                self.model.load_state_dict(ckpt["state_dict"], strict=False)
            elif isinstance(ckpt, dict):
                self.model.load_state_dict(ckpt, strict=False)
            else:
                raise RuntimeError("checkpoint 格式无法识别，应为 state_dict 或包含 state_dict 的字典。")
            self.has_checkpoint = True

    @staticmethod
    def _heuristic_predict(points: np.ndarray, future_steps: int, team_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """无 checkpoint 时的速度基线。"""
        t, n, _ = points.shape
        last = points[-1]
        if t >= 2:
            vel = points[-1] - points[-2]
        else:
            vel = np.zeros_like(last)

        preds = []
        cur = last.copy()
        for _ in range(future_steps):
            cur = cur + vel
            preds.append(cur.copy())
        preds = np.stack(preds, axis=0)
        logvar = np.zeros_like(preds) + 0.25
        return preds, logvar

    def predict(self, points_px: np.ndarray, observed_mask: np.ndarray, team_ids: np.ndarray, frame_w: int, frame_h: int) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        输入历史轨迹，输出未来轨迹。

        参数：
            points_px: [T, N, 2]，像素坐标
            observed_mask: [T, N]
            team_ids: [N]
        返回：
            future_px: [H, N, 2]
            future_logvar: [H, N, 2]
            used_deep_model: 是否真正调用了深度模型
        """
        if points_px.size == 0:
            return np.zeros((self.future_steps, 0, 2), dtype=np.float32), np.zeros((self.future_steps, 0, 2), dtype=np.float32), self.has_checkpoint

        norm = np.array([frame_w, frame_h], dtype=np.float32)
        coords = points_px / np.maximum(norm[None, None, :], 1.0)
        coords = np.clip(coords, 0.0, 1.0)

        if not self.has_checkpoint:
            preds, logvar = self._heuristic_predict(coords, self.future_steps, team_ids)
            preds = preds * norm[None, None, :]
            return preds.astype(np.float32), logvar.astype(np.float32), False

        with torch.no_grad():
            coords_t = torch.from_numpy(coords[None]).float().to(self.device)
            obs_t = torch.from_numpy(observed_mask[None]).float().to(self.device)
            team_t = torch.from_numpy(np.clip(team_ids, 0, 3)[None]).long().to(self.device)
            future, logvar, _ = self.model(coords_t, obs_t, team_t)
            future_np = future[0].cpu().numpy() * norm[None, None, :]
            logvar_np = logvar[0].cpu().numpy()
        return future_np.astype(np.float32), logvar_np.astype(np.float32), True


# ---------------------------
# 由轨迹构建模型输入
# ---------------------------


def build_track_tensor(tracks: List[Track], history_size: int, frame_w: int, frame_h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Track]]:
    """把活跃轨迹组装成模型输入张量。"""
    active = [tr for tr in tracks if tr.confirmed]
    if not active:
        return np.zeros((history_size, 0, 2), dtype=np.float32), np.zeros((history_size, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64), []

    points = np.zeros((history_size, len(active), 2), dtype=np.float32)
    observed = np.zeros((history_size, len(active)), dtype=np.float32)
    team_ids = np.zeros((len(active),), dtype=np.int64)

    for i, tr in enumerate(active):
        hist = list(tr.history)
        start = max(0, history_size - len(hist))
        for j, pt in enumerate(hist[-history_size:]):
            points[start + j, i] = np.asarray(pt, dtype=np.float32)
            observed[start + j, i] = 1.0
        team_ids[i] = -1 if tr.team_id is None else int(tr.team_id)

    return points, observed, team_ids, active


# ---------------------------
# 在线条件变量与可视化
# ---------------------------


def estimate_pseudo_ball(tracks: List[Track], attacking_team: int, frame_w: int, frame_h: int, attack_direction: str) -> np.ndarray:
    """没有真实球检测时，用进攻方前沿与队形中心估计一个伪球位置。"""
    atk = [tr for tr in tracks if tr.team_id == attacking_team and tr.confirmed]
    if not atk:
        return np.array([frame_w * 0.5, frame_h * 0.55], dtype=np.float32)
    pts = np.asarray([tr.point for tr in atk], dtype=np.float32)
    if attack_direction == "left":
        idx = int(np.argmin(pts[:, 0] + 0.18 * np.abs(pts[:, 1] - frame_h * 0.55)))
    else:
        idx = int(np.argmax(pts[:, 0] - 0.18 * np.abs(pts[:, 1] - frame_h * 0.55)))
    front = pts[idx]
    center = pts.mean(axis=0)
    return (0.7 * front + 0.3 * center).astype(np.float32)


TEAM_DRAW_COLORS = [(255, 180, 60), (80, 220, 255)]
UNKNOWN_DRAW_COLOR = (180, 180, 180)


def draw_track_and_prediction(frame: np.ndarray, tracks: List[Track], active_tracks: List[Track], future_px: np.ndarray, future_logvar: np.ndarray, used_deep_model: bool, pseudo_ball: np.ndarray, attacking_team: int, attack_direction: str) -> np.ndarray:
    """把轨迹与预测结果画到视频帧上。"""
    canvas = frame.copy()
    h, w = frame.shape[:2]

    # 先画所有轨迹框与历史轨迹。
    for tr in tracks:
        x, y, bw, bh = tr.bbox
        color = TEAM_DRAW_COLORS[tr.team_id] if tr.team_id in (0, 1) else UNKNOWN_DRAW_COLOR
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, 1, lineType=cv2.LINE_AA)
        px, py = clip_pt(tr.point[0], tr.point[1], w, h)
        cv2.circle(canvas, (px, py), 6, color, 2, lineType=cv2.LINE_AA)
        cv2.putText(canvas, f"ID{tr.track_id}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, lineType=cv2.LINE_AA)

        hist = list(tr.history)
        for i in range(1, len(hist)):
            p0 = clip_pt(hist[i - 1][0], hist[i - 1][1], w, h)
            p1 = clip_pt(hist[i][0], hist[i][1], w, h)
            cv2.line(canvas, p0, p1, color, 1, lineType=cv2.LINE_AA)

    # 再画深度模型（或基线）的未来轨迹。
    if len(active_tracks) > 0 and future_px.size > 0:
        track_to_idx = {tr.track_id: i for i, tr in enumerate(active_tracks)}
        for tr in active_tracks:
            idx = track_to_idx[tr.track_id]
            color = TEAM_DRAW_COLORS[tr.team_id] if tr.team_id in (0, 1) else UNKNOWN_DRAW_COLOR
            prev = clip_pt(tr.point[0], tr.point[1], w, h)
            uncertainty = float(np.exp(future_logvar[:, idx]).mean()) if future_logvar.size > 0 else 1.0
            max_draw_steps = max(2, min(len(future_px), int(round(len(future_px) * (1.2 / max(uncertainty, 1e-3))))))
            for step in range(min(len(future_px), max_draw_steps)):
                nxt = clip_pt(future_px[step, idx, 0], future_px[step, idx, 1], w, h)
                draw_dashed_line(canvas, prev, nxt, color, thickness=2, dash_len=6)
                prev = nxt
            if len(future_px) >= 1:
                last_vis = clip_pt(future_px[min(max_draw_steps - 1, len(future_px) - 1), idx, 0], future_px[min(max_draw_steps - 1, len(future_px) - 1), idx, 1], w, h)
                cv2.arrowedLine(canvas, clip_pt(tr.point[0], tr.point[1], w, h), last_vis, color, 2, tipLength=0.20)

    # 伪球位置与说明。
    bx, by = clip_pt(float(pseudo_ball[0]), float(pseudo_ball[1]), w, h)
    cv2.circle(canvas, (bx, by), 8, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.putText(canvas, "pseudo-ball", (bx + 8, by - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, lineType=cv2.LINE_AA)

    # 图例。
    mode_text = "深度模型" if used_deep_model else "速度基线(未加载checkpoint)"
    atk_text = f"进攻队: Team{attacking_team}  方向: {attack_direction}"
    cv2.rectangle(canvas, (8, 8), (420, 94), (18, 18, 18), -1)
    cv2.putText(canvas, mode_text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, atk_text, (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, "橙/青: 自动分出的两队；绿圈: 伪球位置", (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, f"总轨迹={len(tracks)}  可预测轨迹={len(active_tracks)}", (16, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, lineType=cv2.LINE_AA)

    return canvas


# ---------------------------
# 主流程
# ---------------------------


def choose_attacking_team(tracks: List[Track], mode: str, manual_team: int, attack_direction: str) -> int:
    """自动或手动确定进攻队。"""
    if mode == "manual":
        return int(manual_team)

    team_pts = {0: [], 1: []}
    for tr in tracks:
        if tr.team_id in (0, 1) and tr.confirmed:
            team_pts[int(tr.team_id)].append(tr.point)
    if len(team_pts[0]) == 0 or len(team_pts[1]) == 0:
        return 0

    x0 = float(np.mean([p[0] for p in team_pts[0]]))
    x1 = float(np.mean([p[0] for p in team_pts[1]]))
    if attack_direction == "right":
        return 0 if x0 > x1 else 1
    return 0 if x0 < x1 else 1


class VideoSoccerDeepPipeline:
    """完整的视频处理管线。"""

    def __init__(self, detector_backend: str, yolo_model: str, checkpoint_path: Optional[str], history_size: int, future_steps: int, max_missing: int, max_match_distance: float, attack_direction: str, attacking_team_mode: str, manual_attacking_team: int, warmup_frames: int):
        self.pitch_segmenter = PitchSegmenter()
        if detector_backend == "yolo":
            try:
                self.detector = YOLOPlayerDetector(model_name=yolo_model)
            except Exception as e:
                print(f"[警告] YOLO 初始化失败，自动退回 OpenCV 检测器：{e}")
                self.detector = OpenCVPlayerDetector()
        else:
            self.detector = OpenCVPlayerDetector()

        self.team_cluster = OnlineTeamClustering()
        self.tracker = SimpleTracker(max_missing=max_missing, max_match_distance=max_match_distance)
        self.predictor = DeepPredictorWrapper(future_steps=future_steps, checkpoint_path=checkpoint_path)
        self.history_size = history_size
        self.attack_direction = attack_direction
        self.attacking_team_mode = attacking_team_mode
        self.manual_attacking_team = manual_attacking_team
        self.warmup_frames = warmup_frames

    def run(self, input_video: str, output_video: str) -> None:
        if not os.path.exists(input_video):
            raise FileNotFoundError(f"找不到输入视频: {input_video}")

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise RuntimeError("视频打开失败")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1e-3:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("输出视频创建失败")

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            pitch_mask = self.pitch_segmenter(frame)
            detections = self.detector.detect(frame, pitch_mask)
            detections = self.team_cluster.assign(detections)
            tracks = self.tracker.update(detections)

            if frame_idx <= self.warmup_frames:
                attacking_team = choose_attacking_team(tracks, self.attacking_team_mode, self.manual_attacking_team, self.attack_direction)
                pseudo_ball = estimate_pseudo_ball(tracks, attacking_team, width, height, self.attack_direction)
                canvas = draw_track_and_prediction(frame, tracks, [], np.zeros((0, 0, 2), dtype=np.float32), np.zeros((0, 0, 2), dtype=np.float32), self.predictor.has_checkpoint, pseudo_ball, attacking_team, self.attack_direction)
                writer.write(canvas)
                continue

            points, observed, team_ids, active_tracks = build_track_tensor(tracks, self.history_size, width, height)
            future_px, future_logvar, used_deep_model = self.predictor.predict(points, observed, team_ids, width, height)
            attacking_team = choose_attacking_team(tracks, self.attacking_team_mode, self.manual_attacking_team, self.attack_direction)
            pseudo_ball = estimate_pseudo_ball(tracks, attacking_team, width, height, self.attack_direction)
            canvas = draw_track_and_prediction(frame, tracks, active_tracks, future_px, future_logvar, used_deep_model, pseudo_ball, attacking_team, self.attack_direction)
            writer.write(canvas)

            if frame_idx % 50 == 0:
                print(f"已处理 {frame_idx}/{total_frames if total_frames > 0 else '?'} 帧")

        cap.release()
        writer.release()
        print(f"处理完成，结果已保存到: {output_video}")


# ---------------------------
# 命令行入口
# ---------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="参考 FootBots / TranSPORTmer 思路的足球视频轨迹预测脚本（中文注释）")
    parser.add_argument("--input_video", type=str, required=True, help="输入比赛视频路径")
    parser.add_argument("--output_video", type=str, default="match_video_predict_deep.mp4", help="输出视频路径")
    parser.add_argument("--detector_backend", type=str, default="opencv", choices=["opencv", "yolo"], help="前端检测器类型")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt", help="若使用 YOLO，则指定模型名或权重路径")
    parser.add_argument("--checkpoint", type=str, default="", help="Transformer 预测器的 checkpoint 路径；不提供则退回速度基线")
    parser.add_argument("--history_size", type=int, default=10, help="历史轨迹长度")
    parser.add_argument("--future_steps", type=int, default=8, help="预测未来步数")
    parser.add_argument("--max_missing", type=int, default=10, help="轨迹允许连续丢失的最大帧数")
    parser.add_argument("--max_match_distance", type=float, default=42.0, help="跟踪匹配最大距离")
    parser.add_argument("--attack_direction", type=str, default="right", choices=["left", "right"], help="进攻方向")
    parser.add_argument("--attacking_team_mode", type=str, default="auto", choices=["auto", "manual"], help="自动/手动指定进攻队")
    parser.add_argument("--manual_attacking_team", type=int, default=0, choices=[0, 1], help="手动指定进攻队编号")
    parser.add_argument("--warmup_frames", type=int, default=18, help="前若干帧只做检测跟踪，不做预测，用于稳定背景建模")
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    pipeline = VideoSoccerDeepPipeline(
        detector_backend=args.detector_backend,
        yolo_model=args.yolo_model,
        checkpoint_path=args.checkpoint if args.checkpoint else None,
        history_size=args.history_size,
        future_steps=args.future_steps,
        max_missing=args.max_missing,
        max_match_distance=args.max_match_distance,
        attack_direction=args.attack_direction,
        attacking_team_mode=args.attacking_team_mode,
        manual_attacking_team=args.manual_attacking_team,
        warmup_frames=args.warmup_frames,
    )
    pipeline.run(args.input_video, args.output_video)
