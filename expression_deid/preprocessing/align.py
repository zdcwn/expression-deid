"""把五个关键点相似变换到 RAF 的脸上布局，输出灰度对齐图。"""

from __future__ import annotations

import numpy as np
from PIL import Image

from expression_deid.config import (
    ALIGN_SIZE,
    FIT_ERROR_PX,
    HIGH_SCORE,
    MAX_ROLL_DEG,
    RAF_LANDMARKS_100,
)


def template_for(size: int = ALIGN_SIZE) -> np.ndarray:
    return np.asarray(RAF_LANDMARKS_100, dtype=np.float64) * (size / 100.0)


def landmarks_of(face: np.ndarray) -> np.ndarray:
    return np.asarray(face[4:14], dtype=np.float64).reshape(5, 2)


def similarity_matrix(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """src 到 dst 的相似变换，2×3。不估计反射。"""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if (src.ndim != 2 or src.shape[1:] != (2,) or src.shape != dst.shape
            or len(src) < 2 or not np.isfinite(src).all() or not np.isfinite(dst).all()):
        raise ValueError('相似变换需要两组等长、有限的二维关键点')
    count = src.shape[0]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_delta = src - src_mean
    dst_delta = dst - dst_mean
    src_var = (src_delta ** 2).sum() / count
    if src_var <= 1e-12 or float((dst_delta ** 2).sum()) <= 1e-12:
        raise ValueError('关键点重合，无法计算人脸相似变换')
    cov = dst_delta.T @ src_delta / count
    u, singular, vt = np.linalg.svd(cov)
    sign = np.eye(2)
    if np.linalg.det(u @ vt) < 0:
        sign[1, 1] = -1
    rotation = u @ sign @ vt
    scale = (singular @ np.diag(sign)).sum() / src_var
    translation = dst_mean - scale * (rotation @ src_mean)
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = translation
    return matrix


def warp_gray(gray: np.ndarray, landmarks: np.ndarray, size: int = ALIGN_SIZE) -> tuple[np.ndarray, float]:
    """返回对齐后的 uint8 灰度图，以及五点投到模板上的平均像素误差。"""
    template = template_for(size)
    matrix = similarity_matrix(landmarks, template)
    linear = matrix[:, :2]
    translation = matrix[:, 2]
    inverse = np.linalg.inv(linear)
    offset = -inverse @ translation
    # PIL 的 AFFINE 用输出像素反查输入像素，x 朝右、y 朝下。
    data = (
        float(inverse[0, 0]),
        float(inverse[0, 1]),
        float(offset[0]),
        float(inverse[1, 0]),
        float(inverse[1, 1]),
        float(offset[1]),
    )
    warped = Image.fromarray(gray, mode="L").transform(
        (size, size),
        Image.Transform.AFFINE,
        data,
        resample=Image.Resampling.BILINEAR,
        fillcolor=0,
    )
    projected = landmarks @ linear.T + translation
    fit_error = float(np.linalg.norm(projected - template, axis=1).mean())
    return np.asarray(warped), fit_error


def landmarks_ok(face: np.ndarray, image_hw: tuple[int, int]) -> bool:
    height, width = image_hw
    points = landmarks_of(face)
    if not np.isfinite(points).all():
        return False
    if (points[:, 0] < -0.05 * width).any() or (points[:, 0] > 1.05 * width).any():
        return False
    if (points[:, 1] < -0.05 * height).any() or (points[:, 1] > 1.05 * height).any():
        return False
    x, y, box_w, box_h = face[:4]
    margin = 0.2 * max(box_w, box_h)
    if (points[:, 0] < x - margin).any() or (points[:, 0] > x + box_w + margin).any():
        return False
    if (points[:, 1] < y - margin).any() or (points[:, 1] > y + box_h + margin).any():
        return False
    right_eye, left_eye, nose, right_mouth, left_mouth = points
    eye_dist = float(np.hypot(*(left_eye - right_eye)))
    if box_w <= 1 or not (0.2 * box_w <= eye_dist <= 0.9 * box_w):
        return False
    roll = abs(np.degrees(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])))
    if roll > MAX_ROLL_DEG:
        return False
    eye_y = (right_eye[1] + left_eye[1]) / 2
    mouth_y = (right_mouth[1] + left_mouth[1]) / 2
    if nose[1] < eye_y - 0.1 * eye_dist:
        return False
    if mouth_y < nose[1] - 0.15 * eye_dist:
        return False
    left_x, right_x = sorted((right_eye[0], left_eye[0]))
    if not (left_x - 0.5 * eye_dist <= nose[0] <= right_x + 0.5 * eye_dist):
        return False
    return True


def select_face(faces: np.ndarray, image_hw: tuple[int, int]) -> tuple[np.ndarray | None, str]:
    """多张脸时，在关键点可信的里面留面积最大的。分数够高的优先。

    返回 (face, reason)。reason 为空表示正常；`score_below_0.8` 表示只有低分脸可用。
    """
    if len(faces) == 0:
        return None, "no_face"

    def largest_usable(subset: np.ndarray) -> np.ndarray | None:
        usable = [face for face in subset if landmarks_ok(face, image_hw)]
        if not usable:
            return None
        stacked = np.stack(usable)
        area = stacked[:, 2] * stacked[:, 3]
        return stacked[int(np.argmax(area))]

    high = faces[faces[:, 14] >= HIGH_SCORE]
    chosen = largest_usable(high)
    if chosen is not None:
        return chosen, ""
    chosen = largest_usable(faces)
    if chosen is not None:
        return chosen, "score_below_0.8"
    return None, "landmarks"


def fit_is_extreme(fit_error: float) -> bool:
    return fit_error > FIT_ERROR_PX
