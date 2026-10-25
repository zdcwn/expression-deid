"""YuNet 人脸检测。输出和 OpenCV FaceDetectorYN 同一套列含义。

每行 15 个数：x, y, w, h，然后右眼、左眼、鼻尖、右嘴角、左嘴角，最后是分数。
坐标在原图像素里。模型文件不进仓库，第一次运行时下载。
"""

from __future__ import annotations

import urllib.request

import numpy as np
import onnxruntime as ort
from PIL import Image

from expression_deid.config import (
    DETECT_SCORE,
    MODEL_PATH,
    MODEL_URL,
    NMS_IOU,
    NMS_TOP_K,
    YUNET_INPUT,
    YUNET_STRIDES,
)


def ensure_model(path=MODEL_PATH) -> None:
    if path.exists() and path.stat().st_size > 100_000:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, path)


class YuNet:
    def __init__(self, model_path=MODEL_PATH):
        ensure_model(model_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, rgb: np.ndarray, score_threshold: float = DETECT_SCORE) -> np.ndarray:
        blob, scale = _letterbox_bgr(rgb)
        outputs = self.session.run(None, {self.input_name: blob})
        return _decode(outputs, scale, score_threshold)


def _letterbox_bgr(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = rgb.shape[:2]
    scale = YUNET_INPUT / max(height, width)
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = np.asarray(
        Image.fromarray(rgb).resize((resized_w, resized_h), Image.Resampling.BILINEAR)
    )
    canvas = np.zeros((YUNET_INPUT, YUNET_INPUT, 3), np.float32)
    # OpenCV 读图是 BGR，blobFromImage 不交换通道，也不除以 255。
    canvas[:resized_h, :resized_w] = resized[:, :, ::-1]
    blob = canvas.transpose(2, 0, 1)[None]
    return blob, scale


def _nms(boxes: np.ndarray, scores: np.ndarray) -> list[int]:
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = np.maximum(boxes[:, 2], 0) * np.maximum(boxes[:, 3], 0)
    order = scores.argsort()[::-1][:NMS_TOP_K]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[current] + areas[rest] - inter + 1e-6
        order = rest[inter / union <= NMS_IOU]
    return keep


def _decode(outputs, scale: float, score_threshold: float) -> np.ndarray:
    names = [f"{kind}_{stride}" for kind in ("cls", "obj", "bbox", "kps") for stride in YUNET_STRIDES]
    by_name = {name: outputs[index][0] for index, name in enumerate(names)}
    faces = []
    for stride in YUNET_STRIDES:
        cols = YUNET_INPUT // stride
        cls = np.clip(by_name[f"cls_{stride}"].reshape(-1), 0.0, 1.0)
        obj = np.clip(by_name[f"obj_{stride}"].reshape(-1), 0.0, 1.0)
        score = np.sqrt(cls * obj)
        bbox = by_name[f"bbox_{stride}"].reshape(-1, 4)
        kps = by_name[f"kps_{stride}"].reshape(-1, 10)
        for index in np.flatnonzero(score >= score_threshold):
            row_i, col_i = divmod(int(index), cols)
            cx = (col_i + bbox[index, 0]) * stride
            cy = (row_i + bbox[index, 1]) * stride
            box_w = np.exp(np.clip(bbox[index, 2], -8.0, 8.0)) * stride
            box_h = np.exp(np.clip(bbox[index, 3], -8.0, 8.0)) * stride
            face = np.empty(15, np.float64)
            face[0] = (cx - box_w / 2) / scale
            face[1] = (cy - box_h / 2) / scale
            face[2] = box_w / scale
            face[3] = box_h / scale
            for point in range(5):
                face[4 + 2 * point] = (kps[index, 2 * point] + col_i) * stride / scale
                face[5 + 2 * point] = (kps[index, 2 * point + 1] + row_i) * stride / scale
            face[14] = score[index]
            faces.append(face)
    if not faces:
        return np.zeros((0, 15), np.float64)
    stacked = np.stack(faces)
    return stacked[_nms(stacked[:, :4], stacked[:, 14])]
