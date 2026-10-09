"""路径和已经定下来的处理参数。改这里会影响后面的对齐和读图。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets"
RAF_DIR = DATA_DIR / "raf-db"
KMU_DIR = DATA_DIR / "kmu-fed"
KMU_ALIGNED_DIR = DATA_DIR / "kmu-fed-aligned"
FERPLUS_DIR = DATA_DIR / "fer2013-hf"
PPB_DIR = DATA_DIR / "ppb-emo"
PPB_ALIGNED_DIR = PPB_DIR / "neutral-aligned"

MODEL_PATH = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
STATS_PATH = Path(__file__).resolve().parent / "norm_stats.json"

# KMU 对齐后的边长。表情模型读的时候再缩到 RAF 的 100。
ALIGN_SIZE = 256
EXPR_SIZE = 100

# YuNet 输入固定 640。解码和 OpenCV FaceDetectorYN 一致。
YUNET_INPUT = 640
YUNET_STRIDES = (8, 16, 32)
DETECT_SCORE = 0.5
HIGH_SCORE = 0.8
NMS_IOU = 0.3
NMS_TOP_K = 5000

# 原图里眼睛中点相对前后帧跳开、下一帧又跳回来，才记跳动。
# 人在一段表情里慢慢挪，不算。对齐之后眼睛落在模板上，不靠这个数检查。
JITTER_PX = 40.0
# 五个点投到模板上的平均误差。太大说明这组点不像一张脸。
FIT_ERROR_PX = 25.0
# PPB's central-IR frames are lower contrast than RAF.  A looser detector
# threshold is useful for finding a face, but a frame with this much landmark
# residual is usually a partial crop (for example, only the nose and mouth).
# Keep it out of training and retain it in the manifest for audit.
PPB_FIT_ERROR_PX = 12.0
MAX_ROLL_DEG = 40.0

# 只在 RAF 训练集上用。验证集和 KMU 不开。
AUG_ROTATION_DEG = 10.0
AUG_BRIGHTNESS = 0.15

# RAF 训练集上、YuNet 在 276 张里测到的五点中位数，坐标系是官方 100×100。
# 顺序：右眼、左眼、鼻尖、右嘴角、左嘴角。右眼左眼是脸上的左右，不是画面左右。
# 眼距约 0.435 倍画面宽。官方裁切比常见的三分之一眼距更紧，模板跟这个数走。
RAF_LANDMARKS_100 = (
    (25.75, 32.97),
    (69.23, 31.91),
    (49.07, 54.86),
    (30.41, 70.99),
    (67.45, 70.00),
)
