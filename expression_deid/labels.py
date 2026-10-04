"""RAF 和 KMU 用同一套表情编号。编号跟 RAF 的文件夹一致，表情模型直接拿来当类别。"""

from __future__ import annotations

import re

# 下标就是类别 id。KMU 没有中性。
EXPRESSION_NAMES = (
    "surprise",
    "fear",
    "disgust",
    "happiness",
    "sadness",
    "anger",
    "neutral",
)

EXPRESSION_ZH = ("惊讶", "恐惧", "厌恶", "高兴", "悲伤", "愤怒", "中性")

KMU_CODE_TO_ID = {
    "SU": 0,
    "FE": 1,
    "DI": 2,
    "HA": 3,
    "SA": 4,
    "AN": 5,
}

_KMU_NAME = re.compile(r"^(\d{2})_([A-Z]{2})_([A-Za-z0-9]+)_(\d+)$")


def raf_label(folder_id: int) -> int:
    if folder_id < 0 or folder_id >= len(EXPRESSION_NAMES):
        raise ValueError(f"RAF 文件夹编号超出范围: {folder_id}")
    return folder_id


def parse_kmu_stem(stem: str) -> dict:
    """文件名 `{人}_{表情}_{缩写}_{帧}`。对不上时字段留空，调用方自己决定要不要用这张图。"""
    match = _KMU_NAME.match(stem)
    if match is None:
        return {"person": "", "expression": "", "name": "", "frame": -1, "label": -1}
    code = match.group(2)
    return {
        "person": match.group(1),
        "expression": code,
        "name": match.group(3),
        "frame": int(match.group(4)),
        "label": KMU_CODE_TO_ID.get(code, -1),
    }
