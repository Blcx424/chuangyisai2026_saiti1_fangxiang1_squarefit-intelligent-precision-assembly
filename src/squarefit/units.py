"""项目统一单位换算。

长度在几何与控制中统一使用微米（um），角度在计算中统一使用弧度（rad）。
报告展示时可以换算为毫米（mm）和角秒（arcsec）。
"""

from __future__ import annotations

import math

ARCSEC_TO_RAD = math.pi / (180.0 * 3600.0)


def arcsec_to_rad(value_arcsec: float) -> float:
    """将角秒转换为弧度。"""
    return value_arcsec * ARCSEC_TO_RAD


def rad_to_arcsec(value_rad: float) -> float:
    """将弧度转换为角秒。"""
    return value_rad / ARCSEC_TO_RAD


def mm_to_um(value_mm: float) -> float:
    """将毫米转换为微米。"""
    return value_mm * 1000.0


def um_to_mm(value_um: float) -> float:
    """将微米转换为毫米。"""
    return value_um / 1000.0
