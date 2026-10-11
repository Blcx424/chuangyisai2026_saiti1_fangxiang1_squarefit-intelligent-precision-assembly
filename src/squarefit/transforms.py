"""刚体位姿变换。

旋转采用旋转向量的 Rodrigues 公式。对于小角度，它与常用的一阶近似一致；
但本实现保留完整旋转，避免后续在较大扰动仿真中混淆旋转顺序。

本项目的明确约定：
theta = [theta_x, theta_y, theta_z] 是一个旋转向量，R = exp([theta]_x)。
它不是欧拉角，因而不采用 Rz @ Ry @ Rx 或任何其他“先绕哪一轴”的顺序。
任务书要求的角度很小，theta_x、theta_y、theta_z 在一阶近似下仍分别表示
绕 x、y、z 轴的小转角；报告和代码使用同一约定。
"""

from __future__ import annotations

import math

from .types import Pose6D

Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]


def rotation_matrix_from_pose(pose: Pose6D) -> Matrix3:
    """由单一旋转向量生成 3×3 旋转矩阵。

    这里不把 theta_x、theta_y、theta_z 当作三个按顺序相乘的欧拉角。
    """
    rx, ry, rz = pose.theta_x_rad, pose.theta_y_rad, pose.theta_z_rad
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1e-15:
        return (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )

    ux, uy, uz = rx / angle, ry / angle, rz / angle
    cosine = math.cos(angle)
    sine = math.sin(angle)
    one_minus_cosine = 1.0 - cosine

    return (
        (
            cosine + ux * ux * one_minus_cosine,
            ux * uy * one_minus_cosine - uz * sine,
            ux * uz * one_minus_cosine + uy * sine,
        ),
        (
            uy * ux * one_minus_cosine + uz * sine,
            cosine + uy * uy * one_minus_cosine,
            uy * uz * one_minus_cosine - ux * sine,
        ),
        (
            uz * ux * one_minus_cosine - uy * sine,
            uz * uy * one_minus_cosine + ux * sine,
            cosine + uz * uz * one_minus_cosine,
        ),
    )


def matrix_vector_product(matrix: Matrix3, vector: Vector3) -> Vector3:
    """计算矩阵与列向量的乘积。"""
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def transpose_matrix(matrix: Matrix3) -> Matrix3:
    """返回 3×3 矩阵的转置。旋转矩阵的逆矩阵恰为其转置。"""
    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def matrix_matrix_product(left: Matrix3, right: Matrix3) -> Matrix3:
    """计算两个 3×3 矩阵的乘积 ``left @ right``。"""
    return (
        (
            left[0][0] * right[0][0] + left[0][1] * right[1][0] + left[0][2] * right[2][0],
            left[0][0] * right[0][1] + left[0][1] * right[1][1] + left[0][2] * right[2][1],
            left[0][0] * right[0][2] + left[0][1] * right[1][2] + left[0][2] * right[2][2],
        ),
        (
            left[1][0] * right[0][0] + left[1][1] * right[1][0] + left[1][2] * right[2][0],
            left[1][0] * right[0][1] + left[1][1] * right[1][1] + left[1][2] * right[2][1],
            left[1][0] * right[0][2] + left[1][1] * right[1][2] + left[1][2] * right[2][2],
        ),
        (
            left[2][0] * right[0][0] + left[2][1] * right[1][0] + left[2][2] * right[2][0],
            left[2][0] * right[0][1] + left[2][1] * right[1][1] + left[2][2] * right[2][1],
            left[2][0] * right[0][2] + left[2][1] * right[1][2] + left[2][2] * right[2][2],
        ),
    )


def rotation_vector_from_matrix(matrix: Matrix3) -> Vector3:
    """由旋转矩阵恢复主值旋转向量。

    本项目的角度范围只有角秒级。这里对零附近的旋转做了稳定处理；接近
    180 度的旋转在精密装配赛题中不可能出现，因此显式拒绝，避免产生含糊的
    旋转轴结果。
    """
    cosine = (matrix[0][0] + matrix[1][1] + matrix[2][2] - 1.0) / 2.0
    cosine = max(-1.0, min(1.0, cosine))
    angle = math.acos(cosine)
    skew_vector = (
        matrix[2][1] - matrix[1][2],
        matrix[0][2] - matrix[2][0],
        matrix[1][0] - matrix[0][1],
    )

    if angle < 1e-12:
        # sin(angle)≈angle，因此 0.5*(R-R^T) 的向量部分就是旋转向量。
        return tuple(0.5 * value for value in skew_vector)  # type: ignore[return-value]
    if math.pi - angle < 1e-7:
        raise ValueError("当前补偿器不支持接近 180 度的旋转；赛题角度范围远小于该值。")

    factor = angle / (2.0 * math.sin(angle))
    return tuple(factor * value for value in skew_vector)  # type: ignore[return-value]


def transform_point(pose: Pose6D, local_point_um: Vector3) -> Vector3:
    """将方轴局部坐标中的一点变换至方孔坐标系。"""
    rotated = matrix_vector_product(rotation_matrix_from_pose(pose), local_point_um)
    return (
        rotated[0] + pose.dx_um,
        rotated[1] + pose.dy_um,
        rotated[2] + pose.dz_um,
    )


def inverse_transform_point(pose: Pose6D, world_point_um: Vector3) -> Vector3:
    """将方孔坐标系中的一点变回方轴局部坐标系。"""
    shifted = (
        world_point_um[0] - pose.dx_um,
        world_point_um[1] - pose.dy_um,
        world_point_um[2] - pose.dz_um,
    )
    rotation = rotation_matrix_from_pose(pose)
    # 正交旋转矩阵的逆矩阵为转置矩阵。
    return (
        rotation[0][0] * shifted[0] + rotation[1][0] * shifted[1] + rotation[2][0] * shifted[2],
        rotation[0][1] * shifted[0] + rotation[1][1] * shifted[1] + rotation[2][1] * shifted[2],
        rotation[0][2] * shifted[0] + rotation[1][2] * shifted[1] + rotation[2][2] * shifted[2],
    )
