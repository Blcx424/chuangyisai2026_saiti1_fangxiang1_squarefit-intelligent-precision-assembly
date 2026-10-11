"""方形轴孔的理想解析首轮补偿器。

几何层已定义当前方轴相对方孔的位姿：

    p_world = R_current @ p_local + t_current

本模块求一个在方孔坐标系施加的调整 ``T_adjust``，使其满足：

    T_adjust @ T_current = T_target

因此：

    R_adjust = R_target @ R_current^T
    t_adjust = t_target - R_adjust @ t_current

若目标位姿为零且当前只有平移，调整量才简单等于 ``-dx, -dy, -dz``。
有旋转时，平移和旋转耦合，不能直接把六个数各自取负。

这是假设“测量坐标系、方孔坐标系和执行器调整坐标系已标定一致”时的理想
基线。后续的虚拟执行器会故意加入测量偏置、耦合和回差；含解析补偿先验输入的
物理约束 MLP 学习这些理想模型未覆盖的系统误差，并直接输出完整六维补偿命令。
"""

from __future__ import annotations

from .transforms import (
    Matrix3,
    Vector3,
    matrix_matrix_product,
    matrix_vector_product,
    rotation_matrix_from_pose,
    rotation_vector_from_matrix,
    transpose_matrix,
)
from .types import AnalyticCompensationResult, Pose6D
from .units import rad_to_arcsec


def _translation_of(pose: Pose6D) -> Vector3:
    return pose.dx_um, pose.dy_um, pose.dz_um


def _pose_from_transform(rotation: Matrix3, translation: Vector3) -> Pose6D:
    """把刚体变换重新表示为本项目统一的 ``Pose6D``。"""
    theta_x, theta_y, theta_z = rotation_vector_from_matrix(rotation)
    return Pose6D(
        dx_um=translation[0],
        dy_um=translation[1],
        dz_um=translation[2],
        theta_x_rad=theta_x,
        theta_y_rad=theta_y,
        theta_z_rad=theta_z,
    )


def apply_world_frame_adjustment(current_pose: Pose6D, adjustment: Pose6D) -> Pose6D:
    """在方孔坐标系施加一次调整，并返回更新后的方轴位姿。

    这里采用左乘：``T_new = T_adjust @ T_current``。这是本项目补偿命令的
    唯一约定，后续虚拟执行器、网络输出和报告公式都将沿用它。
    """
    adjustment_rotation = rotation_matrix_from_pose(adjustment)
    current_rotation = rotation_matrix_from_pose(current_pose)
    new_rotation = matrix_matrix_product(adjustment_rotation, current_rotation)
    rotated_current_translation = matrix_vector_product(
        adjustment_rotation,
        _translation_of(current_pose),
    )
    new_translation = tuple(
        rotated_current_translation[index] + _translation_of(adjustment)[index]
        for index in range(3)
    )
    return _pose_from_transform(new_rotation, new_translation)  # type: ignore[arg-type]


def analytic_compensation(
    measured_pose: Pose6D,
    target_pose: Pose6D | None = None,
) -> AnalyticCompensationResult:
    """由测得的六维误差精确反求解析补偿 ``u_geo``。

    默认目标是孔中心线完全对齐的零位姿。若未来需要预留特定的装配偏置或
    非零目标姿态，可通过 ``target_pose`` 显式给出。
    """
    if target_pose is None:
        target_pose = Pose6D()

    measured_rotation = rotation_matrix_from_pose(measured_pose)
    target_rotation = rotation_matrix_from_pose(target_pose)
    adjustment_rotation = matrix_matrix_product(
        target_rotation,
        transpose_matrix(measured_rotation),
    )
    rotated_measurement_translation = matrix_vector_product(
        adjustment_rotation,
        _translation_of(measured_pose),
    )
    adjustment_translation = tuple(
        _translation_of(target_pose)[index] - rotated_measurement_translation[index]
        for index in range(3)
    )
    adjustment = _pose_from_transform(adjustment_rotation, adjustment_translation)  # type: ignore[arg-type]
    predicted_pose_after = apply_world_frame_adjustment(measured_pose, adjustment)
    return AnalyticCompensationResult(
        measured_pose=measured_pose,
        target_pose=target_pose,
        adjustment=adjustment,
        predicted_pose_after=predicted_pose_after,
    )


def format_compensation_result(result: AnalyticCompensationResult) -> str:
    """生成适合 PyCharm 控制台和答辩演示的中文补偿说明。"""
    measured = result.measured_pose
    adjustment = result.adjustment
    after = result.predicted_pose_after

    def describe(name: str, pose: Pose6D) -> str:
        return (
            f"  {name}平移 [dx, dy, dz]（微米）："
            f"[{pose.dx_um:.6f}, {pose.dy_um:.6f}, {pose.dz_um:.6f}]\n"
            f"  {name}转角 [θx, θy, θz]（角秒）："
            f"[{rad_to_arcsec(pose.theta_x_rad):.6f}, "
            f"{rad_to_arcsec(pose.theta_y_rad):.6f}, "
            f"{rad_to_arcsec(pose.theta_z_rad):.6f}]"
        )

    return "\n".join(
        (
            "解析首轮补偿结果",
            describe("测得误差", measured),
            describe("解析补偿 u_geo", adjustment),
            describe("理想补偿后预测位姿", after),
        )
    )
