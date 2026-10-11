"""方形轴孔补偿结果的统一评价指标。

本模块不包含神经网络。它把比赛中“补偿得好”的含义写成稳定、可复核的
数值指标，供三个地方共同使用：

* 虚拟装配器生成最优补偿标签时；
* 后续训练时构造几何物理约束损失时；
* 最终与解析法、普通 BP 网络进行公平对比时。

这里特别区分“合格”和“裕度最大”：所有标注间隙位于 2~4 um 只是合格；
离两个边界都越远，装配对下一次噪声或扰动就越稳健。
"""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import evaluate_lateral_fit, residual_pose_within_requirement
from .types import Dimensions, Pose6D
from .units import rad_to_arcsec


@dataclass(frozen=True)
class AssemblyObjective:
    """一次补偿后的质量指标。

    ``minimum_gap_margin_um`` 为四个标注间隙到 2~4 um 两个边界的最小距离。
    值越大越好；负值表示至少有一个标注间隙已经超出允许范围。
    """

    minimum_gap_margin_um: float
    mean_gap_center_deviation_um: float
    residual_translation_max_um: float
    residual_rotation_max_arcsec: float
    collision: bool
    gaps_qualified: bool
    residual_pose_qualified: bool

    @property
    def fully_qualified(self) -> bool:
        """同时满足间隙、无干涉和六维残余精度要求。"""
        return self.gaps_qualified and self.residual_pose_qualified and not self.collision


def evaluate_assembly_objective(dimensions: Dimensions, pose_after: Pose6D) -> AssemblyObjective:
    """评价补偿后位姿，而不假设其一定已居中。

    目标间隙中心取赛题区间的中点 3 um，仅作为“裕度”的解释指标；真正的
    合格判据仍然是每个 g 位于闭区间 [2, 4] um。
    """
    fit = evaluate_lateral_fit(dimensions, pose_after)
    gaps = fit.labelled_gaps_um
    target_gap = (dimensions.gap_lower_um + dimensions.gap_upper_um) / 2.0
    margin = min(
        min(gap - dimensions.gap_lower_um, dimensions.gap_upper_um - gap)
        for gap in gaps
    )
    return AssemblyObjective(
        minimum_gap_margin_um=margin,
        mean_gap_center_deviation_um=sum(abs(gap - target_gap) for gap in gaps) / len(gaps),
        residual_translation_max_um=max(
            abs(pose_after.dx_um), abs(pose_after.dy_um), abs(pose_after.dz_um)
        ),
        residual_rotation_max_arcsec=max(
            abs(rad_to_arcsec(pose_after.theta_x_rad)),
            abs(rad_to_arcsec(pose_after.theta_y_rad)),
            abs(rad_to_arcsec(pose_after.theta_z_rad)),
        ),
        collision=fit.collision,
        gaps_qualified=fit.labelled_gaps_qualified,
        residual_pose_qualified=residual_pose_within_requirement(dimensions, pose_after),
    )
