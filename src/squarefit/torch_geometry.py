"""PyTorch 版精确方形轴孔几何物理层。

这是主网络之后的不可训练层：它不含任何可学习参数，而是把网络的六维补偿
命令作用到隐藏真实位姿，再用 Rodrigues 刚体变换和四角点计算 ``g1~g4``、
八侧壁最小间隙和六维残余。所有张量均保留计算图，故 ``L_phys`` 可以反向
传播到逆向神经网络。

注意：``true_pose`` 只在仿真训练的物理损失侧使用，绝不作为网络输入；部署时
只能输入测得位姿，执行后通过复测得到新的状态。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


ARCSEC_TO_RAD = torch.pi / 648000.0
RAD_TO_ARCSEC = 648000.0 / torch.pi


def _skew(vector: Tensor) -> Tensor:
    """将 ``[B, 3]`` 旋转向量转换为叉乘矩阵。"""
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            torch.stack((zero, -z, y), dim=-1),
            torch.stack((z, zero, -x), dim=-1),
            torch.stack((-y, x, zero), dim=-1),
        ),
        dim=-2,
    )


def rotation_matrix_from_arcsec(rotation_arcsec: Tensor) -> Tensor:
    """用完整 Rodrigues 公式生成旋转矩阵。

    ``torch.sinc`` 在零角度处连续且可导，避免手写 ``sin(theta)/theta`` 的
    数值不稳定性。这里的三个角度是单一旋转向量分量，并非欧拉角。
    """
    rotation_rad = rotation_arcsec * ARCSEC_TO_RAD
    theta = torch.linalg.vector_norm(rotation_rad, dim=-1, keepdim=True)
    skew = _skew(rotation_rad)
    identity = torch.eye(3, dtype=rotation_arcsec.dtype, device=rotation_arcsec.device).expand(
        rotation_arcsec.shape[0], -1, -1
    )
    coefficient_one = torch.sinc(theta / torch.pi).unsqueeze(-1)
    coefficient_two = (0.5 * torch.sinc(theta / (2.0 * torch.pi)).square()).unsqueeze(-1)
    return identity + coefficient_one * skew + coefficient_two * (skew @ skew)


def rotation_vector_arcsec_from_matrix(rotation: Tensor) -> Tensor:
    """由小角度旋转矩阵恢复旋转向量，供六维残余损失使用。"""
    vee = torch.stack(
        (
            rotation[:, 2, 1] - rotation[:, 1, 2],
            rotation[:, 0, 2] - rotation[:, 2, 0],
            rotation[:, 1, 0] - rotation[:, 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(vee, dim=-1)
    cosine = ((rotation[:, 0, 0] + rotation[:, 1, 1] + rotation[:, 2, 2]) - 1.0) / 2.0
    angle = torch.atan2(sine, cosine.clamp(-1.0, 1.0))
    safe_sine = torch.where(sine.abs() < 1e-12, torch.ones_like(sine), sine)
    factor = angle / (2.0 * safe_sine)
    factor = torch.where(angle < 1e-6, torch.full_like(factor, 0.5), factor)
    return vee * factor.unsqueeze(-1) * RAD_TO_ARCSEC


@dataclass(frozen=True)
class TorchPhysicalState:
    """一次预测补偿后的、可微的几何与残余结果。"""

    pose_after: Tensor
    labelled_gaps_um: Tensor
    minimum_clearance_um: Tensor
    residual_translation_um: Tensor
    residual_rotation_arcsec: Tensor


class MeanActuatorLayer(nn.Module):
    """可配置的执行器均值模型，映射命令到实际平均调整量。

    该层用生成数据时公开的 M0 标定假设。它不是第二个神经网络；M1 的未建模
    偏差由目标域小样本适配吸收。随机执行误差不进入训练标签，只在最终闭环验证中
    通过复测处理。
    """

    def __init__(
        self,
        *,
        translation_coupling: tuple[tuple[float, float, float], ...] = (
            (0.985, 0.012, 0.000), (-0.010, 1.008, 0.000), (0.000, 0.000, 0.992)
        ),
        rotation_coupling: tuple[tuple[float, float, float], ...] = (
            (0.990, 0.010, 0.000), (-0.008, 1.006, 0.006), (0.000, -0.005, 0.995)
        ),
        translation_bias_um: tuple[float, float, float] = (0.025, -0.020, 0.015),
        rotation_bias_arcsec: tuple[float, float, float] = (0.04, -0.03, 0.025),
    ) -> None:
        super().__init__()
        self.register_buffer(
            "translation_coupling",
            torch.tensor(translation_coupling, dtype=torch.float64),
        )
        self.register_buffer(
            "rotation_coupling",
            torch.tensor(rotation_coupling, dtype=torch.float64),
        )
        self.register_buffer("translation_bias_um", torch.tensor(translation_bias_um, dtype=torch.float64))
        self.register_buffer("rotation_bias_arcsec", torch.tensor(rotation_bias_arcsec, dtype=torch.float64))

    def forward(self, command: Tensor) -> Tensor:
        translation = command[:, :3] @ self.translation_coupling.T + self.translation_bias_um
        rotation = command[:, 3:] @ self.rotation_coupling.T + self.rotation_bias_arcsec
        return torch.cat((translation, rotation), dim=-1)


class SquareFitPhysicsLayer(nn.Module):
    """网络输出命令 -> 名义执行器 -> 精确 Rodrigues 四角点几何。"""

    def __init__(
        self,
        *,
        insertion_depth_um: float = 10000.0,
        actuator: MeanActuatorLayer | None = None,
    ) -> None:
        super().__init__()
        self.insertion_depth_um = insertion_depth_um
        self.actuator = actuator if actuator is not None else MeanActuatorLayer()

    @staticmethod
    def _compose_world_adjustment(current_pose: Tensor, adjustment: Tensor) -> Tensor:
        """实现 ``T_new = T_adjust @ T_current`` 的左乘约定。"""
        current_rotation = rotation_matrix_from_arcsec(current_pose[:, 3:])
        adjustment_rotation = rotation_matrix_from_arcsec(adjustment[:, 3:])
        new_rotation = adjustment_rotation @ current_rotation
        new_translation = (
            adjustment_rotation @ current_pose[:, :3].unsqueeze(-1)
        ).squeeze(-1) + adjustment[:, :3]
        return torch.cat((new_translation, rotation_vector_arcsec_from_matrix(new_rotation)), dim=-1)

    def _wall_clearances(self, pose_after: Tensor, effective_dimensions: Tensor, station_um: float) -> Tensor:
        """返回入口或底部的 [right, left, top, bottom] 四壁最小间隙。"""
        half_shaft = effective_dimensions[:, 1] / 2.0
        corner_signs = torch.tensor(
            ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)),
            dtype=pose_after.dtype,
            device=pose_after.device,
        )
        local = torch.stack(
            (
                corner_signs[:, 0].unsqueeze(0) * half_shaft.unsqueeze(1),
                corner_signs[:, 1].unsqueeze(0) * half_shaft.unsqueeze(1),
                torch.full_like(half_shaft.unsqueeze(1), station_um).expand(-1, 4),
            ),
            dim=-1,
        )
        rotation = rotation_matrix_from_arcsec(pose_after[:, 3:])
        world = (rotation @ local.transpose(1, 2)).transpose(1, 2) + pose_after[:, None, :3]
        xs, ys = world[:, :, 0], world[:, :, 1]
        half_hole = effective_dimensions[:, 0] / 2.0
        return torch.stack(
            (
                half_hole - xs.amax(dim=1),
                xs.amin(dim=1) + half_hole,
                half_hole - ys.amax(dim=1),
                ys.amin(dim=1) + half_hole,
            ),
            dim=-1,
        )

    def evaluate_pose(self, pose: Tensor, effective_dimensions: Tensor) -> TorchPhysicalState:
        """直接评价一个当前位姿，不施加执行器命令。

        ``forward`` 的职责是“命令 -> 名义执行器 -> 几何”。在线闭环每次复测后
        还需要先评价*当前*测得位姿；此时若把零命令传给 ``forward``，仍会错误地
        叠加执行器零偏。故把纯几何评价单独公开，既避免这个错误，也让部署端不必
        接触任何 ``true_*`` 数据。
        """
        if pose.ndim != 2 or pose.shape[1] != 6:
            raise ValueError("pose 必须是 [batch, 6]，单位为微米和角秒。")
        if effective_dimensions.shape != (pose.shape[0], 2):
            raise ValueError("effective_dimensions 必须是 [batch, 2]，列为孔边长、轴边长。")
        entry = self._wall_clearances(pose, effective_dimensions, station_um=0.0)
        bottom = self._wall_clearances(pose, effective_dimensions, station_um=self.insertion_depth_um)
        labelled = torch.stack((entry[:, 2], entry[:, 0], bottom[:, 2], bottom[:, 0]), dim=-1)
        minimum = torch.cat((entry, bottom), dim=-1).amin(dim=-1)
        return TorchPhysicalState(
            pose_after=pose,
            labelled_gaps_um=labelled,
            minimum_clearance_um=minimum,
            residual_translation_um=pose[:, :3],
            residual_rotation_arcsec=pose[:, 3:],
        )

    def forward(self, true_pose: Tensor, command: Tensor, effective_dimensions: Tensor) -> TorchPhysicalState:
        if true_pose.ndim != 2 or true_pose.shape[1] != 6:
            raise ValueError("true_pose 必须是 [batch, 6]，单位为微米和角秒。")
        if command.shape != true_pose.shape:
            raise ValueError("command 与 true_pose 的形状必须一致。")
        if effective_dimensions.shape != (true_pose.shape[0], 2):
            raise ValueError("effective_dimensions 必须是 [batch, 2]，列为孔边长、轴边长。")
        adjustment = self.actuator(command)
        pose_after = self._compose_world_adjustment(true_pose, adjustment)
        return self.evaluate_pose(pose_after, effective_dimensions)
