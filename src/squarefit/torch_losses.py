"""单逆网络的三合一可导训练损失。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .torch_geometry import TorchPhysicalState


@dataclass(frozen=True)
class LossTerms:
    total: Tensor
    data: Tensor
    gap: Tensor
    collision: Tensor
    residual: Tensor
    physics: Tensor
    optimality: Tensor


def compute_three_part_loss(
    *,
    normalized_prediction: Tensor,
    normalized_target: Tensor,
    predicted_command: Tensor,
    physical_state: TorchPhysicalState,
    teacher_safety_margin_um: Tensor,
    lambda_physics: float = 2.0,
    lambda_optimality: float = 0.05,
) -> LossTerms:
    """计算数据保真、物理约束与多解择优三部分损失。

    所有物理项都无量纲化：间隙越界按 2 um 归一、碰撞按 1 um 归一、
    平移/转角残余分别按 0.1 um/0.2 arcsec 归一。这样三个量级可以用
    预登记初值 ``L_data + 2 L_phys + 0.05 L_opt`` 合理组合。
    """
    data = torch.mean((normalized_prediction - normalized_target).square())
    gaps = physical_state.labelled_gaps_um
    gap = torch.mean(
        torch.relu((2.0 - gaps) / 2.0).square() + torch.relu((gaps - 4.0) / 2.0).square()
    )
    collision = torch.mean(torch.relu(-physical_state.minimum_clearance_um).square())
    translation = torch.relu(physical_state.residual_translation_um.abs() / 0.1 - 1.0).square().mean()
    rotation = torch.relu(physical_state.residual_rotation_arcsec.abs() / 0.2 - 1.0).square().mean()
    residual = translation + rotation
    physics = gap + collision + residual

    predicted_margin = torch.stack(
        (
            gaps - 2.0,
            4.0 - gaps,
            physical_state.minimum_clearance_um.unsqueeze(-1).expand_as(gaps),
        ),
        dim=-1,
    ).amin(dim=(-1, -2))
    margin_shortfall = torch.relu((teacher_safety_margin_um - predicted_margin) / 1.0).square().mean()
    # 仅作为多解之间的次级偏好，不能压过安全/残余约束。
    command_effort = (
        (predicted_command[:, :3] / 1.0).square().mean()
        + (predicted_command[:, 3:] / 1.0).square().mean()
    )
    optimality = margin_shortfall + 0.01 * command_effort
    total = data + lambda_physics * physics + lambda_optimality * optimality
    return LossTerms(
        total=total,
        data=data,
        gap=gap,
        collision=collision,
        residual=residual,
        physics=physics,
        optimality=optimality,
    )
