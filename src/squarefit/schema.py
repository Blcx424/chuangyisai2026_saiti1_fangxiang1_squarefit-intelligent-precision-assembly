"""神经网络开始前固定的数据列约定。

此文件故意不依赖 NumPy 或 PyTorch。现在先固定“未来每一行 CSV 的什么列是
输入、什么列是主标签”，可避免开始训练后才发现把真值或测试答案泄漏给网络。
"""

from __future__ import annotations

from collections.abc import Iterable


# 特征全集由三块组成：六维传感测量、零件实测尺寸、可由解析模型得到的物理先验。
# 每个实验组只能从本文件定义的集合中选列，避免不同方法不小心使用了不同信息。
MEASURED_POSE_COLUMNS: tuple[str, ...] = (
    "measured_dx_um",
    "measured_dy_um",
    "measured_dz_um",
    "measured_theta_x_arcsec",
    "measured_theta_y_arcsec",
    "measured_theta_z_arcsec",
)

DIMENSION_COLUMNS: tuple[str, ...] = (
    "hole_side_um",
    "shaft_side_um",
)

NOMINAL_CLEARANCE_COLUMNS: tuple[str, ...] = (
    "nominal_single_side_clearance_um",
)

# 这是轴孔选配后、在题设 20 °C 等温条件下的单边间隙。任务书规定铜轴、铜孔
# 热膨胀系数匹配，并明确“仅位姿误差影响间隙”，因此它是主赛题模型允许使用的
# 静态尺寸特征；不把温度扰动混入主模型。
TASK_OPERATING_CLEARANCE_COLUMNS: tuple[str, ...] = NOMINAL_CLEARANCE_COLUMNS

# 下列列仅保留给历史热工况扩展数据集，绝不作为任务书主模型的输入。
# 由 20 °C 参考尺寸、材料线膨胀系数和*测得*孔/轴温度计算得到的运行间隙估计。
# 它不是 true_effective_*；温度读数含传感器偏置与噪声，故只允许扩展实验使用。
ESTIMATED_OPERATING_CLEARANCE_COLUMNS: tuple[str, ...] = (
    "estimated_effective_single_side_clearance_um",
)

ANALYTIC_COMMAND_COLUMNS: tuple[str, ...] = (
    "analytic_command_dx_um",
    "analytic_command_dy_um",
    "analytic_command_dz_um",
    "analytic_command_theta_x_arcsec",
    "analytic_command_theta_y_arcsec",
    "analytic_command_theta_z_arcsec",
)

# 温度是可测工况，不是仿真真值；它只服务于历史热工况扩展，非主赛题输入。
THERMAL_MEASUREMENT_COLUMNS: tuple[str, ...] = (
    "measured_hole_temperature_c",
    "measured_shaft_temperature_c",
)

# A/B/C 三组是任务书主模型的固定特征子集：A 只用可测六维误差和实测尺寸；
# B 加入选配产生的静态单边间隙；C 再加入解析首轮补偿先验。这样既不把温度扩展
# 偷渡为主变量，也能检验“尺寸/间隙先验”和“解析先验”各自的贡献。
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "A_基础BP_8维": (
        *MEASURED_POSE_COLUMNS,
        *DIMENSION_COLUMNS,
    ),
    "B_选配间隙BP_9维": (
        *MEASURED_POSE_COLUMNS,
        *DIMENSION_COLUMNS,
        *TASK_OPERATING_CLEARANCE_COLUMNS,
    ),
    "C_解析先验BP_15维": (
        *MEASURED_POSE_COLUMNS,
        *DIMENSION_COLUMNS,
        *TASK_OPERATING_CLEARANCE_COLUMNS,
        *ANALYTIC_COMMAND_COLUMNS,
    ),
}

# 默认主特征为 B 组 9 维：六维测量、实测轴孔尺寸和选配单边间隙。
NETWORK_INPUT_COLUMNS: tuple[str, ...] = FEATURE_SETS["B_选配间隙BP_9维"]
NETWORK_INPUT_COLUMNS_WITHOUT_OPERATING_CLEARANCE: tuple[str, ...] = FEATURE_SETS["A_基础BP_8维"]

# 主方案：一个网络直接输出完整六维补偿命令 u*。
PRIMARY_TARGET_COLUMNS: tuple[str, ...] = (
    "label_optimal_command_dx_um",
    "label_optimal_command_dy_um",
    "label_optimal_command_dz_um",
    "label_optimal_command_theta_x_arcsec",
    "label_optimal_command_theta_y_arcsec",
    "label_optimal_command_theta_z_arcsec",
)

# 仅用于“解析补偿 + 残差网络”对照组，不是主方案标签。
RESIDUAL_BASELINE_TARGET_COLUMNS: tuple[str, ...] = (
    "label_residual_command_dx_um",
    "label_residual_command_dy_um",
    "label_residual_command_dz_um",
    "label_residual_command_theta_x_arcsec",
    "label_residual_command_theta_y_arcsec",
    "label_residual_command_theta_z_arcsec",
)


# 这些列只允许用于离线造标签、质量门禁或物理损失，不得作为网络输入。
# 训练脚本必须调用 ``validated_training_columns``，不能以“去掉标签列”方式
# 自动猜测输入列；后者会把仿真真值和装配后答案意外喂给网络。
FORBIDDEN_NETWORK_INPUT_PREFIXES: tuple[str, ...] = (
    "true_",
    "physical_adjustment_",
    "before_",
    "after_",
    "label_",
)


def validated_training_columns(
    feature_set_name: str,
    available_columns: Iterable[str],
    *,
    target_columns: tuple[str, ...] = PRIMARY_TARGET_COLUMNS,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """返回经过白名单核验的输入列与标签列。

    此函数是后续 PyTorch DataLoader 的唯一取列入口。它只从 A/B/C 的
    显式白名单中取输入，哪怕 CSV 同时保存了 ``true_*``、装配后间隙和标签
    验证列，也不会发生“drop target 后剩下什么就喂什么”的数据泄露。
    """
    try:
        feature_columns = FEATURE_SETS[feature_set_name]
    except KeyError as error:
        options = "、".join(FEATURE_SETS)
        raise KeyError(f"未知特征组 {feature_set_name!r}；可选：{options}。") from error

    if any(column.startswith(FORBIDDEN_NETWORK_INPUT_PREFIXES) for column in feature_columns):
        raise AssertionError("特征白名单包含仿真真值、标签或装配后验证列。")

    available = set(available_columns)
    missing_features = tuple(column for column in feature_columns if column not in available)
    missing_targets = tuple(column for column in target_columns if column not in available)
    if missing_features or missing_targets:
        details: list[str] = []
        if missing_features:
            details.append(f"缺少输入列：{', '.join(missing_features)}")
        if missing_targets:
            details.append(f"缺少标签列：{', '.join(missing_targets)}")
        raise ValueError("；".join(details))

    return feature_columns, target_columns
