"""方形轴孔装配的可解释虚拟装配器与数据集生成器。

本模块不是神经网络，也不宣称替代真实设备。它的职责是把已经验证过的
几何模型放进一个可控的“虚拟实验台”中，显式区分五件事：

1. ``q_true``：方轴真实的初始六维位姿误差；
2. ``q_meas``：传感器带偏置、噪声后测得的位姿；
3. ``u_geo``：解析模型根据 ``q_meas`` 给出的首轮补偿；
4. 实际执行器动作：存在比例误差、轴间耦合、零偏和随机扰动；
5. ``q_after``：执行首轮补偿后的真实残余位姿。

未来残差网络的监督标签不是凭空编造的数，而是“若完全知道真实状态和
虚拟执行器均值模型时，为达到目标位姿还应追加的命令”。真实硬件阶段则要
把这个标签改为高精度复测得到的残余误差或末端质量标签。

当前参数是用于验证软件流程的*可调整假设*，不是比赛设备的实测参数。
每一个参数都集中在本文件，后续获得传感器/执行器标定数据后可替换。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .compensation import analytic_compensation, apply_world_frame_adjustment
from .geometry import evaluate_lateral_fit, residual_pose_within_requirement
from .objectives import AssemblyObjective, evaluate_assembly_objective
from .schema import FEATURE_SETS, PRIMARY_TARGET_COLUMNS
from .types import AssemblyResult, Dimensions, Pose6D
from .units import arcsec_to_rad, rad_to_arcsec


Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]


def _add_vectors(left: Vector3, right: Vector3) -> Vector3:
    return tuple(left[index] + right[index] for index in range(3))  # type: ignore[return-value]


def _subtract_vectors(left: Vector3, right: Vector3) -> Vector3:
    return tuple(left[index] - right[index] for index in range(3))  # type: ignore[return-value]


def _matrix_vector_product(matrix: Matrix3, vector: Vector3) -> Vector3:
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def _inverse_matrix(matrix: Matrix3) -> Matrix3:
    """返回一个 3×3 非奇异矩阵的逆；用于执行器均值模型的反算。"""
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(determinant) < 1e-12:
        raise ValueError("执行器耦合矩阵不可逆，不能反算理论最优命令。")
    inverse_determinant = 1.0 / determinant
    return (
        ((e * i - f * h) * inverse_determinant, (c * h - b * i) * inverse_determinant, (b * f - c * e) * inverse_determinant),
        ((f * g - d * i) * inverse_determinant, (a * i - c * g) * inverse_determinant, (c * d - a * f) * inverse_determinant),
        ((d * h - e * g) * inverse_determinant, (b * g - a * h) * inverse_determinant, (a * e - b * d) * inverse_determinant),
    )


def _pose_from_parts(translation_um: Vector3, rotation_rad: Vector3) -> Pose6D:
    return Pose6D(
        dx_um=translation_um[0],
        dy_um=translation_um[1],
        dz_um=translation_um[2],
        theta_x_rad=rotation_rad[0],
        theta_y_rad=rotation_rad[1],
        theta_z_rad=rotation_rad[2],
    )


def _translation_of(pose: Pose6D) -> Vector3:
    return pose.dx_um, pose.dy_um, pose.dz_um


def _rotation_of(pose: Pose6D) -> Vector3:
    return pose.theta_x_rad, pose.theta_y_rad, pose.theta_z_rad


def _pose_component_difference(left: Pose6D, right: Pose6D) -> Pose6D:
    """按命令坐标的六个分量求差。

    仅用于角秒级小转角的命令增量 ``delta_u``；实际刚体组合仍统一由
    :func:`apply_world_frame_adjustment` 完成，不会在几何判定中用此近似。
    """
    return _pose_from_parts(
        _subtract_vectors(_translation_of(left), _translation_of(right)),
        _subtract_vectors(_rotation_of(left), _rotation_of(right)),
    )


@dataclass(frozen=True)
class MeasurementModel:
    """六维测量模型：固定零偏 + 每次测量的高斯随机噪声。"""

    translation_bias_um: Vector3 = (0.04, -0.03, 0.02)
    translation_noise_std_um: float = 0.025
    rotation_bias_arcsec: Vector3 = (0.05, -0.04, 0.03)
    rotation_noise_std_arcsec: float = 0.03

    def measure(self, true_pose: Pose6D, rng: random.Random) -> Pose6D:
        rotation_bias_rad = tuple(arcsec_to_rad(value) for value in self.rotation_bias_arcsec)
        translation_noise = tuple(rng.gauss(0.0, self.translation_noise_std_um) for _ in range(3))
        rotation_noise = tuple(
            arcsec_to_rad(rng.gauss(0.0, self.rotation_noise_std_arcsec)) for _ in range(3)
        )
        return _pose_from_parts(
            _add_vectors(_add_vectors(_translation_of(true_pose), self.translation_bias_um), translation_noise),
            _add_vectors(_add_vectors(_rotation_of(true_pose), rotation_bias_rad), rotation_noise),
        )


@dataclass(frozen=True)
class ThermalCondition:
    """孔、轴在一次装配开始时的真实温度，单位为摄氏度。"""

    hole_temperature_c: float
    shaft_temperature_c: float


@dataclass(frozen=True)
class ThermalModel:
    """均匀温度场下的线膨胀模型与温度传感器模型。

    尺寸公差仍统一定义在 ``reference_temperature_c``（20 °C）下。实际侧向
    几何计算使用 ``D(T)=D_ref[1+alpha(T-T_ref)]``。本类默认的钢类参数
    ``alpha=11.5e-6 /°C`` 只服务于历史热工况扩展，不是赛题主数据假设；
    主数据统一由 ``make_task_isothermal_copper_model`` 创建等温、同材铜件。
    本类描述均匀温度造成的边长变化；温度梯度、热弯曲和摩擦热留待真实标定后扩展。
    ``estimated_effective_dimensions`` 使用温度传感器读数而非真实温度，是部署时
    可以计算的运行尺寸估计；它与真实有效尺寸严格分开，避免信息泄漏。
    """

    reference_temperature_c: float = 20.0
    hole_linear_expansion_per_c: float = 11.5e-6
    shaft_linear_expansion_per_c: float = 11.5e-6
    hole_temperature_range_c: tuple[float, float] = (18.0, 22.0)
    shaft_temperature_range_c: tuple[float, float] = (18.0, 22.0)
    hole_temperature_bias_c: float = 0.05
    shaft_temperature_bias_c: float = -0.04
    temperature_noise_std_c: float = 0.10

    def __post_init__(self) -> None:
        if self.hole_temperature_range_c[0] > self.hole_temperature_range_c[1]:
            raise ValueError("孔温度范围下限不能大于上限。")
        if self.shaft_temperature_range_c[0] > self.shaft_temperature_range_c[1]:
            raise ValueError("轴温度范围下限不能大于上限。")
        if self.hole_linear_expansion_per_c < 0 or self.shaft_linear_expansion_per_c < 0:
            raise ValueError("线膨胀系数不能为负数。")
        if self.temperature_noise_std_c < 0:
            raise ValueError("温度测量噪声标准差不能为负数。")

    def sample_true_condition(self, rng: random.Random) -> ThermalCondition:
        return ThermalCondition(
            hole_temperature_c=rng.uniform(*self.hole_temperature_range_c),
            shaft_temperature_c=rng.uniform(*self.shaft_temperature_range_c),
        )

    def coverage_true_condition(self, index: int) -> ThermalCondition:
        """为极少量 M1 标定集提供温度空间的低差异覆盖。"""
        hole = self.hole_temperature_range_c[0] + (
            self.hole_temperature_range_c[1] - self.hole_temperature_range_c[0]
        ) * _halton(index, 31)
        shaft = self.shaft_temperature_range_c[0] + (
            self.shaft_temperature_range_c[1] - self.shaft_temperature_range_c[0]
        ) * _halton(index, 37)
        return ThermalCondition(hole_temperature_c=hole, shaft_temperature_c=shaft)

    def measure(self, true_condition: ThermalCondition, rng: random.Random) -> ThermalCondition:
        return ThermalCondition(
            hole_temperature_c=(
                true_condition.hole_temperature_c
                + self.hole_temperature_bias_c
                + rng.gauss(0.0, self.temperature_noise_std_c)
            ),
            shaft_temperature_c=(
                true_condition.shaft_temperature_c
                + self.shaft_temperature_bias_c
                + rng.gauss(0.0, self.temperature_noise_std_c)
            ),
        )

    def effective_dimensions(self, reference_dimensions: Dimensions, condition: ThermalCondition) -> Dimensions:
        """将 20 °C 尺寸换算为当前真实温度下用于间隙计算的有效尺寸。"""
        hole_scale = 1.0 + self.hole_linear_expansion_per_c * (
            condition.hole_temperature_c - self.reference_temperature_c
        )
        shaft_scale = 1.0 + self.shaft_linear_expansion_per_c * (
            condition.shaft_temperature_c - self.reference_temperature_c
        )
        return replace(
            reference_dimensions,
            hole_side_um=reference_dimensions.hole_side_um * hole_scale,
            shaft_side_um=reference_dimensions.shaft_side_um * shaft_scale,
        )

    def estimated_effective_dimensions(
        self,
        reference_dimensions: Dimensions,
        measured_condition: ThermalCondition,
    ) -> Dimensions:
        """由可测温度估计运行尺寸。

        该函数故意复用同一条公开热膨胀关系，但输入是带偏置/噪声的温度读数；
        因此输出是网络部署时允许使用的工程特征，不是 ``true_effective_*`` 真值。
        """
        return self.effective_dimensions(reference_dimensions, measured_condition)

    def _scale_factors(self, condition: ThermalCondition) -> tuple[float, float]:
        """返回孔、轴由参考温度换算到真实温度的比例。"""
        return (
            1.0 + self.hole_linear_expansion_per_c * (condition.hole_temperature_c - self.reference_temperature_c),
            1.0 + self.shaft_linear_expansion_per_c * (condition.shaft_temperature_c - self.reference_temperature_c),
        )


# 任务书主场景：紫铜方孔/方轴，材料线膨胀系数匹配；装配在统一的 20 °C
# 参考环境下开展。因孔、轴温度相同且等于参考温度，有效尺寸严格等于实测参考
# 尺寸，故间隙的动态变化只来自六维位姿，而不是自加的热工况。
TASK_REFERENCE_TEMPERATURE_C = 20.0
TASK_COPPER_LINEAR_EXPANSION_PER_C = 16.5e-6


def make_task_isothermal_copper_model() -> ThermalModel:
    """返回任务书主数据唯一允许使用的等温、同材热模型。

    仍把铜的匹配热膨胀系数写入 metadata，便于追溯材料假设；但温差为零，
    所以这不是温度扰动模型。历史 v3 热工况模型保留为独立扩展，不覆盖本主线。
    """
    return ThermalModel(
        reference_temperature_c=TASK_REFERENCE_TEMPERATURE_C,
        hole_linear_expansion_per_c=TASK_COPPER_LINEAR_EXPANSION_PER_C,
        shaft_linear_expansion_per_c=TASK_COPPER_LINEAR_EXPANSION_PER_C,
        hole_temperature_range_c=(TASK_REFERENCE_TEMPERATURE_C, TASK_REFERENCE_TEMPERATURE_C),
        shaft_temperature_range_c=(TASK_REFERENCE_TEMPERATURE_C, TASK_REFERENCE_TEMPERATURE_C),
        hole_temperature_bias_c=0.0,
        shaft_temperature_bias_c=0.0,
        temperature_noise_std_c=0.0,
    )


def thermal_clearance_sensitivity(
    reference_dimensions: Dimensions | None = None,
) -> tuple[dict[str, float | str], ...]:
    """返回三种公开材料假设下的热间隙解析敏感性。

    这是报告中的“材料参数敏感性”准备工具，并不改变 M0/M1 主数据集，也不把
    未核验的材料类型伪装成赛场实物。默认取 10 006 / 10 000 um 的代表性轴孔对，
    比较孔、轴分别处于 30 / 10 °C 的 20 °C 温差极端；结果只回答“热间隙可能
    改变多少”，不能替代后续真实材料标定或网络端到端性能实验。
    """
    reference = reference_dimensions or Dimensions(hole_side_um=10006.0, shaft_side_um=10000.0)
    cases = (
        ("殷钢—殷钢（低膨胀敏感性）", 1.2e-6, 1.2e-6),
        ("钢类—钢类（历史热工况扩展）", 11.5e-6, 11.5e-6),
        ("铝合金孔—钢类轴（异材压力假设）", 23.0e-6, 11.5e-6),
    )
    rows: list[dict[str, float | str]] = []
    for name, hole_alpha, shaft_alpha in cases:
        model = ThermalModel(
            hole_linear_expansion_per_c=hole_alpha,
            shaft_linear_expansion_per_c=shaft_alpha,
        )
        for condition_name, condition in (
            ("孔热轴冷：孔30°C、轴10°C", ThermalCondition(30.0, 10.0)),
            ("孔冷轴热：孔10°C、轴30°C", ThermalCondition(10.0, 30.0)),
        ):
            effective = model.effective_dimensions(reference, condition)
            rows.append(
                {
                    "material_assumption": name,
                    "condition": condition_name,
                    "hole_alpha_per_c": hole_alpha,
                    "shaft_alpha_per_c": shaft_alpha,
                    "reference_clearance_um": reference.nominal_single_side_clearance_um,
                    "effective_clearance_um": effective.nominal_single_side_clearance_um,
                    "clearance_shift_um": (
                        effective.nominal_single_side_clearance_um
                        - reference.nominal_single_side_clearance_um
                    ),
                }
            )
    return tuple(rows)


@dataclass(frozen=True)
class ActuatorModel:
    """虚拟执行器的均值模型与随机误差模型。

    ``translation_coupling`` 和 ``rotation_coupling`` 的对角线代表每一轴的
    比例误差，非对角线代表弱轴间耦合。例如 x 命令会少量带动 y。
    ``mean_physical_adjustment`` 是“命令→真实调整”的均值关系；随机噪声只
    加在 ``sample_physical_adjustment`` 中，避免把不可预测随机项当成网络标签。
    """

    translation_coupling: Matrix3 = (
        (0.985, 0.012, 0.000),
        (-0.010, 1.008, 0.000),
        (0.000, 0.000, 0.992),
    )
    rotation_coupling: Matrix3 = (
        (0.990, 0.010, 0.000),
        (-0.008, 1.006, 0.006),
        (0.000, -0.005, 0.995),
    )
    translation_bias_um: Vector3 = (0.025, -0.020, 0.015)
    rotation_bias_arcsec: Vector3 = (0.04, -0.03, 0.025)
    translation_noise_std_um: float = 0.015
    rotation_noise_std_arcsec: float = 0.02

    def mean_physical_adjustment(self, command: Pose6D) -> Pose6D:
        """不含随机噪声的实际调整量。"""
        rotation_bias_rad = tuple(arcsec_to_rad(value) for value in self.rotation_bias_arcsec)
        return _pose_from_parts(
            _add_vectors(_matrix_vector_product(self.translation_coupling, _translation_of(command)), self.translation_bias_um),
            _add_vectors(_matrix_vector_product(self.rotation_coupling, _rotation_of(command)), rotation_bias_rad),
        )

    def sample_physical_adjustment(self, command: Pose6D, rng: random.Random) -> Pose6D:
        """一次真实执行：均值调整量再叠加不可预测的执行器随机扰动。"""
        mean = self.mean_physical_adjustment(command)
        translation_noise = tuple(rng.gauss(0.0, self.translation_noise_std_um) for _ in range(3))
        rotation_noise = tuple(
            arcsec_to_rad(rng.gauss(0.0, self.rotation_noise_std_arcsec)) for _ in range(3)
        )
        return _pose_from_parts(
            _add_vectors(_translation_of(mean), translation_noise),
            _add_vectors(_rotation_of(mean), rotation_noise),
        )

    def inverse_mean_command(self, desired_physical_adjustment: Pose6D) -> Pose6D:
        """反算产生指定平均物理调整量所需的理想命令。

        这是仿真时的“教师/真值”工具。真实系统中不能直接使用它，因为真实
        执行器耦合和偏置未知；这恰好是未来小样本标定与残差网络的意义。
        """
        rotation_bias_rad = tuple(arcsec_to_rad(value) for value in self.rotation_bias_arcsec)
        translation_command = _matrix_vector_product(
            _inverse_matrix(self.translation_coupling),
            _subtract_vectors(_translation_of(desired_physical_adjustment), self.translation_bias_um),
        )
        rotation_command = _matrix_vector_product(
            _inverse_matrix(self.rotation_coupling),
            _subtract_vectors(_rotation_of(desired_physical_adjustment), rotation_bias_rad),
        )
        return _pose_from_parts(translation_command, rotation_command)


@dataclass(frozen=True)
class SimulationDomain:
    """一个可采样的机器/批次场景。

    ``M0``、``M1``、``Mstress`` 都是公开的仿真假设，不是任务书给出的
    传感器或执行器真值。来料尺寸按公开的截断正态假设抽取，再按
    ``selected_clearance_range_um`` 接受，模拟“检测—分组选配”。主数据
    覆盖任务书允许且能由位姿补偿满足的完整单边间隙域 [2.5, 4.0] um；
    总间隙大于 8 um 的来料对不进入补偿网络，而应进入重新选配流程。
    """

    name: str
    scenario: str
    hole_side_range_um: tuple[float, float]
    shaft_side_range_um: tuple[float, float]
    selected_clearance_range_um: tuple[float, float]
    translation_range_um: float
    rotation_range_arcsec: float
    measurement: MeasurementModel
    actuator: ActuatorModel
    thermal: ThermalModel

    def __post_init__(self) -> None:
        if self.hole_side_range_um[0] > self.hole_side_range_um[1]:
            raise ValueError("孔尺寸范围下限不能大于上限。")
        if self.shaft_side_range_um[0] > self.shaft_side_range_um[1]:
            raise ValueError("轴尺寸范围下限不能大于上限。")
        if self.selected_clearance_range_um[0] > self.selected_clearance_range_um[1]:
            raise ValueError("选配单边间隙范围下限不能大于上限。")
        if self.translation_range_um < 0 or self.rotation_range_arcsec < 0:
            raise ValueError("初始误差范围不能为负数。")

    def sample_dimensions(
        self,
        rng: random.Random,
        thermal_condition: ThermalCondition,
    ) -> Dimensions:
        """按截断正态来料分布抽样，并按有效间隙完成选配。

        题设仅给出公差上下限，未给出真实生产尺寸分布；因此此处明确采用
        “中点为均值、6 sigma 覆盖整个公差带”的截断正态公开仿真假设。
        这比均匀抽样更接近受控加工批次，同时不把该假设伪装成实测数据。
        """
        for _ in range(100_000):
            dimensions = Dimensions(
                hole_side_um=_truncated_normal_from_range(rng, self.hole_side_range_um),
                shaft_side_um=_truncated_normal_from_range(rng, self.shaft_side_range_um),
            )
            effective = self.thermal.effective_dimensions(dimensions, thermal_condition)
            if self.selected_clearance_range_um[0] <= effective.nominal_single_side_clearance_um <= self.selected_clearance_range_um[1]:
                return dimensions
        raise RuntimeError("在 100000 次抽样内未获得符合运行温度间隙范围的轴孔对；请检查域参数。")

    def sample_dimensions_for_coverage(
        self,
        index: int,
        thermal_condition: ThermalCondition,
        clearance_range_um: tuple[float, float] | None = None,
    ) -> Dimensions:
        """生成均匀覆盖指定有效单边间隙域的可制造选配组合。"""
        clearance_low, clearance_high = clearance_range_um or self.selected_clearance_range_um
        hole_scale, shaft_scale = self.thermal._scale_factors(thermal_condition)
        # 温度很高或孔、轴温差很大时，参考尺寸公差带不一定能实现预设有效
        # 间隙的整个区间。标定集仍做覆盖，但只能覆盖“本温度条件下可制造、
        # 可选配”的有效子区间，不能伪造物理上不存在的样本。
        feasible_clearance_low = max(
            clearance_low,
            (self.hole_side_range_um[0] * hole_scale - self.shaft_side_range_um[1] * shaft_scale) / 2.0,
        )
        feasible_clearance_high = min(
            clearance_high,
            (self.hole_side_range_um[1] * hole_scale - self.shaft_side_range_um[0] * shaft_scale) / 2.0,
        )
        if feasible_clearance_low > feasible_clearance_high:
            raise ValueError("当前材料公差带、温度条件与选配间隙范围没有可行交集。")
        clearance = feasible_clearance_low + (
            feasible_clearance_high - feasible_clearance_low
        ) * _halton(index, 3)
        shaft_low = max(
            self.shaft_side_range_um[0],
            (self.hole_side_range_um[0] * hole_scale - 2.0 * clearance) / shaft_scale,
        )
        shaft_high = min(
            self.shaft_side_range_um[1],
            (self.hole_side_range_um[1] * hole_scale - 2.0 * clearance) / shaft_scale,
        )
        if shaft_low > shaft_high:
            raise ValueError("当前材料公差带、温度条件与选配间隙范围没有可行交集。")
        shaft = shaft_low + (shaft_high - shaft_low) * _halton(index, 2)
        hole = (shaft * shaft_scale + 2.0 * clearance) / hole_scale
        return Dimensions(hole_side_um=hole, shaft_side_um=shaft)

    def sample_true_pose(self, rng: random.Random, risk_layer: str | None = None) -> Pose6D:
        """按指定风险层采样；``None`` 表示不分层的普通均匀抽样。

        正常层与困难层直接构造，避免在六维立方体中用低效率拒绝采样；只有
        中等覆盖层使用拒绝采样，因其定义本身是“非正常且非困难”的剩余空间。
        """
        if risk_layer == "normal":
            translation_limit = min(0.35, self.translation_range_um)
            rotation_limit = min(0.5, self.rotation_range_arcsec)
            return _pose_from_parts(
                tuple(rng.uniform(-translation_limit, translation_limit) for _ in range(3)),
                tuple(arcsec_to_rad(rng.uniform(-rotation_limit, rotation_limit)) for _ in range(3)),
            )

        if risk_layer == "difficult":
            translation = [rng.uniform(-self.translation_range_um, self.translation_range_um) for _ in range(3)]
            rotation = [rng.uniform(-self.rotation_range_arcsec, self.rotation_range_arcsec) for _ in range(3)]
            eligible = [index for index in range(3) if self.translation_range_um > 0.6]
            eligible.extend(index + 3 for index in range(3) if self.rotation_range_arcsec > 1.2)
            if not eligible:
                raise ValueError("当前域误差范围不足，不能生成困难层。")
            chosen = rng.choice(eligible)
            sign = rng.choice((-1.0, 1.0))
            if chosen < 3:
                translation[chosen] = sign * rng.uniform(0.6, self.translation_range_um)
            else:
                rotation[chosen - 3] = sign * rng.uniform(1.2, self.rotation_range_arcsec)
            return _pose_from_parts(tuple(translation), tuple(arcsec_to_rad(value) for value in rotation))

        for _ in range(100_000):
            translation = tuple(
                rng.uniform(-self.translation_range_um, self.translation_range_um) for _ in range(3)
            )
            rotation = tuple(
                arcsec_to_rad(rng.uniform(-self.rotation_range_arcsec, self.rotation_range_arcsec))
                for _ in range(3)
            )
            pose = _pose_from_parts(translation, rotation)
            if risk_layer is None or classify_risk_layer(pose) == risk_layer:
                return pose
        raise RuntimeError(f"在 100000 次抽样内未获得风险层 {risk_layer!r} 的位姿；请检查阈值和域范围。")

    def coverage_pose(self, index: int) -> Pose6D:
        """Halton 低差异序列覆盖六维目标域，供极少量标定样本使用。"""
        bases = (5, 7, 11, 13, 17, 19)
        values = tuple(2.0 * _halton(index, base) - 1.0 for base in bases)
        translation = tuple(value * self.translation_range_um for value in values[:3])
        rotation = tuple(arcsec_to_rad(value * self.rotation_range_arcsec) for value in values[3:])
        return _pose_from_parts(translation, rotation)

    def coverage_pose_for_layer(self, index: int, risk_layer: str) -> Pose6D:
        """在指定风险层内部做低差异覆盖，避免 6D 均匀采样几乎全落入困难层。"""
        bases = (5, 7, 11, 13, 17, 19)
        values = [_halton(index, base) for base in bases]
        if risk_layer == "normal":
            translation = tuple((2.0 * value - 1.0) * min(0.35, self.translation_range_um) for value in values[:3])
            rotation = tuple(arcsec_to_rad((2.0 * value - 1.0) * min(0.5, self.rotation_range_arcsec)) for value in values[3:])
            return _pose_from_parts(translation, rotation)

        translation = [(2.0 * value - 1.0) * min(0.6, self.translation_range_um) for value in values[:3]]
        rotation = [(2.0 * value - 1.0) * min(1.2, self.rotation_range_arcsec) for value in values[3:]]
        eligible = [item for item in range(3) if self.translation_range_um > 0.35]
        eligible.extend(item + 3 for item in range(3) if self.rotation_range_arcsec > 0.5)
        if not eligible:
            raise ValueError("当前域误差范围不足，不能构造覆盖/困难层。")
        chosen = eligible[index % len(eligible)]
        sign = -1.0 if _halton(index, 23) < 0.5 else 1.0
        if risk_layer == "coverage":
            if chosen < 3:
                translation[chosen] = sign * (0.35 + (min(0.6, self.translation_range_um) - 0.35) * _halton(index, 29))
            else:
                rotation[chosen - 3] = sign * (0.5 + (min(1.2, self.rotation_range_arcsec) - 0.5) * _halton(index, 29))
        elif risk_layer == "difficult":
            difficult_eligible = [item for item in range(3) if self.translation_range_um > 0.6]
            difficult_eligible.extend(item + 3 for item in range(3) if self.rotation_range_arcsec > 1.2)
            if not difficult_eligible:
                raise ValueError("当前域误差范围不足，不能构造困难层。")
            chosen = difficult_eligible[index % len(difficult_eligible)]
            if chosen < 3:
                translation[chosen] = sign * (0.6 + (self.translation_range_um - 0.6) * _halton(index, 29))
            else:
                rotation[chosen - 3] = sign * (1.2 + (self.rotation_range_arcsec - 1.2) * _halton(index, 29))
        else:
            raise ValueError(f"未知风险层：{risk_layer}")
        return _pose_from_parts(tuple(translation), tuple(arcsec_to_rad(value) for value in rotation))


def _halton(index: int, base: int) -> float:
    """返回从 1 开始编号的 Halton 低差异序列值。"""
    value, factor, current = 0.0, 1.0, max(1, index)
    while current:
        factor /= base
        value += factor * (current % base)
        current //= base
    return value


def _truncated_normal_from_range(
    rng: random.Random,
    value_range_um: tuple[float, float],
) -> float:
    """在公差带内采样截断正态值：中点为均值，整个带宽为 6 sigma。"""
    lower, upper = value_range_um
    mean = (lower + upper) / 2.0
    standard_deviation = (upper - lower) / 6.0
    for _ in range(10_000):
        value = rng.gauss(mean, standard_deviation)
        if lower <= value <= upper:
            return value
    raise RuntimeError("在 10000 次尝试内未得到截断正态尺寸；请检查公差范围。")


def classify_size_band(single_side_clearance_um: float) -> str:
    """标注可行域内的尺寸位置；边界样本单独评价，绝不当作异常删除。"""
    if 2.5 <= single_side_clearance_um < 2.6:
        return "lower_boundary"
    if 3.8 < single_side_clearance_um <= 4.0:
        return "upper_boundary"
    if 2.6 <= single_side_clearance_um <= 3.8:
        return "interior"
    raise ValueError("单边间隙不属于任务书可由位姿补偿满足的 [2.5, 4.0] um 域。")


def classify_risk_layer(pose: Pose6D) -> str:
    """按协议的正常/覆盖/困难定义给六维位姿分层。"""
    translation = (abs(pose.dx_um), abs(pose.dy_um), abs(pose.dz_um))
    rotation_arcsec = tuple(abs(rad_to_arcsec(value)) for value in _rotation_of(pose))
    if all(value <= 0.35 for value in translation) and all(value <= 0.5 for value in rotation_arcsec):
        return "normal"
    if any(value > 0.6 for value in translation) or any(value > 1.2 for value in rotation_arcsec):
        return "difficult"
    return "coverage"


def _calibration_layers(count: int) -> list[str]:
    """生成嵌套的 10/30/50 标定层序列，三组均保持 60/30/10 比例。"""
    layers: list[str] = []
    for block_size in (10, 20, 20):
        if len(layers) >= count:
            break
        normal = round(block_size * 0.60)
        difficult = round(block_size * 0.10)
        block = ["normal"] * normal + ["coverage"] * (block_size - normal - difficult) + ["difficult"] * difficult
        # 用确定性交织而不是随机打乱，保证每一段都覆盖三类风险。
        interleaved: list[str] = []
        while block:
            for layer in ("normal", "coverage", "difficult"):
                if layer in block:
                    block.remove(layer)
                    interleaved.append(layer)
        layers.extend(interleaved)
    if len(layers) < count:
        normal = round((count - len(layers)) * 0.60)
        difficult = round((count - len(layers)) * 0.10)
        layers.extend(["normal"] * normal)
        layers.extend(["coverage"] * (count - len(layers) - difficult))
        layers.extend(["difficult"] * difficult)
    return layers[:count]


def make_source_domain() -> SimulationDomain:
    """M0：题设等温铜件下的源域名义机器，用于仿真预训练。"""
    return SimulationDomain(
        name="source_pretrain_M0",
        scenario="M0",
        hole_side_range_um=(10005.0, 10010.0),
        shaft_side_range_um=(9995.0, 10000.0),
        selected_clearance_range_um=(2.5, 4.0),
        translation_range_um=0.8,
        rotation_range_arcsec=1.5,
        measurement=MeasurementModel(),
        actuator=ActuatorModel(),
        thermal=make_task_isothermal_copper_model(),
    )


def make_target_domain() -> SimulationDomain:
    """M1：未见目标机器/批次，用于 10/30/50 条小样本标定。

    它是“少样本迁移”的演示环境，不是对某台真实设备的断言。真实实验时应
    用标定数据替换其中参数。
    """
    return SimulationDomain(
        name="target_adaptation_M1",
        scenario="M1",
        hole_side_range_um=(10005.0, 10010.0),
        shaft_side_range_um=(9995.0, 10000.0),
        selected_clearance_range_um=(2.5, 4.0),
        translation_range_um=1.0,
        rotation_range_arcsec=2.0,
        measurement=MeasurementModel(
            translation_bias_um=(0.07, -0.05, 0.04),
            translation_noise_std_um=0.04,
            rotation_bias_arcsec=(0.08, -0.06, 0.05),
            rotation_noise_std_arcsec=0.05,
        ),
        actuator=ActuatorModel(
            translation_coupling=(
                (0.972, 0.020, 0.000),
                (-0.016, 1.014, 0.004),
                (0.000, 0.000, 0.986),
            ),
            rotation_coupling=(
                (0.982, 0.016, 0.000),
                (-0.012, 1.012, 0.010),
                (0.000, -0.008, 0.990),
            ),
            translation_bias_um=(0.040, -0.030, 0.025),
            rotation_bias_arcsec=(0.06, -0.05, 0.04),
            translation_noise_std_um=0.025,
            rotation_noise_std_arcsec=0.035,
        ),
        thermal=make_task_isothermal_copper_model(),
    )


def make_stress_domain() -> SimulationDomain:
    """Mstress：超出常规任务包络的安全压力场景，绝不混入主结果。"""
    return SimulationDomain(
        name="safety_stress_Mstress",
        scenario="Mstress",
        hole_side_range_um=(10005.0, 10010.0),
        shaft_side_range_um=(9995.0, 10000.0),
        selected_clearance_range_um=(2.5, 4.0),
        # 这是专用于越界安全验证的场景，不是任务书常规误差范围。完整的
        # [2.5,4.0] um 可行尺寸域在此保留，碰撞损失才有可检验的边界样本。
        translation_range_um=4.5,
        rotation_range_arcsec=8.0,
        measurement=MeasurementModel(
            translation_bias_um=(0.12, -0.09, 0.07),
            translation_noise_std_um=0.08,
            rotation_bias_arcsec=(0.16, -0.12, 0.10),
            rotation_noise_std_arcsec=0.10,
        ),
        actuator=ActuatorModel(
            translation_coupling=((0.94, 0.05, 0.01), (-0.04, 1.05, 0.02), (0.01, -0.01, 0.95)),
            rotation_coupling=((0.95, 0.04, 0.01), (-0.03, 1.04, 0.03), (0.01, -0.02, 0.96)),
            translation_bias_um=(0.08, -0.06, 0.05),
            rotation_bias_arcsec=(0.10, -0.08, 0.07),
            translation_noise_std_um=0.05,
            rotation_noise_std_arcsec=0.07,
        ),
        thermal=make_task_isothermal_copper_model(),
    )


def make_size_boundary_domain() -> SimulationDomain:
    """Msize：与 M1 同机器失配、仅把尺寸置于完整可行域两端的专用验收域。"""
    return replace(
        make_target_domain(),
        name="size_boundary_Msize",
        scenario="Msize",
    )


def make_safety_auxiliary_domain() -> SimulationDomain:
    """M0-S：源域的越界安全辅助集。

    它与 M0 共享传感器和执行器均值模型，只扩大初始横向误差的可采样包络。
    该集不代表批量生产分布，也不计入任何主合格率；它只让训练阶段的
    ``L_gap`` 与 ``L_collision`` 见到真实会激活的紧间隙/干涉状态。
    """
    source = make_source_domain()
    return replace(
        source,
        name="source_safety_auxiliary_M0S",
        scenario="M0-S",
        translation_range_um=4.5,
        rotation_range_arcsec=8.0,
        thermal=make_task_isothermal_copper_model(),
    )


@dataclass(frozen=True)
class SimulatedAssembly:
    """一条完整的虚拟装配样本，保留后续报告和网络所需的可追溯中间量。"""

    sample_id: int
    domain_name: str
    machine_scenario: str
    risk_layer: str
    safety_band: str
    size_band: str
    # ``dimensions`` 是 20 °C 参考尺寸；真正进入四角点几何层的是
    # ``effective_dimensions``，后者由真实温度场的线膨胀计算得到。
    dimensions: Dimensions
    effective_dimensions: Dimensions
    thermal_condition: ThermalCondition
    measured_thermal_condition: ThermalCondition
    estimated_effective_dimensions: Dimensions
    hole_linear_expansion_per_c: float
    shaft_linear_expansion_per_c: float
    true_pose: Pose6D
    measured_pose: Pose6D
    analytic_command: Pose6D
    physical_adjustment: Pose6D
    residual_pose_after_analytic: Pose6D
    optimal_command: Pose6D
    residual_command_label: Pose6D
    before_fit: AssemblyResult
    after_analytic_fit: AssemblyResult
    after_analytic_objective: AssemblyObjective
    # ``after_optimal_*`` 是教师标签经过执行器均值模型后的离线回代证据。
    # 它们绝不属于网络输入，保留它们是为了让每一行 CSV 都能独立核验
    # label_optimal_command 是否确实满足联合 PASS。
    optimal_mean_pose: Pose6D
    optimal_mean_fit: AssemblyResult
    optimal_mean_objective: AssemblyObjective
    optimal_safety_margin_um: float
    after_analytic_residual_qualified: bool


def simulate_assembly(
    sample_id: int,
    domain: SimulationDomain,
    rng: random.Random,
    *,
    risk_layer: str | None = None,
    safety_band: str = "production",
    dimensions: Dimensions | None = None,
    thermal_condition: ThermalCondition | None = None,
    true_pose: Pose6D | None = None,
) -> SimulatedAssembly:
    """生成一条“测量→解析补偿→虚拟执行→真值标签”的样本。"""
    thermal_condition = thermal_condition or domain.thermal.sample_true_condition(rng)
    dimensions = dimensions or domain.sample_dimensions(rng, thermal_condition)
    effective_dimensions = domain.thermal.effective_dimensions(dimensions, thermal_condition)
    true_pose = true_pose or domain.sample_true_pose(rng, risk_layer=risk_layer)
    # 常规生产集按六维幅值阈值分层；安全辅助/压力集则按协议统一标为
    # difficult。这里的 difficult 表示“被刻意纳入安全压力课程”，其几何
    # 困难性由 safety_band 给出，而不是靠人为放大 dz 来满足常规幅值阈值。
    actual_risk_layer = "difficult" if safety_band != "production" else classify_risk_layer(true_pose)
    measured_pose = domain.measurement.measure(true_pose, rng)
    measured_thermal_condition = domain.thermal.measure(thermal_condition, rng)
    estimated_effective_dimensions = domain.thermal.estimated_effective_dimensions(
        dimensions,
        measured_thermal_condition,
    )

    before_fit = evaluate_lateral_fit(effective_dimensions, true_pose)
    analytic_command = analytic_compensation(measured_pose).adjustment
    physical_adjustment = domain.actuator.sample_physical_adjustment(analytic_command, rng)
    residual_pose = apply_world_frame_adjustment(true_pose, physical_adjustment)
    after_fit = evaluate_lateral_fit(effective_dimensions, residual_pose)

    # 主方案的仿真教师标签 u*：真实初态需要的物理变换，再通过执行器均值
    # 模型反算出完整的最优命令。对称直方孔中，居中状态同时最大化四侧间隙
    # 的最小裕度；真实硬件阶段这一列要由高精度复测/标定得到，而不能读取
    # true_pose。
    desired_physical_adjustment = analytic_compensation(true_pose).adjustment
    optimal_command = domain.actuator.inverse_mean_command(desired_physical_adjustment)
    residual_label = _pose_component_difference(optimal_command, analytic_command)
    optimal_mean_pose = apply_world_frame_adjustment(
        true_pose,
        domain.actuator.mean_physical_adjustment(optimal_command),
    )
    optimal_mean_fit = evaluate_lateral_fit(effective_dimensions, optimal_mean_pose)
    optimal_mean_objective = evaluate_assembly_objective(effective_dimensions, optimal_mean_pose)

    return SimulatedAssembly(
        sample_id=sample_id,
        domain_name=domain.name,
        machine_scenario=domain.scenario,
        risk_layer=actual_risk_layer,
        safety_band=safety_band,
        size_band=classify_size_band(effective_dimensions.nominal_single_side_clearance_um),
        dimensions=dimensions,
        effective_dimensions=effective_dimensions,
        thermal_condition=thermal_condition,
        measured_thermal_condition=measured_thermal_condition,
        estimated_effective_dimensions=estimated_effective_dimensions,
        hole_linear_expansion_per_c=domain.thermal.hole_linear_expansion_per_c,
        shaft_linear_expansion_per_c=domain.thermal.shaft_linear_expansion_per_c,
        true_pose=true_pose,
        measured_pose=measured_pose,
        analytic_command=analytic_command,
        physical_adjustment=physical_adjustment,
        residual_pose_after_analytic=residual_pose,
        optimal_command=optimal_command,
        residual_command_label=residual_label,
        before_fit=before_fit,
        after_analytic_fit=after_fit,
        after_analytic_objective=evaluate_assembly_objective(effective_dimensions, residual_pose),
        optimal_mean_pose=optimal_mean_pose,
        optimal_mean_fit=optimal_mean_fit,
        optimal_mean_objective=optimal_mean_objective,
        optimal_safety_margin_um=min(
            optimal_mean_objective.minimum_gap_margin_um,
            optimal_mean_fit.minimum_clearance_um,
        ),
        after_analytic_residual_qualified=residual_pose_within_requirement(effective_dimensions, residual_pose),
    )


_SAFETY_BAND_ORDER = ("collision", "near_contact", "tight_clearance", "loose_clearance")


def _safety_band_plan(count: int) -> list[str]:
    """给安全集分配预先固定的四种几何状态。

    * collision：真实几何干涉，``c_min <= 0``；
    * near_contact：未碰撞但距孔壁不超过 0.5 um；
    * tight_clearance：最小间隙在 (0.5, 1.0] um；
    * loose_clearance：至少一个标注间隙超过 4 um、但仍不发生干涉。

    这不是生产发生率，而是用于检查安全约束是否真正被激活的覆盖设计。
    """
    collision = round(count * 0.35)
    near = round(count * 0.25)
    tight = round(count * 0.20)
    loose = count - collision - near - tight
    plan = (
        ["collision"] * collision
        + ["near_contact"] * near
        + ["tight_clearance"] * tight
        + ["loose_clearance"] * loose
    )
    return plan


def _sample_safety_pose(
    domain: SimulationDomain,
    dimensions: Dimensions,
    rng: random.Random,
    safety_band: str,
) -> Pose6D:
    """按可回代验证的几何条件生成侧向危险样本，而不是碰运气地扩大量程。"""
    if safety_band not in _SAFETY_BAND_ORDER:
        raise ValueError(f"未知安全分层：{safety_band}")
    clearance = dimensions.nominal_single_side_clearance_um
    for _ in range(100_000):
        # 用一个横向主导轴主动靠近孔壁；其余自由度仍保留扰动，避免安全集退化成二维。
        translation = [rng.uniform(-0.20, 0.20) for _ in range(3)]
        # dz 不参与直壁方孔的侧向碰撞，因此它和三个转角均独立于横向安全状态。
        # 但“独立”不等于缩小范围：M0-S/Mstress 在 metadata 中公开声明了完整
        # 的六维压力包络，故必须真实覆盖 domain 的 dz/转角范围。横向安全带由
        # 下方精确几何拒绝条件保证；因此数据中也自然保留“纯横向碰撞 + 小 dz”的样本。
        translation[2] = rng.uniform(-domain.translation_range_um, domain.translation_range_um)
        rotations_arcsec = [
            rng.uniform(-domain.rotation_range_arcsec, domain.rotation_range_arcsec)
            for _ in range(3)
        ]
        axis = rng.choice((0, 1))
        sign = rng.choice((-1.0, 1.0))
        if safety_band == "collision":
            magnitude = clearance + rng.uniform(0.08, 1.10)
        elif safety_band == "near_contact":
            magnitude = clearance - rng.uniform(0.05, 0.50)
        elif safety_band == "tight_clearance":
            magnitude = clearance - rng.uniform(0.55, 1.00)
        else:  # loose_clearance：一侧宽松越上界，但另一侧仍保持正间隙。
            lower = max(4.0 - clearance + 0.08, 0.10)
            upper = clearance - 0.60
            if lower >= upper:
                continue
            magnitude = rng.uniform(lower, upper)
        translation[axis] = sign * magnitude
        pose = _pose_from_parts(
            tuple(translation),
            tuple(arcsec_to_rad(value) for value in rotations_arcsec),
        )
        fit = evaluate_lateral_fit(dimensions, pose)
        if safety_band == "collision" and fit.collision:
            return pose
        if safety_band == "near_contact" and 0.0 < fit.minimum_clearance_um <= 0.5:
            return pose
        if safety_band == "tight_clearance" and 0.5 < fit.minimum_clearance_um <= 1.0:
            return pose
        if safety_band == "loose_clearance" and not fit.collision and any(
            gap > dimensions.gap_upper_um for gap in fit.labelled_gaps_um
        ):
            return pose
    raise RuntimeError(f"在 100000 次尝试内未生成安全分层 {safety_band}；请检查范围。")


def _simulate_safety_stress_many(
    count: int,
    domain: SimulationDomain,
    seed: int,
) -> list[SimulatedAssembly]:
    """生成碰撞、近接、紧间隙和宽间隙都被明确覆盖的安全数据集。"""
    rng = random.Random(seed)
    plan = _safety_band_plan(count)
    rng.shuffle(plan)
    samples: list[SimulatedAssembly] = []
    for index, safety_band in enumerate(plan, start=1):
        thermal_condition = domain.thermal.sample_true_condition(rng)
        dimensions = domain.sample_dimensions(rng, thermal_condition)
        effective_dimensions = domain.thermal.effective_dimensions(dimensions, thermal_condition)
        pose = _sample_safety_pose(domain, effective_dimensions, rng, safety_band)
        samples.append(
            simulate_assembly(
                index,
                domain,
                rng,
                dimensions=dimensions,
                thermal_condition=thermal_condition,
                true_pose=pose,
                safety_band=safety_band,
            )
        )
    return samples


def simulate_many(
    count: int,
    domain: SimulationDomain,
    seed: int,
    *,
    sampling_strategy: str = "stratified",
) -> list[SimulatedAssembly]:
    """以固定随机种子生成可复现实验样本。

    ``stratified`` 使用 60/30/10 的正常/覆盖/困难层；``calibration`` 只用于
    目标域 10/30/50 条标定集，在每个风险层内采用六维和尺寸空间覆盖采样；
    ``full_feasible_coverage`` 覆盖完整 [2.5,4.0] um 单边可行尺寸域；
    ``size_boundary`` 以 1:1 比例覆盖其两端边界。
    """
    if count <= 0:
        raise ValueError("样本数量必须为正整数。")
    if sampling_strategy not in {
        "stratified", "calibration", "difficult_only", "safety_stress",
        "full_feasible_coverage", "size_boundary",
    }:
        raise ValueError("未知 sampling_strategy。")
    if sampling_strategy == "safety_stress":
        return _simulate_safety_stress_many(count, domain, seed)
    rng = random.Random(seed)
    if sampling_strategy in {"calibration", "full_feasible_coverage", "size_boundary"}:
        layers = _calibration_layers(count)
        samples: list[SimulatedAssembly] = []
        for index in range(count):
            coverage_index = seed + index + 1
            thermal_condition = domain.thermal.coverage_true_condition(coverage_index)
            clearance_range = None
            if sampling_strategy == "size_boundary":
                clearance_range = (2.5, 2.6) if index % 2 == 0 else (3.8, 4.0)
            samples.append(
                simulate_assembly(
                    index + 1,
                    domain,
                    rng,
                    dimensions=domain.sample_dimensions_for_coverage(
                        coverage_index, thermal_condition, clearance_range
                    ),
                    thermal_condition=thermal_condition,
                    true_pose=domain.coverage_pose_for_layer(coverage_index, layers[index]),
                )
            )
        return samples
    if sampling_strategy == "difficult_only":
        return [
            simulate_assembly(index + 1, domain, rng, risk_layer="difficult")
            for index in range(count)
        ]

    normal_count = round(count * 0.60)
    difficult_count = round(count * 0.10)
    layers = ["normal"] * normal_count + ["difficult"] * difficult_count
    layers.extend(["coverage"] * (count - len(layers)))
    rng.shuffle(layers)
    return [
        simulate_assembly(index + 1, domain, rng, risk_layer=layer)
        for index, layer in enumerate(layers)
    ]


def _pose_columns(prefix: str, pose: Pose6D) -> dict[str, float]:
    return {
        f"{prefix}_dx_um": pose.dx_um,
        f"{prefix}_dy_um": pose.dy_um,
        f"{prefix}_dz_um": pose.dz_um,
        f"{prefix}_theta_x_arcsec": rad_to_arcsec(pose.theta_x_rad),
        f"{prefix}_theta_y_arcsec": rad_to_arcsec(pose.theta_y_rad),
        f"{prefix}_theta_z_arcsec": rad_to_arcsec(pose.theta_z_rad),
    }


def _fit_columns(prefix: str, fit: AssemblyResult) -> dict[str, float | bool]:
    return {
        f"{prefix}_g1_um": fit.g1_um,
        f"{prefix}_g2_um": fit.g2_um,
        f"{prefix}_g3_um": fit.g3_um,
        f"{prefix}_g4_um": fit.g4_um,
        f"{prefix}_minimum_clearance_um": fit.minimum_clearance_um,
        f"{prefix}_collision": fit.collision,
        f"{prefix}_labelled_gaps_qualified": fit.labelled_gaps_qualified,
    }


def sample_to_row(sample: SimulatedAssembly) -> dict[str, float | int | str | bool]:
    """将一条仿真样本展平为 CSV 的一行。

    约定：主方案网络的输入为 ``measured_*``、尺寸与温度传感特征，输出/训练标签为
    ``label_optimal_command_*``。``label_residual_command_*`` 仅保留给“解析
    补偿 + 残差网络”对照组。所有角度均用角秒存储，便于与赛题阈值直接比较；
    代码内部仍统一用弧度计算。
    """
    row: dict[str, float | int | str | bool] = {
        "sample_id": sample.sample_id,
        "domain": sample.domain_name,
        "machine_scenario": sample.machine_scenario,
        "risk_layer": sample.risk_layer,
        "safety_band": sample.safety_band,
        "size_band": sample.size_band,
        "hole_side_um": sample.dimensions.hole_side_um,
        "shaft_side_um": sample.dimensions.shaft_side_um,
        "nominal_single_side_clearance_um": sample.dimensions.nominal_single_side_clearance_um,
        "true_hole_temperature_c": sample.thermal_condition.hole_temperature_c,
        "true_shaft_temperature_c": sample.thermal_condition.shaft_temperature_c,
        "measured_hole_temperature_c": sample.measured_thermal_condition.hole_temperature_c,
        "measured_shaft_temperature_c": sample.measured_thermal_condition.shaft_temperature_c,
        "known_hole_linear_expansion_per_c": sample.hole_linear_expansion_per_c,
        "known_shaft_linear_expansion_per_c": sample.shaft_linear_expansion_per_c,
        "estimated_effective_hole_side_um": sample.estimated_effective_dimensions.hole_side_um,
        "estimated_effective_shaft_side_um": sample.estimated_effective_dimensions.shaft_side_um,
        "estimated_effective_single_side_clearance_um": sample.estimated_effective_dimensions.nominal_single_side_clearance_um,
        "true_effective_hole_side_um": sample.effective_dimensions.hole_side_um,
        "true_effective_shaft_side_um": sample.effective_dimensions.shaft_side_um,
        "true_effective_single_side_clearance_um": sample.effective_dimensions.nominal_single_side_clearance_um,
        "label_optimal_safety_margin_um": sample.optimal_safety_margin_um,
        "after_analytic_residual_qualified": sample.after_analytic_residual_qualified,
        "after_optimal_residual_qualified": sample.optimal_mean_objective.residual_pose_qualified,
        "after_optimal_joint_pass": sample.optimal_mean_objective.fully_qualified,
    }
    for values in (
        _pose_columns("true", sample.true_pose),
        _pose_columns("measured", sample.measured_pose),
        _pose_columns("analytic_command", sample.analytic_command),
        _pose_columns("physical_adjustment", sample.physical_adjustment),
        _pose_columns("after_analytic", sample.residual_pose_after_analytic),
        _pose_columns("label_optimal_command", sample.optimal_command),
        _pose_columns("label_residual_command", sample.residual_command_label),
        _fit_columns("before", sample.before_fit),
        _fit_columns("after_analytic", sample.after_analytic_fit),
        _pose_columns("after_optimal", sample.optimal_mean_pose),
        _fit_columns("after_optimal", sample.optimal_mean_fit),
    ):
        row.update(values)
    return row


def save_dataset_csv(samples: Iterable[SimulatedAssembly], output_path: Path) -> int:
    """将样本保存为无 BOM 的标准 UTF-8 CSV，避免首列名出现隐藏字符。"""
    rows = [sample_to_row(sample) for sample in samples]
    if not rows:
        raise ValueError("没有样本可保存。")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def save_incoming_pair_screening_csv(
    output_path: Path,
    *,
    count: int,
    seed: int,
    hole_side_range_um: tuple[float, float] = (10005.0, 10010.0),
    shaft_side_range_um: tuple[float, float] = (9995.0, 10000.0),
) -> dict[str, object]:
    """保存来料尺寸筛查记录，而非伪造不可补偿件的网络教师标签。

    ``requires_rematching`` 表示总间隙大于 8 um，任何横向位姿调整都不能
    同时使一对相对标注间隙均落在 [2,4] um；该零件对应重新匹配库存，
    不进入六维补偿网络训练或合格率分母。
    """
    if count <= 0:
        raise ValueError("来料筛查样本数必须为正整数。")
    rng = random.Random(seed)
    rows: list[dict[str, float | int | str]] = []
    for candidate_id in range(1, count + 1):
        hole = _truncated_normal_from_range(rng, hole_side_range_um)
        shaft = _truncated_normal_from_range(rng, shaft_side_range_um)
        total_clearance = hole - shaft
        status = "pose_compensation_feasible" if 5.0 <= total_clearance <= 8.0 else "requires_rematching"
        rows.append({"candidate_id": candidate_id, "hole_side_um": hole, "shaft_side_um": shaft, "total_clearance_um": total_clearance, "nominal_single_side_clearance_um": total_clearance / 2.0, "pair_status": status})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    feasible = sum(row["pair_status"] == "pose_compensation_feasible" for row in rows)
    return {"sample_count": count, "pose_compensation_feasible_count": feasible, "requires_rematching_count": count - feasible, "sha256": _sha256(output_path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_fingerprint(sample: SimulatedAssembly) -> str:
    """计算不含 sample_id 的行指纹，用于检查不同数据划分没有重复样本。"""
    row = sample_to_row(sample).copy()
    row.pop("sample_id", None)
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_dataset_samples(
    samples: Iterable[SimulatedAssembly],
    domain: SimulationDomain,
    *,
    require_stratified_layers: bool,
    require_safety_bands: bool = False,
) -> dict[str, object]:
    """实施正式 CSV 的质量门禁；失败时拒绝写入训练数据。"""
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("质量门禁不能检查空数据集。")

    rows = [sample_to_row(sample) for sample in sample_list]
    row_columns = set(rows[0])
    allowed_input_columns = set(FEATURE_SETS["C_解析先验BP_15维"])
    forbidden_feature_prefixes = (
        "true_",
        "physical_adjustment_",
        "after_",
        "before_",
        "label_",
    )
    if any(column.startswith(forbidden_feature_prefixes) for column in allowed_input_columns):
        raise AssertionError("网络特征集合中出现了真值、标签或装配后验证列。")
    if not allowed_input_columns.issubset(row_columns):
        raise AssertionError("CSV 缺少已声明的网络输入列。")
    if not set(PRIMARY_TARGET_COLUMNS).issubset(row_columns):
        raise AssertionError("CSV 缺少完整六维主标签列。")
    for row in rows:
        for column, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise AssertionError(f"CSV 含有非有限数值：{column}={value!r}")

    risk_counts = {"normal": 0, "coverage": 0, "difficult": 0}
    safety_band_counts = {band: 0 for band in _SAFETY_BAND_ORDER}
    size_band_counts = {"lower_boundary": 0, "interior": 0, "upper_boundary": 0}
    for sample in sample_list:
        clearance = sample.effective_dimensions.nominal_single_side_clearance_um
        if not domain.selected_clearance_range_um[0] <= clearance <= domain.selected_clearance_range_um[1]:
            raise AssertionError("存在未通过尺寸选配的轴孔组合。")
        if not sample.optimal_mean_objective.fully_qualified:
            raise AssertionError("教师最优命令回代后未通过联合 PASS。")
        if (
            sample.optimal_mean_fit.collision
            or not sample.optimal_mean_fit.labelled_gaps_qualified
            or not sample.optimal_mean_objective.residual_pose_qualified
        ):
            raise AssertionError("after_optimal 离线验证列与教师联合 PASS 不一致。")
        if sample.risk_layer not in risk_counts:
            raise AssertionError(f"未知风险层：{sample.risk_layer}")
        risk_counts[sample.risk_layer] += 1
        if sample.size_band != classify_size_band(clearance):
            raise AssertionError("size_band 与实际单边间隙不一致。")
        size_band_counts[sample.size_band] += 1
        if sample.safety_band in safety_band_counts:
            safety_band_counts[sample.safety_band] += 1
        elif sample.safety_band != "production":
            raise AssertionError(f"未知安全分层：{sample.safety_band}")

    if require_stratified_layers:
        expected = {
            "normal": round(len(sample_list) * 0.60),
            "difficult": round(len(sample_list) * 0.10),
        }
        expected["coverage"] = len(sample_list) - expected["normal"] - expected["difficult"]
        if risk_counts != expected:
            raise AssertionError(f"风险层样本数不符合 60/30/10 协议：实际 {risk_counts}，期望 {expected}")

    if require_safety_bands:
        expected_bands = {band: _safety_band_plan(len(sample_list)).count(band) for band in _SAFETY_BAND_ORDER}
        if safety_band_counts != expected_bands:
            raise AssertionError(f"安全集分层数不符合协议：实际 {safety_band_counts}，期望 {expected_bands}")
        for sample in sample_list:
            fit = sample.before_fit
            if sample.safety_band == "collision" and not fit.collision:
                raise AssertionError("安全集 collision 样本未发生真实几何干涉。")
            if sample.safety_band == "near_contact" and not (0.0 < fit.minimum_clearance_um <= 0.5):
                raise AssertionError("安全集 near_contact 样本未落入 (0, 0.5] um。")
            if sample.safety_band == "tight_clearance" and not (0.5 < fit.minimum_clearance_um <= 1.0):
                raise AssertionError("安全集 tight_clearance 样本未落入 (0.5, 1.0] um。")
            if sample.safety_band == "loose_clearance" and not (
                not fit.collision and any(gap > sample.dimensions.gap_upper_um for gap in fit.labelled_gaps_um)
            ):
                raise AssertionError("安全集 loose_clearance 样本未出现宽间隙越界。")

    return {
        "sample_count": len(sample_list),
        "risk_layer_counts": risk_counts,
        "safety_band_counts": safety_band_counts,
        "size_band_counts": size_band_counts,
        "operating_clearance_min_um": min(sample.effective_dimensions.nominal_single_side_clearance_um for sample in sample_list),
        "operating_clearance_max_um": max(sample.effective_dimensions.nominal_single_side_clearance_um for sample in sample_list),
        "teacher_joint_pass_rate": sum(sample.optimal_mean_objective.fully_qualified for sample in sample_list) / len(sample_list),
        "row_fingerprint_count": len({_sample_fingerprint(sample) for sample in sample_list}),
    }


def save_dataset_bundle(
    output_dir: Path,
    splits: Iterable[tuple[str, list[SimulatedAssembly], SimulationDomain, bool]],
    *,
    base_seed: int,
    incoming_screening_quality: dict[str, object] | None = None,
) -> dict[str, object]:
    """保存正式数据包、质量门禁摘要和可复现 metadata.json。

    ``splits`` 的每项依次为：文件名、样本、对应场景、是否要求 60/30/10 分层。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    split_metadata: dict[str, object] = {}
    seen_fingerprints: set[str] = set()
    for filename, samples, domain, require_stratified_layers in splits:
        path = output_dir / filename
        safety_dataset = filename in {"source_safety_auxiliary.csv", "target_stress_test.csv"}
        quality = validate_dataset_samples(
            samples,
            domain,
            require_stratified_layers=require_stratified_layers,
            require_safety_bands=safety_dataset,
        )
        fingerprints = {_sample_fingerprint(sample) for sample in samples}
        if len(fingerprints) != len(samples):
            raise AssertionError(f"{filename} 内部存在重复样本。")
        overlap = seen_fingerprints & fingerprints
        if overlap:
            raise AssertionError(f"{filename} 与已有数据划分存在重复样本。")
        seen_fingerprints.update(fingerprints)
        save_dataset_csv(samples, path)
        sampling = "stratified_60_30_10" if require_stratified_layers else "space_coverage"
        if filename in {"source_pretrain.csv", "source_validation.csv"}:
            sampling = "stratified_pose_plus_full_feasible_dimension_coverage"
        elif filename in {"target_operational_test.csv", "monte_carlo_operational_10000.csv"}:
            sampling = "conditional_truncated_normal_selected_pair_plus_stratified_pose"
        elif filename == "target_size_boundary_test.csv":
            sampling = "balanced_lower_upper_feasible_size_boundary"
        elif filename == "target_calibration_50.csv":
            sampling = "nested_stratified_space_coverage"
        if safety_dataset:
            sampling = "balanced_safety_stress"
        split_metadata[filename] = {
            "scenario": domain.scenario,
            "sampling": sampling,
            "quality_gate": quality,
            "sha256": _sha256(path),
        }

    metadata: dict[str, object] = {
        "dataset_protocol": "v5.0_full_feasible_clearance_selective_assembly",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_seed": base_seed,
        "seed_offsets": {
            "source_pretrain": 0,
            "source_validation": 1,
            "target_calibration": 2,
            "target_operational_test": 3,
            "target_stress_test": 4,
            "source_safety_auxiliary": 5,
            "monte_carlo_operational": 100,
            "target_size_boundary": 6,
            "incoming_pair_screening": 200,
        },
        "task_book_dimension_um": {
            "hole_side_range": [10005.0, 10010.0],
            "shaft_side_range": [9995.0, 10000.0],
            "total_clearance_range": [5.0, 15.0],
        },
        "pose_compensation_feasibility": {
            "labelled_gap_um": [2.0, 4.0],
            "total_clearance_um": [5.0, 8.0],
            "single_side_clearance_um": [2.5, 4.0],
            "outside_policy": "requires_rematching_not_neural_compensation",
        },
        "dimension_sampling_assumption": {
            "incoming": "truncated_normal",
            "mean": "midpoint_of_each_task_book_tolerance_band",
            "sigma": "one_sixth_of_each_tolerance_band_width",
            "training_coverage": "low_discrepancy_full_feasible_domain",
            "operational_and_monte_carlo": "conditioned_on_feasible_selected_pairs; this is a production-process distribution, not a boundary-coverage test",
            "size_boundary": "balanced_lower_and_upper_feasible_bands; this is a separate worst-case acceptance set, not a production-yield estimate",
            "note": "公开仿真假设，非实测生产分布；原始独立候选孔轴对的可行比例仅描述随机配对，不可表述为真实来料报废率或选配良率。",
        },
        "domains": {
            "M0": asdict(make_source_domain()),
            "M0-S": asdict(make_safety_auxiliary_domain()),
            "M1": asdict(make_target_domain()),
            "Msize": asdict(make_size_boundary_domain()),
            "Mstress": asdict(make_stress_domain()),
        },
        "network_input_feature_sets": {name: list(columns) for name, columns in FEATURE_SETS.items()},
        "primary_target_columns": list(PRIMARY_TARGET_COLUMNS),
        "files": split_metadata,
        "incoming_pair_screening": incoming_screening_quality,
        "generator_sha256": _sha256(Path(__file__)),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def residual_summary(samples: Iterable[SimulatedAssembly]) -> dict[str, float | int]:
    """汇总解析补偿在虚拟设备上的表现，用于检查数据是否合理。"""
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("没有样本可统计。")
    successful = sum(sample.after_analytic_residual_qualified for sample in sample_list)
    collision = sum(sample.after_analytic_fit.collision for sample in sample_list)
    gap_qualified = sum(sample.after_analytic_objective.gaps_qualified for sample in sample_list)
    fully_qualified = sum(sample.after_analytic_objective.fully_qualified for sample in sample_list)
    average_translation_norm = sum(
        math.sqrt(
            sample.residual_pose_after_analytic.dx_um ** 2
            + sample.residual_pose_after_analytic.dy_um ** 2
            + sample.residual_pose_after_analytic.dz_um ** 2
        )
        for sample in sample_list
    ) / len(sample_list)
    average_rotation_norm_arcsec = sum(
        math.sqrt(
            rad_to_arcsec(sample.residual_pose_after_analytic.theta_x_rad) ** 2
            + rad_to_arcsec(sample.residual_pose_after_analytic.theta_y_rad) ** 2
            + rad_to_arcsec(sample.residual_pose_after_analytic.theta_z_rad) ** 2
        )
        for sample in sample_list
    ) / len(sample_list)
    return {
        "sample_count": len(sample_list),
        "residual_requirement_pass_count": successful,
        "residual_requirement_pass_rate": successful / len(sample_list),
        "after_analytic_collision_count": collision,
        "after_analytic_gap_qualified_rate": gap_qualified / len(sample_list),
        "after_analytic_fully_qualified_rate": fully_qualified / len(sample_list),
        "mean_after_analytic_minimum_gap_margin_um": sum(
            sample.after_analytic_objective.minimum_gap_margin_um for sample in sample_list
        ) / len(sample_list),
        "mean_residual_translation_norm_um": average_translation_norm,
        "mean_residual_rotation_norm_arcsec": average_rotation_norm_arcsec,
    }
