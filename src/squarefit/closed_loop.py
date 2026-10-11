"""在线“测量 -> 推理 -> 补偿 -> 验证”闭环仿真与公平对照。

主方法不是多步 MPC，也不是第二个正向神经网络：先由已经适配的 C15 逆向 MLP
给出首轮六维补偿；每次动作后进行双复测融合，确认存在残余后再由已适配 MLP
与精确 Rodrigues 四角点几何层共同产生安全候选，最多执行三轮物理补偿。

本模块中的 ``true_pose``、真实有效尺寸和真实 M1 执行器仅属于虚拟工厂和离线
验收；控制器本身只读取测得位姿、实测轴孔尺寸、选配间隙和解析补偿先验。
"""

from __future__ import annotations

import csv
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch import Tensor

from .compensation import analytic_compensation, apply_world_frame_adjustment
from .contact import classify_lateral_contact_state
from .schema import FEATURE_SETS, PRIMARY_TARGET_COLUMNS
from .simulation import SimulationDomain, make_size_boundary_domain, make_stress_domain, make_target_domain
from .torch_geometry import SquareFitPhysicsLayer, TorchPhysicalState
from .torch_training import _load_source_bundle
from .types import Dimensions, Pose6D
from .units import arcsec_to_rad, rad_to_arcsec


_POSE_COLUMNS = (
    "measured_dx_um", "measured_dy_um", "measured_dz_um",
    "measured_theta_x_arcsec", "measured_theta_y_arcsec", "measured_theta_z_arcsec",
)
_COMMAND_SCALE = torch.tensor((0.25, 0.25, 0.25, 0.50, 0.50, 0.50), dtype=torch.float64)
_STEP_LIMIT = torch.tensor((1.25, 1.25, 1.25, 2.50, 2.50, 2.50), dtype=torch.float64)
_RESIDUAL_LIMIT = torch.tensor((0.10, 0.10, 0.10, 0.20, 0.20, 0.20), dtype=torch.float64)
_CONTROLLER_ID = "C15_MLP_single_initial_plus_post_action_dual_measurement_confirmed_multiround_geometric_safe_feedback"


@dataclass(frozen=True)
class ClosedLoopConfig:
    """冻结的控制器参数。仅能用开发集预先选定，不能据封存集调整。"""

    max_actions: int = 3
    safety_guard_um: float = 0.05
    line_search_scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)
    pid_p: float = 0.70
    pid_i: float = 0.06
    pid_d: float = 0.02
    pid_integral_limit: float = 2.0
    # 二次补偿必须同时满足“复测确认”和以下较小的残余触发线。触发线仅略高于
    # 任务书残余阈值；它们不再像旧版那样把 0.10--0.16 µm 的真实执行残余
    # 直接当作噪声死区放弃处理。
    feedback_translation_trigger_um: float = 0.12
    feedback_rotation_trigger_arcsec: float = 0.24
    feedback_gap_overshoot_um: float = 0.10
    # 主闭环与标准开环共享一次初测；这项会写入结果元数据，防止将“双初测”
    # 消融结果误表述为执行后反馈带来的收益。
    initial_measurements: int = 1
    confirmation_measurements: int = 2
    # 正常装配工况只允许一步进入预测安全区。仅当“当前预测已干涉”时，
    # 才允许严格改善最小间隙的低速脱困步；对原本安全的样本仍绝不放宽安全线。
    allow_collision_recovery: bool = True
    recovery_clearance_improvement_um: float = 1e-4
    topology_warning_clearance_um: float = 0.50


@dataclass(frozen=True)
class TrialRecord:
    sample_id: int
    method: str
    joint_pass: bool
    gap_pass: bool
    no_collision: bool
    residual_pass: bool
    initial_collision: bool
    post_action_collision: bool
    actions: int
    online_predicted_pass: bool
    predicted_safety_rejections: int
    initial_predicted_contact_topology: str
    final_predicted_contact_topology: str
    topology_guarded_cycles: int
    g1_um: float
    g2_um: float
    g3_um: float
    g4_um: float
    gap_margin_um: float
    minimum_clearance_um: float
    translation_mae_um: float
    rotation_mae_arcsec: float
    maximum_abs_translation_um: float
    maximum_abs_rotation_arcsec: float
    decision_time_ms: float


@dataclass(frozen=True)
class ConvergenceTraceRecord:
    """主闭环每一次真实物理动作后的离线收敛观测。

    这不是控制器的额外输入；只在仿真完成后用虚拟工厂真值统计“违规量是否
    总体下降、过程是否安全”。在线控制器仍只使用测量量和 C15 白名单特征。
    """

    sample_id: int
    action_index: int
    violation_score: float
    joint_pass: bool
    gap_pass: bool
    no_collision: bool
    residual_pass: bool
    gap_margin_um: float
    minimum_clearance_um: float
    maximum_abs_translation_um: float
    maximum_abs_rotation_arcsec: float


def _pose_from_values(values: list[float]) -> Pose6D:
    return Pose6D(
        dx_um=values[0], dy_um=values[1], dz_um=values[2],
        theta_x_rad=arcsec_to_rad(values[3]),
        theta_y_rad=arcsec_to_rad(values[4]), theta_z_rad=arcsec_to_rad(values[5]),
    )


def _pose_values(pose: Pose6D) -> list[float]:
    return [
        pose.dx_um, pose.dy_um, pose.dz_um,
        rad_to_arcsec(pose.theta_x_rad), rad_to_arcsec(pose.theta_y_rad), rad_to_arcsec(pose.theta_z_rad),
    ]


def _row_pose(row: dict[str, str], prefix: str) -> Pose6D:
    suffixes = ("dx_um", "dy_um", "dz_um", "theta_x_arcsec", "theta_y_arcsec", "theta_z_arcsec")
    return _pose_from_values([float(row[f"{prefix}_{suffix}"]) for suffix in suffixes])


def _dimensions(row: dict[str, str], prefix: str) -> Dimensions:
    return Dimensions(hole_side_um=float(row[f"{prefix}_hole_side_um"]), shaft_side_um=float(row[f"{prefix}_shaft_side_um"]))


def _tensor(values: list[float]) -> Tensor:
    return torch.tensor([values], dtype=torch.float64)


def _state_pass(state: TorchPhysicalState) -> bool:
    gaps = state.labelled_gaps_um[0]
    return bool(
        (gaps >= 2.0).all() and (gaps <= 4.0).all()
        and state.minimum_clearance_um[0] > 0.0
        and (state.residual_translation_um[0].abs() <= 0.10).all()
        and (state.residual_rotation_arcsec[0].abs() <= 0.20).all()
    )


def _needs_feedback(state: TorchPhysicalState, config: ClosedLoopConfig) -> bool:
    """对复测噪声设置死区，避免已合格件被边界附近的噪声反复“修坏”。"""
    gaps = state.labelled_gaps_um[0]
    gap_far_outside = bool((gaps < 2.0 - config.feedback_gap_overshoot_um).any() or (gaps > 4.0 + config.feedback_gap_overshoot_um).any())
    large_translation = bool((state.residual_translation_um[0].abs() > config.feedback_translation_trigger_um).any())
    large_rotation = bool((state.residual_rotation_arcsec[0].abs() > config.feedback_rotation_trigger_arcsec).any())
    return gap_far_outside or float(state.minimum_clearance_um[0]) < config.safety_guard_um or large_translation or large_rotation


def _average_pose(measurements: list[Pose6D]) -> Pose6D:
    """将同一静止状态的独立复测作算术融合。

    固定零偏不会被这一操作“凭空消除”，仍由已适配的 MLP 学习；平均只降低
    单次随机测量噪声，避免闭环依据一次偶然读数反向修正。
    """
    if not measurements:
        raise ValueError("至少需要一条测量才能融合位姿。")
    count = float(len(measurements))
    values = [sum(component) / count for component in zip(*(_pose_values(item) for item in measurements), strict=True)]
    return _pose_from_values(values)


def _confirmed_feedback_needed(
    fused_state: TorchPhysicalState,
    individual_states: list[TorchPhysicalState],
    config: ClosedLoopConfig,
) -> bool:
    """仅在两次独立复测均显示明显残余时允许常规二次补偿。

    发生接触/间隙危险时优先安全脱困，不因“确认”而延迟；其余情况要求融合
    读数和每一条原始复测均跨过触发线，降低把随机噪声当执行残余的概率。
    """
    if float(fused_state.minimum_clearance_um[0]) < config.safety_guard_um:
        return True
    return _needs_feedback(fused_state, config) and all(
        _needs_feedback(state, config) for state in individual_states
    )


def _state_metrics(state: TorchPhysicalState) -> tuple[bool, bool, bool, float, float, float, float, float, float]:
    gaps = state.labelled_gaps_um[0]
    gap_pass = bool((gaps >= 2.0).all() and (gaps <= 4.0).all())
    no_collision = bool(state.minimum_clearance_um[0] > 0.0)
    residual_pass = bool(
        (state.residual_translation_um[0].abs() <= 0.10).all()
        and (state.residual_rotation_arcsec[0].abs() <= 0.20).all()
    )
    margin = float(torch.minimum(gaps - 2.0, 4.0 - gaps).amin())
    translation_mae = float(state.residual_translation_um[0].abs().mean())
    rotation_mae = float(state.residual_rotation_arcsec[0].abs().mean())
    maximum_translation = float(state.residual_translation_um[0].abs().amax())
    maximum_rotation = float(state.residual_rotation_arcsec[0].abs().amax())
    return (
        gap_pass, no_collision, residual_pass, margin, float(state.minimum_clearance_um[0]),
        translation_mae, rotation_mae, maximum_translation, maximum_rotation,
    )


def _violation_vector(state: TorchPhysicalState, safety_guard_um: float) -> Tensor:
    """把三个硬约束连续化，供 DLS 雅可比反馈与安全线搜索共同使用。"""
    gaps = state.labelled_gaps_um[0]
    lower = torch.relu(2.0 - gaps) / 2.0
    upper = torch.relu(gaps - 4.0) / 2.0
    collision_guard = torch.relu(safety_guard_um - state.minimum_clearance_um[0]).reshape(1) / safety_guard_um
    residual = torch.cat((state.residual_translation_um[0], state.residual_rotation_arcsec[0]))
    signed_overflow = torch.sign(residual) * torch.relu(residual.abs() - _RESIDUAL_LIMIT) / _RESIDUAL_LIMIT
    return torch.cat((lower, upper, collision_guard, signed_overflow))


def _violation_score(state: TorchPhysicalState, safety_guard_um: float) -> float:
    return float(_violation_vector(state, safety_guard_um).square().sum())


def _trace_record(
    *,
    sample_id: int,
    action_index: int,
    state: TorchPhysicalState,
    config: ClosedLoopConfig,
) -> ConvergenceTraceRecord:
    """把一个真实物理状态转成仅用于离线稳定性分析的轨迹点。"""
    (
        gap_pass, no_collision, residual_pass, margin, clearance, _translation_mae, _rotation_mae,
        maximum_translation, maximum_rotation,
    ) = _state_metrics(state)
    return ConvergenceTraceRecord(
        sample_id=sample_id,
        action_index=action_index,
        violation_score=_violation_score(state, config.safety_guard_um),
        joint_pass=gap_pass and no_collision and residual_pass,
        gap_pass=gap_pass,
        no_collision=no_collision,
        residual_pass=residual_pass,
        gap_margin_um=margin,
        minimum_clearance_um=clearance,
        maximum_abs_translation_um=maximum_translation,
        maximum_abs_rotation_arcsec=maximum_rotation,
    )


def _damping_from_margin(state: TorchPhysicalState, guard_um: float) -> float:
    gaps = state.labelled_gaps_um[0]
    margin = float(torch.cat((gaps - 2.0, 4.0 - gaps, state.minimum_clearance_um[0:1])).amin())
    # 靠近边界时较大阻尼，自动退化成小步保守反馈；边界远时采用较精确的局部逆。
    return 3.0 if margin <= guard_um else (1.0 if margin <= 0.25 else 0.20)


def _clip_command(command: Tensor) -> Tensor:
    return torch.maximum(torch.minimum(command, _STEP_LIMIT), -_STEP_LIMIT)


def _safe_candidate(
    predicted_current: TorchPhysicalState,
    measured_pose: Tensor,
    estimated_dimensions: Tensor,
    desired_command: Tensor,
    physics: SquareFitPhysicsLayer,
    config: ClosedLoopConfig,
    *,
    require_improvement: bool = True,
) -> tuple[Tensor | None, int]:
    """只接受“预测不碰撞且违规量下降”的命令；否则逐级缩步。"""
    baseline_score = _violation_score(predicted_current, config.safety_guard_um)
    rejected = 0
    desired_command = _clip_command(desired_command.detach())
    for scale in config.line_search_scales:
        candidate = desired_command * scale
        state = physics(measured_pose, candidate, estimated_dimensions)
        current_clearance = float(predicted_current.minimum_clearance_um[0])
        candidate_clearance = float(state.minimum_clearance_um[0])
        absolute_safe = candidate_clearance >= config.safety_guard_um
        recovery_progress = (
            config.allow_collision_recovery
            and current_clearance < 0.0
            and candidate_clearance >= current_clearance + config.recovery_clearance_improvement_um
        )
        safe = absolute_safe or recovery_progress
        improves = _violation_score(state, config.safety_guard_um) + 1e-12 < baseline_score
        if safe and (improves or not require_improvement):
            return candidate.detach(), rejected
        rejected += 1
    return None, rejected


def _dls_feedback_command(
    measured_pose: Tensor,
    estimated_dimensions: Tensor,
    physics: SquareFitPhysicsLayer,
    config: ClosedLoopConfig,
    prediction_horizon: int = 1,
) -> Tensor:
    """求局部 DLS 几何反馈的首步命令。

    ``prediction_horizon=1`` 是主方法。H=2/3 只用于开发集消融：规划器会预测
    连续 H 个名义动作的最终几何状态，但在线仍只执行第一个动作并立刻复测。
    这使“单步是否充分”成为可复现实验，而不是口头判断。
    """
    if prediction_horizon <= 0:
        raise ValueError("预测时域必须为正整数。")
    normalized_zero = torch.zeros(6 * prediction_horizon, dtype=torch.float64, requires_grad=True)

    def residual_function(normalized_command: Tensor) -> Tensor:
        pose = measured_pose
        state: TorchPhysicalState | None = None
        for normalized_step in normalized_command.reshape(prediction_horizon, 6):
            command = (normalized_step * _COMMAND_SCALE).unsqueeze(0)
            state = physics(pose, command, estimated_dimensions)
            pose = state.pose_after
        assert state is not None
        return _violation_vector(state, config.safety_guard_um)

    current = physics.evaluate_pose(measured_pose, estimated_dimensions)
    residual = _violation_vector(current, config.safety_guard_um).detach()
    jacobian = torch.autograd.functional.jacobian(residual_function, normalized_zero)
    damping = _damping_from_margin(current, config.safety_guard_um)
    hessian = jacobian.T @ jacobian + damping * torch.eye(6 * prediction_horizon, dtype=torch.float64)
    step = torch.linalg.solve(hessian, -(jacobian.T @ residual))
    return (step.reshape(prediction_horizon, 6)[0] * _COMMAND_SCALE).unsqueeze(0)


class ClosedLoopEvaluator:
    """以同一初始行、同一动作上限和同一噪声轨迹比较四种控制方法。"""

    def __init__(
        self,
        *,
        source_model_directory: Path,
        adaptation_directory: Path,
        config: ClosedLoopConfig = ClosedLoopConfig(),
    ) -> None:
        model, training_config, feature_columns, target_columns, input_scaler, target_scaler = _load_source_bundle(source_model_directory)
        if training_config.feature_set_name != "C_解析先验BP_15维":
            raise ValueError("闭环主方案要求使用 C_解析先验BP_15维 检查点。")
        if target_columns != PRIMARY_TARGET_COLUMNS:
            raise ValueError("补偿检查点目标列不符合六维主标签协议。")
        checkpoint = torch.load(adaptation_directory / "target_adapt_30.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.model = model
        self.feature_columns = feature_columns
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler
        self.config = config
        # 部署端已知的是名义 M0 标定几何/执行器；M1 参数仅在虚拟工厂侧使用。
        self.nominal_physics = SquareFitPhysicsLayer().to(dtype=torch.float64)

    def _feature_tensor(self, measured_pose: Pose6D, row: dict[str, str]) -> Tensor:
        analytic = analytic_compensation(measured_pose).adjustment
        available: dict[str, float] = {
            **dict(zip(_POSE_COLUMNS, _pose_values(measured_pose), strict=True)),
            "hole_side_um": float(row["hole_side_um"]),
            "shaft_side_um": float(row["shaft_side_um"]),
            "nominal_single_side_clearance_um": float(row["nominal_single_side_clearance_um"]),
            **dict(zip(
                (
                    "analytic_command_dx_um", "analytic_command_dy_um", "analytic_command_dz_um",
                    "analytic_command_theta_x_arcsec", "analytic_command_theta_y_arcsec", "analytic_command_theta_z_arcsec",
                ),
                _pose_values(analytic), strict=True,
            )),
        }
        # 部署推理没有教师标签；这里仅对输入白名单做核验，绝不为了复用训练函数
        # 而把 ``label_*`` 人为塞进在线特征字典。
        validated_features = FEATURE_SETS["C_解析先验BP_15维"]
        if any(name not in available for name in validated_features) or validated_features != self.feature_columns:
            raise ValueError("闭环特征白名单与检查点不一致，拒绝推理。")
        values = torch.tensor([[available[name] for name in self.feature_columns]], dtype=torch.float64)
        means = torch.tensor(self.input_scaler.means, dtype=torch.float64)
        stds = torch.tensor(self.input_scaler.standard_deviations, dtype=torch.float64)
        return (values - means) / stds

    def _mlp_command(self, measured_pose: Pose6D, row: dict[str, str]) -> Tensor:
        with torch.no_grad():
            normalized = self.model(self._feature_tensor(measured_pose, row))
            means = torch.tensor(self.target_scaler.means, dtype=torch.float64)
            stds = torch.tensor(self.target_scaler.standard_deviations, dtype=torch.float64)
            return normalized * stds + means

    def _run_method(
        self,
        row: dict[str, str],
        domain: SimulationDomain,
        method: str,
        seed: int,
        *,
        prediction_horizon: int = 1,
        trajectory: list[ConvergenceTraceRecord] | None = None,
    ) -> TrialRecord:
        if method not in {"analytic_open", "mlp_open", "mlp_open_fused_initial", "pid_feedback", "mlp_geometry_closed_loop"}:
            raise ValueError(f"未知方法：{method}")
        true_pose = _row_pose(row, "true")
        measured_pose = _row_pose(row, "measured")
        true_dimensions = _dimensions(row, "true_effective")
        estimated_dimensions = _dimensions(row, "estimated_effective")
        true_dim_tensor = _tensor([true_dimensions.hole_side_um, true_dimensions.shaft_side_um])
        estimated_dim_tensor = _tensor([estimated_dimensions.hole_side_um, estimated_dimensions.shaft_side_um])
        plant_geometry = SquareFitPhysicsLayer().to(dtype=torch.float64)
        measurement_rng = random.Random(seed + 701)
        execution_rng = random.Random(seed + 1701)
        # 主闭环与标准开环共享同一条初测；只有首轮真实动作完成后才进行双复测
        # 融合。这保证优势来自“执行后反馈”，而非给闭环额外初测信息。
        measurement_samples = [measured_pose]
        initial_measurement_count = (
            2 if method == "mlp_open_fused_initial"
            else self.config.initial_measurements if method == "mlp_geometry_closed_loop"
            else 1
        )
        if initial_measurement_count > 1:
            measurement_samples.extend(
                domain.measurement.measure(true_pose, measurement_rng)
                for _ in range(initial_measurement_count - 1)
            )
            measured_pose = _average_pose(measurement_samples)
        start = time.perf_counter()
        actions = 0
        rejections = 0
        topology_guarded_cycles = 0
        initial_state = plant_geometry.evaluate_pose(_tensor(_pose_values(true_pose)), true_dim_tensor)
        initial_collision = not bool(initial_state.minimum_clearance_um[0] > 0.0)
        if trajectory is not None:
            trajectory.append(_trace_record(
                sample_id=int(row["sample_id"]), action_index=0, state=initial_state, config=self.config,
            ))
        post_action_collision = False
        pid_integral = torch.zeros((1, 6), dtype=torch.float64)
        previous_error = torch.zeros((1, 6), dtype=torch.float64)
        predicted_pass = False
        initial_topology = classify_lateral_contact_state(
            estimated_dimensions, measured_pose,
            warning_clearance_um=self.config.topology_warning_clearance_um,
        ).topology

        for cycle in range(self.config.max_actions):
            measured_tensor = _tensor(_pose_values(measured_pose))
            predicted_current = self.nominal_physics.evaluate_pose(measured_tensor, estimated_dim_tensor)
            individual_predicted_states = [
                self.nominal_physics.evaluate_pose(_tensor(_pose_values(item)), estimated_dim_tensor)
                for item in measurement_samples
            ]
            contact_state = classify_lateral_contact_state(
                estimated_dimensions, measured_pose,
                warning_clearance_um=self.config.topology_warning_clearance_um,
            )
            topology_guarded = contact_state.safety_level != "安全"
            topology_guarded_cycles += int(topology_guarded and method == "mlp_geometry_closed_loop")
            predicted_pass = _state_pass(predicted_current)
            if predicted_pass:
                break
            if (
                method == "mlp_geometry_closed_loop"
                and cycle > 0
                and not topology_guarded
                and not _confirmed_feedback_needed(predicted_current, individual_predicted_states, self.config)
            ):
                # 未达到双复测确认条件时停止，避免把一次随机读数当作执行残余。
                break

            if method == "analytic_open":
                if cycle:
                    break
                command = _tensor(_pose_values(analytic_compensation(measured_pose).adjustment))
            elif method in {"mlp_open", "mlp_open_fused_initial"}:
                if cycle:
                    break
                command = self._mlp_command(measured_pose, row)
            elif method == "pid_feedback":
                error = measured_tensor
                pid_integral = torch.clamp(
                    pid_integral + error / _STEP_LIMIT,
                    -self.config.pid_integral_limit,
                    self.config.pid_integral_limit,
                )
                derivative = error - previous_error
                previous_error = error
                command = -(
                    self.config.pid_p * error
                    + self.config.pid_i * pid_integral * _STEP_LIMIT
                    + self.config.pid_d * derivative
                )
            else:
                first_action = cycle == 0
                if first_action and not topology_guarded:
                    desired_candidates = [self._mlp_command(measured_pose, row)]
                elif topology_guarded:
                    # 接触风险时只允许精确几何脱困方向，不能让数据驱动命令冒险。
                    desired_candidates = [
                        _dls_feedback_command(
                            measured_tensor, estimated_dim_tensor, self.nominal_physics,
                            self.config, prediction_horizon=prediction_horizon,
                        )
                    ]
                else:
                    # 二、三轮同时生成“已适配 MLP 残余再推理”和“局部几何 DLS”两种
                    # 候选；安全筛选后选择预测约束违规量更低的一项。旧版只允许 DLS，
                    # 会丢失 MLP 已学到的测量零偏、执行器比例误差和轴间耦合信息。
                    desired_candidates = [
                        self._mlp_command(measured_pose, row),
                        _dls_feedback_command(
                            measured_tensor, estimated_dim_tensor, self.nominal_physics,
                            self.config, prediction_horizon=prediction_horizon,
                        ),
                    ]
                safe_candidates: list[Tensor] = []
                rejected = 0
                for desired in desired_candidates:
                    candidate, candidate_rejected = _safe_candidate(
                        predicted_current, measured_tensor, estimated_dim_tensor, desired,
                        self.nominal_physics, self.config,
                        require_improvement=(not first_action) or topology_guarded,
                    )
                    rejected += candidate_rejected
                    if candidate is not None:
                        safe_candidates.append(candidate)
                command = min(
                    safe_candidates,
                    key=lambda candidate: _violation_score(
                        self.nominal_physics(measured_tensor, candidate, estimated_dim_tensor),
                        self.config.safety_guard_um,
                    ),
                    default=None,
                )
                if command is None and first_action and self.config.allow_collision_recovery:
                    # 前馈首轮无法直接进入安全区时，不冒险强行执行 MLP 指令；改用
                    # 可导几何层给出的局部脱困方向，并只接受“间隙增大 + 违规下降”的步。
                    desired = _dls_feedback_command(
                        measured_tensor, estimated_dim_tensor, self.nominal_physics, self.config,
                        prediction_horizon=prediction_horizon,
                    )
                    command, recovery_rejected = _safe_candidate(
                        predicted_current, measured_tensor, estimated_dim_tensor, desired,
                        self.nominal_physics, self.config, require_improvement=True,
                    )
                    rejections += recovery_rejected
                rejections += rejected
                if command is None:
                    break

            command = _clip_command(command)
            physical_adjustment = domain.actuator.sample_physical_adjustment(_pose_from_values(command[0].tolist()), execution_rng)
            true_pose = apply_world_frame_adjustment(true_pose, physical_adjustment)
            actions += 1
            actual_state = plant_geometry.evaluate_pose(_tensor(_pose_values(true_pose)), true_dim_tensor)
            post_action_collision = post_action_collision or not bool(actual_state.minimum_clearance_um[0] > 0.0)
            if trajectory is not None:
                trajectory.append(_trace_record(
                    sample_id=int(row["sample_id"]), action_index=actions, state=actual_state, config=self.config,
                ))
            if method == "mlp_geometry_closed_loop":
                measurement_samples = [
                    domain.measurement.measure(true_pose, measurement_rng)
                    for _ in range(self.config.confirmation_measurements)
                ]
                measured_pose = _average_pose(measurement_samples)
            else:
                measured_pose = domain.measurement.measure(true_pose, measurement_rng)
                measurement_samples = [measured_pose]

        actual_final = plant_geometry.evaluate_pose(_tensor(_pose_values(true_pose)), true_dim_tensor)
        # 在线系统只能依据最新复测判断停止；离线仿真则以真实虚拟工厂核验最终结果。
        latest_measured = _tensor(_pose_values(measured_pose))
        predicted_pass = _state_pass(self.nominal_physics.evaluate_pose(latest_measured, estimated_dim_tensor))
        final_topology = classify_lateral_contact_state(
            estimated_dimensions, measured_pose,
            warning_clearance_um=self.config.topology_warning_clearance_um,
        ).topology
        (
            gap_pass, no_collision, residual_pass, margin, cmin, trans_mae, rot_mae,
            maximum_translation, maximum_rotation,
        ) = _state_metrics(actual_final)
        final_gaps = actual_final.labelled_gaps_um[0].tolist()
        return TrialRecord(
            sample_id=int(row["sample_id"]), method=method,
            joint_pass=gap_pass and no_collision and residual_pass,
            gap_pass=gap_pass, no_collision=no_collision, residual_pass=residual_pass,
            initial_collision=initial_collision, post_action_collision=post_action_collision,
            actions=actions, online_predicted_pass=predicted_pass,
            predicted_safety_rejections=rejections,
            initial_predicted_contact_topology=initial_topology,
            final_predicted_contact_topology=final_topology,
            topology_guarded_cycles=topology_guarded_cycles,
            gap_margin_um=margin, minimum_clearance_um=cmin,
            g1_um=float(final_gaps[0]), g2_um=float(final_gaps[1]),
            g3_um=float(final_gaps[2]), g4_um=float(final_gaps[3]),
            translation_mae_um=trans_mae, rotation_mae_arcsec=rot_mae,
            maximum_abs_translation_um=maximum_translation, maximum_abs_rotation_arcsec=maximum_rotation,
            decision_time_ms=(time.perf_counter() - start) * 1000.0,
        )

    def evaluate(self, rows: list[dict[str, str]], domain: SimulationDomain, *, seed: int) -> list[TrialRecord]:
        records: list[TrialRecord] = []
        for row in rows:
            sample_seed = seed + int(row["sample_id"]) * 10_000
            for method in ("analytic_open", "mlp_open", "mlp_open_fused_initial", "pid_feedback", "mlp_geometry_closed_loop"):
                records.append(self._run_method(row, domain, method, sample_seed))
        return records

    def evaluate_prediction_horizons(
        self,
        rows: list[dict[str, str]],
        domain: SimulationDomain,
        *,
        horizons: tuple[int, ...] = (1, 2, 3),
        seed: int,
    ) -> list[TrialRecord]:
        """在相同复测闭环与动作上限下，比较 H=1/2/3 的预测时域。"""
        if not horizons or any(horizon <= 0 for horizon in horizons):
            raise ValueError("预测时域必须包含正整数。")
        records: list[TrialRecord] = []
        for row in rows:
            sample_seed = seed + int(row["sample_id"]) * 10_000
            for horizon in horizons:
                record = self._run_method(
                    row,
                    domain,
                    "mlp_geometry_closed_loop",
                    sample_seed,
                    prediction_horizon=horizon,
                )
                records.append(replace(record, method=f"horizon_H{horizon}"))
        return records

    def evaluate_main_method(self, rows: list[dict[str, str]], domain: SimulationDomain, *, seed: int) -> list[TrialRecord]:
        """只评估主闭环，供安全策略消融使用，避免重复运行无关基线。"""
        records: list[TrialRecord] = []
        for row in rows:
            sample_seed = seed + int(row["sample_id"]) * 10_000
            records.append(self._run_method(row, domain, "mlp_geometry_closed_loop", sample_seed))
        return records

    def evaluate_main_method_with_trajectories(
        self,
        rows: list[dict[str, str]],
        domain: SimulationDomain,
        *,
        seed: int,
    ) -> tuple[list[TrialRecord], list[ConvergenceTraceRecord]]:
        """运行主闭环并保存每一次真实动作后的离线稳定性轨迹。"""
        records: list[TrialRecord] = []
        trajectories: list[ConvergenceTraceRecord] = []
        for row in rows:
            sample_seed = seed + int(row["sample_id"]) * 10_000
            records.append(self._run_method(
                row, domain, "mlp_geometry_closed_loop", sample_seed, trajectory=trajectories,
            ))
        return records, trajectories


def summarize_records(
    records: list[TrialRecord],
    *,
    method_order: tuple[str, ...] = (
        "analytic_open", "mlp_open", "mlp_open_fused_initial", "pid_feedback", "mlp_geometry_closed_loop",
    ),
) -> list[dict[str, float | int | str]]:
    """计算报告所需的联合 PASS、分项安全、收敛次数与效率指标。"""
    summaries: list[dict[str, float | int | str]] = []
    for method in method_order:
        selected = [record for record in records if record.method == method]
        if not selected:
            continue
        count = len(selected)
        summaries.append({
            "method": method,
            "sample_count": count,
            "joint_pass_rate": sum(item.joint_pass for item in selected) / count,
            "gap_pass_rate": sum(item.gap_pass for item in selected) / count,
            "no_collision_rate": sum(item.no_collision for item in selected) / count,
            "residual_pass_rate": sum(item.residual_pass for item in selected) / count,
            "initial_collision_rate": sum(item.initial_collision for item in selected) / count,
            "post_action_collision_rate": sum(item.post_action_collision for item in selected) / count,
            "mean_actions": sum(item.actions for item in selected) / count,
            "pass_within_1_action_rate": sum(item.joint_pass and item.actions <= 1 for item in selected) / count,
            "pass_within_2_actions_rate": sum(item.joint_pass and item.actions <= 2 for item in selected) / count,
            "pass_within_3_actions_rate": sum(item.joint_pass and item.actions <= 3 for item in selected) / count,
            "mean_gap_margin_um": sum(item.gap_margin_um for item in selected) / count,
            "mean_minimum_clearance_um": sum(item.minimum_clearance_um for item in selected) / count,
            "translation_mae_um": sum(item.translation_mae_um for item in selected) / count,
            "rotation_mae_arcsec": sum(item.rotation_mae_arcsec for item in selected) / count,
            "mean_decision_time_ms": sum(item.decision_time_ms for item in selected) / count,
            "mean_predicted_safety_rejections": sum(item.predicted_safety_rejections for item in selected) / count,
            "topology_guarded_cycle_count": sum(item.topology_guarded_cycles for item in selected),
        })
    return summaries


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_closed_loop_experiment(
    *,
    data_directory: Path,
    source_model_directory: Path,
    adaptation_directory: Path,
    output_directory: Path,
    operational_limit: int = 1000,
    stress_limit: int = 1000,
    seed: int = 20260918,
    config: ClosedLoopConfig = ClosedLoopConfig(),
) -> dict[str, object]:
    """运行开发阶段闭环评估；不触碰已封存的 M1 最终测试集。"""
    if operational_limit <= 0 or stress_limit <= 0:
        raise ValueError("评估样本数必须为正整数。")
    output_directory.mkdir(parents=True, exist_ok=True)
    with (data_directory / "target_operational_test.csv").open(encoding="utf-8", newline="") as handle:
        operational_rows = list(csv.DictReader(handle))[:operational_limit]
    with (data_directory / "target_stress_test.csv").open(encoding="utf-8", newline="") as handle:
        stress_rows = list(csv.DictReader(handle))[:stress_limit]
    if len(operational_rows) != operational_limit or len(stress_rows) != stress_limit:
        raise ValueError("请求样本数超过可用开发数据。")
    evaluator = ClosedLoopEvaluator(
        source_model_directory=source_model_directory,
        adaptation_directory=adaptation_directory,
        config=config,
    )
    operational = evaluator.evaluate(operational_rows, make_target_domain(), seed=seed)
    stress = evaluator.evaluate(stress_rows, make_stress_domain(), seed=seed + 1_000_000)
    operational_summary = summarize_records(operational)
    stress_summary = summarize_records(stress)
    _write_csv(output_directory / "operational_records.csv", [asdict(item) for item in operational])
    _write_csv(output_directory / "stress_records.csv", [asdict(item) for item in stress])
    _write_csv(output_directory / "summary.csv", [
        {"scenario": "M1_operational_development", **row} for row in operational_summary
    ] + [{"scenario": "Mstress_safety_only", **row} for row in stress_summary])
    result: dict[str, object] = {
        "purpose": "development_closed_loop_comparison_not_final_holdout",
        "method": _CONTROLLER_ID,
        "feature_guard": "C15 whitelist only; true_* only used by virtual plant and offline verification",
        "config": asdict(config),
        "operational": operational_summary,
        "stress_safety_only": stress_summary,
    }
    (output_directory / "closed_loop_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_size_boundary_closed_loop_evaluation(
    *,
    data_directory: Path,
    source_model_directory: Path,
    adaptation_directory: Path,
    output_directory: Path,
    seed: int = 20261021,
    config: ClosedLoopConfig = ClosedLoopConfig(),
) -> dict[str, object]:
    """在 Msize 两端可行尺寸域上冻结评估四种控制方法。

    该测试不混入生产分布合格率，只回答一个单独问题：处于 c=5 或 c=8
    附近、但仍在物理可行域内的选配件，闭环能否保持安全与残余精度。
    """
    source_path = data_directory / "target_size_boundary_test.csv"
    with source_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Msize 尺寸边界测试集不能为空。")
    output_directory.mkdir(parents=True, exist_ok=True)
    evaluator = ClosedLoopEvaluator(source_model_directory=source_model_directory, adaptation_directory=adaptation_directory, config=config)
    records = evaluator.evaluate(rows, make_size_boundary_domain(), seed=seed)
    summaries = summarize_records(records)
    band_by_sample_id = {int(row["sample_id"]): str(row["size_band"]) for row in rows}
    by_band: dict[str, list[TrialRecord]] = {"lower_boundary": [], "upper_boundary": []}
    for record in records:
        by_band[band_by_sample_id[record.sample_id]].append(record)
    by_band_summary = {band: summarize_records(items) for band, items in by_band.items()}
    _write_csv(output_directory / "size_boundary_records.csv", [asdict(item) for item in records])
    _write_csv(output_directory / "size_boundary_summary.csv", summaries)
    payload: dict[str, object] = {"purpose": "full_feasible_dimension_boundary_closed_loop_evaluation", "dataset": str(source_path), "sample_count": len(rows), "dimension_bands": {band: len({item.sample_id for item in items}) for band, items in by_band.items()}, "controller": _CONTROLLER_ID, "config": asdict(config), "overall": summaries, "by_size_band": by_band_summary}
    (output_directory / "size_boundary_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _distribution_summary(values: list[float]) -> dict[str, float]:
    """无外部统计依赖地生成报告所需的分布摘要。"""
    if not values:
        raise ValueError("不能汇总空分布。")
    ordered = sorted(values)

    def quantile(probability: float) -> float:
        index = round((len(ordered) - 1) * probability)
        return float(ordered[index])

    return {
        "min": float(ordered[0]),
        "p01": quantile(0.01),
        "p05": quantile(0.05),
        "median": quantile(0.50),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": float(ordered[-1]),
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
    }


def run_closed_loop_monte_carlo(
    *,
    data_directory: Path,
    source_model_directory: Path,
    adaptation_directory: Path,
    output_directory: Path,
    sample_count: int = 10_000,
    seed: int = 20261020,
    config: ClosedLoopConfig = ClosedLoopConfig(),
) -> dict[str, object]:
    """执行任务书要求的 N>=10000 次主闭环蒙特卡洛仿真。

    仅使用已冻结的 30 条 M1 适配检查点与 H=1 安全闭环控制器。CSV 中的
    ``true_*`` 只由虚拟工厂执行和离线验收读取；在线控制器仍然只接收 C15
    白名单可得特征。逐件 g1~g4 会保存，因而结果可直接统计间隙分布与最坏件。
    """
    if sample_count < 10_000:
        raise ValueError("任务书闭环蒙特卡洛要求 sample_count 不得小于 10000。")
    source_path = data_directory / "monte_carlo_operational_10000.csv"
    with source_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))[:sample_count]
    if len(rows) != sample_count:
        raise ValueError(f"闭环蒙特卡洛数据不足：要求 {sample_count}，实际 {len(rows)}。")
    output_directory.mkdir(parents=True, exist_ok=True)
    evaluator = ClosedLoopEvaluator(
        source_model_directory=source_model_directory,
        adaptation_directory=adaptation_directory,
        config=config,
    )
    records = evaluator.evaluate_main_method(rows, make_target_domain(), seed=seed)
    summary = summarize_records(records, method_order=("mlp_geometry_closed_loop",))
    if len(summary) != 1:
        raise AssertionError("主闭环蒙特卡洛汇总应恰好包含一个控制器。")
    method_summary = summary[0]
    gap_distributions = {
        label: _distribution_summary([float(getattr(record, f"{label}_um")) for record in records])
        for label in ("g1", "g2", "g3", "g4")
    }
    worst_by_margin = min(records, key=lambda item: item.gap_margin_um)
    worst_by_translation = max(records, key=lambda item: item.maximum_abs_translation_um)
    worst_by_rotation = max(records, key=lambda item: item.maximum_abs_rotation_arcsec)
    _write_csv(output_directory / "closed_loop_monte_carlo_records.csv", [asdict(item) for item in records])
    _write_csv(output_directory / "closed_loop_monte_carlo_summary.csv", summary)
    result: dict[str, object] = {
        "purpose": "task_book_N_ge_10000_online_closed_loop_monte_carlo",
        "sample_count": sample_count,
        "evaluation_seed": seed,
        "controller": _CONTROLLER_ID,
        "adaptation_checkpoint": str(adaptation_directory / "target_adapt_30.pt"),
        "feature_guard": "C15 whitelist only; true_* only used by virtual plant and offline verification",
        "config": asdict(config),
        "method": method_summary,
        "final_gap_distribution_um": gap_distributions,
        "worst_cases": {
            "smallest_gap_margin": asdict(worst_by_margin),
            "largest_translation_residual": asdict(worst_by_translation),
            "largest_rotation_residual": asdict(worst_by_rotation),
        },
        "warning": "This is the frozen main controller's closed-loop MC result, not a single-forward-pass network evaluation.",
    }
    (output_directory / "closed_loop_monte_carlo_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _target_domain_with_noise_scale(scale: float) -> SimulationDomain:
    """只放大 M1 的随机测量/执行扰动，固定偏置、耦合、尺寸与初始误差范围。

    这是一项任务范围内的抗扰扫描：它并不伪造新的机器失配，也不扩大题设的
    六维初始误差包络。``scale=1`` 即 M1 名义随机噪声，1.5 和 2 代表更差的
    传感器重复性与执行重复定位性。
    """
    if scale <= 0.0:
        raise ValueError("扰动放大系数必须为正数。")
    base = make_target_domain()
    measurement = replace(
        base.measurement,
        translation_noise_std_um=base.measurement.translation_noise_std_um * scale,
        rotation_noise_std_arcsec=base.measurement.rotation_noise_std_arcsec * scale,
    )
    actuator = replace(
        base.actuator,
        translation_noise_std_um=base.actuator.translation_noise_std_um * scale,
        rotation_noise_std_arcsec=base.actuator.rotation_noise_std_arcsec * scale,
    )
    return replace(base, name=f"M1_noise_{scale:g}x", measurement=measurement, actuator=actuator)


def _rows_with_resampled_initial_measurement(
    rows: list[dict[str, str]],
    domain: SimulationDomain,
    *,
    seed: int,
) -> list[dict[str, str]]:
    """对同一真实初始状态重测，形成同随机标准正态轨迹下的不同噪声强度输入。"""
    resampled: list[dict[str, str]] = []
    suffixes = ("dx_um", "dy_um", "dz_um", "theta_x_arcsec", "theta_y_arcsec", "theta_z_arcsec")
    for row in rows:
        copy = row.copy()
        sample_seed = seed + int(row["sample_id"]) * 10_000 + 401
        measured = domain.measurement.measure(_row_pose(row, "true"), random.Random(sample_seed))
        for suffix, value in zip(suffixes, _pose_values(measured), strict=True):
            copy[f"measured_{suffix}"] = repr(value)
        resampled.append(copy)
    return resampled


def run_disturbance_robustness_scan(
    *,
    data_directory: Path,
    source_model_directory: Path,
    adaptation_directory: Path,
    output_directory: Path,
    noise_scales: tuple[float, ...] = (1.0, 1.5, 2.0),
    operational_limit: int = 1000,
    seed: int = 20261023,
    config: ClosedLoopConfig = ClosedLoopConfig(),
) -> dict[str, object]:
    """在固定 M1 初始状态上扫描测量/执行随机扰动，量化抗扰鲁棒性。

    每个倍率下的四种方法共享同一行、同一初始重测轨迹、同一执行噪声轨迹和
    相同动作上限；所以性能变化可归因于随机扰动强度，而不是样本构成改变。
    """
    if not noise_scales or any(scale <= 0 for scale in noise_scales):
        raise ValueError("噪声倍率必须为正数。")
    if operational_limit <= 0:
        raise ValueError("评估样本数必须为正整数。")
    with (data_directory / "target_operational_test.csv").open(encoding="utf-8", newline="") as handle:
        base_rows = list(csv.DictReader(handle))[:operational_limit]
    if len(base_rows) != operational_limit:
        raise ValueError("请求样本数超过可用 M1 开发运行集。")
    output_directory.mkdir(parents=True, exist_ok=True)
    evaluator = ClosedLoopEvaluator(
        source_model_directory=source_model_directory,
        adaptation_directory=adaptation_directory,
        config=config,
    )
    all_records: list[dict[str, object]] = []
    flattened: list[dict[str, float | int | str]] = []
    summaries: dict[str, list[dict[str, float | int | str]]] = {}
    for scale in noise_scales:
        domain = _target_domain_with_noise_scale(float(scale))
        rows = _rows_with_resampled_initial_measurement(base_rows, domain, seed=seed)
        records = evaluator.evaluate(rows, domain, seed=seed)
        summary = summarize_records(records)
        scenario = f"M1_noise_{scale:g}x"
        summaries[scenario] = summary
        all_records.extend({"noise_scale": scale, **asdict(record)} for record in records)
        flattened.extend({"noise_scale": scale, **item} for item in summary)
    _write_csv(output_directory / "disturbance_scan_records.csv", all_records)
    _write_csv(output_directory / "disturbance_scan_summary.csv", flattened)
    result: dict[str, object] = {
        "purpose": "task_range_measurement_and_actuator_random_disturbance_robustness_scan",
        "noise_scales": list(noise_scales),
        "sample_count_per_scale": operational_limit,
        "controller": _CONTROLLER_ID,
        "config": asdict(config),
        "controlled_variable": "measurement and actuator random noise standard deviations only",
        "fixed_elements": "same true initial states, dimensions, systematic biases, couplings, random-standard-normal trajectories, action limit and C15 checkpoint",
        "feature_guard": "C15 whitelist only; true_* only used by virtual plant and offline verification",
        "summaries": summaries,
    }
    (output_directory / "disturbance_scan_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def run_convergence_stability_analysis(
    *,
    data_directory: Path,
    source_model_directory: Path,
    adaptation_directory: Path,
    output_directory: Path,
    operational_limit: int = 1000,
    seed: int = 20261024,
    config: ClosedLoopConfig = ClosedLoopConfig(),
) -> dict[str, object]:
    """量化主闭环的逐轮收敛与过程安全，服务任务书的稳定性分析。

    采用 M1 开发运行集而非封存集；所有控制器参数、30 条适配检查点和最多三次
    动作限制都已冻结。违规量由间隙越界、0.05 um 安全线与六维残余越界连续化
    得到，故不同量纲可以在同一张收敛图中观察。随机执行扰动下不要求每一件都
    严格单调，但要求总体违规量下降且全过程不产生新的碰撞。
    """
    if operational_limit <= 0:
        raise ValueError("评估样本数必须为正整数。")
    with (data_directory / "target_operational_test.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))[:operational_limit]
    if len(rows) != operational_limit:
        raise ValueError("请求样本数超过可用 M1 开发运行集。")
    output_directory.mkdir(parents=True, exist_ok=True)
    evaluator = ClosedLoopEvaluator(
        source_model_directory=source_model_directory,
        adaptation_directory=adaptation_directory,
        config=config,
    )
    records, traces = evaluator.evaluate_main_method_with_trajectories(
        rows, make_target_domain(), seed=seed,
    )
    if sum(item.action_index == 0 for item in traces) != operational_limit:
        raise AssertionError("每件样本必须恰好记录一个初始收敛状态。")
    if any(not item.no_collision for item in traces):
        # M1 正常工况下，安全线搜索不应让真实虚拟工厂状态发生碰撞。
        raise AssertionError("收敛轨迹发现过程碰撞，不能宣称该配置过程安全。")

    summary_rows: list[dict[str, object]] = []
    for action_index in range(config.max_actions + 1):
        selected = [item for item in traces if item.action_index == action_index]
        if not selected:
            continue
        scores = [item.violation_score for item in selected]
        score_distribution = _distribution_summary(scores)
        summary_rows.append({
            "action_index": action_index,
            "sample_count": len(selected),
            "coverage_of_all_samples": len(selected) / operational_limit,
            "mean_violation_score": score_distribution["mean"],
            "median_violation_score": score_distribution["median"],
            "p95_violation_score": score_distribution["p95"],
            "joint_pass_rate": sum(item.joint_pass for item in selected) / len(selected),
            "gap_pass_rate": sum(item.gap_pass for item in selected) / len(selected),
            "no_collision_rate": sum(item.no_collision for item in selected) / len(selected),
            "residual_pass_rate": sum(item.residual_pass for item in selected) / len(selected),
            "mean_gap_margin_um": statistics.mean(item.gap_margin_um for item in selected),
            "mean_minimum_clearance_um": statistics.mean(item.minimum_clearance_um for item in selected),
        })

    by_sample: dict[int, list[ConvergenceTraceRecord]] = {}
    for item in traces:
        by_sample.setdefault(item.sample_id, []).append(item)
    deltas: list[float] = []
    nonincreasing_count = 0
    transition_groups: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for trajectory in by_sample.values():
        trajectory.sort(key=lambda item: item.action_index)
        for previous, current in zip(trajectory, trajectory[1:], strict=False):
            delta = current.violation_score - previous.violation_score
            deltas.append(delta)
            nonincreasing_count += delta <= 1e-12
            transition_groups.setdefault((previous.action_index, current.action_index), []).append(
                (previous.violation_score, current.violation_score)
            )
    per_transition_summary: list[dict[str, object]] = []
    for (from_action, to_action), pairs in sorted(transition_groups.items()):
        before_scores = [item[0] for item in pairs]
        after_scores = [item[1] for item in pairs]
        score_changes = [after - before for before, after in pairs]
        per_transition_summary.append({
            "from_action_index": from_action,
            "to_action_index": to_action,
            "sample_count": len(pairs),
            "mean_violation_before": statistics.mean(before_scores),
            "mean_violation_after": statistics.mean(after_scores),
            "median_violation_before": statistics.median(before_scores),
            "median_violation_after": statistics.median(after_scores),
            "mean_violation_score_change": statistics.mean(score_changes),
            "nonincreasing_transition_rate": sum(change <= 1e-12 for change in score_changes) / len(score_changes),
        })
    transition_summary = {
        "transition_count": len(deltas),
        "mean_violation_score_change": statistics.mean(deltas) if deltas else 0.0,
        "median_violation_score_change": statistics.median(deltas) if deltas else 0.0,
        "nonincreasing_transition_rate": nonincreasing_count / len(deltas) if deltas else 1.0,
        "process_collision_count": 0,
    }
    _write_csv(output_directory / "convergence_trace_records.csv", [asdict(item) for item in traces])
    _write_csv(output_directory / "convergence_summary.csv", summary_rows)
    _write_csv(output_directory / "convergence_transition_summary.csv", per_transition_summary)
    result: dict[str, object] = {
        "purpose": "development_convergence_and_stability_analysis_not_final_holdout",
        "controller": _CONTROLLER_ID,
        "sample_count": operational_limit,
        "evaluation_seed": seed,
        "adaptation_checkpoint": str(adaptation_directory / "target_adapt_30.pt"),
        "violation_score_definition": "sum of squared normalized gap-boundary, 0.05um safety-guard and six-dimensional residual overflows",
        "config": asdict(config),
        "per_action_summary": summary_rows,
        "per_transition_summary": per_transition_summary,
        "transition_summary": transition_summary,
        "final_control_summary": summarize_records(records, method_order=("mlp_geometry_closed_loop",)),
        "interpretation_guard": "Per-sample scores need not be strictly monotone under fresh execution noise; judge average decrease, pass progression and zero process collision together.",
        "feature_guard": "C15 whitelist only; true_* only used by virtual plant and offline verification",
    }
    (output_directory / "convergence_stability_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def run_prediction_horizon_ablation(
    *,
    data_directory: Path,
    source_model_directory: Path,
    adaptation_directory: Path,
    output_directory: Path,
    operational_limit: int = 1000,
    seed: int = 20260919,
    config: ClosedLoopConfig = ClosedLoopConfig(),
) -> dict[str, object]:
    """固定主模型和闭环参数后，比较 H=1/2/3 的几何预测时域。

    该实验只使用已使用过的 M1 开发运行集，服务于控制器结构选择；它不触碰
    封存最终集，也不据结果反过来改主方法参数。
    """
    if operational_limit <= 0:
        raise ValueError("评估样本数必须为正整数。")
    with (data_directory / "target_operational_test.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))[:operational_limit]
    if len(rows) != operational_limit:
        raise ValueError("请求样本数超过可用 M1 开发运行集。")
    output_directory.mkdir(parents=True, exist_ok=True)
    evaluator = ClosedLoopEvaluator(
        source_model_directory=source_model_directory,
        adaptation_directory=adaptation_directory,
        config=config,
    )
    records = evaluator.evaluate_prediction_horizons(rows, make_target_domain(), seed=seed)
    summary = summarize_records(records, method_order=("horizon_H1", "horizon_H2", "horizon_H3"))
    _write_csv(output_directory / "horizon_records.csv", [asdict(item) for item in records])
    _write_csv(output_directory / "horizon_summary.csv", summary)
    result: dict[str, object] = {
        "purpose": "development_prediction_horizon_ablation_not_final_holdout",
        "fixed_first_action": "C15 MLP adapted with 30 M1 calibration samples",
        "controlled_variable": "geometric DLS prediction horizon H",
        "same_conditions": "same initial rows, maximum three physical actions, measurement/execution noise trajectory and safety guard",
        "controller": _CONTROLLER_ID,
        "config": asdict(config),
        "feature_guard": "C15 whitelist only; true_* only used by virtual plant and offline verification",
        "summary": summary,
    }
    (output_directory / "horizon_ablation_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def evaluate_closed_loop_on_final_holdout(
    *,
    holdout_csv: Path,
    source_model_directory: Path,
    adaptation_directory: Path,
    output_directory: Path,
    seed: int = 20261014,
    config: ClosedLoopConfig = ClosedLoopConfig(),
) -> dict[str, object]:
    """一次性验收冻结 H=1 闭环控制器，不允许据结果再调参。

    该函数专门服务于独立 M1 封存集。它沿用已经冻结的四方法和 H=1 配置，
    将逐件结果、随机种子及控制器参数一并归档，确保报告中的最终数字可以复核。
    """
    if not holdout_csv.is_file():
        raise FileNotFoundError(f"找不到封存测试集：{holdout_csv}")
    with holdout_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("封存测试集不能为空。")
    output_directory.mkdir(parents=True, exist_ok=True)
    evaluator = ClosedLoopEvaluator(
        source_model_directory=source_model_directory,
        adaptation_directory=adaptation_directory,
        config=config,
    )
    records = evaluator.evaluate(rows, make_target_domain(), seed=seed)
    summary = summarize_records(records)
    _write_csv(output_directory / "closed_loop_final_records.csv", [asdict(item) for item in records])
    _write_csv(output_directory / "closed_loop_final_summary.csv", summary)
    result: dict[str, object] = {
        "purpose": "final_holdout_closed_loop_evaluation_model_and_controller_frozen",
        "holdout_csv": str(holdout_csv),
        "sample_count": len(rows),
        "evaluation_seed": seed,
        "controller": _CONTROLLER_ID,
        "config": asdict(config),
        "feature_guard": "C15 whitelist only; true_* only used by virtual plant and offline verification",
        "methods": summary,
        "warning": "This final holdout must not be used to select parameters or modify the model/controller.",
    }
    (output_directory / "closed_loop_final_evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def run_collision_recovery_ablation(
    *,
    data_directory: Path,
    source_model_directory: Path,
    adaptation_directory: Path,
    output_directory: Path,
    seed: int = 20260919,
    config: ClosedLoopConfig = ClosedLoopConfig(),
) -> dict[str, object]:
    """开发阶段比较严格安全 H=1 与受约束脱困 H=1。

    受约束脱困只在“预测已碰撞”时允许最小间隙严格增大且违规量严格下降的步；
    正常 M1 运行集和 Mstress 安全集均使用相同的冻结 MLP、噪声种子、动作上限。
    该消融不得触碰或覆盖封存集。
    """
    with (data_directory / "target_operational_test.csv").open(encoding="utf-8", newline="") as handle:
        operational_rows = list(csv.DictReader(handle))
    with (data_directory / "target_stress_test.csv").open(encoding="utf-8", newline="") as handle:
        stress_rows = list(csv.DictReader(handle))
    output_directory.mkdir(parents=True, exist_ok=True)
    variants = (
        ("strict_H1", config),
        ("recovery_H1", replace(config, allow_collision_recovery=True)),
    )
    all_records: list[dict[str, object]] = []
    result_summaries: dict[str, list[dict[str, float | int | str]]] = {}
    for scenario, rows, domain, scenario_seed in (
        ("M1_operational_development", operational_rows, make_target_domain(), seed),
        ("Mstress_safety_development", stress_rows, make_stress_domain(), seed + 1_000_000),
    ):
        scenario_records: list[TrialRecord] = []
        for variant, variant_config in variants:
            evaluator = ClosedLoopEvaluator(
                source_model_directory=source_model_directory,
                adaptation_directory=adaptation_directory,
                config=variant_config,
            )
            scenario_records.extend(
                replace(record, method=variant)
                for record in evaluator.evaluate_main_method(rows, domain, seed=scenario_seed)
            )
        summaries = summarize_records(scenario_records, method_order=("strict_H1", "recovery_H1"))
        result_summaries[scenario] = summaries
        all_records.extend({"scenario": scenario, **asdict(record)} for record in scenario_records)
    _write_csv(output_directory / "collision_recovery_records.csv", all_records)
    flattened = [
        {"scenario": scenario, **summary}
        for scenario, summaries in result_summaries.items()
        for summary in summaries
    ]
    _write_csv(output_directory / "collision_recovery_summary.csv", flattened)
    result: dict[str, object] = {
        "purpose": "development_collision_recovery_ablation_not_final_holdout",
        "fixed_elements": "same C15 MLP checkpoint, H=1 DLS geometry, seeds, action limit and virtual plant",
        "controlled_variable": "allow_collision_recovery",
        "recovery_rule": "only when predicted current minimum clearance < 0, accept a step only if minimum clearance strictly increases and total constraint violation strictly decreases",
        "scenarios": result_summaries,
    }
    (output_directory / "collision_recovery_ablation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
