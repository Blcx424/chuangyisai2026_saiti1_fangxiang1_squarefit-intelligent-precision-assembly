"""方形轴孔精密装配的基础几何模型。

这一层只负责确定性的坐标变换、间隙计算和判定，不包含随机采样、
控制器或神经网络。这样每一层结果都能单独核对。
"""

from .geometry import (
    labelled_gaps_um,
    minimum_clearance_um,
    evaluate_lateral_fit,
    peg_corners_at_station,
    residual_pose_within_requirement,
    wall_clearances_at_station,
)
from .contact import (
    classify_lateral_contact_state,
    format_contact_state,
    wall_clearance_details_at_station,
    wall_clearance_profile_um,
)
from .compensation import (
    analytic_compensation,
    apply_world_frame_adjustment,
    format_compensation_result,
)
from .objectives import AssemblyObjective, evaluate_assembly_objective
from .schema import (
    ANALYTIC_COMMAND_COLUMNS,
    DIMENSION_COLUMNS,
    ESTIMATED_OPERATING_CLEARANCE_COLUMNS,
    FEATURE_SETS,
    FORBIDDEN_NETWORK_INPUT_PREFIXES,
    MEASURED_POSE_COLUMNS,
    NETWORK_INPUT_COLUMNS,
    NETWORK_INPUT_COLUMNS_WITHOUT_OPERATING_CLEARANCE,
    PRIMARY_TARGET_COLUMNS,
    RESIDUAL_BASELINE_TARGET_COLUMNS,
    THERMAL_MEASUREMENT_COLUMNS,
    validated_training_columns,
)
from .preprocessing import StandardScaler, fit_standard_scaler
from .simulation import (
    ActuatorModel,
    MeasurementModel,
    SimulatedAssembly,
    SimulationDomain,
    ThermalCondition,
    ThermalModel,
    thermal_clearance_sensitivity,
    classify_risk_layer,
    classify_size_band,
    make_safety_auxiliary_domain,
    make_size_boundary_domain,
    make_source_domain,
    make_stress_domain,
    make_target_domain,
    residual_summary,
    save_dataset_bundle,
    save_dataset_csv,
    save_incoming_pair_screening_csv,
    simulate_assembly,
    simulate_many,
    validate_dataset_samples,
)
# 纯几何、接触分类和数据协议不应被 PyTorch 绑死：没有训练环境时仍可独立
# 复核几何与边界样本。训练与闭环入口在实际使用时会自然要求 PyTorch。
try:
    from .torch_model import InverseCompensationMLP
    from .torch_training import (
        TrainingConfig,
        evaluate_saved_source_model,
        evaluate_frozen_target_models_on_holdout,
        run_target_adaptation_experiment,
        run_target_adaptation_replicates,
        train_source_model,
    )
    from .closed_loop import (
        ClosedLoopConfig,
        ConvergenceTraceRecord,
        TrialRecord,
        run_collision_recovery_ablation,
        run_closed_loop_experiment,
        run_size_boundary_closed_loop_evaluation,
        run_closed_loop_monte_carlo,
        run_disturbance_robustness_scan,
        run_convergence_stability_analysis,
        evaluate_closed_loop_on_final_holdout,
        run_prediction_horizon_ablation,
        summarize_records,
    )
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
from .types import (
    AnalyticCompensationResult,
    AssemblyResult,
    ContactState,
    Dimensions,
    Pose6D,
    WallClearance,
)
from .units import arcsec_to_rad, mm_to_um, rad_to_arcsec, um_to_mm

__all__ = [
    "AssemblyResult",
    "AnalyticCompensationResult",
    "ANALYTIC_COMMAND_COLUMNS",
    "AssemblyObjective",
    "ActuatorModel",
    "ContactState",
    "ClosedLoopConfig",
    "ConvergenceTraceRecord",
    "Dimensions",
    "DIMENSION_COLUMNS",
    "ESTIMATED_OPERATING_CLEARANCE_COLUMNS",
    "FEATURE_SETS",
    "FORBIDDEN_NETWORK_INPUT_PREFIXES",
    "MeasurementModel",
    "InverseCompensationMLP",
    "MEASURED_POSE_COLUMNS",
    "Pose6D",
    "SimulatedAssembly",
    "SimulationDomain",
    "StandardScaler",
    "THERMAL_MEASUREMENT_COLUMNS",
    "ThermalCondition",
    "ThermalModel",
    "TrainingConfig",
    "TrialRecord",
    "thermal_clearance_sensitivity",
    "train_source_model",
    "run_target_adaptation_experiment",
    "run_target_adaptation_replicates",
    "run_closed_loop_experiment",
    "run_size_boundary_closed_loop_evaluation",
    "run_closed_loop_monte_carlo",
    "run_disturbance_robustness_scan",
    "run_convergence_stability_analysis",
    "run_collision_recovery_ablation",
    "evaluate_closed_loop_on_final_holdout",
    "run_prediction_horizon_ablation",
    "WallClearance",
    "arcsec_to_rad",
    "analytic_compensation",
    "apply_world_frame_adjustment",
    "classify_lateral_contact_state",
    "classify_risk_layer",
    "classify_size_band",
    "evaluate_lateral_fit",
    "evaluate_saved_source_model",
    "evaluate_frozen_target_models_on_holdout",
    "evaluate_assembly_objective",
    "format_contact_state",
    "format_compensation_result",
    "fit_standard_scaler",
    "labelled_gaps_um",
    "minimum_clearance_um",
    "NETWORK_INPUT_COLUMNS",
    "NETWORK_INPUT_COLUMNS_WITHOUT_OPERATING_CLEARANCE",
    "make_source_domain",
    "make_safety_auxiliary_domain",
    "make_size_boundary_domain",
    "make_stress_domain",
    "make_target_domain",
    "mm_to_um",
    "peg_corners_at_station",
    "PRIMARY_TARGET_COLUMNS",
    "rad_to_arcsec",
    "residual_summary",
    "summarize_records",
    "residual_pose_within_requirement",
    "RESIDUAL_BASELINE_TARGET_COLUMNS",
    "save_dataset_csv",
    "save_dataset_bundle",
    "save_incoming_pair_screening_csv",
    "simulate_assembly",
    "simulate_many",
    "validate_dataset_samples",
    "validated_training_columns",
    "um_to_mm",
    "wall_clearances_at_station",
    "wall_clearance_details_at_station",
    "wall_clearance_profile_um",
]

# 缺少 PyTorch 时，``from squarefit import *`` 仅导出可独立运行的确定性模块；
# 避免纯几何验收被训练依赖阻断。
__all__ = [name for name in __all__ if name in globals()]
