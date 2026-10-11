"""PyTorch 单逆网络的读取、归一化、训练与源域验证流程。"""

from __future__ import annotations

import csv
import json
import random
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from .preprocessing import StandardScaler, fit_standard_scaler
from .schema import PRIMARY_TARGET_COLUMNS, validated_training_columns
from .simulation import SimulationDomain, make_stress_domain, make_target_domain
from .torch_geometry import MeanActuatorLayer, SquareFitPhysicsLayer
from .torch_losses import LossTerms, compute_three_part_loss
from .torch_model import InverseCompensationMLP


TRUE_POSE_COLUMNS = (
    "true_dx_um",
    "true_dy_um",
    "true_dz_um",
    "true_theta_x_arcsec",
    "true_theta_y_arcsec",
    "true_theta_z_arcsec",
)
EFFECTIVE_DIMENSION_COLUMNS = (
    "true_effective_hole_side_um",
    "true_effective_shaft_side_um",
)
TEACHER_MARGIN_COLUMN = "label_optimal_safety_margin_um"


@dataclass(frozen=True)
class TrainingConfig:
    feature_set_name: str = "B_选配间隙BP_9维"
    hidden_size: int = 64
    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs: int = 120
    lambda_physics: float = 2.0
    lambda_optimality: float = 0.05
    seed: int = 20260912


@dataclass(frozen=True)
class PreparedSplit:
    features: Tensor
    targets: Tensor
    true_pose: Tensor
    effective_dimensions: Tensor
    teacher_margin: Tensor


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"数据文件为空：{path}")
    return rows


def _tensor_from_rows(rows: Iterable[dict[str, str]], columns: tuple[str, ...]) -> Tensor:
    return torch.tensor(
        [[float(row[column]) for column in columns] for row in rows],
        dtype=torch.float64,
    )


def _scale_tensor(values: Tensor, scaler: StandardScaler) -> Tensor:
    means = torch.tensor(scaler.means, dtype=values.dtype, device=values.device)
    deviations = torch.tensor(scaler.standard_deviations, dtype=values.dtype, device=values.device)
    return (values - means) / deviations


def prepare_split(
    rows: list[dict[str, str]],
    *,
    feature_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
    input_scaler: StandardScaler,
    target_scaler: StandardScaler,
) -> PreparedSplit:
    """把 CSV 行转换为张量；真实列仅进入训练物理侧，不返回给网络输入。"""
    return PreparedSplit(
        features=_scale_tensor(_tensor_from_rows(rows, feature_columns), input_scaler),
        targets=_scale_tensor(_tensor_from_rows(rows, target_columns), target_scaler),
        true_pose=_tensor_from_rows(rows, TRUE_POSE_COLUMNS),
        effective_dimensions=_tensor_from_rows(rows, EFFECTIVE_DIMENSION_COLUMNS),
        teacher_margin=_tensor_from_rows(rows, (TEACHER_MARGIN_COLUMN,)).squeeze(-1),
    )


def _inverse_target_scale(normalized_prediction: Tensor, target_scaler: StandardScaler) -> Tensor:
    means = torch.tensor(target_scaler.means, dtype=normalized_prediction.dtype, device=normalized_prediction.device)
    deviations = torch.tensor(
        target_scaler.standard_deviations,
        dtype=normalized_prediction.dtype,
        device=normalized_prediction.device,
    )
    return normalized_prediction * deviations + means


def _as_loader(split: PreparedSplit, *, batch_size: int, shuffle: bool, seed: int) -> DataLoader[tuple[Tensor, ...]]:
    dataset = TensorDataset(
        split.features,
        split.targets,
        split.true_pose,
        split.effective_dimensions,
        split.teacher_margin,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _nested_stratified_calibration_rows(
    calibration_path: Path,
    *,
    size: int,
    calibration_seed: int | None,
) -> tuple[list[dict[str, str]], list[str]]:
    """从固定 50 条标定池中取嵌套、分层的 10/30/50 条标定样本。

    ``calibration_seed=None`` 保留初版协议的固定前 N 条选取；给定种子时，
    每一风险层独立打乱后再取 60% normal、30% coverage、10% difficult。
    这使多随机种子复验同时覆盖“优化顺序波动”和“少量标定件组成波动”，但
    不会触碰 M1 运行集、蒙特卡洛集或压力集。
    """
    rows = _read_csv_rows(calibration_path)
    if calibration_seed is None:
        chosen = rows[:size]
    else:
        requested = {
            "normal": int(size * 0.6),
            "coverage": int(size * 0.3),
            "difficult": int(size * 0.1),
        }
        groups: dict[str, list[dict[str, str]]] = {name: [] for name in requested}
        for row in rows:
            layer = row.get("risk_layer")
            if layer not in groups:
                raise ValueError(f"标定池存在未知风险层：{layer!r}")
            groups[layer].append(row)
        rng = random.Random(calibration_seed)
        chosen = []
        for layer, count in requested.items():
            candidates = groups[layer]
            if len(candidates) < count:
                raise ValueError(f"标定池的 {layer} 样本不足：需要 {count}，实际 {len(candidates)}")
            candidates = candidates.copy()
            rng.shuffle(candidates)
            chosen.extend(candidates[:count])
        rng.shuffle(chosen)
    if len(chosen) != size:
        raise AssertionError(f"标定抽样数量错误：期望 {size}，实际 {len(chosen)}")
    return chosen, [row["sample_id"] for row in chosen]


def _loss_for_batch(
    model: InverseCompensationMLP,
    physics_layer: SquareFitPhysicsLayer,
    batch: tuple[Tensor, ...],
    target_scaler: StandardScaler,
    config: TrainingConfig,
) -> tuple[LossTerms, Tensor]:
    features, targets, true_pose, effective_dimensions, teacher_margin = batch
    normalized_prediction = model(features)
    predicted_command = _inverse_target_scale(normalized_prediction, target_scaler)
    physical_state = physics_layer(true_pose, predicted_command, effective_dimensions)
    return (
        compute_three_part_loss(
            normalized_prediction=normalized_prediction,
            normalized_target=targets,
            predicted_command=predicted_command,
            physical_state=physical_state,
            teacher_safety_margin_um=teacher_margin,
            lambda_physics=config.lambda_physics,
            lambda_optimality=config.lambda_optimality,
        ),
        predicted_command,
    )


def evaluate_model(
    model: InverseCompensationMLP,
    physics_layer: SquareFitPhysicsLayer,
    split: PreparedSplit,
    target_scaler: StandardScaler,
    config: TrainingConfig,
) -> dict[str, float]:
    """在冻结验证集上评价损失与端到端物理 PASS，不反传梯度。"""
    model.eval()
    loader = _as_loader(split, batch_size=config.batch_size, shuffle=False, seed=config.seed)
    weighted_terms = {name: 0.0 for name in ("total", "data", "physics", "optimality")}
    total_count = 0
    joint_pass_count = 0
    collision_count = 0
    minimum_margin_sum = 0.0
    translation_absolute_error_sum = 0.0
    translation_squared_error_sum = 0.0
    rotation_absolute_error_sum = 0.0
    rotation_squared_error_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            terms, command = _loss_for_batch(model, physics_layer, batch, target_scaler, config)
            physical = physics_layer(batch[2], command, batch[3])
            count = batch[0].shape[0]
            total_count += count
            for name in weighted_terms:
                weighted_terms[name] += float(getattr(terms, name)) * count
            gaps = physical.labelled_gaps_um
            pass_mask = (
                (gaps >= 2.0).all(dim=1)
                & (gaps <= 4.0).all(dim=1)
                & (physical.minimum_clearance_um > 0.0)
                & (physical.residual_translation_um.abs() <= 0.1).all(dim=1)
                & (physical.residual_rotation_arcsec.abs() <= 0.2).all(dim=1)
            )
            joint_pass_count += int(pass_mask.sum())
            collision_count += int((physical.minimum_clearance_um <= 0.0).sum())
            margins = torch.stack((gaps - 2.0, 4.0 - gaps), dim=-1).amin(dim=(-1, -2))
            minimum_margin_sum += float(margins.sum())
            teacher_command = _inverse_target_scale(batch[1], target_scaler)
            command_error = command - teacher_command
            translation_absolute_error_sum += float(command_error[:, :3].abs().sum())
            translation_squared_error_sum += float(command_error[:, :3].square().sum())
            rotation_absolute_error_sum += float(command_error[:, 3:].abs().sum())
            rotation_squared_error_sum += float(command_error[:, 3:].square().sum())
    return {
        "loss_total": weighted_terms["total"] / total_count,
        "loss_data": weighted_terms["data"] / total_count,
        "loss_physics": weighted_terms["physics"] / total_count,
        "loss_optimality": weighted_terms["optimality"] / total_count,
        "joint_pass_rate": joint_pass_count / total_count,
        "collision_rate": collision_count / total_count,
        "mean_gap_margin_um": minimum_margin_sum / total_count,
        "command_translation_mae_um": translation_absolute_error_sum / (3 * total_count),
        "command_translation_rmse_um": (translation_squared_error_sum / (3 * total_count)) ** 0.5,
        "command_rotation_mae_arcsec": rotation_absolute_error_sum / (3 * total_count),
        "command_rotation_rmse_arcsec": (rotation_squared_error_sum / (3 * total_count)) ** 0.5,
    }


def train_source_model(
    *,
    data_directory: Path,
    output_directory: Path,
    config: TrainingConfig = TrainingConfig(),
) -> dict[str, object]:
    """以 M0 预训练/验证集训练单逆网络，并保存可复现实验产物。"""
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    source_rows = _read_csv_rows(data_directory / "source_pretrain.csv")
    validation_rows = _read_csv_rows(data_directory / "source_validation.csv")
    feature_columns, target_columns = validated_training_columns(
        config.feature_set_name,
        source_rows[0].keys(),
        target_columns=PRIMARY_TARGET_COLUMNS,
    )
    # 此处是唯一拟合均值与标准差的地方；验证集绝不参与该步骤。
    input_scaler = fit_standard_scaler(source_rows, feature_columns)
    target_scaler = fit_standard_scaler(source_rows, target_columns)
    source = prepare_split(
        source_rows,
        feature_columns=feature_columns,
        target_columns=target_columns,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
    )
    validation = prepare_split(
        validation_rows,
        feature_columns=feature_columns,
        target_columns=target_columns,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
    )
    model = InverseCompensationMLP(len(feature_columns), hidden_size=config.hidden_size).to(dtype=torch.float64)
    physics_layer = SquareFitPhysicsLayer().to(dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loader = _as_loader(source, batch_size=config.batch_size, shuffle=True, seed=config.seed)
    history: list[dict[str, float | int]] = []
    best_validation = float("inf")
    best_state: dict[str, Tensor] | None = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        observed = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            terms, _ = _loss_for_batch(model, physics_layer, batch, target_scaler, config)
            terms.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            count = batch[0].shape[0]
            running_loss += float(terms.total.detach()) * count
            observed += count
        validation_metrics = evaluate_model(model, physics_layer, validation, target_scaler, config)
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss_total": running_loss / observed,
            **validation_metrics,
        }
        history.append(record)
        if validation_metrics["loss_total"] < best_validation:
            best_validation = validation_metrics["loss_total"]
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("训练未产生有效模型状态。")
    model.load_state_dict(best_state)
    final_validation = evaluate_model(model, physics_layer, validation, target_scaler, config)
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "input_scaler.json").write_text(
        json.dumps(input_scaler.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_directory / "target_scaler.json").write_text(
        json.dumps(target_scaler.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result: dict[str, object] = {
        "training_config": asdict(config),
        "feature_columns": list(feature_columns),
        "target_columns": list(target_columns),
        "best_validation_loss": best_validation,
        "final_validation": final_validation,
        "history": history,
    }
    (output_directory / "training_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "training_config": asdict(config),
            "feature_columns": list(feature_columns),
            "target_columns": list(target_columns),
        },
        output_directory / "source_inverse_mlp.pt",
    )
    return result


def _load_scaler(path: Path) -> StandardScaler:
    saved = json.loads(path.read_text(encoding="utf-8"))
    columns = tuple(saved)
    return StandardScaler(
        columns=columns,
        means=tuple(float(saved[column]["mean"]) for column in columns),
        standard_deviations=tuple(float(saved[column]["standard_deviation"]) for column in columns),
    )


def _physics_layer_for_domain(domain: SimulationDomain) -> SquareFitPhysicsLayer:
    """构造用于离线验收的真实虚拟执行器层。

    训练期间只能使用 M0 名义执行器，模拟“新机器失配未知”；验收阶段则必须
    使用数据生成域的执行器均值模型，才能公平地计算预测命令在 M1/Mstress
    上的真实几何结果。这一函数绝不向网络暴露执行器参数。
    """
    actuator = domain.actuator
    return SquareFitPhysicsLayer(
        actuator=MeanActuatorLayer(
            translation_coupling=actuator.translation_coupling,
            rotation_coupling=actuator.rotation_coupling,
            translation_bias_um=actuator.translation_bias_um,
            rotation_bias_arcsec=actuator.rotation_bias_arcsec,
        )
    ).to(dtype=torch.float64)


def _load_source_bundle(model_directory: Path) -> tuple[InverseCompensationMLP, TrainingConfig, tuple[str, ...], tuple[str, ...], StandardScaler, StandardScaler]:
    checkpoint = torch.load(model_directory / "source_inverse_mlp.pt", map_location="cpu", weights_only=False)
    config = TrainingConfig(**checkpoint["training_config"])
    feature_columns = tuple(checkpoint["feature_columns"])
    target_columns = tuple(checkpoint["target_columns"])
    input_scaler = _load_scaler(model_directory / "input_scaler.json")
    target_scaler = _load_scaler(model_directory / "target_scaler.json")
    if feature_columns != input_scaler.columns or target_columns != target_scaler.columns:
        raise ValueError("检查点与归一化器的特征列不一致，拒绝加载。")
    model = InverseCompensationMLP(len(feature_columns), hidden_size=config.hidden_size).to(dtype=torch.float64)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config, feature_columns, target_columns, input_scaler, target_scaler


def _prepared_csv(
    path: Path,
    *,
    feature_set_name: str,
    feature_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
    input_scaler: StandardScaler,
    target_scaler: StandardScaler,
    limit: int | None = None,
) -> PreparedSplit:
    rows = _read_csv_rows(path)
    if limit is not None:
        if limit > len(rows):
            raise ValueError(f"请求 {limit} 条，但 {path.name} 只有 {len(rows)} 条。")
        rows = rows[:limit]
    validated_features, validated_targets = validated_training_columns(
        feature_set_name, rows[0].keys(), target_columns=PRIMARY_TARGET_COLUMNS
    )
    if validated_features != feature_columns or validated_targets != target_columns:
        raise ValueError(f"{path.name} 的白名单列与源域检查点不一致，拒绝继续。")
    return prepare_split(
        rows,
        feature_columns=feature_columns,
        target_columns=target_columns,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
    )


def _metric_with_count(metrics: dict[str, float], count: int) -> dict[str, float | int]:
    return {"sample_count": count, **metrics}


def run_target_adaptation_experiment(
    *,
    data_directory: Path,
    source_model_directory: Path,
    output_directory: Path,
    calibration_sizes: tuple[int, ...] = (10, 30, 50),
    adaptation_epochs: int = 80,
    adaptation_learning_rate: float = 5e-4,
    calibration_seed: int | None = None,
) -> dict[str, object]:
    """固定源域检查点，执行 M1 的 zero-shot 与 10/30/50 条小样本适配。

    归一化器始终来自 M0 训练集，M1 运行/蒙特卡洛/压力集全程不参与更新或
    超参数选择。默认每个标定规模从同一个嵌套、分层的 50 条标定池取前 n 条；
    多种子复验可指定 ``calibration_seed``，以分层重抽样检验标定件组成的波动。

    小样本阶段只更新最后输出层的 6 个偏置参数；其余特征提取层完全冻结。
    这是针对 10/30/50 条标定数据的容量约束，避免全网络微调产生灾难性遗忘。
    训练损失仍保留弱几何约束，但其权重从源域的 2.0 降至 0.1，因为 M1 的
    执行器失配正是适配对象，不能让 M0 名义执行器在适配阶段喧宾夺主。
    """
    if not calibration_sizes or any(size not in (10, 30, 50) for size in calibration_sizes):
        raise ValueError("正式协议仅允许嵌套的 10、30、50 条 M1 标定实验。")
    if adaptation_epochs <= 0 or adaptation_learning_rate <= 0:
        raise ValueError("适配轮数和学习率必须为正数。")

    source_model, source_config, feature_columns, target_columns, input_scaler, target_scaler = _load_source_bundle(source_model_directory)
    target_operational = _prepared_csv(
        data_directory / "target_operational_test.csv",
        feature_set_name=source_config.feature_set_name,
        feature_columns=feature_columns, target_columns=target_columns,
        input_scaler=input_scaler, target_scaler=target_scaler,
    )
    monte_carlo = _prepared_csv(
        data_directory / "monte_carlo_operational_10000.csv",
        feature_set_name=source_config.feature_set_name,
        feature_columns=feature_columns, target_columns=target_columns,
        input_scaler=input_scaler, target_scaler=target_scaler,
    )
    stress = _prepared_csv(
        data_directory / "target_stress_test.csv",
        feature_set_name=source_config.feature_set_name,
        feature_columns=feature_columns, target_columns=target_columns,
        input_scaler=input_scaler, target_scaler=target_scaler,
    )
    evaluation_config = replace(
        source_config,
        epochs=adaptation_epochs,
        learning_rate=adaptation_learning_rate,
        lambda_physics=0.1,
        lambda_optimality=0.0,
    )
    target_physics = _physics_layer_for_domain(make_target_domain())
    stress_physics = _physics_layer_for_domain(make_stress_domain())

    def evaluate_all(model: InverseCompensationMLP) -> dict[str, dict[str, float | int]]:
        return {
            "target_operational_1000": _metric_with_count(
                evaluate_model(model, target_physics, target_operational, target_scaler, evaluation_config), 1000
            ),
            "monte_carlo_operational_10000": _metric_with_count(
                evaluate_model(model, target_physics, monte_carlo, target_scaler, evaluation_config), 10000
            ),
            "target_stress_1000": _metric_with_count(
                evaluate_model(model, stress_physics, stress, target_scaler, evaluation_config), 1000
            ),
        }

    result: dict[str, object] = {
        "protocol": "M0_pretrain_then_M1_nested_10_30_50_adaptation_v1",
        "source_checkpoint": str(source_model_directory / "source_inverse_mlp.pt"),
        "feature_columns": list(feature_columns),
        "adaptation_epochs_fixed_before_test": adaptation_epochs,
        "adaptation_learning_rate_fixed_before_test": adaptation_learning_rate,
        "adaptation_strategy": "freeze_feature_extractor_update_final_output_bias_only",
        "adaptation_trainable_parameter_count": 6,
        "adaptation_loss_weights": {"data": 1.0, "nominal_physics": 0.1, "optimality": 0.0},
        "calibration_selection_seed": calibration_seed,
        "zero_shot": evaluate_all(source_model),
        "adapted": {},
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    nominal_physics = SquareFitPhysicsLayer().to(dtype=torch.float64)
    calibration_path = data_directory / "target_calibration_50.csv"

    for size in calibration_sizes:
        run_seed = source_config.seed + size if calibration_seed is None else calibration_seed + size
        torch.manual_seed(run_seed)
        random.seed(run_seed)
        model = InverseCompensationMLP(len(feature_columns), hidden_size=source_config.hidden_size).to(dtype=torch.float64)
        model.load_state_dict(source_model.state_dict())
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        # `network.4.bias` 是 64 → 6 输出层的 6 个偏置，等价于对原补偿策略
        # 施加一个小的六维标定修正；它不是新增网络。
        output_bias = model.network[4].bias
        output_bias.requires_grad_(True)
        calibration_rows, calibration_sample_ids = _nested_stratified_calibration_rows(
            calibration_path, size=size, calibration_seed=calibration_seed,
        )
        calibration = prepare_split(
            calibration_rows,
            feature_columns=feature_columns,
            target_columns=target_columns,
            input_scaler=input_scaler,
            target_scaler=target_scaler,
        )
        loader = _as_loader(calibration, batch_size=min(16, size), shuffle=True, seed=run_seed)
        optimizer = torch.optim.Adam((output_bias,), lr=adaptation_learning_rate)
        epoch_losses: list[float] = []
        for _ in range(adaptation_epochs):
            model.train()
            weighted_loss = 0.0
            observed = 0
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                # 物理正则只用 M0 名义执行器，避免把 M1 真值参数泄漏给适配网络。
                terms, _ = _loss_for_batch(model, nominal_physics, batch, target_scaler, evaluation_config)
                terms.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                weighted_loss += float(terms.total.detach()) * batch[0].shape[0]
                observed += batch[0].shape[0]
            epoch_losses.append(weighted_loss / observed)
        model.eval()
        adapted = result["adapted"]
        assert isinstance(adapted, dict)
        adapted[str(size)] = {
            "calibration_count": size,
            "calibration_risk_layers": {"normal": int(size * 0.6), "coverage": int(size * 0.3), "difficult": int(size * 0.1)},
            "calibration_sample_ids": calibration_sample_ids,
            "calibration_optimization_seed": run_seed,
            "final_calibration_loss": epoch_losses[-1],
            "evaluations": evaluate_all(model),
        }
        torch.save(
            {"model_state_dict": model.state_dict(), "calibration_count": size, "source_training_config": asdict(source_config)},
            output_directory / f"target_adapt_{size}.pt",
        )

    (output_directory / "target_adaptation_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def run_target_adaptation_replicates(
    *,
    data_directory: Path,
    source_model_directory: Path,
    output_directory: Path,
    replicate_seeds: tuple[int, ...] = (20261001, 20261002, 20261003, 20261004, 20261005),
    adaptation_epochs: int = 80,
    adaptation_learning_rate: float = 5e-4,
) -> dict[str, object]:
    """重复五次嵌套分层适配，报告 M1 性能的均值与样本标准差。

    这是开发阶段稳健性复验，不会改写单次协议结果或覆盖其检查点。
    """
    if len(replicate_seeds) < 2 or len(set(replicate_seeds)) != len(replicate_seeds):
        raise ValueError("多种子复验至少需要两个互不重复的随机种子。")
    output_directory.mkdir(parents=True, exist_ok=True)
    replicates: dict[str, dict[str, object]] = {}
    for seed in replicate_seeds:
        replicates[str(seed)] = run_target_adaptation_experiment(
            data_directory=data_directory,
            source_model_directory=source_model_directory,
            output_directory=output_directory / f"seed_{seed}",
            adaptation_epochs=adaptation_epochs,
            adaptation_learning_rate=adaptation_learning_rate,
            calibration_seed=seed,
        )

    aggregate: dict[str, object] = {}
    for size in (10, 30, 50):
        per_split: dict[str, object] = {}
        for split in ("target_operational_1000", "monte_carlo_operational_10000"):
            values = [
                float(replicates[str(seed)]["adapted"][str(size)]["evaluations"][split]["joint_pass_rate"])
                for seed in replicate_seeds
            ]
            per_split[split] = {
                "joint_pass_rate_mean": statistics.mean(values),
                "joint_pass_rate_std": statistics.stdev(values),
                "joint_pass_rates": values,
            }
        aggregate[str(size)] = per_split
    result: dict[str, object] = {
        "protocol": "M0_pretrain_then_M1_stratified_replicated_10_30_50_adaptation_v1",
        "replicate_seeds": list(replicate_seeds),
        "replicate_count": len(replicate_seeds),
        "adaptation_epochs_fixed_before_test": adaptation_epochs,
        "adaptation_learning_rate_fixed_before_test": adaptation_learning_rate,
        "strategy": "freeze_feature_extractor_update_final_output_bias_only",
        "aggregate": aggregate,
        "replicates": replicates,
    }
    (output_directory / "target_adaptation_replicates_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def evaluate_frozen_target_models_on_holdout(
    *,
    holdout_csv: Path,
    source_model_directory: Path,
    adaptation_directory: Path,
) -> dict[str, object]:
    """在从未参与开发的新 M1 封存集上一次性评价已冻结的候选模型。

    本函数不训练、也不选择超参数。调用前必须已经固定特征组、适配策略、轮数
    和模型检查点；它只为技术报告提供最终、独立的 M1 验收数字。
    """
    source_model, source_config, feature_columns, target_columns, input_scaler, target_scaler = _load_source_bundle(source_model_directory)
    holdout = _prepared_csv(
        holdout_csv,
        feature_set_name=source_config.feature_set_name,
        feature_columns=feature_columns,
        target_columns=target_columns,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
    )
    evaluation_config = replace(source_config, lambda_physics=0.1, lambda_optimality=0.0)
    target_physics = _physics_layer_for_domain(make_target_domain())
    result: dict[str, object] = {
        "purpose": "final_frozen_M1_holdout_evaluation",
        "holdout_csv": str(holdout_csv),
        "holdout_sample_count": int(holdout.features.shape[0]),
        "feature_set_name": source_config.feature_set_name,
        "methods": {
            "zero_shot": _metric_with_count(
                evaluate_model(source_model, target_physics, holdout, target_scaler, evaluation_config),
                int(holdout.features.shape[0]),
            )
        },
    }
    methods = result["methods"]
    assert isinstance(methods, dict)
    for size in (10, 30, 50):
        checkpoint_path = adaptation_directory / f"target_adapt_{size}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"缺少冻结适配检查点：{checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = InverseCompensationMLP(len(feature_columns), hidden_size=source_config.hidden_size).to(dtype=torch.float64)
        model.load_state_dict(checkpoint["model_state_dict"])
        methods[f"adapt_{size}"] = _metric_with_count(
            evaluate_model(model, target_physics, holdout, target_scaler, evaluation_config),
            int(holdout.features.shape[0]),
        )
    output_path = adaptation_directory / "final_m1_holdout_evaluation.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def evaluate_saved_source_model(*, data_directory: Path, model_directory: Path) -> dict[str, float]:
    """只读加载已保存 M0 模型，在源验证集上补算完整评价指标。"""
    checkpoint = torch.load(model_directory / "source_inverse_mlp.pt", map_location="cpu", weights_only=False)
    config = TrainingConfig(**checkpoint["training_config"])
    feature_columns = tuple(checkpoint["feature_columns"])
    target_columns = tuple(checkpoint["target_columns"])
    input_scaler = _load_scaler(model_directory / "input_scaler.json")
    target_scaler = _load_scaler(model_directory / "target_scaler.json")
    if feature_columns != input_scaler.columns or target_columns != target_scaler.columns:
        raise ValueError("检查点与归一化器的特征列不一致，拒绝评价。")
    rows = _read_csv_rows(data_directory / "source_validation.csv")
    validated_features, validated_targets = validated_training_columns(config.feature_set_name, rows[0].keys())
    if validated_features != feature_columns or validated_targets != target_columns:
        raise ValueError("检查点白名单与当前数据协议不一致，拒绝评价。")
    validation = prepare_split(
        rows,
        feature_columns=feature_columns,
        target_columns=target_columns,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
    )
    model = InverseCompensationMLP(len(feature_columns), hidden_size=config.hidden_size).to(dtype=torch.float64)
    model.load_state_dict(checkpoint["model_state_dict"])
    return evaluate_model(model, SquareFitPhysicsLayer().to(dtype=torch.float64), validation, target_scaler, config)
