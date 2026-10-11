"""最终提交包的静态完整性校验。

本模块不训练模型、不改写数据或产物，只检查随包 CSV、适配汇总和主特征约定
是否与 metadata 和冻结协议一致。随机种子是确定性整数标识，不表达运行日期。
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from .schema import FEATURE_SETS


_C15_FEATURE_SET = "C_解析先验BP_15维"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 根对象必须是字典：{path}")
    return value


def _verify_dataset_bundle(data_directory: Path) -> list[str]:
    metadata_path = data_directory / "metadata.json"
    metadata = _load_json(metadata_path)
    files = metadata.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"metadata 缺少 files 哈希清单：{metadata_path}")

    checked: list[str] = []
    for filename, file_metadata in files.items():
        if not isinstance(filename, str) or not isinstance(file_metadata, dict):
            raise ValueError(f"metadata 文件记录格式错误：{metadata_path}")
        expected = file_metadata.get("sha256")
        path = data_directory / filename
        if not isinstance(expected, str) or not path.is_file():
            raise ValueError(f"metadata 文件缺失或哈希缺失：{path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"CSV 哈希不一致：{path.name}")
        checked.append(filename)
    return checked


def _verify_c15_schema(data_directory: Path) -> None:
    columns = FEATURE_SETS[_C15_FEATURE_SET]
    if len(columns) != 15 or any(column.startswith("true_") for column in columns):
        raise ValueError("C15 白名单必须恰为 15 个且不包含 true_* 字段。")

    source_csv = data_directory / "source_pretrain.csv"
    with source_csv.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    missing = [column for column in columns if column not in header]
    if missing:
        raise ValueError(f"主数据缺少 C15 字段：{missing}")


def _verify_replicate_summary(summary_path: Path) -> None:
    summary = _load_json(summary_path)
    seeds = summary.get("replicate_seeds")
    if not isinstance(seeds, list) or len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError("多种子适配汇总中的随机种子必须至少两个且互不重复。")
    if summary.get("replicate_count") != len(seeds):
        raise ValueError("多种子适配汇总中的 replicate_count 与随机种子数不一致。")

    aggregate = summary.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError("多种子适配汇总缺少 aggregate。")
    for size in ("10", "30", "50"):
        size_summary = aggregate.get(size)
        if not isinstance(size_summary, dict):
            raise ValueError(f"多种子适配汇总缺少 {size} 条标定结果。")
        for split in ("target_operational_1000", "monte_carlo_operational_10000"):
            split_summary = size_summary.get(split)
            if not isinstance(split_summary, dict):
                raise ValueError(f"多种子适配汇总缺少 {size}/{split}。")
            values = split_summary.get("joint_pass_rates")
            if not isinstance(values, list) or len(values) != len(seeds):
                raise ValueError(f"{size}/{split} 的复验次数与随机种子数不一致。")
            values = [float(value) for value in values]
            if abs(float(split_summary["joint_pass_rate_mean"]) - statistics.mean(values)) > 1e-12:
                raise ValueError(f"{size}/{split} 的均值与明细不一致。")
            if abs(float(split_summary["joint_pass_rate_std"]) - statistics.stdev(values)) > 1e-12:
                raise ValueError(f"{size}/{split} 的样本标准差与明细不一致。")


def verify_release_package(project_root: Path) -> dict[str, object]:
    """校验最终包的主数据、封存集、C15 约定和五种子适配汇总。"""
    project_root = project_root.resolve()
    main_data = project_root / "data" / "generated_task_v5_full_feasible"
    holdout_data = project_root / "data" / "final_holdout_task_v5_final"
    adaptation_directory = project_root / "artifacts" / "task_v5" / "target_adaptation_c15"
    source_directory = project_root / "artifacts" / "task_v5" / "source_pretrain_c15"
    closed_loop_directory = project_root / "artifacts" / "task_v5" / "closed_loop_c15"
    monte_carlo_directory = project_root / "artifacts" / "task_v5" / "closed_loop_monte_carlo_c15"
    replicate_summary = (
        project_root / "artifacts" / "task_v5" / "target_adaptation_replicates_c15"
        / "target_adaptation_replicates_summary.json"
    )
    required_artifacts = (
        source_directory / "source_inverse_mlp.pt",
        source_directory / "input_scaler.json",
        source_directory / "target_scaler.json",
        source_directory / "training_summary.json",
        adaptation_directory / "target_adapt_10.pt",
        adaptation_directory / "target_adapt_30.pt",
        adaptation_directory / "target_adapt_50.pt",
        adaptation_directory / "target_adaptation_summary.json",
        adaptation_directory / "final_m1_holdout_evaluation.json",
        closed_loop_directory / "closed_loop_summary.json",
        monte_carlo_directory / "closed_loop_monte_carlo_summary.json",
        replicate_summary,
    )
    missing = [str(path.relative_to(project_root)) for path in required_artifacts if not path.is_file()]
    if missing:
        raise ValueError(f"最终包缺少冻结产物：{missing}")

    main_files = _verify_dataset_bundle(main_data)
    holdout_files = _verify_dataset_bundle(holdout_data)
    _verify_c15_schema(main_data)
    _verify_replicate_summary(replicate_summary)
    return {
        "main_data_hashes_checked": len(main_files),
        "holdout_data_hashes_checked": len(holdout_files),
        "feature_set": _C15_FEATURE_SET,
        "replicate_summary": str(replicate_summary.relative_to(project_root)),
    }
