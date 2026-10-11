"""神经网络训练前的数据归一化工具。

同一条样本里孔边长约为 10000 um，位姿平移约为 1 um，转角约为 1 arcsec。
若直接送进网络，大数值特征会主导梯度。本模块强制规定：均值和标准差只能
在训练集上拟合；验证集、标定集和测试集只能调用 transform。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


NumericRow = Mapping[str, float | int | bool | str]


@dataclass(frozen=True)
class StandardScaler:
    """按列 z-score 归一化的、可保存统计量。"""

    columns: tuple[str, ...]
    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]

    def transform_row(self, row: NumericRow) -> tuple[float, ...]:
        """归一化一行数据；不会更新训练集统计量。"""
        values = _numeric_values(row, self.columns)
        return tuple(
            (value - mean) / standard_deviation
            for value, mean, standard_deviation in zip(
                values, self.means, self.standard_deviations, strict=True
            )
        )

    def inverse_transform_values(self, normalized_values: Sequence[float]) -> tuple[float, ...]:
        """把网络预测恢复为微米、角秒等工程单位。"""
        if len(normalized_values) != len(self.columns):
            raise ValueError("待反归一化的数值个数与列数不一致。")
        return tuple(
            value * standard_deviation + mean
            for value, mean, standard_deviation in zip(
                normalized_values, self.means, self.standard_deviations, strict=True
            )
        )

    def as_dict(self) -> dict[str, dict[str, float]]:
        """便于后续保存为 JSON，保证推理复用训练集统计量。"""
        return {
            column: {"mean": mean, "standard_deviation": standard_deviation}
            for column, mean, standard_deviation in zip(
                self.columns, self.means, self.standard_deviations, strict=True
            )
        }


def _numeric_values(row: NumericRow, columns: Sequence[str]) -> tuple[float, ...]:
    values: list[float] = []
    for column in columns:
        if column not in row:
            raise KeyError(f"数据行缺少所需列：{column}")
        value = row[column]
        if isinstance(value, bool):
            raise TypeError(f"列 {column} 必须是数值，当前得到：{value!r}")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"列 {column} 必须是数值，当前得到：{value!r}") from error
        if not math.isfinite(numeric_value):
            raise ValueError(f"列 {column} 含有非有限数值：{value!r}")
        values.append(numeric_value)
    return tuple(values)


def fit_standard_scaler(training_rows: Sequence[NumericRow], columns: Sequence[str]) -> StandardScaler:
    """仅依据训练集拟合样本均值和样本标准差。

    零方差列直接报错，而不是悄悄除以 1；它往往意味着特征没有变化、采样范围
    设错，或该列不应进入网络。
    """
    if not training_rows:
        raise ValueError("归一化器必须至少用一条训练样本拟合。")
    if not columns:
        raise ValueError("归一化器至少需要一个特征列。")
    if len(set(columns)) != len(columns):
        raise ValueError("归一化列名不能重复。")

    rows = [_numeric_values(row, columns) for row in training_rows]
    sample_count = len(rows)
    if sample_count < 2:
        raise ValueError("训练集至少需要两条样本，才能估计标准差。")
    means = tuple(sum(row[index] for row in rows) / sample_count for index in range(len(columns)))
    standard_deviations = tuple(
        math.sqrt(sum((row[index] - means[index]) ** 2 for row in rows) / (sample_count - 1))
        for index in range(len(columns))
    )
    for column, standard_deviation in zip(columns, standard_deviations, strict=True):
        if standard_deviation <= 1e-15:
            raise ValueError(f"列 {column} 在训练集上标准差为零，不能进行 z-score 归一化。")
    return StandardScaler(tuple(columns), means, standard_deviations)
