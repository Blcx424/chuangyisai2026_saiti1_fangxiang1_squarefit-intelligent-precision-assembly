from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from squarefit.closed_loop import (
    ClosedLoopConfig,
    _average_pose,
    _confirmed_feedback_needed,
    _dls_feedback_command,
    _violation_score,
)
from squarefit.geometry import evaluate_lateral_fit
from squarefit.torch_geometry import SquareFitPhysicsLayer
from squarefit.types import Dimensions, Pose6D
from squarefit.units import arcsec_to_rad


class ClosedLoopGeometryTests(unittest.TestCase):
    def test_confirmation_requires_both_measurements_to_cross_feedback_trigger(self) -> None:
        """单次偶然噪声越线不能触发二次物理动作，双复测一致才可以。"""
        layer = SquareFitPhysicsLayer().to(dtype=torch.float64)
        dimensions = torch.tensor([[10006.0, 10000.0]], dtype=torch.float64)
        config = ClosedLoopConfig()
        large = layer.evaluate_pose(torch.tensor([[0.14, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64), dimensions)
        small = layer.evaluate_pose(torch.tensor([[0.04, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64), dimensions)
        fused = layer.evaluate_pose(torch.tensor([[0.09, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64), dimensions)
        self.assertFalse(_confirmed_feedback_needed(fused, [large, small], config))
        self.assertTrue(_confirmed_feedback_needed(large, [large, large], config))

    def test_pose_average_reduces_opposite_random_measurement_deviation(self) -> None:
        """复测融合只平均随机项，不改变六维量纲与单位。"""
        fused = _average_pose([
            Pose6D(dx_um=0.14, dy_um=-0.10, dz_um=0.02, theta_x_rad=arcsec_to_rad(0.30)),
            Pose6D(dx_um=0.06, dy_um=-0.02, dz_um=0.04, theta_x_rad=arcsec_to_rad(0.10)),
        ])
        self.assertAlmostEqual(fused.dx_um, 0.10, places=12)
        self.assertAlmostEqual(fused.dy_um, -0.06, places=12)
        self.assertAlmostEqual(fused.dz_um, 0.03, places=12)
        self.assertAlmostEqual(fused.theta_x_rad, arcsec_to_rad(0.20), places=12)

    def test_direct_pose_evaluation_does_not_apply_zero_command_bias(self) -> None:
        """复测状态的纯几何评价不能错误加上执行器零偏。"""
        pose = Pose6D(dx_um=0.5, dy_um=-0.3, dz_um=0.05, theta_x_rad=arcsec_to_rad(0.4))
        dimensions = Dimensions(hole_side_um=10006.0, shaft_side_um=10000.0)
        expected = evaluate_lateral_fit(dimensions, pose)
        layer = SquareFitPhysicsLayer().to(dtype=torch.float64)
        state = layer.evaluate_pose(
            torch.tensor([[0.5, -0.3, 0.05, 0.4, 0.0, 0.0]], dtype=torch.float64),
            torch.tensor([[10006.0, 10000.0]], dtype=torch.float64),
        )
        for actual, reference in zip(state.labelled_gaps_um[0].tolist(), expected.labelled_gaps_um, strict=True):
            self.assertAlmostEqual(actual, reference, places=8)

    def test_damped_geometry_step_reduces_predicted_constraint_violation(self) -> None:
        """局部 DLS 至少应降低一个明显越过六维残余阈值的预测违规量。"""
        layer = SquareFitPhysicsLayer().to(dtype=torch.float64)
        pose = torch.tensor([[0.55, -0.40, 0.25, 0.65, -0.50, 0.35]], dtype=torch.float64)
        dimensions = torch.tensor([[10006.0, 10000.0]], dtype=torch.float64)
        config = ClosedLoopConfig()
        before = layer.evaluate_pose(pose, dimensions)
        command = _dls_feedback_command(pose, dimensions, layer, config)
        after = layer(pose, command, dimensions)
        self.assertLess(_violation_score(after, config.safety_guard_um), _violation_score(before, config.safety_guard_um))


if __name__ == "__main__":
    unittest.main()
