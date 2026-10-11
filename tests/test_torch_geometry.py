from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from squarefit.simulation import make_source_domain, make_target_domain, simulate_many
from squarefit.torch_geometry import MeanActuatorLayer, SquareFitPhysicsLayer


class TorchGeometryTests(unittest.TestCase):
    @staticmethod
    def _tensorize_sample(sample):
        true_pose = torch.tensor(
            [[
                sample.true_pose.dx_um, sample.true_pose.dy_um, sample.true_pose.dz_um,
                sample.true_pose.theta_x_rad * 648000.0 / torch.pi,
                sample.true_pose.theta_y_rad * 648000.0 / torch.pi,
                sample.true_pose.theta_z_rad * 648000.0 / torch.pi,
            ]], dtype=torch.float64,
        )
        command = torch.tensor(
            [[
                sample.optimal_command.dx_um, sample.optimal_command.dy_um, sample.optimal_command.dz_um,
                sample.optimal_command.theta_x_rad * 648000.0 / torch.pi,
                sample.optimal_command.theta_y_rad * 648000.0 / torch.pi,
                sample.optimal_command.theta_z_rad * 648000.0 / torch.pi,
            ]], dtype=torch.float64,
        )
        dimensions = torch.tensor(
            [[sample.effective_dimensions.hole_side_um, sample.effective_dimensions.shaft_side_um]], dtype=torch.float64,
        )
        return true_pose, command, dimensions

    def test_physics_layer_matches_verified_python_geometry_for_teacher_command(self) -> None:
        sample = simulate_many(1, make_source_domain(), seed=501)[0]
        layer = SquareFitPhysicsLayer().to(dtype=torch.float64)
        true_pose = torch.tensor(
            [[
                sample.true_pose.dx_um,
                sample.true_pose.dy_um,
                sample.true_pose.dz_um,
                sample.true_pose.theta_x_rad * 648000.0 / torch.pi,
                sample.true_pose.theta_y_rad * 648000.0 / torch.pi,
                sample.true_pose.theta_z_rad * 648000.0 / torch.pi,
            ]],
            dtype=torch.float64,
        )
        command = torch.tensor(
            [[
                sample.optimal_command.dx_um,
                sample.optimal_command.dy_um,
                sample.optimal_command.dz_um,
                sample.optimal_command.theta_x_rad * 648000.0 / torch.pi,
                sample.optimal_command.theta_y_rad * 648000.0 / torch.pi,
                sample.optimal_command.theta_z_rad * 648000.0 / torch.pi,
            ]],
            dtype=torch.float64,
        )
        dimensions = torch.tensor(
            [[sample.effective_dimensions.hole_side_um, sample.effective_dimensions.shaft_side_um]],
            dtype=torch.float64,
        )
        state = layer(true_pose, command, dimensions)
        expected_gaps = sample.optimal_mean_fit.labelled_gaps_um
        for actual, expected in zip(state.labelled_gaps_um[0].tolist(), expected_gaps, strict=True):
            self.assertAlmostEqual(actual, expected, places=8)
        self.assertAlmostEqual(
            float(state.minimum_clearance_um[0]),
            sample.optimal_mean_fit.minimum_clearance_um,
            places=8,
        )

    def test_target_machine_actuator_layer_matches_target_virtual_plant(self) -> None:
        """M1 验收不得误用 M0 执行器；两种实现须逐项一致。"""
        domain = make_target_domain()
        sample = simulate_many(1, domain, seed=502)[0]
        actuator = domain.actuator
        layer = SquareFitPhysicsLayer(
            actuator=MeanActuatorLayer(
                translation_coupling=actuator.translation_coupling,
                rotation_coupling=actuator.rotation_coupling,
                translation_bias_um=actuator.translation_bias_um,
                rotation_bias_arcsec=actuator.rotation_bias_arcsec,
            )
        ).to(dtype=torch.float64)
        state = layer(*self._tensorize_sample(sample))
        for actual, expected in zip(state.labelled_gaps_um[0].tolist(), sample.optimal_mean_fit.labelled_gaps_um, strict=True):
            self.assertAlmostEqual(actual, expected, places=8)

    def test_gap_loss_has_gradient_with_respect_to_command(self) -> None:
        layer = SquareFitPhysicsLayer().to(dtype=torch.float64)
        true_pose = torch.tensor([[0.7, -0.4, 0.1, 0.8, -0.5, 0.3]], dtype=torch.float64)
        dimensions = torch.tensor([[10006.0, 10000.0]], dtype=torch.float64)
        command = torch.zeros((1, 6), dtype=torch.float64, requires_grad=True)
        state = layer(true_pose, command, dimensions)
        loss = (state.labelled_gaps_um - 3.0).square().mean() + torch.relu(-state.minimum_clearance_um).square().mean()
        loss.backward()
        self.assertIsNotNone(command.grad)
        assert command.grad is not None
        self.assertTrue(torch.isfinite(command.grad).all())
        self.assertGreater(float(command.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
