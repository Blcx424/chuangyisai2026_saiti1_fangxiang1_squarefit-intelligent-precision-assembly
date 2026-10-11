from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from squarefit.compensation import analytic_compensation, apply_world_frame_adjustment
from squarefit.geometry import evaluate_lateral_fit, residual_pose_within_requirement
from squarefit.transforms import rotation_matrix_from_pose
from squarefit.types import Dimensions, Pose6D
from squarefit.units import arcsec_to_rad


class AnalyticCompensationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dimensions = Dimensions(hole_side_um=10006.0, shaft_side_um=10000.0)

    def assert_same_transform(self, actual: Pose6D, expected: Pose6D) -> None:
        for actual_value, expected_value in zip(
            (actual.dx_um, actual.dy_um, actual.dz_um),
            (expected.dx_um, expected.dy_um, expected.dz_um),
        ):
            self.assertAlmostEqual(actual_value, expected_value, places=9)
        actual_rotation = rotation_matrix_from_pose(actual)
        expected_rotation = rotation_matrix_from_pose(expected)
        for actual_row, expected_row in zip(actual_rotation, expected_rotation):
            for actual_value, expected_value in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual_value, expected_value, places=12)

    def test_centered_pose_needs_zero_adjustment(self) -> None:
        result = analytic_compensation(Pose6D())
        self.assert_same_transform(result.adjustment, Pose6D())
        self.assert_same_transform(result.predicted_pose_after, Pose6D())

    def test_pure_translation_is_corrected_in_the_opposite_direction(self) -> None:
        measured = Pose6D(dx_um=0.8, dy_um=-0.3, dz_um=0.05)
        result = analytic_compensation(measured)
        self.assertAlmostEqual(result.adjustment.dx_um, -0.8)
        self.assertAlmostEqual(result.adjustment.dy_um, 0.3)
        self.assertAlmostEqual(result.adjustment.dz_um, -0.05)
        self.assert_same_transform(result.predicted_pose_after, Pose6D())
        fit_after = evaluate_lateral_fit(self.dimensions, result.predicted_pose_after)
        self.assertEqual(fit_after.labelled_gaps_um, (3.0, 3.0, 3.0, 3.0))

    def test_rotation_and_translation_are_corrected_together(self) -> None:
        measured = Pose6D(
            dx_um=0.8,
            dy_um=-0.4,
            dz_um=0.2,
            theta_x_rad=arcsec_to_rad(1.2),
            theta_y_rad=arcsec_to_rad(-0.9),
            theta_z_rad=arcsec_to_rad(0.7),
        )
        result = analytic_compensation(measured)
        self.assert_same_transform(result.predicted_pose_after, Pose6D())
        self.assertTrue(residual_pose_within_requirement(self.dimensions, result.predicted_pose_after))

    def test_nonzero_target_pose_is_supported(self) -> None:
        measured = Pose6D(
            dx_um=0.7,
            dy_um=-0.6,
            dz_um=0.5,
            theta_x_rad=arcsec_to_rad(1.0),
            theta_y_rad=arcsec_to_rad(-0.8),
            theta_z_rad=arcsec_to_rad(0.3),
        )
        target = Pose6D(
            dx_um=0.04,
            dy_um=-0.03,
            dz_um=0.02,
            theta_x_rad=arcsec_to_rad(0.05),
            theta_y_rad=arcsec_to_rad(-0.03),
            theta_z_rad=arcsec_to_rad(0.04),
        )
        result = analytic_compensation(measured, target)
        self.assert_same_transform(result.predicted_pose_after, target)

    def test_apply_adjustment_matches_result(self) -> None:
        measured = Pose6D(dx_um=-0.5, theta_y_rad=arcsec_to_rad(1.5))
        result = analytic_compensation(measured)
        self.assert_same_transform(
            apply_world_frame_adjustment(measured, result.adjustment),
            result.predicted_pose_after,
        )


if __name__ == "__main__":
    unittest.main()
