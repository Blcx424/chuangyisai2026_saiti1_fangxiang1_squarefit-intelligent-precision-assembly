from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from squarefit.geometry import (
    evaluate_lateral_fit,
    labelled_gaps_um,
    residual_pose_within_requirement,
)
from squarefit.transforms import inverse_transform_point, transform_point
from squarefit.types import Dimensions, Pose6D
from squarefit.units import arcsec_to_rad


class GeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dimensions = Dimensions(hole_side_um=10006.0, shaft_side_um=10000.0)

    def test_centered_pose_has_three_um_gaps(self) -> None:
        result = evaluate_lateral_fit(self.dimensions, Pose6D())
        self.assertEqual(result.labelled_gaps_um, (3.0, 3.0, 3.0, 3.0))
        self.assertAlmostEqual(result.minimum_clearance_um, 3.0)
        self.assertFalse(result.collision)
        self.assertTrue(result.labelled_gaps_qualified)

    def test_positive_x_translation_reduces_right_gap(self) -> None:
        result = evaluate_lateral_fit(self.dimensions, Pose6D(dx_um=0.5))
        self.assertEqual(result.labelled_gaps_um, (3.0, 2.5, 3.0, 2.5))
        self.assertAlmostEqual(result.minimum_clearance_um, 2.5)

    def test_x_rotation_changes_top_gap_with_depth(self) -> None:
        theta_x = arcsec_to_rad(1.0)
        result = evaluate_lateral_fit(self.dimensions, Pose6D(dy_um=-0.3, theta_x_rad=theta_x))
        self.assertAlmostEqual(result.g1_um, 3.3, places=6)
        self.assertAlmostEqual(
            result.g3_um,
            3.3 + self.dimensions.insertion_depth_um * math.sin(theta_x),
            places=6,
        )

    def test_y_rotation_changes_right_gap_with_depth(self) -> None:
        theta_y = arcsec_to_rad(1.0)
        result = evaluate_lateral_fit(self.dimensions, Pose6D(theta_y_rad=theta_y))
        self.assertAlmostEqual(result.g2_um, 3.0, places=6)
        self.assertAlmostEqual(
            result.g4_um,
            3.0 - self.dimensions.insertion_depth_um * math.sin(theta_y),
            places=6,
        )

    def test_yaw_reduces_parallel_wall_clearance_symmetrically(self) -> None:
        theta_z = arcsec_to_rad(1.0)
        plus = evaluate_lateral_fit(self.dimensions, Pose6D(theta_z_rad=theta_z))
        minus = evaluate_lateral_fit(self.dimensions, Pose6D(theta_z_rad=-theta_z))
        expected = (
            self.dimensions.hole_side_um / 2.0
            - self.dimensions.shaft_side_um / 2.0
            * (math.cos(theta_z) + math.sin(theta_z))
        )
        for gap_plus, gap_minus in zip(plus.labelled_gaps_um, minus.labelled_gaps_um):
            self.assertAlmostEqual(gap_plus, expected, places=8)
            self.assertAlmostEqual(gap_minus, expected, places=8)

    def test_collision_checks_all_four_walls(self) -> None:
        result = evaluate_lateral_fit(self.dimensions, Pose6D(dx_um=3.2))
        self.assertLess(result.g2_um, 0.0)
        self.assertTrue(result.collision)
        self.assertFalse(result.labelled_gaps_qualified)

    def test_gap_boundaries_are_inclusive(self) -> None:
        lower = Dimensions(hole_side_um=10004.0, shaft_side_um=10000.0)
        upper = Dimensions(hole_side_um=10008.0, shaft_side_um=10000.0)
        below = Dimensions(hole_side_um=10003.998, shaft_side_um=10000.0)
        self.assertTrue(evaluate_lateral_fit(lower, Pose6D()).labelled_gaps_qualified)
        self.assertTrue(evaluate_lateral_fit(upper, Pose6D()).labelled_gaps_qualified)
        self.assertFalse(evaluate_lateral_fit(below, Pose6D()).labelled_gaps_qualified)

    def test_dz_is_lateral_gap_independent_but_still_checked(self) -> None:
        pose = Pose6D(dz_um=0.11)
        self.assertEqual(labelled_gaps_um(self.dimensions, pose), (3.0, 3.0, 3.0, 3.0))
        self.assertFalse(residual_pose_within_requirement(self.dimensions, pose))

    def test_residual_requirement_boundaries(self) -> None:
        limit_pose = Pose6D(
            dx_um=0.1,
            dy_um=-0.1,
            dz_um=0.1,
            theta_x_rad=arcsec_to_rad(0.2),
            theta_y_rad=-arcsec_to_rad(0.2),
            theta_z_rad=arcsec_to_rad(0.2),
        )
        self.assertTrue(residual_pose_within_requirement(self.dimensions, limit_pose))
        self.assertFalse(residual_pose_within_requirement(self.dimensions, Pose6D(dx_um=0.1001)))

    def test_rigid_transform_has_an_inverse(self) -> None:
        pose = Pose6D(
            dx_um=0.4,
            dy_um=-0.7,
            dz_um=1.3,
            theta_x_rad=arcsec_to_rad(0.7),
            theta_y_rad=arcsec_to_rad(-1.2),
            theta_z_rad=arcsec_to_rad(0.5),
        )
        point = (120.0, -250.0, 8000.0)
        reconstructed = inverse_transform_point(pose, transform_point(pose, point))
        for actual, expected in zip(reconstructed, point):
            self.assertAlmostEqual(actual, expected, places=9)


if __name__ == "__main__":
    unittest.main()
