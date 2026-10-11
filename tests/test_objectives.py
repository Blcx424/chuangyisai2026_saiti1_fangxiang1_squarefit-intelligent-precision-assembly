from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from squarefit.objectives import evaluate_assembly_objective
from squarefit.types import Dimensions, Pose6D
from squarefit.units import arcsec_to_rad


class AssemblyObjectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dimensions = Dimensions(hole_side_um=10006.0, shaft_side_um=10000.0)

    def test_centered_pose_has_one_um_gap_margin(self) -> None:
        objective = evaluate_assembly_objective(self.dimensions, Pose6D())
        self.assertAlmostEqual(objective.minimum_gap_margin_um, 1.0)
        self.assertAlmostEqual(objective.mean_gap_center_deviation_um, 0.0)
        self.assertTrue(objective.fully_qualified)

    def test_margin_becomes_negative_when_a_gap_exceeds_limit(self) -> None:
        objective = evaluate_assembly_objective(self.dimensions, Pose6D(dx_um=1.2))
        self.assertAlmostEqual(objective.minimum_gap_margin_um, -0.2)
        self.assertFalse(objective.gaps_qualified)
        self.assertFalse(objective.fully_qualified)

    def test_lateral_gaps_cannot_hide_an_excessive_z_residual(self) -> None:
        objective = evaluate_assembly_objective(
            self.dimensions,
            Pose6D(dz_um=0.11, theta_z_rad=arcsec_to_rad(0.1)),
        )
        self.assertTrue(objective.gaps_qualified)
        self.assertFalse(objective.residual_pose_qualified)
        self.assertFalse(objective.fully_qualified)


if __name__ == "__main__":
    unittest.main()
