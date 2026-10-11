from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from squarefit.contact import (
    classify_lateral_contact_state,
    wall_clearance_profile_um,
)
from squarefit.types import Dimensions, Pose6D
from squarefit.units import arcsec_to_rad


class ContactClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dimensions = Dimensions(hole_side_um=10006.0, shaft_side_um=10000.0)

    def test_profile_contains_eight_side_clearances(self) -> None:
        profile = wall_clearance_profile_um(self.dimensions, Pose6D())
        self.assertEqual(len(profile), 8)
        self.assertEqual(
            [(item.station, item.wall) for item in profile],
            [
                ("entry", "right"),
                ("entry", "left"),
                ("entry", "top"),
                ("entry", "bottom"),
                ("bottom", "right"),
                ("bottom", "left"),
                ("bottom", "top"),
                ("bottom", "bottom"),
            ],
        )
        self.assertTrue(all(item.clearance_um == 3.0 for item in profile))

    def test_centered_pose_is_safe_without_active_constraints(self) -> None:
        state = classify_lateral_contact_state(self.dimensions, Pose6D())
        self.assertEqual(state.safety_level, "安全")
        self.assertEqual(state.topology, "无接触风险")
        self.assertEqual(state.active_constraints, ())

    def test_bottom_only_risk_is_single_side_local(self) -> None:
        state = classify_lateral_contact_state(
            self.dimensions,
            Pose6D(theta_y_rad=arcsec_to_rad(60.0)),
        )
        self.assertEqual(state.safety_level, "接近接触")
        self.assertEqual(state.topology, "单侧局部风险")
        self.assertEqual(state.axial_location, "底部")
        self.assertEqual(
            [(item.station, item.wall) for item in state.active_constraints],
            [("bottom", "right")],
        )

    def test_same_wall_at_entry_and_bottom_is_line_like_risk(self) -> None:
        state = classify_lateral_contact_state(self.dimensions, Pose6D(dx_um=2.9))
        self.assertEqual(state.safety_level, "接近接触")
        self.assertEqual(state.topology, "同侧双端风险（近似线接触）")
        self.assertEqual(state.axial_location, "入口与底部同侧")

    def test_adjacent_walls_are_corner_risk(self) -> None:
        state = classify_lateral_contact_state(
            self.dimensions,
            Pose6D(dx_um=2.9, dy_um=2.9),
        )
        self.assertEqual(state.topology, "相邻双壁角部风险")
        self.assertEqual({item.wall for item in state.active_constraints}, {"right", "top"})

    def test_opposite_walls_at_two_ends_are_wedging_risk(self) -> None:
        state = classify_lateral_contact_state(
            self.dimensions,
            Pose6D(dx_um=2.9, theta_y_rad=arcsec_to_rad(-120.0)),
        )
        self.assertEqual(state.topology, "对向双壁楔紧风险")
        self.assertEqual(state.axial_location, "入口与底部对侧")
        self.assertEqual({item.wall for item in state.active_constraints}, {"right", "left"})

    def test_yaw_can_create_multiwall_risk(self) -> None:
        state = classify_lateral_contact_state(
            self.dimensions,
            Pose6D(theta_z_rad=arcsec_to_rad(120.0)),
        )
        self.assertEqual(state.topology, "多壁复杂风险")
        self.assertEqual(
            {item.wall for item in state.active_constraints},
            {"right", "left", "top", "bottom"},
        )

    def test_contact_and_interference_are_not_conflated(self) -> None:
        contact = classify_lateral_contact_state(self.dimensions, Pose6D(dx_um=3.0))
        interference = classify_lateral_contact_state(self.dimensions, Pose6D(dx_um=3.1))
        self.assertEqual(contact.safety_level, "几何接触")
        self.assertEqual(interference.safety_level, "几何干涉")
        self.assertLess(interference.minimum_clearance_um, 0.0)


if __name__ == "__main__":
    unittest.main()
