from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from squarefit.units import arcsec_to_rad, mm_to_um, rad_to_arcsec, um_to_mm


class UnitConversionTests(unittest.TestCase):
    def test_angle_round_trip(self) -> None:
        self.assertAlmostEqual(rad_to_arcsec(arcsec_to_rad(2.75)), 2.75, places=12)

    def test_length_round_trip(self) -> None:
        self.assertAlmostEqual(um_to_mm(mm_to_um(10.006)), 10.006, places=12)


if __name__ == "__main__":
    unittest.main()
