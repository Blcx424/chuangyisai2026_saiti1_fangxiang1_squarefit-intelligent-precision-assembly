from __future__ import annotations

import csv
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from squarefit.compensation import analytic_compensation, apply_world_frame_adjustment
from squarefit.schema import (
    FEATURE_SETS,
    NETWORK_INPUT_COLUMNS,
    PRIMARY_TARGET_COLUMNS,
    validated_training_columns,
)
from squarefit.simulation import (
    ActuatorModel,
    MeasurementModel,
    ThermalCondition,
    ThermalModel,
    make_source_domain,
    make_safety_auxiliary_domain,
    make_size_boundary_domain,
    make_stress_domain,
    make_target_domain,
    residual_summary,
    save_dataset_bundle,
    save_dataset_csv,
    save_incoming_pair_screening_csv,
    simulate_many,
    thermal_clearance_sensitivity,
)
from squarefit.types import Pose6D
from squarefit.units import arcsec_to_rad


class SimulationTests(unittest.TestCase):
    def test_zero_noise_measurement_only_adds_declared_bias(self) -> None:
        model = MeasurementModel(
            translation_bias_um=(0.1, -0.2, 0.3),
            translation_noise_std_um=0.0,
            rotation_bias_arcsec=(0.4, -0.5, 0.6),
            rotation_noise_std_arcsec=0.0,
        )
        measured = model.measure(Pose6D(dx_um=1.0), random.Random(1))
        self.assertAlmostEqual(measured.dx_um, 1.1)
        self.assertAlmostEqual(measured.dy_um, -0.2)
        self.assertAlmostEqual(measured.dz_um, 0.3)
        self.assertAlmostEqual(measured.theta_x_rad, arcsec_to_rad(0.4))
        self.assertAlmostEqual(measured.theta_y_rad, arcsec_to_rad(-0.5))
        self.assertAlmostEqual(measured.theta_z_rad, arcsec_to_rad(0.6))

    def test_actuator_mean_inverse_returns_requested_physical_adjustment(self) -> None:
        actuator = ActuatorModel(
            translation_noise_std_um=0.0,
            rotation_noise_std_arcsec=0.0,
        )
        desired = Pose6D(
            dx_um=-0.83,
            dy_um=0.41,
            dz_um=-0.15,
            theta_x_rad=arcsec_to_rad(-1.1),
            theta_y_rad=arcsec_to_rad(0.8),
            theta_z_rad=arcsec_to_rad(-0.4),
        )
        command = actuator.inverse_mean_command(desired)
        actual = actuator.mean_physical_adjustment(command)
        for actual_value, desired_value in zip(
            (actual.dx_um, actual.dy_um, actual.dz_um, actual.theta_x_rad, actual.theta_y_rad, actual.theta_z_rad),
            (desired.dx_um, desired.dy_um, desired.dz_um, desired.theta_x_rad, desired.theta_y_rad, desired.theta_z_rad),
        ):
            self.assertAlmostEqual(actual_value, desired_value, places=12)

    def test_optimal_command_reaches_zero_in_noise_free_mean_plant(self) -> None:
        actuator = ActuatorModel(
            translation_noise_std_um=0.0,
            rotation_noise_std_arcsec=0.0,
        )
        true_pose = Pose6D(
            dx_um=0.7,
            dy_um=-0.5,
            dz_um=0.2,
            theta_x_rad=arcsec_to_rad(1.2),
            theta_y_rad=arcsec_to_rad(-0.7),
            theta_z_rad=arcsec_to_rad(0.5),
        )
        desired = analytic_compensation(true_pose).adjustment
        command = actuator.inverse_mean_command(desired)
        after = apply_world_frame_adjustment(true_pose, actuator.mean_physical_adjustment(command))
        for value in (
            after.dx_um,
            after.dy_um,
            after.dz_um,
            after.theta_x_rad,
            after.theta_y_rad,
            after.theta_z_rad,
        ):
            self.assertAlmostEqual(value, 0.0, places=9)

    def test_seed_makes_samples_reproducible(self) -> None:
        domain = make_source_domain()
        first = simulate_many(3, domain, seed=20260912)
        second = simulate_many(3, domain, seed=20260912)
        self.assertEqual(first, second)

    def test_stratified_sampling_follows_60_30_10_protocol(self) -> None:
        samples = simulate_many(100, make_source_domain(), seed=99)
        counts = {layer: sum(sample.risk_layer == layer for sample in samples) for layer in ("normal", "coverage", "difficult")}
        self.assertEqual(counts, {"normal": 60, "coverage": 30, "difficult": 10})

    def test_target_calibration_coverage_respects_clearance_and_nested_layers(self) -> None:
        domain = make_target_domain()
        samples = simulate_many(50, domain, seed=100, sampling_strategy="calibration")
        clearances = [sample.effective_dimensions.nominal_single_side_clearance_um for sample in samples]
        self.assertGreaterEqual(min(clearances), domain.selected_clearance_range_um[0])
        self.assertLessEqual(max(clearances), domain.selected_clearance_range_um[1])
        for prefix, expected in ((10, {"normal": 6, "coverage": 3, "difficult": 1}), (30, {"normal": 18, "coverage": 9, "difficult": 3}), (50, {"normal": 30, "coverage": 15, "difficult": 5})):
            counts = {layer: sum(sample.risk_layer == layer for sample in samples[:prefix]) for layer in expected}
            self.assertEqual(counts, expected)

    def test_safety_stress_samples_activate_all_declared_geometric_bands(self) -> None:
        samples = simulate_many(
            100,
            make_safety_auxiliary_domain(),
            seed=101,
            sampling_strategy="safety_stress",
        )
        counts = {band: sum(sample.safety_band == band for sample in samples) for band in ("collision", "near_contact", "tight_clearance", "loose_clearance")}
        self.assertEqual(counts, {"collision": 35, "near_contact": 25, "tight_clearance": 20, "loose_clearance": 20})
        self.assertTrue(
            all(
                abs(sample.true_pose.dz_um) <= make_safety_auxiliary_domain().translation_range_um
                for sample in samples
            )
        )
        # dz 与横向安全带独立：这里不要求 100 条小样本必然抽到某个极窄 dz 区间，
        # 只验证它不再被固定/截断为旧版的 ±0.8 um。
        self.assertGreater(max(abs(sample.true_pose.dz_um) for sample in samples), 0.8)
        self.assertGreater(
            max(abs(arcsec_to_rad(0.0) + sample.true_pose.theta_x_rad) for sample in samples),
            arcsec_to_rad(1.5),
        )
        self.assertTrue(all(sample.risk_layer == "difficult" for sample in samples))
        for sample in samples:
            minimum = sample.before_fit.minimum_clearance_um
            if sample.safety_band == "collision":
                self.assertTrue(sample.before_fit.collision)
            elif sample.safety_band == "near_contact":
                self.assertTrue(0.0 < minimum <= 0.5)
            elif sample.safety_band == "tight_clearance":
                self.assertTrue(0.5 < minimum <= 1.0)
            else:
                self.assertFalse(sample.before_fit.collision)
                self.assertTrue(any(gap > sample.dimensions.gap_upper_um for gap in sample.before_fit.labelled_gaps_um))

        stress_samples = simulate_many(
            100,
            make_stress_domain(),
            seed=102,
            sampling_strategy="safety_stress",
        )
        self.assertTrue(
            all(abs(sample.true_pose.dz_um) <= make_stress_domain().translation_range_um for sample in stress_samples)
        )
        self.assertGreater(max(abs(sample.true_pose.dz_um) for sample in stress_samples), 0.8)
        self.assertGreater(max(abs(sample.true_pose.theta_y_rad) for sample in stress_samples), arcsec_to_rad(1.5))
        self.assertTrue(all(sample.risk_layer == "difficult" for sample in stress_samples))

    def test_csv_has_network_label_and_measurement_columns(self) -> None:
        samples = simulate_many(3, make_source_domain(), seed=7)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "samples.csv"
            self.assertEqual(save_dataset_csv(samples, output), 3)
            self.assertFalse(output.read_bytes().startswith(b"\xef\xbb\xbf"))
            with output.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 3)
        self.assertIn("measured_dx_um", rows[0])
        self.assertIn("measured_hole_temperature_c", rows[0])
        self.assertIn("estimated_effective_single_side_clearance_um", rows[0])
        self.assertIn("known_hole_linear_expansion_per_c", rows[0])
        self.assertIn("true_effective_single_side_clearance_um", rows[0])
        self.assertIn("analytic_command_theta_z_arcsec", rows[0])
        self.assertIn("label_optimal_command_dx_um", rows[0])
        self.assertIn("label_residual_command_dx_um", rows[0])
        self.assertIn("after_optimal_g1_um", rows[0])
        self.assertEqual(rows[0]["after_optimal_joint_pass"], "True")

    def test_main_network_schema_never_uses_simulation_truth(self) -> None:
        self.assertEqual(len(NETWORK_INPUT_COLUMNS), 9)
        self.assertEqual(len(PRIMARY_TARGET_COLUMNS), 6)
        self.assertFalse(any(column.startswith("true_") for column in NETWORK_INPUT_COLUMNS))
        self.assertTrue(all(column.startswith("label_optimal_command_") for column in PRIMARY_TARGET_COLUMNS))

    def test_feature_sets_define_the_promised_8_9_and_15_dimensions(self) -> None:
        self.assertEqual(len(FEATURE_SETS["A_基础BP_8维"]), 8)
        self.assertEqual(len(FEATURE_SETS["B_选配间隙BP_9维"]), 9)
        self.assertEqual(len(FEATURE_SETS["C_解析先验BP_15维"]), 15)
        self.assertIn("nominal_single_side_clearance_um", FEATURE_SETS["B_选配间隙BP_9维"])
        self.assertFalse(any("temperature" in column for columns in FEATURE_SETS.values() for column in columns))
        self.assertFalse(
            any(column.startswith("true_") for columns in FEATURE_SETS.values() for column in columns)
        )

    def test_training_columns_use_an_explicit_whitelist_not_remaining_csv_columns(self) -> None:
        all_columns = (
            *FEATURE_SETS["B_选配间隙BP_9维"],
            *PRIMARY_TARGET_COLUMNS,
            "true_dx_um",
            "physical_adjustment_dx_um",
            "before_g1_um",
            "after_optimal_g1_um",
            "label_optimal_safety_margin_um",
        )
        features, targets = validated_training_columns("B_选配间隙BP_9维", all_columns)
        self.assertEqual(features, FEATURE_SETS["B_选配间隙BP_9维"])
        self.assertEqual(targets, PRIMARY_TARGET_COLUMNS)
        self.assertFalse(any(column.startswith(("true_", "after_", "label_")) for column in features))

    def test_temperature_difference_changes_effective_single_side_clearance(self) -> None:
        model = ThermalModel(
            hole_linear_expansion_per_c=11.5e-6,
            shaft_linear_expansion_per_c=11.5e-6,
        )
        reference = make_source_domain().sample_dimensions(
            random.Random(7),
            ThermalCondition(hole_temperature_c=20.0, shaft_temperature_c=20.0),
        )
        effective = model.effective_dimensions(
            reference,
            ThermalCondition(hole_temperature_c=30.0, shaft_temperature_c=20.0),
        )
        self.assertAlmostEqual(
            effective.nominal_single_side_clearance_um - reference.nominal_single_side_clearance_um,
            reference.hole_side_um * 11.5e-6 * 10.0 / 2.0,
            places=9,
        )

    def test_task_primary_domains_are_isothermal_matched_copper(self) -> None:
        sample = simulate_many(1, make_source_domain(), seed=117)[0]
        self.assertEqual(sample.thermal_condition.hole_temperature_c, 20.0)
        self.assertEqual(sample.thermal_condition.shaft_temperature_c, 20.0)
        self.assertEqual(sample.measured_thermal_condition.hole_temperature_c, 20.0)
        self.assertEqual(sample.measured_thermal_condition.shaft_temperature_c, 20.0)
        self.assertEqual(sample.hole_linear_expansion_per_c, sample.shaft_linear_expansion_per_c)
        self.assertAlmostEqual(sample.effective_dimensions.hole_side_um, sample.dimensions.hole_side_um)
        self.assertAlmostEqual(sample.effective_dimensions.shaft_side_um, sample.dimensions.shaft_side_um)

    def test_material_sensitivity_contains_low_steel_and_dissimilar_cases(self) -> None:
        rows = thermal_clearance_sensitivity()
        self.assertEqual(len(rows), 6)
        steel_hot_hole = next(row for row in rows if row["material_assumption"].startswith("钢类") and row["condition"].startswith("孔热"))
        invar_hot_hole = next(row for row in rows if row["material_assumption"].startswith("殷钢") and row["condition"].startswith("孔热"))
        self.assertGreater(abs(float(steel_hot_hole["clearance_shift_um"])), abs(float(invar_hot_hole["clearance_shift_um"])))

    def test_optimal_label_is_fully_qualified_in_mean_virtual_plant(self) -> None:
        sample = simulate_many(1, make_source_domain(), seed=18)[0]
        self.assertTrue(sample.optimal_mean_objective.fully_qualified)

    def test_dataset_bundle_writes_metadata_and_hashes(self) -> None:
        source_domain = make_source_domain()
        target_domain = make_target_domain()
        source = simulate_many(10, source_domain, seed=20)
        calibration = simulate_many(3, target_domain, seed=21, sampling_strategy="calibration")
        with tempfile.TemporaryDirectory() as directory:
            metadata = save_dataset_bundle(
                Path(directory),
                (
                    ("source.csv", source, source_domain, True),
                    ("calibration.csv", calibration, target_domain, False),
                ),
                base_seed=20,
            )
            with (Path(directory) / "metadata.json").open("r", encoding="utf-8") as file:
                from_file = json.load(file)
            self.assertEqual(metadata["dataset_protocol"], "v5.0_full_feasible_clearance_selective_assembly")
            self.assertEqual(metadata["pose_compensation_feasibility"]["total_clearance_um"], [5.0, 8.0])
        self.assertEqual(metadata["seed_offsets"]["source_safety_auxiliary"], 5)
        self.assertIn("source.csv", from_file["files"])
        self.assertEqual(len(from_file["files"]["source.csv"]["sha256"]), 64)

    def test_full_feasible_and_size_boundary_sampling_cover_declared_domain(self) -> None:
        domain = make_target_domain()
        samples = simulate_many(300, domain, seed=27, sampling_strategy="full_feasible_coverage")
        clearances = [sample.effective_dimensions.nominal_single_side_clearance_um for sample in samples]
        self.assertLessEqual(min(clearances), 2.51)
        self.assertGreaterEqual(max(clearances), 3.99)
        boundary = simulate_many(20, make_size_boundary_domain(), seed=28, sampling_strategy="size_boundary")
        self.assertEqual(sum(item.size_band == "lower_boundary" for item in boundary), 10)
        self.assertEqual(sum(item.size_band == "upper_boundary" for item in boundary), 10)

    def test_incoming_screening_never_creates_teacher_label_for_rematching_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incoming.csv"
            quality = save_incoming_pair_screening_csv(path, count=200, seed=29)
            with path.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
        self.assertEqual(quality["sample_count"], 200)
        for row in rows:
            total = float(row["total_clearance_um"])
            expected = "pose_compensation_feasible" if 5.0 <= total <= 8.0 else "requires_rematching"
            self.assertEqual(row["pair_status"], expected)

    def test_summary_has_a_valid_rate(self) -> None:
        summary = residual_summary(simulate_many(20, make_source_domain(), seed=11))
        self.assertEqual(summary["sample_count"], 20)
        self.assertGreaterEqual(summary["residual_requirement_pass_rate"], 0.0)
        self.assertLessEqual(summary["residual_requirement_pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
