"""Synthetic public-API/CLI tests; constants here are not a study policy."""

import csv
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from darwin.observation_qc import UNITS, ScreeningError, screen_csv, solar_elevation

START = "2022-08-01T00:00:00Z"
END = "2022-08-02T00:00:00Z"


def toy_policy():
    return {
        "schema": "darwin.aws-screening-policy/1",
        "id": "synthetic-tests-not-study-defaults",
        "status": "provisional_screening_only",
        "cadence_seconds": 600,
        "utc_offset_minutes": -360,
        "timestamp_label": "interval_end",
        "time_basis_status": "inferred_not_operator_confirmed",
        "ranges": {
            "T": [-5, 40],
            "RH": [0, 103],
            "WS": [0, 40],
            "WD": [0, 360],
            "SLR": [-5, 1400],
            **{f: [0, None] for f in ("PCP_tot_bucket", "PCP_diff_radar", "PCP_acoustic")},
        },
        "rh_cap_percent": 100,
        "step_limits": {"T": 5, "RH": 30},
        "persistence_seconds": 10800,
        "saturated_rh_persistence_seconds": 21600,
        "saturated_rh_percent": 99,
        "pressure_reference": {
            "method": "isa_troposphere",
            "sea_level_hpa": 1013.25,
            "sea_level_kelvin": 288.15,
            "lapse_kelvin_per_m": 0.0065,
            "gravity_m_s2": 9.80665,
            "dry_air_gas_constant": 287.05287,
            "tolerance_hpa": 60,
        },
        "night": {
            "method": "noaa_julian_century_geometric_v1",
            "elevation_below_degrees": -6,
            "shortwave_above_w_m2": 20,
        },
        "gauge": {"ratio_above": 2, "larger_total_above_mm": 2},
        "minimum_interval_fraction": 0.8,
        "hour_mean_min_records": 5,
        "day_mean_min_hours": 22,
    }


class ScreeningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="aws-screening-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "synthetic.csv"
        self.policy = self.root / "policy.json"
        self.metadata = self.root / "station.json"
        self.p = toy_policy()

    def write(self, values, *, fields=("T",), first=START):
        text = io.StringIO(newline="")
        writer = csv.writer(text)
        writer.writerow(["datetime", *fields])
        beginning = datetime.fromisoformat(first)
        for index, item in enumerate(values):
            local = beginning + timedelta(minutes=10 * (index + 1) - 360)
            writer.writerow([local.replace(tzinfo=None).isoformat(), *item])
        self.source.write_text(text.getvalue())
        self.m = {
            "schema": "darwin.aws-screening-station/1",
            "id": "synthetic",
            "latitude": -0.7,
            "longitude": -90.3,
            "elevation_m": 2,
            "units": {field: UNITS[field] for field in fields},
            "source_sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
            "metadata_source_sha256": "0" * 64,
        }
        self.save()

    def save(self):
        self.policy.write_text(json.dumps(self.p))
        self.metadata.write_text(json.dumps(self.m))

    def run_screen(self, **kwargs):
        return screen_csv(
            self.source,
            station_metadata=self.metadata,
            policy=self.policy,
            start=START,
            end=kwargs.get("end", END),
        )

    def counts(self, report, field="T"):
        return report["daily"]["2022-08-01"][field]

    def reject(self, code):
        with self.assertRaises(ScreeningError) as result:
            self.run_screen()
        self.assertEqual(result.exception.code, code)

    def test_complete_axis_and_immutable_inputs(self):
        self.write([(20 + index % 2,) for index in range(144)])
        before = [p.read_bytes() for p in (self.source, self.policy, self.metadata)]
        report = self.run_screen()
        self.assertEqual(self.counts(report)["eligible"], 144)
        self.assertTrue(self.counts(report)["daily_aggregation_eligible"])
        self.assertEqual(report["scientific_status"], "not_assessed")
        self.assertEqual(report["missing_times"], [])
        self.assertEqual(
            before, [p.read_bytes() for p in (self.source, self.policy, self.metadata)]
        )

    def test_range_boundaries(self):
        self.write([(-5,), (40,), (40.01,), (-5.01,)], fields=("T",))
        report = self.run_screen()
        self.assertEqual(self.counts(report)["flags"]["range"], 2)

    def test_rh_caps_preserve_original_persistence_values(self):
        self.write([(100 + index % 2,) for index in range(80)], fields=("RH",))
        report = self.run_screen()
        self.assertEqual(len(report["corrections"]), 40)
        self.assertEqual(self.counts(report, "RH")["eligible"], 80)

    def test_humidity_range(self):
        self.write([(103,), (103.01,), (-0.01,)], fields=("RH",))
        report = self.run_screen()
        self.assertEqual(self.counts(report, "RH")["flags"]["range"], 2)
        self.assertEqual(len(report["corrections"]), 1)

    def test_step_strict_threshold_and_later_sample(self):
        self.write([(20,), (25,), (19,)], fields=("T",))
        steps = [i for i in self.run_screen()["issues"] if "step" in i["reasons"]]
        self.assertEqual([i["time"] for i in steps], ["2022-08-01T00:30:00Z"])

    def test_persistence_strict_boundary(self):
        for length, expected in ((19, 19), (20, 0)):
            self.write([(20,)] * length)
            self.assertEqual(self.counts(self.run_screen())["eligible"], expected)

    def test_saturated_humidity_extended_boundary(self):
        for length, expected in ((37, 37), (38, 0)):
            self.write([(99,)] * length, fields=("RH",))
            self.assertEqual(self.counts(self.run_screen(), "RH")["eligible"], expected)

    def test_persistence_context_across_day_start(self):
        self.write([(20,)] * 25, first="2022-07-31T21:00:00Z")
        report = self.run_screen()
        self.assertEqual(self.counts(report)["eligible"], 0)
        self.assertEqual(self.counts(report)["available"], 7)

    def test_null_resets_persistence_and_is_not_numeric(self):
        self.write([(20,)] * 19 + [("NULL",)] + [(20,)] * 19)
        count = self.counts(self.run_screen())
        self.assertEqual(count["eligible"], 38)
        self.assertEqual(count["flags"], {"missing": 1})

    def test_timestamp_gap_resets_step_and_persistence(self):
        self.write([(20,)] * 19 + [(30,)] * 19)
        lines = self.source.read_text().splitlines()
        # Remove the first 30-degree sample, leaving a twenty-minute gap.
        self.source.write_text("\n".join(lines[:20] + lines[21:]))
        self.m["source_sha256"] = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.save()
        self.assertEqual(self.counts(self.run_screen())["eligible"], 37)

    def test_heldout_values_not_inspected(self):
        self.write([(20,)])
        with self.source.open("a") as stream:
            stream.write("2023-08-01T00:00:00,not-a-number\n")
        self.m["source_sha256"] = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.save()
        self.assertEqual(self.counts(self.run_screen())["eligible"], 1)

    def test_missing_timestamps_not_fabricated(self):
        self.write([(20,), (21,)])
        report = self.run_screen()
        self.assertEqual(len(report["missing_times"]), 142)
        self.assertFalse(report["totals"]["T"]["interval_score_eligible"])

    def test_empty_station_is_explicit(self):
        self.write([])
        report = self.run_screen()
        self.assertEqual(self.counts(report)["available"], 0)
        self.assertEqual(len(report["missing_times"]), 144)

    def test_duplicate_and_nonmonotonic_rejected(self):
        for indices in ((1, 1), (2, 1)):
            self.write([(20,), (21,)])
            lines = self.source.read_text().splitlines()
            self.source.write_text("\n".join([lines[0], lines[indices[0]], lines[indices[1]]]))
            self.m["source_sha256"] = hashlib.sha256(self.source.read_bytes()).hexdigest()
            self.save()
            self.reject("timestamp_order")

    def test_off_grid_and_nonfinite_rejected(self):
        self.write([(20,)], first="2022-08-01T00:01:00Z")
        self.reject("off_grid")
        self.write([("inf",)])
        self.reject("numeric")

    def test_unknown_versions_and_units_rejected(self):
        self.write([(20,)])
        self.p["schema"] = "future"
        self.save()
        self.reject("version")
        self.p = toy_policy()
        self.m["units"]["T"] = "K"
        self.save()
        self.reject("units")

    def test_policy_unknown_fields_and_bad_parameters(self):
        self.write([(20,)])
        for key, value in (
            ("extra", True),
            ("cadence_seconds", True),
            ("minimum_interval_fraction", 2),
            ("ranges", {}),
        ):
            self.p = toy_policy()
            self.p[key] = value
            self.save()
            with self.assertRaises(ScreeningError):
                self.run_screen()

    def test_source_mutation_rejected(self):
        self.write([(20,)])
        self.source.write_bytes(self.source.read_bytes().replace(b",20", b",21"))
        self.reject("source_digest")

    def test_duplicate_unused_columns_are_explicitly_projected_out(self):
        self.write([(20,)])
        self.source.write_text("datetime,T,WSmax,WSmax\n2022-07-31T18:10:00,20,1,2\n")
        self.m["source_sha256"] = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.save()
        report = self.run_screen()
        self.assertEqual(report["source_columns"]["duplicate_ignored"], ["WSmax"])
        self.assertEqual(self.counts(report)["eligible"], 1)
        self.source.write_text("datetime,T,T\n2022-07-31T18:10:00,20,21\n")
        self.m["source_sha256"] = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.save()
        self.reject("columns")

    def test_night_check_is_not_a_daytime_ceiling(self):
        self.write([(21,)] * 144, fields=("SLR",))
        report = self.run_screen()
        self.assertGreater(self.counts(report, "SLR")["flags"]["night_radiation"], 50)
        self.assertGreater(self.counts(report, "SLR")["eligible"], 50)

    def test_standard_atmosphere_pressure(self):
        self.write([(1013.25,), (900,), (1100,)], fields=("Pabs",))
        self.assertEqual(self.counts(self.run_screen(), "Pabs")["eligible"], 1)

    def test_gauge_strict_ratio_and_zero(self):
        for lo, hi, state in (
            (1, 2, "passed"),
            (0, 2, "passed"),
            (0, 2.01, "flagged"),
            (1, 2.01, "flagged"),
        ):
            self.write([(lo, hi)] + [(0, 0)] * 143, fields=("PCP_tot_bucket", "PCP_diff_radar"))
            report = self.run_screen()
            self.assertEqual(report["gauge_checks"][0]["status"], state)

    def test_gauges_incomplete_not_passed_and_three_pairs_independent(self):
        self.write([(0, 3)], fields=("PCP_tot_bucket", "PCP_diff_radar"))
        self.assertEqual(self.run_screen()["gauge_checks"][0]["status"], "unevaluated")
        self.write(
            [(0, 3, 8)] + [(0, 0, 0)] * 143,
            fields=("PCP_tot_bucket", "PCP_diff_radar", "PCP_acoustic"),
        )
        self.assertEqual([g["status"] for g in self.run_screen()["gauge_checks"]], ["flagged"] * 3)

    def test_hour_means_and_rain_completeness_differ(self):
        self.write(
            [(20 + index % 2, 0) if index % 6 else ("NULL", "NULL") for index in range(144)],
            fields=("T", "PCP_diff_radar"),
        )
        report = self.run_screen()
        self.assertTrue(self.counts(report)["daily_aggregation_eligible"])
        self.assertFalse(self.counts(report, "PCP_diff_radar")["daily_aggregation_eligible"])

    def test_right_closed_midnight_ownership(self):
        self.write([(20 + index % 2,) for index in range(288)])
        report = self.run_screen(end="2022-08-03T00:00:00Z")
        self.assertEqual([d["T"]["eligible"] for d in report["daily"].values()], [144, 144])

    def test_python_cli_equivalence_and_error(self):
        self.write([(20,), (21,)])
        command = [
            sys.executable,
            "-m",
            "darwin.observation_qc",
            "--source",
            str(self.source),
            "--station-metadata",
            str(self.metadata),
            "--policy",
            str(self.policy),
            "--start",
            START,
            "--end",
            END,
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), self.run_screen())
        self.source.write_text("changed")
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stderr)["error"]["code"], "source_digest")

    def test_solar_reference_and_hemispheres(self):
        # NREL SPA example: geometric elevation approx. 39.872 degrees.
        time = datetime(2003, 10, 17, 19, 30, 30, tzinfo=UTC)
        self.assertAlmostEqual(solar_elevation(time, 39.742476, -105.1786), 39.872, delta=0.05)
        self.assertGreater(solar_elevation(datetime(2024, 3, 20, 12, tzinfo=UTC), 0, 0), 87)
        self.assertLess(solar_elevation(datetime(2024, 3, 20, tzinfo=UTC), 0, 0), -87)
        with self.assertRaises(ScreeningError):
            solar_elevation(time, 91, 0)


if __name__ == "__main__":
    unittest.main()
