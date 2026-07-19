import csv
import os
import tempfile
import unittest

from src.evaluation.controlled_fixture import write_controlled_fixture
from src.evaluation.metrics import binary_metrics, evaluate_on_rois, load_roi_manifest


class BinaryMetricsTests(unittest.TestCase):
    def test_binary_metrics_confusion_matrix(self):
        metrics = binary_metrics(
            [True, True, False, False],
            [True, False, True, False],
        )
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["true_negative"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["f1_score"], 0.5)

    def test_recall_is_undefined_without_positive_labels(self):
        metrics = binary_metrics([False, False], [False, False])
        self.assertIsNone(metrics["recall"])
        self.assertIsNone(metrics["f1_score"])

    def test_f1_is_zero_when_all_positive_samples_are_missed(self):
        metrics = binary_metrics([True, True, False], [False, False, False])
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1_score"], 0.0)


class RoiEvaluationTests(unittest.TestCase):
    def test_controlled_roi_evaluation_writes_comparison_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture_dir = os.path.join(temp_dir, "fixture")
            manifest_path = write_controlled_fixture(fixture_dir)
            records = load_roi_manifest(manifest_path)
            baseline_path = os.path.join(temp_dir, "landmark.csv")
            with open(
                baseline_path, "w", newline="", encoding="utf-8"
            ) as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=[
                        "sample",
                        "eye_closed_pred",
                        "yawn_pred",
                    ],
                )
                writer.writeheader()
                for record in records:
                    writer.writerow(
                        {
                            "sample": record["sample"],
                            "eye_closed_pred": int(record["eye_closed"]),
                            "yawn_pred": 0,
                        }
                    )

            output_dir = os.path.join(temp_dir, "results")
            report = evaluate_on_rois(
                manifest_path,
                output_dir=output_dir,
                landmark_baseline_path=baseline_path,
            )

            self.assertEqual(report["dataset"]["evaluated_samples"], 12)
            self.assertTrue(report["dataset"]["face_detection_bypassed"])
            self.assertEqual(report["edge_feature"]["eye_closed"]["accuracy"], 1.0)
            self.assertEqual(report["edge_feature"]["yawn"]["f1_score"], 1.0)
            self.assertEqual(report["edge_feature"]["analysis_valid_rate"], 1.0)
            self.assertEqual(report["landmark"]["eye_closed"]["accuracy"], 1.0)
            self.assertEqual(report["landmark"]["yawn"]["recall"], 0.0)
            self.assertEqual(report["landmark"]["yawn"]["f1_score"], 0.0)

            for filename in ("predictions.csv", "comparison.json", "comparison.md"):
                self.assertTrue(os.path.exists(os.path.join(output_dir, filename)))


if __name__ == "__main__":
    unittest.main()
