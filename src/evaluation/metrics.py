import argparse
import csv
import json
import logging
import os
import time

import cv2
import numpy as np

from src.configs.config import Config
from src.core.detector import DrowsinessDetector
from src.core.facial_analyzer import FacialAnalyzer
from src.core.feature_classifier import classify_face_features


logger = logging.getLogger(__name__)


def _parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def binary_metrics(ground_truth, predictions):
    ground_truth = np.asarray(ground_truth, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    if ground_truth.shape != predictions.shape:
        raise ValueError("Ground-truth and prediction arrays must have equal length")
    if ground_truth.size == 0:
        return {}

    true_positive = int(np.sum(ground_truth & predictions))
    true_negative = int(np.sum(~ground_truth & ~predictions))
    false_positive = int(np.sum(~ground_truth & predictions))
    false_negative = int(np.sum(ground_truth & ~predictions))
    positive_support = true_positive + false_negative

    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else None

    return {
        "samples": int(len(ground_truth)),
        "positive_support": positive_support,
        "predicted_positive": true_positive + false_positive,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "accuracy": ratio(true_positive + true_negative, len(ground_truth)),
        "precision": ratio(true_positive, true_positive + false_positive),
        "recall": ratio(true_positive, positive_support),
        "specificity": ratio(true_negative, true_negative + false_positive),
        "f1_score": (
            ratio(
                2 * true_positive,
                2 * true_positive + false_positive + false_negative,
            )
            if positive_support
            else None
        ),
        "false_positive_rate": ratio(
            false_positive, false_positive + true_negative
        ),
    }


def load_ground_truth(path):
    records = {}
    with open(path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"frame", "eye_closed", "yawn"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Ground-truth CSV is missing columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            frame = int(row["frame"])
            if frame in records:
                raise ValueError(f"Duplicate ground-truth frame: {frame}")
            records[frame] = {
                "eye_closed": _parse_bool(row["eye_closed"]),
                "yawn": _parse_bool(row["yawn"]),
            }
    return records


def load_roi_manifest(path):
    records = []
    seen_samples = set()
    manifest_dir = os.path.dirname(os.path.abspath(path))
    with open(path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {
            "sample",
            "image",
            "x1",
            "y1",
            "x2",
            "y2",
            "eye_closed",
            "yawn",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"ROI manifest is missing columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            sample = row["sample"].strip()
            if not sample:
                raise ValueError("ROI manifest contains an empty sample identifier")
            if sample in seen_samples:
                raise ValueError(f"Duplicate ROI sample: {sample}")
            seen_samples.add(sample)

            image_value = row["image"].strip()
            image_path = (
                image_value
                if os.path.isabs(image_value)
                else os.path.join(manifest_dir, image_value)
            )
            face_box = tuple(int(row[key]) for key in ("x1", "y1", "x2", "y2"))
            if face_box[2] <= face_box[0] or face_box[3] <= face_box[1]:
                raise ValueError(f"Invalid face box for ROI sample {sample}: {face_box}")
            records.append(
                {
                    "sample": sample,
                    "image": os.path.abspath(image_path),
                    "face_box": face_box,
                    "eye_closed": _parse_bool(row["eye_closed"]),
                    "yawn": _parse_bool(row["yawn"]),
                }
            )
    return records


def load_landmark_baseline(
    path,
    eye_threshold=0.22,
    mouth_threshold=0.30,
    key_field="frame",
):
    records = {}
    with open(path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fields = set(reader.fieldnames or [])
        if key_field not in fields:
            raise ValueError(
                f"Landmark baseline CSV is missing the {key_field} column"
            )

        has_predictions = {"eye_closed_pred", "yawn_pred"}.issubset(fields)
        has_ratios = {"ear", "mar"}.issubset(fields)
        if not has_predictions and not has_ratios:
            raise ValueError(
                "Landmark baseline CSV must contain eye_closed_pred/yawn_pred "
                "or raw ear/mar columns"
            )

        for row in reader:
            key = (
                int(row[key_field])
                if key_field == "frame"
                else row[key_field].strip()
            )
            if key == "":
                raise ValueError(
                    f"Landmark baseline contains an empty {key_field} identifier"
                )
            if key in records:
                raise ValueError(f"Duplicate landmark baseline {key_field}: {key}")
            if has_predictions:
                eye_closed = _parse_bool(row["eye_closed_pred"])
                yawn = _parse_bool(row["yawn_pred"])
            else:
                eye_closed = float(row["ear"]) < eye_threshold
                yawn = float(row["mar"]) > mouth_threshold
            records[key] = {
                "eye_closed_pred": eye_closed,
                "yawn_pred": yawn,
            }
    return records


def _method_summary(rows, prefix, processing_times=None):
    summary = {
        "eye_closed": binary_metrics(
            [row["eye_closed_gt"] for row in rows],
            [row[f"{prefix}_eye_closed_pred"] for row in rows],
        ),
        "yawn": binary_metrics(
            [row["yawn_gt"] for row in rows],
            [row[f"{prefix}_yawn_pred"] for row in rows],
        ),
    }
    if prefix == "edge_feature":
        coverage_fields = {
            "face_detected": "face_detection_rate",
            "analysis_valid": "analysis_valid_rate",
            "eye_analysis_valid": "eye_valid_rate",
            "mouth_analysis_valid": "mouth_valid_rate",
        }
        for row_field, summary_field in coverage_fields.items():
            if all(row_field in row for row in rows):
                summary[summary_field] = float(
                    np.mean([row[row_field] for row in rows])
                )
    if processing_times:
        average_ms = float(np.mean(processing_times))
        summary["average_processing_ms"] = average_ms
        summary["processing_fps"] = float(1000.0 / average_ms) if average_ms else None
    return summary


def _metric_delta(current, baseline):
    delta = {}
    for task in ("eye_closed", "yawn"):
        delta[task] = {}
        for metric in ("accuracy", "precision", "recall", "f1_score"):
            current_value = current[task].get(metric)
            baseline_value = baseline[task].get(metric)
            delta[task][metric] = (
                float(current_value - baseline_value)
                if current_value is not None and baseline_value is not None
                else None
            )
    return delta


def _format_metric(value):
    return "n/a" if value is None else f"{value:.3f}"


def _markdown_report(report):
    dataset = report["dataset"]
    lines = ["# Edge/Feature vs Landmark Evaluation", ""]
    if "video" in dataset:
        lines.extend(
            [
                f"- Video: `{dataset['video']}`",
                f"- Labels: `{dataset['labels']}`",
                f"- Evaluated frames: {dataset['evaluated_frames']}",
            ]
        )
        runtime_unit = "frame"
    else:
        lines.extend(
            [
                f"- ROI manifest: `{dataset['roi_manifest']}`",
                f"- Evaluated samples: {dataset['evaluated_samples']}",
                f"- Source images: {dataset['source_images']}",
                "- Face detection: bypassed; labeled face boxes were used directly",
            ]
        )
        runtime_unit = "sample"
    if "landmark_baseline" in dataset:
        lines.append(f"- Landmark baseline: `{dataset['landmark_baseline']}`")
    lines.append("")

    for task, title in (("eye_closed", "Eye closure"), ("yawn", "Yawn")):
        lines.extend(
            [
                f"## {title}",
                "",
                "| Method | Samples | Accuracy | Precision | Recall | F1 | TP | TN | FP | FN |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method_key, method_name in (
            ("edge_feature", "Edge + feature points"),
            ("landmark", "68-point landmark"),
        ):
            if method_key not in report:
                continue
            metrics = report[method_key][task]
            lines.append(
                "| {name} | {samples} | {accuracy} | {precision} | {recall} | "
                "{f1} | {tp} | {tn} | {fp} | {fn} |".format(
                    name=method_name,
                    samples=metrics["samples"],
                    accuracy=_format_metric(metrics["accuracy"]),
                    precision=_format_metric(metrics["precision"]),
                    recall=_format_metric(metrics["recall"]),
                    f1=_format_metric(metrics["f1_score"]),
                    tp=metrics["true_positive"],
                    tn=metrics["true_negative"],
                    fp=metrics["false_positive"],
                    fn=metrics["false_negative"],
                )
            )
        lines.append("")

    edge_summary = report["edge_feature"]
    lines.extend(
        [
            "## Runtime",
            "",
            f"- Edge/feature average: "
            f"{_format_metric(edge_summary.get('average_processing_ms'))} "
            f"ms/{runtime_unit}",
            f"- Edge/feature throughput: "
            f"{_format_metric(edge_summary.get('processing_fps'))} "
            f"{runtime_unit}s/s",
        ]
    )
    coverage_labels = (
        ("face_detection_rate", "Face detection coverage"),
        ("analysis_valid_rate", "Valid feature coverage"),
        ("eye_valid_rate", "Valid eye coverage"),
        ("mouth_valid_rate", "Valid mouth coverage"),
    )
    for key, label in coverage_labels:
        if key in edge_summary:
            lines.append(f"- {label}: {_format_metric(edge_summary[key])}")
    lines.extend(
        [
            "",
            "Metrics are only comparable when both methods use the same labels.",
            "A task with no positive ground-truth samples has undefined recall/F1 and is shown as n/a.",
            "",
        ]
    )
    return "\n".join(lines)


def _attach_landmark_predictions(rows, baseline, row_key):
    missing = [row[row_key] for row in rows if row[row_key] not in baseline]
    if missing:
        preview = ", ".join(str(value) for value in missing[:10])
        raise ValueError(
            "Landmark baseline does not cover every evaluated sample: " + preview
        )
    for row in rows:
        prediction = baseline[row[row_key]]
        row["landmark_eye_closed_pred"] = prediction["eye_closed_pred"]
        row["landmark_yawn_pred"] = prediction["yawn_pred"]


def _build_report(rows, dataset, processing_times, landmark_baseline_path=None):
    report = {
        "dataset": dataset,
        "edge_feature": _method_summary(
            rows, "edge_feature", processing_times=processing_times
        ),
    }
    if landmark_baseline_path:
        report["landmark"] = _method_summary(rows, "landmark")
        report["delta_edge_feature_minus_landmark"] = _metric_delta(
            report["edge_feature"],
            report["landmark"],
        )
        report["dataset"]["landmark_baseline"] = os.path.abspath(
            landmark_baseline_path
        )
    return report


def _write_report(rows, report, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    predictions_path = os.path.join(output_dir, "predictions.csv")
    report_path = os.path.join(output_dir, "comparison.json")
    markdown_path = os.path.join(output_dir, "comparison.md")

    with open(predictions_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(report_path, "w", encoding="utf-8") as json_file:
        json.dump(report, json_file, indent=2, ensure_ascii=False)
    with open(markdown_path, "w", encoding="utf-8") as markdown_file:
        markdown_file.write(_markdown_report(report))
    logger.info("Evaluation report written to %s", report_path)


def evaluate_on_video(
    video_path,
    labels_path,
    output_dir="evaluation_results",
    landmark_baseline_path=None,
    landmark_eye_threshold=0.22,
    landmark_mouth_threshold=0.30,
    edge_eye_threshold=Config.EYE_OPEN_THRESHOLD,
    edge_mouth_threshold=Config.MOUTH_OPEN_THRESHOLD,
):
    ground_truth = load_ground_truth(labels_path)
    if not ground_truth:
        raise ValueError("Ground-truth CSV has no rows")

    baseline = None
    if landmark_baseline_path:
        baseline = load_landmark_baseline(
            landmark_baseline_path,
            eye_threshold=landmark_eye_threshold,
            mouth_threshold=landmark_mouth_threshold,
        )

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    detector = DrowsinessDetector()
    detector.eye_open_threshold = edge_eye_threshold
    detector.mouth_open_threshold = edge_mouth_threshold
    rows = []
    processing_times = []
    target_frames = set(ground_truth)
    final_frame = max(target_frames)
    frame_index = 0

    try:
        while frame_index <= final_frame:
            success, frame = capture.read()
            if not success:
                break
            if frame_index not in target_frames:
                frame_index += 1
                continue

            started = time.perf_counter()
            _, _, metrics = detector.process_frame_from_frame(frame)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            processing_times.append(elapsed_ms)

            label = ground_truth[frame_index]
            rows.append(
                {
                    "frame": frame_index,
                    "eye_closed_gt": label["eye_closed"],
                    "yawn_gt": label["yawn"],
                    "edge_feature_eye_closed_pred": bool(metrics["eye_closed"]),
                    "edge_feature_yawn_pred": bool(metrics["yawning"]),
                    "eye_openness": float(metrics["eye_openness"]),
                    "mouth_openness": float(metrics["mouth_openness"]),
                    "eye_confidence": float(metrics["eye_confidence"]),
                    "mouth_confidence": float(metrics["mouth_confidence"]),
                    "face_detected": bool(metrics["face_detected"]),
                    "analysis_valid": bool(metrics["analysis_valid"]),
                    "eye_analysis_valid": bool(metrics["eye_analysis_valid"]),
                    "mouth_analysis_valid": bool(metrics["mouth_analysis_valid"]),
                    "processing_ms": elapsed_ms,
                }
            )
            frame_index += 1
    finally:
        capture.release()

    missing_frames = sorted(target_frames.difference(row["frame"] for row in rows))
    if missing_frames:
        raise ValueError(
            "Video ended before labeled frames were reached: "
            + ", ".join(str(frame) for frame in missing_frames[:10])
        )
    if landmark_baseline_path:
        _attach_landmark_predictions(rows, baseline, "frame")

    report = _build_report(
        rows,
        {
            "video": os.path.abspath(video_path),
            "labels": os.path.abspath(labels_path),
            "evaluated_frames": len(rows),
            "edge_eye_threshold": edge_eye_threshold,
            "edge_mouth_threshold": edge_mouth_threshold,
        },
        processing_times,
        landmark_baseline_path=landmark_baseline_path,
    )
    _write_report(rows, report, output_dir)
    return report


def evaluate_on_rois(
    manifest_path,
    output_dir="evaluation_results",
    landmark_baseline_path=None,
    landmark_eye_threshold=0.22,
    landmark_mouth_threshold=0.30,
    edge_eye_threshold=Config.EYE_OPEN_THRESHOLD,
    edge_mouth_threshold=Config.MOUTH_OPEN_THRESHOLD,
):
    records = load_roi_manifest(manifest_path)
    if not records:
        raise ValueError("ROI manifest has no rows")

    baseline = None
    if landmark_baseline_path:
        baseline = load_landmark_baseline(
            landmark_baseline_path,
            eye_threshold=landmark_eye_threshold,
            mouth_threshold=landmark_mouth_threshold,
            key_field="sample",
        )

    config = Config()
    analyzer = FacialAnalyzer()
    rows = []
    processing_times = []
    for record in records:
        image = cv2.imread(record["image"], cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"Cannot open ROI image: {record['image']}")
        image_height, image_width = image.shape[:2]
        x1, y1, x2, y2 = record["face_box"]
        if not (
            0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height
        ):
            raise ValueError(
                f"Face box for {record['sample']} is outside image bounds"
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        started = time.perf_counter()
        analysis = analyzer.analyze_face(gray, record["face_box"])
        feature_state = classify_face_features(
            analysis,
            record["face_box"],
            eye_open_threshold=edge_eye_threshold,
            mouth_open_threshold=edge_mouth_threshold,
            min_face_size=config.FEATURE_MIN_FACE_SIZE,
            eye_min_confidence=config.EYE_FEATURE_MIN_CONFIDENCE,
            mouth_min_confidence=config.MOUTH_FEATURE_MIN_CONFIDENCE,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        processing_times.append(elapsed_ms)
        rows.append(
            {
                "sample": record["sample"],
                "image": record["image"],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "eye_closed_gt": record["eye_closed"],
                "yawn_gt": record["yawn"],
                "edge_feature_eye_closed_pred": feature_state.eye_closed,
                "edge_feature_yawn_pred": feature_state.yawning,
                "eye_openness": float(analysis.eye_openness),
                "mouth_openness": float(analysis.mouth_openness),
                "eye_confidence": float(analysis.eye_confidence),
                "mouth_confidence": float(analysis.mouth_confidence),
                "analysis_valid": feature_state.analysis_valid,
                "eye_analysis_valid": feature_state.eye_valid,
                "mouth_analysis_valid": feature_state.mouth_valid,
                "processing_ms": elapsed_ms,
            }
        )

    if landmark_baseline_path:
        _attach_landmark_predictions(rows, baseline, "sample")
    report = _build_report(
        rows,
        {
            "roi_manifest": os.path.abspath(manifest_path),
            "evaluated_samples": len(rows),
            "source_images": len({record["image"] for record in records}),
            "edge_eye_threshold": edge_eye_threshold,
            "edge_mouth_threshold": edge_mouth_threshold,
            "face_detection_bypassed": True,
        },
        processing_times,
        landmark_baseline_path=landmark_baseline_path,
    )
    _write_report(rows, report, output_dir)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate edge/feature eye-closure and yawn detection."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Path to the evaluation video")
    source.add_argument(
        "--roi-manifest",
        help=(
            "CSV with sample,image,x1,y1,x2,y2,eye_closed,yawn columns; "
            "bypasses face detection"
        ),
    )
    parser.add_argument(
        "--labels",
        help="CSV with frame,eye_closed,yawn columns; required with --video",
    )
    parser.add_argument(
        "--landmark-baseline",
        help=(
            "Optional CSV with predictions or EAR/MAR, keyed by frame for video "
            "or sample for ROI evaluation"
        ),
    )
    parser.add_argument("--output", default="evaluation_results")
    parser.add_argument(
        "--edge-eye-threshold", type=float, default=Config.EYE_OPEN_THRESHOLD
    )
    parser.add_argument(
        "--edge-mouth-threshold", type=float, default=Config.MOUTH_OPEN_THRESHOLD
    )
    parser.add_argument("--landmark-eye-threshold", type=float, default=0.22)
    parser.add_argument("--landmark-mouth-threshold", type=float, default=0.30)
    args = parser.parse_args()

    common = {
        "output_dir": args.output,
        "landmark_baseline_path": args.landmark_baseline,
        "landmark_eye_threshold": args.landmark_eye_threshold,
        "landmark_mouth_threshold": args.landmark_mouth_threshold,
        "edge_eye_threshold": args.edge_eye_threshold,
        "edge_mouth_threshold": args.edge_mouth_threshold,
    }
    if args.video:
        if not args.labels:
            parser.error("--labels is required with --video")
        report = evaluate_on_video(args.video, args.labels, **common)
    else:
        if args.labels:
            parser.error("--labels is only valid with --video")
        report = evaluate_on_rois(args.roi_manifest, **common)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
