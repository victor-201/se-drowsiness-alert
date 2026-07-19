from dataclasses import dataclass
import logging
import threading

import cv2
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class RegionFeatures:
    name: str
    roi_box: tuple
    opening_ratio: float
    confidence: float
    dark_aspect: float
    darkness_score: float
    edge_aspect: float
    keypoint_aspect: float
    edges: np.ndarray
    dark_mask: np.ndarray
    keypoints: np.ndarray


@dataclass
class FaceFeatures:
    eye_openness: float
    mouth_openness: float
    eye_confidence: float
    mouth_confidence: float
    eye_candidate_count: int
    regions: dict


class FacialAnalyzer:
    """Analyze eye and mouth state without a facial-landmark model."""

    EYE_REGIONS = {
        "left_eye": (0.08, 0.24, 0.50, 0.50),
        "right_eye": (0.50, 0.24, 0.92, 0.50),
    }
    MOUTH_REGION = (0.18, 0.56, 0.82, 0.92)

    def __init__(self):
        self._analysis_lock = threading.RLock()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._eye_cascade_path = (
            cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        )
        self._eye_cascade = self._load_eye_cascade()

    def apply_clahe(self, gray_frame):
        with self._analysis_lock:
            if (
                not isinstance(gray_frame, np.ndarray)
                or gray_frame.size == 0
                or gray_frame.ndim != 2
            ):
                raise ValueError("CLAHE requires a non-empty grayscale image")
            return self._clahe.apply(np.ascontiguousarray(gray_frame))

    def _load_eye_cascade(self):
        cascade = cv2.CascadeClassifier(self._eye_cascade_path)
        if cascade.empty():
            logger.warning("Eye cascade is unavailable: %s", self._eye_cascade_path)
        return cascade

    @staticmethod
    def _clip_box(box, image_shape):
        try:
            values = np.asarray(box, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if values.size != 4 or not np.all(np.isfinite(values)):
            return None
        image_h, image_w = image_shape[:2]
        x1 = max(0, min(image_w, int(np.floor(values[0]))))
        y1 = max(0, min(image_h, int(np.floor(values[1]))))
        x2 = max(0, min(image_w, int(np.ceil(values[2]))))
        y2 = max(0, min(image_h, int(np.ceil(values[3]))))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _relative_box(face_box, relative_box, image_shape):
        image_h, image_w = image_shape[:2]
        x1, y1, x2, y2 = [int(value) for value in face_box]
        face_w = max(1, x2 - x1)
        face_h = max(1, y2 - y1)
        rx1, ry1, rx2, ry2 = relative_box
        box = (
            max(0, min(image_w, int(round(x1 + rx1 * face_w)))),
            max(0, min(image_h, int(round(y1 + ry1 * face_h)))),
            max(0, min(image_w, int(round(x1 + rx2 * face_w)))),
            max(0, min(image_h, int(round(y1 + ry2 * face_h)))),
        )
        return box

    @staticmethod
    def _auto_canny(gray):
        blurred = cv2.GaussianBlur(gray, (5, 5), 0.9)
        otsu_threshold, _ = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        high = int(np.clip(otsu_threshold, 45, 180))
        low = max(15, int(high * 0.45))
        return cv2.Canny(blurred, low, high), blurred

    @staticmethod
    def _analysis_mask(shape, region_type):
        height, width = shape
        mask = np.zeros((height, width), dtype=np.uint8)
        if region_type == "eye":
            y1, y2 = int(0.18 * height), int(0.95 * height)
            x1, x2 = int(0.05 * width), int(0.95 * width)
        else:
            y1, y2 = int(0.08 * height), int(0.95 * height)
            x1, x2 = int(0.04 * width), int(0.96 * width)
        mask[y1:y2, x1:x2] = 255
        return mask

    @staticmethod
    def _dark_component(mask, region_type):
        height, width = mask.shape
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0.0

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < height * width * 0.003:
                continue
            x, y, box_w, box_h = cv2.boundingRect(contour)
            if box_w < width * 0.07 or box_h < 2:
                continue

            center_x = (x + box_w / 2.0) / width
            center_y = (y + box_h / 2.0) / height
            if not 0.12 <= center_x <= 0.88:
                continue
            if region_type == "eye" and not 0.22 <= center_y <= 0.88:
                continue
            if region_type == "mouth" and not 0.18 <= center_y <= 0.90:
                continue

            center_weight = max(0.25, 1.0 - abs(center_x - 0.5))
            vertical_weight = 0.5 + min(1.0, box_h / max(1.0, height * 0.35))
            score = area * center_weight * vertical_weight
            if score > best_score:
                best_score = score
                best = (x, y, box_w, box_h, area)

        return best

    @staticmethod
    def _edge_aspect(edges, component):
        height, width = edges.shape
        if component is not None:
            x, y, box_w, box_h, _ = component
            margin_x = max(2, int(box_w * 0.20))
            x1 = max(0, x - margin_x)
            x2 = min(width, x + box_w + margin_x)
            y1 = max(0, y - max(2, int(box_h * 0.35)))
            y2 = min(height, y + box_h + max(2, int(box_h * 0.35)))
        else:
            x1, x2 = int(width * 0.12), int(width * 0.88)
            y1, y2 = int(height * 0.18), int(height * 0.92)

        spans = []
        for x_pos in range(x1, x2):
            rows = np.flatnonzero(edges[y1:y2, x_pos])
            if rows.size >= 2:
                spans.append(float(np.percentile(rows, 90) - np.percentile(rows, 10)))

        if not spans:
            return 0.0
        horizontal_span = max(1.0, float(x2 - x1))
        return float(np.median(spans) / horizontal_span)

    @staticmethod
    def _keypoint_features(gray, feature_mask, roi_box):
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=24,
            qualityLevel=0.025,
            minDistance=max(3, gray.shape[1] // 18),
            mask=feature_mask,
            blockSize=5,
            useHarrisDetector=False,
        )
        if points is None:
            return 0.0, np.empty((0, 2), dtype=np.int32)

        points = points.reshape(-1, 2)
        if len(points) < 2:
            return 0.0, np.empty((0, 2), dtype=np.int32)

        x_span = float(np.percentile(points[:, 0], 90) - np.percentile(points[:, 0], 10))
        y_span = float(np.percentile(points[:, 1], 90) - np.percentile(points[:, 1], 10))
        aspect = y_span / max(1.0, x_span)

        x1, y1, x2, y2 = roi_box
        scale_x = (x2 - x1) / gray.shape[1]
        scale_y = (y2 - y1) / gray.shape[0]
        absolute = np.column_stack(
            (
                np.round(x1 + points[:, 0] * scale_x),
                np.round(y1 + points[:, 1] * scale_y),
            )
        ).astype(np.int32)
        return float(aspect), absolute

    def _analyze_region(self, gray, roi_box, name, region_type):
        target_size = (120, 60) if region_type == "eye" else (140, 84)
        clipped_box = self._clip_box(roi_box, gray.shape)
        if clipped_box is None:
            empty = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
            return RegionFeatures(
                name,
                (0, 0, 0, 0),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                empty,
                empty.copy(),
                np.empty((0, 2), dtype=np.int32),
            )

        x1, y1, x2, y2 = clipped_box
        roi = np.ascontiguousarray(gray[y1:y2, x1:x2])
        if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
            empty = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
            return RegionFeatures(
                name,
                clipped_box,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                empty,
                empty.copy(),
                np.empty((0, 2), dtype=np.int32),
            )

        normalized = cv2.resize(roi, target_size, interpolation=cv2.INTER_LINEAR)
        normalized = self.apply_clahe(normalized)
        edges, blurred = self._auto_canny(normalized)
        feature_mask = self._analysis_mask(normalized.shape, region_type)
        edges = cv2.bitwise_and(edges, feature_mask)

        _, dark_mask = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
        dark_mask = cv2.bitwise_and(dark_mask, feature_mask)

        component = self._dark_component(dark_mask, region_type)
        dark_aspect = 0.0
        darkness_score = 0.0
        component_area_ratio = 0.0
        if component is not None:
            component_x, component_y, box_w, box_h, area = component
            dark_aspect = box_h / max(1.0, float(box_w))
            component_area_ratio = area / float(normalized.size)
            component_pixels = blurred[
                component_y:component_y + box_h,
                component_x:component_x + box_w,
            ]
            if component_pixels.size:
                component_mean = float(np.mean(component_pixels))
                darkness_score = float(np.clip((205.0 - component_mean) / 130.0, 0.10, 1.0))

        edge_aspect = self._edge_aspect(edges, component)
        keypoint_aspect, keypoints = self._keypoint_features(
            normalized, feature_mask, clipped_box
        )

        opening_ratio = (
            0.58 * min(dark_aspect, 1.2)
            + 0.27 * min(edge_aspect, 1.0)
            + 0.15 * min(keypoint_aspect, 1.0)
        )
        if region_type == "mouth":
            opening_ratio *= darkness_score
        opening_ratio = float(np.clip(opening_ratio, 0.0, 1.0))

        edge_density = np.count_nonzero(edges) / float(edges.size)
        point_confidence = min(1.0, len(keypoints) / 8.0)
        component_confidence = min(1.0, component_area_ratio / 0.025)
        confidence = float(
            np.clip(
                0.45 * component_confidence
                + 0.35 * point_confidence
                + 0.20 * min(1.0, edge_density / 0.08),
                0.0,
                1.0,
            )
        )

        return RegionFeatures(
            name=name,
            roi_box=clipped_box,
            opening_ratio=opening_ratio,
            confidence=confidence,
            dark_aspect=float(dark_aspect),
            darkness_score=float(darkness_score),
            edge_aspect=float(edge_aspect),
            keypoint_aspect=float(keypoint_aspect),
            edges=edges,
            dark_mask=dark_mask,
            keypoints=keypoints,
        )

    @staticmethod
    def _overlap_ratio(first, second):
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0.0
        first_area = max(1, (first[2] - first[0]) * (first[3] - first[1]))
        second_area = max(1, (second[2] - second[0]) * (second[3] - second[1]))
        return intersection / float(min(first_area, second_area))

    def _detect_eye_candidates(self, gray, face_box):
        if self._eye_cascade.empty():
            return []

        image_h, image_w = gray.shape[:2]
        clipped_face = self._clip_box(face_box, gray.shape)
        if clipped_face is None:
            return []
        x1, y1, x2, y2 = clipped_face
        face_w = max(1, x2 - x1)
        face_h = max(1, y2 - y1)
        upper_y2 = min(image_h, y1 + int(face_h * 0.62))
        upper_face = gray[max(0, y1):upper_y2, max(0, x1):min(image_w, x2)]
        if upper_face.size == 0:
            return []

        upper_face = self.apply_clahe(np.ascontiguousarray(upper_face))
        min_size = (max(10, int(face_w * 0.13)), max(8, int(face_h * 0.09)))
        try:
            detections = self._eye_cascade.detectMultiScale(
                upper_face,
                scaleFactor=1.10,
                minNeighbors=3,
                minSize=min_size,
            )
        except cv2.error as exc:
            logger.warning("Eye cascade failed; reloading it: %s", exc)
            self._eye_cascade = self._load_eye_cascade()
            return []

        candidates = []
        for ex, ey, ew, eh in detections:
            center_x = (ex + ew / 2.0) / face_w
            center_y = (ey + eh / 2.0) / max(1.0, upper_face.shape[0])
            if not 0.08 <= center_x <= 0.92 or not 0.20 <= center_y <= 0.92:
                continue

            margin_x = int(ew * 0.18)
            margin_y = int(eh * 0.12)
            box = (
                max(0, x1 + ex - margin_x),
                max(0, y1 + ey - margin_y),
                min(image_w, x1 + ex + ew + margin_x),
                min(image_h, y1 + ey + eh + margin_y),
            )
            candidates.append(box)

        candidates.sort(
            key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True
        )
        selected = []
        for candidate in candidates:
            if all(self._overlap_ratio(candidate, other) < 0.35 for other in selected):
                selected.append(candidate)
            if len(selected) == 2:
                break
        return sorted(selected, key=lambda box: box[0])

    def analyze_face(self, gray, face_box):
        with self._analysis_lock:
            if (
                not isinstance(gray, np.ndarray)
                or gray.size == 0
                or gray.ndim != 2
            ):
                return FaceFeatures(0.0, 0.0, 0.0, 0.0, 0, {})
            gray = np.ascontiguousarray(gray)
            clipped_face = self._clip_box(face_box, gray.shape)
            if clipped_face is None:
                return FaceFeatures(0.0, 0.0, 0.0, 0.0, 0, {})
            return self._analyze_face(gray, clipped_face)

    def _analyze_face(self, gray, face_box):
        regions = {}
        eye_values = []
        eye_confidences = []
        eye_candidates = self._detect_eye_candidates(gray, face_box)

        eye_boxes = []
        if eye_candidates:
            eye_boxes.extend(eye_candidates)
        for relative_box in self.EYE_REGIONS.values():
            if len(eye_boxes) >= 2:
                break
            fallback_box = self._relative_box(face_box, relative_box, gray.shape)
            if all(self._overlap_ratio(fallback_box, box) < 0.45 for box in eye_boxes):
                eye_boxes.append(fallback_box)
        eye_boxes = sorted(eye_boxes[:2], key=lambda box: box[0])

        for name, roi_box in zip(self.EYE_REGIONS, eye_boxes):
            region = self._analyze_region(gray, roi_box, name, "eye")
            regions[name] = region
            if region.confidence > 0.10:
                eye_values.append(region.opening_ratio)
                eye_confidences.append(region.confidence)

        mouth_box = self._relative_box(face_box, self.MOUTH_REGION, gray.shape)
        mouth = self._analyze_region(gray, mouth_box, "mouth", "mouth")
        regions["mouth"] = mouth

        geometry_openness = float(np.median(eye_values)) if eye_values else 0.0
        candidate_coverage = min(1.0, len(eye_candidates) / 2.0)
        eye_openness = geometry_openness * (0.35 + 0.65 * candidate_coverage)
        eye_confidence = float(np.mean(eye_confidences)) if eye_confidences else 0.0
        eye_confidence *= 0.55 + 0.45 * candidate_coverage

        return FaceFeatures(
            eye_openness=float(eye_openness),
            mouth_openness=mouth.opening_ratio,
            eye_confidence=float(eye_confidence),
            mouth_confidence=mouth.confidence,
            eye_candidate_count=len(eye_candidates),
            regions=regions,
        )

    @staticmethod
    def draw_features(frame, face_box, analysis):
        x1, y1, x2, y2 = [int(value) for value in face_box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 255), 2)

        colors = {
            "left_eye": (0, 255, 0),
            "right_eye": (0, 255, 0),
            "mouth": (255, 180, 0),
        }
        for name, region in analysis.regions.items():
            rx1, ry1, rx2, ry2 = region.roi_box
            color = colors[name]
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 1)
            for point in region.keypoints:
                cv2.circle(frame, tuple(point), 2, color, -1)
        return frame

    def reset_display(self):
        cv2.destroyAllWindows()
        logger.info("Reset display")
