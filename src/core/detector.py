import logging
import os
import threading
import time
from collections import deque

import cv2
import numpy as np

from src.configs.config import Config
from src.core.alert_system import AlertSystem
from src.core.facial_analyzer import FacialAnalyzer
from src.core.feature_classifier import classify_face_features
from src.core.model_manager import ModelManager


logger = logging.getLogger(__name__)


def manual_bgr_to_gray(bgr):
    b = bgr[:, :, 0].astype(np.float64)
    g = bgr[:, :, 1].astype(np.float64)
    r = bgr[:, :, 2].astype(np.float64)
    return np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0, 255).astype(np.uint8)


def manual_resize(img, new_w, new_h):
    old_h, old_w = img.shape[:2]
    if old_h == new_h and old_w == new_w:
        return img.copy()

    oy = np.arange(new_h, dtype=np.float64)
    ox = np.arange(new_w, dtype=np.float64)
    iy = (oy + 0.5) * old_h / new_h - 0.5
    ix = (ox + 0.5) * old_w / new_w - 0.5
    iy = np.clip(iy, 0, old_h - 1)
    ix = np.clip(ix, 0, old_w - 1)
    iy0 = np.floor(iy).astype(np.intp)
    iy1 = np.minimum(iy0 + 1, old_h - 1)
    ix0 = np.floor(ix).astype(np.intp)
    ix1 = np.minimum(ix0 + 1, old_w - 1)
    dy = (iy - iy0)[:, np.newaxis]
    dx = (ix - ix0)[np.newaxis, :]

    if img.ndim == 2:
        v00 = img[iy0[:, np.newaxis], ix0[np.newaxis, :]]
        v10 = img[iy1[:, np.newaxis], ix0[np.newaxis, :]]
        v01 = img[iy0[:, np.newaxis], ix1[np.newaxis, :]]
        v11 = img[iy1[:, np.newaxis], ix1[np.newaxis, :]]
        result = (
            (v00 * (1 - dx) + v01 * dx) * (1 - dy)
            + (v10 * (1 - dx) + v11 * dx) * dy
        )
        return np.clip(np.round(result), 0, 255).astype(img.dtype)

    channels = img.shape[2]
    result = np.zeros((new_h, new_w, channels), dtype=np.float64)
    for channel in range(channels):
        v00 = img[iy0[:, np.newaxis], ix0[np.newaxis, :], channel]
        v10 = img[iy1[:, np.newaxis], ix0[np.newaxis, :], channel]
        v01 = img[iy0[:, np.newaxis], ix1[np.newaxis, :], channel]
        v11 = img[iy1[:, np.newaxis], ix1[np.newaxis, :], channel]
        result[:, :, channel] = (
            (v00 * (1 - dx) + v01 * dx) * (1 - dy)
            + (v10 * (1 - dx) + v11 * dx) * dy
        )
    return np.clip(np.round(result), 0, 255).astype(img.dtype)


def non_max_suppression(boxes, scores, iou_threshold=0.4):
    if not boxes:
        return [], []

    boxes_array = np.asarray(boxes, dtype=np.float64)
    scores_array = np.asarray(scores, dtype=np.float64)
    x1 = boxes_array[:, 0]
    y1 = boxes_array[:, 1]
    x2 = boxes_array[:, 2]
    y2 = boxes_array[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores_array.argsort()[::-1]
    keep = []

    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        remaining = order[1:]
        xx1 = np.maximum(x1[current], x1[remaining])
        yy1 = np.maximum(y1[current], y1[remaining])
        xx2 = np.minimum(x2[current], x2[remaining])
        yy2 = np.minimum(y2[current], y2[remaining])
        intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[current] + areas[remaining] - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        order = remaining[iou <= iou_threshold]

    return [boxes_array[index] for index in keep], [scores_array[index] for index in keep]


class PipelineStage:
    def __init__(self, output_dir="pipeline_output"):
        self.output_dir = output_dir
        self.frame_count = 0
        os.makedirs(output_dir, exist_ok=True)

    def save_stage(self, name, image, frame_id=None):
        if frame_id is not None:
            self.frame_count = frame_id
        filename = f"frame{self.frame_count:04d}_{name}.png"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, image)
        return filepath


class DrowsinessDetector:
    def __init__(self, save_pipeline=False):
        self.config = Config()
        self.config.load_calibration()
        self.model_manager = ModelManager()
        self.analyzer = FacialAnalyzer()
        self.alert_system = AlertSystem()
        self.camera = None
        self._camera_lock = threading.RLock()
        self._processing_lock = threading.RLock()

        self.eye_open_threshold = self.config.EYE_OPEN_THRESHOLD
        self.eye_closed_consec_frames = self.config.EYE_CLOSED_CONSEC_FRAMES
        self.blink_consec_frames = self.config.BLINK_CONSEC_FRAMES
        self.mouth_open_threshold = self.config.MOUTH_OPEN_THRESHOLD
        self.yawn_consec_frames = self.config.YAWN_CONSEC_FRAMES
        self.no_face_alert_frames = self.config.NO_FACE_ALERT_FRAMES

        self.face_cascade = self._init_face_cascade()
        self.face_net_dnn = self._init_face_detector_dnn()

        self.no_face_counter = 0
        self.face_detected = False
        self.eye_counter = 0
        self.eye_transition_counter = 0
        self.eye_closed = False
        self.drowsiness_start_time = None

        self.blink_total = 0
        self.blink_times = deque(maxlen=100)
        self.blink_per_minute_threshold = self.config.BLINK_PER_MINUTE_THRESHOLD

        self.yawn_counter = 0
        self.yawn_total = 0
        self.yawn_times = deque(maxlen=100)
        self.yawn_per_minute_threshold = self.config.YAWN_PER_MINUTE_THRESHOLD
        self.mouth_open = False

        self.fatigue_alert_count = 0
        self.fatigue_start_time = None
        self.notification_duration = self.config.NOTIFICATION_DURATION
        self.last_reset_time = time.time()

        self.calibration_eye_values = []
        self.save_pipeline = save_pipeline
        self.pipeline = PipelineStage() if save_pipeline else None
        self._last_canny_left = None
        self._last_canny_right = None

    def _init_face_cascade(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            logger.error("Haar face cascade not found at %s", cascade_path)
            return None
        return cv2.CascadeClassifier(cascade_path)

    def _init_face_detector_dnn(self):
        proto = self.config.DNN_PROTOTXT
        model = self.config.DNN_CAFFEMODEL
        if not os.path.exists(proto) or not os.path.exists(model):
            try:
                self.model_manager.download_dnn_models()
            except Exception as exc:
                logger.warning("DNN face model is unavailable: %s", exc)
                return None
        try:
            return cv2.dnn.readNetFromCaffe(proto, model)
        except Exception as exc:
            logger.warning("DNN face detector initialization failed: %s", exc)
            return None

    def detect_faces_haar(self, gray):
        with self._processing_lock:
            if self.face_cascade is None or self.face_cascade.empty():
                return []
            detections = self.face_cascade.detectMultiScale(
                np.ascontiguousarray(gray),
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80),
            )
            return [
                [int(x), int(y), int(x + width), int(y + height)]
                for x, y, width, height in detections
            ]

    def detect_faces_dnn(self, frame):
        with self._processing_lock:
            if self.face_net_dnn is None:
                return [], []
            height, width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                1.0,
                (300, 300),
                (104.0, 177.0, 123.0),
            )
            self.face_net_dnn.setInput(blob)
            detections = self.face_net_dnn.forward()
            boxes = []
            scores = []

            for index in range(detections.shape[2]):
                confidence = float(detections[0, 0, index, 2])
                if confidence < self.config.DNN_CONFIDENCE_THRESHOLD:
                    continue
                box = detections[0, 0, index, 3:7] * np.array(
                    [width, height, width, height]
                )
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 - x1 >= 45 and y2 - y1 >= 45:
                    boxes.append([x1, y1, x2, y2])
                    scores.append(confidence)
            return boxes, scores

    def _detect_face_boxes(self, frame, gray_equalized):
        boxes = []
        scores = []
        try:
            haar_boxes = self.detect_faces_haar(gray_equalized)
            boxes.extend(haar_boxes)
            scores.extend([0.55] * len(haar_boxes))
        except Exception as exc:
            logger.warning("Haar face detection failed: %s", exc)
            self.face_cascade = self._init_face_cascade()

        try:
            dnn_boxes, dnn_scores = self.detect_faces_dnn(frame)
            boxes.extend(dnn_boxes)
            scores.extend(dnn_scores)
        except Exception as exc:
            logger.warning("DNN face detection failed: %s", exc)
            self.face_net_dnn = self._init_face_detector_dnn()

        if boxes:
            boxes, scores = non_max_suppression(
                boxes, scores, self.config.DNN_NMS_THRESHOLD
            )

        image_h = frame.shape[0]
        valid = []
        for box, score in zip(boxes, scores):
            clipped_box = self._clip_box(box, frame.shape)
            if clipped_box is None:
                continue
            x1, y1, x2, y2 = clipped_box
            width, height = x2 - x1, y2 - y1
            aspect = width / max(1.0, float(height))
            if width < 45 or height < 45 or not 0.45 <= aspect <= 1.8:
                continue
            if score <= 0.55 and y1 > image_h * 0.65:
                continue
            valid.append(([x1, y1, x2, y2], float(score)))
        return valid

    @staticmethod
    def _select_primary_face(detections):
        if not detections:
            return None
        return max(
            detections,
            key=lambda item: (
                (item[0][2] - item[0][0])
                * (item[0][3] - item[0][1])
                * max(0.5, item[1])
            ),
        )[0]

    def start_camera(self):
        with self._camera_lock:
            if self.camera and self.camera.isOpened():
                return
            backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF]
            for index in (self.config.CAMERA_ID, 1):
                for backend in backends:
                    try:
                        camera = cv2.VideoCapture(index, backend)
                        if camera.isOpened():
                            self.camera = camera
                            break
                        camera.release()
                    except Exception:
                        continue
                if self.camera and self.camera.isOpened():
                    break
            if not self.camera or not self.camera.isOpened():
                raise IOError("Cannot open camera")

            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
            self.camera.set(cv2.CAP_PROP_FPS, self.config.CAMERA_FPS)
            try:
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

    def stop_camera(self):
        with self._camera_lock:
            if self.camera and self.camera.isOpened():
                self.camera.release()
            self.camera = None

    def read_camera_frame(self):
        with self._camera_lock:
            if not self.camera or not self.camera.isOpened():
                return False, None
            return self.camera.read()

    def detect_blink(self, eye_openness):
        is_closed = eye_openness < self.eye_open_threshold
        if is_closed:
            self.eye_transition_counter += 1
            if self.eye_transition_counter >= self.blink_consec_frames:
                self.eye_closed = True
            return False

        blink_detected = self.eye_closed
        self.eye_closed = False
        self.eye_transition_counter = 0
        if blink_detected:
            now = time.time()
            self.blink_total += 1
            self.blink_times.append(now)
        return blink_detected

    def detect_yawn(self, mouth_openness):
        is_open = mouth_openness > self.mouth_open_threshold
        if is_open and not self.mouth_open:
            self.yawn_counter += 1
            if self.yawn_counter >= self.yawn_consec_frames:
                now = time.time()
                if not self.yawn_times or now - self.yawn_times[-1] >= 4.0:
                    self.mouth_open = True
                    self.yawn_counter = 0
                    self.yawn_total += 1
                    self.yawn_times.append(now)
                    return True
                self.yawn_counter = 0
        elif not is_open:
            self.mouth_open = False
            self.yawn_counter = 0
        return False

    def check_yawn_frequency(self):
        now = time.time()
        return sum(now - timestamp <= 60 for timestamp in self.yawn_times) >= (
            self.yawn_per_minute_threshold
        )

    def check_blink_frequency(self):
        now = time.time()
        return sum(now - timestamp <= 60 for timestamp in self.blink_times) >= (
            self.blink_per_minute_threshold
        )

    def reset_counters_if_needed(self):
        now = time.time()
        if now - self.last_reset_time < 60:
            return
        self.blink_times.clear()
        self.yawn_times.clear()
        self.blink_total = 0
        self.yawn_total = 0
        self.fatigue_alert_count = 0
        self.last_reset_time = now

    def process_frame(self):
        with self._processing_lock:
            self.reset_counters_if_needed()
            if not self.camera or not self.camera.isOpened():
                try:
                    self.start_camera()
                except Exception as exc:
                    logger.error("Failed to initialize camera: %s", exc)
                    return None, False, self._empty_metrics()

            success, frame = self.read_camera_frame()
            if not success or frame is None:
                return None, False, self._empty_metrics()
            frame = cv2.resize(
                frame, (self.config.CAMERA_WIDTH, self.config.CAMERA_HEIGHT)
            )
            return self._process_image(frame)

    def process_frame_from_frame(self, frame):
        with self._processing_lock:
            normalized = self._normalize_frame(frame)
            if normalized is None:
                logger.warning("Skipping invalid input frame")
                return None, False, self._empty_metrics()
            self.reset_counters_if_needed()
            return self._process_image(normalized)

    @staticmethod
    def _normalize_frame(frame):
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim != 3:
            return None
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.shape[2] != 3:
            return None
        if frame.shape[0] < 2 or frame.shape[1] < 2:
            return None
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)

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
    def _feature_montage(regions):
        tiles = []
        for name in ("left_eye", "right_eye", "mouth"):
            region = regions.get(name)
            if region is None:
                continue
            edges = cv2.resize(region.edges, (140, 84), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(
                region.dark_mask, (140, 84), interpolation=cv2.INTER_NEAREST
            )
            tiles.extend([edges, mask])
        if not tiles:
            return np.zeros((84, 140), dtype=np.uint8)
        return np.hstack(tiles)

    def _save_pipeline(self, stage_images):
        if not self.save_pipeline:
            return
        if self.pipeline is None:
            self.pipeline = PipelineStage()
        frame_id = int(time.time() * 1000) % 100000
        for name, image in stage_images.items():
            self.pipeline.save_stage(name, image, frame_id=frame_id)

    def _process_image(self, frame):
        frame = self._normalize_frame(frame)
        if frame is None:
            return None, False, self._empty_metrics()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_equalized = self.analyzer.apply_clahe(gray)
        stage_images = {
            "01_grayscale": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            "02_clahe": cv2.cvtColor(gray_equalized, cv2.COLOR_GRAY2BGR),
        }

        detections = self._detect_face_boxes(frame, gray_equalized)
        face_box = self._select_primary_face(detections)
        face_box = self._clip_box(face_box, frame.shape) if face_box else None
        if face_box is None:
            self.no_face_counter += 1
            self.face_detected = False
            stage_images["03_face_detection"] = frame.copy()
            if self.no_face_counter >= self.no_face_alert_frames:
                frame = self.alert_system.render_distraction_alert(frame)
                metrics = self._empty_metrics()
                metrics["distraction_detected"] = True
                self._save_pipeline(stage_images)
                return frame, True, metrics
            frame = self.alert_system.put_text_unicode(
                frame,
                "Không phát hiện khuôn mặt",
                (20, 30),
                self.config.ALERT_COLOR,
                font_size=24,
            )
            self._save_pipeline(stage_images)
            return frame, False, self._empty_metrics()

        self.no_face_counter = 0
        self.face_detected = True
        face_visualization = frame.copy()
        for detected_box, _ in detections:
            clipped_box = self._clip_box(detected_box, frame.shape)
            if clipped_box is None:
                continue
            dx1, dy1, dx2, dy2 = clipped_box
            cv2.rectangle(
                face_visualization, (dx1, dy1), (dx2, dy2), (0, 255, 0), 1
            )
        stage_images["03_face_detection"] = face_visualization

        analysis = self.analyzer.analyze_face(gray, face_box)
        feature_state = classify_face_features(
            analysis,
            face_box,
            eye_open_threshold=self.eye_open_threshold,
            mouth_open_threshold=self.mouth_open_threshold,
            min_face_size=self.config.FEATURE_MIN_FACE_SIZE,
            eye_min_confidence=self.config.EYE_FEATURE_MIN_CONFIDENCE,
            mouth_min_confidence=self.config.MOUTH_FEATURE_MIN_CONFIDENCE,
        )

        eye_openness = analysis.eye_openness
        mouth_openness = analysis.mouth_openness
        if feature_state.eye_valid:
            self.detect_blink(eye_openness)
            if feature_state.eye_closed:
                self.eye_counter += 1
                if self.eye_counter == 1:
                    self.drowsiness_start_time = time.time()
            else:
                self.eye_counter = 0
                self.drowsiness_start_time = None
        else:
            self.eye_counter = max(0, self.eye_counter - 1)

        if feature_state.mouth_valid:
            self.detect_yawn(mouth_openness)

        drowsiness_detected = self.eye_counter >= self.eye_closed_consec_frames
        fatigue_detected = self.check_blink_frequency() or self.check_yawn_frequency()

        if drowsiness_detected:
            duration = (
                time.time() - self.drowsiness_start_time
                if self.drowsiness_start_time is not None
                else 0.0
            )
            frame = self.alert_system.render_drowsiness_alert(frame, duration)
        if fatigue_detected and self.fatigue_alert_count < 1:
            if self.fatigue_start_time is None:
                self.fatigue_start_time = time.time()
            frame = self.alert_system.render_fatigue_alert(frame)
            if time.time() - self.fatigue_start_time >= self.notification_duration:
                self.fatigue_alert_count += 1
                self.fatigue_start_time = None

        self.analyzer.draw_features(frame, face_box, analysis)
        stage_images["04_edge_feature_rois"] = frame.copy()
        feature_montage = self._feature_montage(analysis.regions)
        stage_images["05_edges_and_masks"] = cv2.cvtColor(
            feature_montage, cv2.COLOR_GRAY2BGR
        )
        stage_images["06_result"] = frame.copy()
        self._save_pipeline(stage_images)

        left_eye = analysis.regions.get("left_eye")
        right_eye = analysis.regions.get("right_eye")
        self._last_canny_left = left_eye.edges if left_eye is not None else None
        self._last_canny_right = right_eye.edges if right_eye is not None else None

        metrics = {
            "eye_openness": eye_openness,
            "mouth_openness": mouth_openness,
            "eye_confidence": analysis.eye_confidence,
            "mouth_confidence": analysis.mouth_confidence,
            "eye_candidate_count": analysis.eye_candidate_count,
            "blink_count": self.blink_total,
            "yawn_count": self.yawn_total,
            "face_detected": True,
            "analysis_valid": feature_state.analysis_valid,
            "eye_analysis_valid": feature_state.eye_valid,
            "mouth_analysis_valid": feature_state.mouth_valid,
            "eye_closed": feature_state.eye_closed,
            "yawning": feature_state.yawning,
            "fatigue_detected": fatigue_detected,
            "drowsiness_detected": drowsiness_detected,
            "distraction_detected": False,
            "eye_counter": self.eye_counter,
        }
        return frame, drowsiness_detected or fatigue_detected, metrics

    def reset_calibration(self):
        self.calibration_eye_values = []

    def process_calibration_frame(self):
        with self._processing_lock:
            if not self.camera or not self.camera.isOpened():
                return None, 0.0
            success, frame = self.read_camera_frame()
            if not success or frame is None:
                return None, 0.0
            frame = cv2.resize(
                frame, (self.config.CAMERA_WIDTH, self.config.CAMERA_HEIGHT)
            )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_equalized = self.analyzer.apply_clahe(gray)
            face_box = self._select_primary_face(
                self._detect_face_boxes(frame, gray_equalized)
            )
            eye_openness = 0.0
            if face_box is not None:
                analysis = self.analyzer.analyze_face(gray, face_box)
                if (
                    analysis.eye_confidence
                    >= self.config.EYE_FEATURE_MIN_CONFIDENCE
                ):
                    eye_openness = analysis.eye_openness
                    self.calibration_eye_values.append(eye_openness)
                self.analyzer.draw_features(frame, face_box, analysis)
            return frame, eye_openness

    def finalize_calibration(self):
        if not self.calibration_eye_values:
            return False, self.eye_open_threshold
        normal_eye_openness = float(np.median(self.calibration_eye_values))
        new_threshold = float(np.clip(normal_eye_openness * 0.90, 0.03, 0.60))
        self.eye_open_threshold = new_threshold
        self.config.save_calibration(new_threshold)
        self.calibration_eye_values = []
        return True, new_threshold

    def _empty_metrics(self):
        return {
            "eye_openness": 0.0,
            "mouth_openness": 0.0,
            "eye_confidence": 0.0,
            "mouth_confidence": 0.0,
            "eye_candidate_count": 0,
            "blink_count": self.blink_total,
            "yawn_count": self.yawn_total,
            "face_detected": False,
            "analysis_valid": False,
            "eye_analysis_valid": False,
            "mouth_analysis_valid": False,
            "eye_closed": False,
            "yawning": False,
            "fatigue_detected": False,
            "drowsiness_detected": False,
            "distraction_detected": False,
            "eye_counter": self.eye_counter,
        }

    def __del__(self):
        self.stop_camera()
