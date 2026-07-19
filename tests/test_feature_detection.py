import os
import threading
import time
import unittest

import cv2
import numpy as np

from src.core.detector import DrowsinessDetector
from src.core.facial_analyzer import FacialAnalyzer


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def synthetic_face(eyes_open=True, mouth_open=False, brightness=210):
    image = np.full((240, 240), brightness, dtype=np.uint8)
    skin = max(80, brightness - 45)
    dark = max(5, brightness - 185)
    cv2.ellipse(image, (120, 120), (96, 112), 0, 0, 360, skin, -1)

    for center_x in (70, 170):
        if eyes_open:
            cv2.ellipse(image, (center_x, 91), (31, 12), 0, 0, 360, dark, 2)
            cv2.circle(image, (center_x, 91), 7, dark, -1)
            cv2.circle(image, (center_x - 2, 89), 2, min(255, brightness + 15), -1)
        else:
            cv2.line(
                image,
                (center_x - 30, 92),
                (center_x + 30, 92),
                dark,
                4,
            )

    if mouth_open:
        cv2.ellipse(image, (120, 174), (37, 28), 0, 0, 360, dark, -1)
        cv2.ellipse(
            image, (120, 174), (38, 29), 0, 0, 360, min(100, dark + 45), 2
        )
    else:
        cv2.line(image, (82, 174), (158, 174), dark, 4)
    return image


class FacialAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = FacialAnalyzer()
        self.face_box = (0, 0, 240, 240)

    def analyze(self, eyes_open=True, mouth_open=False, brightness=210):
        return self.analyzer.analyze_face(
            synthetic_face(eyes_open, mouth_open, brightness),
            self.face_box,
        )

    def test_eye_closure_separates_open_and_closed_eyes(self):
        for brightness in (165, 210, 235):
            with self.subTest(brightness=brightness):
                open_features = self.analyze(True, False, brightness)
                closed_features = self.analyze(False, False, brightness)
                self.assertGreater(
                    open_features.eye_openness,
                    closed_features.eye_openness + 0.06,
                )
                self.assertGreater(open_features.eye_confidence, 0.25)
                self.assertGreater(closed_features.eye_confidence, 0.25)

    def test_yawn_separates_open_and_closed_mouth(self):
        for brightness in (165, 210, 235):
            with self.subTest(brightness=brightness):
                normal_features = self.analyze(True, False, brightness)
                yawn_features = self.analyze(True, True, brightness)
                self.assertGreater(
                    yawn_features.mouth_openness,
                    normal_features.mouth_openness + 0.25,
                )
                self.assertGreater(yawn_features.mouth_openness, 0.35)
                self.assertLess(normal_features.mouth_openness, 0.35)

    def test_real_sample_open_driver_and_closed_passenger(self):
        image_path = os.path.join(PROJECT_ROOT, "notebooks", "test.png")
        image = cv2.imread(image_path)
        self.assertIsNotNone(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        driver = self.analyzer.analyze_face(gray, (981, 214, 1098, 366))
        passenger = self.analyzer.analyze_face(gray, (449, 272, 539, 377))

        self.assertGreaterEqual(driver.eye_openness, 0.20)
        self.assertLess(passenger.eye_openness, 0.20)
        self.assertLess(driver.mouth_openness, 0.35)
        self.assertLess(passenger.mouth_openness, 0.35)

    def test_out_of_bounds_face_box_is_clipped_before_roi_analysis(self):
        analysis = self.analyzer.analyze_face(
            synthetic_face(),
            (-80, -40, 320, 300),
        )

        self.assertEqual(set(analysis.regions), {"left_eye", "right_eye", "mouth"})
        for region in analysis.regions.values():
            x1, y1, x2, y2 = region.roi_box
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, 240)
            self.assertLessEqual(y2, 240)
            self.assertGreater(x2, x1)
            self.assertGreater(y2, y1)

    def test_invalid_face_box_returns_empty_analysis(self):
        analysis = self.analyzer.analyze_face(
            synthetic_face(),
            (220, 220, 10, 10),
        )

        self.assertEqual(analysis.eye_openness, 0.0)
        self.assertEqual(analysis.mouth_openness, 0.0)
        self.assertEqual(analysis.regions, {})


class DetectorStateTests(unittest.TestCase):
    @staticmethod
    def detector_with_fixed_face():
        detector = DrowsinessDetector()
        detector._detect_face_boxes = lambda frame, gray: [
            ([0, 0, frame.shape[1], frame.shape[0]], 1.0)
        ]
        return detector

    def test_consecutive_closed_eye_frames_trigger_drowsiness(self):
        detector = self.detector_with_fixed_face()
        analyzer = detector.analyzer
        face_box = (0, 0, 240, 240)
        open_gray = synthetic_face(True, False)
        closed_gray = synthetic_face(False, False)
        open_score = analyzer.analyze_face(open_gray, face_box).eye_openness
        closed_score = analyzer.analyze_face(closed_gray, face_box).eye_openness
        detector.eye_open_threshold = (open_score + closed_score) / 2.0
        detector.eye_closed_consec_frames = 3

        closed_frame = cv2.cvtColor(closed_gray, cv2.COLOR_GRAY2BGR)
        for _ in range(2):
            _, _, metrics = detector.process_frame_from_frame(closed_frame.copy())
            self.assertFalse(metrics["drowsiness_detected"])
        _, alert, metrics = detector.process_frame_from_frame(closed_frame.copy())

        self.assertTrue(alert)
        self.assertTrue(metrics["eye_closed"])
        self.assertTrue(metrics["drowsiness_detected"])

    def test_consecutive_open_mouth_frames_count_one_yawn(self):
        detector = self.detector_with_fixed_face()
        analyzer = detector.analyzer
        face_box = (0, 0, 240, 240)
        normal_gray = synthetic_face(True, False)
        yawn_gray = synthetic_face(True, True)
        normal_score = analyzer.analyze_face(normal_gray, face_box).mouth_openness
        yawn_score = analyzer.analyze_face(yawn_gray, face_box).mouth_openness
        detector.mouth_open_threshold = (normal_score + yawn_score) / 2.0
        detector.yawn_consec_frames = 3

        yawn_frame = cv2.cvtColor(yawn_gray, cv2.COLOR_GRAY2BGR)
        for _ in range(3):
            _, _, metrics = detector.process_frame_from_frame(yawn_frame.copy())

        self.assertTrue(metrics["yawning"])
        self.assertEqual(metrics["yawn_count"], 1)

    def test_outside_detector_box_is_clipped_before_analysis(self):
        detector = DrowsinessDetector()
        detector._detect_face_boxes = lambda frame, gray: [
            ([-50, -30, frame.shape[1] + 40, frame.shape[0] + 20], 1.0)
        ]
        frame = cv2.cvtColor(synthetic_face(), cv2.COLOR_GRAY2BGR)

        _, _, metrics = detector.process_frame_from_frame(frame)

        self.assertTrue(metrics["face_detected"])
        self.assertTrue(metrics["analysis_valid"])

    def test_invalid_input_frame_is_skipped(self):
        detector = DrowsinessDetector()

        result_frame, alert, metrics = detector.process_frame_from_frame(
            np.empty((0, 0, 3), dtype=np.uint8)
        )

        self.assertIsNone(result_frame)
        self.assertFalse(alert)
        self.assertFalse(metrics["face_detected"])

    def test_concurrent_processing_is_serialized(self):
        detector = DrowsinessDetector()
        frame = cv2.cvtColor(synthetic_face(), cv2.COLOR_GRAY2BGR)
        counter_lock = threading.Lock()
        barrier = threading.Barrier(6)
        active = 0
        max_active = 0
        results = []

        def fake_process(normalized_frame):
            nonlocal active, max_active
            with counter_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with counter_lock:
                active -= 1
            return normalized_frame, False, detector._empty_metrics()

        detector._process_image = fake_process

        def worker():
            barrier.wait()
            results.append(detector.process_frame_from_frame(frame.copy()))

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)

        self.assertEqual(len(results), 6)
        self.assertEqual(max_active, 1)


if __name__ == "__main__":
    unittest.main()
