import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import logging
import time
from src.core.detector import DrowsinessDetector, manual_resize
from src.configs.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('drowsiness_detector.log')
    ]
)
logger = logging.getLogger(__name__)


def draw_metrics_overlay(frame, metrics, config):
    if not frame.flags['C_CONTIGUOUS']:
        frame = np.ascontiguousarray(frame)
    overlay_data = [
        (f"EAR: {metrics['ear']:.2f}", (20, 30), (0, 255, 0) if metrics['ear'] >= config.EAR_THRESHOLD else (0, 0, 255)),
        (f"MAR: {metrics['mar']:.2f}", (20, 55), (0, 255, 0) if metrics['mar'] <= config.YAWN_THRESHOLD else (0, 0, 255)),
        (f"Roll: {metrics['roll_angle']:.1f} | Pitch: {metrics['pitch_angle']:.1f}", (20, 80), (255, 255, 255)),
        (f"Blink: {metrics['blink_count']}  Yawn: {metrics['yawn_count']}", (20, 105), (255, 255, 255)),
        (f"Face: {'Yes' if metrics['face_detected'] else 'No'}", (20, 130),
         (0, 255, 0) if metrics['face_detected'] else (0, 0, 255)),
    ]
    for text, pos, color in overlay_data:
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def show_pipeline_windows(frame, metrics, detector):
    small = manual_resize(frame, 320, 240)
    cv2.imshow("Camera", small)

    if detector._last_canny_left is not None and detector._last_canny_right is not None:
        if detector._last_canny_left.size > 0 and detector._last_canny_right.size > 0:
            h_l, w_l = detector._last_canny_left.shape
            h_r, w_r = detector._last_canny_right.shape
            max_h = max(h_l, h_r)
            combined = np.zeros((max_h, w_l + w_r), dtype=np.uint8)
            combined[:h_l, :w_l] = detector._last_canny_left
            combined[:h_r, w_l:w_l + w_r] = detector._last_canny_right
            canny_big = manual_resize(combined, 320, 120)
            cv2.imshow("Canny Edges", canny_big)


def run_detection(save_pipeline=False):
    detector = DrowsinessDetector(save_pipeline=save_pipeline)
    config = Config()

    try:
        detector.start_camera()
    except Exception as e:
        logger.error(f"Failed to initialize camera: {e}")
        return

    logger.info("Press 'q' to quit, 'p' to toggle pipeline saving")

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera", 640, 480)

    while True:
        frame, alert, metrics = detector.process_frame()
        if frame is None:
            time.sleep(0.03)
            continue

        draw_metrics_overlay(frame, metrics, config)
        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            detector.save_pipeline = not detector.save_pipeline
            logger.info(f"Pipeline saving: {'ON' if detector.save_pipeline else 'OFF'}")

    detector.stop_camera()
    cv2.destroyAllWindows()


def run_calibration():
    config = Config()
    detector = DrowsinessDetector()
    try:
        detector.start_camera()
    except Exception as e:
        logger.error(f"Failed to initialize camera: {e}")
        return

    logger.info(f"Look straight at the camera for {config.CALIBRATION_DURATION} seconds for calibration...")
    logger.info("Press 'q' to skip calibration")

    detector.reset_calibration()
    start_time = time.time()
    duration = config.CALIBRATION_DURATION

    while time.time() - start_time < duration:
        frame, ear = detector.process_calibration_frame()
        if frame is None:
            continue

        remaining = duration - (time.time() - start_time)
        cv2.putText(frame, f"Calibration: {remaining:.0f}s", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Calibration", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    success, new_threshold = detector.finalize_calibration()
    if success:
        logger.info(f"Calibration complete. New EAR threshold: {new_threshold:.3f}")
    else:
        logger.warning("Calibration failed, using default threshold")

    detector.stop_camera()
    cv2.destroyAllWindows()


def run_comparison():
    import dlib
    from src.core.model_manager import ModelManager
    from src.core.facial_analyzer import FacialAnalyzer
    from src.core.custom_landmark_detector import CustomLandmarkDetector

    config = Config()
    model_manager = ModelManager()
    dlib_predictor = model_manager.predictor
    dlib_detector = model_manager.detector
    custom_detector = CustomLandmarkDetector()
    analyzer = FacialAnalyzer()
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Cannot open camera for comparison")
            return
    except Exception as e:
        logger.error(f"Camera init failed: {e}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    logger.info("Comparison mode: LEFT=Custom (Canny+Contour), RIGHT=dlib 68-landmark")
    logger.info("Press 'q' to quit")

    cv2.namedWindow("Comparison", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        frame = cv2.resize(frame, (config.CAMERA_WIDTH, config.CAMERA_HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = analyzer.apply_clahe(gray)

        faces = cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            cv2.putText(frame, "No face", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Comparison", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        x, y, w, h = faces[0]
        face_box = [int(x), int(y), int(x + w), int(y + h)]

        custom_frame = frame.copy()
        dlib_frame = frame.copy()

        custom_shape = custom_detector.detect_landmarks(gray_eq, face_box)
        if custom_shape is not None:
            for i in range(68):
                cv2.circle(custom_frame, tuple(custom_shape[i].astype(int)), 1, (0, 255, 0), -1)
            cv2.polylines(custom_frame, [custom_shape[36:42].astype(int)], True, (0, 255, 0), 1)
            cv2.polylines(custom_frame, [custom_shape[42:48].astype(int)], True, (0, 255, 0), 1)
            cv2.polylines(custom_frame, [custom_shape[48:60].astype(int)], True, (0, 255, 0), 1)
            cv2.polylines(custom_frame, [custom_shape[27:36].astype(int)], True, (0, 255, 0), 1)

            c_left_ear = analyzer.calculate_ear(custom_shape[36:42])
            c_right_ear = analyzer.calculate_ear(custom_shape[42:48])
            c_ear = (c_left_ear + c_right_ear) / 2.0
            c_mar = analyzer.calculate_mar(custom_shape[48:68])
            cv2.putText(custom_frame, f"EAR: {c_ear:.3f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(custom_frame, f"MAR: {c_mar:.3f}", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(custom_frame, "CUSTOM (Canny+Contour)", (10, custom_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        dlib_face = dlib.rectangle(int(face_box[0]), int(face_box[1]), int(face_box[2]), int(face_box[3]))
        dlib_faces = dlib_detector(gray_eq)
        if dlib_faces:
            shape = dlib_predictor(gray_eq, dlib_faces[0])
            dlib_shape = np.array([[p.x, p.y] for p in shape.parts()])
            for i in range(68):
                cv2.circle(dlib_frame, tuple(dlib_shape[i]), 1, (0, 200, 255), -1)
            cv2.polylines(dlib_frame, [dlib_shape[36:42]], True, (0, 200, 255), 1)
            cv2.polylines(dlib_frame, [dlib_shape[42:48]], True, (0, 200, 255), 1)
            cv2.polylines(dlib_frame, [dlib_shape[48:60]], True, (0, 200, 255), 1)
            cv2.polylines(dlib_frame, [dlib_shape[27:36]], True, (0, 200, 255), 1)

            d_left_ear = analyzer.calculate_ear(dlib_shape[36:42])
            d_right_ear = analyzer.calculate_ear(dlib_shape[42:48])
            d_ear = (d_left_ear + d_right_ear) / 2.0
            d_mar = analyzer.calculate_mar(dlib_shape[48:68])
            cv2.putText(dlib_frame, f"EAR: {d_ear:.3f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            cv2.putText(dlib_frame, f"MAR: {d_mar:.3f}", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            cv2.putText(dlib_frame, "DLIB 68-landmark", (10, dlib_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

            if custom_shape is not None:
                diff = np.mean(np.abs(dlib_shape - custom_shape))
                cv2.putText(dlib_frame, f"Mean diff: {diff:.1f}px", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        combined = np.hstack([custom_frame, dlib_frame])
        cv2.imshow("Comparison", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_kivy():
    from src.ui.app import DrowsinessDetectorApp
    DrowsinessDetectorApp().run()


if __name__ == '__main__':
    import sys
    if '--calibrate' in sys.argv:
        run_calibration()
    elif '--compare' in sys.argv:
        run_comparison()
    elif '--opencv' in sys.argv:
        run_detection(save_pipeline='--save-pipeline' in sys.argv)
    else:
        run_kivy()
