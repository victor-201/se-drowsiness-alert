import os
import pickle
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Config:
    FEATURE_PIPELINE_VERSION = 1
    EYE_OPEN_THRESHOLD = 0.20
    EYE_CLOSED_CONSEC_FRAMES = 15
    EYE_FEATURE_MIN_CONFIDENCE = 0.25
    FEATURE_MIN_FACE_SIZE = 80
    BLINK_CONSEC_FRAMES = 3
    BLINK_PER_MINUTE_THRESHOLD = 25
    MOUTH_OPEN_THRESHOLD = 0.35
    MOUTH_FEATURE_MIN_CONFIDENCE = 0.25
    YAWN_CONSEC_FRAMES = 5
    YAWN_PER_MINUTE_THRESHOLD = 3
    NO_FACE_ALERT_FRAMES = 20
    NOTIFICATION_DURATION = 3.0

    CALIBRATION_DURATION = 5
    ALERT_COOLDOWN = 3
    ALERT_STOP_DELAY = 1.0
    CAMERA_ID = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    PRIMARY_COLOR = (0, 255, 0)
    SECONDARY_COLOR = (255, 165, 0)
    ALERT_COLOR = (0, 0, 255)
    TEXT_COLOR = (255, 255, 255)
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    CALIBRATION_FILE = os.path.join(DATA_DIR, "feature_calibration.pkl")
    ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
    IMAGE_DIR = os.path.join(ASSETS_DIR, "images")
    SOUND_DIR = os.path.join(ASSETS_DIR, "sounds")
    SOUND_ALERT_DIR = os.path.join(SOUND_DIR, "alerts")
    SOUND_NOTIFICATION_DIR = os.path.join(SOUND_DIR, "notifications")
    ALERT_SOUND_FILE = os.path.join(SOUND_ALERT_DIR, "alert.wav")
    FATIGUE_SOUND_FILE = os.path.join(SOUND_NOTIFICATION_DIR, "canh_bao_buon_ngu.mp3")
    FONT_DIR = os.path.join(ASSETS_DIR, "fonts")
    FONT_PATH = os.path.join(FONT_DIR, "ARIAL.TTF")
    DNN_CONFIDENCE_THRESHOLD = 0.5
    DNN_NMS_THRESHOLD = 0.4
    DNN_PROTOTXT = os.path.join(DATA_DIR, "deploy.prototxt")
    DNN_CAFFEMODEL = os.path.join(DATA_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

    def save_calibration(self, eye_open_threshold):
        try:
            os.makedirs(self.DATA_DIR, exist_ok=True)
            with open(self.CALIBRATION_FILE, 'wb') as f:
                pickle.dump(
                    {
                        'pipeline_version': self.FEATURE_PIPELINE_VERSION,
                        'eye_open_threshold': eye_open_threshold,
                    },
                    f,
                )
        except Exception as e:
            logger.error(f"Lưu hiệu chỉnh thất bại: {e}")

    def load_calibration(self):
        try:
            if os.path.exists(self.CALIBRATION_FILE):
                with open(self.CALIBRATION_FILE, 'rb') as f:
                    data = pickle.load(f)
                    if data.get('pipeline_version') != self.FEATURE_PIPELINE_VERSION:
                        return False
                    self.EYE_OPEN_THRESHOLD = data.get(
                        'eye_open_threshold', self.EYE_OPEN_THRESHOLD
                    )
                    return True
        except Exception as e:
            logger.error(f"Tải hiệu chỉnh thất bại: {e}")
        return False
