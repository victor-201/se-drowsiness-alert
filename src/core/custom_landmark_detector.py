import cv2
import numpy as np
import logging
from math import cos, sin, pi, sqrt
from src.core.feature_analyzers import EyeStateAnalyzer, YawnDetector

logger = logging.getLogger(__name__)


class CustomLandmarkDetector:
    """
    Custom facial landmark detector using Canny edge detection,
    contour analysis, and Haar cascades (no pre-trained model).

    Outputs (68, 2) numpy array matching dlib's shape_predictor_68 format.
    """

    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml")
        self.eye_analyzer = EyeStateAnalyzer()
        self.yawn_detector = YawnDetector()
        self._gray = None
        self.last_eye_states = {}
        self.last_yawn_state = None
        logger.info("CustomLandmarkDetector initialized (Haar + Canny + Contour + FeatureAnalyzer)")

    def detect_landmarks(self, gray_eq, face_box):
        self._gray = gray_eq
        x1, y1, x2, y2 = [int(v) for v in face_box]
        fw, fh = x2 - x1, y2 - y1
        if fw < 40 or fh < 40:
            return None

        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        s = np.zeros((68, 2), dtype=np.float32)

        re_center, le_center = self._detect_eye_centers(x1, y1, x2, y2)

        re_state = self.eye_analyzer.analyze(gray_eq, re_center, face_box, "right")
        le_state = self.eye_analyzer.analyze(gray_eq, le_center, face_box, "left")
        self.last_eye_states = {'right': re_state, 'left': le_state}

        if re_state['landmarks'] is not None:
            s[36:42] = re_state['landmarks']
        else:
            self._default_eye(s, 36, re_center[0], re_center[1], fw * 0.08, fh * 0.03)

        if le_state['landmarks'] is not None:
            s[42:48] = le_state['landmarks']
        else:
            self._default_eye(s, 42, le_center[0], le_center[1], fw * 0.08, fh * 0.03)

        yawn_result = self.yawn_detector.analyze(gray_eq, face_box)
        self.last_yawn_state = yawn_result

        if yawn_result['outer_landmarks'] is not None:
            s[48:60] = yawn_result['outer_landmarks']
        else:
            self._default_mouth(s, cx, y1 + fh * 0.76, fw, fh)

        if yawn_result['inner_landmarks'] is not None:
            s[60:68] = yawn_result['inner_landmarks']

        self._fit_nose_landmarks(s, re_center, le_center, x1, y1, x2, y2)
        self._fit_eyebrow_landmarks(s, re_center, le_center)
        self._fit_jawline(s, x1, y1, x2, y2)

        return s

    def _detect_eye_centers(self, x1, y1, x2, y2):
        """Detect eye centers using dark-band projection + Haar fallback."""
        fw, fh = x2 - x1, y2 - y1
        eye_y_default = y1 + fh * 0.35
        le_cx_est = x1 + fw * 0.30
        re_cx_est = x1 + fw * 0.70

        # --- Method 1: Dark-band horizontal projection ---
        # Eyes are in the middle-upper band, skip forehead (top 20%)
        er_y1 = y1 + int(fh * 0.18)
        er_y2 = y1 + int(fh * 0.50)
        eye_region = self._gray[er_y1:er_y2, x1:x2]

        re_center = np.array([re_cx_est, eye_y_default])
        le_center = np.array([le_cx_est, eye_y_default])

        if eye_region.size == 0 or eye_region.shape[0] < 15 or eye_region.shape[1] < 15:
            return re_center, le_center

        # Horizontal projection: sum pixel intensities per row
        h_proj = np.mean(eye_region, axis=1)
        # Eyes are darker than forehead and cheeks, find the darkest band
        window = 5
        if len(h_proj) > window:
            rolled = np.convolve(h_proj, np.ones(window), mode='valid')
            darkest_row = np.argmin(rolled) + window // 2
            eye_y = er_y1 + darkest_row
        else:
            eye_y = eye_y_default

        # Within the darkest band, find left/right dark clusters
        band_y1 = max(0, darkest_row - 3) if 'darkest_row' in dir() else 0
        band_y2 = min(eye_region.shape[0], darkest_row + 3) if 'darkest_row' in dir() else eye_region.shape[0]
        band = eye_region[band_y1:band_y2, :]
        if band.size > 0:
            v_proj = np.mean(band, axis=0)
            # Find two darkest columns (left and right eye)
            half_w = len(v_proj) // 2
            left_half = v_proj[:half_w]
            right_half = v_proj[half_w:]
            if len(left_half) > 3 and len(right_half) > 3:
                le_x = x1 + np.argmin(left_half)
                re_x = x1 + half_w + np.argmin(right_half)
                le_center = np.array([float(le_x), float(eye_y)])
                re_center = np.array([float(re_x), float(eye_y)])
                return re_center, le_center

        # --- Method 2: Haar cascade fallback ---
        eyes = self.eye_cascade.detectMultiScale(
            eye_region, scaleFactor=1.03, minNeighbors=2, minSize=(14, 10)
        )

        if len(eyes) >= 2:
            eyes_s = sorted(eyes, key=lambda e: e[0])
            ex0, ey0, ew0, eh0 = eyes_s[0]
            ex1, ey1_r, ew1, eh1 = eyes_s[1]
            re_center = np.array([x1 + ex0 + ew0 / 2.0, er_y1 + ey0 + eh0 / 2.0])
            le_center = np.array([x1 + ex1 + ew1 / 2.0, er_y1 + ey1_r + eh1 / 2.0])
        elif len(eyes) == 1:
            ex, ey, ew, eh = eyes[0]
            ecx = x1 + ex + ew / 2.0
            ecy = er_y1 + ey + eh / 2.0
            fcx = (x1 + x2) / 2.0
            if ecx < fcx:
                re_center = np.array([ecx, ecy])
                le_center = np.array([2 * fcx - ecx, ecy])
            else:
                le_center = np.array([ecx, ecy])
                re_center = np.array([2 * fcx - ecx, ecy])

        return re_center, le_center

    def _default_eye(self, s, base, ecx, ecy, ew, eh):
        angles = [pi, 2 * pi / 3, pi / 3, 0, -pi / 3, -2 * pi / 3]
        for i, a in enumerate(angles):
            s[base + i] = [ecx + ew * cos(a), ecy + eh * sin(a)]

    def _default_mouth(self, s, cx, cy, fw, fh):
        mw = fw * 0.18
        mh = fh * 0.035
        for i in range(12):
            a = i * 2 * pi / 12
            s[48 + i] = [cx + mw * cos(a), cy + mh * sin(a)]
        iw, ih = mw * 0.55, mh * 0.45
        for i in range(8):
            a = i * 2 * pi / 8
            s[60 + i] = [cx + iw * cos(a), cy + ih * sin(a)]

    def _fit_nose_landmarks(self, s, re_center, le_center, x1, y1, x2, y2):
        fw, fh = x2 - x1, y2 - y1
        cx = (x1 + x2) / 2.0
        eye_y = (re_center[1] + le_center[1]) / 2.0
        eye_spacing = abs(re_center[0] - le_center[0])
        nose_tip_y = y1 + fh * 0.62
        nose_w = eye_spacing * 0.22

        for i in range(4):
            t = i / 3.0
            s[27 + i] = [cx, eye_y + t * (nose_tip_y - eye_y)]

        for i in range(5):
            t = (i - 2) / 2.0
            s[31 + i] = [cx + t * nose_w, nose_tip_y + abs(t) * fh * 0.015]

    def _fit_eyebrow_landmarks(self, s, re_center, le_center):
        eye_spacing = abs(re_center[0] - le_center[0])
        for idx, (ecx, ecy) in enumerate([(re_center[0], re_center[1]),
                                          (le_center[0], le_center[1])]):
            base = 17 if idx == 0 else 22
            brow_y = ecy - eye_spacing * 0.18
            bw = eye_spacing * 0.18
            for i in range(5):
                t = (i - 2) / 2.0
                s[base + i] = [ecx + t * bw, brow_y - abs(t) * bw * 0.15]

    def _fit_jawline(self, s, x1, y1, x2, y2):
        fw, fh = x2 - x1, y2 - y1
        cx = (x1 + x2) / 2.0
        for i in range(17):
            t = (i - 8) / 8.0
            curve = (1 - t ** 2) * 0.3
            jw = fw * 0.5 * (1 - curve)
            jx = cx + t * jw
            jy = y2 - fh * (0.05 + 0.08 * (1 - abs(t)))
            s[i] = [jx, jy]
