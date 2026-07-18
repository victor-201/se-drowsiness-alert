import cv2
import numpy as np
import logging
import time
from collections import deque
from math import cos, sin, pi

logger = logging.getLogger(__name__)


class TemporalSmoother:
    def __init__(self, window_size=7, ema_alpha=0.25, dead_zone=0.05):
        self.window = deque(maxlen=window_size)
        self.ema_alpha = ema_alpha
        self.dead_zone = dead_zone
        self._ema_value = None

    def update(self, raw_value):
        self.window.append(raw_value)
        if len(self.window) < 3:
            self._ema_value = raw_value
            return raw_value
        median_val = float(np.median(list(self.window)))
        if self._ema_value is None:
            self._ema_value = median_val
        else:
            diff = median_val - self._ema_value
            if abs(diff) > self.dead_zone:
                self._ema_value += self.ema_alpha * diff
        return self._ema_value


class EyeStateAnalyzer:
    EYE_OPEN = "open"
    EYE_CLOSED = "closed"
    EYE_PARTIAL = "partial"

    def __init__(self):
        self._smoother = TemporalSmoother(window_size=7, ema_alpha=0.20, dead_zone=0.04)
        self._confirmed_state = self.EYE_OPEN
        self._confirmed_openness = 1.0
        self._state_counter = 0
        self._state_hold = 4

    def analyze(self, gray_eq, eye_center, face_box, side="right"):
        x1, y1, x2, y2 = [int(v) for v in face_box]
        fw, fh = x2 - x1, y2 - y1
        ecx, ecy = float(eye_center[0]), float(eye_center[1])

        roi_half_w = int(fw * 0.15)
        roi_half_h = int(fh * 0.10)
        rx1 = max(0, int(ecx - roi_half_w))
        ry1 = max(0, int(ecy - roi_half_h))
        rx2 = min(gray_eq.shape[1], int(ecx + roi_half_w))
        ry2 = min(gray_eq.shape[0], int(ecy + roi_half_h))

        result = {
            'state': self._confirmed_state,
            'openness': self._confirmed_openness,
            'contour_area': 0,
            'vertical_gradient': 0.0,
            'iris_detected': False,
            'landmarks': None,
        }

        if rx2 - rx1 < 12 or ry2 - ry1 < 8:
            return result

        roi = gray_eq[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return result

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(roi)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0.8)

        sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.abs(sobel_y)
        result['vertical_gradient'] = float(np.mean(grad_mag))

        edges_all = np.zeros_like(roi)
        for (lo, hi) in [(15, 50), (25, 75), (35, 100)]:
            e = cv2.Canny(blurred, lo, hi)
            edges_all = cv2.bitwise_or(edges_all, e)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
        edges_closed = cv2.morphologyEx(edges_all, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        roi_area = roi.shape[0] * roi.shape[1]
        best_contour = None
        best_score = -1

        for c in contours:
            area = cv2.contourArea(c)
            if area < 10 or area > roi_area * 0.85:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = bw / bh if bh > 0 else 0
            if aspect < 0.5 or aspect > 8.0:
                continue
            cy_ratio = (by + bh / 2.0) / roi.shape[0]
            pos_score = 1.0 - abs(cy_ratio - 0.5) * 1.5
            pos_score = max(0, pos_score)
            shape_score = min(aspect / 2.0, 1.0)
            score = area * 0.3 + pos_score * 40 + shape_score * 30
            if score > best_score:
                best_score = score
                best_contour = c

        if best_contour is None and contours:
            best_contour = max(contours, key=cv2.contourArea)

        thresh_val = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        dark_mask = cv2.inRange(blurred, 0, int(thresh_val * 0.7))
        dark_contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_dark_area = 0
        for c in dark_contours:
            area = cv2.contourArea(c)
            if area > best_dark_area and area > 5:
                best_dark_area = area

        iris_detected = best_dark_area > 15
        result['iris_detected'] = iris_detected
        result['contour_area'] = cv2.contourArea(best_contour) if best_contour is not None else 0

        raw_openness = self._compute_raw_openness(
            roi, best_contour, result['contour_area'], roi_area,
            grad_mag, iris_detected, best_dark_area
        )

        smoothed_openness = self._smoother.update(raw_openness)
        result['openness'] = smoothed_openness

        raw_state = self._raw_state_from_openness(smoothed_openness)
        result['state'] = self._hysteresis_state(raw_state)

        if best_contour is not None and len(best_contour) >= 5:
            ellipse = cv2.fitEllipse(best_contour)
            (ecx2, ecy2), (ma, MA), angle = ellipse
            ma, MA = max(abs(ma), 1), max(abs(MA), 1)
            if MA > ma:
                ma, MA = MA, ma
                angle = angle + 90
            max_w = roi_half_w * 0.9
            max_h = roi_half_h * 0.9
            ma = min(ma, max_w * 2)
            MA = min(MA, max_h * 2)
            vertical_scale = 0.3 + 0.7 * smoothed_openness
            MA_scaled = MA * vertical_scale
            rad = np.radians(angle)
            angles = [pi, 2 * pi / 3, pi / 3, 0, -pi / 3, -2 * pi / 3]
            pts = np.zeros((6, 2), dtype=np.float32)
            for i, a in enumerate(angles):
                x = ecx2 + (ma / 2) * cos(a) * cos(rad) - (MA_scaled / 2) * sin(a) * sin(rad)
                y = ecy2 + (ma / 2) * cos(a) * sin(rad) + (MA_scaled / 2) * sin(a) * cos(rad)
                pts[i] = [x + rx1, y + ry1]
            result['landmarks'] = pts
        else:
            pts = np.zeros((6, 2), dtype=np.float32)
            eye_w = roi_half_w * 0.65
            eye_h = roi_half_h * smoothed_openness * 1.2
            angles = [pi, 2 * pi / 3, pi / 3, 0, -pi / 3, -2 * pi / 3]
            for i, a in enumerate(angles):
                pts[i] = [ecx + eye_w * cos(a), ecy + eye_h * sin(a)]
            result['landmarks'] = pts

        return result

    def _raw_state_from_openness(self, openness):
        if openness < 0.28:
            return self.EYE_CLOSED
        elif openness < 0.52:
            return self.EYE_PARTIAL
        else:
            return self.EYE_OPEN

    def _hysteresis_state(self, raw_state):
        if raw_state == self._confirmed_state:
            self._state_counter = 0
            return self._confirmed_state
        self._state_counter += 1
        if self._state_counter >= self._state_hold:
            self._confirmed_state = raw_state
            self._state_counter = 0
        return self._confirmed_state

    def _compute_raw_openness(self, roi, contour, contour_area, roi_area,
                              grad_mag, iris_detected, dark_area):
        scores = []
        area_ratio = contour_area / roi_area if roi_area > 0 else 0
        area_score = np.clip(area_ratio * 5.0, 0, 1)
        scores.append(('area', area_score, 0.25))

        grad_mean = np.mean(grad_mag)
        grad_score = np.clip(1.0 - grad_mean / 80.0, 0, 1)
        scores.append(('grad', grad_score, 0.20))

        if contour is not None and len(contour) >= 5:
            bx, by, bw, bh = cv2.boundingRect(contour)
            aspect = bw / bh if bh > 0 else 0
            if aspect < 1.0:
                aspect_score = 1.0
            elif aspect < 3.0:
                aspect_score = 1.0 - (aspect - 1.0) / 4.0
            else:
                aspect_score = 0.1
            scores.append(('aspect', aspect_score, 0.25))
        else:
            scores.append(('aspect', 0.5, 0.25))

        iris_score = 1.0 if iris_detected else 0.0
        scores.append(('iris', iris_score, 0.15))

        dark_ratio = dark_area / roi_area if roi_area > 0 else 0
        dark_score = np.clip(dark_ratio * 8.0, 0, 1)
        scores.append(('dark', dark_score, 0.15))

        total_weight = sum(w for _, _, w in scores)
        openness = sum(s * w for _, s, w in scores) / total_weight
        return float(np.clip(openness, 0.0, 1.0))


class YawnDetector:
    MOUTH_CLOSED = "closed"
    MOUTH_OPEN = "open"
    MOUTH_YAWN = "yawning"

    def __init__(self, yawn_frames=8, cooldown=4.0):
        self.yawn_frames = yawn_frames
        self.cooldown = cooldown
        self.open_counter = 0
        self.mouth_state = self.MOUTH_CLOSED
        self.last_yawn_time = 0.0
        self._smoother = TemporalSmoother(window_size=7, ema_alpha=0.20, dead_zone=0.04)
        self._confirmed_state = self.MOUTH_CLOSED
        self._confirmed_openness = 0.0
        self._state_counter = 0
        self._state_hold = 5

    def analyze(self, gray_eq, face_box):
        x1, y1, x2, y2 = [int(v) for v in face_box]
        fw, fh = x2 - x1, y2 - y1

        result = {
            'state': self._confirmed_state,
            'openness': self._confirmed_openness,
            'mouth_aspect_ratio': 0.0,
            'inner_mouth_area': 0.0,
            'is_yawning': False,
            'outer_landmarks': None,
            'inner_landmarks': None,
        }

        my1 = y1 + int(fh * 0.68)
        my2 = y1 + int(fh * 0.85)
        mx1 = x1 + int(fw * 0.25)
        mx2 = x2 - int(fw * 0.25)

        if my1 >= my2 or mx1 >= mx2:
            return result

        roi = gray_eq[my1:my2, mx1:mx2]
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            return result

        roi_h, roi_w = roi.shape[:2]
        roi_area = roi_h * roi_w
        roi_cx, roi_cy = roi_w / 2.0, roi_h / 2.0

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(roi)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.0)

        adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 8
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
        closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small, iterations=1)

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        best_score = -1
        for c in contours:
            area = cv2.contourArea(c)
            if area < 30 or area > roi_area * 0.6:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = bw / bh if bh > 0 else 0
            if aspect < 1.0:
                continue
            ccx = bx + bw / 2.0
            ccy = by + bh / 2.0
            center_dist = abs(ccx - roi_cx) / roi_w + abs(ccy - roi_cy) / roi_h
            score = area / roi_area * 10 - center_dist * 5 + min(aspect, 5) * 0.5
            if score > best_score:
                best_score = score
                best_contour = c

        if best_contour is None and contours:
            best_contour = max(contours, key=cv2.contourArea)

        thresh_val = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        inner_mask = cv2.inRange(blurred, 0, int(thresh_val * 0.5))
        inner_contours, _ = cv2.findContours(inner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        inner_area = 0
        for c in inner_contours:
            area = cv2.contourArea(c)
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = bw / bh if bh > 0 else 0
            if aspect > 0.8 and area > inner_area:
                inner_area = area

        result['inner_mouth_area'] = inner_area / roi_area if roi_area > 0 else 0

        mar = 0.0
        if best_contour is not None and len(best_contour) >= 5:
            bx, by, bw, bh = cv2.boundingRect(best_contour)
            mar = bh / bw if bw > 0 else 0
            result['mouth_aspect_ratio'] = mar

        raw_openness = self._compute_raw_openness(
            roi, best_contour, roi_area,
            result['inner_mouth_area'], mar
        )

        smoothed_openness = self._smoother.update(raw_openness)
        result['openness'] = smoothed_openness

        raw_state = self._raw_state(smoothed_openness)
        result['state'] = self._hysteresis_state(raw_state)
        result['is_yawning'] = (result['state'] == self.MOUTH_YAWN)

        if best_contour is not None and len(best_contour) >= 5:
            bx, by, bw, bh = cv2.boundingRect(best_contour)
            cx_roi = bx + bw / 2.0
            cy_roi = by + bh / 2.0
            half_w = bw / 2.0
            half_h = bh / 2.0
            outer_pts = np.zeros((12, 2), dtype=np.float32)
            for i in range(12):
                a = i * 2 * pi / 12
                ox = cx_roi + half_w * cos(a)
                oy = cy_roi + half_h * sin(a)
                outer_pts[i] = [ox + mx1, oy + my1]
            result['outer_landmarks'] = outer_pts

            iw = half_w * 0.55
            ih = half_h * 0.45
            inner_pts = np.zeros((8, 2), dtype=np.float32)
            for i in range(8):
                a = i * 2 * pi / 8
                ix = cx_roi + iw * cos(a)
                iy = cy_roi + ih * sin(a)
                inner_pts[i] = [ix + mx1, iy + my1]
            result['inner_landmarks'] = inner_pts

        return result

    def _raw_state(self, openness):
        if openness > 0.65:
            self.open_counter += 1
        else:
            self.open_counter = max(0, self.open_counter - 1)

        if self.open_counter >= self.yawn_frames:
            current_time = time.time()
            if self._confirmed_state != self.MOUTH_YAWN:
                if current_time - self.last_yawn_time >= self.cooldown:
                    self.last_yawn_time = current_time
                    return self.MOUTH_YAWN
                else:
                    return self.MOUTH_OPEN
            return self.MOUTH_YAWN
        elif openness > 0.35:
            return self.MOUTH_OPEN
        else:
            return self.MOUTH_CLOSED

    def _hysteresis_state(self, raw_state):
        if raw_state == self._confirmed_state:
            self._state_counter = 0
            return self._confirmed_state
        self._state_counter += 1
        if self._state_counter >= self._state_hold:
            self._confirmed_state = raw_state
            self._state_counter = 0
        return self._confirmed_state

    def _compute_raw_openness(self, roi, contour, roi_area, inner_ratio, mar):
        scores = []
        if contour is not None:
            area = cv2.contourArea(contour)
            area_ratio = area / roi_area if roi_area > 0 else 0
            area_score = np.clip(area_ratio * 4.0, 0, 1)
        else:
            area_score = 0.0
        scores.append(('area', area_score, 0.25))

        inner_score = np.clip(inner_ratio * 10.0, 0, 1)
        scores.append(('inner', inner_score, 0.30))

        mar_score = np.clip(mar * 3.0, 0, 1)
        scores.append(('mar', mar_score, 0.25))

        roi_std = float(np.std(roi))
        std_score = np.clip(roi_std / 50.0, 0, 1)
        scores.append(('std', std_score, 0.20))

        total_weight = sum(w for _, _, w in scores)
        openness = sum(s * w for _, s, w in scores) / total_weight
        return float(np.clip(openness, 0.0, 1.0))

    def reset(self):
        self.open_counter = 0
        self.mouth_state = self.MOUTH_CLOSED
        self.last_yawn_time = 0.0
        self._confirmed_state = self.MOUTH_CLOSED
        self._confirmed_openness = 0.0
        self._state_counter = 0
