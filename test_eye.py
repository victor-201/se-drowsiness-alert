import cv2
import numpy as np
from src.core.custom_landmark_detector import CustomLandmarkDetector
from src.core.facial_analyzer import FacialAnalyzer

img = cv2.imread('data/test_face.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
gray_eq = clahe.apply(gray)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_eq, 1.1, 4, minSize=(60, 60))
x, y, w, h = faces[0]
face_box = [int(x), int(y), int(x+w), int(y+h)]
print("Face box:", face_box, "size:", w, "x", h)

detector = CustomLandmarkDetector()
landmarks = detector.detect_landmarks(gray_eq, face_box)

vis = img.copy()

# Draw face box
cv2.rectangle(vis, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (255, 0, 0), 2)

# Draw all 68 landmarks
for i in range(68):
    px, py = int(landmarks[i, 0]), int(landmarks[i, 1])
    color = (0, 255, 0)
    if 36 <= i <= 41:
        color = (0, 0, 255)  # right eye - red
    elif 42 <= i <= 47:
        color = (255, 0, 255)  # left eye - magenta
    elif 48 <= i <= 67:
        color = (0, 165, 255)  # mouth - orange
    cv2.circle(vis, (px, py), 3, color, -1)

# Connect eye contours
re = landmarks[36:42].astype(int)
le = landmarks[42:48].astype(int)
cv2.polylines(vis, [re], True, (0, 0, 255), 1)
cv2.polylines(vis, [le], True, (255, 0, 255), 1)

# Connect mouth
outer_mouth = landmarks[48:60].astype(int)
inner_mouth = landmarks[60:68].astype(int)
cv2.polylines(vis, [outer_mouth], True, (0, 165, 255), 1)
cv2.polylines(vis, [inner_mouth], True, (0, 165, 255), 1)

# Connect jaw
jaw = landmarks[0:17].astype(int)
cv2.polylines(vis, [jaw], False, (255, 255, 0), 1)

# Connect brows
rbrow = landmarks[17:22].astype(int)
lbrow = landmarks[22:27].astype(int)
cv2.polylines(vis, [rbrow], False, (255, 255, 255), 1)
cv2.polylines(vis, [lbrow], False, (255, 255, 255), 1)

# Nose
nose_bridge = landmarks[27:31].astype(int)
nose_bottom = landmarks[31:36].astype(int)
cv2.polylines(vis, [nose_bridge], False, (200, 200, 200), 1)
cv2.polylines(vis, [nose_bottom], True, (200, 200, 200), 1)

# Print details
print("\n--- Right Eye (36-41) ---")
for i in range(36, 42):
    print("  pt{}: ({:.1f}, {:.1f})".format(i, landmarks[i,0], landmarks[i,1]))

print("\n--- Left Eye (42-47) ---")
for i in range(42, 48):
    print("  pt{}: ({:.1f}, {:.1f})".format(i, landmarks[i,0], landmarks[i,1]))

print("\n--- Mouth Outer (48-59) ---")
for i in range(48, 60):
    print("  pt{}: ({:.1f}, {:.1f})".format(i, landmarks[i,0], landmarks[i,1]))

print("\n--- Mouth Inner (60-67) ---")
for i in range(60, 68):
    print("  pt{}: ({:.1f}, {:.1f})".format(i, landmarks[i,0], landmarks[i,1]))

analyzer = FacialAnalyzer()
re = landmarks[36:42]
le = landmarks[42:48]
mouth = landmarks[48:68]
print("\nEAR right:", "{:.3f}".format(analyzer.calculate_ear(re)))
print("EAR left:", "{:.3f}".format(analyzer.calculate_ear(le)))
print("MAR:", "{:.3f}".format(analyzer.calculate_mar(mouth)))

re_s = detector.last_eye_states['right']
le_s = detector.last_eye_states['left']
print("\nRight eye: state={}, openness={:.3f}, contour_area={}, iris={}".format(
    re_s['state'], re_s['openness'], re_s['contour_area'], re_s['iris_detected']))
print("Left eye:  state={}, openness={:.3f}, contour_area={}, iris={}".format(
    le_s['state'], le_s['openness'], le_s['contour_area'], le_s['iris_detected']))

yawn = detector.last_yawn_state
print("\nMouth: state={}, openness={:.3f}, mar={:.3f}".format(
    yawn['state'], yawn['openness'], yawn['mouth_aspect_ratio']))

cv2.imwrite('test_result.png', vis)
print("\nSaved test_result.png")
