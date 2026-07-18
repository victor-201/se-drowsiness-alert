<!-- Slide number: 1 -->

Nhóm B:

Ứng dụng cảnh báo phát hiện buồn ngủ khi lái xe

Học phần: Xử lý ảnh và thị giác máy tính
GVHD: Võ Thượng Anh

<!-- Slide number: 2 -->

Vấn đề

Nguyên nhân
~20% tai nạn liên quan buồn ngủ (WHO)

Mục tiêu
Hệ thống real-time phát hiện: nhắm mắt, ngáp, nghiêng đầu

Giả Thuyết
EAR=0.22, MAR=0.30, HEAD_TILT=15° cho F1 ≥ 0.85

<!-- Slide number: 3 -->

Pipeline Tổng quan

Camera (BGR) → Tiền xử lý (Grayscale+CLAHE) → Face Detection (Haar+DNN+CNN MMOD) → Landmarks (68 điểm) → ROI (Mắt/Miệng) → Phân loại (EAR/MAR) → Cảnh Báo (Âm thanh)

Face Detection gồm 3 phương pháp:
- Haar Cascade (nhanh)
- DNN SSD ResNet-10 (chính xác)
- CNN MMOD dlib (fallback khi <3 mặt)
→ Kết hợp bằng NMS (IoU=0.4) + FP Filter

<!-- Slide number: 4 -->

6 Kỹ thuật được sử dụng

01 — Ch.2: Grayscale (thủ công), CLAHE (thủ công), lọc Gaussian
Triển khai: numpy thuần, không dùng OpenCV

02 — Ch.3: Face Detection (Haar + DNN SSD + CNN MMOD + NMS)
Kết hợp 3 phương pháp, NMS bounding box, FP Filter

03 — Ch.3: 68 Facial Landmarks + Canny Edge (thủ công)
dlib shape predictor, Sobel gradient + NMS + Hysteresis, Otsu threshold

04 — Ch.4: Phân đoạn ROI + Connected Components
Otsu threshold tự động, Union-Find 4-connected

05 — Ch.5: Threshold Classification (EAR, MAR, Head Pose)
Dynamic threshold, auto-calibration 30 frame

06 — Ch.5: Cảnh báo âm thanh + hình ảnh
Kivy UI, alert system

<!-- Slide number: 5 -->

Demo  Tính năng

![](Picture22.jpg)

<!-- Slide number: 6 -->

Kết quả khảo sát EAR

| Ngưỡng | Precision | Recall | F1-score |
|--------|-----------|--------|----------|
| 0.16   | ~0.92     | ~0.55  | ~0.69    |
| 0.18   | ~0.88     | ~0.68  | ~0.77    |
| 0.20   | ~0.82     | ~0.78  | ~0.80    |
| **0.22** | **~0.85** | **~0.89** | **~0.87** |
| 0.24   | ~0.75     | ~0.92  | ~0.83    |
| 0.26   | ~0.62     | ~0.95  | ~0.75    |

- EAR=0.22 cân bằng precision-recall tốt nhất
- EAR thấp: bỏ sót nhiều
- EAR cao: dương tính giả

<!-- Slide number: 7 -->

Kết quả khảo sát MAR & Head Tilt

MAR:
| Ngưỡng | Precision | Recall | F1-score |
|--------|-----------|--------|----------|
| 0.25   | ~0.65     | ~0.90  | ~0.75    |
| **0.30** | **~0.83** | **~0.83** | **~0.83** |
| 0.35   | ~0.88     | ~0.70  | ~0.78    |
| 0.40   | ~0.92     | ~0.50  | ~0.65    |

Head Tilt:
| Ngưỡng | Precision | Recall | F1-score |
|--------|-----------|--------|----------|
| 10°    | ~0.68     | ~0.92  | ~0.78    |
| **15°** | **~0.82** | **~0.85** | **~0.83** |
| 20°    | ~0.90     | ~0.65  | ~0.75    |

<!-- Slide number: 8 -->

Kết quả tổng thể

| Chỉ số     | Giá trị |
|------------|---------|
| F1 EAR     | ~0.87   |
| F1 MAR     | ~0.83   |
| Độ trễ     | ~0.5 giây |
| FPS        | ≥15     |

Đạt mục tiêu F1 EAR ≥ 0.85, F1 MAR ≥ 0.80

<!-- Slide number: 9 -->

Đối chiếu giả thuyết

| Giả thuyết | Kết quả | Kết luận |
|------------|---------|----------|
| 1. EAR thấp → giảm nhạy | EAR=0.16 bỏ sót nhiều, EAR=0.26 nhiều FP | Đúng |
| 2. MAR=0.30 cho cân bằng tốt nhất | 0.25 nhiều FP; 0.40 bỏ sót; 0.30 cân bằng | Đúng |
| 3. EAR_CONSEC=15 cho F1 tốt nhất | 5 frame nhầm nháy mắt; 25 frame tăng độ trễ | Đúng |
| 4. HEAD_TILT=15° phù hợp nhất | 10° quá nhạy; 20° bỏ sót; 15° cân bằng | Đúng 1 phần |
| 5. Pipeline F1 ≥ 0.85, trễ <1s | F1≈0.87, trễ≈0.5s | Đúng |

→ 4/5 giả thuyết đúng, 1/5 đúng một phần

<!-- Slide number: 10 -->

Thảo luận

Hiệu quả
- Kết hợp đa chỉ số (EAR + MAR + head pose + tần suất)
- Dynamic threshold cho blink detection
- Auto-calibration head pose (30 frame, median + drift)
- Calibration EAR theo từng người (avg × 0.9)

Hạn chế
- Nhạy với ánh sáng yếu (< 50 lux)
- Khó phát hiện khi đeo kính
- Mất dấu khi góc quay > 45°

<!-- Slide number: 11 -->

Kết luận

- Pipeline hoàn chỉnh với 6 kỹ thuật CV (5 thuộc Ch.3–5)
- Triển khai thủ công: CLAHE, BGR→Gray, Canny edge (Sobel+NMS+hysteresis), Otsu threshold, Connected Components (Union-Find)
- Tham số tối ưu: EAR=0.22, MAR=0.30, HEAD_TILT=15°
- F1-score EAR ≈ 0.87 — vượt mục tiêu ≥ 0.85
- Độ trễ phát hiện ~0.5 giây — đáp ứng yêu cầu thời gian thực
- 2.450+ dòng Python, 21 file, kiến trúc MVC rõ ràng
- 4/5 giả thuyết được kiểm chứng đúng

<!-- Slide number: 12 -->

Thank you
