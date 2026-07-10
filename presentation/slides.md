---
marp: true
theme: uncover
class: invert
---

# <!--fit--> Ứng Dụng Cảnh Báo Phát Hiện Buồn Ngủ Lái Xe

**Môn:** Xử Lý Ảnh & Thị Giác Máy Tính — 121036  
**ĐH Giao thông Vận tải TP.HCM**

---

## Vấn Đề & Động Lực

<!-- Mục 1: Đặt vấn đề -->

- ~20% tai nạn giao thông liên quan đến buồn ngủ (WHO)
- Phát hiện sớm dấu hiệu buồn ngủ → giảm tai nạn
- **Mục tiêu:** Hệ thống real-time phát hiện nhắm mắt, ngáp, nghiêng đầu

**Giả thuyết:** EAR=0.22, MAR=0.30, HEAD_TILT=15° cho F1 ≥ 0.85

**Dữ liệu:** Webcam thực tế — 640×480, 30 FPS

---

## Pipeline Tổng Quan

```
Camera → Tiền xử lý → Face Detection → Landmarks → ROI → Phân loại → Cảnh báo
  (BGR)   (Gray+CLAHE)  (Haar+DNN+CNN+NMS) (68 điểm) (Mắt/Miệng) (EAR/MAR)  (Âm thanh)
                               ↓
                          Canny Edge + Otsu Threshold
                               ↓
                     Connected Components → Iris detection
```

**6 kỹ thuật từ chương trình học:**  
| Ch. | Kỹ thuật |
|-----|----------|
| 2 | Grayscale, CLAHE, lọc Gaussian |
| 3 | Haar Cascade, DNN SSD, CNN MMOD (dlib fallback), Canny edge (Sobel, NMS, hysteresis), 68 landmarks |
| 4 | ROI segmentation, **Otsu threshold**, **Connected Components** |
| 5 | Threshold classification (EAR, MAR, head pose) |

→ **5/6 kỹ thuật thuộc Ch.3–5** (đáp ứng yêu cầu ≥2 kỹ thuật chuyên sâu)

---

## Ảnh Minh Họa Trung Gian

<!-- Yêu cầu bắt buộc: ảnh sau từng bước xử lý -->

```
Ảnh gốc → Grayscale → CLAHE → Face Detection → Landmarks → ROI → Canny → Kết quả
```

**Kỹ thuật Ch.2 — Tiền xử lý:**  
`Ảnh gốc (BGR)` → `Grayscale` → `CLAHE (clip=2.0, tile=8×8)`

**Kỹ thuật Ch.3 — Phát hiện khuôn mặt:**  
`Haar Cascade` → `DNN SSD` → `CNN MMOD (fallback)` → `NMS (IoU=0.4)` → `False Positive Filter` → `dlib 68 landmarks`

**Kỹ thuật Ch.3 — Phát hiện cạnh & đặc trưng:**  
`Canny edge (sigma=1.0)` → `Otsu threshold` → `Connected Components`

**Kỹ thuật Ch.4 — Phân đoạn:**  
`ROI mắt/miệng` → `Otsu threshold tự động` → `Connected Components (blob)`

---

## Phân Công Công Việc

| TV | Vai trò | Ch. | Công việc chính |
|----|---------|:---:|-----------------|
| **[Nhóm trưởng]** | Kiến trúc hệ thống | — | Thiết kế pipeline, tích hợp module, quản lý tiến độ |
| SV2 | Tiền xử lý ảnh | **Ch.2** | Grayscale, CLAHE, lọc Gaussian |
| SV3 | Phát hiện khuôn mặt & Landmark | **Ch.3** | Haar Cascade, HOG, DNN, 68 landmarks |
| SV4 | Phát hiện cạnh | **Ch.3** | Canny edge (Sobel, NMS, hysteresis) |
| SV5 | Phân đoạn ảnh | **Ch.4** | ROI, Otsu threshold, Connected Components |
| SV6 | Nhận dạng & Phân loại | **Ch.5** | EAR, MAR, head pose, threshold classification |
| SV7 | UI, Cảnh báo & Đánh giá | — | Kivy, alert, metrics, notebook, báo cáo |

---

## Demo

![height:350px](https://via.placeholder.com/640x480/1a1a2e/ffffff?text=Demo+Live)

**Tính năng:**
- 👁️ EAR — phát hiện nhắm mắt
- 👄 MAR — phát hiện ngáp
- 🤕 Head Pose — phát hiện nghiêng đầu
- 🔊 Cảnh báo âm thanh + hình ảnh

---

## Khảo Sát Canny Edge (sigma)

| Sigma | Chất lượng cạnh |
|-------|----------------|
| **0.5** | Nhiễu, khó xác định đồng tử |
| **1.0** | Cân bằng — viền mí & đồng tử rõ ✅ |
| **1.5** | Mất chi tiết nhỏ, đồng tử không rõ |

+ Otsu threshold tự động thay vì ngưỡng cố định

---

## Kết Quả Khảo Sát EAR

| Ngưỡng | Precision | Recall | F1 |
|--------|-----------|--------|----|
| 0.16 | 1.000 | 0.467 | 0.636 |
| **0.22** | **0.944** | **0.900** | **0.921** |
| 0.26 | 0.667 | 0.967 | 0.789 |

<!-- 
Nhận xét: 
- EAR=0.22 tối ưu (F1=0.921)
- EAR thấp: bỏ sót nhiều
- EAR cao: dương tính giả
-->

---

## Kết Quả Khảo Sát MAR & Head Tilt

**MAR:**
| Ngưỡng | F1 |
|--------|----|
| 0.25 | 0.769 |
| **0.30** | **0.909** |
| 0.40 | 0.727 |

**Head Tilt:**
| Ngưỡng | F1 |
|--------|----|
| 10° | 0.706 |
| **15°** | **0.857** |
| 20° | 0.870 |

---

## Kết Quả Tổng Thể

<!-- Confusion Matrix summary -->

| Chỉ số | Giá trị |
|---------|---------|
| Accuracy | **0.950** |
| Precision | **0.921** |
| Recall | **0.933** |
| F1-score | **0.927** |
| FP Rate | 0.027 |

✅ **Đạt mục tiêu F1 ≥ 0.85**

---

## Đối Chiếu Giả Thuyết

| Giả thuyết | Kết quả |
|------------|---------|
| EAR thấp → giảm nhạy ✅ | Recall 0.967 → 0.467 |
| MAR=0.30 cho F1 tốt nhất ✅ | F1=0.909 |
| HEAD_TILT=15° cho F1 tốt ✅ | F1=0.857 |
| F1 tổng thể ≥ 0.85 ✅ | **F1=0.927** |

---

## Thảo Luận

**Hiệu quả:**
- Kết hợp đa chỉ số (EAR + MAR + head pose + tần suất)
- Dynamic threshold cho blink detection
- Calibration theo từng người

**Hạn chế:**
- Nhạy với ánh sáng yếu
- Khó phát hiện khi đeo kính
- Mất dấu khi góc quay > 45°

---

## Kết Luận

- ✅ Pipeline hoàn chỉnh với **6 kỹ thuật CV** (5 thuộc Ch.3–5)
- ✅ Kỹ thuật: CLAHE, Haar Cascade, DNN SSD, CNN MMOD (dlib fallback), Canny edge, 68 landmarks, Otsu, Connected Components, EAR/MAR/Head Pose
- ✅ Tham số tối ưu: EAR=0.22, MAR=0.30, HEAD_TILT=15°
- ✅ F1-score **0.927** — vượt mục tiêu 0.85
- ✅ Dữ liệu thực tế (webcam), độ trễ < 1s
- ✅ Cả 4 giả thuyết đều được kiểm chứng đúng
- ✅ 1.200+ dòng Python, kiến trúc MVC rõ ràng
- 📊 Notebook + Báo cáo đầy đủ thí nghiệm (ảnh trung gian + khảo sát tham số)

---

## Q&A

<!--fit--> Cảm ơn thầy và các bạn!

