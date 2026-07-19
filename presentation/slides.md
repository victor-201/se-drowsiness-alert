---
marp: true
theme: uncover
class: invert
---

# Ứng Dụng Cảnh Báo Buồn Ngủ Lái Xe

**Pipeline không dùng facial landmarks**

Môn Xử Lý Ảnh & Thị Giác Máy Tính - 121036

---

## Mục Tiêu

- Phát hiện nhắm mắt kéo dài.
- Phát hiện ngáp.
- Loại bỏ dlib và model 68 landmarks.
- Dùng kỹ thuật cạnh, phân đoạn và điểm đặc trưng có thể giải thích.
- So sánh định lượng với baseline landmark trên cùng ground truth.

---

## Pipeline Mới

```text
Camera
  -> Grayscale + CLAHE
  -> Haar/DNN face detection + NMS
  -> ROI mắt và miệng
  -> Gaussian + Canny + Otsu
  -> Connected components + Shi-Tomasi
  -> Eye/Mouth openness
  -> Bộ đếm frame + cảnh báo
```

Không còn:

- 68 facial landmarks.
- EAR/MAR từ landmark.
- Head pose, roll, pitch.
- CNN MMOD dlib fallback.

---

## ROI Và Đặc Trưng

**Mắt**

- Haar-like eye candidates.
- Fallback ROI theo tỷ lệ bounding box mặt.
- Cạnh mí mắt, vùng tối và phân tán điểm góc.

**Miệng**

- ROI theo tỷ lệ khuôn mặt.
- Khoang miệng tối và độ mở theo phương dọc.
- Hệ số darkness giảm nhầm môi khép/bóng.

---

## Công Thức Chỉ Số

```text
opening = 0.58 * dark_aspect
        + 0.27 * edge_aspect
        + 0.15 * keypoint_aspect
```

- `dark_aspect`: tỷ lệ cao/rộng của vùng tối.
- `edge_aspect`: độ phân tán dọc của cạnh.
- `keypoint_aspect`: độ phân tán dọc/ngang của Shi-Tomasi points.

Miệng nhân thêm hệ số darkness.

---

## Ngưỡng Và Trạng Thái

| Tham số | Giá trị |
|---|---:|
| Eye openness threshold | 0.20 |
| Closed-eye consecutive frames | 15 |
| Mouth openness threshold | 0.35 |
| Yawn consecutive frames | 5 |
| Minimum face size | 80 px |
| Feature confidence | 0.25 |

Calibration thu độ mở mắt bình thường trong 5 giây và đặt ngưỡng bằng median x 0.90.

---

## Test Tự Động

```bash
python -m unittest discover -s tests -v
```

Kết quả ngày 19/07/2026:

- 14/14 test đạt.
- Mắt mở/nhắm tại 3 mức sáng.
- Miệng bình thường/ngáp tại 3 mức sáng.
- Regression trên ảnh thật.
- Bộ đếm frame nhắm mắt.
- Một lần ngáp cho một chu kỳ mở miệng.
- Confusion matrix và F1 edge cases.
- Benchmark ROI xuất CSV/JSON/Markdown.
- Concurrency lock và kiểm tra frame/ROI ngoài biên.

---

## Sanity-Check Ảnh Thật

| Mẫu | Ground truth | Landmark EAR | Landmark | Eye openness | Pipeline mới |
|---|---|---:|---|---:|---|
| Tài xế | Mắt mở | 0.317 | Đúng | 0.237 | Đúng |
| Hành khách nghiêng | Mắt nhắm | 0.273 | Sai | 0.193 | Đúng |

Trên 2 mẫu:

- Landmark eye accuracy: 50%.
- Edge/feature eye accuracy: 100%.
- Không có mẫu ngáp dương tính nên yawn recall/F1 chưa xác định.

---

## Fixture ROI Có Kiểm Soát

12 mẫu = 3 mức sáng x 4 tổ hợp trạng thái.

| Chức năng | Landmark F1 | Edge/feature F1 |
|---|---:|---:|
| Nhắm mắt | 1.000 | 1.000 |
| Ngáp | 0.000 | 1.000 |

- Valid eye/mouth/feature coverage: 100%.
- Face detection được bỏ qua bằng bounding box gán sẵn.
- Chỉ xác nhận test chức năng, không đại diện dữ liệu người thật.

---

## Dung Lượng Và Phụ Thuộc

| Tài nguyên xóa | Kích thước |
|---|---:|
| Shape predictor 68 | 99,693,937 byte |
| CNN MMOD | 729,940 byte |
| Tổng | 95.77 MiB |

`dlib` đã được xóa khỏi dependencies.

---

## Benchmark Trích Đặc Trưng

Sau warm-up, 5 batch x 100 khuôn mặt trên macOS arm64:

| Phương pháp | Median ms/face | FPS tương đương |
|---|---:|---:|
| Landmark cũ | 0.714 | 1400.64 |
| Cạnh + điểm đặc trưng | 17.089 | 58.52 |

Pipeline mới chậm hơn do chạy nhiều phép xử lý ảnh trên ROI.

Phép đo không gồm face detection, camera và UI.

---

## Benchmark Chính Thức

```bash
python -m src.evaluation.metrics \
  --video test.mp4 \
  --labels labels.csv \
  --landmark-baseline landmark.csv \
  --output evaluation_results
```

Hoặc đánh giá ROI gán sẵn:

```bash
python -m src.evaluation.metrics \
  --roi-manifest roi_manifest.csv \
  --landmark-baseline landmark.csv \
  --output evaluation_results
```

Kết quả:

- Confusion matrix từng chức năng.
- Accuracy, precision, recall, specificity, F1, FPR.
- Processing time/FPS.
- Face-detection coverage.
- Eye/mouth/valid-feature coverage.

---

## Đánh Giá Lại

**Đã xác nhận**

- Không còn landmark/dlib trong runtime.
- Giảm 95.77 MiB model.
- Hai chức năng có test tự động.
- Ca mắt nhắm nghiêng trong ảnh mẫu được xử lý tốt hơn baseline.
- Fixture có kiểm soát đạt 100% coverage và tách đủ mắt/ngáp.

**Chưa xác nhận**

- Chưa đủ dữ liệu người thật để kết luận F1 tốt hơn.
- Chưa có ngáp dương tính thật trong repository.
- Tốc độ mới thấp hơn landmark ở bước trích đặc trưng.

---

## Kết Luận

Pipeline mới tốt hơn về:

- Phụ thuộc.
- Dung lượng.
- Khả năng giải thích.
- Khả năng test từng bước.

Cần tập video gán nhãn chung cho hai phương pháp trước khi khẳng định độ chính xác tổng quát.

---

# Q&A
