# Tái Đánh Giá Pipeline Cạnh Và Điểm Đặc Trưng

**Ngày đánh giá:** 19/07/2026

## 1. Phạm vi thay đổi

Pipeline mới loại bỏ:

- `dlib`.
- `shape_predictor_68_face_landmarks.dat`.
- `mmod_human_face_detector.dat`.
- EAR/MAR tính từ landmark.
- Head pose, roll, pitch và cảnh báo nghiêng đầu.

Hai chức năng giữ lại để kiểm tra là nhắm mắt và ngáp.

## 2. Cơ chế mới

Bounding box khuôn mặt vẫn được tìm bằng Haar Cascade và OpenCV DNN SSD. Bên trong khuôn mặt:

1. Vùng mắt được định vị bằng Haar-like eye candidates; ROI theo tỷ lệ khuôn mặt là fallback.
2. Vùng miệng được suy ra theo tỷ lệ khuôn mặt.
3. Mỗi ROI được resize, CLAHE và Gaussian blur.
4. Canny dùng ngưỡng tự động từ Otsu.
5. Otsu inverse threshold cùng morphology tạo mask vùng tối.
6. Connected components lấy thành phần tối phù hợp.
7. Shi-Tomasi lấy tối đa 24 điểm góc.
8. Chỉ số mở được tổng hợp từ:

```text
opening = 0.58 * dark_aspect
        + 0.27 * edge_aspect
        + 0.15 * keypoint_aspect
```

Với miệng, chỉ số trên được nhân thêm hệ số độ tối để hạn chế nhầm môi khép hoặc vùng bóng thành khoang miệng mở.

Ngưỡng mặc định:

- `eye_openness < 0.20`: mắt nhắm.
- `mouth_openness > 0.35`: miệng mở.
- 15 frame nhắm mắt liên tiếp: cảnh báo ngủ gật.
- 5 frame miệng mở liên tiếp: xác nhận một lần ngáp.

## 3. Kết quả test chức năng

Lệnh:

```bash
python -m unittest discover -s tests -v
```

Kết quả ngày 19/07/2026: **14/14 test đạt**.

| Nhóm test | Kết quả |
|---|---:|
| Mắt mở so với mắt nhắm ở brightness 165, 210, 235 | 3/3 đạt |
| Miệng bình thường so với ngáp ở brightness 165, 210, 235 | 3/3 đạt |
| Ảnh thật: tài xế mở mắt, hành khách nhắm mắt | Đạt |
| 3 frame nhắm mắt liên tiếp kích hoạt trạng thái buồn ngủ trong test rút gọn | Đạt |
| 3 frame miệng mở liên tiếp chỉ tính một lần ngáp trong test rút gọn | Đạt |
| Hàm confusion matrix và trường hợp không có positive support | Đạt |
| F1 bằng 0 khi baseline bỏ sót toàn bộ mẫu dương tính | Đạt |
| Benchmark ROI xuất CSV, JSON, Markdown và so sánh baseline | Đạt |
| Box khuôn mặt vượt biên được clip, box đảo được loại bỏ | Đạt |
| Frame rỗng được bỏ qua trước khi gọi OpenCV | Đạt |
| Sáu thread gọi chung detector được tuần tự hóa | Đạt |

Các test tổng hợp xác nhận logic và độ tách chỉ số, không thay thế tập test người thật có ground truth.

Ngoài unit test, stress test 4 thread x 4 lần trên `notebooks/test.png` hoàn tất
16 lượt xử lý Haar/DNN/eye cascade mà không có exception hoặc worker bị treo.

## 4. So sánh sanity-check với landmark cũ

Hai khuôn mặt trong `notebooks/test.png` được gán nhãn thủ công:

Dữ liệu raw được lưu tại `docs/sample_reassessment.csv`.

| Mẫu | Ground truth mắt | Landmark EAR | Landmark dự đoán | Eye openness mới | Pipeline mới dự đoán |
|---|---|---:|---|---:|---|
| Tài xế | Mở | 0.317 | Mở | 0.237 | Mở |
| Hành khách nghiêng đầu | Nhắm | 0.273 | Mở, sai | 0.193 | Nhắm, đúng |

Ngưỡng so sánh là EAR 0.22 và eye openness 0.20.

| Chỉ số trên 2 khuôn mặt | Landmark cũ | Cạnh + điểm đặc trưng |
|---|---:|---:|
| Accuracy nhắm mắt | 50% (1/2) | 100% (2/2) |
| Yawn false positive | 0/2 | 0/2 |
| Yawn recall/F1 | Không xác định | Không xác định |

Hai mẫu đều không ngáp, vì vậy không thể dùng sanity-check này để kết luận recall hoặc F1 cho chức năng ngáp.

Giá trị miệng:

| Mẫu | Landmark MAR | Mouth openness mới | Ground truth |
|---|---:|---:|---|
| Tài xế | 0.033 | 0.097 | Không ngáp |
| Hành khách | 0.245 | 0.147 | Không ngáp |

## 5. Benchmark ROI có kiểm soát

Để kiểm tra đủ hai chức năng mà không bị ảnh hưởng bởi face detector, bộ fixture
`docs/controlled_fixture/` gồm 12 ảnh được sinh xác định:

- Ba mức sáng: 165, 210 và 235.
- Bốn trạng thái: mắt mở/miệng thường, mắt mở/ngáp, mắt nhắm/miệng thường,
  mắt nhắm/ngáp.
- Bounding box khuôn mặt được gán trực tiếp là toàn bộ ảnh 240 x 240.

Raw score của hai phương pháp được lưu tại:

- `docs/controlled_reassessment.csv`.
- `docs/controlled_landmark_baseline.csv`.

Baseline EAR/MAR được đo bằng snapshot model 68 landmark cũ trước khi file model
bị xóa. Ngưỡng là EAR 0.22, MAR 0.30, eye openness 0.20 và mouth openness 0.35.

| Chức năng | Phương pháp | Accuracy | Precision | Recall | F1 | TP/TN/FP/FN |
|---|---|---:|---:|---:|---:|---:|
| Nhắm mắt | Landmark | 1.000 | 1.000 | 1.000 | 1.000 | 6/6/0/0 |
| Nhắm mắt | Cạnh + điểm đặc trưng | 1.000 | 1.000 | 1.000 | 1.000 | 6/6/0/0 |
| Ngáp | Landmark | 0.500 | n/a | 0.000 | 0.000 | 0/6/0/6 |
| Ngáp | Cạnh + điểm đặc trưng | 1.000 | 1.000 | 1.000 | 1.000 | 6/6/0/0 |

Coverage feature, mắt và miệng của pipeline mới đều đạt 100% trên 12 mẫu ROI.
Kết quả này xác nhận việc tách trạng thái và đường đánh giá hoạt động đúng. Vì
ảnh được tạo có kiểm soát và không phải khuôn mặt người thật, không được dùng
bảng này để tuyên bố độ chính xác tổng quát hoặc ưu thế thực tế so với landmark.

Lệnh tái tạo fixture và đánh giá:

```bash
python -m src.evaluation.controlled_fixture \
  --output docs/controlled_fixture

python -m src.evaluation.metrics \
  --roi-manifest docs/controlled_fixture/roi_manifest.csv \
  --landmark-baseline docs/controlled_landmark_baseline.csv \
  --output evaluation_results
```

## 6. Chi phí tài nguyên

Hai file model đã xóa:

| File | Kích thước |
|---|---:|
| `shape_predictor_68_face_landmarks.dat` | 99,693,937 byte |
| `mmod_human_face_detector.dat` | 729,940 byte |
| Tổng | 100,423,877 byte, khoảng 95.77 MiB |

`dlib` cũng đã được xóa khỏi `requirements.txt`.

Benchmark riêng bước trích đặc trưng trên hai bounding box nói trên, sau warm-up, 5 batch x 100 khuôn mặt trên macOS arm64:

Dữ liệu từng batch được lưu tại `docs/sample_runtime_benchmark.csv`.

| Phương pháp | Median ms/khuôn mặt | FPS tương đương |
|---|---:|---:|
| dlib 68 landmarks | 0.714 | 1400.64 |
| Cạnh + điểm đặc trưng | 17.089 | 58.52 |

Phép đo này không gồm face detection, UI, camera hoặc cảnh báo. Kết quả phụ thuộc phần cứng. Pipeline mới chậm hơn rõ rệt ở bước trích đặc trưng vì chạy eye cascade, Canny, morphology, connected components và Shi-Tomasi trên nhiều ROI.

## 7. Kết luận tái đánh giá

Kết luận có thể xác nhận:

- Đã loại bỏ hoàn toàn phụ thuộc landmark/dlib khỏi runtime.
- Giảm khoảng 95.77 MiB model.
- Hai chức năng nhắm mắt và ngáp đã có test tự động.
- Trên sanity-check hai khuôn mặt, pipeline mới xử lý đúng ca nhắm mắt bị nghiêng mà landmark cũ bỏ sót.
- Trên 12 fixture có kiểm soát, pipeline mới tách đúng toàn bộ trạng thái mắt và miệng; baseline landmark bỏ sót toàn bộ sáu mẫu ngáp tổng hợp.
- Tốc độ trích đặc trưng mới thấp hơn landmark cũ, dù phép đo riêng vẫn đạt khoảng 58.52 khuôn mặt/giây trên máy thử.

Kết luận chưa thể xác nhận:

- Không thể tuyên bố pipeline mới có F1 tốt hơn landmark trên người thật.
- Chưa có mẫu ngáp dương tính thật trong repository để đo recall/F1.
- Các F1 khoảng 0.87 cho EAR và 0.83 cho MAR trong báo cáo cũ là số ước lượng, không phải kết quả từ cùng một tập ground truth.

Do đó, đánh giá hiện tại là: **pipeline mới tốt hơn về phụ thuộc, dung lượng và khả năng giải thích; có tín hiệu tốt trên ca mắt nhắm nghiêng; nhưng chưa đủ dữ liệu để kết luận tốt hơn về độ chính xác tổng quát và đang có chi phí xử lý lớn hơn.**

## 8. Benchmark chính thức

Để so sánh công bằng, hai phương pháp phải chạy trên cùng frame và cùng ground truth:

```bash
python -m src.evaluation.metrics \
  --video path/to/test.mp4 \
  --labels path/to/labels.csv \
  --landmark-baseline path/to/landmark_predictions.csv \
  --output evaluation_results
```

Ground truth:

```csv
frame,eye_closed,yawn
0,0,0
1,1,0
2,1,1
```

Baseline landmark có thể chứa dự đoán:

```csv
frame,eye_closed_pred,yawn_pred
0,0,0
1,1,0
2,1,1
```

Hoặc chứa EAR/MAR raw:

```csv
frame,ear,mar
0,0.31,0.12
1,0.18,0.15
2,0.17,0.48
```

Công cụ xuất confusion matrix, accuracy, precision, recall, specificity, F1, FPR, thời gian xử lý, face coverage, eye coverage và mouth coverage.

Nếu face detector làm mất coverage hoặc dữ liệu đã có bounding box chuẩn, dùng
chế độ ROI:

```bash
python -m src.evaluation.metrics \
  --roi-manifest path/to/roi_manifest.csv \
  --landmark-baseline path/to/landmark_predictions.csv \
  --output evaluation_results
```

Manifest ROI có định dạng:

```csv
sample,image,x1,y1,x2,y2,eye_closed,yawn
driver_001,frames/001.png,120,60,340,300,0,0
driver_002,frames/002.png,118,62,342,302,1,1
```
