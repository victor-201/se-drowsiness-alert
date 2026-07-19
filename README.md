# Ứng Dụng Cảnh Báo Buồn Ngủ Lái Xe

Đồ án môn **Xử Lý Ảnh và Thị Giác Máy Tính** - Trường Đại học Giao thông Vận tải TP. Hồ Chí Minh.

## Phạm vi hiện tại

Ứng dụng giám sát khuôn mặt tài xế qua webcam và kiểm tra hai trạng thái:

- Nhắm mắt kéo dài.
- Ngáp.

Pipeline runtime không còn dùng `dlib` hoặc mô hình 68 facial landmarks. Head pose và cảnh báo nghiêng đầu đã được loại bỏ khỏi luồng xử lý.

## Pipeline

1. Chuyển frame sang grayscale và tăng tương phản bằng CLAHE.
2. Tìm bounding box khuôn mặt bằng Haar Cascade và OpenCV DNN SSD, gộp bằng NMS.
3. Xác định vùng mắt bằng Haar-like eye candidates; nếu không đủ ứng viên thì dùng ROI theo tỷ lệ khuôn mặt.
4. Xác định ROI miệng theo tỷ lệ khuôn mặt.
5. Trên từng ROI: Gaussian blur, Canny tự động, Otsu inverse threshold, morphology và connected components.
6. Lấy điểm đặc trưng Shi-Tomasi bằng `cv2.goodFeaturesToTrack`.
7. Kết hợp độ cao vùng tối, độ phân tán cạnh và độ phân tán điểm đặc trưng thành:
   - `eye_openness`: thấp hơn ngưỡng nghĩa là mắt nhắm.
   - `mouth_openness`: cao hơn ngưỡng nghĩa là miệng mở/ngáp.
8. Dùng số frame liên tiếp và tần suất theo phút để phát cảnh báo.

## Cài đặt

Yêu cầu Python 3.8-3.11 và webcam.

```bash
python -m venv venv
source venv/bin/activate       # Linux/macOS
# venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

Không cần tải `shape_predictor_68_face_landmarks.dat`. Hai model dlib cũ đã được xóa, giảm khoảng 95.77 MiB dữ liệu model trong repository.

## Chạy ứng dụng

```bash
# Giao diện Kivy
python src/main.py
# hoặc
python run.py

# Giao diện OpenCV
python src/main.py --opencv

# Lưu ảnh từng bước của pipeline
python src/main.py --opencv --save-pipeline

# Hiệu chỉnh ngưỡng độ mở mắt trong 5 giây
python src/main.py --calibrate
```

Phím trong OpenCV mode:

| Phím | Chức năng |
|---|---|
| `q` | Thoát |
| `p` | Bật/tắt lưu ảnh pipeline |

## Test

```bash
python -m unittest discover -s tests -v
```

Test bao gồm:

- Tách mắt mở/nhắm ở ba mức sáng tổng hợp.
- Tách miệng bình thường/ngáp ở ba mức sáng tổng hợp.
- Regression trên hai khuôn mặt thật trong `notebooks/test.png`.
- Bộ đếm frame liên tiếp cho cảnh báo nhắm mắt và xác nhận một lần ngáp.
- Kiểm tra confusion matrix của công cụ đánh giá.
- Test tích hợp benchmark ROI, baseline landmark và ba file kết quả.
- Chuẩn hóa frame/ROI ngoài biên và bỏ qua dữ liệu không hợp lệ.
- Tuần tự hóa nhiều thread dùng chung detector để bảo vệ Haar, DNN và eye cascade.

## So sánh với landmark cũ

### Đánh giá video

Chuẩn bị CSV ground truth `frame,eye_closed,yawn` và CSV landmark cũ
`frame,eye_closed_pred,yawn_pred`, hoặc raw `frame,ear,mar`.

Ví dụ định dạng nằm tại:

- `docs/evaluation_labels.example.csv`
- `docs/landmark_baseline.example.csv`

Chạy:

```bash
python -m src.evaluation.metrics \
  --video path/to/test.mp4 \
  --labels path/to/labels.csv \
  --landmark-baseline path/to/landmark_predictions.csv \
  --output evaluation_results
```

### Đánh giá ROI đã gán nhãn

Chế độ này bỏ qua face detector và phân tích trực tiếp bounding box đã biết. Manifest
có các cột `sample,image,x1,y1,x2,y2,eye_closed,yawn`; baseline landmark dùng
`sample` thay cho `frame`.

```bash
python -m src.evaluation.metrics \
  --roi-manifest docs/controlled_fixture/roi_manifest.csv \
  --landmark-baseline docs/controlled_landmark_baseline.csv \
  --output evaluation_results
```

Fixture đi kèm có 12 mẫu có kiểm soát: bốn tổ hợp mắt mở/nhắm và miệng
bình thường/ngáp ở ba mức sáng. Đây là test chức năng tổng hợp, không phải bằng
chứng về độ chính xác trên người thật.

Kết quả:

- `evaluation_results/predictions.csv`
- `evaluation_results/comparison.json`
- `evaluation_results/comparison.md`

Công cụ tính riêng accuracy, precision, recall, specificity, F1 và FPR cho nhắm mắt/ngáp. Báo cáo cũng ghi coverage mắt, miệng, feature hợp lệ và thời gian xử lý; chế độ video ghi thêm face-detection coverage. Baseline phải phủ đủ toàn bộ frame/mẫu được đánh giá để bảo đảm hai phương pháp dùng cùng tập nhãn. Recall/F1 được để `n/a` khi tập nhãn không có mẫu dương tính.

Xem kết quả tái đánh giá hiện tại tại `docs/edge_feature_reassessment.md`.

## Cấu hình chính

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `EYE_OPEN_THRESHOLD` | 0.20 | Nhỏ hơn ngưỡng được xem là mắt nhắm |
| `EYE_CLOSED_CONSEC_FRAMES` | 15 | Frame nhắm mắt liên tiếp để cảnh báo |
| `MOUTH_OPEN_THRESHOLD` | 0.35 | Lớn hơn ngưỡng được xem là miệng mở |
| `YAWN_CONSEC_FRAMES` | 5 | Frame miệng mở liên tiếp để tính một lần ngáp |
| `FEATURE_MIN_FACE_SIZE` | 80 | Kích thước mặt tối thiểu để phân tích trạng thái |
| `EYE_FEATURE_MIN_CONFIDENCE` | 0.25 | Confidence tối thiểu của vùng mắt |
| `MOUTH_FEATURE_MIN_CONFIDENCE` | 0.25 | Confidence tối thiểu của vùng miệng |
| `BLINK_PER_MINUTE_THRESHOLD` | 25 | Tần suất chớp mắt cảnh báo mệt mỏi |
| `YAWN_PER_MINUTE_THRESHOLD` | 3 | Tần suất ngáp cảnh báo mệt mỏi |

## Cấu trúc liên quan

```text
src/
  core/
    detector.py          Pipeline camera, trạng thái và cảnh báo
    facial_analyzer.py   Canny, Otsu, vùng tối, Shi-Tomasi
    feature_classifier.py Logic validity/ngưỡng dùng chung
    model_manager.py     Chỉ quản lý model OpenCV DNN face detector
  evaluation/
    controlled_fixture.py Sinh fixture chức năng có kiểm soát
    metrics.py           Benchmark với ground truth và baseline landmark
  ui/
    app.py
    screens/main_screen.py
tests/
  test_feature_detection.py
  test_evaluation_metrics.py
docs/
  edge_feature_reassessment.md
```

`notebooks/experiments.ipynb`, `docs/report.md` và `docs/report.docx` chứa thí nghiệm/báo cáo của pipeline landmark cũ và chỉ được giữ làm baseline lịch sử.
