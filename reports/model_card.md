# MODEL CARD - BANK MARKETING

## 1. Mục đích

Mô hình được xây dựng để dự đoán khả năng khách hàng đăng ký tiền gửi có kỳ hạn của ngân hàng.

Mô hình trả về:

- Xác suất khách hàng đăng ký.
- Kết quả dự đoán có/không.
- Kết quả phụ thuộc vào ngưỡng xác suất được lựa chọn.

---

## 2. Dữ liệu

Bộ dữ liệu Bank Marketing được sử dụng để huấn luyện và đánh giá mô hình.

Biến mục tiêu:

- `y = yes`: khách hàng đăng ký.
- `y = no`: khách hàng không đăng ký.

Biến `duration` được loại bỏ vì có nguy cơ gây data leakage: thời lượng cuộc gọi chỉ được biết sau hoặc trong quá trình liên hệ khách hàng.

---

## 3. Chia dữ liệu

Dữ liệu được chia thành:

- Train: 70%
- Validation: 15%
- Test: 15%

Sử dụng:

- `random_state = 42`
- `stratify = y`

Validation được sử dụng để so sánh threshold.

Test chỉ được sử dụng cho đánh giá cuối cùng.

---

## 4. Tiền xử lý

### Biến số

Các biến số:

- age
- balance
- day
- campaign
- pdays
- previous

Được chuẩn hóa bằng `StandardScaler`.

### Biến phân loại

Các biến:

- job
- marital
- education
- default
- housing
- loan
- contact
- month
- poutcome

Được mã hóa bằng `OneHotEncoder`.

---

## 5. Mô hình

Mô hình sử dụng:

**Logistic Regression**

Mô hình được đặt trong Pipeline cùng với bước preprocessing.

---

## 6. Chỉ số đánh giá

Các chỉ số chính:

- PR-AUC
- Precision
- Recall
- F1-score
- Accuracy
- Confusion Matrix

PR-AUC được ưu tiên vì dữ liệu có sự mất cân bằng giữa hai lớp.

---

## 7. Threshold

Threshold được đánh giá trên validation set.

Các threshold được thử nghiệm:

- 0.20
- 0.30
- 0.40
- 0.50
- 0.60
- 0.70

Threshold cuối cùng được lựa chọn dựa trên validation, sau đó được giữ cố định khi đánh giá test.

---

## 8. Diễn giải lỗi

### False Positive

Mô hình dự đoán khách hàng có khả năng đăng ký nhưng thực tế khách hàng không đăng ký.

### False Negative

Mô hình dự đoán khách hàng không đăng ký nhưng thực tế khách hàng có đăng ký.

FP và FN được phân tích theo:

- Nhóm tuổi.
- Nghề nghiệp.

---

## 9. Giới hạn

- Mô hình chỉ phản ánh dữ liệu được sử dụng trong bộ Bank Marketing.
- Hiệu quả có thể thay đổi trên dữ liệu ngân hàng khác.
- Threshold ảnh hưởng trực tiếp đến Precision và Recall.
- Không nên sử dụng mô hình như quyết định tự động duy nhất trong hoạt động kinh doanh thực tế.
- Dữ liệu có thể chứa các nhóm `unknown`, vì vậy kết quả cần được diễn giải phù hợp.

---

## 10. Khả năng tái lập

Các thành phần chính sử dụng:

- `random_state = 42`
- Train/Validation/Test = 70/15/15
- Stratified split
- Pipeline preprocessing
- Logistic Regression

Kết quả được lưu trong:

`reports/evaluation.json`

Các thí nghiệm threshold được lưu trong:

`reports/threshold_results.csv`
## 11. Kết quả thực nghiệm

### Validation

- PR-AUC: 0.4201
- Threshold được lựa chọn: 0.20

Threshold được lựa chọn dựa trên F1-score trên tập Validation.

### Test

- PR-AUC: 0.4189
- Accuracy: 0.8730
- Precision: 0.4543
- Recall: 0.4262
- F1-score: 0.4398

Confusion Matrix:

- TN: 5583
- FP: 406
- FN: 455
- TP: 338

### Baseline

Baseline dự đoán toàn bộ mẫu là lớp âm tính:

- Accuracy: 0.8829
- Precision: 0
- Recall: 0
- F1-score: 0

Mặc dù baseline có Accuracy cao hơn mô hình, baseline không phát hiện được bất kỳ trường hợp dương tính nào. Vì vậy PR-AUC, Precision, Recall và F1 được sử dụng để đánh giá khả năng phát hiện khách hàng có khả năng đăng ký.