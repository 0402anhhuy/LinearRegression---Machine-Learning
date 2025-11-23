# 📘 LINEAR REGRESSION

### 1. Định nghĩa & Bản chất

**Linear Regression** là thuật toán học máy thuộc nhóm **Học có giám sát (Supervised Learning)**

- **Mục tiêu:** Tìm ra một mối quan hệ tuyến tính (đường thẳng hoặc mặt phẳng) mô tả tốt nhất sự phụ thuộc giữa biến đầu vào ($X$) và biến kết quả ($y$)

- **Bản chất:** Tìm một đường thẳng sao cho khoảng cách từ các điểm dữ liệu thực tế đến đường thẳng đó là nhỏ nhất

[Image of linear regression best fit line]

### 2. Các Thành phần & Công thức Toán học

Để xây dựng mô hình, ta cần 3 thành phần chính: **Hàm giả thuyết**, **Hàm mất mát** và **Thuật toán tối ưu**

#### A. Hàm Giả Thuyết (Hypothesis Function)

$$\hat{y} = wx + b$$

- **$\hat{y}$ (y-hat):** Giá trị máy dự đoán
- **$x$ (Input):** Dữ liệu đầu vào (Feature)
- **$w$ (Weight - Trọng số):** Độ dốc của đường thẳng. Quyết định mức độ ảnh hưởng của $x$ lên $y$
- **$b$ (Bias - Hệ số tự do):** Điểm cắt trục tung. Giúp đường thẳng tịnh tiến lên xuống mà không phụ thuộc $x$

_(Nếu có nhiều biến đầu vào $x_1, x_2...$, công thức là: $\hat{y} = w_1x_1 + w_2x_2 + ... + b$)_

#### B. Hàm Mất Mát (Loss Function - MSE)

Dùng **Mean Squared Error (MSE)** để đánh giá mô hình đang thực hiện tốt hay dở

$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

- **$m$:** Tổng số mẫu dữ liệu
- **$y^{(i)}$:** Giá trị thực tế (nhãn đúng) của mẫu thứ $i$
- **$\hat{y}^{(i)}$:** Giá trị máy vừa đoán cho mẫu thứ $i$
- **Bình phương $(\dots)^2$:** Giúp triệt tiêu dấu âm và "trừng phạt" nặng các sai số lớn (Outliers)
- **$\frac{1}{2m}$:** Chia trung bình. Số $2$ ở mẫu số giúp triệt tiêu số mũ $2$ khi tính đạo hàm

#### C. Thuật toán Tối ưu (Gradient Descent)

Dùng đạo hàm để biết hướng "xuống dốc" nhằm giảm thiểu sai số $J$

**Quy tắc cập nhật (Vòng lặp):**

$$w_{new} = w_{old} - \alpha \cdot \frac{\partial J}{\partial w}$$
$$b_{new} = b_{old} - \alpha \cdot \frac{\partial J}{\partial b}$$

- **$\alpha$ (Learning Rate):** Tốc độ học
  - Lớn quá: Bước nhảy dài, dễ trượt qua đáy
  - Nhỏ quá: Học rất chậm
- **$\frac{\partial J}{\partial w}, \frac{\partial J}{\partial b}$ (Gradient):** Đạo hàm riêng, cho biết hướng dốc

**Công thức Gradient cụ thể (khi đã đạo hàm xong):**

$$dw = \frac{1}{m} \sum (\hat{y} - y) \cdot x$$
$$db = \frac{1}{m} \sum (\hat{y} - y)$$

### 3. Ví dụ Tính toán

Giả sử dữ liệu có 1 mẫu duy nhất: **Input $x=2$, Output thực tế $y=10$**

- Khởi tạo: $w=3, b=1$. Learning rate $\alpha = 0.1$

**Bước 1: Dự đoán (Forward Pass)**
$$\hat{y} = w \cdot x + b = 3 \cdot 2 + 1 = 7$$

**Bước 2: Tính sai số (Loss)**
- Sai số $e = \hat{y} - y = 7 - 10 = -3$
$$MSE = \frac{1}{2} (-3)^2 = 4.5$$

**Bước 3: Tính Gradient (Đạo hàm)**

- $dw = (\hat{y} - y) \cdot x = (-3) \cdot 2 = -6$
- $db = (\hat{y} - y) = -3$

**Bước 4: Cập nhật tham số (Backward Pass)**

- $w_{mới} = 3 - 0.1 \cdot (-6) = 3 + 0.6 = 3.6$
- $b_{mới} = 1 - 0.1 \cdot (-3) = 1 + 0.3 = 1.3$

$\rightarrow$ **Kết quả:** Sau 1 bước học, $w$ tăng từ 3 lên 3.6, $b$ tăng từ 1 lên 1.3. Dự đoán lần sau sẽ là $3.6(2) + 1.3 = 8.5$ (Gần với 10 hơn so với số 7 ban đầu)

### 4. Xây dựng mô hình

[Image of machine learning workflow steps]

1.  **Thu thập & Tải dữ liệu:**

    - Đọc file (CSV, Excel...)
    - Xác định đâu là Feature (X), đâu là Target (y)

2.  **Tiền xử lý dữ liệu (Preprocessing):**

    - **Làm sạch:** Xử lý dữ liệu thiếu (NaN), dữ liệu rác
    - **Chuẩn hóa (Normalization/Standardization):** Đưa dữ liệu về cùng một khoảng (thường dùng Mean/Std). Nếu không làm bước này, thuật toán Gradient Descent sẽ rất khó hội tụ
    - **Chia tập dữ liệu:** Train set và Test set

3.  **Thiết kế Mô hình:**

    - Chọn thuật toán: Linear Regression (`nn.Linear`)
    - Xác định Input size (số lượng feature) và Output size (thường là 1)

4.  **Thiết lập huấn luyện:**

    - Chọn Loss Function: `MSELoss`
    - Chọn Optimizer: `SGD` hoặc `Adam`
    - Chọn Hyperparameters: Learning rate, Epochs

5.  **Vòng lặp huấn luyện (Training Loop):**

    - Dự đoán (Forward) $\rightarrow$ Tính Loss $\rightarrow$ Đạo hàm (Backward) $\rightarrow$ Cập nhật (Optimizer Step)

6.  **Đánh giá & Dự đoán:**
    - Dùng tập Test để kiểm tra độ chính xác
    - Khi dự đoán thực tế: **Phải chuẩn hóa input mới** theo quy tắc của tập train, sau đó **giải chuẩn hóa output** để ra kết quả cuối cùng

### 5. Các lỗi thường gặp & Lưu ý

1.  **Underfitting (Chưa học được gì):**

    - _Biểu hiện:_ Đường dự đoán nằm ngang, Loss không giảm
    - _Lý do:_ Ít dữ liệu, chọn sai feature (feature rác), hoặc model quá đơn giản
    - _Khắc phục:_ Thêm dữ liệu, chọn feature tốt hơn (như ví dụ đổi từ `bedrooms` sang `income`), tăng Epochs, tăng Learning rate

2.  **Quên chuẩn hóa (Normalization):**

    - Dẫn đến việc $w$ và $b$ bị lệch lạc, Loss nhảy lung tung (`NaN` hoặc vô cực)

3.  **Data Leakage (Rò rỉ dữ liệu):**
    - Lấy thông tin của tập Test để tính toán cho tập Train (ví dụ: tính Mean/Std trên toàn bộ dữ liệu trước khi chia tập). Phải chia tập trước, rồi mới tính Mean/Std trên tập Train
