# Chi tiết các cải tiến so với baseline

Repo này được nâng cấp từ baseline [CLIP-CAER](https://github.com/zgsfer/CLIP-CAER) với các chiến lược huấn luyện nâng cao để cải thiện hiệu suất, bao gồm **Expression-Aware Adapters (EAA)**, **Instance-Enhanced Classifiers (IEC)**, **Mutual Information (MI) Loss**, **Decorrelation (DC) Loss**, và các phương pháp xử lý mất cân bằng dữ liệu.

## Tổng quan kiến trúc mới

Kiến trúc nâng cao được xây dựng dựa trên sườn của CLIP-CAER, kết hợp thêm nhiều module mới để tạo ra một mô hình mạnh mẽ và chính xác hơn.

### 1. Backbone Thị giác Hai luồng (Dual-Stream)
Mô hình xử lý hai luồng hình ảnh riêng biệt:
- **Luồng Gương mặt (Face Stream):** Các vùng mặt được cắt (crop) để nắm bắt các biểu cảm chi tiết, tinh vi.
- **Luồng Ngữ cảnh (Context Stream):** Toàn bộ khung hình (full-frame) được sử dụng để nắm bắt bối cảnh và hành vi xung quanh (có thể bật/tắt bằng cờ `--crop-body`).

Cả hai luồng đều được xử lý bởi cùng một bộ mã hóa hình ảnh của CLIP (CLIP Visual Encoder).

### 2. EAA (Expression-Aware Adapter)
Để nắm bắt tốt hơn các chi tiết cảm xúc tinh vi trên gương mặt mà không làm mất đi khả năng tổng quát hóa của mô hình CLIP đã được huấn luyện trước, một module **Expression-Aware Adapter** gọn nhẹ được tích hợp vào luồng xử lý gương mặt.
- **Cách triển khai:** Một module adapter theo kiến trúc "bottleneck" được chèn vào sau bộ mã hóa hình ảnh của luồng gương mặt.
- **Trainable:** Chỉ các tham số của adapter được tinh chỉnh (fine-tune), trong khi phần lớn bộ mã hóa hình ảnh được giữ đông lạnh (frozen).

### 3. Mô hình hóa Thời gian và Kết hợp (Temporal Modeling and Fusion)
- **Bộ mã hóa thời gian (Temporal Encoders):** Chuỗi các đặc trưng (features) từ mỗi khung hình của cả hai luồng được đưa qua các mô hình Temporal Transformer riêng biệt để nắm bắt mối quan hệ theo thời gian.
- **Kết hợp (Fusion):** Các embedding cấp độ video cho luồng mặt (`z_f`) và luồng ngữ cảnh (`z_c`) sau đó được kết hợp bằng cách ghép nối (concatenation) và qua một lớp chiếu (linear projection) để tạo ra một embedding thị giác cuối cùng, hợp nhất là `z`.

### 4. Prompt Hai góc nhìn (Dual-View Prompting) & MI Loss
Để ngăn các prompt có thể học (learnable prompts) bị overfitting và đi chệch khỏi ngữ nghĩa mong muốn, một chiến lược prompt hai góc nhìn được sử dụng.
- **Góc nhìn "Mô tả" thủ công (Hand-Crafted "Descriptive" View):** Các prompt mô tả chi tiết, giàu thông tin cho mỗi lớp cảm xúc, tập trung vào các biểu hiện vi mô trên gương mặt giống như Action Unit (AU) (ví dụ: "A person with furrowed eyebrows and a puzzled gaze"). Các prompt này là cố định.
- **Góc nhìn "Mềm" có thể học (Learnable "Soft" View):** Các vector ngữ cảnh theo kiểu CoOp có thể được tối ưu trong quá trình huấn luyện.
- **Mutual Information (MI) Loss:** Một hàm loss dựa trên InfoNCE được sử dụng để tối đa hóa thông tin tương hỗ (mutual information) giữa các embedding của prompt mô tả và prompt mềm (`t_desc` và `t_soft`), đảm bảo các prompt học được luôn bám sát ngữ nghĩa gốc.

### 5. IEC (Instance-Enhanced Classifier)
Để làm cho bộ phân loại dựa trên văn bản có khả năng thích ứng tốt hơn với các đặc trưng thị giác của từng mẫu video cụ thể, module IEC được sử dụng.
- **Cách triển khai:** Thay vì dùng một mẫu văn bản (text prototype) tĩnh cho mỗi lớp, một prototype động, được "tăng cường" theo từng mẫu, được tạo ra bằng cách sử dụng phép nội suy tuyến tính cầu (**Spherical Linear Interpolation - Slerp**).
- **Công thức:** `t_mix(k) = slerp(t_desc(k), z, λ_slerp)`, trong đó `t_desc(k)` là prompt mô tả cho lớp `k`, `z` là embedding thị giác của mẫu video, và `λ_slerp` là một trọng số có thể điều chỉnh.
- Việc phân loại cuối cùng được thực hiện bằng cách tính toán độ tương đồng giữa embedding thị giác `z` và các text prototype đã được trộn `t_mix` này.

### 6. Chiến lược Huấn luyện Ổn định (Stable Training Strategy)
Để chống lại hiện tượng "sập" mô hình (model collapse) khi huấn luyện các kiến trúc phức tạp, một chiến lược huấn luyện end-to-end an toàn được áp dụng:
`L_total = L_classification + (weight_mi * L_mi) + (weight_dc * L_dc)`
- **Đóng băng CLIP Encoder:** Bộ mã hóa hình ảnh gốc của CLIP (`image_encoder`) được đóng băng hoàn toàn (`lr=0`) để tạo ra một backbone trích xuất đặc trưng ổn định, tránh nguy cơ bất ổn lớn nhất.
- **Learning Rate Thấp và Thống nhất:** Sử dụng một learning rate chung và rất thấp (ví dụ `1e-5`) cho tất cả các module mới (Adapter, Prompt Learner, Temporal) để đảm bảo không có thành phần nào học quá nhanh và gây bất ổn.
- **Loss Warmup Kéo dài:** Tăng thời gian "khởi động" cho MI và DC loss (ví dụ: `--mi-warmup 5`). Trong các epoch đầu, mô hình sẽ chỉ tập trung vào hàm loss phân loại chính, giúp nó hội tụ một cách ổn định trước khi các hàm loss phức tạp hơn được thêm vào.
- **Gradient Clipping:** Luôn bật để giới hạn độ lớn tối đa của gradient, hoạt động như một "lưới an toàn" để ngăn các cập nhật trọng số đột biến gây sập mô hình.

### 7. Logging chi tiết và Biểu đồ
Quá trình huấn luyện giờ đây sẽ in ra và lưu lại các thông tin chi tiết hơn sau mỗi epoch:
- **Biểu đồ:** Một file `log.png` được tạo ra, vẽ biểu đồ của **Train/Valid Loss**, **Train/Valid WAR**, và **Train/Valid UAR** qua các epoch.
- **Thông số Console:**
    -   `Train WAR`, `Train UAR` của epoch hiện tại.
    -   `Valid WAR`, `Valid UAR` của epoch hiện tại.
    -   `Best Train/Valid WAR/UAR` tốt nhất từ đầu đến giờ.
    -   Ma trận nhầm lẫn (Confusion Matrix) của tập validation sau mỗi epoch.

## Hướng dẫn Sử dụng

Quá trình huấn luyện có thể được tùy chỉnh với các tham số dòng lệnh mới để điều khiển các tính năng nâng cao.

### Local (MacOS)
```bash
bash train_local.sh
```
File này được cấu hình để chạy với `--gpu mps`. Bạn cần chỉnh sửa đường dẫn trong file cho phù hợp với máy của mình.

### Google Colab / Linux
```bash
bash train_colab.sh
```
File này được cấu hình để chạy với GPU CUDA (`--gpu 0`).

### Các tham số chính
- `--lr`: Learning rate chung cho các module chính (khuyến nghị: `1e-5`).
- `--lr-image-encoder`: Learning rate cho bộ mã hóa ảnh (khuyến nghị: `0.0` để đóng băng).
- `--lr-prompt-learner`: Learning rate cho prompt learner (khuyến nghị: `1e-5`).
- `--lr-adapter`: Learning rate cho adapter (khuyến nghị: `1e-5`).
- `--lambda_mi`: Trọng số cho Mutual Information loss (ví dụ: `0.7`).
- `--lambda_dc`: Trọng số cho Decorrelation loss (ví dụ: `1.2`).
- `--mi-warmup`, `--dc-warmup` (int): Số epoch "khởi động" cho loss (khuyến nghị: `5`).
- `--grad-clip` (float): Giá trị giới hạn cho gradient clipping (khuyến nghị: `1.0`).
- `--use-amp`: (cờ) Bật để sử dụng Automatic Mixed Precision (chỉ hoạt động trên CUDA).
- `--crop-body`: (cờ) Bật để cắt vùng body thay vì dùng toàn bộ khung hình.
- `--use-weighted-sampler`: (cờ) Bật để sử dụng `WeightedRandomSampler` xử lý mất cân bằng dữ liệu.
- `--logit-adj`: (cờ) Bật để sử dụng Logit Adjustment.