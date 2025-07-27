# Phân Loại Ảnh Trái Cây

## Tổng Quan

Dự án này triển khai một giải pháp phân loại ảnh tiên tiến sử dụng mô hình MobileNetV2 đã được huấn luyện trước để phân loại chính xác hình ảnh các loại trái cây. Giải pháp bao gồm xử lý dữ liệu toàn diện, huấn luyện mô hình và khả năng dự đoán thời gian thực.

## Cấu Trúc Dự Án

```
CV/
│
├── data/
│   └── MY_data/
│       ├── train/           # Ảnh huấn luyện
│       ├── test/            # Ảnh kiểm thử
│       └── prediction/      # Ảnh để dự đoán
│
├── src/
│   ├── preprocessing.py     # Xử lý và trực quan hóa dữ liệu
│   ├── model_training.py    # Script huấn luyện mô hình
│   ├── classify.py          # Script dự đoán hình ảnh
│   └── config.py            # Cấu hình dự án tập trung
│
├── results/                 # Thư mục lưu kết quả
│   ├── plots/
│   │   └── sample_plot.png
│   ├── model_architecture.json
│   └── classifier_history.json
│
├── trained_models/
│   └── fruit_classifier_model.h5
│
├── web_app/
│   ├── app.py               # Backend Flask
│   ├── templates/           # Giao diện HTML
│   └── static/
│       ├── uploads/         # Ảnh người dùng upload
|       └── style.css        # Làm đẹp giao diện web
│
├── requirements.txt         # Danh sách thư viện cần cài
└── README.md                # Tài liệu hướng dẫn dự án

```

## Tính Năng Nổi Bật

* **Mô hình học sâu**: Sử dụng MobileNetV2 được huấn luyện trước cho bài toán phân loại ảnh trái cây.
* **Tiền xử lý dữ liệu nâng cao**: Chuẩn hóa ảnh đầu vào, tăng cường dữ liệu.
* **Tối ưu hóa quá trình huấn luyện**: 
  - Callback EarlyStopping
  - Theo dõi hiệu suất theo từng epoch
* **Trực quan hóa dữ liệu**: Vẽ mẫu dữ liệu và lịch sử huấn luyện.
* **Dự đoán linh hoạt**: Dự đoán ảnh thời gian thực từ dòng lệnh hoặc qua giao diện web.
* **Triển khai web**: Tích hợp Flask để đưa mô hình lên giao diện web – cho phép người dùng upload ảnh và nhận kết quả phân loại.

## Cài Đặt

### Yêu Cầu Hệ Thống

* Python 3.8 trở lên
* Trình quản lý gói `pip`

### Các Bước Thiết Lập

1. **Tải mã nguồn**
   ```bash
   git clone https://github.com/HideoTakayan/CV.git
   cd CV
   ```

2. **Cài đặt thư viện**
   ```bash
   pip install -r requirements.txt
   ```

3. **Chuẩn bị dữ liệu**
   * Dữ liệu đã được chuẩn bị sẵn trong thư mục data/
   * Bạn không cần tải thêm hay giải nén gì cả
   * Có thể sử dụng trực tiếp để huấn luyện và chạy web/app

## Sử dụng:

### 1. Huấn luyện mô hình

```bash
python src/model_training.py
```

### 2. Dự đoán ảnh mới

```bash
python src/classify.py
```
**Lưu ý**: Cập nhật biến `demo_anh` trong file `classify.py` thành đường dẫn tới ảnh bạn muốn dự đoán.

### 3. Trực quan hóa tập dữ liệu

```bash
python src/preprocessing.py
```
### 4. Chạy ứng dụng web

```bash
python web_app/app.py
```
Sau khi chạy, mở trình duyệt và truy cập http://127.0.0.1:5000 để sử dụng giao diện web phân loại trái cây.  

## Cấu hình dự án

Tệp  `config.py` dùng để quản lý tập trung toàn bộ cấu hình của dự án:

```python
# Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAINING_DATA = os.path.join(DATA_DIR, "MY_data", "train")
TESTING_DATA = os.path.join(DATA_DIR, "MY_data", "test")
PREDICTION_DATA = os.path.join(DATA_DIR, "MY_data", "predict")

MODEL_SAVE_PATH = os.path.join(BASE_DIR, "trained_models", "fruit_classifier_model.h5")
MODEL_HISTORY_PATH = os.path.join(BASE_DIR, "results", "classifier_history.json")
MODEL_ARCHITECTURE_PATH = os.path.join(BASE_DIR, "results", "model_architecture.json")
SAMPLE_PLOT_PATH = os.path.join(BASE_DIR, "results", "plots", "sample_plot.png")


# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 30
OPTIMIZER = "adam"
SHUFFLE = True
SEED = 42

# Model Parameters
INPUT_SHAPE = (224, 224, 3)
ACTIVATION_FUNCTION = "relu"
```

## Kết Quả Huấn Luyện

* **Mô hình đã huấn luyện**: `fruit_classifier_model.h5`  
* **Kiến trúc mô hình**: `results/model_architecture.json`  
* **Chỉ số đánh giá hiệu năng**: 
  - Độ chính xác trên tập huấn luyện (Training Accuracy)  
  - Độ chính xác trên tập kiểm thử (Validation Accuracy)  


## Định Hướng Phát Triển Tương Lai  
🔍 Về mô hình phân loại  
- [ ] Mở rộng số lượng và chủng loại trái cây được nhận diện
- [ ] Tinh chỉnh kiến trúc mô hình để nâng cao độ chính xác
- [ ] Áp dụng các kỹ thuật tăng cường dữ liệu nâng cao
- [ ] Tích hợp tính năng giải thích dự đoán của mô hình (Model Interpretability)

🌐 Về giao diện và trải nghiệm web  
- [ ] Thiết kế giao diện người dùng (UI) hiện đại, thân thiện với thiết bị di động  
- [ ] Cho phép chụp ảnh trực tiếp từ webcam để phân loại  
- [ ] Hiển thị biểu đồ tỷ lệ dự đoán cho từng loại trái cây  
- [ ] Thêm lịch sử dự đoán và thống kê người dùng  



## Tài liệu tham khảo

1. [Fruit Image Classification](https://github.com/Nirikshan95/FruitClassifier)
2. MobileNetV2 Research Paper
3. Dataset (Quên nguồn)
## Đóng Góp & Góp Ý
Chúng tôi luôn hoan nghênh mọi đóng góp từ cộng đồng!
