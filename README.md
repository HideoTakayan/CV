# Phân Loại Ảnh Trái Cây

## Tổng Quan

Dự án này triển khai một giải pháp phân loại ảnh tiên tiến sử dụng mô hình MobileNetV2 đã được huấn luyện trước để phân loại chính xác hình ảnh các loại trái cây. Giải pháp bao gồm xử lý dữ liệu toàn diện, huấn luyện mô hình và khả năng dự đoán thời gian thực.

## Cấu Trúc Dự Án


```
CV/
│
├── data/
│ ├── archive (6).zip
│ └── MY_data
│ ├── train/  # Ảnh huấn luyện
│ ├── test/ # Ảnh kiểm thử
│ └── prediction/ # Ảnh để dự đoán
│
├── src/
│ ├── preprocessing.py # Xử lý và trực quan hóa dữ liệu
│ ├── model_training.py # Script huấn luyện mô hình
│ ├── classify.py # Script dự đoán hình ảnh
│ └── config.py # Cấu hình dự án tập trung
│
├── results/ # Thư mục lưu kết quả
│ ├── plots/
│ │ └── sample_plot.png
│ ├── model_architecture.json
│ └── classifier_history.json
│
├── trained models/
│ └── fruit_classifier_model.h5
│
├── web_app/
│ ├── app.py # Backend Flask
│ ├── templates/ # Giao diện HTML
│ └── static/uploads/ # Ảnh người dùng upload
│
├── requirements.txt
└── README.md


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
   git clone https://github.com/Nirikshan95/FruitClassifier.git
   cd FruitClassifier
   ```

2. **Cài đặt thư viện**
   ```bash
   pip install -r requirements.txt
   ```

3. **Chuẩn bị dữ liệu**
   * Download "Fruit Classification (10 Class)" dataset from Kaggle
   * Place `archive (6).zip` in the `data/` directory
   * The preprocessing script will handle extraction

## Usage

### 1. Train the Model

```bash
python src/model_training.py
```

### 2. Predict New Images

```bash
python src/classify.py
```
*Note: Update `demo_img` path in `classify.py` with your image*

### 3. Visualize Dataset

```bash
python src/preprocessing.py
```

## Configuration

The `config.py` file centralizes project configurations:

```python
# Paths
DATA_DIR = "./data"
ZIP_FILE_PATH = "./data/archive (6).zip"
TRAINING_DATA = "./data/MY_data/train/"
TESTING_DATA = "./data/MY_data/test/"
PREDICTION_DATA = "./data/MY_data/prediction/"
MODEL_SAVE_PATH = "./trained models/fruit_classifier_model.h5"
MODEL_HISTORY_PATH = "./results/classifier_history.json"
MODEL_ARCHITECTURE_PATH="./results/model_architecture.json"
SAMPLE_PLOT_PATH="./results/plots/sample_plot.png"

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
OPTIMIZER = "adam"

# Model Parameters
INPUT_SHAPE = (224,224,3)
ACTIVATION_FUNCTION = "relu"
```

## Results

* **Trained Model**: `fruit_classifier_model.h5`
* **Model Architecture**: `results/model_architecture.json`
* **Performance Metrics**: 
  - Training accuracy
  - Validation accuracy

## Future Roadmap

- [ ] Expand fruit class diversity
- [ ] Fine-tune model architecture
- [ ] Implement advanced data augmentation
- [ ] Add model interpretability features

## References

1. [Kaggle Fruit Classification Dataset](https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class)
2. MobileNetV2 Research Paper

## Contributions

Contributions, issues, and feature requests are welcome!
