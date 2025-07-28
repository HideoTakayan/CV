import os

# Lấy thư mục gốc của project (1 cấp trên thư mục src/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Đường dẫn dữ liệu
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAINING_DATA = os.path.join(DATA_DIR, "MY_data", "train")       # Dữ liệu huấn luyện
TESTING_DATA = os.path.join(DATA_DIR, "MY_data", "test")         # Dữ liệu kiểm tra
PREDICTION_DATA = os.path.join(DATA_DIR, "MY_data", "predict")   # Dữ liệu dự đoán

# Đường dẫn lưu mô hình và kết quả
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "trained_models", "fruit_classifier_model.h5")      # Mô hình đã huấn luyện
MODEL_HISTORY_PATH = os.path.join(BASE_DIR, "results", "classifier_history.json")            # Lịch sử huấn luyện
MODEL_ARCHITECTURE_PATH = os.path.join(BASE_DIR, "results", "model_architecture.json")       # Kiến trúc mô hình
SAMPLE_PLOT_PATH = os.path.join(BASE_DIR, "results", "plots", "sample_plot.png")             # Biểu đồ mẫu

# Các siêu tham số huấn luyện
BATCH_SIZE = 64
EPOCHS = 30
OPTIMIZER = "adam"
SHUFFLE = True
SEED = 42

# Tham số của mô hình
INPUT_SHAPE = (224, 224, 3)      # Kích thước đầu vào ảnh
ACTIVATION_FUNCTION = "relu"     # Hàm kích hoạt cho các lớp ẩn
