import os

# ===== Cấu hình huấn luyện =====
INPUT_SHAPE = (224, 224, 3)
ACTIVATION_FUNCTION = 'relu'
OPTIMIZER = 'adam'
EPOCHS = 50          # Đề xuất tăng để mô hình học tốt hơn do ít dữ liệu
BATCH_SIZE = 4       # Batch nhỏ vì số lượng ảnh ít

# ===== Đường dẫn dữ liệu =====
DATA_DIR = 'images'  # Thư mục chứa các thư mục trái cây tên tiếng Việt (vd: 'chuối', 'xoài', ...)

# Nếu không có file zip, có thể để None hoặc xóa dòng này
ZIP_FILE_PATH = None

# Cùng 1 thư mục cho cả train và test (sẽ chia tự động 80/20)
TRAINING_DATA = DATA_DIR
TESTING_DATA = DATA_DIR

# ===== Tạo thư mục lưu kết quả nếu chưa có =====
os.makedirs('results', exist_ok=True)
os.makedirs('trained_models', exist_ok=True)

# ===== Đường dẫn lưu kết quả =====
SAMPLE_PLOT_PATH = os.path.join('results', 'sample_images.png')
MODEL_SAVE_PATH = os.path.join('trained_models', 'fruit_classifier_model.h5')
MODEL_HISTORY_PATH = os.path.join('trained_models', 'history.json')
MODEL_ARCHITECTURE_PATH = os.path.join('trained_models', 'architecture.json')
