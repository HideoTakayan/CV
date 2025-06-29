import os
import tensorflow as tf

# ===== Cấu hình huấn luyện =====
INPUT_SHAPE = (224, 224, 3)
ACTIVATION_FUNCTION = 'relu'

# Learning rate phù hợp cho dữ liệu vừa phải
LEARNING_RATE = 0.001
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# Đào tạo đủ lâu để học tốt (EarlyStopping sẽ cắt sớm nếu cần)
EPOCHS = 50

# BATCH_SIZE tối ưu cho máy có RAM 16GB
BATCH_SIZE = 32  # Nếu máy yếu, bạn có thể giảm xuống 16

# ===== Đường dẫn dữ liệu =====
DATA_DIR = 'images'  # Thư mục chứa ảnh, mỗi thư mục con là 1 loại trái cây

# Không dùng file zip (nếu có thì chỉ định ở đây)
ZIP_FILE_PATH = None

# Dùng chung thư mục cho cả training/testing (80/20 chia trong code)
TRAINING_DATA = DATA_DIR
TESTING_DATA = DATA_DIR

# ===== Tạo thư mục lưu kết quả nếu chưa tồn tại =====
os.makedirs('results', exist_ok=True)
os.makedirs('trained_models', exist_ok=True)

# ===== Đường dẫn lưu kết quả =====
SAMPLE_PLOT_PATH = os.path.join('results', 'sample_images.png')
MODEL_SAVE_PATH = os.path.join('trained_models', 'fruit_classifier_model.h5')
MODEL_HISTORY_PATH = os.path.join('trained_models', 'history.json')
MODEL_ARCHITECTURE_PATH = os.path.join('trained_models', 'architecture.json')
LABEL_MAP_PATH = os.path.join('trained_models', 'label_map.json')

# ===== Dùng để tối ưu hiệu suất khi load ảnh =====
AUTOTUNE = tf.data.AUTOTUNE
