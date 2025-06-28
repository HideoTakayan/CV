import os
import tensorflow as tf

# ===== Training configuration =====
INPUT_SHAPE = (224, 224, 3)
ACTIVATION_FUNCTION = 'relu'
LEARNING_RATE = 0.0005
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
EPOCHS = 50
BATCH_SIZE = 4  # Nhỏ vì số ảnh ít

# ===== Dataset paths =====
DATA_DIR = 'images'  # Folder chứa các thư mục trái cây (đặt tên không dấu để tránh lỗi encode)

# If using zip, set path here (else keep None)
ZIP_FILE_PATH = None

# Split will be handled inside code (80/20)
TRAINING_DATA = DATA_DIR
TESTING_DATA = DATA_DIR

# ===== Create folders if not exist =====
os.makedirs('results', exist_ok=True)
os.makedirs('trained_models', exist_ok=True)

# ===== Paths to save outputs =====
SAMPLE_PLOT_PATH = os.path.join('results', 'sample_images.png')
MODEL_SAVE_PATH = os.path.join('trained_models', 'fruit_classifier_model.h5')
MODEL_HISTORY_PATH = os.path.join('trained_models', 'history.json')
MODEL_ARCHITECTURE_PATH = os.path.join('trained_models', 'architecture.json')
LABEL_MAP_PATH = os.path.join('trained_models', 'label_map.json')
