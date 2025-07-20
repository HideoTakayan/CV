import os
import tensorflow as tf

INPUT_SHAPE = (224, 224, 3)
ACTIVATION_FUNCTION = 'relu'
LEARNING_RATE = 0.001
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
EPOCHS = 35
BATCH_SIZE = 16

# ===== Dữ liệu =====
DATA_DIR = 'images'
TRAINING_DATA = DATA_DIR
TESTING_DATA = DATA_DIR
ZIP_FILE_PATH = None

# ===== Lưu kết quả =====
os.makedirs('results/plots', exist_ok=True)
os.makedirs('trained_models', exist_ok=True)

SAMPLE_PLOT_PATH = os.path.join('results', 'sample_images.png')
MODEL_SAVE_PATH = os.path.join('trained_models', 'fruit_classifier_model')
MODEL_HISTORY_PATH = os.path.join('trained_models', 'history.json')
MODEL_ARCHITECTURE_PATH = os.path.join('trained_models', 'architecture.json')
LABEL_MAP_PATH = os.path.join('trained_models', 'label_map.json')

AUTOTUNE = tf.data.AUTOTUNE
