import os

# Lấy thư mục gốc của project (1 cấp trên thư mục src/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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
