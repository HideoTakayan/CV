import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.callbacks import EarlyStopping
import config

class DataLoader:
    """
    Class for loading, preprocessing and visualizing image data.
    """

    def __init__(self):
        self.classes = []
        self.train_data = None
        self.test_data = None
        self.train_data_org = None
        self.test_data_org = None

    def load_data(self):
        """
        Load and preprocess dataset from directory.
        """
        data_path = config.DATA_DIR

        if not os.path.exists(data_path) or not os.listdir(data_path):
            raise FileNotFoundError(f"Dataset not found at: {data_path}. Please check your image folder.")

        print(f"✅ Found dataset at: {data_path}")
        print("📁 Class folders:", sorted(os.listdir(data_path)))

        # Load training and validation sets using split
        self.train_data_org = tf.keras.utils.image_dataset_from_directory(
            data_path,
            validation_split=0.2,
            subset="training",
            seed=42,
            shuffle=True,
            image_size=config.INPUT_SHAPE[:2],
            batch_size=config.BATCH_SIZE
        )

        self.test_data_org = tf.keras.utils.image_dataset_from_directory(
            data_path,
            validation_split=0.2,
            subset="validation",
            seed=42,
            shuffle=True,
            image_size=config.INPUT_SHAPE[:2],
            batch_size=config.BATCH_SIZE
        )

        self.classes = self.train_data_org.class_names

        # Normalize images to range [0, 1]
        self.train_data = self.train_data_org.map(lambda x, y: (Rescaling(1./255)(x), y))
        self.test_data = self.test_data_org.map(lambda x, y: (Rescaling(1./255)(x), y))

    def plot_samples(self):
        """
        Display sample training images (9 random images).
        """
        if not self.train_data_org:
            raise ValueError("Dataset not loaded. Call load_data() before plotting samples.")

        print("📷 Displaying sample training images...")
        os.makedirs('results/plots', exist_ok=True)

        plt.figure(figsize=(10, 10))
        for images, labels in self.train_data_org.take(1):
            for i in range(min(9, len(images))):
                plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(self.classes[labels[i]])
                plt.axis("off")
        plt.tight_layout()
        plt.savefig(config.SAMPLE_PLOT_PATH)
        plt.show()

    def get_classes(self):
        """
        Return list of class names.
        """
        return self.classes


class call_back:
    """
    Callback configuration for model training.
    """
    def __init__(self):
        self.callback = None

    def get_callbacks(self):
        self.callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.01,
            patience=3,
            verbose=1,
            restore_best_weights=True
        )
        return self.callback


# Test when running this file directly
if __name__ == "__main__":
    loader = DataLoader()
    loader.load_data()
    loader.plot_samples()
