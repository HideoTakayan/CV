import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Rescaling
from src import config

class DataLoader:
    def __init__(self, data_path=config.DATA_DIR, batch_size=config.BATCH_SIZE, image_size=config.INPUT_SHAPE[:2]):
        self.data_path = data_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.classes = []
        self.train_data = None
        self.test_data = None
        self.train_data_org = None
        self.test_data_org = None

    def load_data(self):
        if not os.path.exists(self.data_path) or not os.listdir(self.data_path):
            raise FileNotFoundError(f"Khong tim thay du lieu hoac thu muc rong: {self.data_path}")

        self.train_data_org = tf.keras.utils.image_dataset_from_directory(
            self.data_path,
            validation_split=0.2,
            subset="training",
            seed=42,
            shuffle=True,
            image_size=self.image_size,
            batch_size=self.batch_size
        )

        self.test_data_org = tf.keras.utils.image_dataset_from_directory(
            self.data_path,
            validation_split=0.2,
            subset="validation",
            seed=42,
            shuffle=True,
            image_size=self.image_size,
            batch_size=self.batch_size
        )

        self.classes = self.train_data_org.class_names

        rescale = Rescaling(1.0 / 255)

        self.train_data = (
            self.train_data_org
            .map(lambda x, y: (rescale(x), y), num_parallel_calls=config.AUTOTUNE)
            .cache()
            .prefetch(buffer_size=config.AUTOTUNE)
        )

        self.test_data = (
            self.test_data_org
            .map(lambda x, y: (rescale(x), y), num_parallel_calls=config.AUTOTUNE)
            .cache()
            .prefetch(buffer_size=config.AUTOTUNE)
        )

        return self  # ✅ FIXED: trả về chính đối tượng để gọi tiếp get_classes()

    def plot_samples(self):
        if not self.train_data_org:
            raise ValueError("Chua load du lieu. Goi load_data() truoc.")

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
        return self.classes
