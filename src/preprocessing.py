import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.callbacks import EarlyStopping
import config


class DataLoader:
    """
    Class dung de load, tien xu ly va ve du lieu anh trai cay.
    """

    def __init__(self):
        self.classes = []
        self.train_data = None
        self.test_data = None
        self.train_data_org = None
        self.test_data_org = None

    def load_data(self):
        """
        Tai va xu ly du lieu anh tu thu muc goc.
        Cac thu muc con la ten lop (label), chia thanh 80% train va 20% validation.
        """
        data_path = config.DATA_DIR

        if not os.path.exists(data_path) or not os.listdir(data_path):
            raise FileNotFoundError(f"Khong tim thay du lieu hoac thu muc rong: {data_path}")

        class_folders = sorted(os.listdir(data_path))
        print(f"Thu muc du lieu: {data_path}")
        print(f"Cac lop phat hien: {class_folders}")

        # Load du lieu train/validation
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

        # Rescale ve khoang [0,1] va toi uu voi cache + prefetch
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

    def plot_samples(self):
        """
        Hien thi 9 anh ngau nhien tu tap huan luyen.
        """
        if not self.train_data_org:
            raise ValueError("Chua load du lieu. Goi load_data() truoc khi ve.")

        print("Dang ve anh mau tu tap train...")
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
        Tra ve danh sach lop.
        """
        return self.classes


class call_back:
    """
    Callback dung de dung som neu model khong cai thien.
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


# Chay thu nghiem khi run truc tiep file
if __name__ == "__main__":
    loader = DataLoader()
    loader.load_data()
    loader.plot_samples()
