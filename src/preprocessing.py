import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.callbacks import EarlyStopping
import config


class DataLoader:
    """
    Lớp xử lý tải và tiền xử lý dữ liệu ảnh.
    """

    def __init__(self):
        self.classes = []
        self.train_data = None
        self.test_data = None
        self.train_data_org = None
        self.test_data_org = None

    def load_data(self):
        """
        Tải và chuẩn hóa dữ liệu từ thư mục chứa ảnh.
        Mỗi thư mục con tương ứng với một class.
        """
        data_path = config.DATA_DIR

        if not os.path.exists(data_path) or not os.listdir(data_path):
            raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {data_path}")

        print(f"Found dataset folder: {data_path}")
        print("Detected class folders:", sorted(os.listdir(data_path)))

        # Chia train / validation tự động từ thư mục gốc
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

        # Scale ảnh về [0, 1]
        rescale_layer = Rescaling(1. / 255)
        self.train_data = self.train_data_org.map(lambda x, y: (rescale_layer(x), y))
        self.test_data = self.test_data_org.map(lambda x, y: (rescale_layer(x), y))

    def plot_samples(self):
        """
        Hiển thị 9 ảnh mẫu từ tập huấn luyện để kiểm tra.
        """
        if not self.train_data_org:
            raise ValueError("Chưa load dữ liệu. Gọi load_data() trước.")

        print("Displaying sample training images...")
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
        Trả về danh sách các class (tên thư mục con).
        """
        return self.classes


class call_back:
    """
    Cài đặt callback cho quá trình huấn luyện.
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


# Nếu chạy file này riêng lẻ
if __name__ == "__main__":
    loader = DataLoader()
    loader.load_data()
    loader.plot_samples()
