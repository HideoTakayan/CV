import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import config

class TaiDuLieu:
    """
    Lớp xử lý tải và tiền xử lý dữ liệu ảnh trái cây.
    """
    def __init__(self):
        self.nhan = []
        self.du_lieu_train = None
        self.du_lieu_test = None

    def tai_du_lieu(self):
        """
        Tải dữ liệu từ thư mục đã có sẵn.
        """
        if not os.path.exists(config.TRAINING_DATA) or not os.path.exists(config.TESTING_DATA):
            raise FileNotFoundError("Khong tim thay du lieu huan luyen hoac kiem thu. Hay kiem tra lai thu muc MY_data.")

        print("Da tim thay du lieu huan luyen va kiem thu.")

        # Tạo dataset từ thư mục
        self.du_lieu_train_raw = tf.keras.utils.image_dataset_from_directory(
            config.TRAINING_DATA,
            shuffle=config.SHUFFLE,
            image_size=(224, 224),
            batch_size=config.BATCH_SIZE,
            seed=config.SEED
        )
        self.du_lieu_test_raw = tf.keras.utils.image_dataset_from_directory(
            config.TESTING_DATA,
            shuffle=config.SHUFFLE,
            image_size=(224, 224),
            batch_size=config.BATCH_SIZE,
            seed=config.SEED
        )

        # Lưu nhãn
        self.nhan = self.du_lieu_train_raw.class_names

        # Tiền xử lý: chuẩn hóa ảnh về [0, 1]
        def chuan_hoa(anh, nhan):
            return Rescaling(1./255)(anh), nhan

        self.du_lieu_train = self.du_lieu_train_raw.map(chuan_hoa).prefetch(tf.data.AUTOTUNE)
        self.du_lieu_test = self.du_lieu_test_raw.map(chuan_hoa).prefetch(tf.data.AUTOTUNE)

    def hien_thi_mau(self):
        """
        Hiển thị một số ảnh mẫu từ dữ liệu huấn luyện.
        """
        if self.du_lieu_train is None:
            raise ValueError("Du lieu chua duoc tai. Hay goi tai_du_lieu() truoc.")

        print("Dang hien thi anh mau...")
        os.makedirs(os.path.dirname(config.SAMPLE_PLOT_PATH), exist_ok=True)

        plt.figure(figsize=(10, 10))
        for anh, nhan in self.du_lieu_train_raw.take(1):
            for i in range(9):
                plt.subplot(3, 3, i + 1)
                plt.imshow(anh[i].numpy().astype('uint8'))
                plt.title(self.nhan[nhan[i]])
                plt.axis('off')
        plt.savefig(config.SAMPLE_PLOT_PATH)
        plt.show()

    def lay_nhan(self):
        return self.nhan


class GoiLai:
    """
    Lớp callback cho huấn luyện mô hình.
    """
    def __init__(self):
        self.callbacks = []

    def lay_callback(self):
        self.callbacks = [
            EarlyStopping(
                monitor="val_accuracy",
                min_delta=0.001,
                patience=12,
                verbose=1,
                mode="max",
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor="val_accuracy",
                factor=0.5,
                patience=5,
                verbose=1,
                mode="max",
                min_lr=1e-6
            )
        ]
        return self.callbacks


if __name__ == "__main__":
    bo_tai = TaiDuLieu()
    bo_tai.tai_du_lieu()
    bo_tai.hien_thi_mau()
