import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from preprocessing import DataLoader
import config

def xu_ly_anh(duong_dan):
    img = image.load_img(duong_dan, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def hien_thi(duong_dan, label):
    img = cv2.imread(duong_dan)
    cv2.putText(img, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Ket qua", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load model đã train
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)

    # Lấy danh sách class
    loader = DataLoader()
    loader.load_data()
    class_names = loader.get_classes()

    # Đường dẫn thư mục chứa ảnh test
    thu_muc = "images/test"

    for ten_file in os.listdir(thu_muc):
        if ten_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            duong_dan_anh = os.path.join(thu_muc, ten_file)
            dau_vao = xu_ly_anh(duong_dan_anh)
            du_doan = model.predict(dau_vao, verbose=0)
            chi_so = np.argmax(du_doan)
            label = class_names[chi_so]
            print(f"{ten_file}: {label}")
            hien_thi(duong_dan_anh, label)
