import os
import numpy as np
import tensorflow as tf
import cv2
import config
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from preprocessing import DataLoader

def xu_ly_anh_du_doan(duong_dan_anh):
    if not os.path.exists(duong_dan_anh):
        raise FileNotFoundError(f"Khong tim thay anh: {duong_dan_anh}")
    
    img = image.load_img(duong_dan_anh, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def du_doan_trai_cay(duong_dan_anh, model, class_names):
    img = xu_ly_anh_du_doan(duong_dan_anh)
    ket_qua = model.predict(img, verbose=0)
    chi_so = np.argmax(ket_qua)
    return class_names[chi_so]

def hien_thi_anh_va_nhan(duong_dan_anh, nhan):
    anh = cv2.imread(duong_dan_anh)
    if anh is None:
        raise ValueError("Anh khong ton tai hoac bi hong.")

    cv2.putText(anh, f"Du doan: {nhan}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Ket qua du doan", anh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def du_doan_thu_muc(duong_dan_thu_muc, hien_thi=False):
    """
    Duyet tat ca anh trong thu muc va du doan
    """
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
    data_loader = DataLoader()
    data_loader.load_data()
    class_names = data_loader.get_classes()

    danh_sach_anh = [f for f in os.listdir(duong_dan_thu_muc) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not danh_sach_anh:
        print("Khong tim thay anh nao trong thu muc.")
        return

    for ten_anh in danh_sach_anh:
        duong_dan = os.path.join(duong_dan_thu_muc, ten_anh)
        ket_qua = du_doan_trai_cay(duong_dan, model, class_names)
        print(f"{ten_anh}: {ket_qua}")
        if hien_thi:
            hien_thi_anh_va_nhan(duong_dan, ket_qua)

# === Chay thu khi run file ===
if __name__ == "__main__":
    # Du doan toan bo thu muc test
    du_doan_thu_muc("images/test", hien_thi=True)
