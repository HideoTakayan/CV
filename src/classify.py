import os
import numpy as np
import tensorflow as tf
import cv2
import config
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Danh sách các nhãn tương ứng với các lớp
NHAN = [
    "bo", "ca chua", "cam", "chanh", "cherry", "chom chom", "chuoi", "dau tay", "du du",
    "dua", "dua hau", "dua luoi", "hat dieu", "hat ngo", "hong", "kiwi", "khe", "le",
    "luu", "mit to nu", "nhan", "ot chuong", "tao", "thanh long", "vai thieu", "xoai"
]

def xu_ly_anh(duong_dan): 
    """
    Doc va tien xu ly anh tu duong dan.
    """
    if not os.path.exists(duong_dan):
        raise FileNotFoundError(f"Khong tim thay anh tai: {duong_dan}")
    
    anh = image.load_img(duong_dan, target_size=(224, 224))
    anh_array = image.img_to_array(anh)
    anh_array = np.expand_dims(anh_array, axis=0) # Thêm chiều batch
    anh_array = preprocess_input(anh_array) # Chuẩn hóa ảnh theo yêu cầu của MobileNetV2
    return anh_array

def du_doan(duong_dan):
    """
    Du doan loai trai cay tu anh.
    """
    # Tải mô hình
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError("Khong tim thay mo hinh da train. Hay train truoc hoac kiem tra duong dan.")
    
    mo_hinh = tf.keras.models.load_model(config.MODEL_SAVE_PATH)

    # Xử lý đầu vào
    dau_vao = xu_ly_anh(duong_dan)

    # Dự đoán
    ket_qua = mo_hinh.predict(dau_vao) # Dự đoán trả về xác suất cho từng lớp
    chi_so = np.argmax(ket_qua) # Lấy chỉ số lớp có xác suất cao nhất
    xac_suat = float(np.max(ket_qua)) * 100

    if chi_so >= len(NHAN): 
        raise IndexError(f"Chi so du doan {chi_so} vuot qua so luong nhan ({len(NHAN)}).")

    ten_loai = NHAN[chi_so]
    print(f"Du doan: {ten_loai} ({xac_suat:.2f}%)")
    return ten_loai, xac_suat

def hien_thi_anh(duong_dan):
    """
    Mo va hien thi anh bang OpenCV.
    """
    anh = cv2.imread(duong_dan)
    if anh is None:
        raise ValueError(f"Khong doc duoc anh tai: {duong_dan}")
    
    cv2.imshow("Anh da chon", anh)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    # Đường dẫn tới ảnh dùng để kiểm thử
    demo_anh = r"D:\CV\data\MY_data\predict\1 (19).jpeg"

    # Dự đoán và hiển thị ảnh
    loai_qua, ti_le = du_doan(demo_anh)
    hien_thi_anh(demo_anh)
