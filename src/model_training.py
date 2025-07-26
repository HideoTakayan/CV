import os
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import config
from preprocessing import TaiDuLieu, GoiLai

# Tải dữ liệu
bo_tai = TaiDuLieu()
bo_tai.tai_du_lieu()
nhan = bo_tai.lay_nhan()
go_lai = GoiLai()

# Khởi tạo base model MobileNetV2
mo_hinh_goc = MobileNetV2(weights="imagenet", include_top=False, input_shape=config.INPUT_SHAPE)

# Fine-tune: khóa 100 layer đầu, mở phần còn lại
for layer in mo_hinh_goc.layers[:100]:
    layer.trainable = False
for layer in mo_hinh_goc.layers[100:]:
    layer.trainable = True

# Xây dựng mô hình phân loại
mo_hinh = Sequential([
    mo_hinh_goc,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(123, kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Activation(config.ACTIVATION_FUNCTION),
    Dropout(0.3),
    Dense(64, kernel_regularizer=l2(0.005)),
    BatchNormalization(),
    Activation(config.ACTIVATION_FUNCTION),
    Dropout(0.3),
    Dense(len(nhan), activation='softmax')
])

# Biên dịch mô hình
mo_hinh.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='SparseCategoricalCrossentropy',
    metrics=['accuracy']
)

# Callback giảm learning rate khi chững
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=1e-6
)

# Huấn luyện mô hình
lich_su = mo_hinh.fit(
    bo_tai.du_lieu_train,
    epochs=config.EPOCHS,
    validation_data=bo_tai.du_lieu_test,
    batch_size=config.BATCH_SIZE,
    callbacks=[go_lai.lay_callback(), reduce_lr]
)

# Lưu mô hình
mo_hinh.save(config.MODEL_SAVE_PATH)
print("Da luu mo hinh thanh cong")

# Lưu lịch sử huấn luyện
lich_su_clean = {k: [float(v_) for v_ in v] for k, v in lich_su.history.items()}
with open(config.MODEL_HISTORY_PATH, "w") as tep_lich_su:
    json.dump(lich_su_clean, tep_lich_su)

# Lưu cấu trúc mô hình
mo_hinh_json = mo_hinh.to_json()
with open(config.MODEL_ARCHITECTURE_PATH, "w") as tep_kien_truc:
    json.dump(mo_hinh_json, tep_kien_truc)
