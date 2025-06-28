import os
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import config
from preprocessing import DataLoader, call_back

# === Bước 1: Tải dữ liệu ===
print("\n=== Bước 1: Tải dữ liệu ===")
data_loader = DataLoader()
data_loader.load_data()
class_names = data_loader.get_classes()
print(f"Số lớp: {len(class_names)}")

# === Bước 2: Tải mô hình gốc (MobileNetV2) ===
print("\n=== Bước 2: Tải mô hình MobileNetV2 ===")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=config.INPUT_SHAPE)
base_model.trainable = False  # Giữ nguyên trọng số của mô hình tiền huấn luyện

# === Bước 3: Xây dựng mô hình phân loại ===
print("\n=== Bước 3: Xây dựng mô hình phân loại ===")
def add_dense_block(model, units, l2_rate):
    model.add(Dense(units, kernel_regularizer=l2(l2_rate)))
    model.add(BatchNormalization())
    model.add(Activation(config.ACTIVATION_FUNCTION))
    model.add(Dropout(0.3))

model = Sequential(name="FruitClassifier")
model.add(base_model)
model.add(GlobalAveragePooling2D())
add_dense_block(model, 256, 0.01)
add_dense_block(model, 128, 0.005)
model.add(Dense(len(class_names), activation='softmax'))

model.compile(
    optimizer=config.OPTIMIZER,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# === Bước 4: Huấn luyện mô hình ===
print("\n=== Bước 4: Huấn luyện mô hình ===")
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    data_loader.train_data,
    validation_data=data_loader.test_data,
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    callbacks=[call_back().get_callbacks(), early_stop]
)

# === Bước 5: Lưu mô hình ===
print("\n=== Bước 5: Lưu mô hình ===")
model.save(config.MODEL_SAVE_PATH)

with open(config.MODEL_HISTORY_PATH, 'w') as f:
    json.dump(history.history, f)

with open(config.MODEL_ARCHITECTURE_PATH, 'w') as f:
    json.dump(model.to_json(), f)

print("\n✅ Huấn luyện và lưu mô hình hoàn tất.")
