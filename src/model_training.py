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

# === Buoc 1: Tai du lieu ===
print("\n Buoc 1: Dang tai du lieu...")
data_loader = DataLoader()
data_loader.load_data()
class_names = data_loader.get_classes()
print(f" So lop: {len(class_names)}")
print(" Cac lop:", class_names)

# === Buoc 2: Tai mo hinh goc ===
print("\n Buoc 2: Dang tai mo hinh MobileNetV2...")
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=config.INPUT_SHAPE
)
base_model.trainable = False  # Dong bang mo hinh goc

# === Buoc 3: Xay dung mo hinh phan loai ===
print("\n Buoc 3: Dang xay dung mo hinh phan loai...")

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

# Bien dich mo hinh
model.compile(
    optimizer=config.OPTIMIZER,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# === Buoc 4: Huan luyen mo hinh ===
print("\n Buoc 4: Dang huan luyen mo hinh...")

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

# === Buoc 5: Luu mo hinh va thong tin ===
print("\n Buoc 5: Dang luu mo hinh va thong tin...")

# Luu mo hinh .h5
model.save(config.MODEL_SAVE_PATH)

# Luu lich su huan luyen
with open(config.MODEL_HISTORY_PATH, 'w', encoding='utf-8') as f:
    json.dump(history.history, f, ensure_ascii=False, indent=2)

# Luu cau truc mo hinh (.json string)
with open(config.MODEL_ARCHITECTURE_PATH, 'w', encoding='utf-8') as f:
    f.write(model.to_json())

# Luu danh sach ten lop
with open(config.LABEL_MAP_PATH, 'w', encoding='utf-8') as f:
    json.dump({i: label for i, label in enumerate(class_names)}, f, ensure_ascii=False, indent=2)

print("\n Da huan luyen xong. Mo hinh va thong tin da duoc luu.")
