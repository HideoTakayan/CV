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
print("\n=== Buoc 1: Tai du lieu ===")
data_loader = DataLoader()
data_loader.load_data()
class_names = data_loader.get_classes()
print(f"So lop: {len(class_names)}")

# === Buoc 2: Tai mo hinh goc (MobileNetV2) ===
print("\n=== Buoc 2: Tai mo hinh MobileNetV2 ===")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=config.INPUT_SHAPE)
base_model.trainable = False  # Giu nguyen trong so cua mo hinh tien huan luyen

# === Buoc 3: Xay dung mo hinh phan loai ===
print("\n=== Buoc 3: Xay dung mo hinh phan loai ===")
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

# === Buoc 4: Huan luyen mo hinh ===
print("\n=== Buoc 4: Huan luyen mo hinh ===")
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

# === Buoc 5: Luu mo hinh ===
print("\n=== Buoc 5: Luu mo hinh ===")
model.save(config.MODEL_SAVE_PATH)

with open(config.MODEL_HISTORY_PATH, 'w') as f:
    json.dump(history.history, f)

with open(config.MODEL_ARCHITECTURE_PATH, 'w') as f:
    json.dump(model.to_json(), f)

print("\nHoan tat huan luyen va luu mo hinh.")
