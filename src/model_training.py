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

# === Step 1: Load data ===
print("\n=== Step 1: Load data ===")
data_loader = DataLoader()
data_loader.load_data()
class_names = data_loader.get_classes()
print(f"Number of classes: {len(class_names)}")

# === Step 2: Load base model (MobileNetV2) ===
print("\n=== Step 2: Load base model (MobileNetV2) ===")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=config.INPUT_SHAPE)
base_model.trainable = False  # Freeze pretrained weights

# === Step 3: Build classification model ===
print("\n=== Step 3: Build classification model ===")

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

# === Step 4: Train model ===
print("\n=== Step 4: Train model ===")
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

# === Step 5: Save model and metadata ===
print("\n=== Step 5: Save model ===")
model.save(config.MODEL_SAVE_PATH)

# Save training history
with open(config.MODEL_HISTORY_PATH, 'w', encoding='utf-8') as f:
    json.dump(history.history, f)

# Save model architecture
with open(config.MODEL_ARCHITECTURE_PATH, 'w', encoding='utf-8') as f:
    json.dump(model.to_json(), f)

# Save label mapping (class index to label)
with open(config.LABEL_MAP_PATH, 'w', encoding='utf-8') as f:
    json.dump({i: label for i, label in enumerate(class_names)}, f, ensure_ascii=False)

print("\nTraining complete. Model and metadata saved.")
