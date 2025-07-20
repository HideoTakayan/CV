# model.py
import config
import json
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

def create_model(num_classes):
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=config.INPUT_SHAPE
    )
    base_model.trainable = False

    model = Sequential(name="FruitClassifier")
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.005)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=config.OPTIMIZER,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_and_save_model(model, train_data, val_data, class_names):
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)
    ]

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks
    )

    model.save(config.MODEL_SAVE_PATH, save_format="tf")

    with open(config.MODEL_HISTORY_PATH, 'w') as f:
        json.dump(history.history, f, indent=2)

    with open(config.MODEL_ARCHITECTURE_PATH, 'w') as f:
        f.write(model.to_json())

    with open(config.LABEL_MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump({i: name for i, name in enumerate(class_names)}, f, ensure_ascii=False, indent=2)

    print(" Model và thong tin da duoc luu.")
