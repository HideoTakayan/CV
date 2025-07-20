import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json
import config
import os

model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)

with open(config.LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    label_map = json.load(f)
    class_names = [label_map[str(i)] for i in range(len(label_map))]

def predict_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Khong tim thay anh: {img_path}")

    img = image.load_img(img_path, target_size=config.INPUT_SHAPE[:2])
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions)
    return class_names[predicted_index], predictions[0][predicted_index]
