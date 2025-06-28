import os
import numpy as np
import tensorflow as tf
import cv2
import config
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from preprocessing import DataLoader

def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def predict(image_path):
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
    data_loader = DataLoader()
    data_loader.load_data()
    class_names = data_loader.get_classes()

    img = preprocess_image(image_path)
    prediction = model.predict(img)
    index = np.argmax(prediction)
    print("\nPrediction:", class_names[index])
    return class_names[index]

def show_image(image_path, label):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or corrupted.")
    cv2.putText(img, f"Prediction: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_image = "images/test/fruit2.jpg"
    result = predict(input_image)
    show_image(input_image, result)
