from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf

# === THÊM ĐOẠN NÀY để import được src/
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import DataLoader
from web.utils import xu_ly_anh
import src.config as config

app = Flask(__name__)
UPLOAD_FOLDER = "web/static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model và class
model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
class_names = DataLoader().load_data().get_classes()

from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
from src.preprocessing import DataLoader
from web.utils import xu_ly_anh
import src.config as config

app = Flask(__name__)
UPLOAD_FOLDER = "web/static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model và class
model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
class_names = DataLoader().load_data().get_classes()

@app.route("/", methods=["GET", "POST"])
def index():
    label = None
    confidence = None
    image_filename = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            dau_vao = xu_ly_anh(filepath)
            prediction = model.predict(dau_vao, verbose=0)
            class_id = np.argmax(prediction)
            label = class_names[class_id]
            confidence = float(np.max(prediction)) * 100  # chuyển sang %
            image_filename = file.filename

    return render_template("index.html", label=label, confidence=confidence, image=image_filename)

if __name__ == "__main__":
    app.run(debug=True)
