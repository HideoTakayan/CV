from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import sys

# Thêm thư mục src vào path để import classify.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from classify import du_doan

# Cấu hình đường dẫn upload
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Tạo thư mục uploads nếu chưa tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Hàm kiểm tra định dạng file hợp lệ
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Dự đoán kết quả
            try:
                loai_qua, ti_le = du_doan(filepath)
                ket_qua = f"{loai_qua} ({ti_le:.2f}%)"
            except Exception as e:
                ket_qua = f"Lỗi dự đoán: {e}"

            return render_template("index.html", filename=filename, ket_qua=ket_qua)

    return render_template("index.html", filename=None)
if __name__ == "__main__":
    app.run(debug=True)
