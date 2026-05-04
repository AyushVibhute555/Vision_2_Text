import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from backend.utils.caption_generator import generate_caption

# =========================
# BASE DIRECTORY
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# APP INIT
# =========================
app = Flask(__name__)

# Enable CORS (important for frontend)
CORS(app, resources={r"/*": {"origins": "*"}})

# =========================
# UPLOAD FOLDER (FIXED PATH)
# =========================
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =========================
# HEALTH CHECK ROUTE (IMPORTANT)
# =========================
@app.route("/")
def home():
    return "🚀 VisionNarrate API is running!"

# =========================
# GENERATE CAPTION API
# =========================
@app.route("/generate-caption", methods=["POST"])
def generate_caption_api():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        # Debug log
        print("📥 Saving image to:", image_path)

        file.save(image_path)

        # Generate caption
        caption = generate_caption(image_path)

        return jsonify({"caption": caption})

    except Exception as e:
        print("🔥 BACKEND ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)