from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import io
import base64
import numpy as np
import cv2
from PIL import Image
from agents.agent import solve_captcha

app = Flask(__name__, static_folder="dist", static_url_path="/")
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/", defaults={'path': ''})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

@app.route("/solve", methods=["POST"])
def solve():
    """Handle CAPTCHA solving via uploaded image and user-provided correct answer."""
    try:
        if "image" not in request.files or "correct_answer" not in request.form:
            return jsonify({"error": "Image and correct answer are required"}), 400

        file = request.files["image"]
        correct_answer = request.form["correct_answer"].strip().lower()
        image = Image.open(file.stream)

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        gemini_text, gemini_time = solve_captcha(base64_image, model="gemini")
        mistral_text, mistral_time = solve_captcha(base64_image, model="mistral")

        gemini_text = gemini_text.strip().lower()
        mistral_text = mistral_text.strip().lower()

        response = {
            "correct_response": correct_answer,
            "results": [
                {"agent": "Gemini", "response": gemini_text, "time": f"{gemini_time}s", "correct": gemini_text == correct_answer},
                {"agent": "Mistral", "response": mistral_text, "time": f"{mistral_time}s", "correct": mistral_text == correct_answer}
            ]
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/obfuscate", methods=["POST"])
def obfuscate():
    """Apply obfuscation techniques to the uploaded CAPTCHA image."""
    try:
        if "image" not in request.files:
            return jsonify({"error": "Image is required"}), 400

        file = request.files["image"]
        image = Image.open(file.stream).convert("RGB")
        img_np = np.array(image)

        obfuscated_img = apply_obfuscation(img_np)
        obfuscated_pil = Image.fromarray(obfuscated_img)
        obfuscated_path = "static/obfuscated_captcha.png"
        obfuscated_pil.save(obfuscated_path)

        return jsonify({"image_url": f"http://127.0.0.1:5000/{obfuscated_path}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def apply_obfuscation(img_np):
    """Apply obfuscation techniques: distortions, noise, and blurring."""
    height, width, _ = img_np.shape
    for i in range(height):
        offset = int(10 * np.sin(2.0 * np.pi * i / 30))
        img_np[i] = np.roll(img_np[i], offset, axis=0)

    noise = np.random.randint(0, 50, img_np.shape, dtype="uint8")
    img_np = cv2.add(img_np, noise)
    img_np = cv2.GaussianBlur(img_np, (3, 3), 0)

    for _ in range(5):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        cv2.line(img_np, (x, 0), (x, height), (50, 50, 50), 1)
        cv2.line(img_np, (0, y), (width, y), (50, 50, 50), 1)
    
    return img_np

if __name__ == "__main__":
    app.run(debug=True)
