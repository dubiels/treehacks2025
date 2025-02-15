from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import io
import base64
import os
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
    """Handle CAPTCHA solving via an image URL."""
    try:
        data = request.get_json()
        if "image_url" not in data or "correct_answer" not in data:
            return jsonify({"error": "Image URL and correct answer are required"}), 400

        image_url = data["image_url"]
        correct_answer = data["correct_answer"].strip().lower()

        # Render image on the front end by returning the image URL
        display_image_url = image_url  

        # Solve CAPTCHA using OpenAI (GPT-4o with direct image URL)
        openai_text, openai_time = solve_captcha(image_url, model="openai")

        # Fetch image from URL for Gemini & Mistral
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch image from URL"}), 400

        image = Image.open(io.BytesIO(response.content))

        # Convert image to Base64 for Gemini & Mistral
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Solve CAPTCHA using Gemini & Mistral
        gemini_text, gemini_time = solve_captcha(base64_image, model="gemini")
        mistral_text, mistral_time = solve_captcha(base64_image, model="mistral")

        # Normalize responses
        gemini_text = gemini_text.strip().lower()
        mistral_text = mistral_text.strip().lower()
        openai_text = openai_text.strip().lower()

        response = {
            "display_image": display_image_url,
            "correct_response": correct_answer,
            "results": [
                {"agent": "OpenAI", "response": openai_text, "time": f"{openai_time}s", "correct": openai_text == correct_answer},
                {"agent": "Gemini", "response": gemini_text, "time": f"{gemini_time}s", "correct": gemini_text == correct_answer},
                {"agent": "Mistral", "response": mistral_text, "time": f"{mistral_time}s", "correct": mistral_text == correct_answer}
            ]
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
