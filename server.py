from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from agents.agent import solve_captcha
from PIL import Image
import io
import base64

app = Flask(__name__, static_folder="dist", static_url_path="/")
CORS(app)

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
        # Ensure an image was uploaded
        if "image" not in request.files or "correct_answer" not in request.form:
            return jsonify({"error": "Image and correct answer are required"}), 400

        file = request.files["image"]
        correct_answer = request.form["correct_answer"].strip().lower()

        image = Image.open(file.stream)

        # Convert image to Base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Solve CAPTCHA using both AI models
        gemini_text, gemini_time = solve_captcha(base64_image, model="gemini")
        mistral_text, mistral_time = solve_captcha(base64_image, model="mistral")

        # Normalize AI responses for comparison
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


if __name__ == "__main__":
    app.run(debug=True)
