from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import io
import base64
import os
import numpy as np
import cv2
from PIL import Image
from agents.agent import solve_captcha, check_text_correctness, check_multiselect_correctness
import uuid
import sqlite3

# Flask app setup
app = Flask(__name__, static_folder="dist", static_url_path="/")
CORS(app, resources={r"/*": {"origins": "*"}})

# Imgur API credentials (replace with your actual client ID)
IMGUR_CLIENT_ID = "your_client_id"
DATABASE = "captcha.db"  # SQLite database file

# ------------------ DATABASE SETUP ------------------
def initialize_db():
    """Creates the database and table for storing Imgur URLs."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()

initialize_db()  # Initialize DB when server starts

def save_image_url_to_db(image_url):
    """Stores the uploaded Imgur URL in SQLite."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO images (url) VALUES (?)", (image_url,))
    conn.commit()
    conn.close()

@app.route("/", defaults={'path': ''})
@app.route("/")
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

# ------------------ IMAGE OBFUSCATION & UPLOAD ------------------
def upload_to_imgur(image_path):
    """Uploads an image to Imgur and returns the public URL."""
    headers = {"Authorization": f"Client-ID {IMGUR_CLIENT_ID}"}
    with open(image_path, "rb") as file:
        response = requests.post(
            "https://api.imgur.com/3/upload",
            headers=headers,
            files={"image": file},
        )

    data = response.json()
    if data.get("success"):
        return data["data"]["link"]  # Public Imgur link
    else:
        raise Exception(f"Imgur upload failed: {data}")

@app.route("/obfuscate", methods=["POST"])
def obfuscate():
    """Apply obfuscation, upload to Imgur, and return the public link."""
    try:
        data = request.get_json()
        if "image_url" not in data:
            return jsonify({"error": "Image URL is required"}), 400

        image_url = data["image_url"]
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch image"}), 400

        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        img_np = np.array(image)

        # Apply obfuscation
        obfuscated_img = apply_obfuscation(img_np)
        obfuscated_pil = Image.fromarray(obfuscated_img)

        # Save temporarily (on Windows-friendly path)
        filename = f"{uuid.uuid4().hex}.png"
        temp_dir = "public/temp"  # Change /tmp/ to public/temp/
        
        # Ensure temp directory exists
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        temp_path = os.path.join(temp_dir, filename)
        obfuscated_pil.save(temp_path)

        # Upload to Imgur
        imgur_url = upload_to_imgur(temp_path)

        # Save Imgur URL to SQLite
        save_image_url_to_db(imgur_url)

        return jsonify({"image_url": imgur_url})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------ RETRIEVE STORED IMAGE LINKS ------------------
@app.route("/get_images", methods=["GET"])
def get_images():
    """Retrieve stored Imgur image links from SQLite."""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT url FROM images ORDER BY id DESC LIMIT 10")  # Get last 10 images
        images = cursor.fetchall()
        conn.close()

        return jsonify({"images": [img[0] for img in images]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------ CAPTCHA SOLVER ------------------
@app.route("/solve", methods=["POST"])
def solve():
    """Handle CAPTCHA solving via an image URL."""
    try:
        data = request.get_json()
        is_multiselect = data.get("is_multiselect", False)

        validation_function = check_multiselect_correctness if is_multiselect else check_text_correctness

        if "image_url" not in data or "correct_answer" not in data:
            return jsonify({"error": "Image URL and correct answer are required"}), 400

        image_url = data["image_url"]
        correct_answer = data["correct_answer"].strip().lower()

        # Solve CAPTCHA using OpenAI (GPT-4o with direct image URL)
        _, openai_text, openai_time = solve_captcha(image_url, model_name="gpt-4o", is_multiselect=is_multiselect)

        # Fetch image from URL for Gemini & Mistral
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch image from URL"}), 400

        image = Image.open(io.BytesIO(response.content))

        # Convert image to Base64 for Gemini & Mistral
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Solve CAPTCHA using Gemini & Mistral (Base64)
        gemini_models = ["gemini-1.5-flash", "gemini-1.5-pro"]
        mistral_models = ["pixtral-12b-2409"]
        groq_models = ["llama-3.2-90b-vision-preview", "llama-3.2-11b-vision-preview"]

        results = [
            {"agent": "OpenAI GPT-4o", "response": openai_text.strip().lower(), "time": f"{openai_time}s", "correct": validation_function(correct_answer, openai_text.strip().lower())}
        ]

        # Process Gemini models
        for gemini_model in gemini_models:
            _, gemini_text, gemini_time = solve_captcha(base64_image, model_name=gemini_model, is_multiselect=is_multiselect)
            results.append({"agent": f"Google {gemini_model}", "response": gemini_text.strip().lower(), "time": f"{gemini_time}s", "correct": validation_function(correct_answer, gemini_text.strip().lower())})

        # Process Mistral models
        for mistral_model in mistral_models:
            _, mistral_text, mistral_time = solve_captcha(base64_image, model_name=mistral_model, is_multiselect=is_multiselect)
            results.append({"agent": f"Mistral {mistral_model}", "response": mistral_text.strip().lower(), "time": f"{mistral_time}s", "correct": validation_function(correct_answer, mistral_text.strip().lower())})

        response = {
            "display_image": image_url,
            "correct_response": correct_answer,
            "results": results
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------ IMAGE OBFUSCATION FUNCTION ------------------
def apply_obfuscation(img_np):
    """Apply AI-avoidant obfuscation techniques."""
    height, width, _ = img_np.shape
    for i in range(height):
        offset = int(5 * np.sin(2.0 * np.pi * i / 50))
        img_np[i] = np.roll(img_np[i], offset, axis=0)
    noise = np.random.normal(0, 15, img_np.shape).astype("uint8")
    img_np = cv2.add(img_np, noise)
    img_np = cv2.GaussianBlur(img_np, (3, 3), 0)
    return img_np

if __name__ == "__main__":
    app.run(debug=True)
