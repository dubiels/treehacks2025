from dotenv import load_dotenv
import pyimgur
import sqlite3
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import io
import base64
import os
import uuid
import numpy as np
import cv2
from PIL import Image
from agents.agent import solve_captcha


load_dotenv()


app = Flask(__name__, static_folder="dist", static_url_path="/")
CORS(app, resources={r"/*": {"origins": "*"}})

client_id = os.getenv("IMGUR_CLIENT_ID")
im = pyimgur.Imgur(client_id)

DB_PATH = "captcha_images.db"  # SQLite database file
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()


@app.route("/", defaults={'path': ''})
@app.route("/")
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')


# create sqlite db
cursor.execute('''CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_image_url TEXT NOT NULL,
    obfuscated_image_url TEXT NOT NULL
)''')
conn.commit()


# insert data into sqlite db
def insert_image_urls(original_url, obfuscated_url):
    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO images (original_url, obfuscated_url)
        VALUES (?, ?)
    ''', (original_url, obfuscated_url))
    conn.commit()
    conn.close()


@app.route("/obfuscate", methods=["POST"])
def obfuscate():
    """Apply obfuscation techniques and return a temporary CAPTCHA link."""
    try:
        data = request.get_json()
        if "image_url" not in data:
            return jsonify({"error": "Image URL is required"}), 400

        image_url = data["image_url"]
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch image from URL"}), 400

        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        img_np = np.array(image)

        # Apply obfuscation
        obfuscated_img = apply_obfuscation(img_np)
        obfuscated_pil = Image.fromarray(obfuscated_img)

        # Upload via imgur
        temp_image_path = f"temp_{uuid.uuid4().hex}.png"
        obfuscated_pil.save(temp_image_path)
        uploaded_image = im.upload_image(
            temp_image_path, title="Obfuscated CAPTCHA")
        os.remove(temp_image_path)

        cursor.execute('''INSERT INTO images (original_image_url, obfuscated_image_url) 
                        VALUES (?, ?)''', (image_url, uploaded_image.link))
        conn.commit()

        return jsonify({
            "original_image_url": image_url,
            "obfuscated_image_url": uploaded_image.link
        })

        # Karolina's code
        # # Generate a unique filename
        # unique_filename = f"{uuid.uuid4().hex}.png"

        # # Ensure the static directory exists
        # static_dir = "public/temp"
        # if not os.path.exists(static_dir):
        #     os.makedirs(static_dir)

        # # Save the obfuscated image
        # obfuscated_path = os.path.join(static_dir, unique_filename)
        # obfuscated_pil.save(obfuscated_path)

        # return jsonify({
        #     "image_url": f"https://treehacks2025-one.vercel.app/temp/{unique_filename}"

        #     "obfuscated_url": obfuscated_image
        # })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Serve obfuscated CAPTCHAs dynamically
@app.route("/temp/")
def serve_obfuscated(filename):
    """Serve temporary CAPTCHA images."""
    return send_from_directory("public/temp", filename)


@app.route("/solve", methods=["POST"])
def solve():
    """Handle CAPTCHA solving via an image URL."""
    try:
        data = request.get_json()
        if "image_url" not in data or "correct_answer" not in data:
            return jsonify({"error": "Image URL and correct answer are required"}), 400

        image_url = data["image_url"]
        correct_answer = data["correct_answer"].strip().lower()

        # Solve CAPTCHA using OpenAI (GPT-4o with direct image URL)
        _, openai_text, openai_time = solve_captcha(
            image_url, model_name="gpt-4o")

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
        groq_models = ["llama-3.2-90b-vision-preview",
                       "llama-3.2-11b-vision-preview"]

        results = [
            {"agent": "OpenAI GPT-4o", "response": openai_text.strip().lower(), "time": f"{openai_time}s",
             "correct": openai_text.strip().lower() == correct_answer}
        ]

        # Process Gemini models
        for gemini_model in gemini_models:
            _, gemini_text, gemini_time = solve_captcha(
                base64_image, model_name=gemini_model)
            results.append({"agent": f"Google {gemini_model}", "response": gemini_text.strip().lower(
            ), "time": f"{gemini_time}s", "correct": gemini_text.strip().lower() == correct_answer})

        # Process Mistral models
        for mistral_model in mistral_models:
            _, mistral_text, mistral_time = solve_captcha(
                base64_image, model_name=mistral_model)
            results.append({"agent": f"Mistral {mistral_model}", "response": mistral_text.strip().lower(
            ), "time": f"{mistral_time}s", "correct": mistral_text.strip().lower() == correct_answer})

        # Process Groq models
        for groq_model in groq_models:
            _, groq_text, groq_time = solve_captcha(
                image_url, model_name=groq_model)
            results.append({"agent": f"Mistral {groq_model}", "response": groq_text.strip().lower(
            ), "time": f"{groq_time}s", "correct": groq_text.strip().lower() == correct_answer})

        response = {
            "display_image": image_url,
            "correct_response": correct_answer,
            "results": results
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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


# get all the images
@app.route("/get_images", methods=["GET"])
def get_images():
    """Fetch all saved images (original and obfuscated) from the database."""
    cursor.execute("SELECT * FROM images")
    images = cursor.fetchall()
    images_list = [{"original_image_url": img[1],
                    "obfuscated_image_url": img[2]} for img in images]
    return jsonify(images_list)


if __name__ == "__main__":
    app.run(debug=True)
