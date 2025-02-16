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
from PIL import Image


# Flask app setup
app = Flask(__name__, static_folder="dist", static_url_path="/")
CORS(app, resources={r"/*": {"origins": "*"}})

# Imgur API credentials (replace with your actual client ID)
DATABASE = "captcha.db"  # SQLite database file
IMGUR_CLIENT_ID = os.getenv('IMGUR_CLIENT_ID')

# ------------------ DATABASE SETUP ------------------

def print_db_contents():
    """Reads all stored image URLs from the database and prints them."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM images")  # Fetch all rows
    rows = cursor.fetchall()  # Retrieve data

    conn.close()  # Close connection

    if rows:
        print("\n🔹 **Stored Image URLs in Database** 🔹\n")
        for row in rows:
            print(f"ID: {row[0]}, URL: {row[1]}")
    else:
        print("\n⚠️ No images found in the database.\n")

def initialize_db():
    """Clears the database and recreates the images table on every server restart."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # ✅ Drop the table if it exists (this wipes all stored data)
    cursor.execute("DROP TABLE IF EXISTS images")

    # ✅ Recreate the table
    cursor.execute("""
        CREATE TABLE images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print("✅ Database wiped and reinitialized on server restart.")

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
    """Uploads an image to Imgur and returns the correct direct URL."""
    headers = {"Authorization": f"Client-ID {IMGUR_CLIENT_ID}"}

    # ✅ Detect the actual image format (PNG, JPEG, etc.)
    with Image.open(image_path) as img:
        img_format = img.format.lower()  # Convert to lowercase ('png', 'jpeg', etc.)

    # ✅ Upload the image to Imgur
    with open(image_path, "rb") as file:
        response = requests.post(
            "https://api.imgur.com/3/upload",
            headers=headers,
            files={"image": file},
        )

    data = response.json()
    if data.get("success"):
        imgur_id = data["data"]["id"]  # ✅ Get the Imgur image ID

        # ✅ Ensure the correct extension
        if img_format in ["jpeg", "jpg"]:
            return f"https://i.imgur.com/{imgur_id}.jpg"
        elif img_format == "png":
            return f"https://i.imgur.com/{imgur_id}.png"
        else:
            return f"https://i.imgur.com/{imgur_id}"  # Default (just in case)

    else:
        raise Exception(f"Imgur upload failed: {data}")

@app.route("/save_image_url", methods=["POST"])
def save_image_url():
    """Saves a new image URL to the database."""
    try:
        data = request.get_json()
        if "image_url" not in data:
            return jsonify({"error": "Image URL is required"}), 400

        image_url = data["image_url"]
        
        # ✅ Save to database
        save_image_url_to_db(image_url)
        
        print(f"✅ Image URL saved: {image_url}")  # Debugging
        
        return jsonify({"message": "Image URL saved successfully."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
@app.route("/obfuscate", methods=["POST"])
def obfuscate():
    """Apply obfuscation, upload to Imgur, and return the public link."""
    try:
        data = request.get_json()
        if "image_url" not in data:
            return jsonify({"error": "Image URL is required"}), 400

        image_url = data["image_url"]

        # ✅ Fetch the image the same way React does
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        }
        response = requests.get(image_url, headers=headers, stream=True)

        if response.status_code != 200:
            return jsonify({"error": f"Failed to fetch image, status code: {response.status_code}"}), 400

        print("✅ Image fetched successfully from:", image_url)  # Debugging

        # ✅ Convert the image to an array
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        img_np = np.array(image)

        # Apply obfuscation
        obfuscated_img = apply_obfuscation(img_np)
        obfuscated_pil = Image.fromarray(obfuscated_img)

        # Save temporarily
        filename = f"{uuid.uuid4().hex}.png"
        temp_dir = "public/temp"

        # Ensure directory exists
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        temp_path = os.path.join(temp_dir, filename)
        obfuscated_pil.save(temp_path)

        # ✅ Upload to Imgur
        imgur_url = upload_to_imgur(temp_path)

        # ✅ Save the URL in SQLite
        save_image_url_to_db(imgur_url)
        print_db_contents()

        return jsonify({"image_url": imgur_url})

    except Exception as e:
        print(f"❌ Error in /obfuscate: {e}")  # Debugging
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
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        }
        response = requests.get(image_url, headers=headers, stream=True)

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
            # {"agent": "OpenAI GPT-4o", "response": openai_text.strip().lower(), "time": f"{openai_time}s", "correct": validation_function(correct_answer, openai_text.strip().lower())}
        ]

        # Process Gemini models
        for gemini_model in gemini_models:
            _, gemini_text, gemini_time = solve_captcha(base64_image, model_name=gemini_model, is_multiselect=is_multiselect)
            results.append({"agent": f"Google {gemini_model}", "response": gemini_text.strip().lower(), "time": f"{gemini_time}s", "correct": validation_function(correct_answer, gemini_text.strip().lower())})

        # Process Mistral models
        for mistral_model in mistral_models:
            _, mistral_text, mistral_time = solve_captcha(base64_image, model_name=mistral_model, is_multiselect=is_multiselect)
            results.append({"agent": f"Mistral {mistral_model}", "response": mistral_text.strip().lower(), "time": f"{mistral_time}s", "correct": validation_function(correct_answer, mistral_text.strip().lower())})

        # Process Groq models
        if is_multiselect:
            for groq_model in groq_models:
                _, groq_text, groq_time = solve_captcha(image_url, model_name=groq_model, is_multiselect=is_multiselect)
                results.append({"agent": f"Groq {groq_model}", "response": groq_text.strip().lower(), "time": f"{groq_time}s", "correct": validation_function(correct_answer, groq_text.strip().lower())})
        
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
