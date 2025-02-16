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

IMAGE_STORAGE_PATH = "public/temp"

# Imgur API credentials (replace with your actual client ID)
DATABASE = "captcha.db"  # SQLite database file
IMGUR_CLIENT_ID = os.getenv('IMGUR_CLIENT_ID')

# ------------------ DATABASE SETUP ------------------


@app.route("/get_images", methods=["GET"])
def get_images():
    """Retrieve all stored image URLs from the database, ordered by oldest first."""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        # Fetch URLs in order
        cursor.execute("SELECT url FROM images ORDER BY id ASC")
        rows = cursor.fetchall()
        conn.close()

        image_urls = [row[0] for row in rows]  # Extract URLs from rows

        return jsonify({"images": image_urls}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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


@app.route("/obfuscate2", methods=["POST"])
def obfuscate2():
    """Apply style transfer and warping obfuscation, upload to Imgur, and return the public link."""
    try:
        data = request.get_json()
        if "image_url" not in data:
            return jsonify({"error": "Image URL is required"}), 400

        image_url = data["image_url"]

        # Fetch the image
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        }
        response = requests.get(image_url, headers=headers, stream=True)

        if response.status_code != 200:
            return jsonify({"error": f"Failed to fetch image, status code: {response.status_code}"}), 400

        # Convert the image to an array
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        img_np = np.array(image)

        # Apply the new obfuscation method
        obfuscated_img = apply_obfuscation2(img_np)
        obfuscated_pil = Image.fromarray(obfuscated_img)

        # Save temporarily
        filename = f"{uuid.uuid4().hex}.png"
        temp_dir = "public/temp"

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        temp_path = os.path.join(temp_dir, filename)
        obfuscated_pil.save(temp_path)

        # Upload to Imgur
        imgur_url = upload_to_imgur(temp_path)

        # Save the URL in SQLite
        save_image_url_to_db(imgur_url)

        return jsonify({"image_url": imgur_url})

    except Exception as e:
        print(f"❌ Error in /obfuscate2: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/obfuscate3", methods=["POST"])
def obfuscate3():
    """Apply U-Net and diffusion-based obfuscation, upload to Imgur, and return the public link."""
    try:
        data = request.get_json()
        if "image_url" not in data:
            return jsonify({"error": "Image URL is required"}), 400

        image_url = data["image_url"]

        # Fetch the image
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        }
        response = requests.get(image_url, headers=headers, stream=True)

        if response.status_code != 200:
            return jsonify({"error": f"Failed to fetch image, status code: {response.status_code}"}), 400

        # Convert the image to an array
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        img_np = np.array(image)

        # Apply the U-Net and diffusion-based obfuscation
        obfuscated_img = apply_obfuscation3(img_np)
        obfuscated_pil = Image.fromarray(obfuscated_img)

        # Save temporarily
        filename = f"{uuid.uuid4().hex}.png"
        temp_dir = "public/temp"

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        temp_path = os.path.join(temp_dir, filename)
        obfuscated_pil.save(temp_path)

        # Upload to Imgur
        imgur_url = upload_to_imgur(temp_path)

        # Save the URL in SQLite
        save_image_url_to_db(imgur_url)

        return jsonify({"image_url": imgur_url})

    except Exception as e:
        print(f"❌ Error in /obfuscate3: {e}")
        return jsonify({"error": str(e)}), 500

    @app.route("/clear_database", methods=["POST"])
    def clear_database():
        """Deletes all stored images and returns success."""
        try:
            for file in os.listdir(IMAGE_STORAGE_PATH):
                file_path = os.path.join(IMAGE_STORAGE_PATH, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            return jsonify({"success": True, "message": "Database cleared."})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    if __name__ == "__main__":
        app.run(debug=True)

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
        # TODO uncomment
        # _, openai_text, openai_time = solve_captcha(image_url, model_name="gpt-4o", is_multiselect=is_multiselect)

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
        gemini_models = ["gemini-1.5-flash", "gemini-2.0-flash"]
        mistral_models = ["pixtral-12b-2409"]
        groq_models = ["llama-3.2-90b-vision-preview",
                       "llama-3.2-11b-vision-preview"]

        results = [
            # TODO uncomment
            # {"agent": "OpenAI GPT-4o", "response": openai_text.strip().lower(), "time": f"{openai_time}s", "correct": validation_function(correct_answer, openai_text.strip().lower())}
        ]

        # Process Gemini models
        for gemini_model in gemini_models:
            _, gemini_text, gemini_time = solve_captcha(
                base64_image, model_name=gemini_model, is_multiselect=is_multiselect)
            results.append({"agent": f"Google {gemini_model}", "response": gemini_text.strip().lower(
            ), "time": f"{gemini_time}s", "correct": validation_function(correct_answer, gemini_text.strip().lower())})

        # Process Mistral models
        for mistral_model in mistral_models:
            _, mistral_text, mistral_time = solve_captcha(
                base64_image, model_name=mistral_model, is_multiselect=is_multiselect)
            results.append({"agent": f"Mistral {mistral_model}", "response": mistral_text.strip().lower(
            ), "time": f"{mistral_time}s", "correct": validation_function(correct_answer, mistral_text.strip().lower())})

        # Process Groq models
        if is_multiselect:
            for groq_model in groq_models:
                _, groq_text, groq_time = solve_captcha(
                    image_url, model_name=groq_model, is_multiselect=is_multiselect)
                results.append({"agent": f"Groq {groq_model}", "response": groq_text.strip().lower(
                ), "time": f"{groq_time}s", "correct": validation_function(correct_answer, groq_text.strip().lower())})

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


def apply_obfuscation2(img_np):
    """
    Apply a simulated Diff-CAPTCHA obfuscation effect.
    This version uses OpenCV's stylization to give the image an artistic look,
    then applies a random non-linear warp to further distort the image.
    """
    # Apply artistic stylization to the image
    styled_img = cv2.stylization(img_np, sigma_s=60, sigma_r=0.45)

    # Get image dimensions
    height, width, _ = styled_img.shape

    # Create a mesh grid of pixel indices
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Create random displacement fields
    dx = (np.random.rand(height, width) - 0.5) * 10
    dy = (np.random.rand(height, width) - 0.5) * 10

    # Calculate the new mapping for each pixel
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    # Warp the styled image using the computed mappings
    warped_img = cv2.remap(styled_img, map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)

    return warped_img


def apply_obfuscation3(img_np):
    """
    Apply Diff-CAPTCHA obfuscation using U-Net segmentation and diffusion.
    Maintains better text readability while still confusing AI models.
    """
    # Convert input to appropriate format
    img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Step 1: U-Net Segmentation with improved text detection
    text_mask = segment_text(img)
    text_mask = cv2.resize(text_mask, (img.shape[1], img.shape[0]))

    # Dilate the text mask slightly to protect text edges
    kernel = np.ones((3, 3), np.uint8)
    text_mask = cv2.dilate(text_mask, kernel, iterations=1)

    # Properly expand dimensions for broadcasting
    text_mask = np.repeat(text_mask[:, :, np.newaxis], 3, axis=2)

    # Step 2: Apply gentler diffusion to background
    background_mask = 1 - text_mask
    background = img * background_mask

    # Apply a more subtle diffusion effect
    diffused_background = apply_gentle_diffusion(background)

    # Step 3: Apply subtle style transfer to background
    styled_background = apply_subtle_style(diffused_background)

    # Step 4: Combine text and background with better blending
    text_region = img * text_mask

    # Add a slight blur to the edges for better blending
    blurred_mask = cv2.GaussianBlur(text_mask, (3, 3), 0)
    final_image = text_region + styled_background * (1 - blurred_mask)

    return cv2.cvtColor(final_image.astype(np.uint8), cv2.COLOR_BGR2RGB)


def apply_gentle_diffusion(image):
    """
    Apply a gentler diffusion process that maintains more structure.
    """
    # Convert to float32 and normalize
    img_float = image.astype(np.float32) / 255.0

    # Reduced parameters for gentler effect
    timesteps = 20  # Reduced from 50
    beta_start = 0.0001
    beta_end = 0.01  # Reduced from 0.02

    # Create noise schedule
    betas = np.linspace(beta_start, beta_end, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)

    # Add noise more gently
    noisy = img_float.copy()
    for t in range(timesteps):
        noise = np.random.normal(size=image.shape) * \
            0.5  # Reduced noise intensity
        noise_strength = np.sqrt(
            1.0 - alphas_cumprod[t]) * 0.7  # Reduced strength
        noisy = np.sqrt(alphas_cumprod[t]) * img_float + noise_strength * noise

    # Very light denoising to maintain structure
    denoised = cv2.fastNlMeansDenoisingColored(
        (noisy * 255).astype(np.uint8),
        None,
        5,    # Reduced from 10
        5,    # Reduced from 10
        7,    # Template window size
        21    # Search window size
    )

    return denoised


def apply_subtle_style(image):
    """
    Apply a more subtle artistic style that maintains clarity.
    """
    # Ensure image is in uint8 format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # 1. Gentle edge-preserving filter
    smoothed = cv2.edgePreservingFilter(
        image,
        flags=1,
        sigma_s=40,  # Reduced from 60
        sigma_r=0.3  # Reduced from 0.4
    )

    # 2. Lighter stylization
    styled = cv2.stylization(
        smoothed,
        sigma_s=40,  # Reduced from 60
        sigma_r=0.3  # Reduced from 0.45
    )

    # 3. Subtle detail enhancement
    enhanced = cv2.detailEnhance(
        styled,
        sigma_s=5,   # Reduced from 10
        sigma_r=0.1  # Reduced from 0.15
    )

    # 4. Add very subtle noise for texture
    noise = np.random.normal(0, 2, enhanced.shape).astype(
        np.uint8)  # Reduced from 5
    textured = cv2.add(enhanced, noise)

    # 5. Gentle contrast adjustment
    lab = cv2.cvtColor(textured, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(8, 8))  # Reduced from 3.0
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    final = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Blend with original image to maintain more clarity
    return cv2.addWeighted(image, 0.3, final, 0.7, 0)


def segment_text(image):
    """
    U-Net based text segmentation.
    Returns a binary mask identifying text regions.
    """
    # Get original dimensions
    original_h, original_w = image.shape[:2]

    # Convert to grayscale for simpler processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,  # Block size
        2    # Constant subtracted from mean
    )

    # Clean up noise
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Normalize to [0, 1]
    mask = binary.astype(np.float32) / 255.0

    return mask


@app.route("/clear_database", methods=["POST"])
def clear_database():
    """Clears all stored images from the database and refreshes history."""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        # Delete all rows
        cursor.execute("DELETE FROM images")
        conn.commit()
        conn.close()

        print("✅ Database cleared successfully.")
        return jsonify({"success": True, "message": "Database cleared."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
