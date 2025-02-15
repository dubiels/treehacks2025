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
import uuid

app = Flask(__name__, static_folder="dist", static_url_path="/")
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/", defaults={'path': ''})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

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

        # Generate a unique filename
        unique_filename = f"{uuid.uuid4().hex}.png"

        # Ensure the static directory exists
        static_dir = "public/temp"
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)

        # Save the obfuscated image
        obfuscated_path = os.path.join(static_dir, unique_filename)
        obfuscated_pil.save(obfuscated_path)

        return jsonify({
            "image_url": f"https://treehacks2025-one.vercel.app/temp/{unique_filename}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Serve obfuscated CAPTCHAs dynamically
@app.route("/temp/<filename>")
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
        _, openai_text, openai_time = solve_captcha(image_url, model_name="gpt-4o")

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
            {"agent": "OpenAI GPT-4o", "response": openai_text.strip().lower(), "time": f"{openai_time}s", "correct": openai_text.strip().lower() == correct_answer}
        ]

        # Process Gemini models
        for gemini_model in gemini_models:
            _, gemini_text, gemini_time = solve_captcha(base64_image, model_name=gemini_model)
            results.append({"agent": f"Google {gemini_model}", "response": gemini_text.strip().lower(), "time": f"{gemini_time}s", "correct": gemini_text.strip().lower() == correct_answer})

        # Process Mistral models
        for mistral_model in mistral_models:
            _, mistral_text, mistral_time = solve_captcha(base64_image, model_name=mistral_model)
            results.append({"agent": f"Mistral {mistral_model}", "response": mistral_text.strip().lower(), "time": f"{mistral_time}s", "correct": mistral_text.strip().lower() == correct_answer})

        # Process Groq models
        for groq_model in groq_models:
            _, groq_text, groq_time = solve_captcha(image_url, model_name=groq_model)
            results.append({"agent": f"Mistral {groq_model}", "response": groq_text.strip().lower(), "time": f"{groq_time}s", "correct": groq_text.strip().lower() == correct_answer})

        response = {
            "display_image": image_url,
            "correct_response": correct_answer,
            "results": results
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# def apply_obfuscation(img_np):
#     """Apply AI-avoidant obfuscation techniques."""
#     height, width, _ = img_np.shape
#     for i in range(height):
#         offset = int(5 * np.sin(2.0 * np.pi * i / 50))
#         img_np[i] = np.roll(img_np[i], offset, axis=0)

#     noise = np.random.normal(0, 15, img_np.shape).astype("uint8")
#     img_np = cv2.add(img_np, noise)
#     img_np = cv2.GaussianBlur(img_np, (3, 3), 0)

#     return img_np

# if __name__ == "__main__":
#     app.run(debug=True)





def apply_obfuscation(img_np):
    """
    Apply Diff-CAPTCHA obfuscation using a combination of:
    1. U-Net segmentation to protect text regions
    2. Denoising diffusion for background transformation
    3. Style transfer for artistic background manipulation
    """
    # Convert input to appropriate format
    img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    original_shape = img.shape[:2]  # Store original dimensions
    
    # Step 1: U-Net Segmentation to separate text and background
    text_mask = segment_text(img)  # Returns binary mask
    
    # Resize mask back to original dimensions
    text_mask = cv2.resize(text_mask, (img.shape[1], img.shape[0]))
    text_mask = np.expand_dims(text_mask, axis=-1)  # Add channel dimension
    
    # Step 2: Apply diffusion model to background
    background_mask = 1 - text_mask
    background = img * background_mask
    diffused_background = apply_diffusion(background)
    
    # Step 3: Style transfer on diffused background
    styled_background = apply_style_transfer(diffused_background)
    
    # Step 4: Combine text and transformed background
    text_region = img * text_mask
    final_image = text_region + styled_background * background_mask
    
    return cv2.cvtColor(final_image.astype(np.uint8), cv2.COLOR_BGR2RGB)

def segment_text(image):
    """
    U-Net based text segmentation.
    Returns a binary mask identifying text regions.
    """
    # Get original dimensions
    original_h, original_w = image.shape[:2]
    
    # Preprocess image
    processed = preprocess_for_unet(image)
    
    # Get U-Net model
    unet = load_unet_model()
    
    if unet is None:
        # Fallback to basic thresholding if model fails to load
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask
    
    # Get text mask prediction
    mask = unet.predict(processed)[0]  # Get first item from batch
    
    # Resize mask back to original dimensions
    mask = cv2.resize(mask, (original_w, original_h))
    
    # Threshold the mask to get binary values
    _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
    
    return mask

def preprocess_for_unet(image, target_size=(256, 256)):
    """
    Preprocess image for U-Net input.
    Includes resizing, normalization, and formatting.
    """
    # Resize image
    processed = cv2.resize(image, target_size)
    
    # Normalize pixel values to [0, 1]
    processed = processed.astype(np.float32) / 255.0
    
    # Expand dimensions for batch processing
    processed = np.expand_dims(processed, axis=0)
    
    return processed

def apply_diffusion(image):
    """
    Apply denoising diffusion model to transform the background.
    """
    # Initialize diffusion model
    diffusion_model = load_diffusion_model()
    
    # Normalize image
    image_norm = image.astype(np.float32) / 255.0
    
    # Apply diffusion process
    diffused = diffusion_model.denoise(image_norm)
    
    # Convert back to uint8
    return (diffused * 255).astype(np.uint8)

def apply_style_transfer(image):
    """
    Apply artistic style transfer to the background.
    Ensures proper handling of image types and ranges.
    """
    # Ensure image is in uint8 format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Apply style transfer with adjusted parameters
    styled = cv2.stylization(image, sigma_s=60, sigma_r=0.45)
    
    return styled

def load_unet_model():
    """
    Load pretrained U-Net model for text segmentation.
    Uses a simplified U-Net architecture suitable for CAPTCHA text segmentation.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model

        def create_unet(input_shape=(256, 256, 3)):
            # Encoder
            inputs = layers.Input(input_shape)
            
            # Encoder path
            conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
            conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
            pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
            
            conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
            conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
            pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
            
            # Bridge
            conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
            conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
            
            # Decoder path
            up1 = layers.UpSampling2D(size=(2, 2))(conv3)
            up1 = layers.concatenate([up1, conv2], axis=3)
            conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(up1)
            conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv4)
            
            up2 = layers.UpSampling2D(size=(2, 2))(conv4)
            up2 = layers.concatenate([up2, conv1], axis=3)
            conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(up2)
            conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv5)
            
            # Output
            outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv5)
            
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            return model

        model = create_unet()
        
        # Load pretrained weights if they exist
        weights_path = 'models/unet_captcha_weights.h5'
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
        
        return model
    
    except Exception as e:
        print(f"Error loading U-Net model: {str(e)}")
        return None

def load_diffusion_model():
    """
    Load pretrained diffusion model for background transformation.
    Implements a simplified diffusion process using a custom noise schedule.
    """
    class SimpleDiffusion:
        def __init__(self, timesteps=1000):
            self.timesteps = timesteps
            self.beta = np.linspace(0.0001, 0.02, timesteps)
            self.alpha = 1 - self.beta
            self.alpha_hat = np.cumprod(self.alpha)
            
        def add_noise(self, x, t):
            """Add noise to the input image at timestep t"""
            sqrt_alpha_hat = np.sqrt(self.alpha_hat[t])
            sqrt_one_minus_alpha_hat = np.sqrt(1 - self.alpha_hat[t])
            eps = np.random.normal(size=x.shape)
            return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps
        
        def denoise(self, x, steps=100):
            """Denoise the input image using the reverse diffusion process"""
            x_noisy = x.copy()
            for t in reversed(range(steps)):
                z = np.random.normal(size=x.shape) if t > 0 else 0
                alpha = self.alpha[t]
                alpha_hat = self.alpha_hat[t]
                beta = self.beta[t]
                
                x_noisy = 1 / np.sqrt(alpha) * (
                    x_noisy - (beta / np.sqrt(1 - alpha_hat)) * z
                )
                x_noisy = np.clip(x_noisy, 0, 1)
            
            return x_noisy
    
    return SimpleDiffusion()

if __name__ == "__main__":
    app.run(debug=True)







#  def apply_obfuscation(img_np):
#     """
#     Apply a simulated Diff-CAPTCHA obfuscation effect.
#     This version uses OpenCV's stylization to give the image an artistic look,
#     then applies a random non-linear warp to further distort the image.
#     """
#     # Apply artistic stylization to the image (simulate style transfer)
#     # sigma_s and sigma_r control the level of stylization
#     styled_img = cv2.stylization(img_np, sigma_s=60, sigma_r=0.45)
    
#     # Get image dimensions
#     height, width, _ = styled_img.shape

#     # Create a mesh grid of pixel indices
#     x, y = np.meshgrid(np.arange(width), np.arange(height))
    
#     # Create random displacement fields (simulate diffusion-like distortions)
#     dx = (np.random.rand(height, width) - 0.5) * 10  # up to ±5 pixels horizontally
#     dy = (np.random.rand(height, width) - 0.5) * 10  # up to ±5 pixels vertically

#     # Calculate the new mapping for each pixel
#     map_x = (x + dx).astype(np.float32)
#     map_y = (y + dy).astype(np.float32)
    
#     # Warp the styled image using the computed mappings
#     warped_img = cv2.remap(styled_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
#     return warped_img
