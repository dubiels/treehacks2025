import time
import requests
import google.generativeai as genai
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os

# 🔹 Load API Key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ ERROR: GOOGLE_API_KEY not found. Make sure you have a .env file and it contains the correct API key.")

# 🔹 Initialize Gemini API (Using 1.5 Flash)
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")  # Use updated model

# 🔹 Function to Download CAPTCHA Image from URL
def download_captcha(url):
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        raise Exception("Failed to download CAPTCHA image.")

# 🔹 Function to Convert Image to Raw Bytes (Needed for Gemini)
def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # Save as PNG format
    return buffered.getvalue()  # Return raw bytes

# 🔹 Function to Upload CAPTCHA to Google Gemini and Get the Result
def solve_captcha(image_url):
    try:
        start_time = time.time()  # Start timing

        # Download CAPTCHA image
        captcha_image = download_captcha(image_url)

        # Convert image to raw bytes
        image_bytes = encode_image(captcha_image)

        # Send to Gemini for text recognition
        response = model.generate_content(
            [
                {"mime_type": "image/png", "data": image_bytes},  # Image content
                "Extract the text from this CAPTCHA image."       # Text prompt
            ]
        )
        
        end_time = time.time()  # End timing
        time_taken = round(end_time - start_time, 3)  # Time in seconds

        return response.text, time_taken
    
    except Exception as e:
        return f"Error: {str(e)}", None

# 🔹 Example Usage
captcha_url = "https://cf-assets.www.cloudflare.com/slt3lc6tev37/3pwMuJ55jpErAafgrWbyTr/e6c487ac6e4288dfe284db72b88ea3d1/captcha.png"
captcha_text, time_taken = solve_captcha(captcha_url)

print(f"CAPTCHA Solved: {captcha_text}")
print(f"Time Taken: {time_taken} seconds")