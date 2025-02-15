import requests
import os
import time
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import base64
import google.generativeai as genai
from mistralai import Mistral

# 🔹 Load API Keys from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ ERROR: GOOGLE_API_KEY not found in .env file.")
if not MISTRAL_API_KEY:
    raise ValueError("❌ ERROR: MISTRAL_API_KEY not found in .env file.")

# 🔹 Initialize APIs
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

# 🔹 Function to Download CAPTCHA Image
def download_captcha(url):
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        raise Exception("Failed to download CAPTCHA image.")

# 🔹 Function to Convert Image to Base64
def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# 🔹 Function to Solve CAPTCHA using Google Gemini
def solve_with_gemini(image_bytes):
    try:
        response = gemini_model.generate_content(
            [
                {"mime_type": "image/png", "data": image_bytes},
                "Extract the text from this CAPTCHA image."
            ]
        )
        return response.text
    except Exception as e:
        return f"Gemini Error: {str(e)}"

# 🔹 Function to Solve CAPTCHA using Mistral AI
def solve_with_mistral(base64_image):
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract only the CAPTCHA text from this image. Do not include any extra instructions or commentary."
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{base64_image}"
                    }
                ]
            }
        ]

        chat_response = mistral_client.chat.complete(
            model="pixtral-12b-2409",
            messages=messages
        )

        return chat_response.choices[0].message.content.strip()
    except Exception as e:
        return f"Mistral Error: {str(e)}"


# 🔹 Main Function to Solve CAPTCHA (Choose AI Model)
def solve_captcha(image_data, model="gemini"):
    try:
        start_time = time.time()

        # Check if input is already Base64
        if image_data.startswith("http"):
            # If a URL is given, download and convert to Base64
            captcha_image = download_captcha(image_data)
            base64_image = encode_image(captcha_image)
        else:
            # If already Base64, use it directly
            base64_image = image_data

        # Solve based on the model
        if model == "gemini":
            result = solve_with_gemini(base64_image)
        elif model == "mistral":
            result = solve_with_mistral(base64_image)
        else:
            raise ValueError("Invalid model. Choose 'gemini' or 'mistral'.")

        end_time = time.time()
        time_taken = round(end_time - start_time, 3)
        return result, time_taken
    
    except Exception as e:
        return f"Error: {str(e)}", None


# 🔹 Example Usage (Run with Gemini or Mistral)
captcha_url = "https://cf-assets.www.cloudflare.com/slt3lc6tev37/3pwMuJ55jpErAafgrWbyTr/e6c487ac6e4288dfe284db72b88ea3d1/captcha.png"

# Run with Google Gemini
captcha_text_gemini, time_taken_gemini = solve_captcha(captcha_url, model="gemini")
print(f"🔵 Gemini CAPTCHA Solved: {captcha_text_gemini} (Time: {time_taken_gemini}s)")

# Run with Mistral AI
captcha_text_mistral, time_taken_mistral = solve_captcha(captcha_url, model="mistral")
print(f"🟠 Mistral CAPTCHA Solved: {captcha_text_mistral} (Time: {time_taken_mistral}s)")