import requests
import os
import time
import base64
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import google.generativeai as genai
from mistralai import Mistral
import openai  # OpenAI library

# 🔹 Load API Keys from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ ERROR: GOOGLE_API_KEY not found in .env file.")
if not MISTRAL_API_KEY:
    raise ValueError("❌ ERROR: MISTRAL_API_KEY not found in .env file.")
if not OPENAI_API_KEY:
    raise ValueError("❌ ERROR: OPENAI_API_KEY not found in .env file.")

# 🔹 Initialize APIs
genai.configure(api_key=GOOGLE_API_KEY)
gemini_models = {
    "gemini-1.5-flash": genai.GenerativeModel("gemini-1.5-flash"),
    "gemini-1.5-pro": genai.GenerativeModel("gemini-1.5-pro")
}
mistral_client = Mistral(api_key=MISTRAL_API_KEY)
openai.api_key = OPENAI_API_KEY

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
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# 🔹 Function to Solve CAPTCHA using Google Gemini (Base64)
def solve_with_gemini(image_bytes, model_name="gemini-1.5-flash"):
    try:
        model = gemini_models[model_name]
        response = model.generate_content(
            [
                {"mime_type": "image/png", "data": image_bytes},
                "Extract the text from this CAPTCHA image."
            ]
        )
        return response.text
    except Exception as e:
        return f"{model_name} Error: {str(e)}"

# 🔹 Function to Solve CAPTCHA using Mistral AI (Base64)
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
        return f"pixtral-12b-2409 Error: {str(e)}"

# 🔹 Function to Solve CAPTCHA using OpenAI GPT-4o (Direct Image URL)
def solve_with_openai(image_url):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an OCR tool that extracts text from images."},
                {"role": "user", "content": [
                    {"type": "text", "text": 
                     "This image contains a unique alphanumeric code that I need to redeem. "
                     "Extract only the exact code from the image. **Do not add any other words or explanations.** "
                     "Return only the extracted text, without quotation marks or formatting."},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]}
            ],
            max_tokens=50,
            temperature=0,
            n=1
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"gpt-4o Error: {str(e)}"

# 🔹 Main Function to Solve CAPTCHA (Gemini/Mistral use Base64, OpenAI uses URL)
def solve_captcha(image_data, model_name):
    try:
        start_time = time.time()

        if image_data.startswith("http"):
            if model_name == "gpt-4o":
                image_url = image_data  # ✅ Use direct URL for OpenAI
            else:
                captcha_image = download_captcha(image_data)
                image_bytes = encode_image(captcha_image)
        else:
            image_bytes = image_data

        if model_name in ["gemini-1.5-flash", "gemini-1.5-pro"]:
            result = solve_with_gemini(image_bytes, model_name)
        elif model_name == "pixtral-12b-2409":
            result = solve_with_mistral(image_bytes)
        elif model_name == "gpt-4o":
            result = solve_with_openai(image_data)
        else:
            raise ValueError(f"Invalid model: {model_name}")

        end_time = time.time()
        time_taken = round(end_time - start_time, 3)
        return model_name, result, time_taken
    
    except Exception as e:
        return model_name, f"Error: {str(e)}", None

# 🔹 Example Usage
captcha_url = "https://cf-assets.www.cloudflare.com/slt3lc6tev37/4wCmCWsWiTB8ZG64tBVEKY/0499192ff9baf249fa2b45843c5d2948/recaptcha.png"

models_to_test = [
    "gemini-1.5-flash",  # ✅ Gemini Flash
    "gemini-1.5-pro",  # ✅ Gemini Pro
    "pixtral-12b-2409",  # ✅ Mistral
    "gpt-4o"  # ✅ GPT-4o (OpenAI)
]

# Run with all models
for model in models_to_test:
    model_name, captcha_text, time_taken = solve_captcha(captcha_url, model)
    print(f"🔹 {model_name.upper()} CAPTCHA Solved: {captcha_text} (Time: {time_taken}s)\n")
