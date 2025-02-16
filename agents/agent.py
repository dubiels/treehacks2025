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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ ERROR: GOOGLE_API_KEY not found in .env file.")
if not MISTRAL_API_KEY:
    raise ValueError("❌ ERROR: MISTRAL_API_KEY not found in .env file.")
if not OPENAI_API_KEY:
    raise ValueError("❌ ERROR: OPENAI_API_KEY not found in .env file.")
if not GROQ_API_KEY:
    raise ValueError("❌ ERROR: GROQ_API_KEY not found in .env file.")

# 🔹 Initialize APIs
genai.configure(api_key=GOOGLE_API_KEY)
gemini_models = {
    "gemini-1.5-flash": genai.GenerativeModel("gemini-1.5-flash"),
    "gemini-1.5-pro": genai.GenerativeModel("gemini-1.5-pro"),
    "gemini-2.0-flash": genai.GenerativeModel("gemini-2.0-flash")
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


def solve_with_gemini(image_bytes, model_name="gemini-1.5-flash", is_multiselect=False):
    try:
        model = gemini_models[model_name]
        prompt = (
            "Extract only the exact CAPTCHA text from this image. "
            "**Do not add any other words, explanations, formatting, or punctuation.** "
            "Return only the exact alphanumeric code exactly as seen in the image."
        ) if not is_multiselect else (
            "This is a multi-select CAPTCHA image. Identify the correct images matching the category and respond with a comma-separated list of their positions. "
            "The grid layout may be `3x3` or `4x4`. Make sure to not respond with any other characters, only the comma separated numbers. Determine the best layout based on the image and respond in the format below:\n"
            "**3x3 Example (if the grid is 3x3)**\n"
            "1 2 3\n"
            "4 5 6\n"
            "7 8 9\n"
            "**4x4 Example (if the grid is 4x4)**\n"
            "1 2 3 4\n"
            "5 6 7 8\n"
            "9 10 11 12\n"
            "13 14 15 16\n"
            "If no images match, respond with 'None'."
            "Example response that I expect from you: 2, 7, 9"
        )
        # print("PROMPT", prompt)

        response = model.generate_content(
            [
                {"mime_type": "image/png", "data": image_bytes},
                prompt
            ]
        )
        return response.text.strip()
    except Exception as e:
        return f"{model_name} Error: {str(e)}"


# 🔹 Function to Solve CAPTCHA using Mistral AI (Base64)
def solve_with_mistral(base64_image, is_multiselect=False):
    try:
        prompt = (
            "Extract only the CAPTCHA text from this image. Do not include any extra instructions or commentary."
        ) if not is_multiselect else (
            "This is a multi-select CAPTCHA image. Identify the correct images matching the category and respond with a comma-separated list of their positions. "
            "The grid layout may be `3x3` or `4x4`. Make sure to not respond with any other characters, only the comma separated numbers. Determine the best layout based on the image and respond in the format below:\n"
            "**3x3 Example (if the grid is 3x3)**\n"
            "1 2 3\n"
            "4 5 6\n"
            "7 8 9\n"
            "**4x4 Example (if the grid is 4x4)**\n"
            "1 2 3 4\n"
            "5 6 7 8\n"
            "9 10 11 12\n"
            "13 14 15 16\n"
            "If no images match, respond with 'None'."
            "Example response that I expect from you: 2, 7, 9"
        )
        # print("PROMPT", prompt)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
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


def solve_with_openai(image_url, is_multiselect=False):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    try:
        prompt = (
            "Extract only the exact CAPTCHA text from this image. Do not add any other words or explanations."
        ) if not is_multiselect else (
            "This is a multi-select CAPTCHA image. Identify the correct images matching the category and respond with a comma-separated list of their positions. "
            "The grid layout may be `3x3` or `4x4`. Make sure to not respond with any other characters, only the comma separated numbers. Determine the best layout based on the image and respond in the format below:\n"
            "**3x3 Example (if the grid is 3x3)**\n"
            "1 2 3\n"
            "4 5 6\n"
            "7 8 9\n"
            "**4x4 Example (if the grid is 4x4)**\n"
            "1 2 3 4\n"
            "5 6 7 8\n"
            "9 10 11 12\n"
            "13 14 15 16\n"
            "If no images match, respond with 'None'."
            "Example response that I expect from you: 2, 7, 9"
        )
        # print("PROMPT", prompt)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                    "content": "You are an OCR tool that extracts text from images."},
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

# 🔹 Function to Solve CAPTCHA using Groq API (Direct Image URL)


def solve_with_groq(image_url, model_name, is_multiselect=False):
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        prompt = (
            "Extract only the exact CAPTCHA text from this image. Do not add any other words or explanations. Do not include phrases like 'The CAPTCHA text is' or 'Extracted text:' or extra characters like ** or things like *Answer*."
        ) if not is_multiselect else (
            "This is a multi-select CAPTCHA image. Identify the correct images matching the category and respond with a comma-separated list of their positions. "
            "The grid layout may be `3x3` or `4x4`. DON'T GIVE ME ANY OF YOUR REASONING, ONLY THE NUMBERS. Make sure to not respond with any other characters, only the comma separated numbers. Determine the best layout based on the image and respond in the format below:\n"
            "**3x3 Example (if the grid is 3x3)**\n"
            "1 2 3\n"
            "4 5 6\n"
            "7 8 9\n"
            "**4x4 Example (if the grid is 4x4)**\n"
            "1 2 3 4\n"
            "5 6 7 8\n"
            "9 10 11 12\n"
            "13 14 15 16\n"
            "If no images match, respond with 'None'."
            "Example response that I expect from you: 2, 7, 9"
            "No periods at the end please, only comma separated numbers"
        )
        # print("PROMPT", prompt)

        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text":
                        "Extract only the exact CAPTCHA text from this image. "
                        "Do not add any explanations, formatting, or extra words. "
                        "Do not include phrases like 'The CAPTCHA text is' or 'Extracted text:' or extra characters like ** or things like *Answer*. "
                        "Only return the raw CAPTCHA code as it appears in the image."
                     },
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]}
            ],
            "max_tokens": 50
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)

        if response.status_code != 200:
            return f"{model_name} Error: HTTP {response.status_code} - {response.text}"

        response_data = response.json()

        if "choices" not in response_data or not response_data["choices"]:
            return f"{model_name} Error: No response received"

        return response_data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"{model_name} Error: {str(e)}"

# 🔹 Main Function to Solve CAPTCHA


def solve_captcha(image_data, model_name, is_multiselect=False):
    try:
        start_time = time.time()

        if image_data.startswith("http"):
            if model_name in ["gpt-4o", "llama-3.2-90b-vision-preview", "llama-3.2-11b-vision-preview"]:
                image_url = image_data
            else:
                captcha_image = download_captcha(image_data)
                image_bytes = encode_image(captcha_image)
        else:
            image_bytes = image_data

        if model_name in gemini_models:
            result = solve_with_gemini(image_bytes, model_name, is_multiselect)
        elif model_name == "pixtral-12b-2409":
            result = solve_with_mistral(image_bytes, is_multiselect)
        elif model_name == "gpt-4o":
            result = solve_with_openai(image_data, is_multiselect)
        elif model_name in ["llama-3.2-90b-vision-preview", "llama-3.2-11b-vision-preview"]:
            result = solve_with_groq(image_data, model_name, is_multiselect)
        else:
            raise ValueError(f"Invalid model: {model_name}")

        return model_name, result, round(time.time() - start_time, 3)
    except Exception as e:
        return model_name, f"Error: {str(e)}", None


def check_multiselect_correctness(correct_response, ai_response):
    """
    Checks if the AI model's multi-select CAPTCHA response matches the correct response.

    Args:
        correct_response (str): The correct selection string, e.g., "1,4,5".
        ai_response (str): The AI model's response, e.g., "1, 4, 5" or "**3x3**\n1, 2, 3".

    Returns:
        bool: True if the AI response matches the correct response, False otherwise.
    """
    try:
        # Convert correct response into a set of integers
        correct_set = set(map(int, correct_response.split(",")))

        # Extract only numeric values from AI response
        ai_response = ai_response.strip()
        # Take the last line if format includes **3x3**
        ai_numbers = ai_response.split("\n")[-1]
        ai_set = set(map(int, ai_numbers.replace(" ", "").split(",")))

        # Compare sets
        return ai_set == correct_set

    except Exception:
        # Handle errors (e.g., invalid AI response)
        return False


def check_text_correctness(correct_response, ai_response):
    """
    Checks if the AI model's text CAPTCHA response matches the correct response,
    allowing flexible spacing but preserving character order.

    Args:
        correct_response (str): The correct text-based CAPTCHA answer.
        ai_response (str): The AI model's response.

    Returns:
        bool: True if the AI response contains all correct characters in order, False otherwise.
    """
    try:
        # Remove all whitespace from correct response
        # Remove spaces/tabs/newlines
        correct_cleaned = "".join(correct_response.split())
        # Remove spaces/tabs/newlines
        ai_cleaned = "".join(ai_response.split())

        # Check if the AI response preserves character order
        correct_index = 0
        for char in ai_cleaned:
            if correct_index < len(correct_cleaned) and char == correct_cleaned[correct_index]:
                correct_index += 1

        # If all characters in correct_cleaned are found in order, it's correct
        return correct_index == len(correct_cleaned)

    except Exception:
        # Handle unexpected errors (e.g., empty responses, None values)
        return False


def test_agents():
    # 🔹 Example Usage
    captcha_url = "https://cf-assets.www.cloudflare.com/slt3lc6tev37/4wCmCWsWiTB8ZG64tBVEKY/0499192ff9baf249fa2b45843c5d2948/recaptcha.png"
    multiselect_url = "https://miro.medium.com/v2/resize:fit:1092/1*jcXPqzruCRYHItBKcVolrw.jpeg"

    models_to_test = [
        "gemini-1.5-flash",  # ✅ Google Gemini Flash
        "gemini-2.0-flash",
        "pixtral-12b-2409",  # ✅ Mistral
        "gpt-4o",  # ✅ OpenAI GPT-4o
        "llama-3.2-90b-vision-preview",  # ✅ Groq LLaMA 3 (8B)
        "llama-3.2-11b-vision-preview"  # ✅ Groq LLaMA 3 (70B)
    ]

    # Run with all models
    results = []
    for model in models_to_test:
        model_name, captcha_text, time_taken = solve_captcha(
            captcha_url, model)
        results.append((model_name, captcha_text, time_taken))

    # Print results in a clean format
    print("\n🔹 CAPTCHA SOLVING RESULTS 🔹\n")
    for model_name, captcha_text, time_taken in results:
        print(f"✅ {model_name.upper()} Solved: {captcha_text}")
        print(f"⏱ Time Taken: {time_taken}s\n")

    # Run with all models
    results = []
    for model in models_to_test:
        model_name, captcha_text, time_taken = solve_captcha(
            multiselect_url, model, is_multiselect=True)
        results.append((model_name, captcha_text, time_taken))

    # Print results in a clean format
    print("\n🔹 MULTISELECT CAPTCHA SOLVING RESULTS 🔹\n")
    for model_name, captcha_text, time_taken in results:
        print(f"✅ {model_name.upper()} Solved: {captcha_text}")
        print(f"⏱ Time Taken: {time_taken}s\n")

# test_agents()
