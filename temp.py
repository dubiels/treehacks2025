import asyncio
import concurrent.futures

@app.route("/solve", methods=["POST"])
def solve():
    """Handle CAPTCHA solving via an image URL asynchronously."""
    try:
        data = request.get_json()
        is_multiselect = data.get("is_multiselect", False)
        validation_function = check_multiselect_correctness if is_multiselect else check_text_correctness

        if "image_url" not in data or "correct_answer" not in data:
            return jsonify({"error": "Image URL and correct answer are required"}), 400

        image_url = data["image_url"]
        correct_answer = data["correct_answer"].strip().lower()

        print(f"🔹 Processing CAPTCHA: {image_url} (Multiselect: {is_multiselect})")

        # ✅ Use ThreadPoolExecutor to run models concurrently
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []

            # ✅ Process OpenAI GPT-4o (Direct URL)
            futures.append(loop.run_in_executor(executor, solve_captcha, image_url, "gpt-4o", is_multiselect))

            # ✅ Process Gemini and Mistral (Base64)
            response = requests.get(image_url)
            if response.status_code != 200:
                return jsonify({"error": "Failed to fetch image from URL"}), 400

            image = Image.open(io.BytesIO(response.content))
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

            gemini_models = ["gemini-1.5-flash", "gemini-1.5-pro"]
            mistral_models = ["pixtral-12b-2409"]
            groq_models = ["llama-3.2-90b-vision-preview", "llama-3.2-11b-vision-preview"]

            for model in gemini_models + mistral_models:
                futures.append(loop.run_in_executor(executor, solve_captcha, base64_image, model, is_multiselect))

            for model in groq_models:
                futures.append(loop.run_in_executor(executor, solve_captcha, image_url, model, is_multiselect))

            # ✅ Collect results as they complete
            results = []
            completed_tasks = loop.run_until_complete(asyncio.gather(*futures))

            for (model_name, response_text, time_taken) in completed_tasks:
                response_text = response_text.strip().lower()
                results.append({
                    "agent": model_name,
                    "response": response_text,
                    "time": f"{time_taken}s",
                    "correct": validation_function(correct_answer, response_text),
                })

        return jsonify({
            "display_image": image_url,
            "correct_response": correct_answer,
            "results": results
        })

    except Exception as e:
        print(f"❌ Error in /solve: {e}")  # ✅ Debugging
        return jsonify({"error": str(e)}), 500
