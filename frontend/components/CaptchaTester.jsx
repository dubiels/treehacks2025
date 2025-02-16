import React, { useState, useEffect, useMemo } from "react";
import { Clock, Check, X, AlertTriangle, Circle, Loader } from "lucide-react";

const CaptchaTester = () => {
    const [imageUrl, setImageUrl] = useState("");
    const [correctAnswer, setCorrectAnswer] = useState("");
    const [isMultiselect, setIsMultiselect] = useState(false); // ✅ New state for CAPTCHA type
    const [results, setResults] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [obfuscatedImage, setObfuscatedImage] = useState(null);
    const [isObfuscating, setIsObfuscating] = useState(false);
    const [errorMessage, setErrorMessage] = useState("");

    // ✅ Dynamically determine models based on isMultiselect
    const models = useMemo(() => {
      const baseModels = [
          "OpenAI GPT-4o",
          "Google gemini-1.5-flash",
          "Google gemini-2.0-flash",
          "Mistral pixtral-12b-2409",
      ];
      const groqModels = [
          "Groq llama-3.2-90b-vision-preview",
          "Groq llama-3.2-11b-vision-preview",
      ];
      return isMultiselect ? [...baseModels, ...groqModels] : baseModels;
    }, [isMultiselect]);
    useEffect(() => {
        setResults(
            models.map((model) => ({
                agent: model,
                response: "",
                time: "",
                correct: null,
                status: "idle",
            }))
        );
    }, [models]);

    const isValidImageUrl = (url) => /\.(jpg|jpeg|png)$/i.test(url);

    const handleImageUrlChange = async (e) => {
      const url = e.target.value;
      setImageUrl(url);
  
      // ✅ Validate URL format
      const isValid = /\.(jpg|jpeg|png)$/i.test(url);
      setErrorMessage(!isValid && url.trim() !== "" ? "Invalid image URL. Only .jpg and .png are supported." : "");
  
      if (isValid) {
          try {
              // ✅ Send URL to backend to save in DB
              await fetch("http://127.0.0.1:5000/save_image_url", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ image_url: url }),
              });
  
              console.log("✅ Image URL saved:", url); // Debugging
          } catch (error) {
              console.error("❌ Error saving image URL:", error);
          }
      }
  };  

    const handleSubmit = async () => {
        if (!imageUrl || !correctAnswer || errorMessage) return;

        setIsLoading(true);
        setResults((prev) =>
            prev.map((result) => ({ ...result, status: "loading" }))
        );

        try {
            const response = await fetch("http://127.0.0.1:5000/solve", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    image_url: imageUrl,
                    correct_answer: correctAnswer,
                    is_multiselect: isMultiselect, // ✅ Pass the toggle state to the backend
                }),
            });

            const data = await response.json();

            if (data.error) {
                console.error("Error:", data.error);
                setErrorMessage(`Server Error: ${data.error}`);
            } else {
                setResults((prev) =>
                    prev.map((result) => {
                        const matchingResult = data.results.find(
                            (r) => r.agent === result.agent
                        );
                        return matchingResult
                            ? { ...matchingResult, status: "completed" }
                            : { ...result, status: "idle" };
                    })
                );
            }
        } catch (error) {
            console.error("Submission error:", error);
            setErrorMessage("Network error. Please try again.");
        } finally {
            setIsLoading(false);
        }
    };

    const handleObfuscate = async () => {
        if (!imageUrl || errorMessage) return;

        setIsObfuscating(true);

        try {
            const response = await fetch("http://127.0.0.1:5000/obfuscate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image_url: imageUrl }),
            });

            const data = await response.json();
            if (data.image_url) {
                setObfuscatedImage(data.image_url);
                setImageUrl(data.image_url);
            } else {
                console.error("Obfuscation failed:", data.error);
                setErrorMessage(`Server Error: ${data.error}`);
            }
        } catch (error) {
            console.error("Error obfuscating:", error);
            setErrorMessage("Network error. Please try again.");
        } finally {
            setIsObfuscating(false);
        }
    };

    const getStatusIcon = (status, correct) => {
        const iconSize = 24; // Set a consistent size for all icons
        switch (status) {
            case "idle":
                return <Circle className="text-gray-500" size={iconSize} />;
            case "loading":
                return (
                    <Loader
                        className="text-blue-500 animate-spin"
                        size={iconSize}
                    />
                );
            case "completed":
                return correct ? (
                    <Check className="text-green-500" size={iconSize} />
                ) : (
                    <X className="text-red-500" size={iconSize} />
                );
            default:
                return null;
        }
    };

    return (
        <div className="min-h-screen bg-gray-900 text-white p-8">
            <div className="max-w-6xl mx-auto">
                <h1 className="text-4xl font-bold text-center mb-2 mt-12">
                    Are your CAPTCHAs AI-resistant?
                </h1>
                <p className="text-gray-400 text-center mb-12">
                    Instantly test against multiple industry-standard models 👇
                </p>

                {errorMessage && (
                    <div className="bg-red-500 text-white px-4 py-2 rounded-lg text-center mb-6 flex items-center gap-2">
                        <AlertTriangle size={20} />
                        {errorMessage}
                    </div>
                )}

                {/* ✅ Improved Toggle for CAPTCHA Type */}
                <div className="flex items-center justify-center mb-6">
                    <span
                        className={`text-sm font-medium ${
                            !isMultiselect ? "text-white" : "text-gray-400"
                        }`}
                    >
                        Text CAPTCHA
                    </span>
                    <div
                        className={`relative flex items-center w-14 h-7 mx-3 bg-gray-600 rounded-full cursor-pointer transition-all ${
                            isMultiselect ? "bg-green-500" : "bg-gray-500"
                        }`}
                        onClick={() => setIsMultiselect(!isMultiselect)}
                    >
                        <div
                            className={`absolute w-6 h-6 bg-white rounded-full shadow-md transform transition-all ${
                                isMultiselect
                                    ? "translate-x-7"
                                    : "translate-x-1"
                            }`}
                        />
                    </div>
                    <span
                        className={`text-sm font-medium ${
                            isMultiselect ? "text-white" : "text-gray-400"
                        }`}
                    >
                        Image Multi-Select CAPTCHA
                    </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div className="space-y-6">
                        <div className="bg-gray-800 rounded-lg p-6">
                            <h2 className="text-xl font-semibold mb-4">
                                Enter CAPTCHA Image URL:
                            </h2>
                            <input
                                type="text"
                                value={imageUrl}
                                onChange={handleImageUrlChange}
                                className="w-full bg-gray-700 border border-gray-600 rounded px-4 py-2"
                                placeholder="Paste CAPTCHA image URL (must end in .jpg or .png)"
                            />
                            {imageUrl && isValidImageUrl(imageUrl) && (
                                <div className="mt-4">
                                    <img
                                        src={imageUrl}
                                        alt="CAPTCHA preview"
                                        className="max-w-full h-auto"
                                    />
                                </div>
                            )}
                        </div>

                        <div className="bg-gray-800 rounded-lg p-6">
                            <h2 className="text-xl font-semibold mb-4">
                                {isMultiselect
                                    ? "Correct Boxes (Start with 1, Comma Separated)"
                                    : "Correct Response:"}
                            </h2>
                            <input
                                type="text"
                                value={correctAnswer}
                                onChange={(e) =>
                                    setCorrectAnswer(e.target.value)
                                }
                                className="w-full bg-gray-700 border border-gray-600 rounded px-4 py-2"
                                placeholder={
                                    isMultiselect
                                        ? "e.g., 2,5,8"
                                        : "Enter the correct CAPTCHA text"
                                }
                            />
                        </div>

                        <button
                            onClick={handleSubmit}
                            disabled={
                                !imageUrl ||
                                !correctAnswer ||
                                isLoading ||
                                errorMessage
                            }
                            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed py-3 rounded-lg font-semibold"
                        >
                            {isLoading ? "Testing..." : "Test CAPTCHA"}
                        </button>

                        <button
                            onClick={handleObfuscate}
                            disabled={
                                !imageUrl || isObfuscating || errorMessage
                            }
                            className="w-full bg-orange-600 hover:bg-orange-700 disabled:bg-gray-600 disabled:cursor-not-allowed py-3 rounded-lg font-semibold mt-4"
                        >
                            {isObfuscating ? "Obfuscating..." : "Obfuscate"}
                        </button>
                    </div>

                    <div className="bg-gray-800 rounded-lg p-6">
                        <div className="grid grid-cols-3 gap-4 mb-4">
                            <h3 className="font-semibold">Agent</h3>
                            <h3 className="font-semibold">Response:</h3>
                            <h3 className="font-semibold">Time:</h3>
                        </div>
                        <div className="space-y-4">
                            {results.map((result, index) => (
                                <div
                                    key={index}
                                    className="grid grid-cols-3 gap-4 items-center"
                                >
                                    <div className="flex items-center gap-2">
                                        <div className="w-6 h-6 flex items-center justify-center">
                                            {getStatusIcon(
                                                result.status,
                                                result.correct
                                            )}
                                        </div>
                                        {result.agent}
                                    </div>

                                    <div>{result.response}</div>
                                    <div className="flex items-center gap-2">
                                        <Clock size={16} />
                                        {result.time}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default CaptchaTester;
