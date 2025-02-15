import React, { useState } from 'react';
import { Upload, Clock, Check, X } from 'lucide-react';

const CaptchaTester = () => {
  const [image, setImage] = useState(null);
  const [correctAnswer, setCorrectAnswer] = useState('');
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [obfuscatedImage, setObfuscatedImage] = useState(null);
  const [isObfuscating, setIsObfuscating] = useState(false);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => setImage(e.target.result);
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async () => {
    if (!image || !correctAnswer) return;
  
    setIsLoading(true);
    setResults(null);
  
    const formData = new FormData();
    formData.append("image", dataURLtoBlob(image));
    formData.append("correct_answer", correctAnswer);
  
    try {
      const response = await fetch("http://127.0.0.1:5000/solve", {
        method: "POST",
        body: formData,
      });
  
      const data = await response.json();
  
      if (data.error) {
        console.error("Error:", data.error);
      } else {
        setResults(data.results);
      }
    } catch (error) {
      console.error("Submission error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleObfuscate = async () => {
    if (!image) return;
    
    setIsObfuscating(true);
  
    const formData = new FormData();
    formData.append("image", dataURLtoBlob(image));
  
    try {
      const response = await fetch("http://127.0.0.1:5000/obfuscate", {
        method: "POST",
        body: formData,
      });
  
      const data = await response.json();
      if (data.image_url) {
        setObfuscatedImage(data.image_url);
      } else {
        console.error("Obfuscation failed:", data.error);
      }
    } catch (error) {
      console.error("Error obfuscating:", error);
    } finally {
      setIsObfuscating(false);
    }
  };
  
  const dataURLtoBlob = (dataURL) => {
    const byteString = atob(dataURL.split(",")[1]);
    const mimeString = dataURL.split(",")[0].split(":")[1].split(";")[0];
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
    }
    return new Blob([ab], { type: mimeString });
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

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="space-y-6">
            <div className="bg-gray-800 rounded-lg p-6">
              <h2 className="text-xl font-semibold mb-4">Upload CAPTCHA:</h2>
              <div className="border-2 border-dashed border-gray-600 rounded-lg p-4 text-center">
                <input type="file" onChange={handleImageUpload} className="hidden" id="captcha-upload" accept="image/*" />
                <label htmlFor="captcha-upload" className="cursor-pointer block">
                  <Upload className="mx-auto mb-2" />
                  <span className="text-gray-400">Click to upload or drag and drop</span>
                </label>
              </div>
              {image && <img src={image} alt="CAPTCHA preview" className="mt-4 max-w-full h-auto" />}
            </div>

            <div className="bg-gray-800 rounded-lg p-6">
              <h2 className="text-xl font-semibold mb-4">Correct Response:</h2>
              <input type="text" value={correctAnswer} onChange={(e) => setCorrectAnswer(e.target.value)} className="w-full bg-gray-700 border border-gray-600 rounded px-4 py-2" placeholder="Enter the correct CAPTCHA text" />
            </div>

            <button onClick={handleSubmit} disabled={!image || !correctAnswer || isLoading} className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed py-3 rounded-lg font-semibold">
              {isLoading ? 'Testing...' : 'Test CAPTCHA'}
            </button>
            
            <button onClick={handleObfuscate} disabled={!image || isObfuscating} className="w-full bg-orange-600 hover:bg-orange-700 disabled:bg-gray-600 disabled:cursor-not-allowed py-3 rounded-lg font-semibold mt-4">
              {isObfuscating ? 'Obfuscating...' : 'Obfuscate'}
            </button>
          </div>
          
          <div className="bg-gray-800 rounded-lg p-6">
            <div className="grid grid-cols-3 gap-4 mb-4">
              <h3 className="font-semibold">Agent</h3>
              <h3 className="font-semibold">Response:</h3>
              <h3 className="font-semibold">Time:</h3>
            </div>
            <div className="space-y-4">
              {results?.map((result, index) => (
                <div key={index} className="grid grid-cols-3 gap-4 items-center">
                  <div className="flex items-center gap-2">
                    {result.correct ? <Check className="text-green-500" size={20} /> : <X className="text-red-500" size={20} />}
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
            {obfuscatedImage && <img src={obfuscatedImage} alt="Obfuscated CAPTCHA" className="mt-4 max-w-full h-auto" />}
          </div>
        </div>
      </div>
    </div>
  );
};

export default CaptchaTester;
