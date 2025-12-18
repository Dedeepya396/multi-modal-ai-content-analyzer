# 🧠 Multi modal AI content analyzer
An end-to-end multi-modal AI analyzer system that estimates the likelihood of content being AI-generated across text, image, video, and audio modalities.
This project is deployed as an interactive Gradio application on Hugging Face Spaces.

---

## 🚀 Live Demo

🔗 Hugging Face Space:  
👉 https://huggingface.co/spaces/satyahaha/B2B

---

## 📌 Features

### ✅ Text Detection
- Transformer-based AI vs Human text classification
- Confidence estimation (High / Medium / Low)
- LLM-generated human-readable explanation

### ✅ Image Detection
- Vision transformer-based AI vs Human image classifier
- Softmax calibration for stable probabilities
- Confidence scoring with explanation

### ✅ Video Detection
- Frame-wise extraction and classification
- Aggregated prediction across video frames
- Robust to short or low-FPS videos

### ✅ Audio Detection
- CNN + LSTM architecture on Mel-Spectrograms
- Temporal modeling for synthetic speech detection
---

## 🧩 Tech Stack

### Core Libraries
- PyTorch
- Hugging Face Transformers
- Gradio
- OpenCV
- Librosa
- NumPy

### Models Used
#### Text
- `roberta-base-openai-detector`
- GPT-2 (Perplexity-based analysis)

#### Image & Video
- `Ateeqq/ai-vs-human-image-detector`

#### Audio
- Custom CNN + LSTM neural network

#### Explanation Generation
- Groq LLMs (`llama-3.3-70b-versatile`, `gemma2-9b-it`)

---

## ⚙️ Installation (Local Setup)

```bash
git clone https://github.com/dedeepya396/ai-multimodal-detector.git
cd ai-multimodal-detector
pip install -r requirements.txt
```

## 🔑 Environment Variables:
Set the following environment variables before running the app:

export HF_TOKEN=your_huggingface_token

export GROQ_API_KEY=your_groq_api_key

## 🖥️ Usage:
1. Open the application
2. Select modality
    - Text
    - Image
    - Audio
    - Video
3. Upload or paste the inputs
4. Click Analyze
5. View:
    - AI probability
    - Confidence level
    - Explanation

