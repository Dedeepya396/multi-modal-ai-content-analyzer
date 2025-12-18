import os
import torch
import math
import gradio as gr
from PIL import Image
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoImageProcessor, AutoModelForImageClassification,
    logging
)
from openai import OpenAI
from groq import Groq
import cv2
import numpy as np
import torch.nn as nn
import librosa

logging.set_verbosity_error()

# -----------------------------
# API Keys (set via Space secrets)
# -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# TEXT DETECTION
# -----------------------------
def run_hf_detector(text, model_id="roberta-base-openai-detector"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, token=HF_TOKEN).to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    human_score, ai_score = float(probs[0]), float(probs[1])
    label = "AI-generated" if ai_score > human_score else "Human-generated"
    return {"ai_score": ai_score, "human_score": human_score, "hf_label": label}

def calculate_perplexity(text):
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    encodings = tokenizer(text, return_tensors="pt").to(device)
    max_length = model.config.n_positions
    if encodings.input_ids.size(1) > max_length:
        encodings.input_ids = encodings.input_ids[:, :max_length]
        encodings.attention_mask = encodings.attention_mask[:, :max_length]
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings.input_ids)
    loss = outputs.loss
    perplexity = math.exp(loss.item())
    label = "AI-generated" if perplexity < 60 else "Human-generated"
    return {"perplexity": perplexity, "perplexity_label": label}

def generate_text_explanation(text, ai_score, human_score):
    decision = "AI-generated" if ai_score > human_score else "Human-generated"
    prompt = f"""
    You are an AI text analysis expert. Explain concisely why this text was classified as '{decision}'.
    Text: "{text}"
    Explanation:"""
    response = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[{"role":"user","content":prompt}],
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def analyze_text(text):
    try:
        hf_out = run_hf_detector(text)
        hf_out["ai_score"] = float(hf_out["ai_score"])
        hf_out["human_score"] = float(hf_out["human_score"])
        diff = abs(hf_out["ai_score"] - hf_out["human_score"])
        confidence = "High" if diff>0.8 else "Medium" if diff>=0.3 else "Low"
        perp_out = calculate_perplexity(text)
        explanation = generate_text_explanation(text, hf_out["ai_score"], hf_out["human_score"])
        return {"ai_score": hf_out["ai_score"], "confidence": confidence, "explanation": explanation}
    except:
        return {"ai_score":0.0,"confidence":"Low","explanation":"Error analyzing text."}

# -----------------------------
# IMAGE DETECTION
# -----------------------------
image_model_name = "Ateeqq/ai-vs-human-image-detector"
image_processor = AutoImageProcessor.from_pretrained(image_model_name)
image_model = AutoModelForImageClassification.from_pretrained(image_model_name)
image_model.eval()

def generate_image_explanation(ai_probability,human_probability,confidence):
    prompt = f"""
    You are an AI image analysis expert.
    AI: {ai_probability:.4f}, Human: {human_probability:.4f}, Confidence: {confidence}
    Explain in 1-2 sentences why it was classified as {'AI-generated' if ai_probability>human_probability else 'Human-generated'}.
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":prompt}],
        temperature=0.6
    )
    return response.choices[0].message.content.strip()

def analyze_image(image):
    image = image.convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = image_model(**inputs).logits
    probabilities = torch.nn.functional.softmax(logits/6.0, dim=-1)[0]
    ai_prob, human_prob = probabilities[0].item(), probabilities[1].item()
    diff = abs(ai_prob-human_prob)
    confidence = "High" if diff>=0.7 else "Medium" if diff>=0.3 else "Low"
    explanation = generate_image_explanation(ai_prob, human_prob, confidence)
    return {"ai_probability": ai_prob, "confidence": confidence, "explanation": explanation}

# -----------------------------
# VIDEO DETECTION
# -----------------------------
def extract_frames(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps*frame_rate)
    frames,count = [],0
    while cap.isOpened():
        ret,frame = cap.read()
        if not ret: break
        if count%interval==0: frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        count+=1
    cap.release()
    return frames

def analyze_video(video_path):
    frames = extract_frames(video_path, frame_rate=1)
    if not frames: return {"error":"No frames extracted."}
    ai_probs,human_probs = [],[]
    for img in frames:
        inputs = image_processor(images=img, return_tensors="pt")
        with torch.no_grad(): logits = image_model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        ai_probs.append(probs[0].item())
        human_probs.append(probs[1].item())
    avg_ai,avg_human = float(np.mean(ai_probs)), float(np.mean(human_probs))
    diff = abs(avg_ai-avg_human)
    confidence = "High" if diff>=0.7 else "Medium" if diff>=0.3 else "Low"
    prompt = f"Video processed {len(frames)} frames. AI: {avg_ai:.4f}, Human: {avg_human:.4f}. Confidence: {confidence}. Explain why it was {'AI-generated' if avg_ai>avg_human else 'Human-generated'}."
    response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role":"user","content":prompt}], temperature=0.6)
    explanation = response.choices[0].message.content.strip()
    return {"ai_probability":avg_ai,"confidence":confidence,"explanation":explanation}

# -----------------------------
# AUDIO DETECTION
# -----------------------------
class AudioCNNRNN(nn.Module):
    def __init__(self,lstm_hidden_size=128,num_classes=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,32,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden_size,batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size,num_classes)
    def forward(self,x):
        b,s,c,h,w = x.size()
        x = self.cnn(x.view(b*s,c,h,w)).mean(dim=[2,3]).view(b,s,-1)
        out,_ = self.lstm(x)
        return self.fc(out[:,-1,:])

def extract_mel_spectrogram(audio_path, sr=16000, n_mels=64):
    waveform,_ = librosa.load(audio_path,sr=sr)
    mel_spec = librosa.feature.melspectrogram(waveform,sr,n_mels=n_mels)
    return librosa.power_to_db(mel_spec,ref=np.max)

def slice_spectrogram(mel_spec,slice_size=128,step=64):
    return [mel_spec[:,i:i+slice_size] for i in range(0, mel_spec.shape[1]-slice_size, step)]

def analyze_audio(audio_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNNRNN().to(device).eval()
    mel_spec = extract_mel_spectrogram(audio_path)
    slices = slice_spectrogram(mel_spec)
    if not slices: return {"ai_probability":0,"confidence":"Low","explanation":"Audio too short."}
    data = torch.stack([torch.tensor(s).unsqueeze(0) for s in slices]).unsqueeze(0).to(device)
    with torch.no_grad(): logits = model(data)
    probabilities = torch.nn.functional.softmax(logits/3.0, dim=-1)[0]
    ai_prob,human_prob = probabilities[0].item(),probabilities[1].item()
    diff = abs(ai_prob-human_prob)
    confidence = "High" if diff>=0.7 else "Medium" if diff>=0.3 else "Low"
    prompt = f"Audio AI:{ai_prob:.4f} Human:{human_prob:.4f} Confidence:{confidence}. Explain reasoning."
    response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role":"user","content":prompt}], temperature=0.6)
    return {"ai_probability":ai_prob,"confidence":confidence,"explanation":response.choices[0].message.content.strip()}

# -----------------------------
# GRADIO UI
# -----------------------------
def format_text_results(text):
    res = analyze_text(text)
    conf_map = {"High":"🟢 High","Medium":"🟡 Medium","Low":"🔴 Low"}
    return f"### Text Detection\nAI Score: {res['ai_score']:.4f}\nConfidence: {conf_map.get(res['confidence'],res['confidence'])}\nExplanation: {res['explanation']}"

def format_image_results(image):
    res = analyze_image(image)
    return f"### Image Detection\nAI Probability: {res['ai_probability']:.4f}\nConfidence: {res['confidence']}\nExplanation: {res['explanation']}"

def format_video_results(video_file):
    res = analyze_video(video_file)
    if "error" in res: return res["error"]
    return f"### Video Detection\nAI Probability: {res['ai_probability']:.4f}\nConfidence: {res['confidence']}\nExplanation: {res['explanation']}"

def format_audio_results(audio_file):
    res = analyze_audio(audio_file)
    return f"### Audio Detection\nAI Probability: {res['ai_probability']:.4f}\nConfidence: {res['confidence']}\nExplanation: {res['explanation']}"

with gr.Blocks() as app:
    home = gr.Column(visible=True)
    with home:
        gr.Markdown("## AI Multi-Modal Detector")
        with gr.Row():
            t_btn = gr.Button("Text")
            i_btn = gr.Button("Image")
            v_btn = gr.Button("Video")
            a_btn = gr.Button("Audio")

    text_page = gr.Column(visible=False)
    with text_page:
        inp = gr.Textbox(lines=5, placeholder="Paste text...", label="Text")
        out = gr.Markdown()
        gr.Button("Analyze").click(format_text_results, inputs=inp, outputs=out)
        gr.Button("Back").click(lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[home,text_page])

    image_page = gr.Column(visible=False)
    with image_page:
        inp = gr.Image(type="pil")
        out = gr.Markdown()
        gr.Button("Analyze").click(format_image_results, inputs=inp, outputs=out)
        gr.Button("Back").click(lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[home,image_page])

    video_page = gr.Column(visible=False)
    with video_page:
        inp = gr.Video()
        out = gr.Markdown()
        gr.Button("Analyze").click(format_video_results, inputs=inp, outputs=out)
        gr.Button("Back").click(lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[home,video_page])

    audio_page = gr.Column(visible=False)
    with audio_page:
        inp = gr.Audio(type="filepath")
        out = gr.Markdown()
        gr.Button("Analyze").click(format_audio_results, inputs=inp, outputs=out)
        gr.Button("Back").click(lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[home,audio_page])

    t_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True)), outputs=[home,text_page])
    i_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True)), outputs=[home,image_page])
    v_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True)), outputs=[home,video_page])
    a_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True)), outputs=[home,audio_page])

app.launch()
