from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import re
import string
import os

app = FastAPI(title="Emotion Intent Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and vectorizer
model  = joblib.load("best_model.pkl")
tfidf  = joblib.load("tfidf_vectorizer.pkl")

EMOTION_LABELS = {
    0: 'Sadness',
    1: 'Joy',
    2: 'Love',
    3: 'Anger',
    4: 'Fear',
    5: 'Surprise'
}

EMOTION_EMOJIS = {
    'Sadness': '😢',
    'Joy': '😊',
    'Love': '🥰',
    'Anger': '😠',
    'Fear': '😨',
    'Surprise': '😲'
}

EMOTION_ADVICE = {
    'Sadness': "It's okay to feel sad. Take it one step at a time. 🌸",
    'Joy': "That's amazing!! Spread that positive energy!! ✨",
    'Love': "Love is beautiful — cherish this feeling. 💕",
    'Anger': "Take a deep breath. Your feelings are valid. 🌿",
    'Fear': "You're braver than you think. One step at a time. 🌙",
    'Surprise': "Life loves to surprise us — embrace it!! 🦋"
}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

class TextInput(BaseModel):
    text: str

@app.get("/")
def root():
    return FileResponse("index.html")

@app.post("/predict")
def predict(input: TextInput):
    clean = preprocess_text(input.text)
    vec   = tfidf.transform([clean])
    pred  = int(model.predict(vec)[0])

    try:
        proba = model.predict_proba(vec)[0]
        confidence = round(float(max(proba)) * 100, 1)
        all_probs = {EMOTION_LABELS[i]: round(float(p)*100, 1) for i, p in enumerate(proba)}
    except:
        confidence = 100.0
        all_probs = {EMOTION_LABELS[pred]: 100.0}

    emotion = EMOTION_LABELS[pred]
    return {
        "emotion":     emotion,
        "emoji":       EMOTION_EMOJIS[emotion],
        "confidence":  confidence,
        "advice":      EMOTION_ADVICE[emotion],
        "all_scores":  all_probs,
        "input_text":  input.text
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": "Emotion Intent Classifier"}
