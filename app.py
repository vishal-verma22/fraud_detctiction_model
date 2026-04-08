from flask import Flask, request, jsonify
from flask_cors import CORS
from pickle import load
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from google.cloud import speech
import nltk
import os
import tempfile

# --------------------------
# Setup NLTK data directory
# --------------------------
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.download("punkt", download_dir=nltk_data_dir, quiet=True)
nltk.download("punkt_tab", download_dir=nltk_data_dir, quiet=True)
nltk.download("stopwords", download_dir=nltk_data_dir, quiet=True)

nltk.data.path.append(nltk_data_dir)

# --------------------------
# Google Speech-to-Text Setup
# --------------------------
# Credentials are automatically picked from GOOGLE_APPLICATION_CREDENTIALS environment variable
stt_client = speech.SpeechClient()
print("✅ Google Speech-to-Text client initialized")

# --------------------------
# Flask app setup
# --------------------------
app = Flask(__name__)
CORS(app)

# --------------------------
# Load ML model & vectorizer
# --------------------------
with open("Generated_pkl.pkl", "rb") as f:
    model = load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = load(f)

# --------------------------
# Text preprocessing
# --------------------------
sb = SnowballStemmer("english")
sw = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in punctuation]
    tokens = [t for t in tokens if t not in sw]
    tokens = [sb.stem(t) for t in tokens]
    return " ".join(tokens)

# --------------------------
# Routes
# --------------------------
@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "stt": "Google Speech-to-Text",
        "message": "Use /predict_text for text or /analyze_audio for audio transcription"
    })

@app.route("/predict_text", methods=["POST"])
def predict_text():
    """Original text-based prediction endpoint"""
    try:
        data = request.json
        text = data.get("text", "")

        if text.strip() == "":
            return jsonify({"prediction": ""})

        cleaned = clean_text(text)
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]

        return jsonify({"prediction": str(prediction)})

    except Exception as e:
        return jsonify({"error": str(e), "prediction": ""})

@app.route("/analyze_audio", methods=["POST"])
def analyze_audio():
    """Audio transcription using Google Speech-to-Text + Fraud Detection"""
    try:
        # Debug logging
        print(f"Request headers: {dict(request.headers)}")
        print(f"Files in request: {list(request.files.keys()) if request.files else 'None'}")
        print(f"Raw data length: {len(request.data) if request.data else 0}")
        
        # Get audio file
        audio_bytes = None
        
        if 'audio' in request.files:
            audio_file = request.files['audio']
            audio_bytes = audio_file.read()
            print(f"✅ Received audio file: {len(audio_bytes)} bytes, filename: {audio_file.filename}")
        elif request.data:
            audio_bytes = request.data
            print(f"✅ Received raw audio data: {len(audio_bytes)} bytes")
        else:
            return jsonify({"error": "No audio file provided"}), 400
        
        # Validate audio size
        if not audio_bytes or len(audio_bytes) < 100:
            return jsonify({"error": f"Audio file too small: {len(audio_bytes)} bytes"}), 400
        
        # Configure Google Speech-to-Text
        audio = speech.RecognitionAudio(content=audio_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-IN",
            enable_automatic_punctuation=True,
        )
        
        # Transcribe
        print("🔄 Sending to Google Speech-to-Text...")
        response = stt_client.recognize(config=config, audio=audio)
        
        if not response.results:
            print("⚠️ No speech detected in audio")
            return jsonify({"error": "No speech detected", "transcript": ""}), 400
        
        text = response.results[0].alternatives[0].transcript
        print(f"🎤 Google STT: {text}")
        
        if not text.strip():
            return jsonify({"prediction": "normal", "transcript": ""})
        
        # Run fraud detection
        cleaned = clean_text(text)
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]
        
        # Convert to readable format
        is_fraud = prediction in ["1", "fraud", "scam"]
        result = "fraud" if is_fraud else "normal"
        
        print(f"🔍 Prediction: {result} (raw: {prediction})")
        
        return jsonify({
            "prediction": result,
            "transcript": text,
            "raw_prediction": str(prediction)
        })
        
    except Exception as e:
        print(f"❌ Error in analyze_audio: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return "OK"

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", debug=True, port=port)