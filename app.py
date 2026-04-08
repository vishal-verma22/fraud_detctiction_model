from flask import Flask, request, jsonify
from flask_cors import CORS
from pickle import load
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
import os
import tempfile
import assemblyai as aai

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
# AssemblyAI Setup
# --------------------------
ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
if ASSEMBLYAI_API_KEY:
    aai.api_key = ASSEMBLYAI_API_KEY  # ✅ FIXED: changed from aai.settings.api_key
    print("✅ AssemblyAI API key loaded")
else:
    print("⚠️ WARNING: ASSEMBLYAI_API_KEY not set! Audio transcription will fail.")

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
    api_status = "✅ Configured" if ASSEMBLYAI_API_KEY else "❌ Missing API Key"
    return jsonify({
        "status": "running",
        "assemblyai": api_status,
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
    """New endpoint for audio transcription + fraud detection"""
    # Check if API key is set
    if not ASSEMBLYAI_API_KEY:
        return jsonify({"error": "ASSEMBLYAI_API_KEY not configured on server"}), 500

    # Get audio file
    if 'audio' not in request.files and not request.data:
        return jsonify({"error": "No audio file provided"}), 400

    if 'audio' in request.files:
        audio_file = request.files['audio']
        audio_bytes = audio_file.read()
    else:
        audio_bytes = request.data

    # Save temporarily
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # Transcribe with AssemblyAI
        transcriber = aai.Transcriber()
        config = aai.TranscriptionConfig(
            language_code="en",
            speech_model=aai.SpeechModel.best,
        )
        transcript = transcriber.transcribe(tmp_path, config)

        if transcript.status == aai.TranscriptStatus.error:
            return jsonify({"error": transcript.error}), 500

        text = transcript.text
        if not text.strip():
            return jsonify({"prediction": "normal", "transcript": ""})

        # Run fraud detection
        cleaned = clean_text(text)
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]

        # Convert to readable format
        is_fraud = prediction in ["1", "fraud", "scam"]
        result = "fraud" if is_fraud else "normal"

        return jsonify({
            "prediction": result,
            "transcript": text,
            "raw_prediction": str(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.route("/health", methods=["GET"])
def health():
    return "OK"

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", debug=False, port=port)