from flask import Flask, request, jsonify
from flask_cors import CORS
from pickle import load
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
import os
import re

# --------------------------
# Hindi Transliteration Library
# --------------------------
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

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
# Auto Hindi to English Transliteration (NO DICTIONARY)
# --------------------------
def contains_hindi(text):
    """Check if text contains Hindi/Devanagari characters"""
    return any('\u0900' <= char <= '\u097F' for char in text)

def hindi_to_english(text):
    """
    Automatically convert ANY Hindi script to English/Roman script
    Works for ALL Hindi words - no dictionary needed!
    """
    if not contains_hindi(text):
        return text
    
    try:
        # Convert Devanagari to ITRANS (Roman script)
        roman = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
        
        # Remove special characters, keep only letters and spaces
        roman = re.sub(r'[^a-zA-Z\s]', '', roman)
        
        # Convert to lowercase
        roman = roman.lower()
        
        return roman
        
    except Exception as e:
        print(f"Transliteration error: {e}")
        # Fallback: remove Hindi characters
        return re.sub(r'[^\x00-\x7F]+', ' ', text)

# --------------------------
# Routes
# --------------------------
@app.route("/")
def home():
    return "Server is running ✅ with AUTO Hindi transliteration (no dictionary)!"

@app.route("/predict_text", methods=["POST"])
def predict_text():
    try:
        data = request.json
        text = data.get("text", "")
        
        if text.strip() == "":
            return jsonify({"prediction": ""})
        
        print(f"📝 Original: {text}")
        
        # AUTO convert Hindi to English (NO DICTIONARY)
        english_text = hindi_to_english(text)
        print(f"🔄 Converted: {english_text}")
        
        # Clean and predict
        cleaned = clean_text(english_text)
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]
        
        print(f"🎯 Prediction: {prediction}")
        
        return jsonify({
            "prediction": str(prediction),
            "original": text,
            "converted": english_text
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e), "prediction": ""})

# --------------------------
# Test endpoint
# --------------------------
@app.route("/test", methods=["POST"])
def test():
    try:
        data = request.json
        text = data.get("text", "")
        result = hindi_to_english(text)
        return jsonify({
            "original": text,
            "converted": result,
            "contains_hindi": contains_hindi(text)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)