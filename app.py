from flask import Flask, request, jsonify
from flask_cors import CORS
from pickle import load
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk

# Download required NLTK data if not present
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
CORS(app)

# Load model and vectorizer
with open("Generated_pkl.pkl", "rb") as f:
    model = load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = load(f)

sb = SnowballStemmer("english")
sw = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in punctuation]
    tokens = [t for t in tokens if t not in sw]
    tokens = [sb.stem(t) for t in tokens]
    return " ".join(tokens)

@app.route("/")
def home():
    return "Server is running ✅"

@app.route("/predict_text", methods=["POST"])
def predict_text():
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
        # Return error as JSON instead of crashing
        return jsonify({"error": str(e), "prediction": ""})

if __name__ == "__main__":
    # 0.0.0.0 makes it accessible from other devices
    app.run(host="0.0.0.0", debug=True, port=5000)