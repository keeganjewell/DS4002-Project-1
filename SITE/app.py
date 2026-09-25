"""
Flask website for the lyrics -> genre classifier.

Loads the TF-IDF vectorizer + multinomial logistic regression model saved
by SCRIPTS/06_train_model.py (trained on all 400 labeled songs) and serves
a single page: paste lyrics, get a predicted genre with per-genre
confidence scores.

Run:
    pip install flask   (already done if you're seeing this after setup)
    python SITE/app.py
Then open http://localhost:5000
"""

import os
import re
import string
import sys

import joblib
from flask import Flask, render_template, request

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(BASE_DIR, "MODEL")

VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier.joblib")

MIN_WORDS = 15  # below this, TF-IDF features are too sparse to trust

if not (os.path.exists(VECTORIZER_PATH) and os.path.exists(CLASSIFIER_PATH)):
    sys.exit(
        "Model files not found in MODEL/. Run `python SCRIPTS/06_train_model.py` "
        "first to train and save the model."
    )

vectorizer = joblib.load(VECTORIZER_PATH)
classifier = joblib.load(CLASSIFIER_PATH)

app = Flask(__name__)

# --------------------------------------------------------------------------
# Preprocessing -- must exactly match SCRIPTS/06_train_model.py's clean_lyrics
# --------------------------------------------------------------------------

PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
WHITESPACE_RE = re.compile(r"\s+")


def clean_lyrics(text):
    text = str(text).lower()
    text = PUNCT_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def predict_genre(lyrics_text):
    cleaned = clean_lyrics(lyrics_text)
    word_count = len(cleaned.split())

    X = vectorizer.transform([cleaned])
    probs = classifier.predict_proba(X)[0]
    classes = classifier.classes_

    ranked = sorted(zip(classes, probs), key=lambda p: p[1], reverse=True)
    predicted_genre, top_prob = ranked[0]

    return {
        "predicted_genre": predicted_genre,
        "confidence": round(top_prob * 100, 1),
        "ranked": [(genre, round(p * 100, 1)) for genre, p in ranked],
        "word_count": word_count,
        "low_confidence_warning": word_count < MIN_WORDS,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    lyrics_input = ""

    if request.method == "POST":
        lyrics_input = request.form.get("lyrics", "")
        if lyrics_input.strip():
            result = predict_genre(lyrics_input)

    return render_template("index.html", result=result, lyrics_input=lyrics_input)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
