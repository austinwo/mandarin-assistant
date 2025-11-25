import json
from pathlib import Path

from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util

# ----- Load data -----
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "phrases.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    phrases = json.load(f)

if not phrases:
    raise ValueError("phrases.json is empty – add some entries first.")

# ----- Load model & embeddings -----
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

eng_sentences = [p["en"] for p in phrases]
embeddings = model.encode(eng_sentences, convert_to_tensor=True)

def query(q: str):
    q_emb = model.encode(q, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, embeddings)[0]
    best_idx = scores.argmax().item()
    return phrases[best_idx], float(scores[best_idx])

# ----- Flask app -----
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    score = None
    user_input = ""

    if request.method == "POST":
        user_input = request.form.get("query", "").strip()
        if user_input:
            match, s = query(user_input)
            result = match
            score = round(float(s), 4)  # nicer display

    return render_template("index.html", result=result, score=score, user_input=user_input)

if __name__ == "__main__":
    # debug=True is fine on your local dev box
    app.run(host="127.0.0.1", port=5000, debug=True)
