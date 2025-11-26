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

# ----- Build sentences + index mapping -----
# sentences: list of all English triggers (main + variants)
# sentence_owner: for each sentence, which phrase index it belongs to
sentences = []
sentence_owner = []

for i, p in enumerate(phrases):
    # main canonical English sentence
    sentences.append(p["en"])
    sentence_owner.append(i)

    # optional variants
    for v in p.get("en_variants", []):
        sentences.append(v)
        sentence_owner.append(i)

# Precompute embeddings for all triggers
embeddings = model.encode(sentences, convert_to_tensor=True)

def query(q: str):
    """
    Given an English query, find the closest sentence embedding,
    then map that back to the corresponding phrase entry.
    """
    q_emb = model.encode(q, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, embeddings)[0]
    best_flat_idx = scores.argmax().item()

    # Map from flattened sentence index -> phrase index
    phrase_idx = sentence_owner[best_flat_idx]
    return phrases[phrase_idx], float(scores[best_flat_idx])

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

            # Build a clean result object for the template,
            result = {
                "en": match.get("en"),
                "thai": match.get("thai"),
                "zh": match.get("zh"),
                "zh_trad": match.get("zh_trad"),
                "pinyin": match.get("pinyin"),
                "zh_thai": match.get("zh_thai"),
            }

            score = round(float(s), 4)  # nicer display

    return render_template("index.html", result=result, score=score, user_input=user_input)

if __name__ == "__main__":
    # debug=True is fine on your local dev box
    app.run(host="127.0.0.1", port=5000, debug=True)
