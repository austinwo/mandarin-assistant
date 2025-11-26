import json
from pathlib import Path

from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from random import choice


# ----- Config -----
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "phrases.json"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "phrases"

# minimum acceptable similarity (0–1)
MIN_SIMILARITY = 0.60
TOP_K = 5  # how many nearest neighbors to inspect / suggest


# ----- Load phrases -----
with open(DATA_PATH, "r", encoding="utf-8") as f:
    phrases = json.load(f)

if not phrases:
    raise ValueError("phrases.json is empty – add some entries first.")


# ----- Embedding model -----
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)


# ----- Chroma client -----
client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False),
)


def rebuild_index():
    """
    Drop and rebuild the Chroma collection from phrases.json.
    This guarantees the DB always matches your dataset.
    """
    # delete old collection if exists
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    ids, docs, metadatas = [], [], []

    for i, p in enumerate(phrases):
        # main English
        ids.append(f"{i}-0")
        docs.append(p["en"])
        metadatas.append({"phrase_index": i, "source": "en"})

        # variants
        for j, v in enumerate(p.get("en_variants", []), start=1):
            ids.append(f"{i}-{j}")
            docs.append(v)
            metadatas.append({"phrase_index": i, "source": "en_variant"})

    embeddings = model.encode(docs, convert_to_numpy=True).tolist()

    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return collection


collection = rebuild_index()


def query(q: str, top_k: int = TOP_K):
    """
    Return:
      - best matching phrase dict (or None)
      - best similarity (float or None)
      - list of suggestion strings (English canonical phrases)
    """
    q = q.strip()
    if not q:
        return None, None, []

    q_emb = model.encode([q], convert_to_numpy=True).tolist()

    res = collection.query(
        query_embeddings=q_emb,
        n_results=top_k,
    )

    # safety checks
    if not res.get("metadatas") or not res["metadatas"][0]:
        return None, None, []

    metadatas = res["metadatas"][0]
    distances = res["distances"][0]

    # convert distances -> similarities
    sims = [1.0 - float(d) for d in distances]
    phrase_indices = [m["phrase_index"] for m in metadatas]

    # best match
    best_phrase_idx = phrase_indices[0]
    best_sim = sims[0]
    best_phrase = phrases[best_phrase_idx]

    # build suggestion list of unique canonical EN phrases
    seen = set()
    suggestions = []
    for idx in phrase_indices:
        en = phrases[idx]["en"]
        if en not in seen:
            seen.add(en)
            suggestions.append(en)

    return best_phrase, best_sim, suggestions


# ----- Flask -----
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    user_input = ""
    result = None
    score = None
    no_good_match = False
    suggestions = []

    if request.method == "POST":
        user_input = request.form.get("query", "").strip()
        if user_input:
            match, sim, suggestions = query(user_input)

            if match is not None and sim is not None:
                if sim < MIN_SIMILARITY:
                    # We *did* find something, but confidence is low:
                    # tell the user and show suggestions instead of pretending nothing happened.
                    no_good_match = True
                    score = round(sim, 3)
                    result = None
                else:
                    result = match
                    score = round(sim, 3)
                    # suggestions still passed so you can show "other options" if you want

    return render_template(
        "index.html",
        user_input=user_input,
        result=result,
        score=score,
        no_good_match=no_good_match,
        suggestions=suggestions,
    )


@app.route("/random", methods=["POST"])
def random_phrase():
    # Pick a random canonical phrase
    p = choice(phrases)

    return render_template(
        "index.html",
        user_input=p["en"],   # pre-fill the input with the English phrase
        result=p,             # directly show the translations
        score=None,           # no similarity score for random
        no_good_match=False,
        suggestions=[],       # optional, used by the template
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
