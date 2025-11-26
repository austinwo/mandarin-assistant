import json
from pathlib import Path

from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ----- Config -----
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "phrases.json"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "phrases"

# minimum acceptable similarity (0–1)
MIN_SIMILARITY = 0.60

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
    except:
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

    print(f"ids {ids}")

    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return collection


collection = rebuild_index()


def query(q: str):
    q = q.strip()
    if not q:
        return None, None

    q_emb = model.encode([q], convert_to_numpy=True).tolist()

    res = collection.query(
        query_embeddings=q_emb,
        n_results=1,
    )

    if not res["metadatas"] or not res["metadatas"][0]:
        return None, None

    meta = res["metadatas"][0][0]
    distance = float(res["distances"][0][0])

    # distance = 1 - cosine similarity
    similarity = 1.0 - distance
    phrase_idx = meta["phrase_index"]

    return phrases[phrase_idx], similarity


# ----- Flask -----
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    user_input = ""
    result = None
    score = None
    no_good_match = False

    if request.method == "POST":
        user_input = request.form.get("query", "").strip()
        if user_input:
            match, sim = query(user_input)
            if match:
                if sim < MIN_SIMILARITY:
                    no_good_match = True
                    score = round(sim, 3)
                else:
                    result = match
                    score = round(sim, 3)

    return render_template(
        "index.html",
        user_input=user_input,
        result=result,
        score=score,
        no_good_match=no_good_match,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
