import json
from pathlib import Path

from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ----- Paths & data -----
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "phrases.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    phrases = json.load(f)

if not phrases:
    raise ValueError("phrases.json is empty – add some entries first.")

# ----- Embedding model -----
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# ----- Chroma client -----
client = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db"),
    settings=Settings(anonymized_telemetry=False),
)

def rebuild_index():
    """Drop and rebuild the Chroma collection from phrases.json."""
    # delete old collection if it exists
    try:
        client.delete_collection("phrases")
    except Exception:
        pass  # first run or already gone, ignore

    collection = client.create_collection(name="phrases")

    ids = []
    docs = []
    metadatas = []

    for i, p in enumerate(phrases):
        # main English sentence
        ids.append(f"{i}-0")
        docs.append(p["en"])
        metadatas.append({"phrase_index": i, "source": "en"})

        # optional English variants
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

    client.persist()
    return collection

# build fresh collection on every startup
collection = rebuild_index()

def query(q: str):
    """Query Chroma using an embedding from the same SentenceTransformer."""
    if not q.strip():
        return None, None

    q_emb = model.encode([q], convert_to_numpy=True).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=1)

    if not res["metadatas"] or not res["metadatas"][0]:
        return None, None

    meta = res["metadatas"][0][0]
    distance = res["distances"][0][0]
    phrase_idx = meta["phrase_index"]

    return phrases[phrase_idx], distance
