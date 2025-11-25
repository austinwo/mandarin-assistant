import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

DATA_PATH = Path(__file__).parent / "phrases.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    phrases = json.load(f)

if not phrases:
    raise ValueError("phrases.json is empty - add some entries first.")

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

eng_sentences = [p["en"] for p in phrases]
embeddings = model.encode(eng_sentences, convert_to_tensor=True)

def query(q):
    q_emb = model.encode(q, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, embeddings)[0]
    best_idx = scores.argmax().item()
    return phrases[best_idx], float(scores[best_idx])

if __name__ == "__main__":
    while True:
        q = input("Type what you want to say in English: ")
        match, score = query(q)
        print("\n=== RESULT ===")
        print("EN      :", match["en"])
        print("TH      :", match["thai"])
        print("ZH      :", match["zh"])
        print("PINYIN  :", match["pinyin"])
        print("ZH-TH   :", match["zh_thai"])
        print("SCORE   :", score)
        print()
