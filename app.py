"""
Flask app for an English → Mandarin/Thai assistant with:

- Semantic search over a curated phrase corpus (Chroma + SentenceTransformers).
- RAG-style explanations for known phrases.
- Direct OpenAI-backed translation + examples as a fallback.

Designed as a small but realistic AI product surface:
- Structured retrieval layer
- Model-backed generation
- Clear separation of concerns
"""

from collections import defaultdict
import time
import json
import logging
import os
from pathlib import Path
from random import choice
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from openai import OpenAI

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR: Path = Path(__file__).parent
DATA_PATH: Path = BASE_DIR / "phrases.json"
CHROMA_DIR: Path = BASE_DIR / "chroma_db"
COLLECTION_NAME: str = "phrases"

# Retrieval behavior
MIN_SIMILARITY: float = float(os.getenv("MIN_SIMILARITY", "0.60")) # minimum acceptable similarity (0–1)
TOP_K: int = int(os.getenv("TOP_K", "5")) # how many nearest neighbors to inspect / suggest


# Model config (can be overridden via environment)
OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Load phrases
# ---------------------------------------------------------------------------

with open(DATA_PATH, "r", encoding="utf-8") as f:
    phrases: List[Dict[str, Any]] = json.load(f)

if not phrases:
    raise ValueError("phrases.json is empty – add some entries first.")

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# ---------------------------------------------------------------------------
# Chroma client & index
# ---------------------------------------------------------------------------

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False),
)


def rebuild_index() -> chromadb.api.models.Collection.Collection:
    """
    Rebuild the Chroma collection from phrases.json.

    Each phrase contributes:
    - One canonical English sentence.
    - Zero or more English variants (if present in "en_variants").
    """
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted existing Chroma collection '%s'.", COLLECTION_NAME)
    except Exception:
        # Collection may not exist on first run.
        logger.info("No existing Chroma collection '%s' to delete.", COLLECTION_NAME)

    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    ids: List[str] = []
    docs: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for i, phrase in enumerate(phrases):
        # main English
        ids.append(f"{i}-0")
        docs.append(phrase["en"])
        metadatas.append({"phrase_index": i, "source": "en"})

        # optional variants
        for j, variant in enumerate(phrase.get("en_variants", []), start=1):
            ids.append(f"{i}-{j}")
            docs.append(variant)
            metadatas.append({"phrase_index": i, "source": "en_variant"})

    embeddings = model.encode(docs, convert_to_numpy=True).tolist()

    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    logger.info(
        "Rebuilt Chroma collection '%s' with %d base phrases and %d total entries.",
        COLLECTION_NAME,
        len(phrases),
        len(docs),
    )

    return collection


def should_rebuild_index() -> bool:
    """Check if index needs rebuilding."""
    if not CHROMA_DIR.exists():
        return True

    # Check if phrases.json is newer than chroma_db
    if DATA_PATH.stat().st_mtime > CHROMA_DIR.stat().st_mtime:
        return True

    return False

if should_rebuild_index():
    collection = rebuild_index()
    logger.info("Index rebuilt.")
else:
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    logger.info("Using existing Chroma collection '%s'.", COLLECTION_NAME)

# ---------------------------------------------------------------------------
# OpenAI client (for RAG + fallback generation)
# ---------------------------------------------------------------------------

def validate_environment():
    """Validate required environment variables."""
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable is required")

validate_environment()

oa_client = OpenAI()  # uses OPENAI_API_KEY from env


def generate_rag_answer(user_input: str, phrase: Dict[str, Any]) -> Optional[str]:
    """
    Explain / give usage based on a retrieved phrase (RAG style).

    Returns a concise English explanation + exactly two example sentences
    with Mandarin, pinyin, and English meanings.
    """
    system_msg = (
        "You are a friendly Mandarin and Thai language tutor. "
        "Given the user's intent and the retrieved phrase with translations, "
        "explain how to say what they want in Mandarin, and then give exactly "
        "two example usages. For each example, you MUST include:\n"
        "- Mandarin sentence\n"
        "- Pinyin\n"
        "- English meaning\n\n"
        "Format them clearly, for example:\n"
        "Example 1: 我很興奮學習普通話。 (Wǒ hěn xīngfèn xuéxí pǔtōnghuà.) – I'm excited to learn Mandarin.\n"
        "Example 2: ...\n\n"
        "Keep the explanation concise (under ~6 sentences total)."
    )

    zh_trad = phrase.get("zh_trad", "")

    user_msg = f"""
User input: "{user_input}"

Retrieved phrase and translations:

- English: {phrase.get("en", "")}
- Thai: {phrase.get("thai", "")}
- Chinese (Simplified): {phrase.get("zh", "")}
- Chinese (Traditional): {zh_trad}
- Pinyin: {phrase.get("pinyin", "")}
- Chinese written with Thai letters (ZH-TH): {phrase.get("zh_thai", "")}

First, briefly explain in English how the user would naturally say what they want in Mandarin.
Then give EXACTLY two example sentences in this format:

Example 1: <Mandarin> (<Pinyin>) – <English meaning>
Example 2: <Mandarin> (<Pinyin>) – <English meaning>
"""

    try:
        completion = oa_client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )
        content = completion.choices[0].message.content
        return content.strip() if content else None
    except Exception as e:
        logger.exception("[generate_rag_answer] Error calling OpenAI: %s", e)
        return None


def generate_direct_phrase(user_input: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Fallback: let OpenAI translate any English input directly.

    Returns:
        phrase_dict: same shape as entries in phrases.json (or minimal fallback).
        explanation_text: explanation + examples, or None on failure.
    """

    system_msg = (
        "You are a bilingual English–Mandarin–Thai assistant. "
        "Given an English sentence, produce natural conversational Mandarin plus Thai "
        "and pronunciation info. Return ONLY a compact JSON object, no extra text."
    )

    user_msg = f"""
Translate this English sentence into Mandarin and Thai, and provide pronunciation info.

English input: "{user_input}"

Return JSON with exactly these keys:
- "en": original English
- "thai": Thai translation
- "zh": Mandarin Chinese (Simplified)
- "zh_trad": Mandarin Chinese (Traditional)
- "pinyin": Mandarin pinyin with tone marks
- "zh_thai": Mandarin pronunciation written using Thai letters
- "explanation": a short English explanation PLUS EXACTLY two example usages.

For the "explanation" field:

1. Start with 1–2 short English sentences explaining the meaning/nuance.
2. Then include EXACTLY two usage examples, each on its own line, and each MUST have:
   - Mandarin sentence
   - Pinyin
   - English meaning

Use this format inside the explanation string:

Example 1: <Mandarin> (<Pinyin>) – <English meaning>
Example 2: <Mandarin> (<Pinyin>) – <English meaning>

You MUST return only valid JSON. No markdown, no comments, no extra text.
"""

    raw: Optional[str] = None

    try:
        completion = oa_client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
        )

        raw = (completion.choices[0].message.content or "").strip()

        logger.debug("OPENAI RAW RESPONSE (direct phrase): %s", raw)

        text = raw

        # Strip code fences defensively if the model ignores strict JSON instructions.
        if text.startswith("```"):
            text = text.strip("`")
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start : end + 1]

        data = json.loads(text)

        phrase: Dict[str, Any] = {
            "en": data.get("en", user_input),
            "thai": data.get("thai", ""),
            "zh": data.get("zh", ""),
            "zh_trad": data.get("zh_trad", data.get("zh", "")),
            "pinyin": data.get("pinyin", ""),
            "zh_thai": data.get("zh_thai", ""),
        }

        explanation: str = data.get("explanation", "")

        logger.debug(
            "PARSED PHRASE JSON: %s\nExplanation: %s",
            json.dumps(phrase, indent=2, ensure_ascii=False),
            explanation,
        )

        return phrase, explanation or None

    except Exception as e:
        logger.exception("Error in generate_direct_phrase: %s", e)
        if raw is not None:
            logger.error("Raw OpenAI response: %s", raw)

        # Minimal fallback phrase so the UI can still render something.
        fallback_phrase: Dict[str, Any] = {
            "en": user_input,
            "thai": "",
            "zh": "",
            "zh_trad": "",
            "pinyin": "",
            "zh_thai": "",
        }
        return fallback_phrase, None


def query_corpus(q: str, top_k: int = TOP_K) -> Tuple[Optional[Dict[str, Any]], Optional[float], List[str]]:
    """
    Query the Chroma collection with the given English string.

    Returns:
        best_phrase: the best matching phrase dict (or None)
        best_similarity: cosine similarity (0–1) or None
        suggestions: unique canonical English phrases from the top-k hits
    """
    q = q.strip()
    if not q:
        return None, None, []

    q_emb = model.encode([q], convert_to_numpy=True).tolist()

    res = collection.query(
        query_embeddings=q_emb,
        n_results=top_k,
    )

    if not res.get("metadatas") or not res["metadatas"][0]:
        return None, None, []

    metadatas = res["metadatas"][0]
    distances = res["distances"][0]

    # distance = 1 - cosine similarity
    sims = [1.0 - float(d) for d in distances]
    phrase_indices = [m["phrase_index"] for m in metadatas]

    best_phrase_idx = phrase_indices[0]
    best_sim = sims[0]
    best_phrase = phrases[best_phrase_idx]

    # Build suggestion list from top-k canonical English phrases (deduplicated).
    seen: set[str] = set()
    suggestions: List[str] = []
    for idx in phrase_indices:
        en = phrases[idx]["en"]
        if en not in seen:
            seen.add(en)
            suggestions.append(en)

    return best_phrase, best_sim, suggestions


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""
    
    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for given client."""
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        self.requests[client_id] = [
            t for t in self.requests[client_id] if t > window_start
        ]
        
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        self.requests[client_id].append(now)
        return True


rate_limiter = RateLimiter(max_requests=30, window_seconds=60)

def get_client_id():
    """Get client identifier for rate limiting."""
    return request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")


def rate_limit(f):
    """Decorator to apply rate limiting to routes."""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_id = get_client_id()
        if not rate_limiter.is_allowed(client_id):
            return render_template(
                "index.html",
                error="Too many requests. Please wait a moment and try again."
            ), 429
        return f(*args, **kwargs)
    return decorated_function


@app.route("/", methods=["GET", "POST"])
@rate_limit
def index():
    user_input: str = ""
    result: Optional[Dict[str, Any]] = None
    score: Optional[float] = None
    suggestions: List[str] = []
    rag_answer: Optional[str] = None
    no_good_match: bool = False  # for template logic if you want to surface it

    if request.method == "POST":
        action = request.form.get("action", "translate")

        if action == "random":
            # Ignore user_input, just pick a random phrase from corpus.
            phrase = choice(phrases)
            user_input = phrase["en"]
            result = phrase
            score = None
            suggestions = []
            no_good_match = False
            rag_answer = generate_rag_answer(user_input, phrase)
            logger.info("Random phrase selected: %s", user_input)

        else:
            # Normal translate flow.
            MAX_INPUT_LENGTH = 500

            user_input = request.form.get("query", "").strip()

            # Input validation
            if not user_input:
                return render_template(
                    "index.html",
                    error="Please enter something to translate."
                )

            if len(user_input) > MAX_INPUT_LENGTH:
                return render_template(
                    "index.html",
                    error=f"Input too long. Maximum {MAX_INPUT_LENGTH} characters."
                )

            if user_input:
                match, sim, suggestions = query_corpus(user_input)

                if match is None or sim is None:
                    # Nothing useful in DB → full OpenAI translation.
                    logger.info(
                        "[index] No semantic match for '%s', using direct OpenAI translation.",
                        user_input,
                    )
                    phrase, explanation = generate_direct_phrase(user_input)
                    result = phrase
                    score = None
                    rag_answer = explanation
                    suggestions = []
                    no_good_match = True

                elif sim < MIN_SIMILARITY:
                    # Low similarity → treat as "new phrase", let OpenAI handle it.
                    logger.info(
                        "[index] Low similarity (%.3f) for '%s', using direct OpenAI translation.",
                        sim,
                        user_input,
                    )
                    phrase, explanation = generate_direct_phrase(user_input)
                    result = phrase
                    score = None          # don't show similarity pill
                    suggestions = []      # hide suggestions in this path
                    rag_answer = explanation
                    no_good_match = True

                else:
                    # Good semantic match → classic RAG over phrases.json.
                    logger.info(
                        "[index] Good match (sim=%.3f) for '%s', using corpus + RAG.",
                        sim,
                        user_input,
                    )
                    result = match
                    score = round(sim, 3)
                    rag_answer = generate_rag_answer(user_input, match)
                    no_good_match = False

    return render_template(
        "index.html",
        user_input=user_input,
        result=result,
        score=score,
        no_good_match=no_good_match,
        suggestions=suggestions,
        rag_answer=rag_answer,
    )

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return jsonify({
        "status": "healthy",
        "service": "mandarin-assistant",
        "version": "0.1.0"
    })

if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(host="127.0.0.1", port=5000, debug=debug)

