# Mandarin AI — Write English Like Yourself, Get Mandarin Like a Local

Learn how Chinese speakers actually talk — not how textbooks translate.

A Retrieval-Augmented Generation (RAG) system that transforms everyday English into **grounded, native-style Mandarin phrasing** with contextual examples. Built to solve a real user problem: conversational Mandarin acquisition that mirrors how people speak in real life.

---

## ✨ What it gives you
- **Native-style Mandarin phrasing** (not literal translation)
- **Chinese output in both ZH Simplified + ZH Traditional**
- **Pinyin pronunciation**
- **Thai-based phonetic pronunciation (ZH-TH)**
- **Mandarin + English usage examples with relevant context**

This moves language learning from **vocabulary memorization** to **situational fluency**.

---

## 🚀 Why it's different
Most language tools fail at conversation because they sit at the extremes:

- **Pure translation → rigid, literal, unnatural**
- **Pure LLM → hallucinations, tone drift, no grounding**

Mandarin AI uses a **hybrid retrieval + generation approach**:

**Retrieve → Evaluate → Generate**
1. User query → embedded using a multilingual transformer
2. Vector search for the closest conversational phrase
3. If similarity ≥ threshold → return grounded examples
4. If similarity < threshold → GPT-4o-mini generates a natural phrasing

Generation becomes **a controlled fallback**, not a default.

The outcome is real-world Mandarin phrasing that matches how native speakers would actually speak in social settings.

---

# 🔥 Architecture Overview

### UI Layer (Flask + Bootstrap)
- English input
- “Translate”
- “🎲 Inspire me”
- Loading state
- Prior output cleared during inference

### Retrieval Layer (Chroma + MPNet)
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- Multilingual semantic embeddings
- k-nearest neighbor cosine similarity
- Operates without LLM cost for high-confidence matches

### LLM Layer (OpenAI)
- Model: **gpt-4o-mini**
- Structured JSON output:
  - Simplified
  - Traditional
  - Pinyin
  - Thai phonetics
  - Mandarin + English usage examples

### Orchestration
- Retrieval anchors the phrasing
- Generation only extends or interpolates
- The system avoids literal translation and uncontrolled creativity

---

# 🧠 Model Choice: Why `gpt-4o-mini`
- Strong latency performance for user-facing workflows
- Consistent JSON-style completions
- Excellent multilingual inference
- Cost-effective for iteration

### When to upgrade
| Scenario | Model |
|---|---|
| Conversational usage | **gpt-4o-mini** |
| Cultural nuance / tutoring | **gpt-4o** |
| Curriculum / reasoning tasks | **gpt-4.1** |

---

# 🛠️ Setup

Install dependencies:
```bash
poetry install
```

---

# 🔑 Configure OpenAI API Key

macOS / Linux / WSL:
```bash
export OPENAI_API_KEY="sk-xxxx"
```

Windows PowerShell:
```bash
setx OPENAI_API_KEY "sk-xxxx"
```

Restart your terminal — no `.env` required.

---

# ▶️ Run the app
```bash
poetry run python app.py
```

Visit:
```
http://127.0.0.1:5000
```

---

# 💡 Features

### 1. Conversational translation
```
Where are we going later?
```
Output includes:
- EN reference
- TH
- ZH Traditional
- ZH Simplified
- Pinyin
- ZH-TH (Thai phonetics)
- Usage examples in both Mandarin + English

### 2. 🎲 Inspiration mode
Shows real-life Mandarin examples when users don’t know what to ask.

### 3. UX design around LLMs
- Results hidden while generating
- Buttons disabled during inference
- Loading indicator
- Dark UI for long-session readability

User experience respects **latency and cognitive load**.

---

# ⚙️ Internal Logic (high-level)
```
User Input → Embedding (MPNet) → Vector Search (Chroma) → Similarity Threshold → 
Retrieval Output OR GPT-4o-mini JSON → Rendered UI
```

Retrieval constrains the model to **grounded examples**.  
Generation only activates when retrieval confidence is insufficient.

---

# 🛡️ Reliability & Guardrails
If LLM inference fails:
- Retrieval-only output is returned
- UI remains operational
- User still learns something

Graceful degradation > hard failures.

---

# 📈 Future Roadmap

**1. Personalized memory**
- Store user embeddings to build a language profile over time
- Adapt tone and phrasing to each learner’s style

**2. Speech mode**
- Whisper input → Mandarin TTS output
- Real conversation practice loops

**3. Anki export**
- Characters, Pinyin, ZH-TH, and usage examples
- Lower friction for spaced repetition

**4. Adaptive difficulty**
- Track familiarity and exposure
- Recommend new phrases intelligently
- Avoid random drilling

**5. Session-aware Mandarin**
- Multi-turn context retention
- Travel, social, and practical domains
- Persona-aware phrasing
