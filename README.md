<img width="896" height="679" alt="image" src="https://github.com/user-attachments/assets/87185d27-43d3-41cb-9d8f-ed7a13ed6209" />


# English → Mandarin Assistant

A minimal web app that translates everyday English phrases into:
- Thai
- Mandarin Chinese (Simplified)
- Pinyin
- Mandarin pronunciation written using Thai phonetics (ZH-TH)

This is built for real-world communication, not textbook Mandarin.  
The purpose is to learn how people actually speak in normal conversations with friends, locals, and native speakers.

---

## 🚀 Features

- Type an English sentence and get:
  - **EN** — original
  - **TH** — Thai
  - **ZH** — Mandarin (Simplified)
  - **Pinyin** — pronunciation
  - **ZH-TH** — Mandarin pronunciation expressed using Thai letters

This reinforces pronunciation, meaning, and context. It is optimized for casual social use and memory building, not academic instruction.

---

## 🧠 Approach

The app uses:
- Sentence embeddings via SentenceTransformer
- A curated JSON phrase bank
- Cosine similarity scoring
- A simple Flask backend and browser UI

Every phrase is intentional and context-aware. No hallucinated translations.

Goals moving forward:
- Expand the phrase library
- Improve ZH-TH phonetic accuracy
- Add slang, nightlife, and culture-specific expressions
- Support conversational memory
- Allow voice input and text-to-speech output

This project is iterative by design: ship early, improve constantly.

---

## 🛠️ Setup (Poetry)

This project uses Poetry. Do not install pip packages globally.

1. Install dependencies:
   poetry install

2. Activate environment:
   poetry shell

3. Run the backend:
   poetry run python app.py

4. Open the UI:
   http://localhost:5000

---

## 🌱 Roadmap

- Add Traditional Chinese
- Expand phrase bank (nightlife, gym, dating, logistics)
- Improve ZH-TH phonetic mapping accuracy
- Add context memory (vector DB: Chroma, Qdrant, LanceDB, etc.)
- Microphone input and text-to-speech output
- Deployment to hosting providers (Vercel, Flask backend, etc.)
