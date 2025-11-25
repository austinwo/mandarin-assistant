# English -> Mandarin Assistant

A minimal web app that translates everyday English phrases into:
- Thai
- Mandarin Chinese (Simplified)
- Pinyin
- Mandarin pronounced using Thai phonetics (ZH-TH)

This is built for real-world communication, not textbook Mandarin.  
The purpose is to learn how people actually speak in normal conversations with friends, locals, and native speakers.

---

## 🚀 Features

- Type an English sentence and get:
  - **EN** — original
  - **TH** — Thai
  - **ZH** — Mandarin (Simplified)
  - **Pinyin** — pronunciation
  - **ZH-TH** — Mandarin pronunciation written using Thai letters

This tool helps reinforce pronunciation, context, and memory.
It is optimized for daily use and casual language, not formal academic material.

---

## 🧠 Approach

The current version uses:
- Sentence embeddings (e.g., MPNet or similar)
- A phrase dictionary (JSON)
- Similarity scoring
- Template-based translation output

Goals moving forward:
- Expand the phrase dataset
- Improve pronunciation mapping
- Introduce conversational memory
- Support slang, casual speech, and nightlife vocabulary
- Allow voice input and audio output

This project is intentionally iterative.  
The goal is to ship early, then improve repeatedly.

---

## 🛠️ Setup

### Create a virtual environment
```bash
python -m venv venv
```

### Activate the environment

macOS / Linux:
```bash
source venv/bin/activate
```

Windows:
```bash
venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the backend
```bash
python assistant.py
```

### Open the UI
Open `index.html` in your browser.

---

## 🌱 Roadmap
- Add Traditional Chinese support
- Expand phrase bank for slang and nightlife language
- Improve ZH-TH phonetic mapping accuracy
- Add context memory (vector DB)
- Add microphone input and text-to-speech output
- Deploy to Vercel or small Flask backend
