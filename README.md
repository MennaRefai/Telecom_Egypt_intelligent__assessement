# Telecom_Egypt_intelligent__assessement
RAG-powered assistant for Telecom Egypt with voice, text, and document input. Whisper ASR + bge-m3 embeddings + FAISS + Qwen2.5. Handles Arabic, English, and Egyptian dialect with source-cited answers and voice playback.


# 📞 Telecom Egypt Intelligent Assistant

A production-ready AI assistant that answers customer questions using **Telecom Egypt (te.eg)** resources and **user-uploaded documents**. Supports **voice, text, and document** input — with **source citations** and **voice playback**.

Handles **Arabic, English, and Egyptian dialect**.

---

## 📖 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Setup](#setup)
6. [Running the App](#running-the-app)
7. [Building the Knowledge Base](#building-the-knowledge-base)
8. [How to Use](#how-to-use)
9. [Configuration](#configuration)
10. [On-Premises Deployment](#on-premises-deployment)
11. [Evaluation](#evaluation)
12. [Future Work](#future-work)
13. [Author](#author)

---

## Overview

**Telecom Egypt Intelligent Assistant** is an end-to-end Retrieval-Augmented Generation (RAG) system with an ASR front-end and a TTS back-end. It is designed to:

- Answer customer questions from the official **te.eg** knowledge base
- Handle queries in **Arabic, English, and Egyptian dialect**
- Accept **voice**, **text**, and **document** inputs
- Speak the answer back when input is voice-based, while always displaying text
- Ground all answers in source documents with **citations**
- Run **fully on-premises** — no cloud APIs in the core pipeline

---

## Features

| Feature | Description |
|---|---|
| 🎤 **Voice input** | Whisper (medium) ASR with domain-biased prompting |
| 💬 **Text chat** | Interactive chat with history |
| 📄 **Document upload** | PDF, DOCX, TXT, Images (OCR) |
| 🔍 **RAG pipeline** | FAISS vector search over te.eg + user docs |
| 📚 **Source citations** | Every answer grounded in retrieved chunks |
| 🔊 **Text-to-speech** | Voice responses for voice queries |
| 🌍 **Bilingual** | Automatic Arabic / English handling |
| 🧠 **Dialect handling** | Arabic normalization + Egyptian → MSA mapping |
| 💬 **Chat history** | Full conversation preserved per session |

---

## Architecture

```
                        ┌─────────────────────┐
                        │       User          │
                        └──────────┬──────────┘
                                   │
             ┌─────────────────────┼─────────────────────┐
             ▼                     ▼                     ▼
        ┌─────────┐           ┌─────────┐           ┌─────────┐
        │  Voice  │           │  Text   │           │  Docs   │
        └────┬────┘           └────┬────┘           └────┬────┘
             │                     │                     │
             ▼                     │                     ▼
     ┌───────────────┐             │           ┌──────────────────┐
     │ Whisper ASR   │             │           │ Parser / OCR     │
     └───────┬───────┘             │           │ (pdf, docx, txt, │
             │                     │           │  png, jpg)       │
             ▼                     │           └────────┬─────────┘
     ┌───────────────┐             │                    │
     │ Normalize +   │             │                    │
     │ Egy→MSA map   │             │                    │
     └───────┬───────┘             │                    │
             │                     │                    │
             └──────────┬──────────┴────────────────────┘
                        ▼
              ┌──────────────────────┐
              │  Embedding (bge-m3)  │
              │      1024-dim        │
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │   FAISS Vector Index │
              │     (top-k search)   │
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │  LLM (Qwen2.5-0.5B)  │
              │   Grounded Answer    │
              │   + Citations        │
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │  TTS (voice) + Text  │
              └──────────────────────┘
```

---

## Project Structure

```
telecom-egypt-assistant/
├── app.py                     # Streamlit UI
├── backend.py                 # All pipeline functions
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── baseline_index.faiss       # FAISS vector index (generated)
├── baseline_chunks.pkl        # Chunk metadata store (generated)
└── data/
    └── te_eg_data.json        # Scraped te.eg content (optional)
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/telecom-egypt-assistant.git
cd telecom-egypt-assistant
```

### 2. Install system dependencies

**Ubuntu / Debian:**
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg tesseract-ocr tesseract-ocr-ara poppler-utils
```

**macOS (Homebrew):**
```bash
brew install ffmpeg tesseract tesseract-lang poppler
```

**Windows:**
- **FFmpeg**: https://ffmpeg.org/download.html — add to PATH.
- **Tesseract**: https://github.com/UB-Mannheim/tesseract/wiki — install **Arabic** language pack.

### 3. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate           # Linux / macOS
venv\Scripts\activate              # Windows
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Add the FAISS index files

Place these two files in the project root:
- `baseline_index.faiss`
- `baseline_chunks.pkl`

*(See [Building the Knowledge Base](#building-the-knowledge-base) to generate them.)*

---

## Running the App

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

> **First launch** may take 30–60 seconds — models (Whisper, bge-m3, Qwen) load into memory.

---

## requirements.txt

```
streamlit
faiss-cpu
sentence-transformers
transformers==4.46.3
langchain-text-splitters
openai-whisper
pymupdf
python-docx
pytesseract
Pillow
gTTS
torch
numpy
```

---

## Building the Knowledge Base

The knowledge base is built once by scraping te.eg, chunking the content, embedding it, and saving the FAISS index.

```python
from backend import create_index, add_to_index, chunk_text, save_index
import json

with open('te_eg_data.json', 'r', encoding='utf-8') as f:
    scraped = json.load(f)

create_index()
for page in scraped:
    chunks = chunk_text(page['content'], metadata={
        'source': page['url'],
        'title':  page['title'],
        'type':   'website'
    })
    add_to_index(chunks)

save_index('baseline_index.faiss', 'baseline_chunks.pkl')
print("Knowledge base built.")
```

---

## How to Use

### 💬 Text Mode
Type a question in the chat box → receive a text answer with citations.

### 🎤 Voice Mode
1. Select **🎤 Voice** in the sidebar.
2. Either **upload an audio file** or **record from the mic**.
3. Whisper transcribes → RAG retrieves → answer + voice playback.

### 📄 Document Mode
1. Select **📄 Document** in the sidebar.
2. Upload a PDF, DOCX, TXT, or image.
3. Choose:
   - **Ingest only** — add to knowledge base.
   - **Ingest + Query** — add and ask a question.
   - **Query only** — ask without storing.

---

## Configuration

Key settings — all in `backend.py`:

| Setting | Value | Where |
|---|---|---|
| Embedding model | `BAAI/bge-m3` | `get_embed_model()` |
| Embedding dimension | 1024 | `create_index()` |
| Chunk size | 500 chars | `_text_splitter` |
| Chunk overlap | 50 chars | `_text_splitter` |
| Whisper model | `medium` | `get_whisper()` |
| LLM | `Qwen/Qwen2.5-0.5B-Instruct` | `llm = pipeline(...)` |
| Top-k retrieval | 3 | `generate_with_citations()` |

---

## On-Premises Deployment

The entire pipeline runs **locally** — no external APIs required:

| Component | Tool | On-Prem? |
|---|---|---|
| ASR | Whisper | ✅ |
| Embeddings | SentenceTransformers | ✅ |
| Vector search | FAISS | ✅ |
| LLM | Qwen2.5 (transformers) | ✅ |
| TTS | gTTS | ⚠️ uses Google TTS |

**To make TTS fully on-prem**, replace `gTTS` in `text_to_speech()` with:

```python
# Option 1: pyttsx3 (offline, fast)
import pyttsx3
engine = pyttsx3.init()
engine.save_to_file(text, output_path)
engine.runAndWait()

# Option 2: Coqui TTS (higher quality, multilingual)
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.tts_to_file(text=text, file_path=output_path, language="ar")
```

### Running on GPU

```python
llm = pipeline(..., device=0)   # 0 = first GPU
```

Install CUDA PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Docker (optional)

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg tesseract-ocr tesseract-ocr-ara poppler-utils
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t telecom-egypt-assistant .
docker run -p 8501:8501 telecom-egypt-assistant
```

---

## Evaluation

| Metric | Description |
|---|---|
| **ASR accuracy (WER)** | Whisper transcript vs. ground-truth |
| **Retrieval hit-rate** | Are relevant chunks in the top-k? |
| **Citation accuracy** | Do citations match retrieved chunks? |
| **Answer correctness** | Does the LLM answer match retrieved context? |
| **Latency** | End-to-end response time (voice + text) |

---

## Future Work

- **VAD** (Voice Activity Detection) — better handling of noisy recordings
- **Larger Whisper model** (`large-v3`) — improve Egyptian dialect accuracy
- **Cross-encoder reranker** — higher retrieval precision
- **Fully on-prem TTS** (Coqui / pyttsx3) — remove gTTS dependency
- **Expanded Egyptian dialect dictionary** — beyond the current map
- **Fine-tuned Arabic intent classifier** — structured query routing
- **Streaming LLM output** — reduce perceived latency

---

## Author

**Menna Refai** — Telecom Egypt AI Assessment

---

## License

Internal assessment project. Not for public distribution.
