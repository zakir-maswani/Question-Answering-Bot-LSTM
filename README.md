<div align="center">

# 🧠 Question-Answering-Bot-LSTM

### A Closed-Domain Question Answering System powered by PyTorch, FastAPI & Vanilla JS

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

[![JavaScript](https://img.shields.io/badge/JavaScript-Vanilla-F7DF1E?style=flat-square&logo=javascript&logoColor=black)](.)
[![HTML5](https://img.shields.io/badge/HTML5-Frontend-E34F26?style=flat-square&logo=html5&logoColor=white)](.)
[![CSS3](https://img.shields.io/badge/CSS3-Styling-1572B6?style=flat-square&logo=css3&logoColor=white)](.)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)](.)

Ask a question → get an instant answer, predicted by a hand-trained LSTM classifier.

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#️-tech-stack)
- [Project Structure](#-project-structure)
- [Model Architecture](#-model-architecture)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [Known Limitations](#️-known-limitations)
- [Roadmap](#️-roadmap)
- [License](#-license)

---

## 🔍 Overview

Most tutorials treat QA as text generation. This project takes a simpler, more practical route for **closed-domain bots** (FAQ bots, trivia bots, internal knowledge assistants): treat every unique answer as a **class**, and train the model to classify which answer a question maps to.

```
CSV dataset → tokenization → vocabulary → LSTM classifier → FastAPI → HTML/CSS/JS chat UI
```

When the model isn't confident, a fuzzy string-matching fallback (`difflib`) catches reworded questions instead of giving up.

---

## ✨ Features

<table>
<tr>
<td width="33%" valign="top">

### 🔤 Full ML Pipeline
Tokenization, vocabulary building, `Dataset`/`DataLoader`, and training loop — all documented in one notebook.

</td>
<td width="33%" valign="top">

### 🧬 LSTM Classifier
Sequence encoder with dropout regularization, trained to classify questions into known answers.

</td>
<td width="33%" valign="top">

### 🎯 Full-Data Training
Trains on the entire dataset for deployment, so every known question is correctly recognized.

</td>
</tr>
<tr>
<td width="33%" valign="top">

### 🔍 Fuzzy Fallback
`difflib`-based closest-match search catches reworded questions the model isn't confident about.

</td>
<td width="33%" valign="top">

### ⚡ FastAPI Backend
REST API with a `/predict` endpoint and auto-generated interactive docs at `/docs`.

</td>
<td width="33%" valign="top">

### 🎨 Lightweight Frontend
Zero-dependency HTML/CSS/JS chat UI — no frameworks, no build step.

</td>
</tr>
</table>

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model | PyTorch (`nn.Embedding`, `nn.LSTM`, `nn.Linear`) |
| Training | Jupyter Notebook, scikit-learn |
| Backend / API | FastAPI, Uvicorn, Pydantic |
| Frontend | HTML, CSS, vanilla JavaScript |
| Data | pandas, CSV |

---

## 📁 Project Structure

```
question-answering-bot-lstm/
├── notebook.ipynb          # Training pipeline: EDA → vocab → model → training → export
├── main.py                 # FastAPI app: loads model, serves /predict + frontend
├── requirements.txt
├── model/
│   ├── qa_model.pth          # Trained LSTM weights
│   ├── vocab.pkl              # Word-level vocabulary
│   ├── answer_vocab.pkl       # Answer-class vocabulary
│   └── qa_dataset.csv         # Used by the fuzzy fallback
└── static/
    ├── index.html
    ├── style.css
    └── script.js
```

---

## 🧩 Model Architecture

```
Question text
     │
     ▼
Tokenize (lowercase, strip punctuation, split)
     │
     ▼
Embedding (dim=64) → Dropout (0.3)
     │
     ▼
LSTM (hidden_dim=128)
     │
     ▼
Final hidden state → Dropout (0.3) → Linear
     │
     ▼
Softmax → predicted answer + confidence
```

If confidence is low, a `difflib`-based closest-question search over the dataset kicks in as a fallback.

---

## 🚀 Getting Started

```bash
git clone <repo-url>
cd question-answering-bot-lstm
pip install -r requirements.txt
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000** for the chat UI, or **http://127.0.0.1:8000/docs** for interactive API docs.

To retrain: open `notebook.ipynb` and run all cells — it regenerates everything in `model/`.

---

## 🔌 API Reference

**`POST /predict`**

```json
// Request
{ "question": "What is the capital of France?" }

// Response
{ "answer": "Paris", "confidence": 0.994 }
```

---

## ⚠️ Known Limitations

- Classification-based and **closed-domain** — it can only return answers seen during training, never generates new text.
- Most answers in the dataset are unique to one question, so the model largely **memorizes** rather than generalizes. Truly novel questions won't be answered correctly even with the fuzzy fallback.
- Confidence reflects certainty over the fixed answer set, not factual correctness.

---

## 🗺️ Roadmap

- [ ] Sequence-to-sequence decoder for free-form, multi-word generated answers
- [ ] Data augmentation (paraphrased questions per answer) for better generalization
- [ ] Semantic similarity fallback (sentence-transformers) instead of `difflib`
- [ ] Auth, rate limiting, logging for production
- [ ] Dockerize for one-command deployment

---

## 📄 License

Released under the [MIT License](LICENSE).

---

<div align="center">
Made with 🧠 + ☕
</div>
