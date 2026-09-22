import difflib
import pickle
import uvicorn
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
STATIC_DIR = BASE_DIR / "static"

# Same preprocessing used during training — must stay identical so that
# questions are tokenized/encoded the same way the model was trained on.
def tokenize(text: str):
    text = text.lower()
    text = text.replace("?", "")
    text = text.replace("'", "")
    return text.split()


def text_to_indices(text: str, vocab: dict):
    indexed_text = []
    for token in tokenize(text):
        if token in vocab:
            indexed_text.append(vocab[token])
        else:
            indexed_text.append(vocab["<UNK>"])
    return indexed_text

# Same model architecture used during training — must match exactly so the
# saved weights (state_dict) load correctly.
class RNN(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim=64, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc_layer = nn.Linear(hidden_dim, output_size)

    def forward(self, question):
        embedded_question = self.dropout1(self.embedding(question))
        output, (hidden, cell) = self.lstm(embedded_question)
        hidden = self.dropout2(hidden.squeeze(0))
        return self.fc_layer(hidden)


# Load vocab + answer_vocab + trained weights once, at startup.
with open(MODEL_DIR / "vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open(MODEL_DIR / "answer_vocab.pkl", "rb") as f:
    answer_vocab = pickle.load(f)

idx_to_answer = {index: answer for answer, index in answer_vocab.items()}

model = RNN(vocab_size=len(vocab), output_size=len(answer_vocab))
model.load_state_dict(torch.load(MODEL_DIR / "qa_model.pth", map_location="cpu"))
model.eval()  # inference mode: disables dropout

# Fallback: the model only recognizes questions worded (almost) exactly like
# training data. If its confidence is low, fall back to finding the most
# similar question in the original dataset and return that answer instead —
# this makes reworded/paraphrased questions ("capital city of France") still
# work as long as something close exists in the dataset.
qa_df = pd.read_csv(MODEL_DIR / "qa_dataset.csv")
known_questions = qa_df["question"].tolist()


def find_closest_question(question: str, cutoff: float = 0.6):
    matches = difflib.get_close_matches(question, known_questions, n=1, cutoff=cutoff)
    if not matches:
        return None, None
    matched_question = matches[0]
    matched_answer = qa_df.loc[qa_df["question"] == matched_question, "answer"].iloc[0]
    return matched_question, matched_answer


def predict(question: str, threshold: float = 0.3):
    numerical_question = text_to_indices(question, vocab)
    if not numerical_question:
        return "Please ask a question.", 0.0

    question_tensor = torch.tensor(numerical_question).unsqueeze(0)

    with torch.no_grad():
        output = model(question_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_index = torch.max(probs, dim=1)

    if confidence.item() >= threshold:
        return idx_to_answer[predicted_index.item()], confidence.item()

    # Model isn't confident — try a fuzzy match against known questions
    matched_question, matched_answer = find_closest_question(question)
    if matched_answer is not None:
        return matched_answer, confidence.item()

    return "I don't know", confidence.item()


# FastAPI app
app = FastAPI(title="QA Model API")


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    confidence: float


@app.post("/predict", response_model=AnswerResponse)
def get_prediction(payload: QuestionRequest):
    answer, confidence = predict(payload.question)
    return AnswerResponse(answer=answer, confidence=round(confidence, 4))


@app.get("/")
def serve_frontend():
    return FileResponse(STATIC_DIR / "index.html")


# Serve style.css / script.js etc.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)