# ML Buddy 🤖
A domain-specific AI/ML knowledge assistant built with RAG, LangGraph, FAISS, and Gemini.

---

## What It Does

| Mode | Description |
|---|---|
| 💬 Chat | RAG-powered Q&A over a curated AI/ML knowledge base |
| 🎯 Model Recommender | Describe your problem → get the right model + evaluation metric |
| 🔍 Code Explainer | Paste ML code → get a beginner-friendly explanation |

---

## Architecture

```
User Query
    │
    ▼
[Streamlit UI]
    │
    ├── Chat Mode
    │       ├── Google Embeddings → FAISS similarity search (top-3 chunks)
    │       └── Gemini LLM answers using retrieved context only
    │
    ├── Model Recommender → Prompt Engineering → Gemini API
    └── Code Explainer    → Prompt Engineering → Gemini API
```

---

## Tech Stack

- **Python** — core language
- **LangGraph** — RAG pipeline orchestration
- **LangChain** — document loading, text splitting, vector store
- **FAISS** — vector similarity search
- **Google Embeddings** (`text-embedding-004`) — document & query embeddings
- **Gemini API** (`gemini-2.0-flash`) — LLM for answer generation
- **Streamlit** — web UI

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/ml-buddy.git
cd ml-buddy
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your API key
Create a `.env` file in the root folder:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```
Get your free key at [aistudio.google.com](https://aistudio.google.com)

> ⚠️ Never push `.env` to GitHub — it's already in `.gitignore`

### 4. Run the app
```bash
streamlit run app.py
```

---

## Project Structure

```
ml-buddy/
├── knowledge_base/
│   ├── core_ml.txt          # Supervised/unsupervised, overfitting, regularization
│   ├── deep_learning.txt    # CNNs, RNNs, Transformers, dropout, backprop
│   ├── nlp.txt              # Tokenization, embeddings, BERT, RAG, prompt engineering
│   └── evaluation.txt       # Metrics, confusion matrix, hyperparameter tuning
├── src/
│   ├── rag.py               # LangGraph + FAISS RAG pipeline
│   ├── tasks.py             # Model recommender + code explainer
│   └── prompts.py           # All prompt templates
├── app.py                   # Streamlit UI
├── .env                     # API key (never commit this)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## How RAG Works

1. **Indexing** — `.txt` files loaded, split into 500-word chunks, embedded via Google Embeddings, stored in FAISS. Index saved to disk so it only builds once.
2. **Retrieval** — User question embedded → FAISS finds top 3 similar chunks.
3. **Generation** — Retrieved chunks passed as context to Gemini → answers using only that context, reducing hallucination.

---

## Knowledge Base Covers

Core ML · Deep Learning · NLP · Model Evaluation

---

## .gitignore
Make sure your `.gitignore` includes:
```
.env
faiss_index/
__pycache__/
*.pyc
```