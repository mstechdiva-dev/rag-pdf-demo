# rag-pdf

A self-hosted RAG (Retrieval-Augmented Generation) application that lets you upload PDF documents and ask questions about their contents. Built to search my own cover letters, portfolio documents, and career materials.

**Live Demo Use Case:** "Ask me anything about my background and I'll query my own vector database."

## How It Works

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Upload    │───▶│   Chunk     │───▶│   Embed     │───▶│   Store     │
│    PDF      │    │   Text      │    │   Vectors   │    │  ChromaDB   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Ask      │───▶│  Retrieve   │───▶│   Send to   │───▶│   Answer    │
│  Question   │    │   Chunks    │    │    LLM      │    │ + Sources   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### RAG Pipeline Explained

1. **Chunking**: PDFs are split into ~1000 character pieces with 200 char overlap
2. **Embedding**: Each chunk becomes a 1536-dimensional vector via OpenAI
3. **Storage**: Vectors stored in ChromaDB (file-based, no server needed)
4. **Retrieval**: Questions are vectorized and matched against stored chunks
5. **Generation**: Top 4 matches + question sent to GPT-4o for grounded answer

## Tech Stack

- **Backend**: FastAPI (Python)
- **RAG**: LangChain + OpenAI API
- **Vector DB**: ChromaDB (local, file-based)
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Deployment**: Railway, Render, or any Linux server

## Deployment (Railway - Browser Only)

No terminal required. Everything through your browser.

### Step 1: Push to GitHub

1. Go to [github.com](https://github.com) → Click **+** → **New repository**
2. Name it `rag-pdf`, keep it Public, click **Create repository**
3. On the next page, click **uploading an existing file**
4. Drag all the files from the unzipped `rag-pdf` folder into the browser
5. Click **Commit changes**

### Step 2: Deploy on Railway

1. Go to [railway.app](https://railway.app) → Sign in with GitHub
2. Click **New Project** → **Deploy from GitHub repo**
3. Select your `rag-pdf` repository
4. Railway auto-detects Python and starts building

### Step 3: Add Your API Key

1. In Railway, click on your project
2. Go to **Variables** tab
3. Click **+ New Variable**
4. Add: `OPENAI_API_KEY` = `your-openai-api-key-here`
5. Railway will auto-redeploy

### Step 4: Get Your URL

1. Go to **Settings** tab
2. Under **Domains**, click **Generate Domain**
3. You'll get a URL like: `rag-pdf-production.up.railway.app`

**Done.** Share that URL. Upload PDFs through the web interface.

## Project Structure

```
rag-pdf/
├── app/
│   ├── main.py           # FastAPI routes and endpoints
│   ├── rag_engine.py     # LangChain document processing and QA
│   └── templates/
│       └── index.html    # Web UI
├── uploads/              # Where PDFs are stored
├── chroma_db/            # Vector database files (auto-created)
├── requirements.txt      # Python dependencies
├── .env.example          # Environment template
└── README.md
```

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Serve web UI |
| POST | `/upload` | Upload PDF |
| POST | `/rebuild-index` | Reprocess all PDFs |
| POST | `/ask` | Submit a question |
| GET | `/documents` | List uploaded PDFs |
| DELETE | `/documents/{name}` | Remove a PDF |
| GET | `/status` | System health check |
| POST | `/export-chat` | Download Q&A session |

## Features

- **Drag & drop uploads** - Simple PDF upload via browser
- **Chat interface** - Ask questions in a conversational UI  
- **Source citations** - See which documents each answer came from
- **Document management** - View and delete uploaded files
- **Persistent index** - Vector database saved to disk, survives restarts

## Cost Estimate

- **Hosting**: Free tier (Railway/Render)
- **OpenAI API**: 
  - Indexing: ~$0.01-0.05 per document
  - Questions: ~$0.01-0.03 per question (GPT-4o)
- **Total**: A few dollars per month for moderate use

## Limitations

- Requires OpenAI API (not fully offline)
- Scanned PDFs need OCR preprocessing
- No user authentication (add if exposing publicly)
- Single-user design (no multi-tenancy)

## Why LangChain?

For this use case, LangChain provides:
- Clean abstractions for chunking and embedding
- Built-in ChromaDB integration
- Retrieval chain with source tracking
- Easy swap between models/providers

For a simple RAG app like this, LangChain reduces boilerplate without adding unnecessary complexity.

---

## Run Locally (Optional)

If you prefer to run on your own machine:

```bash
cd rag-pdf
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env
uvicorn app.main:app --reload
```

Open http://localhost:8000

---

A RAG demonstration project.
