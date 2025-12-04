"""
PDF Q&A System - FastAPI Application

A self-hosted web application that lets you upload PDF documents and ask questions
about their contents using AI. The system only answers based on your uploaded 
documents - it won't make things up or pull from outside sources.

Endpoints:
    GET  /              - Serve web UI
    POST /upload        - Upload PDF
    POST /rebuild-index - Process all PDFs into vector DB
    POST /ask           - Submit a question
    POST /export-chat   - Download Q&A as text file
    GET  /documents     - List uploaded PDFs
    DELETE /documents/{name} - Remove a PDF
    GET  /status        - System health check
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY not set. Create a .env file with your key.")

from app.rag_engine import RAGEngine, UPLOAD_DIR

# Initialize FastAPI app
app = FastAPI(
    title="PDF Q&A System",
    description="Ask questions about your PDF documents using RAG",
    version="1.0.0"
)

# Initialize RAG engine
rag = RAGEngine()

# Templates
templates = Jinja2Templates(directory="app/templates")


# Request/Response models
class QuestionRequest(BaseModel):
    question: str
    k: int = 4  # Number of chunks to retrieve


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ExportRequest(BaseModel):
    messages: list[ChatMessage]


# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main web UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF document for indexing.
    
    The file is saved to the uploads folder and immediately processed
    into the vector database.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Save file
    file_path = UPLOAD_DIR / file.filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process into vector store
        result = rag.process_pdf(str(file_path))
        
        return JSONResponse({
            "success": True,
            "message": f"Uploaded and indexed {file.filename}",
            "details": result
        })
    
    except Exception as e:
        # Clean up on failure
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebuild-index")
async def rebuild_index():
    """
    Rebuild the entire vector index from all PDFs in uploads folder.
    
    Use this when:
    - You've deleted files manually
    - The index seems corrupted
    - You want to start fresh
    """
    try:
        result = rag.rebuild_index()
        return JSONResponse({
            "success": True,
            "message": f"Rebuilt index with {result['documents_processed']} documents",
            "details": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Ask a question about the uploaded documents.
    
    The system will:
    1. Convert your question to a vector
    2. Find the most relevant chunks from your documents
    3. Send those chunks + your question to the LLM
    4. Return an answer with source citations
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        result = rag.ask(request.question, k=request.k)
        return JSONResponse({
            "success": True,
            "answer": result["answer"],
            "sources": result["sources"]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    """List all uploaded PDF documents."""
    documents = rag.get_documents()
    return JSONResponse({
        "success": True,
        "documents": documents
    })


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """
    Delete a document and rebuild the index.
    
    Note: This triggers a full reindex to remove the document's
    chunks from the vector store.
    """
    success = rag.delete_document(filename)
    if success:
        return JSONResponse({
            "success": True,
            "message": f"Deleted {filename} and rebuilt index"
        })
    else:
        raise HTTPException(status_code=404, detail=f"Document {filename} not found")


@app.get("/status")
async def get_status():
    """Get system status and health check."""
    status = rag.get_status()
    return JSONResponse(status)


@app.post("/export-chat")
async def export_chat(request: ExportRequest):
    """
    Export Q&A session as a text file.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages to export")
    
    # Create export content
    lines = [
        "PDF Q&A Session Export",
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 50,
        ""
    ]
    
    for msg in request.messages:
        role = "Q" if msg.role == "user" else "A"
        lines.append(f"{role}: {msg.content}")
        lines.append("")
    
    content = "\n".join(lines)
    
    # Save to temp file
    export_path = Path("/tmp") / f"qa_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    export_path.write_text(content)
    
    return FileResponse(
        path=str(export_path),
        filename=f"qa_session_{datetime.now().strftime('%Y%m%d')}.txt",
        media_type="text/plain"
    )


# Run with: uvicorn app.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
