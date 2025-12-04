"""
RAG Engine - PDF processing and question answering with LangChain
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Use persistent volume paths for Railway
UPLOAD_DIR = Path("/app/data/uploads")
CHROMA_DIR = Path("/app/data/chroma_db")

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Text splitter for chunking documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)


def get_vector_store() -> Optional[Chroma]:
    """Get or create the vector store."""
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        return None
    
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )


def process_pdf(file_path: Path) -> Dict[str, Any]:
    """Process a PDF and add it to the vector store."""
    # Load PDF
    loader = PyPDFLoader(str(file_path))
    documents = loader.load()
    
    # Add source metadata
    for doc in documents:
        doc.metadata["source"] = file_path.name
    
    # Split into chunks
    chunks = text_splitter.split_documents(documents)
    
    # Add to vector store
    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )
    
    vector_store.add_documents(chunks)
    
    return {
        "filename": file_path.name,
        "pages": len(documents),
        "chunks": len(chunks)
    }


def rebuild_index() -> Dict[str, Any]:
    """Rebuild the entire vector index from all uploaded PDFs."""
    # Clear existing index
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all PDFs
    pdf_files = list(UPLOAD_DIR.glob("*.pdf"))
    
    if not pdf_files:
        return {"status": "no files", "documents": 0, "chunks": 0}
    
    all_chunks = []
    
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        
        for doc in documents:
            doc.metadata["source"] = pdf_path.name
        
        chunks = text_splitter.split_documents(documents)
        all_chunks.extend(chunks)
    
    # Create new vector store with all chunks
    if all_chunks:
        Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory=str(CHROMA_DIR)
        )
    
    return {
        "status": "rebuilt",
        "documents": len(pdf_files),
        "chunks": len(all_chunks)
    }


def ask_question(question: str) -> Dict[str, Any]:
    """Ask a question and get an answer with sources."""
    vector_store = get_vector_store()
    
    if vector_store is None:
        return {
            "answer": "No documents have been indexed yet. Please upload a PDF first.",
            "sources": []
        }
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # Get relevant documents
    relevant_docs = retriever.get_relevant_documents(question)
    
    if not relevant_docs:
        return {
            "answer": "I couldn't find any relevant information in the uploaded documents.",
            "sources": []
        }
    
    # Build context from relevant docs
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Create prompt
    prompt = f"""Based on the following context from uploaded documents, answer the question. 
If the answer cannot be found in the context, say "I don't have enough information to answer that based on the uploaded documents."

Context:
{context}

Question: {question}

Answer:"""
    
    # Get answer from LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    response = llm.invoke(prompt)
    
    # Format sources
    sources = []
    seen = set()
    for doc in relevant_docs:
        source_key = (doc.metadata.get("source", "Unknown"), doc.metadata.get("page", 0))
        if source_key not in seen:
            seen.add(source_key)
            sources.append({
                "document": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 0) + 1,  # 1-indexed for display
                "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
    
    return {
        "answer": response.content,
        "sources": sources
    }


def get_document_list() -> List[Dict[str, Any]]:
    """Get list of uploaded documents."""
    documents = []
    
    for pdf_path in UPLOAD_DIR.glob("*.pdf"):
        documents.append({
            "name": pdf_path.name,
            "size": pdf_path.stat().st_size
        })
    
    return documents


def delete_document(filename: str) -> bool:
    """Delete a document and rebuild index."""
    file_path = UPLOAD_DIR / filename
    
    if file_path.exists():
        file_path.unlink()
        rebuild_index()
        return True
    
    return False


def get_status() -> Dict[str, Any]:
    """Get current system status."""
    pdf_count = len(list(UPLOAD_DIR.glob("*.pdf")))
    
    vector_store = get_vector_store()
    chunk_count = 0
    
    if vector_store is not None:
        try:
            chunk_count = vector_store._collection.count()
        except:
            pass
    
    return {
        "documents": pdf_count,
        "chunks": chunk_count,
        "index_exists": vector_store is not None
    }
