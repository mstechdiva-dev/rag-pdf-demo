"""
RAG Engine - Core document processing and question-answering logic

This module handles:
1. PDF text extraction
2. Text chunking (splitting documents into manageable pieces)
3. Embedding generation (converting text to vectors)
4. Vector storage and retrieval (ChromaDB)
5. Answer generation with source citations

How RAG works:
1. CHUNKING: Documents are split into smaller pieces (chunks)
   - Why? LLMs have context limits, and smaller chunks = more precise retrieval
   - We use 1000 chars with 200 char overlap to maintain context between chunks

2. EMBEDDINGS: Each chunk is converted to a vector (list of numbers)
   - Why? Vectors allow semantic search - finding similar meaning, not just keywords
   - OpenAI's embedding model converts text to 1536-dimensional vectors

3. STORAGE: Vectors are stored in ChromaDB (a vector database)
   - Why? Fast similarity search across thousands of chunks
   - ChromaDB persists to disk - survives restarts

4. RETRIEVAL: When user asks a question, we:
   - Convert question to vector
   - Find most similar chunks (default: top 4)
   - Return those chunks as context

5. GENERATION: Send question + retrieved chunks to LLM
   - LLM answers ONLY from provided context
   - Includes source citations (which document, which page)
"""

import os
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Directories
UPLOAD_DIR = Path("uploads")
CHROMA_DIR = Path("chroma_db")

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for PDF documents.
    """
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=0  # Deterministic answers for factual retrieval
        )
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,      # Characters per chunk
            chunk_overlap=200,    # Overlap between chunks for context continuity
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # Split priority: paragraphs > lines > words
        )
        self._load_existing_index()
    
    def _load_existing_index(self):
        """Load existing ChromaDB index if it exists."""
        if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
            try:
                self.vectorstore = Chroma(
                    persist_directory=str(CHROMA_DIR),
                    embedding_function=self.embeddings
                )
                print(f"Loaded existing index from {CHROMA_DIR}")
            except Exception as e:
                print(f"Could not load existing index: {e}")
                self.vectorstore = None
    
    def process_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single PDF file and add to the vector store.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with processing details
        """
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        # Add source metadata
        for page in pages:
            page.metadata["source"] = Path(file_path).name
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(pages)
        
        # Add to vector store
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=str(CHROMA_DIR)
            )
        else:
            self.vectorstore.add_documents(chunks)
        
        return {
            "file": Path(file_path).name,
            "pages": len(pages),
            "chunks": len(chunks)
        }
    
    def rebuild_index(self) -> Dict[str, Any]:
        """
        Rebuild the entire vector index from all PDFs in uploads folder.
        
        Use this when:
        - You've deleted files manually
        - The index seems corrupted
        - You want to start fresh
        """
        # Clear existing index
        if CHROMA_DIR.exists():
            import shutil
            shutil.rmtree(CHROMA_DIR)
            CHROMA_DIR.mkdir()
        
        self.vectorstore = None
        
        # Process all PDFs
        pdf_files = list(UPLOAD_DIR.glob("*.pdf"))
        results = []
        
        for pdf_path in pdf_files:
            result = self.process_pdf(str(pdf_path))
            results.append(result)
        
        return {
            "documents_processed": len(results),
            "details": results
        }
    
    def ask(self, question: str, k: int = 4) -> Dict[str, Any]:
        """
        Ask a question about the uploaded documents.
        
        Args:
            question: The question to ask
            k: Number of chunks to retrieve (default 4)
            
        Returns:
            Dictionary with answer and source citations
        """
        if self.vectorstore is None:
            return {
                "answer": "No documents have been uploaded yet. Please upload some PDFs first.",
                "sources": []
            }
        
        # Custom prompt that emphasizes answering from context only
        prompt_template = """Use the following pieces of context to answer the question. 
If you don't know the answer based on the context, say "I couldn't find information about that in the uploaded documents."
Always cite which document(s) your answer comes from.

Context:
{context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": k}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        # Get answer
        result = qa_chain.invoke({"query": question})
        
        # Extract source information
        sources = []
        seen_sources = set()
        for doc in result.get("source_documents", []):
            source_name = doc.metadata.get("source", "Unknown")
            page_num = doc.metadata.get("page", "?")
            source_key = f"{source_name}-{page_num}"
            
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append({
                    "document": source_name,
                    "page": page_num,
                    "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
        
        return {
            "answer": result["result"],
            "sources": sources
        }
    
    def get_documents(self) -> List[Dict[str, Any]]:
        """List all uploaded PDF documents."""
        documents = []
        for pdf_path in UPLOAD_DIR.glob("*.pdf"):
            stat = pdf_path.stat()
            documents.append({
                "name": pdf_path.name,
                "size_kb": round(stat.st_size / 1024, 1),
                "modified": stat.st_mtime
            })
        return sorted(documents, key=lambda x: x["modified"], reverse=True)
    
    def delete_document(self, filename: str) -> bool:
        """Delete a document and rebuild index."""
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            file_path.unlink()
            self.rebuild_index()  # Rebuild to remove from vector store
            return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        doc_count = len(list(UPLOAD_DIR.glob("*.pdf")))
        chunk_count = 0
        
        if self.vectorstore is not None:
            try:
                chunk_count = self.vectorstore._collection.count()
            except Exception:
                pass
        
        return {
            "status": "ready",
            "documents": doc_count,
            "chunks_indexed": chunk_count,
            "index_exists": self.vectorstore is not None
        }
