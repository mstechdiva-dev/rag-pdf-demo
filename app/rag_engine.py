"""
RAG Engine for PDF document processing and question answering.
Uses LangChain, ChromaDB, and OpenAI.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# Use Railway volume paths for persistence
UPLOAD_DIR = Path("/app/data/uploads")
CHROMA_DIR = Path("/app/data/chroma_db")

# Create directories if they don't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


class RAGEngine:
    """RAG Engine for processing PDFs and answering questions."""
    
    def __init__(self):
        """Initialize the RAG engine with OpenAI embeddings."""
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self._vector_store: Optional[Chroma] = None
    
    def get_vector_store(self) -> Chroma:
        """Get or create the Chroma vector store."""
        if self._vector_store is None:
            self._vector_store = Chroma(
                persist_directory=str(CHROMA_DIR),
                embedding_function=self.embeddings,
                collection_name="pdf_documents"
            )
        return self._vector_store
    
    def process_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Process a PDF file: load, chunk, embed, and store.
        
        Args:
            file_path: Path to the PDF file (string)
            
        Returns:
            Dict with processing details (chunks count, etc.)
        """
        # Convert string to Path for consistent handling
        file_path = Path(file_path)
        
        # Load PDF
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()
        
        # Add source metadata
        for doc in documents:
            doc.metadata["source"] = file_path.name
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Add to vector store
        vector_store = self.get_vector_store()
        vector_store.add_documents(chunks)
        
        return {
            "filename": file_path.name,
            "pages": len(documents),
            "chunks": len(chunks)
        }
    
    def rebuild_index(self) -> Dict[str, Any]:
        """
        Rebuild the entire vector index from all PDFs in upload directory.
        
        Returns:
            Dict with rebuild details
        """
        # Clear existing vector store
        if CHROMA_DIR.exists():
            import shutil
            shutil.rmtree(CHROMA_DIR)
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Reset vector store reference
        self._vector_store = None
        
        # Process all PDFs
        total_chunks = 0
        processed_files = []
        
        for pdf_file in UPLOAD_DIR.glob("*.pdf"):
            result = self.process_pdf(str(pdf_file))
            total_chunks += result["chunks"]
            processed_files.append(result["filename"])
        
        return {
            "documents_processed": len(processed_files),
            "total_chunks": total_chunks,
            "files": processed_files
        }
    
    def ask(self, question: str, k: int = 4) -> Dict[str, Any]:
        """
        Ask a question and get an answer based on the indexed documents.
        
        Args:
            question: The question to ask
            k: Number of chunks to retrieve (default 4)
            
        Returns:
            Dict with answer and source documents
        """
        vector_store = self.get_vector_store()
        
        # Check if we have any documents
        collection = vector_store._collection
        if collection.count() == 0:
            return {
                "answer": "No documents have been uploaded yet. Please upload some PDF documents first.",
                "sources": []
            }
        
        # Create retriever with k parameter
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        # Create prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create LLM
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        # Get answer
        result = qa_chain.invoke({"query": question})
        
        # Format sources
        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "document": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 0) + 1,
                "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
        
        return {
            "answer": result["result"],
            "sources": sources
        }
    
    def get_documents(self) -> List[Dict[str, str]]:
        """
        Get list of uploaded documents.
        
        Returns:
            List of dicts with document info
        """
        documents = []
        for pdf_file in UPLOAD_DIR.glob("*.pdf"):
            documents.append({
                "name": pdf_file.name,
                "size": f"{pdf_file.stat().st_size / 1024:.1f} KB"
            })
        return documents
    
    def delete_document(self, filename: str) -> bool:
        """
        Delete a document and rebuild the index.
        
        Args:
            filename: Name of the file to delete
            
        Returns:
            True if successful, False otherwise
        """
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            file_path.unlink()
            # Rebuild index without this document
            self.rebuild_index()
            return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the RAG engine.
        
        Returns:
            Dict with status information
        """
        vector_store = self.get_vector_store()
        collection = vector_store._collection
        
        return {
            "documents": len(list(UPLOAD_DIR.glob("*.pdf"))),
            "chunks_indexed": collection.count(),
            "index_exists": collection.count() > 0
        }
