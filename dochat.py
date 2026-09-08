"""
DoChat — RAG Document Chatbot Backend
=======================================
LangChain + FAISS + HuggingFace Embeddings + Mistral LLM
Implements full Retrieval-Augmented Generation pipeline.

Usage:
    python dochat.py
    curl -X POST http://localhost:8001/add -d '{"text": "...", "title": "Doc 1"}'
    curl -X POST http://localhost:8001/query -d '{"question": "What is X?"}'

Requirements:
    pip install langchain langchain-community langchain-huggingface
    pip install sentence-transformers faiss-cpu
    pip install fastapi uvicorn python-dotenv
"""

import os
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── LangChain / Vector Store ──────────────────────────────────────────────────
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── Config ───────────────────────────────────────────────────────────────────
HF_TOKEN       = os.getenv("HF_API_TOKEN", "")
LLM_MODEL_ID   = "mistralai/Mistral-7B-Instruct-v0.2"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # runs locally, no API key
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 50
TOP_K          = 3

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="DoChat RAG API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Global state ──────────────────────────────────────────────────────────────
_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectorstore: Optional[FAISS] = None
_qa_chain = None
_documents: List[dict] = []   # [{id, title, text, chunks}]


# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model (cached after first call)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_ID,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_llm() -> HuggingFaceEndpoint:
    return HuggingFaceEndpoint(
        repo_id=LLM_MODEL_ID,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
        temperature=0.1,
        task="text-generation",
    )


RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are DoChat, an expert document analyst.
Answer the question using ONLY the context provided.
If the answer is not in the context, say: "This information is not in the provided documents."

Context:
{context}

Question: {question}

Answer (be specific, cite document sources when possible):""",
)


def rebuild_chain():
    """Rebuild the RetrievalQA chain from the current vector store."""
    global _qa_chain
    if _vectorstore is None:
        _qa_chain = None
        return
    retriever = _vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    _qa_chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": RAG_PROMPT},
        return_source_documents=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

class AddDocRequest(BaseModel):
    text: str
    title: Optional[str] = None


class QueryRequest(BaseModel):
    question: str


@app.post("/add")
async def add_document(req: AddDocRequest):
    """Add a document to the knowledge base and update the vector store."""
    global _vectorstore

    if len(req.text.strip()) < 20:
        raise HTTPException(400, "Document text is too short (min 20 chars).")

    # Chunk the text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(req.text)
    doc_id = str(uuid.uuid4())[:8]
    title  = req.title or f"Document {len(_documents) + 1}"

    _documents.append({
        "id": doc_id,
        "title": title,
        "text": req.text,
        "chunk_count": len(chunks),
    })

    # Add chunks to FAISS
    metadatas = [{"doc_id": doc_id, "title": title, "chunk_idx": i} for i in range(len(chunks))]
    embeddings = get_embeddings()

    if _vectorstore is None:
        _vectorstore = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)
    else:
        _vectorstore.add_texts(chunks, metadatas=metadatas)

    rebuild_chain()

    return {
        "status": "ok",
        "doc_id": doc_id,
        "title": title,
        "chunk_count": len(chunks),
        "total_docs": len(_documents),
    }


@app.post("/query")
async def query_documents(req: QueryRequest):
    """Query the knowledge base using RAG."""
    if _qa_chain is None:
        raise HTTPException(400, "No documents loaded. Add documents first.")

    result = _qa_chain.invoke(req.question)
    answer = result["result"]
    sources = result.get("source_documents", [])

    citations = []
    seen = set()
    for doc in sources:
        meta = doc.metadata
        key  = (meta.get("doc_id"), meta.get("chunk_idx"))
        if key not in seen:
            seen.add(key)
            citations.append({
                "doc_id":  meta.get("doc_id"),
                "title":   meta.get("title", "Unknown"),
                "excerpt": doc.page_content[:200],
            })

    return {
        "answer":   answer,
        "question": req.question,
        "citations": citations[:3],
    }


@app.get("/documents")
async def list_documents():
    """List all documents in the knowledge base."""
    return {"documents": _documents, "total": len(_documents)}


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Remove a document from the knowledge base."""
    global _documents
    _documents = [d for d in _documents if d["id"] != doc_id]
    # Note: FAISS does not support incremental deletes easily.
    # In production, rebuild the index or use Chroma/Weaviate instead.
    rebuild_chain()
    return {"status": "ok", "remaining_docs": len(_documents)}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "docs_loaded": len(_documents),
        "vector_store": _vectorstore is not None,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
