# main.py

# --- Core Imports ---
import os
import logging
import hashlib
import json
import re
import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Optional, Dict

# --- Third-party Imports ---
# FastAPI
from fastapi import FastAPI, HTTPException, Depends, status, APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl

# Environment Management
from dotenv import load_dotenv

# Database (SQLAlchemy & PostgreSQL with pgvector)
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector

# PDF & Text Processing
import requests
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# AI & Embeddings
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
import numpy as np

# --- Initial Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables ---
# These will be initialized in the lifespan manager
embedding_model = None
gemini_model = None
db_engine = None
SessionLocal = None

# --- Database Model Setup ---
Base = declarative_base()

class Document(Base):
    """SQLAlchemy model for storing document metadata."""
    __tablename__ = "documents"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    url = Column(String, unique=True, index=True, nullable=False)
    url_hash = Column(String, unique=True, index=True, nullable=False)
    content_hash = Column(String, nullable=False)
    processed_at = Column(DateTime, default=datetime.utcnow)
    chunk_count = Column(Integer, default=0)

class DocumentChunk(Base):
    """SQLAlchemy model for storing document chunks and their embeddings."""
    __tablename__ = "document_chunks"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(384))  # all-MiniLM-L6-v2 produces 384-dim vectors

# --- Lifespan Manager for Resource Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the FastAPI application."""
    global embedding_model, gemini_model, db_engine, SessionLocal
    logger.info("Application startup: Initializing resources...")
    try:
        # 1. Database Connection
        DATABASE_URL = os.getenv("DATABASE_URL")
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL environment variable not set.")
        db_engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
        Base.metadata.create_all(bind=db_engine)
        logger.info("Database connection successful and tables verified.")

        # 2. Embedding Model
        model_name = 'all-MiniLM-L6-v2'
        embedding_model = SentenceTransformer(model_name)
        logger.info(f"Embedding model '{model_name}' loaded successfully.")

        # 3. Google Gemini Model
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=GOOGLE_API_KEY)

        # Safer, more effective system prompt
        system_instruction = """You are an expert Q&A assistant for legal and policy documents. Your task is to answer questions based *only* on the provided context.

INSTRUCTIONS:
1. Read the user's questions and the provided context chunks carefully.
2. For each question, find the answer directly within the context.
3. Your answers must be concise, ideally 1-2 sentences.
4. **Crucially, do not add any information that is not present in the context.** Do not make assumptions or invent facts.
5. **If the answer to a question cannot be found in the provided context, you MUST respond with the exact phrase: "Information not available in the provided context."** This is not a failure; it is the correct action when the information is missing.
6. You must provide an answer for every question, even if it's the "Information not available" response.

Respond in a valid JSON object with a single key "answers" that contains a list of strings. For example:
{
  "answers": [
    "Answer to question 1.",
    "Information not available in the provided context.",
    "Answer to question 3."
  ]
}"""

        gemini_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            system_instruction=system_instruction,
            generation_config={"response_mime_type": "application/json"}
        )
        logger.info("Google Gemini 1.5 Flash model initialized successfully.")

        yield  # Application is now running

    except Exception as e:
        logger.error(f"Fatal error during application startup: {e}")
        raise
    finally:
        logger.info("Application shutdown: Cleaning up resources.")
        # Cleanup can be added here if needed

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Advanced RAG API with Gemini 1.5 Flash",
    description="A robust RAG system for document Q&A, featuring intelligent re-ranking and Google's Gemini 1.5 Flash.",
    version="3.0.0",
    lifespan=lifespan
)

# --- Security and Authentication ---
security = HTTPBearer()
VALID_TOKEN = os.getenv("BEARER_TOKEN")
if not VALID_TOKEN:
    raise ValueError("BEARER_TOKEN environment variable is not set.")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validates the bearer token provided in the request."""
    if credentials.scheme != "Bearer" or credentials.credentials != VALID_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials

# --- Pydantic Models for API Schema ---
class ProcessRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class ProcessResponse(BaseModel):
    answers: List[str]

# --- Helper Functions ---
def get_db():
    """Dependency to get a new database session for each request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_content_hash(content: str) -> str:
    """Generates a SHA256 hash for a string to detect content changes."""
    return hashlib.sha256(content.encode()).hexdigest()

# --- Core Logic Classes ---

class PDFProcessor:
    """Handles downloading and extracting text from PDF documents."""
    @staticmethod
    def download_and_extract(url: str) -> str:
        logger.info(f"Downloading PDF from {url}")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, timeout=60)
                response.raise_for_status()
                temp_file.write(response.content)
                temp_file_path = temp_file.name

            logger.info("PDF downloaded. Extracting text with PyPDFLoader...")
            loader = PyPDFLoader(temp_file_path)
            pages = loader.load()
            text = "\n".join([page.page_content for page in pages])
            logger.info(f"Text extraction complete. Total characters: {len(text)}")
            return text
        except requests.RequestException as e:
            logger.error(f"Failed to download PDF: {e}")
            raise HTTPException(status_code=400, detail=f"Could not download PDF from URL: {e}")
        except Exception as e:
            logger.error(f"Failed to process PDF: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing PDF file: {e}")
        finally:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

class TextChunker:
    """Splits text into manageable, overlapping chunks."""
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk(self, text: str) -> List[str]:
        logger.info("Chunking text...")
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Text split into {len(chunks)} chunks.")
        return chunks

class DatabaseManager:
    """Manages all database interactions for documents and chunks."""
    @staticmethod
    def get_document_by_hash(db: Session, url_hash: str, content_hash: str) -> Optional[Document]:
        return db.query(Document).filter(
            Document.url_hash == url_hash,
            Document.content_hash == content_hash
        ).first()

    @staticmethod
    def save_document_and_chunks(db: Session, url: str, text_content: str, chunks: List[str]) -> Document:
        url_hash = get_content_hash(url)
        content_hash = get_content_hash(text_content)
        
        try:
            logger.info("Saving new document and chunks to the database.")
            doc = Document(
                url=url,
                url_hash=url_hash,
                content_hash=content_hash,
                chunk_count=len(chunks)
            )
            db.add(doc)
            db.flush()  # To get the generated doc.id

            embeddings = embedding_model.encode(chunks, show_progress_bar=True)
            
            chunk_objects = []
            for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_obj = DocumentChunk(
                    document_id=doc.id,
                    chunk_index=i,
                    content=chunk_text,
                    embedding=embedding.tolist()
                )
                chunk_objects.append(chunk_obj)
            
            db.bulk_save_objects(chunk_objects)
            db.commit()
            logger.info(f"Successfully saved document {doc.id} with {len(chunks)} chunks.")
            return doc
        except Exception as e:
            db.rollback()
            logger.error(f"Database error while saving document: {e}")
            raise HTTPException(status_code=500, detail="Failed to save document to database.")

    @staticmethod
    def get_chunks_for_document(db: Session, document_id: uuid.UUID) -> List[Dict]:
        chunks = db.query(DocumentChunk.content, DocumentChunk.embedding).filter(
            DocumentChunk.document_id == document_id
        ).all()
        return [{"content": c.content, "embedding": np.array(c.embedding)} for c in chunks]

class EnhancedRAGRetriever:
    """Implements an advanced retrieval strategy with vector search and re-ranking."""
    def __init__(self, all_chunks: List[Dict]):
        self.all_chunks = all_chunks
        self.corpus_embeddings = np.array([chunk['embedding'] for chunk in all_chunks])
        logger.info(f"RAG Retriever initialized with {len(all_chunks)} chunks.")

    def retrieve(self, query: str, top_k: int = 20) -> List[str]:
        """
        Retrieves the most relevant chunks for a given query.
        1. Encodes the query.
        2. Performs a semantic search to find the top_k initial candidates.
        3. Re-ranks these candidates for better contextual relevance.
        """
        logger.info(f"Retrieving context for query: '{query[:80]}...'")
        query_embedding = embedding_model.encode(query, convert_to_tensor=True)

        # 1. Semantic Search (Initial Retrieval)
        # Using cosine similarity to find the top_k most similar chunks
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        top_results = np.argpartition(-cos_scores, range(top_k))[:top_k]
        
        # 2. Re-ranking
        # Here, we could use a more advanced cross-encoder, but for simplicity and speed,
        # we'll re-rank based on the initial cosine scores which is often sufficient.
        # The key is retrieving a larger pool first (top_k) and then selecting the best.
        
        ranked_chunks = sorted(
            [
                (cos_scores[i], self.all_chunks[i]['content']) for i in top_results
            ], 
            key=lambda x: x[0], 
            reverse=True
        )

        # Select the top N chunks after re-ranking (e.g., top 10)
        final_context_chunks = [content for score, content in ranked_chunks[:10]]
        logger.info(f"Retrieved and re-ranked, selected {len(final_context_chunks)} final chunks.")
        
        return final_context_chunks

class GeminiProcessor:
    """Handles interaction with the Google Gemini API."""
    @staticmethod
    def generate_answers(questions: List[str], context_chunks: List[str]) -> List[str]:
        if not context_chunks:
            logger.warning("LLM processor called with no context. Returning default message.")
            return ["Information not available in the provided context." for _ in questions]

        context_str = "\n\n".join([f"--- Context Chunk {i+1} ---\n{chunk}" for i, chunk in enumerate(context_chunks)])
        questions_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

        prompt = f"""
CONTEXT:
{context_str}

QUESTIONS:
Based *only* on the context provided above, answer the following questions:
{questions_str}
"""
        logger.info(f"Sending prompt to Gemini. Context length: {len(context_str)}, Questions: {len(questions)}")
        
        try:
            response = gemini_model.generate_content(prompt)
            response_text = response.text
            
            # The model is configured for JSON output, so we parse it directly.
            parsed_json = json.loads(response_text)
            answers = parsed_json.get("answers", [])

            if not isinstance(answers, list) or len(answers) != len(questions):
                raise ValueError("LLM returned malformed JSON or incorrect number of answers.")
            
            logger.info("Successfully received and parsed answers from Gemini.")
            return answers

        except Exception as e:
            logger.error(f"Error communicating with Gemini API: {e}")
            logger.error(f"Raw response text (if available): {getattr(response, 'text', 'N/A')}")
            # Fallback in case of API or parsing error
            return [f"Error generating answer: {e}" for _ in questions]


# --- API Endpoint ---
router = APIRouter(prefix="/api/v1")

@router.post("/process", response_model=ProcessResponse, dependencies=[Depends(verify_token)])
async def process_documents_and_questions(request: ProcessRequest, db: Session = Depends(get_db)):
    """
    Main endpoint to process a document and answer questions about it.
    - Downloads and processes the PDF if not in cache.
    - Uses an advanced RAG retriever to find the best context.
    - Uses Gemini 1.5 Flash to generate answers.
    """
    url = str(request.documents)
    logger.info(f"--- New Request --- URL: {url}, Questions: {len(request.questions)}")

    # Step 1: Download and get content hash
    text_content = PDFProcessor.download_and_extract(url)
    if not text_content or len(text_content) < 100:
        raise HTTPException(status_code=400, detail="Failed to extract sufficient text from the document.")
    
    content_hash = get_content_hash(text_content)
    url_hash = get_content_hash(url)

    # Step 2: Check cache
    doc = DatabaseManager.get_document_by_hash(db, url_hash, content_hash)
    
    if doc:
        logger.info(f"Cache hit. Using existing document chunks for doc ID: {doc.id}")
        all_chunks = DatabaseManager.get_chunks_for_document(db, doc.id)
    else:
        logger.info("Cache miss. Processing and storing new document.")
        # Step 2a: Chunk and save if not in cache
        chunker = TextChunker()
        chunks = chunker.chunk(text_content)
        new_doc = DatabaseManager.save_document_and_chunks(db, url, text_content, chunks)
        all_chunks = DatabaseManager.get_chunks_for_document(db, new_doc.id)

    # Step 3: Retrieve relevant context for all questions
    retriever = EnhancedRAGRetriever(all_chunks)
    
    # Consolidate context from all questions to provide a broader view to the LLM
    consolidated_context = set()
    for question in request.questions:
        chunks_for_q = retriever.retrieve(question)
        for chunk in chunks_for_q:
            consolidated_context.add(chunk)
            
    final_context = list(consolidated_context)
    logger.info(f"Consolidated context from all questions into {len(final_context)} unique chunks.")

    # Step 4: Generate answers using Gemini
    answers = GeminiProcessor.generate_answers(request.questions, final_context)

    return ProcessResponse(answers=answers)

app.include_router(router)

@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Welcome to the Advanced RAG API with Gemini 1.5 Flash",
        "api_docs": "/docs"
    }

# To run this application:
# uvicorn main:app --reload
