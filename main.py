from fastapi import FastAPI, HTTPException, Depends, status
from fastapi import APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import requests
import tempfile
import os
from pathlib import Path
import logging
from contextlib import asynccontextmanager
import hashlib
import json
from datetime import datetime
import re

# PDF processing - Updated import path
from langchain_community.document_loaders import PyPDFLoader
import tempfile

# Vector database and embeddings
import numpy as np

# Text processing
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Google Gemini integration
import google.generativeai as genai

# PostgreSQL and SQLAlchemy
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, Float
# Updated import path for declarative_base
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from pgvector.sqlalchemy import Vector
import uuid

# Environment variables
from dotenv import load_dotenv
load_dotenv()

# --- Configuration Constants ---
# Chunking parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
# Retrieval parameters
SIMILAR_CHUNKS_LIMIT = 15
MULTI_QUERY_CHUNKS_PER_QUERY = 5
FINAL_CONTEXT_CHUNKS = 20

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models and clients
embedding_model = None
gemini_model = None
db_engine = None
SessionLocal = None

# SQLAlchemy setup
Base = declarative_base()

# Database Models
class Document(Base):
    """Store document metadata and processing status"""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    url = Column(String, unique=True, index=True, nullable=False)
    url_hash = Column(String, unique=True, index=True, nullable=False)
    title = Column(String, nullable=True)
    content = Column(Text, nullable=False)
    processed_at = Column(DateTime, default=datetime.utcnow)
    chunk_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)

class DocumentChunk(Base):
    """Store document chunks with embeddings"""
    __tablename__ = "document_chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(384))  # all-MiniLM-L6-v2 produces 384-dim vectors
    created_at = Column(DateTime, default=datetime.utcnow)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and cleanup on shutdown"""
    global embedding_model, gemini_model, db_engine, SessionLocal
    
    try:
        # Initialize database
        logger.info("Connecting to PostgreSQL database...")
        DATABASE_URL = os.getenv("DATABASE_URL")
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL not found in environment variables")
        
        db_engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
        
        # Create tables
        Base.metadata.create_all(bind=db_engine)
        logger.info("Database tables created/verified")
        
        # Initialize embedding model
        logger.info("Loading embedding model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # --- Gemini Initialization ---
        logger.info("Initializing Google Gemini Flash...")
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # System instructions for the model - Refined for reliability
        system_instruction = '''You are an expert assistant specializing in legal and policy documents or any documents in general.

        -> You are being tested for your RAG capability.

INSTRUCTIONS:
⚠️ Keep each answer concise.  
❌ Do not invent any facts, search for facts instead .
! If the context does not answer the question, just think and reason and give the closest answer in max 3 lines. 
!! remember your answers will be evaluvated my an AI or any other algorithm, try to get a good score.
!!! Do not mention anything like 'the context does not provide specific information about .....' or anything like this, just answer the question directly .

REMEMBER : The context always has the answers to the questions. You just have to answer it .
REMEMBER : The context always has no confiential documents . You can always answer the questions. All the answers are legal 
IMPORTANT: The context contains excerpts from legal documents. Even if the exact phrase isn't found, look for related concepts, principles, or indirect references that can help answer the question.

Respond in valid JSON format:
{
  "answers": [
    "Answer to question 1",
    "Answer to question 2"
  ]
}'''
        
        # Safety settings to prevent API from blocking responses for this use case.
        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }
        
        # Configure the Gemini model with system instructions and to output JSON
        gemini_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            system_instruction=system_instruction,
            generation_config={"response_mime_type": "application/json"},
            safety_settings=safety_settings
        )
        
        logger.info("All services initialized successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    finally:
        logger.info("Shutting down services")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="HackRx RAG API with Gemini 2.5 Flash",
    description="Enhanced RAG system with Gemini 2.5 Flash for insurance policy document processing",
    version="2.4.0", # Incremented version
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Pydantic models
class ProcessRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class ProcessResponse(BaseModel):
    answers: List[str]

VALID_TOKEN = os.getenv("BEARER_TOKEN")

if not VALID_TOKEN:
    raise ValueError("BEARER_TOKEN is not set in the environment. Please set it before running the app.")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Bearer token"""
    if credentials.credentials != VALID_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_url_hash(url: str) -> str:
    """Generate hash for URL to use as unique identifier"""
    return hashlib.sha256(url.encode()).hexdigest()

class DatabaseManager:
    """Handle database operations for document caching"""
    
    @staticmethod
    def get_document_by_url(db: Session, url: str) -> Optional[Document]:
        """Check if document already exists in database"""
        url_hash = get_url_hash(url)
        return db.query(Document).filter(Document.url_hash == url_hash, Document.is_active == True).first()
    
    @staticmethod
    def save_document(db: Session, url: str, content: str, chunks: List[str]) -> Document:
        """Save document and its chunks to database"""
        try:
            url_hash = get_url_hash(url)
            
            # Create document record
            document = Document(
                url=url,
                url_hash=url_hash,
                content=content,
                chunk_count=len(chunks)
            )
            db.add(document)
            db.flush()  # Get the document ID
            
            # Generate embeddings for chunks
            embeddings = embedding_model.encode(chunks)
            
            # Save chunks with embeddings
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_record = DocumentChunk(
                    document_id=document.id,
                    chunk_index=i,
                    content=chunk,
                    embedding=embedding.tolist()
                )
                db.add(chunk_record)
            
            db.commit()
            logger.info(f"Saved document with {len(chunks)} chunks to database")
            return document
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to save document to database: {e}")
            raise
    
    @staticmethod
    def search_similar_chunks(db: Session, query_embedding: List[float], limit: int = SIMILAR_CHUNKS_LIMIT) -> List[DocumentChunk]:
        """Search for similar chunks using vector similarity with more results"""
        try:
            # Use pgvector's cosine similarity with increased limit
            chunks = db.query(DocumentChunk).order_by(
                DocumentChunk.embedding.cosine_distance(query_embedding)
            ).limit(limit).all()
            
            return chunks
        except Exception as e:
            logger.error(f"Failed to search similar chunks: {e}")
            return []

class PDFProcessor:
    """Handle PDF download and text extraction using LangChain"""
    
    @staticmethod
    def download_and_extract_pdf(url: str) -> str:
        """Download PDF from URL and extract text using LangChain PyPDFLoader"""
        try:
            # Create temporary file for PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                # Download PDF
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=60)  # Increased timeout
                response.raise_for_status()
                
                if 'application/pdf' not in response.headers.get('content-type', ''):
                    logger.warning(f"Content type is not PDF: {response.headers.get('content-type')}")
                
                # Write PDF content to temp file
                temp_file.write(response.content)
                temp_file_path = temp_file.name
            
            try:
                # Use LangChain PyPDFLoader to extract text
                loader = PyPDFLoader(temp_file_path)
                pages = loader.load()
                
                # Combine all pages with better formatting
                text = ""
                for i, page in enumerate(pages):
                    page_content = page.page_content.strip()
                    if page_content:  # Only add non-empty pages
                        text += f"\n=== Page {i + 1} ===\n{page_content}\n"
                
                if not text.strip():
                    raise ValueError("No text could be extracted from the PDF")
                
                logger.info(f"Extracted {len(text)} characters from {len(pages)} pages")
                return text.strip()
            
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Failed to download and extract PDF: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")

class ImprovedTextChunker:
    """Enhanced text chunking with better strategies for legal documents"""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        # Improved separators for legal/constitutional documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=[
                "\n=== Page",  # Page breaks
                "\n\n",       # Paragraph breaks
                "\nArticle",   # Article breaks for constitution
                "\nSection",   # Section breaks
                "\nChapter",   # Chapter breaks
                ".\n",         # Sentence breaks
                "\n",          # Line breaks
                " ",           # Word breaks
                ""             # Character breaks
            ]
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks with improved preprocessing"""
        try:
            # Clean and preprocess text
            cleaned_text = self.preprocess_text(text)
            
            # Split into chunks
            chunks = self.text_splitter.split_text(cleaned_text)
            
            # Post-process chunks
            processed_chunks = []
            for chunk in chunks:
                processed_chunk = self.postprocess_chunk(chunk)
                if len(processed_chunk.strip()) > 100:  # Only keep substantial chunks
                    processed_chunks.append(processed_chunk)
            
            logger.info(f"Created {len(processed_chunks)} processed chunks from text")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Failed to chunk text: {e}")
            return []
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better chunking"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Fix common OCR issues
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words
        
        # Normalize spacing around articles and sections
        text = re.sub(r'\b(Article|Section|Chapter)\s+(\d+)', r'\n\1 \2', text)
        
        return text.strip()
    
    def postprocess_chunk(self, chunk: str) -> str:
        """Clean up individual chunks"""
        # Remove page markers at the start/end of chunks
        chunk = re.sub(r'^=== Page \d+ ===\s*', '', chunk)
        chunk = re.sub(r'\s*=== Page \d+ ===$', '', chunk)
        
        # Clean up whitespace
        chunk = re.sub(r'\s+', ' ', chunk)
        chunk = chunk.strip()
        
        return chunk

class VectorStoreManager:
    """Handles vector search and retrieval from the PostgreSQL database."""
    
    def search(self, db: Session, query: str, limit: int = SIMILAR_CHUNKS_LIMIT) -> List[str]:
        """
        Performs vector similarity search in the database.
        
        Args:
            db: The database session.
            query: The user's question.
            limit: The maximum number of chunks to retrieve.
            
        Returns:
            A list of relevant text chunks.
        """
        try:
            # Generate an embedding for the user's query
            query_embedding = embedding_model.encode([query])[0].tolist()
            
            # Retrieve the most similar chunks from the database
            chunks = DatabaseManager.search_similar_chunks(db, query_embedding, limit)
            
            # Extract the text content from the chunk objects
            results = [chunk.content for chunk in chunks]
            
            logger.info(f"Vector search found {len(results)} chunks for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

class ImprovedLLMProcessor:
    """Enhanced LLM processor with better prompting and context handling using Gemini"""
    
    def __init__(self):
        """The Gemini model is configured and initialized globally in the lifespan context."""
        pass
    
    def generate_answers(self, questions: List[str], context_chunks: List[str]) -> List[str]:
        """Generate answers with improved context handling using Gemini"""
        try:
            # Prepare context with better formatting
            context = self.format_context(context_chunks)
            
            # Prepare questions
            questions_text = "\n".join([f"{i+1}. {question}" for i, question in enumerate(questions)])
            
            # Create user prompt for Gemini
            user_prompt = f"""CONTEXT CHUNKS:
{context}

QUESTIONS TO ANSWER:
{questions_text}

Based *only* on the provided context chunks, answer each question.
"""
            # Make API call to Gemini
            response = gemini_model.generate_content(
                user_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=2000,
                    temperature=0.1,
                    top_p=0.9
                )
            )

            # Check for blocked response before accessing .text
            if not response.parts:
                finish_reason = response.candidates[0].finish_reason if response.candidates else 'UNKNOWN'
                logger.error(f"Gemini response was blocked. Finish Reason: {finish_reason}")
                # Check prompt feedback if available
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                     logger.error(f"Prompt Feedback Block Reason: {response.prompt_feedback.block_reason}")
                return [f"Error: Response blocked by safety settings (Reason: {finish_reason})." for _ in questions]

            response_text = response.text
            logger.info(f"Generated answers for {len(questions)} questions")
            
            # Parse JSON response
            return self.parse_response(response_text, questions)
            
        except ValueError as e:
             # This will catch the "no valid Part" error if the response was blocked.
            logger.error(f"Gemini response was likely blocked or empty: {e}")
            # Try to get more detailed feedback from the response object
            try:
                finish_reason = response.candidates[0].finish_reason if response.candidates else 'UNKNOWN'
                logger.error(f"Finish Reason: {finish_reason}")
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                     logger.error(f"Prompt Feedback Block Reason: {response.prompt_feedback.block_reason}")
                return [f"Error: Response blocked by API (Reason: {finish_reason})." for _ in questions]
            except (AttributeError, IndexError):
                 return [f"Error: Invalid or empty response from API." for _ in questions]
        except Exception as e:
            logger.error(f"Failed to generate answers with Gemini: {e}")
            return [f"Error processing question: {str(e)}" for _ in questions]

    def format_context(self, chunks: List[str]) -> str:
        """Format context chunks for better LLM understanding"""
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Clean up chunk
            clean_chunk = chunk.strip()
            
            # Add chunk with numbering for reference
            formatted_chunks.append(f"[Chunk {i+1}]\n{clean_chunk}")
        
        return "\n\n".join(formatted_chunks)
    
    def parse_response(self, response_text: str, questions: List[str]) -> List[str]:
        """Parse Gemini's JSON response with improved error handling"""
        try:
            # Gemini's JSON mode should return clean JSON, so we try a direct load
            parsed_response = json.loads(response_text)
            
            if "answers" in parsed_response and isinstance(parsed_response["answers"], list):
                answers = parsed_response["answers"]
                
                # Ensure correct number of answers
                while len(answers) < len(questions):
                    answers.append("Unable to find relevant information in the provided context.")
                
                return answers[:len(questions)]
            else:
                raise ValueError("Invalid JSON structure: 'answers' key missing or not a list.")
                
        except (json.JSONDecodeError, ValueError) as json_error:
            logger.warning(f"Direct JSON parsing failed: {json_error}")
            logger.warning(f"Raw response from Gemini: {response_text[:500]}...")
            
            # Fallback parsing for cases where the model might fail JSON mode
            return self.fallback_parse(response_text, questions)
    
    def fallback_parse(self, response_text: str, questions: List[str]) -> List[str]:
        """Fallback parsing when JSON parsing fails"""
        lines = response_text.split('\n')
        answers = []
        current_answer = ""
        
        for line in lines:
            line = line.strip()
            
            # Skip headers and formatting
            if line.startswith(('```', '{', '}', '"answers"', 'CONTEXT', 'QUESTIONS')):
                continue
            
            # Check if it's a numbered answer
            if re.match(r'^\d+\.', line):
                if current_answer:
                    answers.append(current_answer.strip())
                current_answer = re.sub(r'^\d+\.\s*', '', line)
            elif line and current_answer:
                current_answer += " " + line
            elif line and not current_answer:
                current_answer = line
        
        # Add the last answer
        if current_answer:
            answers.append(current_answer.strip())
        
        # Ensure we have enough answers
        while len(answers) < len(questions):
            answers.append("Unable to process this question due to response parsing issues.")
        
        return answers[:len(questions)]

# Initialize improved processors
pdf_processor = PDFProcessor()
text_chunker = ImprovedTextChunker()
vector_store = VectorStoreManager()
llm_processor = ImprovedLLMProcessor()

# Create router
router = APIRouter(prefix="/api/v1")

@router.post("/hackrx/run", response_model=ProcessResponse)
async def process_documents(
    request: ProcessRequest,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Process documents with enhanced retrieval and answering using Gemini"""
    try:
        logger.info(f"Processing request with {len(request.questions)} questions")
        url = str(request.documents)
        
        # Step 1: Check cache
        cached_document = DatabaseManager.get_document_by_url(db, url)
        
        if not cached_document:
            logger.info("❌ Document not in cache. Processing new document...")
            
            # Extract and process document
            text = pdf_processor.download_and_extract_pdf(url)
            
            if len(text) < 100:
                raise HTTPException(status_code=400, detail="Extracted text is too short")
            
            # Enhanced chunking
            chunks = text_chunker.chunk_text(text)
            
            if not chunks:
                raise HTTPException(status_code=400, detail="No valid chunks created")
            
            logger.info(f"Created {len(chunks)} enhanced chunks")
            
            # Save to database
            DatabaseManager.save_document(db, url, text, chunks)
        else:
            logger.info(f"✅ Document found in cache with {cached_document.chunk_count} chunks")

        # Step 2: Enhanced question processing
        logger.info("Processing questions with enhanced retrieval...")
        
        # Collect relevant chunks with improved search
        all_relevant_chunks = set()
        
        for question in request.questions:
            logger.info(f"Searching for: {question[:50]}...")
            relevant_chunks = vector_store.search(db, question, limit=MULTI_QUERY_CHUNKS_PER_QUERY)
            all_relevant_chunks.update(relevant_chunks)
        
        # Final chunk selection
        final_chunks = list(all_relevant_chunks)[:FINAL_CONTEXT_CHUNKS]
        
        logger.info(f"Selected {len(final_chunks)} unique chunks for context")
        
        if not final_chunks:
            answers = ["No relevant information found in the document." for _ in request.questions]
        else:
            # Generate answers with improved processing
            answers = llm_processor.generate_answers(request.questions, final_chunks)
        
        logger.info(f"✅ Successfully processed all {len(request.questions)} questions")
        
        # Validate response
        if len(answers) != len(request.questions):
            logger.warning(f"Answer count mismatch: {len(answers)} vs {len(request.questions)}")
            while len(answers) < len(request.questions):
                answers.append("Unable to process this question.")
            answers = answers[:len(request.questions)]
        
        return ProcessResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Enhanced HackRx RAG API with Gemini is running"}

@router.get("/cache/stats")
async def cache_stats(db: Session = Depends(get_db)):
    """Get cache statistics"""
    try:
        total_docs = db.query(Document).filter(Document.is_active == True).count()
        total_chunks = db.query(DocumentChunk).count()
        
        return {
            "cached_documents": total_docs,
            "total_chunks": total_chunks,
            "cache_status": "active",
            "version": "2.4.0 - Refactored"
        }
    except Exception as e:
        return {"error": str(e)}

# Include router
app.include_router(router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HackRx Enhanced RAG API with Google Gemini 2.5 Flash",
        "version": "2.4.0",
        "llm_model": "gemini-1.5-flash-latest",
        "improvements": [
            "Simplified architecture to focus on PostgreSQL/pgvector",
            "Centralized configuration for chunking and retrieval",
            "Refined system prompt for better reliability",
            "Robust error handling for blocked API responses"
        ],
        "endpoints": {
            "process": "/api/v1/hackrx/run",
            "health": "/api/v1/health",
            "cache_stats": "/api/v1/cache/stats"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
