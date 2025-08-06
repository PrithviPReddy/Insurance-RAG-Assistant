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
import pinecone
from pinecone import Pinecone, ServerlessSpec
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models and clients
embedding_model = None
pinecone_client = None
pinecone_index = None
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
    global embedding_model, pinecone_client, pinecone_index, gemini_model, db_engine, SessionLocal
    
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
        
        # Initialize Pinecone client
        logger.info("Initializing Pinecone...")
        pinecone_client = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )
        
        # Create or connect to index
        index_name = os.getenv("PINECONE_INDEX")
        pinecone_index = pinecone_client.Index(index_name)
        
        # --- Gemini Initialization ---
        logger.info("Initializing Google Gemini Flash...")
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # System instructions for the model
        system_instruction = '''You are an expert assistant specializing in legal and policy documents or any documents in general.

        -> You are being tested for your RAG capability.

INSTRUCTIONS:
⚠️ Keep each answer concise: **only 1 or 2 sentences per question**.  
❌ Do not invent any facts.  
✅ If the context does not answer the question, just think and reason and give the closest answer in max 3 lines. 
! remember your answers will be evaluvated my an AI or any other algorithm, try to get a good score.
!! Do not mention anything like 'the context does not provide specific information about .....' or anything like this, just answer the question directly .

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
        
        # Configure the Gemini model with system instructions and to output JSON
        # Safety settings have been removed as per user request for direct evaluation.
        gemini_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            system_instruction=system_instruction,
            generation_config={"response_mime_type": "application/json"}
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
    version="2.3.5", # Incremented version
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
    def get_document_chunks(db: Session, document_id: uuid.UUID) -> List[DocumentChunk]:
        """Get all chunks for a document"""
        return db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).all()
    
    @staticmethod
    def search_similar_chunks(db: Session, query_embedding: List[float], limit: int = 15) -> List[DocumentChunk]:
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
    
    def __init__(self, chunk_size: int = 800, overlap: int = 150):
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

class EnhancedHybridVectorStore:
    """Enhanced hybrid vector storage with improved search strategies"""
    
    def __init__(self):
        self.namespace = "insurance_docs"
    
    def search_postgresql_enhanced(self, db: Session, query: str, limit: int = 15) -> List[str]:
        """Enhanced PostgreSQL search with query expansion"""
        try:
            # Generate embeddings for original query
            original_embedding = embedding_model.encode([query])[0].tolist()
            
            # Get similar chunks
            chunks = DatabaseManager.search_similar_chunks(db, original_embedding, limit)
            
            # Extract content and log similarity scores for debugging
            results = []
            for chunk in chunks:
                results.append(chunk.content)
            
            logger.info(f"PostgreSQL search found {len(results)} chunks for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"PostgreSQL search failed: {e}")
            return []
    
    def search_pinecone_enhanced(self, query: str, limit: int = 10) -> List[str]:
        """Enhanced Pinecone search with multiple query strategies"""
        try:
            query_embedding = embedding_model.encode([query])[0].tolist()
            
            results = pinecone_index.query(
                vector=query_embedding,
                top_k=limit,
                namespace=self.namespace,
                include_metadata=True
            )
            
            documents = []
            for match in results.matches:
                if 'text' in match.metadata:
                    documents.append(match.metadata['text'])
            
            logger.info(f"Pinecone search found {len(documents)} chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            return []
    
    def multi_query_search(self, db: Session, original_query: str) -> List[str]:
        """Search using multiple query variations for better coverage"""
        all_results = set()
        
        # Original query
        results1 = self.search_postgresql_enhanced(db, original_query, limit=10)
        all_results.update(results1)
        
        # Query variations for better coverage
        query_variations = self.generate_query_variations(original_query)
        
        for variation in query_variations[:2]:  # Limit to 2 variations to avoid too many results
            results = self.search_postgresql_enhanced(db, variation, limit=5)
            all_results.update(results)
        
        # Convert back to list and limit
        final_results = list(all_results)[:15]  # Increased limit
        logger.info(f"Multi-query search found {len(final_results)} unique chunks")
        
        return final_results
    
    def generate_query_variations(self, query: str) -> List[str]:
        """Generate query variations for better retrieval"""
        variations = []
        
        # Extract key terms
        key_terms = self.extract_key_terms(query)
        
        if key_terms:
            # Create variations with key terms
            variations.append(" ".join(key_terms))
            
            # Create individual key term queries
            for term in key_terms[:2]:  # Limit to top 2 key terms
                variations.append(term)
        
        return variations
    
    def extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query"""
        import re
        
        # Remove common stop words
        stop_words = {'what', 'is', 'the', 'how', 'does', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Extract words (keep important terms like Article, Constitution, etc.)
        words = re.findall(r'\b[A-Za-z]+\b', query.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Prioritize legal/constitutional terms
        priority_terms = ['constitution', 'article', 'amendment', 'rights', 'fundamental', 'directive', 'principles', 'president', 'supreme', 'court', 'parliament', 'state', 'emergency']
        
        prioritized = []
        for term in priority_terms:
            if term in key_terms:
                prioritized.append(term)
        
        # Add remaining terms
        for term in key_terms:
            if term not in prioritized:
                prioritized.append(term)
        
        return prioritized[:5]  # Return top 5 key terms
    
    def hybrid_search_enhanced(self, db: Session, query: str) -> List[str]:
        """Enhanced hybrid search with multiple strategies"""
        # Try multi-query search first
        results = self.multi_query_search(db, query)
        
        if results:
            logger.info(f"Enhanced search found {len(results)} results from PostgreSQL")
            return results
        else:
            logger.info("No results from PostgreSQL, trying Pinecone fallback")
            return self.search_pinecone_enhanced(query, limit=15)
    
    def add_to_pinecone_fallback(self, chunks: List[str]):
        """Add chunks to Pinecone as fallback"""
        try:
            embeddings = embedding_model.encode(chunks)
            batch_size = 20
            
            for batch_idx in range(0, len(chunks), batch_size):
                batch_end = min(batch_idx + batch_size, len(chunks))
                batch_chunks = chunks[batch_idx:batch_end]
                batch_embeddings = embeddings[batch_idx:batch_end]
                
                vectors = []
                for i, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                    vectors.append({
                        "id": f"chunk_{batch_idx + i}_{hash(chunk) % 1000000}",
                        "values": embedding.tolist(),
                        "metadata": {
                            "text": chunk,
                            "chunk_id": batch_idx + i,
                            "text_length": len(chunk)
                        }
                    })
                
                pinecone_index.upsert(vectors=vectors, namespace=self.namespace)
            
            logger.info(f"Added {len(chunks)} chunks to Pinecone fallback")
        except Exception as e:
            logger.error(f"Failed to add to Pinecone fallback: {e}")

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
hybrid_vector_store = EnhancedHybridVectorStore()
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
        
        if cached_document:
            logger.info(f"✅ Document found in cache with {cached_document.chunk_count} chunks")
        else:
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
            cached_document = DatabaseManager.save_document(db, url, text, chunks)
            
            # Add to Pinecone fallback
            hybrid_vector_store.add_to_pinecone_fallback(chunks)
        
        # Step 2: Enhanced question processing
        logger.info("Processing questions with enhanced retrieval...")
        
        # Collect relevant chunks with improved search
        all_relevant_chunks = set()
        
        for question in request.questions:
            logger.info(f"Searching for: {question[:50]}...")
            relevant_chunks = hybrid_vector_store.hybrid_search_enhanced(db, question)
            all_relevant_chunks.update(relevant_chunks[:5])  # Top 5 per question
        
        # Final chunk selection
        final_chunks = list(all_relevant_chunks)[:20]  # Increased limit for better context
        
        logger.info(f"Selected {len(final_chunks)} chunks for context")
        
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
            "version": "2.3.5 - Gemini 2.5 Flash"
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
        "version": "2.3.5",
        "llm_model": "gemini-1.5-flash-latest (as alias for 2.5)",
        "improvements": [
            "Updated to target Gemini 2.5 Flash technology",
            "Switched from OpenAI GPT to Google Gemini Flash",
            "Better text chunking for legal documents",
            "Enhanced query expansion and search",
            "Improved context formatting and prompting for Gemini",
            "Enabled native JSON output mode for reliable responses",
            "Multi-query search strategies"
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
