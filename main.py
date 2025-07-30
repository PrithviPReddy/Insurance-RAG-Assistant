from fastapi import FastAPI, HTTPException, Depends, status
from fastapi import APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List
import requests
import tempfile
import os
from pathlib import Path
import logging
from contextlib import asynccontextmanager

# pdf processing
from langchain.document_loaders import PyPDFLoader
import tempfile

# vector database and embeddings
import pinecone
from pinecone import Pinecone, ServerlessSpec
import numpy as np

# text processing
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# openAI integration
import openai
from openai import OpenAI

# environment variables
from dotenv import load_dotenv
load_dotenv()

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# global variables for models and clients
embedding_model = None
pinecone_client = None
pinecone_index = None
openai_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and cleanup on shutdown"""
    global embedding_model, pinecone_client, pinecone_index, openai_client
    
    try:
        # initialize embedding model
        logger.info("Loading embedding model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
       
        # initialize Pinecone client
        logger.info("Initializing Pinecone...")
        pinecone_client = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )
        
        # create or connect to index
        index_name = os.getenv("PINECONE_INDEX")
        pinecone_index = pinecone_client.Index(index_name)
        
        # initialize OpenAI client
        logger.info("Initializing OpenAI GPT-4...")
        openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        logger.info("All services initialized successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    finally:
        logger.info("Shutting down services")

# initialize FastAPI app with lifespan
app = FastAPI(
    title="HackRx RAG API",
    description="RAG system for insurance policy document processing",
    version="1.0.0",
    lifespan=lifespan
)

# security
security = HTTPBearer()

# pydantic models
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

class PDFProcessor:
    """Handle PDF download and text extraction using LangChain"""
    
    @staticmethod
    def download_and_extract_pdf(url: str) -> str:
        """Download PDF from URL and extract text using LangChain PyPDFLoader"""
        try:
            # create temporary file for PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                # download PDF
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                if 'application/pdf' not in response.headers.get('content-type', ''):
                    logger.warning(f"Content type is not PDF: {response.headers.get('content-type')}")
                
                # write PDF content to temp file
                temp_file.write(response.content)
                temp_file_path = temp_file.name
            
            try:
                # use LangChain PyPDFLoader to extract text
                loader = PyPDFLoader(temp_file_path)
                pages = loader.load()
                
                # combine all pages
                text = ""
                for i, page in enumerate(pages):
                    text += f"\n--- Page {i + 1} ---\n{page.page_content}"
                
                if not text.strip():
                    raise ValueError("No text could be extracted from the PDF")
                
                return text.strip()
            
            finally:
                # clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Failed to download and extract PDF: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")

class TextChunker:
    """Handle text chunking using LangChain's RecursiveCharacterTextSplitter"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks using LangChain"""
        try:
            chunks = self.text_splitter.split_text(text)
            # filter out very short chunks
            filtered_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
            logger.info(f"Created {len(filtered_chunks)} chunks from text")
            return filtered_chunks
        except Exception as e:
            logger.error(f"Failed to chunk text: {e}")
            return []

class VectorStore:
    """Handle vector storage and retrieval using Pinecone"""
    
    def __init__(self):
        self.namespace = "insurance_docs"
    
    def add_documents(self, chunks: List[str]):
        """Add document chunks to Pinecone vector store"""
        try:
            # generate embeddings
            embeddings = embedding_model.encode(chunks)
            
            # prepare vectors for upsert
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vectors.append({
                    "id": f"chunk_{i}_{hash(chunk) % 1000000}",  # Unique ID
                    "values": embedding.tolist(),
                    "metadata": {
                        "text": chunk,
                        "chunk_id": i,
                        "text_length": len(chunk)
                    }
                })
            
            # upsert vectors to Pinecone
            pinecone_index.upsert(
                vectors=vectors,
                namespace=self.namespace
            )
            
            logger.info(f"Added {len(chunks)} chunks to Pinecone vector store")
        except Exception as e:
            logger.error(f"Failed to add documents to Pinecone: {e}")
            raise
    
    def search(self, query: str, n_results: int = 5) -> List[str]:
        """Search for relevant chunks in Pinecone"""
        try:
            # generate query embedding
            query_embedding = embedding_model.encode([query])[0].tolist()
            
            # search in Pinecone
            results = pinecone_index.query(
                vector=query_embedding,
                top_k=n_results,    
                namespace=self.namespace,
                include_metadata=True
            )
            
            # extract documents from results
            documents = []
            for match in results.matches:
                if 'text' in match.metadata:
                    documents.append(match.metadata['text'])
            
            logger.info(f"Found {len(documents)} relevant chunks for query")
            return documents
        except Exception as e:
            logger.error(f"Failed to search Pinecone: {e}")
            return []
    
    def clear_namespace(self):
        """Clear all vectors in the namespace"""
        try:
            pinecone_index.delete(delete_all=True, namespace=self.namespace)
            logger.info(f"Cleared namespace: {self.namespace}")
        except Exception as e:
            logger.error(f"Failed to clear namespace: {e}")

class LLMProcessor:
    """Handle LLM interactions using OpenAI GPT-4"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.system_prompt = '''You are a helpful insurance assistant. For each user question, answer using ONLY the provided context.
⚠️ Keep each answer concise: **only 1 or 2 sentences per question**.  
❌ Do not invent any facts.  
✅ If the context does not answer the question, just think and reason and give the closest answer in max 3 lines"

Respond in **valid JSON** format like this:
{
  "answers": [
    "Short answer to question 1.",
    "Short answer to question 2."
  ]
}'''
    
    def generate_answers(self, questions: List[str], context_chunks: List[str]) -> List[str]:
        """Generate answers for multiple questions using OpenAI GPT-4"""
        try:
            # prepare context
            context = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])
            
            # prepare questions list
            questions_text = "\n".join([f"{i+1}. {question}" for i, question in enumerate(questions)])
            
            # create user message
            user_message = f"""Context:
{context}

Questions:
{questions_text}"""

            # make API call to OpenAI GPT-4
            response = openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=1000,
                temperature=0,
                top_p=0.8
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.info(f"Generated answers for {len(questions)} questions using OpenAI GPT-4")
            
            # try to extract JSON from response
            try:
                import json
                import re
                
                # extract JSON from markdown code block if present
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # try to find JSON object in response
                    json_match = re.search(r'\{.*?"answers"\s*:\s*\[.*?\].*?\}', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                    else:
                        # if no JSON brackets found, assume the whole response is JSON
                        json_str = response_text
                
                parsed_response = json.loads(json_str)
                
                if "answers" in parsed_response and isinstance(parsed_response["answers"], list):
                    answers = parsed_response["answers"]
                    # ensure we have the right number of answers
                    while len(answers) < len(questions):
                        answers.append("Not mentioned in the document.")
                    return answers[:len(questions)]  # Trim to exact number needed
                else:
                    raise ValueError("Invalid JSON structure")
                    
            except Exception as json_error:
                logger.warning(f"Failed to parse JSON response: {json_error}")
                logger.warning(f"Raw response: {response_text}")
                
                # fallback: try to extract answers without JSON
                lines = response_text.split('\n')
                answers = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith(('Context', 'Questions:', '```', '{', '}', '"answers"')):
                        # clean up the line
                        if line.startswith('"') and line.endswith('",'):
                            line = line[1:-2]
                        elif line.startswith('"') and line.endswith('"'):
                            line = line[1:-1]
                        if line and len(line) > 10:  # reasonable answer length
                            answers.append(line)
                
                # ensure we have enough answers
                while len(answers) < len(questions):
                    answers.append("Not mentioned in the document.")
                
                return answers[:len(questions)]
            
        except Exception as e:
            logger.error(f"Failed to generate answers with OpenAI GPT-4: {e}")
            return [f"I apologize, but I encountered an error while processing your question: {str(e)}" for _ in questions]

# initialize processors
pdf_processor = PDFProcessor()
text_chunker = TextChunker()
vector_store = VectorStore()
llm_processor = LLMProcessor()

# create router AFTER initializing processors
router = APIRouter(prefix="/api/v1")

@router.post("/hackrx/run", response_model=ProcessResponse)
async def process_documents(
    request: ProcessRequest,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Process insurance documents and answer questions using RAG"""
    try:
        logger.info(f"Processing request with {len(request.questions)} questions")
        
        # step 1:download and extract text from PDF
        logger.info(f"Downloading and extracting PDF from: {request.documents}")
        text = pdf_processor.download_and_extract_pdf(str(request.documents))
        
        if len(text) < 100:
            raise HTTPException(status_code=400, detail="Extracted text is too short. PDF may be empty or corrupted.")
        
        # Step 2:chunk the text
        logger.info("Chunking text")
        chunks = text_chunker.chunk_text(text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid chunks could be created from the document")
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 3:store chunks in vector database
        logger.info("Storing chunks in Pinecone vector database")
        vector_store.add_documents(chunks)
        
        # Step 4:process all questions together
        logger.info(f"Processing {len(request.questions)} questions together")
        
        #collect relevant chunks for all questions
        all_relevant_chunks = set()  #use set to avoid duplicates
        question_chunks_map = {}
        
        for question in request.questions:
            relevant_chunks = vector_store.search(question, n_results=3)  # reduced to 3 per question
            question_chunks_map[question] = relevant_chunks
            all_relevant_chunks.update(relevant_chunks)
        
        #convert back to list and limit total chunks
        final_chunks = list(all_relevant_chunks)[:10]  #max 10 chunks total for efficiency
        
        if not final_chunks:
            #if no chunks found, return generic responses
            answers = ["Not mentioned in the document." for _ in request.questions]
        else:
            #generate answers using LLM with batch processing
            answers = llm_processor.generate_answers(request.questions, final_chunks)
        
        logger.info(f"Successfully processed all {len(request.questions)} questions")
        
        #validate answers format
        if len(answers) != len(request.questions):
            logger.warning(f"Answer count mismatch: {len(answers)} answers for {len(request.questions)} questions")
            #pad or trim answers to match questions
            while len(answers) < len(request.questions):
                answers.append("Not mentioned in the document.")
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
    return {"status": "healthy", "message": "HackRx RAG API is running"}

#include router in app
app.include_router(router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HackRx RAG API",
        "version": "1.0.0",
        "endpoints": {
            "process": "/api/v1/hackrx/run",
            "health": "/api/v1/health"
        }
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

#to run the code, type this in the cmd:
#uvicorn main:app --reload --host 0.0.0.0 --port 8000
