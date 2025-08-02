import os
import asyncio
import hashlib
import tempfile
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging
import time
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import uvicorn
import fitz  # PyMuPDF

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from groq import AsyncGroq

# Enhanced logging (moved up before API setup)
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

# Reduce httpx logging verbosity
logging.getLogger("httpx").setLevel(logging.WARNING)

# API Authentication token for HackRX - loaded from environment
HACKRX_AUTH_TOKEN = os.getenv("HACKRX_AUTH_TOKEN")
if not HACKRX_AUTH_TOKEN:
    raise ValueError("HACKRX_AUTH_TOKEN environment variable is required")

# API Key rotation setup
API_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"), 
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4")
]

# API rotation state
api_call_counter = 0
current_api_index = 0

def get_next_groq_client() -> AsyncGroq:
    """Get the next Groq client based on rotation strategy"""
    global api_call_counter, current_api_index
    
    # Switch API every 2 calls
    if api_call_counter % 2 == 0 and api_call_counter > 0:
        current_api_index = (current_api_index + 1) % len(API_KEYS)
    
    api_call_counter += 1
    
    return AsyncGroq(
        api_key=API_KEYS[current_api_index],
        max_retries=3,
        timeout=30.0
    )

# Initialize with first client
groq_client = get_next_groq_client()

app = FastAPI(
    title="Advanced RAG System",
    description="State-of-the-art RAG with optimized accuracy and performance",
    version="3.0"
)

# Security scheme for Bearer token authentication
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the Bearer token for authentication"""
    if credentials.credentials != HACKRX_AUTH_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    return credentials.credentials

# Global caches for performance
document_cache: Dict[str, Dict[str, Any]] = {}
answer_cache: Dict[str, str] = {}
embedding_cache: Dict[str, np.ndarray] = {}

# Configuration constants
MAX_CONTEXT_TOKENS = 4000  # Conservative limit
CHUNK_SIZE = 600
CHUNK_OVERLAP = 150
MAX_RETRIEVAL_CANDIDATES = 12
CACHE_EXPIRY_HOURS = 24

# Server configuration from environment variables
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8008"))

class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL of the document to process")
    questions: List[str] = Field(..., description="List of questions to answer")

class QueryResponse(BaseModel):
    answers: List[str] = Field(..., description="List of generated answers")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")

def create_cache_key(text: str) -> str:
    """Create a hash-based cache key"""
    return hashlib.md5(text.encode()).hexdigest()

def is_cache_valid(timestamp: float) -> bool:
    """Check if cache entry is still valid"""
    return time.time() - timestamp < (CACHE_EXPIRY_HOURS * 3600)

def enhanced_document_parsing(doc_url: str) -> str:
    """Advanced document parsing with caching and optimization"""
    cache_key = create_cache_key(doc_url)
    
    # Check cache first
    if cache_key in document_cache and is_cache_valid(document_cache[cache_key]["timestamp"]):
        return document_cache[cache_key]["text"]
    
    try:
        logger.info(f"Fetching document from: {doc_url[:50]}...")
        
        response = requests.get(doc_url, timeout=30)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name
        
        try:
            # Enhanced PDF parsing with better structure preservation
            doc = fitz.open(temp_path)
            text_blocks = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text with layout information
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if "lines" in block:
                        block_text = []
                        for line in block["lines"]:
                            line_text = []
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if len(text) > 1:  # Filter single characters
                                    line_text.append(text)
                            if line_text:
                                block_text.append(" ".join(line_text))
                        
                        if block_text:
                            text_blocks.append(" ".join(block_text))
            
            doc.close()
            
            # Advanced text cleaning and normalization
            full_text = " ".join(text_blocks)
            
            # Normalize whitespace
            full_text = re.sub(r'\s+', ' ', full_text)
            
            # Fix common PDF extraction issues
            full_text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', full_text)  # Add space between camelCase
            full_text = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', full_text)  # Space between numbers and letters
            full_text = re.sub(r'([A-Za-z])(\d+)', r'\1 \2', full_text)  # Space between letters and numbers
            
            # Clean special characters while preserving important punctuation
            full_text = re.sub(r'[^\w\s\.\,\:\;\(\)\-\%\$\'\"]', ' ', full_text)
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            
            # Cache the result
            document_cache[cache_key] = {
                "text": full_text,
                "timestamp": time.time(),
                "length": len(full_text)
            }
            
            logger.info(f"Document parsed successfully ({len(full_text)} chars)")
            return full_text
            
        finally:
            os.unlink(temp_path)
            
    except Exception as e:
        logger.error(f"Error parsing document: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Document parsing failed: {str(e)}")

def intelligent_chunking(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Intelligent chunking with semantic boundary detection"""
    
    # Define semantic boundaries in order of preference
    boundary_patterns = [
        r'\n\d+\.\d+\s+',  # Numbered sections (highest priority)
        r'\n[A-Z][A-Z\s]{2,}:',  # All caps headers
        r'\n(?:ARTICLE|SECTION|CLAUSE|DEFINITION|BENEFIT|EXCLUSION|CONDITION)',  # Key section headers
        r'(?<=\.)\s+(?=[A-Z])',  # Sentence boundaries
        r'(?<=;)\s+',  # Semicolon boundaries
        r'(?<=,)\s+(?=\w+\s+(?:shall|will|must|may|can))',  # Clause boundaries
    ]
    
    chunks = []
    current_chunk = ""
    
    # First, try to split on strong semantic boundaries
    sections = []
    remaining_text = text
    
    for pattern in boundary_patterns[:3]:  # Use only the strongest patterns for initial split
        if re.search(pattern, remaining_text):
            sections = re.split(pattern, remaining_text)
            break
    
    if not sections:
        sections = [text]
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
        
        # If section fits in chunk size, add it
        if len(section) <= chunk_size:
            if section:
                chunks.append(section)
            continue
        
        # For large sections, use sentence-level chunking
        sentences = re.split(r'(?<=[.!?])\s+', section)
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding this sentence would exceed chunk size
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(test_chunk) <= chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Start new chunk with current sentence
                if len(sentence) <= chunk_size:
                    current_chunk = sentence
                else:
                    # Split very long sentences by words
                    words = sentence.split()
                    word_chunk = ""
                    for word in words:
                        test_word_chunk = word_chunk + " " + word if word_chunk else word
                        if len(test_word_chunk) <= chunk_size:
                            word_chunk = test_word_chunk
                        else:
                            if word_chunk:
                                chunks.append(word_chunk)
                            word_chunk = word
                    if word_chunk:
                        current_chunk = word_chunk
        
        # Add any remaining chunk
        if current_chunk:
            chunks.append(current_chunk)
    
    # Add intelligent overlap
    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0 and len(chunk) > 100:
            # Add overlap from previous chunk
            prev_words = chunks[i-1].split()
            overlap_size = min(overlap // 5, len(prev_words))  # 20% of overlap size in words
            if overlap_size > 0:
                overlap_text = " ".join(prev_words[-overlap_size:])
                chunk = overlap_text + " " + chunk
        overlapped_chunks.append(chunk)
    
    # Filter out very short or repetitive chunks
    final_chunks = []
    seen_chunks = set()
    
    for chunk in overlapped_chunks:
        chunk_clean = re.sub(r'\s+', ' ', chunk.strip())
        if len(chunk_clean) > 50:  # Minimum meaningful length
            # Check for near-duplicates
            chunk_signature = create_cache_key(chunk_clean[:100])
            if chunk_signature not in seen_chunks:
                seen_chunks.add(chunk_signature)
                final_chunks.append(chunk_clean)
    
    return final_chunks

class AdvancedVectorStore:
    """Advanced vector store with multiple retrieval strategies"""
    
    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        
        # Multiple specialized vectorizers
        self.primary_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=1,
            max_df=0.9,
            sublinear_tf=True,
            norm='l2'
        )
        
        self.semantic_vectorizer = TfidfVectorizer(
            max_features=6000,
            ngram_range=(2, 4),  # Focus on phrases
            stop_words='english',
            min_df=1,
            max_df=0.8,
            analyzer='word',
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
        )
        
        self.entity_vectorizer = TfidfVectorizer(
            max_features=4000,
            ngram_range=(1, 2),
            stop_words=None,  # Keep entities
            min_df=1,
            lowercase=False,  # Preserve case for entities
            token_pattern=r'\b[A-Z][a-zA-Z0-9]*\b|\b\d+\b|\b\d+%\b|\b\d+\s*(?:days?|months?|years?)\b'
        )
        
        # Create vector matrices
        self.primary_vectors = self.primary_vectorizer.fit_transform(chunks)
        self.semantic_vectors = self.semantic_vectorizer.fit_transform(chunks)
        self.entity_vectors = self.entity_vectorizer.fit_transform(chunks)
        
        # Create BM25 index for keyword search
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        self.bm25_index = BM25Okapi(tokenized_chunks)
    
    def hybrid_search(self, query: str, n_results: int = MAX_RETRIEVAL_CANDIDATES) -> List[Tuple[str, float]]:
        """Advanced hybrid search with multiple scoring strategies"""
        
        # Primary TF-IDF search
        primary_query = self.primary_vectorizer.transform([query])
        primary_scores = cosine_similarity(primary_query, self.primary_vectors)[0]
        
        # Semantic search (phrase-focused)
        semantic_query = self.semantic_vectorizer.transform([query])
        semantic_scores = cosine_similarity(semantic_query, self.semantic_vectors)[0]
        
        # Entity search (numbers, proper nouns)
        entity_query = self.entity_vectorizer.transform([query])
        entity_scores = cosine_similarity(entity_query, self.entity_vectors)[0]
        
        # BM25 keyword search
        tokenized_query = query.lower().split()
        bm25_scores = np.array(self.bm25_index.get_scores(tokenized_query))
        
        # Normalize all scores to [0, 1]
        primary_scores = (primary_scores - primary_scores.min()) / (primary_scores.max() - primary_scores.min() + 1e-8)
        semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-8)
        entity_scores = (entity_scores - entity_scores.min()) / (entity_scores.max() - entity_scores.min() + 1e-8)
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        
        # Intelligent score combination based on query characteristics
        query_lower = query.lower()
        
        # Adjust weights based on query type
        if any(term in query_lower for term in ['what is', 'define', 'meaning', 'definition']):
            # Definition queries - favor semantic and entity matches
            weights = [0.3, 0.4, 0.2, 0.1]
        elif any(term in query_lower for term in ['how much', 'what amount', 'cost', 'price', 'percentage', 'period']):
            # Quantitative queries - favor entity and BM25 matches
            weights = [0.2, 0.2, 0.4, 0.2]
        elif any(term in query_lower for term in ['process', 'procedure', 'how to', 'steps']):
            # Process queries - favor primary and semantic matches
            weights = [0.4, 0.3, 0.1, 0.2]
        else:
            # General queries - balanced approach
            weights = [0.35, 0.3, 0.2, 0.15]
        
        # Combine scores
        combined_scores = (
            weights[0] * primary_scores +
            weights[1] * semantic_scores +
            weights[2] * entity_scores +
            weights[3] * bm25_scores
        )
        
        # Get top results
        top_indices = np.argsort(combined_scores)[::-1][:n_results]
        
        results = []
        for idx in top_indices:
            if combined_scores[idx] > 0.1:  # Minimum relevance threshold
                results.append((self.chunks[idx], combined_scores[idx]))
        
        return results

def extract_key_entities(text: str) -> List[str]:
    """Extract key entities like numbers, percentages, time periods"""
    entities = []
    
    # Numbers and percentages
    entities.extend(re.findall(r'\b\d+(?:\.\d+)?%?\b', text))
    
    # Time periods
    entities.extend(re.findall(r'\b\d+\s*(?:days?|months?|years?|hours?)\b', text, re.IGNORECASE))
    
    # Currency amounts
    entities.extend(re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\b\d+(?:,\d{3})*\s*(?:dollars?|rupees?)\b', text, re.IGNORECASE))
    
    # Proper nouns (likely important terms)
    entities.extend(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))
    
    return list(set(entities))

def intelligent_context_selection(query: str, candidates: List[Tuple[str, float]], max_tokens: int = MAX_CONTEXT_TOKENS) -> List[str]:
    """Intelligently select the best context chunks within token limit"""
    
    # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
    char_limit = max_tokens * 4
    
    selected_chunks = []
    total_chars = 0
    
    # Extract query entities for relevance boosting
    query_entities = set(extract_key_entities(query))
    query_words = set(query.lower().split())
    
    # Sort candidates by enhanced relevance score
    enhanced_candidates = []
    for chunk, score in candidates:
        # Boost score based on entity matches
        chunk_entities = set(extract_key_entities(chunk))
        entity_overlap = len(query_entities.intersection(chunk_entities))
        
        # Boost score based on exact word matches
        chunk_words = set(chunk.lower().split())
        word_overlap = len(query_words.intersection(chunk_words))
        
        # Calculate enhanced score
        enhanced_score = score + (entity_overlap * 0.1) + (word_overlap * 0.05)
        enhanced_candidates.append((chunk, enhanced_score))
    
    # Sort by enhanced score
    enhanced_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Select chunks within token limit
    for chunk, score in enhanced_candidates:
        chunk_chars = len(chunk)
        if total_chars + chunk_chars <= char_limit:
            selected_chunks.append(chunk)
            total_chars += chunk_chars
        else:
            # Try to fit a truncated version if it's highly relevant
            if score > 0.7 and len(selected_chunks) < 2:
                remaining_chars = char_limit - total_chars
                if remaining_chars > 200:  # Minimum meaningful chunk size
                    truncated_chunk = chunk[:remaining_chars-3] + "..."
                    selected_chunks.append(truncated_chunk)
                    break
    
    # Ensure we have at least one chunk
    if not selected_chunks and candidates:
        first_chunk = candidates[0][0]
        if len(first_chunk) > char_limit:
            selected_chunks.append(first_chunk[:char_limit-3] + "...")
        else:
            selected_chunks.append(first_chunk)
    
    return selected_chunks

async def advanced_answer_generation(query: str, context_chunks: List[str]) -> str:
    """Advanced answer generation with optimized prompting"""
    
    # Create optimized context
    context = "\n\n--- RELEVANT SECTION ---\n".join(context_chunks)
    
    # Determine query type for specialized prompting
    query_lower = query.lower()
    
    if any(term in query_lower for term in ['define', 'what is', 'meaning of']):
        prompt_type = "definition"
        instruction = "Provide a clear, precise definition based exactly on the policy text."
    elif any(term in query_lower for term in ['how much', 'what amount', 'cost', 'percentage', 'period', 'time']):
        prompt_type = "quantitative"
        instruction = "Extract and state the exact numbers, amounts, percentages, or time periods mentioned."
    elif any(term in query_lower for term in ['process', 'procedure', 'how to', 'steps', 'requirements']):
        prompt_type = "procedural"
        instruction = "List the specific steps, requirements, or procedures as stated in the policy."
    elif any(term in query_lower for term in ['coverage', 'benefits', 'includes', 'covers']):
        prompt_type = "coverage"
        instruction = "Specify exactly what is covered or included according to the policy."
    elif any(term in query_lower for term in ['exclusions', 'not covered', 'limitations', 'restrictions']):
        prompt_type = "exclusions"
        instruction = "List the specific exclusions, limitations, or restrictions stated in the policy."
    else:
        prompt_type = "general"
        instruction = "Provide a comprehensive answer based on the policy information."
    
    # Create cache key for answer caching
    context_hash = create_cache_key(context + query)
    
    # Check answer cache
    if context_hash in answer_cache:
        return answer_cache[context_hash]
    
    # Optimized prompt based on query type
    prompt = f"""You are an expert insurance policy analyst. {instruction}

CRITICAL RULES:
1. Answer ONLY from the provided context
2. Quote exact numbers, percentages, time periods, and amounts
3. If specific information is not in the context, state "not specified in the provided policy sections"
4. Be precise and factual - no assumptions or interpretations
5. Use the exact terminology from the policy

POLICY CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    try:
        # Get the next client for API rotation
        current_groq_client = get_next_groq_client()
        
        # Use optimized model parameters
        response = await current_groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.0,  
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stream=False
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Post-process the answer
        answer = re.sub(r'\n+', '\n', answer)
        answer = re.sub(r'\s+', ' ', answer)
        answer = answer.strip()
        
        # Cache the answer
        answer_cache[context_hash] = answer
        
        return answer
        
    except asyncio.TimeoutError:
        logger.error(f"Timeout error in answer generation for query: {query[:50]}...")
        return "unable to process"
    except Exception as e:
        error_msg = str(e).lower()
        if any(timeout_indicator in error_msg for timeout_indicator in ['timeout', 'timed out', 'time out']):
            logger.error(f"Timeout-related error in answer generation: {str(e)}")
            return "unable to process"
        else:
            logger.error(f"Error in answer generation: {str(e)}")
            return "unable to process"

@app.post("/hackrx/run", response_model=QueryResponse)
async def advanced_hackrx_run(request: QueryRequest, token: str = Depends(verify_token)) -> QueryResponse:
    """Advanced RAG endpoint with state-of-the-art optimizations"""
    try:
        start_time = time.time()
        
        # Enhanced document processing with caching - PARSE ONCE FOR ALL QUESTIONS
        document_text = enhanced_document_parsing(request.documents)
        if not document_text.strip():
            raise HTTPException(status_code=400, detail="No text content extracted")
        
        # Intelligent chunking - CREATE ONCE FOR ALL QUESTIONS
        chunks = intelligent_chunking(document_text)
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid chunks created")
        
        # Create advanced vector store - CREATE ONCE FOR ALL QUESTIONS
        vector_store = AdvancedVectorStore(chunks)
        
        # Process questions with advanced techniques - REUSE VECTOR STORE FOR ALL QUESTIONS
        async def process_advanced_question(question: str) -> str:
            try:
                # Advanced hybrid search using the shared vector store
                search_results = vector_store.hybrid_search(question, n_results=MAX_RETRIEVAL_CANDIDATES)
                
                # Intelligent context selection
                selected_chunks = intelligent_context_selection(question, search_results)
                
                # Generate answer with advanced prompting (with timeout handling)
                answer = await asyncio.wait_for(
                    advanced_answer_generation(question, selected_chunks),
                    timeout=30.0  # 45 second timeout per question
                )
                
                return answer
                
            except asyncio.TimeoutError:
                logger.error(f"Timeout processing question: {question[:50]}...")
                return "unable to process"
            except Exception as e:
                error_msg = str(e).lower()
                if any(timeout_indicator in error_msg for timeout_indicator in ['timeout', 'timed out', 'time out']):
                    logger.error(f"Timeout-related error processing question '{question[:50]}...': {str(e)}")
                    return "unable to process"
                else:
                    logger.error(f"Error processing question '{question[:50]}...': {str(e)}")
                    return "unable to process"
        
        # Process all questions concurrently using the same vector store with overall timeout
        try:
            tasks = [process_advanced_question(q) for q in request.questions]
            answers = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=120.0  # 2 minute overall timeout
            )
        except asyncio.TimeoutError:
            logger.error("Overall processing timeout exceeded")
            # Return "unable to process" for all questions
            answers = ["unable to process"] * len(request.questions)
        
        # Handle any exceptions in results
        final_answers = []
        for answer in answers:
            if isinstance(answer, Exception):
                error_msg = str(answer).lower()
                if any(timeout_indicator in error_msg for timeout_indicator in ['timeout', 'timed out', 'time out']):
                    final_answers.append("unable to process")
                else:
                    final_answers.append("unable to process")
            else:
                final_answers.append(answer)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Enhanced metadata
        metadata = {
            "total_chunks": len(chunks),
            "processing_time_seconds": processing_time,
            "average_time_per_question": processing_time / len(request.questions),
            "system_type": "advanced_rag",
            "cache_hits": len([k for k in document_cache.keys()]) + len([k for k in answer_cache.keys()]),
            "model_version": "llama-3.1-8b-instant",
            "chunking_strategy": "intelligent_semantic",
            "retrieval_method": "hybrid_multi_vector",
            "accuracy_optimizations": "enabled",
            "api_rotation": {
                "total_api_keys": len(API_KEYS),
                "api_calls_made": len(request.questions),
                "rotation_strategy": "Every 2 calls"
            }
        }
        
        # Validate response format compliance with HackRX requirements
        response_data = QueryResponse(answers=final_answers, metadata=metadata)
        
        # Log format compliance
        logger.info(f"HackRX format compliance: answers={len(final_answers)}, all_strings={all(isinstance(a, str) for a in final_answers)}")
        
        return response_data
        
    except asyncio.TimeoutError:
        logger.error("Request timeout in advanced hackrx_run")
        # Return "unable to process" for all questions
        timeout_answers = ["unable to process"] * len(request.questions)
        return QueryResponse(
            answers=timeout_answers, 
            metadata={"error": "timeout", "system_type": "advanced_rag"}
        )
    except Exception as e:
        error_msg = str(e).lower()
        if any(timeout_indicator in error_msg for timeout_indicator in ['timeout', 'timed out', 'time out']):
            logger.error(f"Timeout-related error in advanced hackrx_run: {str(e)}")
            timeout_answers = ["unable to process"] * len(getattr(request, 'questions', []))
            return QueryResponse(
                answers=timeout_answers, 
                metadata={"error": "timeout", "system_type": "advanced_rag"}
            )
        else:
            logger.error(f"Error in advanced hackrx_run: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/usage")
async def get_api_usage():
    """Get API usage documentation and example"""
    base_url = f"http://{API_HOST}:{API_PORT}"
    return {
        "api_documentation": {
            "base_url": base_url,
            "authentication": {
                "type": "Bearer Token",
                "header": f"Authorization: Bearer {HACKRX_AUTH_TOKEN}"
            },
            "endpoints": {
                "main_endpoint": "/hackrx/run",
                "method": "POST",
                "content_type": "application/json"
            }
        },
        "example_request": {
            "url": f"POST {base_url}/hackrx/run",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {HACKRX_AUTH_TOKEN}"
            },
            "body": {
                "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
                "questions": [
                    "What is the grace period for premium payment?",
                    "What is the waiting period for pre-existing diseases?"
                ]
            }
        },
        "expected_response": {
            "answers": ["Response to question 1", "Response to question 2"],
            "metadata": {"processing_time_seconds": 2.5, "total_chunks": 150}
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    base_url = f"http://{API_HOST}:{API_PORT}"
    return {
        "status": "healthy",
        "message": "Advanced RAG system operational",
        "authentication": "Bearer token required for /hackrx/run endpoint",
        "api_base_url": base_url,
        "cache_stats": {
            "document_cache_size": len(document_cache),
            "answer_cache_size": len(answer_cache),
            "embedding_cache_size": len(embedding_cache)
        },
        "api_rotation": {
            "total_api_keys": len(API_KEYS),
            "current_api_index": current_api_index + 1,
            "total_api_calls": api_call_counter,
            "rotation_strategy": "Every 2 calls"
        },
        "performance_target": "<10s latency",
        "accuracy_target": "Maximum"
    }

@app.get("/cache/clear")
async def clear_cache(token: str = Depends(verify_token)):
    """Clear all caches"""
    global document_cache, answer_cache, embedding_cache
    document_cache.clear()
    answer_cache.clear()
    embedding_cache.clear()
    return {"message": "All caches cleared"}

@app.get("/api/rotation/reset")
async def reset_api_rotation(token: str = Depends(verify_token)):
    """Reset API rotation counter"""
    global api_call_counter, current_api_index
    api_call_counter = 0
    current_api_index = 0
    return {
        "message": "API rotation counter reset",
        "current_api_index": current_api_index + 1,
        "api_call_counter": api_call_counter
    }

@app.get("/api/rotation/status")
async def get_api_rotation_status():
    """Get current API rotation status"""
    return {
        "total_api_keys": len(API_KEYS),
        "current_api_index": current_api_index + 1,
        "total_api_calls": api_call_counter,
        "rotation_strategy": "Every 2 calls",
        "next_rotation_at_call": ((api_call_counter // 2) + 1) * 2
    }

if __name__ == "__main__":
    uvicorn.run(
        "advanced_rag:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info"
    )