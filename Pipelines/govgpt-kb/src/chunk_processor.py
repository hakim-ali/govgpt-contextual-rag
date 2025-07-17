"""
Knowledge Base chunk processor for token-based chunking and embeddings
Handles chunking, section mapping, contextualization, and embedding generation
"""
import os
import json
import hashlib
import tiktoken
import openai
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, wait_exponential, stop_after_attempt

from models import KBChunkEmbedding
from processors import ProcessedDocument, DocumentSection, DocumentProcessor

# Load environment variables
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path, override=True)

# Token encoder for chunking
encoder = tiktoken.encoding_for_model("text-embedding-3-large")

class EnrichedChunkCache:
    """Handles caching of enriched chunks to avoid repeated LLM calls"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir) / "enriched_chunks_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_document_hash(self, document_name: str, full_text: str) -> str:
        """Get MD5 hash of document content for caching purposes"""
        content = f"{document_name}_{full_text}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get_cache_path(self, document_name: str, full_text: str) -> Path:
        """Get cache file path for enriched chunks"""
        doc_hash = self._get_document_hash(document_name, full_text)
        return self.cache_dir / f"{document_name}_{doc_hash}_enriched.json"
    
    def is_cached(self, document_name: str, full_text: str) -> bool:
        """Check if enriched chunks are already cached"""
        cache_path = self.get_cache_path(document_name, full_text)
        return cache_path.exists()
    
    def save_to_cache(self, document_name: str, full_text: str, enriched_chunks: List[str], chunk_metadata: List[Dict[str, Any]]):
        """Save enriched chunks to cache"""
        cache_path = self.get_cache_path(document_name, full_text)
        
        cache_data = {
            'enriched_chunks': enriched_chunks,
            'chunk_metadata': chunk_metadata,
            'document_name': document_name,
            'total_chunks': len(enriched_chunks),
            'processing_timestamp': __import__('time').time()
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Cached {len(enriched_chunks)} enriched chunks for {document_name}")
    
    def load_from_cache(self, document_name: str, full_text: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Load enriched chunks from cache"""
        cache_path = self.get_cache_path(document_name, full_text)
        
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        print(f"📦 Loaded {len(cache_data['enriched_chunks'])} enriched chunks from cache for {document_name}")
        return cache_data['enriched_chunks'], cache_data['chunk_metadata']

@dataclass
class ChunkWithSectionMapping:
    """Represents a chunk with its mapped section information"""
    chunk_id: str
    text: str
    start_char: int
    end_char: int
    mapped_section: DocumentSection
    document_title: str
    page_numbers: List[int]

class ChunkProcessor:
    """
    Dedicated chunk processor for Knowledge Base system
    Handles token-based chunking, section mapping, contextualization, and embedding generation
    """
    
    def __init__(self):
        # Initialize OpenAI client for embeddings and contextualization
        self.llm_client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        
        # Configuration
        self.embed_model = os.getenv("KB_EMBEDDING_MODEL", "text-embedding-3-large")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "250"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "30"))
        self.max_workers = int(os.getenv("MAX_WORKERS", "4"))
        
        # Initialize caching
        cache_dir = os.getenv("CACHE_DIR", "./cache")
        self.enriched_cache = EnrichedChunkCache(cache_dir)
        
        # Initialize DocumentProcessor for page number calculations
        self.document_processor = DocumentProcessor()
        
        print(f"🧩 ChunkProcessor initialized")
        print(f"🧠 Embedding model: {self.embed_model}")
        print(f"📐 Chunk size: {self.chunk_size}, overlap: {self.chunk_overlap}")
        print(f"⚡ Max workers: {self.max_workers}")
        print(f"💾 Cache directory: {cache_dir}")
    
    def chunk_text_by_tokens(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Chunk text by tokens and return chunks with character positions
        Returns: List of (chunk_text, start_char, end_char)
        """
        token_ids = encoder.encode(text)
        chunks = []
        
        for i in range(0, len(token_ids), self.chunk_size - self.chunk_overlap):
            chunk_token_ids = token_ids[i:i + self.chunk_size]
            chunk_text = encoder.decode(chunk_token_ids)
            
            # Find character positions in original text
            start_char = len(encoder.decode(token_ids[:i])) if i > 0 else 0
            end_char = len(encoder.decode(token_ids[:i + len(chunk_token_ids)]))
            
            chunks.append((chunk_text, start_char, end_char))
        
        return chunks
    
    def map_chunks_to_sections(self, chunks_with_positions: List[Tuple[str, int, int]], 
                              sections: List[DocumentSection], 
                              processed_doc: ProcessedDocument) -> List[ChunkWithSectionMapping]:
        """Map chunks to their corresponding sections with page numbers"""
        chunk_mappings = []
        
        for idx, (chunk_text, start_char, end_char) in enumerate(chunks_with_positions):
            chunk_id = str(idx + 1)  # Simple numeric ID: 1, 2, 3, etc.
            
            # Find best matching section by character position
            best_section = None
            best_overlap = 0
            
            for section in sections:
                # Calculate overlap between chunk and section
                overlap_start = max(start_char, section.start_char)
                overlap_end = min(end_char, section.end_char)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_section = section
            
            # Fallback to first section if no overlap found
            if best_section is None and sections:
                best_section = sections[0]
            
            # Create default section if none exist
            if best_section is None:
                best_section = DocumentSection(
                    title="Unknown Section",
                    content=chunk_text,
                    hierarchy=["Unknown Section"],
                    level=0,
                    start_char=start_char,
                    end_char=end_char,
                    page_numbers=[1]  # Default fallback
                )
            
            # Calculate accurate page numbers using DocumentProcessor
            page_numbers = self.document_processor.calculate_chunk_page_numbers(
                processed_doc, start_char, end_char
            )
            
            chunk_mapping = ChunkWithSectionMapping(
                chunk_id=chunk_id,
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
                mapped_section=best_section,
                document_title=processed_doc.title,
                page_numbers=page_numbers
            )
            
            chunk_mappings.append(chunk_mapping)
        
        return chunk_mappings
    
    @retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(5))
    def contextualize_chunk_with_section(self, section: DocumentSection, chunk: str) -> str:
        """
        Enhanced contextualization using section-based context or full document context.
        Follows exact implementation from preprocess_rag.py
        """
        try:
            if section.title != "Full Document":
                # Section-based context for Docling mode
                prompt = (
                    "Given the Chunk section information below, we want to explain what context the text chunk captures in the document."
                    f"Section Content:\n{section.content}\n\n"
                    f"Here is the text chunk we want to explain:\n{chunk}\n\n"
                    "Answer ONLY with a succinct explanation of the meaning of the chunk in the context of the whole section above. "
                    "for improved search retrieval. "
                    "Answer only with the context and nothing else."
                )
            else:
                # Full document context for PyPDF2 mode
                prompt = (
                    "Given the document information below, we want to explain what context the text chunk captures in the document."
                    f"Document Content (excerpt):\n{section.content[:2000]}...\n\n"
                    f"Here is the text chunk we want to explain:\n{chunk}\n\n"
                    "Answer ONLY with a succinct explanation of the meaning of the chunk in the context of the whole document above. "
                    "for improved search retrieval. "
                    "Answer only with the context and nothing else."
                )
            
            response = self.llm_client.chat.completions.create(
                model=os.getenv("RAG_MODEL", "gpt-3.5-turbo"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"⚠️  Warning: Contextualization failed: {e}")
            return ""
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken for text-embedding-3-large"""
        return len(encoder.encode(text))
    
    def _count_batch_tokens(self, batch_texts: List[str]) -> int:
        """Count total tokens in a batch"""
        return sum(self._count_tokens(text) for text in batch_texts)
    
    def _truncate_text_safely(self, text: str, max_tokens: int = 7992) -> str:
        """Truncate text to fit within token limit with safety margin"""
        tokens = encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate to max_tokens and decode back to text
        truncated_tokens = tokens[:max_tokens]
        truncated_text = encoder.decode(truncated_tokens)
        
        print(f"⚠️  Warning: Text truncated from {len(tokens)} to {len(truncated_tokens)} tokens")
        return truncated_text
    
    def _process_batch_safely(self, batch_texts: List[str]) -> List[List[float]]:
        """Process batch with token validation and fallback strategies"""
        # Constants
        MAX_TOKENS_PER_REQUEST = 8192
        SAFETY_MARGIN = 200
        MAX_SAFE_TOKENS = MAX_TOKENS_PER_REQUEST - SAFETY_MARGIN
        
        # Count tokens in batch
        total_tokens = self._count_batch_tokens(batch_texts)
        
        # If batch is within limits, process normally
        if total_tokens <= MAX_SAFE_TOKENS:
            try:
                response = self.llm_client.embeddings.create(
                    model=self.embed_model,
                    input=batch_texts,
                    encoding_format="float"
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                print(f"⚠️  Batch API call failed: {e}")
                # Fall back to individual processing
                return self._process_texts_individually(batch_texts)
        
        else:
            print(f"⚠️  Batch exceeds token limit ({total_tokens} > {MAX_SAFE_TOKENS}), reducing batch size")
            # Fallback 1: Reduce batch size
            return self._process_with_reduced_batch_size(batch_texts)
    
    def _process_with_reduced_batch_size(self, texts: List[str]) -> List[List[float]]:
        """Process texts with progressively smaller batch sizes"""
        MAX_SAFE_TOKENS = 8192 - 200
        
        # Try different batch sizes
        for batch_size in [16, 8, 4, 2, 1]:
            try:
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_tokens = self._count_batch_tokens(batch_texts)
                    
                    if batch_tokens <= MAX_SAFE_TOKENS:
                        # Process this batch
                        response = self.llm_client.embeddings.create(
                            model=self.embed_model,
                            input=batch_texts,
                            encoding_format="float"
                        )
                        batch_embeddings = [item.embedding for item in response.data]
                        all_embeddings.extend(batch_embeddings)
                    else:
                        # Process individually with truncation
                        for text in batch_texts:
                            individual_embedding = self._process_single_text_with_fallbacks(text)
                            all_embeddings.append(individual_embedding)
                
                print(f"✅ Successfully processed {len(texts)} texts with batch_size={batch_size}")
                return all_embeddings
                
            except Exception as e:
                print(f"⚠️  Batch size {batch_size} failed: {e}")
                continue
        
        # If all batch sizes fail, process individually
        return self._process_texts_individually(texts)
    
    def _process_texts_individually(self, texts: List[str]) -> List[List[float]]:
        """Process each text individually with fallback strategies"""
        print(f"🔧 Processing {len(texts)} texts individually...")
        all_embeddings = []
        
        for i, text in enumerate(texts):
            try:
                embedding = self._process_single_text_with_fallbacks(text)
                all_embeddings.append(embedding)
                
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1}/{len(texts)} texts individually")
            
            except Exception as e:
                print(f"❌ Failed to process text {i+1}: {e}")
                # Create zero embedding as last resort
                zero_embedding = [0.0] * 3072  # text-embedding-3-large dimension
                all_embeddings.append(zero_embedding)
                print(f"⚠️  Created zero embedding for failed text {i+1}")
        
        return all_embeddings
    
    def _process_single_text_with_fallbacks(self, text: str) -> List[float]:
        """Process single text with multiple fallback strategies"""
        try:
            # Check if text needs truncation
            token_count = self._count_tokens(text)
            if token_count > 7992:  # Leave margin for safety
                text = self._truncate_text_safely(text, 7992)
            
            # Try to get embedding
            response = self.llm_client.embeddings.create(
                model=self.embed_model,
                input=[text],
                encoding_format="float"
            )
            return response.data[0].embedding
            
        except Exception as e:
            print(f"⚠️  Single text processing failed: {e}")
            
            # Fallback: Try with more aggressive truncation
            try:
                truncated_text = self._truncate_text_safely(text, 4000)
                response = self.llm_client.embeddings.create(
                    model=self.embed_model,
                    input=[truncated_text],
                    encoding_format="float"
                )
                return response.data[0].embedding
            
            except Exception as e2:
                print(f"❌ Aggressive truncation also failed: {e2}")
                # Return zero embedding as last resort
                return [0.0] * 3072

    @retry(wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(10))
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with token-aware batching and robust fallback strategies"""
        try:
            print(f"🔄 Generating embeddings for {len(texts)} chunks...")
            
            # Configuration
            BATCH_SIZE = 32  # Keep as requested
            MAX_TOKENS_PER_REQUEST = 8192
            SAFETY_MARGIN = 200
            
            all_embeddings = []
            total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
            
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i + BATCH_SIZE]
                batch_num = i // BATCH_SIZE + 1
                
                print(f"🔍 Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
                
                # Process batch with token safety checks
                batch_embeddings = self._process_batch_safely(batch_texts)
                all_embeddings.extend(batch_embeddings)
                
                print(f"📊 Generated embeddings for batch {batch_num}/{total_batches}")
            
            print(f"✅ Generated {len(all_embeddings)} embeddings with token-aware processing")
            return all_embeddings
            
        except Exception as e:
            print(f"❌ Failed to generate embeddings: {e}")
            raise
    
    def process_chunks_with_contextualization(self, chunk_mappings: List[ChunkWithSectionMapping], document_name: str, full_text: str) -> Tuple[List[str], List[str]]:
        """
        Process chunks with contextualization using parallel processing and caching
        Returns: (enriched_texts, contextual_texts)
        """
        # Check if enriched chunks are already cached
        if self.enriched_cache.is_cached(document_name, full_text):
            cached_enriched, cached_metadata = self.enriched_cache.load_from_cache(document_name, full_text)
            
            # Extract contextual texts from cached metadata
            contextual_texts = [meta.get('contextual_text', '') for meta in cached_metadata]
            
            return cached_enriched, contextual_texts
        
        def process_chunk_mapping(chunk_mapping: ChunkWithSectionMapping):
            try:
                # Use mapped section for contextualization
                context = self.contextualize_chunk_with_section(
                    chunk_mapping.mapped_section,
                    chunk_mapping.text
                )
                
                # Combine context with original chunk (following preprocess_rag.py pattern)
                enriched_text = f"{context}\n\n{chunk_mapping.text}" if context else chunk_mapping.text
                
                # Prepare metadata for caching
                metadata = {
                    'chunk_id': chunk_mapping.chunk_id,
                    'document_title': chunk_mapping.document_title,
                    'section_title': chunk_mapping.mapped_section.title,
                    'section_hierarchy': chunk_mapping.mapped_section.hierarchy,
                    'contextual_text': context,
                    'page_numbers': chunk_mapping.page_numbers
                }
                
                return enriched_text, context, metadata
                
            except Exception as e:
                print(f"❌ Error enriching chunk {chunk_mapping.chunk_id}: {e}")
                fallback_metadata = {
                    'chunk_id': chunk_mapping.chunk_id,
                    'document_title': chunk_mapping.document_title,
                    'section_title': chunk_mapping.mapped_section.title,
                    'section_hierarchy': chunk_mapping.mapped_section.hierarchy,
                    'contextual_text': '',
                    'page_numbers': chunk_mapping.page_numbers
                }
                return chunk_mapping.text, "", fallback_metadata
        
        # Process chunks in parallel
        print(f"🚀 Enriching {len(chunk_mappings)} chunks with {self.max_workers} workers...")
        enriched_chunks = []
        contextual_texts = []
        chunk_metadata = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(process_chunk_mapping, chunk_mapping): idx 
                for idx, chunk_mapping in enumerate(chunk_mappings)
            }
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    enriched_text, context, metadata = future.result()
                    enriched_chunks.append((idx, enriched_text))
                    contextual_texts.append((idx, context))
                    chunk_metadata.append((idx, metadata))
                except Exception as e:
                    print(f"   ❌ Error processing chunk {idx+1}: {e}")
                    # Fallback to original text
                    enriched_chunks.append((idx, chunk_mappings[idx].text))
                    contextual_texts.append((idx, ""))
                    fallback_metadata = {
                        'chunk_id': chunk_mappings[idx].chunk_id,
                        'document_title': chunk_mappings[idx].document_title,
                        'section_title': chunk_mappings[idx].mapped_section.title,
                        'section_hierarchy': chunk_mappings[idx].mapped_section.hierarchy,
                        'contextual_text': '',
                        'page_numbers': chunk_mappings[idx].page_numbers
                    }
                    chunk_metadata.append((idx, fallback_metadata))
        
        # Sort results by original order
        enriched_chunks.sort(key=lambda x: x[0])
        contextual_texts.sort(key=lambda x: x[0])
        chunk_metadata.sort(key=lambda x: x[0])
        
        final_enriched = [enriched_text for _, enriched_text in enriched_chunks]
        final_contextual = [context for _, context in contextual_texts]
        final_metadata = [metadata for _, metadata in chunk_metadata]
        
        # Cache the enriched chunks
        self.enriched_cache.save_to_cache(document_name, full_text, final_enriched, final_metadata)
        
        return final_enriched, final_contextual
    
    def create_kb_embeddings(self, 
                           chunk_mappings: List[ChunkWithSectionMapping],
                           embeddings: List[List[float]], 
                           contextual_texts: List[str],
                           enriched_texts: List[str],
                           entity: str, 
                           domain: str, 
                           category: str,
                           document_name: str) -> List[KBChunkEmbedding]:
        """
        Create KBChunkEmbedding objects from processed chunks
        """
        kb_embeddings = []
        
        for i, (chunk_mapping, embedding) in enumerate(zip(chunk_mappings, embeddings)):
            contextual_text = contextual_texts[i] if i < len(contextual_texts) else ""
            enriched_chunk_text = enriched_texts[i] if i < len(enriched_texts) else ""
            
            kb_embedding = KBChunkEmbedding(
                chunk_id=chunk_mapping.chunk_id,
                embedding=embedding,
                chunk_text=chunk_mapping.text,
                contextual_text=contextual_text,
                enriched_chunk_text=enriched_chunk_text,
                document_title=chunk_mapping.document_title,
                section_title=chunk_mapping.mapped_section.title,
                section_hierarchy=chunk_mapping.mapped_section.hierarchy,
                section_level=chunk_mapping.mapped_section.level,
                chunk_start_char=chunk_mapping.start_char,
                chunk_end_char=chunk_mapping.end_char,
                section_start_char=chunk_mapping.mapped_section.start_char,
                section_end_char=chunk_mapping.mapped_section.end_char,
                page_numbers=chunk_mapping.page_numbers,
                embedding_model=self.embed_model,
                entity=entity,
                domain=domain,
                category=category,
                document_name=document_name
            )
            kb_embeddings.append(kb_embedding)
        
        return kb_embeddings
    
    def process_document_to_chunks(self, 
                                  processed_doc: ProcessedDocument,
                                  entity: str, 
                                  domain: str, 
                                  category: str,
                                  document_name: str) -> List[KBChunkEmbedding]:
        """
        Complete processing pipeline from ProcessedDocument to KBChunkEmbedding objects
        
        Args:
            processed_doc: Document processed by DocumentProcessor
            entity: Government entity name
            domain: Functional domain name
            category: Document category name
            document_name: Clean document name for identification
        
        Returns:
            List of KBChunkEmbedding objects ready for database storage
        
        Note: Contextualization is always enabled as this is the basis of our contextual RAG
        """
        print(f"🧩 Processing document to chunks: {processed_doc.title}")
        
        # 1. Chunk text using token-based method
        chunks_with_positions = self.chunk_text_by_tokens(processed_doc.full_text)
        print(f" • Generated {len(chunks_with_positions)} chunks using token-based chunking")
        
        # 2. Map chunks to sections with page numbers
        chunk_mappings = self.map_chunks_to_sections(
            chunks_with_positions, processed_doc.sections, processed_doc
        )
        print(f" • Mapped {len(chunk_mappings)} chunks to {len(processed_doc.sections)} sections")
        
        # 3. Process chunks with mandatory contextualization and caching
        enriched_texts, contextual_texts = self.process_chunks_with_contextualization(
            chunk_mappings, document_name, processed_doc.full_text
        )
        print(f" • Enhanced {len(enriched_texts)} chunks with contextualization")
        
        # 4. Generate embeddings
        embeddings = self.generate_embeddings(enriched_texts)
        print(f" • Generated {len(embeddings)} embeddings")
        
        # 5. Create KBChunkEmbedding objects
        kb_embeddings = self.create_kb_embeddings(
            chunk_mappings, embeddings, contextual_texts, enriched_texts,
            entity, domain, category, document_name
        )
        print(f" • Created {len(kb_embeddings)} KBChunkEmbedding objects")
        
        return kb_embeddings