import os
import time
import json
import pickle
import hashlib
from pathlib import Path
import requests
import openai
import numpy as np
from rank_bm25 import BM25Okapi
import faiss
import tiktoken
from tenacity import retry, wait_exponential, stop_after_attempt
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import litellm
# Docling imports
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc.document import DoclingDocument, TextItem, TableItem, PictureItem
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
import subprocess

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

llm_client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),             
    base_url=os.getenv("OPENAI_API_BASE")
)

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")

# ─── CONFIG from environment ────────────────────────────────────────────────────
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL")
RAG_MODEL       = os.getenv("RAG_MODEL")
EMBED_MODEL     = os.getenv("EMBED_MODEL")
print(f"Using RAG model: {RAG_MODEL}")
print(f"Using embedding model: {EMBED_MODEL}")
PDF_FOLDER      = os.getenv("PDF_FOLDER")
ARTIFACT_DIR    = os.getenv("ARTIFACT_DIR")
VECTOR_K        = int(os.getenv("VECTOR_K"))
BM25_K          = int(os.getenv("BM25_K"))
RRF_K           = int(os.getenv("RRF_K"))
TOP_K           = int(os.getenv("TOP_K"))
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP"))
MAX_WORKERS     = int(os.getenv("MAX_WORKERS", 4))
print(f"Using {MAX_WORKERS} parallel workers for processing.")
MAX_SUMMARY_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", 1024))
EMBEDDING_URL = os.getenv("EMBEDDING_URL")
CHAT_URL = os.getenv("CHAT_URL")
USE_DOCLING     = os.getenv("USE_DOCLING", "true").lower() == "true"
DOCLING_SERVER_URL = os.getenv("DOCLING_SERVER_URL", "")  # Remote Docling GPU server URL

# Model-specific artifact naming
MODEL_SUFFIX = EMBED_MODEL.replace("/", "_").replace("-", "_")

# Token encoder
encoder = tiktoken.encoding_for_model("text-embedding-3-large")

@dataclass
class DocumentSection:
    """Represents a document section with its metadata"""
    title: str
    content: str
    hierarchy: List[str]
    level: int
    start_char: int
    end_char: int
    page_numbers: List[int]

@dataclass
class ChunkWithSectionMapping:
    """Represents a chunk with its mapped section information"""
    chunk_id: str
    text: str
    start_char: int
    end_char: int
    mapped_section: DocumentSection
    document_title: str

# ─── CACHING UTILITIES ─────────────────────────────────────────────────────────
class DocumentCache:
    """Handles caching of processed documents and sections"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir) / "document_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get MD5 hash of file for caching purposes"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_cache_path(self, file_path: str) -> Path:
        """Get cache file path for a given document and cache type"""
        file_hash = self._get_file_hash(file_path)
        filename = Path(file_path).stem
        return self.cache_dir / f"{filename}_{file_hash}.json"
    
    def is_cached(self, file_path: str) -> bool:
        """Check if document is already cached"""
        cache_path = self.get_cache_path(file_path)
        return cache_path.exists()
    
    def save_to_cache(self, file_path: str, data: Any):
        """Save data to cache"""
        cache_path = self.get_cache_path(file_path)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Cached {Path(file_path).name}")
    
    def load_from_cache(self, file_path: str) -> Any:
        """Load data from cache"""
        cache_path = self.get_cache_path(file_path)
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)

# ─── DOCLING PROCESSOR FOR SECTION DETECTION ──────────────────────────────────
class DoclingProcessor:
    """Handles document processing with Docling for section detection"""
    
    def __init__(self, cache_dir: str):
        from docling.datamodel.pipeline_options import TesseractCliOcrOptions
        os.environ["TESSDATA_PREFIX"] = subprocess.check_output("dpkg -L tesseract-ocr-eng | grep tessdata$", shell=True, text=True).strip()
        
        # Use Tesseract CLI instead of tesserocr binding
        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            ocr_options=TesseractCliOcrOptions(
                force_full_page_ocr=True,
                lang=["eng"]
            )
        )
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        self.cache = DocumentCache(cache_dir)
    
    def extract_text_and_sections(self, file_path: str) -> Tuple[str, List[DocumentSection], str]:
        """
        Extract full text and section information from document
        Returns: (full_text, sections, document_title)
        """
        print(f"🔍 Processing document with Docling: {os.path.basename(file_path)}")
        
        # Check cache first
        if self.cache.is_cached(file_path):
            cached_data = self.cache.load_from_cache(file_path)
            print(f"📋 Loaded cached text and sections for {os.path.basename(file_path)}")
            
            # Reconstruct DocumentSection objects
            sections = [
                DocumentSection(
                    title=s['title'],
                    content=s['content'],
                    hierarchy=s['hierarchy'],
                    level=s['level'],
                    start_char=s['start_char'],
                    end_char=s['end_char'],
                    page_numbers=s['page_numbers']
                )
                for s in cached_data['sections']
            ]
            return cached_data['full_text'], sections, cached_data['document_title']
        
        # Parse document with Docling
        print(f"🔄 Converting document with Docling...")
        conversion_result = self.converter.convert(file_path)
        
        doc = conversion_result.document
        
        # Extract document title
        doc_title = self._extract_document_title(doc, file_path)
        
        # Extract full text with character positions
        full_text = self._extract_full_text(doc)
        
        # Extract sections with character positions
        sections = self._extract_sections_with_positions(doc, full_text)
        
        # Cache the results
        cache_data = {
            'full_text': full_text,
            'sections': [
                {
                    'title': s.title,
                    'content': s.content,
                    'hierarchy': s.hierarchy,
                    'level': s.level,
                    'start_char': s.start_char,
                    'end_char': s.end_char,
                    'page_numbers': s.page_numbers
                }
                for s in sections
            ],
            'document_title': doc_title
        }
        self.cache.save_to_cache(file_path, cache_data)
        
        print(f" • Extracted {len(sections)} sections from document")
        return full_text, sections, doc_title
    
    def _extract_document_title(self, doc: DoclingDocument, file_path: str) -> str:
        """Extract document title from the first text item or use filename"""
        try:
            for item, level in doc.iterate_items():
                if isinstance(item, TextItem) and item.text.strip():
                    title = item.text.strip()
                    if len(title) > 100:
                        title = title[:100] + "..."
                    return title
        except Exception as e:
            print(f"Warning: Could not extract title: {e}")
        
        return os.path.splitext(os.path.basename(file_path))[0]
    
    def _extract_full_text(self, doc: DoclingDocument) -> str:
        """Extract full text from document maintaining character positions"""
        text_parts = []
        
        try:
            for item, level in doc.iterate_items():
                if isinstance(item, TextItem) and item.text.strip():
                    text_parts.append(item.text.strip())
                elif isinstance(item, TableItem):
                    # Convert table to text representation
                    try:
                        table_df = item.export_to_dataframe()
                        table_text = table_df.to_string(index=False)
                        text_parts.append(f"[TABLE]\n{table_text}\n[/TABLE]")
                    except:
                        text_parts.append("[TABLE - content not extractable]")
                elif isinstance(item, PictureItem):
                    # Skip pictures as they don't have text content
                    continue
                else:
                    # Handle other items that may have text
                    try:
                        other_text = item.text.strip()
                        if other_text:
                            text_parts.append(other_text)
                    except:
                        text_parts.append("[OtherItem - content not extractable]")
        except Exception as e:
            print(f"Warning: Error extracting full text: {e}")
            # Fallback: try to get text from texts attribute
            if hasattr(doc, 'texts'):
                text_parts = [text.text for text in doc.texts if text.text.strip()]
        
        return '\n\n'.join(text_parts)
    
    def _extract_sections_with_positions(self, doc: DoclingDocument, full_text: str) -> List[DocumentSection]:
        """Extract sections with their character positions in the full text"""
        sections = []
        current_section = None
        text_buffer = []
        char_position = 0
        
        try:
            for item, level in doc.iterate_items():
                if isinstance(item, TextItem):
                    text_content = item.text.strip()
                    if not text_content:
                        continue
                    
                    # Check if this is a heading
                    # is_heading = (
                    #     hasattr(item, 'label') and 
                    #     item.label in [DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER]
                    # ) or self._is_likely_heading(text_content)

                    is_heading = (
                        hasattr(item, 'label') and 
                        item.label in [DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER]
                    ) 
                    
                    
                    if is_heading:
                        # Save previous section if exists
                        if current_section and text_buffer:
                            section_content = '\n\n'.join(text_buffer)
                            end_pos = char_position
                            
                            current_section.content = section_content
                            current_section.end_char = end_pos
                            sections.append(current_section)
                        
                        # Start new section
                        start_pos = full_text.find(text_content, char_position)
                        if start_pos == -1:
                            start_pos = char_position
                        
                        current_section = DocumentSection(
                            title=text_content,
                            content="",
                            hierarchy=[text_content],  # Simplified hierarchy
                            level=level,
                            start_char=start_pos,
                            end_char=0,  # Will be set when section ends
                            page_numbers=[1]  # Simplified page numbering
                        )
                        text_buffer = [text_content[:100]]
                    else:
                        # Add to current section
                        text_buffer.append(text_content)
                    
                    # Update character position
                    char_position = full_text.find(text_content, char_position)
                    if char_position != -1:
                        char_position += len(text_content) + 2  # +2 for \n\n
                
                elif isinstance(item, TableItem):
                    # Handle tables
                    try:
                        table_df = item.export_to_dataframe()
                        # table_text = f"[TABLE]\n{table_df.to_string(index=False)}\n[/TABLE]"
                        table_text = f"[TABLE]\n{table_df.to_markdown(index=False)}\n[/TABLE]"
                        text_buffer.append(table_text)
                        char_position += len(table_text) + 2
                    except:
                        table_placeholder = "[TABLE - content not extractable]"
                        text_buffer.append(table_placeholder)
                        char_position += len(table_placeholder) + 2

                elif isinstance(item, PictureItem):
                    continue  # Pictures not have text content, skip them
                
                else:
                    try:
                        other_text = item.text.strip()
                        text_buffer.append(other_text)
                        char_position += len(other_text) + 2
                    except:
                        table_placeholder = "[OtherItem - content not extractable]"
                        text_buffer.append(table_placeholder)
                        char_position += len(table_placeholder) + 2
        
        except Exception as e:
            print(f"Warning: Error during section extraction: {e}")
        
        # Add final section
        if current_section and text_buffer:
            section_content = '\n\n'.join(text_buffer)
            current_section.content = section_content
            current_section.end_char = len(full_text)
            sections.append(current_section)
        
        # If no sections found, create default section
        if not sections:
            sections.append(DocumentSection(
                title="Document Content",
                content=full_text,
                hierarchy=["Document Content"],
                level=0,
                start_char=0,
                end_char=len(full_text),
                page_numbers=[1]
            ))
        
        return sections
    
    def _is_likely_heading(self, text: str) -> bool:
        """Simple heuristic to identify headings"""
        text = text.strip()
        if not text:
            return False
        
        # Check for common heading patterns
        heading_patterns = [
            lambda t: t.isupper() and len(t) < 100,
            lambda t: t.startswith(('Chapter', 'Section', 'Part', 'Article')),
            lambda t: len(t.split()) <= 10 and not t.endswith('.'),
            lambda t: any(t.startswith(f"{i}.") for i in range(1, 10)),
        ]
        
        return any(pattern(text) for pattern in heading_patterns)

# ─── ENHANCED REMOTE DOCLING PROCESSOR (GPU SERVER) V2 ─────────────────────────
class RemoteDoclingProcessor:
    """Enhanced version using native DoclingDocument objects from remote Docling GPU server"""
    
    def __init__(self, cache_dir: str, server_url: str = None):
        self.server_url = server_url or DOCLING_SERVER_URL
        if not self.server_url:
            raise ValueError("DOCLING_SERVER_URL must be provided for remote processing")
        self.cache = DocumentCache(cache_dir)
        
        # Create a local processor instance to reuse its extraction methods
        self._local_processor = DoclingProcessor(cache_dir)
        print(f"🌐 Using enhanced remote Docling server v2: {self.server_url}")
    
    def extract_text_and_sections(self, file_path: str) -> Tuple[str, List[DocumentSection], str]:
        """
        Extract full text and section information from document using remote server
        Returns: (full_text, sections, document_title)
        """
        print(f"🔍 Processing document with Enhanced Remote Docling v2: {os.path.basename(file_path)}")
        
        # Check cache first
        if self.cache.is_cached(file_path):
            cached_data = self.cache.load_from_cache(file_path)
            print(f"📋 Loaded cached text and sections for {os.path.basename(file_path)}")
            
            # Reconstruct DocumentSection objects
            sections = [
                DocumentSection(
                    title=s['title'],
                    content=s['content'],
                    hierarchy=s['hierarchy'],
                    level=s['level'],
                    start_char=s['start_char'],
                    end_char=s['end_char'],
                    page_numbers=s['page_numbers']
                )
                for s in cached_data['sections']
            ]
            return cached_data['full_text'], sections, cached_data['document_title']
        
        # Process with Remote Docling API - try native format first, fallback to JSON
        print(f"🔄 Converting document with Enhanced Remote Docling API...")
        
        try:
            # Try to get native DoclingDocument object first
            doc = self._call_remote_docling_api_native(file_path)
            print("✅ Successfully obtained native DoclingDocument from remote server")
            
            # Use the same extraction methods as local processor
            doc_title = self._local_processor._extract_document_title(doc, file_path)
            full_text = self._local_processor._extract_full_text(doc)
            sections = self._local_processor._extract_sections_with_positions(doc, full_text)
            
        except Exception as e:
            print(f"⚠️  Native DoclingDocument approach failed: {e}")
            print("🔄 Falling back to JSON parsing approach...")
            
            # Fallback to original JSON parsing approach
            # doc_data = self._call_remote_docling_api_json(file_path)
            # doc_title = self._extract_document_title_from_response(doc_data, file_path)
            # full_text, sections = self._parse_docling_response(doc_data)
        
        # Cache the results
        cache_data = {
            'full_text': full_text,
            'sections': [
                {
                    'title': s.title,
                    'content': s.content,
                    'hierarchy': s.hierarchy,
                    'level': s.level,
                    'start_char': s.start_char,
                    'end_char': s.end_char,
                    'page_numbers': s.page_numbers
                }
                for s in sections
            ],
            'document_title': doc_title
        }
        self.cache.save_to_cache(file_path, cache_data)
        
        print(f" • Extracted {len(sections)} sections from document via enhanced GPU server")
        return full_text, sections, doc_title
    
    def _call_remote_docling_api_native(self, file_path: str) -> DoclingDocument:
        """

        Try to get native DoclingDocument object from remote server
        The server returns DoclingDocument in JSON format under 'json_content'
        """
        options = {
            "from_formats": ["pdf", "docx", "pptx", "html", "image", "asciidoc", "md", "xlsx"],
            "to_formats": ["json"],  # Request JSON format which contains DoclingDocument
            "image_export_mode": "placeholder",
            "ocr": True,
            "force_ocr": False,
            "ocr_engine": "tesseract",
            "ocr_lang": ["en"],
            "pdf_backend": "dlparse_v2",
            "table_mode": "fast",
            "abort_on_error": False,
            "return_as_file": False,
        }
        
        file_name = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            files = {
                "files": (file_name, f, "application/pdf")
            }
            
            try:
                response = requests.post(
                    self.server_url, 
                    files=files, 
                    data=options,
                    timeout=300  # 5 minute timeout for large files
                )
                response.raise_for_status()
                
                # Try to deserialize as DoclingDocument
                response_data = response.json()
                return self._deserialize_docling_document(response_data)
                
            except requests.exceptions.RequestException as e:
                raise Exception(f"Remote Docling native API call failed: {e}")
            except Exception as e:
                raise Exception(f"Failed to deserialize DoclingDocument: {e}")
    
    def _deserialize_docling_document(self, json_data: Dict) -> DoclingDocument:
        """
        Convert JSON response back to DoclingDocument object
        The server returns DoclingDocument structure in 'document.json_content'
        """
        try:
            # The server response format: {"document": {"json_content": <DoclingDocument>}}
            if isinstance(json_data, dict) and 'document' in json_data:
                document = json_data['document']
                if 'json_content' in document and document['json_content']:
                    # This is the actual DoclingDocument structure
                    return DoclingDocument.model_validate(document['json_content'])
            
            # Fallback: try direct validation
            if isinstance(json_data, dict) and 'body' in json_data:
                return DoclingDocument.model_validate(json_data)
            
            # If neither approach works, raise an exception
            raise ValueError("Unable to deserialize response as DoclingDocument")
            
        except Exception as e:
            raise Exception(f"DoclingDocument deserialization failed: {e}")
        
# ─── SIMPLE PDF PROCESSOR (FALLBACK) ───────────────────────────────────────────
class SimplePDFProcessor:
    """Simple PDF processing using PyPDF2 when Docling is disabled"""
    
    def __init__(self, cache_dir: str):
        self.cache = DocumentCache(cache_dir)
    
    def extract_text_and_sections(self, file_path: str) -> Tuple[str, List[DocumentSection], str]:
        """
        Extract full text without section detection
        Returns: (full_text, sections, document_title)
        """
        print(f"🔍 Processing document with PyPDF2: {os.path.basename(file_path)}")
        
        # Check cache first
        if self.cache.is_cached(file_path):
            cached_data = self.cache.load_from_cache(file_path)
            print(f"📋 Loaded cached text and sections for {os.path.basename(file_path)}")
            
            # Reconstruct DocumentSection objects
            sections = [
                DocumentSection(
                    title=s['title'],
                    content=s['content'],
                    hierarchy=s['hierarchy'],
                    level=s['level'],
                    start_char=s['start_char'],
                    end_char=s['end_char'],
                    page_numbers=s['page_numbers']
                )
                for s in cached_data['sections']
            ]
            return cached_data['full_text'], sections, cached_data['document_title']
        
        # Process with PyPDF2
        try:
            import PyPDF2
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                full_text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        full_text += f"\n\n{page_text.strip()}"
                
                full_text = full_text.strip()
        
        except ImportError:
            raise ImportError("PyPDF2 is required when USE_DOCLING=false. Install with: pip install PyPDF2")
        except Exception as e:
            raise Exception(f"Error processing PDF with PyPDF2: {e}")
        
        # Extract document title
        doc_title = os.path.splitext(os.path.basename(file_path))[0]
        
        # Create single section with entire document
        section = DocumentSection(
            title="Full Document",
            content=full_text,
            hierarchy=["Full Document"],
            level=0,
            start_char=0,
            end_char=len(full_text),
            page_numbers=list(range(1, len(pdf_reader.pages) + 1))
        )
        
        sections = [section]
        
        # Cache the results
        cache_data = {
            'full_text': full_text,
            'sections': [
                {
                    'title': s.title,
                    'content': s.content,
                    'hierarchy': s.hierarchy,
                    'level': s.level,
                    'start_char': s.start_char,
                    'end_char': s.end_char,
                    'page_numbers': s.page_numbers
                }
                for s in sections
            ],
            'document_title': doc_title
        }
        self.cache.save_to_cache(file_path, cache_data)
        
        print(f" • Extracted 1 section (full document) from PDF")
        return full_text, sections, doc_title
        
# ─── CHUNK TO SECTION MAPPING ──────────────────────────────────────────────────
def chunk_text_by_tokens(text: str) -> List[Tuple[str, int, int]]:
    """
    Chunk text by tokens (your existing method) and return chunks with character positions
    Returns: List of (chunk_text, start_char, end_char)
    """
    token_ids = encoder.encode(text)
    chunks_with_positions = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    
    for i in range(0, len(token_ids), step):
        slice_ids = token_ids[i:i+CHUNK_SIZE]
        chunk_text = encoder.decode(slice_ids)
        
        # Find character positions in original text
        start_char = text.find(chunk_text.strip())
        if start_char == -1:
            # Fallback: estimate position
            start_char = int((i / len(token_ids)) * len(text))
        
        end_char = start_char + len(chunk_text)
        chunks_with_positions.append((chunk_text, start_char, end_char))
    
    return chunks_with_positions

def map_chunks_to_sections(chunks_with_positions: List[Tuple[str, int, int]], 
                          sections: List[DocumentSection],
                          doc_title: str) -> List[ChunkWithSectionMapping]:
    """
    Map each chunk to its most relevant section based on character overlap
    """
    chunk_mappings = []
    
    for i, (chunk_text, chunk_start, chunk_end) in enumerate(chunks_with_positions):
        # Find the section with maximum overlap
        best_section = None
        max_overlap = 0
        
        for section in sections:
            # Calculate character overlap
            overlap_start = max(chunk_start, section.start_char)
            overlap_end = min(chunk_end, section.end_char)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_section = section
        
        # If no overlap found, use text similarity as fallback
        if best_section is None or max_overlap == 0:
            best_section = _find_section_by_text_similarity(chunk_text, sections)
        
        chunk_mapping = ChunkWithSectionMapping(
            chunk_id=f"chunk_{i}",
            text=chunk_text,
            start_char=chunk_start,
            end_char=chunk_end,
            mapped_section=best_section,
            document_title=doc_title
        )
        
        chunk_mappings.append(chunk_mapping)
    
    return chunk_mappings

def _find_section_by_text_similarity(chunk_text: str, sections: List[DocumentSection]) -> DocumentSection:
    """Fallback method to find section by text similarity"""
    if not sections:
        return DocumentSection(
            title="Unknown Section",
            content=chunk_text,
            hierarchy=["Unknown Section"],
            level=0,
            start_char=0,
            end_char=len(chunk_text),
            page_numbers=[1]
        )
    
    chunk_words = set(chunk_text.lower().split())
    best_section = sections[0]
    best_score = 0
    
    for section in sections:
        section_words = set(section.content.lower().split())
        overlap = len(chunk_words.intersection(section_words))
        score = overlap / max(len(chunk_words), 1)
        
        if score > best_score:
            best_score = score
            best_section = section
    
    return best_section

# ─── CONTEXTUALIZE CHUNK ────────────────────────────────────────────────────

@retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(5))
def contextualize_chunk_with_section(section: DocumentSection, chunk: str) -> str:
    """
    Enhanced contextualization using section-based context or full document context.
    """
    if USE_DOCLING and section.title != "Full Document":
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
    response = llm_client.chat.completions.create(
        model=RAG_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.7
    )
    context = response.choices[0].message.content.strip()
    return context

# ─── UTILITY FUNCTIONS ──────────────────────────────────────────────────────────
@retry(wait=wait_exponential(multiplier=1, min=2, max=60*60), stop=stop_after_attempt(10))
def get_embeddings(
    texts: list[str], model: str = EMBED_MODEL
) -> np.ndarray:
    # Batch requests if using azure_ai/embed-v-4-0, else single call
    embs = []
    if model == "azure_ai/embed-v-4-0":
        batch_size = 96
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            response = llm_client.embeddings.create(
                model=model, input=batch_texts, encoding_format="float"
            )
            embs.extend(item.embedding for item in response.data)
    elif model == "huggingface/Qwen/Qwen3-Embedding-8B":
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            response = llm_client.embeddings.create(
                model=model,
                input=batch_texts,
                encoding_format="float"
            )
            embs.extend(item.embedding for item in response.data)
    else:
        response = llm_client.embeddings.create(
            model=model, input=texts, encoding_format="float"
        )
        embs.extend(item.embedding for item in response.data)
    return embs

# ─── MAIN PIPELINE ──────────────────────────────────────────────────────────────
def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    chunks_path = os.path.join(ARTIFACT_DIR, f"enriched_chunks.json")
    
    # Initialize Docling processor
    if USE_DOCLING:
        if DOCLING_SERVER_URL:
            print("🔧 Using Enhanced Remote Docling GPU server v2 for advanced document processing")
            docling_processor = RemoteDoclingProcessor(ARTIFACT_DIR, DOCLING_SERVER_URL)
        else:
            print("🔧 Using Local Docling processor for advanced document processing")
            docling_processor = DoclingProcessor(ARTIFACT_DIR)
    else:
        print("🔧 Using PyPDF2 processor for simple text extraction")
        docling_processor = SimplePDFProcessor(ARTIFACT_DIR)
    
    # 1. Load or build enriched chunks
    if os.path.exists(chunks_path):
        print(f"✅ Found existing enriched_chunks.json at {chunks_path}, loading...")
        with open(chunks_path, 'r') as fp:
            saved_data = json.load(fp)
            enriched_chunks = saved_data['enriched_chunks']
            chunk_metadata = saved_data['chunk_metadata']
    else:
        enriched_chunks = []
        chunk_metadata = []
        print("▶ Starting preprocessing with section-based context mapping...")

        for pdf in sorted(os.listdir(PDF_FOLDER)):
            if not pdf.lower().endswith('.pdf'): 
                continue
            
            path = os.path.join(PDF_FOLDER, pdf)
            print(f"\n🔄 Processing document: {pdf}")
            
            # Extract text and sections using Docling
            try:
                full_text, sections, doc_title = docling_processor.extract_text_and_sections(path)
            except Exception as e:
                print(f"❌ Error processing {pdf}: {e}")
                continue

            # Chunk text using your existing token-based method
            chunks_with_positions = chunk_text_by_tokens(full_text)
            print(f" • Generated {len(chunks_with_positions)} chunks using token-based chunking")
            
            # Map chunks to sections
            chunk_mappings = map_chunks_to_sections(chunks_with_positions, sections, doc_title)
            print(f" • Mapped {len(chunk_mappings)} chunks to {len(sections)} sections")

            # Enhanced parallel enrichment with section-based context
            def process_chunk_mapping(chunk_mapping: ChunkWithSectionMapping):
                try:
                    # Use mapped section for contextualization
                    context = contextualize_chunk_with_section(
                        chunk_mapping.mapped_section,
                        chunk_mapping.text
                    )
                    
                    # Combine context with original chunk
                    enriched_text = f"{context}\n\n{chunk_mapping.text}"
                    
                    # Prepare metadata
                    metadata = {
                        'chunk_id': chunk_mapping.chunk_id,
                        'document_title': chunk_mapping.document_title,
                        'section_title': chunk_mapping.mapped_section.title,
                        'section_hierarchy': chunk_mapping.mapped_section.hierarchy,
                        'section_level': chunk_mapping.mapped_section.level,
                        'chunk_start_char': chunk_mapping.start_char,
                        'chunk_end_char': chunk_mapping.end_char,
                        'section_start_char': chunk_mapping.mapped_section.start_char,
                        'section_end_char': chunk_mapping.mapped_section.end_char,
                        'original_text': chunk_mapping.text,
                        'contextual_text': context,
                        'page_numbers': chunk_mapping.mapped_section.page_numbers
                    }
                    
                    return enriched_text, metadata
                    
                except Exception as e:
                    print(f"❌ Error enriching chunk {chunk_mapping.chunk_id}:", e)
                    # Fallback to original text
                    metadata = {
                        'chunk_id': chunk_mapping.chunk_id,
                        'document_title': chunk_mapping.document_title,
                        'section_title': chunk_mapping.mapped_section.title,
                        'section_hierarchy': chunk_mapping.mapped_section.hierarchy,
                        'section_level': chunk_mapping.mapped_section.level,
                        'chunk_start_char': chunk_mapping.start_char,
                        'chunk_end_char': chunk_mapping.end_char,
                        'original_text': chunk_mapping.text,
                        'contextual_text': "",
                        'page_numbers': chunk_mapping.mapped_section.page_numbers
                    }
                    return chunk_mapping.text, metadata

            # Process chunks in parallel
            print(f"🚀 Enriching {len(chunk_mappings)} chunks with {MAX_WORKERS} workers...")
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(process_chunk_mapping, chunk_mapping): idx 
                    for idx, chunk_mapping in enumerate(chunk_mappings)
                }
                
                processed_count = 0
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        enriched_text, metadata = future.result()
                        enriched_chunks.append(enriched_text)
                        chunk_metadata.append(metadata)
                        processed_count += 1
                        print(f"   ✅ Chunk {processed_count}/{len(chunk_mappings)} enriched.")
                        
                    except Exception as e:
                        print(f"   ❌ Error processing chunk {idx+1}: {e}")
                    

        # Persist enriched corpus with metadata
        save_data = {
            'enriched_chunks': enriched_chunks,
            'chunk_metadata': chunk_metadata,
            'total_documents': len([f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]),
            'processing_timestamp': time.time(),
            'chunk_size': CHUNK_SIZE,
            'chunk_overlap': CHUNK_OVERLAP
        }
        with open(chunks_path, 'w') as fp:
            json.dump(save_data, fp, indent=2)
        print(f"\n✅ Enrichment complete. Saved {len(enriched_chunks)} chunks to {chunks_path}")

    # Build BM25 index
    bm25_path = os.path.join(ARTIFACT_DIR, f"bm25.pkl")
    if os.path.exists(bm25_path):
        print(f"✅ Found existing BM25 index at {bm25_path}, loading...")     
    else:
        print("\n▶ Building BM25 index...")
        tokenized = [c.split() for c in enriched_chunks]
        bm25 = BM25Okapi(tokenized)
        bm25_path = os.path.join(ARTIFACT_DIR, f'bm25.pkl')
        with open(bm25_path, 'wb') as fp:
            pickle.dump(bm25, fp)

    # Save metadata separately
    # metadata_path = os.path.join(ARTIFACT_DIR, f'chunk_metadata.json')
    # with open(metadata_path, 'w') as fp:
    #     json.dump(chunk_metadata, fp, indent=2)

    # Build FAISS index
    faiss_path = os.path.join(ARTIFACT_DIR, f"faiss_{MODEL_SUFFIX}.idx")
    if os.path.exists(faiss_path):
        print(f"✅ Found existing FAISS index at {faiss_path}, loading...")
    else:
        print("▶ Generating embeddings and building FAISS index...")
        embs = get_embeddings(enriched_chunks, EMBED_MODEL)
        embs = np.array(embs, dtype=np.float32)
        faiss.normalize_L2(embs)
        dim = embs.shape[1]
        idx = faiss.IndexFlatIP(dim)
        idx.add(embs)
        faiss_path = os.path.join(ARTIFACT_DIR, f'faiss_{MODEL_SUFFIX}.idx')
        faiss.write_index(idx, faiss_path)

    # Create summary report
    unique_sections = len(set(meta['section_title'] for meta in chunk_metadata))
    unique_docs = len(set(meta['document_title'] for meta in chunk_metadata))
    
    print("\n" + "="*60)
    print("📊 PROCESSING SUMMARY")
    print("="*60)
    print(f"📁 Artifacts directory: {ARTIFACT_DIR}")
    print(f"📄 Documents processed: {unique_docs}")
    print(f"📝 Total chunks processed: {len(enriched_chunks)}")
    print(f"🔧 Chunk size: {CHUNK_SIZE} tokens")
    print(f"🔄 Chunk overlap: {CHUNK_OVERLAP} tokens")
    print(f"🏷️  Unique sections identified: {unique_sections}")
    print(f"💾 Cache directory: {docling_processor.cache.cache_dir}")
    print(f"📊 BM25 index: {bm25_path}")
    print(f"🔍 FAISS index: {faiss_path}")
    print("="*60)
    print("✅ Section-based contextual RAG preprocessing complete!")

if __name__ == '__main__':
    main()