"""
Knowledge Base document processors with integrated caching
"""
import os
import json
import hashlib
import requests
import openai
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import time
from datetime import datetime
import shutil

# Docling imports for remote processing
try:
    from docling_core.types.doc.document import DoclingDocument, TextItem
    from docling_core.types.doc.labels import DocItemLabel
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

# Load environment variables
from dotenv import load_dotenv
# Load .env from the project root (govgpt-kb directory)
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path, override=True)

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
class ProcessedDocument:
    """Represents a processed document with sections and metadata"""
    title: str
    full_text: str
    sections: List[DocumentSection]
    processing_method: str
    processing_timestamp: float
    file_path: str
    file_size: int

class DocumentProcessor:
    """
    Unified document processor with integrated caching support
    Handles both Docling and PyPDF2 processing with intelligent caching
    """
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir or os.getenv("CACHE_DIR", "./cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.use_docling = os.getenv("USE_DOCLING", "false").lower() == "true"
        self.cache_enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
        self.use_cache = os.getenv("USE_CACHE", "true").lower() == "true"
        self.docling_server_url = os.getenv("DOCLING_SERVER_URL", "")
        
        # Initialize OpenAI client for contextualization
        self.llm_client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        
        print(f"🔧 DocumentProcessor initialized")
        print(f"📁 Cache directory: {self.cache_dir}")
        print(f"🚀 Processing method: {'Docling' if self.use_docling else 'PyPDF2'}")
        print(f"🌐 Docling server URL: '{self.docling_server_url}'")
        print(f"💾 Cache enabled: {self.cache_enabled}")
        print(f"🗂️  Use cache: {self.use_cache}")
        
        # Clear cache if USE_CACHE is false
        if not self.use_cache and self.cache_enabled:
            self.clear_cache()
    
    def clear_cache(self):
        """Clear all cached files"""
        try:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                print("🧹 Cache cleared due to USE_CACHE=false")
        except Exception as e:
            print(f"⚠️  Warning: Could not clear cache: {e}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get MD5 hash of file for caching purposes"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_cache_path(self, file_path: str) -> Path:
        """Get cache file path for a given document"""
        file_hash = self._get_file_hash(file_path)
        filename = Path(file_path).stem
        return self.cache_dir / f"{filename}_{file_hash}.json"
    
    def _is_cached(self, file_path: str) -> bool:
        """Check if document is already cached and valid"""
        if not self.cache_enabled:
            return False
        
        cache_path = self._get_cache_path(file_path)
        return cache_path.exists()
    
    def _save_to_cache(self, file_path: str, processed_doc: ProcessedDocument):
        """Save processed document to cache"""
        if not self.cache_enabled:
            return
        
        try:
            cache_path = self._get_cache_path(file_path)
            cache_data = {
                'title': processed_doc.title,
                'full_text': processed_doc.full_text,
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
                    for s in processed_doc.sections
                ],
                'processing_method': processed_doc.processing_method,
                'processing_timestamp': processed_doc.processing_timestamp,
                'file_path': processed_doc.file_path,
                'file_size': processed_doc.file_size
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Cached: {Path(file_path).name}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not cache document: {e}")
    
    def _load_from_cache(self, file_path: str) -> Optional[ProcessedDocument]:
        """Load processed document from cache"""
        if not self.cache_enabled:
            return None
        
        try:
            cache_path = self._get_cache_path(file_path)
            if not cache_path.exists():
                return None
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
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
                for s in cache_data['sections']
            ]
            
            processed_doc = ProcessedDocument(
                title=cache_data['title'],
                full_text=cache_data['full_text'],
                sections=sections,
                processing_method=cache_data['processing_method'],
                processing_timestamp=cache_data['processing_timestamp'],
                file_path=cache_data['file_path'],
                file_size=cache_data['file_size']
            )
            
            print(f"📋 Loaded from cache: {Path(file_path).name}")
            return processed_doc
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load from cache: {e}")
            return None
    
    def parse_document(self, file_path: str, force_reprocess: bool = False) -> Optional[ProcessedDocument]:
        """
        Parse document with caching support
        Returns ProcessedDocument or None if parsing failed
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check cache first (unless USE_CACHE is false or force_reprocess)
        if self.use_cache and not force_reprocess:
            cached_doc = self._load_from_cache(file_path)
            if cached_doc:
                return cached_doc
        
        # Process document
        try:
            print(f"🔄 Processing: {Path(file_path).name}")
            
            if self.use_docling:
                if self.docling_server_url:
                    print(f"🌐 Using remote Docling server: {self.docling_server_url}")
                    processed_doc = self._process_with_docling(file_path)
                else:
                    print(f"🔧 Using local Docling processing")
                    processed_doc = self._process_with_local_docling(file_path)
            else:
                print(f"📄 Using PyPDF2 processing")
                processed_doc = self._process_with_pypdf2(file_path)
            
            # Cache the results
            self._save_to_cache(file_path, processed_doc)
            
            return processed_doc
            
        except Exception as e:
            print(f"❌ Failed to process {Path(file_path).name}: {e}")
            return None
    
    def _process_with_docling(self, file_path: str) -> ProcessedDocument:
        """Process PDF using remote Docling service with correct API format"""
        try:
            print(f"🔍 Processing with Docling: {Path(file_path).name}")
            
            # Use correct API format from preprocess_rag.py _call_remote_docling_api_native
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
                    "files": (file_name, f, "application/pdf")  # Note: "files" not "file"
                }
                
                response = requests.post(
                    self.docling_server_url, 
                    files=files, 
                    data=options,  # Important: options as data parameter
                    timeout=300
                )
                response.raise_for_status()
                
                # Parse response - server returns {"document": {"json_content": <DoclingDocument>}}
                response_data = response.json()
                
                # Extract text content from multiple possible fields
                content = None
                doc_data = response_data.get('document', {})
                
                # print(f"📋 Available fields in document: {list(doc_data.keys())}")
                
                # # Debug: Show content of each field
                # for field in ['json_content', 'text_content', 'md_content', 'html_content']:
                #     if field in doc_data:
                #         value = doc_data[field]
                #         if value:
                #             print(f"✅ {field} has content: {len(value)} chars")
                #         else:
                #             print(f"❌ {field} is None or empty")
                
                # # Priority order: json_content.main_text > text_content > md_content > html_content
                # if 'json_content' in doc_data and doc_data['json_content']:
                #     json_content = doc_data['json_content']
                #     if isinstance(json_content, dict) and 'main_text' in json_content:
                #         content = json_content['main_text']
                #         print(f"✅ Extracted text from json_content.main_text: {len(content)} chars")
                
                # if not content and 'text_content' in doc_data and doc_data['text_content']:
                #     content = doc_data['text_content']
                #     print(f"✅ Extracted text from text_content: {len(content)} chars")
                
                # if not content and 'md_content' in doc_data and doc_data['md_content']:
                #     content = doc_data['md_content']
                #     print(f"✅ Extracted text from md_content: {len(content)} chars")
                
                # if not content and 'html_content' in doc_data and doc_data['html_content']:
                #     content = doc_data['html_content']
                #     print(f"✅ Extracted text from html_content: {len(content)} chars")
                
                # # Last resort: try to get text from json_content using different methods
                # if not content and 'json_content' in doc_data and doc_data['json_content']:
                #     json_content = doc_data['json_content']
                #     if isinstance(json_content, dict):
                #         print(f"🔍 json_content keys: {list(json_content.keys())}")
                        
                #         # Try different text extraction methods
                #         # First try: texts field at root level
                #         if 'texts' in json_content and json_content['texts']:
                #             texts = json_content['texts']
                #             if texts:
                #                 # Combine all text items
                #                 all_text = []
                #                 for text_item in texts:
                #                     if isinstance(text_item, dict) and 'text' in text_item:
                #                         all_text.append(text_item['text'])
                                
                #                 if all_text:
                #                     content = '\n'.join(all_text)
                #                     print(f"✅ Extracted text from json_content.texts: {len(content)} chars")
                        
                #         # Second try: texts field inside body
                #         if not content and 'body' in json_content and 'texts' in json_content['body']:
                #             texts = json_content['body']['texts']
                #             if texts:
                #                 # Combine all text items
                #                 all_text = []
                #                 for text_item in texts:
                #                     if isinstance(text_item, dict) and 'text' in text_item:
                #                         all_text.append(text_item['text'])
                                
                #                 if all_text:
                #                     content = '\n'.join(all_text)
                #                     print(f"✅ Extracted text from json_content.body.texts: {len(content)} chars")
                
                # if not content:
                #     available_fields = list(doc_data.keys()) if doc_data else []
                #     raise Exception(f"No content found in Docling result. Available fields: {available_fields}")
                
                # Use proper DoclingDocument deserialization and section extraction
                doc_title = None
                sections = []
                
                # Try to deserialize DoclingDocument and use advanced section extraction
                if DOCLING_AVAILABLE:
                    try:
                        doc = self._deserialize_docling_document(response_data)
                        print("✅ Successfully deserialized DoclingDocument from remote server")
                        
                        # Use the same extraction methods as RemoteDoclingProcessor
                        doc_title = self._extract_document_title(doc, file_path)
                        full_text = self._extract_full_text(doc)
                        sections = self._extract_sections_with_positions(doc, full_text)
                        
                        # Update content with properly extracted full text
                        content = full_text
                        
                    except Exception as e:
                        print(f"⚠️  DoclingDocument deserialization failed: {e}")
                        print("🔄 Falling back to simple text extraction...")
                
                # Fallback: use simple title extraction from content
                if not doc_title:
                    doc_title = self._extract_title_from_content(content, file_path)
                
                # Fallback to single section if no sections extracted
                if not sections:
                    sections = [
                        DocumentSection(
                            title="Full Document",
                            content=content,
                            hierarchy=["Full Document"],
                            level=0,
                            start_char=0,
                            end_char=len(content),
                            page_numbers=[1]  # Remote Docling doesn't provide detailed page info
                        )
                    ]
                
                return ProcessedDocument(
                    title=doc_title,
                    full_text=content,
                    sections=sections,
                    processing_method="remote_docling",
                    processing_timestamp=time.time(),
                    file_path=file_path,
                    file_size=os.path.getsize(file_path)
                )
                
        except Exception as e:
            raise Exception(f"Docling processing failed: {e}")
    
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
                if isinstance(item, TextItem):
                    text_parts.append(item.text)
                # elif isinstance(item, TableItem):
                #     # Extract table content if needed
                #     pass
                # elif isinstance(item, PictureItem):
                #     # Handle images if needed
                #     pass
        except Exception as e:
            print(f"Warning: Could not extract full text: {e}")
        
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
                            hierarchy=[text_content],
                            level=level,
                            start_char=start_pos,
                            end_char=0,
                            page_numbers=[]  # Will be calculated later
                        )
                        text_buffer = []
                    
                    # Add text to current section
                    text_buffer.append(text_content)
                    char_position = full_text.find(text_content, char_position)
                    if char_position != -1:
                        char_position += len(text_content)
            
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
            
            print(f"📋 Extracted {len(sections)} sections from DoclingDocument")
            return sections
            
        except Exception as e:
            print(f"⚠️  Could not extract sections from DoclingDocument: {e}")
            # Fallback to single section
            return [DocumentSection(
                title="Document Content",
                content=full_text,
                hierarchy=["Document Content"],
                level=0,
                start_char=0,
                end_char=len(full_text),
                page_numbers=[1]
            )]
    
    def _extract_sections_from_docling_document(self, json_content: dict, full_text: str) -> List[DocumentSection]:
        """Extract sections from DoclingDocument structure if available"""
        sections = []
        try:
            # Try to extract sections from DoclingDocument body
            if 'body' in json_content and 'texts' in json_content['body']:
                texts = json_content['body']['texts']
                
                for i, text_item in enumerate(texts):
                    if isinstance(text_item, dict) and 'text' in text_item:
                        text_content = text_item['text']
                        
                        # Try to find title/hierarchy information
                        title = f"Section {i+1}"
                        hierarchy = [title]
                        level = 0
                        
                        # Look for label information
                        if 'label' in text_item:
                            label = text_item['label']
                            if label in ['title', 'section_header', 'subtitle']:
                                title = text_content[:50] + "..." if len(text_content) > 50 else text_content
                                level = 1 if label == 'title' else 2
                        
                        # Calculate character positions in full text
                        start_char = full_text.find(text_content) if text_content in full_text else 0
                        end_char = start_char + len(text_content)
                        
                        sections.append(DocumentSection(
                            title=title,
                            content=text_content,
                            hierarchy=hierarchy,
                            level=level,
                            start_char=start_char,
                            end_char=end_char,
                            page_numbers=[1]  # Remote Docling doesn't provide detailed page info
                        ))
            
            print(f"📋 Extracted {len(sections)} sections from DoclingDocument")
            return sections
            
        except Exception as e:
            print(f"⚠️  Could not extract sections from DoclingDocument: {e}")
            return []
    
    # Page count calculation moved to chunk level
    
    def _process_with_local_docling(self, file_path: str) -> ProcessedDocument:
        """Process PDF using local Docling installation - exact same approach as preprocess_rag.py"""
        try:
            print(f"🔍 Processing with local Docling: {Path(file_path).name}")
            
            # Import docling components exactly as in preprocess_rag.py
            print("🔧 Importing docling components...")
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
            from docling.document_converter import PdfFormatOption
            from docling_core.types.doc.labels import DocItemLabel
            from docling_core.types.doc.document import DoclingDocument, TextItem, TableItem, PictureItem
            print("✅ Docling imports successful")
            
            # Initialize converter with OCR disabled (since Tesseract is not available)
            pipeline_options = PdfPipelineOptions(
                do_ocr=False  # Disable OCR since Tesseract is not installed
            )
            
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            
            # Convert document
            print(f"🔄 Converting document with Docling...")
            conversion_result = converter.convert(file_path)
            doc = conversion_result.document
            
            # Extract document title - exact same method as preprocess_rag.py
            doc_title = self._extract_document_title_docling(doc, file_path)
            
            # Extract full text with character positions - exact same method
            full_text = self._extract_full_text_docling(doc)
            
            # Extract sections with character positions - no page numbers needed for sections
            sections = self._extract_sections_with_positions_docling(doc, full_text)
            
            print(f" • Extracted {len(sections)} sections from document")
            
            return ProcessedDocument(
                title=doc_title,
                full_text=full_text,
                sections=sections,
                processing_method="local_docling",
                processing_timestamp=time.time(),
                file_path=file_path,
                file_size=os.path.getsize(file_path)
            )
            
        except ImportError as e:
            print(f"❌ Local Docling ImportError: {e}")
            print(f"⚠️  Falling back to PyPDF2")
            return self._process_with_pypdf2(file_path)
        except Exception as e:
            print(f"❌ Local Docling processing failed: {e}")
            print(f"⚠️  Falling back to PyPDF2")
            return self._process_with_pypdf2(file_path)
    
    def _extract_document_title_docling(self, doc, file_path: str) -> str:
        """Extract document title from the first text item or use filename - exact same as preprocess_rag.py"""
        try:
            from docling_core.types.doc.document import TextItem
            
            for item, level in doc.iterate_items():
                if isinstance(item, TextItem) and item.text.strip():
                    title = item.text.strip()
                    if len(title) > 100:
                        title = title[:100] + "..."
                    return title
        except Exception as e:
            print(f"Warning: Could not extract title: {e}")
        
        return os.path.splitext(os.path.basename(file_path))[0]
    
    def _extract_full_text_docling(self, doc) -> str:
        """Extract full text from document maintaining character positions - exact same as preprocess_rag.py"""
        from docling_core.types.doc.document import TextItem, TableItem, PictureItem
        
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
    
    # Old page-related methods removed - using PyPDF2 approach instead
    
    def _extract_sections_with_positions_docling(self, doc, full_text: str) -> List[DocumentSection]:
        """Extract sections with their character positions in the full text - exact same as preprocess_rag.py"""
        from docling_core.types.doc.document import TextItem, TableItem, PictureItem
        from docling_core.types.doc.labels import DocItemLabel
        
        # Extract sections for document structure only
        
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
                    
                    # Check if this is a heading - exact same logic as preprocess_rag.py
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
                            page_numbers=[1]  # Default - page numbers calculated at chunk level
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
                    # Handle tables - exact same as preprocess_rag.py
                    try:
                        table_df = item.export_to_dataframe()
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
            # No page number calculation needed for sections
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
                page_numbers=[1]  # Default - page numbers calculated at chunk level
            ))
        
        return sections
    
    def _process_with_pypdf2(self, file_path: str) -> ProcessedDocument:
        """Process PDF using PyPDF2"""
        try:
            print(f"📄 Processing with PyPDF2: {Path(file_path).name}")
            
            import PyPDF2
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                full_text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        full_text += f"\n\n{page_text.strip()}"
                
                full_text = full_text.strip()
                
                if not full_text:
                    raise Exception("No text content extracted from PDF")
                
                # Extract document title
                doc_title = self._extract_title_from_content(full_text, file_path)
                
                # Create single section with entire document
                sections = [
                    DocumentSection(
                        title="Full Document",
                        content=full_text,
                        hierarchy=["Full Document"],
                        level=0,
                        start_char=0,
                        end_char=len(full_text),
                        page_numbers=list(range(1, len(pdf_reader.pages) + 1))
                    )
                ]
                
                return ProcessedDocument(
                    title=doc_title,
                    full_text=full_text,
                    sections=sections,
                    processing_method="pypdf2",
                    processing_timestamp=time.time(),
                    file_path=file_path,
                    file_size=os.path.getsize(file_path)
                )
                
        except ImportError:
            raise ImportError("PyPDF2 is required when USE_DOCLING=false. Install with: pip install PyPDF2")
        except Exception as e:
            raise Exception(f"PyPDF2 processing failed: {e}")
    
    def _extract_title_from_content(self, content: str, file_path: str) -> str:
        """Extract document title from content or use filename as fallback"""
        try:
            # Try to extract title from first few lines
            lines = content.split('\n')
            for line in lines[:10]:  # Check first 10 lines
                line = line.strip()
                if line and len(line) > 10 and len(line) < 200:
                    return line
        except Exception:
            pass
        
        # Fallback to filename
        return os.path.splitext(os.path.basename(file_path))[0]
    
    def _calculate_chunk_page_numbers_with_pypdf2(self, file_path: str, chunk_start: int, chunk_end: int, full_text: str) -> List[int]:
        """Calculate page numbers for a chunk using PyPDF2"""
        try:
            import PyPDF2
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Build page-to-character mapping
                page_boundaries = [0]  # Start with position 0
                current_pos = 0
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    if page_text.strip():
                        # Find this page's text in the full text
                        # This is approximate but should work for most cases
                        current_pos += len(page_text.strip()) + 2  # +2 for \n\n
                        page_boundaries.append(current_pos)
                
                # Calculate which pages the chunk spans
                pages = []
                for i in range(len(page_boundaries) - 1):
                    page_start = page_boundaries[i]
                    page_end = page_boundaries[i + 1]
                    
                    # Check if chunk overlaps with this page
                    if chunk_start < page_end and chunk_end > page_start:
                        pages.append(i + 1)
                
                # Handle last page if chunk extends beyond last boundary
                if chunk_end > page_boundaries[-1] and len(pdf_reader.pages) > 0:
                    pages.append(len(pdf_reader.pages))
                
                return pages if pages else [1]
                
        except Exception as e:
            print(f"⚠️  Warning: Could not calculate page numbers with PyPDF2: {e}")
            return [1]
    
    def calculate_chunk_page_numbers(self, processed_doc: ProcessedDocument, chunk_start: int, chunk_end: int) -> List[int]:
        """Calculate page numbers for a chunk based on character positions"""
        # Use PyPDF2 to get accurate page numbers
        page_numbers = self._calculate_chunk_page_numbers_with_pypdf2(
            processed_doc.file_path, chunk_start, chunk_end, processed_doc.full_text
        )
        
        # Also find the best matching section for contextual RAG
        best_section = None
        best_overlap = 0
        
        for section in processed_doc.sections:
            # Calculate overlap between chunk and section
            overlap_start = max(chunk_start, section.start_char)
            overlap_end = min(chunk_end, section.end_char)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_section = section
        
        # Store section info for contextual RAG (this could be expanded)
        if best_section:
            # Could store section info in chunk metadata for contextual RAG
            pass
        
        return page_numbers
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            if not self.cache_enabled or not self.cache_dir.exists():
                return {"cache_enabled": False, "cached_files": 0}
            
            cache_files = list(self.cache_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                "cache_enabled": True,
                "cached_files": len(cache_files),
                "cache_size_mb": round(total_size / (1024 * 1024), 2),
                "cache_directory": str(self.cache_dir)
            }
            
        except Exception as e:
            return {"cache_enabled": False, "error": str(e)}