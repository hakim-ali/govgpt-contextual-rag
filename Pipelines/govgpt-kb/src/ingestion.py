"""
Knowledge Base document ingestion and processing with YAML-based configuration
"""
import os
import sys
import json
import yaml
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from models import KBChunkEmbedding, get_kb_model_dimension
from database import get_kb_manager
from processors import DocumentProcessor, ProcessedDocument, DocumentSection
from chunk_processor import ChunkProcessor
from logger import ProcessingLogger

# Load environment variables
from dotenv import load_dotenv
# Load .env from the project root (govgpt-kb directory)
# env_path = Path(__file__).parent.parent / '.env'
load_dotenv(override=True)


@dataclass
class DocumentMapping:
    """Document mapping from YAML configuration"""
    excel_title: str
    entity: str
    domain: str
    category: str
    language: str
    file_path: str
    filename: str
    file_size: Optional[int]


class KBDocumentProcessor:
    """Knowledge Base document processor with YAML-based configuration"""
    
    def __init__(self, version: str = "1.0"):
        self.version = version
        self.kb_base_dir = Path(__file__).parent.parent
        self.config_dir = self.kb_base_dir / "config"
        self.mapping_file = self.config_dir / f"mapping_v{version}.yaml"
        
        # Initialize components
        self.kb_manager = get_kb_manager()
        if not self.kb_manager:
            raise RuntimeError("Knowledge Base manager not available")
        
        self.document_processor = DocumentProcessor()
        self.chunk_processor = ChunkProcessor()
        self.logger = ProcessingLogger(version)
        
        # Configuration
        self.force_populate = os.getenv("FORCE_POPULATE", "false").lower() == "true"
        self.use_cache = os.getenv("USE_CACHE", "true").lower() == "true"
        
        print(f"🔧 Knowledge Base processor initialized")
        print(f"📝 Version: {self.version}")
        print(f"📁 Mapping file: {self.mapping_file}")
        print(f"🔄 Force populate: {self.force_populate}")
        print(f"🗂️  Use cache: {self.use_cache}")
        print(f"🧩 Using ChunkProcessor for all processing operations")
        
        # Clear database if force populate is enabled OR if USE_CACHE is false
        if self.force_populate or not self.use_cache:
            self.clear_database()
    
    def clear_database(self):
        """Clear knowledge_base table when FORCE_POPULATE=true or USE_CACHE=false"""
        try:
            with self.kb_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM knowledge_base")
                    deleted_count = cur.rowcount
                    conn.commit()
                    print(f"🧹 Cleared {deleted_count} records from knowledge_base table")
                    self.logger.log_cache_cleared()
        except Exception as e:
            print(f"❌ Failed to clear database: {e}")
    
    def load_yaml_mapping(self) -> Dict[str, Any]:
        """Load YAML mapping file"""
        try:
            if not self.mapping_file.exists():
                raise FileNotFoundError(f"Mapping file not found: {self.mapping_file}")
            
            print(f"📖 Loading YAML mapping: {self.mapping_file}")
            
            with open(self.mapping_file, 'r', encoding='utf-8') as f:
                mapping_data = yaml.safe_load(f)
            
            print(f"✅ Loaded mapping v{mapping_data.get('version', 'unknown')}")
            return mapping_data
            
        except Exception as e:
            print(f"❌ Failed to load YAML mapping: {e}")
            raise
    
    def get_document_mappings_from_yaml(self) -> List[DocumentMapping]:
        """Get document mappings from YAML file"""
        mapping_data = self.load_yaml_mapping()
        document_mappings = []
        
        for doc_mapping in mapping_data.get('document_mappings', []):
            # Only process documents with file_path (ignore None/null paths)
            if doc_mapping.get('file_path'):
                document_mappings.append(DocumentMapping(
                    excel_title=doc_mapping['excel_title'],
                    entity=doc_mapping['entity'],
                    domain=doc_mapping['domain'],
                    category=doc_mapping['category'],
                    language=doc_mapping['language'],
                    file_path=doc_mapping['file_path'],
                    filename=doc_mapping['filename'],
                    file_size=doc_mapping.get('file_size')
                ))
        
        return document_mappings
    
    
    
    
    
    def ingest_document(self, mapping: DocumentMapping) -> bool:
        """Ingest a single document with hierarchy mapping"""
        try:
            print(f"\n🔄 Processing document: {mapping.excel_title}")
            
            # Check if file exists
            full_path = self.kb_base_dir / mapping.file_path
            if not full_path.exists():
                error_msg = f"File not found: {mapping.file_path}"
                self.logger.log_failed_file(
                    mapping.file_path, mapping.excel_title, error_msg,
                    mapping.entity, mapping.domain, mapping.category
                )
                return False
            
            print(f"📋 Mapped to: {mapping.entity} > {mapping.domain} > {mapping.category}")
            
            # Process document with DocumentProcessor
            processed_doc = self.document_processor.parse_document(str(full_path))
            if not processed_doc:
                error_msg = "Document processing failed"
                self.logger.log_failed_file(
                    mapping.file_path, mapping.excel_title, error_msg,
                    mapping.entity, mapping.domain, mapping.category
                )
                return False
            
            # Log if document was loaded from cache
            from_cache = processed_doc.processing_timestamp != time.time()
            
            # Use ChunkProcessor for all processing operations
            print(f"🧩 Processing document with ChunkProcessor...")
            kb_embeddings = self.chunk_processor.process_document_to_chunks(
                processed_doc=processed_doc,
                entity=mapping.entity,
                domain=mapping.domain,
                category=mapping.category,
                document_name=mapping.filename.replace('.PDF', '').replace('.pdf', '')  # Clean filename
            )
            
            if not kb_embeddings:
                error_msg = "ChunkProcessor failed to generate embeddings"
                self.logger.log_failed_file(
                    mapping.file_path, mapping.excel_title, error_msg,
                    mapping.entity, mapping.domain, mapping.category
                )
                return False
            
            # Store in Knowledge Base database
            success = self.kb_manager.store_kb_embeddings(kb_embeddings)
            
            if success:
                print(f"✅ Successfully processed {mapping.excel_title}")
                print(f"📊 Stored {len(kb_embeddings)} chunks in Knowledge Base")
                self.logger.log_processed_file(mapping.file_path, from_cache)
                return True
            else:
                error_msg = "Failed to store embeddings in database"
                self.logger.log_failed_file(
                    mapping.file_path, mapping.excel_title, error_msg,
                    mapping.entity, mapping.domain, mapping.category
                )
                return False
                
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            self.logger.log_failed_file(
                mapping.file_path, mapping.excel_title, error_msg,
                mapping.entity, mapping.domain, mapping.category
            )
            return False
    
    def ingest_all_documents(self) -> bool:
        """Ingest all documents from YAML mapping"""
        print("🚀 Starting Knowledge Base document ingestion...")
        
        # Get document mappings from YAML
        mappings = self.get_document_mappings_from_yaml()
        total_count = len(mappings)
        
        print(f"📋 Found {total_count} documents to process from YAML mapping")
        
        success_count = 0
        
        for i, mapping in enumerate(mappings, 1):
            print(f"\n[{i}/{total_count}] Processing: {mapping.excel_title}")
            self.logger.print_progress(i, total_count, mapping.file_path)
            
            if self.ingest_document(mapping):
                success_count += 1
            
            # Small delay between documents
            time.sleep(0.5)
        
        # Save processing summary
        stats = self.kb_manager.get_kb_stats()
        additional_info = {}
        if stats:
            additional_info = {
                "knowledge_base_stats": stats,
                "cache_stats": self.document_processor.get_cache_stats()
            }
        
        self.logger.save_processing_summary(total_count, additional_info)
        
        print(f"\n🎯 Ingestion completed: {success_count}/{total_count} documents processed successfully")
        
        # Show final statistics
        if stats:
            print("\n📊 Knowledge Base Statistics:")
            print(f"   Total chunks: {stats.get('total_chunks', 0)}")
            print(f"   Total documents: {stats.get('total_documents', 0)}")
            print(f"   Entities: {stats.get('total_entities', 0)}")
            print(f"   Domains: {stats.get('total_domains', 0)}")
            print(f"   Categories: {stats.get('total_categories', 0)}")
        
        return success_count == total_count

def main():
    """Main function for document ingestion"""
    try:
        processor = KBDocumentProcessor()
        success = processor.ingest_all_documents()
        
        if success:
            print("\n✅ Knowledge Base ingestion completed successfully!")
            return 0
        else:
            print("\n❌ Knowledge Base ingestion completed with errors!")
            return 1
            
    except Exception as e:
        print(f"❌ Fatal error during ingestion: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)