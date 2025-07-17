#!/usr/bin/env python3
"""
Database population script from versioned YAML mapping
Processes documents from YAML mapping into Knowledge Base database
"""
import os
import sys
import yaml
import requests
import openai
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models import KBChunkEmbedding
from src.database import get_kb_manager
from src.processors import DocumentProcessor
from src.chunk_processor import ChunkProcessor
from src.logger import ProcessingLogger

# Load environment variables
from dotenv import load_dotenv
load_dotenv('./.env')

@dataclass
class DocumentEntry:
    """Document entry from YAML mapping"""
    excel_title: str
    entity: str
    domain: str
    category: str
    language: str
    file_path: Optional[str]
    filename: Optional[str]
    file_size: Optional[int]
    status: str

class KBDatabasePopulator:
    """Populates Knowledge Base database from YAML mapping"""
    
    def __init__(self, mapping_file: str = None, version: str = "1.0"):
        self.version = version
        self.kb_base_dir = Path(__file__).parent.parent
        self.config_dir = self.kb_base_dir / "config"
        
        # Default mapping file
        if not mapping_file:
            mapping_file = self.config_dir / f"mapping_v{version}.yaml"
        
        self.mapping_file = Path(mapping_file)
        
        # Initialize KB manager
        self.kb_manager = get_kb_manager()
        if not self.kb_manager:
            raise RuntimeError("Knowledge Base manager not available")
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.chunk_processor = ChunkProcessor()
        self.logger = ProcessingLogger(version)
        
        print(f"🔧 Knowledge Base populator initialized")
        print(f"📁 Mapping file: {self.mapping_file}")
        print(f"📝 Version: v{self.version}")
        print(f"🧩 Using ChunkProcessor for chunking and embeddings")
    
    def load_yaml_mapping(self) -> Dict[str, Any]:
        """Load YAML mapping file"""
        try:
            if not self.mapping_file.exists():
                raise FileNotFoundError(f"Mapping file not found: {self.mapping_file}")
            
            print(f"📖 Loading YAML mapping: {self.mapping_file}")
            
            with open(self.mapping_file, 'r', encoding='utf-8') as f:
                mapping_data = yaml.safe_load(f)
            
            print(f"✅ Loaded mapping v{mapping_data.get('version', 'unknown')}")
            print(f"📊 Total documents: {mapping_data.get('total_documents', 0)}")
            print(f"📊 Available documents: {mapping_data.get('available_documents', 0)}")
            
            return mapping_data
            
        except Exception as e:
            print(f"❌ Failed to load YAML mapping: {e}")
            raise
    
    def populate_hierarchy_tables(self, mapping_data: Dict[str, Any]) -> bool:
        """Populate entities, domains, and categories tables"""
        try:
            print("🔄 Populating hierarchy tables...")
            
            # Populate entities
            entities = mapping_data.get('entities', [])
            for entity in entities:
                success = self.kb_manager.ensure_entity(
                    entity['name'], 
                    entity.get('display_name', entity['name']),
                    entity.get('code', entity['name'][:3].upper())
                )
                if success:
                    print(f"✅ Entity: {entity['name']}")
                else:
                    print(f"⚠️  Entity exists: {entity['name']}")
            
            # Populate domains
            domains = mapping_data.get('functional_domains', [])
            for domain in domains:
                success = self.kb_manager.ensure_domain(
                    domain['name'],
                    domain.get('display_name', domain['name']),
                    domain.get('folder_name', domain['name'])
                )
                if success:
                    print(f"✅ Domain: {domain['name']}")
                else:
                    print(f"⚠️  Domain exists: {domain['name']}")
            
            # Populate categories
            categories = mapping_data.get('document_categories', [])
            for category in categories:
                success = self.kb_manager.ensure_category(
                    category['name'],
                    category.get('description', f"{category['name']} documents")
                )
                if success:
                    print(f"✅ Category: {category['name']}")
                else:
                    print(f"⚠️  Category exists: {category['name']}")
            
            print("✅ Hierarchy tables populated successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to populate hierarchy tables: {e}")
            return False
    
    
    def populate_document(self, doc_entry: DocumentEntry) -> bool:
        """Populate a single document from YAML mapping using ChunkProcessor"""
        try:
            if doc_entry.status != 'available' or not doc_entry.file_path:
                print(f"⏭️  Skipping {doc_entry.excel_title} - {doc_entry.status}")
                return True
            
            print(f"\n🔄 Processing: {doc_entry.excel_title}")
            print(f"📁 File: {doc_entry.file_path}")
            
            # Check if file exists
            file_path = self.kb_base_dir / doc_entry.file_path
            if not file_path.exists():
                error_msg = f"File not found: {file_path}"
                print(f"❌ {error_msg}")
                self.logger.log_failed_file(
                    doc_entry.file_path, doc_entry.excel_title, error_msg,
                    doc_entry.entity, doc_entry.domain, doc_entry.category
                )
                return False
            
            print(f"📋 Mapped to: {doc_entry.entity} > {doc_entry.domain} > {doc_entry.category}")
            
            # Process document with DocumentProcessor
            processed_doc = self.document_processor.parse_document(str(file_path))
            if not processed_doc:
                error_msg = "Document processing failed"
                print(f"❌ {error_msg}")
                self.logger.log_failed_file(
                    doc_entry.file_path, doc_entry.excel_title, error_msg,
                    doc_entry.entity, doc_entry.domain, doc_entry.category
                )
                return False
            
            # Log if document was loaded from cache
            from_cache = hasattr(processed_doc, 'processing_timestamp') and processed_doc.processing_timestamp != time.time()
            
            # Extract document name from filename (remove extension)
            document_name = doc_entry.filename.replace('.pdf', '').replace('.PDF', '') if doc_entry.filename else processed_doc.title
            
            # Process document to chunks using ChunkProcessor
            # Note: Contextualization is always enabled as this is the basis of our contextual RAG
            kb_embeddings = self.chunk_processor.process_document_to_chunks(
                processed_doc=processed_doc,
                entity=doc_entry.entity,
                domain=doc_entry.domain, 
                category=doc_entry.category,
                document_name=document_name
            )
            
            # Store in Knowledge Base database
            success = self.kb_manager.store_kb_embeddings(kb_embeddings)
            
            if success:
                print(f"✅ Successfully processed {doc_entry.excel_title}")
                print(f"📊 Stored {len(kb_embeddings)} chunks in Knowledge Base")
                self.logger.log_processed_file(doc_entry.file_path, from_cache)
                return True
            else:
                error_msg = "Failed to store embeddings in database"
                print(f"❌ {error_msg}")
                self.logger.log_failed_file(
                    doc_entry.file_path, doc_entry.excel_title, error_msg,
                    doc_entry.entity, doc_entry.domain, doc_entry.category
                )
                return False
                
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            print(f"❌ {error_msg}")
            self.logger.log_failed_file(
                doc_entry.file_path, doc_entry.excel_title, error_msg,
                doc_entry.entity, doc_entry.domain, doc_entry.category
            )
            return False
    
    def populate_from_yaml(self) -> bool:
        """Populate database from YAML mapping"""
        try:
            print("🚀 Starting Knowledge Base population from YAML mapping...")
            
            # Load YAML mapping
            mapping_data = self.load_yaml_mapping()
            
            # Populate hierarchy tables
            if not self.populate_hierarchy_tables(mapping_data):
                return False
            
            # Process documents
            document_mappings = mapping_data.get('document_mappings', [])
            available_docs = [doc for doc in document_mappings if doc.get('status') == 'available']
            
            print(f"\n📋 Processing {len(available_docs)} available documents...")
            
            success_count = 0
            for i, doc_data in enumerate(available_docs, 1):
                doc_entry = DocumentEntry(
                    excel_title=doc_data['excel_title'],
                    entity=doc_data['entity'],
                    domain=doc_data['domain'],
                    category=doc_data['category'],
                    language=doc_data['language'],
                    file_path=doc_data['file_path'],
                    filename=doc_data['filename'],
                    file_size=doc_data['file_size'],
                    status=doc_data['status']
                )
                
                print(f"\n[{i}/{len(available_docs)}] Processing: {doc_entry.excel_title}")
                
                if self.populate_document(doc_entry):
                    success_count += 1
                
                # Small delay between documents
                time.sleep(1)
            
            print(f"\n🎯 Population completed: {success_count}/{len(available_docs)} documents processed successfully")
            
            # Show final statistics
            stats = self.kb_manager.get_kb_stats()
            if stats:
                print("\n📊 Knowledge Base Statistics:")
                print(f"   Total chunks: {stats['total_stats']['total_chunks']}")
                print(f"   Total documents: {stats['total_stats']['total_documents']}")
                print(f"   Entities: {stats['total_stats']['total_entities']}")
                print(f"   Domains: {stats['total_stats']['total_domains']}")
                print(f"   Categories: {stats['total_stats']['total_categories']}")
            
            # Save processing summary
            additional_info = {}
            if stats:
                additional_info = {
                    "knowledge_base_stats": stats,
                    "cache_stats": self.document_processor.get_cache_stats()
                }
            
            self.logger.save_processing_summary(len(available_docs), additional_info)
            
            return success_count == len(available_docs)
            
        except Exception as e:
            print(f"❌ Database population failed: {e}")
            return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Populate Knowledge Base from YAML mapping')
    parser.add_argument('--mapping', '-m', help='Path to YAML mapping file')
    parser.add_argument('--version', '-v', default='1.0', help='Mapping version (default: 1.0)')
    
    args = parser.parse_args()
    
    print("🚀 Knowledge Base Database Populator")
    print("=" * 50)
    
    try:
        populator = KBDatabasePopulator(args.mapping, args.version)
        success = populator.populate_from_yaml()
        
        if success:
            print("\n✅ Database population completed successfully!")
            return 0
        else:
            print("\n❌ Database population failed!")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)