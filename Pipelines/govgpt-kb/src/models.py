"""
Knowledge Base database models and configuration for WoG PostgreSQL + PGVector integration
"""
import os
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
import openai

# Load environment variables
from dotenv import load_dotenv
load_dotenv(override=True)

# Knowledge Base Database configuration
KB_USE_DATABASE = os.getenv("KB_USE_DATABASE", "true").lower() == "true"
KB_DATABASE_URL = os.getenv("KB_DATABASE_URL", "postgresql://mustaqmollah@localhost:5432/WoG")
KB_DB_POOL_SIZE = int(os.getenv("KB_DB_POOL_SIZE", "10"))

# Embedding model dimensions registry (focusing on text-embedding-3-large)
KB_EMBEDDING_DIMENSIONS = {
    "text-embedding-3-large": 3072,
    "text_embedding_3_large": 3072,  # Alternative naming convention
    "text-embedding-3-small": 1536,
    "text_embedding_3_small": 1536,
    "azure_ai/embed-v-4-0": 1536,
    "azure_ai_embed_v_4_0": 1536,
}

@dataclass
class KBChunkEmbedding:
    """Knowledge Base chunk embedding - matches existing ChunkEmbedding structure"""
    chunk_id: str
    embedding: List[float]
    chunk_text: str
    contextual_text: str
    enriched_chunk_text: str
    document_title: str
    section_title: str
    section_hierarchy: List[str]
    section_level: int
    chunk_start_char: int
    chunk_end_char: int
    section_start_char: int
    section_end_char: int
    page_numbers: List[int]
    embedding_model: str
    
    # Knowledge Base specific fields
    entity: str = ""
    domain: str = ""
    category: str = ""
    document_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage"""
        return {
            'chunk_id': self.chunk_id,
            'chunk_text': self.chunk_text,
            'contextual_text': self.contextual_text,
            'enriched_chunk_text': self.enriched_chunk_text,
            'document_title': self.document_title,
            'section_title': self.section_title,
            'section_hierarchy': self.section_hierarchy,
            'section_level': self.section_level,
            'chunk_start_char': self.chunk_start_char,
            'chunk_end_char': self.chunk_end_char,
            'section_start_char': self.section_start_char,
            'section_end_char': self.section_end_char,
            'page_numbers': self.page_numbers,
            'embedding_model': self.embedding_model,
            'entity': self.entity,
            'domain': self.domain,
            'category': self.category,
            'document_name': self.document_name
        }

def get_kb_model_dimension(model_name: str) -> int:
    """Get embedding dimension for a model with fallback to auto-detection"""
    
    # Check known dimensions first
    if model_name in KB_EMBEDDING_DIMENSIONS:
        return KB_EMBEDDING_DIMENSIONS[model_name]
    
    # Fallback to auto-detection
    print(f"Unknown model {model_name}, auto-detecting dimension...")
    return auto_detect_kb_dimension(model_name)

def auto_detect_kb_dimension(model_name: str) -> int:
    """Auto-detect embedding dimension by creating a test embedding"""
    try:
        llm_client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        
        # Create a test embedding to detect dimension
        response = llm_client.embeddings.create(
            model=model_name,
            input=["test"],
            encoding_format="float"
        )
        dimension = len(response.data[0].embedding)
        
        # Add to registry for future use
        KB_EMBEDDING_DIMENSIONS[model_name] = dimension
        print(f"Auto-detected dimension {dimension} for model {model_name}")
        
        return dimension
        
    except Exception as e:
        print(f"Could not detect dimension for {model_name}: {e}")
        # Return default dimension for text-embedding-3-large
        return 3072

def get_create_entities_table_sql() -> str:
    """Create entities table SQL"""
    return """
    CREATE TABLE IF NOT EXISTS entities (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100) UNIQUE NOT NULL,
        display_name VARCHAR(200),
        code VARCHAR(10) UNIQUE,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """

def get_create_domains_table_sql() -> str:
    """Create functional_domains table SQL"""
    return """
    CREATE TABLE IF NOT EXISTS functional_domains (
        id SERIAL PRIMARY KEY,
        name VARCHAR(50) UNIQUE NOT NULL,
        display_name VARCHAR(100),
        folder_name VARCHAR(50),
        created_at TIMESTAMP DEFAULT NOW()
    );
    """

def get_create_categories_table_sql() -> str:
    """Create document_categories table SQL"""
    return """
    CREATE TABLE IF NOT EXISTS document_categories (
        id SERIAL PRIMARY KEY,
        name VARCHAR(50) UNIQUE NOT NULL,
        description TEXT,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """

def get_create_knowledge_base_table_sql(model_name: str) -> str:
    """Create main knowledge_base table SQL with exact ChunkEmbedding schema"""
    dimension = get_kb_model_dimension(model_name)
    
    return f"""
    CREATE TABLE IF NOT EXISTS knowledge_base (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        
        -- Hierarchy fields (direct names instead of foreign keys) - Allow NULL for flexible hierarchy
        entity VARCHAR(100),
        domain VARCHAR(100),
        category VARCHAR(100),
        
        -- Document identification
        document_name VARCHAR(255),
        
        -- Exact ChunkEmbedding fields
        chunk_id VARCHAR(255) NOT NULL,
        embedding vector({dimension}),
        chunk_text TEXT NOT NULL,
        contextual_text TEXT DEFAULT '',
        enriched_chunk_text TEXT DEFAULT '',
        document_title TEXT NOT NULL,
        section_title TEXT DEFAULT '',
        section_hierarchy TEXT[] DEFAULT '{{}}',
        section_level INTEGER DEFAULT 0,
        chunk_start_char INTEGER DEFAULT 0,
        chunk_end_char INTEGER DEFAULT 0,
        section_start_char INTEGER DEFAULT 0,
        section_end_char INTEGER DEFAULT 0,
        page_numbers INTEGER[] DEFAULT '{{}}',
        embedding_model VARCHAR(100) NOT NULL,
        
        -- System metadata
        ingestion_ts TIMESTAMP DEFAULT NOW(),
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        
        -- Composite unique constraint
        UNIQUE (document_name, chunk_id)
    );
    """

def get_create_kb_indexes_sql(model_name: str) -> str:
    """Create indexes for knowledge_base table"""
    dimension = get_kb_model_dimension(model_name)
    
    # For 3072 dimensions, we use cosine similarity with sequential scan (no HNSW)
    if dimension > 2000:
        vector_index_comment = f"""
        -- Vector index skipped for {model_name} ({dimension}D) - using cosine similarity with sequential scan
        -- This is because HNSW index doesn't support >2000 dimensions
        """
    else:
        vector_index_comment = f"""
        CREATE INDEX IF NOT EXISTS knowledge_base_embedding_hnsw_idx 
        ON knowledge_base USING hnsw (embedding vector_cosine_ops);
        """
    
    return f"""
    {vector_index_comment}
    
    CREATE INDEX IF NOT EXISTS knowledge_base_chunk_id_idx 
    ON knowledge_base (chunk_id);
    
    CREATE INDEX IF NOT EXISTS knowledge_base_document_title_idx 
    ON knowledge_base (document_title);
    
    CREATE INDEX IF NOT EXISTS knowledge_base_section_title_idx 
    ON knowledge_base (section_title);
    
    CREATE INDEX IF NOT EXISTS knowledge_base_entity_domain_idx 
    ON knowledge_base (entity, domain);
    
    CREATE INDEX IF NOT EXISTS knowledge_base_category_idx 
    ON knowledge_base (category);
    
    CREATE INDEX IF NOT EXISTS knowledge_base_document_name_idx 
    ON knowledge_base (document_name);
    
    CREATE INDEX IF NOT EXISTS knowledge_base_embedding_model_idx 
    ON knowledge_base (embedding_model);
    """

def get_seed_entities_sql() -> str:
    """Get SQL to seed entities table with real Excel data"""
    return """
    INSERT INTO entities (name, display_name, code) VALUES
    ('General', 'General Government', 'GEN'),
    ('DGE', 'Department of Government Enablement', 'DGE'),
    ('Department of Finance', 'Department of Finance', 'DOF'),
    ('Human Resource Authority', 'Human Resource Authority', 'HRA')
    ON CONFLICT (name) DO NOTHING;
    """

def get_seed_domains_sql() -> str:
    """Get SQL to seed functional_domains table with real Excel data"""
    return """
    INSERT INTO functional_domains (name, display_name, folder_name) VALUES
    ('CX', 'Customer Experience', 'CX'),
    ('Finance', 'Finance', 'Finance'),
    ('HR', 'Human Resources', 'HR'),
    ('Procurement', 'Procurement', 'Procurement'),
    ('IT', 'Information Technology', 'IT')
    ON CONFLICT (name) DO NOTHING;
    """

def get_seed_categories_sql() -> str:
    """Get SQL to seed document_categories table with real Excel data"""
    return """
    INSERT INTO document_categories (name, description) VALUES
    ('Guide', 'Instructional guides and manuals'),
    ('Glossary', 'Glossaries and definitions'),
    ('Templates', 'Document templates and forms'),
    ('Policy', 'Government policies and guidelines'),
    ('Law', 'Legal documents and regulations'),
    ('Framework', 'Frameworks and standards'),
    ('Charter', 'Organizational charters'),
    ('Manual', 'Operational manuals and procedures'),
    ('FAQ', 'Frequently Asked Questions')
    ON CONFLICT (name) DO NOTHING;
    """

def validate_kb_database_config() -> bool:
    """Validate Knowledge Base database configuration"""
    if not KB_USE_DATABASE:
        return False
        
    if not KB_DATABASE_URL:
        print("❌ KB_DATABASE_URL not configured")
        return False
        
    return True

def print_kb_model_info():
    """Print information about supported embedding models for Knowledge Base"""
    print("\\n📊 KNOWLEDGE BASE SUPPORTED EMBEDDING MODELS")
    print("=" * 60)
    for model, dimension in KB_EMBEDDING_DIMENSIONS.items():
        index_type = "cosine similarity (sequential)" if dimension > 2000 else "HNSW"
        print(f"• {model:<30} | {dimension:>4}D | {index_type}")
    print("=" * 60)
    print("Note: For >2000D models, using cosine similarity with sequential scan")