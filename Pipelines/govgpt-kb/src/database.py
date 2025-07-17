"""
Knowledge Base PostgreSQL + PGVector database operations for WoG system
"""
import os
import json
import numpy as np
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector
from typing import List, Dict, Any, Optional, Tuple
import time
from contextlib import contextmanager
from models import (
    KBChunkEmbedding,
    get_kb_model_dimension,
    get_create_entities_table_sql,
    get_create_domains_table_sql,
    get_create_categories_table_sql,
    get_create_knowledge_base_table_sql,
    get_create_kb_indexes_sql,
    get_seed_entities_sql,
    get_seed_domains_sql,
    get_seed_categories_sql,
    validate_kb_database_config,
    KB_DATABASE_URL,
    KB_DB_POOL_SIZE,
    KB_USE_DATABASE
)

class KnowledgeBaseManager:
    """Manages PostgreSQL + PGVector operations for Knowledge Base system"""
    
    def __init__(self):
        self.pool: Optional[ConnectionPool] = None
        self.database_initialized: bool = False
        
        if validate_kb_database_config():
            self._initialize_connection_pool()
            self._initialize_database()
    
    def _initialize_connection_pool(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.pool = ConnectionPool(
                KB_DATABASE_URL,
                min_size=1,
                max_size=KB_DB_POOL_SIZE,
                check=ConnectionPool.check_connection
            )
            print(f"✅ Knowledge Base PostgreSQL connection pool initialized (max_size={KB_DB_POOL_SIZE})")
            
            # Test connection and enable pgvector
            with self.get_connection() as conn:
                self._enable_pgvector(conn)
                
        except Exception as e:
            print(f"❌ Failed to initialize Knowledge Base database connection: {e}")
            self.pool = None
    
    def _enable_pgvector(self, conn):
        """Enable pgvector extension"""
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(conn)
            print("✅ PGVector extension enabled for Knowledge Base")
        except Exception as e:
            print(f"❌ Failed to enable pgvector for Knowledge Base: {e}")
            raise
    
    def _initialize_database(self):
        """Initialize database schema and seed data"""
        if self.database_initialized:
            return
            
        try:
            with self.get_connection() as conn:
                # Create hierarchy tables
                print("🔄 Creating Knowledge Base hierarchy tables...")
                conn.execute(get_create_entities_table_sql())
                conn.execute(get_create_domains_table_sql())
                conn.execute(get_create_categories_table_sql())
                
                # Create main knowledge_base table (default to text-embedding-3-large)
                embed_model = os.getenv("KB_EMBEDDING_MODEL", "text-embedding-3-large")
                conn.execute(get_create_knowledge_base_table_sql(embed_model))
                
                # Create indexes (with cosine similarity for 3072D)
                conn.execute(get_create_kb_indexes_sql(embed_model))
                
                # Grant PUBLIC access to all tables (no permission restrictions)
                print("🔧 Granting PUBLIC access to all WoG tables...")
                tables = ['entities', 'functional_domains', 'document_categories', 'knowledge_base']
                for table in tables:
                    conn.execute(f'GRANT ALL PRIVILEGES ON TABLE {table} TO PUBLIC')
                
                # Grant PUBLIC access to sequences
                conn.execute('GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO PUBLIC')
                
                # Seed hierarchy data from Excel
                print("🌱 Seeding Knowledge Base with Excel hierarchy data...")
                conn.execute(get_seed_entities_sql())
                conn.execute(get_seed_domains_sql())
                conn.execute(get_seed_categories_sql())
                
                conn.commit()
                self.database_initialized = True
                
                print("✅ Knowledge Base database initialized with hierarchy data")
                
        except Exception as e:
            print(f"❌ Failed to initialize Knowledge Base database: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get database connection from pool"""
        if not self.pool:
            raise RuntimeError("Knowledge Base database pool not initialized")
        
        with self.pool.connection() as conn:
            register_vector(conn)
            yield conn
    
    def store_kb_embeddings(self, embeddings: List[KBChunkEmbedding]) -> bool:
        """Store Knowledge Base chunk embeddings in the database"""
        if not embeddings:
            return True
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Prepare batch insert
                    insert_sql = """
                    INSERT INTO knowledge_base (
                        entity, domain, category, document_name,
                        chunk_id, embedding, chunk_text, contextual_text, enriched_chunk_text,
                        document_title, section_title, section_hierarchy,
                        section_level, chunk_start_char, chunk_end_char,
                        section_start_char, section_end_char, page_numbers,
                        embedding_model
                    ) VALUES %s
                    ON CONFLICT (document_name, chunk_id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        chunk_text = EXCLUDED.chunk_text,
                        contextual_text = EXCLUDED.contextual_text,
                        enriched_chunk_text = EXCLUDED.enriched_chunk_text,
                        entity = EXCLUDED.entity,
                        domain = EXCLUDED.domain,
                        category = EXCLUDED.category,
                        document_name = EXCLUDED.document_name,
                        updated_at = NOW()
                    """
                    
                    # Prepare data for batch insert
                    data = []
                    for emb in embeddings:
                        data.append((
                            emb.entity,
                            emb.domain,
                            emb.category,
                            emb.document_name,
                            emb.chunk_id,
                            np.array(emb.embedding),
                            emb.chunk_text,
                            emb.contextual_text,
                            emb.enriched_chunk_text,
                            emb.document_title,
                            emb.section_title,
                            emb.section_hierarchy,  # Use TEXT[] directly
                            emb.section_level,
                            emb.chunk_start_char,
                            emb.chunk_end_char,
                            emb.section_start_char,
                            emb.section_end_char,
                            emb.page_numbers,  # Use INTEGER[] directly
                            emb.embedding_model
                        ))
                    
                    # Execute batch insert
                    from psycopg import sql
                    cur.executemany(
                        insert_sql.replace('%s', '(' + ','.join(['%s'] * 19) + ')'),
                        data
                    )
                    
                conn.commit()
                print(f"✅ Stored {len(embeddings)} Knowledge Base embeddings")
                return True
                
        except Exception as e:
            print(f"❌ Failed to store Knowledge Base embeddings: {e}")
            return False
    
    def get_kb_embeddings_count(self) -> int:
        """Get total count of embeddings in Knowledge Base"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM knowledge_base")
                    return cur.fetchone()[0]
        except Exception as e:
            print(f"❌ Failed to get Knowledge Base embeddings count: {e}")
            return 0
    
    def get_all_kb_chunk_texts(self) -> List[str]:
        """Get all original texts for BM25 index building"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT chunk_text 
                        FROM knowledge_base 
                        ORDER BY chunk_id
                    """)
                    return [row[0] for row in cur.fetchall()]
        except Exception as e:
            print(f"❌ Failed to get Knowledge Base chunk texts: {e}")
            return []
    
    def search_kb_similar_vectors(
        self, 
        query_embedding: List[float], 
        limit: int = 50,
        entity_filter: Optional[List[str]] = None,
        domain_filter: Optional[List[str]] = None,
        category_filter: Optional[List[str]] = None,
        document_name_filter: Optional[List[str]] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Search for similar vectors using cosine similarity with metadata filtering
        Returns: List of (chunk_id, chunk_text, distance)
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    # Build WHERE clause for filtering
                    where_conditions = []
                    params = [np.array(query_embedding), np.array(query_embedding)]
                    
                    if entity_filter:
                        where_conditions.append("kb.entity = ANY(%s)")
                        params.append(entity_filter)
                    
                    if domain_filter:
                        where_conditions.append("kb.domain = ANY(%s)")
                        params.append(domain_filter)
                    
                    if category_filter:
                        where_conditions.append("kb.category = ANY(%s)")
                        params.append(category_filter)
                    
                    if document_name_filter:
                        where_conditions.append("kb.document_name = ANY(%s)")
                        params.append(document_name_filter)
                    
                    where_clause = " AND " + " AND ".join(where_conditions) if where_conditions else ""
                    
                    # Add limit parameter
                    params.append(limit)
                    
                    query_sql = f"""
                        SELECT 
                            kb.chunk_id,
                            kb.chunk_text,
                            kb.embedding <=> %s as distance,
                            kb.entity as entity_name,
                            kb.domain as domain_name,
                            kb.category as category_name,
                            kb.document_name
                        FROM knowledge_base kb
                        WHERE 1=1 {where_clause}
                        ORDER BY kb.embedding <=> %s
                        LIMIT %s
                    """
                    
                    cur.execute(query_sql, params)
                    
                    results = []
                    for row in cur.fetchall():
                        results.append((
                            row['chunk_id'],
                            row['chunk_text'],
                            float(row['distance'])
                        ))
                    
                    return results
                    
        except Exception as e:
            print(f"❌ Failed to search Knowledge Base similar vectors: {e}")
            return []
    
    def get_kb_chunk_metadata(
        self,
        entity_filter: Optional[List[str]] = None,
        domain_filter: Optional[List[str]] = None,
        category_filter: Optional[List[str]] = None,
        document_name_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get Knowledge Base chunk metadata with optional filtering"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    # Build WHERE clause for filtering
                    where_conditions = []
                    params = []
                    
                    if entity_filter:
                        where_conditions.append("kb.entity = ANY(%s)")
                        params.append(entity_filter)
                    
                    if domain_filter:
                        where_conditions.append("kb.domain = ANY(%s)")
                        params.append(domain_filter)
                    
                    if category_filter:
                        where_conditions.append("kb.category = ANY(%s)")
                        params.append(category_filter)
                    
                    if document_name_filter:
                        where_conditions.append("kb.document_name = ANY(%s)")
                        params.append(document_name_filter)
                    
                    where_clause = " AND " + " AND ".join(where_conditions) if where_conditions else ""
                    
                    query_sql = f"""
                        SELECT 
                            kb.chunk_id,
                            kb.chunk_text,
                            kb.contextual_text,
                            kb.document_title,
                            kb.section_title,
                            kb.section_hierarchy,
                            kb.section_level,
                            kb.chunk_start_char,
                            kb.chunk_end_char,
                            kb.section_start_char,
                            kb.section_end_char,
                            kb.page_numbers,
                            kb.entity as entity_name,
                            kb.domain as domain_name,
                            kb.category as category_name,
                            kb.document_name
                        FROM knowledge_base kb
                        WHERE 1=1 {where_clause}
                        ORDER BY kb.chunk_id
                    """
                    
                    cur.execute(query_sql, params)
                    
                    metadata = []
                    for row in cur.fetchall():
                        metadata.append({
                            'chunk_id': row['chunk_id'],
                            'chunk_text': row['chunk_text'],
                            'contextual_text': row['contextual_text'],
                            'document_title': row['document_title'],
                            'section_title': row['section_title'],
                            'section_hierarchy': row['section_hierarchy'] or [],
                            'section_level': row['section_level'],
                            'chunk_start_char': row['chunk_start_char'],
                            'chunk_end_char': row['chunk_end_char'],
                            'section_start_char': row['section_start_char'],
                            'section_end_char': row['section_end_char'],
                            'page_numbers': row['page_numbers'] or [],
                            'entity_name': row['entity_name'],
                            'domain_name': row['domain_name'],
                            'category_name': row['category_name'],
                            'document_name': row['document_name']
                        })
                    
                    return metadata
                    
        except Exception as e:
            print(f"❌ Failed to get Knowledge Base chunk metadata: {e}")
            return []
    
    def get_entity_id(self, entity_name: str) -> Optional[int]:
        """Get entity ID by name"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM entities WHERE name = %s", (entity_name,))
                    result = cur.fetchone()
                    return result[0] if result else None
        except Exception as e:
            print(f"❌ Failed to get entity ID: {e}")
            return None
    
    def get_domain_id(self, domain_name: str) -> Optional[int]:
        """Get functional domain ID by name"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM functional_domains WHERE name = %s", (domain_name,))
                    result = cur.fetchone()
                    return result[0] if result else None
        except Exception as e:
            print(f"❌ Failed to get domain ID: {e}")
            return None
    
    def get_category_id(self, category_name: str) -> Optional[int]:
        """Get document category ID by name"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM document_categories WHERE name = %s", (category_name,))
                    result = cur.fetchone()
                    return result[0] if result else None
        except Exception as e:
            print(f"❌ Failed to get category ID: {e}")
            return None
    
    def ensure_entity(self, name: str, display_name: str, code: str) -> bool:
        """Ensure entity exists in database, create if not exists"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Check if entity already exists
                    cur.execute("SELECT id FROM entities WHERE name = %s", (name,))
                    if cur.fetchone():
                        return False  # Already exists
                    
                    # Insert new entity
                    cur.execute("""
                        INSERT INTO entities (name, display_name, code) 
                        VALUES (%s, %s, %s)
                    """, (name, display_name, code))
                    conn.commit()
                    return True  # Successfully created
        except Exception as e:
            print(f"❌ Failed to ensure entity {name}: {e}")
            return False
    
    def ensure_domain(self, name: str, display_name: str, folder_name: str) -> bool:
        """Ensure functional domain exists in database, create if not exists"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Check if domain already exists
                    cur.execute("SELECT id FROM functional_domains WHERE name = %s", (name,))
                    if cur.fetchone():
                        return False  # Already exists
                    
                    # Insert new domain
                    cur.execute("""
                        INSERT INTO functional_domains (name, display_name, folder_name) 
                        VALUES (%s, %s, %s)
                    """, (name, display_name, folder_name))
                    conn.commit()
                    return True  # Successfully created
        except Exception as e:
            print(f"❌ Failed to ensure domain {name}: {e}")
            return False
    
    def ensure_category(self, name: str, description: str) -> bool:
        """Ensure document category exists in database, create if not exists"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Check if category already exists
                    cur.execute("SELECT id FROM document_categories WHERE name = %s", (name,))
                    if cur.fetchone():
                        return False  # Already exists
                    
                    # Insert new category
                    cur.execute("""
                        INSERT INTO document_categories (name, description) 
                        VALUES (%s, %s)
                    """, (name, description))
                    conn.commit()
                    return True  # Successfully created
        except Exception as e:
            print(f"❌ Failed to ensure category {name}: {e}")
            return False
    
    def test_kb_performance(self) -> float:
        """Test Knowledge Base query performance"""
        import time
        start_time = time.time()
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM knowledge_base")
                    cur.fetchone()
            
            end_time = time.time()
            return (end_time - start_time) * 1000  # Convert to milliseconds
            
        except Exception as e:
            print(f"❌ Knowledge Base performance test failed: {e}")
            return -1
    
    def get_kb_stats(self) -> Dict[str, Any]:
        """Get Knowledge Base statistics"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Get counts by entity, domain, and category
                    cur.execute("""
                        SELECT 
                            COUNT(*) as total_chunks,
                            COUNT(DISTINCT document_title) as total_documents,
                            COUNT(DISTINCT entity) as total_entities,
                            COUNT(DISTINCT domain) as total_domains,
                            COUNT(DISTINCT category) as total_categories
                        FROM knowledge_base
                    """)
                    stats_row = cur.fetchone()
                    stats = {
                        'total_chunks': stats_row[0],
                        'total_documents': stats_row[1],
                        'total_entities': stats_row[2],
                        'total_domains': stats_row[3],
                        'total_categories': stats_row[4]
                    }
                    
                    # Get breakdown by entity
                    cur.execute("""
                        SELECT kb.entity, COUNT(*) as chunk_count
                        FROM knowledge_base kb
                        GROUP BY kb.entity
                        ORDER BY chunk_count DESC
                    """)
                    entity_breakdown = cur.fetchall()
                    
                    # Get breakdown by domain  
                    cur.execute("""
                        SELECT kb.domain, COUNT(*) as chunk_count
                        FROM knowledge_base kb
                        GROUP BY kb.domain
                        ORDER BY chunk_count DESC
                    """)
                    domain_breakdown = cur.fetchall()
                    
                    return {
                        'total_stats': stats,
                        'entity_breakdown': [{'name': row[0], 'chunk_count': row[1]} for row in entity_breakdown],
                        'domain_breakdown': [{'name': row[0], 'chunk_count': row[1]} for row in domain_breakdown]
                    }
                    
        except Exception as e:
            print(f"❌ Failed to get Knowledge Base stats: {e}")
            return {}
    
    def close(self):
        """Close Knowledge Base database connection pool"""
        if self.pool:
            self.pool.close()
            print("✅ Knowledge Base database connection pool closed")

# Global Knowledge Base manager instance
kb_manager = KnowledgeBaseManager() if KB_USE_DATABASE else None

def get_kb_manager() -> Optional[KnowledgeBaseManager]:
    """Get the global Knowledge Base manager instance"""
    return kb_manager