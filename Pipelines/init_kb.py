#!/usr/bin/env python3
"""
Initialize Knowledge Base for Docker deployment
This script populates the PGVector database with documents when the container starts
"""
import os
import sys
import time
import subprocess
from pathlib import Path

def wait_for_postgres(max_retries=30, retry_delay=2):
    """Wait for PostgreSQL to be ready"""
    print("🔄 Waiting for PostgreSQL to be ready...")
    
    for attempt in range(max_retries):
        try:
            # Simple test to check if we can connect to the database
            result = subprocess.run([
                "python", "-c", 
                """
import psycopg
import os
try:
    conn = psycopg.connect(os.getenv('KB_DATABASE_URL', 'postgresql://mustaqmollah@host.docker.internal:5432/WoG'))
    conn.close()
    print('✅ PostgreSQL is ready')
    exit(0)
except Exception as e:
    print(f'❌ PostgreSQL not ready: {e}')
    exit(1)
"""
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print("✅ PostgreSQL is ready!")
                return True
            else:
                print(f"❌ PostgreSQL not ready (attempt {attempt + 1}/{max_retries})")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Connection timeout (attempt {attempt + 1}/{max_retries})")
        except Exception as e:
            print(f"❌ Error checking PostgreSQL: {e}")
        
        if attempt < max_retries - 1:
            print(f"⏳ Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    print("❌ PostgreSQL not available after maximum retries")
    return False

def check_kb_data():
    """Check if knowledge base already has data"""
    print("🔍 Checking if Knowledge Base has data...")
    
    try:
        result = subprocess.run([
            "python", "-c", 
            """
import sys
sys.path.append('/app/govgpt-kb/src')
from database import get_kb_manager
kb_manager = get_kb_manager()
if kb_manager:
    count = kb_manager.get_kb_embeddings_count()
    print(f'Found {count} chunks in Knowledge Base')
    exit(0 if count > 0 else 1)
else:
    print('KB manager not available')
    exit(1)
"""
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Knowledge Base already has data")
            return True
        else:
            print("📝 Knowledge Base is empty, will populate with data")
            return False
            
    except Exception as e:
        print(f"❌ Error checking Knowledge Base data: {e}")
        return False

def populate_kb():
    """Populate Knowledge Base with documents"""
    print("🚀 Populating Knowledge Base with documents...")
    
    try:
        # Change to govgpt-kb directory
        kb_dir = Path("/app/govgpt-kb")
        os.chdir(kb_dir)
        
        # Run the populate script
        result = subprocess.run([
            "python", "scripts/populate_database.py", "--version", "prod"
        ], timeout=600)  # 10 minute timeout
        
        if result.returncode == 0:
            print("✅ Knowledge Base populated successfully!")
            return True
        else:
            print("❌ Knowledge Base population failed!")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Knowledge Base population timed out!")
        return False
    except Exception as e:
        print(f"❌ Error populating Knowledge Base: {e}")
        return False

def main():
    """Main initialization function"""
    print("🚀 Knowledge Base Initialization for Docker")
    print("=" * 50)
    
    # Wait for PostgreSQL
    if not wait_for_postgres():
        print("❌ PostgreSQL not available, exiting")
        sys.exit(1)
    
    # Check if KB already has data
    if check_kb_data():
        print("✅ Knowledge Base already initialized")
        return 0
    
    # Populate Knowledge Base
    if populate_kb():
        print("✅ Knowledge Base initialization completed successfully!")
        return 0
    else:
        print("❌ Knowledge Base initialization failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)