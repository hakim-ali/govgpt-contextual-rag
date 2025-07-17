#!/usr/bin/env python3
"""
Excel-to-YAML generator for Knowledge Base versioned mapping system
"""
import os
import sys
import pandas as pd
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class KBMappingGenerator:
    """Generates versioned YAML mappings from Excel hierarchy data"""
    
    def __init__(self, excel_file: Optional[str] = None, version: str = "1.0", knowledge_base_folder: str = "knowledge_base_prod"):
        self.excel_file = excel_file
        self.version = version
        self.knowledge_base_folder = knowledge_base_folder
        self.kb_base_dir = Path(__file__).parent.parent
        self.config_dir = self.kb_base_dir / "config"
        self.knowledge_base_dir = self.kb_base_dir / knowledge_base_folder
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        print(f"🔧 KB Mapping Generator initialized")
        print(f"📁 Knowledge Base dir: {self.knowledge_base_dir}")
        print(f"📝 Version: v{self.version}")
        print(f"📊 Using folder: {knowledge_base_folder}")
    
    def read_excel_hierarchy(self) -> Optional[pd.DataFrame]:
        """Read Excel file with hierarchy data"""
        if not self.excel_file or not os.path.exists(self.excel_file):
            print(f"⚠️  Excel file not found: {self.excel_file}")
            print("🔄 Using hardcoded data from previous analysis...")
            return None
            
        try:
            print(f"📖 Reading Excel file: {self.excel_file}")
            df = pd.read_excel(self.excel_file)
            print(f"✅ Read {len(df)} rows from Excel file")
            return df
            
        except Exception as e:
            print(f"❌ Failed to read Excel file: {e}")
            return None
    
    def get_hardcoded_excel_data(self) -> Dict[str, Any]:
        """Get hardcoded Excel data from previous analysis"""
        return {
            'entities': [
                {'name': 'General', 'display_name': 'General Government', 'code': 'GEN'},
                {'name': 'DGE', 'display_name': 'Department of Government Enablement', 'code': 'DGE'},
                {'name': 'Department of Finance', 'display_name': 'Department of Finance', 'code': 'DOF'},
                {'name': 'Human Resource Authority', 'display_name': 'Human Resource Authority', 'code': 'HRA'}
            ],
            'functional_domains': [
                {'name': 'CX', 'display_name': 'Customer Experience', 'folder_name': 'CX'},
                {'name': 'Finance', 'display_name': 'Finance', 'folder_name': 'Finance'},
                {'name': 'HR', 'display_name': 'Human Resources', 'folder_name': 'HR'},
                {'name': 'Procurement', 'display_name': 'Procurement', 'folder_name': 'Procurement'},
                {'name': 'IT', 'display_name': 'Information Technology', 'folder_name': 'IT'}
            ],
            'document_categories': [
                {'name': 'Guide', 'description': 'Instructional guides and manuals'},
                {'name': 'Glossary', 'description': 'Glossaries and definitions'},
                {'name': 'Templates', 'description': 'Document templates and forms'},
                {'name': 'Policy', 'description': 'Government policies and guidelines'},
                {'name': 'Law', 'description': 'Legal documents and regulations'},
                {'name': 'Framework', 'description': 'Frameworks and standards'},
                {'name': 'Charter', 'description': 'Organizational charters'},
                {'name': 'Manual', 'description': 'Operational manuals and procedures'},
                {'name': 'FAQ', 'description': 'Frequently Asked Questions'}
            ],
            'excel_documents': [
                {'title': 'CX Abu Dhabi Government Tone of Voice Document', 'entity': 'General', 'domain': 'CX', 'category': 'Guide'},
                {'title': 'CX Effortless_Guide_AR', 'entity': 'DGE', 'domain': 'CX', 'category': 'Guide'},
                {'title': 'CX Effortless_Guide_EN_V2', 'entity': 'DGE', 'domain': 'CX', 'category': 'Guide'},
                {'title': 'CX Glossary', 'entity': 'DGE', 'domain': 'CX', 'category': 'Glossary'},
                {'title': 'CX TOV templates', 'entity': 'DGE', 'domain': 'CX', 'category': 'Templates'},
                {'title': 'Abu Dhabi Government Finance Policy Manual_AR', 'entity': 'Department of Finance', 'domain': 'Finance', 'category': 'Policy'},
                {'title': 'Abu Dhabi Government Finance Policy Manual_EN', 'entity': 'Department of Finance', 'domain': 'Finance', 'category': 'Policy'},
                {'title': 'HR - Government Employee Guide Version 1 - AR', 'entity': 'Human Resource Authority', 'domain': 'HR', 'category': 'Guide'},
                {'title': 'HR - Government Employee Guide Version 2 - AR', 'entity': 'Human Resource Authority', 'domain': 'HR', 'category': 'Guide'},
                {'title': 'HR - HR Law - EN', 'entity': 'Human Resource Authority', 'domain': 'HR', 'category': 'Law'},
                {'title': 'HR - HR Law- AR', 'entity': 'Human Resource Authority', 'domain': 'HR', 'category': 'Law'},
                {'title': 'HR - Implementation Regulation for HR Law No 6 Year 2016 - AR', 'entity': 'Human Resource Authority', 'domain': 'HR', 'category': 'Law'},
                {'title': 'HR - Implementation Regulation for HR Law No 6 Year 2016 - EN', 'entity': 'Human Resource Authority', 'domain': 'HR', 'category': 'Law'},
                {'title': 'Proc - DGE Procurement Framework - EN', 'entity': 'DGE', 'domain': 'Procurement', 'category': 'Framework'},
                {'title': 'Proc - Policies and Procedures for Sales Auctions and Warehouses - AR', 'entity': 'DGE', 'domain': 'Procurement', 'category': 'Policy'},
                {'title': 'Proc - Policies and Procedures for Sales Auctions and Warehouses - EN', 'entity': 'DGE', 'domain': 'Procurement', 'category': 'Policy'},
                {'title': 'Proc - Procurement Charter - AR', 'entity': 'DGE', 'domain': 'Procurement', 'category': 'Charter'},
                {'title': 'Proc - Procurement Charter - EN', 'entity': 'DGE', 'domain': 'Procurement', 'category': 'Charter'},
                {'title': 'Proc - Procurement Manual (Ariba Aligned) - EN', 'entity': 'DGE', 'domain': 'Procurement', 'category': 'Manual'},
                {'title': 'Proc - Procurement Manual (Business Process) - AR', 'entity': 'DGE', 'domain': 'Procurement', 'category': 'Manual'},
                {'title': 'Proc - Procurement Manual (Business Process) - EN', 'entity': 'DGE', 'domain': 'Procurement', 'category': 'Manual'},
                {'title': 'Proc - Procurement Standard Regulations - AR', 'entity': 'DGE', 'domain': 'Procurement', 'category': 'Manual'},
                {'title': 'Proc - Procurement Standard Regulations - EN', 'entity': 'DGE', 'domain': 'Procurement', 'category': 'Manual'},
                {'title': 'Proc - Support Frequently Asked Questions - EN', 'entity': 'DGE', 'domain': 'Procurement', 'category': 'FAQ'}
            ]
        }
    
    def extract_excel_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract hierarchy data from Excel DataFrame with null value handling"""
        try:
            # Handle null values by filling with defaults
            df = df.fillna({
                'government_entity': 'Unknown',
                'functional_domain': 'General',
                'document_category': 'Document'
            })
            
            # Extract unique entities (excluding null/empty values)
            entities = []
            unique_entities = df['government_entity'].dropna().unique()
            for entity in unique_entities:
                if entity and str(entity).strip():  # Skip empty strings
                    code = ''.join([word[0].upper() for word in str(entity).split()[:3]])  # Generate code
                    entities.append({
                        'name': str(entity),
                        'display_name': str(entity),
                        'code': code
                    })
            
            # Extract unique functional domains (excluding null/empty values)
            domains = []
            unique_domains = df['functional_domain'].dropna().unique()
            for domain in unique_domains:
                if domain and str(domain).strip():  # Skip empty strings
                    domains.append({
                        'name': str(domain),
                        'display_name': str(domain),
                        'folder_name': str(domain)
                    })
            
            # Extract unique document categories (excluding null/empty values)
            categories = []
            unique_categories = df['document_category'].dropna().unique()
            for category in unique_categories:
                if category and str(category).strip():  # Skip empty strings
                    categories.append({
                        'name': str(category),
                        'description': f"{category} documents"
                    })
            
            # Extract document mappings with null-safe value extraction and language filtering
            excel_documents = []
            allowed_languages = ['en', 'ar+en']
            skipped_count = 0
            
            for _, row in df.iterrows():
                # Handle null values with defaults
                entity = row['government_entity'] if pd.notna(row['government_entity']) and str(row['government_entity']).strip() else 'Unknown'
                domain = row['functional_domain'] if pd.notna(row['functional_domain']) and str(row['functional_domain']).strip() else 'General'
                category = row['document_category'] if pd.notna(row['document_category']) and str(row['document_category']).strip() else 'Document'
                
                # Extract language with null handling
                language = row['language'] if pd.notna(row['language']) and str(row['language']).strip() else 'en'
                language = str(language).strip().lower()  # Normalize to lowercase
                
                # Filter by allowed languages
                if language not in allowed_languages:
                    skipped_count += 1
                    title = str(row['title']) if pd.notna(row['title']) else 'Untitled'
                    print(f"⏭️  Skipping document '{title}' - language '{language}' not in allowed list {allowed_languages}")
                    continue
                
                excel_documents.append({
                    'title': str(row['title']) if pd.notna(row['title']) else 'Untitled',
                    'entity': str(entity),
                    'domain': str(domain),
                    'category': str(category),
                    'language': str(language)
                })
            
            print(f"📋 Processed {len(excel_documents)} documents ({skipped_count} skipped due to language filtering)")
            
            return {
                'entities': entities,
                'functional_domains': domains,
                'document_categories': categories,
                'excel_documents': excel_documents
            }
            
        except Exception as e:
            print(f"❌ Failed to extract Excel data: {e}")
            return self.get_hardcoded_excel_data()
    
    def scan_available_files(self) -> List[Dict[str, str]]:
        """Scan knowledge_base directory for available files"""
        available_files = []
        
        if not self.knowledge_base_dir.exists():
            print(f"⚠️  Knowledge base directory not found: {self.knowledge_base_dir}")
            return available_files
        
        print(f"🔍 Scanning knowledge base directory: {self.knowledge_base_dir}")
        
        for root, dirs, files in os.walk(self.knowledge_base_dir):
            for file in files:
                if file.lower().endswith(('.pdf', '.docx', '.doc', '.txt')):
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(self.kb_base_dir)
                    
                    # Extract domain from path
                    path_parts = file_path.relative_to(self.knowledge_base_dir).parts
                    domain = path_parts[0] if path_parts else 'Unknown'
                    
                    available_files.append({
                        'filename': file,
                        'file_path': str(relative_path),
                        'domain': domain,
                        'size': file_path.stat().st_size
                    })
        
        print(f"📁 Found {len(available_files)} files in knowledge base")
        return available_files
    
    def find_matching_file(self, excel_title: str, available_files: List[Dict]) -> Optional[Dict]:
        """Find matching file for Excel document using intelligent matching"""
        import re
        
        # Clean the Excel title for matching
        clean_title = excel_title.lower().replace(' ', '').replace('-', '').replace('_', '')
        
        # Try exact filename match first
        for file_info in available_files:
            filename = file_info['filename']
            clean_filename = filename.lower().replace(' ', '').replace('-', '').replace('_', '')
            
            # Remove file extension for comparison
            clean_filename = re.sub(r'\.(pdf|docx|doc|txt)$', '', clean_filename)
            
            if clean_title == clean_filename:
                return file_info
        
        # Try partial matching with key terms
        title_words = excel_title.lower().split()
        key_terms = []
        
        # Extract key terms from title
        for word in title_words:
            if len(word) > 2 and word not in ['the', 'and', 'for', 'of', 'in', 'on', 'at', 'to', 'by']:
                key_terms.append(word)
        
        # Score files based on matching key terms
        best_match = None
        best_score = 0
        
        for file_info in available_files:
            filename = file_info['filename'].lower()
            score = 0
            
            for term in key_terms:
                if term in filename:
                    score += 1
            
            # Bonus for language match
            if 'en' in excel_title.lower() and 'en' in filename:
                score += 0.5
            elif 'ar' in excel_title.lower() and 'ar' in filename:
                score += 0.5
            
            # Bonus for domain match
            domain_path = file_info['file_path'].lower()
            if any(term in domain_path for term in key_terms):
                score += 0.3
            
            if score > best_score and score >= len(key_terms) * 0.6:  # At least 60% match
                best_score = score
                best_match = file_info
        
        return best_match

    def map_excel_to_files(self, excel_documents: List[Dict], available_files: List[Dict]) -> List[Dict[str, Any]]:
        """Map Excel documents to available files using intelligent matching"""
        mapped_documents = []
        matched_files = set()
        
        # Try to match each Excel document to available files
        for excel_doc in excel_documents:
            matching_file = self.find_matching_file(excel_doc['title'], available_files)
            
            if matching_file and matching_file['filename'] not in matched_files:
                # Mark file as matched to avoid duplicates
                matched_files.add(matching_file['filename'])
                
                mapped_documents.append({
                    'excel_title': excel_doc['title'],
                    'entity': excel_doc['entity'],
                    'domain': excel_doc['domain'],
                    'category': excel_doc['category'],
                    'language': excel_doc['language'],
                    'file_path': matching_file['file_path'],
                    'filename': matching_file['filename'],
                    'file_size': matching_file['size'],
                    'status': 'available'
                })
                
                print(f"✅ Matched: '{excel_doc['title']}' → {matching_file['filename']}")
            else:
                # Document not found in available files
                mapped_documents.append({
                    'excel_title': excel_doc['title'],
                    'entity': excel_doc['entity'],
                    'domain': excel_doc['domain'],
                    'category': excel_doc['category'],
                    'language': excel_doc['language'],
                    'file_path': None,
                    'filename': None,
                    'file_size': None,
                    'status': 'missing'
                })
                
                print(f"❌ Not found: '{excel_doc['title']}'")
        
        # Report unmatched files
        unmatched_files = [f for f in available_files if f['filename'] not in matched_files]
        if unmatched_files:
            print(f"📋 Unmatched files ({len(unmatched_files)}):")
            for file_info in unmatched_files[:10]:  # Show first 10
                print(f"   - {file_info['filename']}")
            if len(unmatched_files) > 10:
                print(f"   ... and {len(unmatched_files) - 10} more")
        
        print(f"\n🔗 Mapped {len([d for d in mapped_documents if d['status'] == 'available'])} available files")
        print(f"❓ {len([d for d in mapped_documents if d['status'] == 'missing'])} documents marked as missing")
        
        return mapped_documents
    
    def generate_yaml_mapping(self) -> bool:
        """Generate versioned YAML mapping file"""
        try:
            print(f"🔄 Generating YAML mapping v{self.version}...")
            
            # Read Excel data
            df = self.read_excel_hierarchy()
            if df is not None:
                excel_data = self.extract_excel_data(df)
            else:
                excel_data = self.get_hardcoded_excel_data()
            
            # Scan available files
            available_files = self.scan_available_files()
            
            # Map Excel documents to files
            document_mappings = self.map_excel_to_files(
                excel_data['excel_documents'], 
                available_files
            )
            
            # Create YAML structure
            yaml_data = {
                'version': self.version,
                'generated_at': datetime.now().isoformat(),
                'source_excel': self.excel_file or 'hardcoded_data',
                'total_documents': len(document_mappings),
                'available_documents': len([d for d in document_mappings if d['status'] == 'available']),
                'missing_documents': len([d for d in document_mappings if d['status'] == 'missing']),
                
                'entities': excel_data['entities'],
                'functional_domains': excel_data['functional_domains'],
                'document_categories': excel_data['document_categories'],
                'document_mappings': document_mappings
            }
            
            # Write YAML file
            yaml_file = self.config_dir / f"mapping_v{self.version}.yaml"
            with open(yaml_file, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            print(f"✅ Generated YAML mapping: {yaml_file}")
            print(f"📊 Summary:")
            print(f"   - Total documents: {yaml_data['total_documents']}")
            print(f"   - Available: {yaml_data['available_documents']}")
            print(f"   - Missing: {yaml_data['missing_documents']}")
            print(f"   - Entities: {len(yaml_data['entities'])}")
            print(f"   - Domains: {len(yaml_data['functional_domains'])}")
            print(f"   - Categories: {len(yaml_data['document_categories'])}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to generate YAML mapping: {e}")
            return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Knowledge Base YAML mapping')
    parser.add_argument('--excel', '-e', help='Path to Excel hierarchy file')
    parser.add_argument('--version', '-v', default='1.0', help='Mapping version (default: 1.0)')
    parser.add_argument('--folder', '-f', default='knowledge_base_prod', help='Knowledge base folder name (default: knowledge_base_prod)')
    
    args = parser.parse_args()
    
    print("🚀 Knowledge Base Mapping Generator")
    print("=" * 50)
    
    try:
        generator = KBMappingGenerator(args.excel, args.version, args.folder)
        success = generator.generate_yaml_mapping()
        
        if success:
            print("\\n✅ YAML mapping generation completed successfully!")
            return 0
        else:
            print("\\n❌ YAML mapping generation failed!")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)