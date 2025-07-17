"""
Processing logger for version-specific error tracking
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

@dataclass
class FailedFile:
    """Represents a file that failed processing"""
    file_path: str
    excel_title: str
    error: str
    entity: str
    domain: str
    category: str
    timestamp: str

@dataclass
class ProcessingStats:
    """Processing statistics summary"""
    total_documents: int
    processed: int
    cached: int
    failed: int
    success_rate: str

class ProcessingLogger:
    """
    Version-specific processing logger
    Tracks processed files, failed files, and provides comprehensive reporting
    """
    
    def __init__(self, version: str):
        self.version = version
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"processing_v{version}.log"
        
        # Initialize tracking lists
        self.processed_files: List[str] = []
        self.cached_files: List[str] = []
        self.failed_files: List[FailedFile] = []
        
        # Processing metadata
        self.start_time = datetime.now()
        self.force_populate = os.getenv("FORCE_POPULATE", "false").lower() == "true"
        self.cache_cleared = False
        
        print(f"📊 ProcessingLogger initialized for version {version}")
        print(f"📁 Log file: {self.log_file}")
    
    def log_processed_file(self, file_path: str, from_cache: bool = False):
        """Log a successfully processed file"""
        filename = Path(file_path).name
        
        if from_cache:
            self.cached_files.append(filename)
            print(f"📋 Cached: {filename}")
        else:
            self.processed_files.append(filename)
            print(f"✅ Processed: {filename}")
    
    def log_failed_file(self, file_path: str, excel_title: str, error: str, 
                       entity: str, domain: str, category: str):
        """Log a file that failed processing"""
        failed_file = FailedFile(
            file_path=file_path,
            excel_title=excel_title,
            error=error,
            entity=entity,
            domain=domain,
            category=category,
            timestamp=datetime.now().isoformat()
        )
        
        self.failed_files.append(failed_file)
        filename = Path(file_path).name
        print(f"❌ Failed: {filename} - {error}")
    
    def log_cache_cleared(self):
        """Log that cache was cleared"""
        self.cache_cleared = True
        print("🧹 Cache cleared")
    
    def get_processing_stats(self, total_documents: int) -> ProcessingStats:
        """Get current processing statistics"""
        processed_count = len(self.processed_files)
        cached_count = len(self.cached_files)
        failed_count = len(self.failed_files)
        
        # Calculate success rate
        if total_documents > 0:
            success_rate = f"{((processed_count + cached_count) / total_documents * 100):.1f}%"
        else:
            success_rate = "0%"
        
        return ProcessingStats(
            total_documents=total_documents,
            processed=processed_count,
            cached=cached_count,
            failed=failed_count,
            success_rate=success_rate
        )
    
    def save_processing_summary(self, total_documents: int, additional_info: Dict[str, Any] = None):
        """Save final processing summary to log file"""
        try:
            stats = self.get_processing_stats(total_documents)
            end_time = datetime.now()
            processing_duration = (end_time - self.start_time).total_seconds()
            
            log_data = {
                "version": self.version,
                "timestamp": self.start_time.isoformat(),
                "processing_duration_seconds": round(processing_duration, 2),
                "force_populate": self.force_populate,
                "cache_cleared": self.cache_cleared,
                "processed_files": self.processed_files,
                "cached_files": self.cached_files,
                "failed_files": [asdict(f) for f in self.failed_files],
                "summary": asdict(stats)
            }
            
            # Add any additional information
            if additional_info:
                log_data.update(additional_info)
            
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            print(f"📊 Processing summary saved to {self.log_file}")
            
            # Print final summary
            print("\n" + "="*50)
            print("📊 PROCESSING SUMMARY")
            print("="*50)
            print(f"Version: {self.version}")
            print(f"Total documents: {stats.total_documents}")
            print(f"Processed: {stats.processed}")
            print(f"Cached: {stats.cached}")
            print(f"Failed: {stats.failed}")
            print(f"Success rate: {stats.success_rate}")
            print(f"Processing time: {processing_duration:.2f} seconds")
            
            if self.failed_files:
                print(f"\n❌ Failed files ({len(self.failed_files)}):")
                for failed in self.failed_files:
                    print(f"  • {failed.excel_title}: {failed.error}")
            
            print("="*50)
            
        except Exception as e:
            print(f"❌ Failed to save processing summary: {e}")
    
    def get_failed_files_by_error(self) -> Dict[str, List[str]]:
        """Group failed files by error type"""
        error_groups = {}
        
        for failed in self.failed_files:
            error_type = failed.error.split(':')[0]  # Get error type before colon
            if error_type not in error_groups:
                error_groups[error_type] = []
            error_groups[error_type].append(failed.excel_title)
        
        return error_groups
    
    def print_progress(self, current: int, total: int, current_file: str = ""):
        """Print processing progress"""
        percentage = (current / total * 100) if total > 0 else 0
        processed_count = len(self.processed_files)
        cached_count = len(self.cached_files)
        failed_count = len(self.failed_files)
        
        status = f"[{current}/{total}] {percentage:.1f}% - "
        status += f"✅ {processed_count} processed, 📋 {cached_count} cached, ❌ {failed_count} failed"
        
        if current_file:
            status += f" - Current: {Path(current_file).name}"
        
        print(status)
    
    def load_previous_log(self) -> Optional[Dict[str, Any]]:
        """Load previous processing log if exists"""
        try:
            if self.log_file.exists():
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Could not load previous log: {e}")
        
        return None