"""
System metrics: Index build time and size
"""
import os
from pathlib import Path
from typing import Dict


class SystemMetricsCollector:
    """
    Collect system-level metrics for RAG pipeline.
    """
    
    def get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB."""
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)  # Convert bytes to MB
    
    def get_index_size(self, index_path: str) -> float:
        """
        Get total index size including metadata file.
        
        Args:
            index_path: Path to .faiss index file
        
        Returns:
            Total size in MB
        """
        index_path = Path(index_path)
        
        # Get index file size
        index_size = self.get_file_size_mb(str(index_path))
        
        # Get metadata file size (.meta.json or .json)
        meta_path = index_path.with_suffix(".meta.json")
        if not meta_path.exists():
            meta_path = index_path.with_suffix(".json")
        
        meta_size = self.get_file_size_mb(str(meta_path)) if meta_path.exists() else 0.0
        
        total_size = index_size + meta_size
        
        return {
            "index_size_mb": index_size,
            "metadata_size_mb": meta_size,
            "total_size_mb": total_size,
        }
    
    def collect_metrics(self, 
                       index_path: str, 
                       index_build_time_sec: float = None) -> Dict:
        """
        Collect all system metrics.
        
        Args:
            index_path: Path to .faiss index file
            index_build_time_sec: Index build time in seconds (optional)
        
        Returns:
            Dict with metrics
        """
        size_metrics = self.get_index_size(index_path)
        
        metrics = {
            **size_metrics,
        }
        
        if index_build_time_sec is not None:
            metrics["index_build_time_sec"] = index_build_time_sec
        
        return metrics
