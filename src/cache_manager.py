"""
Cache Manager for AnimatePodcasting

This module provides caching functionality for expensive operations like transcription
and image generation to improve performance and reduce redundant processing.
"""

import os
import json
import hashlib
import logging
import pickle
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages caching of expensive operations like transcription and image generation."""
    
    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir (str): Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.stats = {
            "hits": 0,
            "misses": 0,
            "saved_time": 0  # in seconds
        }
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create subdirectories for different cache types
        self.transcription_cache = self.cache_dir / "transcriptions"
        self.image_cache = self.cache_dir / "images"
        self.analysis_cache = self.cache_dir / "analysis"
        
        os.makedirs(self.transcription_cache, exist_ok=True)
        os.makedirs(self.image_cache, exist_ok=True)
        os.makedirs(self.analysis_cache, exist_ok=True)
        
        # Load cache stats if they exist
        self._load_stats()
        
        logger.info(f"Cache initialized at {self.cache_dir}")
    
    def _load_stats(self):
        """Load cache statistics from disk."""
        stats_file = self.cache_dir / "stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, "r") as f:
                    self.stats = json.load(f)
                logger.info(f"Loaded cache stats: hits={self.stats['hits']}, misses={self.stats['misses']}")
            except Exception as e:
                logger.warning(f"Failed to load cache stats: {str(e)}")
    
    def _save_stats(self):
        """Save cache statistics to disk."""
        stats_file = self.cache_dir / "stats.json"
        try:
            with open(stats_file, "w") as f:
                json.dump(self.stats, f)
        except Exception as e:
            logger.warning(f"Failed to save cache stats: {str(e)}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """
        Generate a hash for a file based on its content and metadata.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            str: Hash string for the file
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # For large files, use a combination of file size, modification time, and partial content
        file_stat = file_path.stat()
        size = file_stat.st_size
        mtime = file_stat.st_mtime
        
        # For audio files, read the first 1MB for hashing
        with open(file_path, "rb") as f:
            content = f.read(1024 * 1024)  # Read first 1MB
        
        # Create hash using size, mtime, and partial content
        hasher = hashlib.md5()
        hasher.update(f"{size}_{mtime}".encode())
        hasher.update(content)
        
        return hasher.hexdigest()
    
    def _get_cache_path(self, file_hash: str, cache_type: str) -> Path:
        """
        Get the cache file path for a given hash and cache type.
        
        Args:
            file_hash (str): Hash of the original file
            cache_type (str): Type of cache (transcription, image, analysis)
            
        Returns:
            Path: Path to the cache file
        """
        if cache_type == "transcription":
            return self.transcription_cache / f"{file_hash}.json"
        elif cache_type == "image":
            return self.image_cache / f"{file_hash}.pkl"
        elif cache_type == "analysis":
            return self.analysis_cache / f"{file_hash}.json"
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
    
    def get_transcription(self, audio_path: str, model_size: str = "base") -> Optional[Dict]:
        """
        Get cached transcription for an audio file if it exists.
        
        Args:
            audio_path (str): Path to the audio file
            model_size (str): Whisper model size used
            
        Returns:
            dict: Cached transcription data or None if not found
        """
        try:
            # Generate hash from audio file and model size
            file_hash = self._get_file_hash(audio_path) + f"_{model_size}"
            cache_path = self._get_cache_path(file_hash, "transcription")
            
            if cache_path.exists():
                # Load cached transcription
                with open(cache_path, "r") as f:
                    transcription = json.load(f)
                
                # Update cache stats
                self.stats["hits"] += 1
                if "processing_time" in transcription["metadata"]:
                    self.stats["saved_time"] += transcription["metadata"]["processing_time"]
                self._save_stats()
                
                logger.info(f"Cache hit for transcription of {audio_path}")
                return transcription["data"]
            else:
                # Cache miss
                self.stats["misses"] += 1
                self._save_stats()
                logger.info(f"Cache miss for transcription of {audio_path}")
                return None
        except Exception as e:
            logger.warning(f"Error checking transcription cache: {str(e)}")
            return None
    
    def save_transcription(self, audio_path: str, transcription: Dict, 
                          model_size: str = "base", processing_time: float = 0):
        """
        Save transcription data to cache.
        
        Args:
            audio_path (str): Path to the audio file
            transcription (dict): Transcription data to cache
            model_size (str): Whisper model size used
            processing_time (float): Time taken to process the transcription in seconds
        """
        try:
            # Generate hash from audio file and model size
            file_hash = self._get_file_hash(audio_path) + f"_{model_size}"
            cache_path = self._get_cache_path(file_hash, "transcription")
            
            # Create metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "audio_file": os.path.basename(audio_path),
                "model_size": model_size,
                "processing_time": processing_time
            }
            
            # Save transcription with metadata
            cache_data = {
                "metadata": metadata,
                "data": transcription
            }
            
            with open(cache_path, "w") as f:
                json.dump(cache_data, f)
            
            logger.info(f"Cached transcription for {audio_path}")
            
        except Exception as e:
            logger.warning(f"Failed to cache transcription: {str(e)}")
    
    def get_analysis(self, transcription_hash: str) -> Optional[Dict]:
        """
        Get cached content analysis if it exists.
        
        Args:
            transcription_hash (str): Hash of the transcription data
            
        Returns:
            dict: Cached analysis data or None if not found
        """
        try:
            cache_path = self._get_cache_path(transcription_hash, "analysis")
            
            if cache_path.exists():
                # Load cached analysis
                with open(cache_path, "r") as f:
                    analysis = json.load(f)
                
                # Update cache stats
                self.stats["hits"] += 1
                if "processing_time" in analysis["metadata"]:
                    self.stats["saved_time"] += analysis["metadata"]["processing_time"]
                self._save_stats()
                
                logger.info(f"Cache hit for analysis {transcription_hash[:8]}")
                return analysis["data"]
            else:
                # Cache miss
                self.stats["misses"] += 1
                self._save_stats()
                logger.info(f"Cache miss for analysis {transcription_hash[:8]}")
                return None
        except Exception as e:
            logger.warning(f"Error checking analysis cache: {str(e)}")
            return None
    
    def save_analysis(self, transcription_hash: str, analysis_data: Dict, processing_time: float = 0):
        """
        Save content analysis data to cache.
        
        Args:
            transcription_hash (str): Hash of the transcription data
            analysis_data (dict): Analysis data to cache
            processing_time (float): Time taken to process the analysis in seconds
        """
        try:
            cache_path = self._get_cache_path(transcription_hash, "analysis")
            
            # Create metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "processing_time": processing_time
            }
            
            # Save analysis with metadata
            cache_data = {
                "metadata": metadata,
                "data": analysis_data
            }
            
            with open(cache_path, "w") as f:
                json.dump(cache_data, f)
            
            logger.info(f"Cached analysis {transcription_hash[:8]}")
            
        except Exception as e:
            logger.warning(f"Failed to cache analysis: {str(e)}")
    
    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            dict: Cache statistics
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests) * 100 if total_requests > 0 else 0
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": f"{hit_rate:.2f}%",
            "saved_time": f"{self.stats['saved_time']:.2f} seconds",
            "cache_size": self._get_cache_size()
        }
    
    def _get_cache_size(self) -> str:
        """
        Calculate the total size of the cache.
        
        Returns:
            str: Formatted cache size (KB, MB, etc.)
        """
        total_size = 0
        
        for dirpath, _, filenames in os.walk(self.cache_dir):
            for f in filenames:
                fp = Path(dirpath) / f
                if fp.exists():
                    total_size += fp.stat().st_size
        
        # Convert to human-readable format
        for unit in ['bytes', 'KB', 'MB', 'GB']:
            if total_size < 1024:
                return f"{total_size:.2f} {unit}"
            total_size /= 1024
        
        return f"{total_size:.2f} TB"
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """
        Clear the cache.
        
        Args:
            cache_type (str, optional): Type of cache to clear (transcription, image, analysis)
                                      If None, clear all caches.
        """
        try:
            if cache_type == "transcription" or cache_type is None:
                for f in self.transcription_cache.glob("*.json"):
                    f.unlink()
                logger.info("Cleared transcription cache")
            
            if cache_type == "image" or cache_type is None:
                for f in self.image_cache.glob("*.pkl"):
                    f.unlink()
                logger.info("Cleared image cache")
            
            if cache_type == "analysis" or cache_type is None:
                for f in self.analysis_cache.glob("*.json"):
                    f.unlink()
                logger.info("Cleared analysis cache")
            
            if cache_type is None:
                # Reset stats
                self.stats = {"hits": 0, "misses": 0, "saved_time": 0}
                self._save_stats()
                logger.info("Reset cache statistics")
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
            raise 