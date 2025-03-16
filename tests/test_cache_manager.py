"""
Tests for the cache manager module.
"""

import unittest
import os
import sys
import json
import shutil
import tempfile
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cache_manager import CacheManager


class TestCacheManager(unittest.TestCase):
    """Test cases for the CacheManager class."""
    
    def setUp(self):
        """Set up for tests."""
        # Create a temporary directory for cache
        self.temp_dir = tempfile.mkdtemp()
        self.cache_manager = CacheManager(cache_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test initialization."""
        # Check that cache directories were created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "transcriptions")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "images")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "analysis")))
        
        # Check that stats were initialized
        stats = self.cache_manager.stats
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)
        self.assertEqual(stats["saved_time"], 0)
    
    def test_generate_file_hash(self):
        """Test file hash generation."""
        # Create a temporary file with known content
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"test content")
            file_path = temp_file.name
        
        try:
            # Generate hash
            file_hash = self.cache_manager._generate_file_hash(file_path)
            
            # Check that hash is a non-empty string
            self.assertIsInstance(file_hash, str)
            self.assertTrue(len(file_hash) > 0)
            
            # Generate hash again and check for consistency
            file_hash2 = self.cache_manager._generate_file_hash(file_path)
            self.assertEqual(file_hash, file_hash2)
            
            # Modify the file and check that hash changes
            with open(file_path, 'wb') as f:
                f.write(b"modified content")
            
            file_hash3 = self.cache_manager._generate_file_hash(file_path)
            self.assertNotEqual(file_hash, file_hash3)
        
        finally:
            # Clean up
            os.remove(file_path)
    
    def test_get_transcription_not_cached(self):
        """Test retrieving a transcription that's not in the cache."""
        result = self.cache_manager.get_transcription("nonexistent.mp3")
        
        # Check that None is returned for non-cached file
        self.assertIsNone(result)
        
        # Check that stats were updated
        self.assertEqual(self.cache_manager.stats["hits"], 0)
        self.assertEqual(self.cache_manager.stats["misses"], 1)
    
    def test_save_and_get_transcription(self):
        """Test saving and retrieving a transcription."""
        # Create a test transcription
        transcription = {
            "text": "Test transcription",
            "segments": [{"text": "Test segment"}]
        }
        
        # Save it to cache
        self.cache_manager.save_transcription("test.mp3", transcription, processing_time=1.5)
        
        # Retrieve it from cache
        cached_transcription = self.cache_manager.get_transcription("test.mp3")
        
        # Check that retrieved transcription matches original
        self.assertEqual(cached_transcription, transcription)
        
        # Check that stats were updated
        self.assertEqual(self.cache_manager.stats["hits"], 1)
        self.assertEqual(self.cache_manager.stats["saved_time"], 1.5)
    
    def test_get_analysis_not_cached(self):
        """Test retrieving analysis that's not in the cache."""
        result = self.cache_manager.get_analysis("nonexistent_hash")
        
        # Check that None is returned for non-cached analysis
        self.assertIsNone(result)
        
        # Check that stats were updated
        self.assertEqual(self.cache_manager.stats["hits"], 0)
        self.assertEqual(self.cache_manager.stats["misses"], 1)
    
    def test_save_and_get_analysis(self):
        """Test saving and retrieving analysis."""
        # Create test data
        analysis = ["Segment 1", "Segment 2"]
        
        # Save it to cache
        self.cache_manager.save_analysis("test_hash", analysis, processing_time=0.5)
        
        # Retrieve it from cache
        cached_analysis = self.cache_manager.get_analysis("test_hash")
        
        # Check that retrieved analysis matches original
        self.assertEqual(cached_analysis, analysis)
        
        # Check that stats were updated
        self.assertEqual(self.cache_manager.stats["hits"], 1)
        self.assertEqual(self.cache_manager.stats["saved_time"], 0.5)
    
    def test_get_cache_stats(self):
        """Test retrieving cache statistics."""
        # Add some test data to affect stats
        self.cache_manager.stats["hits"] = 5
        self.cache_manager.stats["misses"] = 3
        self.cache_manager.stats["saved_time"] = 10.5
        
        # Get stats
        stats = self.cache_manager.get_cache_stats()
        
        # Check that stats are correct
        self.assertEqual(stats["hits"], 5)
        self.assertEqual(stats["misses"], 3)
        self.assertEqual(stats["hit_rate"], 62.5)  # 5 / (5 + 3) * 100
        self.assertEqual(stats["saved_time"], 10.5)
    
    def test_clear_cache(self):
        """Test clearing the cache."""
        # Add some test data
        transcription = {"text": "Test"}
        analysis = ["Segment"]
        
        self.cache_manager.save_transcription("test.mp3", transcription, processing_time=1.0)
        self.cache_manager.save_analysis("test_hash", analysis, processing_time=0.5)
        
        # Verify data was added
        self.assertEqual(self.cache_manager.get_transcription("test.mp3"), transcription)
        self.assertEqual(self.cache_manager.get_analysis("test_hash"), analysis)
        
        # Clear cache
        self.cache_manager.clear_cache()
        
        # Verify data was removed
        self.assertIsNone(self.cache_manager.get_transcription("test.mp3"))
        self.assertIsNone(self.cache_manager.get_analysis("test_hash"))
        
        # Check that stats were reset
        stats = self.cache_manager.stats
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 2)  # Two misses from the verification gets
        self.assertEqual(stats["saved_time"], 0)
    
    def test_save_and_load_stats(self):
        """Test saving and loading cache statistics."""
        # Set some test stats
        self.cache_manager.stats["hits"] = 10
        self.cache_manager.stats["misses"] = 5
        self.cache_manager.stats["saved_time"] = 20.5
        
        # Save stats
        self.cache_manager._save_stats()
        
        # Create a new cache manager that will load the saved stats
        new_cache = CacheManager(cache_dir=self.temp_dir)
        
        # Check that stats were loaded correctly
        self.assertEqual(new_cache.stats["hits"], 10)
        self.assertEqual(new_cache.stats["misses"], 5)
        self.assertEqual(new_cache.stats["saved_time"], 20.5)
    
    def test_compute_transcription_hash(self):
        """Test computing a hash for transcription."""
        # Create a transcription
        transcription = {
            "text": "Test transcription",
            "segments": [{"text": "Segment 1"}, {"text": "Segment 2"}]
        }
        
        # Compute hash
        hash1 = self.cache_manager._compute_transcription_hash(transcription)
        
        # Check that hash is a non-empty string
        self.assertIsInstance(hash1, str)
        self.assertTrue(len(hash1) > 0)
        
        # Compute hash again and check for consistency
        hash2 = self.cache_manager._compute_transcription_hash(transcription)
        self.assertEqual(hash1, hash2)
        
        # Modify the transcription and check that hash changes
        transcription["text"] = "Modified transcription"
        hash3 = self.cache_manager._compute_transcription_hash(transcription)
        self.assertNotEqual(hash1, hash3)
    
    def test_cache_hit_rate_calculation(self):
        """Test calculation of cache hit rate."""
        # Test with no accesses
        stats = self.cache_manager.get_cache_stats()
        self.assertEqual(stats["hit_rate"], 0)  # No accesses = 0% hit rate
        
        # Test with some hits and misses
        self.cache_manager.stats["hits"] = 3
        self.cache_manager.stats["misses"] = 1
        
        stats = self.cache_manager.get_cache_stats()
        self.assertEqual(stats["hit_rate"], 75)  # 3 / (3 + 1) * 100
        
        # Test with all hits
        self.cache_manager.stats["hits"] = 5
        self.cache_manager.stats["misses"] = 0
        
        stats = self.cache_manager.get_cache_stats()
        self.assertEqual(stats["hit_rate"], 100)  # 5 / (5 + 0) * 100
        
        # Test with all misses
        self.cache_manager.stats["hits"] = 0
        self.cache_manager.stats["misses"] = 5
        
        stats = self.cache_manager.get_cache_stats()
        self.assertEqual(stats["hit_rate"], 0)  # 0 / (0 + 5) * 100


if __name__ == '__main__':
    unittest.main() 