"""
Tests for the cache manager module.
"""

import unittest
import os
import sys
import json
import shutil
import tempfile
from unittest.mock import patch, MagicMock, call
import pathlib

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
    
    @patch('src.cache_manager.CacheManager._get_file_hash')
    def test_get_transcription_not_cached(self, mock_get_hash):
        """Test retrieving a transcription that's not in the cache."""
        # Mock the hash function to avoid file not found error
        mock_get_hash.return_value = "test_hash"
        
        result = self.cache_manager.get_transcription("nonexistent.mp3")
        
        # Check that None is returned for non-cached file
        self.assertIsNone(result)
        
        # Check that stats were updated
        self.assertEqual(self.cache_manager.stats["misses"], 1)
    
    @patch('src.cache_manager.CacheManager._get_file_hash')
    @patch('src.cache_manager.Path.exists')
    @patch('builtins.open')
    def test_save_and_get_transcription(self, mock_open, mock_exists, mock_get_hash):
        """Test saving and retrieving a transcription."""
        # Create a test transcription
        transcription = {
            "text": "Test transcription",
            "segments": [{"text": "Test segment"}]
        }
        
        # Mock the hash function and file operations
        mock_get_hash.return_value = "test_hash"
        mock_exists.return_value = True
        
        # Mock file open for reading JSON
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_file.read.return_value = json.dumps({
            "metadata": {"processing_time": 1.5},
            "data": transcription
        })
        
        # Save and retrieve transcription
        self.cache_manager.save_transcription("test.mp3", transcription, processing_time=1.5)
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
        self.assertEqual(self.cache_manager.stats["misses"], 1)
    
    @patch('src.cache_manager.Path.exists')
    @patch('builtins.open')
    def test_save_and_get_analysis(self, mock_open, mock_exists):
        """Test saving and retrieving analysis."""
        # Create test data
        analysis = ["Segment 1", "Segment 2"]
        
        # Mock file operations
        mock_exists.return_value = True
        
        # Mock file open for reading JSON
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_file.read.return_value = json.dumps({
            "metadata": {"processing_time": 0.5},
            "data": analysis
        })
        
        # Save and retrieve analysis
        self.cache_manager.save_analysis("test_hash", analysis, processing_time=0.5)
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
        self.assertEqual(stats["hit_rate"], "62.50%") # String format in actual implementation
        self.assertEqual(stats["saved_time"], "10.50 seconds") # String format in actual implementation
    
    def test_clear_cache(self):
        """Test clearing the cache."""
        # Add some test data to stats
        self.cache_manager.stats['hits'] = 10
        self.cache_manager.stats['misses'] = 5
        self.cache_manager.stats['saved_time'] = 20.5
        
        # Clear the cache
        self.cache_manager.clear_cache()
        
        # Verify that stats were reset
        self.assertEqual(self.cache_manager.stats['hits'], 0)
        self.assertEqual(self.cache_manager.stats['misses'], 0)
        self.assertEqual(self.cache_manager.stats['saved_time'], 0)
    
    def test_save_and_load_stats(self):
        """Test saving and loading cache statistics."""
        # Create a temp directory that will persist through test
        persistent_temp_dir = tempfile.mkdtemp()
        try:
            # Create a cache manager in the persistent directory
            cache_mgr = CacheManager(cache_dir=persistent_temp_dir)
            
            # Set some test stats
            cache_mgr.stats["hits"] = 10
            cache_mgr.stats["misses"] = 5
            cache_mgr.stats["saved_time"] = 20.5
            
            # Save stats
            cache_mgr._save_stats()
            
            # Create a new cache manager that will load the saved stats
            new_cache = CacheManager(cache_dir=persistent_temp_dir)
            
            # Check that stats were loaded correctly
            self.assertEqual(new_cache.stats["hits"], 10)
            self.assertEqual(new_cache.stats["misses"], 5)
            self.assertEqual(new_cache.stats["saved_time"], 20.5)
        finally:
            # Clean up
            shutil.rmtree(persistent_temp_dir)
    
    @patch('src.cache_manager.CacheManager._get_file_hash')
    def test_cache_hit_rate_calculation(self, mock_get_hash):
        """Test calculation of cache hit rate."""
        # Mock the hash function to avoid file not found error
        mock_get_hash.return_value = "test_hash"
        
        # Test with no accesses
        stats = self.cache_manager.get_cache_stats()
        self.assertEqual(stats["hit_rate"], "0.00%")  # String format in actual implementation
        
        # Test with some hits and misses
        self.cache_manager.stats["hits"] = 3
        self.cache_manager.stats["misses"] = 1
        
        stats = self.cache_manager.get_cache_stats()
        self.assertEqual(stats["hit_rate"], "75.00%")  # String format in actual implementation
        
        # Test with all hits
        self.cache_manager.stats["hits"] = 5
        self.cache_manager.stats["misses"] = 0
        
        stats = self.cache_manager.get_cache_stats()
        self.assertEqual(stats["hit_rate"], "100.00%")  # String format in actual implementation
        
        # Test with all misses
        self.cache_manager.stats["hits"] = 0
        self.cache_manager.stats["misses"] = 5
        
        stats = self.cache_manager.get_cache_stats()
        self.assertEqual(stats["hit_rate"], "0.00%")  # String format in actual implementation


if __name__ == '__main__':
    unittest.main() 