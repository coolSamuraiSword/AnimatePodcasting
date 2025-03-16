"""
Tests for the performance tracking module.
"""

import unittest
import time
import os
import json
from unittest.mock import patch, MagicMock
import tempfile
import sys

# Add the parent directory to the path so we can import the src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.performance import PerformanceTracker, track, track_operation


class TestPerformanceTracker(unittest.TestCase):
    """Test cases for the PerformanceTracker class."""
    
    def setUp(self):
        """Set up for tests."""
        self.tracker = PerformanceTracker()
    
    def tearDown(self):
        """Clean up after tests."""
        self.tracker = None
    
    def test_track_function(self):
        """Test tracking function execution."""
        @self.tracker.track_function
        def test_func(x, y):
            time.sleep(0.01)  # Small delay for consistent timing
            return x + y
        
        # Call the function
        result = test_func(5, 3)
        
        # Check the result
        self.assertEqual(result, 8)
        
        # Check that metrics were recorded
        self.assertIn("test_func", self.tracker.metrics["function_calls"])
        stats = self.tracker.metrics["function_calls"]["test_func"]
        self.assertEqual(stats["calls"], 1)
        self.assertGreater(stats["total_time"], 0)
        self.assertEqual(stats["successful_calls"], 1)
        self.assertEqual(stats["failed_calls"], 0)
    
    def test_track_function_exception(self):
        """Test tracking a function that raises an exception."""
        @self.tracker.track_function
        def failing_func():
            time.sleep(0.01)  # Small delay for consistent timing
            raise ValueError("Test exception")
        
        # Call the function and expect an exception
        with self.assertRaises(ValueError):
            failing_func()
        
        # Check that metrics were recorded
        self.assertIn("failing_func", self.tracker.metrics["function_calls"])
        stats = self.tracker.metrics["function_calls"]["failing_func"]
        self.assertEqual(stats["calls"], 1)
        self.assertGreater(stats["total_time"], 0)
        self.assertEqual(stats["successful_calls"], 0)
        self.assertEqual(stats["failed_calls"], 1)
    
    def test_track_operation(self):
        """Test tracking operation execution."""
        with self.tracker.track_operation("test_operation"):
            time.sleep(0.01)  # Small delay for consistent timing
        
        # Check that metrics were recorded
        self.assertIn("test_operation", self.tracker.metrics["operations"])
        stats = self.tracker.metrics["operations"]["test_operation"]
        self.assertEqual(stats["executions"], 1)
        self.assertGreater(stats["total_time"], 0)
        self.assertEqual(stats["successful_executions"], 1)
        self.assertEqual(stats["failed_executions"], 0)
    
    def test_track_operation_exception(self):
        """Test tracking an operation that raises an exception."""
        with self.assertRaises(ValueError):
            with self.tracker.track_operation("failing_operation"):
                time.sleep(0.01)  # Small delay for consistent timing
                raise ValueError("Test exception")
        
        # Check that metrics were recorded
        self.assertIn("failing_operation", self.tracker.metrics["operations"])
        stats = self.tracker.metrics["operations"]["failing_operation"]
        self.assertEqual(stats["executions"], 1)
        self.assertGreater(stats["total_time"], 0)
        self.assertEqual(stats["successful_executions"], 0)
        self.assertEqual(stats["failed_executions"], 1)
    
    def test_benchmark(self):
        """Test benchmarking a function."""
        def bench_func():
            time.sleep(0.01)  # Small delay for consistent timing
            return True
        
        # Run benchmark with 3 iterations
        result = self.tracker.benchmark(bench_func, iterations=3)
        
        # Check result structure
        self.assertEqual(result["function"], "bench_func")
        self.assertEqual(result["iterations"], 3)
        self.assertEqual(len(result["execution_times"]), 3)
        self.assertGreater(result["average_time"], 0)
        self.assertGreater(result["min_time"], 0)
        self.assertGreater(result["max_time"], 0)
        
        # Check that metrics were recorded
        self.assertEqual(len(self.tracker.metrics["benchmarks"]), 1)
    
    def test_get_report(self):
        """Test generating a performance report."""
        # Add some test data
        @self.tracker.track_function
        def test_func():
            time.sleep(0.01)
            return True
        
        test_func()
        
        with self.tracker.track_operation("test_op"):
            time.sleep(0.01)
        
        # Get the report
        report = self.tracker.get_report()
        
        # Check report structure
        self.assertIn("timestamp", report)
        self.assertIn("function_stats", report)
        self.assertIn("operation_stats", report)
        self.assertIn("benchmark_summary", report)
        
        # Check function stats
        self.assertIn("test_func", report["function_stats"])
        self.assertEqual(report["function_stats"]["test_func"]["calls"], 1)
        
        # Check operation stats
        self.assertIn("test_op", report["operation_stats"])
        self.assertEqual(report["operation_stats"]["test_op"]["executions"], 1)
    
    def test_save_report(self):
        """Test saving a performance report to file."""
        # Add some test data
        @self.tracker.track_function
        def test_func():
            time.sleep(0.01)
            return True
        
        test_func()
        
        # Create a temporary file for the report
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            report_path = temp_file.name
        
        try:
            # Save the report
            self.tracker.save_report(report_path)
            
            # Check that the file exists and has content
            self.assertTrue(os.path.exists(report_path))
            self.assertGreater(os.path.getsize(report_path), 0)
            
            # Load and check the report
            with open(report_path, 'r') as f:
                saved_report = json.load(f)
            
            self.assertIn("function_stats", saved_report)
            self.assertIn("test_func", saved_report["function_stats"])
        
        finally:
            # Clean up
            if os.path.exists(report_path):
                os.remove(report_path)
    
    def test_reset(self):
        """Test resetting performance metrics."""
        # Add some test data
        @self.tracker.track_function
        def test_func():
            return True
        
        test_func()
        
        # Verify data was added
        self.assertIn("test_func", self.tracker.metrics["function_calls"])
        
        # Reset metrics
        self.tracker.reset()
        
        # Verify data was cleared
        self.assertEqual(self.tracker.metrics["function_calls"], {})
        self.assertEqual(self.tracker.metrics["operations"], {})
        self.assertEqual(self.tracker.metrics["benchmarks"], [])


class TestDecorators(unittest.TestCase):
    """Test cases for the decorator functions."""
    
    def setUp(self):
        """Set up for tests."""
        # Clear the default tracker metrics
        from src.performance import default_tracker
        default_tracker.reset()
    
    def test_track_decorator(self):
        """Test the track decorator."""
        @track
        def decorated_func():
            time.sleep(0.01)
            return "result"
        
        # Call the decorated function
        result = decorated_func()
        
        # Check the result
        self.assertEqual(result, "result")
        
        # Check that metrics were recorded in the default tracker
        from src.performance import default_tracker
        self.assertIn("decorated_func", default_tracker.metrics["function_calls"])
    
    def test_track_operation_context_manager(self):
        """Test the track_operation context manager."""
        with track_operation("test_context_op"):
            time.sleep(0.01)
        
        # Check that metrics were recorded in the default tracker
        from src.performance import default_tracker
        self.assertIn("test_context_op", default_tracker.metrics["operations"])


if __name__ == '__main__':
    unittest.main() 