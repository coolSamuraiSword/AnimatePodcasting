"""
Performance Logging and Benchmarking for AnimatePodcasting

This module provides tools for monitoring application performance,
benchmarking operations, and identifying bottlenecks.
"""

import time
import logging
import functools
import json
import os
from typing import Dict, Any, Optional, Callable, List, Union
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """
    Track performance metrics across application components.
    
    This class provides methods to measure execution time of functions and code blocks,
    accumulate statistics, and generate reports on application performance.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize performance tracker.
        
        Args:
            log_file (str, optional): Path to save performance logs
        """
        self.metrics = {
            "function_calls": {},
            "operations": {},
            "benchmarks": []
        }
        
        self.log_file = log_file
        if log_file:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
    
    def track_function(self, func: Callable) -> Callable:
        """
        Decorator to track function execution time.
        
        Args:
            func (callable): Function to track
            
        Returns:
            callable: Wrapped function with performance tracking
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                success = False
                raise e
            finally:
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Record metrics
                func_name = func.__name__
                if func_name not in self.metrics["function_calls"]:
                    self.metrics["function_calls"][func_name] = {
                        "calls": 0,
                        "total_time": 0,
                        "min_time": float('inf'),
                        "max_time": 0,
                        "successful_calls": 0,
                        "failed_calls": 0
                    }
                
                stats = self.metrics["function_calls"][func_name]
                stats["calls"] += 1
                stats["total_time"] += execution_time
                stats["min_time"] = min(stats["min_time"], execution_time)
                stats["max_time"] = max(stats["max_time"], execution_time)
                
                if success:
                    stats["successful_calls"] += 1
                else:
                    stats["failed_calls"] += 1
                
                # Log the call
                logger.debug(f"Function {func_name} executed in {execution_time:.4f}s")
                
                # Save to log file if specified
                if self.log_file:
                    self._append_to_log("function", {
                        "name": func_name,
                        "execution_time": execution_time,
                        "timestamp": datetime.now().isoformat(),
                        "success": success
                    })
            
            return result
        
        return wrapper
    
    @contextmanager
    def track_operation(self, operation_name: str):
        """
        Context manager to track execution time of a code block.
        
        Args:
            operation_name (str): Name of the operation to track
        """
        start_time = time.time()
        success = True
        
        try:
            yield
        except Exception as e:
            success = False
            raise e
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Record metrics
            if operation_name not in self.metrics["operations"]:
                self.metrics["operations"][operation_name] = {
                    "executions": 0,
                    "total_time": 0,
                    "min_time": float('inf'),
                    "max_time": 0,
                    "successful_executions": 0,
                    "failed_executions": 0
                }
            
            stats = self.metrics["operations"][operation_name]
            stats["executions"] += 1
            stats["total_time"] += execution_time
            stats["min_time"] = min(stats["min_time"], execution_time)
            stats["max_time"] = max(stats["max_time"], execution_time)
            
            if success:
                stats["successful_executions"] += 1
            else:
                stats["failed_executions"] += 1
            
            # Log the operation
            logger.debug(f"Operation {operation_name} completed in {execution_time:.4f}s")
            
            # Save to log file if specified
            if self.log_file:
                self._append_to_log("operation", {
                    "name": operation_name,
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat(),
                    "success": success
                })
    
    def benchmark(self, func: Callable, *args, iterations: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Benchmark a function by running it multiple times and measuring performance.
        
        Args:
            func (callable): Function to benchmark
            *args: Arguments to pass to the function
            iterations (int): Number of iterations to run
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            dict: Benchmark results with execution times and statistics
        """
        if iterations < 1:
            raise ValueError("Iterations must be at least 1")
        
        func_name = func.__name__
        execution_times = []
        
        logger.info(f"Starting benchmark of {func_name} with {iterations} iterations")
        
        try:
            # Run warm-up iteration (not counted in results)
            func(*args, **kwargs)
            
            # Run actual benchmark iterations
            for i in range(iterations):
                start_time = time.time()
                func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time
                execution_times.append(execution_time)
                logger.debug(f"Iteration {i+1}/{iterations}: {execution_time:.4f}s")
            
            # Calculate statistics
            avg_time = sum(execution_times) / len(execution_times)
            result = {
                "function": func_name,
                "iterations": iterations,
                "execution_times": execution_times,
                "average_time": avg_time,
                "min_time": min(execution_times),
                "max_time": max(execution_times),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add to metrics
            self.metrics["benchmarks"].append(result)
            
            # Log results
            logger.info(f"Benchmark results for {func_name}: "
                       f"avg={avg_time:.4f}s, min={result['min_time']:.4f}s, "
                       f"max={result['max_time']:.4f}s")
            
            # Save to log file if specified
            if self.log_file:
                self._append_to_log("benchmark", result)
            
            return result
            
        except Exception as e:
            logger.error(f"Benchmark of {func_name} failed: {str(e)}")
            raise
    
    def get_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Returns:
            dict: Performance metrics and statistics
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "function_stats": {},
            "operation_stats": {},
            "benchmark_summary": []
        }
        
        # Process function statistics
        for func_name, stats in self.metrics["function_calls"].items():
            avg_time = stats["total_time"] / stats["calls"] if stats["calls"] > 0 else 0
            
            report["function_stats"][func_name] = {
                "calls": stats["calls"],
                "average_time": avg_time,
                "min_time": stats["min_time"] if stats["min_time"] != float('inf') else 0,
                "max_time": stats["max_time"],
                "total_time": stats["total_time"],
                "success_rate": (stats["successful_calls"] / stats["calls"]) * 100 if stats["calls"] > 0 else 0
            }
        
        # Process operation statistics
        for op_name, stats in self.metrics["operations"].items():
            avg_time = stats["total_time"] / stats["executions"] if stats["executions"] > 0 else 0
            
            report["operation_stats"][op_name] = {
                "executions": stats["executions"],
                "average_time": avg_time,
                "min_time": stats["min_time"] if stats["min_time"] != float('inf') else 0,
                "max_time": stats["max_time"],
                "total_time": stats["total_time"],
                "success_rate": (stats["successful_executions"] / stats["executions"]) * 100 if stats["executions"] > 0 else 0
            }
        
        # Include benchmark summaries
        for benchmark in self.metrics["benchmarks"]:
            report["benchmark_summary"].append({
                "function": benchmark["function"],
                "iterations": benchmark["iterations"],
                "average_time": benchmark["average_time"],
                "min_time": benchmark["min_time"],
                "max_time": benchmark["max_time"],
                "timestamp": benchmark["timestamp"]
            })
        
        return report
    
    def save_report(self, file_path: str):
        """
        Save performance report to a JSON file.
        
        Args:
            file_path (str): Path to save the report
        """
        report = self.get_report()
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Performance report saved to {file_path}")
        
        except Exception as e:
            logger.error(f"Failed to save performance report: {str(e)}")
    
    def reset(self):
        """Reset all performance metrics."""
        self.metrics = {
            "function_calls": {},
            "operations": {},
            "benchmarks": []
        }
        logger.info("Performance metrics reset")
    
    def _append_to_log(self, entry_type: str, data: Dict[str, Any]):
        """
        Append an entry to the performance log file.
        
        Args:
            entry_type (str): Type of entry (function, operation, benchmark)
            data (dict): Entry data
        """
        try:
            with open(self.log_file, 'a') as f:
                log_entry = {
                    "type": entry_type,
                    "data": data
                }
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.warning(f"Failed to write to performance log: {str(e)}")


# Create a default tracker instance for easy import
default_tracker = PerformanceTracker()

# Decorator for tracking function performance using default tracker
def track(func):
    """Decorator for tracking function performance using the default tracker."""
    return default_tracker.track_function(func)

# Context manager for tracking operations
@contextmanager
def track_operation(operation_name: str):
    """Context manager for tracking operations using the default tracker."""
    with default_tracker.track_operation(operation_name):
        yield 