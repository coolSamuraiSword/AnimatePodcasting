"""
Error Handling and Recovery System for AnimatePodcasting

This module provides robust error handling, automatic recovery strategies,
and detailed logging for debugging and issue resolution.
"""

import os
import sys
import logging
import traceback
import functools
import json
import time
from enum import Enum
from typing import Dict, Any, Optional, Callable, List, Union, Type, Tuple
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Enumeration of error severity levels."""
    INFO = 1       # Informational, non-critical
    WARNING = 2    # Warning, potential issue but operation can continue
    ERROR = 3      # Error, operation failed but application can continue
    CRITICAL = 4   # Critical error, may require application restart
    FATAL = 5      # Fatal error, application cannot continue

class ErrorType(Enum):
    """Enumeration of common error types for categorization."""
    NETWORK = 1        # Network-related errors
    FILE_SYSTEM = 2    # File system errors (permissions, missing files)
    INPUT = 3          # Invalid user input
    MODEL = 4          # AI model-related errors
    DEPENDENCY = 5     # Missing or incompatible dependency
    RESOURCE = 6       # Resource limitation (memory, disk space)
    DATA = 7           # Data processing/validation errors
    CONFIG = 8         # Configuration errors
    UNKNOWN = 9        # Uncategorized errors

class RecoveryStrategy(Enum):
    """Enumeration of recovery strategies."""
    RETRY = 1          # Retry the operation
    ALTERNATE = 2      # Use an alternate method
    SKIP = 3           # Skip the operation and continue
    ROLLBACK = 4       # Rollback to previous state
    USER_INPUT = 5     # Prompt for user input to resolve
    ABORT = 6          # Abort the current operation
    RESTART = 7        # Restart the application component
    NONE = 8           # No recovery possible

class ApplicationError(Exception):
    """
    Base exception class for application-specific errors.
    
    Provides rich context for error handling and recovery.
    """
    
    def __init__(
        self, 
        message: str,
        error_type: ErrorType = ErrorType.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.NONE,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize application error with rich context.
        
        Args:
            message (str): Error message
            error_type (ErrorType): Type of error for categorization
            severity (ErrorSeverity): Severity level
            recovery_strategy (RecoveryStrategy): Suggested recovery strategy
            context (dict, optional): Additional context for debugging
            original_exception (Exception, optional): Original exception if this is a wrapper
        """
        self.message = message
        self.error_type = error_type
        self.severity = severity
        self.recovery_strategy = recovery_strategy
        self.context = context or {}
        self.original_exception = original_exception
        self.timestamp = datetime.now().isoformat()
        
        # Call the base class constructor
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error to dictionary representation.
        
        Returns:
            dict: Dictionary representation of the error
        """
        result = {
            "message": self.message,
            "error_type": self.error_type.name,
            "severity": self.severity.name,
            "recovery_strategy": self.recovery_strategy.name,
            "context": self.context,
            "timestamp": self.timestamp
        }
        
        if self.original_exception:
            result["original_exception"] = {
                "type": type(self.original_exception).__name__,
                "message": str(self.original_exception)
            }
        
        return result
    
    def __str__(self) -> str:
        """
        Get string representation of the error.
        
        Returns:
            str: Formatted error message
        """
        base_msg = f"{self.severity.name}: {self.message}"
        
        if self.original_exception:
            base_msg += f" (Caused by {type(self.original_exception).__name__}: {self.original_exception})"
        
        return base_msg

class ErrorHandler:
    """
    Centralized error handling and recovery system.
    
    Provides methods for handling errors, implementing recovery strategies,
    logging errors for debugging, and generating error reports.
    """
    
    def __init__(
        self, 
        error_log_file: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_error_reporting: bool = True
    ):
        """
        Initialize error handler.
        
        Args:
            error_log_file (str, optional): Path to save error logs
            max_retries (int): Maximum number of retry attempts
            retry_delay (float): Delay between retry attempts in seconds
            enable_error_reporting (bool): Whether to enable error reporting
        """
        self.error_log_file = error_log_file
        if error_log_file:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(error_log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_error_reporting = enable_error_reporting
        self.errors = []  # List of errors encountered
    
    def handle_error(
        self, 
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        recovery_strategy: Optional[RecoveryStrategy] = None
    ) -> Tuple[bool, Optional[Any]]:
        """
        Handle an exception with appropriate logging and recovery.
        
        Args:
            exception (Exception): The exception to handle
            context (dict, optional): Additional context about the error
            recovery_strategy (RecoveryStrategy, optional): Override the default recovery strategy
            
        Returns:
            tuple: (success, result) - Whether recovery was successful and any result
        """
        # Convert to ApplicationError if it's not already
        if not isinstance(exception, ApplicationError):
            if isinstance(exception, FileNotFoundError):
                app_error = ApplicationError(
                    str(exception),
                    error_type=ErrorType.FILE_SYSTEM,
                    severity=ErrorSeverity.ERROR,
                    recovery_strategy=RecoveryStrategy.SKIP,
                    context=context,
                    original_exception=exception
                )
            elif isinstance(exception, PermissionError):
                app_error = ApplicationError(
                    str(exception),
                    error_type=ErrorType.FILE_SYSTEM,
                    severity=ErrorSeverity.ERROR,
                    recovery_strategy=RecoveryStrategy.USER_INPUT,
                    context=context,
                    original_exception=exception
                )
            elif isinstance(exception, (ConnectionError, TimeoutError)):
                app_error = ApplicationError(
                    str(exception),
                    error_type=ErrorType.NETWORK,
                    severity=ErrorSeverity.WARNING,
                    recovery_strategy=RecoveryStrategy.RETRY,
                    context=context,
                    original_exception=exception
                )
            elif isinstance(exception, ValueError):
                app_error = ApplicationError(
                    str(exception),
                    error_type=ErrorType.INPUT,
                    severity=ErrorSeverity.WARNING,
                    recovery_strategy=RecoveryStrategy.USER_INPUT,
                    context=context,
                    original_exception=exception
                )
            elif isinstance(exception, MemoryError):
                app_error = ApplicationError(
                    str(exception),
                    error_type=ErrorType.RESOURCE,
                    severity=ErrorSeverity.CRITICAL,
                    recovery_strategy=RecoveryStrategy.ABORT,
                    context=context,
                    original_exception=exception
                )
            else:
                app_error = ApplicationError(
                    str(exception),
                    context=context,
                    original_exception=exception
                )
        else:
            app_error = exception
        
        # Override recovery strategy if specified
        if recovery_strategy:
            app_error.recovery_strategy = recovery_strategy
        
        # Log the error
        self._log_error(app_error)
        
        # Add to error list
        self.errors.append(app_error)
        
        # Implement recovery strategy
        return self._implement_recovery(app_error)
    
    def _log_error(self, error: ApplicationError):
        """
        Log an error with appropriate severity level.
        
        Args:
            error (ApplicationError): The error to log
        """
        # Determine logging level based on severity
        log_level = logging.INFO
        if error.severity == ErrorSeverity.WARNING:
            log_level = logging.WARNING
        elif error.severity == ErrorSeverity.ERROR:
            log_level = logging.ERROR
        elif error.severity in (ErrorSeverity.CRITICAL, ErrorSeverity.FATAL):
            log_level = logging.CRITICAL
        
        # Create log message
        log_message = f"{error}"
        if error.context:
            context_str = ", ".join(f"{k}={v}" for k, v in error.context.items())
            log_message += f" [Context: {context_str}]"
        
        # Log the error
        logger.log(log_level, log_message)
        
        # If it's an exception, log the stack trace
        if error.original_exception:
            logger.log(log_level, "".join(traceback.format_exception(
                type(error.original_exception), 
                error.original_exception,
                error.original_exception.__traceback__
            )))
        
        # Write to error log file if configured
        if self.error_log_file:
            try:
                with open(self.error_log_file, 'a') as f:
                    json.dump(error.to_dict(), f)
                    f.write('\n')
            except Exception as e:
                logger.warning(f"Failed to write to error log: {e}")
    
    def _implement_recovery(self, error: ApplicationError) -> Tuple[bool, Optional[Any]]:
        """
        Implement the recovery strategy for an error.
        
        Args:
            error (ApplicationError): The error to recover from
            
        Returns:
            tuple: (success, result) - Whether recovery was successful and any result
        """
        strategy = error.recovery_strategy
        
        # Handle different recovery strategies
        if strategy == RecoveryStrategy.NONE:
            return False, None
        
        elif strategy == RecoveryStrategy.RETRY:
            # Retry the operation if context includes the function and args
            if 'function' in error.context and callable(error.context['function']):
                func = error.context['function']
                args = error.context.get('args', [])
                kwargs = error.context.get('kwargs', {})
                
                for attempt in range(self.max_retries):
                    try:
                        logger.info(f"Retry attempt {attempt + 1}/{self.max_retries}...")
                        time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                        result = func(*args, **kwargs)
                        logger.info(f"Retry successful on attempt {attempt + 1}")
                        return True, result
                    except Exception as e:
                        logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
                
                logger.error(f"All {self.max_retries} retry attempts failed")
                return False, None
            else:
                logger.warning("Cannot retry: missing function in context")
                return False, None
        
        elif strategy == RecoveryStrategy.ALTERNATE:
            # Use alternate method if provided
            if 'alternate_function' in error.context and callable(error.context['alternate_function']):
                try:
                    alt_func = error.context['alternate_function']
                    alt_args = error.context.get('alternate_args', [])
                    alt_kwargs = error.context.get('alternate_kwargs', {})
                    
                    logger.info("Attempting alternate method...")
                    result = alt_func(*alt_args, **alt_kwargs)
                    logger.info("Alternate method successful")
                    return True, result
                except Exception as e:
                    logger.error(f"Alternate method failed: {e}")
                    return False, None
            else:
                logger.warning("Cannot use alternate: missing alternate_function in context")
                return False, None
        
        elif strategy == RecoveryStrategy.SKIP:
            logger.info("Skipping failed operation and continuing")
            return True, None
        
        elif strategy == RecoveryStrategy.ABORT:
            logger.warning("Aborting current operation")
            return False, None
        
        elif strategy == RecoveryStrategy.USER_INPUT:
            # This would require interaction, which depends on the application's UI
            logger.info("Recovery requires user input (not implemented in this context)")
            return False, None
        
        # Other strategies would require more complex implementation
        logger.warning(f"Recovery strategy {strategy.name} not fully implemented")
        return False, None
    
    def with_error_handling(
        self,
        error_types: Optional[List[Type[Exception]]] = None,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
        error_type: ErrorType = ErrorType.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR
    ):
        """
        Decorator for functions that need error handling.
        
        Args:
            error_types (list): List of exception types to catch
            recovery_strategy (RecoveryStrategy): Recovery strategy to use
            error_type (ErrorType): Type of error for categorization
            severity (ErrorSeverity): Severity level
            
        Returns:
            callable: Decorated function with error handling
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Set up error context
                context = {
                    "function": func,
                    "function_name": func.__name__,
                    "args": args,
                    "kwargs": kwargs
                }
                
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check if we should handle this error type
                    if error_types is not None and not any(isinstance(e, t) for t in error_types):
                        raise
                    
                    # Create ApplicationError
                    app_error = ApplicationError(
                        f"Error in {func.__name__}: {str(e)}",
                        error_type=error_type,
                        severity=severity,
                        recovery_strategy=recovery_strategy,
                        context=context,
                        original_exception=e
                    )
                    
                    # Handle the error
                    success, result = self.handle_error(app_error)
                    
                    if success:
                        return result
                    
                    # If recovery failed and it's a critical or fatal error, re-raise
                    if severity in (ErrorSeverity.CRITICAL, ErrorSeverity.FATAL):
                        raise
                    
                    # For non-critical errors, we might want to return a default value
                    # This could be customized based on the function's signature
                    return None
            
            return wrapper
        
        return decorator
    
    @contextmanager
    def error_context(
        self,
        context_name: str,
        error_types: Optional[List[Type[Exception]]] = None,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
        error_type: ErrorType = ErrorType.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for error handling.
        
        Args:
            context_name (str): Name of the context (for logging)
            error_types (list): List of exception types to catch
            recovery_strategy (RecoveryStrategy): Recovery strategy to use
            error_type (ErrorType): Type of error for categorization
            severity (ErrorSeverity): Severity level
            context (dict): Additional context information
        """
        error_context = context or {}
        error_context["context_name"] = context_name
        
        try:
            yield
        except Exception as e:
            # Check if we should handle this error type
            if error_types is not None and not any(isinstance(e, t) for t in error_types):
                raise
            
            # Create ApplicationError
            app_error = ApplicationError(
                f"Error in {context_name}: {str(e)}",
                error_type=error_type,
                severity=severity,
                recovery_strategy=recovery_strategy,
                context=error_context,
                original_exception=e
            )
            
            # Handle the error
            success, _ = self.handle_error(app_error)
            
            # If recovery failed and it's a critical or fatal error, re-raise
            if not success and severity in (ErrorSeverity.CRITICAL, ErrorSeverity.FATAL):
                raise
    
    def get_error_report(self) -> Dict[str, Any]:
        """
        Generate an error report with all encountered errors.
        
        Returns:
            dict: Error report with statistics and error details
        """
        error_counts = {
            error_type.name: 0 for error_type in ErrorType
        }
        
        severity_counts = {
            severity.name: 0 for severity in ErrorSeverity
        }
        
        recovery_counts = {
            recovery.name: 0 for recovery in RecoveryStrategy
        }
        
        # Count errors by type and severity
        for error in self.errors:
            error_counts[error.error_type.name] += 1
            severity_counts[error.severity.name] += 1
            recovery_counts[error.recovery_strategy.name] += 1
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_errors": len(self.errors),
            "error_counts_by_type": error_counts,
            "error_counts_by_severity": severity_counts,
            "recovery_strategy_counts": recovery_counts,
            "errors": [error.to_dict() for error in self.errors]
        }
    
    def save_error_report(self, file_path: str):
        """
        Save error report to a JSON file.
        
        Args:
            file_path (str): Path to save the report
        """
        report = self.get_error_report()
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Error report saved to {file_path}")
        
        except Exception as e:
            logger.error(f"Failed to save error report: {str(e)}")
    
    def clear_errors(self):
        """Clear the list of errors."""
        self.errors = []
        logger.info("Error list cleared")


# Create a default error handler instance for easy import
default_handler = ErrorHandler(error_log_file="logs/errors.log")

# Decorator for error handling using default handler
def handle_errors(
    error_types: Optional[List[Type[Exception]]] = None,
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
    error_type: ErrorType = ErrorType.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.ERROR
):
    """Decorator for error handling using the default handler."""
    return default_handler.with_error_handling(
        error_types=error_types,
        recovery_strategy=recovery_strategy,
        error_type=error_type,
        severity=severity
    )

# Context manager for error handling
@contextmanager
def error_context(
    context_name: str,
    error_types: Optional[List[Type[Exception]]] = None,
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
    error_type: ErrorType = ErrorType.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    context: Optional[Dict[str, Any]] = None
):
    """Context manager for error handling using the default handler."""
    with default_handler.error_context(
        context_name=context_name,
        error_types=error_types,
        recovery_strategy=recovery_strategy,
        error_type=error_type,
        severity=severity,
        context=context
    ):
        yield 