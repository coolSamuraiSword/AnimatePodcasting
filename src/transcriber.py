"""
Transcriber module for AnimatePodcasting

This module handles audio transcription using OpenAI's Whisper model,
optimized for Apple Silicon M4 Pro.
"""

import os
import logging
import time
from typing import Dict, Any, Optional

import whisper
import torch

# Import the cache manager
try:
    from .cache_manager import CacheManager
except ImportError:
    from cache_manager import CacheManager

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    """Audio transcription using OpenAI's Whisper model."""
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None, use_cache: bool = True):
        """
        Initialize the transcriber with the specified model.
        
        Args:
            model_size (str): Size of the Whisper model ('tiny', 'base', 'small', 'medium', 'large')
            device (str, optional): Device to use for inference ('cpu', 'cuda', etc.)
            use_cache (bool): Whether to use cache for transcriptions
        """
        # Initialize cache manager if caching is enabled
        self.use_cache = use_cache
        if use_cache:
            self.cache = CacheManager()
        
        # Model loading is an expensive operation, so we log it
        logger.info(f"Loading Whisper model: {model_size}")
        
        # Set up the device
        self.device = device
        start_time = time.time()
        
        try:
            self.model = whisper.load_model(model_size, device=device)
            load_time = time.time() - start_time
            logger.info(f"Whisper model '{model_size}' loaded in {load_time:.2f}s")
            self.model_size = model_size
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            raise
    
    def transcribe(self, audio_path: str, language: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
        """
        Transcribe the audio file using Whisper, with caching.
        
        Args:
            audio_path (str): Path to the audio file
            language (str): Language code (e.g., 'en' for English)
            verbose (bool): Whether to print progress information
            
        Returns:
            dict: The transcription result with text, segments, etc.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Check cache first if enabled
        if self.use_cache:
            cached_transcription = self.cache.get_transcription(audio_path, self.model_size)
            if cached_transcription:
                logger.info(f"Using cached transcription for {audio_path}")
                return cached_transcription
        
        logger.info(f"Transcribing {audio_path}...")
        options = {
            "verbose": verbose,
        }
        
        if language:
            options["language"] = language
        
        start_time = time.time()
        
        try:
            result = self.model.transcribe(audio_path, **options)
            
            processing_time = time.time() - start_time
            logger.info(f"Transcription completed in {processing_time:.2f}s")
            
            # Cache the result if caching is enabled
            if self.use_cache:
                self.cache.save_transcription(
                    audio_path, 
                    result, 
                    self.model_size, 
                    processing_time
                )
            
            return result
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise
    
    def get_text(self, result: Dict[str, Any]) -> str:
        """
        Extract just the text from the transcription result.
        
        Args:
            result (dict): The transcription result
            
        Returns:
            str: The transcribed text
        """
        return result.get("text", "")

    def get_segments(self, result: Dict[str, Any]) -> list:
        """
        Extract the segments from the transcription result.
        
        Args:
            result (dict): The transcription result
            
        Returns:
            list: The transcribed segments with timing information
        """
        return result.get("segments", [])
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the transcription cache.
        
        Returns:
            dict: Cache statistics or None if caching is disabled
        """
        if not self.use_cache:
            return {"status": "Caching disabled"}
        
        return self.cache.get_cache_stats()
    
    def clear_cache(self):
        """Clear the transcription cache."""
        if self.use_cache:
            self.cache.clear_cache("transcription")
            logger.info("Transcription cache cleared")
        else:
            logger.warning("Caching is disabled, no cache to clear")
