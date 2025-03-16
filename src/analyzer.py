"""
Content Analyzer module for AnimatePodcasting

This module handles text analysis and key segment extraction using SentenceTransformers.
"""

import logging
import hashlib
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional

# Import the cache manager
try:
    from .cache_manager import CacheManager
except ImportError:
    from cache_manager import CacheManager

logger = logging.getLogger(__name__)

class ContentAnalyzer:
    """Text analysis and key segment extraction using SentenceTransformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_cache: bool = True):
        """
        Initialize the content analyzer with the specified model.
        
        Args:
            model_name (str): Name of the SentenceTransformer model to use
            use_cache (bool): Whether to use cache for analysis results
        """
        # Initialize cache manager if caching is enabled
        self.use_cache = use_cache
        if use_cache:
            self.cache = CacheManager()
        
        start_time = time.time()
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name)
            # Ensure the model is using the fastest available device
            self.model.to(self.model.device)
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully on {self.model.device} in {load_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {str(e)}")
            raise
    
    def _compute_transcription_hash(self, transcription: Dict[str, Any]) -> str:
        """
        Compute a hash for the transcription data for caching purposes.
        
        Args:
            transcription (dict): Whisper transcription result
            
        Returns:
            str: Hash string for the transcription
        """
        # Create a hash based on the transcript text
        text = transcription.get("text", "")
        hasher = hashlib.md5()
        hasher.update(text.encode())
        return hasher.hexdigest()
    
    def extract_key_segments(self, transcription: Dict[str, Any], top_n: int = 5) -> List[str]:
        """
        Extract key segments from a transcription based on semantic relevance.
        
        Args:
            transcription (dict): Whisper transcription result
            top_n (int): Number of key segments to extract
            
        Returns:
            list: List of key segment texts
        """
        # Early return for empty transcription
        segments = transcription.get("segments", [])
        if not segments:
            logger.warning("No segments found in transcription")
            return []
        
        # Check cache first if enabled
        if self.use_cache:
            # Compute hash for the transcription
            transcription_hash = self._compute_transcription_hash(transcription)
            # Look for cached analysis result
            cached_analysis = self.cache.get_analysis(transcription_hash)
            if cached_analysis and "key_segments" in cached_analysis:
                logger.info("Using cached analysis results")
                return cached_analysis["key_segments"]
        
        start_time = time.time()
        
        # Extract segment texts
        segment_texts = [segment["text"].strip() for segment in segments]
        
        # Filter out very short segments
        filtered_segments = [text for text in segment_texts if len(text.split()) > 3]
        
        if not filtered_segments:
            logger.warning("No substantial segments found after filtering")
            return segment_texts[:top_n]  # Return original segments if no substantial ones
        
        # If we have fewer segments than requested, return all
        if len(filtered_segments) <= top_n:
            return filtered_segments
        
        # Encode segments for clustering
        embeddings = self.model.encode(filtered_segments, convert_to_tensor=True)
        
        # Use a simple clustering approach to find diverse key segments
        selected_indices = self._mmr_selection(embeddings, top_n)
        
        # Get the selected segments
        key_segments = [filtered_segments[i] for i in selected_indices]
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Extracted {len(key_segments)} key segments in {processing_time:.2f}s")
        
        # Cache the result if caching is enabled
        if self.use_cache:
            transcription_hash = self._compute_transcription_hash(transcription)
            analysis_data = {
                "key_segments": key_segments,
                "segment_count": len(segments),
                "filtered_count": len(filtered_segments)
            }
            self.cache.save_analysis(transcription_hash, analysis_data, processing_time)
        
        return key_segments
    
    def _mmr_selection(self, embeddings, top_n, lambda_param=0.5):
        """
        Maximal Marginal Relevance selection to select diverse and relevant segments.
        
        Args:
            embeddings: Tensor of segment embeddings
            top_n: Number of segments to select
            lambda_param: Trade-off between relevance and diversity (0-1)
            
        Returns:
            list: List of selected indices
        """
        # Convert embeddings to numpy for easier manipulation
        embeddings_np = embeddings.cpu().numpy()
        
        # Calculate cosine similarity matrix
        similarities = np.dot(embeddings_np, embeddings_np.T)
        
        # Normalize similarities to [0, 1]
        similarities = (similarities + 1) / 2
        
        # Calculate relevance scores (similarity to the average embedding)
        avg_embedding = np.mean(embeddings_np, axis=0, keepdims=True)
        relevance_scores = np.dot(embeddings_np, avg_embedding.T).flatten()
        
        # Initialize selection
        selected_indices = []
        unselected_indices = list(range(len(embeddings_np)))
        
        # Select the most relevant segment first
        first_idx = np.argmax(relevance_scores)
        selected_indices.append(first_idx)
        unselected_indices.remove(first_idx)
        
        # Select remaining segments
        for _ in range(min(top_n - 1, len(unselected_indices))):
            mmr_scores = []
            
            for idx in unselected_indices:
                # Relevance term
                relevance = relevance_scores[idx]
                
                # Diversity term (negative of maximum similarity to already selected)
                max_sim = max([similarities[idx, sel_idx] for sel_idx in selected_indices])
                diversity = -max_sim
                
                # MMR score as weighted sum
                mmr = lambda_param * relevance + (1 - lambda_param) * diversity
                mmr_scores.append((idx, mmr))
            
            # Select segment with highest MMR score
            selected_idx, _ = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(selected_idx)
            unselected_indices.remove(selected_idx)
        
        return selected_indices
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze the sentiment of a text (placeholder for future functionality).
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment analysis result
        """
        # This is a placeholder for future sentiment analysis functionality
        # Could be implemented using HuggingFace Transformers sentiment models
        logger.info("Sentiment analysis not yet implemented")
        return {"positive": 0.5, "negative": 0.5}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the analysis cache.
        
        Returns:
            dict: Cache statistics or None if caching is disabled
        """
        if not self.use_cache:
            return {"status": "Caching disabled"}
        
        return self.cache.get_cache_stats()
    
    def clear_cache(self):
        """Clear the analysis cache."""
        if self.use_cache:
            self.cache.clear_cache("analysis")
            logger.info("Analysis cache cleared")
        else:
            logger.warning("Caching is disabled, no cache to clear")
