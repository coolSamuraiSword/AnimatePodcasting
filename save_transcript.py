#!/usr/bin/env python
"""
Save Whisper transcription to a text file
"""
import os
import logging
from src.transcriber import WhisperTranscriber

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Configure file paths
    audio_path = "data/CattleRaid.mp3" 
    output_file = "outputs/transcript.txt"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Initialize transcriber and transcribe audio
    logger.info(f"Transcribing {audio_path}...")
    transcriber = WhisperTranscriber(model_size="base")
    transcription = transcriber.transcribe(audio_path)
    
    # Get full text
    full_text = transcription.get("text", "")
    
    # Write transcript to file
    with open(output_file, "w") as f:
        f.write(full_text)
    
    logger.info(f"Transcript saved to {output_file}")
    
    # Optional: Print segments information
    logger.info("Segment information:")
    for i, segment in enumerate(transcription.get("segments", [])[:5]):
        start_time = segment.get("start", 0)
        end_time = segment.get("end", 0)
        text = segment.get("text", "")
        logger.info(f"Segment {i+1}: [{start_time:.2f}s - {end_time:.2f}s] {text[:50]}...")
    
    if len(transcription.get("segments", [])) > 5:
        logger.info(f"... and {len(transcription.get('segments', [])) - 5} more segments")

if __name__ == "__main__":
    main()
