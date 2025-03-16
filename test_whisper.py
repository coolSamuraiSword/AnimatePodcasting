#!/usr/bin/env python3
import os
import whisper
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Audio file path
    audio_path = "data/sample_audio1.mp3"
    
    # Ensure the file exists
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return
    
    logger.info(f"Testing Whisper transcription on {audio_path}")
    
    # Load model (using CPU for stability)
    logger.info("Loading Whisper 'tiny' model (smallest model for quick testing)...")
    model = whisper.load_model("tiny", device="cpu")
    logger.info("Model loaded successfully")
    
    # Transcribe a short segment
    logger.info("Transcribing...")
    result = model.transcribe(audio_path, fp16=False)
    
    # Print the result
    logger.info("Transcription complete!")
    logger.info(f"Transcription text: {result['text'][:200]}...")

if __name__ == "__main__":
    main()
