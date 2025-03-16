#!/usr/bin/env python
"""
AnimatePodcasting - Main application file

This script demonstrates the core functionality of the AnimatePodcasting project:
1. Transcribe audio using OpenAI Whisper
2. Analyze content using SentenceTransformers
3. Generate images using Diffusers
4. Create video with ffmpeg
5. Sync with Notion project page
"""

import os
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Fix imports to work when running from project directory
try:
    from src.transcriber import WhisperTranscriber
    from src.analyzer import ContentAnalyzer
    from src.generator import ImageGenerator
    from src.project_notion import ProjectNotionManager
except ModuleNotFoundError:
    # When running from within src directory
    from transcriber import WhisperTranscriber
    from analyzer import ContentAnalyzer
    from generator import ImageGenerator
    from project_notion import ProjectNotionManager

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AnimatePodcasting - Create animated videos from podcast audio')
    parser.add_argument('--audio', type=str, help='Path to audio file')
    parser.add_argument('--output', type=str, default='outputs/output.mp4', 
                        help='Path to output video file')
    parser.add_argument('--model', type=str, default='base', 
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Custom prompt for image generation')
    parser.add_argument('--num-images', type=int, default=5,
                        help='Number of images to generate')
    parser.add_argument('--style', type=str, default=None,
                        help='Animation style to use for image generation')
    parser.add_argument('--setup-notion', action='store_true',
                        help='Set up or update the Notion project page structure')
    
    return parser.parse_args()

def main():
    """Main function to orchestrate the podcast animation process."""
    try:
        args = parse_args()
        
        # Initialize Notion project manager
        try:
            notion = ProjectNotionManager()
            if args.setup_notion:
                notion.setup_project_page()
        except Exception as e:
            logger.error(f"Failed to initialize Notion integration: {str(e)}")
            logger.info("Continuing without Notion integration...")
            notion = None
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output)
        os.makedirs(output_dir, exist_ok=True)
        
        if args.audio:
            audio_path = args.audio
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return
        else:
            # For demo purposes, use a sample audio file
            logger.info("No audio file provided, using sample audio for demonstration")
            sample_dir = Path(__file__).parent.parent / "data"
            os.makedirs(sample_dir, exist_ok=True)
            audio_path = sample_dir / "sample.mp3"
            
            if not os.path.exists(audio_path):
                logger.warning(f"Sample audio file not found at {audio_path}")
                logger.info("Please provide an audio file with --audio option")
                return
        
        # Step 1: Transcribe audio
        logger.info("Transcribing audio...")
        logger.info(f"Using audio file: {audio_path}")
        transcriber = WhisperTranscriber(model_size=args.model)
        transcription = transcriber.transcribe(audio_path)
        
        # Log transcription activity to Notion
        if notion:
            notion.log_processing_activity(
                "transcription",
                {
                    "Audio File": os.path.basename(audio_path),
                    "Model": args.model,
                    "Duration": f"{transcription.get('segments', [])[-1].get('end', 0):.2f}s"
                }
            )
        
        # Display a small sample of the transcription
        if transcription and "text" in transcription:
            text_sample = transcription["text"][:200] + "..." if len(transcription["text"]) > 200 else transcription["text"]
            logger.info(f"Transcription sample: {text_sample}")
        
        # Step 2: Analyze content
        logger.info("Analyzing content...")
        analyzer = ContentAnalyzer()
        key_segments = analyzer.extract_key_segments(transcription)
        
        # Log analysis activity to Notion
        if notion:
            notion.log_processing_activity(
                "content_analysis",
                {
                    "Key Segments Found": len(key_segments),
                    "Sample Segment": key_segments[0][:100] + "..." if key_segments else "None"
                }
            )
        
        # Step 3: Generate images
        logger.info("Generating images...")
        generator = ImageGenerator(animation_style=args.style)
        image_paths = []
        
        for i, segment in enumerate(key_segments[:args.num_images]):
            prompt = args.prompt if args.prompt else segment
            image_path = os.path.join(output_dir, f"frame_{i:03d}.png")
            
            logger.info(f"Generating image {i+1}/{args.num_images}")
            logger.info(f"Prompt: {prompt[:100]}...")
            
            image = generator.generate_image(prompt)
            image.save(image_path)
            image_paths.append(image_path)
            
            # Log image generation to Notion
            if notion:
                notion.log_processing_activity(
                    "image_generation",
                    {
                        "Image": f"frame_{i:03d}.png",
                        "Style": args.style or "default",
                        "Prompt": prompt[:100] + "..."
                    }
                )
            
            logger.info(f"Image saved to: {image_path}")
        
        # Step 4: Create video with ffmpeg
        if image_paths:
            logger.info(f"Creating video at {args.output}...")
            generator.create_video(audio_path, image_paths, args.output)
            
            # Log video creation to Notion
            if notion:
                notion.log_processing_activity(
                    "video_creation",
                    {
                        "Output": os.path.basename(args.output),
                        "Images Used": len(image_paths),
                        "Audio": os.path.basename(audio_path)
                    }
                )
            
            logger.info("Video creation complete!")
        
        # Update project status in Notion
        if notion:
            notion.update_project_status({
                "Last Processing": "Complete",
                "Audio Processed": os.path.basename(audio_path),
                "Images Generated": len(image_paths),
                "Video Output": os.path.basename(args.output),
                "Model Used": args.model,
                "Animation Style": args.style or "default"
            })
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        if notion:
            notion.log_processing_activity(
                "error",
                {
                    "Error Type": type(e).__name__,
                    "Error Message": str(e)
                }
            )
        raise

if __name__ == "__main__":
    print("AnimatePodcasting - Create animated videos from podcast audio")
    print("=" * 60)
    print("This is a demo application showing the integration of:")
    print("- OpenAI Whisper for audio transcription")
    print("- SentenceTransformers for content analysis")
    print("- Diffusers for image generation")
    print("- FFmpeg for video creation")
    print("- Notion API for project tracking")
    print("=" * 60)
    main()
