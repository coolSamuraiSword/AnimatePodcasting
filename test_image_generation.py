#!/usr/bin/env python
"""
Test script for image generation to verify our fixes
"""
import os
import logging
from src.generator import ImageGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Create output directory
    output_dir = "outputs/test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the image generator
    logger.info("Initializing image generator...")
    image_gen = ImageGenerator()
    
    # Generate a test image
    test_prompt = "A colorful painting of a podcast microphone with audio waves in the background"
    test_image_path = os.path.join(output_dir, "test_image.png")
    
    logger.info(f"Generating test image with prompt: '{test_prompt}'")
    image = image_gen.generate_image(
        prompt=test_prompt,
        output_path=test_image_path,
        num_inference_steps=20  # Fewer steps for faster testing
    )
    
    # Check if the image was successfully generated
    if os.path.exists(test_image_path):
        file_size = os.path.getsize(test_image_path)
        logger.info(f"Test image successfully generated! File size: {file_size} bytes")
        if file_size < 1000:
            logger.warning("Image file size is suspiciously small, it might still be blank or corrupted")
        else:
            logger.info("Image file size looks normal")
    else:
        logger.error("Failed to generate test image")

if __name__ == "__main__":
    main()
