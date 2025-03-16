"""
Image Generator module for AnimatePodcasting

This module handles image generation using Diffusers and video creation using ffmpeg.
"""

import os
import logging
import subprocess
import torch
from diffusers import StableDiffusionPipeline

logger = logging.getLogger(__name__)

class ImageGenerator:
    """Image generation using Diffusers and video creation using ffmpeg."""
    
    # Define available animation styles
    ANIMATION_STYLES = {
        "cartoon": "cartoon style, vibrant colors, simple shapes, bold outlines",
        "anime": "anime style, Japanese animation, detailed characters, expressive eyes",
        "pixel_art": "pixel art style, 8-bit graphics, retro gaming aesthetic",
        "3d_animation": "3D animation style, Pixar-like, smooth surfaces, depth",
        "watercolor": "watercolor animation style, soft edges, flowing colors, artistic",
        "clay_animation": "claymation style, stop motion, textured surfaces",
        "line_drawing": "animated line drawing, simple black lines on white background",
        "vector": "vector art animation, clean lines, flat colors, geometric shapes"
    }
    
    @classmethod
    def get_available_styles(cls):
        """Return a list of available animation styles."""
        return list(cls.ANIMATION_STYLES.keys())
    
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device=None, animation_style=None):
        """
        Initialize the image generator with the specified model.
        
        Args:
            model_id (str): Hugging Face model ID for Stable Diffusion
            device (str): Device to use for inference ('cpu', 'cuda', 'mps', None)
                          If None, will automatically select the best available device.
            animation_style (str): Animation style to use for image generation.
                                   Must be one of the keys in ANIMATION_STYLES.
        """
        # Set animation style
        self.animation_style = animation_style
        if animation_style and animation_style not in self.ANIMATION_STYLES:
            logger.warning(f"Animation style '{animation_style}' not found. Using default prompting.")
            self.animation_style = None
        # Set device automatically if not specified
        if device is None:
            # For Stable Diffusion specifically, use CPU on Apple Silicon to avoid issues
            # The extra performance of MPS is not worth the potential instability
            if torch.backends.mps.is_available():
                device = "cpu"  
                logger.info("Apple Silicon detected, but using CPU for stability with Stable Diffusion")
            elif torch.cuda.is_available():
                device = "cuda"
                logger.info("Using CUDA for inference")
            else:
                device = "cpu"
                logger.info("Using CPU for inference")
        
        self.device = device
        
        # Load Stable Diffusion pipeline with settings for stability
        logger.info(f"Loading {model_id} on {device}...")
        
        # Always use float32 for maximum compatibility
        pipeline_kwargs = {
            "safety_checker": None,  # Disable safety checker for speed
            "torch_dtype": torch.float32  # Always use float32 for stability
        }
        
        try:
            # Load the pipeline
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_id, 
                **pipeline_kwargs
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(device)
            
            # Enable memory efficiency techniques for all devices
            self.pipeline.enable_attention_slicing()
                
            logger.info(f"Successfully loaded model on {device}")
            
        except Exception as e:
            # If loading fails, always fall back to CPU
            logger.warning(f"Error loading model on {device}: {e}")
            logger.info("Falling back to CPU for compatibility")
            
            # Update device setting
            self.device = "cpu"
            device = "cpu"
            
            # Reload with CPU settings
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                safety_checker=None
            )
            self.pipeline = self.pipeline.to("cpu")
            self.pipeline.enable_attention_slicing()
        
        logger.info("Model loaded successfully")
    
    def generate_image(self, prompt, negative_prompt=None, output_path=None, 
                       width=512, height=512, num_inference_steps=30, guidance_scale=7.5):
        """
        Generate an image using Stable Diffusion.
        
        Args:
            prompt (str): The text prompt for image generation
            negative_prompt (str): Negative prompt to guide what to avoid
            output_path (str): Path to save the generated image
            width (int): Image width
            height (int): Image height
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): How closely to follow the prompt
            
        Returns:
            PIL.Image: The generated image
        """
        logger.info(f"Generating image for prompt: '{prompt}'")
        
        # Default negative prompt if none provided
        if negative_prompt is None:
            negative_prompt = "blurry, bad quality, distorted, deformed"
        
        # Augment prompt with animation style if specified
        if self.animation_style and self.animation_style in self.ANIMATION_STYLES:
            style_description = self.ANIMATION_STYLES[self.animation_style]
            prompt = f"{prompt}, {style_description}"
            logger.info(f"Using animation style: {self.animation_style}")
            logger.info(f"Enhanced prompt: '{prompt}'")
        
        # Generate the image with error handling
        try:
            with torch.no_grad():
                # Generate image
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                )
                
                # Get the first image from results
                image = result.images[0]
                
                # Validate image data (check if it's empty/corrupted)
                if hasattr(image, 'getextrema'):
                    # Check if all RGB channels have same min/max value which indicates blank image
                    extrema = image.getextrema()
                    if isinstance(extrema, tuple) and len(extrema) == 3:
                        # For RGB images
                        is_blank = all(min_val == max_val for min_val, max_val in extrema)
                    else:
                        # For grayscale
                        is_blank = extrema[0] == extrema[1]
                        
                    if is_blank:
                        logger.warning("Generated image appears to be blank. Creating placeholder image instead.")
                        # Create a colored placeholder with text instead of blank image
                        from PIL import Image, ImageDraw, ImageFont
                        image = Image.new('RGB', (width, height), color=(245, 245, 245))
                        draw = ImageDraw.Draw(image)
                        try:
                            # Try to use a system font
                            font = ImageFont.truetype("Arial", 20)
                        except Exception:
                            # Fall back to default font
                            font = ImageFont.load_default()
                        # Add some text to the image
                        prompt_text = prompt[:100] + '...' if len(prompt) > 100 else prompt
                        draw.text((20, height//2 - 30), "Image generation failed", fill=(0, 0, 0), font=font)
                        draw.text((20, height//2), prompt_text, fill=(80, 80, 80), font=font)
                
                # Save the image if output path is provided
                if output_path:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    image.save(output_path)
                    logger.info(f"Image saved to {output_path}")
        
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            # Create an error image instead of failing
            from PIL import Image, ImageDraw
            image = Image.new('RGB', (width, height), color=(245, 245, 245))
            draw = ImageDraw.Draw(image)
            draw.text((20, height//2), f"Error: {str(e)}", fill=(255, 0, 0))
            
            # Still try to save the error image
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                try:
                    image.save(output_path)
                    logger.info(f"Error image saved to {output_path}")
                except Exception as save_error:
                    logger.error(f"Could not save error image: {str(save_error)}")
        
        return image
    
    def create_video(self, audio_path, image_paths, output_path, 
                     transition_duration=1.0, image_duration=5.0, fps=30):
        """
        Create a video from images and audio using ffmpeg.
        
        Args:
            audio_path (str): Path to the audio file
            image_paths (list): List of paths to image files
            output_path (str): Path to save the output video
            transition_duration (float): Duration of transitions in seconds
            image_duration (float): Duration each image should be shown in seconds
            fps (int): Frames per second of the output video
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not image_paths:
            logger.error("No images provided for video creation")
            return False
        
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return False
        
        # Create temporary directory for intermediate files
        temp_dir = os.path.join(os.path.dirname(output_path), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a video from the images with crossfade transitions
        try:
            # Create the ffmpeg command for video creation
            # Using the complex xfade filter for transitions between images
            
            # First create a file with the list of images and durations
            concat_file = os.path.join(temp_dir, "concat.txt")
            with open(concat_file, "w") as f:
                for img_path in image_paths:
                    # Use absolute paths to ensure ffmpeg can find the files
                    abs_img_path = os.path.abspath(img_path)
                    f.write(f"file '{abs_img_path}'\n")
                    f.write(f"duration {image_duration}\n")
                # Write the last image path without duration
                abs_last_img_path = os.path.abspath(image_paths[-1])
                f.write(f"file '{abs_last_img_path}'\n")
            
            # Create slideshow video
            slideshow_path = os.path.join(temp_dir, "slideshow.mp4")
            
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", concat_file,
                "-vsync", "vfr", "-pix_fmt", "yuv420p",
                "-vf", f"fps={fps},format=yuv420p,scale=trunc(iw/2)*2:trunc(ih/2)*2",
                slideshow_path
            ]
            
            logger.info("Creating slideshow with ffmpeg...")
            logger.debug(f"Command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            # Add audio to the slideshow
            cmd = [
                "ffmpeg", "-y",
                "-i", slideshow_path,
                "-i", audio_path,
                "-c:v", "copy", "-c:a", "aac", "-shortest",
                output_path
            ]
            
            logger.info("Adding audio to slideshow...")
            logger.debug(f"Command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            logger.info(f"Video created successfully: {output_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error creating video: {str(e)}")
            return False
