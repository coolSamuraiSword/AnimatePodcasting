#!/usr/bin/env python
"""
Notion Demo Script for AnimatePodcasting

This script demonstrates how to use the AnimatePodcasting system with Notion integration.
It will guide you through setting up the environment and running the system.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.project_notion import ProjectNotionManager
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from project_notion import ProjectNotionManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if the environment is properly set up."""
    env_file = Path('.env')
    if not env_file.exists():
        logger.error("No .env file found. Please ensure you have set up your Notion credentials.")
        logger.info("The .env file should contain:\n  NOTION_TOKEN=your_token\n  NOTION_PAGE_ID=your_page_id")
        return False
    
    return True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Notion Demo for AnimatePodcasting')
    parser.add_argument('--audio', type=str, help='Path to audio file')
    parser.add_argument('--output', type=str, default='outputs/demo_output.mp4', 
                        help='Path to output video file')
    parser.add_argument('--model', type=str, default='tiny', 
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size (tiny is faster for demo purposes)')
    parser.add_argument('--num-images', type=int, default=3,
                        help='Number of images to generate')
    
    # Add Notion management options
    parser.add_argument('--setup-notion', action='store_true',
                       help='Set up or update Notion project page structure')
    parser.add_argument('--add-task', type=str, 
                       help='Add a new task to the Notion task manager')
    parser.add_argument('--priority', type=str, default='normal',
                       choices=['low', 'normal', 'high'],
                       help='Priority for the new task')
    parser.add_argument('--complete-task', type=str,
                       help='Mark a task as completed')
    
    # Add Git integration options
    parser.add_argument('--init-git', action='store_true',
                       help='Initialize a Git repository and log to Notion')
    parser.add_argument('--log-git', action='store_true',
                       help='Log the latest Git activity to Notion')
    
    return parser.parse_args()

def run_notion_tasks(notion_manager, args):
    """Run Notion-specific tasks based on arguments."""
    if args.setup_notion:
        logger.info("Setting up Notion project page...")
        notion_manager.setup_project_page()
        logger.info("Notion project page setup complete!")
    
    if args.add_task:
        logger.info(f"Adding task with priority {args.priority}: {args.add_task}")
        notion_manager.add_task(args.add_task, args.priority)
        logger.info("Task added successfully!")
    
    if args.complete_task:
        logger.info(f"Marking task as complete: {args.complete_task}")
        notion_manager.complete_task(args.complete_task)
        logger.info("Task marked as complete!")
    
    if args.init_git:
        logger.info("Initializing Git repository...")
        notion_manager.init_git_repository()
        logger.info("Git repository initialized and logged to Notion!")
    
    if args.log_git:
        logger.info("Logging Git activity to Notion...")
        notion_manager.log_git_activity()
        logger.info("Git activity logged to Notion!")

def run_demo():
    """Run the demo."""
    # Check environment
    if not check_environment():
        return
    
    args = parse_args()
    
    # Initialize Notion manager
    try:
        notion_manager = ProjectNotionManager()
    except Exception as e:
        logger.error(f"Failed to initialize Notion integration: {str(e)}")
        return
    
    # Handle Notion-specific tasks
    run_notion_tasks(notion_manager, args)
    
    # If no audio file provided, we're only doing Notion tasks
    if not args.audio:
        if not any([args.setup_notion, args.add_task, args.complete_task, args.init_git, args.log_git]):
            logger.info("No audio file or Notion tasks specified. Use --help to see available options.")
        return
    
    audio_path = args.audio
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return
    
    # Create output directory
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the system with Notion integration
    cmd = f"cd src && python animate_podcast.py --audio {audio_path} --output ../{args.output} --model {args.model} --num-images {args.num_images}"
    logger.info(f"Running command: {cmd}")
    
    # Execute the command
    exit_code = os.system(cmd)
    
    if exit_code == 0:
        logger.info("===============================")
        logger.info("Demo completed successfully!")
        logger.info(f"Generated video: {args.output}")
        logger.info("Check your Notion page to see the updates")
        logger.info("===============================")
        
        # Log Git activity after successful processing
        if os.path.exists('.git'):
            logger.info("Logging Git activity to Notion...")
            notion_manager.log_git_activity()
    else:
        logger.error("Demo failed. Check the logs for details.")

if __name__ == "__main__":
    print("AnimatePodcasting - Notion Integration Demo")
    print("=" * 50)
    print("This demo will process an audio file and update your Notion page.")
    print("=" * 50)
    
    if len(sys.argv) == 1:
        print("\nUSAGE EXAMPLES:")
        print("  # Setup Notion page structure:")
        print("  python notion_demo.py --setup-notion")
        print("\n  # Task management:")
        print("  python notion_demo.py --add-task \"Implement new feature\" --priority high")
        print("  python notion_demo.py --complete-task \"Implement new feature\"")
        print("\n  # Git integration:")
        print("  python notion_demo.py --init-git")
        print("  python notion_demo.py --log-git")
        print("\n  # Process audio:")
        print("  python notion_demo.py --audio your_audio_file.mp3")
        print("\nRun with --help for all available options")
    else:
        run_demo() 