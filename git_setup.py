#!/usr/bin/env python
"""
Git Setup Script for AnimatePodcasting

This script initializes a Git repository for the AnimatePodcasting project,
creates a proper .gitignore file, and performs the initial commit.
It also logs the Git activity to the Notion project page.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project path to PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from src.project_notion import ProjectNotionManager
except ImportError:
    print("Could not import ProjectNotionManager. Make sure the project structure is correct.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_gitignore():
    """Create a comprehensive .gitignore file for the project."""
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
.venv
.conda/
.python-version

# Environment variables
.env

# IDE files
.idea/
.vscode/
*.swp
*.swo
.DS_Store

# Project specific
outputs/
models/
data/
*.mp3
*.mp4
*.wav
*.png
*.jpg
*.jpeg
"""
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    logger.info("Created .gitignore file")

def setup_git_repo():
    """Initialize the Git repository and make the initial commit."""
    try:
        # Check if Git is already initialized
        if os.path.exists('.git'):
            logger.info("Git repository already initialized")
            return True
        
        # Initialize Git repository
        subprocess.run(["git", "init"], check=True)
        logger.info("Git repository initialized")
        
        # Create .gitignore
        create_gitignore()
        
        # Create data and outputs directories if they don't exist
        os.makedirs('data', exist_ok=True)
        os.makedirs('outputs', exist_ok=True)
        
        # Add files to Git
        subprocess.run(["git", "add", "."], check=True)
        
        # Commit
        subprocess.run(["git", "commit", "-m", "Initial commit: AnimatePodcasting project setup"], check=True)
        logger.info("Initial commit created")
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Git command failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Failed to set up Git repository: {str(e)}")
        return False

def setup_git_hooks():
    """Set up Git hooks for the project."""
    try:
        hooks_dir = Path('.git/hooks')
        if not hooks_dir.exists():
            logger.warning("Git hooks directory not found")
            return False
        
        # Create pre-commit hook to log activity to Notion
        pre_commit_hook = hooks_dir / 'pre-commit'
        with open(pre_commit_hook, 'w') as f:
            f.write("""#!/bin/sh
# Pre-commit hook to log Git activity to Notion
python -c "from src.project_notion import ProjectNotionManager; ProjectNotionManager().log_git_activity()" || true
exit 0
""")
        
        # Make the hook executable
        os.chmod(pre_commit_hook, 0o755)
        logger.info("Git pre-commit hook created to log activity to Notion")
        
        return True
    except Exception as e:
        logger.error(f"Failed to set up Git hooks: {str(e)}")
        return False

def log_to_notion():
    """Log the Git initialization to Notion."""
    try:
        notion = ProjectNotionManager()
        notion.log_git_activity(commit_message="Initial commit: AnimatePodcasting project setup")
        notion.complete_task("Initialize Git repository")
        logger.info("Git initialization logged to Notion")
        return True
    except Exception as e:
        logger.error(f"Failed to log to Notion: {str(e)}")
        return False

def main():
    """Main function to set up Git and log to Notion."""
    print("AnimatePodcasting - Git Setup")
    print("=" * 50)
    print("This script will initialize a Git repository for the AnimatePodcasting project")
    print("and log the activity to the Notion project page.")
    print("=" * 50)
    
    # Set up Git repository
    if not setup_git_repo():
        print("Failed to set up Git repository")
        return
    
    # Set up Git hooks
    setup_git_hooks()
    
    # Log to Notion
    if not log_to_notion():
        print("Failed to log to Notion, but Git repository is set up")
        return
    
    print("=" * 50)
    print("Git repository initialized successfully!")
    print("The activity has been logged to your Notion project page.")
    print("\nNext steps:")
    print("1. Add a remote repository:")
    print("   git remote add origin https://github.com/yourusername/AnimatePodcasting.git")
    print("2. Push your code:")
    print("   git push -u origin main")
    print("=" * 50)

if __name__ == "__main__":
    main() 