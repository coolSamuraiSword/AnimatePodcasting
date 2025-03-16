"""
Project Notion Manager for AnimatePodcasting

This module handles the project's Notion page, keeping it in sync with the local development.
"""

import os
import logging
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from notion_client import Client

logger = logging.getLogger(__name__)

class ProjectNotionManager:
    """Manages the project's Notion page and keeps it in sync with local development."""
    
    def __init__(self):
        """Initialize the Project Notion Manager."""
        self.token = os.getenv('NOTION_TOKEN')
        self.page_id = os.getenv('NOTION_PAGE_ID')
        
        if not self.token or not self.page_id:
            raise ValueError(
                "Notion configuration not found. "
                "Please ensure NOTION_TOKEN and NOTION_PAGE_ID are set in .env file."
            )
        
        logger.info("Initializing Notion client for project management...")
        self.client = Client(auth=self.token)
        logger.info("Notion client initialized successfully")
    
    def setup_project_page(self):
        """
        Set up or update the project page structure with comprehensive sections.
        """
        try:
            # Define the main sections of the project page
            sections = [
                # Project Overview Section
                {
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": [{"type": "text", "text": {"content": "🎯 Project Overview"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {
                                    "content": "AnimatePodcasting is a Python-based application that transforms podcast audio into engaging animated videos using AI technologies. The system processes audio, extracts key segments, generates relevant imagery, and compiles everything into a shareable video format."
                                }
                            }
                        ]
                    }
                },
                
                # Project Architecture
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"type": "text", "text": {"content": "Project Architecture"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {
                                    "content": "The project follows a modular architecture with specialized components for each stage of the pipeline:"
                                }
                            }
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": "WhisperTranscriber"},
                                "annotations": {"bold": True}
                            },
                            {
                                "type": "text",
                                "text": {"content": ": Uses OpenAI's Whisper to transform spoken audio into text with accurate timing information."}
                            }
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": "ContentAnalyzer"},
                                "annotations": {"bold": True}
                            },
                            {
                                "type": "text",
                                "text": {"content": ": Employs sentence transformers to identify key segments from the transcript based on semantic relevance."}
                            }
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": "ImageGenerator"},
                                "annotations": {"bold": True}
                            },
                            {
                                "type": "text",
                                "text": {"content": ": Leverages Stable Diffusion to create images that visually represent the content of key segments."}
                            }
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": "VideoCreator"},
                                "annotations": {"bold": True}
                            },
                            {
                                "type": "text",
                                "text": {"content": ": Uses FFmpeg to combine the audio and generated images into a cohesive video presentation."}
                            }
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": "NotionManager"},
                                "annotations": {"bold": True}
                            },
                            {
                                "type": "text",
                                "text": {"content": ": Integrates with Notion API to track project progress, manage tasks, and document the development process."}
                            }
                        ]
                    }
                },
                
                # Technical Stack
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"type": "text", "text": {"content": "Technical Stack"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {
                                    "content": "The project utilizes the following technologies:"
                                }
                            }
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": "Python 3.11+ as the primary programming language"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": "OpenAI Whisper for audio transcription"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": "SentenceTransformers for semantic analysis"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": "Hugging Face Diffusers for image generation"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": "FFmpeg for video processing"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": "Notion API for project management"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": "Git for version control"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "divider",
                    "divider": {}
                },
                
                # Task Manager Section
                {
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": [{"type": "text", "text": {"content": "📋 Task Manager"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": "Track and manage project tasks here. Use the functions in ProjectNotionManager to add, update, and complete tasks."}
                            }
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "heading_3",
                    "heading_3": {
                        "rich_text": [{"type": "text", "text": {"content": "Pending Tasks"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "to_do",
                    "to_do": {
                        "rich_text": [{"type": "text", "text": {"content": "Initialize Git repository"}}],
                        "checked": False,
                        "color": "blue"
                    }
                },
                {
                    "object": "block",
                    "type": "to_do",
                    "to_do": {
                        "rich_text": [{"type": "text", "text": {"content": "Set up development environment"}}],
                        "checked": False,
                        "color": "blue"
                    }
                },
                {
                    "object": "block",
                    "type": "to_do",
                    "to_do": {
                        "rich_text": [{"type": "text", "text": {"content": "Gather sample podcast data"}}],
                        "checked": False,
                        "color": "blue"
                    }
                },
                {
                    "object": "block",
                    "type": "to_do",
                    "to_do": {
                        "rich_text": [{"type": "text", "text": {"content": "Implement caching for transcriptions"}}],
                        "checked": False,
                        "color": "blue"
                    }
                },
                {
                    "object": "block",
                    "type": "to_do",
                    "to_do": {
                        "rich_text": [{"type": "text", "text": {"content": "Improve image generation prompts"}}],
                        "checked": False,
                        "color": "blue"
                    }
                },
                {
                    "object": "block",
                    "type": "to_do",
                    "to_do": {
                        "rich_text": [{"type": "text", "text": {"content": "Add customizable video templates"}}],
                        "checked": False,
                        "color": "blue"
                    }
                },
                {
                    "object": "block",
                    "type": "to_do",
                    "to_do": {
                        "rich_text": [{"type": "text", "text": {"content": "Create REST API for web interface"}}],
                        "checked": False,
                        "color": "blue"
                    }
                },
                {
                    "object": "block",
                    "type": "heading_3",
                    "heading_3": {
                        "rich_text": [{"type": "text", "text": {"content": "Completed Tasks"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "to_do",
                    "to_do": {
                        "rich_text": [{"type": "text", "text": {"content": "Set up Notion integration"}}],
                        "checked": True,
                        "color": "green"
                    }
                },
                {
                    "object": "block",
                    "type": "divider",
                    "divider": {}
                },
                
                # Git Activity Section
                {
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": [{"type": "text", "text": {"content": "🔄 Git Activity"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": "Track code changes and development progress. Recent commits and key development milestones will be logged here."}
                            }
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "divider",
                    "divider": {}
                },
                
                # Project Status Section
                {
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": [{"type": "text", "text": {"content": "📊 Project Status"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": "Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")}}]
                    }
                },
                {
                    "object": "block",
                    "type": "divider",
                    "divider": {}
                },
                
                # Development Tips & Best Practices
                {
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": [{"type": "text", "text": {"content": "💡 Development Tips & Best Practices"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": "Git Workflow"},
                                "annotations": {"bold": True}
                            },
                            {
                                "type": "text",
                                "text": {"content": ": Use feature branches for new development. Create a branch for each new feature or bug fix using the format 'feature/name' or 'bugfix/issue-number'."}
                            }
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": "Commit Messages"},
                                "annotations": {"bold": True}
                            },
                            {
                                "type": "text",
                                "text": {"content": ": Follow the conventional commits format (feat:, fix:, docs:, refactor:, etc.) to maintain clarity in the project history."}
                            }
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": "Task Management"},
                                "annotations": {"bold": True}
                            },
                            {
                                "type": "text",
                                "text": {"content": ": Use the built-in task tracking in Notion. Add new tasks with the add_task() method and mark them complete with complete_task()."}
                            }
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": "Cache Management"},
                                "annotations": {"bold": True}
                            },
                            {
                                "type": "text",
                                "text": {"content": ": Transcriptions and analysis results should be cached to avoid redundant processing of the same audio files."}
                            }
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": "Modular Design"},
                                "annotations": {"bold": True}
                            },
                            {
                                "type": "text",
                                "text": {"content": ": Keep components loosely coupled. Each module should have a single responsibility and clear interfaces."}
                            }
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": "Testing"},
                                "annotations": {"bold": True}
                            },
                            {
                                "type": "text",
                                "text": {"content": ": Write unit tests for core functionality. Use pytest for test automation. Aim for >80% code coverage."}
                            }
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": "Documentation"},
                                "annotations": {"bold": True}
                            },
                            {
                                "type": "text",
                                "text": {"content": ": Keep code documentation up-to-date. Use docstrings for functions and classes. Update this Notion page with key architectural decisions."}
                            }
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "divider",
                    "divider": {}
                },
                
                # Processing History Section
                {
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": [{"type": "text", "text": {"content": "📝 Processing History"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": "Recent processing activities will be logged here."}}]
                    }
                }
            ]
            
            # Update the page with the sections
            self.client.blocks.children.append(
                block_id=self.page_id,
                children=sections
            )
            
            logger.info("Project page structure set up successfully")
            
        except Exception as e:
            logger.error(f"Failed to set up project page: {str(e)}")
            raise
    
    def log_processing_activity(self, activity_type: str, details: Dict):
        """
        Log a processing activity to the Processing History section.
        
        Args:
            activity_type (str): Type of activity (e.g., 'transcription', 'image_generation')
            details (dict): Activity details
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create the activity log entry
            activity_block = {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": f"[{timestamp}] {activity_type.upper()}\n",
                            },
                            "annotations": {"bold": True}
                        },
                        {
                            "type": "text",
                            "text": {
                                "content": "\n".join(f"{k}: {v}" for k, v in details.items())
                            }
                        }
                    ]
                }
            }
            
            # Find the Processing History section and add the entry
            self._append_to_section("📝 Processing History", [activity_block])
            
            logger.info(f"Logged {activity_type} activity to Notion")
            
        except Exception as e:
            logger.error(f"Failed to log activity to Notion: {str(e)}")
            # Don't raise the exception - logging failure shouldn't stop the main process
    
    def update_project_status(self, status_updates: Dict):
        """
        Update the Project Status section.
        
        Args:
            status_updates (dict): Status information to update
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create the status update block
            status_block = {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": f"Status as of {timestamp}\n",
                            },
                            "annotations": {"bold": True}
                        },
                        {
                            "type": "text",
                            "text": {
                                "content": "\n".join(f"{k}: {v}" for k, v in status_updates.items())
                            }
                        }
                    ]
                }
            }
            
            # Update the Project Status section
            self._append_to_section("📊 Project Status", [status_block])
            
            logger.info("Updated project status in Notion")
            
        except Exception as e:
            logger.error(f"Failed to update project status in Notion: {str(e)}")
            # Don't raise the exception - status update failure shouldn't stop the main process
    
    def add_task(self, task_description: str, priority: str = "normal"):
        """
        Add a new task to the Task Manager section.
        
        Args:
            task_description (str): Description of the task
            priority (str): Priority level (low, normal, high)
        """
        try:
            # Map priority to color
            color_map = {
                "low": "gray",
                "normal": "blue",
                "high": "red"
            }
            color = color_map.get(priority.lower(), "blue")
            
            # Create a section heading and task block
            task_blocks = [
                {
                    "object": "block",
                    "type": "heading_3",
                    "heading_3": {
                        "rich_text": [{"type": "text", "text": {"content": "New Task"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "to_do",
                    "to_do": {
                        "rich_text": [{"type": "text", "text": {"content": task_description}}],
                        "checked": False,
                        "color": color
                    }
                }
            ]
            
            # Add directly to the page under the Task Manager section
            self._append_to_section("📋 Task Manager", task_blocks)
            
            logger.info(f"Added new task: {task_description}")
            
        except Exception as e:
            logger.error(f"Failed to add task to Notion: {str(e)}")
            raise
    
    def complete_task(self, task_description: str):
        """
        Mark a task as completed and move it to the Completed Tasks section.
        
        Args:
            task_description (str): Description of the task to complete
        """
        try:
            # Find the task in the Pending Tasks section
            blocks = self.client.blocks.children.list(block_id=self.page_id)
            task_block_id = None
            
            # Search for the task
            for block in blocks["results"]:
                if block["type"] == "to_do" and not block["to_do"]["checked"]:
                    text_content = "".join([rt["text"]["content"] for rt in block["to_do"]["rich_text"]])
                    if task_description.lower() in text_content.lower():
                        task_block_id = block["id"]
                        break
            
            if task_block_id:
                # Mark the task as completed
                self.client.blocks.update(
                    block_id=task_block_id,
                    to_do={
                        "checked": True,
                        "color": "green"
                    }
                )
                
                # Move task to Completed Tasks section (create a new task and delete the old one)
                completed_task = {
                    "object": "block",
                    "type": "to_do",
                    "to_do": {
                        "rich_text": [{"type": "text", "text": {"content": task_description}}],
                        "checked": True,
                        "color": "green"
                    }
                }
                
                # Add to Completed Tasks section
                self._append_to_heading("Completed Tasks", [completed_task])
                
                # Delete the original task
                self.client.blocks.delete(block_id=task_block_id)
                
                logger.info(f"Marked task as completed: {task_description}")
            else:
                logger.warning(f"Task not found: {task_description}")
            
        except Exception as e:
            logger.error(f"Failed to complete task in Notion: {str(e)}")
            raise
    
    def log_git_activity(self, commit_message: Optional[str] = None, branch: Optional[str] = None):
        """
        Log Git activity to the Git Activity section.
        
        Args:
            commit_message (str, optional): Specific commit message to log
            branch (str, optional): Branch name to log
        """
        try:
            git_details = {}
            
            # If no specific commit message, get the latest commit
            if not commit_message:
                try:
                    commit_message = subprocess.check_output(
                        ["git", "log", "-1", "--pretty=%B"], 
                        stderr=subprocess.STDOUT,
                        universal_newlines=True
                    ).strip()
                    git_details["Last Commit"] = commit_message
                except subprocess.CalledProcessError:
                    git_details["Status"] = "No git repository found or no commits yet"
            else:
                git_details["Commit"] = commit_message
            
            # Get current branch if not specified
            if not branch:
                try:
                    branch = subprocess.check_output(
                        ["git", "branch", "--show-current"],
                        stderr=subprocess.STDOUT,
                        universal_newlines=True
                    ).strip()
                    git_details["Branch"] = branch
                except subprocess.CalledProcessError:
                    pass
            else:
                git_details["Branch"] = branch
            
            # Get additional git stats
            try:
                # Total number of commits
                commit_count = subprocess.check_output(
                    ["git", "rev-list", "--count", "HEAD"],
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                ).strip()
                git_details["Total Commits"] = commit_count
                
                # Files changed in last commit
                files_changed = subprocess.check_output(
                    ["git", "show", "--stat", "--oneline", "HEAD"],
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                ).strip()
                
                # Extract just the summary line with file counts
                summary_lines = [line for line in files_changed.split('\n') if ' file' in line and 'changed' in line]
                if summary_lines:
                    git_details["Changes"] = summary_lines[0]
            except subprocess.CalledProcessError:
                pass
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create the git activity block
            activity_block = {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": f"[{timestamp}] GIT ACTIVITY\n",
                            },
                            "annotations": {"bold": True}
                        },
                        {
                            "type": "text",
                            "text": {
                                "content": "\n".join(f"{k}: {v}" for k, v in git_details.items())
                            }
                        }
                    ]
                }
            }
            
            # Add to Git Activity section
            self._append_to_section("🔄 Git Activity", [activity_block])
            
            logger.info("Logged git activity to Notion")
            
        except Exception as e:
            logger.error(f"Failed to log git activity to Notion: {str(e)}")
            # Don't raise the exception - git logging failure shouldn't stop the main process
    
    def init_git_repository(self, directory: Optional[str] = None):
        """
        Initialize a git repository and log the activity to Notion.
        
        Args:
            directory (str, optional): Directory to initialize git repository
        """
        try:
            # Set the directory
            if directory:
                os.chdir(directory)
            
            # Check if git repo already exists
            if os.path.exists('.git'):
                logger.info("Git repository already initialized")
                self.log_git_activity(commit_message="Git repository already initialized")
                return
            
            # Initialize git repository
            subprocess.run(["git", "init"], check=True)
            logger.info("Git repository initialized")
            
            # Create .gitignore file
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
.env

# Project specific
outputs/
models/
.DS_Store
*.mp3
*.mp4
*.wav
*.png
*.jpg
"""
            with open('.gitignore', 'w') as f:
                f.write(gitignore_content)
            
            # Add files and make initial commit
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit: Project setup"], check=True)
            
            # Log to Notion
            self.log_git_activity(commit_message="Initial commit: Project setup")
            
            # Complete the task if it exists
            self.complete_task("Initialize Git repository")
            
        except Exception as e:
            logger.error(f"Failed to initialize git repository: {str(e)}")
            raise
    
    def add_project_recommendation(self, title: str, description: str):
        """
        Add a project recommendation or best practice to the Development Tips section.
        
        Args:
            title (str): Title of the recommendation
            description (str): Description of the recommendation
        """
        try:
            # Create the recommendation block
            recommendation_block = {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {"content": title},
                            "annotations": {"bold": True}
                        },
                        {
                            "type": "text",
                            "text": {"content": f": {description}"}
                        }
                    ]
                }
            }
            
            # Add to Development Tips section
            self._append_to_section("💡 Development Tips & Best Practices", [recommendation_block])
            
            logger.info(f"Added project recommendation: {title}")
            
        except Exception as e:
            logger.error(f"Failed to add project recommendation: {str(e)}")
            raise
    
    def _append_to_section(self, section_title: str, blocks: List[Dict]):
        """
        Helper method to append blocks to a section with the given title.
        
        Args:
            section_title (str): Title of the section to append to
            blocks (list): List of blocks to append
        """
        try:
            # Find the section
            notion_blocks = self.client.blocks.children.list(block_id=self.page_id)
            section_id = None
            
            for block in notion_blocks["results"]:
                if block["type"] == "heading_1":
                    rich_text = block["heading_1"]["rich_text"]
                    if any(text.get("text", {}).get("content") == section_title for text in rich_text):
                        section_id = block["id"]
                        break
            
            if section_id:
                try:
                    # Try to append blocks after the section heading
                    self.client.blocks.children.append(
                        block_id=section_id,
                        children=blocks
                    )
                except Exception as e:
                    logger.warning(f"Could not append directly to section: {str(e)}")
                    # Fallback: Add after section at page level
                    section_index = next((i for i, block in enumerate(notion_blocks["results"]) 
                                        if block["type"] == "heading_1" and 
                                        any(text.get("text", {}).get("content") == section_title 
                                            for text in block["heading_1"]["rich_text"])), -1)
                    
                    if section_index != -1 and section_index < len(notion_blocks["results"]) - 1:
                        # Add content at page level
                        self.client.blocks.children.append(
                            block_id=self.page_id,
                            children=blocks
                        )
                    else:
                        logger.warning(f"Could not find appropriate position to add blocks")
            else:
                logger.warning(f"Section '{section_title}' not found, adding to page")
                # Add to page if section not found
                self.client.blocks.children.append(
                    block_id=self.page_id,
                    children=blocks
                )
        except Exception as e:
            logger.error(f"Failed to append to section: {str(e)}")
            # Last resort fallback - add directly to the page
            try:
                self.client.blocks.children.append(
                    block_id=self.page_id,
                    children=blocks
                )
            except Exception as append_error:
                logger.error(f"Failed to append to page: {str(append_error)}")
    
    def _append_to_heading(self, heading_text: str, blocks: List[Dict]):
        """
        Helper method to append blocks after a specific heading.
        
        Args:
            heading_text (str): Text of the heading to append after
            blocks (list): List of blocks to append
        """
        try:
            # Find all blocks to locate the heading
            notion_blocks = self.client.blocks.children.list(block_id=self.page_id)
            heading_block = None
            
            # First try to find the heading
            for block in notion_blocks["results"]:
                # Check all heading types (h1, h2, h3)
                for heading_type in ["heading_1", "heading_2", "heading_3"]:
                    if block["type"] == heading_type:
                        rich_text = block[heading_type]["rich_text"]
                        if any(text.get("text", {}).get("content") == heading_text for text in rich_text):
                            heading_block = block
                            break
                
                if heading_block:
                    break
            
            if heading_block:
                try:
                    # First try to append directly to the heading block
                    self.client.blocks.children.append(
                        block_id=heading_block["id"],
                        children=blocks
                    )
                except Exception as e:
                    logger.warning(f"Could not append to heading block: {str(e)}")
                    # Fallback: append to page level instead
                    self.client.blocks.children.append(
                        block_id=self.page_id,
                        children=blocks
                    )
            else:
                # If heading not found, append to page level
                logger.warning(f"Heading '{heading_text}' not found, adding to page level")
                self.client.blocks.children.append(
                    block_id=self.page_id,
                    children=blocks
                )
        except Exception as e:
            logger.error(f"Failed to append to heading: {str(e)}")
            # Last resort fallback - add directly to the page
            try:
                self.client.blocks.children.append(
                    block_id=self.page_id,
                    children=blocks
                )
            except Exception as append_error:
                logger.error(f"Failed to append to page: {str(append_error)}") 