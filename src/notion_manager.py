"""
Notion Manager module for AnimatePodcasting

This module handles integration with Notion, allowing the application to
read from and write to Notion databases and pages.
"""

import os
import logging
from typing import Dict, List, Optional, Union
from notion_client import Client

logger = logging.getLogger(__name__)

class NotionManager:
    """Handles integration with Notion API."""
    
    def __init__(self, auth_token: Optional[str] = None):
        """
        Initialize the Notion manager with authentication.
        
        Args:
            auth_token (str, optional): Notion integration token. If not provided,
                                      will look for NOTION_TOKEN environment variable.
        """
        self.auth_token = auth_token or os.getenv('NOTION_TOKEN')
        if not self.auth_token:
            raise ValueError(
                "Notion authentication token not provided. "
                "Please set NOTION_TOKEN environment variable or pass token to constructor."
            )
        
        logger.info("Initializing Notion client...")
        self.client = Client(auth=self.auth_token)
        logger.info("Notion client initialized successfully")
    
    def get_database(self, database_id: str) -> Dict:
        """
        Retrieve a Notion database by its ID.
        
        Args:
            database_id (str): The ID of the Notion database
            
        Returns:
            dict: The database object
        """
        try:
            return self.client.databases.retrieve(database_id=database_id)
        except Exception as e:
            logger.error(f"Error retrieving database {database_id}: {str(e)}")
            raise
    
    def query_database(
        self, 
        database_id: str,
        filter_conditions: Optional[Dict] = None,
        sorts: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Query a Notion database with optional filters and sorting.
        
        Args:
            database_id (str): The ID of the Notion database
            filter_conditions (dict, optional): Filter conditions for the query
            sorts (list, optional): Sort conditions for the query
            
        Returns:
            list: List of database items matching the query
        """
        try:
            query_params = {}
            if filter_conditions:
                query_params['filter'] = filter_conditions
            if sorts:
                query_params['sorts'] = sorts
                
            response = self.client.databases.query(
                database_id=database_id,
                **query_params
            )
            return response['results']
        except Exception as e:
            logger.error(f"Error querying database {database_id}: {str(e)}")
            raise
    
    def create_page(
        self,
        parent_id: str,
        title: str,
        content: Optional[List[Dict]] = None,
        properties: Optional[Dict] = None,
        is_database: bool = False
    ) -> Dict:
        """
        Create a new page in Notion.
        
        Args:
            parent_id (str): ID of the parent page or database
            title (str): Title of the new page
            content (list, optional): List of content blocks
            properties (dict, optional): Additional page properties
            is_database (bool): Whether the parent is a database
            
        Returns:
            dict: The created page object
        """
        try:
            # Prepare the parent reference
            parent = {
                "type": "database_id" if is_database else "page_id",
                "database_id" if is_database else "page_id": parent_id
            }
            
            # Prepare properties
            page_properties = properties or {}
            if not is_database:
                page_properties["title"] = {
                    "title": [{"text": {"content": title}}]
                }
            
            # Create the page
            page = self.client.pages.create(
                parent=parent,
                properties=page_properties,
                children=content or []
            )
            
            logger.info(f"Created new page: {title}")
            return page
        except Exception as e:
            logger.error(f"Error creating page: {str(e)}")
            raise
    
    def update_page(
        self,
        page_id: str,
        properties: Optional[Dict] = None,
        content: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Update an existing Notion page.
        
        Args:
            page_id (str): ID of the page to update
            properties (dict, optional): Properties to update
            content (list, optional): New content blocks
            
        Returns:
            dict: The updated page object
        """
        try:
            # Update properties if provided
            if properties:
                self.client.pages.update(
                    page_id=page_id,
                    properties=properties
                )
            
            # Update content if provided
            if content:
                self.client.blocks.children.append(
                    block_id=page_id,
                    children=content
                )
            
            logger.info(f"Updated page: {page_id}")
            return self.client.pages.retrieve(page_id=page_id)
        except Exception as e:
            logger.error(f"Error updating page {page_id}: {str(e)}")
            raise
    
    def create_transcript_page(
        self,
        parent_id: str,
        title: str,
        transcript_text: str,
        metadata: Optional[Dict] = None,
        is_database: bool = False
    ) -> Dict:
        """
        Create a new page with podcast transcript content.
        
        Args:
            parent_id (str): ID of the parent page or database
            title (str): Title of the transcript
            transcript_text (str): The transcript content
            metadata (dict, optional): Additional metadata about the transcript
            is_database (bool): Whether the parent is a database
            
        Returns:
            dict: The created page object
        """
        # Prepare properties
        properties = metadata or {}
        if is_database:
            properties["Name"] = {"title": [{"text": {"content": title}}]}
        
        # Prepare content blocks
        content = [
            {
                "object": "block",
                "type": "heading_1",
                "heading_1": {
                    "rich_text": [{"type": "text", "text": {"content": "Transcript"}}]
                }
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": transcript_text}}]
                }
            }
        ]
        
        # Add metadata section if provided
        if metadata:
            content.insert(0, {
                "object": "block",
                "type": "heading_1",
                "heading_1": {
                    "rich_text": [{"type": "text", "text": {"content": "Metadata"}}]
                }
            })
            
            # Add each metadata field
            for key, value in metadata.items():
                if key != "Name":  # Skip the title property
                    content.insert(1, {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {"type": "text", "text": {"content": f"{key}: {value}"}}
                            ]
                        }
                    })
        
        return self.create_page(
            parent_id=parent_id,
            title=title,
            content=content,
            properties=properties,
            is_database=is_database
        ) 