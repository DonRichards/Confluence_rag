from bs4 import BeautifulSoup
import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import gradio as gr
import html
from utils.pinecone_logic import delete_pinecone_index, get_pinecone_index, upsert_data
from utils.data_prep import import_csv, clean_data_pinecone_schema, generate_embeddings_and_add_to_df
from utils.openai_logic import get_embeddings, create_prompt, add_prompt_messages, get_chat_completion_messages, create_system_prompt
from utils.auth import get_confluence_client
import sys
import time
import json
from datetime import datetime
import logging
import psutil
import socket
import threading
from typing import Optional
import errno

# load environment variables
load_dotenv(find_dotenv())

# Set up logging directory
LOGS_DIR = os.path.join(os.getcwd(), "conversation_logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure logging based on DEBUG environment variable
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
log_level = logging.DEBUG if DEBUG else logging.INFO
log_format = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s' if DEBUG else '%(asctime)s - %(levelname)s - %(message)s'

# Set up logging
logging.basicConfig(
    level=log_level,
    format=log_format,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "app_debug.log")),
        logging.StreamHandler()
    ]
)

logging.info(f"Application starting with DEBUG={DEBUG}")

# Monkey patch the Gradio client utils to fix the "bool is not iterable" error
# This is needed for Gradio 4.44.1
if DEBUG:
    try:
        import gradio_client.utils as client_utils
        
        # Also patch typing_extensions to fix TypedDict issues
        try:
            import typing_extensions
            original_typeddict = typing_extensions.TypedDict
            
            # Create a safer TypedDict replacement that doesn't choke on bool values
            def safe_typeddict(*args, **kwargs):
                try:
                    return original_typeddict(*args, **kwargs)
                except Exception as e:
                    logging.debug(f"Caught error in TypedDict: {e}")
                    # Return a simplified version that won't cause errors
                    return dict
                    
            # Apply the patch
            typing_extensions.TypedDict = safe_typeddict
            logging.debug("Applied monkey patch to typing_extensions.TypedDict")
        except Exception as typeddict_error:
            logging.warning(f"Failed to patch TypedDict: {typeddict_error}")
            
        original_json_schema_to_python_type = client_utils.json_schema_to_python_type
        
        # Check the actual signature of the original function
        import inspect
        func_sig = inspect.signature(original_json_schema_to_python_type)
        param_count = len(func_sig.parameters)
        logging.debug(f"Original json_schema_to_python_type signature has {param_count} parameters: {func_sig}")
        
        # Create a compatible wrapper based on the original signature
        if param_count == 1:
            def patched_json_schema_to_python_type(schema):
                try:
                    # Handle specific case of schema being boolean
                    if isinstance(schema, bool):
                        logging.debug(f"Received boolean schema value: {schema}")
                        return "any"  # Return a simple type as fallback
                    return original_json_schema_to_python_type(schema)
                except TypeError as e:
                    if "argument of type 'bool' is not iterable" in str(e):
                        logging.debug("Caught bool is not iterable error in json_schema_to_python_type, returning default schema")
                        return "any"  # Return a simple type as fallback
                    raise
        else:
            def patched_json_schema_to_python_type(schema, *args, **kwargs):
                try:
                    # Handle specific case of schema being boolean
                    if isinstance(schema, bool):
                        logging.debug(f"Received boolean schema value: {schema}")
                        return "any"  # Return a simple type as fallback
                    return original_json_schema_to_python_type(schema, *args, **kwargs)
                except TypeError as e:
                    if "argument of type 'bool' is not iterable" in str(e):
                        logging.debug("Caught bool is not iterable error in json_schema_to_python_type, returning default schema")
                        return "any"  # Return a simple type as fallback
                    raise
                
        # Apply the patch
        client_utils.json_schema_to_python_type = patched_json_schema_to_python_type
        logging.debug("Applied monkey patch to gradio_client.utils.json_schema_to_python_type")
        
        # Also patch the internal function if we can find it
        if hasattr(client_utils, '_json_schema_to_python_type'):
            original_internal = client_utils._json_schema_to_python_type
            
            # Check signature of internal function
            internal_sig = inspect.signature(original_internal)
            internal_param_count = len(internal_sig.parameters)
            logging.debug(f"Original _json_schema_to_python_type signature has {internal_param_count} parameters: {internal_sig}")
            
            # Create a compatible wrapper
            if internal_param_count == 1:
                def patched_internal_func(schema):
                    try:
                        # Handle specific case of schema being boolean (True/False)
                        if isinstance(schema, bool):
                            logging.debug(f"Handling boolean schema value in internal func: {schema}")
                            return "any"  # Return a simple type as fallback
                            
                        # Handle additionalProperties being True (key source of the error)
                        if isinstance(schema, dict):
                            # If additionalProperties is True, modify it to a dict with "type": "any"
                            if schema.get("additionalProperties") is True:
                                logging.debug("Schema has additionalProperties=True, converting to dict")
                                modified_schema = schema.copy()
                                modified_schema["additionalProperties"] = {"type": "any"}
                                return original_internal(modified_schema)
                            
                            # Handle $ref to a boolean additionalProperties
                            if "$ref" in schema and isinstance(schema["$ref"], str):
                                ref_key = schema["$ref"].split("/")[-1]
                                if ref_key in schema:
                                    ref_schema = schema[ref_key]
                                    if isinstance(ref_schema, dict) and ref_schema.get("additionalProperties") is True:
                                        logging.debug(f"Reference schema has additionalProperties=True, fixing reference")
                                        defs = schema.copy()  # Create a copy of defs to modify
                                        defs[ref_key] = ref_schema.copy()
                                        defs[ref_key]["additionalProperties"] = {"type": "any"}
                                        return original_internal(defs)
                        
                        return original_internal(schema)
                    except TypeError as e:
                        if "argument of type 'bool' is not iterable" in str(e):
                            logging.debug("Caught bool is not iterable error in _json_schema_to_python_type, returning default schema")
                            return "any"  # Return a simple type as fallback
                        if "not iterable" in str(e):
                            logging.debug(f"Caught iteration error in _json_schema_to_python_type: {e}")
                            return "any"  # Return a simple type as fallback
                        raise
                    except Exception as e:
                        logging.debug(f"Caught unexpected error in internal schema parser: {e}")
                        if "Cannot parse schema" in str(e):
                            logging.debug(f"Cannot parse schema error with schema: {schema}")
                            return "any"  # Return a simple type as fallback
                        raise
            else:
                def patched_internal_func(schema, *args, **kwargs):
                    try:
                        # Handle specific case of schema being boolean (True/False)
                        if isinstance(schema, bool):
                            logging.debug(f"Handling boolean schema value in internal func: {schema}")
                            return "any"  # Return a simple type as fallback
                            
                        # Handle additionalProperties being True (key source of the error)
                        if isinstance(schema, dict):
                            # If additionalProperties is True, modify it to a dict with "type": "any"
                            if schema.get("additionalProperties") is True:
                                logging.debug("Schema has additionalProperties=True, converting to dict")
                                modified_schema = schema.copy()
                                modified_schema["additionalProperties"] = {"type": "any"}
                                return original_internal(modified_schema, *args, **kwargs)
                            
                            # Handle $ref to a boolean additionalProperties
                            if "$ref" in schema and args and isinstance(args[0], dict):
                                ref_key = schema["$ref"].split("/")[-1]
                                if ref_key in args[0]:
                                    ref_schema = args[0][ref_key]
                                    if isinstance(ref_schema, dict) and ref_schema.get("additionalProperties") is True:
                                        logging.debug(f"Reference schema has additionalProperties=True, fixing reference")
                                        defs = args[0].copy()  # Create a copy of defs to modify
                                        defs[ref_key] = ref_schema.copy()
                                        defs[ref_key]["additionalProperties"] = {"type": "any"}
                                        return original_internal(schema, defs, **kwargs)
                        
                        return original_internal(schema, *args, **kwargs)
                    except TypeError as e:
                        if "argument of type 'bool' is not iterable" in str(e):
                            logging.debug("Caught bool is not iterable error in _json_schema_to_python_type, returning default schema")
                            return "any"  # Return a simple type as fallback
                        if "not iterable" in str(e):
                            logging.debug(f"Caught iteration error in _json_schema_to_python_type: {e}")
                            return "any"  # Return a simple type as fallback
                        raise
                    except Exception as e:
                        logging.debug(f"Caught unexpected error in internal schema parser: {e}")
                        if "Cannot parse schema" in str(e):
                            logging.debug(f"Cannot parse schema error with schema: {schema}")
                            return "any"  # Return a simple type as fallback
                        raise
                    
            # Apply the patch to the internal function
            client_utils._json_schema_to_python_type = patched_internal_func
            logging.debug("Applied monkey patch to gradio_client.utils._json_schema_to_python_type")
            
            # Try to patch the get_type function which is the source of the "bool is not iterable" error
            if hasattr(client_utils, 'get_type'):
                original_get_type = client_utils.get_type
                
                def patched_get_type(schema):
                    try:
                        # The specific error happens when trying to check if "const" in schema
                        # when schema is a bool, so handle that case directly
                        if isinstance(schema, bool):
                            logging.debug("Bypassing bool schema in get_type to prevent 'not iterable' error")
                            return "bool"
                        # Also handle dictionaries with additionalProperties: true 
                        if isinstance(schema, dict) and schema.get("additionalProperties") is True:
                            logging.debug("Handling additionalProperties=True in get_type")
                            modified_schema = schema.copy()
                            modified_schema["additionalProperties"] = {"type": "any"}
                            return original_get_type(modified_schema)
                        return original_get_type(schema)
                    except TypeError as e:
                        if "not iterable" in str(e):
                            logging.debug(f"Caught iteration error in get_type: {e}")
                            # If schema is not iterable, return a sensible default based on type
                            if isinstance(schema, bool):
                                return "bool"
                            elif isinstance(schema, (int, float)):
                                return "number"
                            elif schema is None:
                                return "null"
                            return "any"
                        raise
                    except Exception as e:
                        logging.debug(f"Caught unexpected error in get_type: {e}")
                        return "any"  # Return a generic type as fallback
                        
                client_utils.get_type = patched_get_type
                logging.debug("Applied monkey patch to gradio_client.utils.get_type")
                
            # Also patch the APIInfoParseError to prevent terminal crashes
            try:
                # Import needed for custom patch
                from gradio_client.utils import APIInfoParseError
                
                # Create a safe wrapper for the get_api_info method to catch schema errors
                def safe_get_api_info_wrapper(original_get_api_info):
                    def wrapped_get_api_info(self):
                        try:
                            return original_get_api_info(self)
                        except APIInfoParseError as e:
                            logging.error(f"API info parse error: {e}")
                            # Return a minimal valid API info dictionary that won't crash
                            return {
                                "title": "Fallback API Info",
                                "version": "0.1.0",
                                "description": "Error in API info generation, using fallback",
                                "components": []
                            }
                        except Exception as e:
                            logging.error(f"Unexpected error in get_api_info: {e}")
                            # Return a minimal valid API info dictionary
                            return {
                                "title": "Emergency Fallback API Info",
                                "version": "0.1.0",
                                "description": f"Error: {str(e)}",
                                "components": []
                            }
                    return wrapped_get_api_info
                
                # Apply the wrapper to the Blocks class if it exists
                import gradio.blocks
                if hasattr(gradio.blocks.Blocks, 'get_api_info'):
                    original_api_info = gradio.blocks.Blocks.get_api_info
                    gradio.blocks.Blocks.get_api_info = safe_get_api_info_wrapper(original_api_info)
                    logging.debug("Applied safe wrapper to Blocks.get_api_info")
            except Exception as e:
                logging.warning(f"Failed to patch API info error handling: {e}")
            
    except Exception as patch_error:
        logging.warning(f"Failed to apply Gradio client utils patch: {patch_error}")

# Check that all required environment variables are set
required_vars = ["PINECONE_API_KEY", "OPENAI_API_KEY", "CONFLUENCE_DOMAIN", "CONFLUENCE_ACCESS_TOKEN", "WEBSITE_PORT"]
missing_vars = [var for var in required_vars if os.getenv(var) is None]
if missing_vars:
    logging.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

# Function to check if a port is available
def is_port_available(port: int) -> bool:
    """Check if a port is available for use.
    
    This function attempts to bind to the specified port to verify it's available.
    It uses proper socket options to ensure reliable results.
    
    Args:
        port: The port number to check
        
    Returns:
        bool: True if port is available, False otherwise
    """
    try:
        logging.debug(f"Checking if port {port} is available")
        
        # Create socket with appropriate options
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Set SO_REUSEADDR to allow the socket to be reused immediately
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            logging.debug("Set SO_REUSEADDR on socket")
            
            # Timeout to prevent hanging
            s.settimeout(2.0)
            logging.debug(f"Set socket timeout to {s.gettimeout()} seconds")
            
            # Try to bind to the port
            logging.debug(f"Attempting to bind to port {port}")
            try:
                s.bind(('localhost', port))
                logging.debug(f"Successfully bound to port {port}")
                return True
            except socket.error as e:
                if e.errno == errno.EADDRINUSE:
                    logging.debug(f"Port {port} is already in use")
                else:
                    logging.debug(f"Failed to bind to port {port}: {e}")
                return False
    except Exception as e:
        logging.warning(f"Error checking port {port}: {str(e)}")
        return False  # Assume not available if there's an error

# Utility function to find an available port
def find_available_port(start_port: int, end_port: int) -> Optional[int]:
    """Find an available port in the given range.
    
    Args:
        start_port: Lower bound of port range to check (inclusive)
        end_port: Upper bound of port range to check (inclusive)
        
    Returns:
        Optional[int]: Available port number or None if none found
    """
    for port in range(start_port, end_port + 1):
        if is_port_available(port):
            return port
    return None

# Check that WEBSITE_PORT is a valid integer and isn't already in use
GRADIO_SERVER_PORT = os.getenv("WEBSITE_PORT")
if not GRADIO_SERVER_PORT:
    logging.error("GRADIO_SERVER_PORT environment variable is not set")
    print("Error: GRADIO_SERVER_PORT environment variable is not set")
    sys.exit(1)

try:
    port_number = int(GRADIO_SERVER_PORT)
    logging.debug(f"Parsed GRADIO_SERVER_PORT as integer: {port_number}")
    if not 1 <= port_number <= 65535:
        logging.error(f"GRADIO_SERVER_PORT must be between 1 and 65535, got {GRADIO_SERVER_PORT}")
        print(f"Error: GRADIO_SERVER_PORT must be between 1 and 65535, got {GRADIO_SERVER_PORT}")
        sys.exit(1)
except ValueError:
    logging.error(f"GRADIO_SERVER_PORT must be a valid integer, got {GRADIO_SERVER_PORT}")
    print(f"Error: GRADIO_SERVER_PORT must be a valid integer, got {GRADIO_SERVER_PORT}")
    sys.exit(1)

# Global variable to store the current conversation history
current_history = []

# Function to extract information
def extract_info(data):
    logging.debug(f"Extracting information from {len(data['matches']) if data and 'matches' in data else 'empty'} matches")
    extracted_info = []
    for match in data['matches']:
        source = match['metadata']['source']
        text = match['metadata']['text']
        score = match['score']
        extracted_info.append((source, text, score))
    logging.debug(f"Extracted {len(extracted_info)} pieces of information")
    return extracted_info

# Function to get full page content
def get_full_content(source_url):
    logging.debug(f"Getting full content for URL: {source_url}")
    try:
        # Get authenticated Confluence client
        confluence = get_confluence_client()
        print(f"Successfully connected to Confluence client")
        
        # Extract page ID from URL
        page_id = source_url.split('pageId=')[-1].split('&')[0]  # Handle any additional URL parameters
        print(f"Attempting to fetch content for page ID: {page_id}")
        
        try:
            # Try to get the page directly by ID first
            try:
                page_content = confluence.get_page_by_id(page_id, expand='body.storage,title')
                if page_content:
                    content = page_content['body']['storage']['value']
                    title = page_content['title']
                    print(f"Successfully retrieved content for page: {title}")
                    
                    # Clean the content before returning
                    print(f"Content: {content}")
                    cleaned_content = clean_confluence_markup(display_content(content))
                    if os.getenv('DEBUG') == 'True':
                        print(f"Cleaned content before returning: {cleaned_content}")
                    return f"Content of '{title}':\n\n{cleaned_content}"
            except Exception as id_error:
                print(f"Could not fetch page directly by ID: {str(id_error)}")
        
            # If direct ID access fails, try searching by title
            print("\nAttempting to search for page by title...")
            
            # First try to get all spaces
            print("Fetching available spaces...")
            spaces = confluence.get_all_spaces(start=0, limit=50)
            
            if not spaces or 'results' not in spaces:
                print("No spaces found")
                return "I couldn't access any Confluence spaces. Please check your permissions."
            
            # Try each space to find the page
            for space in spaces['results']:
                # Skip personal spaces (they start with ~)
                if space['key'].startswith('~'):
                    continue
                    
                print(f"\nSearching in space: {space['key']} ({space.get('name', 'Unknown')})")
                try:
                    # Use CQL to search for pages in this space
                    cql = f'space = "{space["key"]}" AND title ~ "AI Resources"'
                    print(f"Searching with CQL: {cql}")
                    search_results = confluence.cql(cql, expand='body.storage,title', include_archived_spaces=False, limit=10)
                    
                    if not search_results or 'results' not in search_results:
                        print(f"No matching pages found in space {space['key']}")
                        continue
                        
                    results = search_results.get('results', [])
                    print(f"Found {len(results)} potential matches in space {space['key']}")
                    
                    for result in results:
                        try:
                            content = result['body']['storage']['value']
                            title = result['title']
                            print(f"Found matching page: {title}")
                            
                            # Clean the content before returning
                            cleaned_content = clean_confluence_markup(content)
                            print(f"Cleaned content before returning: {cleaned_content}")  # Debugging output
                            return f"Content of '{title}':\n\n{cleaned_content}"
                        except Exception as content_error:
                            print(f"Error extracting content from search result: {str(content_error)}")
                            continue
                            
                except Exception as space_error:
                    print(f"Error searching space {space['key']}: {str(space_error)}")
                    continue
            
            print("\nPage not found in any space")
            return "I couldn't find the page in any accessible space. This could be because:\n1. The page has been deleted or moved\n2. The page ID is incorrect\n3. The page is in a restricted space\n\nPlease verify the page exists and try again."
            
        except Exception as e:
            error_msg = str(e)
            print(f"Detailed error message: {error_msg}")
            
            if "permission" in error_msg.lower():
                return "I apologize, but I don't have permission to access this page. This could be because:\n1. The page requires specific access permissions\n2. The authentication token has expired\n3. The page is in a restricted space\n\nPlease try accessing the page directly through your browser."
            elif "no content" in error_msg.lower():
                return "I couldn't find the page you're looking for. This could be because:\n1. The page has been deleted or moved\n2. The page ID is incorrect\n3. The page is in a different space\n\nPlease verify the page exists and try again."
            else:
                return f"Sorry, I encountered an error while trying to fetch the page content: {error_msg}"
                
    except Exception as e:
        print(f"Error fetching page content: {str(e)}")
        return "Sorry, I encountered an error while trying to connect to Confluence. Please try again later."

# Function to clean Confluence markup
def clean_confluence_markup(content):
    """Remove Confluence-specific markup from content while preserving structure."""
    logging.debug("Cleaning Confluence markup")
    try:
        import re
        
        # First, extract content from within ac:rich-text-body tags
        rich_text_matches = re.finditer(r'<ac:rich-text-body[^>]*>(.*?)</ac:rich-text-body>', content, re.DOTALL)
        extracted_content = []
        for match in rich_text_matches:
            extracted_content.append(match.group(1))
        
        # If we found rich-text-body content, use that
        if extracted_content:
            content = '\n'.join(extracted_content)
        
        # Remove user references
        content = re.sub(r'<ac:link<ri:user[^>]*>.*?</ac:link>', '', content)
        content = re.sub(r'<ri:user[^>]*>.*?</ri:user>', '', content)
        
        # Remove image tags
        content = re.sub(r'<ac:image[^>]*>.*?</ac:image>', '', content)
        
        # Replace Confluence-specific tags with appropriate markdown
        replacements = {
            r'<h1[^>]*>': '# ',
            r'</h1>': '\n',
            r'<h2[^>]*>': '## ',
            r'</h2>': '\n',
            r'<h3[^>]*>': '### ',
            r'</h3>': '\n',
            r'<p[^>]*>': '',
            r'</p>': '\n\n',
            r'<br[^>]*>': '\n',
            r'<strong[^>]*>': '**',
            r'</strong>': '**',
            r'<em[^>]*>': '*',
            r'</em>': '*',
            r'<ul[^>]*>': '',
            r'</ul>': '',
            r'<ol[^>]*>': '',
            r'</ol>': '',
            r'<li[^>]*>': '- ',
            r'</li>': '\n',
            r'<a[^>]*href="([^"]*)"[^>]*>': r'[\1](',
            r'</a>': ')',
            r'<ac:layout[^>]*>': '',
            r'</ac:layout>': '',
            r'<ac:layout-section[^>]*>': '',
            r'</ac:layout-section>': '',
            r'<ac:layout-cell[^>]*>': '',
            r'</ac:layout-cell>': '',
            r'<ac:structured-macro[^>]*>': '',
            r'</ac:structured-macro>': '',
            r'<ac:parameter[^>]*>none</ac:parameter>': '',  # Remove the "none" parameter
            r'<ac:parameter[^>]*>': '',  # Remove any other parameters
            r'</ac:parameter>': '',
            r'<ri:page[^>]*>': '',
            r'</ri:page>': '',
            r'<ri:attachment[^>]*>': '',
            r'</ri:attachment>': '',
            r'<ri:card-appearance[^>]*>': '',
            r'</ri:card-appearance>': '',
            r'<ri:version-at-save[^>]*>': '',
            r'</ri:version-at-save>': '',
            r'<ri:space-key[^>]*>': '',
            r'</ri:space-key>': '',
            r'<ri:content-title[^>]*>': '',
            r'</ri:content-title>': '',
            r'<ri:filename[^>]*>': '',
            r'</ri:filename>': '',
            r'<ri:alt[^>]*>': '',
            r'</ri:alt>': '',
            r'<ri:width[^>]*>': '',
            r'</ri:width>': '',
            r'<ri:custom-width[^>]*>': '',
            r'</ri:custom-width>': '',
            r'<ri:original-height[^>]*>': '',
            r'</ri:original-height>': '',
            r'<ri:original-width[^>]*>': '',
            r'</ri:original-width>': '',
            r'<ri:align[^>]*>': '',
            r'</ri:align>': ''
        }
        
        # Apply replacements
        for pattern, replacement in replacements.items():
            content = re.sub(pattern, replacement, content)
        
        # Clean up extra whitespace and newlines
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = content.strip()
        
        # Remove any remaining HTML-like tags
        content = re.sub(r'<[^>]+>', '', content)
        
        # Clean up any remaining user references
        content = re.sub(r'\(shared by.*?\)', '', content)
        content = re.sub(r'\(Shared by.*?\)', '', content)
        
        return content
    except Exception as e:
        print(f"Error cleaning markup: {str(e)}")
        return content

# Function to format content in a structured way
def format_content(content, title):
    """Format the content in a structured, readable way."""
    try:
        # Clean up Confluence markup first
        content = clean_confluence_markup(content)
        
        # Split content into sections based on headers
        sections = []
        current_section = {"title": "Overview", "content": []}
        
        # Split content into lines and process
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line is a header (starts with # or is all caps)
            if line.startswith('#') or line.isupper():
                # Only add the current section if it has content
                if current_section["content"]:
                    sections.append(current_section)
                current_section = {"title": line.lstrip('#').strip(), "content": []}
            else:
                current_section["content"].append(line)
        
        # Add the last section if it has content
        if current_section["content"]:
            sections.append(current_section)
        
        # Format the output - only include the title
        output = f"# {title}\n\n"
        
        # If no sections were found or they're all empty, just return the content directly
        if not sections:
            return f"# {title}\n\n{content}"
        
        # Add each section with its content
        for section in sections:
            # Skip empty sections
            if not section["content"]:
                continue
                
            # Add proper section heading if it's not 'Overview' or if Overview has content
            if section["title"] and section["title"] != "Overview":
                output += f"## {section['title']}\n"
            
            # Add the section content
            output += '\n'.join(section['content'])
            output += '\n\n'
        
        return output.strip()  # Remove any trailing whitespace
    except Exception as e:
        print(f"Error formatting content: {str(e)}")
        return f"# {title}\n\n{content}"

# Function to process user message for special commands
def process_message(message, history):
    logging.debug(f"Processing message: {message}")
    # Check for content rearrangement requests
    rearrangement_keywords = ["rearrange", "reorganize", "restructure", "format", "organize"]
    if any(keyword in message.lower() for keyword in rearrangement_keywords):
        try:
            # First, check if we have a URL in the message
            if "pageId=" in message:
                # Extract the URL and use it directly
                url_start = message.find("http")
                url_end = message.find(" ", url_start) if message.find(" ", url_start) != -1 else len(message)
                source_url = message[url_start:url_end]
                print(f"Found URL in message, attempting to fetch content: {source_url}")
                content = get_full_content(source_url)
                if content and "Content of '" in content:
                    # Extract title and content
                    title_start = content.find("'") + 1
                    title_end = content.find("'", title_start)
                    title = content[title_start:title_end]
                    actual_content = content[content.find("\n\n") + 2:]
                    
                    # Format the content
                    formatted_content = format_content(actual_content, title)
                    return f'<div class="question-banner" style="background-color: #002D72 !important; color: white !important;">Content of \'{title}\'</div>\n{formatted_content}'
            
            # If no URL or URL fetch failed, try to find the page in recent history
            if history and len(history) > 0:
                last_response = history[-1][1]
                if "Similarity:" in last_response:
                    # Try to find a matching reference
                    page_title = message.split("of")[-1].strip().strip('"').strip("'").strip("?")
                    print(f"Looking for page with title similar to: '{page_title}' in references")
                    
                    # Parse the links from the HTML
                    import re
                    links = re.findall(r'<a href="([^"]+)"[^>]*>([^<]+)</a>', last_response)
                    
                    # Try each reference
                    for link_url, link_text in links:
                        if page_title.lower() in link_text.lower() or page_title.lower() in link_url.lower():
                            source_url = link_url
                            print(f"Found matching reference, attempting to fetch content: {source_url}")
                            content = get_full_content(source_url)
                            if content and "Content of '" in content:
                                # Extract title and content
                                title_start = content.find("'") + 1
                                title_end = content.find("'", title_start)
                                title = content[title_start:title_end]
                                actual_content = content[content.find("\n\n") + 2:]
                                
                                # Format the content
                                formatted_content = format_content(actual_content, title)
                                return f'<div class="question-banner" style="background-color: #002D72 !important; color: white !important;">Content of \'{title}\'</div>\n{formatted_content}'
            
            return (
                "I couldn't find the content to rearrange. Please try:\n"
                "1. First ask a question about the topic to get reference links\n"
                "2. Then use 'rearrange content of [page title]' with the page title\n"
                "For example: 'rearrange content of Project Guidelines'"
            )
                
        except Exception as e:
            print(f"Error rearranging content: {str(e)}")
            return (
                "Sorry, I encountered an error while trying to rearrange the content. Please try:\n"
                "1. First ask a question about the topic to get reference links\n"
                "2. Then use 'rearrange content of [page title]' with the page title\n"
                "For example: 'rearrange content of Project Guidelines'"
            )

    # Check for "show content" command
    if message.lower().startswith("show content"):
        try:
            # Extract reference number or text
            ref_selector = message.lower().replace("show content", "").strip().strip(":").strip()
            
            # Try to treat it as a number first
            try:
                ref_num = int(ref_selector)
                # Find the URL in the references
                source_url = None
                if history and len(history) > 0:
                    last_response = history[-1][1]
                    if "Similarity:" in last_response:
                        # Parse the links from the HTML
                        import re
                        links = re.findall(r'<a href="([^"]+)"[^>]*>([^<]+)</a>', last_response)
                        
                        if 0 < ref_num <= len(links):
                            source_url = links[ref_num-1][0]
                print(f"Attempting to fetch content from URL: {source_url}")
                
                if not source_url:
                    return "Could not find the reference number in the previous response. Please check the number and try again."
                
                # Get full content
                content = get_full_content(source_url)
                if content:
                    # Extract title if available
                    if "Content of '" in content:
                        title_start = content.find("'") + 1
                        title_end = content.find("'", title_start)
                        title = content[title_start:title_end]
                        # Format with blue header
                        content_start = content.find("\n\n") + 2
                        banner = f'<div class="question-banner" style="background-color: #002D72 !important; color: white !important;">Content of \'{title}\'</div>'
                        return f'<div style="background-color: white !important; color: black !important;">{banner}\n{content[content_start:]}</div>'
                    return f'<div style="background-color: white !important; color: black !important;">{content}</div>'
                else:
                    return f'<div style="background-color: white !important; color: black !important;">Sorry, I couldn\'t retrieve the full content of the page. Please try accessing the URL directly.</div>'
            except ValueError:
                # If not a number, try to find a matching link by text
                if history and len(history) > 0:
                    last_response = history[-1][1]
                    if "Similarity:" in last_response:
                        # Parse the links from the HTML
                        import re
                        links = re.findall(r'<a href="([^"]+)"[^>]*>([^<]+)</a>', last_response)
                        
                        for link_url, link_text in links:
                            if ref_selector.lower() in link_text.lower() or ref_selector.lower() in link_url.lower():
                                print(f"Attempting to fetch content from URL: {link_url}")
                                
                                # Get full content
                                content = get_full_content(link_url)
                                if content:
                                    # Extract title if available
                                    if "Content of '" in content:
                                        title_start = content.find("'") + 1
                                        title_end = content.find("'", title_start)
                                        title = content[title_start:title_end]
                                        # Format with blue header
                                        content_start = content.find("\n\n") + 2
                                        banner = f'<div class="question-banner">Content of \'{title}\'</div>'
                                        return banner + '\n' + content[content_start:]
                    return content
                else:
                    return "Sorry, I couldn't retrieve the full content of the page. Please try accessing the URL directly."
                
                return "Please use the format 'show content <number>' where number is the reference number, or 'show content [page title]'."
        except Exception as e:
            print(f"Error processing show content command: {str(e)}")
            return "Please use the format 'show content <number>' where number is the reference number, or 'show content [page title]'."
    
    # Check for direct page content request
    if any(phrase in message.lower() for phrase in ["show me the content of", "output the contents of", "display the content of", "show the content of"]):
        try:
            # First, check if we have a URL in the message
            if "pageId=" in message:
                # Extract the URL and use it directly
                url_start = message.find("http")
                url_end = message.find(" ", url_start) if message.find(" ", url_start) != -1 else len(message)
                source_url = message[url_start:url_end]
                print(f"Found URL in message, attempting to fetch content: {source_url}")
                content = get_full_content(source_url)
                if content:
                    # Extract title if available
                    if "Content of '" in content:
                        title_start = content.find("'") + 1
                        title_end = content.find("'", title_start)
                        title = content[title_start:title_end]
                        # Format with blue header
                        content_start = content.find("\n\n") + 2
                        banner = f'<div class="question-banner">Content of \'{title}\'</div>'
                        return banner + '\n' + content[content_start:]
                    return content
            
            # If no URL or URL fetch failed, try to find the page in recent history
            if history and len(history) > 0:
                last_response = history[-1][1]
                if "Similarity:" in last_response:
                    # Parse the links from the HTML
                    import re
                    links = re.findall(r'<a href="([^"]+)"[^>]*>([^<]+)</a>', last_response)
                    
                    # Try to find a matching reference
                    page_title = message.split("of")[-1].strip().strip('"').strip("'").strip("?")
                    print(f"Looking for page with title similar to: '{page_title}' in references")
                    
                    # Try each reference
                    for link_url, link_text in links:
                        if page_title.lower() in link_text.lower() or page_title.lower() in link_url.lower():
                            print(f"Found matching reference, attempting to fetch content: {link_url}")
                            content = get_full_content(link_url)
                            if content:
                                # Extract title if available
                                if "Content of '" in content:
                                    title_start = content.find("'") + 1
                                    title_end = content.find("'", title_start)
                                    title = content[title_start:title_end]
                                    # Format with blue header
                                    content_start = content.find("\n\n") + 2
                                    banner = f'<div class="question-banner">Content of \'{title}\'</div>'
                                    return banner + '\n' + content[content_start:]
                                return content
            
            return (
                "I couldn't find the page you're looking for. Please try:\n"
                "1. First ask a question about the topic to get reference links\n"
                "2. Then use 'show content <number>' or 'show content [page title]'\n"
                "For example: 'show content 1' or 'show content Project Guidelines'"
            )
                
        except Exception as e:
            print(f"Error fetching page content: {str(e)}")
            return (
                "Sorry, I encountered an error. Please try:\n"
                "1. First ask a question about the topic to get reference links\n"
                "2. Then use 'show content <number>' or 'show content [page title]'\n"
                "For example: 'show content 1' or 'show content Project Guidelines'"
            )
    
    return None

# Function to initialize Pinecone
def init_pinecone():
    logging.info("Initializing Pinecone")
    try:
        # Get Pinecone API key from environment variables
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        if not pinecone_api_key:
            logging.error("PINECONE_API_KEY not found in environment variables")
            print("Error: PINECONE_API_KEY not found in environment variables")
            return None
            
        # Get Pinecone environment from environment variables
        pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')
        if not pinecone_environment:
            logging.error("PINECONE_ENVIRONMENT not found in environment variables")
            print("Error: PINECONE_ENVIRONMENT not found in environment variables")
            return None
            
        # Initialize Pinecone
        index_name = os.getenv('PINECONE_INDEX_NAME', 'default-index')
        logging.debug(f"Getting Pinecone index: {index_name}")
        index, _ = get_pinecone_index(index_name)
        
        logging.info("Pinecone initialized successfully")
        print("Done: Pinecone initialized successfully")
        return index
    except Exception as e:
        logging.exception(f"Error initializing Pinecone: {str(e)}")
        print(f"Error initializing Pinecone: {str(e)}")
        return None

# Function to query Pinecone
def query_pinecone(index, query):
    logging.info(f"Querying Pinecone with: {query}")
    
    # Generate embedding for the query with specific error handling
    try:
        logging.debug("Generating embedding for query")
        embedding_response = get_embeddings(query, "text-embedding-ada-002")
        query_embedding = embedding_response.data[0].embedding
        logging.debug(f"Generated embedding with length: {len(query_embedding)}")
    except Exception as embed_err:
        logging.error(f"Error generating embeddings: {str(embed_err)}", exc_info=True)
        raise RuntimeError(f"Failed to generate embedding: {str(embed_err)}") from embed_err
    
    # Query Pinecone with specific error handling for the API call
    try:
        logging.debug("Executing Pinecone query")
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
    except Exception as pinecone_err:
        logging.error(f"Error in Pinecone API call: {str(pinecone_err)}", exc_info=True)
        raise RuntimeError(f"Failed to query Pinecone: {str(pinecone_err)}") from pinecone_err
    
    # Process results with error handling
    try:
        logging.debug(f"Received query results with {len(results['matches']) if 'matches' in results else 'no'} matches")
        
        if 'matches' not in results or not results['matches']:
            logging.warning("No matches found in Pinecone results")
            return []
            
        extracted_info = extract_info(results)
        
        if not extracted_info:
            logging.warning("No information extracted from results")
            return []
            
        logging.info(f"Pinecone query completed successfully with {len(extracted_info)} results")
        print("Done: Pinecone query completed")
        return extracted_info
    except Exception as process_err:
        logging.error(f"Error processing Pinecone results: {str(process_err)}", exc_info=True)
        raise RuntimeError(f"Failed to process search results: {str(process_err)}") from process_err

# Function to generate response with OpenAI
def generate_answer(query, context_data, chat_history=[]):
    logging.info(f"Generating answer for: {query}")
    
    # Validate context data
    if not context_data:
        logging.warning("No context data provided for answer generation")
        return "I couldn't find relevant information to answer your question."
    
    try:
        # Create system prompt
        logging.debug("Creating system prompt")
        system_prompt = create_system_prompt()
        
        # Create chat messages
        messages = []
        messages = add_prompt_messages("system", system_prompt, messages)
        
        # Add chat history for context (up to 5 past exchanges)
        history_context = ""
        recent_history = chat_history[-5:] if len(chat_history) > 5 else chat_history
        if recent_history:
            logging.debug(f"Adding {len(recent_history)} history items to context")
            history_context = "Previous conversation:\n"
            for h_user, h_bot in recent_history:
                if h_user:  # Only add if user message exists
                    history_context += f"User: {h_user}\n"
                    if h_bot:  # Only add if bot message exists
                        # Strip HTML if present
                        if isinstance(h_bot, str) and ("<div" in h_bot or "<span" in h_bot):
                            try:
                                from bs4 import BeautifulSoup
                                soup = BeautifulSoup(h_bot, 'html.parser')
                                h_bot_clean = soup.get_text(separator=' ', strip=True)
                                history_context += f"Assistant: {h_bot_clean}\n\n"
                            except Exception as html_err:
                                logging.warning(f"Could not parse HTML in history: {str(html_err)}")
                                history_context += f"Assistant: [Previous response]\n\n"
                        else:
                            history_context += f"Assistant: {h_bot}\n\n"
        
        # Create context string from retrieved data
        search_context = ""
        logging.debug(f"Adding {len(context_data)} search results to context")
        for i, (source, text, score) in enumerate(context_data, 1):
            # Skip invalid data
            if not text:
                logging.warning(f"Skipping empty text for source {source}")
                continue
                
            # Add some of the text as context (trim if too long)
            content_preview = text[:1500] + "..." if len(text) > 1500 else text
            search_context += f"Source {i} ({source}):\n{content_preview}\n\n"
        
        # Check if we have any valid search context
        if not search_context:
            logging.warning("No valid search context found after processing")
            return "I couldn't extract useful information from the search results."
        
        # Create the prompt with context
        prompt = f"Based on the following information, please answer the question. If referencing specific sources, mention them in your answer.\n\n"
        
        if history_context:
            prompt += f"{history_context}\n"
            
        prompt += f"Search results:\n{search_context}\n\nCurrent question: {query}\n\nAnswer:"
        messages = add_prompt_messages("user", prompt, messages)
        
        # Get response from OpenAI with specific error handling
        try:
            logging.debug("Sending request to OpenAI")
            response = get_chat_completion_messages(messages, "gpt-3.5-turbo", temperature=0.3)
            
            if not response or not response.strip():
                logging.warning("Received empty response from OpenAI")
                return "I received an empty response. Please try again."
                
            logging.debug(f"Received response from OpenAI: {response[:100]}...")
            return response
        except Exception as openai_err:
            logging.error(f"Error in OpenAI API call: {str(openai_err)}", exc_info=True)
            raise RuntimeError(f"Failed to get response from OpenAI: {str(openai_err)}") from openai_err
            
    except Exception as e:
        logging.exception(f"Error generating answer: {str(e)}")
        raise RuntimeError(f"Error generating answer: {str(e)}") from e

# Function to log conversation
def log_conversation(user_input, assistant_response, session_id, results=None):
    logging.debug(f"Logging conversation for session: {session_id}")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get the full conversation history
    conversation_history = []
    for user_msg, bot_msg in current_history:
        conversation_history.append({
            "user": user_msg,
            "assistant": bot_msg
        })
    
    log_entry = {
        "timestamp": timestamp,
        "session_id": session_id,
        "user_input": user_input,
        "assistant_response": assistant_response,
        "search_results": [(source, score) for source, _, score in results] if results else [],
        "conversation_history": conversation_history
    }
    
    # Create log file with session ID in the name
    log_file = os.path.join(LOGS_DIR, f"conversation_{session_id}.jsonl")
    
    # Append the log entry to the file
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    print(f"Conversation logged to {log_file}")

# Function to handle message selection
def select_message(evt: gr.SelectData):
    global current_history
    selected_index = evt.index[0]
    if selected_index < len(current_history):
        return current_history[selected_index][0]  # Return the user message
    return ""

# Function to fork conversation
def fork_conversation(message, history, selected_message):
    global current_history
    if not selected_message:
        return history
    
    # Find the index of the selected message
    for i, (user_msg, _) in enumerate(history):
        if user_msg == selected_message:
            # Truncate history to the selected message
            history = history[:i]
            # Add the new message
            history.append((message, ""))
            current_history = history.copy()
            return history
    
    return history

# Main chat function for Gradio
def chat_function(message, history):
    logging.info(f"Chat function called with message: {message}")
    try:
        # Process the message and get response
        special_response = process_message(message, history)
        
        if special_response:
            # Wrap special response for consistency and add copy button if needed
            # For simplicity, let's assume special responses don't need a copy button for now
            # Or we can add it here too if desired
            return history + [(message, special_response)]
            
        # Initialize Pinecone
        index = init_pinecone()
        if not index:
            logging.error("Failed to initialize Pinecone index")
            return history + [(message, "Sorry, I couldn't connect to the knowledge base. Please try again later.")]
            
        # Define SVG for the copy button
        copy_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="16" height="16"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>"""
        copy_button_html = f'<button class="copy-button" title="Copy response" onclick="copyToClipboard(this)">{copy_svg}</button>'

        # Generate session ID early for consistent logging
        session_id_str = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Query Pinecone with enhanced error handling
        try:
            results = query_pinecone(index, message)
            if not results:
                logging.warning(f"No matches found in Pinecone for query: {message}")
                not_found_answer = "I couldn't find any relevant information. Please try rephrasing your question."
                formatted_response = f"""
                <div class="chat-response">
                    {copy_button_html}
                    <div class="question-banner">
                        {html.escape(message) if message else "Initial Query Response"}
                    </div>
                    <div class="answer-content">
                        {html.escape(not_found_answer)}
                    </div>
                    <div class="session-info">
                        Session ID: {session_id_str}
                    </div>
                </div>
                """
                log_conversation(message, formatted_response, session_id_str, results)
                return history + [(None, formatted_response)] # Use None for user msg part
        except Exception as pinecone_err:
            logging.error(f"Error during Pinecone query: {str(pinecone_err)}")
            error_answer = f"Failed to search the knowledge base: {str(pinecone_err)}"
            formatted_response = f"""
            <div class="chat-response">
                {copy_button_html}
                <div class="question-banner error-banner" style="background-color: #f8d7da !important; color: #721c24 !important;">
                    {html.escape(message) if message else "Query Error"}
                </div>
                <div class="answer-content error-content" style="color: #721c24 !important;">
                    {html.escape(error_answer)}
                </div>
                <div class="session-info">
                    Session ID: {session_id_str}
                </div>
            </div>
            """
            log_conversation(message, formatted_response, session_id_str, None)
            return history + [(None, formatted_response)]
            
        # Log the number of matches found
        logging.info(f"Found {len(results)} matches in Pinecone")
            
        # Generate answer with OpenAI with enhanced error handling
        try:
            answer = generate_answer(message, results, history)
            if not answer:
                logging.warning("OpenAI returned empty response")
                error_answer = "I couldn't generate an answer. Please try again."
                formatted_response = f"""
                <div class="chat-response">
                    {copy_button_html}
                    <div class="question-banner">
                        {html.escape(message) if message else "Initial Query Response"}
                    </div>
                    <div class="answer-content">
                        {html.escape(error_answer)}
                    </div>
                    <div class="session-info">
                        Session ID: {session_id_str}
                    </div>
                </div>
                """
                log_conversation(message, formatted_response, session_id_str, results)
                return history + [(None, formatted_response)]
        except Exception as openai_err:
            logging.error(f"Error during OpenAI answer generation: {str(openai_err)}")
            error_answer = f"Failed to generate answer: {str(openai_err)}"
            formatted_response = f"""
            <div class="chat-response">
                {copy_button_html}
                <div class="question-banner error-banner" style="background-color: #f8d7da !important; color: #721c24 !important;">
                    {html.escape(message) if message else "Generation Error"}
                </div>
                <div class="answer-content error-content" style="color: #721c24 !important;">
                    {html.escape(error_answer)}
                </div>
                <div class="session-info">
                    Session ID: {session_id_str}
                </div>
            </div>
            """
            log_conversation(message, formatted_response, session_id_str, results)
            return history + [(None, formatted_response)]
        
        # Build sources HTML
        sources_html = ""
        if results:
            sources_items = []
            for idx, (source, _, score) in enumerate(results, 1):
                if source and score is not None:
                    similarity_percentage = score * 100
                    sources_items.append(f'<li>Source {idx}: <a href="{source}" class="source-link" target="_blank">{html.escape(source)}</a> <span class="similarity-score">[Similarity: {similarity_percentage:.2f}%]</span></li>')
                else:
                    sources_items.append("<li>Invalid source data</li>")
            sources_list = "\n".join(sources_items)
            sources_html = f"""
            <div class="sources-section">
                <p class="sources-header">Sources:</p>
                <ul class="sources-list">
                    {sources_list}
                </ul>
            </div>
            """
        else:
            sources_html = """
            <div class="sources-section">
                <p class="sources-header">Sources:</p>
                <ul class="sources-list">
                    <li>No sources found.</li>
                </ul>
            </div>
            """

        # Format the response with the copy button and structure
        formatted_response = f"""
        <div class="chat-response">
            {copy_button_html}
            <div class="question-banner">
                 {html.escape(message) if message else "Initial Query Response"}
            </div>
            <div class="answer-content">
                {html.escape(answer)}
            </div>
            {sources_html}
            <div class="session-info">
                Session ID: {session_id_str}
            </div>
            <div class="instruction-text">
                To view the full content of any page, type "show content" followed by the reference number.
            </div>
        </div>
        """
        
        # Validate the HTML response (optional but good practice)
        try:
            soup = BeautifulSoup(formatted_response, 'html.parser')
            # Ensure the main div has the class
            if not soup.div.has_attr('class') or 'chat-response' not in soup.div['class']:
                 soup.div['class'] = soup.div.get('class', []) + ['chat-response']
            formatted_response = str(soup)
        except Exception as e:
            logging.error(f"Error validating HTML response: {str(e)}")

        # Log the conversation
        # Make sure logged response is the final HTML string
        log_conversation(message, formatted_response, session_id_str, results)

        # Return None for the user message part to avoid duplication in chatbot UI
        return history + [(None, formatted_response)]

    except Exception as e:
        logging.error(f"Unhandled error in chat function: {str(e)}", exc_info=True)
        # Return a user-friendly error message in the chat history format
        # Also wrap error message for copy button consistency
        session_id_str = datetime.now().strftime("%Y%m%d%H%M%S")
        copy_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="16" height="16"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>"""
        copy_button_html_error = f'<button class="copy-button" title="Copy response" onclick="copyToClipboard(this)">{copy_svg}</button>'
        error_message_content = f"An error occurred: {html.escape(str(e))}"
        formatted_error_response = f"""
        <div class="chat-response">
             {copy_button_html_error}
             <div class="question-banner error-banner" style="background-color: #f8d7da !important; color: #721c24 !important;">
                 {html.escape(message) if message else "Initial Query Error"}
             </div>
             <div class="answer-content error-content" style="color: #721c24 !important;">
                 {error_message_content}
             </div>
              <div class="session-info">
                Session ID: {session_id_str}
            </div>
        </div>
        """
        log_conversation(message, formatted_error_response, session_id_str) # Log error too
        return history + [(None, formatted_error_response)] # Use None for user msg part

# Function to run the initial query on page load
def run_initial_query():
    """Runs an initial query to describe the first Confluence space."""
    logging.info("Attempting to run initial query...")
    try:
        # 1. Get Confluence client
        confluence = get_confluence_client()
        if not confluence:
            print("Initial Query: Failed to get Confluence client.")
            return [] # Return empty history update

        # 2. Fetch spaces
        print("Initial Query: Fetching Confluence spaces...")
        # Fetch a small number of spaces to find the first one quickly
        spaces = confluence.get_all_spaces(start=0, limit=10, expand='name')

        if not spaces or 'results' not in spaces or not spaces['results']:
            print("Initial Query: No spaces found or accessible.")
            # Optional: Return a message indicating no spaces found
            # return [(None, "Could not find any Confluence spaces to describe.")]
            return []

        # 3. Find the first non-personal space
        first_space_name = None
        for space in spaces['results']:
            # Skip personal spaces (keys usually start with '~')
            if not space['key'].startswith('~'):
                # Use the space name if available, otherwise fallback to the key
                first_space_name = space.get('name', space['key'])
                print(f"Initial Query: Found first non-personal space: {first_space_name} (Key: {space['key']})")
                break

        if not first_space_name:
            print("Initial Query: No non-personal spaces found among the first fetched spaces.")
            # Optional: Return a message indicating no suitable spaces found
            # return [(None, "Could not find a suitable Confluence space to describe.")]
            return []

        # 4. Construct the initial query
        initial_query = f"Give me a description of the {first_space_name} space."
        print(f"Initial Query: Constructed query: '{initial_query}'")

        # 5. Call chat_function with the initial query and empty history
        # chat_function expects (message, history) and returns updated history
        initial_history_update = chat_function(initial_query, [])
        print("Initial Query: chat_function completed.")

        # 6. Return the result to update the chatbot
        # The result should be in the format Gradio expects for Chatbot output: List[Tuple[str | None, str | None]]
        return initial_history_update

    except Exception as e:
        error_message = f"Error during initial query setup: {str(e)}"
        print(error_message)
        # Format error for Gradio Chatbot
        formatted_error = f'<div style="color: red; padding: 10px;">Failed to run initial query: {html.escape(str(e))}</div>'
        # Return the error message in the expected history format
        return [(None, formatted_error)]

# Legacy main function for backward compatibility
def main(query):
    logging.info(f"Legacy main function called with query: {query}")
    # Create mock history for single query usage
    history = []
    response = chat_function(query, history)
    return response

def display_content(content):
    logging.debug(f"Display content called with content length: {len(content) if content else 0}")
    # Debugging output to check the content before any modifications
    if DEBUG:
        logging.debug(f"Original content before processing: {content}")
    print(f"Original content before processing: {content}")
    
    if content.startswith("none#"):
        content = content[5:]  # Remove the "none#" prefix
        logging.debug(f"Content after removing 'none#': {content[:100]}...")  # Truncated debug output
        print(f"Content after removing 'none#': {content}")  # Debugging output
    
    # Continue with the rest of the display logic
    return content

# Socket monitoring thread function for DEBUG mode
def monitor_socket_health(port: int, check_interval: int = 30) -> None:
    """Periodically check if the server socket is still healthy.
    
    Args:
        port: The port to monitor
        check_interval: How often to check in seconds
    """
    logging.debug(f"Starting socket health monitoring for port {port}")
    failures = 0
    
    while True:
        try:
            # Wait for check interval
            time.sleep(check_interval)
            
            # Check if socket is responsive
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                result = s.connect_ex(('localhost', port))
                
                if result == 0:
                    # Connection succeeded - socket is responsive
                    if failures > 0:
                        logging.info(f"Socket on port {port} is responsive again after {failures} failures")
                    failures = 0
                else:
                    # Connection failed - socket not responding
                    failures += 1
                    logging.warning(f"Socket health check failed on port {port} (failure #{failures})")
                    
                    if failures >= 3:
                        logging.error(f"Socket health check failed {failures} times in a row")
                        print(f"Warning: Server on port {port} appears to be unresponsive")
        except Exception as e:
            logging.warning(f"Error in socket monitoring: {str(e)}")
            failures += 1

# Function to test Gradio configuration
def test_gradio_config() -> bool:
    """Test the Gradio configuration to ensure it's valid.
    
    Tests API info generation and socket binding, providing useful
    diagnostics for common configuration issues.
    
    Returns:
        bool: True if configuration test passes, False otherwise
    """
    try:
        # Import needed modules
        import inspect
        import importlib
        import gradio
        
        # Log Gradio version
        logging.info(f"Gradio version: {gradio.__version__}")
        
        # Check for known problematic versions
        if gradio.__version__ == "4.44.1":
            logging.warning("You're using Gradio 4.44.1 which has known issues with JSON schema handling")
            logging.info("Applied patches should help, but consider upgrading to a newer version if problems persist")
        
        # Test basic API info generation to catch compatibility issues
        logging.debug("Testing API info generation (common failure point)")
        
        # Create a minimal test blocks instance
        test_blocks = gradio.Blocks()
        with test_blocks:
            gradio.Textbox(label="Test")
            
        # Try to generate API info without fully launching
        try:
            if hasattr(test_blocks, "get_api_info"):
                api_info = test_blocks.get_api_info()
                logging.debug("API info generation successful")
            else:
                logging.debug("Blocks.get_api_info not found, skipping test")
        except Exception as e:
            logging.warning(f"API info generation test failed: {str(e)}")
            logging.warning("Applying emergency patch to bypass API info generation")
            
            # Apply emergency patch to bypass API info
            try:
                import gradio.blocks
                
                # Create a safe fallback version of get_api_info
                def safe_minimal_api_info(self):
                    return {
                        "title": "Emergency API Info",
                        "version": "1.0",
                        "description": "Minimal API info due to error in normal generation",
                        "components": []
                    }
                
                # Apply the emergency patch
                gradio.blocks.Blocks.get_api_info = safe_minimal_api_info
                logging.info("Applied emergency API info patch")
            except Exception as patch_err:
                logging.error(f"Failed to apply emergency patch: {patch_err}")
                return False
            
        # Test socket binding
        try:
            # Try to find an available port first
            test_port = find_available_port(10000, 10100)
            if not test_port:
                logging.warning("Could not find test port for binding test")
                return False
                
            logging.debug(f"Testing socket binding on port {test_port}")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # Set socket options for better reuse
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('localhost', test_port))
                s.listen(1)
                logging.debug("Socket binding test passed")
        except Exception as e:
            logging.warning(f"Socket binding test failed: {str(e)}")
            return False
        
        # All tests passed
        return True
    except Exception as e:
        logging.warning(f"Gradio configuration test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Define JavaScript for copy functionality
    js_copy_code = """
    function copyToClipboard(button) {
        const chatResponse = button.closest('.chat-response');
        const answerContent = chatResponse.querySelector('.answer-content');
        if (answerContent) {
            const textToCopy = answerContent.innerText || answerContent.textContent;
            navigator.clipboard.writeText(textToCopy).then(() => {
                // Optional: Provide feedback to the user, e.g., change button text/icon
                const originalContent = button.innerHTML;
                button.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="16" height="16"><path d="M9 16.17 4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/></svg>'; // Checkmark icon
                setTimeout(() => {
                    button.innerHTML = originalContent;
                }, 1500); // Revert after 1.5 seconds
            }).catch(err => {
                console.error('Failed to copy text: ', err);
                alert('Failed to copy text.');
            });
        } else {
            console.error('Could not find answer content to copy.');
        }
    }
    """

    # Create a custom light theme for Gradio
    light_theme = gr.themes.Default(
        primary_hue="gray",
        secondary_hue="gray",
        neutral_hue="slate",
        text_size=gr.themes.sizes.text_md,
    ).set(
        body_background_fill="#FFFFFF",
        background_fill_primary="#FFFFFF",
        background_fill_secondary="#FFFFFF",
        block_background_fill="#FFFFFF",
        block_label_background_fill="#FFFFFF",
        block_title_background_fill="#FFFFFF",
        border_color_accent="#FFFFFF",
        border_color_primary="#FFFFFF",
        color_accent="#FFFFFF",
        input_background_fill="#FFFFFF",
        body_text_color="black",
        block_shadow="none",
        button_primary_background_fill="#FFFFFF",
        button_primary_text_color="black",
    )
    
    # Create Gradio interface with chat components
    gr.close_all()
    
    with gr.Blocks(
        title="Confluence Knowledge Assistant", 
        theme=light_theme,
        css="web_resources/main.css",
        js=js_copy_code # Uncomment to inject the JavaScript code
    ) as demo:
        # Create a header with JHU branding
        with gr.Row(elem_id="header"):
            with gr.Column(scale=1):
                # Construct src using f-string to avoid literal /file= sequence
                src_path = "=web_resources/university.shield.rgb.blue.svg"
                header_html = f'''<div class="header-container">    <img src="/file{src_path}" alt="JHU Logo" class="jhu-logo" style="max-width: 100px;">    <div class="title-container">        <h1 class="jhu-title">JOHNS HOPKINS UNIVERSITY</h1>        <h2 class="assistant-title">Confluence Knowledge Assistant</h2>    </div></div>'''
                gr.HTML(header_html)
        
        # Create the chatbot interface
        chatbot = gr.Chatbot(
            elem_id="chatbot",
            avatar_images=(None, None),
            show_label=False,
            height=600,
            sanitize_html=False,
        )
        
        # Create message input and clear button
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<div class="input-prefix">Query</div>')
            with gr.Column(scale=8):
                msg = gr.Textbox(
                    placeholder="Ask a question about your content...",
                    container=False,
                    scale=8,
                )
            with gr.Column(scale=1):
                submit = gr.Button("Submit")
        
        # Bind the submit button and text input to the chat function
        submit_click_event = submit.click(
            chat_function,
            inputs=[msg, chatbot],
            outputs=[chatbot],
            api_name="chat",
        )
        
        # Bind the Enter key to submit
        enter_event = msg.submit(
            chat_function,
            inputs=[msg, chatbot],
            outputs=[chatbot],
            api_name="chat_submit",
        )
        
        # Add JavaScript to clear the input after submission
        enter_event.then(lambda: "", None, msg)
        submit_click_event.then(lambda: "", None, msg)

        # Run the initial query when the Gradio app loads
        demo.load(run_initial_query, inputs=None, outputs=chatbot)

    try:
        # Test Gradio configuration before launching
        if DEBUG:
            test_result = test_gradio_config()
            if not test_result:
                logging.warning("Gradio configuration test failed, but attempting to launch anyway")
            else:
                logging.info("Gradio configuration test passed")
                
        # Check port availability and find alternative if needed
        if not is_port_available(port_number):
            logging.error(f"Port {port_number} is already in use")
            print(f"Error: Port {port_number} is already in use")
            
            # Try to find an alternative port when in DEBUG mode
            logging.debug("Attempting to find an alternative port")
            alt_port = find_available_port(port_number + 1, port_number + 100)
            if alt_port:
                logging.info(f"Found alternative port {alt_port} - will use this instead")
                print(f"Found alternative port {alt_port} - using this instead of {port_number}")
                port_number = alt_port
            else:
                logging.error("Could not find an alternative port - exiting")
                sys.exit(1)
        
        # Log server launch details
        logging.info(f"Launching Gradio app on port {port_number}")
        
        # Use a simpler launch approach
        try:
            logging.debug("Launching Gradio with basic configuration")
            demo.launch(
                server_name="0.0.0.0",  # Allow external connections
                server_port=port_number,
                share=False,
                show_api=False,
                quiet=not DEBUG
            )
            logging.info(f"Gradio launch successful on port {port_number}")
        except Exception as launch_error:
            logging.error(f"Failed to launch Gradio: {str(launch_error)}")
            
            # Try with alternative configuration if first attempt fails
            try:
                logging.debug("Trying alternative launch configuration")
                alt_port = find_available_port(port_number + 100, port_number + 200)
                if alt_port:
                    logging.info(f"Attempting to launch on alternative port {alt_port}")
                    print(f"Trying alternative port {alt_port}")
                    
                    demo.launch(
                        server_name="0.0.0.0",
                        server_port=alt_port,
                        share=False,
                        show_api=False,
                        auth=lambda username, password: True,  # Disable auth checks
                        prevent_thread_lock=True
                    )
                    port_number = alt_port  # Update port for monitoring
                    logging.info(f"Gradio launch successful on port {alt_port}")
                else:
                    logging.error("No alternative ports available")
                    print("Error: Could not find an available port to launch the application")
                    sys.exit(1)
            except Exception as e:
                logging.error(f"All launch attempts failed: {str(e)}")
                print(f"Error: Failed to start Gradio application: {str(e)}")
                sys.exit(1)
        
        # Start socket monitoring if DEBUG is enabled
        if DEBUG:
            try:
                monitoring_thread = threading.Thread(
                    target=monitor_socket_health, 
                    args=(port_number,), 
                    daemon=True
                )
                monitoring_thread.start()
                logging.debug("Socket monitoring thread started")
            except Exception as e:
                logging.warning(f"Failed to start monitoring thread: {str(e)}")
        
    except Exception as e:
        logging.exception(f"Error during Gradio server setup: {str(e)}")
        print(f"Error: {e}")
        sys.exit(1)
