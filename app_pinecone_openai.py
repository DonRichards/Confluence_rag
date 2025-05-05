from bs4 import BeautifulSoup
import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import gradio as gr
import html
import re  # Add import for regular expressions
from utils.pinecone_logic import delete_pinecone_index, get_pinecone_index, upsert_data, init_pinecone, query_pinecone
from utils.data_prep import import_csv, clean_data_pinecone_schema, generate_embeddings_and_add_to_df
from utils.openai_logic import get_embeddings, create_prompt, add_prompt_messages, get_chat_completion_messages, create_system_prompt
from utils.auth import get_confluence_client
import sys
import time
import json
from datetime import datetime, timedelta
import logging
from openai import OpenAI
from gradio import HTML

# load environment variables
load_dotenv(find_dotenv())

# Set up logging directory
LOGS_DIR = os.path.join(os.getcwd(), "conversation_logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Global variables
current_history = []
session_id = None
current_space = None  # Track the currently selected space
current_time_filter = None  # Track the time-based filter

# Function to extract information
def extract_info(data):
    extracted_info = []
    for match in data['matches']:
        source = match['metadata']['source']
        text = match['metadata']['text']
        score = match['score']
        extracted_info.append((source, text, score))
    return extracted_info

# Function to get full page content
def get_full_content(source_url):
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

def calculate_date_range(time_filter):
    """Calculate start and end dates based on a time filter string."""
    today = datetime.now()
    
    # Default to last 7 days if no valid filter
    start_date = today - timedelta(days=7)
    end_date = today
    
    if time_filter:
        time_filter = time_filter.lower()
        if "this week" in time_filter:
            # Get the start of the current week (Monday)
            days_since_monday = today.weekday()
            # Modified: Include the previous two weeks to ensure we get enough content
            start_date = today - timedelta(days=days_since_monday + 14)
            # Add helpful debug output
            print(f"Modified 'this week' date range to include recent days: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        elif "last week" in time_filter:
            # Get the start of last week (Monday)
            days_since_monday = today.weekday()
            start_date = today - timedelta(days=days_since_monday + 7)
            end_date = start_date + timedelta(days=6)
        elif "this month" in time_filter:
            # Start of current month
            start_date = today.replace(day=1)
        elif "last month" in time_filter:
            # Start of last month
            if today.month == 1:
                start_date = today.replace(year=today.year-1, month=12, day=1)
            else:
                start_date = today.replace(month=today.month-1, day=1)
            # End of last month
            end_date = start_date.replace(day=28) + timedelta(days=4)  # This will get us to the next month
            end_date = end_date.replace(day=1) - timedelta(days=1)  # Back up one day to the end of the month
        elif "today" in time_filter:
            start_date = today
        elif "yesterday" in time_filter:
            start_date = today - timedelta(days=1)
            end_date = start_date
        elif "recent" in time_filter:
            # Modified: Make "recent" include last 14 days instead of 7
            start_date = today - timedelta(days=14)
            print(f"Using extended 'recent' date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        elif "past" in time_filter and "days" in time_filter:
            # Parse "past X days"
            try:
                days = int(''.join(filter(str.isdigit, time_filter)))
                start_date = today - timedelta(days=days)
            except ValueError:
                # Default to 7 days if we can't parse the number
                start_date = today - timedelta(days=7)
    
    # Make sure both dates are timezone-naive to avoid comparison issues
    if start_date.tzinfo is not None:
        start_date = start_date.replace(tzinfo=None)
    if end_date.tzinfo is not None:
        end_date = end_date.replace(tzinfo=None)
        
    return start_date, end_date

def extract_meeting_content(content):
    """Extract structured information from meeting content."""
    import re
    
    # Initialize extracted data
    extracted = {
        "topics": [],
        "decisions": [],
        "action_items": [],
        "participants": []
    }
    
    # Convert to lowercase for case-insensitive matches, but keep original for extraction
    content_lower = content.lower()
    
    # Extract topics
    topic_patterns = [
        r'(?:agenda|topics?|discussed?|discussion|covered)[\s\:]+([^\n]+)',
        r'(?:topics|agenda)[\s\:]*\n+((?:.+\n)+)',
        r'(?<=\n)[\*\-\•]\s+([^\n]+)',  # Bullet points often indicate topics
        r'^\s*[\*\-\•]\s+([^\n]+)',     # Bullet points at start of lines
        r'<h\d>[^<]*(?:topic|agenda|discussion)[^<]*</h\d>\s*([^<]+)',  # HTML headings
    ]
    
    for pattern in topic_patterns:
        matches = re.finditer(pattern, content_lower)
        for match in matches:
            topic = match.group(1).strip()
            if topic and len(topic) > 5 and topic not in extracted["topics"]:
                # Extract the actual text from the original content to preserve casing
                original_text = content[match.start(1):match.end(1)].strip()
                if len(original_text) > 5:  # Ensure we don't add empty or too short topics
                    extracted["topics"].append(original_text)
    
    # Extract decisions
    decision_patterns = [
        r'(?:decision|decided|agreed|agreement|conclusion)[\s\:]+([^\n]+)',
        r'(?:decision|decided|agreed|agreement|conclusion)[\s\:]*\n+((?:.+\n)+)',
        r'<h\d>[^<]*(?:decision|outcome)[^<]*</h\d>\s*([^<]+)',  # HTML headings
    ]
    
    for pattern in decision_patterns:
        matches = re.finditer(pattern, content_lower)
        for match in matches:
            decision = match.group(1).strip()
            if decision and len(decision) > 5 and decision not in extracted["decisions"]:
                # Extract the actual text from the original content
                original_text = content[match.start(1):match.end(1)].strip()
                if len(original_text) > 5:
                    extracted["decisions"].append(original_text)
    
    # Extract action items
    action_patterns = [
        r'(?:action|task|todo|to do|action item|next step)[\s\:]+([^\n]+)',
        r'(?:action|task|todo|to do|action item|next step)[\s\:]*\n+((?:.+\n)+)',
        r'(?:\n)[\*\-\•]\s+([A-Z][^:]+:[^\n]+)',  # Bullet points with names often indicate action items
        r'<h\d>[^<]*(?:action|task)[^<]*</h\d>\s*([^<]+)',  # HTML headings
    ]
    
    for pattern in action_patterns:
        matches = re.finditer(pattern, content_lower)
        for match in matches:
            action = match.group(1).strip()
            if action and len(action) > 5 and action not in extracted["action_items"]:
                # Extract the actual text from the original content
                original_text = content[match.start(1):match.end(1)].strip()
                if len(original_text) > 5:
                    extracted["action_items"].append(original_text)
    
    # Extract participants
    participant_patterns = [
        r'(?:attendees|participants|present|attended|attendance)[\s\:]+([^\n]+)',
        r'(?:attendees|participants|present|attended|attendance)[\s\:]*\n+((?:.+\n)+)',
        r'<h\d>[^<]*(?:attendees|participants)[^<]*</h\d>\s*([^<]+)',  # HTML headings
    ]
    
    for pattern in participant_patterns:
        matches = re.finditer(pattern, content_lower)
        for match in matches:
            participants = match.group(1).strip()
            if participants and len(participants) > 3 and participants not in extracted["participants"]:
                # Extract the actual text from the original content
                original_text = content[match.start(1):match.end(1)].strip()
                if len(original_text) > 3:
                    extracted["participants"].append(original_text)
    
    # If no structured data was found, extract the full text as a fallback
    if not any(extracted.values()):
        # Get the first 500 characters as a summary if nothing structured was found
        extracted["full_text"] = content[:500] + "..." if len(content) > 500 else content
    
    return extracted

# Add a function to replace future years in text
def replace_future_dates(text):
    """Replace future year references in text with current year."""
    import re
    
    # Get current year
    current_year = datetime.now().year
    
    # Look for year patterns: standalone years or within dates
    year_patterns = [
        (r'\b(20[2-9][5-9])\b', str(current_year)),  # Standalone years 2025-2099
        (r'(\d{1,2}/\d{1,2}/)20[2-9][5-9]', f'\\1{current_year}'),  # MM/DD/2025+
        (r'(\d{1,2}-\d{1,2}-)20[2-9][5-9]', f'\\1{current_year}'),  # MM-DD-2025+
        (r'(\w+ \d{1,2}, )20[2-9][5-9]', f'\\1{current_year}'),     # Month DD, 2025+
        (r'(\d{1,2} \w+ )20[2-9][5-9]', f'\\1{current_year}'),      # DD Month 2025+
    ]
    
    # Apply replacements
    processed_text = text
    for pattern, replacement in year_patterns:
        processed_text = re.sub(pattern, replacement, processed_text)
    
    # Replace specific month-year patterns like "March 2025"
    month_year_pattern = r'(\b(?:January|February|March|April|May|June|July|August|September|October|November|December) 20[2-9][5-9]\b)'
    
    def replace_month_year(match):
        month = match.group(0).split()[0]
        return f"{month} {current_year}"
    
    processed_text = re.sub(month_year_pattern, replace_month_year, processed_text)
    
    # Replace date formats like 2025-03-31 with current year
    iso_date_pattern = r'\b(20[2-9][5-9]-\d{2}-\d{2})\b'
    
    def replace_iso_date(match):
        date_parts = match.group(0).split('-')
        return f"{current_year}-{date_parts[1]}-{date_parts[2]}"
    
    processed_text = re.sub(iso_date_pattern, replace_iso_date, processed_text)
    
    return processed_text

# Add a new function for enhanced meeting content handling
def enhance_meeting_content(content, source_url):
    """Enhance meeting content by adding more structure and extracting key information."""
    import re
    
    # Process the content to replace future years first
    content = replace_future_dates(content)
    
    # Check if this is a meeting note
    meeting_patterns = [
        r'meeting notes', r'meeting minutes', r'standup', r'check-in',
        r'sprint (meeting|review)', r'daily (meeting|scrum)',
        r'sync', r'retrospective'
    ]
    
    is_meeting = False
    for pattern in meeting_patterns:
        if re.search(pattern, content.lower()) or re.search(pattern, source_url.lower()):
            is_meeting = True
            break
    
    if not is_meeting:
        return content  # Return original if not a meeting note
    
    # Try to extract meeting date
    date_patterns = [
        r'(\d{4}-\d{1,2}-\d{1,2})',  # YYYY-MM-DD
        r'(\d{1,2}/\d{1,2}/\d{4})',  # MM/DD/YYYY
        r'(\w+ \d{1,2},? \d{4})',    # Month DD, YYYY
        r'(\d{1,2} \w+ \d{4})'       # DD Month YYYY
    ]
    
    meeting_date = None
    for pattern in date_patterns:
        match = re.search(pattern, content)
        if match:
            meeting_date = match.group(1)
            break
    
    # Extract structured information
    extracted = extract_meeting_content(content)
    
    # Prepare enhanced content
    enhanced = ""
    
    # Add the meeting date if found
    if meeting_date:
        # Check if this is a date in the current year
        current_year = str(datetime.now().year)
        if current_year in meeting_date:
            enhanced += f"Meeting date: {meeting_date}\n\n"
        else:
            # Remove the year entirely to avoid confusion
            meeting_date = re.sub(r'\b\d{4}\b', '', meeting_date).strip().rstrip(',')
            enhanced += f"Meeting date: {meeting_date} (recent)\n\n"
    
    # Add a title if one can be extracted from the source URL or content
    title = None
    # Try to extract title from source URL
    title_match = re.search(r'([^/]+)(?:\.html?|\.aspx)?$', source_url)
    if title_match:
        potential_title = title_match.group(1)
        # Clean up the title - replace hyphens with spaces, etc.
        potential_title = potential_title.replace('-', ' ').replace('_', ' ')
        if len(potential_title) > 3:
            title = potential_title
    
    # If title found, add it to enhanced content
    if title:
        # Remove any future years from the title
        title = replace_future_dates(title)
        enhanced = f"Meeting: {title}\n\n" + enhanced
    
    # Add topics
    if extracted.get("topics"):
        enhanced += "Topics discussed:\n"
        for topic in extracted["topics"]:
            enhanced += f"- {topic}\n"
        enhanced += "\n"
    
    # Add decisions
    if extracted.get("decisions"):
        enhanced += "Decisions made:\n"
        for decision in extracted["decisions"]:
            enhanced += f"- {decision}\n"
        enhanced += "\n"
    
    # Add action items
    if extracted.get("action_items"):
        enhanced += "Action items:\n"
        for action in extracted["action_items"]:
            enhanced += f"- {action}\n"
        enhanced += "\n"
    
    # Add participants
    if extracted.get("participants"):
        enhanced += "Participants:\n"
        for participant in extracted["participants"]:
            enhanced += f"- {participant}\n"
        enhanced += "\n"
    
    # If no structured data was found, include the full text
    if not any([extracted.get("topics"), extracted.get("decisions"), 
                extracted.get("action_items"), extracted.get("participants")]) and extracted.get("full_text"):
        enhanced += "Full meeting content:\n"
        enhanced += extracted["full_text"]
    
    # If we couldn't extract anything useful, return the processed content
    if not enhanced.strip():
        return content
    
    return enhanced

# Function to generate a response using OpenAI
def generate_response(query, query_results, space_filter=None, time_filter=None):
    """Generate a response based on a query and the retrieved documents."""
    try:
        # Get OpenAI API key
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            logging.error("OpenAI API key not found")
            return "Error: OpenAI API key not found"
            
        client = OpenAI(api_key=openai_api_key)
        
        # Preprocess the results to replace future years
        processed_results = []
        for i, result in enumerate(query_results):
            if len(result) >= 4:
                source_url, text_content, score, date_info = result
                # Replace future years in content
                processed_text = replace_future_dates(text_content)
                # Replace future years in the source URL if it contains them
                processed_source = replace_future_dates(source_url)
                processed_results.append((processed_source, processed_text, score, date_info))
            elif len(result) >= 3:
                source_url, text_content, score = result
                # Replace future years in content
                processed_text = replace_future_dates(text_content)
                # Replace future years in the source URL if it contains them
                processed_source = replace_future_dates(source_url)
                processed_results.append((processed_source, processed_text, score))
            else:
                processed_results.append(result)  # Keep as is if format is unexpected
        
        # Use the processed results for the rest of the function
        query_results = processed_results
        
        # Check if this is a date-related query
        date_related_query = time_filter is not None or any(term in query.lower() for term in [
            "recent", "latest", "this week", "last week", "this month", 
            "created", "updated", "modified", "edited", "last modified"
        ])
        
        # Check if this is a summary or context-heavy query
        needs_context = any(term in query.lower() for term in [
            "summarize", "summary", "overview", "recap", 
            "what's been happening", "what's going on", 
            "changes", "updates", "progress"
        ])
        
        # Calculate date range based on time filter
        start_date, end_date = calculate_date_range(time_filter)
        if time_filter:
            print(f"Filtering content between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}")
            # Add more verbose date debugging
            print(f"DEBUG: Current system time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"DEBUG: Using time filter: '{time_filter}'")
        
        # Check if we need to discover dependencies and relationships
        meeting_notes_context = False
        if needs_context and query_results:
            # Check if the top results are meeting notes
            meeting_notes_patterns = [
                r'meeting notes', r'meeting minutes', r'standup', r'check-in',
                r'sprint (meeting|review)', r'daily (meeting|scrum)',
                r'sync', r'retrospective'
            ]
            
            # Content based check for meeting notes
            for i, result in enumerate(query_results[:3]):  # Check top 3 results
                if len(result) >= 3:  # Make sure we have content
                    content = result[1] if len(result) > 1 else ""
                    # Check if content matches meeting notes patterns
                    if any(re.search(pattern, content.lower()) for pattern in meeting_notes_patterns):
                        print(f"Detected meeting notes in result {i+1}")
                        meeting_notes_context = True
                        break
        
        # Prepare context from documents
        context = ""
        has_date_info = False
        has_future_dates = False
        all_dates_are_future = True  # Track if all dates are future
        
        # Get current date info for replacement
        today = datetime.now()
        current_week_start = today - timedelta(days=today.weekday())
        yesterday = today - timedelta(days=1)
        
        # Enhanced processing for meeting notes
        processed_results = []
        date_filtered_count = 0
        
        if query_results:
            print(f"DEBUG: Processing {len(query_results)} query results")
            # First, filter and preprocess results based on date if this is a time-based query
            for i, result in enumerate(query_results, 1):
                is_within_date_range = True
                date_info = None
                is_future_date = False
                original_date = None
                
                # Extract date if available
                if len(result) >= 4:  # Has date information
                    source_url, text_content, score, date_info = result
                    original_date = date_info  # Store original for reference
                    has_date_info = True
                    
                    # Try to parse the date (various formats)
                    if date_info and date_related_query:
                        try:
                            from dateutil import parser
                            result_date = parser.parse(date_info)
                            
                            # Check if date is in the future
                            if result_date.date() > datetime.now().date():
                                print(f"DEBUG: Document {i} has a future date: {result_date.date()}")
                                has_future_dates = True
                                is_future_date = True
                                
                                # Replace future date with current date for context clarity
                                # Use yesterday or start of current week for modified dates
                                adjusted_date = yesterday.strftime('%Y-%m-%d')
                                print(f"DEBUG: Replacing future date {date_info} with current date {adjusted_date}")
                                date_info = adjusted_date
                            else:
                                # If at least one date is not in the future, update tracking
                                all_dates_are_future = False
                            
                            # Check if within the date range for time-filtered queries
                            # For future dates, treat them as current/recent automatically
                            if is_future_date:
                                is_within_date_range = True  # Always include future-dated content
                            else:
                                # Modified: For "this week" filter, be more lenient with dates from the few days before
                                if time_filter and "this week" in time_filter.lower():
                                    # Include content from up to a week before the start date
                                    buffer_date = start_date - timedelta(days=7)
                                    is_within_date_range = buffer_date.date() <= result_date.date() <= end_date.date()
                                    if is_within_date_range and result_date.date() < start_date.date():
                                        print(f"DEBUG: Including recent document from just before 'this week' ({result_date.date()})")
                                else:
                                    is_within_date_range = start_date.date() <= result_date.date() <= end_date.date()
                            
                            print(f"DEBUG: Document {i} date: {result_date.date()}, Within range: {is_within_date_range}")
                            if is_within_date_range:
                                date_filtered_count += 1
                        except Exception as e:
                            # If we can't parse the date, include it anyway
                            print(f"DEBUG: Could not parse date: {date_info}, Error: {str(e)}")
                            all_dates_are_future = False  # Can't determine if it's future
                else:
                    source_url, text_content, score = result
                    print(f"DEBUG: Document {i} has no date information")
                    all_dates_are_future = False  # No date means not future
                
                # Process the content, extract meeting details if it looks like meeting notes
                extracted_content = None
                enhanced_content = None
                if len(text_content) > 0:
                    # Always enhance meeting content for better presentation
                    if meeting_notes_context or "meeting" in source_url.lower() or "minutes" in source_url.lower():
                        enhanced_content = enhance_meeting_content(text_content, source_url)
                        extracted_content = extract_meeting_content(text_content)
                
                # Special handling for future dates or regular date filtering
                if is_future_date:
                    # For documents with future dates, include them but mark them specially
                    processed_results.append({
                        "source": source_url,
                        "content": text_content,
                        "enhanced_content": enhanced_content,
                        "score": score,
                        "date": date_info,  # Use the adjusted date
                        "original_date": original_date,  # Store original
                        "extracted": extracted_content,
                        "is_within_range": True,  # Always include future dates
                        "is_future_date": True
                    })
                elif is_within_date_range or not date_related_query:
                    # Regular processing for properly dated documents
                    processed_results.append({
                        "source": source_url,
                        "content": text_content,
                        "enhanced_content": enhanced_content,
                        "score": score,
                        "date": date_info,
                        "original_date": original_date,
                        "extracted": extracted_content,
                        "is_within_range": is_within_date_range,
                        "is_future_date": False
                    })
            
            print(f"DEBUG: After date filtering, {len(processed_results)} results remain (matched date filter: {date_filtered_count})")
            print(f"DEBUG: Future dates detected: {has_future_dates}")
            print(f"DEBUG: All dates are future: {all_dates_are_future}")
            
            # NEW: If no results matched the date filter but we're looking for "this week" content,
            # include the most recent documents anyway (likely from late April)
            if date_related_query and len(processed_results) == 0 and time_filter:
                print("DEBUG: No documents matched strict date filter. Including most recent documents as fallback.")
                # Get the most recent documents (up to 5)
                recent_docs = []
                
                for result in query_results:
                    if len(result) >= 4:  # Has date information
                        source, text_content, score, date_info = result
                        try:
                            from dateutil import parser
                            date_obj = parser.parse(date_info)
                            recent_docs.append((date_obj, {"source": source, "content": text_content, "score": score, "date": date_info}))
                        except:
                            # If we can't parse the date, add it with a very old date so it's at the end
                            recent_docs.append((datetime.min, {"source": source, "content": text_content, "score": score, "date": date_info}))
                
                # Sort by date (most recent first)
                recent_docs.sort(key=lambda x: x[0], reverse=True)
                
                # Add the most recent documents (up to 5)
                for _, doc in recent_docs[:5]:
                    processed_results.append({
                        "source": doc["source"],
                        "content": doc["content"],
                        "enhanced_content": None,
                        "score": doc["score"],
                        "date": doc["date"],
                        "original_date": doc["date"],
                        "extracted": None,
                        "is_within_range": True,  # Force inclusion
                        "is_future_date": False
                    })
                
                print(f"DEBUG: Added {len(processed_results)} recent documents as fallback")
            
            # Sort results by date if date information is available
            if has_date_info and date_related_query:
                from dateutil import parser
                
                def get_date(item):
                    if item["date"]:
                        try:
                            return parser.parse(item["date"])
                        except:
                            return datetime.min
                    return datetime.min
                
                processed_results.sort(key=get_date, reverse=True)
            
            # Build the context string
            for i, result in enumerate(processed_results, 1):
                source = result["source"]
                content = result.get("enhanced_content") or result["content"]  # Use enhanced content if available
                score = result["score"]
                date = result["date"]
                original_date = result.get("original_date")
                extracted = result["extracted"]
                is_future_date = result.get("is_future_date", False)
                
                # Add debug output to view content
                content_preview = content[:100] + "..." if len(content) > 100 else content
                print(f"DEBUG: Document {i} content preview: {content_preview}")
                print(f"DEBUG: Document {i} content length: {len(content)}")
                
                # Add a note for future-dated content
                future_note = ""
                if is_future_date:
                    # Explicitly indicate that the original date was in the future and was replaced
                    future_note = f" (NOTE: Original date {original_date} was incorrect and has been adjusted for context)"
                
                if date:
                    context += f"Page {i} (Score: {score:.2f}, Last Modified: {date}){future_note}:\n"
                else:
                    context += f"Page {i} (Score: {score:.2f}){future_note}:\n"
                    
                context += f"Source: {source}\n"
                context += f"Content: {content}\n"
                
                # Only add extracted meeting data separately if not using enhanced content
                if extracted and not result.get("enhanced_content"):
                    context += "Extracted Meeting Data:\n"
                    if extracted.get("topics"):
                        context += "  Topics: " + ", ".join(extracted["topics"][:5]) + "\n"
                    if extracted.get("decisions"):
                        context += "  Decisions: " + ", ".join(extracted["decisions"][:3]) + "\n"
                    if extracted.get("action_items"):
                        context += "  Action Items: " + ", ".join(extracted["action_items"][:3]) + "\n"
                    if extracted.get("participants"):
                        context += "  Participants: " + ", ".join(extracted["participants"][:5]) + "\n"
                
                context += "\n"
        
        # Recursive context discovery for meeting notes if needed
        additional_context = ""
        historical_context = ""
        if meeting_notes_context and space_filter:
            print("Performing recursive context discovery for meeting notes...")
            
            # Look for project information or previous meetings
            from utils.pinecone_logic import init_pinecone, query_pinecone
            
            index = init_pinecone()
            if index:
                # 1. Find project context - search for project descriptions in the same space
                project_query = "project overview description goals objectives roadmap"
                project_results = query_pinecone(
                    index, 
                    project_query, 
                    filter_by_space=space_filter,
                    similarity_threshold=0.3, 
                    top_k=3
                )
                
                # 2. Find previous meeting notes (looking back further)
                # IMPORTANT: Do not apply date filtering to historical context
                extended_time_window = "last month" if time_filter == "this week" else "last quarter"
                previous_meetings_query = "previous meeting notes minutes agenda"
                previous_results = query_pinecone(
                    index, 
                    previous_meetings_query, 
                    filter_by_space=space_filter,
                    similarity_threshold=0.3, 
                    top_k=5
                )
                
                # 3. Look for action items and decisions
                # IMPORTANT: Do not apply date filtering to action items
                action_items_query = "action items decisions next steps followup"
                action_items_results = query_pinecone(
                    index, 
                    action_items_query, 
                    filter_by_space=space_filter,
                    similarity_threshold=0.3, 
                    top_k=3
                )
                
                # Add project context
                if project_results:
                    additional_context += "=== ADDITIONAL PROJECT CONTEXT ===\n\n"
                    for i, result in enumerate(project_results, 1):
                        if len(result) >= 3:
                            source, text, score = result[:3]
                            additional_context += f"Project Context {i} (Score: {score:.2f}):\n"
                            additional_context += f"Source: {source}\n"
                            additional_context += f"Content: {text}\n\n"
                
                # Process historical meeting notes
                # Store previous meetings and action items as historical context instead
                # We'll track it separately to ensure it doesn't override time-filtered results
                if previous_results:
                    historical_context += "=== PREVIOUS MEETING NOTES (FOR REFERENCE ONLY) ===\n\n"
                    
                    # Process each result, check for future dates
                    for i, result in enumerate(previous_results, 1):
                        # Extract all available information
                        source = content = score = date_info = None
                        is_future_date = False
                        
                        if len(result) >= 4:  # With date info
                            source, content, score, date_info = result
                            
                            # Check for future dates
                            try:
                                from dateutil import parser
                                parsed_date = parser.parse(date_info)
                                if parsed_date.date() > datetime.now().date():
                                    is_future_date = True
                                    print(f"DEBUG: Historical context doc {i} has future date: {parsed_date.date()}")
                            except:
                                pass
                        elif len(result) >= 3:
                            source, content, score = result
                        
                        # Add a note for future dates
                        future_note = " (NOTE: Has future date)" if is_future_date else ""
                        
                        # Add to historical context with appropriate details
                        if date_info:
                            historical_context += f"Previous Meeting {i} (Score: {score:.2f}, Date: {date_info}){future_note}:\n"
                        else:
                            historical_context += f"Previous Meeting {i} (Score: {score:.2f}){future_note}:\n"
                        
                        historical_context += f"Source: {source}\n"
                        historical_context += f"Content: {content}\n\n"
                
                # Process action items, checking for future dates
                if action_items_results:
                    historical_context += "=== ACTION ITEMS AND DECISIONS (FOR REFERENCE ONLY) ===\n\n"
                    
                    # Process each result, check for future dates
                    for i, result in enumerate(action_items_results, 1):
                        # Extract all available information
                        source = content = score = date_info = None
                        is_future_date = False
                        
                        if len(result) >= 4:  # With date info
                            source, content, score, date_info = result
                            
                            # Check for future dates
                            try:
                                from dateutil import parser
                                parsed_date = parser.parse(date_info)
                                if parsed_date.date() > datetime.now().date():
                                    is_future_date = True
                                    print(f"DEBUG: Action item doc {i} has future date: {parsed_date.date()}")
                            except:
                                pass
                        elif len(result) >= 3:
                            source, content, score = result
                        
                        # Add a note for future dates
                        future_note = " (NOTE: Has future date)" if is_future_date else ""
                        
                        # Add to historical context with appropriate details
                        if date_info:
                            historical_context += f"Action Items Source {i} (Score: {score:.2f}, Date: {date_info}){future_note}:\n"
                        else:
                            historical_context += f"Action Items Source {i} (Score: {score:.2f}){future_note}:\n"
                        
                        historical_context += f"Source: {source}\n"
                        historical_context += f"Content: {content}\n\n"
        
        # Add special instructions for date-related queries
        date_instructions = ""
        if date_related_query:
            if has_date_info:
                # If we have an active time filter, make it explicit
                time_filter_instruction = ""
                if time_filter:
                    # Special handling for future dates
                    future_date_instruction = ""
                    if has_future_dates:
                        future_date_instruction = f"""
IMPORTANT: Some documents in this context have metadata dates from the future (year 2025 or beyond).
These dates have been replaced with current dates in the context you see.
The original content is recent/current - just ignore any specific future years or dates mentioned in the content itself.

When you see documents marked with a notice about "Original date was incorrect", treat that content as:
1. CURRENT/RECENT - as in created or modified within the requested time period
2. Created within the date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
3. Relevant to the query about activity in {time_filter}

DO NOT mention the future dates or the date adjustment in your response.
"""
                    
                    time_filter_instruction = f"""
The user has an active time filter: '{time_filter}'.
You should focus on content that was created or modified within this time period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.
{future_date_instruction}
"""
                
                date_instructions = f"""
This query is about recent or time-sensitive content. Please:
1. Pay special attention to the 'Last Modified' dates in the page metadata
2. Sort and prioritize results by recency if appropriate
3. Focus on content from the requested time period ({time_filter if time_filter else "recent"})
{time_filter_instruction}
"""
            else:
                date_instructions = """
This query is about recent or time-sensitive content, but the page metadata doesn't include modification dates.
Please explain that the system cannot currently determine content creation/modification dates and suggest alternatives.
"""
                
        # Add instructions for meeting notes context
        meeting_notes_instructions = ""
        if meeting_notes_context:
            meeting_notes_instructions = """
You're summarizing changes related to meeting notes. For this type of content:

1. EXTRACT AND PRESENT SPECIFIC DETAILS from the meeting notes - don't just mention document names
2. For each meeting within the requested time period, provide:
   - The exact topics discussed and decisions made
   - Specific action items and assigned responsibilities
   - Key participants and their contributions, if mentioned
   - Any specific project updates or status changes mentioned

3. When using the structured content:
   - Focus on the DETAILS in the "Content" section of each document
   - If you see "Topics discussed:", "Decisions made:", etc. sections, use those details directly
   - Quote specific items when particularly important
   - Use the "enhanced content" that has already been extracted for you

4. FORMAT YOUR RESPONSE for clarity:
   - Begin with a summary statement about activity this week
   - Use bullet points for key information organized by meeting
   - Highlight important decisions and action items
   - NEVER say "content descriptions not available" when content is present

5. BE SPECIFIC AND DETAILED:
   - If you see meeting content, always summarize what's there
   - Don't say "I cannot provide specific details" when content is available
   - If content seems partial or unclear, summarize what IS available

6. CRITICAL INSTRUCTION ABOUT DATES:
   - DO NOT mention specific dates like "March 31" in your summary
   - Refer to all meetings as "recent" or "this week" regardless of any dates mentioned in the content
   - Ignore any years like "2025" that might still appear in the content
   - Treat ALL meeting content as current/recent activity THIS WEEK

7. IMPORTANT: If you have processed content from meeting notes, DO NOT claim there were no meetings or no details available.
"""
        
        # Create system message
        system_message = f"""You are a knowledgeable AI assistant that provides helpful and accurate information based on Confluence pages.
If the information isn't in the provided pages, admit that you don't know rather than making up an answer.
When answering, cite the source URLs from the context when appropriate.
Be direct and focus on answering the user's query WITHOUT using phrases like:
- "Based on the provided context..."
- "According to the information available..."
- "From the content provided..."

Current filters active:
- Space filter: {space_filter if space_filter else "None (searching all spaces)"}
- Time filter: {time_filter if time_filter else "None"}

Important terminology guidelines:
- Always refer to sources as "pages" (not "documents") since they come from Confluence
- When listing sources, use "Link #" instead of "Document #"
- Use "View Page" instead of "View Document" for source links
- For each page mentioned, provide specific content and details from that page

When summarizing content:
1. Focus on SPECIFIC DETAILS, not generalities
2. Quote important information directly
3. Extract definite facts, dates, numbers, and names
4. Group information by topic rather than just listing documents
5. Start directly with your summary - no introductory phrases

When dealing with structured content:
1. Recognize and distinguish between different document types (meeting notes, reports, specifications, etc.)
2. Preserve hierarchical structure when presenting information (main ideas → supporting details)
3. Notice when multiple documents are related and synthesize information across them
4. Pay attention to document metadata like dates, authors, and spaces to understand context
5. Identify and highlight key relationships between people, projects, and topics mentioned

When analyzing meeting notes:
1. Identify key decisions, action items, and responsible parties
2. Track discussion points across multiple meetings if available
3. Identify recurring themes or issues
4. Highlight unresolved matters that appear in multiple meetings
5. Link project discussions to relevant specifications or documentation

When the user asks about a specific space or collection:
1. Look for information in the pages that mentions that space by name, key, or ID
2. Pay attention to metadata like URLs containing space information (e.g., spaceKey=SPACENAME)
3. Analyze the content from pages in that space to infer the purpose of the space
4. Provide an overall overview of what content is contained in the space

When the user asks about time-based content (e.g., "content created this week" or "recently updated pages"):
1. Check the 'last_modified' dates in the page metadata
2. Filter and present content based on the requested time period
3. For each page, include its title, last modified date, and a concrete summary of its content with specific details
4. Provide an overall summary of what changed in relation to the space

IMPORTANT FOR TIME-FILTERED QUERIES WITH FEW OR NO RESULTS:
- If no documents match the exact time filter (e.g., "this week"), use information from the most recent available documents 
- Make it clear to the user that you're showing information from a different time period than requested
- Never respond with "I don't have access to specific Confluence pages" when space and relevant content is available
- Always provide the most helpful information you can, even if it's from outside the exact requested time range

IMPORTANT FOR HANDLING DOCUMENT CONTENT:
- If document content appears empty or incomplete, check if metadata like dates and URLs are still available
- Never say "content sections are empty" - instead provide whatever information you do have, like document titles, URLs, and dates
- If multiple documents have limited content, focus on providing a summary of what's available (dates, document names, etc.)
- The fact that documents exist with recent dates is information itself - report on the timing and quantity of updates
""" + date_instructions + meeting_notes_instructions
        
        # Format messages for OpenAI API
        messages = [
            {"role": "system", "content": system_message},
        ]
        
        # Add the main context
        messages.append({"role": "user", "content": f"Given the following Confluence pages:\n\n{context}\n\nQuestion: {query}"})
        
        # Add additional guidance if we have few or no results in the requested date range
        if date_related_query and date_filtered_count == 0 and len(processed_results) > 0:
            messages.append({"role": "assistant", "content": "I notice there are no documents exactly matching the requested time period."})
            messages.append({"role": "user", "content": f"""
IMPORTANT: No documents match the exact requested time period ({time_filter}). 
However, I'm showing you the most recent documents available that are relevant to the query.

Please still provide a helpful response based on these documents, but make sure to:
1. Acknowledge that you're showing information from outside the requested time period
2. Still provide specific details from the available content
3. Do NOT respond with "I don't have access to specific Confluence pages" - you do have access to the content shown above

Your goal is to provide the most helpful response possible with the information available.
"""})
            
        # Check if any of the processed results have empty content and add specific guidance
        has_empty_content = False
        for result in processed_results:
            content = result.get("enhanced_content") or result.get("content", "")
            if not content or len(content.strip()) < 10:  # Consider content empty if less than 10 chars
                has_empty_content = True
                break
                
        if has_empty_content:
            messages.append({"role": "assistant", "content": "I notice some of the documents have limited or no content."})
            messages.append({"role": "user", "content": f"""
IMPORTANT: Some documents have limited or no content text, but the metadata (dates, URLs, titles) is still valuable.

Please provide a response that:
1. Acknowledges the limited content but still uses what information is available
2. Focuses on the metadata - report the dates of recent activity in the {current_space} space
3. Summarizes what we can determine from the document titles and dates
4. Suggests what this activity might indicate (e.g., "There have been 6 document updates in the last week")
5. NEVER says "content sections are empty" - this is unhelpful
6. Frames the information positively in terms of what IS available rather than what isn't

Your goal is to be informative about the recent activity based on document existence and metadata, even if content details are limited.
"""})
        
        # Add additional context if we have it
        if additional_context:
            messages.append({"role": "assistant", "content": "I need some additional context about these meeting notes to provide a complete answer."})
            messages.append({"role": "user", "content": f"Here is additional context to help with understanding the meeting notes and related projects:\n\n{additional_context}"})
        
        # Add historical context as a separate message with a clear warning about time period
        if historical_context and date_related_query:
            messages.append({"role": "assistant", "content": "I'd like to understand the historical context to better interpret the recent information."})
            messages.append({"role": "user", "content": f"""Here is HISTORICAL context from outside the requested time period. 

IMPORTANT: This historical information is PROVIDED ONLY FOR BACKGROUND UNDERSTANDING.
You MUST NOT include this older information directly in your response.
Use it only to understand the significance of the current content in {context}.

{historical_context}

REMEMBER: For a query about '{time_filter}', your response should ONLY summarize content from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.

IF all content has future dates (like years 2025+):
- These are date errors in the metadata
- Treat ALL content with future dates as current/recent content
- Do NOT say "no content matches the time filter" when you have content with future dates
"""})
        
        # If we have content with future dates only, add a clear instruction
        if has_future_dates and len(processed_results) > 0 and all_dates_are_future:
            messages.append({"role": "assistant", "content": "I notice all the content has adjusted dates. How should I handle this?"})
            messages.append({"role": "user", "content": f"""
IMPORTANT CLARIFICATION: All the content found in response to this query originally had future dates (like years 2025+).
These have been adjusted in the context to current dates, but you need to understand:

1. These documents ARE CURRENT - they were created/modified RECENTLY (within {time_filter})
2. The actual content IS relevant to the query about: "{time_filter}"
3. You should summarize this content as recent activity
4. DO NOT mention the date inconsistency in your answer
5. DO NOT claim there were "no updates this week" or similar phrases
6. DO NOT reference specific future years (like 2025) in your summary

For example, if content mentions "a meeting on March 31, 2025", treat it as "a recent meeting" without the specific date.

Focus on THE ACTUAL CONTENT - what was discussed, what decisions were made, what actions were taken, etc.
Your task is to summarize the SUBSTANCE of these documents as if they were all created within {time_filter}.
"""})
        
        # Call OpenAI API to generate response
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            temperature=0.3,
            max_tokens=1000,
            presence_penalty=0.6,  # Encourages model to introduce new information
            frequency_penalty=0.5  # Reduces repetition
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        return f"Error generating response: {str(e)}"

# Function to log conversation
def log_conversation(user_input, assistant_response, session_id, history=None, results=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get the full conversation history
    conversation_history = []
    if history:
        for user_msg, bot_msg in history:
            conversation_history.append({
                "user": user_msg,
                "assistant": bot_msg
            })
    else:
        # Fallback to global current_history
            for user_msg, bot_msg in current_history:
                conversation_history.append({
                    "user": user_msg,
                "assistant": bot_msg
            })
    
    # Process search results to handle both 3-element and 4-element tuples
    search_results = []
    if results:
        for result in results:
            if len(result) >= 4:  # Handle 4-element tuples (with date info)
                source, _, score, date_info = result
                search_results.append({"source": source, "score": score, "date": date_info})
            else:  # Handle 3-element tuples
                source, _, score = result
                search_results.append({"source": source, "score": score})
    
    log_entry = {
        "timestamp": timestamp,
        "session_id": session_id,
        "user_input": user_input,
        "assistant_response": assistant_response,
        "search_results": search_results,
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
            new_history = history[:i+1]
            # Add the new message
            if message:
                # Process the new message with the truncated history
                result = chat_function(message, new_history)
                current_history = result
                return result
            return new_history
    
    return history

# Process space-related queries to improve retrieval
def preprocess_space_query(message):
    """
    Preprocess space-related queries to improve retrieval effectiveness.
    
    Args:
        message: The original user query
        
    Returns:
        str: The preprocessed query optimized for retrieval
    """
    # Check if this is a space-related query
    space_related_keywords = ['space', 'collection', 'namespace', 'group', 'team', 'community', 'project']
    
    # Check if query is specifically about what a space is or its purpose
    purpose_patterns = [
        r'what is the (purpose|point|goal|objective) of',
        r'what is the .* space',
        r'what is .* space for',
        r'what is .* space about',
        r'what does .* space do',
        r'tell me about .* space',
        r'purpose of .* space',
        r'why does .* space exist'
    ]
    
    # Check if query is about time-based content
    time_related_patterns = [
        r'(content|pages|documents) (created|updated|modified|edited) (this|last) (week|month|day)',
        r'(recent|latest|new) (content|pages|documents|updates)',
        r'(what|which) (content|pages|documents) (were|have been) (created|updated|modified|edited)',
        r'show me (content|pages|documents) from (this|last) (week|month|day)',
        r'list (content|pages|documents) (created|updated|modified|edited) (recently|lately)'
    ]
    
    is_purpose_query = False
    is_time_query = False
    import re
    
    # Check for purpose queries
    for pattern in purpose_patterns:
        if re.search(pattern, message.lower()):
            is_purpose_query = True
            break
    
    # Check for time-based queries
    for pattern in time_related_patterns:
        if re.search(pattern, message.lower()):
            is_time_query = True
            break
    
    # If it's a space purpose query, expand it
    if is_purpose_query:
        # Extract space name
        space_name = None
        for pattern in purpose_patterns:
            match = re.search(pattern + r' (the |)([a-zA-Z0-9_\-]+)', message.lower())
            if match:
                space_name = match.group(2)
                break
        
        if space_name:
            # Generate an expanded query
            expanded_query = f"{message} purpose objective focus topics documents content description"
            print(f"Original query: '{message}' expanded to: '{expanded_query}'")
            return expanded_query
    
    # If it's a time-based query, expand it with time-relevant keywords
    if is_time_query:
        # Extract time period
        time_periods = ['recent', 'latest', 'new', 'this week', 'last week', 'this month', 'last month', 'today', 'yesterday']
        detected_period = None
        
        for period in time_periods:
            if period in message.lower():
                detected_period = period
                break
        
        # Generate an expanded query for time-based searches
        expanded_query = f"{message} last_modified date timestamp recently updated created"
        
        if detected_period:
            expanded_query += f" {detected_period}"
            
        print(f"Original time query: '{message}' expanded to: '{expanded_query}'")
        return expanded_query
    
    return message

# Main chat function for Gradio
def chat_function(message, history=None):
    """Generate a chat response based on the user's message and conversation history."""
    # Initialize or update conversation history
    if history is None:
        history = []
        current_history.clear()
    
    # Save the original message for error reporting
    original_message = message
    
    try:
        # Generate a random session ID if none exists yet
        global session_id
        global current_space  # Access the global space selection
        global current_time_filter  # Access the global time filter
        if not session_id:
            import uuid
            session_id = str(uuid.uuid4())
            
        # Create the logs directory if it doesn't exist
        os.makedirs(LOGS_DIR, exist_ok=True)
        
        # Check for content processing requests
        content_response = process_message(message, history)
        if content_response:
            updated_history = history + [(message, content_response)]
            current_history.clear()
            for item in updated_history:
                current_history.append(item)
            log_conversation(message, content_response, session_id, history)
            return updated_history
        
        # Check for conversation forking
        if message.startswith("/fork"):
            try:
                parts = message.split(" ", 2)
                if len(parts) >= 2:
                    selected_index = int(parts[1]) - 1
                    new_message = parts[2] if len(parts) > 2 else ""
                    
                    if 0 <= selected_index < len(history):
                        return fork_conversation(new_message, history, selected_index)
                else:
                        error_message = f"Invalid message index {selected_index + 1}. Please specify a valid message number."
                        updated_history = history + [(message, error_message)]
                        return updated_history
            except Exception as e:
                error_message = f"Error forking conversation: {str(e)}"
                updated_history = history + [(message, error_message)]
                return updated_history
        
        # Check for reset commands
        if message.lower() in ["reset space", "clear space", "/reset space", "/clear space"]:
            old_space = current_space
            current_space = None
            response = f"Space selection cleared. Previously searching in space: '{old_space}'" if old_space else "No space was selected."
            updated_history = history + [(message, response)]
            current_history = updated_history.copy()
            return updated_history
        
        if message.lower() in ["reset time", "clear time", "/reset time", "/clear time"]:
            old_time = current_time_filter
            current_time_filter = None
            response = f"Time filter cleared. Previously filtering by: '{old_time}'" if old_time else "No time filter was set."
            updated_history = history + [(message, response)]
            current_history = updated_history.copy()
            return updated_history
            
        if message.lower() in ["reset all", "clear all", "/reset all", "/clear all", "reset filters", "clear filters"]:
            old_space = current_space
            old_time = current_time_filter
            current_space = None
            current_time_filter = None
            space_msg = f"Space selection '{old_space}' cleared." if old_space else "No space was selected."
            time_msg = f"Time filter '{old_time}' cleared." if old_time else "No time filter was set."
            response = f"All filters reset. {space_msg} {time_msg}"
            updated_history = history + [(message, response)]
            current_history = updated_history.copy()
            return updated_history
        
        # Add debug command for date ranges
        if message.lower() in ["debug time", "debug date", "/debug time", "/debug date"]:
            start_date, end_date = calculate_date_range(current_time_filter)
            today = datetime.now()
            response = f"""
<div class="debug-info">
<h3>Date Information</h3>
<p><strong>Current system time:</strong> {today.strftime('%Y-%m-%d %H:%M:%S')}</p>
<p><strong>Active time filter:</strong> {current_time_filter if current_time_filter else "None"}</p>
<p><strong>Calculated date range:</strong> {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}</p>
</div>
"""
            updated_history = history + [(message, response)]
            current_history = updated_history.copy()
            return updated_history
                
        # Add debug command to check data in Pinecone for date issues
        if message.lower() in ["debug data", "/debug data"]:
            try:
                from utils.pinecone_logic import init_pinecone, query_pinecone
                index = init_pinecone()
                
                # Prepare a generic query to retrieve recent documents
                test_query = "meeting notes recent updates"
                
                # Get results with the current space filter if set
                results = query_pinecone(
                    index, 
                    test_query, 
                    filter_by_space=current_space,
                    similarity_threshold=0.3, 
                    top_k=10
                )
                
                if not results:
                    response = """
<div class="debug-info">
<h3>Data Inspection</h3>
<p>No results found. This could indicate:
<ul>
<li>No data in the index</li>
<li>The space filter is too restrictive</li>
<li>The query isn't matching any content</li>
</ul>
</p>
<p>Try resetting the space filter with "reset space" and trying again.</p>
</div>
"""
                else:
                    # Extract and analyze date information
                    date_info = []
                    for i, result in enumerate(results, 1):
                        if len(result) >= 4:  # Has date information
                            source_url, text_content, score, doc_date = result
                            
                            # Try to parse the date
                            date_status = "Unknown format"
                            try:
                                from dateutil import parser
                                parsed_date = parser.parse(doc_date)
                                date_status = f"Valid ({parsed_date.strftime('%Y-%m-%d')})"
                                
                                # Check if date is in the future
                                if parsed_date.date() > datetime.now().date():
                                    date_status += " - WARNING: Future date!"
                            except Exception as e:
                                date_status = f"Invalid: {str(e)}"
                                
                            # Extract space info
                            space_info = "Unknown"
                            try:
                                import re
                                # Extract space key from Confluence URL pattern
                                space_match = re.search(r'/spaces/([^/]+)', source_url)
                                if space_match:
                                    space_info = space_match.group(1)
                                else:
                                    # Try alternative pattern with spaceKey parameter
                                    space_key_match = re.search(r'spaceKey=([^&]+)', source_url)
                                    if space_key_match:
                                        space_info = space_key_match.group(1)
                            except:
                                pass
                                
                            date_info.append({
                                "index": i,
                                "source": source_url,
                                "date": doc_date,
                                "status": date_status,
                                "space": space_info
                            })
                        else:
                            date_info.append({
                                "index": i,
                                "source": result[0] if len(result) > 0 else "Unknown",
                                "date": "No date information",
                                "status": "Missing",
                                "space": "Unknown"
                            })
                    
                    # Generate HTML table with date information
                    table_rows = ""
                    for item in date_info:
                        source_short = item["source"]
                        if len(source_short) > 60:
                            source_short = source_short[:57] + "..."
                            
                        table_rows += f"""
<tr>
  <td>{item["index"]}</td>
  <td>{source_short}</td>
  <td>{item["date"]}</td>
  <td>{item["status"]}</td>
  <td>{item["space"]}</td>
</tr>"""
                    
                    response = f"""
<div class="debug-info">
<h3>Data Inspection Results</h3>
<p>Found {len(results)} documents in the index.</p>
<table class="debug-table" style="width:100%; border-collapse: collapse;">
  <tr>
    <th style="border:1px solid #ddd; padding:8px;">#</th>
    <th style="border:1px solid #ddd; padding:8px;">Source</th>
    <th style="border:1px solid #ddd; padding:8px;">Date</th>
    <th style="border:1px solid #ddd; padding:8px;">Status</th>
    <th style="border:1px solid #ddd; padding:8px;">Space</th>
  </tr>
  {table_rows}
</table>
</div>
"""
                
                updated_history = history + [(message, response)]
                current_history = updated_history.copy()
                return updated_history
                
            except Exception as e:
                error_message = f"Error analyzing data: {str(e)}"
                updated_history = history + [(message, error_message)]
                current_history = updated_history.copy()
                return updated_history
                
        # Process space-related queries
        enhanced_message = preprocess_space_query(message)
        
        # Extract filter by space if specified in the message
        filter_by_space = current_space  # Default to the current space if set
        
        # Check for space filter in format "search in space: SPACE_KEY query"
        import re
        space_match = re.search(r'search in (space|namespace|collection): (\w+)(.*)', enhanced_message, re.IGNORECASE)
        
        if space_match:
            filter_by_space = space_match.group(2).strip()
            enhanced_message = space_match.group(3).strip()
            # Remember this space selection for future queries
            current_space = filter_by_space
            print(f"Detected and saved space filter: {filter_by_space}, modified query: {enhanced_message}")
        
        # Alternative format: "in SPACE_KEY: query"
        space_match2 = re.search(r'in (\w+):(.*)', enhanced_message, re.IGNORECASE)
        if space_match2:
            filter_by_space = space_match2.group(1).strip()
            enhanced_message = space_match2.group(2).strip()
            # Remember this space selection for future queries
            current_space = filter_by_space
            print(f"Detected and saved alternative space filter: {filter_by_space}, modified query: {enhanced_message}")
        
        # Check for time-based filters
        time_patterns = [
            (r'this week', 'this week'),
            (r'last week', 'last week'),
            (r'this month', 'this month'),
            (r'last month', 'last month'),
            (r'today', 'today'),
            (r'yesterday', 'yesterday'),
            (r'recent(ly)?', 'recent'),
            (r'past (\d+) days?', lambda m: f'past {m.group(1)} days'),
            (r'since (\w+ \d+)', lambda m: f'since {m.group(1)}')
        ]
        
        # Look for time filters in the query
        time_filter = None
        for pattern, filter_value in time_patterns:
            match = re.search(pattern, enhanced_message.lower())
            if match:
                if callable(filter_value):
                    time_filter = filter_value(match)
                else:
                    time_filter = filter_value
                print(f"Detected time filter: {time_filter}")
                # Save the time filter for future reference
                current_time_filter = time_filter
        
        # If no explicit time filter in this query but we have one saved, use it for time-sensitive follow-ups
        if not time_filter and current_time_filter:
            # Check if this is a follow-up that would benefit from the time filter
            follow_up_patterns = [
                r'(what|which) (changed|updated|modified)',
                r'(tell|show) me (the|those|these) (changes|updates)',
                r'summarize (the|those|these) (changes|updates)',
                r'(what|list|show) are the (changes|updates|modifications)',
                r'what happened',
                r'what\'s new',
                r'any (changes|updates|modifications)',
                r'what is different'
            ]
            
            is_time_followup = False
            for pattern in follow_up_patterns:
                if re.search(pattern, enhanced_message.lower()):
                    is_time_followup = True
                    break
                    
            if is_time_followup:
                print(f"Applying saved time filter: {current_time_filter} to follow-up query")
                # Add time filter to the query - only log this in system but don't actually modify user query
                # This will be passed as part of the context to the LLM
                time_filter = current_time_filter
        
        # Initialize Pinecone
        print(f"Processing message: {enhanced_message}")
        from utils.pinecone_logic import init_pinecone, query_pinecone
        index = init_pinecone()
        
        namespaces = []
        # Get namespaces for filtering suggestions
        try:
            stats = index.describe_index_stats()
            if hasattr(stats, 'namespaces'):
                namespaces = list(stats.namespaces.keys())
            elif 'namespaces' in stats:
                namespaces = list(stats.get('namespaces', {}).keys())
        except:
            pass
            
        # Query Pinecone for relevant information
        query_results = query_pinecone(index, enhanced_message, filter_by_space=filter_by_space, similarity_threshold=0.3, top_k=8)
        
        # Log the current interaction
        log_conversation(enhanced_message, "Processing...", session_id, history, query_results)
        
        if not query_results:
            # If filtering by space caused no results, give specific feedback
            if filter_by_space:
                spaces_info = f"Available spaces: {', '.join(namespaces)}" if namespaces else ""
                space_note = f"Currently searching in space: '{filter_by_space}'. Type 'reset space' to search across all spaces."
                error_message = f"""No results found when filtering by space key '{filter_by_space}'. {spaces_info}
                    <br><br>{space_note}
                    <br><br>Try:
                    <ul>
                        <li>Rephrasing your query</li>
                        <li>Using more general keywords</li>
                        <li>Resetting the space filter with 'reset space'</li>
                    </ul>"""
                updated_history = history + [(original_message, error_message)]
                current_history = updated_history.copy()
                return updated_history
            
            # Generic no results message
            error_message = """I couldn't find any relevant information for your query. 
                <br><br>Try:
                <ul>
                    <li>Rephrasing your question</li>
                    <li>Using more specific keywords</li>
                    <li>Check if the information exists in your database</li>
                </ul>"""
            updated_history = history + [(original_message, error_message)]
            current_history = updated_history.copy()
            return updated_history
        
        # Generate response with OpenAI, passing both space and time filters for context
        response = generate_response(enhanced_message, query_results, space_filter=current_space, time_filter=current_time_filter)
        
        if not response:
            return "Failed to generate a response. Please try again."
        
        # Format sources for display
        formatted_sources = format_sources(query_results)
        
        # Add tips about space filtering if multiple namespaces exist
        space_filtering_tip = ""
        if namespaces and len(namespaces) > 1:
            current_space_msg = f"<p><strong>Currently searching in space:</strong> {current_space}</p>" if current_space else ""
            reset_tip = "<p>Type 'reset space' to search across all spaces.</p>" if current_space else ""
            
            spaces_list = ", ".join(namespaces)
            space_filtering_tip = f"""<div class="tip-section">
<h3>Search Tips</h3>
{current_space_msg}
<p><strong>Filter by space:</strong> You can filter results by space key using: "search in space: [space_key] [your query]"</p>
<p><strong>Available spaces:</strong> {spaces_list}</p>
{reset_tip}
</div>"""

        # Add time filter info if applicable
        time_filter_tip = ""
        if current_time_filter:
            time_filter_tip = f"""<div class="tip-section time-active">
<h3>Active Time Filter</h3>
<p><strong>Currently filtering by time:</strong> {current_time_filter}</p>
<p>Type 'reset time' to clear this filter.</p>
</div>"""

        # Add time-based query tip if date information is available
        time_query_tip = ""
        has_date_info = any(len(result) >= 4 for result in query_results)
        if has_date_info and not current_time_filter:
            time_query_tip = """<div class="tip-section time-tip">
<h3>Time-Based Search</h3>
<p>You can search for recently updated content by using phrases like:</p>
<ul>
<li>"show content created this week"</li>
<li>"what pages were updated recently"</li>
<li>"list documents modified this month"</li>
</ul>
</div>"""

        # Final response with sources and optional tips
        final_response = f"""<div class="chat-response">
<div class="response-content">{response}</div>
            {space_filtering_tip}
{time_filter_tip}
{time_query_tip}
{formatted_sources}
</div>"""
        
        # Update history with the new message pair and return
        updated_history = history + [(original_message, final_response)]
        current_history = updated_history.copy()
        return updated_history

    except Exception as e:
        error_message = f"Error processing your request: {str(e)}"
        print(error_message)
        import traceback
        traceback.print_exc()
        
        # Return error message in the correct format for the chatbot
        updated_history = history + [(original_message, f"I encountered an error while processing your request. Please try again or contact support if the issue persists. Error details: {str(e)}")]
        current_history = updated_history.copy()
        return updated_history

# Legacy main function for backward compatibility
def main(query):
    # Create mock history for single query usage
    history = []
    response = chat_function(query)
    return response

def display_content(content):
    # Debugging output to check the content before any modifications
    print(f"Original content before processing: {content}")
    
    if content.startswith("none#"):
        content = content[5:]  # Remove the "none#" prefix
        print(f"Content after removing 'none#': {content}")  # Debugging output
    
    # Continue with the rest of the display logic
    return content

# Function to debug Pinecone connection and data
def debug_pinecone():
    """
    Debug utility to verify Pinecone connection and data.
    """
    print("=== Starting Pinecone Debug ===")
    
    # Check environment variables
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_environment = os.getenv('PINECONE_ENVIRONMENT') 
    pinecone_index_name = os.getenv('PINECONE_INDEX_NAME', 'default-index')
    
    print(f"Environment Variables:")
    print(f"- PINECONE_API_KEY: {'Set' if pinecone_api_key else 'Missing'}")
    print(f"- PINECONE_ENVIRONMENT: {'Set: ' + pinecone_environment if pinecone_environment else 'Missing'}")
    print(f"- PINECONE_INDEX_NAME: {pinecone_index_name}")
    
    if not pinecone_api_key or not pinecone_environment:
        print("ERROR: Missing required environment variables. Please check your .env file.")
        return False
    
    # Initialize Pinecone and get index
    try:
        from utils.pinecone_logic import get_pinecone_index, verify_pinecone_upsert
        index, index_created = get_pinecone_index(pinecone_index_name)
        
        if index is None:
            print("ERROR: Failed to initialize Pinecone index.")
            return False
            
        print(f"Successfully connected to Pinecone index '{pinecone_index_name}'")
        print(f"Index was {'created during this session' if index_created else 'already existing'}")
        
        # Check index stats
        try:
            stats = index.describe_index_stats()
            if not stats:
                print("ERROR: Could not retrieve index stats.")
                return False
                
            print("\nIndex Stats:")
            print(json.dumps(stats, indent=2))
            
            if 'total_vector_count' in stats:
                vector_count = stats['total_vector_count']
                if vector_count <= 0:
                    print("\nWARNING: Index exists but contains no vectors.")
                    print("You need to populate your database by running the update_database script.")
                    return False
                else:
                    print(f"\nIndex contains {vector_count} vectors.")
            else:
                print("\nWARNING: Index stats does not contain vector count information.")
        except Exception as e:
            print(f"ERROR checking index stats: {str(e)}")
            return False
            
        # Test query
        print("\nTesting a sample query...")
        try:
            # Generate embedding for a simple test query
            from utils.openai_logic import get_embeddings
            
            test_query = "test query"
            print(f"Using test query: '{test_query}'")
            
            embedding_response = get_embeddings(test_query, "text-embedding-ada-002")
            query_embedding = embedding_response.data[0].embedding
            
            # Query Pinecone
            results = index.query(
                vector=query_embedding,
                top_k=1,
                include_metadata=True
            )
            
            print("Query results:")
            print(json.dumps(results, indent=2))
            
            if 'matches' in results and len(results['matches']) > 0:
                print(f"Query successful: {len(results['matches'])} matches found.")
                return True
            else:
                print("WARNING: Query returned no matches. This could be expected if your database doesn't contain relevant data.")
                return True  # Still return true as the connection works
        except Exception as e:
            print(f"ERROR during test query: {str(e)}")
            return False
            
    except Exception as e:
        print(f"ERROR initializing Pinecone: {str(e)}")
        return False
        
    print("=== Pinecone Debug Complete ===")
    return True

# Creating the RAG application
def create_rag_app():
    """Create and configure the RAG application."""
    # Initialize Pinecone
    init_pinecone()
    
    # Create the application interface
    with gr.Blocks(title="RAG Demo with Pinecone and OpenAI") as app:
        gr.Markdown("# RAG Demo with Pinecone and OpenAI")
        gr.Markdown("Ask questions about the documents stored in your Pinecone index.")
        
        with gr.Row():
            with gr.Column(scale=4):
                query_input = gr.Textbox(
                    label="Enter your question",
                    placeholder="What would you like to know?",
                    lines=1
                )
                submit_btn = gr.Button("Submit", variant="primary")
                
        with gr.Row():
            with gr.Column(scale=3):
                answer_output = gr.Markdown(label="Answer", value="")
            with gr.Column(scale=2):
                sources_output = gr.JSON(label="Retrieved Documents", value={})
                
        def process_query(query):
            """Process a user query and return an answer with sources."""
            try:
                # Log the query
                logging.info(f"Processing query: {query}")
                
                # Query Pinecone for relevant documents
                query_results = query_pinecone(query, top_k=5)
                
                # Format sources for display
                sources_display = []
                for i, (source, content, score) in enumerate(query_results, 1):
                    sources_display.append({
                        "rank": i,
                        "source": source,
                        "score": float(score),
                        "snippet": content[:150] + "..." if len(content) > 150 else content
                    })
                
                # Generate response
                response = generate_response(query, query_results)
                
                return response, sources_display
            except Exception as e:
                logging.error(f"Error processing query: {str(e)}")
                return f"Error: {str(e)}", []
        
        # Set up the event handler for query submission
        submit_btn.click(
            fn=process_query,
            inputs=[query_input],
            outputs=[answer_output, sources_output]
        )
        
        query_input.submit(
            fn=process_query,
            inputs=[query_input],
            outputs=[answer_output, sources_output]
        )
        
    return app

# Format sources section with double quotes instead of single quotes
def format_sources(query_results):
    if not query_results:
        return ""
    
    sources_html = '<div class="sources-section">\n<p class="sources-header">Sources:</p>\n<ul class="sources-list">'
    
    # Get namespaces information for display
    for i, result in enumerate(query_results):
        # Extract date information if available
        if len(result) >= 4:
            source, _, score, date_info = result
            has_date = True
        else:
            source, _, score = result
            has_date = False
            date_info = None
            
        # Extract namespace/space key information from the source URL
        namespace = "default"
        
        # Try to extract space key from URL
        try:
            import re
            # Extract space key from Confluence URL pattern
            space_match = re.search(r'/spaces/([^/]+)', source)
            if space_match:
                namespace = space_match.group(1)
            else:
                # Try alternative pattern with spaceKey parameter
                space_key_match = re.search(r'spaceKey=([^&]+)', source)
                if space_key_match:
                    namespace = space_key_match.group(1)
        except:
            pass
        
        # Add source entry with similarity score, namespace, and date if available
        similarity_percentage = int(score * 100)
        date_html = f', Last Modified: {date_info}' if has_date else ''
        sources_html += f'\n<li class="source-item">Source {i+1} [Space Key: <em>{namespace}</em>] - <a href="{source}" class="source-link" target="_blank">{source}</a> <span class="similarity-score">(Similarity: {similarity_percentage}%{date_html})</span></li>'
    
    sources_html += "\n</ul>\n</div>"
    return sources_html

def create_prompt(query, context, space_filter=None, time_filter=None):
    """Create a prompt for the OpenAI API based on the query and context."""
    import re
    
    # Check if we have an empty context
    if not context or context.strip() == "":
        # Custom handler for time-based queries with no results
        if time_filter:
            return f"""The user asked: "{query}"
            
You were not able to find any documents that exactly match the time frame "{time_filter}". 
However, you should still respond with the most relevant information available from recent documents.
If you have documents from a slightly different time period, you should still use that information 
to provide the best possible answer, making clear the timeframe the information comes from.

If you truly have no relevant information, explain that you don't have information for that specific timeframe
but offer to help with related information from other time periods.
"""
        else:
            return f"""The user asked: "{query}"
            
You don't have access to specific Confluence pages or their content directly. Please explain that you would need more specific details from the user to provide a helpful response. Suggest the user provide some context from the relevant pages, and you can help analyze and summarize that information.

Remember to be polite and helpful in your response, suggesting ways the user could refine their query to get better results.
"""
    
    # For space-filtered queries, add context about the space
    space_context = ""
    if space_filter:
        space_context = f"The user is specifically asking about content in the {space_filter} space. "
        
    # For time-filtered queries, add context about the time period    
    time_context = ""
    if time_filter:
        # Check if we have results from outside the requested time period
        dates_in_context = re.findall(r'Last Modified: (\d{4}-\d{2}-\d{2})', context)
        all_within_period = True
        
        if dates_in_context:
            # Check if the time filter is "this week" but we're returning older data
            if "this week" in time_filter.lower():
                today = datetime.now()
                start_of_week = today - timedelta(days=today.weekday())
                
                from dateutil import parser
                for date_str in dates_in_context:
                    try:
                        date = parser.parse(date_str).date()
                        if date < start_of_week.date():
                            all_within_period = False
                            break
                    except:
                        continue
            
            if not all_within_period:
                time_context = f"The user requested information from {time_filter}, but you're providing the most recent available information which may be from before this time period. Make sure to acknowledge this in your response. "
        
    # Create the prompt with our additional context
    prompt = f"""{space_context}{time_context}The user asked: "{query}"
    
Based on the following information from Confluence:

{context}

Provide a helpful, accurate response that directly addresses the user's query. Use the content from the pages to provide specific details, examples, and context.

If the information provided doesn't completely answer the query, clearly state what aspects you can address and what remains uncertain.

Do not mention "Confluence" directly in your response. Present the information as if you have direct knowledge of the content.
"""
    return prompt

if __name__ == "__main__":
    # Add debug option via command line
    if len(sys.argv) > 1 and sys.argv[1] == '--debug-pinecone':
        debug_pinecone()
        sys.exit(0)
        
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
    ) as demo:
        # Create a header with JHU branding
        with gr.Row(elem_id="header"):
            with gr.Column(scale=1):
                gr.HTML("""
                    <div class="header-container">
                        <img src="/file=web_resources/university.shield.rgb.blue.svg" alt="JHU Logo" class="jhu-logo" style="max-width: 100px;">

                        <div class="title-container">
                            <h1 class="jhu-title">JOHNS HOPKINS UNIVERSITY</h1>
                            <h2 class="assistant-title">Confluence Knowledge Assistant</h2>
                        </div>
                    </div>
                """)
        
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
        # Always clear the input field regardless of whether the function succeeds or fails
        enter_event.then(lambda: "", None, msg, show_progress=False)
        submit_click_event.then(lambda: "", None, msg, show_progress=False)
    
    # Launch the interface
    demo.launch(server_name="localhost", server_port=8888, share=False)