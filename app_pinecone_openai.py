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

# load environment variables
load_dotenv(find_dotenv())

# Set up logging directory
LOGS_DIR = os.path.join(os.getcwd(), "conversation_logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Global variable to store the current conversation history
current_history = []

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

# Function to initialize Pinecone
def init_pinecone():
    print("Start: Initializing Pinecone")
    try:
        # Get Pinecone API key from environment variables
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        if not pinecone_api_key:
            print("Error: PINECONE_API_KEY not found in environment variables")
            return None
            
        # Get Pinecone environment from environment variables
        pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')
        if not pinecone_environment:
            print("Error: PINECONE_ENVIRONMENT not found in environment variables")
            return None
            
        # Initialize Pinecone
        index_name = os.getenv('PINECONE_INDEX_NAME', 'default-index')
        index, _ = get_pinecone_index(index_name)
        
        # Verify the index has data
        try:
            stats = index.describe_index_stats()
            if not stats or 'total_vector_count' not in stats:
                print("Warning: Could not retrieve index stats")
            else:
                vector_count = stats['total_vector_count']
                if vector_count <= 0:
                    print(f"Warning: Index contains no vectors. Database is empty and needs to be populated.")
                else:
                    print(f"Index contains {vector_count} vectors. Database is ready for queries.")
        except Exception as e:
            print(f"Warning: Could not verify index data: {e}")
        
        print("Done: Pinecone initialized successfully")
        return index
    except Exception as e:
        print(f"Error initializing Pinecone: {str(e)}")
        return None

# Function to query Pinecone
def query_pinecone(index, query):
    print(f"Start: Querying Pinecone with: {query}")
    try:
        # Generate embedding for the query
        embedding_response = get_embeddings(query, "text-embedding-ada-002")
        query_embedding = embedding_response.data[0].embedding
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        
        # Extract information from results
        extracted_info = extract_info(results)
        
        print("Done: Pinecone query completed")
        return extracted_info
    except Exception as e:
        print(f"Error querying Pinecone: {str(e)}")
        return []

# Function to generate response with OpenAI
def generate_answer(query, context_data, chat_history=[]):
    try:
        # Create system prompt
        system_prompt = create_system_prompt()
        
        # Create chat messages
        messages = []
        messages = add_prompt_messages("system", system_prompt, messages)
        
        # Add chat history for context (up to 5 past exchanges)
        history_context = ""
        recent_history = chat_history[-5:] if len(chat_history) > 5 else chat_history
        if recent_history:
            history_context = "Previous conversation:\n"
            for h_user, h_bot in recent_history:
                history_context += f"User: {h_user}\nAssistant: {h_bot}\n\n"
        
        # Create context string from retrieved data
        search_context = ""
        for i, (source, text, score) in enumerate(context_data, 1):
            # Add some of the text as context (trim if too long)
            content_preview = text[:1500] + "..." if len(text) > 1500 else text
            search_context += f"Source {i} ({source}):\n{content_preview}\n\n"
        
        # Create the prompt with context
        prompt = f"Based on the following information, please answer the question. If referencing specific sources, mention them in your answer.\n\n"
        
        if history_context:
            prompt += f"{history_context}\n"
            
        prompt += f"Search results:\n{search_context}\n\nCurrent question: {query}\n\nAnswer:"
        messages = add_prompt_messages("user", prompt, messages)
        
        # Get response from OpenAI
        response = get_chat_completion_messages(messages, "gpt-3.5-turbo", temperature=0.3)
        
        return response
    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        return f"Error generating answer: {str(e)}"

# Function to log conversation
def log_conversation(user_input, assistant_response, session_id, results=None):
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
    try:
        # Process the message and get response
        special_response = process_message(message, history)
        
        if special_response:
            # Return the message and response in the format Gradio expects
            return history + [(message, special_response)]
            
        # Initialize Pinecone
        index = init_pinecone()
        if not index:
            return history + [(message, "Sorry, I couldn't connect to the knowledge base. Please check that your Pinecone API key and environment variables are set correctly.")]
            
        # Verify the index has data
        try:
            stats = index.describe_index_stats()
            if stats and 'total_vector_count' in stats and stats['total_vector_count'] <= 0:
                error_message = (
                    "Your Pinecone database is empty. No vectors have been added yet. "
                    "Please run the update_database.py script to populate your database with data. "
                    "You can do this by running: python update_database.py --csv-file=your_data.csv"
                )
                return history + [(message, error_message)]
        except Exception as e:
            print(f"Error checking index stats: {str(e)}")
            # Continue execution since this is just a warning
            
        # Query Pinecone
        results = query_pinecone(index, message)
        if not results:
            return history + [(message, "I couldn't find any relevant information. This could mean your question is outside the scope of the knowledge base, or the knowledge base doesn't contain the necessary data. Please try rephrasing your question or updating the database with more relevant content.")]
            
        # Generate answer with OpenAI
        answer = generate_answer(message, results, history)
        if not answer:
            return history + [(message, "I couldn't generate an answer. Please try again or check the OpenAI API key.")]
        
        # --- ORIGINAL CODE (Restored) ---
        # Create session ID for this specific interaction
        session_id_str = datetime.now().strftime("%Y%m%d%H%M%S")

        # Format the response with proper HTML structure
        formatted_response = f"""
        <div class="chat-response">
            <div class="question-banner">
                {message}
            </div>
            <div class="answer-content">
                {html.escape(answer)}
            </div>
            
            <div class="sources-section">
                <p class="sources-header">Sources:</p>
                <ul class="sources-list">
        """

        # Add sources with similarity scores
        # Handle case where results might be empty or not contain expected data
        if results:
            for idx, (source, _, score) in enumerate(results, 1):
                if source and score is not None:
                    similarity_percentage = score * 100
                    # Ensure closing </li> tag for valid HTML
                    formatted_response += f"""
                            <li>Source {idx}: <a href="{source}" class="source-link" target="_blank">{source}</a>
                            <span class="similarity-score">[Similarity: {similarity_percentage:.2f}%]</span></li> 
                    """
                else:
                    formatted_response += "<li>Invalid source data</li>" # Removed invalid {idx}
        else:
            formatted_response += "<li>No sources found.</li>"

        formatted_response += f"""
                </ul>
            </div>
            
            <div class="session-info">
                Session ID: {session_id_str}
            </div>
            
            <div class="instruction-text">
                To view the full content of any page, type "show content" followed by the reference number.
            </div>
        </div>
        """
        # Validate the HTML response

        try:
            soup = BeautifulSoup(formatted_response, 'html.parser')
            formatted_response = str(soup)
        except Exception as e:
            logging.error(f"Error validating HTML response: {str(e)}")
        # Log the conversation
        log_conversation(message, formatted_response, session_id_str, results)

        # Return None for the user message part to avoid duplication
        return history + [(None, formatted_response)]

    except Exception as e:
        logging.error(f"Error in chat function: {str(e)}")
        # Return a user-friendly error message in the chat history format
        error_message = f'<div style="color: red; padding: 10px;">An error occurred: {str(e)}</div>'
        
        # Provide more helpful message for specific error types
        if "pinecone" in str(e).lower():
            error_message = f'<div style="color: red; padding: 10px;">Pinecone error: {str(e)}<br><br>To troubleshoot, run: python app_pinecone_openai.py --debug-pinecone</div>'
        elif "openai" in str(e).lower():
            error_message = f'<div style="color: red; padding: 10px;">OpenAI API error: {str(e)}<br><br>Please check your OpenAI API key and account status.</div>'
        
        return history + [(message, error_message)]

# Legacy main function for backward compatibility
def main(query):
    # Create mock history for single query usage
    history = []
    response = chat_function(query, history)
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
        enter_event.then(lambda: "", None, msg)
        submit_click_event.then(lambda: "", None, msg)
    
    # Launch the interface
    demo.launch(server_name="localhost", server_port=8888, share=False)