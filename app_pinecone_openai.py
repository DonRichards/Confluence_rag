from bs4 import BeautifulSoup
import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import gradio as gr
import html
from utils.pinecone_logic import delete_pinecone_index, get_pinecone_index, upsert_data, init_pinecone, query_pinecone
from utils.data_prep import import_csv, clean_data_pinecone_schema, generate_embeddings_and_add_to_df
from utils.openai_logic import get_embeddings, create_prompt, add_prompt_messages, get_chat_completion_messages, create_system_prompt
from utils.auth import get_confluence_client
import sys
import time
import json
from datetime import datetime
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

# Function to generate a response using OpenAI
def generate_response(query, query_results):
    """Generate a response based on a query and the retrieved documents."""
    try:
        # Get OpenAI API key
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            logging.error("OpenAI API key not found")
            return "Error: OpenAI API key not found"
            
        client = OpenAI(api_key=openai_api_key)
        
        # Prepare context from documents
        context = ""
        if query_results:
            for i, (source_url, text_content, score) in enumerate(query_results, 1):
                if text_content:
                    context += f"Document {i} (Score: {score:.2f}):\n"
                    context += f"Source: {source_url}\n"
                    context += f"Content: {text_content}\n\n"
        
        # Create system message
        system_message = """You are a knowledgeable AI assistant that provides helpful and accurate information based on the available context.
If the information isn't in the provided context, admit that you don't know rather than making up an answer.
When answering, cite the source URLs from the context when appropriate.
Be concise and focus on directly answering the user's query.

When the user asks about a specific space or collection:
1. Look for information in the context that mentions that space by name, key, or ID
2. Pay attention to metadata like URLs containing space information (e.g., spaceKey=SPACENAME)
3. Analyze the content from documents in that space to infer the purpose of the space
4. If you find multiple documents from the same space, this may indicate the space's focus
5. If you can make reasonable inferences about a space based on document titles or content, do so while noting it's an inference."""
        
        # Format messages for OpenAI API
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Given the following context:\n\n{context}\n\nQuestion: {query}"}
        ]
        
        # Call OpenAI API to generate response
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            temperature=0.3,
            max_tokens=1000
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
    
    is_purpose_query = False
    import re
    for pattern in purpose_patterns:
        if re.search(pattern, message.lower()):
            is_purpose_query = True
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
                else:
                    error_message = "Please use the format: /fork [message_number] [new_message]"
                    updated_history = history + [(message, error_message)]
                    return updated_history
            except Exception as e:
                error_message = f"Error forking conversation: {str(e)}"
                updated_history = history + [(message, error_message)]
                return updated_history
        
        # Process space-related queries
        enhanced_message = preprocess_space_query(message)
        
        # Extract filter by space if specified in the message
        filter_by_space = None
        
        # Check for space filter in format "search in space: SPACE_KEY query"
        import re
        space_match = re.search(r'search in (space|namespace|collection): (\w+)(.*)', enhanced_message, re.IGNORECASE)
        
        if space_match:
            filter_by_space = space_match.group(2).strip()
            enhanced_message = space_match.group(3).strip()
            print(f"Detected space filter: {filter_by_space}, modified query: {enhanced_message}")
        
        # Alternative format: "in SPACE_KEY: query"
        space_match2 = re.search(r'in (\w+):(.*)', enhanced_message, re.IGNORECASE)
        if space_match2:
            filter_by_space = space_match2.group(1).strip()
            enhanced_message = space_match2.group(2).strip()
            print(f"Detected alternative space filter: {filter_by_space}, modified query: {enhanced_message}")
        
        # Initialize Pinecone
        print(f"Processing message: {enhanced_message}")
        from utils.pinecone_logic import init_pinecone, query_pinecone
        index = init_pinecone()
        
        # Get namespaces for filtering suggestions
        namespaces = []
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
                error_message = f"""No results found when filtering by space key '{filter_by_space}'. {spaces_info}
                    <br><br>Try:
                    <ul>
                        <li>Searching without a space filter</li>
                        <li>Using a different space key</li>
                        <li>Rephrasing your query</li>
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
        
        # Generate response with OpenAI
        response = generate_response(enhanced_message, query_results)
        
        if not response:
            return "Failed to generate a response. Please try again."
        
        # Format sources for display
        formatted_sources = format_sources(query_results)
        
        # Add tips about space filtering if multiple namespaces exist
        space_filtering_tip = ""
        if namespaces and len(namespaces) > 1:
            spaces_list = ", ".join(namespaces)
            space_filtering_tip = f"""<div class="tip-section">
<p><strong>Tip:</strong> You can filter results by space key using: "search in space: [space_key] [your query]"</p>
<p>Available space keys: {spaces_list}</p>
</div>"""
        
        # Final response with sources and optional tip
        final_response = f"""<div class="chat-response">
<div class="response-content">{response}</div>
{formatted_sources}
{space_filtering_tip}
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
    for i, (source, _, score) in enumerate(query_results):
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
        
        # Add source entry with similarity score and namespace
        similarity_percentage = int(score * 100)
        sources_html += f'\n<li class="source-item">Source {i+1} [Space Key: <em>{namespace}</em>] - <a href="{source}" class="source-link" target="_blank">{source}</a> <span class="similarity-score">(Similarity: {similarity_percentage}%)</span></li>'
    
    sources_html += "\n</ul>\n</div>"
    return sources_html

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
    demo.launch(server_name="localhost", server_port=8889, share=False)