import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import gradio as gr
from utils.pinecone_logic import delete_pinecone_index, get_pinecone_index, upsert_data
from utils.data_prep import import_csv, clean_data_pinecone_schema, generate_embeddings_and_add_to_df
from utils.openai_logic import get_embeddings, create_prompt, add_prompt_messages, get_chat_completion_messages, create_system_prompt
from utils.auth import get_confluence_client
import sys
import time
import json
from datetime import datetime

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
                    return formatted_content
            
            # If no URL or URL fetch failed, try to find the page in recent history
            if history and len(history) > 0:
                last_response = history[-1][1]
                if "References:" in last_response:
                    refs = last_response.split("\n\nReferences:\n")[-1].split("\n")
                    # Try to find a matching reference
                    page_title = message.split("of")[-1].strip().strip('"').strip("'").strip("?")
                    print(f"Looking for page with title similar to: '{page_title}' in references")
                    
                    # Try each reference
                    for ref in refs:
                        if page_title.lower() in ref.lower():
                            source_url = ref.split("] ")[-1].strip()
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
                                return formatted_content
            
            return (
                "I couldn't find the content to rearrange. Please try:\n"
                "1. First ask a question about the topic to get reference numbers\n"
                "2. Then use 'rearrange content <number>' with the reference number\n"
                "For example: 'rearrange content 1'"
            )
                
        except Exception as e:
            print(f"Error rearranging content: {str(e)}")
            return (
                "Sorry, I encountered an error while trying to rearrange the content. Please try:\n"
                "1. First ask a question about the topic to get reference numbers\n"
                "2. Then use 'rearrange content <number>' with the reference number\n"
                "For example: 'rearrange content 1'"
            )

    # Check for "show content" command
    if message.lower().startswith("show content"):
        try:
            # Extract reference number
            ref_num = int(message.split()[-1])
            if ref_num > 0 and ref_num <= len(history[-1][1].split("\n\nReferences:\n")[-1].split("\n")):
                # Get the source URL from the last response
                refs = history[-1][1].split("\n\nReferences:\n")[-1].split("\n")
                source_url = refs[ref_num-1].split("] ")[-1].strip()
                
                print(f"Attempting to fetch content from URL: {source_url}")
                
                # Get full content
                content = get_full_content(source_url)
                if content:
                    return content
                else:
                    return "Sorry, I couldn't retrieve the full content of the page. Please try accessing the URL directly."
            else:
                return "Please provide a valid reference number from the last response."
        except (ValueError, IndexError) as e:
            print(f"Error processing show content command: {str(e)}")
            return "Please use the format 'show content <number>' where number is the reference number from the last response."
    
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
                    return content
            
            # If no URL or URL fetch failed, try to find the page in recent history
            if history and len(history) > 0:
                last_response = history[-1][1]
                if "References:" in last_response:
                    refs = last_response.split("\n\nReferences:\n")[-1].split("\n")
                    # Try to find a matching reference
                    page_title = message.split("of")[-1].strip().strip('"').strip("'").strip("?")
                    print(f"Looking for page with title similar to: '{page_title}' in references")
                    
                    # Try each reference
                    for ref in refs:
                        if page_title.lower() in ref.lower():
                            source_url = ref.split("] ")[-1].strip()
                            print(f"Found matching reference, attempting to fetch content: {source_url}")
                            content = get_full_content(source_url)
                            if content:
                                return content
            
            return (
                "I couldn't find the page you're looking for. Please try:\n"
                "1. First ask a question about the topic to get reference numbers\n"
                "2. Then use 'show content <number>' with the reference number\n"
                "For example: 'show content 1'"
            )
                
        except Exception as e:
            print(f"Error fetching page content: {str(e)}")
            return (
                "Sorry, I encountered an error. Please try:\n"
                "1. First ask a question about the topic to get reference numbers\n"
                "2. Then use 'show content <number>' with the reference number\n"
                "For example: 'show content 1'"
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
    global current_history
    
    # Generate a session ID only if this is the first message
    if not history:
        # DEBUG is False then use the current time else use a fixed session ID
        if os.getenv('DEBUG') != 'True':
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            session_id = "31415926_53589"
            
        # Initialize current_history for a new conversation
        current_history = []
    else:
        # Extract session ID from the first message's response
        session_id = history[0][1].split("\n\nReferences:\n")[0].split("Session ID: ")[-1].strip()
    
    # Initialize Pinecone if not already initialized
    index = init_pinecone()
    if index is None:
        response = "Error: Failed to initialize Pinecone. Please check your API keys and environment settings."
        # Update current_history before logging
        current_history.append((message, response))
        # Log the conversation
        log_conversation(message, response, session_id)
        # Append to history
        history.append((message, response))
        return history
    
    if not message or message.strip() == "":
        response = "Please enter a question to search the knowledge base."
        # Update current_history before logging
        current_history.append((message, response))
        # Log the conversation
        log_conversation(message, response, session_id)
        # Append to history
        history.append((message, response))
        return history
    
    # Check for special commands
    special_response = process_message(message, history)
    if special_response:
        # Update current_history before logging
        current_history.append((message, special_response))
        # Log the conversation
        log_conversation(message, special_response, session_id)
        # Append to history
        history.append((message, special_response))
        return history
        
    # Query Pinecone
    results = query_pinecone(index, message)
    
    if not results:
        response = "Sorry, I couldn't find any relevant information to answer your question."
    else:
        # Generate answer based on retrieved content and history
        answer = generate_answer(message, results, history)
        
        # Format links as references
        references = "\n\nReferences:\n"
        for i, (source, _, score) in enumerate(results, 1):
            # Format score as percentage
            similarity = f"{score * 100:.1f}%"
            references += f"{i}. [Similarity: {similarity}] {source}\n"
        
        # Add instruction for viewing full content
        references += "\nTo view the full content of any page, type 'show content <number>' where <number> is the reference number."
        
        # Add session ID to the response
        response = f"{answer}\n\nSession ID: {session_id}\n{references}"
    
    # Update current_history before logging
    current_history.append((message, response))
    
    # Log the conversation
    log_conversation(message, response, session_id, results)
    
    # Print results to console
    print(f"Query processed: '{message}'. Response generated.")
    print(f"Current history size: {len(current_history)}")
    
    # Append the new message and response to the history
    history.append((message, response))
    return history

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

if __name__ == "__main__":
    # Create Gradio interface with chat components
    gr.close_all()
    
    with gr.Blocks(title="Confluence Knowledge Assistant", theme=gr.themes.Base()) as demo:
        # Add JHU logo and header section using the local logo file
        gr.HTML("""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <svg style="max-width: 260.704px; padding-right: 20px;" role="img" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 260.704 49.406">
                <path d="M91.861 39.828a10.32 10.32 0 0 0-.269-2.945.88.88 0 0 0-.713-.3l-.355-.028a.259.259 0 0 1 .026-.342c.543.029 1.068.042 1.624.042.6 0 .981-.014 1.494-.042a.248.248 0 0 1 .03.342l-.343.028a.831.831 0 0 0-.711.343 12.81 12.81 0 0 0-.2 2.9v1.61a4.943 4.943 0 0 1-1 3.4 3.75 3.75 0 0 1-2.72 1.042 4.049 4.049 0 0 1-2.66-.783c-.753-.626-1.111-1.664-1.111-3.358v-3.36c0-1.48-.029-1.723-.854-1.794l-.354-.028c-.086-.057-.059-.3.026-.342.711.029 1.2.042 1.765.042s1.054-.014 1.749-.042c.086.042.115.285.03.342l-.341.028c-.826.071-.854.314-.854 1.794v3.075c0 2.29.711 3.8 2.888 3.8 2.064 0 2.847-1.619 2.847-3.784Z"></path>
                <path d="M110.459 43.231c0 .426 0 2.12.042 2.49a.254.254 0 0 1-.269.157c-.171-.242-.583-.741-1.821-2.149l-3.3-3.758c-.386-.442-1.353-1.608-1.651-1.923h-.029a3.646 3.646 0 0 0-.072.925v3.1a11.13 11.13 0 0 0 .258 2.948c.085.156.37.241.727.269l.44.045a.259.259 0 0 1-.03.355 36.81 36.81 0 0 0-1.664-.044c-.6 0-.982.016-1.48.044a.26.26 0 0 1-.029-.355l.384-.045c.33-.042.556-.127.627-.285a13.684 13.684 0 0 0 .185-2.932v-4.109a1.308 1.308 0 0 0-.312-1.026 1.554 1.554 0 0 0-.883-.34l-.24-.03a.25.25 0 0 1 .026-.356c.6.042 1.353.042 1.608.042a4.569 4.569 0 0 0 .656-.042 19.576 19.576 0 0 0 2.434 3.131l1.379 1.553c.983 1.094 1.68 1.893 2.351 2.577h.026a1.411 1.411 0 0 0 .059-.6v-3.045a10.057 10.057 0 0 0-.288-2.946c-.085-.131-.314-.216-.883-.285l-.242-.03c-.1-.085-.086-.314.029-.356.655.029 1.139.042 1.68.042.61 0 .982-.014 1.464-.042a.25.25 0 0 1 .03.356l-.2.03c-.456.069-.741.184-.8.3a11.525 11.525 0 0 0-.215 2.932Z"></path>
                <path d="M121.06 38.42c0-1.509-.029-1.751-.868-1.821l-.356-.029c-.086-.059-.056-.315.029-.357.711.03 1.2.042 1.794.042.568 0 1.052-.013 1.764-.042.085.042.114.3.029.357l-.355.029c-.841.07-.869.312-.869 1.821v5.067c0 1.508.029 1.707.869 1.805l.355.045c.085.056.056.312-.029.355-.712-.026-1.2-.042-1.764-.042-.6 0-1.083.016-1.794.042a.275.275 0 0 1-.029-.355l.356-.045c.84-.1.868-.3.868-1.805Z"></path>
                <path d="M132.587 37.764c-.327-.8-.541-1.1-1.153-1.167l-.258-.029a.241.241 0 0 1 .03-.357c.412.03.868.044 1.48.044s1.123-.014 1.723-.044a.252.252 0 0 1 .029.357l-.215.029c-.541.07-.653.155-.668.27a17.4 17.4 0 0 0 .711 1.994c.655 1.652 1.31 3.286 2.024 4.91.439-.939 1.038-2.4 1.366-3.159.411-.967 1.082-2.576 1.323-3.188a1.493 1.493 0 0 0 .129-.556c0-.1-.144-.214-.642-.27l-.255-.029a.248.248 0 0 1 .029-.357c.4.03.939.044 1.478.044.471 0 .913-.014 1.382-.044a.286.286 0 0 1 .03.357l-.427.029a.912.912 0 0 0-.742.5 34.7 34.7 0 0 0-1.593 3.273l-.769 1.749c-.57 1.313-1.238 2.961-1.48 3.716a.289.289 0 0 1-.154.042.6.6 0 0 1-.216-.042 16.663 16.663 0 0 0-.655-1.893Z"></path>
                <path d="M149.72 38.393c0-1.483-.029-1.7-.868-1.794l-.229-.03c-.085-.059-.056-.314.029-.356.613.029 1.1.042 1.68.042h2.676a19.158 19.158 0 0 0 1.92-.042 16.157 16.157 0 0 1 .229 1.895.28.28 0 0 1-.356.026c-.214-.668-.341-1.166-1.082-1.352a6.689 6.689 0 0 0-1.382-.086h-1.024c-.424 0-.424.03-.424.571v2.846c0 .4.042.4.466.4h.828a5.292 5.292 0 0 0 1.208-.086c.173-.057.272-.144.343-.5l.115-.583a.28.28 0 0 1 .368.014c0 .343-.059.9-.059 1.44 0 .511.059 1.054.059 1.363a.277.277 0 0 1-.368.016l-.131-.554a.6.6 0 0 0-.439-.541 4.665 4.665 0 0 0-1.1-.072h-.828c-.424 0-.466.014-.466.386v2.005c0 .755.042 1.238.266 1.478.172.171.471.328 1.725.328a4.152 4.152 0 0 0 1.821-.214 3.56 3.56 0 0 0 1.009-1.379.258.258 0 0 1 .356.1 12.187 12.187 0 0 1-.641 1.975 187.565 187.565 0 0 0-3.814-.042h-1.281c-.612 0-1.1.016-1.937.042a.273.273 0 0 1-.028-.354l.469-.045c.813-.07.884-.285.884-1.777Z"></path>
                <path d="M165.143 38.406c0-1.353-.042-1.6-.625-1.668l-.456-.057a.239.239 0 0 1 .016-.357c.794-.072 1.779-.115 3.171-.115a5.042 5.042 0 0 1 2.378.428 2.122 2.122 0 0 1 1.18 1.995 2.631 2.631 0 0 1-1.777 2.375c-.071.085 0 .229.07.343a15.243 15.243 0 0 0 2.862 3.784 1.706 1.706 0 0 0 .982.4.119.119 0 0 1 .014.2 2.211 2.211 0 0 1-.626.072c-1.211 0-1.937-.357-2.945-1.793-.372-.527-.956-1.509-1.4-2.149a1.015 1.015 0 0 0-1.01-.455c-.641 0-.668.013-.668.314v1.793c0 1.492.028 1.664.854 1.777l.3.045a.277.277 0 0 1-.026.354 39.424 39.424 0 0 0-1.694-.042c-.6 0-1.113.016-1.779.042a.273.273 0 0 1-.029-.354l.354-.045c.826-.1.854-.285.854-1.777Zm1.169 2.034a1.137 1.137 0 0 0 .043.471c.042.042.256.07.981.07a2.374 2.374 0 0 0 1.467-.371 2.038 2.038 0 0 0 .711-1.763 2.1 2.1 0 0 0-2.279-2.194c-.88 0-.922.056-.922.456Z"></path>
                <path d="M182.344 45.878a4.45 4.45 0 0 1-2.221-.527 6.642 6.642 0 0 1-.386-1.993c.072-.1.286-.129.343-.042a2.588 2.588 0 0 0 2.447 2.119 1.617 1.617 0 0 0 1.781-1.635 2.136 2.136 0 0 0-1.169-1.991l-1.349-.883a3.036 3.036 0 0 1-1.539-2.447c0-1.353 1.054-2.45 2.9-2.45a5.568 5.568 0 0 1 1.325.184 1.824 1.824 0 0 0 .5.085 6.268 6.268 0 0 1 .261 1.739c-.056.085-.285.127-.355.042a1.862 1.862 0 0 0-1.935-1.61 1.493 1.493 0 0 0-1.694 1.581 2.232 2.232 0 0 0 1.209 1.749l1.139.713a3.2 3.2 0 0 1 1.779 2.732c0 1.565-1.181 2.633-3.033 2.633"></path>
                <path d="M195.052 38.42c0-1.509-.028-1.751-.868-1.821l-.356-.029c-.086-.059-.056-.315.029-.357.713.03 1.2.042 1.794.042.57 0 1.052-.013 1.766-.042.085.042.114.3.026.357l-.357.029c-.836.07-.866.312-.866 1.821v5.067c0 1.508.03 1.707.866 1.805l.357.045c.087.056.059.312-.026.355-.714-.026-1.2-.042-1.766-.042-.6 0-1.081.016-1.794.042a.275.275 0 0 1-.029-.355l.356-.045c.841-.1.868-.3.868-1.805Z"></path>
                <path d="M210.855 43.516c0 1.492.026 1.707.85 1.777l.458.045c.085.056.055.311-.03.355-.812-.028-1.294-.044-1.864-.044s-1.067.016-1.98.044a.248.248 0 0 1 0-.355l.513-.045c.812-.07.883-.285.883-1.777v-6.364c0-.442 0-.455-.428-.455h-.782a2.917 2.917 0 0 0-1.75.356 2.6 2.6 0 0 0-.641.983.279.279 0 0 1-.371-.1 14.749 14.749 0 0 0 .541-2.122.363.363 0 0 1 .271 0c.086.455.557.44 1.212.44h5.765c.767 0 .895-.028 1.108-.4.071-.026.229-.014.255.042a8.59 8.59 0 0 0-.213 2.165c-.056.113-.3.113-.371.026a1.834 1.834 0 0 0-.354-1.038 2.84 2.84 0 0 0-1.654-.356h-1.01c-.426 0-.41.013-.41.484Z"></path>
                <path d="M227.075 43.487c0 1.521.085 1.749.854 1.805l.543.045a.277.277 0 0 1-.03.355 58.187 58.187 0 0 0-1.937-.042c-.6 0-1.108.015-1.864.042a.276.276 0 0 1-.03-.355l.444-.045c.823-.085.85-.285.85-1.805v-.727a2.872 2.872 0 0 0-.452-1.763l-1.68-3.275c-.485-.939-.7-1.011-1.124-1.081l-.4-.071a.264.264 0 0 1 .029-.357c.454.029.967.042 1.652.042.655 0 1.166-.014 1.523-.042.127.042.127.271.043.357l-.184.029c-.5.07-.6.141-.6.256a7.159 7.159 0 0 0 .442 1.1c.525 1.052 1.052 2.175 1.609 3.161.439-.757.91-1.581 1.337-2.407a12.915 12.915 0 0 0 .923-1.88c0-.085-.255-.184-.6-.226l-.256-.029a.239.239 0 0 1 .029-.357 22.046 22.046 0 0 0 2.689 0 .24.24 0 0 1 .03.357l-.4.071c-.741.127-1.169 1.026-1.88 2.25l-.895 1.55a3.274 3.274 0 0 0-.671 2.3Z"></path>
                <path d="M24.118 49.407a36.482 36.482 0 0 0 4.278-2.8V32.213l-4.278-2.846Z"></path>
                <path d="m24.118 19.401 4.278 2.848v-7.273q-2.136-.1-4.278-.116Z"></path>
                <path d="M36.959 15.733a105.155 105.155 0 0 0-4.28-.456v9.821l4.28 2.849Z"></path>
                <path d="M32.679 35.062v7.828a43.411 43.411 0 0 0 4.28-4.968v-.013Z"></path>
                <path d="m36.959 37.922.006-.011h-.006Z"></path>
                <path d="M41.231 16.377v14.416l.013.007a49.2 49.2 0 0 0 4.22-13.623l-.146-.029a110.63 110.63 0 0 0-4.086-.771"></path>
                <path d="M28.402 22.253v9.963l4.278 2.846v-9.963Z"></path>
                <path d="M36.959 27.947v9.966h.006a44.284 44.284 0 0 0 4.269-7.088v-.029Z"></path>
                <path d="M41.231 30.823a.114.114 0 0 0 .013-.023l-.013-.006Z"></path>
                <path d="M14.734 32.864a6.732 6.732 0 0 0 3.224-2.005 13.077 13.077 0 0 0-1.64-.475 10.338 10.338 0 0 1-1.584 2.481"></path>
                <path d="M7.945 30.861a6.723 6.723 0 0 0 3.228 2.005 10.411 10.411 0 0 1-1.586-2.485 13.269 13.269 0 0 0-1.642.48"></path>
                <path d="M12.331 20.483a9.525 9.525 0 0 0-1.463 2.141c.458.061.945.1 1.463.122Z"></path>
                <path d="M16.749 29.216a14.147 14.147 0 0 1 1.966.61 6.653 6.653 0 0 0 .926-2.817h-2.5a9.657 9.657 0 0 1-.4 2.207"></path>
                <path d="M11.173 19.919a6.725 6.725 0 0 0-3.228 2.006 13.4 13.4 0 0 0 1.641.477 10.381 10.381 0 0 1 1.587-2.483"></path>
                <path d="M17.958 21.925a6.716 6.716 0 0 0-3.224-2 10.373 10.373 0 0 1 1.585 2.481 13.38 13.38 0 0 0 1.639-.477"></path>
                <path d="M17.145 25.775h2.5a6.637 6.637 0 0 0-.926-2.817 14.267 14.267 0 0 1-1.966.609 9.668 9.668 0 0 1 .4 2.209"></path>
                <path d="M13.567 32.308a9.519 9.519 0 0 0 1.471-2.147c-.461-.061-.952-.1-1.471-.124Z"></path>
                <path d="M12.331 27.009H9.992a8.325 8.325 0 0 0 .39 1.972 16.812 16.812 0 0 1 1.944-.178Z"></path>
                <path d="M13.567 28.804a16.739 16.739 0 0 1 1.952.177 8.3 8.3 0 0 0 .389-1.971h-2.341Z"></path>
                <path d="M15.044 22.623a9.316 9.316 0 0 0-1.477-2.153v2.276c.523-.017 1.016-.06 1.477-.121"></path>
                <path d="M15.909 25.777a8.314 8.314 0 0 0-.387-1.973 17.275 17.275 0 0 1-1.955.179v1.794Z"></path>
                <path d="M10.868 30.161a9.688 9.688 0 0 0 1.463 2.14v-2.263q-.777.033-1.463.123"></path>
                <path d="M9.156 23.568a14.42 14.42 0 0 1-1.972-.609 6.714 6.714 0 0 0-.923 2.817h2.5a9.668 9.668 0 0 1 .4-2.209"></path>
                <path d="M12.331 23.977a16.812 16.812 0 0 1-1.944-.178 8.3 8.3 0 0 0-.395 1.978h2.335Z"></path>
                <path d="M8.76 27.009h-2.5a6.7 6.7 0 0 0 .923 2.817 14.545 14.545 0 0 1 1.972-.612 9.614 9.614 0 0 1-.4-2.205"></path>
                <path d="M.76 17.185c3.138 19.411 15.293 28.884 21.346 32.221V14.86A107.686 107.686 0 0 0 .908 17.152Zm12.189 1.252a7.955 7.955 0 1 1-7.951 7.955 7.965 7.965 0 0 1 7.954-7.955"></path>
                <path d="m45.305 2.345-.016-.008a106.323 106.323 0 0 0-44.378.008l-.031.008A1.16 1.16 0 0 0 0 3.48v4.013a61.564 61.564 0 0 0 .45 7.482 110.216 110.216 0 0 1 45.329-.007 62.135 62.135 0 0 0 .446-7.475V3.48a1.157 1.157 0 0 0-.921-1.135m-5 9.932A96.723 96.723 0 0 0 23.115 10.9a96.693 96.693 0 0 0-17.19 1.38.251.251 0 0 1-.28-.372L9.3 3.824c.037-.075 2.4-1.245 2.4-1.245a23.777 23.777 0 0 1 11.407.921h.013a23.773 23.773 0 0 1 11.4-.916s2.364 1.17 2.4 1.245l3.658 8.081a.249.249 0 0 1-.277.372"></path>
                <path d="m33.573 3.966 1.539 4.121a22.261 22.261 0 0 0-10.943 1.168v.386c4.145-.156 9.386-.387 13.624.524l-2.309-5.312Z"></path>
                <path d="m11.067 8.087 1.539-4.121-1.91.888-2.312 5.311c4.236-.912 9.478-.681 13.622-.524v-.385a22.245 22.245 0 0 0-10.939-1.169"></path>
                <path d="M59.89 10.71c0-3.628-.325-4.02-2.254-4.184l-.815-.064c-.2-.131-.131-.719.065-.817 1.894.064 3 .1 4.377.1 1.308 0 2.419-.033 3.725-.1.2.1.263.686.067.817l-.49.064c-1.927.263-1.993.718-1.993 4.184v12.12a18.192 18.192 0 0 1-.751 6.336 7.1 7.1 0 0 1-6.443 4.709c-.39 0-1.4-.033-1.4-.687 0-.556.488-1.5 1.175-1.5a3.99 3.99 0 0 1 1.21.2 5.056 5.056 0 0 0 1.371.229 1.4 1.4 0 0 0 1.308-.85c.751-1.537.848-6.436.848-8.2Z"></path>
                <path d="M64.985 18.403a9.529 9.529 0 0 1 9.816-9.73c6.367 0 9.565 4.6 9.565 9.454a9.409 9.409 0 0 1-9.565 9.619c-6.119 0-9.813-4.381-9.813-9.343m16.621.579c0-4.547-2.012-9.454-7.277-9.454-2.867 0-6.587 1.958-6.587 7.995 0 4.077 1.983 9.372 7.415 9.372 3.308 0 6.449-2.481 6.449-7.912"></path>
                <path d="M90.624 18.404c-1.322 0-1.378.056-1.378.884v3.88c0 2.894.138 3.28 1.68 3.445l.8.084c.165.109.11.605-.055.688a87.801 87.801 0 0 0-3.5-.083c-1.184 0-2.122.056-3.2.083a.535.535 0 0 1-.054-.688l.469-.084c1.542-.276 1.6-.551 1.6-3.445v-9.919c0-2.894-.193-3.362-1.626-3.473l-.717-.055c-.167-.11-.11-.606.054-.689 1.351.026 2.289.083 3.473.083 1.076 0 2.013-.026 3.2-.083.167.083.223.579.055.689l-.524.055c-1.6.165-1.653.579-1.653 3.473v3.171c0 .853.056.882 1.378.882h7.883c1.323 0 1.377-.029 1.377-.882v-3.171c0-2.894-.054-3.308-1.68-3.473l-.524-.055c-.165-.11-.11-.606.055-.689 1.268.056 2.2.083 3.334.083 1.076 0 2.013-.026 3.254-.083.164.083.221.579.055.689l-.579.055c-1.6.165-1.654.579-1.654 3.473v9.919c0 2.894.055 3.252 1.654 3.445l.662.084c.164.109.11.605-.056.688a74.179 74.179 0 0 0-3.335-.083c-1.13 0-2.121.026-3.334.083a.534.534 0 0 1-.055-.688l.524-.084c1.68-.276 1.68-.551 1.68-3.445v-3.88c0-.828-.054-.884-1.377-.884Z"></path>
                <path d="M123.287 22.621c0 .828 0 4.107.083 4.822a.5.5 0 0 1-.524.3c-.332-.466-1.13-1.431-3.527-4.161l-6.395-7.277c-.747-.854-2.619-3.115-3.2-3.719h-.055a6.978 6.978 0 0 0-.138 1.789v6.01c0 1.294.028 4.879.5 5.7.168.305.717.47 1.407.525l.852.084a.5.5 0 0 1-.053.688 70.196 70.196 0 0 0-3.225-.083c-1.158 0-1.9.028-2.868.083a.51.51 0 0 1-.055-.688l.744-.084c.635-.083 1.075-.248 1.213-.551.385-.991.359-4.354.359-5.677v-7.959a2.517 2.517 0 0 0-.608-1.985 2.984 2.984 0 0 0-1.709-.663l-.469-.054a.486.486 0 0 1 .055-.69c1.158.083 2.62.083 3.116.083a8.694 8.694 0 0 0 1.266-.083c.554 1.407 3.806 5.045 4.716 6.065l2.671 3c1.9 2.122 3.252 3.666 4.549 4.989h.055a2.75 2.75 0 0 0 .111-1.158v-5.9c0-1.3-.03-4.878-.552-5.706-.164-.248-.608-.413-1.709-.551l-.467-.054c-.193-.165-.168-.608.053-.69 1.27.056 2.206.083 3.254.083 1.185 0 1.9-.026 2.838-.083a.485.485 0 0 1 .056.69l-.386.054c-.882.138-1.434.359-1.545.579-.467.992-.412 4.41-.412 5.678Z"></path>
                <path d="M130.572 27.746a8.611 8.611 0 0 1-4.3-1.019 12.8 12.8 0 0 1-.745-3.858c.14-.2.552-.249.663-.083.413 1.4 1.542 4.107 4.741 4.107a3.132 3.132 0 0 0 3.447-3.17 4.142 4.142 0 0 0-2.261-3.86l-2.619-1.71a5.879 5.879 0 0 1-2.976-4.74c0-2.619 2.039-4.74 5.622-4.74a10.723 10.723 0 0 1 2.565.357 3.6 3.6 0 0 0 .962.167 12.044 12.044 0 0 1 .5 3.362c-.11.167-.551.248-.69.084-.357-1.324-1.1-3.116-3.747-3.116-2.7 0-3.28 1.792-3.28 3.062 0 1.6 1.324 2.754 2.343 3.389l2.205 1.377c1.738 1.075 3.445 2.674 3.445 5.292 0 3.032-2.288 5.1-5.871 5.1"></path>
                <path d="M152.804 16.755c-1.568 0-1.633.063-1.633 1.043v4.608c0 3.429.163 3.888 1.993 4.085l.949.1c.2.13.131.718-.067.813-1.764-.064-2.875-.1-4.148-.1-1.406 0-2.517.064-3.791.1a.629.629 0 0 1-.065-.813l.556-.1c1.828-.328 1.895-.656 1.895-4.085V10.643c0-3.43-.229-3.986-1.927-4.116l-.85-.065c-.2-.13-.131-.719.065-.817 1.6.032 2.712.1 4.117.1 1.274 0 2.384-.032 3.79-.1.2.1.261.687.063.817l-.62.065c-1.894.2-1.959.686-1.959 4.116v3.758c0 1.013.065 1.044 1.633 1.044h9.345c1.568 0 1.633-.031 1.633-1.044v-3.758c0-3.43-.065-3.919-1.994-4.116l-.621-.065c-.2-.13-.13-.719.067-.817 1.5.065 2.614.1 3.953.1a78.82 78.82 0 0 0 3.857-.1c.2.1.261.687.063.817l-.685.068c-1.9.2-1.96.686-1.96 4.116v11.76c0 3.429.065 3.855 1.96 4.085l.783.1c.2.13.132.718-.065.813a91.858 91.858 0 0 0-3.953-.1c-1.339 0-2.515.031-3.953.1a.63.63 0 0 1-.067-.813l.621-.1c1.994-.328 1.994-.656 1.994-4.085v-4.608c0-.98-.065-1.043-1.633-1.043Z"></path>
                <path d="M168.998 18.403a9.53 9.53 0 0 1 9.813-9.73c6.368 0 9.565 4.6 9.565 9.454a9.409 9.409 0 0 1-9.565 9.618c-6.118 0-9.813-4.381-9.813-9.343m16.621.579c0-4.547-2.011-9.454-7.277-9.454-2.865 0-6.587 1.958-6.587 7.995 0 4.077 1.985 9.372 7.415 9.372 3.309 0 6.449-2.481 6.449-7.912"></path>
                <path d="M193.365 23.173c0 2.894.055 3.252 1.819 3.445l.744.083a.538.538 0 0 1-.055.689 93.976 93.976 0 0 0-3.583-.083c-1.158 0-2.15.026-3.362.083a.533.533 0 0 1-.055-.689l.579-.083c1.6-.219 1.653-.551 1.653-3.445v-10.17c0-2.344-.055-2.813-1.3-2.95l-.992-.111a.458.458 0 0 1 .056-.688 43.187 43.187 0 0 1 5.456-.222 9.943 9.943 0 0 1 5.292 1.1 4.762 4.762 0 0 1 2.205 4.133 4.67 4.67 0 0 1-2.756 4.384 9.3 9.3 0 0 1-3.8.826c-.193-.08-.193-.495-.028-.549 2.977-.552 4.052-2.235 4.052-4.632a4.063 4.063 0 0 0-4.437-4.409c-1.462 0-1.49.108-1.49.992Z"></path>
                <path d="M204.833 13.249c0-2.894-.056-3.335-1.654-3.472l-.69-.055c-.165-.111-.109-.608.055-.69 1.351.056 2.233.083 3.472.083 1.076 0 2.013-.026 3.2-.083.165.083.222.579.054.69l-.523.055c-1.6.164-1.654.578-1.654 3.472v3.172c0 .523.055 1.045.359 1.045a1.561 1.561 0 0 0 .771-.3c.386-.33 1.1-1.047 1.407-1.323l2.976-2.948c.524-.5 1.874-1.9 2.15-2.288a.815.815 0 0 0 .194-.442c0-.109-.11-.192-.47-.274l-.744-.165a.462.462 0 0 1 .055-.69c.965.056 2.067.083 3.032.083s1.9-.026 2.728-.083a.511.511 0 0 1 .056.69 6.8 6.8 0 0 0-2.562.772 27.406 27.406 0 0 0-3.86 3.2l-2.481 2.342c-.386.388-.634.635-.634.854 0 .2.165.415.524.854 2.619 3 4.741 5.485 6.892 7.745a3.193 3.193 0 0 0 2.1 1.131l.532.083a.48.48 0 0 1-.056.689c-.717-.056-1.472-.084-2.768-.084-1.13 0-2.1.028-3.364.084-.193-.056-.274-.526-.11-.689l.634-.113c.387-.054.661-.137.661-.273 0-.167-.193-.386-.387-.635-.524-.662-1.239-1.406-2.287-2.591l-2.205-2.481c-1.571-1.763-2.012-2.314-2.674-2.314-.413 0-.469.358-.469 1.323v3.556c0 2.893.055 3.279 1.6 3.445l.745.083c.163.11.11.605-.056.689a76.564 76.564 0 0 0-3.364-.084 61.17 61.17 0 0 0-3.2.084.533.533 0 0 1-.055-.689l.552-.083c1.46-.221 1.516-.552 1.516-3.445Z"></path>
                <path d="M223.752 13.305c0-2.922-.055-3.391-1.682-3.528l-.687-.055c-.167-.111-.111-.606.053-.69 1.379.056 2.316.084 3.474.084 1.1 0 2.04-.028 3.417-.084.167.084.222.579.056.69l-.689.055c-1.626.137-1.681.606-1.681 3.528v9.811c0 2.923.055 3.309 1.681 3.5l.689.083c.165.11.11.605-.056.689a78.794 78.794 0 0 0-3.417-.084c-1.158 0-1.99.028-3.367.084a.536.536 0 0 1-.055-.689l.582-.083c1.627-.193 1.682-.579 1.682-3.5Z"></path>
                <path d="M247.594 22.621c0 .828 0 4.107.084 4.822a.5.5 0 0 1-.524.3c-.331-.466-1.13-1.431-3.528-4.161l-6.395-7.277c-.744-.854-2.62-3.115-3.2-3.719h-.054a6.9 6.9 0 0 0-.14 1.789v6.01c0 1.294.029 4.879.5 5.7.166.305.717.47 1.406.525l.854.084a.506.506 0 0 1-.055.688 70.16 70.16 0 0 0-3.225-.083c-1.157 0-1.9.028-2.865.083a.508.508 0 0 1-.056-.688l.745-.084c.633-.083 1.074-.248 1.212-.551.386-.991.358-4.354.358-5.677v-7.959a2.524 2.524 0 0 0-.606-1.985 2.99 2.99 0 0 0-1.709-.663l-.469-.054a.483.483 0 0 1 .056-.69c1.158.083 2.617.083 3.113.083a8.737 8.737 0 0 0 1.268-.083c.552 1.407 3.8 5.045 4.712 6.065l2.675 3c1.9 2.122 3.251 3.666 4.548 4.989h.055a2.776 2.776 0 0 0 .11-1.158v-5.9c0-1.3-.028-4.878-.551-5.706-.165-.248-.605-.413-1.708-.551l-.469-.054c-.194-.165-.167-.608.053-.69 1.269.056 2.205.083 3.254.083 1.186 0 1.9-.026 2.838-.083a.485.485 0 0 1 .056.69l-.386.054c-.882.138-1.433.359-1.545.579-.469.992-.413 4.41-.413 5.678Z"></path>
                <path d="M254.834 27.746a8.612 8.612 0 0 1-4.3-1.019 12.8 12.8 0 0 1-.745-3.858c.14-.2.554-.249.663-.083.413 1.4 1.544 4.107 4.742 4.107a3.132 3.132 0 0 0 3.445-3.17 4.144 4.144 0 0 0-2.261-3.86l-2.617-1.71a5.876 5.876 0 0 1-2.978-4.74c0-2.619 2.04-4.74 5.624-4.74a10.73 10.73 0 0 1 2.563.357 3.611 3.611 0 0 0 .962.167 12 12 0 0 1 .5 3.362c-.111.167-.55.248-.689.084-.358-1.324-1.1-3.116-3.747-3.116-2.7 0-3.281 1.792-3.281 3.062 0 1.6 1.324 2.754 2.342 3.389l2.206 1.377c1.736 1.075 3.445 2.674 3.445 5.292 0 3.032-2.288 5.1-5.87 5.1"></path>
            </svg>
            <div>
                <h1 style="margin: 0; color: #002D72; font-size: 24px;">Confluence Knowledge Assistant</h1>
                <p style="margin: 5px 0 0; color: #333; font-weight: 400;">
                    Ask questions about your organization's knowledge base. The assistant will search through the content and provide relevant answers with source links.
                </p>
            </div>
        </div>
        """)
        
        chatbot = gr.Chatbot(
            show_label=False,
            avatar_images=(None, "logos-vertical.jpg"),  # Use the local logo file for the chatbot avatar
            bubble_full_width=False,
            height=600,
        )
        msg = gr.Textbox(
            placeholder="Ask a question about your organization...",
            container=False,
            scale=8,
        )
        clear = gr.ClearButton([msg, chatbot], value="Clear chat")
        
        # Add message selection and editing functionality
        chatbot.select(select_message, None, msg)
        
        # Connect the chat function to the interface
        msg.submit(chat_function, [msg, chatbot], [chatbot], queue=False).then(
            lambda: "", None, [msg], queue=False
        )
        
        # Add a minimal button for the copy action
        copy_button = gr.Button(
            "📋",  # Using a clipboard emoji as a minimal visual indicator
            elem_id="copy-button",  # For styling
            visible=True
        )
        
        # Custom CSS to style the interface with JHU branding
        gr.HTML("""
        <style>
        /* JHU Brand Colors */
        :root {
            --heritage-blue: #002D72;
            --spirit-blue: #68ACE5;
            --medium-blue: #0077D8;
            --gold: #F1C400;
            --white: #FFFFFF;
            --light-gray: #F5F5F5;
            --text-gray: #333333;
        }
        
        /* Global styling */
        body {
            background-color: var(--white) !important;
            font-family: 'Open Sans', 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
            color: var(--text-gray) !important;
        }
        
        /* Header styling */
        h1, h2, h3 {
            color: var(--heritage-blue) !important;
            font-weight: 600 !important;
        }
        
        /* Chat container */
        .gradio-container {
            background-color: var(--white) !important;
            max-width: 1200px !important;
            margin: 0 auto !important;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1) !important;
            border-radius: 8px !important;
        }
        
        /* Chatbot messages */
        .gradio-chatbot {
            border-radius: 8px !important;
            border: 1px solid #E0E0E0 !important;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05) !important;
        }
        
        /* Bot message styling */
        .gradio-chatbot [class*="bot"] {
            background-color: var(--light-gray) !important;
            border-radius: 12px !important;
            padding: 12px 16px !important;
            color: var(--text-gray) !important;
            font-size: 15px !important;
            line-height: 1.5 !important;
        }
        
        /* User message styling */
        .gradio-chatbot .user-message {
            background-color: var(--heritage-blue) !important;
            color: white !important;
            border-radius: 12px !important;
            padding: 12px 16px !important;
            font-size: 15px !important;
            line-height: 1.5 !important;
        }
        
        /* Input area styling */
        .gradio-textbox textarea {
            border: 2px solid #E0E0E0 !important;
            border-radius: 8px !important;
            padding: 12px !important;
            font-size: 16px !important;
        }
        
        /* Button styling */
        button {
            background-color: var(--heritage-blue) !important;
            color: white !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            padding: 10px 15px !important;
        }
        
        button:hover {
            background-color: #003d9e !important;
            transform: translateY(-1px) !important;
        }
        
        /* Clear button with special styling */
        button.secondary {
            background-color: #E0E0E0 !important;
            color: var(--text-gray) !important;
        }
        
        button.secondary:hover {
            background-color: #CCCCCC !important;
        }
        
        /* Source links styling */
        a {
            color: var(--medium-blue) !important;
            text-decoration: none !important;
            font-weight: 600 !important;
        }
        
        a:hover {
            color: var(--heritage-blue) !important;
            text-decoration: underline !important;
        }
        
        /* Code blocks styling */
        pre, code {
            background-color: #f0f4f8 !important;
            border: 1px solid #e6eaef !important;
            border-radius: 4px !important;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
            font-size: 14px !important;
            color: #333 !important;
            padding: 2px 4px !important;
        }
        
        pre {
            padding: 12px !important;
            overflow-x: auto !important;
        }
        
        /* Improved contrast for message content */
        .message {
            font-weight: 500 !important;
        }
        
        /* Emphasized text */
        strong, b {
            font-weight: 700 !important;
        }
        
        /* Chatbot scrollbar styling */
        .gradio-chatbot::-webkit-scrollbar {
            width: 8px !important;
        }
        
        .gradio-chatbot::-webkit-scrollbar-track {
            background: #f1f1f1 !important;
        }
        
        .gradio-chatbot::-webkit-scrollbar-thumb {
            background: #888 !important;
            border-radius: 10px !important;
        }
        
        .gradio-chatbot::-webkit-scrollbar-thumb:hover {
            background: #555 !important;
        }
        </style>
        """)
        
        # Add the enhanced copy button functionality
        copy_button.click(
            fn=lambda: None,
            _js="""
            () => {
                try {
                    console.log("Starting enhanced formatted copy operation");
                    
                    // Find all message elements in the chatbot
                    const allMessages = document.querySelectorAll('[class*="message"], [class*="bot"], .message');
                    console.log("All message elements found:", allMessages.length);
                    
                    // Filter to only get bot/assistant messages
                    const botMessages = Array.from(allMessages).filter(msg => {
                        // Not a user message
                        const hasUserClass = msg.classList.contains('user') || 
                                           msg.classList.contains('user-message') || 
                                           msg.parentElement.classList.contains('user');
                        
                        // Check content for bot indicators
                        const hasBot = msg.innerHTML.includes('gstatic.com/images') || 
                                      msg.innerHTML.includes('assistant') ||
                                      msg.innerHTML.includes('jhu.edu');
                        
                        return !hasUserClass && hasBot;
                    });
                    
                    console.log("Filtered bot messages:", botMessages.length);
                    
                    // Standard fallback search if needed
                    if (botMessages.length === 0) {
                        const userMessages = document.querySelectorAll('.user, .user-message');
                        for (const userMsg of userMessages) {
                            const nextElement = userMsg.nextElementSibling;
                            if (nextElement && !nextElement.classList.contains('user')) {
                                botMessages.push(nextElement);
                            }
                        }
                    }
                    
                    if (botMessages.length === 0) {
                        alert("No assistant responses found to copy");
                        return;
                    }
                    
                    // Get only the LAST bot message (most recent)
                    const lastBotMessage = botMessages[botMessages.length - 1];
                    
                    // Create hidden rich text element to maintain formatting
                    const hiddenDiv = document.createElement('div');
                    hiddenDiv.style.position = 'absolute';
                    hiddenDiv.style.left = '-9999px';
                    hiddenDiv.style.top = '0';
                    
                    // Check if this is Confluence content
                    const isConfluence = lastBotMessage.textContent.includes("Content of '");
                    
                    if (isConfluence) {
                        // For Confluence content, use a more structured format
                        console.log("Formatting Confluence content");
                        
                        // Extract the raw content
                        const rawText = lastBotMessage.textContent;
                        
                        // Create styled HTML 
                        hiddenDiv.innerHTML = rawText
                            // Style headings
                            .replace(/^# (.*)$/gm, '<h1>$1</h1>')
                            .replace(/^## (.*)$/gm, '<h2>$1</h2>')
                            .replace(/^### (.*)$/gm, '<h3>$1</h3>')
                            
                            // Style lists
                            .replace(/^- (.*)$/gm, '<ul><li>$1</li></ul>')
                            
                            // Style links
                            .replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g, '<a href="$2">$1</a>')
                            
                            // Style code blocks
                            .replace(/```([\\s\\S]*?)```/g, '<pre><code>$1</code></pre>')
                            
                            // Style bold and italic
                            .replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>')
                            .replace(/\\*([^*]+)\\*/g, '<em>$1</em>')
                            
                            // Preserve paragraphs
                            .replace(/\\n\\n/g, '<br/><br/>');
                    } else {
                        // For regular messages
                        hiddenDiv.innerHTML = lastBotMessage.innerHTML;
                    }
                    
                    // Add to document
                    document.body.appendChild(hiddenDiv);
                    
                    // Try both copy methods for maximum compatibility
                    
                    // Method 1: Selection and execCommand
                    const range = document.createRange();
                    range.selectNode(hiddenDiv);
                    const selection = window.getSelection();
                    selection.removeAllRanges();
                    selection.addRange(range);
                    
                    const execCommandSuccess = document.execCommand('copy');
                    
                    // Clean up selection
                    selection.removeAllRanges();
                    
                    // Method 2: Try clipboard API with HTML if the first method fails
                    if (!execCommandSuccess) {
                        try {
                            // Modern clipboard API with both text and HTML
                            navigator.clipboard.write([
                                new ClipboardItem({
                                    'text/plain': new Blob([hiddenDiv.textContent], {type: 'text/plain'}),
                                    'text/html': new Blob([hiddenDiv.innerHTML], {type: 'text/html'})
                                })
                            ]);
                        } catch (clipErr) {
                            // Final fallback to text-only
                            navigator.clipboard.writeText(hiddenDiv.textContent);
                        }
                    }
                    
                    // Clean up
                    document.body.removeChild(hiddenDiv);
                    
                    // Show success feedback
                    const feedbackElem = document.createElement('div');
                    feedbackElem.textContent = '✓ Copied!';
                    feedbackElem.style.position = 'fixed';
                    feedbackElem.style.bottom = '20px';
                    feedbackElem.style.left = '50%';
                    feedbackElem.style.transform = 'translateX(-50%)';
                    feedbackElem.style.backgroundColor = '#002D72';  // JHU Heritage Blue
                    feedbackElem.style.color = 'white';
                    feedbackElem.style.padding = '10px 20px';
                    feedbackElem.style.borderRadius = '4px';
                    feedbackElem.style.zIndex = '1000';
                    
                    document.body.appendChild(feedbackElem);
                    
                    setTimeout(() => {
                        feedbackElem.style.opacity = '0';
                        feedbackElem.style.transition = 'opacity 0.5s';
                        setTimeout(() => document.body.removeChild(feedbackElem), 500);
                    }, 2000);
                    
                } catch (e) {
                    console.error("Enhanced copy operation failed:", e);
                    alert("Error during copy operation: " + e.message);
                }
            }
            """
        )
    
    # Launch the interface
    demo.launch(server_name="localhost", server_port=8888, share=False)