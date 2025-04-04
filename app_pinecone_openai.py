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
                    return f"Content of '{title}':\n\n{content}"
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
                            return f"Content of '{title}':\n\n{content}"
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

# Function to format content in a structured way
def format_content(content, title):
    """Format the content in a structured, readable way."""
    try:
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
                if current_section["content"]:
                    sections.append(current_section)
                current_section = {"title": line.lstrip('#').strip(), "content": []}
            else:
                current_section["content"].append(line)
        
        # Add the last section
        if current_section["content"]:
            sections.append(current_section)
        
        # Format the output
        output = f"# {title}\n\n"
        for section in sections:
            output += f"## {section['title']}\n"
            output += '\n'.join(section['content'])
            output += '\n\n'
        
        return output
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
    log_entry = {
        "timestamp": timestamp,
        "session_id": session_id,
        "user_input": user_input,
        "assistant_response": assistant_response,
        "search_results": [(source, score) for source, _, score in results] if results else []
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
    
    # Initialize Pinecone if not already initialized
    index = init_pinecone()
    if index is None:
        history.append((message, "Error: Failed to initialize Pinecone. Please check your API keys and environment settings."))
        current_history = history.copy()
        return history
    
    if not message or message.strip() == "":
        history.append((message, "Please enter a question to search the knowledge base."))
        current_history = history.copy()
        return history
    
    # Check for special commands
    special_response = process_message(message, history)
    if special_response:
        history.append((message, special_response))
        current_history = history.copy()
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
        
        # Combine answer with references
        response = f"{answer}\n{references}"
    
    # Generate a session ID if it doesn't exist in the thread
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Log the conversation
    log_conversation(message, response, session_id, results)
    
    # Print results to console
    print(f"Query processed: '{message}'. Response generated.")
    
    # Append the new message and response to the history
    history.append((message, response))
    current_history = history.copy()
    return history

# Legacy main function for backward compatibility
def main(query):
    # Create mock history for single query usage
    history = []
    response = chat_function(query, history)
    return response

if __name__ == "__main__":
    # Create Gradio interface with chat components
    gr.close_all()
    
    with gr.Blocks(title="Confluence Knowledge Assistant") as demo:
        gr.Markdown("# Confluence Knowledge Assistant")
        gr.Markdown("Ask questions about your organization's knowledge base. The assistant will search through the content and provide relevant answers with source links.")
        
        chatbot = gr.Chatbot(
            show_label=False,
            avatar_images=(None, "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSg3bGnGE50nrIQ1Z7uBYTzDa9ri_h5vJ9B_A&usqp=CAU"),
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
    
    # Launch the interface
    demo.launch(server_name="localhost", server_port=8888, share=False)