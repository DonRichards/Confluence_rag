import requests
from requests.auth import HTTPBasicAuth
import os
from dotenv import load_dotenv
import pandas as pd
import sys
from bs4 import BeautifulSoup
from utils.auth import get_confluence_client

# Load environment variables from .env file
load_dotenv()
confluence_domain = os.getenv("CONFLUENCE_DOMAIN")
# username = os.getenv("username")
# password = os.getenv("password")
confluence_api_key = os.getenv("CONFLUENCE_API_KEY")
# Set your Confluence details here
space_key = 'SUP'  # replace with your info

# Function to fetch pages from Confluence
def fetch_pages(start=0, limit=25):
    """Fetch pages from Confluence"""
    try:
        # Get authenticated client
        confluence = get_confluence_client()
        
        # Get spaces from environment variable
        space_keys = os.getenv('SPACES', '').split(',')
        if not space_keys or space_keys[0] == '':
            print("No spaces specified in SPACES environment variable. Using all global spaces.")
            
            # Get all spaces if none specified
            print("Fetching all spaces from Confluence...")
            spaces_response = confluence.get_all_spaces(start=0, limit=50)
            
            if isinstance(spaces_response, dict) and 'results' in spaces_response:
                spaces = spaces_response['results']
                print(f"Found {len(spaces)} spaces")
            else:
                print(f"Unexpected API response format: {spaces_response}")
                return []
        else:
            print(f"Using specified spaces: {', '.join(space_keys)}")
            spaces = []
            
            # First, get all available spaces to check keys
            all_spaces_response = confluence.get_all_spaces(limit=100)
            available_spaces = {}
            
            if isinstance(all_spaces_response, dict) and 'results' in all_spaces_response:
                for space in all_spaces_response['results']:
                    available_spaces[space.get('key', '').upper()] = space
            
            # Now try to match the requested spaces
            for space_key in space_keys:
                space_key = space_key.strip()
                if not space_key:
                    continue
                
                # Try exact match first
                try:
                    space_info = confluence.get_space(space_key)
                    if space_info:
                        spaces.append(space_info)
                        print(f"Found space: {space_info.get('name', 'Unknown')} ({space_key})")
                        continue
                except Exception:
                    # Try case-insensitive match
                    upper_key = space_key.upper()
                    if upper_key in available_spaces:
                        space_info = available_spaces[upper_key]
                        spaces.append(space_info)
                        print(f"Found space with different case: {space_info.get('name', 'Unknown')} ({space_info.get('key', 'Unknown')})")
                    else:
                        print(f"Space not found: {space_key}")
        
        all_pages = []
        
        for space in spaces:
            print(f"Processing space: {space.get('name', 'Unknown')} ({space.get('key', 'Unknown')})")
            if space['type'] != 'global':
                print(f"Skipping non-global space: {space.get('name', 'Unknown')}")
                continue
                
            # Get pages from space
            try:
                print(f"Fetching pages from space: {space.get('key', 'Unknown')}")
                pages = confluence.get_all_pages_from_space(
                    space['key'], 
                    start=start, 
                    limit=limit, 
                    status='current'
                )
                
                print(f"Found {len(pages)} pages in space {space.get('key', 'Unknown')}")
                
                for page in pages:
                    try:
                        page_content = confluence.get_page_by_id(
                            page['id'], 
                            expand='body.storage'
                        )
                        
                        # Extract the content
                        content = page_content['body']['storage']['value']
                        
                        # Add to our pages list
                        all_pages.append({
                            'id': page['id'],
                            'title': page['title'],
                            'url': f"{os.getenv('CONFLUENCE_DOMAIN')}/pages/viewpage.action?pageId={page['id']}",
                            'content': content
                        })
                    except Exception as e:
                        print(f"Error processing page {page.get('id', 'Unknown')}: {str(e)}")
            except Exception as e:
                print(f"Error fetching pages from space {space.get('key', 'Unknown')}: {str(e)}")
        
        return all_pages
        
    except Exception as e:
        print(f"Error fetching from Confluence: {str(e)}")
        return []

# Function to make an API call
def api_call(url):
    """Make an API call to Confluence"""
    confluence_token = os.getenv('CONFLUENCE_ACCESS_TOKEN').strip("'")
    
    # For Atlassian API tokens, use the token as the password with email as username
    username = os.getenv('USERNAME')  # Your Atlassian account email
    
    response = requests.get(
        url, 
        auth=HTTPBasicAuth(username, confluence_token)
    )
    
    if response.status_code != 200:
        print(f"Error: API call failed with status code {response.status_code}")
        print(f"Response: {response.text}")
        return None
    
    return response.json()


# Function to fetch labels from Confluence
def fetch_labels(page_id):
    url = f'{confluence_domain}/rest/api/content/{page_id}/label'
    json_data = api_call(url)

    if json_data:
        try:
            internal_only = False
            for item in json_data.get("results", []):
                if item.get("name") == 'internal_only':
                    internal_only = True

            return internal_only
        except KeyError:
            print("Error processing JSON data.")
            return None
    else:
        print("Failed to fetch labels.")
        return None


# Function to fetch page content from Confluence
def fetch_page_content(page_id):
    url = f'{confluence_domain}/rest/api/content/{page_id}?expand=body.storage'
    json_data = api_call(url)

    if json_data:
        try:
            return json_data['body']['storage']['value']
        except KeyError:
            print("Error: Unable to access page content in the returned JSON.")
            return None
    else:
        print("Failed to fetch page content.")
        return None
    

# Function to create an empty DataFrame    
def create_dataframe():
    try:
        columns = ['id', 'type', 'status', 'tiny_link', 'title', 'content', 'is_internal']
        df = pd.DataFrame(columns=columns)
        return df
    except Exception as e:
        print(f"An error occurred while creating the DataFrame: {e}")
        return None


# Function to add all pages to the DataFrame
def add_all_pages_to_dataframe(df, all_pages):
    if not isinstance(df, pd.DataFrame):
        print("Error: The first argument must be a pandas DataFrame.")
        return None

    if not isinstance(all_pages, list):
        print("Error: The second argument must be a list.")
        return None

    for page in all_pages:
        try:
            new_record = [{
                'id': page.get('id', ''),
                'type': page.get('type', ''),
                'status': page.get('status', ''),
                'tiny_link': page.get('_links', {}).get('tinyui', ''),
                'title': page.get('title', '')
            }]

            # Add new records to the DataFrame
            df = pd.concat([df, pd.DataFrame(new_record)], ignore_index=True)
        except Exception as e:
            print(f"An error occurred while adding a page to the DataFrame: {e}")

    return df


# Function index of the DataFrame
def set_index_of_dataframe(df):
    if not isinstance(df, pd.DataFrame):
        print("Error: The argument must be a pandas DataFrame.")
        return None

    if 'id' not in df.columns:
        print("Error: 'id' column not found in the DataFrame.")
        return None

    try:
        df.set_index('id', inplace=True)
        return df
    except Exception as e:
        print(f"An error occurred while setting the index: {e}")
        return None

# Function to fetch by limit
def fetch_pages_by_limit(all_pages, start, limit):
    if not isinstance(all_pages, list):
        print("Error: 'all_pages' must be a list.")
        return None

    while True:
        response_data = fetch_pages(start, limit)
        if response_data:
            results = response_data.get('results')
            if results:
                all_pages.extend(results)
                start += limit
                if start >= response_data.get('size', 0):
                    break
            else:
                print("Warning: No results found in the response.")
                break
        else:
            print("Error: Failed to fetch pages.")
            return None

    return all_pages

from tqdm import tqdm  # Make sure to import tqdm at the top of your script

def fetch_all_pages(all_pages, start, limit, max_chunk_size=200):
    if not isinstance(all_pages, list):
        print("Error: 'all_pages' must be a list.")
        return None

    # Calculate the total number of chunks to fetch based on the limit and max_chunk_size
    total_chunks = (limit + max_chunk_size - 1) // max_chunk_size

    # Initialize the tqdm progress bar
    with tqdm(total=limit, desc="Fetching pages") as pbar:
        while True:
            chunk_size = min(limit, max_chunk_size)  # Determine the size of the next chunk
            response_data = fetch_pages(start, chunk_size)
            if response_data:
                results = response_data.get('results')
                if results is not None:
                    all_pages.extend(results)
                    fetched_count = len(results)
                    pbar.update(fetched_count)  # Update the progress bar with the number of fetched results
                    if fetched_count < chunk_size:
                        break  # Break the loop if the number of results is less than the chunk size
                    start += fetched_count
                    limit -= fetched_count  # Decrease the remaining limit by the number of fetched results
                    if limit <= 0:
                        break  # If the remaining limit is 0 or less, we've fetched everything needed
                else:
                    print("Warning: No results found in the response.")
                    break
            else:
                print("Error: Failed to fetch pages.")
                return None
            
    return all_pages


# Function to delete internal_only records
def delete_internal_only_records(df):
    # Ensure df is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: The variable 'df' must be a pandas DataFrame.")
        return df
    
    # Loop through the DataFrame with a tqdm progress bar
    if 'is_internal' in df.columns:
        for page_id, row in tqdm(df.iterrows(), total=df.shape[0], desc="Updating is_internal status"):
            is_internal_page = fetch_labels(page_id)
            
            if is_internal_page is not None:
                df.loc[page_id, 'is_internal'] = is_internal_page
            else:
                print(f"Warning: Could not fetch labels for page ID {page_id}.")
    else:
        print("Error: 'is_internal' column not found in the DataFrame.")
        return df
    
    # Delete internal_only records
    df = df[df['is_internal'] != True]

    return df


def add_content_to_dataframe(df):
    # Check if the input is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: The variable 'df' must be a pandas DataFrame.")
        return df

    # Wrap the loop in tqdm for progress tracking
    for page_id, row in tqdm(df.iterrows(), total=df.shape[0], desc="Updating DataFrame"):
        html_content = fetch_page_content(page_id)

        if html_content is not None:
            try:
                # Parse the HTML content
                soup = BeautifulSoup(html_content, "lxml")

                # Extract text with proper spacing
                text_parts = []
                for element in soup.stripped_strings:
                    text_parts.append(element)

                page_content = ' '.join(text_parts)

                # Update the DataFrame with the extracted content
                df.loc[page_id, 'content'] = page_content
            except Exception as e:
                print(f"Error processing HTML content for page ID {page_id}: {e}")
        else:
            print(f"Warning: Could not fetch content for page ID {page_id}.")

    return df



def save_dataframe_to_csv(df, filename):
    if not isinstance(df, pd.DataFrame):
        print("Error: The variable 'df' must be a pandas DataFrame.")
    else:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            df.to_csv(filename, index=True)
            print("Data successfully saved " + str(len(df)) + " records to " + filename)
        except Exception as e:
            print(f"An error occurred while saving the DataFrame to CSV: {e}")

def test_confluence_connection():
    """Test the connection to Confluence"""
    try:
        print("Testing Confluence connection...")
        confluence = get_confluence_client()
        
        # Get spaces from environment variable
        space_keys = os.getenv('SPACES', '').split(',')
        if space_keys and space_keys[0] != '':
            print(f"Testing with specified space: {space_keys[0]}")
            space_key = space_keys[0].strip()
            
            try:
                space_info = confluence.get_space(space_key)
                if space_info:
                    print(f"Successfully connected to Confluence!")
                    print(f"Found space: {space_info.get('name', 'Unknown')} ({space_key})")
                    return True
                else:
                    print(f"Space not found: {space_key}")
                    # Still return True since connection works
                    return True
            except Exception as e:
                print(f"Error fetching space {space_key}: {str(e)}")
                # Try the default method as fallback
        
        # Fallback to getting any space
        print("Fetching any space...")
        spaces_response = confluence.get_all_spaces(limit=1)
        
        if isinstance(spaces_response, dict) and 'results' in spaces_response:
            spaces = spaces_response['results']
            print(f"Found {len(spaces)} spaces")
            
            if spaces:
                print(f"Successfully connected to Confluence!")
                print(f"Found space: {spaces[0]['name']}")
            else:
                print("No spaces found, but connection is working.")
                
            return True
        else:
            print(f"Unexpected API response format: {spaces_response}")
            return False
            
    except Exception as e:
        print(f"Connection failed with error: {str(e)}")
        print(f"Exception type: {type(e)}")
        return False

def list_all_available_spaces():
    """List all spaces the API token can access with proper pagination"""
    try:
        confluence = get_confluence_client()
        all_spaces = []
        start = 0
        limit = 50  # Number of spaces per page
        
        print("Fetching all accessible spaces (paginating through results)...")
        
        while True:
            print(f"Fetching spaces {start} to {start+limit}...")
            spaces_response = confluence.get_all_spaces(start=start, limit=limit)
            
            if not isinstance(spaces_response, dict) or 'results' not in spaces_response:
                print(f"Unexpected API response format: {spaces_response}")
                break
                
            spaces = spaces_response['results']
            if not spaces:
                break  # No more spaces to fetch
                
            all_spaces.extend(spaces)
            print(f"Retrieved {len(spaces)} spaces in this page")
            
            # Check if we've reached the end
            if 'size' in spaces_response and start + len(spaces) >= spaces_response['size']:
                print(f"Reached end of results (total: {spaces_response['size']})")
                break
                
            start += len(spaces)
        
        print(f"\nFound {len(all_spaces)} total accessible spaces:")
        for space in all_spaces:
            print(f"  - {space.get('name', 'Unknown')} (Key: {space.get('key', 'Unknown')})")
        
        return all_spaces
    except Exception as e:
        print(f"Error listing spaces: {str(e)}")
        return []

def fetch_all_content():
    """Fetch all content directly using the content endpoint with pagination"""
    try:
        confluence = get_confluence_client()
        all_content = []
        space_content_counts = {}
        start = 0
        limit = 25  # Number of items per page
        
        print("Fetching all content (paginating through results)...")
        
        while True:
            print(f"Fetching content items {start} to {start+limit}...")
            
            # Use the correct method for fetching content
            # The method is get_content() or get_all_pages() depending on what you need
            url = f"{os.getenv('CONFLUENCE_DOMAIN')}/rest/api/content"
            params = {
                'start': start,
                'limit': limit,
                'type': 'page',
                'status': 'current',
                'expand': 'body.storage,space'
            }
            
            # Use the REST API directly since the wrapper doesn't have the method we need
            response = requests.get(
                url,
                params=params,
                auth=(os.getenv('USERNAME'), os.getenv('CONFLUENCE_ACCESS_TOKEN').strip("'"))
            )
            
            if response.status_code != 200:
                print(f"Error: API call failed with status code {response.status_code}")
                print(f"Response: {response.text}")
                break
                
            content_response = response.json()
            
            if 'results' not in content_response:
                print(f"Unexpected API response format: {content_response}")
                break
                
            content_items = content_response['results']
            if not content_items:
                break  # No more content to fetch
                
            all_content.extend(content_items)
            print(f"Retrieved {len(content_items)} content items in this page")
            
            # Track which spaces the content comes from
            for item in content_items:
                if 'space' in item and 'key' in item['space']:
                    space_key = item['space']['key']
                    if space_key not in space_content_counts:
                        space_content_counts[space_key] = 0
                    space_content_counts[space_key] += 1
            
            # Check if we've reached the end
            if 'size' in content_response and start + len(content_items) >= content_response['size']:
                print(f"Reached end of results (total: {content_response['size']})")
                break
                
            start += len(content_items)
        
        print(f"\nFound {len(all_content)} total content items across {len(space_content_counts)} spaces")
        print("\nContent distribution by space:")
        for space_key, count in sorted(space_content_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {space_key}: {count} items")
        
        # Process the content items
        processed_content = []
        for item in all_content:
            try:
                # Extract the content
                content = item['body']['storage']['value']
                space_key = item.get('space', {}).get('key', 'Unknown')
                space_name = item.get('space', {}).get('name', 'Unknown')
                
                processed_content.append({
                    'id': item['id'],
                    'title': item['title'],
                    'url': f"{os.getenv('CONFLUENCE_DOMAIN')}/pages/viewpage.action?pageId={item['id']}",
                    'content': content,
                    'space_key': space_key,
                    'space_name': space_name
                })
            except Exception as e:
                print(f"Error processing content item {item.get('id', 'Unknown')}: {str(e)}")
        
        return processed_content
        
    except Exception as e:
        print(f"Error fetching content: {str(e)}")
        return []

def fetch_all_content_alternative():
    """Fetch all content using existing Atlassian Python API methods"""
    try:
        confluence = get_confluence_client()
        all_content = []
        space_content_counts = {}
        
        # First get all spaces
        spaces_response = confluence.get_all_spaces(limit=100)
        spaces = []
        
        if isinstance(spaces_response, dict) and 'results' in spaces_response:
            spaces = spaces_response['results']
        
        print(f"Fetching content from {len(spaces)} spaces...")
        
        # For each space, get all pages
        for space in spaces:
            space_key = space.get('key')
            space_name = space.get('name', 'Unknown')
            print(f"Fetching pages from space: {space_name} ({space_key})")
            
            try:
                # Get all pages from this space
                pages = confluence.get_all_pages_from_space(space_key, start=0, limit=500)
                
                print(f"Found {len(pages)} pages in space {space_key}")
                
                if space_key not in space_content_counts:
                    space_content_counts[space_key] = 0
                space_content_counts[space_key] += len(pages)
                
                # For each page, get the content
                for page in pages:
                    try:
                        page_content = confluence.get_page_by_id(
                            page['id'], 
                            expand='body.storage'
                        )
                        
                        # Extract the content
                        content = page_content['body']['storage']['value']
                        
                        all_content.append({
                            'id': page['id'],
                            'title': page['title'],
                            'url': f"{os.getenv('CONFLUENCE_DOMAIN')}/pages/viewpage.action?pageId={page['id']}",
                            'content': content,
                            'space_key': space_key,
                            'space_name': space_name
                        })
                    except Exception as e:
                        print(f"Error processing page {page.get('id', 'Unknown')}: {str(e)}")
            except Exception as e:
                print(f"Error fetching pages from space {space_key}: {str(e)}")
        
        print(f"\nFound {len(all_content)} total content items across {len(space_content_counts)} spaces")
        print("\nContent distribution by space:")
        for space_key, count in sorted(space_content_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {space_key}: {count} items")
        
        return all_content
        
    except Exception as e:
        print(f"Error fetching content: {str(e)}")
        return []

def fetch_content_direct():
    """Fetch content directly using the content API with search"""
    try:
        all_content = []
        space_content_counts = {}
        
        print("Fetching content directly via content API...")
        
        # Get credentials
        confluence_domain = os.getenv('CONFLUENCE_DOMAIN')
        username = os.getenv('USERNAME')
        token = os.getenv('CONFLUENCE_ACCESS_TOKEN').strip("'")
        
        # Get spaces to include from environment variable (if any)
        space_filter = os.getenv('SPACES', '').split(',')
        space_filter = [s.strip() for s in space_filter if s.strip()]
        
        # Start with page 0
        start = 0
        limit = 50
        has_more = True
        
        # Keep fetching pages until there are no more
        page_num = 1
        while has_more and page_num <= 20:  # Limit to 20 pages for now
            print(f"Fetching content page {page_num} (items {start} to {start+limit-1})...")
            
            # Construct the URL with explicit pagination parameters
            url = f"{confluence_domain}/rest/api/content"
            params = {
                'type': 'page',
                'status': 'current',
                'expand': 'body.storage,space,version',
                'limit': limit,
                'start': start
            }
            
            # Make the API call
            try:
                response = requests.get(
                    url,
                    params=params,
                    auth=HTTPBasicAuth(username, token)
                )
                
                if response.status_code != 200:
                    print(f"Error fetching content: {response.status_code}")
                    print(f"Response: {response.text}")
                    break
                
                # Parse the response
                content_data = response.json()
                
                # Process the results
                results = content_data.get('results', [])
                if not results:
                    print("No more results found.")
                    has_more = False
                    break
                    
                print(f"Found {len(results)} items in page {page_num}")
                
                # Process each result
                for item in results:
                    try:
                        # Get space information
                        space_key = item.get('space', {}).get('key', 'Unknown')
                        space_name = item.get('space', {}).get('name', 'Unknown')
                        
                        # Skip if we're filtering by space and this space isn't in our filter
                        if space_filter and space_key not in space_filter:
                            continue
                        
                        # Track content by space
                        if space_key not in space_content_counts:
                            space_content_counts[space_key] = 0
                        space_content_counts[space_key] += 1
                        
                        # Get the content body
                        content = ""
                        if 'body' in item and 'storage' in item['body']:
                            content = item['body']['storage']['value']
                        
                        # Add to our content list
                        all_content.append({
                            'id': item['id'],
                            'title': item['title'],
                            'url': f"{confluence_domain}/pages/viewpage.action?pageId={item['id']}",
                            'content': content,
                            'space_key': space_key,
                            'space_name': space_name,
                            'content_type': 'page'
                        })
                        
                    except Exception as e:
                        print(f"Error processing content item {item.get('id', 'Unknown')}: {str(e)}")
                
                # Check if we have more results to fetch
                if len(results) < limit:
                    has_more = False
                else:
                    # Move to the next page
                    start += limit
                    page_num += 1
                    
            except Exception as e:
                print(f"Error fetching content: {str(e)}")
                has_more = False
        
        # Try using search API to find content in specific spaces
        if space_filter and (not all_content or len(space_content_counts) < len(space_filter)):
            print("\nTrying search API to find content in specific spaces...")
            
            for space_key in space_filter:
                if space_key in space_content_counts:
                    continue  # Already have content for this space
                    
                print(f"Searching for content in space: {space_key}")
                
                # Use CQL search
                cql = f"space = {space_key} AND type = page"
                search_url = f"{confluence_domain}/rest/api/search"
                search_params = {
                    'cql': cql,
                    'limit': 50,
                    'expand': 'content.body.storage,content.space'
                }
                
                try:
                    search_response = requests.get(
                        search_url,
                        params=search_params,
                        auth=HTTPBasicAuth(username, token)
                    )
                    
                    if search_response.status_code != 200:
                        print(f"Error searching content for space {space_key}: {search_response.status_code}")
                        continue
                    
                    search_data = search_response.json()
                    search_results = search_data.get('results', [])
                    
                    print(f"Found {len(search_results)} items in space {space_key} via search")
                    
                    for result in search_results:
                        try:
                            # The content is in the content property
                            content_item = result.get('content', {})
                            if not content_item:
                                continue
                                
                            # Get space information
                            result_space_key = content_item.get('space', {}).get('key', 'Unknown')
                            result_space_name = content_item.get('space', {}).get('name', 'Unknown')
                            
                            # Track content by space
                            if result_space_key not in space_content_counts:
                                space_content_counts[result_space_key] = 0
                            space_content_counts[result_space_key] += 1
                            
                            # Get the content body
                            content = ""
                            if 'body' in content_item and 'storage' in content_item['body']:
                                content = content_item['body']['storage']['value']
                            
                            # Add to our content list
                            all_content.append({
                                'id': content_item['id'],
                                'title': content_item['title'],
                                'url': f"{confluence_domain}/pages/viewpage.action?pageId={content_item['id']}",
                                'content': content,
                                'space_key': result_space_key,
                                'space_name': result_space_name,
                                'content_type': 'page'
                            })
                            
                        except Exception as e:
                            print(f"Error processing search result: {str(e)}")
                    
                except Exception as e:
                    print(f"Error searching content for space {space_key}: {str(e)}")
        
        print(f"\nFound {len(all_content)} total content items across {len(space_content_counts)} spaces")
        print("\nContent distribution by space:")
        for space_key, count in sorted(space_content_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {space_key}: {count} items")
        
        # If we found content from spaces that weren't in the original filter, update the filter
        found_spaces = list(space_content_counts.keys())
        if found_spaces:
            print("\nSpaces with content: {', '.join(found_spaces)}")
            update_env_spaces(found_spaces)
        
        return all_content
        
    except Exception as e:
        print(f"Error fetching content: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def fetch_labels_for_content(content_items):
    """Fetch labels for a list of content items"""
    print("Fetching labels for content items...")
    
    confluence_domain = os.getenv('CONFLUENCE_DOMAIN')
    username = os.getenv('USERNAME')
    token = os.getenv('CONFLUENCE_ACCESS_TOKEN').strip("'")
    
    for item in content_items:
        try:
            content_id = item['id']
            url = f"{confluence_domain}/rest/api/content/{content_id}/label"
            
            response = requests.get(
                url,
                auth=HTTPBasicAuth(username, token)
            )
            
            if response.status_code == 200:
                labels_data = response.json()
                labels = [label.get('name', '') for label in labels_data.get('results', [])]
                
                # Add labels to the content item
                item['labels'] = labels
                
                # Check for internal_only label
                item['is_internal'] = 'internal_only' in labels
            else:
                item['labels'] = []
                item['is_internal'] = False
                
        except Exception as e:
            print(f"Error fetching labels for content {item.get('id', '')}: {str(e)}")
            item['labels'] = []
            item['is_internal'] = False
    
    return content_items

def main():
    csv_file = './data/kb.csv'
    
    # Create the data directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    print("Fetching content directly...")
    
    # Use the direct content API approach
    all_content = fetch_content_direct()
    
    if not all_content:
        print("Failed to fetch content. Exiting.")
        return
        
    print(f"Total content items fetched: {len(all_content)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_content)
    
    # Save to CSV
    if not df.empty:
        df.to_csv(csv_file, index=False)
        print(f"Data successfully saved: {len(df)} records to {csv_file}")
    else:
        print("No content to save after filtering")

def update_env_spaces(space_keys):
    """Update the SPACES variable in the .env file"""
    try:
        # Read the current .env file
        with open('.env', 'r') as f:
            lines = f.readlines()
        
        # Update or add the SPACES line
        spaces_line = f"SPACES={','.join(space_keys)}\n"
        spaces_found = False
        
        for i, line in enumerate(lines):
            if line.startswith('SPACES='):
                lines[i] = spaces_line
                spaces_found = True
                break
        
        if not spaces_found:
            lines.append(spaces_line)
        
        # Write the updated content back to the .env file
        with open('.env', 'w') as f:
            f.writelines(lines)
            
        print(f"Updated SPACES in .env file to: {','.join(space_keys)}")
        
    except Exception as e:
        print(f"Error updating .env file: {str(e)}")

def fetch_content_from_spaces():
    """Fetch content directly from spaces using explicit URL parameter pagination"""
    try:
        all_content = []
        space_content_counts = {}
        
        print("Fetching content directly from spaces...")
        
        # Get credentials
        confluence_domain = os.getenv('CONFLUENCE_DOMAIN')
        username = os.getenv('USERNAME')
        token = os.getenv('CONFLUENCE_ACCESS_TOKEN').strip("'")
        
        # Get spaces to include from environment variable (if any)
        space_filter = os.getenv('SPACES', '').split(',')
        space_filter = [s.strip() for s in space_filter if s.strip()]
        
        if not space_filter:
            print("No spaces specified in SPACES environment variable.")
            # Try to get a list of spaces to use
            try:
                spaces_url = f"{confluence_domain}/rest/api/space"
                spaces_response = requests.get(
                    spaces_url,
                    params={'limit': 100},
                    auth=HTTPBasicAuth(username, token)
                )
                
                if spaces_response.status_code == 200:
                    spaces_data = spaces_response.json()
                    if 'results' in spaces_data:
                        space_filter = [space.get('key') for space in spaces_data.get('results', [])]
                        print(f"Found {len(space_filter)} spaces to use: {', '.join(space_filter)}")
            except Exception as e:
                print(f"Error getting spaces list: {str(e)}")
                
            if not space_filter:
                print("No spaces found. Please specify space keys in the SPACES environment variable.")
                return []
        
        # Process each space
        for space_key in space_filter:
            print(f"\nFetching content from space: {space_key}")
            space_content_count = 0
            
            # Process each content type we want to fetch
            for content_type in ['page', 'blogpost']:
                print(f"Fetching {content_type}s from space {space_key}...")
                
                # Start with page 0
                start = 0
                limit = 100
                has_more = True
                
                # Keep fetching pages until there are no more
                page_num = 1
                while has_more:
                    print(f"Fetching {content_type} page {page_num} from space {space_key} (items {start} to {start+limit-1})...")
                    
                    # Construct the URL with explicit pagination parameters
                    url = f"{confluence_domain}/rest/api/space/{space_key}/content/{content_type}"
                    params = {
                        'depth': 'all',
                        'expand': 'body.storage,version,space',
                        'limit': limit,
                        'start': start
                    }
                    
                    # Make the API call
                    try:
                        response = requests.get(
                            url,
                            params=params,
                            auth=HTTPBasicAuth(username, token)
                        )
                        
                        if response.status_code != 200:
                            print(f"Error fetching {content_type}s from space {space_key}: {response.status_code}")
                            if response.status_code == 404:
                                print(f"Space {space_key} not found or you don't have access to it.")
                            else:
                                print(f"Response: {response.text}")
                            break
                        
                        # Parse the response
                        content_data = response.json()
                        
                        # Debug the response structure
                        print(f"Response structure: {list(content_data.keys())}")
                        if content_type in content_data:
                            print(f"Links in response: {content_data[content_type].get('_links', {})}")
                        
                        # Process the content items
                        if content_type in content_data:
                            results = content_data[content_type].get('results', [])
                            print(f"Found {len(results)} {content_type} items in page {page_num}")
                            
                            # Process each result
                            for item in results:
                                try:
                                    # Get the content body if needed
                                    content = ""
                                    if 'body' in item and 'storage' in item['body']:
                                        content = item['body']['storage']['value']
                                    else:
                                        # If body isn't expanded, make another API call to get it
                                        content_url = f"{confluence_domain}/rest/api/content/{item['id']}?expand=body.storage"
                                        content_response = requests.get(
                                            content_url,
                                            auth=HTTPBasicAuth(username, token)
                                        )
                                        
                                        if content_response.status_code == 200:
                                            content_data = content_response.json()
                                            if 'body' in content_data and 'storage' in content_data['body']:
                                                content = content_data['body']['storage']['value']
                                    
                                    # Get space name
                                    space_name = space_key
                                    if 'space' in item and 'name' in item['space']:
                                        space_name = item['space']['name']
                                    elif '_expandable' in item and 'space' in item['_expandable']:
                                        # Try to get space info from the expandable URL
                                        space_url = item['_expandable']['space']
                                        if space_url.startswith('/'):
                                            space_url = f"{confluence_domain}{space_url}"
                                        
                                        space_response = requests.get(
                                            space_url,
                                            auth=HTTPBasicAuth(username, token)
                                        )
                                        
                                        if space_response.status_code == 200:
                                            space_data = space_response.json()
                                            if 'name' in space_data:
                                                space_name = space_data['name']
                                    
                                    # Add to our content list
                                    all_content.append({
                                        'id': item['id'],
                                        'title': item['title'],
                                        'url': f"{confluence_domain}/pages/viewpage.action?pageId={item['id']}",
                                        'content': content,
                                        'space_key': space_key,
                                        'space_name': space_name,
                                        'content_type': content_type
                                    })
                                    
                                    # Increment our counter
                                    space_content_count += 1
                                    
                                except Exception as e:
                                    print(f"Error processing content item {item.get('id', '')}: {str(e)}")
                            
                            # Check if we have more results to fetch
                            if len(results) < limit:
                                has_more = False
                            else:
                                # Move to the next page
                                start += limit
                                page_num += 1
                        else:
                            print(f"No {content_type} content found in response")
                            has_more = False
                            
                    except Exception as e:
                        print(f"Error fetching {content_type}s from space {space_key}: {str(e)}")
                        has_more = False
            
            # Update the space content counts
            if space_content_count > 0:
                space_content_counts[space_key] = space_content_count
                print(f"Total content items fetched from space {space_key}: {space_content_count}")
            else:
                print(f"No content found in space {space_key}")
        
        print(f"\nFound {len(all_content)} total content items across {len(space_content_counts)} spaces")
        print("\nContent distribution by space:")
        for space_key, count in sorted(space_content_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {space_key}: {count} items")
        
        # If we found content from spaces that weren't in the original filter, update the filter
        found_spaces = list(space_content_counts.keys())
        if found_spaces and set(found_spaces) != set(space_filter):
            print("\nUpdating space filter to include only spaces with content:")
            print(f"Original spaces: {', '.join(space_filter)}")
            print(f"Spaces with content: {', '.join(found_spaces)}")
            update_env_spaces(found_spaces)
        
        return all_content
        
    except Exception as e:
        print(f"Error fetching content: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def fetch_with_atlassian_api():
    """Fetch content using the Atlassian Python API"""
    try:
        from atlassian import Confluence
        
        # Get credentials
        confluence_domain = os.getenv('CONFLUENCE_DOMAIN')
        username = os.getenv('USERNAME')
        token = os.getenv('CONFLUENCE_ACCESS_TOKEN').strip("'")
        
        # Create Confluence client
        confluence = Confluence(
            url=confluence_domain,
            username=username,
            password=token,
            cloud=True  # Set to True if using Atlassian Cloud
        )
        
        # Get spaces to include from environment variable (if any)
        space_filter = os.getenv('SPACES', '').split(',')
        space_filter = [s.strip() for s in space_filter if s.strip()]
        
        all_content = []
        space_content_counts = {}
        
        # Try to get all spaces
        try:
            all_spaces = confluence.get_all_spaces()
            if isinstance(all_spaces, dict) and 'results' in all_spaces:
                spaces = all_spaces['results']
                print(f"Found {len(spaces)} spaces")
                
                # If no space filter, use all spaces
                if not space_filter:
                    space_filter = [space['key'] for space in spaces]
            else:
                print(f"Unexpected API response format: {all_spaces}")
        except Exception as e:
            print(f"Error getting spaces: {str(e)}")
        
        # Process each space
        for space_key in space_filter:
            print(f"\nFetching content from space: {space_key}")
            
            try:
                # Get all pages from space
                pages = confluence.get_all_pages_from_space(space_key)
                
                print(f"Found {len(pages)} pages in space {space_key}")
                
                for page in pages:
                    try:
                        # Get page content
                        page_content = confluence.get_page_by_id(
                            page['id'], 
                            expand='body.storage'
                        )
                        
                        # Extract the content
                        content = page_content['body']['storage']['value']
                        
                        # Track content by space
                        if space_key not in space_content_counts:
                            space_content_counts[space_key] = 0
                        space_content_counts[space_key] += 1
                        
                        # Add to our content list
                        all_content.append({
                            'id': page['id'],
                            'title': page['title'],
                            'url': f"{confluence_domain}/pages/viewpage.action?pageId={page['id']}",
                            'content': content,
                            'space_key': space_key,
                            'space_name': page.get('space', {}).get('name', space_key),
                            'content_type': 'page'
                        })
                        
                    except Exception as e:
                        print(f"Error processing page {page.get('id', 'Unknown')}: {str(e)}")
                
            except Exception as e:
                print(f"Error fetching pages from space {space_key}: {str(e)}")
        
        print(f"\nFound {len(all_content)} total content items across {len(space_content_counts)} spaces")
        print("\nContent distribution by space:")
        for space_key, count in sorted(space_content_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {space_key}: {count} items")
        
        return all_content
        
    except Exception as e:
        print(f"Error fetching with Atlassian API: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def check_api_access():
    """Check API access, permissions, and limits"""
    try:
        print("\n=== CHECKING CONFLUENCE API ACCESS ===\n")
        
        # Get credentials
        confluence_domain = os.getenv('CONFLUENCE_DOMAIN')
        username = os.getenv('USERNAME')
        token = os.getenv('CONFLUENCE_ACCESS_TOKEN').strip("'")
        
        # 1. Check authentication
        print("1. Testing authentication...")
        auth_url = f"{confluence_domain}/rest/api/user/current"
        auth_response = requests.get(
            auth_url,
            auth=HTTPBasicAuth(username, token)
        )
        
        if auth_response.status_code == 200:
            user_data = auth_response.json()
            print(f"✓ Authentication successful")
            print(f"   Logged in as: {user_data.get('displayName', 'Unknown')} ({user_data.get('username', 'Unknown')})")
            print(f"   User type: {user_data.get('type', 'Unknown')}")
        else:
            print(f"✗ Authentication failed: {auth_response.status_code}")
            print(f"  Response: {auth_response.text}")
            return
        
        # 2. Check spaces access with proper pagination
        print("\n2. Checking spaces access with pagination...")
        all_spaces = []
        start = 0
        limit = 100
        has_more = True
        
        while has_more:
            spaces_url = f"{confluence_domain}/rest/api/space?next=true&limit={limit}&start={start}"
            spaces_response = requests.get(
                spaces_url,
                auth=HTTPBasicAuth(username, token)
            )
            
            if spaces_response.status_code == 200:
                spaces_data = spaces_response.json()
                spaces = spaces_data.get('results', [])
                all_spaces.extend(spaces)
                
                print(f"Retrieved {len(spaces)} spaces in this page (total: {len(all_spaces)})")
                
                # Check if there are more spaces to fetch
                if '_links' in spaces_data and 'next' in spaces_data['_links']:
                    start += limit
                else:
                    has_more = False
            else:
                print(f"✗ Failed to retrieve spaces: {spaces_response.status_code}")
                print(f"  Response: {spaces_response.text}")
                break
        
        if all_spaces:
            print(f"\n✓ Successfully retrieved {len(all_spaces)} total spaces")
            
            # List all spaces
            print("\n   Available spaces:")
            for space in all_spaces:
                print(f"   - {space.get('name', 'Unknown')} (Key: {space.get('key', 'Unknown')}, Type: {space.get('type', 'Unknown')})")
            
            # Check specific spaces from .env
            space_filter = os.getenv('SPACES', '').split(',')
            space_filter = [s.strip() for s in space_filter if s.strip()]
            
            if space_filter:
                print("\n   Checking access to specified spaces:")
                available_space_keys = [space.get('key', '').upper() for space in all_spaces]
                
                for space_key in space_filter:
                    if space_key.upper() in available_space_keys:
                        print(f"   ✓ Space {space_key} is accessible")
                    else:
                        print(f"   ✗ Space {space_key} is NOT accessible")
                        
                        # Try direct access to check error
                        direct_space_url = f"{confluence_domain}/rest/api/space/{space_key}"
                        direct_response = requests.get(
                            direct_space_url,
                            auth=HTTPBasicAuth(username, token)
                        )
                        
                        if direct_response.status_code != 200:
                            print(f"     Error: {direct_response.status_code} - {direct_response.text}")
        
        # 3. Check rate limits (if available in headers)
        print("\n3. Checking API rate limits...")
        rate_limit_remaining = spaces_response.headers.get('X-RateLimit-Remaining')
        rate_limit = spaces_response.headers.get('X-RateLimit-Limit')
        
        if rate_limit and rate_limit_remaining:
            print(f"✓ Rate limit: {rate_limit_remaining}/{rate_limit} remaining")
        else:
            # Check for any rate limit related headers
            rate_headers = {k: v for k, v in spaces_response.headers.items() if 'rate' in k.lower()}
            if rate_headers:
                print("✓ Rate limit headers found:")
                for k, v in rate_headers.items():
                    print(f"   {k}: {v}")
            else:
                print("ℹ No explicit rate limit headers found")
        
        # 4. Test content access
        print("\n4. Testing content access...")
        
        # Try to get a list of content
        content_url = f"{confluence_domain}/rest/api/content"
        content_response = requests.get(
            content_url,
            params={'limit': 10, 'expand': 'space'},
            auth=HTTPBasicAuth(username, token)
        )
        
        if content_response.status_code == 200:
            content_data = content_response.json()
            content_items = content_data.get('results', [])
            print(f"✓ Successfully retrieved {len(content_items)} content items")
            
            if content_items:
                # Try to get details of the first item
                first_item = content_items[0]
                item_id = first_item.get('id')
                
                print(f"\n   Testing detailed content access for item {item_id}...")
                detail_url = f"{confluence_domain}/rest/api/content/{item_id}?expand=body.storage,space,version"
                
                detail_response = requests.get(
                    detail_url,
                    auth=HTTPBasicAuth(username, token)
                )
                
                if detail_response.status_code == 200:
                    print(f"✓ Successfully retrieved detailed content")
                    
                    # Check if we can get the body
                    detail_data = detail_response.json()
                    if 'body' in detail_data and 'storage' in detail_data['body']:
                        print(f"✓ Content body is accessible")
                    else:
                        print(f"✗ Content body is NOT accessible")
                else:
                    print(f"✗ Failed to retrieve detailed content: {detail_response.status_code}")
        else:
            print(f"✗ Failed to retrieve content: {content_response.status_code}")
            print(f"  Response: {content_response.text}")
        
        # 5. Check permissions (if available)
        print("\n5. Checking permissions...")
        
        # Try to get permissions for a space
        if all_spaces:
            test_space = all_spaces[0]['key']
            perm_url = f"{confluence_domain}/rest/api/space/{test_space}/permission"
            
            perm_response = requests.get(
                perm_url,
                auth=HTTPBasicAuth(username, token)
            )
            
            if perm_response.status_code == 200:
                print(f"✓ Successfully retrieved permissions for space {test_space}")
                
                # Try to parse permissions
                perm_data = perm_response.json()
                if isinstance(perm_data, dict):
                    for perm_type, perms in perm_data.items():
                        if isinstance(perms, list):
                            print(f"   {perm_type}: {len(perms)} permissions")
            else:
                print(f"ℹ Permission endpoint not accessible: {perm_response.status_code}")
                print(f"  This is normal if you don't have admin permissions")
        
        print("\n=== API ACCESS CHECK COMPLETE ===\n")
        
    except Exception as e:
        print(f"Error checking API access: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check API access and limits
    check_api_access()
    
    # First, list all spaces to see what's directly accessible
    print("Checking directly accessible spaces:")
    list_all_available_spaces()
    
    # Now run the main function to fetch all content
    print("\n\nFetching all accessible content:")
    if test_confluence_connection():
        main()
    else:
        print("Exiting due to connection issues.")