import os
from dotenv import load_dotenv
from atlassian import Confluence

def test_connection():
    # Load environment variables
    print("Loading .env environment variables...")
    load_dotenv()
    
    # Get credentials
    confluence_domain = os.getenv('CONFLUENCE_DOMAIN')
    access_token = os.getenv('CONFLUENCE_ACCESS_TOKEN').strip("'")
    username = os.getenv('USERNAME')
    
    print(f"Connecting to: {confluence_domain}")
    print(f"Using username: {username}")
    
    try:
        # Create Confluence client
        # Try with basic authentication instead of token authentication
        confluence = Confluence(
            url=confluence_domain,
            username=username,
            password=access_token,
            cloud=True  # Set to True for Atlassian Cloud
        )
        
        # First, get all available spaces to show what's accessible
        print("\nFetching all available spaces to check access...")
        all_spaces_response = confluence.get_all_spaces(limit=10000)
        
        if isinstance(all_spaces_response, dict) and 'results' in all_spaces_response:
            available_spaces = all_spaces_response['results']
            print(f"Found {len(available_spaces)} accessible spaces:")
            
            if available_spaces:
                print("Available spaces:")
                for space in available_spaces:
                    print(f"  - {space.get('name', 'Unknown')} (Key: {space.get('key', 'Unknown')})")
            else:
                print("No spaces available with your current access token.")
                return False
        else:
            print(f"Unexpected API response format: {all_spaces_response}")
            return False
        
        # Now check for the spaces specified in .env
        print("\nChecking spaces specified in .env file...")
        space_keys = os.getenv('SPACES', '').split(',')
        space_keys = [key.strip() for key in space_keys if key.strip()]
        
        if not space_keys:
            print("No spaces specified in SPACES environment variable.")
            print("Please add space keys to your .env file.")
            return False
        
        print(f"Spaces specified in .env: {', '.join(space_keys)}")
        
        # Check each specified space
        found_spaces = []
        available_space_keys = [space.get('key', '').upper() for space in available_spaces]
        
        for space_key in space_keys:
            # Try case-insensitive match
            if space_key.upper() in available_space_keys:
                # Find the actual space with its correct case
                for space in available_spaces:
                    if space.get('key', '').upper() == space_key.upper():
                        print(f"Found space: {space.get('name', 'Unknown')} (Key: {space.get('key', 'Unknown')})")
                        found_spaces.append(space)
                        break
            else:
                print(f"Space not found or no access: {space_key}")
        
        if found_spaces:
            print(f"\nSuccessfully confirmed access to {len(found_spaces)} of {len(space_keys)} specified spaces.")
            return True
        else:
            print("\nNone of the specified spaces in your .env file were found.")
            print("Please check the space keys and ensure your token has access to these spaces.")
            return False
            
    except Exception as e:
        print(f"Connection failed with error: {str(e)}")
        print(f"Exception type: {type(e)}")
        return False

def list_all_spaces():
    """List all spaces the user has access to"""
    # Load environment variables
    load_dotenv()
    
    # Get credentials
    confluence_domain = os.getenv('CONFLUENCE_DOMAIN')
    access_token = os.getenv('CONFLUENCE_ACCESS_TOKEN').strip("'")
    username = os.getenv('USERNAME')
    
    try:
        # Create Confluence client
        # Try with basic authentication instead of token authentication
        confluence = Confluence(
            url=confluence_domain,
            username=username,
            password=access_token,
            cloud=True  # Set to True for Atlassian Cloud
        )
        
        # Get all spaces with proper pagination
        print("Fetching all accessible spaces (paginating through results)...")
        all_spaces = []
        start = 0
        limit = 1000
        
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
            if len(spaces) < limit or ('size' in spaces_response and start + len(spaces) >= spaces_response['size']):
                print(f"Reached end of results (total: {len(all_spaces)})")
                break
                
            start += limit
        
        print(f"\nFound {len(all_spaces)} total accessible spaces:")
        for idx, space in enumerate(all_spaces, 1):
            print(f"{idx:3}. {space.get('name', 'Unknown')} (Key: {space.get('key', 'Unknown')})")
        
        return all_spaces
    except Exception as e:
        print(f"Error listing spaces: {str(e)}")
        return []

if __name__ == "__main__":
    print("=== Testing Confluence Connection ===")
    test_connection()
    
    # list_all_spaces()