import os
from dotenv import load_dotenv
from atlassian import Confluence

def get_confluence_client():
    """
    Get an authenticated Confluence client using basic authentication.
    """
    # Load environment variables
    load_dotenv()
    
    # Get credentials
    confluence_domain = os.getenv('CONFLUENCE_DOMAIN')
    access_token = os.getenv('CONFLUENCE_ACCESS_TOKEN').strip("'")
    username = os.getenv('USERNAME')
    
    print(f"Connecting to: {confluence_domain}")
    print(f"Using username: {username}")
    
    try:
        # Create Confluence client with basic authentication
        confluence = Confluence(
            url=confluence_domain,
            username=username,
            password=access_token,
            cloud=True  # Set to True for Atlassian Cloud
        )
        
        # Test the connection by getting available spaces
        spaces = confluence.get_all_spaces(start=0, limit=50)
        if spaces and 'results' in spaces:
            print(f"Successfully connected to Confluence. Found {len(spaces['results'])} accessible spaces.")
            return confluence
        else:
            print("Failed to get spaces from Confluence")
            return None
            
    except Exception as e:
        print(f"Error connecting to Confluence: {str(e)}")
        return None

if __name__ == "__main__":
    try:
        confluence = get_confluence_client()
        # Test with a simple API call
        spaces = confluence.get_all_spaces(limit=1)
        print("Successfully connected to Confluence!")
        print(f"Found space: {spaces[0]['name'] if spaces else 'No spaces found'}")
    except Exception as e:
        print(f"Connection failed: {str(e)}") 