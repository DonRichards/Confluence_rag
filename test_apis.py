import os
import logging
from dotenv import load_dotenv
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
logging.info("Environment variables loaded")

def test_pinecone():
    """Test connection to Pinecone service."""
    try:
        import pinecone

        # Check for required environment variables
        api_key = os.getenv('PINECONE_API_KEY')
        if not api_key:
            logging.error("PINECONE_API_KEY not found in environment variables")
            return False

        environment = os.getenv('PINECONE_ENVIRONMENT')
        if not environment:
            logging.error("PINECONE_ENVIRONMENT not found in environment variables")
            return False

        index_name = os.getenv('PINECONE_INDEX_NAME')
        if not index_name:
            logging.error("PINECONE_INDEX_NAME not found in environment variables")
            return False

        # Initialize Pinecone
        logging.info(f"Initializing Pinecone with environment: {environment}")
        pinecone.init(api_key=api_key, environment=environment)
        
        # List indexes to check connection
        logging.info("Listing Pinecone indexes...")
        indexes = pinecone.list_indexes()
        logging.info(f"Available indexes: {indexes}")
        
        # Check if our index exists
        if index_name not in indexes:
            logging.error(f"Index '{index_name}' not found in available indexes")
            return False
            
        # Connect to index
        logging.info(f"Connecting to index: {index_name}")
        index = pinecone.Index(index_name)
        
        # Get index stats
        logging.info("Retrieving index stats...")
        stats = index.describe_index_stats()
        logging.info(f"Index stats: {stats}")
        
        logging.info("Pinecone connection successful")
        return True
    except Exception as e:
        logging.error(f"Error testing Pinecone connection: {str(e)}")
        return False

def test_openai():
    """Test connection to OpenAI API."""
    try:
        import openai
        
        # Check for required environment variables
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logging.error("OPENAI_API_KEY not found in environment variables")
            return False
            
        # Set API key
        openai.api_key = api_key
        
        # Test embedding generation
        logging.info("Testing OpenAI embeddings API...")
        test_text = "This is a test query for embeddings"
        
        # Use client for OpenAI
        client = openai.Client(api_key=api_key)
        
        # Generate embedding
        response = client.embeddings.create(
            input=test_text,
            model="text-embedding-ada-002"
        )
        
        # Check response
        if hasattr(response, 'data') and len(response.data) > 0:
            embedding_length = len(response.data[0].embedding)
            logging.info(f"Successfully generated embedding with {embedding_length} dimensions")
            
            # Test completions API
            logging.info("Testing OpenAI chat completions API...")
            chat_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, how are you?"}
                ]
            )
            
            if hasattr(chat_response, 'choices') and len(chat_response.choices) > 0:
                logging.info(f"Chat API response: {chat_response.choices[0].message.content[:50]}...")
                logging.info("OpenAI API connection successful")
                return True
            else:
                logging.error("No valid response from chat completions API")
                return False
        else:
            logging.error("No valid response from embeddings API")
            return False
    except Exception as e:
        logging.error(f"Error testing OpenAI connection: {str(e)}")
        return False

def test_confluence():
    """Test connection to Confluence API."""
    try:
        from utils.auth import get_confluence_client
        
        # Check for required environment variables
        domain = os.getenv('CONFLUENCE_DOMAIN')
        if not domain:
            logging.error("CONFLUENCE_DOMAIN not found in environment variables")
            return False
            
        token = os.getenv('CONFLUENCE_ACCESS_TOKEN')
        if not token:
            logging.error("CONFLUENCE_ACCESS_TOKEN not found in environment variables")
            return False
            
        # Get authenticated client
        logging.info("Getting Confluence client...")
        confluence = get_confluence_client()
        
        if not confluence:
            logging.error("Failed to initialize Confluence client")
            return False
            
        # Test API by listing spaces
        logging.info("Listing Confluence spaces...")
        spaces = confluence.get_all_spaces(start=0, limit=10)
        
        if not spaces or 'results' not in spaces:
            logging.error("No spaces found or invalid response from Confluence API")
            return False
            
        space_count = len(spaces['results'])
        logging.info(f"Found {space_count} Confluence spaces")
        
        # List first few spaces
        for i, space in enumerate(spaces['results'][:3], 1):
            space_key = space.get('key', 'Unknown')
            space_name = space.get('name', 'Unknown')
            logging.info(f"Space {i}: {space_name} (Key: {space_key})")
            
        logging.info("Confluence API connection successful")
        return True
    except Exception as e:
        logging.error(f"Error testing Confluence connection: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing API connections...")
    
    print("\n=== Testing Pinecone Connection ===")
    pinecone_result = test_pinecone()
    print(f"Pinecone test {'PASSED' if pinecone_result else 'FAILED'}")
    
    print("\n=== Testing OpenAI Connection ===")
    openai_result = test_openai()
    print(f"OpenAI test {'PASSED' if openai_result else 'FAILED'}")
    
    print("\n=== Testing Confluence Connection ===")
    confluence_result = test_confluence()
    print(f"Confluence test {'PASSED' if confluence_result else 'FAILED'}")
    
    print("\n=== Summary ===")
    all_passed = pinecone_result and openai_result and confluence_result
    print(f"Pinecone: {'✓' if pinecone_result else '✗'}")
    print(f"OpenAI: {'✓' if openai_result else '✗'}")
    print(f"Confluence: {'✓' if confluence_result else '✗'}")
    print(f"Overall status: {'All tests PASSED' if all_passed else 'Some tests FAILED'}")
    
    sys.exit(0 if all_passed else 1) 