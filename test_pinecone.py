import os
import logging
from dotenv import load_dotenv
import pinecone
from openai import OpenAI
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
logging.info("Environment variables loaded")

def test_pinecone():
    # Get environment variables
    api_key = os.getenv('PINECONE_API_KEY')
    env = os.getenv('PINECONE_ENVIRONMENT') 
    index_name = os.getenv('PINECONE_INDEX_NAME')
    
    if not api_key or not env or not index_name:
        print("Missing Pinecone environment variables")
        return False
        
    # Initialize Pinecone
    print(f"Initializing Pinecone with env: {env}")
    try:
        pc = pinecone.Pinecone(api_key=api_key)
        indexes = pc.list_indexes()
        
        print(f"Available indexes: {indexes}")
        
        # Check if index exists - need to extract names from index objects
        index_names = [idx.name for idx in indexes]
        print(f"Index names: {index_names}")
        
        if index_name not in index_names:
            print(f"Index {index_name} not found in available indexes: {index_names}")
            return False
            
        # Connect to index
        index = pc.Index(index_name)
        
        # Get index stats
        stats = index.describe_index_stats()
        print(f"Index stats: {stats}")
        
        return index
    except Exception as e:
        print(f"Pinecone error: {str(e)}")
        return False

def test_openai():
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Missing OpenAI API key")
        return False
        
    try:
        # Create client
        client = OpenAI(api_key=api_key)
        
        # Test embedding 
        response = client.embeddings.create(
            input="Test query",
            model="text-embedding-ada-002"
        )
        
        embedding = response.data[0].embedding
        print(f"Generated embedding with dimension: {len(embedding)}")
        
        return embedding
    except Exception as e:
        print(f"OpenAI error: {str(e)}")
        return False

def test_query(index, embedding):
    if not index or not embedding:
        return False
        
    try:
        # Run query
        results = index.query(
            vector=embedding,
            top_k=3,
            include_metadata=True
        )
        
        # Check results
        matches = results.get('matches', [])
        print(f"Query returned {len(matches)} matches")
        
        if matches:
            print(f"Top match score: {matches[0].get('score')}")
            
        return results
    except Exception as e:
        print(f"Query error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Pinecone and OpenAI...")
    
    # Test Pinecone
    index = test_pinecone()
    if not index:
        print("Pinecone test failed")
        sys.exit(1)
        
    # Test OpenAI
    embedding = test_openai()
    if not embedding:
        print("OpenAI test failed")
        sys.exit(1)
        
    # Test query
    results = test_query(index, embedding)
    if not results:
        print("Query test failed")
        sys.exit(1)
        
    print("All tests passed successfully!") 