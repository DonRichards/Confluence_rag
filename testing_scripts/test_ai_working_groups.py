#!/usr/bin/env python
"""
Test script to diagnose AI Working Groups retrieval issues
"""
import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()
print("Loading .env environment variables...")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Initialize OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text):
    """Get embedding for text using OpenAI API"""
    response = openai_client.embeddings.create(
        input=text,
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    )
    return response.data[0].embedding

def test_variations():
    """Test various query variations to see what works"""
    index_name = os.getenv("PINECONE_INDEX", "default-index")
    index = pc.Index(index_name)
    
    # Test different query variations
    queries = [
        "AI Working Groups",
        "AI Workgroups",
        "AI Interest Group working groups",
        "AI Working Groups list",
        "Library Staff AI Training Group",
        "JHU Libraries AI groups"
    ]
    
    # Parameters to test
    top_k = 10
    namespace = "all"  # Try different namespaces if needed
    
    print(f"Testing queries against index: {index_name}, namespace: {namespace}")
    print("-" * 80)
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        query_embedding = get_embedding(query)
        
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
        
        print(f"Found {len(results.matches)} results")
        
        for i, match in enumerate(results.matches[:3]):  # Show top 3
            print(f"  Result {i+1}: Score {match.score:.4f}")
            print(f"    ID: {match.id}")
            if hasattr(match, 'metadata') and match.metadata:
                title = match.metadata.get('title', 'No title')
                url = match.metadata.get('url', 'No URL')
                print(f"    Title: {title}")
                print(f"    URL: {url}")
            print("  " + "-" * 40)

def test_namespaces():
    """Test which namespaces contain the data"""
    index_name = os.getenv("PINECONE_INDEX", "default-index")
    index = pc.Index(index_name)
    
    # Get available namespaces
    stats = index.describe_index_stats()
    namespaces = stats.namespaces
    
    print("Available namespaces:")
    for ns, count in namespaces.items():
        print(f"  {ns}: {count} vectors")
    
    # Use a targeted query
    query = "AI Working Groups"
    query_embedding = get_embedding(query)
    
    print(f"\nSearching for '{query}' across all namespaces:")
    for namespace in namespaces:
        results = index.query(
            vector=query_embedding,
            top_k=3,
            namespace=namespace,
            include_metadata=True
        )
        
        print(f"\nNamespace: {namespace} - Found {len(results.matches)} results")
        for i, match in enumerate(results.matches[:1]):  # Show top result only
            print(f"  Top result: Score {match.score:.4f}")
            if hasattr(match, 'metadata') and match.metadata:
                title = match.metadata.get('title', 'No title')
                print(f"    Title: {title}")

if __name__ == "__main__":
    print("Testing AI Working Groups retrieval")
    print("=" * 80)
    test_variations()
    print("\n" + "=" * 80)
    test_namespaces() 