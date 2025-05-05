#!/usr/bin/env python
"""
Script to verify the AI Working Groups fix worked
"""
import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
import json

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

def verify_fix():
    """Verify that our AI Working Groups fix worked"""
    index_name = os.getenv("PINECONE_INDEX", "default-index")
    index = pc.Index(index_name)
    
    # Test different terminology variations
    queries = [
        "AI Working Groups",
        "AI Workgroups",
        "Tell me about the AI Workgroups. Is there a list?",
        "Tell me about the AI Working Groups. Is there a list?",
        "What are the AI Interest Group working groups?",
        "List the AI workgroups"
    ]
    
    print("Testing queries with both terminology variations")
    print("=" * 80)
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        
        # Generate embedding
        query_embedding = get_embedding(query)
        
        # Search in the index
        results = index.query(
            vector=query_embedding,
            top_k=3,
            namespace="all",
            include_metadata=True
        )
        
        print(f"Found {len(results.matches)} results")
        
        # Check for AI Working Groups page in top results
        ai_working_groups_found = False
        for i, match in enumerate(results.matches[:3]):
            print(f"  Result {i+1}: Score {match.score:.4f}")
            
            if hasattr(match, 'metadata') and match.metadata:
                title = match.metadata.get('title', 'No title')
                print(f"    Title: {title}")
                
                if "AI Working Groups" in title or "AI Workgroups" in title:
                    ai_working_groups_found = True
            
            print("  " + "-" * 40)
        
        if ai_working_groups_found:
            print("✅ AI Working Groups page found in top results!")
        else:
            print("❌ AI Working Groups page NOT found in top results!")
    
    # Test app's chat function directly
    try:
        from app_pinecone_openai import chat_function
        
        print("\nTesting chat_function directly")
        print("=" * 80)
        
        # Try a query that previously failed
        test_query = "Tell me about the AI Workgroups. Is there a list?"
        print(f"Query: '{test_query}'")
        
        # Run the chat function with empty history
        response = chat_function(test_query, [])
        
        # Check if any results were returned (response will be non-empty)
        if response and len(response) > 0 and len(response[0]) > 1:
            user_message, assistant_response = response[0]
            
            # Check if the response contains content (not just an error message)
            if "couldn't find any relevant information" not in assistant_response.lower():
                print("✅ chat_function successfully returned content!")
                print(f"Response length: {len(assistant_response)} characters")
            else:
                print("❌ chat_function still returned 'no relevant information'")
        else:
            print("❌ chat_function did not return a valid response")
            
    except Exception as e:
        print(f"Error testing chat_function: {e}")

if __name__ == "__main__":
    print("Verifying AI Working Groups fix")
    verify_fix() 