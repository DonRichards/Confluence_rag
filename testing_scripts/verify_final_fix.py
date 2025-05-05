#!/usr/bin/env python
"""
Final verification script to test AI Working Groups fixes
"""
import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
import json
from datetime import datetime, timedelta

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
    """Verify that our AI Working Groups fixes worked"""
    index_name = os.getenv("PINECONE_INDEX", "default-index")
    index = pc.Index(index_name)
    
    # Test different terminology variations and time-filtered queries
    queries = [
        "AI Working Groups",
        "AI Workgroups",
        "Tell me about the AI Workgroups. Is there a list?",
        "Tell me about the AI Working Groups this week",
        "What are the AI Interest Group working groups this week?",
        "List the AI workgroups from this week"
    ]
    
    print("Testing queries with both terminology variations and time filtering")
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
        recent_date_found = False
        
        for i, match in enumerate(results.matches[:3]):
            print(f"  Result {i+1}: Score {match.score:.4f}")
            
            if hasattr(match, 'metadata') and match.metadata:
                title = match.metadata.get('title', 'No title')
                last_modified = match.metadata.get('last_modified', 'No date')
                print(f"    Title: {title}")
                print(f"    Last Modified: {last_modified}")
                
                if "AI Working Groups" in title or "AI Workgroups" in title:
                    ai_working_groups_found = True
                
                # Check if date is within last week
                try:
                    from dateutil import parser
                    date_obj = parser.parse(last_modified)
                    one_week_ago = datetime.now() - timedelta(days=7)
                    if date_obj > one_week_ago:
                        recent_date_found = True
                except Exception:
                    pass
            
            print("  " + "-" * 40)
        
        if ai_working_groups_found:
            print("✅ AI Working Groups page found in top results!")
        else:
            print("❌ AI Working Groups page NOT found in top results!")
            
        if recent_date_found:
            print("✅ Recent dates (within last week) found!")
        else:
            print("❌ Recent dates NOT found!")
    
    # Test app's chat function directly with time filter
    try:
        print("\nTesting app_pinecone_openai.py with time filtering")
        print("=" * 80)
        
        # Test the app directly
        from app_pinecone_openai import chat_function
        
        # Set global variables needed for testing
        import app_pinecone_openai
        app_pinecone_openai.current_time_filter = "this week"
        app_pinecone_openai.current_space = None
        
        # Try time-filtered query
        test_query = "Tell me about the AI Workgroups this week. Is there a list?"
        print(f"Query with time filter: '{test_query}'")
        
        # Run the chat function with empty history
        response = chat_function(test_query, [])
        
        # Check if any results were returned
        if response and len(response) > 0 and len(response[0]) > 1:
            user_message, assistant_response = response[0]
            
            # Check if the response contains content (not just an error message)
            if len(assistant_response) > 100 and "couldn't find any relevant information" not in assistant_response.lower():
                print("✅ Time-filtered query successfully returned content!")
                print(f"Response snippet: {assistant_response[:100]}...")
            else:
                print("❌ Time-filtered query failed or returned 'no relevant information'")
                if assistant_response:
                    print(f"Response: {assistant_response[:100]}...")
        else:
            print("❌ chat_function did not return a valid response")
            
    except Exception as e:
        print(f"Error testing app with time filter: {e}")

if __name__ == "__main__":
    print("Running final verification of AI Working Groups fixes")
    verify_fix() 