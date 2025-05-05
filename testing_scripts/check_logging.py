#!/usr/bin/env python3

import os
import sys
import json
from datetime import datetime

# Add parent directory to path to import from project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pinecone_logic import init_pinecone, query_pinecone
from app_pinecone_openai import log_conversation

# Set up log directory
LOGS_DIR = "test_logs"
os.makedirs(LOGS_DIR, exist_ok=True)

def test_log_conversation():
    """Test the log_conversation function with different kinds of query results"""
    
    print("Testing log_conversation function...")
    
    # Initialize Pinecone
    print("Initializing Pinecone...")
    index = init_pinecone()
    if not index:
        print("Failed to initialize Pinecone")
        return
    
    # Query for results that might include date information
    test_query = "What are the guidelines for AI resource usage?"
    print(f"Querying Pinecone with: '{test_query}'")
    query_results = query_pinecone(
        index, 
        test_query, 
        top_k=3,
        filter_by_space=None,
        similarity_threshold=0.3
    )
    
    # Generate a test session ID
    session_id = "test_session_" + datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Log the conversation with the query results
    print("Logging conversation with query results...")
    log_conversation(
        user_input=test_query,
        assistant_response="Test response",
        session_id=session_id,
        history=None,
        results=query_results
    )
    
    # Check the logged results
    log_file = os.path.join(LOGS_DIR, f"conversation_{session_id}.jsonl")
    print(f"Checking log file: {log_file}")
    
    try:
        with open(log_file, "r") as f:
            log_entry = json.loads(f.readline().strip())
            
        # Verify the log entry has the correct structure
        print("\nLog entry structure:")
        print(f"- timestamp: {type(log_entry.get('timestamp', None))}")
        print(f"- session_id: {log_entry.get('session_id', None)}")
        print(f"- user_input: {log_entry.get('user_input', None)}")
        print(f"- assistant_response: {log_entry.get('assistant_response', None)}")
        
        # Verify search results structure
        search_results = log_entry.get('search_results', [])
        print(f"\nNumber of search results: {len(search_results)}")
        
        for i, result in enumerate(search_results):
            print(f"\nResult {i+1}:")
            print(f"- source: {result.get('source', None)[:50]}...")
            print(f"- score: {result.get('score', None)}")
            if 'date' in result:
                print(f"- date: {result.get('date', None)}")
        
        # Verify conversation history structure
        conversation_history = log_entry.get('conversation_history', [])
        print(f"\nConversation history entries: {len(conversation_history)}")
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error testing log_conversation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Override LOGS_DIR for testing
    import app_pinecone_openai
    app_pinecone_openai.LOGS_DIR = LOGS_DIR
    
    test_log_conversation() 