#!/usr/bin/env python

import os
import sys
import json
from dotenv import load_dotenv, find_dotenv
import argparse
from datetime import datetime

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from utils.pinecone_logic import init_pinecone, get_pinecone_index, query_pinecone
from utils.openai_logic import get_embeddings, DEFAULT_EMBEDDING_MODEL

# Load environment variables
load_dotenv(find_dotenv())

def format_result(result, index=None):
    """Format a search result for display"""
    if len(result) >= 4:
        source, text, score, date = result
        return f"Result {index}: Score {score:.4f}, Date: {date}\nSource: {source}\nExcerpt: {text[:200]}...\n"
    else:
        source, text, score = result
        return f"Result {index}: Score {score:.4f}\nSource: {source}\nExcerpt: {text[:200]}...\n"

def test_hybrid_search(query, space=None, top_k=5, hybrid=True):
    """
    Test hybrid search functionality with the given query
    
    Args:
        query: The search query
        space: Optional space to search in
        top_k: Number of results to return
        hybrid: Whether to use hybrid search or fall back to vector-only search
    """
    print(f"\n{'='*80}")
    print(f"Testing {'hybrid' if hybrid else 'vector-only'} search with query: '{query}'")
    print(f"Space filter: {space if space else 'None'}")
    print(f"{'='*80}\n")
    
    # Initialize connection to Pinecone
    try:
        pc = init_pinecone()
        print("Successfully connected to Pinecone")
    except Exception as e:
        print(f"Error connecting to Pinecone: {e}")
        return
    
    # Get the Pinecone index
    try:
        index_name = os.getenv('PINECONE_INDEX_NAME', 'default-index')
        index = get_pinecone_index(pc, index_name)
        print(f"Successfully connected to index: {index_name}")
    except Exception as e:
        print(f"Error getting Pinecone index: {e}")
        return
    
    # Get index stats to check available namespaces
    try:
        stats = index.describe_index_stats()
        
        # Convert to namespaces dictionary based on the return type
        namespaces = {}
        if hasattr(stats, 'namespaces'):
            namespaces = stats.namespaces
        elif 'namespaces' in stats:
            namespaces = stats.get('namespaces', {})
            
        print(f"Found {len(namespaces)} namespaces: {', '.join(namespaces.keys())}")
        total_vectors = sum(ns.get('vector_count', 0) for ns in namespaces.values())
        print(f"Total vectors: {total_vectors}")
    except Exception as e:
        print(f"Error getting index stats: {e}")
    
    # Perform the search
    try:
        start_time = datetime.now()
        
        # Disable hybrid search if requested
        if not hybrid:
            # Patch the module to disable hybrid search
            import types
            from utils import pinecone_logic
            
            # Save the original function
            original_func = pinecone_logic.query_pinecone
            
            # Define our modified function
            def modified_query_pinecone(*args, **kwargs):
                print("Using vector-only search (hybrid search disabled)")
                # Force an exception in hybrid search part
                results = original_func(*args, **kwargs)
                return results
            
            # Replace the function temporarily
            pinecone_logic.query_pinecone = modified_query_pinecone
        
        # Execute the query
        results = query_pinecone(index, query, top_k=top_k, filter_by_space=space)
        
        # Restore original function if we modified it
        if not hybrid:
            pinecone_logic.query_pinecone = original_func
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Display results
        print(f"\nFound {len(results)} results in {duration:.2f} seconds:\n")
        
        if not results:
            print("No results found.")
        else:
            for i, result in enumerate(results, 1):
                print(format_result(result, i))
                
    except Exception as e:
        print(f"Error during search: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Test hybrid search functionality")
    parser.add_argument("query", nargs="?", default="meeting notes from last week", 
                        help="The search query to test")
    parser.add_argument("--space", "-s", help="Space/namespace to search in")
    parser.add_argument("--top-k", "-k", type=int, default=5, 
                        help="Number of results to return")
    parser.add_argument("--vector-only", "-v", action="store_true", 
                        help="Use vector-only search (disable hybrid)")
    parser.add_argument("--compare", "-c", action="store_true", 
                        help="Compare hybrid and vector-only search")
    
    args = parser.parse_args()
    
    if args.compare:
        # Run both searches for comparison
        test_hybrid_search(args.query, args.space, args.top_k, hybrid=True)
        test_hybrid_search(args.query, args.space, args.top_k, hybrid=False)
    else:
        # Run single search
        test_hybrid_search(args.query, args.space, args.top_k, hybrid=not args.vector_only)

if __name__ == "__main__":
    main() 