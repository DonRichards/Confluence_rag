#!/usr/bin/env python3

import os
import sys

# Add parent directory to path to import from project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pinecone_logic import query_pinecone, init_pinecone
from utils.openai_logic import get_embeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_query(query_text, similarity_threshold=0.7, namespace=None):
    """
    Test querying Pinecone with different configurations.
    
    Args:
        query_text: Text to query
        similarity_threshold: Score threshold to filter results (default: 0.7)
        namespace: Optional namespace to query (default: None, which will use all)
    """
    try:
        print(f"\n=== Testing Query ===")
        print(f"Query: '{query_text}'")
        print(f"Similarity threshold: {similarity_threshold}")
        print(f"Namespace: {namespace if namespace else 'all'}")
        
        # Initialize Pinecone
        index = init_pinecone()
        if not index:
            print("Failed to initialize Pinecone")
            return
        
        # Get index stats
        stats = index.describe_index_stats()
        print(f"\nTotal vectors in index: {stats.total_vector_count}")
        
        # Print namespaces
        print("\nNamespaces:")
        for ns, count in stats.namespaces.items():
            print(f"  - {ns}: {count} vectors")
        
        # Query Pinecone
        print("\nQuerying Pinecone...")
        results = query_pinecone(
            index, 
            query_text, 
            top_k=5, 
            filter_by_space=namespace,
            similarity_threshold=similarity_threshold
        )
        
        # Print results
        print(f"\nResults: {len(results)} matches")
        for i, result in enumerate(results):
            if len(result) >= 4:
                source, text, score, date_info = result
                print(f"\nResult {i+1}:")
                print(f"Score: {score:.4f}")
                print(f"Date: {date_info}")
                print(f"Source: {source}")
                print(f"Text snippet: {text[:200]}...")
            else:
                source, text, score = result
                print(f"\nResult {i+1}:")
                print(f"Score: {score:.4f}")
                print(f"Source: {source}")
                print(f"Text snippet: {text[:200]}...")
        
        # If no results, try with a lower threshold
        if not results and similarity_threshold > 0.1:
            print("\n=== No results found. Trying with lower threshold ===")
            lower_threshold = max(0.1, similarity_threshold / 2)
            print(f"New threshold: {lower_threshold}")
            
            lower_results = query_pinecone(
                index, 
                query_text, 
                top_k=5, 
                filter_by_space=namespace,
                similarity_threshold=lower_threshold
            )
            
            print(f"\nResults with lower threshold: {len(lower_results)} matches")
            for i, result in enumerate(lower_results):
                if len(result) >= 4:
                    source, text, score, date_info = result
                    print(f"\nResult {i+1}:")
                    print(f"Score: {score:.4f}")
                    print(f"Date: {date_info}")
                    print(f"Source: {source}")
                    print(f"Text snippet: {text[:200]}...")
                else:
                    source, text, score = result
                    print(f"\nResult {i+1}:")
                    print(f"Score: {score:.4f}")
                    print(f"Source: {source}")
                    print(f"Text snippet: {text[:200]}...")
        
    except Exception as e:
        print(f"Error testing query: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Handle command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Test Pinecone query functionality')
    parser.add_argument('query', nargs='?', default="What are the guidelines for AI resource usage?", 
                        help='Query text to search for')
    parser.add_argument('threshold', nargs='?', type=float, default=0.5, 
                        help='Similarity threshold (default: 0.5)')
    parser.add_argument('namespace', nargs='?', default=None, 
                        help='Optional namespace to filter results')
    
    args = parser.parse_args()
    
    # Run test with provided arguments
    test_query(args.query, args.threshold, args.namespace) 