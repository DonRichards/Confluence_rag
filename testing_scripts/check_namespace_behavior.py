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

def test_namespace_behavior():
    """
    Test Pinecone query behavior with different namespace configurations
    to confirm our fix addresses the issue where queries weren't returning results.
    """
    try:
        print("\n=== Testing Namespace Behavior in Queries ===")
        
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
        
        # Test queries
        test_query = "What is the AI resources policy?"
        
        # Test 1: Query with no namespace (should search all namespaces)
        print("\nTest 1: Query with no namespace specified")
        results1 = query_pinecone(
            index, 
            test_query, 
            top_k=3,
            filter_by_space=None,
            similarity_threshold=0.3
        )
        
        print(f"Results found: {len(results1)}")
        
        # Test 2: Query with 'all' namespace (should use the all namespace)
        print("\nTest 2: Query with 'all' namespace specified")
        results2 = query_pinecone(
            index, 
            test_query, 
            top_k=3,
            filter_by_space='all',
            similarity_threshold=0.3
        )
        
        print(f"Results found: {len(results2)}")
        
        # Test 3: Query with specific namespace
        print("\nTest 3: Query with specific namespace (use first available)")
        specific_namespace = next(iter(stats.namespaces))
        results3 = query_pinecone(
            index, 
            test_query, 
            top_k=3,
            filter_by_space=specific_namespace,
            similarity_threshold=0.3
        )
        
        print(f"Results found with namespace '{specific_namespace}': {len(results3)}")
        
        # Test 4: Query with low similarity threshold
        print("\nTest 4: Query with low similarity threshold (0.1)")
        results4 = query_pinecone(
            index, 
            test_query, 
            top_k=3,
            filter_by_space=None,
            similarity_threshold=0.1
        )
        
        print(f"Results found: {len(results4)}")
        
        # Test 5: Query with adaptive threshold
        print("\nTest 5: Query with adaptive threshold (starts at 0.5, falls back to lower)")
        results5 = query_pinecone(
            index, 
            test_query, 
            top_k=3,
            filter_by_space=None,
            similarity_threshold=0.5
        )
        
        print(f"Results found: {len(results5)}")
        
        # Compare the different approaches
        print("\nSummary of tests:")
        print(f"Test 1 (no namespace): {len(results1)} results")
        print(f"Test 2 ('all' namespace): {len(results2)} results")
        print(f"Test 3 (specific namespace '{specific_namespace}'): {len(results3)} results")
        print(f"Test 4 (low threshold): {len(results4)} results")
        print(f"Test 5 (adaptive threshold): {len(results5)} results")
        
    except Exception as e:
        print(f"Error testing namespace behavior: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_namespace_behavior() 