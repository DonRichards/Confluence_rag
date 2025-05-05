#!/usr/bin/env python3

import os
import sys

# Add parent directory to path to import from project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

def check_index_stats():
    """Check and display Pinecone index statistics"""
    try:
        # Get the index name from environment variable
        index_name = os.getenv("PINECONE_INDEX_NAME", "default-index")
        print(f"Checking stats for index: {index_name}")
        
        # Use the same pinecone_logic helper to avoid import issues
        from utils.pinecone_logic import init_pinecone
        
        # Initialize index using utility function
        index = init_pinecone()
        if not index:
            print("Failed to initialize Pinecone index")
            return False
        
        # Get index statistics
        stats = index.describe_index_stats()
        
        # Print the stats in a readable format
        print("\nIndex Statistics:")
        print(f"Dimension: {stats.dimension}")
        print(f"Index fullness: {stats.index_fullness}")
        print(f"Total vector count: {stats.total_vector_count}")
        
        # Print namespaces
        print("\nNamespaces:")
        for namespace, vector_count in stats.namespaces.items():
            print(f"  - {namespace}: {vector_count} vectors")
        
        # Print additional information if available
        if hasattr(stats, 'index_config'):
            print("\nIndex Configuration:")
            print(json.dumps(stats.index_config, indent=2))
            
        return True
    except Exception as e:
        print(f"Error checking index stats: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    check_index_stats() 