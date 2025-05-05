#!/usr/bin/env python3
"""
Test script to check if date information is properly stored in Pinecone vectors.
This script will examine vectors in the Pinecone index to see if last_modified dates are present.
"""

import os
from dotenv import load_dotenv
import json
from utils.pinecone_logic import init_pinecone
from utils.openai_logic import get_embeddings

# Load environment variables
load_dotenv()

def test_date_metadata():
    """Test if date information is available in Pinecone vectors"""
    print("\n=== TESTING DATE METADATA IN VECTORS ===")
    
    # Initialize Pinecone
    index = init_pinecone()
    if not index:
        print("Failed to initialize Pinecone")
        return
    
    # Get stats to check namespaces
    try:
        stats = index.describe_index_stats()
        
        # Print index stats
        print(f"Total vectors in index: {stats.get('total_vector_count', 'unknown')}")
        
        # Convert to namespaces dictionary based on the return type
        namespaces = {}
        if hasattr(stats, 'namespaces'):
            namespaces = stats.namespaces
        elif 'namespaces' in stats:
            namespaces = stats.get('namespaces', {})
            
        print(f"Namespaces: {list(namespaces.keys())}")
        
        # Check each namespace
        for namespace, ns_stats in namespaces.items():
            print(f"\nChecking namespace: {namespace}")
            print(f"Vector count: {ns_stats.get('vector_count', 'unknown')}")
            
            # Skip empty namespaces
            if ns_stats.get('vector_count', 0) == 0:
                print("Namespace is empty, skipping")
                continue
                
            # Query a few vectors to check metadata
            try:
                # Generate a generic embedding for querying
                embedding_response = get_embeddings("test", "text-embedding-ada-002")
                query_embedding = embedding_response.data[0].embedding
                
                # Query some vectors
                results = index.query(
                    vector=query_embedding,
                    top_k=10,
                    include_metadata=True,
                    namespace=namespace
                )
                
                matches = results.get('matches', [])
                print(f"Retrieved {len(matches)} vectors for inspection")
                
                # Check for date information
                date_count = 0
                for i, match in enumerate(matches):
                    metadata = match.get('metadata', {})
                    vector_id = match.get('id', 'unknown')
                    
                    # Check if last_modified exists
                    if 'last_modified' in metadata and metadata['last_modified']:
                        date_count += 1
                        print(f"Vector {i+1} (ID: {vector_id}) has last_modified: {metadata['last_modified']}")
                    else:
                        print(f"Vector {i+1} (ID: {vector_id}) does NOT have last_modified")
                        print(f"  Metadata keys: {list(metadata.keys())}")
                        
                print(f"\nSummary for namespace {namespace}:")
                print(f"Vectors with date information: {date_count}/{len(matches)} ({date_count/len(matches)*100 if matches else 0:.1f}%)")
                
                # If no dates found, try to fetch specific vector data for deeper inspection
                if date_count == 0 and matches:
                    print("\nInspecting first vector in more detail:")
                    vector_id = matches[0].get('id', 'unknown')
                    vector_data = index.fetch(ids=[vector_id], namespace=namespace)
                    
                    if 'vectors' in vector_data and vector_id in vector_data['vectors']:
                        full_metadata = vector_data['vectors'][vector_id].get('metadata', {})
                        print(f"Full metadata for vector {vector_id}:")
                        for key, value in full_metadata.items():
                            print(f"  {key}: {type(value).__name__} = {value if len(str(value)) < 50 else str(value)[:50]+'...'}")
            
            except Exception as e:
                print(f"Error inspecting namespace {namespace}: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"Error getting index stats: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    test_date_metadata()
    print("\nDate metadata test complete") 