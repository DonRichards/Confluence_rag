import os
import json
import argparse
from utils.pinecone_logic import init_pinecone, query_pinecone
from utils.openai_logic import get_embeddings

def test_namespaces():
    """Test queries across different namespaces"""
    print("\n=== TESTING DIFFERENT NAMESPACES ===")
    
    # Initialize Pinecone
    index = init_pinecone()
    
    print('Testing query with different namespaces...')
    for ns in ['DIAS', 'AIG', 'all', None]:
        print(f'\nTesting namespace {ns}:')
        results = query_pinecone(index, 'AI Working Groups', filter_by_space=ns, similarity_threshold=0.0)
        print(f'Results count: {len(results)}')
        if results:
            print(f'First result score: {results[0][2]}')
            print(f'First result source: {results[0][0]}')
            print(f'First result text preview: {results[0][1][:150]}...')

def test_query_variations():
    """Test different query variations"""
    print("\n=== TESTING QUERY VARIATIONS ===")
    
    # Initialize Pinecone
    index = init_pinecone()
    
    queries = [
        "AI Working Groups",
        "artificial intelligence working groups",
        "teams working on AI",
        "AI initiatives",
        "machine learning projects",
        "data science teams"
    ]
    
    for query in queries:
        print(f'\nTesting query: "{query}"')
        results = query_pinecone(index, query, filter_by_space='all', similarity_threshold=0.0)
        print(f'Results count: {len(results)}')
        if results:
            for i, (source, text, score) in enumerate(results[:3]):  # Show top 3 results
                print(f'Result {i+1} - Score: {score:.4f}')
                print(f'Source: {source}')
                print(f'Text snippet: {text[:100]}...\n')

def test_direct_query():
    """Test direct querying of Pinecone with raw embeddings"""
    print("\n=== TESTING DIRECT PINECONE QUERYING ===")
    
    # Initialize Pinecone
    index = init_pinecone()
    
    # Get stats to check namespaces
    stats = index.describe_index_stats()
    if hasattr(stats, 'namespaces'):
        namespaces = list(stats.namespaces.keys())
    elif 'namespaces' in stats:
        namespaces = list(stats.get('namespaces', {}).keys())
    else:
        namespaces = ['']
        
    print(f'Available namespaces: {namespaces}')
    
    # Test queries
    test_queries = ["AI Working Groups", "machine learning", "data"]
    
    for query in test_queries:
        print(f'\nTesting direct query: "{query}"')
        
        # Generate embedding for the query
        embedding_response = get_embeddings(query, "text-embedding-ada-002")
        query_embedding = embedding_response.data[0].embedding
        
        # Test with different namespaces
        for namespace in ['all'] + [ns for ns in namespaces if ns != 'all'][:3]:  # Test 'all' plus up to 3 other namespaces
            print(f'  Namespace: {namespace}')
            
            # Direct Pinecone query
            results = index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True,
                namespace=namespace
            )
            
            matches = results.get('matches', [])
            print(f'  Found {len(matches)} matches')
            
            # Show details of matches
            for i, match in enumerate(matches[:2]):  # Show top 2 matches
                score = match.get('score', 0)
                metadata = match.get('metadata', {})
                print(f'    Match {i+1} - Score: {score:.4f}')
                print(f'    ID: {match.get("id", "N/A")}')
                print(f'    Space: {metadata.get("space", "N/A")}')
                if 'source' in metadata:
                    print(f'    Source: {metadata["source"]}')
                if 'text' in metadata:
                    print(f'    Text snippet: {metadata["text"][:100]}...\n')

def test_metadata_filters():
    """Test querying with metadata filters"""
    print("\n=== TESTING METADATA FILTERS ===")
    
    # Initialize Pinecone
    index = init_pinecone()
    
    # Get a sample query embedding
    embedding_response = get_embeddings("AI", "text-embedding-ada-002")
    query_embedding = embedding_response.data[0].embedding
    
    # Try different metadata filters
    print("\nChecking metadata structure in first few vectors...")
    try:
        # Get some vector IDs first
        initial_results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            namespace='all'
        )
        
        # Get one vector ID to check its metadata structure
        if initial_results and 'matches' in initial_results and initial_results['matches']:
            vector_id = initial_results['matches'][0]['id']
            vector_data = index.fetch(ids=[vector_id], namespace='all')
            
            if 'vectors' in vector_data and vector_id in vector_data['vectors']:
                metadata = vector_data['vectors'][vector_id].get('metadata', {})
                print(f"Sample metadata structure for vector {vector_id}:")
                for key, value in metadata.items():
                    print(f"  {key}: {type(value).__name__} = {value if len(str(value)) < 50 else str(value)[:50]+'...'}")
                
                # Try querying with metadata filters based on what we found
                if 'space' in metadata:
                    print("\nTesting filter on 'space' field...")
                    space_value = metadata['space']
                    filter_query = {"space": {"$eq": space_value}}
                    
                    results = index.query(
                        vector=query_embedding,
                        top_k=5,
                        include_metadata=True,
                        namespace='all',
                        filter=filter_query
                    )
                    
                    matches = results.get('matches', [])
                    print(f"Found {len(matches)} matches with space={space_value}")
                
                # Try other metadata fields
                for key in metadata.keys():
                    if key != 'space' and key != 'text' and not isinstance(metadata[key], (list, dict)):
                        print(f"\nTesting filter on '{key}' field...")
                        field_value = metadata[key]
                        filter_query = {key: {"$eq": field_value}}
                        
                        results = index.query(
                            vector=query_embedding,
                            top_k=5,
                            include_metadata=True,
                            namespace='all',
                            filter=filter_query
                        )
                        
                        matches = results.get('matches', [])
                        print(f"Found {len(matches)} matches with {key}={field_value}")
            else:
                print("Could not fetch vector data for metadata analysis")
        else:
            print("No initial results to check metadata structure")
    
    except Exception as e:
        print(f"Error testing metadata filters: {e}")
        import traceback
        traceback.print_exc()

def examine_vector_metadata():
    """Examine metadata of vectors in the index"""
    print("\n=== EXAMINING VECTOR METADATA IN DETAIL ===")
    
    # Initialize Pinecone
    index = init_pinecone()
    
    # Get stats to check namespaces
    stats = index.describe_index_stats()
    if hasattr(stats, 'namespaces'):
        namespaces = list(stats.namespaces.keys())
    elif 'namespaces' in stats:
        namespaces = list(stats.get('namespaces', {}).keys())
    else:
        namespaces = ['']
    
    # Focus on the 'all' namespace
    namespace = 'all' if 'all' in namespaces else namespaces[0]
    print(f"Examining vectors in namespace: {namespace}")
    
    # Get some vectors for examination
    embedding_response = get_embeddings("AI", "text-embedding-ada-002")
    query_embedding = embedding_response.data[0].embedding
    
    results = index.query(
        vector=query_embedding,
        top_k=10,
        include_metadata=True,
        namespace=namespace
    )
    
    matches = results.get('matches', [])
    print(f"Found {len(matches)} vectors for examination")
    
    # Check each vector's metadata in detail
    for i, match in enumerate(matches[:5]):  # Look at first 5 matches
        vector_id = match.get('id', 'unknown')
        score = match.get('score', 0)
        metadata = match.get('metadata', {})
        
        print(f"\nVector {i+1} (ID: {vector_id}, Score: {score:.4f}):")
        print(f"  Metadata keys: {list(metadata.keys())}")
        
        # Check critical fields
        source = metadata.get('source', 'No source available')
        text = metadata.get('text', 'No content available')
        space = metadata.get('space', 'No space info')
        
        print(f"  source: {source[:100] + '...' if len(str(source)) > 100 else source}")
        print(f"  space: {space}")
        print(f"  text length: {len(text)} chars")
        if len(text) > 0:
            print(f"  text sample: {text[:150]}...")
        
        # Try to fetch actual vector data for more details
        try:
            vector_data = index.fetch(ids=[vector_id], namespace=namespace)
            if 'vectors' in vector_data and vector_id in vector_data['vectors']:
                full_metadata = vector_data['vectors'][vector_id].get('metadata', {})
                if full_metadata != metadata:
                    print("  Additional metadata from fetch:")
                    for key, value in full_metadata.items():
                        if key not in metadata:
                            print(f"    {key}: {str(value)[:100] + '...' if len(str(value)) > 100 else value}")
        except Exception as e:
            print(f"  Error fetching vector data: {e}")
    
    # Check if source and text are consistently missing
    print("\nAnalyzing metadata issues across more vectors...")
    results = index.query(
        vector=query_embedding,
        top_k=50,  # Check more vectors
        include_metadata=True,
        namespace=namespace
    )
    
    matches = results.get('matches', [])
    missing_source_count = 0
    missing_text_count = 0
    missing_space_count = 0
    has_source_count = 0
    has_text_count = 0
    
    for match in matches:
        metadata = match.get('metadata', {})
        if metadata.get('source', 'No source available') == 'No source available':
            missing_source_count += 1
        else:
            has_source_count += 1
            
        if metadata.get('text', 'No content available') == 'No content available':
            missing_text_count += 1
        else:
            has_text_count += 1
            
        if 'space' not in metadata:
            missing_space_count += 1
    
    print(f"\nOut of {len(matches)} vectors examined:")
    print(f"  Missing source: {missing_source_count} ({missing_source_count/len(matches)*100:.1f}%)")
    print(f"  Has source: {has_source_count} ({has_source_count/len(matches)*100:.1f}%)")
    print(f"  Missing text: {missing_text_count} ({missing_text_count/len(matches)*100:.1f}%)")
    print(f"  Has text: {has_text_count} ({has_text_count/len(matches)*100:.1f}%)")
    print(f"  Missing space field: {missing_space_count} ({missing_space_count/len(matches)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Test Pinecone queries and debug RAG system')
    parser.add_argument('--test-all', action='store_true', help='Run all tests')
    parser.add_argument('--test-namespaces', action='store_true', help='Test queries across different namespaces')
    parser.add_argument('--test-queries', action='store_true', help='Test different query variations')
    parser.add_argument('--test-direct', action='store_true', help='Test direct Pinecone querying')
    parser.add_argument('--test-filters', action='store_true', help='Test metadata filters')
    parser.add_argument('--check-metadata', action='store_true', help='Examine vector metadata in detail')
    parser.add_argument('--query', type=str, help='Custom query to test')
    
    args = parser.parse_args()
    
    # If no specific test is selected, run all tests
    run_all = args.test_all or not (args.test_namespaces or args.test_queries or args.test_direct or args.test_filters or args.check_metadata or args.query)
    
    print("=== PINECONE RAG SYSTEM DIAGNOSTIC TOOL ===")
    
    if args.query:
        print(f"\n=== TESTING CUSTOM QUERY: '{args.query}' ===")
        index = init_pinecone()
        
        # Test across all namespaces
        print("\nTesting across all namespaces:")
        stats = index.describe_index_stats()
        if hasattr(stats, 'namespaces'):
            namespaces = list(stats.namespaces.keys())
        elif 'namespaces' in stats:
            namespaces = list(stats.get('namespaces', {}).keys())
        else:
            namespaces = ['']
        
        for ns in namespaces:
            print(f"\nNamespace: {ns}")
            results = query_pinecone(index, args.query, filter_by_space=ns, top_k=5, similarity_threshold=0.0)
            print(f"Found {len(results)} results")
            
            for i, (source, text, score) in enumerate(results[:3]):
                print(f"Result {i+1} - Score: {score:.4f}")
                print(f"Source: {source}")
                print(f"Text snippet: {text[:150]}...\n")
    
    if run_all or args.test_namespaces:
        test_namespaces()
    
    if run_all or args.test_queries:
        test_query_variations()
    
    if run_all or args.test_direct:
        test_direct_query()
    
    if run_all or args.test_filters:
        test_metadata_filters()
        
    if run_all or args.check_metadata:
        examine_vector_metadata()
    
    print("\n=== DIAGNOSTIC COMPLETE ===")

if __name__ == "__main__":
    main() 