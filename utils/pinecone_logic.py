import os
import pinecone
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
import ast

#Global variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pinecone = Pinecone(api_key=PINECONE_API_KEY)
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_DB_NAME = os.getenv("PINECONE_DB_NAME")

# Initialize Pinecone connection and return index
def init_pinecone():
    """Initialize the Pinecone connection and return the index object."""
    try:
        index_name = 'default-index'
        index, _ = get_pinecone_index(index_name)
        return index
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        return None

# delete index
def delete_pinecone_index(index_name):
    print(f"Deleting index '{index_name}' if it exists.")
    try:
        if index_name in [index.name for index in pinecone.list_indexes()]:
            print(f"   Index '{index_name}' found, deleting...")
            pinecone.delete_index(index_name)
            if index_name in [index.name for index in pinecone.list_indexes()]:
                print(f"   ERROR: Index '{index_name}' was not deleted.")
                exit(1)
            else:
                print(f"   Index '{index_name}' successfully deleted.")
        else:
            print(f"   Index '{index_name}' not found no action taken.")        
    except Exception as e:
        print(f"   Index '{index_name}' not found no action taken. {e}")
        exit(1)

# create index if needed
def get_pinecone_index(index_name):
    print(f"Checking if index {index_name} exists.")
    index_created = False
    if index_name in [index.name for index in pinecone.list_indexes()]:
        print(f"Index {index_name} already exists, good to go.")
        index = pinecone.Index(index_name)
    else:
        print(f"Index {index_name} does not exist, need to create it.")
        index_created = True
        pinecone.create_index(
            name=index_name, 
            dimension=1536, 
            metric='cosine', 
            spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT))
            
        print(f"Index {index_name} created.")

        index = pinecone.Index(index_name)
    return index, index_created


def validate_namespace_count(index, expected_spaces=None):
    """
    Validate that the index has the expected number of namespaces.
    
    Args:
        index: The Pinecone index to verify
        expected_spaces: Optional list of expected space names. If None, will read from SPACES env var
        
    Returns:
        tuple: (bool, dict) - Success status and dict with details about namespace counts
    """
    try:
        # Get configured spaces from environment if not provided
        if expected_spaces is None:
            expected_spaces = os.getenv('SPACES', '').split(',')
            expected_spaces = [space.strip() for space in expected_spaces if space.strip()]
        
        # Add 1 for the default namespace if no spaces configured
        expected_count = len(expected_spaces) if expected_spaces else 1
        
        # Get actual namespace count from index
        stats = index.describe_index_stats()
        actual_namespaces = []
        if hasattr(stats, 'namespaces'):
            actual_namespaces = list(stats.namespaces.keys())
        elif 'namespaces' in stats:
            actual_namespaces = list(stats['namespaces'].keys())
        
        actual_count = len(actual_namespaces)
        
        # Prepare detailed report
        report = {
            'expected_count': expected_count,
            'actual_count': actual_count,
            'expected_spaces': expected_spaces,
            'actual_spaces': actual_namespaces,
            'missing_spaces': [space for space in expected_spaces if space not in actual_namespaces],
            'unexpected_spaces': [space for space in actual_namespaces if space not in expected_spaces]
        }
        
        # Validation successful if counts match
        success = actual_count == expected_count
        
        return success, report
        
    except Exception as e:
        print(f"Error validating namespace count: {e}")
        return False, {'error': str(e)}

# Function to upsert data
def upsert_data(index, df, namespace):
    print("Start: Upserting data to Pinecone index")
    
    # Add detailed debugging of the input dataframe
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    
    # Validate expected namespaces before starting
    success, namespace_report = validate_namespace_count(index)
    if not success:
        print("\nWARNING: Namespace count validation failed!")
        print(f"Expected {namespace_report['expected_count']} namespaces, found {namespace_report['actual_count']}")
        if namespace_report['missing_spaces']:
            print(f"Missing spaces: {namespace_report['missing_spaces']}")
        if namespace_report['unexpected_spaces']:
            print(f"Unexpected spaces: {namespace_report['unexpected_spaces']}")
        print("Proceeding with upsert anyway...\n")
    
    # Check if the space column exists and show its values
    if 'space' in df.columns:
        space_values = df['space'].value_counts().to_dict()
        print(f"Space column values distribution: {space_values}")
    
    prepped = []
    skipped = 0
    success_count = 0
    metadata_size_errors = 0
    upsert_success = False

    # Get initial count for verification
    try:
        initial_stats = index.describe_index_stats()
        initial_count = initial_stats.get('total_vector_count', 0) if initial_stats else 0
        print(f"Initial vector count in index: {initial_count}")
    except Exception as e:
        print(f"Warning: Could not get initial index stats: {e}")
        initial_count = 0

    # Use default-index if namespace is "all", otherwise use the provided namespace
    target_namespace = "default-index" if namespace == "all" else namespace
    print(f"\nProcessing namespace: '{target_namespace}' with {len(df)} records")
    
    namespace_prepped = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        # Skip rows with None values
        if row['values'] is None:
            skipped += 1
            continue
        
        try:
            # Ensure the metadata is properly loaded - debug the first few rows
            if i < 5:
                print(f"\nDEBUG row {i} metadata before processing: {row['metadata'][:100]}...")
            
            meta = None
            if isinstance(row['metadata'], str):
                meta = ast.literal_eval(row['metadata'])
            elif isinstance(row['metadata'], dict):
                meta = row['metadata']
            else:
                print(f"Warning: Metadata for row {i} is neither string nor dict: {type(row['metadata'])}")
                skipped += 1
                continue
            
            # Ensure space is in metadata if it exists in DataFrame
            if 'space' in df.columns:
                meta['space'] = row['space']
            
            # Convert metadata values to strings if they aren't already
            meta = {k: str(v) if not isinstance(v, (str, bool, int, float)) else v 
                   for k, v in meta.items()}
            
            # Check metadata size
            metadata_str = str(meta)
            if len(metadata_str) > 40000:  # Pinecone metadata size limit
                print(f"\nWarning: Metadata size ({len(metadata_str)}) exceeds limit for row {i}")
                metadata_size_errors += 1
                continue
            
            # Prepare the vector for upsert
            vector_data = {
                'id': str(row['id']),
                'values': row['values'],
                'metadata': meta
            }
            namespace_prepped.append(vector_data)
            
            # Batch upsert when we reach 100 vectors
            if len(namespace_prepped) >= 100:
                try:
                    index.upsert(vectors=namespace_prepped, namespace=target_namespace)
                    success_count += len(namespace_prepped)
                    namespace_prepped = []
                except Exception as e:
                    print(f"\nError during batch upsert: {e}")
                    namespace_prepped = []
            
        except Exception as e:
            print(f"\nError processing row {i}: {e}")
            continue
    
    # Upsert any remaining vectors
    if namespace_prepped:
        try:
            index.upsert(vectors=namespace_prepped, namespace=target_namespace)
            success_count += len(namespace_prepped)
        except Exception as e:
            print(f"\nError during final batch upsert: {e}")
    
    print(f"Successfully upserted {success_count} vectors to namespace '{target_namespace}'")
    
    # Verify the upsert was successful
    try:
        final_stats = index.describe_index_stats()
        final_count = final_stats.get('total_vector_count', 0) if final_stats else 0
        expected_increase = success_count
        actual_increase = final_count - initial_count
        
        print("\nUpsert Summary:")
        print(f"Total vectors processed successfully: {success_count}")
        print(f"Skipped vectors: {skipped}")
        print(f"Metadata size errors: {metadata_size_errors}")
        print(f"\nInitial vector count: {initial_count}")
        print(f"Final vector count: {final_count}")
        print(f"Expected increase: {expected_increase}")
        print(f"Actual increase: {actual_increase}")
        
        upsert_success = True
        
    except Exception as e:
        print(f"\nError verifying upsert: {e}")
    
    return upsert_success, success_count, skipped, metadata_size_errors

# Function to verify upsert was successful
def verify_pinecone_upsert(index, query_sample=None):
    """
    Verify that data was successfully upserted to Pinecone.
    
    Args:
        index: The Pinecone index to verify
        query_sample: Optional sample query to test search functionality
        
    Returns:
        bool: True if verification was successful, False otherwise
    """
    try:
        # Check if index is accessible
        stats = index.describe_index_stats()
        
        if not stats or 'total_vector_count' not in stats:
            print("Verification failed: Could not retrieve index stats")
            return False
            
        vector_count = stats['total_vector_count']
        if vector_count <= 0:
            print(f"Verification failed: Index contains no vectors (count: {vector_count})")
            return False
            
        print(f"Index contains {vector_count} vectors")
        
        # If a sample query is provided, test search functionality
        if query_sample:
            from utils.openai_logic import get_embeddings
            
            print(f"Testing search with sample query: '{query_sample}'")
            embedding_response = get_embeddings(query_sample, "text-embedding-ada-002")
            query_embedding = embedding_response.data[0].embedding
            
            results = index.query(
                vector=query_embedding,
                top_k=1,
                include_metadata=True
            )
            
            if not results or 'matches' not in results or len(results['matches']) == 0:
                print("Verification failed: No results returned for sample query")
                return False
                
            print(f"Search test successful: Retrieved {len(results['matches'])} matches")
        
        return True
    except Exception as e:
        print(f"Verification failed with error: {e}")
        return False

# Function to query Pinecone index
def query_pinecone(index, query_text, top_k=5, filter_by_space=None):
    """
    Query Pinecone index with an embedding of the query text.
    
    Args:
        index: The Pinecone index to query
        query_text: The text query to search for
        top_k: Number of results to return (default: 5)
        filter_by_space: Optional space/namespace to filter results by
        
    Returns:
        List of tuples (source_url, text_content, score) of the top matching documents
    """
    try:
        from utils.openai_logic import get_embeddings
        
        # Get configured spaces from environment
        configured_spaces = os.getenv('SPACES', '').split(',')
        configured_spaces = [space.strip() for space in configured_spaces if space.strip()]
        
        # First check index stats to see available namespaces
        try:
            stats = index.describe_index_stats()
            print("Available namespaces in index:")
            if hasattr(stats, 'namespaces'):
                namespaces = list(stats.namespaces.keys())
            elif 'namespaces' in stats:
                namespaces = list(stats.get('namespaces', {}).keys())
            else:
                namespaces = ['']
                
            for ns in namespaces:
                ns_display = f"'{ns}'" if ns else "''"
                if 'namespaces' in stats and ns in stats['namespaces']:
                    count = stats['namespaces'][ns].get('vector_count', 0)
                    print(f"  {ns_display}: {count} vectors")
                else:
                    print(f"  {ns_display}")
                    
            # Print configured spaces for comparison
            if configured_spaces:
                print(f"Configured spaces from environment: {configured_spaces}")
                # Check which configured spaces exist in the index
                for space in configured_spaces:
                    if space in namespaces:
                        print(f"  Configured space '{space}' exists in index")
                    else:
                        print(f"  Warning: Configured space '{space}' not found in index")
        except Exception as e:
            print(f"Error checking index stats: {e}")
            namespaces = ['']
        
        namespace = None
        if filter_by_space:
            # If specific space is requested, find best match among available namespaces
            print(f"Filtering by space: '{filter_by_space}'")
            
            # Check for exact match first
            if filter_by_space in namespaces:
                namespace = filter_by_space
                print(f"Found exact namespace match: '{namespace}'")
            else:
                # Check for case-insensitive match
                for ns in namespaces:
                    if ns.lower() == filter_by_space.lower():
                        namespace = ns
                        print(f"Found case-insensitive namespace match: '{namespace}'")
                        break
                
                # If still no match, look for partial matches
                if not namespace:
                    partial_matches = [ns for ns in namespaces if filter_by_space.lower() in ns.lower()]
                    if partial_matches:
                        namespace = partial_matches[0]
                        print(f"Using partial namespace match: '{namespace}'")
                    else:
                        print(f"No matching namespace found for '{filter_by_space}'")
                        # Instead of defaulting to '' which searches everything, return empty
                        return []
        elif configured_spaces and len(configured_spaces) == 1:
            # If only one space is configured and no specific filter is requested, use that space
            space = configured_spaces[0]
            if space in namespaces:
                namespace = space
                print(f"Using single configured space: '{namespace}'")
            
        print(f"Querying Pinecone with: '{query_text}' using namespace: {namespace if namespace else 'all'}")
        
        # Generate embedding for the query
        embedding_response = get_embeddings(query_text, "text-embedding-ada-002")
        if not embedding_response or not hasattr(embedding_response, 'data') or not embedding_response.data:
            print("Error: Failed to generate embedding for query")
            return []
            
        query_embedding = embedding_response.data[0].embedding
        print(f"Dimension of query embedding: ", len(query_embedding))
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
        
        if not results or 'matches' not in results:
            print("No results returned from Pinecone query")
            return []
            
        # Process results
        extracted_info = []
        for match in results['matches']:
            # Extract metadata
            metadata = match.get('metadata', {})
            source = metadata.get('source', 'No source available')
            text = metadata.get('text', 'No content available')
            score = match.get('score', 0.0)
            
            # Debug namespaces
            namespace_info = ""
            if 'space' in metadata:
                namespace_info = f" [space: {metadata['space']}]"
            
            print(f"Match{namespace_info} - score: {score:.4f}, source: {source[:50]}...")
            extracted_info.append((source, text, score))
        
        print(f"Query returned {len(extracted_info)} results")
        return extracted_info
        
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        import traceback
        traceback.print_exc()
        return []

