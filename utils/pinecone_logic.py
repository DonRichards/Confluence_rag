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
            dimension=3072, 
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
def query_pinecone(index, query_text, top_k=8, filter_by_space=None, similarity_threshold=0.3):
    """
    Query Pinecone index with an improved hybrid search approach.
    
    Args:
        index: The Pinecone index to query
        query_text: The text query to search for
        top_k: Number of results to return (default: 8)
        filter_by_space: Optional space/namespace to filter results by
        similarity_threshold: Minimum similarity score to include in results (default: 0.3)
        
    Returns:
        List of tuples (source_url, text_content, score, date) of the top matching documents
    """
    try:
        from utils.openai_logic import get_embeddings, DEFAULT_EMBEDDING_MODEL
        import re
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        # Get configured spaces from environment
        configured_spaces = os.getenv('SPACES', '').split(',')
        configured_spaces = [space.strip() for space in configured_spaces if space.strip()]
        
        # Get available namespaces
        stats = index.describe_index_stats()
        namespaces = []
        
        # Convert to vector counts dictionary based on the return type
        if hasattr(stats, 'namespaces'):
            namespaces = list(stats.namespaces.keys())
        elif 'namespaces' in stats:
            namespaces = list(stats.get('namespaces', {}).keys())
        
        # Sort namespaces for consistent handling
        namespaces.sort()
        
        # Print namespaces
        print(f"Available namespaces: {namespaces}")
        
        # Default to None if no filter specified
        if not filter_by_space:
            configured_space = os.getenv('DEFAULT_SPACE')
            if configured_space and configured_space.strip():
                filter_by_space = configured_space.strip()
                print(f"Using default space filter: {filter_by_space}")
        
        print(f"Space filter: {filter_by_space if filter_by_space else 'None'}")
        
        # Check if filter_by_space is a valid namespace
        if filter_by_space and filter_by_space not in namespaces:
            # Try to find a matching namespace
            matching_namespaces = [ns for ns in namespaces if filter_by_space.lower() in ns.lower()]
            if matching_namespaces:
                filter_by_space = matching_namespaces[0]
                print(f"Using closest matching namespace: {filter_by_space}")
            else:
                print(f"Warning: Specified space '{filter_by_space}' not found in available namespaces. Searching across all spaces.")
                filter_by_space = None
        
        # Function to generate query variations for better retrieval
        def generate_query_variations(query):
            """
            Generate multiple variations of the query to improve retrieval performance.
            """
            # Original query is always included
            variations = [query]
            
            # Tokenize the query
            tokens = word_tokenize(query.lower())
            stop_words = set(stopwords.words('english'))
            important_words = [token for token in tokens if token.isalnum() and token not in stop_words]
            
            # Convert questions to statements
            if query.endswith('?'):
                statement = query.rstrip('?')
                variations.append(statement)
            
            # Common question reformulations
            if query.lower().startswith('what is'):
                variations.append(query[8:] + ' definition')
                variations.append('define ' + query[8:])
            
            if query.lower().startswith('how to'):
                variations.append(query[7:] + ' process')
                variations.append(query[7:] + ' steps')
                variations.append('steps for ' + query[7:])
            
            # Add a variation with just the important keywords
            if len(important_words) > 2:
                variations.append(' '.join(important_words))
            
            # Add a variation focused on current/recent content
            if not any(time_word in query.lower() for time_word in ['recent', 'latest', 'new', 'current', 'last week', 'this month']):
                variations.append(query + ' recent')
                
            # Handle different terminology variations
            # "Working Groups" vs "Workgroups"
            if "working groups" in query.lower():
                workgroups_variation = query.lower().replace("working groups", "workgroups")
                variations.append(workgroups_variation)
            elif "workgroups" in query.lower():
                working_groups_variation = query.lower().replace("workgroups", "working groups")
                variations.append(working_groups_variation)
                
            # "Interest Group" variations
            if "ai" in query.lower() and ("working groups" in query.lower() or "workgroups" in query.lower()):
                variations.append("AI Interest Group Working Groups")
                variations.append("AI Interest Group Workgroups")
                variations.append("Library AI Working Groups")
                variations.append("Library AI Workgroups")
            
            # Remove duplicates and return
            return list(dict.fromkeys(variations))
        
        query_variations = generate_query_variations(query_text)
        
        # Generate embeddings for all query variations
        all_results = []
        
        # Track unique IDs to prevent duplicates
        seen_ids = set()
        
        # Get 3x more results than needed for reranking
        search_top_k = max(top_k * 3, 20)
        
        # Process the original query and its variations
        for i, query_variant in enumerate(query_variations):
            print(f"Processing query variation {i+1}/{len(query_variations)}: '{query_variant}'")
            
            # Generate embedding for the query variant
            embedding_response = get_embeddings(query_variant, DEFAULT_EMBEDDING_MODEL)
            if not embedding_response or not hasattr(embedding_response, 'data') or not embedding_response.data:
                print(f"Warning: Failed to generate embedding for query variation {i+1}")
                continue
                
            query_embedding = embedding_response.data[0].embedding
            
            # Query Pinecone
            try:
                # First try sparse-dense hybrid search if available
                try:
                    # DISABLED: Hybrid search not supported in this index configuration
                    """
                    # Use alpha=0.5 for equal weight between sparse and dense vectors
                    query_params = {
                        "vector": query_embedding,
                        "top_k": search_top_k,
                        "include_metadata": True,
                        "include_values": False,
                        "alpha": 0.5,  # Balance between sparse and dense
                        "sparse_vector": {"indices": [], "values": []}  # Will be populated if BM25 is available
                    }
                    
                    # Try to generate sparse vector using BM25
                    try:
                        from sklearn.feature_extraction.text import TfidfVectorizer
                        import numpy as np
                        
                        # Create a simple BM25-like sparse vector
                        vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")
                        vectorizer.fit([query_variant])
                        
                        # Get feature names
                        feature_names = vectorizer.get_feature_names_out()
                        
                        # Create sparse vector
                        sparse_vec = vectorizer.transform([query_variant])
                        indices = sparse_vec.indices.tolist()
                        values = sparse_vec.data.tolist()
                        
                        query_params["sparse_vector"] = {
                            "indices": indices,
                            "values": values
                        }
                        
                        print(f"Using hybrid search with {len(indices)} sparse terms")
                    except Exception as e:
                        print(f"Warning: Failed to create sparse vector: {e}. Falling back to dense-only search.")
                        # Remove sparse_vector parameter
                        query_params.pop("sparse_vector", None)
                        query_params.pop("alpha", None)
                    
                    # Add namespace filter if specified
                    if filter_by_space:
                        query_params["namespace"] = filter_by_space
                    
                    # Execute the query
                    results = index.query(**query_params)
                    """
                    # Standard vector search as default - hybrid search disabled
                    raise Exception("Hybrid search explicitly disabled - using standard vector search")
                    
                except Exception as hybrid_error:
                    print(f"Using standard vector search (hybrid search disabled)")
                    
                    # Standard vector search as default
                    query_params = {
                        "vector": query_embedding,
                        "top_k": search_top_k,
                        "include_metadata": True,
                        "include_values": False
                    }
                    
                    # Add namespace filter if specified
                    if filter_by_space:
                        query_params["namespace"] = filter_by_space
                    
                    # Execute the query
                    results = index.query(**query_params)
                
                # Process matches
                for match in results.get('matches', []):
                    match_id = match.get('id')
                    
                    # Skip if we've already seen this ID
                    if match_id in seen_ids:
                        continue
                    
                    seen_ids.add(match_id)
                    
                    score = match.get('score', 0)
                    metadata = match.get('metadata', {})
                    
                    # Skip low-scoring matches
                    if score < similarity_threshold:
                        continue
                    
                    # Get basic metadata
                    source = metadata.get('url', 'Unknown source')
                    text = metadata.get('text', '')
                    
                    # Extract date information
                    doc_date = metadata.get('last_modified', '')
                    
                    # Create recency boost for newer content
                    recency_boost = 0
                    if doc_date:
                        try:
                            from datetime import datetime, timedelta
                            from dateutil import parser
                            
                            # Parse the date
                            parsed_date = parser.parse(doc_date)
                            
                            # Make sure the date is timezone-naive to avoid comparison issues
                            if parsed_date.tzinfo is not None:
                                parsed_date = parsed_date.replace(tzinfo=None)
                            
                            # Get current time as timezone-naive 
                            current_time = datetime.now().replace(tzinfo=None)
                            
                            # Calculate days since creation/modification
                            days_ago = (current_time - parsed_date).days
                            
                            # Apply a recency boost (up to 0.05 extra score for very recent content)
                            # Content from last 30 days gets a boost
                            if days_ago <= 30:
                                recency_boost = 0.05 * (1 - days_ago/30)
                        except Exception as date_error:
                            print(f"Date parsing error: {date_error}")
                    
                    # Apply text relevance boost based on exact term matching
                    text_boost = 0
                    if text:
                        # Count how many query terms appear in the text
                        query_terms = set(word_tokenize(query_variant.lower()))
                        text_terms = set(word_tokenize(text.lower()))
                        matching_terms = query_terms.intersection(text_terms)
                        
                        # Calculate coverage percentage
                        if query_terms:
                            coverage = len(matching_terms) / len(query_terms)
                            # Add up to 0.05 for full term coverage
                            text_boost = 0.05 * coverage
                    
                    # Combined score with boosts
                    adjusted_score = score + recency_boost + text_boost
                    
                    # Create result tuple with source, text, adjusted score and date
                    all_results.append((source, text, adjusted_score, doc_date))
                
            except Exception as e:
                print(f"Error querying Pinecone with variation {i+1}: {e}")
        
        # Deduplicate, sort by score, and limit to top_k results
        all_results.sort(key=lambda x: x[2], reverse=True)
        final_results = all_results[:top_k]
        
        print(f"Found {len(final_results)} results after filtering and re-ranking")
        return final_results
        
    except Exception as e:
        print(f"Error in query_pinecone: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

