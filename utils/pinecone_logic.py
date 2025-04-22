from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
import ast
import os

#Global variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pinecone = Pinecone(api_key=PINECONE_API_KEY)
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# delete index
def delete_pinecone_index(index_name):
    print(f"Deleting index '{index_name}' if it exists.")
    try:
        pinecone.delete_index(index_name)
        print(f"Index '{index_name}' successfully deleted.")
    except Exception as e:
        print(f"index '{index_name}' not found no action taken.")


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


# Function to upsert data
def upsert_data(index, df):
    print("Start: Upserting data to Pinecone index")
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

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Skip rows with None values
        if row['values'] is None:
            skipped += 1
            continue
        
        try:
            meta = ast.literal_eval(row['metadata'])
            
            # Validate that values is a list type (needed by Pinecone)
            values = row['values']
            if not isinstance(values, list):
                print(f"Warning: Skipping row {i} - 'values' is not a list: {type(values)}")
                skipped += 1
                continue
                
            # Check metadata size before adding to batch
            metadata_str = str(meta)
            metadata_size = len(metadata_str)
            if metadata_size > 40000:  # 40KB limit
                print(f"Warning: Metadata for row {i} is too large ({metadata_size} bytes). Truncating...")
                # Truncate text field if present
                if 'text' in meta and isinstance(meta['text'], str):
                    max_size = 39000  # Leave room for other fields
                    reduction_needed = metadata_size - max_size
                    current_text_length = len(meta['text'])
                    new_text_length = max(10, current_text_length - reduction_needed - 100)  # Extra buffer
                    meta['text'] = meta['text'][:new_text_length] + "... [TRUNCATED]"
                else:
                    # If no text field, we can't do much
                    print(f"Error: Cannot truncate metadata for row {i} - no text field found")
                    skipped += 1
                    continue
            
            prepped.append({
                'id': row['id'], 
                'values': values,
                'metadata': meta
            })
            
            if len(prepped) >= 200: # batching upserts
                try:
                    upsert_response = index.upsert(prepped)
                    if upsert_response and 'upserted_count' in upsert_response:
                        success_count += upsert_response['upserted_count']
                        upsert_success = True
                        print(f"Batch upsert successful: {upsert_response['upserted_count']} records upserted")
                    else:
                        print(f"Warning: Batch upsert response did not contain expected data: {upsert_response}")
                except Exception as e:
                    if "metadata size" in str(e).lower() and "exceeds the limit" in str(e).lower():
                        metadata_size_errors += 1
                        print(f"Error during batch upsert: {e}")
                        
                        # Handle batch with metadata size issues by upserting one by one
                        print("Attempting to upsert records individually...")
                        for record in prepped:
                            try:
                                # Further truncate text if needed
                                if 'text' in record['metadata'] and isinstance(record['metadata']['text'], str):
                                    record['metadata']['text'] = record['metadata']['text'][:20000] + "... [SEVERELY TRUNCATED]"
                                
                                single_response = index.upsert([record])
                                if single_response and 'upserted_count' in single_response:
                                    success_count += single_response['upserted_count']
                                    upsert_success = True
                            except Exception as single_error:
                                print(f"Error upserting individual record {record['id']}: {single_error}")
                    else:
                        print(f"Error during batch upsert: {e}")
                prepped = []
        except Exception as e:
            print(f"Error processing row {i} for upsert: {e}")
            skipped += 1

    # Upsert any remaining entries after the loop
    if len(prepped) > 0:
        try:
            upsert_response = index.upsert(prepped)
            if upsert_response and 'upserted_count' in upsert_response:
                success_count += upsert_response['upserted_count']
                upsert_success = True
                print(f"Final batch upsert successful: {upsert_response['upserted_count']} records upserted")
            else:
                print(f"Warning: Final batch upsert response did not contain expected data: {upsert_response}")
        except Exception as e:
            if "metadata size" in str(e).lower() and "exceeds the limit" in str(e).lower():
                metadata_size_errors += 1
                print(f"Error during final batch upsert: {e}")
                
                # Handle batch with metadata size issues by upserting one by one
                print("Attempting to upsert remaining records individually...")
                for record in prepped:
                    try:
                        # Further truncate text if needed
                        if 'text' in record['metadata'] and isinstance(record['metadata']['text'], str):
                            record['metadata']['text'] = record['metadata']['text'][:20000] + "... [SEVERELY TRUNCATED]"
                        
                        single_response = index.upsert([record])
                        if single_response and 'upserted_count' in single_response:
                            success_count += single_response['upserted_count']
                            upsert_success = True
                    except Exception as single_error:
                        print(f"Error upserting individual record {record['id']}: {single_error}")
            else:
                print(f"Error during final batch upsert: {e}")
    
    # Print summary statistics
    print("\nUpsert Summary:")
    print(f"- Total records processed: {df.shape[0]}")
    print(f"- Records skipped before upsert: {skipped}")
    print(f"- Records successfully upserted: {success_count}")
    print(f"- Metadata size errors encountered: {metadata_size_errors}")
    
    # Verify upsert was successful by checking index stats
    try:
        stats = index.describe_index_stats()
        if stats and 'total_vector_count' in stats:
            final_count = stats['total_vector_count']
            print(f"Index now contains {final_count} vectors")
            vectors_added = final_count - initial_count
            print(f"Net vectors added in this operation: {vectors_added}")
            
            if final_count > initial_count:
                upsert_success = True
        else:
            print("Warning: Could not verify index stats")
    except Exception as e:
        print(f"Error checking index stats: {e}")
    
    if upsert_success:
        print(f"Done: Data upserted to Pinecone index successfully. {success_count} records upserted")
    else:
        print("Warning: Data upsert to Pinecone index could not be verified")
    
    return upsert_success, success_count

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

