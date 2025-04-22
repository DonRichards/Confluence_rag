import pandas as pd
import json
from tqdm.auto import tqdm
from utils.openai_logic import create_embeddings
import os, sys
from .auth import get_confluence_client

# Function to get dataset
def import_csv(df, csv_file, max_rows=None):
    """
    Fetch content from Confluence or fallback to CSV
    """
    try:
        # Get authenticated Confluence client
        confluence = get_confluence_client()
        
        # Get confluence domain for building URLs
        confluence_domain = os.getenv('CONFLUENCE_DOMAIN')
        
        # Get all spaces
        spaces = confluence.get_all_spaces(start=0, limit=50, expand='description.plain')
        
        data = []
        id_counter = 1
        
        for space in spaces['results']:
            # Skip archival or personal spaces if needed
            if space['type'] != 'global':
                continue
                
            # Get pages from space
            pages = confluence.get_all_pages_from_space(space['key'], start=0, limit=100, status='current')
            
            for page in pages['results']:
                page_content = confluence.get_page_by_id(page['id'], expand='body.storage')
                content = page_content['body']['storage']['value']
                
                data.append({
                    'id': str(id_counter),
                    'tiny_link': f"{confluence_domain}/pages/viewpage.action?pageId={page['id']}",
                    'content': content
                })
                id_counter += 1
                
                if max_rows and id_counter > max_rows:
                    break
            
            if max_rows and id_counter > max_rows:
                break
        
        # Convert to DataFrame
        new_df = pd.DataFrame(data)
        return new_df
        
    except Exception as e:
        print(f"Error fetching from Confluence: {str(e)}")
        print("Falling back to CSV file...")
        
        # Fallback to CSV if Confluence fetch fails
        if csv_file and os.path.exists(csv_file):
            try:
                if max_rows:
                    df = pd.read_csv(csv_file, nrows=max_rows)
                else:
                    df = pd.read_csv(csv_file)
                
                # Validate required columns and check for either tiny_link or url
                required_columns = {'id', 'content'}
                if not required_columns.issubset(df.columns):
                    missing_columns = required_columns - set(df.columns)
                    print(f"Error: CSV file is missing required columns: {missing_columns}")
                    return None
                
                # Check for link column (either tiny_link or url)
                if 'tiny_link' not in df.columns and 'url' not in df.columns:
                    print("Error: CSV file is missing a link column - either 'tiny_link' or 'url' is required")
                    return None
                    
                return df
            except Exception as e:
                print(f"Error reading CSV file: {e}")
                return None
        else:
            print(f"CSV file not found: {csv_file}")
            return None

def clean_data_pinecone_schema(df):
    # Check if df is None
    if df is None:
        return "Error: No data was loaded from either Confluence or CSV"
    
    # Update to check for either 'tiny_link' or 'url' column
    if 'tiny_link' in df.columns:
        link_column = 'tiny_link'
    elif 'url' in df.columns:
        link_column = 'url'
    else:
        # If neither 'tiny_link' nor 'url' is present
        return "Error: CSV file is missing required link column: either 'tiny_link' or 'url' must be present"
    
    # Ensure necessary columns are present
    required_columns = {'id', 'content'}
    if not required_columns.issubset(df.columns):
        missing_columns = required_columns - set(df.columns)
        return f"Error: CSV file is missing required columns: {missing_columns}"
    
    # Filter out rows where 'content' is empty
    df_filtered = df[df['content'].notna() & (df['content'] != '')].copy()
    
    if df_filtered.empty:
        return "Error: No valid data found in the CSV file after filtering empty content."
    
    # Proceed with the function's main logic - operate on the copy
    df_filtered['id'] = df_filtered['id'].astype(str)
    
    # Rename the link column to 'source'
    df_filtered.rename(columns={link_column: 'source'}, inplace=True)
    
    # Limit content size to avoid exceeding Pinecone's metadata size limit (40KB)
    # A safe limit would be around 30KB to account for other metadata
    max_content_length = 30000  # ~30KB in characters
    
    # Function to truncate text and add indicator if truncated
    def truncate_text(text):
        if len(text) > max_content_length:
            return text[:max_content_length] + "... [TRUNCATED]"
        return text
    
    # Apply truncation to content
    df_filtered['content'] = df_filtered['content'].apply(truncate_text)
    
    # Create metadata JSON
    import json
    df_filtered['metadata'] = df_filtered.apply(
        lambda row: json.dumps({'source': row['source'], 'text': row['content']}), 
        axis=1
    )
    
    # Additional check to ensure metadata size is within limits
    # Estimate JSON size in bytes and filter out any that are still too large
    max_metadata_size = 40000  # Just under 40KB
    df_filtered['metadata_size'] = df_filtered['metadata'].apply(len)
    oversized_count = df_filtered[df_filtered['metadata_size'] > max_metadata_size].shape[0]
    
    if oversized_count > 0:
        print(f"Warning: {oversized_count} records still exceed Pinecone's metadata size limit after truncation.")
        print("These records will be further truncated or removed.")
        
        # Further truncate oversized records
        def ensure_size_limit(metadata_json, row_id):
            try:
                metadata = json.loads(metadata_json)
                current_size = len(metadata_json)
                
                if current_size <= max_metadata_size:
                    return metadata_json
                
                # Calculate how much to reduce
                reduction_needed = current_size - max_metadata_size + 1000  # Extra 1KB buffer
                current_text_len = len(metadata['text'])
                new_text_len = max(10, current_text_len - reduction_needed)  # Ensure some text remains
                
                metadata['text'] = metadata['text'][:new_text_len] + "... [SEVERELY TRUNCATED]"
                return json.dumps(metadata)
            except Exception as e:
                print(f"Error processing row {row_id}: {e}")
                return json.dumps({'source': 'error', 'text': 'Error processing content'})
        
        df_filtered['metadata'] = df_filtered.apply(
            lambda row: ensure_size_limit(row['metadata'], row['id']), 
            axis=1
        )
    
    # Drop the temporary size column and content column (no longer needed)
    if 'metadata_size' in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=['metadata_size'])
    
    # Keep only the essential columns for Pinecone
    result_df = df_filtered[['id', 'metadata']].copy()
    
    print("Done: Dataset retrieved")
    return result_df


# Function to generate embeddings and add to DataFrame
def generate_embeddings_and_add_to_df(df, model_for_openai_embedding):
    # Ensure df is a DataFrame
    if isinstance(df, str):
        try:
            df = pd.read_csv(df)  # If df is a file path
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None
            
    if df is None or not isinstance(df, pd.DataFrame):
        print("Error: Input is not a valid DataFrame")
        return None
        
    if 'metadata' not in df.columns:
        print("Error: DataFrame is missing 'metadata' column")
        return None
    
    print("Start: Generating embeddings and adding to DataFrame")
    
    df['values'] = None
    # OpenAI's text-embedding-ada-002 has a token limit of 8191
    # But we'll be conservative since token count estimation is not exact
    # 1 token is roughly 4 chars in English, so 8000 tokens ≈ 32,000 chars
    # We'll be more conservative and use 28,000 chars initially
    safe_token_limit = 7000  # Characters (approx 7000 tokens)
    max_retries = 3  # Maximum number of retries for each row
    
    # We'll track failed rows to report at the end
    failed_rows = []
    retry_success = 0
    truncated_rows = 0

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            content = row['metadata']
            meta = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for row {index}: {e}")
            failed_rows.append(index)
            continue  # Skip to the next iteration

        text = meta.get('text', '')
        if not text:
            print(f"Warning: Missing 'text' in metadata for row {index}. Skipping.")
            failed_rows.append(index)
            continue

        # Pre-emptively truncate text to avoid token limit errors
        original_length = len(text)
        if original_length > safe_token_limit * 4:  # 4 chars per token approximation
            text = text[:safe_token_limit * 4]  # Initial conservative truncation
            truncated_rows += 1
            # Only log truncation if it's significant 
            if original_length > 1.5 * safe_token_limit * 4:
                print(f"Warning: Text for row {index} was truncated from {original_length} to {len(text)} chars")

        # Try embedding with progressive truncation if needed
        success = False
        retry_count = 0
        current_text = text
        
        while not success and retry_count < max_retries:
            try:
                response = create_embeddings(current_text, model_for_openai_embedding)
                if response is not None:
                    df.at[index, 'values'] = response
                    success = True
                    break
                else:
                    print(f"Warning: Empty embedding response for row {index}")
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a token limit error
                if "maximum context length" in error_msg or "token" in error_msg:
                    # Calculate a more aggressive truncation
                    # Each retry, reduce by half until we're under the limit
                    current_length = len(current_text)
                    new_length = current_length // 2
                    
                    print(f"Retry {retry_count+1} for row {index}: Truncating from {current_length} to {new_length} chars")
                    current_text = current_text[:new_length]
                    
                    # If we're getting very short and still having issues, give up
                    if new_length < 1000 and retry_count > 0:
                        print(f"Text too short to retry for row {index}, giving up")
                        break
                else:
                    # If it's not a token limit error, no point retrying
                    print(f"Non-token error for row {index}: {error_msg}")
                    break
                    
            retry_count += 1
        
        if success and retry_count > 0:
            retry_success += 1
        
        if not success:
            print(f"Failed to generate embedding for row {index} after {retry_count} retries")
            failed_rows.append(index)

    # Remove rows with None values for 'values' column before returning
    original_count = df.shape[0]
    df = df.dropna(subset=['values'])
    final_count = df.shape[0]
    dropped_count = original_count - final_count
    
    # Print summary statistics
    print("\nEmbedding Generation Summary:")
    print(f"- Total rows processed: {original_count}")
    print(f"- Rows where text was truncated: {truncated_rows}")
    print(f"- Rows that succeeded after retry: {retry_success}")
    print(f"- Rows that failed completely: {len(failed_rows)}")
    print(f"- Success rate: {(final_count / original_count) * 100:.2f}%")
    
    if dropped_count > 0:
        print(f"Warning: {dropped_count} rows were dropped due to failed embedding generation")
    
    print("Done: Generating embeddings and adding to DataFrame")
    return df

