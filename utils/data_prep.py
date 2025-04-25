import pandas as pd
import json
from tqdm.auto import tqdm
from utils.openai_logic import create_embeddings
import os, sys
from .auth import get_confluence_client
import re
import urllib.parse

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
        
        # Get all spaces with pagination
        all_spaces = []
        space_start = 0
        space_limit = 1000
        
        while True:
            spaces = confluence.get_all_spaces(
                start=space_start, 
                limit=space_limit, 
                expand='description.plain'
            )
            
            if not isinstance(spaces, dict) or 'results' not in spaces:
                break
                
            current_spaces = spaces['results']
            if not current_spaces:
                break
                
            all_spaces.extend(current_spaces)
            print(f"Found {len(current_spaces)} more spaces (total: {len(all_spaces)})")
            
            if len(current_spaces) < space_limit:
                break
                
            space_start += len(current_spaces)
        
        data = []
        id_counter = 1
        
        for space in all_spaces:
            # Skip archival or personal spaces if needed
            if space['type'] != 'global':
                continue
                
            print(f"\nProcessing space: {space['key']}")
            
            # Get pages from space with pagination
            page_start = 0
            while True:
                pages = confluence.get_all_pages_from_space(
                    space['key'], 
                    start=page_start,
                    limit=100,  # Can increase this if needed
                    status='current'
                )
                
                if not isinstance(pages, dict) or 'results' not in pages:
                    current_pages = pages if isinstance(pages, list) else []
                else:
                    current_pages = pages['results']
                
                if not current_pages:
                    break
                    
                print(f"Processing {len(current_pages)} pages from {space['key']} (starting at {page_start})")
                
                for page in current_pages:
                    try:
                        page_content = confluence.get_page_by_id(
                            page['id'], 
                            expand='body.storage'
                        )
                        content = page_content['body']['storage']['value']
                        
                        data.append({
                            'id': str(id_counter),
                            'tiny_link': f"{confluence_domain}/pages/viewpage.action?pageId={page['id']}",
                            'content': content,
                            'space_key': space['key']  # Added space key for tracking
                        })
                        id_counter += 1
                        
                        if max_rows and id_counter > max_rows:
                            break
                    except Exception as e:
                        print(f"Error processing page {page['id']} in space {space['key']}: {str(e)}")
                
                if max_rows and id_counter > max_rows:
                    break
                    
                # Move to next page of results
                page_start += len(current_pages)
                
                # If we got less than the limit, we've reached the end
                if len(current_pages) < 100:
                    break
            
            print(f"Completed space {space['key']} with {id_counter} total pages")
        
        # Convert to DataFrame
        new_df = pd.DataFrame(data)
        print("\nFinal statistics:")
        space_counts = new_df.groupby('space_key').size()
        print("\nPages per space:")
        for space_key, count in space_counts.items():
            print(f"  {space_key}: {count} pages")
            
        return new_df
        
    except Exception as e:
        print(f"Error in import_csv: {str(e)}")
        if csv_file and os.path.exists(csv_file):
            print(f"Falling back to CSV file: {csv_file}")
            return pd.read_csv(csv_file)
        raise

def clean_data_pinecone_schema(df):
    # Check if df is None
    if df is None:
        return "Error: No data was loaded from either Confluence or CSV"
    
    # Add detailed debugging of input dataframe
    print(f"Input DataFrame shape: {df.shape}")
    print(f"Input DataFrame columns: {df.columns.tolist()}")
    
    # Load spaces configuration from environment variable
    configured_spaces = os.getenv('SPACES', '').split(',')
    configured_spaces = [space.strip() for space in configured_spaces if space.strip()]
    print(f"Configured spaces from environment: {configured_spaces}")
    
    # Update to check for either 'tiny_link' or 'url' column
    if 'tiny_link' in df.columns:
        link_column = 'tiny_link'
        print(f"Using 'tiny_link' column for URLs")
    elif 'url' in df.columns:
        link_column = 'url'
        print(f"Using 'url' column for URLs")
    else:
        # If neither 'tiny_link' nor 'url' is present
        return "Error: CSV file is missing required link column: either 'tiny_link' or 'url' must be present"
    
    # Ensure necessary columns are present
    required_columns = {'id', 'content'}
    if not required_columns.issubset(df.columns):
        missing_columns = required_columns - set(df.columns)
        return f"Error: CSV file is missing required columns: {missing_columns}"
    
    # Check if space column exists
    space_column_exists = 'space' in df.columns
    if not space_column_exists:
        print("Warning: 'space' column not found. Will extract space keys from URLs or use configured spaces.")
    else:
        space_counts = df['space'].value_counts().to_dict()
        print(f"Found existing 'space' column with values: {space_counts}")
    
    # Filter out rows where 'content' is empty
    df_filtered = df[df['content'].notna() & (df['content'] != '')].copy()
    
    if df_filtered.empty:
        return "Error: No valid data found in the CSV file after filtering empty content."
    
    # Proceed with the function's main logic - operate on the copy
    df_filtered['id'] = df_filtered['id'].astype(str)
    
    # Rename the link column to 'source'
    df_filtered.rename(columns={link_column: 'source'}, inplace=True)
    
    # If space column doesn't exist, create it with default value
    if not space_column_exists:
        df_filtered['space'] = 'default-index'
    
    # Extract space key from URL whether space column exists or not
    # This ensures we always use the space key, not the space name
    if 'source' in df_filtered.columns:
        # Print sample URLs for debugging
        print("\nSample URLs for space key extraction:")
        for i, url in enumerate(df_filtered['source'].head(10).tolist()):
            print(f"  URL {i}: {url}")
        
        # Try to extract space from Confluence URL
        try:
            def extract_space_key_from_url(url):
                # First, convert to string if not already
                if not isinstance(url, str):
                    url = str(url)
                
                # Get configured spaces from environment
                configured_spaces = os.getenv('SPACES', '').split(',')
                configured_spaces = [space.strip() for space in configured_spaces if space.strip()]
                
                # Print the URL for debugging first 100 chars
                url_preview = url[:100] + ("..." if len(url) > 100 else "")
                
                # Various patterns for Confluence URLs
                # 1. Standard spaces pattern: /spaces/SPACEKEY/
                spaces_pattern = r'/spaces/([^/]+)'
                
                # 2. SpaceKey parameter: spaceKey=SPACEKEY
                space_key_param = r'spaceKey=([^&]+)'
                
                # 3. Wiki URL with space in path: /display/SPACEKEY/
                wiki_pattern = r'/display/([^/]+)'
                
                # 4. Space in path component: /space/SPACEKEY/
                space_pattern = r'/space/([^/]+)'
                
                # 5. Direct space prefix: space=SPACEKEY
                direct_space = r'space=([^&\s]+)'
                
                # Try all patterns in order
                patterns = [
                    (spaces_pattern, "spaces pattern"),
                    (space_key_param, "spaceKey parameter"),
                    (wiki_pattern, "wiki pattern"),
                    (space_pattern, "space pattern"),
                    (direct_space, "direct space")
                ]
                
                extracted_space = None
                
                for pattern, pattern_name in patterns:
                    match = re.search(pattern, url)
                    if match:
                        space_key = match.group(1)
                        # In case the URL has encoded characters
                        space_key = urllib.parse.unquote(space_key)
                        # Only use the space key if it's in configured spaces
                        if space_key in configured_spaces:
                            print(f"  Matched {pattern_name}: {space_key} from {url_preview}")
                            extracted_space = space_key
                            break
                        else:
                            print(f"  Found space key {space_key} but it's not in configured spaces, skipping")
                
                # Remove the subdomain extraction fallback completely
                
                return extracted_space or 'default-index'  # Return 'default-index' if no valid space found
            
            # Apply extraction to URLs with verbose debugging
            print("\nExtracting space keys from URLs:")
            space_keys = []
            for i, row in df_filtered.head(10).iterrows():
                url = row['source']
                space_key = extract_space_key_from_url(url)
                space_keys.append(space_key)
                print(f"  Row {i} - URL: {url} -> Space key: {space_key}")
            
            # Apply to full dataset
            df_filtered['space'] = df_filtered['source'].apply(extract_space_key_from_url)
            unique_spaces = set(df_filtered['space'].tolist())
            space_counts = df_filtered['space'].value_counts().to_dict()
            
            print(f"\nExtracted {len(unique_spaces)} unique space keys from URLs")
            print(f"Space key counts: {space_counts}")
            print(f"Space keys found: {', '.join(list(unique_spaces)[:10])}" + 
                  (f"... and {len(unique_spaces) - 10} more" if len(unique_spaces) > 10 else ""))
            
        except Exception as e:
            print(f"Error extracting spaces from URLs: {e}")
            import traceback
            traceback.print_exc()
            # If extraction fails but we have configured spaces, use the first one
            if configured_spaces:
                print(f"Using first configured space as fallback: {configured_spaces[0]}")
                df_filtered['space'] = configured_spaces[0]
            else:
                # Otherwise default
                df_filtered['space'] = 'default-index'
    
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
        lambda row: json.dumps({'source': row['source'], 'text': row['content'], 'space': row['space']}), 
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
                return json.dumps({'source': 'error', 'text': 'Error processing content', 'space': 'default-index'})
        
        df_filtered['metadata'] = df_filtered.apply(
            lambda row: ensure_size_limit(row['metadata'], row['id']), 
            axis=1
        )
    
    # Drop the temporary size column and content column (no longer needed)
    if 'metadata_size' in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=['metadata_size'])
    
    # Keep the essential columns for Pinecone, including space for namespace
    result_df = df_filtered[['id', 'metadata', 'space']].copy()
    
    # Get a count of records per space for reporting
    spaces_count = result_df['space'].value_counts().to_dict()
    print(f"Records by space: {spaces_count}")
    
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
    
    # Store columns we want to preserve
    preserve_columns = ['id', 'metadata']
    if 'space' in df.columns:
        preserve_columns.append('space')
        print(f"Will preserve 'space' column with {df['space'].nunique()} unique values")
    
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

    # Print before drop
    print(f"DataFrame before dropping failed rows: {df.shape}")
    print(f"Columns before drop: {df.columns.tolist()}")

    # Remove rows with None values for 'values' column before returning
    original_count = df.shape[0]
    df = df.dropna(subset=['values'])
    final_count = df.shape[0]
    dropped_count = original_count - final_count
    
    # Ensure we preserved all important columns
    print(f"Columns after drop: {df.columns.tolist()}")
    for col in preserve_columns:
        if col not in df.columns:
            print(f"WARNING: {col} column was lost during processing!")
    
    # Debug space column if it exists
    if 'space' in df.columns:
        space_counts = df['space'].value_counts().to_dict()
        print(f"Space column values after processing: {space_counts}")
    
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

