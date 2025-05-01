import os
import pandas as pd
import argparse
from dotenv import load_dotenv, find_dotenv
from utils.pinecone_logic import get_pinecone_index, upsert_data, verify_pinecone_upsert, init_pinecone, delete_pinecone_index
from utils.data_prep import import_csv, clean_data_pinecone_schema, generate_embeddings_and_add_to_df
from utils.split_spaces import split_kb_by_space
import sys
from tqdm import tqdm
import openai
import tiktoken
import json
from datetime import datetime, timezone

# load environment variables
load_dotenv(find_dotenv())

# Constants for timestamp tracking
TIMESTAMP_FILE = "last_update_timestamp.json"

def get_last_update_timestamp():
    """
    Get the timestamp of the last successful update.
    
    Returns:
        str: ISO formatted timestamp string or None if no previous update
    """
    if os.path.exists(TIMESTAMP_FILE):
        try:
            with open(TIMESTAMP_FILE, 'r') as f:
                data = json.load(f)
                return data.get('last_update')
        except Exception as e:
            print(f"Error reading timestamp file: {e}")
    return None

def save_update_timestamp():
    """
    Save the current timestamp as the last update time.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        with open(TIMESTAMP_FILE, 'w') as f:
            json.dump({'last_update': timestamp}, f)
        print(f"Update timestamp saved: {timestamp}")
    except Exception as e:
        print(f"Error saving timestamp: {e}")

def prepare_data_for_upsert(df, model_name):
    """
    Clean data and generate embeddings for upsert to Pinecone.
    
    Args:
        df: Input DataFrame
        model_name: Name of the OpenAI embedding model to use
        
    Returns:
        DataFrame ready for upserting to Pinecone, or None if error
    """
    # Clean and prepare the data
    df_cleaned = clean_data_pinecone_schema(df)
    if isinstance(df_cleaned, str):  # Error message returned
        print(f"Error cleaning data: {df_cleaned}")
        return None
        
    # Generate embeddings
    df_with_embeddings = generate_embeddings_and_add_to_df(df_cleaned, model_name)
    if df_with_embeddings is None:
        print("Error generating embeddings")
        return None
        
    return df_with_embeddings

# Function to truncate text based on token limit
def truncate_text_by_tokens(text, model_name, max_tokens=8000):
    """Truncates text to a maximum number of tokens using tiktoken."""
    try:
        # Get the encoding for the specified model
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback for models not explicitly known by tiktoken
        print(f"Warning: Encoding for model '{model_name}' not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
        
    tokens = encoding.encode(text)
    
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        truncated_text = encoding.decode(truncated_tokens)
        # Optional: Add indication that text was truncated
        # truncated_text += " [Truncated]"
        return truncated_text
    else:
        return text

def process_space_file(file_path, index, model_name, namespace_name, incremental_update=False, last_update=None):
    try:
        df = pd.read_csv(file_path)
        
        # Check if DataFrame is empty
        if df.empty:
            print(f"Skipping empty file: {file_path}")
            return True # Return True to indicate successful handling (skipping)

        # Original print statement (now safe due to the check above)
        print(f"Read {len(df)} records for space {df['space_key'].iloc[0] if 'space_key' in df.columns else os.path.basename(file_path).replace('.csv', '')}")

        # Filter for records modified since last update if incremental update is enabled
        if incremental_update and last_update and 'last_modified' in df.columns:
            try:
                # Convert the last_modified column to datetime
                df['last_modified'] = pd.to_datetime(df['last_modified'])
                last_update_dt = pd.to_datetime(last_update)
                
                # Filter for records modified since last update
                original_count = len(df)
                df = df[df['last_modified'] > last_update_dt]
                
                print(f"Incremental update: Found {len(df)} records modified since {last_update}")
                print(f"Original record count: {original_count}, Updated records: {len(df)}")
                
                if len(df) == 0:
                    print(f"No updates needed for {namespace_name}.")
                    return True
            except Exception as e:
                print(f"Warning: Error filtering by last_modified date: {e}")
                print("Processing all records...")
        
        # Ensure 'id' column exists and is suitable for Pinecone ID
        if 'id' not in df.columns:
            print(f"Error: 'id' column missing in {file_path}. Skipping.")
            return False
            
        df['pinecone_id'] = df['id'].astype(str) # Ensure ID is string for Pinecone

        print(f"Processing {len(df)} records from {file_path} into namespace '{namespace_name}'...")

        # Add embedding generation and upsert logic here
        # Reduce batch size to 1 to ensure individual API calls stay under token limit
        batch_size = 1

        for i in tqdm(range(0, len(df), batch_size)):
            batch = df.iloc[i:i + batch_size]
            
            # Combine title and content for embedding
            # Ensure content is string and handle potential NaN values
            batch_loc = batch.index
            batch_combined_text = batch.loc[batch_loc, 'title'].fillna('') + " " + batch.loc[batch_loc, 'content'].fillna('').astype(str)

            # Truncate text for each item in the batch before sending to OpenAI
            truncated_batch_text = [
                truncate_text_by_tokens(text, model_name) 
                for text in batch_combined_text.tolist()
            ]

            # Generate embeddings using OpenAI API with truncated text
            try:
                # Use the truncated_batch_text list
                response = openai.embeddings.create(
                    input=truncated_batch_text, 
                    model=model_name 
                )
                # Extract embeddings from the response object
                embeddings = [item.embedding for item in response.data]
                
                # Check if the number of embeddings matches the batch size
                if len(embeddings) != len(batch):
                    print(f"Warning: Mismatch in embeddings count for batch {i//batch_size}. Expected {len(batch)}, got {len(embeddings)}.")
                    # Handle mismatch, e.g., skip batch or try to reconcile
                    continue 
                    
            except Exception as e:
                print(f"Error generating OpenAI embeddings for batch {i//batch_size}: {e}")
                continue # Skip this batch on embedding error

            # Prepare metadata, ensuring values are suitable types (string, number, boolean, list of strings)
            metadata = []
            for _, row in batch.iterrows():
                 meta = {
                    'title': str(row['title']) if pd.notna(row['title']) else '',
                    'url': str(row['url']) if pd.notna(row['url']) else '',
                    'space_key': str(row['space_key']) if pd.notna(row['space_key']) else '',
                    # Add other relevant metadata fields here, converting types if necessary
                    # Example: 'last_modified': str(row['last_modified']) if pd.notna(row['last_modified']) else ''
                 }
                 # Ensure no None values in metadata
                 metadata.append({k: v for k, v in meta.items() if v is not None})


            # Prepare vectors for upsert
            vectors_to_upsert = list(zip(batch['pinecone_id'].tolist(), embeddings, metadata))

            # Upsert to Pinecone
            try:
                index.upsert(vectors=vectors_to_upsert, namespace=namespace_name)
                # print(f"Upserted batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")
            except Exception as e:
                print(f"Error upserting batch {i//batch_size}: {e}")
                # Optional: add more detailed error handling or logging here

        print(f"Finished processing {file_path}.")
        return True
        
    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
        return False
    except pd.errors.EmptyDataError:
        print(f"Error: File is empty {file_path}")
        return True # Treat empty file as successfully handled (skipped)
    except Exception as e:
        print(f"An error occurred processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Update Pinecone database with Confluence data")
    parser.add_argument("--reset", action="store_true", help="Reset the Pinecone index before updating")
    parser.add_argument("--incremental", action="store_true", help="Only process records that have changed since the last update")
    args = parser.parse_args()
    
    reset_index = args.reset
    incremental_update = args.incremental
    
    # Get the last update timestamp if incremental update is enabled
    last_update = None
    if incremental_update:
        last_update = get_last_update_timestamp()
        if last_update:
            print(f"Performing incremental update. Last update: {last_update}")
        else:
            print("No previous update timestamp found. Performing full update.")
            incremental_update = False
    
    # Configure OpenAI client
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return
    openai.api_key = openai_api_key
    
    model_name = "text-embedding-ada-002" # OpenAI model name
    
    # Initialize Pinecone
    # Corrected index name based on log output checking for 'default-index'
    pinecone_index_name = os.getenv('PINECONE_INDEX_NAME', 'default-index') 
    
    if reset_index:
        print("Resetting Pinecone index...")
        # Use the correct index name for deletion
        delete_pinecone_index(pinecone_index_name) 
    
    # Pass the correct index name to init - REMOVED argument, function reads from env
    index = init_pinecone() # No argument needed here
    if not index:
        print("Failed to initialize Pinecone")
        return
        
    # First, split the data by space if kb.csv exists
    if os.path.exists('data/kb.csv'):
        print("Splitting data by space...")
        split_kb_by_space()
    
    # Process each space file
    space_files_dir = './data/spaces'
    if not os.path.exists(space_files_dir):
        print(f"Error: Directory '{space_files_dir}' not found.")
        sys.exit(1)
        
    space_files = [f for f in os.listdir(space_files_dir) if f.endswith('.csv')]
    
    if not space_files:
        print(f"No CSV files found in '{space_files_dir}'. Ensure app_confluence.py ran successfully.")
        sys.exit(1)

    all_successful = True
    for space_file in space_files:
        file_path = os.path.join(space_files_dir, space_file)
        # Generate namespace from filename (e.g., 'DIAS.csv' -> 'DIAS')
        namespace_name = space_file.replace('.csv', '') 
        print(f"\nProcessing {file_path} into namespace '{namespace_name}'...")
        
        # Pass the model_name string to the function along with incremental update flag and last update timestamp
        if not process_space_file(file_path, index, model_name, namespace_name, incremental_update, last_update):
            all_successful = False
            print(f"Failed to process {file_path}")

    if all_successful:
        print("\nAll space files processed successfully.")
        # Save the current timestamp for future incremental updates
        save_update_timestamp()
    else:
        print("\nSome space files failed to process.")

if __name__ == "__main__":
    main()
