import os
import pandas as pd
import argparse
from dotenv import load_dotenv, find_dotenv
from utils.pinecone_logic import get_pinecone_index, upsert_data, verify_pinecone_upsert, init_pinecone, delete_pinecone_index
from utils.data_prep import import_csv, clean_data_pinecone_schema, generate_embeddings_and_add_to_df
from utils.split_spaces import split_kb_by_space
import sys

# load environment variables
load_dotenv(find_dotenv())

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

def process_space_file(file_path, index, model_for_openai_embedding, namespace_name):
    """Process a single space's CSV file and upsert to Pinecone."""
    print(f"\nProcessing {file_path}...")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    print(f"Read {len(df)} records for space {df['space'].iloc[0] if 'space' in df.columns else 'default-index'}")
    
    # Generate embeddings and prepare for upsert
    df_with_embeddings = prepare_data_for_upsert(df, model_for_openai_embedding)
    if df_with_embeddings is None or len(df_with_embeddings) == 0:
        print("Error: No valid data to upsert")
        return False
        
    # Upsert the data to Pinecone
    success, count, skipped, metadata_errors = upsert_data(index, df_with_embeddings, namespace_name)
    
    if success:
        print(f"Successfully processed {count} records (skipped: {skipped}, metadata errors: {metadata_errors})")
        return True
    else:
        print(f"Failed to process file {file_path}")
        return False

def main():
    # Load environment variables
    load_dotenv()
    
    # Get command line arguments
    reset_index = "--reset" in sys.argv
    model_for_openai_embedding = "text-embedding-ada-002"
    
    # Initialize Pinecone
    if reset_index:
        print("Resetting Pinecone index...")
        # This deletes the index and all namespaces.
        delete_pinecone_index('demo-agi-testing')
    
    index = init_pinecone()
    if not index:
        print("Failed to initialize Pinecone")
        return
    
    # First, split the data by space if kb.csv exists
    if os.path.exists('data/kb.csv'):
        print("Splitting data by space...")
        split_kb_by_space()
    
    # Process each space file
    spaces_dir = 'data/spaces'
    if not os.path.exists(spaces_dir):
        print(f"Error: {spaces_dir} directory not found")
        return
    
    # Get list of space files (excluding all.csv)
    space_files = [f for f in os.listdir(spaces_dir) 
                   if f.endswith('.csv') and f != 'all.csv']
    
    print(f"\nFound {len(space_files)} space files to process")
    
    # Process each space file
    success_count = 0
    for space_file in space_files:
        file_path = os.path.join(spaces_dir, space_file)
        # Filename excluding the path and the dot extension.
        namespace_name = os.path.splitext(os.path.basename(space_file))[0]
        if process_space_file(file_path, index, model_for_openai_embedding, namespace_name):
            success_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed {success_count} out of {len(space_files)} spaces")

if __name__ == "__main__":
    main()
