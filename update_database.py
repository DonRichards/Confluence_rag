import os
import pandas as pd
import argparse
from dotenv import load_dotenv, find_dotenv
from utils.pinecone_logic import get_pinecone_index, upsert_data, verify_pinecone_upsert
from utils.data_prep import import_csv, clean_data_pinecone_schema, generate_embeddings_and_add_to_df
import sys

# load environment variables
load_dotenv(find_dotenv())

def update_database(csv_file="./data/kb.csv"):
    """
    Updates the Pinecone database with data from the specified CSV file.
    """
    print("Starting database update process...")
    
    # Initialize Pinecone
    index_name = os.getenv('PINECONE_INDEX_NAME', 'default-index')
    index, _ = get_pinecone_index(index_name)
    
    if index is None:
        print("Error: Failed to initialize Pinecone index.")
        return False
    
    # Import data
    print(f"Importing data from {csv_file}...")
    df = import_csv(None, csv_file)
    
    # Check if data import was successful
    if df is None or isinstance(df, str):
        print(f"Error: Failed to import data: {df if isinstance(df, str) else 'No data returned'}")
        return False
    
    # Clean data
    print("Cleaning data...")
    df = clean_data_pinecone_schema(df)
    
    # Check if data cleaning was successful
    if df is None or isinstance(df, str):
        print(f"Error: Failed to clean data: {df if isinstance(df, str) else 'No data returned'}")
        return False
    
    # Generate embeddings
    print("Generating embeddings...")
    df = generate_embeddings_and_add_to_df(df, "text-embedding-ada-002")
    
    # Check if embedding generation was successful
    if df is None:
        print("Error: Failed to generate embeddings")
        return False
    
    # Upsert data to Pinecone
    print("Upserting data to Pinecone index...")
    upsert_success, records_upserted = upsert_data(index, df)
    
    if not upsert_success:
        print("Error: Failed to upsert data to Pinecone")
        return False
        
    print(f"Successfully upserted {records_upserted} records to Pinecone")
    
    # Verify the data was successfully added with a sample query
    sample_row = df.iloc[0] if not df.empty else None
    sample_query = None
    
    if sample_row is not None:
        try:
            # Use the first row's content as a sample query
            metadata = eval(sample_row['metadata'])
            if metadata and 'text' in metadata:
                # Take just the first 10 words as the sample query
                sample_text = metadata['text'][:500]
                words = sample_text.split()[:10]
                sample_query = ' '.join(words)
        except Exception as e:
            print(f"Error creating sample query: {e}")
    
    if sample_query:
        print(f"Verifying upsert with sample query: '{sample_query}'")
        if not verify_pinecone_upsert(index, sample_query):
            print("Warning: Upsert verification failed with sample query")
            # Don't return False here as the upsert might have succeeded even if verification fails
    else:
        print("Verifying index statistics only (no sample query available)")
        if not verify_pinecone_upsert(index):
            print("Warning: Upsert verification failed")
            # Don't return False here as the upsert might have succeeded even if verification fails
    
    print("Database update completed successfully!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update Pinecone database with data from a CSV file")
    parser.add_argument("--csv-file", dest="csv_file", default="./data/kb.csv",
                        help="Path to the CSV file containing data to upsert (default: ./data/kb.csv)")
    parser.add_argument("--reset", action="store_true", 
                        help="Reset the database by deleting the index before upserting data")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify the database without updating it")
    
    args = parser.parse_args()
    
    # If verify_only is specified, just check if the index exists and has data
    if args.verify_only:
        print("Verifying Pinecone database...")
        index_name = os.getenv('PINECONE_INDEX_NAME', 'default-index')
        
        try:
            from utils.pinecone_logic import get_pinecone_index, verify_pinecone_upsert
            index, _ = get_pinecone_index(index_name)
            
            if index is None:
                print("Error: Failed to initialize Pinecone index.")
                sys.exit(1)
            
            verification_result = verify_pinecone_upsert(index)
            if verification_result:
                print("Verification successful! Database is ready for queries.")
                sys.exit(0)
            else:
                print("Verification failed. Database may be empty or inaccessible.")
                sys.exit(1)
        except Exception as e:
            print(f"Error during verification: {e}")
            sys.exit(1)
    
    # If reset is specified, delete the index first
    if args.reset:
        print("Resetting database...")
        index_name = os.getenv('PINECONE_INDEX_NAME', 'default-index')
        
        try:
            from utils.pinecone_logic import delete_pinecone_index
            delete_pinecone_index(index_name)
            print(f"Index '{index_name}' has been reset.")
        except Exception as e:
            print(f"Error resetting index: {e}")
            sys.exit(1)
    
    # Check if the CSV file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' does not exist.")
        sys.exit(1)
    
    # Run the update process
    result = update_database(args.csv_file)
    
    if result:
        print("Database update completed successfully!")
        sys.exit(0)
    else:
        print("Database update failed.")
        sys.exit(1)
