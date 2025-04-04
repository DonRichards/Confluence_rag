import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from utils.pinecone_logic import get_pinecone_index, upsert_data
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
    upsert_data(index, df)
    
    print("Database update completed successfully!")
    return True

if __name__ == "__main__":
    # Check if csv file was provided as an argument
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        update_database(csv_file)
    else:
        update_database()