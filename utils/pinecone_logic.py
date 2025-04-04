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
                
            prepped.append({
                'id': row['id'], 
                'values': values,
                'metadata': meta
            })
            
            if len(prepped) >= 200: # batching upserts
                index.upsert(prepped)
                prepped = []
        except Exception as e:
            print(f"Error processing row {i} for upsert: {e}")
            skipped += 1

    # Upsert any remaining entries after the loop
    if len(prepped) > 0:
        index.upsert(prepped)
    
    if skipped > 0:
        print(f"Warning: {skipped} rows were skipped during upsert due to invalid data")
    
    print("Done: Data upserted to Pinecone index")
    return index

