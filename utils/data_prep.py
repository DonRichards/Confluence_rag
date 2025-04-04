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
    df = df[df['content'].notna() & (df['content'] != '')]
    
    if df.empty:
        return "Error: No valid data found in the CSV file after filtering empty content."
    
    # Proceed with the function's main logic
    df['id'] = df['id'].astype(str)
    # Rename the link column to 'source'
    df.rename(columns={link_column: 'source'}, inplace=True)
    df['metadata'] = df.apply(lambda row: json.dumps({'source': row['source'], 'text': row['content']}), axis=1)
    df = df[['id', 'metadata']]
    # print(df.head())
    print("Done: Dataset retrieved")
    return df


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
    # Max tokens the model can handle - setting a safe limit
    max_tokens = 4000  

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            content = row['metadata']
            meta = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for row {index}: {e}")
            continue  # Skip to the next iteration

        text = meta.get('text', '')
        if not text:
            print(f"Warning: Missing 'text' in metadata for row {index}. Skipping.")
            continue

        try:
            # Truncate text if it's likely too long
            if len(text) > max_tokens * 4:  # Approximation: 4 chars ~= 1 token
                text = text[:max_tokens * 4]
                print(f"Warning: Text for row {index} was truncated due to length")
            
            response = create_embeddings(text, model_for_openai_embedding)
            embedding = response   # .data[0].embedding -- Embedding Error? this may be it  # Each in bedding format may be different
            
            # Skip rows with None embeddings
            if embedding is None:
                print(f"Warning: Empty embedding for row {index}. Skipping.")
                continue
                
            df.at[index, 'values'] = embedding
        except Exception as e:
            print(f"Error generating embedding for row {index}: {e}")

    # Remove rows with None values for 'values' column before returning
    df = df.dropna(subset=['values'])
    
    print("Done: Generating embeddings and adding to DataFrame")
    return df

