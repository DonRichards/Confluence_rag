import pandas as pd
import os
from dotenv import load_dotenv

def split_kb_by_space():
    # Load environment variables
    load_dotenv()
    
    print("Starting to split knowledge base by space...")
    
    # Create spaces directory if it doesn't exist
    os.makedirs('data/spaces', exist_ok=True)
    
    # Read the original CSV
    try:
        df = pd.read_csv('data/kb.csv')
        print(f"Read {len(df)} records from kb.csv")
        print(f"Columns found: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error reading kb.csv: {e}")
        return
    
    # Create a copy of all data
    df.to_csv('data/spaces/all.csv', index=False)
    print(f"Created backup all.csv with {len(df)} records")
    
    # Get the configured spaces
    spaces = os.getenv('SPACES', '').split(',')
    spaces = [space.strip() for space in spaces if space.strip()]
    print(f"Found {len(spaces)} configured spaces")
    
    # Print unique space keys in the data
    if 'space_key' in df.columns:
        unique_spaces = df['space_key'].unique()
        print(f"\nFound {len(unique_spaces)} unique space keys in data:")
        for space in unique_spaces:
            count = len(df[df['space_key'] == space])
            print(f"  {space}: {count} records")
    
    # Track what we've processed
    processed_indices = set()
    
    # Process each space
    for space in spaces:
        print(f"\nProcessing space: {space}")
        
        # Filter rows for this space using space_key column
        if 'space_key' in df.columns:
            space_mask = df['space_key'].str.upper() == space.upper()
        else:
            space_mask = pd.Series([False] * len(df))
        
        space_df = df[space_mask].copy()
        
        if not space_df.empty:
            # Add space column if it doesn't exist
            space_df['space'] = space
            # Save to CSV
            output_file = f'data/spaces/{space}.csv'
            space_df.to_csv(output_file, index=False)
            print(f'Created {space}.csv with {len(space_df)} records')
            
            # Track these indices as processed
            processed_indices.update(space_df.index)
        else:
            print(f'No records found for space {space}')
    
    # Remaining rows go to default
    default_df = df[~df.index.isin(processed_indices)].copy()
    default_df['space'] = 'default-index'
    default_df.to_csv('data/spaces/default-index.csv', index=False)
    print(f'\nCreated default-index.csv with {len(default_df)} records')
    
    print("\nSummary:")
    print(f"Total records: {len(df)}")
    print(f"Records assigned to spaces: {len(processed_indices)}")
    print(f"Records in default: {len(default_df)}")
    
    # Print sample unmatched records
    if len(default_df) > 0:
        print("\nSample unmatched records:")
        sample = default_df.head(5)
        for _, row in sample.iterrows():
            print(f"  Space Key: {row.get('space_key', 'N/A')}, Space Name: {row.get('space_name', 'N/A')}")

if __name__ == "__main__":
    split_kb_by_space() 