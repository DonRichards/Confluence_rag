#!/usr/bin/env python
import re
import ast
import json

# Read the file
with open('utils/data_prep.py', 'r') as f:
    lines = f.readlines()

# Find the start of the process_metadata function
start_line = 0
for i, line in enumerate(lines):
    if line.strip().startswith('def process_metadata(row):'):
        start_line = i
        break

# Find the end of the function (the next function definition)
end_line = len(lines)
for i in range(start_line + 1, len(lines)):
    if line.strip().startswith('def '):
        end_line = i
        break

# Replace the function with a corrected version
fixed_function = '''def process_metadata(row):
    """Process metadata to ensure it's in a consistent format."""
    try:
        # First check if it's a string and convert to dict
        if isinstance(row['metadata'], str):
            try:
                metadata = ast.literal_eval(row['metadata'])
            except:
                # If parsing fails, create a basic metadata dict
                metadata = {
                    'source': row['source'],
                    'text': row['content']
                }
        else:
            # Already a dict
            metadata = row['metadata']
            
        # Ensure text is not too long
        if 'text' in metadata:
            metadata['text'] = truncate_text(metadata['text'])
            
        # Add space information to metadata
        metadata['space'] = row['space']
        
        try:
            # Convert back to string
            return json.dumps(metadata)
        except Exception as e:
            print(f"Error processing metadata for row {row['id']}: {str(e)}")
            # Return a basic metadata if processing fails
            return json.dumps({
                'source': row['source'],
                'text': truncate_text(row['content']),
                'space': row['space']
            })
    except Exception as e:
        print(f"Error in process_metadata: {str(e)}")
        return json.dumps({
            'source': row.get('source', 'unknown'),
            'text': truncate_text(row.get('content', '')),
            'space': row.get('space', 'unknown')
        })
'''

# Combine the parts
fixed_content = ''.join(lines[:start_line]) + fixed_function + ''.join(lines[end_line:])

# Write the fixed content back
with open('utils/data_prep.py', 'w') as f:
    f.write(fixed_content)

print("Fixed indentation issues in data_prep.py") 