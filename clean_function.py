#!/usr/bin/env python

# Read the correct function from our fixed_function.txt file
with open('fixed_function.txt', 'r') as f:
    fixed_function = f.read()

# Read the entire data_prep.py file
with open('utils/data_prep.py', 'r') as f:
    content = f.read()

# Find where to insert the function (before generate_embeddings_and_add_to_df)
insertion_point = content.find("def generate_embeddings_and_add_to_df")

if insertion_point > 0:
    # Build the new content
    new_content = content[:insertion_point] + fixed_function + "\n\n# Function to generate embeddings and add to DataFrame\n" + content[insertion_point:]
    
    # Write the updated content back to the file
    with open('utils/data_prep.py', 'w') as f:
        f.write(new_content)
    
    print("Successfully added fixed process_metadata function to data_prep.py")
else:
    print("Could not find insertion point in the file") 