#!/usr/bin/env python

with open('utils/data_prep.py', 'r') as f:
    lines = f.readlines()

with open('utils/data_prep.py.new', 'w') as f:
    for line in lines:
        if line.strip() == "try:":
            # Check if previous line has "metadata['space'] = row['space']"
            if any("metadata['space'] = row['space']" in l for l in lines[-5:]):
                # We're in the problematic section
                f.write("        try:\n")
                f.write("            # Convert back to string\n")
                f.write("            return json.dumps(metadata)\n")
                f.write("        except Exception as e:\n")
                f.write("            print(f\"Error processing metadata for row {row['id']}: {str(e)}\")\n")
                f.write("            # Return a basic metadata if processing fails\n")
                f.write("            return json.dumps({\n")
                f.write("                'source': row['source'],\n")
                f.write("                'text': truncate_text(row['content']),\n")
                f.write("                'space': row['space']\n")
                f.write("            })\n")
                
                # Skip the problematic try block and its content
                skip_lines = 8  # Adjust as needed
                for _ in range(skip_lines):
                    next(lines, None)
            else:
                f.write(line)
        else:
            f.write(line)

# Now replace the original file with the fixed one
import os
os.rename('utils/data_prep.py.new', 'utils/data_prep.py')

print("Fixed data_prep.py") 