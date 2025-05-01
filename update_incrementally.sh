#!/bin/bash

set -e

echo "Starting incremental update of Confluence content and embeddings..."

# Step 1: Fetch the latest content from Confluence
echo "Step 1: Fetching latest content from Confluence..."
pipenv run python app_confluence.py

# Step 2: Run the update_database.py script with the incremental flag
echo "Step 2: Updating database incrementally..."
pipenv run python update_database.py --incremental

echo "Incremental update completed!" 