# Incremental Updates for RAG System

## Overview

The RAG system now supports incremental updates to efficiently manage the synchronization between Confluence content and the Pinecone vector database. 

Instead of reprocessing all content during each update, the system tracks when content was last updated and only processes documents that have been modified since the last synchronization.

## How It Works

1. **Timestamp Tracking**:
   - The system maintains a `last_update_timestamp.json` file to record when the last successful update occurred.
   - This timestamp is stored in ISO 8601 format in UTC timezone.

2. **Confluence Content Extraction**:
   - When fetching content from Confluence, the system captures the `last_modified` timestamp for each page.
   - This timestamp comes from the `version.when` field in the Confluence API response.

3. **Filtering During Updates**:
   - When the `--incremental` flag is used, the system compares each document's `last_modified` timestamp with the stored timestamp.
   - Only documents modified since the last update are processed and sent to OpenAI for embedding generation.
   - This significantly reduces API calls, processing time, and costs.

## Benefits

- **Efficiency**: Processes only what has changed, saving time and computational resources.
- **Cost Reduction**: Minimizes OpenAI API calls for embedding generation.
- **Up-to-date Knowledge**: Ensures the vector database stays current with Confluence content.
- **Performance**: Reduces the load on both Confluence and Pinecone during updates.

## How to Use

### Running Incremental Updates

Use the provided shell script:

```bash
./update_incrementally.sh
```

Or run manually:

```bash
# Step 1: Update content from Confluence
pipenv run python app_confluence.py

# Step 2: Perform incremental update
pipenv run python update_database.py --incremental
```

### Full Reindexing

If you need to reindex all content (e.g., after schema changes):

```bash
pipenv run python update_database.py --reset
```

### Checking Last Update Time

To see when the system was last updated:

```bash
cat last_update_timestamp.json
```

## Troubleshooting

- If `last_modified` timestamps are missing in your Confluence data, the incremental update will process all records.
- If no previous timestamp exists, the system will perform a full update.
- If you encounter issues with incremental updates, you can always run a full update with `--reset` flag.

## Technical Implementation

The incremental update feature is implemented across two main files:

1. **app_confluence.py**:
   - Captures and includes `last_modified` timestamps in the extracted Confluence content.

2. **update_database.py**:
   - Stores the timestamp of successful updates in `last_update_timestamp.json`.
   - Reads the timestamp file at the start of incremental updates.
   - Filters content based on modification dates when the `--incremental` flag is used.

## Scheduling Updates

For automated updates, consider setting up a cron job:

```bash
# Example: Run incremental updates every hour
0 * * * * cd /path/to/rag && ./update_incrementally.sh >> /path/to/logs/update.log 2>&1
``` 