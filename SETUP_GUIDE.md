# RAG System Setup and Usage Guide

This guide will help you set up and use our enhanced Retrieval Augmented Generation (RAG) system for Confluence content.

## System Overview

Our RAG system connects Confluence to Pinecone (vector database) and OpenAI to create an intelligent knowledge assistant that can:

1. Answer questions about your Confluence content
2. Track changes across time periods
3. Understand the structure and relationships in your documents
4. Follow hierarchical document organization
5. Provide space-specific and time-filtered searches

## Installation

### Prerequisites

- Python 3.9+ installed
- Pinecone account and API key
- OpenAI account and API key
- Confluence access with appropriate permissions

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd rag
```

### Step 2: Install Dependencies

Using pip:
```bash
pip install -r requirements.txt
```

Or using Pipenv (recommended):
```bash
pip install pipenv
pipenv install
```

### Step 3: Configure Environment Variables

Copy the sample environment file:
```bash
cp sample.env .env
```

Edit `.env` and add your credentials:
```
# Confluence settings
CONFLUENCE_URL=https://your-instance.atlassian.net
CONFLUENCE_API_TOKEN=your-api-token
CONFLUENCE_USERNAME=your-email@example.com

# OpenAI settings
OPENAI_API_KEY=your-openai-api-key

# Pinecone settings
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=your-pinecone-environment
PINECONE_INDEX_NAME=your-index-name

# Optional settings
DEFAULT_SPACE=your-default-space
SPACES=space1,space2,space3
```

## Initial Setup

### Step 1: Extract Confluence Content

This will extract content from your Confluence spaces:

```bash
pipenv run python app_confluence.py
```

### Step 2: Create and Populate Pinecone Index

```bash
pipenv run python update_database.py --reset
```

This will:
- Initialize a Pinecone index
- Generate embeddings for your Confluence content
- Upload the embeddings to Pinecone

## Usage

### Running the Chat Interface

Start the Gradio web interface:

```bash
pipenv run python app_pinecone_openai.py
```

Access the chat interface at http://localhost:7860

### Search Commands

The following special commands can be used in the chat:

- `search in space: SPACE_NAME` - Filter searches to a specific space
- `reset space` - Clear space filtering
- `last week` or `this month` - Apply time-based filtering
- `reset time` - Clear time filtering
- `reset all` or `clear filters` - Clear all filters

### Example Queries

- "What are the latest meeting notes from the Engineering team?"
- "Show me documentation about the authentication system"
- "Search in space: MARKETING What campaigns are planned for Q4?"
- "What changes were made to the project this week?"
- "Who is responsible for the database migration task?"

## Updating Content

### Incremental Updates

To efficiently update only changed content:

```bash
./update_incrementally.sh
```

Or manually:

```bash
pipenv run python app_confluence.py
pipenv run python update_database.py --incremental
```

### Full Reindexing

If you need to completely rebuild the index:

```bash
pipenv run python update_database.py --reset
```

## New Features

### Enhanced Search Capabilities

- **Hybrid Search**: Combines vector similarity and BM25 keyword matching
- **Query Variations**: Automatically generates variations of user queries
- **Recency Boost**: Prioritizes more recent content
- **Re-ranking**: Enhances results based on term matching and relevance

### Structured Content Understanding

- **Document Hierarchy**: Preserves and understands document structure
- **Relationship Tracking**: Identifies connections between documents
- **Meeting Notes Analysis**: Extracts action items and decisions
- **Cross-Document Synthesis**: Combines information across related documents

### Space and Time Filtering

- **Persistent Space Selection**: Remembers which space to search throughout a conversation
- **Time-Based Filters**: Understands and maintains time constraints across queries
- **Filter Commands**: Simple natural language commands to set and clear filters

## Advanced Configuration

### Embedding Model Selection

The system uses OpenAI's `text-embedding-3-large` model for embeddings. To change this:

1. Edit `utils/openai_logic.py` to update `DEFAULT_EMBEDDING_MODEL`
2. Edit `update_database.py` to update the `model_name` variable

### Response Model Selection

The system uses `gpt-4-turbo-preview` for generating responses. To change this:

1. Edit `utils/openai_logic.py` to update `DEFAULT_CHAT_MODEL`

### Custom System Prompts

To customize how the system responds:

1. Edit the `system_message` variable in `app_pinecone_openai.py`

## Troubleshooting

### Common Issues

1. **Authentication Errors**: Check your API keys and credentials in `.env`
2. **No Results Found**: Verify that content was successfully extracted and indexed
3. **Index Not Found**: Check that your Pinecone index name matches in `.env`
4. **Space Not Found**: Confirm the space exists and is accessible to your Confluence user

### Debugging

Run these test scripts to diagnose issues:

```bash
# Check Pinecone index
pipenv run python testing_scripts/check_index_stats.py

# Test query functionality
pipenv run python testing_scripts/check_query.py

# Verify space persistence
pipenv run python testing_scripts/test_space_persistence.py
```

## Maintenance

- Run incremental updates regularly to keep content fresh
- Monitor your Pinecone and OpenAI usage
- Check logs in `conversation_logs/` for errors or unexpected behavior 