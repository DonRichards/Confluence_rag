# General purpose

I'm using this to pull all of the pages from Confluence and push it into Pionecone as vector databases and this uses OpenAI to chat with the data. 


## app_confluence.py

This script interacts with the Confluence API to fetch content from specified spaces. It uses environment variables to configure the connection (domain, credentials, space keys) and provides multiple methods to retrieve content:

1. Direct API calls using requests library
2. Atlassian Python API client
3. Content-specific endpoints for pages and blog posts

The script includes features for:
- Pagination handling for large result sets
- Content cleaning and normalization
- Space filtering and management
- Error handling and logging
- Progress tracking with tqdm
- DataFrame creation and manipulation for structured data

The content is processed and prepared for vector database ingestion, with support for:
- HTML cleaning and text extraction
- Label management
- Internal content filtering
- Space-based organization
- Using Confluence space names as Pinecone namespaces for organized vector storage


## utils/pinecone_logic.py

This script manages the interaction with Pinecone vector database, providing core functionality for vector storage and retrieval. It handles index management (creation, deletion), data upsertion with namespace support, and includes robust error handling and progress tracking. The script is designed to work with pandas DataFrames containing vector embeddings and metadata, organizing data by Confluence space names as namespaces. Key features include:

- Index lifecycle management (creation/deletion)
- Batch processing with automatic chunking
- Namespace-based data organization
- Comprehensive error handling and logging
- Progress tracking with tqdm
- Data validation and type checking
- Rate limiting and backoff strategies