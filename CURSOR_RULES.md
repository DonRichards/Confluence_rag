# Cursor Rules for RAG Application

## Environment Setup
- All API keys must be stored in .env file
- Never commit API keys to git
- Required environment variables: OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME

## Database Management
- Run `pipenv run python update_database.py --reset --csv-file=path/to/file.csv` for schema changes
- Use `pipenv run python update_database.py --verify-only` to check database status without making changes
- Run `pipenv run python app_pinecone_openai.py --debug-pinecone` to diagnose connection issues

## Data Handling
- Content exceeding 30KB will be truncated automatically
- CSV files require 'id', 'content', and 'url'/'tiny_link' columns
- HTML content is supported but may affect embedding quality
- Token limit errors are handled with progressive truncation

## Error Management
- Use --debug-pinecone flag for connection issues
- Check console output for truncation warnings
- Most metadata size errors are handled automatically
- Failed embeddings are reported in the summary statistics

## Performance Optimization
- Pre-process large files in batches (200 records per batch)
- Update vector database weekly to maintain freshness
- Pre-truncate large text content before embedding to reduce OpenAI API costs
- Consider caching common queries for high-traffic applications

## Security Considerations
- Use read-only Confluence credentials when possible
- Implement namespace isolation for multi-tenant deployments
- Verify all API permissions are properly scoped
- Rotate API keys periodically

## Maintenance Procedures
- Monitor vector counts regularly to ensure database growth remains manageable
- Log all database operations for audit purposes
- Check for outdated vectors and update as needed
- Periodically clean up unused vectors to reduce storage costs 