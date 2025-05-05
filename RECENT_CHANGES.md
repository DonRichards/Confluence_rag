# Recent Changes to the RAG System

## New Features

1. **Upgraded Models and Embeddings**
   - Enhanced OpenAI model from text-embedding-ada-002 to text-embedding-3-large
   - Increased token limit to 4096 for chat responses with GPT-4-Turbo
   - Improved embedding dimension and quality for better search results

2. **Advanced Hybrid Search**
   - Implemented BM25/TF-IDF keyword matching alongside vector search
   - Added intelligent query variation generation for more robust retrieval
   - Implemented content-aware re-ranking with recency and term matching boosts
   - Enhanced relevance scoring based on query term coverage

3. **Semantic Document Processing**
   - Added intelligent recognition of document types and structures
   - Enhanced hierarchical content preservation
   - Added relationship tracking between related documents
   - Improved meeting notes analysis with action item extraction

4. **Recursive Context Discovery for Meeting Notes**
   - System now detects when queries are about meeting notes summaries and changes
   - Automatically fetches additional context like project descriptions and previous meetings
   - Provides more complete and accurate summaries by understanding the broader context
   - Identifies recurring topics and tracks progress across multiple meetings
   - Presents information synthesis rather than just individual document summaries

5. **Persistent Space Selection**
   - The system now remembers which space to search in throughout a conversation
   - Once a user specifies "search in space: SPACE_NAME", all subsequent queries will use that space
   - Users can type "reset space" to clear the space selection and search across all spaces again
   - The UI displays the currently selected space and provides reset instructions

6. **Persistent Time Filters**
   - The system now remembers time-based constraints (like "this week" or "recent") across queries
   - Time context is preserved for follow-up questions about changes, updates, or summaries
   - Supports various time formats: "this week", "last month", "past 7 days", etc.
   - Users can clear time filters with "reset time" command
   - UI shows active time filters and provides reset instructions

7. **Enhanced Context Management**
   - Added "reset all" and "clear filters" commands to clear both space and time filters at once
   - Follow-up detection for time-sensitive queries (e.g., "summarize the changes" will maintain previous time context)
   - Improved system prompts to respect active filters in responses

## Bug Fixes

1. **Fixed future date handling in meeting notes**
   - Improved detection and processing of content with incorrect future dates (2025+)
   - System now properly handles dates in metadata and content extraction
   - Enhanced date comparison logic for time-based filtering

2. **Fixed logging error with query results**
   - Modified `log_conversation` function to handle both 3-element and 4-element result tuples
   - Properly converts query results to JSON-serializable format
   - Added error handling for different result formats

3. **Added error handling to query_pinecone function**
   - Added try/except block to catch errors when processing individual matches
   - Prevents one bad result from causing the entire query to fail
   - Added graceful fallback from hybrid search to standard vector search

4. **Improved Confluence terminology in responses**
   - Updated system prompt to use more accurate terminology (pages instead of documents)
   - Enhanced response format for improved readability and accuracy
   - Added contextual information in responses like page descriptions

## Documentation

1. **Added SETUP_GUIDE.md**
   - Comprehensive setup instructions for new installations
   - Usage examples for common tasks
   - Documentation of new features and search commands
   - Troubleshooting section for common issues

2. **Updated README.md**
   - Added overview of new hybrid search capabilities
   - Updated model information and dependencies
   - Added performance benchmarks and recommendations

## Reorganization

1. **Moved test scripts to testing_scripts/ directory**
   - Created proper executable Python scripts with proper imports
   - Made all scripts executable with chmod +x
   - Added command-line argument handling

2. **Test scripts now available:**
   - `testing_scripts/check_index_stats.py` - Check Pinecone index statistics
   - `testing_scripts/check_logging.py` - Test logging functionality
   - `testing_scripts/check_query.py` - Test query functionality with different thresholds
   - `testing_scripts/check_namespace_behavior.py` - Test namespace behavior
   - `testing_scripts/test_space_persistence.py` - Test persistent space selection feature
   - `testing_scripts/test_time_persistence.py` - Test persistent time filter feature

## Usage Notes

- When running test scripts, always use `pipenv run python testing_scripts/script_name.py`
- Or make them executable and run directly with `pipenv run testing_scripts/script_name.py`
- All test scripts include detailed output to help diagnose issues
- Use "reset space", "reset time", or "reset all" to clear active filters in the chatbot
- The hybrid search now requires scikit-learn, which has been added to requirements.txt 