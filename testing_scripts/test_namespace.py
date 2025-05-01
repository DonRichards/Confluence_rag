from pinecone.grpc import PineconeGRPC as Pinecone
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os

# load environment variables
load_dotenv(find_dotenv("../.env"))

# Check if values exist for PINECONE_API_KEY and PINECONE_DB_NAME FIRST
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_db_name = os.getenv("PINECONE_DB_NAME")
pinecone_host = os.getenv("PINECONE_HOST")

if not pinecone_api_key or not pinecone_db_name:
    print("PINECONE_API_KEY and PINECONE_DB_NAME must be set in the environment or .env file")
    exit()

# Now initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(host=pinecone_host)

namespace = "AIG"

# Check if namespace exists
# Note: The original check 'if not index.describe_index_stats(namespace=namespace):' might not be the best way.
# describe_index_stats returns stats, not a boolean for existence.
# A better check might involve trying a simple operation or checking the stats dictionary.
try:
    stats = index.describe_index_stats()
    if namespace not in stats.namespaces:
         print(f"Namespace '{namespace}' does not exist in index '{pinecone_db_name}'")
         exit()
except Exception as e:
    print(f"Error checking index/namespace: {e}")
    exit()

print(f"Querying namespace '{namespace}' in index '{pinecone_db_name}'...")

query=f'Tell me about the {namespace}?'

client = OpenAI()
response = client.embeddings.create(
    input=query,
    model="text-embedding-3-small"
)
query_embedding = response.data[0].embedding

# Then use the embedding vector in the Pinecone query
results = index.query(
    namespace=namespace,
    vector=query_embedding,  # Use the embedding vector here
    top_k=2,
    include_metadata=True
)
print(results)

# client = OpenAI()
# response = client.embeddings.create(
#     model="text-embedding-3-small",
#     input=query
# )
# query_embedding = response.data[0].embedding

# # Now use the generated embedding in the query
# query_results = index.query(
#     namespace=namespace,
#     vector=query_embedding,
#     top_k=2,
#     include_values=True,
#     include_metadata=True
# )

# # Extract and print the text content from matches
# for match in query_results.matches:
#     print(f"\nScore: {match.score}")
#     print(f"ID: {match.id}")
#     if hasattr(match, 'metadata') and 'text' in match.metadata:
#         print(f"Text: {match.metadata['text']}")
#     print("-" * 50)