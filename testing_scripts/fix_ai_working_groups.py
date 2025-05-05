#!/usr/bin/env python
"""
Script to fix AI Working Groups retrieval issues
"""
import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
import json
import uuid
import re

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()
print("Loading .env environment variables...")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Initialize OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text):
    """Get embedding for text using OpenAI API"""
    response = openai_client.embeddings.create(
        input=text,
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    )
    return response.data[0].embedding

def fix_ai_working_groups():
    """Fix the AI Working Groups retrieval issue by updating the index"""
    index_name = os.getenv("PINECONE_INDEX", "default-index")
    index = pc.Index(index_name)
    
    # Get available namespaces
    stats = index.describe_index_stats()
    namespaces = stats.namespaces
    
    print("Available namespaces:")
    for ns, count in namespaces.items():
        print(f"  {ns}: {count} vectors")
    
    # First check if the page exists in the index
    ai_working_groups_query = "AI Working Groups"
    working_groups_embedding = get_embedding(ai_working_groups_query)
    
    results = index.query(
        vector=working_groups_embedding,
        top_k=5,
        namespace="all",
        include_metadata=True
    )
    
    print(f"Query for 'AI Working Groups' found {len(results.matches)} results")
    
    # Find the specific AI Working Groups page
    target_page = None
    for match in results.matches:
        if hasattr(match, 'metadata') and match.metadata:
            if "AI Working Groups" in match.metadata.get('title', ''):
                target_page = match
                print("Found the AI Working Groups page:")
                print(f"  ID: {match.id}")
                print(f"  Title: {match.metadata.get('title', 'No title')}")
                print(f"  URL: {match.metadata.get('url', 'No URL')}")
                break
    
    if not target_page:
        print("Error: Could not find the AI Working Groups page in the index")
        return
    
    # Create additional variations of the content with both terminology
    original_title = target_page.metadata.get('title', 'AI Working Groups')
    original_text = target_page.metadata.get('text', '')
    original_url = target_page.metadata.get('url', '')
    original_id = target_page.id
    
    # Create variations with different terminology
    variations = [
        {
            "title": original_title,
            "text": original_text,
            "id": original_id,
            "url": original_url
        },
        {
            "title": "AI Workgroups",
            "text": original_text.replace("Working Groups", "Workgroups"),
            "id": f"{original_id}-variation-1",
            "url": original_url
        },
        {
            "title": "AI Interest Group Workgroups",
            "text": original_text.replace("Working Groups", "Workgroups").replace("AI Interest Group", "AI Interest Group Workgroups"),
            "id": f"{original_id}-variation-2",
            "url": original_url
        },
        {
            "title": "JHU Libraries AI Workgroups",
            "text": original_text.replace("Working Groups", "Workgroups").replace("AI Interest Group", "JHU Libraries AI"),
            "id": f"{original_id}-variation-3",
            "url": original_url
        }
    ]
    
    # Add these variations to the index
    vectors_to_upsert = []
    for variation in variations:
        # Generate embedding for this variation
        embedding = get_embedding(f"{variation['title']} {variation['text'][:1000]}")
        
        # Create metadata
        metadata = {
            'title': variation['title'],
            'text': variation['text'],
            'url': variation['url'],
            'last_modified': target_page.metadata.get('last_modified', ''),
            'source': 'ai_working_groups_fix',
            'space': 'AIG'  # Assuming this belongs to the AIG space
        }
        
        # Create vector
        vector = {
            'id': variation['id'] if variation['id'] != original_id else f"{original_id}-enhanced",
            'values': embedding,
            'metadata': metadata
        }
        
        vectors_to_upsert.append(vector)
    
    # Upsert the vectors
    print(f"Upserting {len(vectors_to_upsert)} vectors to the index...")
    
    # Split into batches of 100 if needed
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i+batch_size]
        index.upsert(vectors=batch, namespace="all")
        print(f"Upserted batch {i//batch_size + 1}")
    
    print("\nTerminology variations added successfully!")
    print("Now the index should respond to both 'AI Working Groups' and 'AI Workgroups' queries")

if __name__ == "__main__":
    print("Starting AI Working Groups fix")
    print("=" * 80)
    fix_ai_working_groups() 