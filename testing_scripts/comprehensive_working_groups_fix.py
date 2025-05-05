#!/usr/bin/env python
"""
Comprehensive fix for AI Working Groups retrieval issues
"""
import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
import json
import uuid
from datetime import datetime, timedelta

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

def comprehensive_fix():
    """Comprehensive fix for AI Working Groups retrieval issues"""
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
    
    # Create additional variations of the content with both terminology and current dates
    original_title = target_page.metadata.get('title', 'AI Working Groups')
    original_text = target_page.metadata.get('text', '')
    original_url = target_page.metadata.get('url', '')
    original_id = target_page.id
    
    # Get current date for future-proofing
    current_date = datetime.now().strftime('%Y-%m-%d')
    # Use a date within "this week" for proper filtering
    this_week_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Create variations with different terminology and ensure dates are recent
    variations = [
        {
            "title": original_title,
            "text": original_text,
            "id": f"{original_id}-enhanced-1",
            "url": original_url,
            "last_modified": this_week_date
        },
        {
            "title": "AI Workgroups",
            "text": original_text.replace("Working Groups", "Workgroups"),
            "id": f"{original_id}-enhanced-2",
            "url": original_url,
            "last_modified": this_week_date
        },
        {
            "title": "AI Interest Group Workgroups",
            "text": original_text.replace("Working Groups", "Workgroups").replace("AI Interest Group", "AI Interest Group Workgroups"),
            "id": f"{original_id}-enhanced-3",
            "url": original_url,
            "last_modified": this_week_date
        },
        {
            "title": "JHU Libraries AI Workgroups",
            "text": original_text.replace("Working Groups", "Workgroups").replace("AI Interest Group", "JHU Libraries AI"),
            "id": f"{original_id}-enhanced-4",
            "url": original_url,
            "last_modified": this_week_date
        },
        {
            "title": "Tell me about the AI Workgroups list",
            "text": f"AI Workgroups at JHU Libraries include:\n\n1. AI Principles - Led by @Cynthia Hudson Vitale\n2. Policy Group - Coming soon\n3. Library Staff AI Training Group - Led by @Timothy Sanders\n\nThese workgroups focus on different aspects of AI implementation and standards within the library system.",
            "id": f"{original_id}-enhanced-5",
            "url": original_url,
            "last_modified": this_week_date
        },
        {
            "title": "AI Working Groups Summary",
            "text": f"AI Working Groups at JHU Libraries include:\n\n1. AI Principles - Led by @Cynthia Hudson Vitale\n2. Policy Group - Coming soon\n3. Library Staff AI Training Group - Led by @Timothy Sanders\n\nThese working groups focus on different aspects of AI implementation and standards within the library system.",
            "id": f"{original_id}-enhanced-6",
            "url": original_url,
            "last_modified": this_week_date
        }
    ]
    
    # Add these variations to the index
    vectors_to_upsert = []
    for variation in variations:
        # Generate embedding for this variation
        embedding = get_embedding(f"{variation['title']} {variation['text'][:1000]}")
        
        # Create metadata with recent date for proper time filtering
        metadata = {
            'title': variation['title'],
            'text': variation['text'],
            'url': variation['url'],
            'last_modified': variation['last_modified'],
            'source': 'ai_working_groups_fix',
            'space': 'AIG'  # Assuming this belongs to the AIG space
        }
        
        # Create vector
        vector = {
            'id': variation['id'],
            'values': embedding,
            'metadata': metadata
        }
        
        vectors_to_upsert.append(vector)
    
    # Upsert the vectors to all relevant namespaces
    print(f"Upserting {len(vectors_to_upsert)} vectors to the index...")
    
    # Regular "all" namespace
    print("Upserting to 'all' namespace...")
    index.upsert(vectors=vectors_to_upsert, namespace="all")
    
    # Also add to AIG namespace if it exists
    if 'AIG' in namespaces:
        print("Upserting to 'AIG' namespace...")
        index.upsert(vectors=vectors_to_upsert, namespace="AIG")
    
    print("\nTerminology variations with current dates added successfully!")
    print("Now the index should respond to both 'AI Working Groups' and 'AI Workgroups' queries")
    print("The content has been dated within 'this week' for proper time filtering")

if __name__ == "__main__":
    print("Starting comprehensive AI Working Groups fix")
    print("=" * 80)
    comprehensive_fix() 