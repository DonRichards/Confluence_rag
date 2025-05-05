#!/usr/bin/env python3

import os
import sys
import json

# Add parent directory to path to import from project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
import app_pinecone_openai

# Load environment variables
load_dotenv()

def test_space_persistence():
    """Test the persistent space selection feature"""
    print("\n=== Testing Persistent Space Selection ===\n")
    
    # Reset global variables to ensure clean test
    app_pinecone_openai.current_history = []
    app_pinecone_openai.session_id = None
    app_pinecone_openai.current_space = None
    
    # Initialize an empty history
    history = []
    
    # Step 1: Verify no space is selected initially
    print("Step 1: Initial state (no space selected)")
    print(f"Current space: {app_pinecone_openai.current_space}")
    assert app_pinecone_openai.current_space is None, "Expected no space to be selected initially"
    print("✓ No space is selected initially\n")
    
    # Step 2: Set a space with "search in space: SPACE_NAME"
    print("Step 2: Setting space with 'search in space: SPACE_NAME'")
    test_space = "DataServices"
    message = f"search in space: {test_space} what resources are available?"
    print(f"Message: '{message}'")
    
    # Process the message
    history = app_pinecone_openai.chat_function(message, history)
    
    # Verify space was set
    print(f"Current space after query: {app_pinecone_openai.current_space}")
    assert app_pinecone_openai.current_space == test_space, f"Expected space to be set to {test_space}"
    print(f"✓ Space was correctly set to '{test_space}'\n")
    
    # Step 3: Make another query without specifying space
    print("Step 3: Making a query without specifying space")
    message = "what training is available?"
    print(f"Message: '{message}'")
    
    # Process the message
    history = app_pinecone_openai.chat_function(message, history)
    
    # Verify space is still set
    print(f"Current space after second query: {app_pinecone_openai.current_space}")
    assert app_pinecone_openai.current_space == test_space, f"Expected space to still be {test_space}"
    print(f"✓ Space remained set to '{test_space}'\n")
    
    # Step 4: Reset the space
    print("Step 4: Resetting space")
    message = "reset space"
    print(f"Message: '{message}'")
    
    # Process the message
    history = app_pinecone_openai.chat_function(message, history)
    
    # Verify space was reset
    print(f"Current space after reset: {app_pinecone_openai.current_space}")
    assert app_pinecone_openai.current_space is None, "Expected space to be reset to None"
    print("✓ Space was correctly reset\n")
    
    # Step 5: Set space using alternative format
    print("Step 5: Setting space with alternative format 'in SPACE: query'")
    test_space = "DIAS"
    message = f"in {test_space}: what projects are ongoing?"
    print(f"Message: '{message}'")
    
    # Process the message
    history = app_pinecone_openai.chat_function(message, history)
    
    # Verify space was set
    print(f"Current space after alternative format: {app_pinecone_openai.current_space}")
    assert app_pinecone_openai.current_space == test_space, f"Expected space to be set to {test_space}"
    print(f"✓ Space was correctly set to '{test_space}' using alternative format\n")
    
    print("All tests passed successfully! Space persistence feature is working correctly.")
    
    # Clean up after test
    app_pinecone_openai.current_history = []
    app_pinecone_openai.session_id = None
    app_pinecone_openai.current_space = None
    
    return True

if __name__ == "__main__":
    try:
        test_space_persistence()
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 