#!/usr/bin/env python3

import os
import sys
import re

# Add parent directory to path to import from project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
import app_pinecone_openai

# Load environment variables
load_dotenv()

def test_time_persistence():
    """Test the time filter persistence feature"""
    print("\n=== Testing Time Filter Persistence ===\n")
    
    # Reset global variables to ensure clean test
    app_pinecone_openai.current_history = []
    app_pinecone_openai.session_id = None
    app_pinecone_openai.current_space = None
    app_pinecone_openai.current_time_filter = None
    
    # Initialize an empty history
    history = []
    
    # Step 1: Verify no time filter is set initially
    print("Step 1: Initial state (no time filter set)")
    print(f"Current time filter: {app_pinecone_openai.current_time_filter}")
    assert app_pinecone_openai.current_time_filter is None, "Expected no time filter to be set initially"
    print("✓ No time filter is set initially\n")
    
    # Step 2: Set a time filter with explicit mention
    print("Step 2: Setting time filter with explicit mention")
    test_message = "What changed this week in the AI space?"
    print(f"Message: '{test_message}'")
    
    # Process the message
    history = app_pinecone_openai.chat_function(test_message, history)
    
    # Verify time filter was set
    print(f"Current time filter after query: {app_pinecone_openai.current_time_filter}")
    assert app_pinecone_openai.current_time_filter == "this week", f"Expected time filter to be set to 'this week'"
    print(f"✓ Time filter was correctly set to 'this week'\n")
    
    # Step 3: Make a follow-up query without specifying time
    print("Step 3: Making a follow-up query without specifying time")
    follow_up_message = "Summarize these changes"
    print(f"Message: '{follow_up_message}'")
    
    # Process the message
    history = app_pinecone_openai.chat_function(follow_up_message, history)
    
    # Verify time filter is still applied
    print(f"Current time filter after follow-up: {app_pinecone_openai.current_time_filter}")
    assert app_pinecone_openai.current_time_filter == "this week", f"Expected time filter to still be 'this week'"
    print(f"✓ Time filter remained set to 'this week'\n")
    
    # Step 4: Reset the time filter
    print("Step 4: Resetting time filter")
    reset_message = "reset time"
    print(f"Message: '{reset_message}'")
    
    # Process the message
    history = app_pinecone_openai.chat_function(reset_message, history)
    
    # Verify time filter was reset
    print(f"Current time filter after reset: {app_pinecone_openai.current_time_filter}")
    assert app_pinecone_openai.current_time_filter is None, "Expected time filter to be reset to None"
    print("✓ Time filter was correctly reset\n")
    
    # Step 5: Test with different time format
    print("Step 5: Setting time filter with alternative format")
    alt_message = "Show documents modified in the past 7 days"
    print(f"Message: '{alt_message}'")
    
    # Process the message
    history = app_pinecone_openai.chat_function(alt_message, history)
    
    # Verify time filter was set
    print(f"Current time filter after alternative format: {app_pinecone_openai.current_time_filter}")
    assert app_pinecone_openai.current_time_filter == "past 7 days", f"Expected time filter to be set to 'past 7 days'"
    print(f"✓ Time filter was correctly set to 'past 7 days' using alternative format\n")
    
    # Step 6: Test reset all command
    print("Step 6: Setting both space and time filters, then resetting all")
    
    # Set space filter
    space_message = "search in space: DataServices what's new?"
    print(f"Message: '{space_message}'")
    history = app_pinecone_openai.chat_function(space_message, history)
    
    # Verify both filters are set
    print(f"Current space: {app_pinecone_openai.current_space}")
    print(f"Current time filter: {app_pinecone_openai.current_time_filter}")
    assert app_pinecone_openai.current_space == "DataServices", "Expected space to be set to 'DataServices'"
    assert app_pinecone_openai.current_time_filter == "past 7 days", "Expected time filter to still be 'past 7 days'"
    
    # Reset all filters
    reset_message = "reset all"
    print(f"Message: '{reset_message}'")
    history = app_pinecone_openai.chat_function(reset_message, history)
    
    # Verify all filters were reset
    print(f"Current space after reset all: {app_pinecone_openai.current_space}")
    print(f"Current time filter after reset all: {app_pinecone_openai.current_time_filter}")
    assert app_pinecone_openai.current_space is None, "Expected space to be reset to None"
    assert app_pinecone_openai.current_time_filter is None, "Expected time filter to be reset to None"
    print("✓ All filters were correctly reset\n")
    
    print("All tests passed successfully! Time filter persistence feature is working correctly.")
    
    # Clean up after test
    app_pinecone_openai.current_history = []
    app_pinecone_openai.session_id = None
    app_pinecone_openai.current_space = None
    app_pinecone_openai.current_time_filter = None
    
    return True

if __name__ == "__main__":
    try:
        test_time_persistence()
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 