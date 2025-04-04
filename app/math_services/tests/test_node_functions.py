"""
Test script to manually check each node function with a controlled state.

This script creates a manual MathTutorState and runs each function separately 
to identify where issues might be occurring.
"""

import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal, Union
import getpass
from datetime import datetime

# Load environment variables
load_dotenv()

# Import the MathTutorState model and functions
from app.main import (
    MathTutorState, 
    initialize_session,
    classify_workflow,
    parse_variables,
    define_context,
    generate_solution,
    verify_solution,
    regenerate_solution,
    generate_hints,
    provide_progressive_hint,
    process_user_answer,
    process_general_question,
    determine_next_suggestion,
    format_response,
    record_attempt
)

def get_api_key():
    """Make sure we have an API key"""
    if not os.getenv("OPENAI_API_KEY"):
        api_key = getpass.getpass("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    return os.getenv("OPENAI_API_KEY")

def print_state_info(state, label="State"):
    """Print key information about the state for debugging"""
    print(f"\n{label} Information:")
    print(f"Problem: {state.problem}")
    print(f"Workflow Type: {state.workflow_type}")
    print(f"Variables: {state.variables}")
    print(f"Subject Area: {state.subject_area}")
    print(f"Solution Steps: {len(state.solution_steps)} steps" if state.solution_steps else "No solution steps")
    print(f"Final Answer: {state.final_answer}")
    print(f"Verification Results: {len(state.verification_results)} results" if state.verification_results else "No verification results")
    print(f"Needs Regeneration: {state.needs_regeneration}")
    print(f"Hints: {len(state.hints)} hints" if state.hints else "No hints")
    print(f"Current Hint Level: {state.current_hint_level}")

def test_function(state, function, name):
    """Test a single function with proper error handling"""
    print(f"\n=== Testing {name} ===")
    try:
        result_state = function(state)
        print(f"✅ {name} executed successfully")
        print_state_info(result_state, f"After {name}")
        return result_state
    except Exception as e:
        print(f"❌ Error in {name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return state

def main():
    """Run tests for each function with a manually created state"""
    api_key = get_api_key()
    print(f"Using API key: {api_key[:5]}...")
    
    # Create a simple math problem state
    initial_state = MathTutorState(
        problem="What is the present value of $10,000 to be received in 5 years, assuming an interest rate of 8% per year?",
        problem_id=None,
        workflow_type=None,
        session_id=None,
        attempt_number=1
    )
    
    print("Created initial state:")
    print_state_info(initial_state, "Initial State")
    
    # Test each function in sequence
    state = initial_state
    
    # Initialize session
    state = test_function(state, initialize_session, "initialize_session")
    
    # Classify the problem
    state = test_function(state, classify_workflow, "classify_workflow")
    
    # Check if it's a math problem
    if state.workflow_type == "math":
        # Test math workflow functions
        state = test_function(state, parse_variables, "parse_variables")
        state = test_function(state, define_context, "define_context")
        state = test_function(state, generate_solution, "generate_solution")
        state = test_function(state, verify_solution, "verify_solution")
        
        # Only test regeneration if needed
        if state.needs_regeneration:
            state = test_function(state, regenerate_solution, "regenerate_solution")
        
        # Continue with other functions
        state = test_function(state, generate_hints, "generate_hints")
        state = test_function(state, provide_progressive_hint, "provide_progressive_hint")
        
        # Test user answer processing
        state.user_answer = "The present value is $6,805.83"
        state = test_function(state, process_user_answer, "process_user_answer")
        
        # Test suggestion mechanism
        state = test_function(state, determine_next_suggestion, "determine_next_suggestion")
    else:
        # Test general question function
        state = test_function(state, process_general_question, "process_general_question")
    
    # Test final functions
    state = test_function(state, format_response, "format_response")
    state = test_function(state, record_attempt, "record_attempt")
    
    print("\n=== Test Complete ===")
    print(f"Final workflow type: {state.workflow_type}")
    print(f"Solution generated: {'Yes' if state.solution_steps else 'No'}")
    print(f"Final answer: {state.final_answer}")
    print(f"Hints generated: {len(state.hints) if state.hints else 0}")
    
if __name__ == "__main__":
    main() 