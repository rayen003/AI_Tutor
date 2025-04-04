#!/usr/bin/env python3
"""
A simplified version of the math tutor that doesn't use LangGraph.
This script isolates the node functions and calls them directly in sequence
to avoid the 'Can receive only one value per step' error.
"""

import os
import json
import sys
from dotenv import load_dotenv
import getpass
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

# Import the necessary components directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from app.main import (
        MathTutorState,
        initialize_session,
        classify_workflow,
        parse_variables,
        define_context,
        generate_solution,
        verify_solution,
        generate_hints,
        provide_progressive_hint,
        process_user_answer,
        determine_next_suggestion,
        format_response,
        record_attempt
    )
except ModuleNotFoundError:
    # Try direct import if app module doesn't work
    from main import (
        MathTutorState,
        initialize_session,
        classify_workflow,
        parse_variables,
        define_context,
        generate_solution,
        verify_solution,
        generate_hints,
        provide_progressive_hint,
        process_user_answer,
        determine_next_suggestion,
        format_response,
        record_attempt
    )

def get_api_key() -> str:
    """Get the OpenAI API key from environment variables or prompt the user for it."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    return api_key

def print_state_info(state, label="Current State"):
    """Print key information about the state for debugging."""
    print(f"\n=== {label} ===")
    print(f"Problem: {state.problem}")
    print(f"Workflow Type: {state.workflow_type}")
    print(f"Variables: {state.variables}")
    print(f"Solution Steps: {len(state.solution_steps) if state.solution_steps else 0} steps")
    if state.solution_steps:
        for i, step in enumerate(state.solution_steps):
            print(f"  Step {i+1}: {step[:50]}...")
    print(f"Final Answer: {state.final_answer}")
    print(f"Verification Results: {state.verification_results}")
    if state.hints:
        print(f"Hints: {len(state.hints)} available")
        for i, hint in enumerate(state.hints):
            if isinstance(hint, dict) and 'hint' in hint:
                print(f"  Hint {i+1}: {hint['hint'][:50]}...")
            elif isinstance(hint, str):
                print(f"  Hint {i+1}: {hint[:50]}...")
            else:
                print(f"  Hint {i+1}: {str(hint)[:50]}...")
    print(f"Current Hint Level: {state.current_hint_level}")
    print(f"User Answer: {state.user_answer}")
    print(f"Is Correct: {state.is_correct}")
    print(f"Feedback: {state.feedback}")
    print("=" * (len(label) + 8))

def solve_problem_simplified(problem: str) -> MathTutorState:
    """
    Process a math problem in a linear sequence without using the state graph.
    This avoids concurrent state updates that can cause the 'Can receive only one value per step' error.
    """
    # Ensure API key is set
    get_api_key()
    
    # Create initial state
    state = MathTutorState(
        problem=problem,
        workflow_type="math",  # Set default workflow type to match validation
        variables={},
        subject_area="",
        solution_steps=[],
        final_answer="",
        verification_results=[],  # Changed from string to empty list
        hints=[],
        current_hint_level=0,  # Changed from current_hint_index
        user_answer="",
        is_correct=None,  # Added to match the model
        feedback=None,  # Added to match the model
        problem_id=None,
        session_id=None,
        attempt_number=1,
        timestamp=None,
        context=None,
        solution_generated=False,
        solution_revealed=False,
        steps_to_regenerate=[],
        needs_regeneration=False,
        regeneration_attempts=0,
        max_regeneration_attempts=3,
        max_hint_level=3,
        final_response=None,
        error_message=None,
        attempt_history=[],
        requested_action="none",
        system_suggestion="none",
        suggestion_message=None,
        answer_proximity=None
    )
    
    print_state_info(state, "Initial State")
    
    # Step 1: Initialize session
    print("\nTesting initialize_session...")
    try:
        state = initialize_session(state)
        print("✅ initialize_session executed successfully")
    except Exception as e:
        print(f"❌ Error in initialize_session: {str(e)}")
    
    # Step 2: Classify workflow
    print("\nTesting classify_workflow...")
    try:
        state = classify_workflow(state)
        print(f"✅ classify_workflow executed successfully. Workflow type: {state.workflow_type}")
    except Exception as e:
        print(f"❌ Error in classify_workflow: {str(e)}")
    
    # If it's a math problem, continue with math workflow
    if state.workflow_type == "math":
        # Step 3: Parse variables
        print("\nTesting parse_variables...")
        try:
            state = parse_variables(state)
            print(f"✅ parse_variables executed successfully. Variables: {state.variables}")
        except Exception as e:
            print(f"❌ Error in parse_variables: {str(e)}")
        
        # Step 4: Define context
        print("\nTesting define_context...")
        try:
            state = define_context(state)
            print(f"✅ define_context executed successfully. Subject area: {state.subject_area}")
        except Exception as e:
            print(f"❌ Error in define_context: {str(e)}")
        
        # Step 5: Generate solution
        print("\nTesting generate_solution...")
        try:
            state = generate_solution(state)
            print(f"✅ generate_solution executed successfully. Steps: {len(state.solution_steps) if state.solution_steps else 0}")
        except Exception as e:
            print(f"❌ Error in generate_solution: {str(e)}")
        
        # Step 6: Verify solution
        print("\nTesting verify_solution...")
        try:
            state = verify_solution(state)
            print(f"✅ verify_solution executed successfully.")
        except Exception as e:
            print(f"❌ Error in verify_solution: {str(e)}")
        
        # Step 7: Generate hints
        print("\nTesting generate_hints...")
        try:
            state = generate_hints(state)
            print(f"✅ generate_hints executed successfully. Hints: {len(state.hints) if state.hints else 0}")
        except Exception as e:
            print(f"❌ Error in generate_hints: {str(e)}")
        
        # Step 8: Process user answer - simulate a user answer
        user_answer = "The present value is $6,756.30"
        state.user_answer = user_answer
        print(f"\nTesting process_user_answer with answer: '{user_answer}'...")
        try:
            state = process_user_answer(state)
            print(f"✅ process_user_answer executed successfully. Assessment: {state.assessment}")
        except Exception as e:
            print(f"❌ Error in process_user_answer: {str(e)}")
    
    # For general questions
    else:
        # Process general question workflow
        print("\nProcessing general question...")
        
    # Common final steps
    
    # Step 9: Determine next suggestion
    print("\nTesting determine_next_suggestion...")
    try:
        state = determine_next_suggestion(state)
        print(f"✅ determine_next_suggestion executed successfully. Suggestion: {state.next_suggestion}")
    except Exception as e:
        print(f"❌ Error in determine_next_suggestion: {str(e)}")
    
    # Step 10: Format response
    print("\nTesting format_response...")
    try:
        state = format_response(state)
        print(f"✅ format_response executed successfully.")
    except Exception as e:
        print(f"❌ Error in format_response: {str(e)}")
    
    # Step 11: Record attempt
    print("\nTesting record_attempt...")
    try:
        state = record_attempt(state)
        print(f"✅ record_attempt executed successfully. Attempt count: {state.attempt_count}")
    except Exception as e:
        print(f"❌ Error in record_attempt: {str(e)}")
    
    # Final state summary
    print_state_info(state, "Final State")
    
    return state

def main():
    # Test with sample problem
    problem = "What is the present value of $10,000 to be received in 5 years if the interest rate is 8%?"
    print(f"Testing with problem: {problem}")
    
    final_state = solve_problem_simplified(problem)
    
    # Test progressive hint functionality
    print("\n=== Testing Progressive Hints ===")
    state_with_hint = provide_progressive_hint(final_state)
    
    if state_with_hint.hints and len(state_with_hint.hints) > 0:
        current_hint = state_with_hint.hints[state_with_hint.current_hint_level - 1]
        if isinstance(current_hint, dict) and 'hint' in current_hint:
            print(f"Hint provided: {current_hint['hint']}")
        else:
            print(f"Hint provided: {current_hint}")
        
        if len(state_with_hint.hints) > 1 and state_with_hint.current_hint_level < len(state_with_hint.hints):
            state_with_hint = provide_progressive_hint(state_with_hint)
            current_hint = state_with_hint.hints[state_with_hint.current_hint_level - 1]
            if isinstance(current_hint, dict) and 'hint' in current_hint:
                print(f"Next hint provided: {current_hint['hint']}")
            else:
                print(f"Next hint provided: {current_hint}")
    else:
        print("No hints were generated.")
    
    print("\n=== Testing Another User Answer ===")
    state_with_hint.user_answer = "The present value is approximately $6,756"
    updated_state = process_user_answer(state_with_hint)
    print(f"Assessment of new answer: {updated_state.assessment if hasattr(updated_state, 'assessment') else updated_state.feedback}")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 