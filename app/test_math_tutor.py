"""
Simple test script for the MathTutor platform.

This script tests basic functionality of the MathTutor platform with example problems.
"""
import sys
import os
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Import the solve_problem function from main
from app.main import solve_problem

# Test cases covering different problem types relevant to BBA students
test_cases = [
    {
        "name": "Finance - Present Value",
        "problem": "What is the present value of $10,000 to be received in 5 years, assuming an interest rate of 8% per year?",
        "expected_subject": "Finance"
    },
    {
        "name": "Statistics - Mean Calculation",
        "problem": "Calculate the mean and standard deviation of the following data set: 10, 15, 20, 25, 30.",
        "expected_subject": "Statistics"
    },
    {
        "name": "Economics - Price Elasticity",
        "problem": "If the price of a product increases from $20 to $25 and the quantity demanded decreases from 100 units to 80 units, what is the price elasticity of demand?",
        "expected_subject": "Economics"
    },
    {
        "name": "Accounting - Depreciation",
        "problem": "A company purchases equipment for $50,000 with an expected useful life of 5 years and a salvage value of $5,000. Calculate the annual depreciation expense using the straight-line method.",
        "expected_subject": "Accounting"
    },
    {
        "name": "General Question",
        "problem": "What is the difference between FIFO and LIFO inventory accounting methods?",
        "expected_subject": "Accounting",
        "expected_type": "general"
    }
]

def run_basic_test():
    """Run basic test on solve_problem function"""
    print("=== MathTutor Basic Test ===")
    print(f"Testing {len(test_cases)} problems...\n")
    
    session_id = f"test_session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"Test {i+1}: {test_case['name']}")
        print(f"Problem: {test_case['problem']}")
        
        # Process the problem
        try:
            start_time = datetime.now()
            result = solve_problem(
                problem_text=test_case['problem'],
                session_id=session_id
            )
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Print basic info
            print(f"Processing time: {processing_time:.2f} seconds")
            print(f"Problem ID: {result.get('problem_id', 'N/A')}")
            
            # Check if solution or hint was generated
            if result.get('solution_steps'):
                print(f"Solution generated: {len(result['solution_steps'])} steps")
            
            if result.get('current_hint'):
                print(f"Hint generated: Level {result.get('hint_level', 'N/A')}")
            
            # Add result to list
            results.append({
                "test_case": test_case,
                "result": result,
                "processing_time": processing_time,
                "success": True
            })
            
        except Exception as e:
            print(f"Error: {str(e)}")
            results.append({
                "test_case": test_case,
                "error": str(e),
                "success": False
            })
        
        print("-" * 60)
    
    # Print summary
    print("\n=== Test Summary ===")
    successful = sum(1 for r in results if r["success"])
    print(f"Total tests: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    
    if successful > 0:
        avg_time = sum(r["processing_time"] for r in results if r["success"]) / successful
        print(f"Average processing time: {avg_time:.2f} seconds")
    
    return results

def test_hint_progression():
    """Test the hint progression functionality"""
    print("\n=== Testing Hint Progression ===")
    session_id = f"test_session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    problem = "Calculate the present value of $5,000 to be received in 3 years with an interest rate of 7%."
    
    print(f"Problem: {problem}")
    
    # Initial solution generation
    result = solve_problem(
        problem_text=problem,
        session_id=session_id
    )
    
    problem_id = result.get("problem_id")
    print(f"Problem ID: {problem_id}")
    
    # Request hints in sequence
    for i in range(1, 4):
        print(f"\nRequesting hint level {i}...")
        result = solve_problem(
            problem_text=problem,
            session_id=session_id,
            problem_id=problem_id,
            attempt_number=i+1,
            requested_action="hint"
        )
        
        if result.get("current_hint"):
            print(f"Hint level {result.get('hint_level')}:")
            print(result["current_hint"])
        else:
            print("No hint generated.")
    
    # Finally request full solution
    print("\nRequesting full solution...")
    result = solve_problem(
        problem_text=problem,
        session_id=session_id,
        problem_id=problem_id,
        attempt_number=5,
        requested_action="solution"
    )
    
    if result.get("solution_steps"):
        print(f"Solution has {len(result['solution_steps'])} steps")
        print(f"First step: {result['solution_steps'][0][:100]}...")
    else:
        print("No solution generated.")

def test_answer_assessment():
    """Test the answer assessment functionality"""
    print("\n=== Testing Answer Assessment ===")
    session_id = f"test_session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    problem = "What is 25 + 17?"
    
    print(f"Problem: {problem}")
    
    # Generate solution first
    result = solve_problem(
        problem_text=problem,
        session_id=session_id
    )
    
    problem_id = result.get("problem_id")
    
    # Test cases: correct, close, and wrong answers
    test_answers = [
        {"answer": "42", "expected": "correct"},
        {"answer": "41", "expected": "close"},
        {"answer": "25", "expected": "wrong"}
    ]
    
    for i, test in enumerate(test_answers):
        print(f"\nTesting answer: {test['answer']} (expected: {test['expected']})")
        
        result = solve_problem(
            problem_text=problem,
            session_id=session_id,
            problem_id=problem_id,
            attempt_number=i+2,
            user_answer=test["answer"],
            requested_action="check_answer"
        )
        
        print(f"Is correct: {result.get('is_correct')}")
        if result.get("feedback"):
            print(f"Feedback: {result['feedback']}")
        
        if result.get("suggestion"):
            print(f"System suggestion: {result['suggestion']}")
            if result.get("suggestion_message"):
                print(f"Suggestion message: {result['suggestion_message']}")

if __name__ == "__main__":
    run_basic_test()
    test_hint_progression()
    test_answer_assessment() 