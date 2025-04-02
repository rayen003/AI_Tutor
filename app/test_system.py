"""
Test script for the MathTutor platform.

This script runs a series of test cases to verify that the MathTutor system
is functioning correctly, including math problem solving, hint generation,
and answer assessment.
"""

import sys
import os
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Import the MathTutor system
from app.math_services.main import solve_problem

# Test cases - different types of problems for different subjects
TEST_CASES = [
    {
        "name": "Finance - Present Value",
        "problem": "What is the present value of $10,000 to be received in 5 years, assuming an interest rate of 8% per year?",
        "subject": "Finance"
    },
    {
        "name": "Statistics - Mean and Standard Deviation",
        "problem": "Calculate the mean and standard deviation of the following data set: 10, 15, 20, 25, 30.",
        "subject": "Statistics"
    },
    {
        "name": "Economics - Price Elasticity",
        "problem": "If the price of a product increases from $20 to $25 and the quantity demanded decreases from 100 units to 80 units, what is the price elasticity of demand?",
        "subject": "Economics"
    },
    {
        "name": "Accounting - Depreciation",
        "problem": "A company purchases equipment for $50,000 with an expected useful life of 5 years and a salvage value of $5,000. Calculate the annual depreciation expense using the straight-line method.",
        "subject": "Accounting"
    },
    {
        "name": "General Question",
        "problem": "What is the difference between FIFO and LIFO inventory accounting methods?",
        "subject": "General"
    }
]

def run_tests():
    """Run all test cases and display results."""
    results = []
    
    print("=== MathTutor System Test ===")
    print(f"Running {len(TEST_CASES)} test cases...")
    print("=" * 60)
    
    for i, test_case in enumerate(TEST_CASES):
        print(f"\nTest {i+1}: {test_case['name']}")
        print(f"Problem: {test_case['problem']}")
        
        # Create a unique session and problem ID for this test
        session_id = f"test_session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        problem_id = f"test_prob_{i}"
        
        try:
            # Process the problem
            start_time = datetime.now()
            result = solve_problem(
                problem_text=test_case['problem'],
                session_id=session_id,
                problem_id=problem_id
            )
            end_time = datetime.now()
            
            # Calculate processing time
            processing_time = (end_time - start_time).total_seconds()
            
            # Print results summary
            print(f"Subject: {result.get('subject_area') or 'Not determined'}")
            print(f"Workflow Type: {result.get('workflow_type') or 'Not determined'}")
            print(f"Processing Time: {processing_time:.2f} seconds")
            
            if result.get('solution_steps'):
                print(f"Solution Steps: {len(result.get('solution_steps'))} steps generated")
            
            if result.get('error_message'):
                print(f"Error: {result.get('error_message')}")
                
            # Store results
            results.append({
                "test_case": test_case,
                "success": not bool(result.get('error_message')),
                "processing_time": processing_time,
                "workflow_type": result.get('workflow_type'),
                "subject_area": result.get('subject_area'),
                "steps_count": len(result.get('solution_steps') or []),
                "result": result
            })
            
        except Exception as e:
            print(f"Test Failed: {str(e)}")
            results.append({
                "test_case": test_case,
                "success": False,
                "error": str(e)
            })
    
    # Print test summary
    print("\n" + "=" * 60)
    print("=== Test Summary ===")
    successful_tests = sum(1 for r in results if r.get('success', False))
    print(f"Tests Run: {len(results)}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {len(results) - successful_tests}")
    
    # Calculate average processing time
    avg_time = sum(r.get('processing_time', 0) for r in results if 'processing_time' in r) / len(results)
    print(f"Average Processing Time: {avg_time:.2f} seconds")
    
    return results

if __name__ == "__main__":
    run_tests() 