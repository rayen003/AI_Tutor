"""
Test script to directly check API responses from the language model.

This script tests the raw responses from the language model for 
different prompts to see if we're getting valid JSON back.
"""

import os
import json
from dotenv import load_dotenv
import getpass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

def get_api_key():
    """Make sure we have an API key"""
    if not os.getenv("OPENAI_API_KEY"):
        api_key = getpass.getpass("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    return os.getenv("OPENAI_API_KEY")

def test_json_response(prompt_template, variables, test_name):
    """Test if a prompt returns valid JSON"""
    print(f"\n=== Testing {test_name} ===")
    
    # Initialize LLM with explicit parameters
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0,
        request_timeout=60,
        max_retries=2
    )
    
    # Create prompt
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    try:
        # Invoke the model
        response = llm.invoke(prompt.format(**variables))
        
        # Print raw response for inspection
        print("\nRaw API Response:")
        print(response.content)
        
        # Try to parse as JSON
        try:
            parsed = json.loads(response.content)
            print("\n✅ Response is valid JSON")
            print(f"JSON Type: {type(parsed).__name__}")
            if isinstance(parsed, list):
                print(f"Array Length: {len(parsed)}")
            elif isinstance(parsed, dict):
                print(f"Dictionary Keys: {list(parsed.keys())}")
            return True
        except json.JSONDecodeError as e:
            print(f"\n❌ Not valid JSON: {str(e)}")
            return False
    except Exception as e:
        print(f"\n❌ API Error: {str(e)}")
        return False

def main():
    """Run tests for different API prompts"""
    api_key = get_api_key()
    print(f"Using API key: {api_key[:5]}...")
    
    # Test verification prompt
    verification_prompt = """
    You are a mathematical verification expert. Your job is to carefully check each step of the following solution.
    
    Problem: What is the present value of $10,000 to be received in 5 years, assuming an interest rate of 8% per year?
    Known variables: future_value=10000, interest_rate=0.08, years=5
    
    Solution steps:
    Step 1: To find the present value, we need to use the formula: PV = FV / (1 + r)^t
    Step 2: Substituting the values: PV = $10,000 / (1 + 0.08)^5
    Step 3: PV = $10,000 / (1.08)^5
    Step 4: PV = $10,000 / 1.4693
    Step 5: PV = $6,805.83
    
    For each step, verify:
    1. Mathematical correctness (calculations, formulas, etc.)
    2. Logical progression (each step follows from previous ones)
    3. Clarity and educational value
    
    Return ONLY a JSON array with objects containing step_number, verification (CORRECT/INCORRECT), confidence, explanation, and suggestion.
    Example: 
    [
      {
        "step_number": 1,
        "verification": "CORRECT",
        "confidence": 95,
        "explanation": "The formula is correctly applied...",
        "suggestion": null
      }
    ]
    
    Do not include any text before or after the JSON array.
    """
    
    test_json_response(verification_prompt, {}, "Verification Prompt")
    
    # Test variable parsing prompt
    variable_parsing_prompt = """
    You are a math tutor analyzing a business mathematics problem.
    Extract all the relevant variables and their values from the following problem:
    
    Problem: What is the present value of $10,000 to be received in 5 years, assuming an interest rate of 8% per year?
    
    Identify each variable, its value, and its meaning in the context of the problem.
    For example, if the problem mentions "an interest rate of 5%", extract 'interest_rate': 0.05.
    
    Format your response as a JSON dictionary with variable names as keys and their values as values.
    Only include the JSON dictionary in your response, without any additional text.
    """
    
    test_json_response(variable_parsing_prompt, {}, "Variable Parsing Prompt")
    
    # Test context prompt
    context_prompt = """
    You are a specialized math tutor for BBA students. You need to identify the subject area 
    and relevant mathematical concepts for the following problem:
    
    Problem: What is the present value of $10,000 to be received in 5 years, assuming an interest rate of 8% per year?
    Variables identified: {"future_value": 10000, "interest_rate": 0.08, "years": 5}
    
    First, determine the primary subject area this problem belongs to:
    1. Finance
    2. Statistics
    3. Economics
    4. Accounting
    5. General Business Mathematics
    6. Other (specify)
    
    Then, identify the specific concepts, formulas, or frameworks that are relevant to solving this problem.
    
    Format your response as a JSON object with the following structure:
    {
      "subject_area": "The primary subject area",
      "context": "A detailed description of the relevant concepts, formulas, and approach needed to solve this problem"
    }
    
    Only include the JSON object in your response, without any additional text.
    """
    
    test_json_response(context_prompt, {}, "Context Prompt")
    
    # Test hint generation prompt
    hint_prompt = """
    You are a math tutor helping a BBA student. Create a series of progressive hints for the following problem:
    
    Problem: What is the present value of $10,000 to be received in 5 years, assuming an interest rate of 8% per year?
    Subject Area: Finance
    Context: Time value of money concepts, specifically present value calculation
    Variables: {"future_value": 10000, "interest_rate": 0.08, "years": 5}
    
    Create 3 levels of hints that provide increasing guidance:
    
    Hint Level 1: A very subtle hint that points to the core concept or approach needed, without giving away the solution method directly.
    
    Hint Level 2: A more direct hint that identifies the specific formula or concept to apply, but doesn't show how to apply it to this problem.
    
    Hint Level 3: A detailed hint that shows part of the solution process, but still leaves the final steps for the student to complete.
    
    Format your response as a JSON array with 3 hint objects, each containing a level and hint text.
    Only return the JSON array, without any additional text.
    """
    
    test_json_response(hint_prompt, {}, "Hint Generation Prompt")
    
    print("\n=== All Tests Complete ===")

if __name__ == "__main__":
    main() 