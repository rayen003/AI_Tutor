from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any

def verify_solution_steps(state):
    """
    Verify each step of a mathematical solution for correctness and logical flow.
    If errors are found, mark steps for regeneration.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Nothing to verify if there are no steps
    if not state.solution_steps:
        return state.copy()
    
    problem = state.problem
    variables = state.variables
    steps = state.solution_steps
    
    # Create context string from variables
    var_context = ", ".join([f"{k}={v}" for k, v in variables.items()])
    
    verification_prompt = ChatPromptTemplate.from_template("""
    You are a mathematical verification expert. Your job is to carefully check each step of the following solution.
    
    Problem: {problem}
    Known variables: {var_context}
    
    Solution steps:
    {steps}
    
    For each step, verify:
    1. Mathematical correctness (calculations, formulas, etc.)
    2. Logical progression (each step follows from previous ones)
    3. Clarity and educational value
    
    For each step, provide ONLY a JSON array with the following format:
    [
        {{
            "step_number": 1,
            "verification": "CORRECT",
            "confidence": 95,
            "explanation": "The formula is correctly applied...",
            "suggestion": null
        }},
        {{
            "step_number": 2,
            "verification": "INCORRECT",
            "confidence": 85,
            "explanation": "There's an error in the calculation...",
            "suggestion": "Step should be: x = 5 instead of x = 4"
        }}
    ]
    
    Do not include any text before or after the JSON array. Only return the JSON array.
    """)
    
    # Get verification results
    try:
        response = llm.invoke(
            verification_prompt.format(
                problem=problem, 
                var_context=var_context,
                steps="\n".join([f"Step {i+1}: {step}" for i, step in enumerate(steps)])
            )
        )
        
        # Parse verification results
        import json
        content = response.content.strip()
        
        # Handle empty responses
        if not content:
            print("Warning: Empty verification response")
            new_state = state.copy()
            new_state.verification_results = []
            new_state.needs_regeneration = False
            return new_state
            
        verification_results = json.loads(content)
        
        # Create new state with verification results
        new_state = state.copy()
        
        # Add verification metadata to our state
        new_state.verification_results = verification_results
        
        # Determine if any steps need regeneration
        steps_to_regenerate = []
        for result in verification_results:
            if result["verification"] == "INCORRECT":
                steps_to_regenerate.append(result["step_number"])
        
        new_state.steps_to_regenerate = steps_to_regenerate
        
        # Set flag if regeneration is needed
        new_state.needs_regeneration = len(steps_to_regenerate) > 0
        
        return new_state
    except Exception as e:
        # Fallback for parsing failure or API error
        print(f"Error in verification: {e}")
        new_state = state.copy()
        new_state.verification_results = []
        new_state.needs_regeneration = False
        new_state.verification_error = str(e)
        return new_state


def regenerate_solution_steps(state):
    """
    Regenerate incorrect solution steps based on verification results.
    Maintains context between steps.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Nothing to regenerate if no steps are marked for regeneration
    if not state.needs_regeneration or not state.steps_to_regenerate:
        return state.copy()
    
    problem = state.problem
    variables = state.variables
    steps = state.solution_steps
    steps_to_regenerate = state.steps_to_regenerate
    verification_results = state.verification_results
    
    # Create context from correct steps and verification feedback
    correct_steps = []
    for i, step in enumerate(steps):
        step_num = i + 1
        if step_num not in steps_to_regenerate:
            correct_steps.append(f"Step {step_num}: {step}")
    
    # Get feedback for incorrect steps
    feedback = []
    for result in verification_results:
        if result["step_number"] in steps_to_regenerate:
            feedback.append(
                f"Step {result['step_number']}: INCORRECT - {result['explanation']}\n" +
                f"Suggestion: {result['suggestion'] or 'Please revise this step'}"
            )
    
    regeneration_prompt = ChatPromptTemplate.from_template("""
    You are a mathematics expert. You need to correct some steps in a mathematical solution.
    
    Problem: {problem}
    Known variables: {var_context}
    
    Correct steps (keep these unchanged):
    {correct_steps}
    
    Steps that need correction:
    {feedback}
    
    Please provide a complete corrected solution, including both the unchanged steps and your 
    corrected steps in the proper order. Make sure your corrections address the specific issues mentioned.
    
    Format each step as "Step X: [content]" where X is the step number.
    """)
    
    # Create context string from variables
    var_context = ", ".join([f"{k}={v}" for k, v in variables.items()])
    
    # Get regenerated solution
    response = llm.invoke(
        regeneration_prompt.format(
            problem=problem,
            var_context=var_context,
            correct_steps="\n".join(correct_steps),
            feedback="\n".join(feedback)
        )
    )
    
    # Extract regenerated steps
    try:
        new_steps = []
        # Simple regex to extract steps from response
        import re
        step_pattern = r"Step (\d+): (.+)(?:\n|$)"
        matches = re.findall(step_pattern, response.content, re.MULTILINE)
        
        # Convert to dict for easier lookup by step number
        step_dict = {int(num): content for num, content in matches}
        
        # Rebuild steps in the correct order
        for i in range(1, len(steps) + 1):
            if i in step_dict:
                new_steps.append(step_dict[i])
            else:
                # Keep the original step if not regenerated
                new_steps.append(steps[i-1])
        
        # Create new state with regenerated steps
        new_state = state.copy()
        new_state.solution_steps = new_steps
        
        # Reset regeneration flags
        new_state.needs_regeneration = False
        new_state.steps_to_regenerate = []
        
        return new_state
    except Exception as e:
        # Fallback for parsing failure
        print(f"Error parsing regenerated steps: {e}")
        new_state = state.copy()
        new_state.regeneration_error = str(e)
        return new_state 