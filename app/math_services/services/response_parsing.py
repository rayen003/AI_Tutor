from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import re

# This is a simplified example - real implementation would be more robust
def parse_variables(state):
    """Extract variables from a math problem"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    parse_prompt = ChatPromptTemplate.from_template("""
    Extract all variables and their values from this math problem.
    Format your response as a JSON dictionary with variable names as keys.
    
    Problem: {problem}
    
    Example output for "If a loan of $5000 has an interest rate of 7% for 3 years, what is the future value?":
    {{"principal": 5000, "rate": 0.07, "time": 3}}
    
    JSON Output:
    """)
    
    response = llm.invoke(parse_prompt.format(problem=state.problem))
    
    # In a real implementation, you'd use a proper JSON parser with error handling
    # This is simplified
    try:
        import json
        variables = json.loads(response.content)
        
        new_state = state.model_copy()
        new_state.workflow_type = "math"
        new_state.variables = variables
        return new_state
    except:
        # Fallback for parsing failure
        new_state = state.model_copy()
        new_state.workflow_type = "general"  # If we can't parse variables, treat as general
        return new_state
