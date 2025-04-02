from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import re

def generate_response(state):
    """Generate a detailed solution using Chain of Thought reasoning"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    problem = state.problem
    variables = state.variables
    context = state.context
    
    # Create context string from variables
    var_context = ", ".join([f"{k}={v}" for k, v in variables.items()])
    
    solution_prompt = ChatPromptTemplate.from_template("""
    You are a mathematics tutor for a BBA student. Solve the following problem using step-by-step reasoning.
    
    Problem: {problem}
    
    Known variables: {var_context}
    
    Additional context: {context}
    
    Provide a clear Chain of Thought solution with these guidelines:
    1. Start by identifying the key concepts and formulas needed
    2. Break down the solution into clear, logical steps
    3. Show all calculations explicitly
    4. Explain the reasoning behind each step
    5. Verify your final answer
    
    Format each step as "Step X: [content]" where X is the step number.
    
    Solution:
    """)
    
    response = llm.invoke(
        solution_prompt.format(
            problem=problem,
            var_context=var_context,
            context=context or "No additional context provided."
        )
    )
    
    # Extract solution steps
    try:
        # Simple regex to extract steps from response
        import re
        step_pattern = r"Step (\d+): (.+)(?:\n|$)"
        matches = re.findall(step_pattern, response.content, re.MULTILINE)
        
        # Extract just the step content
        steps = [content for _, content in matches]
        
        # If no steps were found, use a simple split by newlines
        if not steps:
            steps = [line.strip() for line in response.content.split('\n') if line.strip()]
        
        new_state = state.model_copy()
        new_state.solution_steps = steps
        return new_state
    except Exception as e:
        print(f"Error parsing solution steps: {e}")
        new_state = state.model_copy()
        new_state.solution_steps = [response.content]  # Fallback: use entire response as a single step
        return new_state

def define_context(state):
    """Define the mathematical context for the problem"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    problem = state.problem
    variables = state.variables
    
    context_prompt = ChatPromptTemplate.from_template("""
    Analyze this math problem and provide context that would help solve it.
    Identify the subject area (e.g., finance, algebra, statistics) and any key concepts or formulas needed.
    
    Problem: {problem}
    Variables identified: {variables}
    
    Output format:
    Subject: [subject area]
    Key Concepts: [list of relevant concepts]
    Relevant Formulas: [any formulas that apply]
    Approach: [brief description of how to approach this problem]
    """)
    
    response = llm.invoke(
        context_prompt.format(
            problem=problem,
            variables=str(variables)
        )
    )
    
    # Extract subject area if possible
    subject_match = re.search(r"Subject: ([^\n]+)", response.content)
    subject = subject_match.group(1) if subject_match else "General Mathematics"
    
    new_state = state.model_copy()
    new_state.context = response.content
    new_state.subject_area = subject
    return new_state
