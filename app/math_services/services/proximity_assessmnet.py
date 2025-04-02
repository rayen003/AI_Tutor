from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import re

def assess_correctness(state):
    """Assess the correctness of the solution and extract the final answer"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    if not state.solution_steps:
        # No steps to assess
        new_state = state.model_copy()
        new_state.is_correct = False
        new_state.feedback = "No solution steps provided."
        return new_state
    
    problem = state.problem
    steps = state.solution_steps
    
    # Extract final answer from last step if possible
    final_step = steps[-1]
    
    assessment_prompt = ChatPromptTemplate.from_template("""
    You are evaluating a mathematical solution. 
    
    Problem: {problem}
    
    Solution steps:
    {steps}
    
    Please provide:
    1. Is the overall solution correct? (Yes/No)
    2. What is the final answer? (Extract it from the solution)
    3. Brief feedback on the solution quality
    
    Format your response as:
    Correct: [Yes/No]
    Final Answer: [extracted answer]
    Feedback: [brief assessment]
    """)
    
    response = llm.invoke(
        assessment_prompt.format(
            problem=problem,
            steps="\n".join([f"Step {i+1}: {step}" for i, step in enumerate(steps)])
        )
    )
    
    # Parse the response
    correct_match = re.search(r"Correct: (Yes|No)", response.content, re.IGNORECASE)
    is_correct = False
    if correct_match:
        is_correct = correct_match.group(1).lower() == "yes"
    
    answer_match = re.search(r"Final Answer: ([^\n]+)", response.content)
    final_answer = answer_match.group(1) if answer_match else None
    
    feedback_match = re.search(r"Feedback: ([^\n]+)", response.content)
    feedback = feedback_match.group(1) if feedback_match else "No feedback provided."
    
    new_state = state.model_copy()
    new_state.is_correct = is_correct
    new_state.final_answer = final_answer
    new_state.feedback = feedback
    return new_state
