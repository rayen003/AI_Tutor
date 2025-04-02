"""
Answer proximity assessment module for the MathTutor platform.

This module evaluates how close a student's answer is to the correct solution,
providing detailed feedback and a numerical proximity score.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import re
import json
from app.math_services.prompts import ANSWER_ASSESSMENT_PROMPT

def assess_correctness(state):
    """
    Assess how close the student's answer is to the correct solution.
    
    Args:
        state: The current MathTutorState containing the problem, solution, and user's answer
        
    Returns:
        Updated state with assessment results
    """
    # Skip if no user answer provided or no solution generated
    if not state.user_answer or not state.solution_steps:
        return state.model_copy()
    
    # Use a more capable model for assessment
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Extract final answer from solution steps if not explicitly set
    final_answer = state.final_answer
    if not final_answer and state.solution_steps:
        # Try to extract from the last solution step
        last_step = state.solution_steps[-1]
        # Use simple heuristic - look for phrases like "final answer", "therefore", "thus"
        answer_indicators = ["final answer", "therefore", "thus", "so", "="]
        for indicator in answer_indicators:
            if indicator in last_step.lower():
                # Extract everything after the indicator
                final_answer = last_step[last_step.lower().find(indicator) + len(indicator):].strip()
                break
    
    # Create assessment prompt
    assessment_prompt = ChatPromptTemplate.from_template(ANSWER_ASSESSMENT_PROMPT)
    
    # Get assessment from LLM
    response = llm.invoke(
        assessment_prompt.format(
            problem=state.problem,
            solution_steps="\n".join([f"Step {i+1}: {step}" for i, step in enumerate(state.solution_steps)]),
            final_answer=final_answer or "Not explicitly stated",
            user_answer=state.user_answer
        )
    )
    
    # Parse assessment results
    try:
        assessment_data = json.loads(response.content)
        
        # Create new state with assessment results
        new_state = state.model_copy()
        
        # Extract assessment data
        new_state.answer_proximity = assessment_data.get("proximity_score", 0.0)
        new_state.is_correct = assessment_data.get("correctness") == "CORRECT"
        new_state.feedback = assessment_data.get("feedback", "")
        
        return new_state
    except Exception as e:
        # Fallback for parsing failures
        print(f"Error parsing assessment results: {e}")
        
        # Use regex as fallback
        new_state = state.model_copy()
        
        # Basic regex matching for common output formats
        proximity_match = re.search(r"proximity_score[\"']?\s*:\s*(0\.\d+|1\.0)", response.content)
        if proximity_match:
            new_state.answer_proximity = float(proximity_match.group(1))
        
        # Check if correct based on keywords
        is_correct = "CORRECT" in response.content.upper() and not "INCORRECT" in response.content.upper()
        new_state.is_correct = is_correct
        
        # Extract feedback using patterns
        feedback_match = re.search(r"feedback[\"']?\s*:\s*[\"']([^\"']+)[\"']", response.content)
        if feedback_match:
            new_state.feedback = feedback_match.group(1)
        else:
            new_state.feedback = "Unable to parse detailed feedback."
        
        return new_state
