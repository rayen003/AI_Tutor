"""
Simple MathTutor implementation for BBA students.
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
import json
import os
import re
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Prompts
CLASSIFY_PROMPT = """
You are a math tutor for BBA students. You will be given a problem and you need to determine 
if it is a math problem or a general question.

If it is a math problem, you should return "math". If it is a general question, you should return "general".
Here is the problem: {problem}
"""

SOLUTION_PROMPT = """
You are a math tutor for a BBA student. Provide a detailed, step-by-step solution for this problem:

Problem: {problem}

Create a solution that:
1. Breaks down the problem into logical steps
2. Explains each step clearly, including formulas
3. Shows all calculations
4. Provides the final answer with appropriate units

Format your response as numbered steps.
"""

HINT_PROMPT = """
You are a math tutor helping a BBA student. Create a series of progressive hints for this problem:

Problem: {problem}

Create 3 levels of hints with increasing guidance:
1. A subtle hint pointing to the approach
2. A more direct hint about the formula to use
3. A detailed hint showing part of the solution process

Format your response as a list of 3 hints, from vague to specific.
"""

ASSESSMENT_PROMPT = """
You are a math tutor evaluating a student's answer.

Problem: {problem}
Correct answer: {correct_answer}
Student's answer: {student_answer}

Evaluate how close the student's answer is to the correct answer.
Provide a score from 0.0 to 1.0 where 1.0 is completely correct.
Include feedback on what's correct and what needs improvement.
"""

GENERAL_PROMPT = """
You are a tutor for BBA students. Answer this general question:

Question: {problem}

Provide a clear, educational response with examples or applications in a business context.
"""

# State class
class TutorState(BaseModel):
    # Problem information
    problem: str
    is_math_problem: Optional[bool] = None
    
    # Solution information
    solution_steps: List[str] = Field(default_factory=list)
    final_answer: Optional[str] = None
    
    # User interaction
    user_answer: Optional[str] = None
    feedback: Optional[str] = None
    answer_score: Optional[float] = None
    
    # Hints
    hints: List[str] = Field(default_factory=list)
    current_hint_level: int = 0
    
    # General response (for non-math questions)
    general_response: Optional[str] = None
    
    # What to return to the user
    response_to_user: Optional[str] = None
    
# Functions for the workflow
def classify_problem(state: TutorState) -> TutorState:
    """Determine if this is a math problem or general question"""
    response = llm.invoke(CLASSIFY_PROMPT.format(problem=state.problem))
    result = response.content.strip().lower()
    
    state.is_math_problem = result == "math"
    return state

def router(state: TutorState) -> str:
    """Route based on problem type"""
    if state.is_math_problem:
        return "math_workflow"
    else:
        return "general_workflow"

def solve_math_problem(state: TutorState) -> TutorState:
    """Generate a solution for the math problem"""
    response = llm.invoke(SOLUTION_PROMPT.format(problem=state.problem))
    
    # Parse steps from response
    steps = []
    answer = None
    
    # Simple regex to extract steps
    step_pattern = r"(?:Step|^\d+)[\.:\)]+(.*?)(?=(?:Step|^\d+)[\.:\)]|$)"
    matches = re.findall(step_pattern, response.content, re.MULTILINE | re.DOTALL)
    
    if matches:
        steps = [step.strip() for step in matches]
    else:
        # Fallback: use the whole response
        steps = [response.content.strip()]
    
    # Extract final answer from the last step if possible
    if steps:
        last_step = steps[-1].lower()
        for marker in ["answer is", "final answer", "result is", "therefore"]:
            if marker in last_step:
                pos = last_step.find(marker) + len(marker)
                answer = last_step[pos:].strip()
                break
    
    state.solution_steps = steps
    state.final_answer = answer
    return state

def generate_hints(state: TutorState) -> TutorState:
    """Generate progressive hints for the problem"""
    response = llm.invoke(HINT_PROMPT.format(problem=state.problem))
    
    # Split the response into separate hints
    hints = []
    
    # Try to parse numbered hints
    hint_pattern = r"(?:Hint|Level) \d+:?\s*(.*?)(?=(?:Hint|Level) \d+:|$)"
    matches = re.findall(hint_pattern, response.content, re.MULTILINE | re.DOTALL)
    
    if matches:
        hints = [hint.strip() for hint in matches]
    else:
        # Fallback: split by lines or paragraphs
        hints = [p.strip() for p in response.content.split("\n\n") if p.strip()]
    
    # Ensure we have at least one hint
    if not hints:
        hints = ["Consider what formulas might be relevant to this problem."]
    
    state.hints = hints
    return state

def handle_general_question(state: TutorState) -> TutorState:
    """Process a general (non-math) question"""
    response = llm.invoke(GENERAL_PROMPT.format(problem=state.problem))
    state.general_response = response.content.strip()
    return state

def format_response(state: TutorState) -> TutorState:
    """Format the final response to the user"""
    if state.is_math_problem:
        if state.current_hint_level > 0 and state.hints:
            # Show hint
            hint_index = min(state.current_hint_level - 1, len(state.hints) - 1)
            state.response_to_user = f"Hint: {state.hints[hint_index]}"
        elif state.user_answer:
            # Show feedback on answer
            feedback = state.feedback or "No feedback available."
            state.response_to_user = f"Feedback on your answer: {feedback}"
        else:
            # Show solution
            steps_text = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(state.solution_steps)])
            state.response_to_user = f"Problem: {state.problem}\n\n{steps_text}"
            if state.final_answer:
                state.response_to_user += f"\n\nFinal Answer: {state.final_answer}"
    else:
        # General question
        state.response_to_user = state.general_response
    
    return state

# Build the graph
def build_graph():
    workflow = StateGraph(TutorState)
    
    # Add nodes
    workflow.add_node("classify", classify_problem)
    workflow.add_node("solve_math", solve_math_problem)
    workflow.add_node("generate_hints", generate_hints)
    workflow.add_node("handle_general", handle_general_question)
    workflow.add_node("format_response", format_response)
    
    # Add edges
    workflow.add_edge(START, "classify")
    
    # Route based on problem type
    workflow.add_conditional_edges(
        "classify",
        router,
        {
            "math_workflow": "solve_math",
            "general_workflow": "handle_general"
        }
    )
    
    # Complete the math workflow
    workflow.add_edge("solve_math", "generate_hints")
    workflow.add_edge("generate_hints", "format_response")
    
    # Complete the general workflow
    workflow.add_edge("handle_general", "format_response")
    
    workflow.add_edge("format_response", END)
    
    return workflow.compile()

# Main function
def process_problem(
    problem_text: str,
    hint_level: int = 0,
    user_answer: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a math or general question
    
    Args:
        problem_text: The problem or question to solve
        hint_level: Level of hint to provide (0 = solution, 1-3 = progressive hints)
        user_answer: Optional user answer to evaluate
        
    Returns:
        Dictionary with processed results
    """
    # Create initial state
    state = TutorState(
        problem=problem_text,
        current_hint_level=hint_level,
        user_answer=user_answer
    )
    
    # Build graph
    graph = build_graph()
    
    # Process through the graph
    result = graph.invoke(state)
    
    # Convert to dictionary
    return {
        "problem": result.problem,
        "is_math_problem": result.is_math_problem,
        "solution_steps": result.solution_steps if hint_level == 0 else None,
        "final_answer": result.final_answer if hint_level == 0 else None,
        "hint": result.hints[hint_level-1] if hint_level > 0 and hint_level <= len(result.hints) else None,
        "feedback": result.feedback,
        "response": result.response_to_user
    }

# Simple test
if __name__ == "__main__":
    # Test with a math problem
    math_problem = "Calculate the present value of $5,000 to be received in 3 years with an interest rate of 7%."
    print("\nTesting math problem:")
    result = process_problem(math_problem)
    print(f"Response: {result['response']}")
    
    # Test with a hint
    print("\nTesting with hint level 1:")
    result = process_problem(math_problem, hint_level=1)
    print(f"Hint: {result['hint']}")
    
    # Test with a general question
    general_question = "What is the difference between FIFO and LIFO inventory accounting methods?"
    print("\nTesting general question:")
    result = process_problem(general_question)
    print(f"Response: {result['response']}") 