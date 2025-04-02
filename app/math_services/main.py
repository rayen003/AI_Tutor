from pydantic import BaseModel, Field, FieldValidationInfo
from typing import List, Dict, Any, Optional, Literal, Union
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from pydantic.v1 import root_validator, validator
import os
import re
import json
import uuid
from datetime import datetime

# Load environment variables
load_dotenv()

# Define the comprehensive state model
class MathTutorState(BaseModel):
    # Current problem info
    problem: str
    problem_id: Optional[str] = None
    workflow_type: Optional[Literal["math", "general"]] = None
    
    # Session tracking
    session_id: Optional[str] = None
    attempt_number: int = 1
    timestamp: Optional[str] = None
    
    # Problem context
    variables: Dict[str, Any] = Field(default_factory=dict)
    context: Optional[str] = None
    subject_area: Optional[str] = None
    
    # Solution tracking
    solution_steps: List[str] = Field(default_factory=list)
    final_answer: Optional[str] = None
    
    # Verification
    verification_results: List[Dict[str, Any]] = Field(default_factory=list)
    steps_to_regenerate: List[int] = Field(default_factory=list)
    needs_regeneration: bool = False
    regeneration_attempts: int = 0
    max_regeneration_attempts: int = 3
    
    # User interaction
    user_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    feedback: Optional[str] = None
    
    # Hints
    hints: List[Dict[str, Any]] = Field(default_factory=list)
    current_hint_index: int = 0
    
    # Response formatting
    final_response: Optional[str] = None
    
    # Error handling
    error_message: Optional[str] = None
    
    # Comprehensive history - stores entire attempt history
    attempt_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # User interaction tracking
    requested_action: Optional[Literal["hint", "solution", "check_answer", "none"]] = "none"
    
    # System recommendations
    system_suggestion: Optional[Literal["hint", "solution", "continue", "none"]] = "none"
    suggestion_message: Optional[str] = None
    
    # Hint progression tracking
    current_hint_level: int = 0
    max_hint_level: int = 3
    
    # Answer assessment
    answer_proximity: Optional[float] = None  # 0.0 to 1.0

    @validator("problem")
    def validate_problem(problem):
        if not problem:
            raise ValueError("Problem cannot be empty")
        return problem
    
    @validator("workflow_type")
    def validate_workflow_type(workflow_type):
        if workflow_type not in ["math", "general"]:
            raise ValueError("Workflow type must be either 'math' or 'general'")
        else:
            return workflow_type
        

prompts= {
    "classify_workflow": """
    You are a math tutor. You will be given a problem and you need to determine if it is a math problem or a general question.
    If it is a math problem, you should return "math". If it is a general question, you should return "general".
    Here is the problem: {state.problem}
    """,
    "general_question": """
    You are a math tutor. You will be given a general question and you need to answer it.
    Here is the question: {state.problem}
    """
}

# Initialize the LLM - we'll use the same instance across services
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Import service functions
from app.math_services.services.response_parsing import parse_variables
from app.math_services.services.reasoning import define_context, generate_response
from app.math_services.services.verification import verify_solution_steps, regenerate_solution_steps
from app.math_services.services.proximity_assessmnet import assess_correctness
from app.math_services.services.hint_generation import generate_hint

# Define workflow classification - determines math vs general
def classify_workflow(state: MathTutorState) -> MathTutorState:
    """Determine if this is a math problem or general question"""


    question_type = state.get("problem","").lower()

    prompt = prompts["classify_workflow"].format(state=state)

    classification = llm.invoke(prompt)
    state.workflow_type = classification
    return state

# Route to appropriate workflow based on classification
def router(state: MathTutorState) -> str:
    """Router to determine next processing step"""
    if state.workflow_type == "math":
        return "math_workflow"
    else:
        return "general_workflow"

# Determine if regeneration is needed
def check_regeneration(state: MathTutorState) -> str:
    """Check if solution steps need regeneration"""
    if state.needs_regeneration and state.regeneration_attempts < state.max_regeneration_attempts:
        # Increment the attempt counter
        new_state = state.model_copy()
        new_state.regeneration_attempts += 1
        return "regenerate"
    else:
        return "continue"

# General workflow placeholder
def process_general_question(state: MathTutorState) -> MathTutorState:
    """Handle general (non-math) questions"""
    new_state = state.model_copy()
    response = llm.invoke(prompts["general_question"].format(state=state))
    new_state.final_response = response
    return new_state

# Format final response
def format_final_response(state: MathTutorState) -> MathTutorState:
    """Format the verified solution into a comprehensive response"""
    new_state = state.model_copy()
    
    if state.workflow_type == "math":
        steps_text = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(state.solution_steps)])
        new_state.final_response = f"""
        Problem: {state.problem}
        
        Solution:
        {steps_text}
        
        Final Answer: {state.final_answer or "Not provided"}
        """
    
    # For general questions, the response was already set
    
    return new_state

# Build the workflow graph
def build_graph():
    workflow = StateGraph(MathTutorState)
    
    # Add session initialization node
    workflow.add_node("initialize_session", initialize_session)
    
    # Add existing nodes
    workflow.add_node("classify", classify_workflow)
    workflow.add_node("parse_variables", parse_variables)
    workflow.add_node("define_context", define_context)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("verify_steps", verify_solution_steps)
    workflow.add_node("regenerate_steps", regenerate_solution_steps)
    workflow.add_node("assess_correctness", assess_correctness)
    workflow.add_node("generate_hint", generate_hint)
    workflow.add_node("general_process", process_general_question)
    workflow.add_node("format_response", format_final_response)
    
    # Add record attempt node
    workflow.add_node("record_attempt", record_attempt)
    
    # Add edges with session init first
    workflow.add_edge(START, "initialize_session")
    workflow.add_edge("initialize_session", "classify")
    
    # Existing edges
    workflow.add_conditional_edges(
        "classify",
        router,
        {
            "math_workflow": "parse_variables",
            "general_workflow": "general_process"
        }
    )
    
    # Math workflow path
    workflow.add_edge("parse_variables", "define_context")
    workflow.add_edge("define_context", "generate_response")
    workflow.add_edge("generate_response", "verify_steps")
    
    # Conditional regeneration path
    workflow.add_conditional_edges(
        "verify_steps",
        check_regeneration,
        {
            "regenerate": "regenerate_steps",
            "continue": "assess_correctness"
        }
    )
    
    # After regeneration, verify again
    workflow.add_edge("regenerate_steps", "verify_steps")
    
    # Finish the math workflow
    workflow.add_edge("assess_correctness", "generate_hint")
    workflow.add_edge("generate_hint", "record_attempt")
    workflow.add_edge("record_attempt", "format_response")
    
    # General workflow path
    workflow.add_edge("general_process", "record_attempt")
    
    # End the workflow
    workflow.add_edge("format_response", END)
    
    return workflow.compile()

# Create the workflow app
math_tutor_app = build_graph()

# Main execution function for testing
def solve_problem(problem_text: str, session_id: Optional[str] = None, 
                 problem_id: Optional[str] = None, attempt_number: int = 1) -> Dict[str, Any]:
    """Process a problem through the math tutor workflow with session tracking"""
    initial_state = MathTutorState(
        problem=problem_text,
        session_id=session_id,
        problem_id=problem_id,
        attempt_number=attempt_number
    )
    
    result = math_tutor_app.invoke(initial_state)
    
    return {
        "problem_id": result.problem_id,
        "session_id": result.session_id,
        "attempt_number": result.attempt_number,
        "timestamp": result.timestamp,
        "problem": result.problem,
        "workflow_type": result.workflow_type,
        "solution_steps": result.solution_steps,
        "final_response": result.final_response,
        "hints": result.hints,
        "is_correct": result.is_correct,
        "regeneration_attempts": result.regeneration_attempts
    }

# Example usage for multiple attempts
if __name__ == "__main__":
    session_id = str(uuid.uuid4())  # Create a single session ID
    
    test_problems = [
        "Calculate the NPV of $5000 invested for 3 years at 7% interest.",
        "Solve the equation 2x + 3 = 7."
    ]
    
    problem_tracking = {}  # Track problem IDs and attempts
    
    # First attempt for each problem
    for problem in test_problems:
        print(f"\nFirst attempt for: {problem}")
        result = solve_problem(problem, session_id=session_id)
        
        # Store the problem ID for future attempts
        problem_tracking[problem] = {
            "problem_id": result["problem_id"],
            "attempts": 1
        }
        
        print(f"Problem ID: {result['problem_id']}")
        print(f"Workflow type: {result['workflow_type']}")
        print(f"Correct: {result['is_correct']}")
        print(f"Regeneration attempts: {result['regeneration_attempts']}")
    
    # Second attempt for the first problem
    second_problem = test_problems[0]
    print(f"\nSecond attempt for: {second_problem}")
    
    # Get tracking info
    tracking_info = problem_tracking[second_problem]
    tracking_info["attempts"] += 1
    
    # Make the second attempt
    result = solve_problem(
        second_problem,
        session_id=session_id,
        problem_id=tracking_info["problem_id"],
        attempt_number=tracking_info["attempts"]
    )
    
    print(f"Problem ID: {result['problem_id']}")
    print(f"Attempt number: {result['attempt_number']}")
    print(f"Timestamp: {result['timestamp']}")
    print(f"Correct: {result['is_correct']}")

def initialize_session(state: MathTutorState) -> MathTutorState:
    """Initialize or continue a session"""
    new_state = state.model_copy()
    
    # Generate session ID if not present
    if not new_state.session_id:
        new_state.session_id = str(uuid.uuid4())
    
    # Generate problem ID if not present
    if not new_state.problem_id:
        new_state.problem_id = str(uuid.uuid4())
    
    # Add timestamp
    new_state.timestamp = datetime.now().isoformat()
    
    return new_state

def record_attempt(state: MathTutorState) -> MathTutorState:
    """Record the current attempt in history before finishing"""
    new_state = state.model_copy()
    
    # Create an attempt record
    attempt_record = {
        "problem_id": new_state.problem_id,
        "problem_text": new_state.problem,
        "attempt_number": new_state.attempt_number,
        "timestamp": new_state.timestamp,
        "workflow_type": new_state.workflow_type,
        "solution_steps": new_state.solution_steps,
        "verification_results": new_state.verification_results,
        "regeneration_attempts": new_state.regeneration_attempts,
        "final_answer": new_state.final_answer,
        "is_correct": new_state.is_correct,
        "feedback": new_state.feedback,
        "hints_used": new_state.current_hint_index
    }
    
    # Add to history
    new_state.attempt_history.append(attempt_record)
    
    return new_state

def process_user_answer(state: MathTutorState) -> MathTutorState:
    """Assess the user's answer and calculate proximity score"""
    new_state = state.model_copy()
    
    # Use LLM to assess correctness and proximity
    assessment_prompt = ChatPromptTemplate.from_template("""
    Evaluate this answer to the given math problem.
    
    Problem: {problem}
    Correct Solution: {solution}
    User's Answer: {user_answer}
    
    Provide:
    1. Is the answer correct? (Yes/No)
    2. On a scale of 0.0 to 1.0, how close is the user to the correct answer?
    3. Brief feedback explaining any errors
    
    Format: 
    Correct: [Yes/No]
    Proximity: [0.0-1.0]
    Feedback: [explanation]
    """)
    
    response = llm.invoke(
        assessment_prompt.format(
            problem=state.problem,
            solution=state.final_answer, 
            user_answer=state.user_answer
        )
    )
    
    # Extract assessment using regex
    correct_match = re.search(r"Correct: (Yes|No)", response.content)
    is_correct = correct_match.group(1).lower() == "yes" if correct_match else False
    
    proximity_match = re.search(r"Proximity: (0\.\d+|1\.1)", response.content)
    proximity = float(proximity_match.group(1)) if proximity_match else 0.0
    
    feedback_match = re.search(r"Feedback: (.+)", response.content)
    feedback = feedback_match.group(1) if feedback_match else "No feedback provided."
    
    # Update state with assessment
    new_state.is_correct = is_correct
    new_state.answer_proximity = proximity
    new_state.feedback = feedback
    
    return new_state

def determine_next_suggestion(state: MathTutorState) -> MathTutorState:
    """Determine what the system should suggest next based on answer assessment"""
    new_state = state.model_copy()
    
    # Already correct - congratulate
    if state.is_correct:
        new_state.system_suggestion = "continue"
        new_state.suggestion_message = "Great job! Would you like to try another problem?"
        return new_state
    
    # Close but not quite - suggest hint
    if state.answer_proximity and state.answer_proximity >= 0.7:
        new_state.system_suggestion = "hint"
        new_state.suggestion_message = "You're on the right track! Would you like a hint to help you solve this?"
        return new_state
    
    # Far off - suggest solution
    new_state.system_suggestion = "solution"
    new_state.suggestion_message = "You seem to be having difficulty. Would you like to see the solution?"
    return new_state

def generate_progressive_hint(state: MathTutorState) -> MathTutorState:
    """Generate or retrieve the next level hint"""
    new_state = state.model_copy()
    
    # If we don't have hints yet, generate them
    if not state.hints:
        # Use existing hint generation code here
        new_state = generate_hint(new_state)
    
    # Increment hint level (if not at max)
    if new_state.current_hint_level < new_state.max_hint_level:
        new_state.current_hint_level += 1
    
    return new_state

def user_input_router(state: MathTutorState) -> str:
    """Route based on what the user has provided or requested"""
    if state.requested_action == "hint":
        return "requested_hint"
    elif state.requested_action == "solution":
        return "requested_solution"
    elif state.user_answer:
        return "has_answer"
    else:
        return "no_input"
    


graph = build_graph()

# Example usage
if __name__ == "__main__":
    # Test with a problem
    problem = "What is the sum of the first 100 natural numbers?"
    result = graph.invoke({
        "problem": problem,
        "workflow_type": "math",
        "attempt_number": 1,
        "session_id": str(uuid.uuid4())
    })  
    

        