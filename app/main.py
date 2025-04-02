import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, START, END
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal, Union
from dotenv import load_dotenv
import os
import re
import json
import uuid
from datetime import datetime

# Load environment variables
load_dotenv()

# Define the internal state model - complete processing details
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
    solution_generated: bool = False
    solution_revealed: bool = False
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
    current_hint_level: int = 0
    max_hint_level: int = 3
    
    # Response formatting
    final_response: Optional[str] = None
    
    # Error handling
    error_message: Optional[str] = None
    
    # History tracking
    attempt_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # User interaction tracking
    requested_action: Optional[Literal["hint", "solution", "check_answer", "none"]] = "none"
    
    # System recommendations
    system_suggestion: Optional[Literal["hint", "solution", "continue", "none"]] = "none"
    suggestion_message: Optional[str] = None
    
    # Answer assessment
    answer_proximity: Optional[float] = None  # 0.0 to 1.0

# Define the user-facing response model - only what users need to see
class UserFacingResponse(BaseModel):
    # Problem information
    problem: str
    problem_id: str
    
    # Solution information (only shown when appropriate)
    solution_steps: Optional[List[str]] = None
    final_answer: Optional[str] = None
    
    # User feedback
    is_correct: Optional[bool] = None
    feedback: Optional[str] = None
    
    # Learning support
    current_hint: Optional[str] = None
    hint_level: Optional[int] = None
    
    # System suggestion
    suggestion: Optional[str] = None
    suggestion_message: Optional[str] = None
    
    # Session tracking (minimal)
    session_id: str
    attempt_number: int

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Prompts (will be moved to a separate file later)
CLASSIFY_PROMPT = """
You are a math tutor for BBA (Bachelor of Business Administration) students. 
You will be given a problem and you need to determine if it is a math problem or a general question.

If it is a math problem (requiring calculations, formulas, numerical reasoning, etc.), you should return "math".
If it is a general conceptual question, you should return "general".

Here is the problem: {problem}

Respond with only "math" or "general".
"""

VARIABLE_PARSING_PROMPT = """
You are a math tutor analyzing a business mathematics problem.
Extract all the relevant variables and their values from the following problem:

Problem: {problem}

Identify each variable, its value, and its meaning in the context of the problem.
For example, if the problem mentions "an interest rate of 5%", extract 'interest_rate': 0.05.

Format your response as a JSON dictionary with variable names as keys and their values as values.
Only include the JSON dictionary in your response, without any additional text.
"""

CONTEXT_PROMPT = """
You are a specialized math tutor for BBA students. You need to identify the subject area 
and relevant mathematical concepts for the following problem:

Problem: {problem}
Variables identified: {variables}

First, determine the primary subject area this problem belongs to:
1. Finance
2. Statistics
3. Economics
4. Accounting
5. General Business Mathematics
6. Other (specify)

Then, identify the specific concepts, formulas, or frameworks that are relevant to solving this problem.

Format your response as a JSON object with the following structure:
{{
  "subject_area": "The primary subject area",
  "context": "A detailed description of the relevant concepts, formulas, and approach needed to solve this problem"
}}

Only include the JSON object in your response, without any additional text.
"""

SOLUTION_PROMPT = """
You are a math tutor for a BBA student working on a business mathematics problem.
Provide a detailed, step-by-step solution that clearly explains your reasoning.

Problem: {problem}
Subject Area: {subject_area}
Relevant Context: {context}
Known Variables: {variables}

Create a comprehensive solution that:
1. Breaks down the problem into logical steps
2. Explains each step clearly, including the formulas or concepts being applied
3. Shows all calculations
4. Provides the final answer with appropriate units

Focus on clarity and educational value - your goal is to help the student understand 
the problem and solution thoroughly.

Format your response as a list of solution steps, with each step clearly explaining one part of the solution.
I will use these steps directly to teach the student, so make each step self-contained and informative.
"""

VERIFICATION_PROMPT = """
You are a mathematical verification expert. Your job is to carefully check each step of the following solution.

Problem: {problem}
Known variables: {var_context}

Solution steps:
{steps}

For each step, verify:
1. Mathematical correctness (calculations, formulas, etc.)
2. Logical progression (each step follows from previous ones)
3. Clarity and educational value

For each step, provide:
- Verification status: CORRECT or INCORRECT
- Confidence score (0-100%)
- Explanation (especially for errors)
- Suggested correction (if needed)

Format your response as a JSON array of step verifications.
"""

HINT_PROMPT = """
You are a math tutor helping a BBA student. Create a series of progressive hints for the following problem:

Problem: {problem}
Subject Area: {subject_area}
Context: {context}
Variables: {variables}
Solution Steps: {solution_steps}

Create 3 levels of hints that provide increasing guidance:

Hint Level 1: A very subtle hint that points to the core concept or approach needed, without giving away the solution method directly.

Hint Level 2: A more direct hint that identifies the specific formula or concept to apply, but doesn't show how to apply it to this problem.

Hint Level 3: A detailed hint that shows part of the solution process, but still leaves the final steps for the student to complete.

Format your response as a JSON array with 3 hint objects.
"""

ASSESSMENT_PROMPT = """
You are a math tutor evaluating a student's answer to a business mathematics problem.

Problem: {problem}
Correct solution steps: {solution_steps}
Correct final answer: {final_answer}

Student's answer: {user_answer}

Evaluate how close the student's answer is to the correct answer:
1. Calculate a proximity score from 0.0 to 1.0 where:
   - 1.0 means the answer is completely correct (exact match or mathematically equivalent)
   - 0.0 means the answer is completely unrelated or wrong
   - Intermediate values represent partial correctness

2. Provide specific feedback about what parts of the answer are correct and what needs improvement.

3. Determine if the student's answer should be considered correct, partially correct, or incorrect.

Format your response as a JSON object with proximity_score, correctness, feedback, and improvement_suggestion fields.
"""

GENERAL_PROMPT = """
You are a knowledgeable tutor for BBA students. You've been asked a general question about business or mathematics concepts.

Question: {problem}

Provide a clear, comprehensive, and educational response that:
1. Directly answers the question
2. Explains relevant concepts and principles
3. Provides examples or applications in a business context where appropriate
4. Cites relevant frameworks or theories if applicable

Your goal is to provide not just an answer, but an explanation that deepens the student's understanding of the subject matter.
Keep your response focused on business education and appropriate for undergraduate business students.
"""

# Core processing functions
def initialize_session(state: MathTutorState) -> MathTutorState:
    """Initialize or continue a session with proper tracking IDs"""
    new_state = state.model_copy()
    
    # Generate session ID if not present
    if not new_state.session_id:
        new_state.session_id = str(uuid.uuid4())
    
    # Generate problem ID if not present
    if not new_state.problem_id:
        new_state.problem_id = f"prob_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Add timestamp
    new_state.timestamp = datetime.now().isoformat()
    
    return new_state

def classify_workflow(state: MathTutorState) -> MathTutorState:
    """Determine if this is a math problem or general question"""
    new_state = state.model_copy()
    
    prompt = ChatPromptTemplate.from_template(CLASSIFY_PROMPT)
    response = llm.invoke(prompt.format(problem=state.problem))
    
    # Extract classification
    classification = response.content.strip().lower()
    if classification in ["math", "general"]:
        new_state.workflow_type = classification
    else:
        # Default to math if unclear
        new_state.workflow_type = "math"
    
    return new_state

def router(state: MathTutorState) -> str:
    """Router to determine next processing step"""
    if state.workflow_type == "math":
        return "math_workflow"
    else:
        return "general_workflow"

def parse_variables(state: MathTutorState) -> MathTutorState:
    """Extract variables and their values from the problem"""
    new_state = state.model_copy()
    
    prompt = ChatPromptTemplate.from_template(VARIABLE_PARSING_PROMPT)
    response = llm.invoke(prompt.format(problem=state.problem))
    
    try:
        # Parse the JSON response
        variables = json.loads(response.content)
        new_state.variables = variables
    except Exception as e:
        print(f"Error parsing variables: {e}")
        # Create empty variables as fallback
        new_state.variables = {}
    
    return new_state

def define_context(state: MathTutorState) -> MathTutorState:
    """Identify subject area and relevant mathematical context"""
    new_state = state.model_copy()
    
    prompt = ChatPromptTemplate.from_template(CONTEXT_PROMPT)
    response = llm.invoke(prompt.format(
        problem=state.problem,
        variables=json.dumps(state.variables)
    ))
    
    try:
        # Parse the JSON response
        context_data = json.loads(response.content)
        new_state.subject_area = context_data.get("subject_area")
        new_state.context = context_data.get("context")
    except Exception as e:
        print(f"Error parsing context: {e}")
        # Set defaults as fallback
        new_state.subject_area = "General Business Mathematics"
        new_state.context = "General problem-solving approach"
    
    return new_state

def generate_solution(state: MathTutorState) -> MathTutorState:
    """Generate a detailed step-by-step solution"""
    new_state = state.model_copy()
    
    prompt = ChatPromptTemplate.from_template(SOLUTION_PROMPT)
    response = llm.invoke(prompt.format(
        problem=state.problem,
        subject_area=state.subject_area or "Business Mathematics",
        context=state.context or "General problem-solving",
        variables=json.dumps(state.variables)
    ))
    
    # Split the response into steps
    step_pattern = r"(?:Step \d+:|^\d+\.)(.*?)(?=Step \d+:|^\d+\.|$)"
    steps = re.findall(step_pattern, response.content, re.MULTILINE | re.DOTALL)
    
    # Clean up the steps
    cleaned_steps = [step.strip() for step in steps if step.strip()]
    
    if not cleaned_steps:
        # Fallback: just use the entire response as a single step
        cleaned_steps = [response.content.strip()]
    
    new_state.solution_steps = cleaned_steps
    new_state.solution_generated = True
    
    # Extract final answer from the last step if possible
    last_step = cleaned_steps[-1]
    final_answer_markers = ["answer is", "final answer", "result is", "we get", "therefore"]
    
    for marker in final_answer_markers:
        if marker in last_step.lower():
            # Extract everything after the marker
            try:
                final_answer = last_step[last_step.lower().find(marker) + len(marker):].strip()
                new_state.final_answer = final_answer
                break
            except:
                pass
    
    return new_state

def verify_solution(state: MathTutorState) -> MathTutorState:
    """Verify the solution for correctness and clarity"""
    new_state = state.model_copy()
    
    if not state.solution_steps:
        return new_state
    
    # Create context from variables
    var_context = ", ".join([f"{k}={v}" for k, v in state.variables.items()])
    
    prompt = ChatPromptTemplate.from_template(VERIFICATION_PROMPT)
    response = llm.invoke(prompt.format(
        problem=state.problem,
        var_context=var_context,
        steps="\n".join([f"Step {i+1}: {step}" for i, step in enumerate(state.solution_steps)])
    ))
    
    try:
        # Try to parse structured response
        verification_results = json.loads(response.content)
        new_state.verification_results = verification_results
        
        # Check if any steps need regeneration
        steps_to_regenerate = []
        for result in verification_results:
            if (isinstance(result, dict) and 
                result.get("verification") == "INCORRECT" and
                "step_number" in result):
                steps_to_regenerate.append(result["step_number"])
        
        new_state.steps_to_regenerate = steps_to_regenerate
        new_state.needs_regeneration = len(steps_to_regenerate) > 0
    except Exception as e:
        print(f"Error parsing verification results: {e}")
        # Set default verification as fallback
        new_state.verification_results = []
        new_state.needs_regeneration = False
    
    return new_state

def check_regeneration(state: MathTutorState) -> str:
    """Check if solution steps need regeneration"""
    if state.needs_regeneration and state.regeneration_attempts < state.max_regeneration_attempts:
        return "regenerate"
    else:
        return "continue"

def regenerate_solution(state: MathTutorState) -> MathTutorState:
    """Regenerate incorrect solution steps"""
    new_state = state.model_copy()
    new_state.regeneration_attempts += 1
    
    # For this MVP, we'll simply regenerate the entire solution
    # In a more advanced implementation, we would regenerate only the problematic steps
    
    # Generate a new solution
    return generate_solution(new_state)

def generate_hints(state: MathTutorState) -> MathTutorState:
    """Generate progressive hints for the problem"""
    new_state = state.model_copy()
    
    # Skip if hints already exist
    if state.hints:
        return new_state
    
    prompt = ChatPromptTemplate.from_template(HINT_PROMPT)
    response = llm.invoke(prompt.format(
        problem=state.problem,
        subject_area=state.subject_area or "Business Mathematics",
        context=state.context or "General problem-solving",
        variables=json.dumps(state.variables),
        solution_steps=json.dumps(state.solution_steps)
    ))
    
    try:
        # Parse the hints
        hints = json.loads(response.content)
        new_state.hints = hints
    except Exception as e:
        print(f"Error parsing hints: {e}")
        # Create default hints as fallback
        new_state.hints = [
            {"level": 1, "hint": "Think about the key concepts related to this problem."},
            {"level": 2, "hint": "Consider what formulas might be applicable here."},
            {"level": 3, "hint": "Start by organizing the given variables and determining what you need to find."}
        ]
    
    return new_state

def provide_progressive_hint(state: MathTutorState) -> MathTutorState:
    """Provide the next level hint based on progression"""
    new_state = state.model_copy()
    
    # Generate hints if they don't exist yet
    if not new_state.hints:
        new_state = generate_hints(new_state)
    
    # Increment the hint level if not at max
    if new_state.current_hint_level < new_state.max_hint_level:
        new_state.current_hint_level += 1
    
    return new_state

def process_user_answer(state: MathTutorState) -> MathTutorState:
    """Assess the user's answer and provide feedback"""
    new_state = state.model_copy()
    
    # Skip if no user answer provided
    if not state.user_answer:
        return new_state
    
    prompt = ChatPromptTemplate.from_template(ASSESSMENT_PROMPT)
    response = llm.invoke(prompt.format(
        problem=state.problem,
        solution_steps=json.dumps(state.solution_steps),
        final_answer=state.final_answer or "Not explicitly stated",
        user_answer=state.user_answer
    ))
    
    try:
        # Parse the assessment
        assessment = json.loads(response.content)
        new_state.answer_proximity = assessment.get("proximity_score", 0.0)
        new_state.is_correct = assessment.get("correctness") == "CORRECT"
        new_state.feedback = assessment.get("feedback", "")
    except Exception as e:
        print(f"Error parsing assessment: {e}")
        # Fallback assessment
        new_state.answer_proximity = 0.0
        new_state.is_correct = False
        new_state.feedback = "Unable to assess your answer. Please try again."
    
    return new_state

def process_general_question(state: MathTutorState) -> MathTutorState:
    """Handle general (non-math) questions"""
    new_state = state.model_copy()
    
    prompt = ChatPromptTemplate.from_template(GENERAL_PROMPT)
    response = llm.invoke(prompt.format(problem=state.problem))
    
    new_state.final_response = response.content
    
    return new_state

def determine_next_suggestion(state: MathTutorState) -> MathTutorState:
    """Determine what the system should suggest based on user progress"""
    new_state = state.model_copy()
    
    # If user has the correct answer, suggest continuing
    if state.is_correct:
        new_state.system_suggestion = "continue"
        new_state.suggestion_message = "Great job! Would you like to try another problem?"
        return new_state
    
    # If user is close, suggest a hint
    if state.answer_proximity and state.answer_proximity >= 0.7:
        new_state.system_suggestion = "hint"
        new_state.suggestion_message = "You're on the right track! Would you like a hint?"
        return new_state
    
    # If user has made multiple attempts with low proximity, suggest solution
    if state.attempt_number > 2 and (not state.answer_proximity or state.answer_proximity < 0.3):
        new_state.system_suggestion = "solution"
        new_state.suggestion_message = "Would you like to see the complete solution?"
        return new_state
    
    # Default to hint for most situations
    new_state.system_suggestion = "hint"
    new_state.suggestion_message = "Would you like a hint to approach this problem?"
    return new_state

def process_user_action(state: MathTutorState) -> str:
    """Route based on user's requested action"""
    if state.requested_action == "hint":
        return "provide_hint"
    elif state.requested_action == "solution":
        return "reveal_solution"
    elif state.requested_action == "check_answer":
        return "process_user_answer"
    else:
        return "determine_suggestion"

def reveal_solution(state: MathTutorState) -> MathTutorState:
    """Reveal the solution to the user"""
    new_state = state.model_copy()
    new_state.solution_revealed = True
    return new_state

def format_response(state: MathTutorState) -> MathTutorState:
    """Format the final response"""
    new_state = state.model_copy()
    
    if state.workflow_type == "math":
        if state.solution_revealed:
            # Show full solution
            steps_text = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(state.solution_steps)])
            new_state.final_response = f"""
            Problem: {state.problem}
            
            Solution:
            {steps_text}
            
            Final Answer: {state.final_answer or "Not explicitly stated"}
            """
        elif state.current_hint_level > 0 and state.hints:
            # Show current hint level
            try:
                current_hint = next((h for h in state.hints if h.get("level") == state.current_hint_level), None)
                if current_hint:
                    new_state.final_response = f"""
                    Problem: {state.problem}
                    
                    Hint (Level {state.current_hint_level}):
                    {current_hint.get("hint", "No hint available at this level.")}
                    """
            except:
                new_state.final_response = f"Problem: {state.problem}\n\nHint unavailable."
        else:
            # Show problem and feedback if available
            new_state.final_response = f"""
            Problem: {state.problem}
            
            {state.feedback if state.feedback else ''}
            """
    # For general questions, the response was already set
    
    return new_state

def record_attempt(state: MathTutorState) -> MathTutorState:
    """Record the current attempt in history"""
    new_state = state.model_copy()
    
    # Create an attempt record
    attempt_record = {
        "problem_id": new_state.problem_id,
        "attempt_number": new_state.attempt_number,
        "timestamp": datetime.now().isoformat(),
        "requested_action": new_state.requested_action,
        "user_answer": new_state.user_answer,
        "is_correct": new_state.is_correct,
        "hint_level": new_state.current_hint_level,
        "solution_revealed": new_state.solution_revealed
    }
    
    # Add to history
    new_state.attempt_history.append(attempt_record)
    
    # Increment attempt counter for next time
    new_state.attempt_number += 1
    
    return new_state

def convert_to_user_response(state: MathTutorState) -> UserFacingResponse:
    """Convert internal state to user-facing response"""
    # Extract current hint if available
    current_hint = None
    if state.current_hint_level > 0 and state.hints:
        try:
            hint_obj = next((h for h in state.hints if h.get("level") == state.current_hint_level), None)
            if hint_obj:
                current_hint = hint_obj.get("hint")
        except:
            pass
    
    # Only include solution steps if they should be revealed
    solution_steps = state.solution_steps if state.solution_revealed else None
    
    return UserFacingResponse(
        problem=state.problem,
        problem_id=state.problem_id or "",
        solution_steps=solution_steps,
        final_answer=state.final_answer if state.solution_revealed else None,
        is_correct=state.is_correct,
        feedback=state.feedback,
        current_hint=current_hint,
        hint_level=state.current_hint_level if state.current_hint_level > 0 else None,
        suggestion=state.system_suggestion,
        suggestion_message=state.suggestion_message,
        session_id=state.session_id or "",
        attempt_number=state.attempt_number
    )

# Build the graph
def build_graph():
    """Build the workflow graph for the math tutor"""
    workflow = StateGraph(MathTutorState)
    
    # Add all nodes
    workflow.add_node("initialize_session", initialize_session)
    workflow.add_node("classify", classify_workflow)
    workflow.add_node("parse_variables", parse_variables)
    workflow.add_node("define_context", define_context)
    workflow.add_node("generate_solution", generate_solution)
    workflow.add_node("verify_solution", verify_solution)
    workflow.add_node("regenerate_solution", regenerate_solution)
    workflow.add_node("provide_hint", provide_progressive_hint)
    workflow.add_node("reveal_solution", reveal_solution)
    workflow.add_node("process_user_answer", process_user_answer)
    workflow.add_node("determine_suggestion", determine_next_suggestion)
    workflow.add_node("format_response", format_response)
    workflow.add_node("record_attempt", record_attempt)
    workflow.add_node("general_process", process_general_question)
    
    # Define the edges
    # Initial flow
    workflow.add_edge(START, "initialize_session")
    workflow.add_edge("initialize_session", "classify")
    
    # Branch based on problem type
    workflow.add_conditional_edges(
        "classify",
        router,
        {
            "math_workflow": "parse_variables",
            "general_workflow": "general_process"
        }
    )
    
    # Math problem processing flow
    workflow.add_edge("parse_variables", "define_context")
    workflow.add_edge("define_context", "generate_solution")
    workflow.add_edge("generate_solution", "verify_solution")
    
    # Verification and potential regeneration
    workflow.add_conditional_edges(
        "verify_solution",
        check_regeneration,
        {
            "regenerate": "regenerate_solution",
            "continue": "determine_suggestion"
        }
    )
    workflow.add_edge("regenerate_solution", "verify_solution")
    
    # User action processing
    workflow.add_conditional_edges(
        "determine_suggestion",
        process_user_action,
        {
            "provide_hint": "provide_hint",
            "reveal_solution": "reveal_solution",
            "check_answer": "process_user_answer",
            "determine_suggestion": "determine_suggestion"
        }
    )
    
    # Post-action flows
    workflow.add_edge("provide_hint", "format_response")
    workflow.add_edge("reveal_solution", "format_response")
    workflow.add_edge("process_user_answer", "determine_suggestion")
    workflow.add_edge("determine_suggestion", "format_response")
    workflow.add_edge("general_process", "format_response")
    
    # Final steps
    workflow.add_edge("format_response", "record_attempt")
    workflow.add_edge("record_attempt", END)
    
    return workflow.compile()

# Main entry point
def solve_problem(
    problem_text: str, 
    session_id: Optional[str] = None,
    problem_id: Optional[str] = None, 
    attempt_number: int = 1,
    user_answer: Optional[str] = None,
    requested_action: str = "none"
) -> Dict[str, Any]:
    """
    Process a problem through the math tutor workflow
    
    Args:
        problem_text: The problem or question to solve
        session_id: Optional session ID for continuing a session
        problem_id: Optional problem ID for tracking this specific problem
        attempt_number: The attempt number for this problem
        user_answer: Optional user-provided answer to assess
        requested_action: What the user wants to do (hint, solution, check_answer, none)
        
    Returns:
        User-facing response with appropriate information
    """
    # Create initial state
    initial_state = MathTutorState(
        problem=problem_text,
        session_id=session_id,
        problem_id=problem_id,
        attempt_number=attempt_number,
        user_answer=user_answer,
        requested_action=requested_action
    )
    
    # Build graph (could be cached in production)
    graph = build_graph()
    
    # Process through the graph
    result = graph.invoke(initial_state)
    
    # Convert to user-facing response
    user_response = convert_to_user_response(result)
    
    # Return as dictionary
    return user_response.dict()

# Streamlit UI (basic interface for testing)
def main():
    st.title("MathTutor - AI-Powered Learning for BBA Students")
    st.subheader("Ask any question about finance, statistics, economics, or accounting")
    
    # Initialize session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        
    if "problem_history" not in st.session_state:
        st.session_state.problem_history = []
        
    if "current_problem" not in st.session_state:
        st.session_state.current_problem = {"id": None, "text": "", "attempt": 1}
    
    # User input section
    with st.form("problem_form"):
        problem_text = st.text_area("Enter your math problem or question:", 
                               value=st.session_state.current_problem["text"],
                               height=100)
        
        # Optional user answer for checking
        user_answer = st.text_area("Your answer (optional):", height=50)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            submit_button = st.form_submit_button("Solve Problem")
        with col2:
            hint_button = st.form_submit_button("Get Hint")
        with col3:
            solution_button = st.form_submit_button("Show Solution")
    
    # Handle form submission
    if submit_button and problem_text:
        with st.spinner("Processing your problem..."):
            # New problem or continuation?
            if problem_text != st.session_state.current_problem["text"]:
                # New problem
                st.session_state.current_problem = {
                    "id": None, 
                    "text": problem_text,
                    "attempt": 1
                }
            
            # Process the problem
            result = solve_problem(
                problem_text=problem_text,
                session_id=st.session_state.session_id,
                problem_id=st.session_state.current_problem["id"],
                attempt_number=st.session_state.current_problem["attempt"],
                user_answer=user_answer if user_answer else None,
                requested_action="check_answer" if user_answer else "none"
            )
            
            # Update problem tracking
            st.session_state.current_problem["id"] = result["problem_id"]
            st.session_state.current_problem["attempt"] += 1
            
            # Display results
            st.markdown("### Response")
            st.markdown(result.get("final_response", "No response generated."))
            
            # Show feedback if provided answer
            if user_answer and result.get("feedback"):
                st.markdown("### Feedback on Your Answer")
                
                # Display correctness with color
                if result.get("is_correct"):
                    st.success("Your answer is correct! ✓")
                else:
                    st.error("Your answer needs revision. ✗")
                
                st.markdown(result["feedback"])
            
            # Show system suggestion
            if result.get("suggestion") and result.get("suggestion") != "none":
                suggestion = result.get("suggestion_message", "")
                st.info(f"Suggestion: {suggestion}")

    # Handle hint request
    elif hint_button and problem_text:
        with st.spinner("Generating hint..."):
            result = solve_problem(
                problem_text=problem_text,
                session_id=st.session_state.session_id,
                problem_id=st.session_state.current_problem["id"],
                attempt_number=st.session_state.current_problem["attempt"],
                requested_action="hint"
            )
            
            # Update problem tracking
            st.session_state.current_problem["id"] = result["problem_id"]
            st.session_state.current_problem["attempt"] += 1
            
            # Display hint
            st.markdown("### Hint")
            if result.get("current_hint"):
                st.markdown(f"**Hint Level {result.get('hint_level')}:** {result['current_hint']}")
            else:
                st.markdown("No hint available.")

    # Handle solution request
    elif solution_button and problem_text:
        with st.spinner("Generating solution..."):
            result = solve_problem(
                problem_text=problem_text,
                session_id=st.session_state.session_id,
                problem_id=st.session_state.current_problem["id"],
                attempt_number=st.session_state.current_problem["attempt"],
                requested_action="solution"
            )
            
            # Update problem tracking
            st.session_state.current_problem["id"] = result["problem_id"]
            st.session_state.current_problem["attempt"] += 1
            
            # Display solution
            st.markdown("### Complete Solution")
            
            if result.get("solution_steps"):
                for i, step in enumerate(result["solution_steps"]):
                    st.markdown(f"**Step {i+1}:** {step}")
                
                if result.get("final_answer"):
                    st.markdown(f"**Final Answer:** {result['final_answer']}")
            else:
                st.markdown("No solution available.")

# Run the Streamlit app when executed directly
if __name__ == "__main__":
    main()











