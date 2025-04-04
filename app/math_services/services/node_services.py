"""
Node functions for the MathTutor graph.
These functions serve as the nodes in the MathTutor workflow graph.
"""

import os
import re
import json
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.models import MathTutorInternalState, MathTutorOutputState
from app.prompts import PROMPTS

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0,
    request_timeout=60,
    max_retries=2
)

# IMPROVED JSON PARSING FUNCTION
def safe_json_parse(content, default_value):
    """Safely parse JSON content from API responses."""
    if not content or not content.strip():
        return default_value
    
    # Try direct parsing first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks
    json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(json_pattern, content)
    
    if matches:
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # Try to extract anything that looks like a JSON object or array
    json_object_pattern = r"(\{[\s\S]*\}|\[[\s\S]*\])"
    matches = re.findall(json_object_pattern, content)
    
    if matches:
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    return default_value

# Core processing functions
def initialize_session(state: MathTutorInternalState) -> MathTutorInternalState:
    """Initialize or continue a session with proper tracking IDs"""
    new_state = state.copy()
    
    if not new_state.session_id:
        new_state.session_id = str(uuid.uuid4())
    
    if not new_state.problem_id:
        new_state.problem_id = f"prob_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    new_state.timestamp = datetime.now().isoformat()
    
    return new_state

def classify_workflow(state: MathTutorInternalState) -> MathTutorInternalState:
    """Determine if this is a math problem or general question"""
    new_state = state.copy()
    
    prompt = ChatPromptTemplate.from_template(PROMPTS["classify"])
    response = llm.invoke(prompt.format(problem=state.problem))
    
    classification = response.content.strip().lower()
    if classification in ["math", "general"]:
        new_state.workflow_type = classification
    else:
        new_state.workflow_type = "math"
    
    return new_state

def parse_variables(state: MathTutorInternalState) -> MathTutorInternalState:
    """Extract variables and their values from the problem"""
    new_state = state.copy()
    
    prompt = ChatPromptTemplate.from_template(PROMPTS["variable_parsing"])
    
    try:
        response = llm.invoke(prompt.format(problem=state.problem))
        variables = safe_json_parse(response.content, {})
        new_state.variables = variables
    except Exception as e:
        print(f"Error parsing variables: {e}")
        new_state.variables = {}
    
    return new_state

def define_context(state: MathTutorInternalState) -> MathTutorInternalState:
    """Identify subject area and relevant mathematical context"""
    new_state = state.copy()
    
    prompt = ChatPromptTemplate.from_template(PROMPTS["context"])
    
    try:
        response = llm.invoke(prompt.format(
            problem=state.problem,
            variables=json.dumps(state.variables)
        ))
        
        context_data = safe_json_parse(response.content, {
            "subject_area": "General Business Mathematics",
            "context": "General problem-solving approach"
        })
        
        new_state.subject_area = context_data.get("subject_area")
        new_state.context = context_data.get("context")
    except Exception as e:
        print(f"Error defining context: {e}")
        new_state.subject_area = "General Business Mathematics"
        new_state.context = "General problem-solving approach"
    
    return new_state

def extract_final_answer(steps: List[str]) -> Optional[str]:
    """Extract the final answer from solution steps"""
    for step in reversed(steps):
        step = step.lower()
        if "final answer:" in step:
            return step.split("final answer:")[-1].strip()
        if "final answer is" in step:
            return step.split("final answer is")[-1].strip()
        if "answer:" in step:
            return step.split("answer:")[-1].strip()
        if "answer is" in step:
            return step.split("answer is")[-1].strip()
        if "$" in step and any(x.isdigit() for x in step):
            # Try to extract a dollar amount as last resort
            matches = re.findall(r'\$[\d,]+\.?\d*', step)
            if matches:
                return matches[-1]
    return None

def generate_solution(state: MathTutorInternalState) -> MathTutorInternalState:
    """Generate a step-by-step solution for the problem"""
    new_state = state.copy()
    
    prompt = ChatPromptTemplate.from_template(PROMPTS["solution"])
    
    try:
        response = llm.invoke(prompt.format(
            problem=state.problem,
            subject_area=state.subject_area,
            context=state.context,
            variables=json.dumps(state.variables)
        ))
        
        # Parse the response into steps
        steps = [step.strip() for step in response.content.split("\n") if step.strip()]
        new_state.solution_steps = steps
        new_state.solution_generated = True
        
        # Extract final answer
        final_answer = extract_final_answer(steps)
        if final_answer:
            new_state.final_answer = final_answer
            new_state.final_response = f"# Solution\n\n{''.join(steps)}\n\n**Final Answer:** {final_answer}"
        else:
            new_state.final_response = f"# Solution\n\n{''.join(steps)}"
            
    except Exception as e:
        print(f"Error generating solution: {e}")
        new_state.error_message = "Failed to generate solution"
    
    return new_state

def generate_hints(state: MathTutorInternalState) -> MathTutorInternalState:
    """Generate a sequence of hints for the problem"""
    new_state = state.copy()
    
    prompt = ChatPromptTemplate.from_template(PROMPTS["hints"])
    
    try:
        response = llm.invoke(prompt.format(
            problem=state.problem,
            subject_area=state.subject_area,
            context=state.context,
            variables=json.dumps(state.variables)
        ))
        
        # Parse hints from response
        hints = []
        current_hint = []
        
        for line in response.content.split("\n"):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("Hint") or line.startswith("#"):
                if current_hint:
                    hints.append("\n".join(current_hint))
                current_hint = [line]
            else:
                current_hint.append(line)
                
        if current_hint:
            hints.append("\n".join(current_hint))
            
        new_state.hints = hints
        new_state.max_hint_level = len(hints)
        
    except Exception as e:
        print(f"Error generating hints: {e}")
        new_state.error_message = "Failed to generate hints"
        
    return new_state

def provide_progressive_hint(state: MathTutorInternalState) -> MathTutorInternalState:
    """Provide the next hint in the sequence"""
    new_state = state.copy()
    
    # Generate hints if they don't exist
    if not state.hints:
        new_state = generate_hints(new_state)
        if new_state.error_message:
            return new_state
    
    if state.current_hint_level >= state.max_hint_level:
        new_state.error_message = "Maximum hint level reached"
        return new_state
    
    new_state.current_hint_level += 1
    hint_index = min(new_state.current_hint_level - 1, len(state.hints) - 1)
    
    try:
        current_hint = state.hints[hint_index]
        if isinstance(current_hint, dict) and "hint" in current_hint:
            new_state.final_response = f"# Hint Level {new_state.current_hint_level}\n\n{current_hint['hint']}"
        else:
            new_state.final_response = f"# Hint Level {new_state.current_hint_level}\n\n{str(current_hint)}"
    except Exception as e:
        print(f"Error providing hint: {e}")
        new_state.error_message = "Failed to provide hint"
    
    return new_state

def check_answer(state: MathTutorInternalState) -> MathTutorInternalState:
    """Check if the user's answer is correct"""
    new_state = state.copy()
    
    if not new_state.solution_generated:
        new_state = generate_solution(new_state)
        if new_state.error_message:
            return new_state
            
    if not new_state.final_answer:
        new_state.error_message = "Cannot check answer: solution not available"
        return new_state
        
    try:
        # Clean up answers for comparison
        user_answer = re.sub(r'[^\d.]', '', state.user_answer)
        correct_answer = re.sub(r'[^\d.]', '', new_state.final_answer)
        
        # Convert to float for numerical comparison
        user_value = float(user_answer)
        correct_value = float(correct_answer)
        
        # Calculate relative difference
        relative_diff = abs(user_value - correct_value) / correct_value
        new_state.answer_proximity = 1 - relative_diff
        
        # Check if answer is correct within 1% margin
        if relative_diff <= 0.01:
            new_state.is_correct = True
            new_state.feedback = "Correct! Your answer matches the solution."
            new_state.final_response = f"# Correct!\n\nYour answer of {state.user_answer} is correct!"
        else:
            new_state.is_correct = False
            if relative_diff <= 0.1:
                new_state.feedback = "Close, but not quite correct. Try reviewing your calculations."
                new_state.final_response = "# Not quite there yet\n\nYour answer is close, but not quite correct. Would you like a hint?"
            else:
                new_state.feedback = "Your answer is incorrect. Consider using a different approach."
                new_state.final_response = "# Incorrect\n\nYour answer is not correct. Would you like a hint to help you solve this problem?"
                
    except Exception as e:
        print(f"Error checking answer: {e}")
        new_state.error_message = "Failed to check answer"
        
    return new_state

def determine_next_suggestion(state: MathTutorInternalState) -> MathTutorInternalState:
    """Determine what to suggest to the user next based on their progress"""
    new_state = state.copy()
    
    if new_state.requested_action == "solution":
        new_state.system_suggestion = "none"
        new_state.suggestion_message = "Would you like to try another problem?"
        return new_state
    
    if new_state.is_correct:
        new_state.system_suggestion = "none"
        new_state.suggestion_message = "Great job! You've solved this problem correctly. Would you like to try another one?"
        return new_state
    
    if new_state.requested_action == "hint":
        if new_state.current_hint_level >= new_state.max_hint_level:
            new_state.system_suggestion = "solution"
            new_state.suggestion_message = "Would you like to see the complete solution?"
        else:
            new_state.system_suggestion = "hint"
            new_state.suggestion_message = "Would you like another hint?"
        return new_state
    
    if new_state.attempt_number <= 1 or new_state.answer_proximity is not None and new_state.answer_proximity < 0.5:
        new_state.system_suggestion = "hint"
        new_state.suggestion_message = "Would you like a hint to help you solve this problem?"
    elif new_state.attempt_number >= 3:
        new_state.system_suggestion = "solution"
        new_state.suggestion_message = "Would you like to see the solution to this problem?"
    else:
        new_state.system_suggestion = "continue"
        new_state.suggestion_message = "You're on the right track. Try once more with the feedback in mind."
    
    return new_state

def format_response(state: MathTutorInternalState) -> MathTutorInternalState:
    """Format the response to present to the user"""
    new_state = state.copy()
    
    if state.requested_action == "solution":
        solution_text = "\n\n".join([f"Step {i+1}: {step}" for i, step in enumerate(state.solution_steps)])
        new_state.final_response = f"# Solution\n\n{solution_text}\n\n**Final Answer:** {state.final_answer}"
        new_state.solution_revealed = True
    
    elif state.requested_action == "hint" and state.current_hint_level > 0 and state.hints:
        hint_index = min(state.current_hint_level - 1, len(state.hints) - 1)
        current_hint = state.hints[hint_index]
        
        if isinstance(current_hint, dict) and "hint" in current_hint:
            hint_text = current_hint["hint"]
        else:
            hint_text = str(current_hint)
            
        new_state.final_response = f"# Hint\n\n{hint_text}"
    
    elif state.requested_action == "check_answer" and state.user_answer:
        if state.is_correct:
            new_state.final_response = f"# Correct!\n\n{state.feedback}"
        else:
            new_state.final_response = f"# Not quite there yet\n\n{state.feedback}"
            
            if state.suggestion_message:
                new_state.final_response += f"\n\n{state.suggestion_message}"
    
    else:
        new_state.final_response = "I'm ready to help with your math problem. What would you like to do next?"
    
    return new_state

def record_attempt(state: MathTutorInternalState) -> MathTutorInternalState:
    """Record the current attempt in the history"""
    new_state = state.copy()
    
    if not state.user_answer:
        return new_state
    
    attempt = {
        "attempt_number": state.attempt_number,
        "timestamp": datetime.now().isoformat(),
        "user_answer": state.user_answer,
        "is_correct": state.is_correct,
        "feedback": state.feedback,
        "hints_used": state.current_hint_level
    }
    
    new_state.attempt_history.append(attempt)
    new_state.attempt_number += 1
    
    return new_state

def general_question(state: MathTutorInternalState) -> MathTutorInternalState:
    """Process a general question rather than a math problem"""
    new_state = state.copy()
    
    prompt = ChatPromptTemplate.from_template(PROMPTS["general"])
    response = llm.invoke(prompt.format(problem=state.problem))
    
    new_state.final_response = response.content
    
    return new_state

def convert_to_output_state(state: dict) -> MathTutorOutputState:
    """Convert internal state to output state for the user"""
    return MathTutorOutputState(
        problem=state.get("problem", ""),
        workflow_type=state.get("workflow_type"),
        subject_area=state.get("subject_area"),
        solution_steps=state.get("solution_steps", []),
        final_answer=state.get("final_answer"),
        current_hint=state.get("hints", [])[state.get("current_hint_level", 0) - 1] if state.get("hints") and state.get("current_hint_level", 0) > 0 else None,
        is_correct=state.get("is_correct"),
        feedback=state.get("feedback"),
        final_response=state.get("final_response"),
        error_message=state.get("error_message"),
        suggestion_message=state.get("suggestion_message")
    )

# Routing functions
def router(state: MathTutorInternalState) -> str:
    """Router to determine next processing step"""
    if state.workflow_type == "math":
        return "math_workflow"
    else:
        return "general_workflow"

def check_regeneration(state: MathTutorInternalState) -> str:
    """Check if solution needs regeneration"""
    if state.needs_regeneration and state.regeneration_attempts < state.max_regeneration_attempts:
        return "regenerate"
    return "continue" 