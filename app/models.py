"""
Contains the state models for the MathTutor application.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal, Union
from datetime import datetime

class MathTutorInputState(BaseModel):
    """Input state for the MathTutor - what the user provides."""
    problem: str
    requested_action: Optional[Literal["hint", "solution", "check_answer", "none"]] = "none"
    user_answer: Optional[str] = None

class MathTutorInternalState(BaseModel):
    """Internal state for the MathTutor - used for processing."""
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

class MathTutorOutputState(BaseModel):
    """Output state for the MathTutor - what the user sees."""
    problem: str
    workflow_type: Optional[Literal["math", "general"]] = None
    subject_area: Optional[str] = None
    solution_steps: List[str] = Field(default_factory=list)
    final_answer: Optional[str] = None
    current_hint: Optional[str] = None
    is_correct: Optional[bool] = None
    feedback: Optional[str] = None
    final_response: Optional[str] = None
    error_message: Optional[str] = None
    suggestion_message: Optional[str] = None 