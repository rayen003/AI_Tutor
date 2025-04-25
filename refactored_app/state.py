from typing import TypedDict, List, Optional, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
# from pydantic.v1 import BaseModel, Field # Use pydantic.v1 compatibility
from pydantic import BaseModel, Field # Use Pydantic v2

# --- State Definition ---
class SimpleTutorState(TypedDict):
    thread_id: str
    problem: str
    user_answer: Optional[str]
    action: str # 'solve', 'hint', 'assess'
    complexity: Optional[str] # 'simple', 'medium', 'complex' - determined by router
    plan: Optional[List[str]] # Steps for complex problems
    solution_steps: Optional[List[str]]
    correct_answer: Optional[str]
    feedback: Optional[str] # For user answer assessment
    hint: Optional[str] # Current hint
    hint_level: Optional[int] # Index of the last hint provided
    error: Optional[str] # To capture errors in nodes
    is_correct: Optional[bool] # Flag indicating if the user provided the correct answer
    # Verification fields
    is_solution_valid: Optional[bool] # Status of the tutor-generated solution
    verification_reason: Optional[str] # Reason if status is invalid
    retry_count: Optional[int] # Counter for verification retries
    # Use add_messages for history - it handles accumulation correctly
    chat_history: Annotated[list[BaseMessage], add_messages]


# --- Structured Output Models (Pydantic v2) ---
class TutorSolution(BaseModel):
    """Structured format for the step-by-step solution and final answer."""
    steps: List[str] = Field(description="The detailed step-by-step solution.")
    final_answer: str = Field(description="The final answer derived from the steps.")

class PlanningSteps(BaseModel):
    """Structured format for planning steps."""
    plan: List[str] = Field(description="A list of high-level steps to solve the problem.")

class VerificationResult(BaseModel):
    """Structured format for the solution verification result."""
    is_valid: bool = Field(description="True if the solution is valid, False otherwise.")
    reason: Optional[str] = Field(default=None, description="The reason why the solution is invalid (if applicable). Should be None if is_valid is True.") 