"""
Test file for the MathTutor graph with various scenarios and logging.
"""

import os
import logging
from dotenv import load_dotenv
from app.main import create_math_tutor_graph, MathTutorInputState, process_problem
from IPython.display import Image, display
import graphviz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('math_tutor_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def visualize_graph():
    """Visualize the MathTutor graph using graphviz."""
    graph = create_math_tutor_graph()
    dot = graph.get_graph().draw_mermaid_png()
    display(Image(dot))

def test_math_problem(problem: str, action: str = "none", user_answer: str = None):
    """Test a math problem scenario with logging."""
    logger.info(f"Testing math problem: {problem}")
    logger.info(f"Action: {action}, User answer: {user_answer}")
    
    # Create input state
    input_state = MathTutorInputState(
        problem=problem,
        requested_action=action,
        user_answer=user_answer
    )
    
    # Process the problem
    result = process_problem(input_state)
    
    # Log results
    logger.info(f"Workflow type: {result.workflow_type}")
    logger.info(f"Subject area: {result.subject_area}")
    logger.info(f"Solution steps generated: {len(result.solution_steps) if result.solution_steps else 0}")
    logger.info(f"Final answer: {result.final_answer}")
    logger.info(f"Current hint: {result.current_hint}")
    logger.info(f"Is correct: {result.is_correct}")
    logger.info(f"Feedback: {result.feedback}")
    logger.info(f"Final response: {result.final_response}")
    logger.info(f"Error message: {result.error_message}")
    logger.info(f"Suggestion message: {result.suggestion_message}")
    
    return result

def test_general_question(question: str):
    """Test a general question scenario with logging."""
    logger.info(f"Testing general question: {question}")
    
    # Create input state
    input_state = MathTutorInputState(
        problem=question,
        requested_action="none",
        user_answer=None
    )
    
    # Process the question
    result = process_problem(input_state)
    
    # Log results
    logger.info(f"Workflow type: {result.workflow_type}")
    logger.info(f"Final response: {result.final_response}")
    logger.info(f"Error message: {result.error_message}")
    
    return result

def run_test_scenarios():
    """Run various test scenarios for the MathTutor."""
    logger.info("Starting MathTutor test scenarios")
    
    # Test 1: Simple math problem with solution request
    logger.info("\nTest 1: Simple math problem with solution request")
    test_math_problem(
        "What is the present value of $10,000 to be received in 5 years at an 8% interest rate?",
        action="solution"
    )
    
    # Test 2: Math problem with hint request
    logger.info("\nTest 2: Math problem with hint request")
    test_math_problem(
        "Calculate the future value of $5,000 invested at 6% for 3 years.",
        action="hint"
    )
    
    # Test 3: Math problem with answer check
    logger.info("\nTest 3: Math problem with answer check")
    test_math_problem(
        "If you invest $2,000 at 5% interest for 4 years, what is the future value?",
        action="check_answer",
        user_answer="$2,431.01"
    )
    
    # Test 4: General question
    logger.info("\nTest 4: General question")
    test_general_question(
        "What is the difference between simple and compound interest?"
    )
    
    logger.info("All test scenarios completed")

if __name__ == "__main__":
    # Run test scenarios
    run_test_scenarios() 