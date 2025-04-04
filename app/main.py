"""
Main file for the MathTutor application.
Contains graph definitions and workflow composition.
"""

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

from app.models import MathTutorInputState, MathTutorInternalState, MathTutorOutputState
from app.math_services.services.node_services import (
    # Core processing functions
    initialize_session,
    classify_workflow,
    parse_variables,
    define_context,
    generate_solution,
    generate_hints,
    provide_progressive_hint,
    check_answer,
    determine_next_suggestion,
    format_response,
    record_attempt,
    general_question,
    convert_to_output_state,
    
    # Routing functions
    router,
    check_regeneration
)

# Import the verification service
from app.math_services.services.verification import verify_solution_steps, regenerate_solution_steps

# Import tool integrations
from app.math_services.tools.tool_integration import MathToolkit

# Load environment variables
load_dotenv()

# Initialize the LLM with tools
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0,
    request_timeout=60,
    max_retries=2
)

# Set up tools for enhanced math tutoring
math_tools = MathToolkit.get_all_tools()

# Create the math workflow
def create_math_workflow():
    """Create the workflow for math problems"""
    math_workflow = StateGraph(MathTutorInternalState)
    
    # Define the nodes
    math_workflow.add_node("parse_variables", parse_variables)
    math_workflow.add_node("define_context", define_context)
    math_workflow.add_node("generate_solution", generate_solution)
    math_workflow.add_node("verify_solution", verify_solution_steps)
    math_workflow.add_node("regenerate_solution", regenerate_solution_steps)
    math_workflow.add_node("generate_hints", generate_hints)
    math_workflow.add_node("provide_hint", provide_progressive_hint)
    math_workflow.add_node("process_user_answer", check_answer)
    math_workflow.add_node("determine_next_suggestion", determine_next_suggestion)
    math_workflow.add_node("format_response", format_response)
    math_workflow.add_node("record_attempt", record_attempt)
    
    # Define the edges
    math_workflow.add_edge(START, "parse_variables")
    math_workflow.add_edge("parse_variables", "define_context")
    math_workflow.add_edge("define_context", "generate_solution")
    math_workflow.add_edge("generate_solution", "verify_solution")
    
    # Conditional edge for regeneration
    math_workflow.add_conditional_edges(
        "verify_solution",
        check_regeneration,
        {
            "regenerate": "regenerate_solution",
            "continue": "generate_hints"
        }
    )
    
    math_workflow.add_edge("regenerate_solution", "verify_solution")
    math_workflow.add_edge("generate_hints", "determine_next_suggestion")
    
    # Add edges for requested actions
    math_workflow.add_edge("provide_hint", "determine_next_suggestion")
    math_workflow.add_edge("process_user_answer", "determine_next_suggestion")
    
    math_workflow.add_edge("determine_next_suggestion", "format_response")
    math_workflow.add_edge("format_response", "record_attempt")
    math_workflow.add_edge("record_attempt", END)
    
    return math_workflow.compile()

# Create the main graph
def create_math_tutor_graph():
    """Create the main graph for the math tutor application"""
    graph = StateGraph(MathTutorInternalState)
    
    # Add the main nodes
    graph.add_node("initialize_session", initialize_session)
    graph.add_node("classify_workflow", classify_workflow)
    graph.add_node("general_question", general_question)
    
    # Add the math workflow as a subgraph
    math_workflow = create_math_workflow()
    graph.add_node("math_workflow", math_workflow)
    
    # Add the edges
    graph.add_edge(START, "initialize_session")
    graph.add_edge("initialize_session", "classify_workflow")
    
    # Conditional edge based on workflow type
    graph.add_conditional_edges(
        "classify_workflow",
        router,
        {
            "math_workflow": "math_workflow",
            "general_workflow": "general_question"
        }
    )
    
    graph.add_edge("general_question", END)
    graph.add_edge("math_workflow", END)
    
    return graph.compile()

# Create the workflow app
app = create_math_tutor_graph()

# Main function to process a problem
def process_problem(input_state: MathTutorInputState) -> MathTutorOutputState:
    """Process a math problem through the workflow"""
    # Convert input state to internal state
    internal_state = MathTutorInternalState(
        problem=input_state.problem,
        workflow_type="math",  # Default to math until classified
        requested_action=input_state.requested_action,
        user_answer=input_state.user_answer
    )
    
    # Run the workflow
    result = app.invoke(internal_state)
    
    # Convert to output state
    return convert_to_output_state(result)

# Function to get all available math tools
def get_available_tools():
    """Get a list of all available tools for the MathTutor"""
    return MathToolkit.get_interactive_tools()

def run_app():
    """
    Run the math tutor application.
    """
    app = create_math_tutor_graph()
    return app

if __name__ == "__main__":
    # For local development, test a specific problem
    from app.models import MathTutorInputState
    from app.prompts import create_test_problems
    
    # Create a test problem
    problems = create_test_problems()
    
    # Initialize with the first problem
    state = MathTutorInputState(
        problem=problems[0]["problem"],
        context=problems[0]["context"],
        user_answer=None,
        action="solve"
    )
    
    print(f"Testing Math Tutor with problem: {state.problem}")
    
    # Create and run the graph
    graph = create_math_tutor_graph()
    final_state = graph.invoke(state)
    
    print("\nFinal state:")
    print(f"Problem: {final_state.get('problem')}")
    print(f"Solution steps: {len(final_state.get('solution_steps', []))}")
    print(f"Final answer: {final_state.get('final_answer')}")
    print(f"Current hint: {final_state.get('current_hint')}")
    print(f"Is correct: {final_state.get('is_correct')}")
    print(f"Feedback: {final_state.get('feedback')}")
