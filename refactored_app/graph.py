from langgraph.graph import StateGraph, END, START
# from langgraph.checkpoint.sqlite import SqliteSaver 
# from langgraph_checkpoint.sqlite import SqliteSaver 
from langgraph.checkpoint.memory import MemorySaver # Use in-memory saver

# Local imports
from .state import SimpleTutorState
from .nodes import (
    initialize,
    action_router_node_logic,
    route_action,
    classify_complexity,
    plan_cot,
    generate_cot,
    assess_answer,
    generate_hint,
    verify_solution # Import new node
)

# --- Graph Definition ---
workflow = StateGraph(SimpleTutorState)

# Add nodes
workflow.add_node("initialize", initialize)
workflow.add_node("action_router", action_router_node_logic) # Dummy node
workflow.add_node("classify_complexity", classify_complexity)
workflow.add_node("plan_cot", plan_cot)
workflow.add_node("generate_cot", generate_cot)
workflow.add_node("assess_answer", assess_answer)
workflow.add_node("generate_hint", generate_hint)
workflow.add_node("verify_solution", verify_solution) # Add the verification node

# Add edges
workflow.add_edge(START, "initialize")

# Routing from initialization based on action
workflow.add_conditional_edges(
    "initialize",
    route_action, # Use the routing logic function
    {
        "classify_complexity": "classify_complexity", # Initial solve action
        "assess_answer": "assess_answer",
        "generate_hint": "generate_hint",
        END: END # Handle unknown actions
    }
)

# After complexity classification, decide on planning or direct CoT
workflow.add_conditional_edges(
    "classify_complexity",
    lambda state: state.get("complexity", "medium"), # Branch on complexity
    {
        "simple": "generate_cot",
        "medium": "plan_cot",
        "complex": "plan_cot"
    }
)

# After planning, generate the detailed solution
workflow.add_edge("plan_cot", "generate_cot")

# After generating CoT, go to verification
# workflow.add_edge("generate_cot", END) # Old edge
workflow.add_edge("generate_cot", "verify_solution") # New edge

# Add conditional edge after verification
def check_verification(state: SimpleTutorState):
    is_valid = state.get("is_solution_valid") # New boolean check
    reason = state.get("verification_reason", "Unknown reason") # Get reason for logging
    retry_count = state.get("retry_count", 0) # Get current retry count
    max_retries = 2 # Define max retries

    # Handle case where verification itself errored
    if state.get("error") and "Verification failed" in state.get("error", ""):
         print(f"DEBUG: Verification node encountered an error: {state['error']}. Stopping flow.")
         # Clear retry count on stopping flow?
         # return {"retry_count": 0, "end": True} # Need to check how to update state AND route
         # Routing function only returns the next node name or END
         return END # Stop the flow if verification itself failed catastrophically

    if is_valid is True:
        print("DEBUG: Verification successful. Ending solve path.")
        # It's valid, reset retry count for future operations if needed
        # We should return the state update separately if graph allowed
        # Since it doesn't, reset happens in `initialize` on next call
        return END
    else: # is_valid is False or None
        current_retry = retry_count + 1
        print(f"DEBUG: Verification failed (Attempt {current_retry}/{max_retries}, Reason='{reason}')")
        if current_retry > max_retries:
            print(f"DEBUG: Max retries ({max_retries}) exceeded. Ending solve path with error.")
            # Need a way to signal this specific failure state
            # For now, just end. Consider adding a specific error message node later.
            # TODO: Set a final error message in state?
            return END
        else:
            print(f"DEBUG: Looping back to generate_cot.")
            # Need to update retry_count in the state. How?
            # Node return values update state. Routing functions don't.
            # We need to add retry_count update to the node *before* this edge.
            # --> Let's modify verify_solution to return the incremented count on failure.
            return "regenerate"

# --- Modification needed in verify_solution node too! ---

workflow.add_conditional_edges(
    "verify_solution",
    check_verification,
    {
        END: END,
        "regenerate": "generate_cot"
    }
)

# After assessing the answer, end the 'assess' path
workflow.add_edge("assess_answer", END)

# After generating a hint, end the 'hint' path
workflow.add_edge("generate_hint", END)


# --- Compile the Graph ---
# Nodes after which the graph should wait for human input
# Note: Verification happens *before* interrupt usually
interrupt_nodes = ["assess_answer", "generate_hint"]
# If we want to show the user the verified solution before they interact,
# we could potentially interrupt *after* verify_solution if status is 'valid'.
# For now, let's assume the 'solve' action runs fully through generation and verification.

# Add persistence using MemorySaver
checkpointer = MemorySaver() # Use MemorySaver

# Compile the graph with the checkpointer
app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_after=interrupt_nodes
)

# Optional: Generate a diagram
try:
    app.get_graph().draw_mermaid_png(output_file_path="refactored_app_graph.png")
    print("Graph diagram saved to refactored_app_graph.png")
except Exception as e:
    print(f"Could not generate graph diagram: {e}") 