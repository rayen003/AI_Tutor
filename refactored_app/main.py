import os
import uuid
import sys

# Ensure the parent directory is in the Python path
# to allow imports like `from refactored_app.graph import app`
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Local imports
from .graph import app # Import the compiled graph
from .state import SimpleTutorState # Import state definition if needed for initial input typing

def run_simulation():
    """Runs the interactive command-line simulation for the Math Tutor."""
    # --- Disable LangSmith Tracing (Optional) ---
    # os.environ["LANGCHAIN_TRACING_V2"] = "false"

    # print("--- Math Tutor Simulation (Refactored - Proactive Plot Test) ---")
    print("--- Math Tutor Simulation (Refactored) ---") # Revert title

    # --- Restore User Input --- #
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    problem = input("Enter the math problem: ").strip()
    # ------------------------ #

    print(f"Problem: {problem}")

    current_state_dict = None # Store the latest state dictionary

    # --- Turn 1: Ask Tutor to Solve --- (Automatically first step)
    print("\n--- Asking Tutor to Solve --- ")
    initial_input = {
        "problem": problem,
        "action": "solve",
        "chat_history": [] # Start clean
    }
    try:
        # --- Use stream instead of invoke --- #
        print("[Streaming...]\n")
        last_state_after_solve = None
        for event in app.stream(initial_input, config=config, stream_mode="values"):
            # stream_mode="values" yields the state after each step
            # print(f"Event keys: {event.keys()}") # For debugging stream content
            # The event is the full state dict after the node runs
            current_state_dict = event
            # You could print node executions here if desired, but the node logs do that
            # Example: find the node that just ran (needs more complex event parsing)
            print(".", end="", flush=True) # Print dot for progress
            last_state_after_solve = current_state_dict # Keep track of last state in solve phase

        print("\n[...Stream Complete]")

        # --- Process final state after initial solve --- #
        current_state_dict = last_state_after_solve # Use the state after solve finishes

        if current_state_dict:
            if current_state_dict.get("error"):
                 error_msg = current_state_dict['error']
                 print(f"\n>> Tutor Error: {error_msg}")
                 # Check if error was due to max retries
                 # if "Max retries" in current_state_dict.get("error", "") or state.get("retry_count", 0) > 2:
                 if "verification failed after" in error_msg.lower() and "retries" in error_msg.lower():
                     print(">> Tutor: I tried several times but couldn't generate a verified solution. Please check the problem or try rephrasing.")
                 # Reset error in state? No, let it persist until next node potentially clears it.

            # Display solution ONLY if no critical error occurred (like max retries)
            if not current_state_dict.get("error") or ("verification failed after" not in current_state_dict.get("error", "").lower()):
                if current_state_dict.get("solution_steps"):
                    print("\n>> Tutor Solution:")
                    for i, step in enumerate(current_state_dict["solution_steps"]):
                        print(f"  Step {i+1}: {step}")
                    print(f"  Final Answer: {current_state_dict.get('correct_answer', 'Not found')}")
                elif not current_state_dict.get("error"):
                    print("\n>> Tutor: I couldn't generate a solution.")
        else:
            print("\n>> Tutor: Failed to process the initial request.")
            return

    except Exception as e:
        print(f"\n>> An unexpected error occurred during initial processing: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Subsequent Turns (Loop with User Input & Streaming) ---
    turn_counter = 1
    while True:
        turn_counter += 1
        print(f"\n--- Turn {turn_counter} --- ")
        user_input = input("Your answer (or type 'hint' or 'quit'): ").strip()

        if user_input.lower() == 'quit':
            print("Quitting simulation.")
            break
        elif user_input.lower() == 'hint':
            action = "hint"
            user_answer = None
            print("Asking for hint...")
        else:
            action = "assess"
            user_answer = user_input
            print(f"Submitting answer: {user_answer}")

        if not current_state_dict or "problem" not in current_state_dict:
             print("Error: Lost problem context. Cannot continue.")
             break

        turn_input = {
            "problem": current_state_dict["problem"],
            "action": action,
            "user_answer": user_answer,
            "hint": None,
            "feedback": None
        }

        try:
            # --- Use stream for subsequent turns --- #
            print("[Streaming...]\n")
            last_state_this_turn = None
            final_node_output = None
            for event in app.stream(turn_input, config=config, stream_mode="updates"):
                 # stream_mode="updates" yields delta {node_name: output}
                 # print(f"Event: {event}") # For debugging stream content
                 print(".", end="", flush=True)
                 # Capture the actual output of the *last* node before interrupt/end
                 # This requires knowing the node name that produced the final output
                 # For simplicity now, we just grab the state after the stream ends.
                 pass # Processing updates is complex, get final state below

            print("\n[...Stream Complete]")
            # Get the final state after the stream finishes for this turn
            # Note: current_state_dict from *previous* turn is used if stream fails/empty
            final_state = app.get_state(config)
            if final_state and final_state.values:
                 current_state_dict = final_state.values # Update state from checkpointer
            else:
                 print("Warning: Could not retrieve final state after stream.")
                 # Keep using previous state? May lead to issues.

            # --- Process final state for this turn --- #
            if current_state_dict:
                if current_state_dict.get("error"):
                     print(f"\n>> Tutor Error: {current_state_dict['error']}")
                if action == "assess" and current_state_dict.get("feedback"):
                     print(f"\n>> Tutor Feedback: {current_state_dict['feedback']}")
                elif action == "hint" and current_state_dict.get("hint"):
                     # Hint is now printed token-by-token within the node
                     # print(f"\n>> {current_state_dict['hint']}") # Remove this line
                     pass # Node handles printing, maybe just ensure newline if needed
                elif not current_state_dict.get("error") and not current_state_dict.get("feedback") and not current_state_dict.get("hint"):
                    print("\n>> Tutor: OK.")
            else:
                print("\n>> Tutor: Failed to process your request this turn.")
                break

        except Exception as e:
             print(f"\n>> An unexpected error occurred this turn: {e}")
             import traceback
             traceback.print_exc()

    print("\n--- Simulation Complete --- ")

if __name__ == "__main__":
    run_simulation() 