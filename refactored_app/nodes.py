import uuid
from langgraph.graph import END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import OutputParserException

# Local imports from the refactored app structure
from .state import SimpleTutorState, TutorSolution, PlanningSteps, VerificationResult
from .prompts import (
    CLASSIFY_COMPLEXITY_SYSTEM,
    PLAN_COT_SYSTEM,
    GENERATE_COT_SIMPLE_SYSTEM,
    GENERATE_COT_PLAN_SYSTEM,
    ASSESS_ANSWER_SYSTEM,
    CHECK_CORRECTNESS_SYSTEM,
    GENERATE_HINT_SYSTEM,
    VERIFY_SOLUTION_SYSTEM
)
from .tools import tools # Import the tool list

# --- LLM Setup (Consider moving to a config or graph file if shared across nodes)
# For now, keep it here as nodes directly use it.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0) # Use low temp for verification
llm_with_tools = llm.bind_tools(tools)

# --- Graph Node Functions ---

def initialize(state: SimpleTutorState) -> dict:
    """Sets up the initial state, only applying defaults if keys are missing."""
    update_dict = {}

    # Ensure thread_id exists - essential for checkpointer config
    # StateGraph ensures thread_id is in config, but good practice to handle here if needed.
    # If checkpointer loads state, thread_id will be present.
    # If first run, thread_id from config will be used by checkpointer.
    # No need to explicitly set thread_id here if using checkpointer correctly.

    # Define defaults for fields that should have a value if missing
    defaults = {
        "complexity": None,
        "plan": None,
        "solution_steps": None,
        "correct_answer": None,
        "feedback": None,
        "hint": None,
        "hint_level": -1,
        "error": None,
        "is_solution_valid": None,
        "verification_reason": None,
        "retry_count": 0 # Initialize retry count
        # chat_history managed below and by add_messages
    }

    # Apply defaults only if the key is MISSING in the current state
    for key, default_value in defaults.items():
        # Reset retry count whenever initialize runs (start of new external call)
        if key == "retry_count":
            update_dict[key] = 0
            continue

        if key not in state or state[key] is None: # Also apply if key exists but is None (e.g., reset error)
             # Exception: don't reset hint_level if it's >= 0
             if key == "hint_level" and state.get("hint_level", -1) >= 0:
                 continue
             # Exception: don't reset solution_steps/correct_answer if they exist
             if key in ["solution_steps", "correct_answer"] and state.get(key):
                 continue

             update_dict[key] = default_value

    # Ensure chat_history list exists if missing (add_messages handles accumulation)
    if "chat_history" not in state:
         update_dict["chat_history"] = []

    # Add the initial problem to history only ONCE
    # Check the potentially loaded history
    history = state.get("chat_history", [])
    problem = state.get("problem") # Problem comes from invoke input
    if problem and not any(isinstance(msg, HumanMessage) and msg.content == problem for msg in history):
        # Use add_messages semantics: return the message to be added
        # Note: This assumes 'problem' is consistently passed in invoke calls where relevant
        # If problem is ONLY passed on the first invoke, this check is sufficient.
        if "chat_history" in update_dict: # If we just created the list
             update_dict["chat_history"].append(HumanMessage(content=problem))
        else: # If history exists but missing problem, return message for add_messages
             update_dict["chat_history"] = [HumanMessage(content=problem)]

    # Return only the things that NEED to be updated/defaulted
    return update_dict

# Dummy node logic - the routing happens in the conditional edge
def action_router_node_logic(state: SimpleTutorState) -> dict:
     """Dummy node, routing is handled by conditional edges."""
     return {}

# Actual routing logic function (used by conditional edges)
def route_action(state: SimpleTutorState) -> str:
    """Determines the next node based on the requested action and problem complexity."""
    action = state.get("action")
    complexity = state.get("complexity")
    is_correct = state.get("is_correct") # Get correctness flag

    if action == "assess":
        return "assess_answer"
    elif action == "hint":
        # --- Add Check: Prevent hints if already correct ---
        if is_correct is True:
            print("DEBUG: Problem already solved correctly, ending hint request.")
            # TODO: Could route to a node that gives a specific message
            # For now, just end the turn.
            return END
        else:
            return "generate_hint"
    elif action == "solve":
        if complexity is None:
             return "classify_complexity"
        elif complexity == "simple":
             return "generate_cot"
        else:
             return "plan_cot"
    else:
        return END

def classify_complexity(state: SimpleTutorState) -> dict:
    """Uses LLM to classify problem complexity."""
    problem = state.get("problem")
    if not problem:
        return {"error": "Cannot classify complexity without a problem."}

    prompt = CLASSIFY_COMPLEXITY_SYSTEM.format(problem=problem)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        complexity = response.content.strip().lower()
        if complexity in ["simple", "medium", "complex"]:
            return {"complexity": complexity}
        else:
            return {"complexity": "medium", "error": "Complexity classification failed, defaulted to medium."}
    except Exception as e:
        return {"complexity": "medium", "error": f"Error classifying complexity: {e}"}


def plan_cot(state: SimpleTutorState) -> dict:
    """Generates a high-level plan for medium/complex problems."""
    problem = state.get("problem")
    if not problem:
        return {"error": "Cannot generate plan without a problem."}

    messages = list(state.get("chat_history", [])[-5:])
    prompt_messages = [SystemMessage(content=PLAN_COT_SYSTEM)] + messages # History includes problem

    plan_llm = llm.with_structured_output(PlanningSteps)
    try:
        structured_response = plan_llm.invoke(prompt_messages)
        if isinstance(structured_response, PlanningSteps):
             plan_message = AIMessage(content=f"Okay, here's the plan to solve '{problem}':\n" + "\n".join(f"- {p}" for p in structured_response.plan))
             return {
                 "plan": structured_response.plan,
                 "chat_history": messages + [plan_message]
                }
        else:
             return {"error": f"Planning failed. Invalid response: {repr(structured_response)}"}
    except OutputParserException as e:
        return {"error": f"Error parsing plan: {e}"}
    except Exception as e:
        return {"error": f"Error generating plan: {e}"}

def generate_cot(state: SimpleTutorState) -> dict:
    """Generates CoT based on complexity, optionally using a plan. Handles tool calls."""
    problem = state.get("problem")
    if not problem:
         return {"error": "Cannot generate solution without a problem."}

    complexity = state.get("complexity", "medium")
    plan = state.get("plan")
    messages = list(state.get("chat_history", []))

    if complexity == 'simple':
        system_prompt = GENERATE_COT_SIMPLE_SYSTEM
        prompt_messages = [SystemMessage(content=system_prompt)] + messages
    else:
        if not plan:
            return {"error": "Cannot generate CoT without a plan for medium/complex problems."}
        system_prompt = GENERATE_COT_PLAN_SYSTEM
        plan_already_in_history = any(isinstance(msg, AIMessage) and msg.content.startswith("Okay, here's the plan") for msg in messages)
        if plan_already_in_history:
             prompt_messages = [SystemMessage(content=system_prompt)] + messages
        else:
             plan_str = "\n".join(f"- {p}" for p in plan)
             prompt_messages = [
                 SystemMessage(content=system_prompt),
                 AIMessage(content=f"Here is the plan to follow:\n{plan_str}"),
             ] + messages

    try:
        response = llm_with_tools.invoke(prompt_messages)
        current_call_messages = prompt_messages + [response]

        while response.tool_calls:
            tool_messages = []
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id")
                print(f"\n  [generate_cot] Calling Tool: {tool_name}") # Log tool name
                print(f"  [generate_cot]   Arguments: {tool_args}") # Log args

                selected_tool = None
                for t in tools:
                    if t.name == tool_name:
                        selected_tool = t
                        break

                if selected_tool:
                    try:
                        tool_output = selected_tool.invoke(tool_args)
                        print(f"  [generate_cot]   Output: {tool_output[:100]}{'...' if len(tool_output)>100 else ''}") # Log truncated output
                        tool_messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_id))
                    except Exception as tool_err:
                        print(f"  [generate_cot]   Tool Error: {tool_err}") # Log tool error
                        tool_messages.append(ToolMessage(content=f"Error executing tool: {str(tool_err)}", tool_call_id=tool_id))
                else:
                    print(f"  [generate_cot]   Error: Tool '{tool_name}' not found.") # Log tool not found
                    tool_messages.append(ToolMessage(content=f"Error: Tool '{tool_name}' not found.", tool_call_id=tool_id))

            current_call_messages.extend(tool_messages)
            response = llm_with_tools.invoke(current_call_messages)
            current_call_messages.append(response)

        if hasattr(response, 'content') and response.content:
            structured_llm = llm.with_structured_output(TutorSolution)
            try:
                 final_structured_response = structured_llm.invoke([AIMessage(content=response.content)])
                 if isinstance(final_structured_response, TutorSolution):
                     final_ai_message = AIMessage(content=response.content)
                     # Return only the fields relevant to this node's update
                     return {
                         "solution_steps": final_structured_response.steps,
                         "correct_answer": final_structured_response.final_answer,
                         "error": None,
                         "chat_history": messages + [final_ai_message]
                     }
                 else:
                     return {"error": f"Failed to get structured output. Returned object: {repr(final_structured_response)}"}

            except (OutputParserException, Exception) as struct_err:
                return {
                    "error": f"Structured output failed: {struct_err}. Raw content: {response.content}",
                    "solution_steps": ["Could not parse steps."],
                    "correct_answer": "Could not parse answer.",
                    "chat_history": messages + [AIMessage(content=response.content)]
                    }
        else:
             return {"error": f"Final CoT LLM response missing content. Final Response Object: {repr(response)}"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Error generating CoT with tool handling: {str(e)}"}

def assess_answer(state: SimpleTutorState) -> dict:
    """Compares user answer, determines correctness, and generates feedback."""
    user_answer = state.get("user_answer")
    correct_answer = state.get("correct_answer")
    problem = state.get("problem")
    messages = list(state.get("chat_history", []))

    if user_answer is None:
        return {"feedback": "You didn't provide an answer to assess.", "is_correct": None, "error": "No user answer provided for assessment."}
    if correct_answer is None:
        return {"feedback": "Sorry, I couldn't determine the correct answer to compare yours against.", "is_correct": None, "error": "Correct answer missing for assessment."}
    if not problem:
        return {"feedback": "Sorry, I seem to have lost the original problem context.", "is_correct": None, "error": "Problem context missing for assessment."}

    # Step 1: Check correctness using LLM
    is_correct = None
    try:
        check_prompt = CHECK_CORRECTNESS_SYSTEM.format(problem=problem, correct_answer=correct_answer, user_answer=user_answer)
        check_response = llm.invoke([SystemMessage(content=check_prompt)])
        check_result = check_response.content.strip().upper()
        if check_result == "CORRECT":
            is_correct = True
        elif check_result == "INCORRECT":
            is_correct = False
        else:
            # LLM failed to follow instructions, maybe default to incorrect?
            print(f"Warning: Correctness check returned unexpected result: {check_result}")
            is_correct = False # Default to incorrect if check is ambiguous

    except Exception as check_err:
        print(f"Error during correctness check: {check_err}")
        # Cannot determine correctness, proceed without setting the flag
        pass # is_correct remains None

    # Step 2: Generate feedback using LLM
    try:
        feedback_prompt = ASSESS_ANSWER_SYSTEM.format(problem=problem, correct_answer=correct_answer, user_answer=user_answer)
        feedback_response = llm.invoke([SystemMessage(content=feedback_prompt)])
        feedback = feedback_response.content

        # Add assessment interaction to history using add_messages semantics
        assessment_history_update = [AIMessage(content=feedback)]
        # Return feedback AND the determined correctness flag
        return {"feedback": feedback, "is_correct": is_correct, "chat_history": assessment_history_update}

    except Exception as feedback_err:
        # Still return correctness if determined, even if feedback fails
        return {
            "feedback": "Sorry, I encountered an error while assessing your answer.",
            "is_correct": is_correct, # Return correctness if available
            "error": f"Feedback generation failed: {feedback_err}"
            }

def generate_hint(state: SimpleTutorState) -> dict:
    """Generates a contextual hint using the LLM based on the next solution step."""
    solution_steps = state.get("solution_steps")
    last_hint_level = state.get("hint_level", -1)
    messages = list(state.get("chat_history", []))
    problem = state.get("problem")

    if not problem:
        return {"hint": "I seem to have lost the problem context.", "error": "Problem context missing for hint generation."}
    if not solution_steps:
        hint_msg = AIMessage(content="I need to generate the solution steps first before I can give a hint.")
        return {
            "hint": "I need to generate the solution steps first.",
            "error": "Hint requested but no solution steps available.",
            "chat_history": [HumanMessage(content="Can I have a hint?"), hint_msg]
            }

    next_hint_level = last_hint_level + 1

    if next_hint_level < len(solution_steps):
        # Prepare context for the LLM
        solution_steps_formatted = "\n".join(f"  Step {i+1}: {step}" for i, step in enumerate(solution_steps))
        # Get last ~3 messages for recent history
        recent_history_raw = messages[-3:]
        recent_history = "\n".join([f"  {type(m).__name__}: {m.content}" for m in recent_history_raw])

        # Create the prompt
        prompt = GENERATE_HINT_SYSTEM.format(
            problem=problem,
            solution_steps_formatted=solution_steps_formatted,
            last_hint_level=last_hint_level,
            next_hint_level=next_hint_level + 1, # Use 1-based index for display in prompt
            recent_history=recent_history if recent_history else "(No recent history)"
        )

        try:
            # --- Use stream instead of invoke for token-by-token output --- #
            # response = llm.invoke([SystemMessage(content=prompt)]) # Old invoke
            hint_text_accumulated = ""
            print(">> ", end="") # Print prompt prefix before streaming
            for chunk in llm.stream([SystemMessage(content=prompt)]):
                print(chunk.content, end="", flush=True)
                hint_text_accumulated += chunk.content
            print() # Add a newline after streaming is complete

            hint_text = hint_text_accumulated.strip() # Use the full accumulated text

            # Add hint request and response to history via add_messages
            hint_history_update = [
                HumanMessage(content="Can I have a hint?"),
                AIMessage(content=hint_text) # Use accumulated text for history
            ]
            return {
                "hint": hint_text, # Store the generated hint
                "hint_level": next_hint_level,
                "chat_history": hint_history_update
                }
        except Exception as e:
            hint_text = f"(Sorry, I encountered an error generating the hint: {e})"
            hint_history_update = [
                 HumanMessage(content="Can I have a hint?"),
                 AIMessage(content=hint_text)
            ]
            return {
                "hint": hint_text,
                "error": f"Hint generation failed: {e}",
                # Update history even if error occurred
                "chat_history": hint_history_update
            }

    else:
        # No more steps left
        hint_text = "You've reached the end of the solution steps! Try solving it now."
        hint_history_update = [
            HumanMessage(content="Can I have a hint?"),
            AIMessage(content=hint_text)
            ]
        return {
            "hint": hint_text,
            "hint_level": next_hint_level, # Keep level pointing past the end
            "chat_history": hint_history_update
            }

# --- Verification Node (Corrected Tool Handling + Structured Output) ---
def verify_solution(state: SimpleTutorState) -> dict:
    """Uses an LLM judge to verify the generated solution, handling potential tool calls before parsing structured output."""
    print("--- Node: verify_solution (Structured) ---")
    problem = state.get("problem")
    solution_steps = state.get("solution_steps")
    correct_answer = state.get("correct_answer")
    messages = list(state.get("chat_history", []))

    if not problem or not solution_steps or correct_answer is None:
        return {
            "is_solution_valid": False,
            "verification_reason": "Missing problem, steps, or answer for verification.",
            "error": "Verification prerequisite missing.",
            "retry_count": state.get("retry_count", 0) + 1 # Increment retry on prerequisite error
        }

    solution_steps_formatted = "\n".join(f"  Step {i+1}: {step}" for i, step in enumerate(solution_steps))
    prompt = VERIFY_SOLUTION_SYSTEM.format(
        problem=problem,
        solution_steps_formatted=solution_steps_formatted,
        correct_answer=correct_answer
    )
    prompt_messages = [SystemMessage(content=prompt)]

    try:
        # --- Step 1: Invoke LLM with tools, handle calls --- #
        response = llm_with_tools.invoke(prompt_messages)
        current_call_messages = prompt_messages + [response]

        while response.tool_calls:
            print("  [verify_solution] Verifier LLM requested tool calls. Executing...")
            tool_messages = []
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id")
                print(f"    [verify_solution] Calling Tool: {tool_name}")
                print(f"    [verify_solution]   Arguments: {tool_args}")
                selected_tool = None
                for t in tools: 
                    if t.name == tool_name: selected_tool = t; break
                if selected_tool:
                    try:
                        tool_output = selected_tool.invoke(tool_args)
                        print(f"    [verify_solution]   Output: {tool_output[:100]}{'...' if len(tool_output)>100 else ''}")
                        tool_messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_id))
                    except Exception as tool_err:
                        print(f"    [verify_solution]   Tool Error: {tool_err}")
                        tool_messages.append(ToolMessage(content=f"Error executing tool: {str(tool_err)}", tool_call_id=tool_id))
                else:
                    print(f"    [verify_solution]   Error: Tool '{tool_name}' not found.")
                    tool_messages.append(ToolMessage(content=f"Error: Tool '{tool_name}' not found.", tool_call_id=tool_id))
            
            current_call_messages.extend(tool_messages)
            response = llm_with_tools.invoke(current_call_messages)
            current_call_messages.append(response)
        
        # --- Step 2: Parse the FINAL response using structured output --- # 
        final_response_message = response # This should be the last AIMessage
        if not isinstance(final_response_message, AIMessage):
             # Should not happen if invoke returns correctly
             raise TypeError(f"Expected AIMessage after tool handling, got {type(final_response_message)}")

        if not final_response_message.content:
            raise ValueError("Verifier LLM returned empty content after tool handling.")

        # Now use the structured output LLM on the final content
        structured_llm = llm.with_structured_output(VerificationResult) # Use base LLM
        verification_result: VerificationResult = structured_llm.invoke([final_response_message]) # Pass final message

        # --- Step 3: Process parsed result --- #
        is_valid = verification_result.is_valid
        reason = verification_result.reason

        if is_valid and reason is not None: reason = None
        elif not is_valid and reason is None: reason = "Verifier marked invalid but provided no specific reason."

        print(f"--- Verification Result: is_valid={is_valid}, Reason='{reason}' --- ")
        current_retry_count = state.get("retry_count", 0)
        update_dict = {"is_solution_valid": is_valid, "verification_reason": reason}
        if not is_valid:
            next_retry_count = current_retry_count + 1
            update_dict["retry_count"] = next_retry_count
            max_retries = 2 
            if next_retry_count > max_retries:
                print(f"ERROR: Max verification retries ({max_retries}) exceeded.")
                update_dict["error"] = f"Solution verification failed after {max_retries} retries."

        return update_dict

    except Exception as e:
        print(f"Error during structured solution verification: {e}")
        import traceback
        traceback.print_exc()
        current_retry_count = state.get("retry_count", 0)
        next_retry_count = current_retry_count + 1
        error_message = f"Verification failed: {e}"
        max_retries = 2
        if next_retry_count > max_retries:
            print(f"ERROR: Max verification retries ({max_retries}) exceeded after exception.")
            error_message = f"Solution verification failed after {max_retries} retries (exception occurred)."

        return {
            "is_solution_valid": False, 
            "verification_reason": f"Exception during structured verification: {e}",
            "error": error_message, 
            "retry_count": next_retry_count 
        }