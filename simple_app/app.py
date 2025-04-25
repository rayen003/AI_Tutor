import os
from typing import TypedDict, List, Optional, Dict
from dotenv import load_dotenv
import sympy # Import sympy for calculator tool
import re # For parsing steps

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field # Import Pydantic
from langchain.tools import tool # Import tool decorator
from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage
from langgraph.checkpoint.memory import MemorySaver # Import MemorySaver
import warnings
import uuid # Import uuid for thread IDs

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
load_dotenv()

# --- State Definition ---
Messages = List[BaseMessage]

class SimpleTutorState(TypedDict):
    """
    Represents the state of our simple tutor graph for turn-based execution.
    """
    # Inputs for a given turn
    problem: str
    user_answer: Optional[str] = None # Can be updated by the user/simulation
    action: Optional[str] = None      # Added: Goal for this turn (e.g., 'solve', 'assess', 'hint')

    # History / Context
    chat_history: Messages = []
    request_details: Optional[str] = None

    # Graph Control & Intermediate
    complexity: Optional[str] = None
    plan: Optional[List[str]] = None

    # Generated Content - Persists across turns
    solution_steps: Optional[List[str]] = None # Added back for CoT/Hints
    correct_answer: Optional[str] = None
    hint: Optional[str] = None # Holds the hint *generated* in a turn
    error: Optional[str] = None # Holds error *from* a turn

    # Assessment Results - Persists across turns
    verification_result: Optional[Dict[str, str]] = None
    is_correct: Optional[bool] = None
    feedback: Optional[str] = None # Holds feedback *generated* in a turn

# --- Pydantic Model for Structured Output ---
class TutorSolution(BaseModel):
    """Structured format for the step-by-step solution and final answer."""
    steps: List[str] = Field(description="List of strings, each representing a step in the solution process.")
    final_answer: str = Field(description="The final numerical or symbolic answer.")

# --- Initialize LLM ---
# Use a cost-effective model for now
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0, 
    request_timeout=60 # Add a 60-second timeout
)

# --- Tool Definition (Placeholder for now) ---
# We'll define a tool for solving/checking math problems later

# --- Tool Definition ---
@tool
def calculator(expression: str) -> str:
    """Evaluates or solves a mathematical expression/equation using sympy.
    Use for arithmetic, algebra simplification, or solving simple equations.
    Examples:
    calculator('2*x + 5*x - 3') -> '7*x - 3'
    calculator('3*y - 12 = 0') -> 'y = 4' (or just '4')
    calculator('12 / 3') -> '4'
    Args: expression (str): The mathematical expression or equation to evaluate/solve.
    Returns: str: The result or an error message.
    """
    print(f"--- Tool: calculator called with '{expression}' ---")
    try:
        # Check if it looks like an equation
        if '=' in expression:
            try:
                lhs, rhs = map(sympy.sympify, expression.split('=', 1))
                # Assume solving for a single variable (e.g., x, y, etc.)
                variables = lhs.free_symbols.union(rhs.free_symbols)
                if len(variables) == 1:
                    var = variables.pop()
                    solution = sympy.solve(sympy.Eq(lhs, rhs), var)
                    if solution:
                        # Return just the value for simplicity, maybe add var later if needed
                        result_str = str(solution[0])
                        print(f"--- Tool: calculator solved equation result = {result_str} ---")
                        return result_str
                    else:
                         return f"Could not solve equation: {expression}"
                else:
                    return f"Equation has multiple variables or no variables: {expression}"
            except Exception as eq_err:
                 return f"Error solving equation '{expression}': {str(eq_err)}"
        else:
            # Treat as an expression to simplify/evaluate
            result = sympy.sympify(expression, evaluate=True)
            simplified_result = sympy.simplify(result)
            result_str = str(simplified_result)
            print(f"--- Tool: calculator evaluated expression result = {result_str} ---")
            return result_str
    except (sympy.SympifyError, TypeError, Exception) as e:
        error_msg = f"Error processing '{expression}': {str(e)}"
        print(f"--- Tool: calculator error: {error_msg} ---")
        return error_msg # Return error message

# List of tools for the agent
tools = [calculator]
# Bind tools to LLM for agentic behavior
llm_with_tools = llm.bind_tools(tools)

# --- Helper Function for Cleaning ---
def clean_math_string(input_str: str) -> str:
    """Removes common prefixes, LaTeX math delimiters, and extra spaces."""
    if not isinstance(input_str, str):
        return ""
    # Remove prefixes like "x = ", "y = ", etc.
    cleaned = re.sub(r'^[a-zA-Z]\s*=\s*', '', input_str.strip()).strip()
    # Remove LaTeX delimiters \(\), \[\] and standard brackets for safety
    cleaned = cleaned.replace('\\(', '').replace('\\)', '') # LaTeX inline
    cleaned = cleaned.replace('\\[', '').replace('\\]', '') # LaTeX display - Corrected escape
    cleaned = cleaned.replace('[', '').replace(']', '')   # Standard brackets
    cleaned = cleaned.replace('{', '').replace('}', '')   # Standard braces
    # Remove any remaining leading/trailing whitespace
    cleaned = cleaned.strip()
    return cleaned

# --- NEW CoT GENERATION NODE ---
def generate_cot(state: SimpleTutorState) -> dict:
    """Generates CoT based on complexity, optionally using a plan. Handles tool calls."""
    print("--- Node: generate_cot ---")
    problem = state["problem"]
    complexity = state.get("complexity", "medium") # Default if missing
    plan = state.get("plan")
    messages = list(state.get("chat_history", [])[-5:]) # Get recent history

    # Define prompts based on complexity
    if complexity == 'simple':
        print("--- Using SIMPLE prompt for CoT generation ---")
        system_prompt = """You are a helpful math tutor. Solve the user's problem step-by-step (Chain-of-Thought).
Keep the steps concise for this simple problem.
Use the available tools (like the calculator) if necessary *within* your steps for calculations.
Think step by step before outputting the final structured response in the required format."""
        prompt_messages = [SystemMessage(content=system_prompt)] + messages + [HumanMessage(content=f"Solve this simple problem: {problem}")]
    else: # medium or complex
        print("--- Using PLAN-BASED prompt for CoT generation ---")
        if not plan:
            print("Error: Plan-based generation requested but no plan found in state.")
            return {"error": "Cannot generate CoT without a plan for medium/complex problems."}
        plan_str = "\n".join(f"- {p}" for p in plan)
        system_prompt = """You are a helpful math tutor. Solve the user's problem step-by-step (Chain-of-Thought), following the provided plan.
Use the available tools (like the calculator) if necessary for calculations within your steps.
Think step by step, elaborating on the plan, before outputting the final structured response in the required format."""
        prompt_messages = [
            SystemMessage(content=system_prompt),
            AIMessage(content=f"Here is the plan to follow:\n{plan_str}"), # Inject plan
        ] + messages # Include history

    # --- LLM Call with Tool Handling ---
    try:
        print("DEBUG: Invoking LLM for CoT generation (with tool handling).")
        response = llm_with_tools.invoke(prompt_messages)
        print(f"--- Initial CoT LLM Response: {repr(response)} ---")
        
        messages.append(response) # Add initial AI response to messages

        # --- Handle Tool Calls if Present ---
        if response.tool_calls:
            print("--- Tool calls detected. Executing tools... ---")
            tool_messages = []
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id")
                
                print(f"  - Executing tool '{tool_name}' with args: {tool_args} (ID: {tool_id})")
                
                # Find the tool function (simple lookup for now)
                selected_tool = None
                for t in tools:
                    if t.name == tool_name:
                        selected_tool = t
                        break
                
                if selected_tool:
                    try:
                        # Pass arguments from tool_call['args']
                        tool_output = selected_tool.invoke(tool_args) 
                        print(f"  - Tool '{tool_name}' output: {tool_output}")
                        tool_messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_id))
                    except Exception as tool_err:
                        print(f"  - Error executing tool '{tool_name}': {tool_err}")
                        tool_messages.append(ToolMessage(content=f"Error executing tool: {str(tool_err)}", tool_call_id=tool_id))
                else:
                    print(f"  - Error: Tool '{tool_name}' not found.")
                    tool_messages.append(ToolMessage(content=f"Error: Tool '{tool_name}' not found.", tool_call_id=tool_id))

            # Add tool results to messages and call LLM again
            messages.extend(tool_messages)
            print("DEBUG: Re-invoking LLM after tool execution.")
            response = llm_with_tools.invoke(messages) # Call again with tool results
            print(f"--- Second CoT LLM Response (after tools): {repr(response)} ---")
        
        # --- Process Final Response for Structured Output ---
        if hasattr(response, 'content') and response.content:
            print("--- Attempting Structured Output on final CoT response ---")
            # Ensure we use the LLM *without* tools for structured output parsing
            structured_llm = llm.with_structured_output(TutorSolution) 
            # Invoke with the final response content (which should contain the structured info)
            # Wrap content in a HumanMessage or similar if needed, but try invoking directly first if it's a BaseMessage
            try:
                 # Check if response is already a BaseMessage, otherwise wrap content
                 if isinstance(response, BaseMessage):
                     invocation_input = [response] 
                 else: 
                     invocation_input = [AIMessage(content=response.content)] # Fallback

                 final_structured_response = structured_llm.invoke(invocation_input)
                 print(f"--- Structured Output Object Returned: {repr(final_structured_response)} ---")

                 if isinstance(final_structured_response, TutorSolution):
                     print("--- Structured Output Success! --- ")
                     return {
                         "solution_steps": final_structured_response.steps,
                         "correct_answer": final_structured_response.final_answer,
                         "error": None,
                         "chat_history": messages + [AIMessage(content=response.content)] # Add final response
                     }
                 else:
                     print("Error: Structured output call did not return TutorSolution.")
                     return {"error": f"Failed to get structured output. Returned object: {repr(final_structured_response)}"}

            except Exception as struct_err:
                print(f"Error during structured output parsing: {struct_err}")
                # Fallback: Try to extract from raw content if possible (less reliable)
                return {
                    "error": f"Structured output failed: {struct_err}. Raw content: {response.content}",
                    "solution_steps": ["Could not parse steps."], 
                    "correct_answer": "Could not parse answer."
                    }

        else:
             print("Error: Final LLM response after potential tool calls had no content.")
             return {"error": f"Final CoT LLM response missing content. Final Response Object: {repr(response)}"}

    except Exception as e:
        print(f"Error in generate_cot (tool handling): {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Error generating CoT with tool handling: {str(e)}"}

def assess_answer(state: SimpleTutorState) -> dict:
    print("--- Node: assess_answer ---")
    user_ans = state.get("user_answer")
    correct_ans = state.get("correct_answer")

    if not user_ans:
        return {"feedback": "No user answer provided to assess.", "is_correct": None}

    if not correct_ans:
        return {"feedback": "Cannot assess answer, the correct answer wasn't generated.", "is_correct": None}

    is_correct = False
    feedback = "That doesn't seem right. Try again or ask for a hint."
    try:
        # Clean both user answer and correct answer before comparison
        user_ans_cleaned = clean_math_string(user_ans)
        correct_ans_cleaned = clean_math_string(correct_ans)
        print(f"--- Comparing Cleaned Answers: User='{user_ans_cleaned}', Correct='{correct_ans_cleaned}' ---")

        # Use sympy to check for numerical or symbolic equality
        # Allow evaluate=False initially to prevent premature calculation errors
        if sympy.simplify(sympy.sympify(user_ans_cleaned, evaluate=False) - sympy.sympify(correct_ans_cleaned, evaluate=False)) == 0:
            is_correct = True
            feedback = "Correct!"
        else:
             # Try evaluating if possible, in case of things like 12/3 vs 4
             try:
                 user_eval = sympy.sympify(user_ans_cleaned, evaluate=True)
                 correct_eval = sympy.sympify(correct_ans_cleaned, evaluate=True)
                 if sympy.simplify(user_eval - correct_eval) == 0:
                     is_correct = True
                     feedback = "Correct!"
                 else:
                     print(f"Sympy comparison failed: Cleaned='{user_ans_cleaned}' vs '{correct_ans_cleaned}', Evaluated='{user_eval}' vs '{correct_eval}'")
             except Exception as eval_err:
                 print(f"Evaluation during comparison failed: {eval_err}")
                 print(f"Sympy comparison failed (non-eval): Cleaned='{user_ans_cleaned}' vs '{correct_ans_cleaned}'")

    except (sympy.SympifyError, TypeError, Exception) as e:
        print(f"Error during answer comparison ({type(e).__name__}): {e}. Falling back to cleaned string comparison.")
        # Fallback to simple string comparison of *cleaned* strings
        if user_ans_cleaned.lower() == correct_ans_cleaned.lower():
             is_correct = True
             feedback = "Correct!"

    return {"is_correct": is_correct, "feedback": feedback}

def generate_hint(state: SimpleTutorState) -> dict:
    print("--- Node: generate_hint ---")
    problem = state["problem"]
    # Use solution_steps if available
    steps = state.get("solution_steps")
    history = state.get("chat_history", [])

    hint_text = f"Hint for '{problem}': Think about the first step."
    if steps:
        # Example: Provide the next step as a hint (needs step tracking later)
        # For now, just use the first step if available
        hint_text = f"Hint for '{problem}': The first step involves '{steps[0]}'."

    current_history = list(history)
    new_history = current_history + [AIMessage(content=hint_text)]
    # Return *only* the hint and updated history for this turn's action
    # Clear other fields that might be left over from previous turns
    return {
        "hint": hint_text, 
        "chat_history": new_history, 
        "feedback": None, 
        "error": None,
        "is_correct": None # Clear assessment from previous turns
    }

def generate_solution_response(state: SimpleTutorState) -> dict:
    print("--- Node: generate_solution_response ---")
    correct_ans = state.get("correct_answer", "Could not determine correct answer.")
    steps = state.get("solution_steps")
    history = state.get("chat_history", [])

    response = f"The correct answer is: {correct_ans}\n"
    if steps:
        response += "Steps:\n" + "\n".join(f"- {s}" for s in steps)
    else:
        response += "(Could not generate detailed steps)"

    current_history = list(history)
    new_history = current_history + [AIMessage(content=response)]
     # Return the full solution in 'feedback', plus updated history
     # Clear other fields
    return {
        "feedback": response, 
        "chat_history": new_history, 
        "hint": None, 
        "error": None,
        "is_correct": None
    }

def assess_complexity(state: SimpleTutorState) -> dict:
    """Assess the complexity of the math problem."""
    print("--- Node: assess_complexity ---")
    problem = state["problem"]

    # Simple prompt for complexity assessment
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert math problem evaluator. Classify the complexity of solving the following math problem for a typical high school or early college student. Respond ONLY with one word: 'simple', 'medium', or 'complex'."),
        ("human", "Problem: {problem_text}")
    ])
    evaluator_chain = prompt | llm

    try:
        response = evaluator_chain.invoke({"problem_text": problem})
        complexity = response.content.strip().lower()
        # Basic validation
        if complexity not in ['simple', 'medium', 'complex']:
            print(f"Warning: Unexpected complexity response: {complexity}. Defaulting to medium.")
            complexity = 'medium'
        print(f"--- Assessed Complexity: {complexity} ---")
        return {"complexity": complexity}
    except Exception as e:
        print(f"Error assessing complexity: {e}")
        return {"error": f"Failed to assess complexity: {str(e)}", "complexity": "medium"} # Default on error

def generate_plan(state: SimpleTutorState) -> dict:
    """Generate a high-level plan for medium/complex problems."""
    print("--- Node: generate_plan ---")
    problem = state["problem"]
    # Use state complexity check if needed, though routing handles this
    # complexity = state.get("complexity", "medium") 

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strategic math planner. Outline the key steps (2-5 steps) needed to solve the following problem. Do NOT perform detailed calculations or give the final answer. Just provide the high-level strategy. Start each step with 'STEP X:'."),
        ("human", "Problem: {problem_text}")
    ])
    planner_chain = prompt | llm

    try:
        response = planner_chain.invoke({"problem_text": problem})
        response_content = response.content.strip()
        print(f"--- Plan LLM Response ---\n{response_content}\n------------------------")
        # Basic parsing for steps starting with STEP X:
        plan_steps = []
        step_pattern = re.compile(r"^STEP\s*\d+[:]?\s*(.*)", re.IGNORECASE)
        for line in response_content.split('\n'):
            match = step_pattern.match(line.strip())
            if match:
                plan_steps.append(match.group(1).strip())
        
        if not plan_steps:
            print("Warning: Failed to parse plan steps from LLM response.")
            # Use the whole response as a single plan step?
            plan_steps = [response_content] if response_content else ["Plan generation failed."]
            
        print(f"--- Generated Plan Steps: {plan_steps} ---")
        return {"plan": plan_steps}
    except Exception as e:
        print(f"Error generating plan: {e}")
        return {"error": f"Failed to generate plan: {str(e)}", "plan": ["Plan generation failed due to error."]}

def interpret_request(state: SimpleTutorState) -> dict:
    """Analyzes the user's input to understand the specific request details."""
    print("--- Node: interpret_request ---")
    user_input = state["problem"] # The raw input is in the 'problem' field
    # history = state["chat_history"] # History not strictly needed for basic interpretation

    system_message_content = "Analyze the user's request. Identify the core math problem and any specific instructions like 'explain steps', 'show work', or 'define term'. Output a brief description of these details."
    human_message_content = f"User request: {user_input}"
    
    # Construct the list of messages directly
    messages_to_invoke = [
        SystemMessage(content=system_message_content),
        HumanMessage(content=human_message_content)
    ]
    
    try:
        # Use the base LLM without tools for this simple interpretation
        response = llm.invoke(messages_to_invoke) # Pass the list of messages
        details = response.content.strip()
        print(f"--- Interpreted Request Details: {details} ---")
        # Store details, potentially clear previous errors if interpretation succeeds
        return {"request_details": details, "error": None} 
    except Exception as e:
        print(f"Error in interpret_request: {e}")
        # Proceed even if interpretation fails, but log an error state
        return {"request_details": "Interpretation failed.", "error": f"Interpretation error: {e}"}

# Main action router logic - This function is ONLY for conditional edges
def route_next_action(state: SimpleTutorState):
    action = state.get("action")
    user_answer_provided = state.get("user_answer") is not None
    print(f"--- Routing Condition: Action='{action}', UserAnswerProvided={user_answer_provided} ---")
    # Determine the next step based on the action
    if action == "assess" and user_answer_provided:
        return "generate_cot_then_assess"
    elif action == "hint":
        return "generate_cot_then_hint"
    elif action == "solve":
        return "generate_cot_then_solution"
    else:
        print(f"--- Routing Condition: No specific action ('{action}') or prerequisite missing, ending turn. ---")
        return "__end__"

# --- Dummy function for the router node --- 
def action_router_node_logic(state: SimpleTutorState) -> dict:
    """This node only acts as a branching point. Logic is in edges."""
    print("--- Node: action_router (Branching Point) ---")
    return {} # Nodes must return a dictionary

# --- Graph Definition ---
builder = StateGraph(SimpleTutorState)

# Add nodes
builder.add_node("interpret_request", interpret_request)
builder.add_node("assess_complexity", assess_complexity)
builder.add_node("generate_plan", generate_plan)
builder.add_node("generate_cot", generate_cot) # Shared node
builder.add_node("assess_answer", assess_answer)
builder.add_node("generate_hint", generate_hint)
builder.add_node("generate_solution_response", generate_solution_response)
# Intermediate nodes calling generate_cot
builder.add_node("generate_cot_then_assess", generate_cot)
builder.add_node("generate_cot_then_hint", generate_cot)
builder.add_node("generate_cot_then_solution", generate_cot)
# Add the router node with the dummy logic function
builder.add_node("action_router", action_router_node_logic) 

# --- Define Edges and Routing --- 
builder.set_entry_point("interpret_request")
builder.add_edge("interpret_request", "assess_complexity")

# Routing after complexity assessment
def route_after_complexity(state: SimpleTutorState):
    complexity = state.get("complexity", "medium")
    print(f"--- Routing after Complexity: '{complexity}' ---")
    # Point to plan or the actual router node
    return "generate_plan" if complexity != 'simple' else "action_router"

builder.add_conditional_edges(
    "assess_complexity",
    route_after_complexity,
    {
        "generate_plan": "generate_plan",
        "action_router": "action_router" # Route to the router node
    }
)

# After plan, go to the action router node
builder.add_edge("generate_plan", "action_router")

# Edges FROM the action router node
builder.add_conditional_edges(
    "action_router",     # Source node is the actual router node
    route_next_action,   # Condition function determines the path
    {
        # Map the return values to actual target nodes
        "generate_cot_then_assess": "generate_cot_then_assess",
        "generate_cot_then_hint": "generate_cot_then_hint",
        "generate_cot_then_solution": "generate_cot_then_solution",
        "__end__": END
    }
)

# Edges AFTER the generate_cot run for specific actions
builder.add_edge("generate_cot_then_assess", "assess_answer")
builder.add_edge("generate_cot_then_hint", "generate_hint")
builder.add_edge("generate_cot_then_solution", "generate_solution_response")

# Edges after final action nodes to END
builder.add_edge("assess_answer", END)
builder.add_edge("generate_hint", END)
builder.add_edge("generate_solution_response", END)

# Compile the graph
app = builder.compile()
print("Simple Tutor Graph Compiled (Turn-Based, No Interrupts).")

# --- Interactive Simulation Run ---
if __name__ == "__main__":
    import asyncio
    import pprint
    import uuid
    # Keep graph drawing imports if desired
    from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod

    # ... (Optional graph drawing) ...

    print("--- Math Tutor Simulation --- (LangSmith Tracing Disabled)")

    # Initial problem setup
    thread_id = str(uuid.uuid4()) # Start a new thread/session
    config = {"configurable": {"thread_id": thread_id}}
    problem = "Solve for y: 3*y - 5 = 7"
    print(f"Problem: {problem}")

    current_state = None # Keep track of the latest state

    # --- Turn 1: Ask Tutor to Solve ---
    print("\n--- Turn 1: Asking Tutor to Solve ---")
    initial_input = {
        "problem": problem,
        "action": "solve",
        "chat_history": [] # Start clean
    }
    current_state = app.invoke(initial_input, config=config)

    if current_state:
        if current_state.get("error"):
             print(f"\n>> Tutor Error: {current_state['error']}")
        if current_state.get("solution_steps"):
             print("\n>> Tutor Solution:")
             for i, step in enumerate(current_state["solution_steps"]):
                 print(f"  Step {i+1}: {step}")
             print(f"  Final Answer: {current_state.get('correct_answer', 'Not found')}")
        else:
             print("\n>> Tutor: I couldn't generate a solution.") # Should have error above

    else:
        print("\n>> Tutor: Failed to process the initial request.")


    # --- Subsequent Turns (Loop) ---
    turn_counter = 1
    while True:
        turn_counter += 1
        print(f"\n--- Turn {turn_counter} ---")
        user_input = input("Your answer (or type 'hint' or 'quit'): ").strip()

        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'hint':
            action = "hint"
            user_answer = None
            print("Asking for hint...")
        else:
            action = "assess"
            user_answer = user_input
            print(f"Submitting answer: {user_answer}")

        # --- FIX: Ensure problem is passed in subsequent turns ---
        if not current_state or "problem" not in current_state:
            print("Error: Cannot continue without problem context from previous state.")
            break
        turn_input = {
            "problem": current_state["problem"], # Carry over the problem
            "action": action,
            "user_answer": user_answer
            # The graph state (history, solution) is maintained via thread_id
        }
        current_state = app.invoke(turn_input, config=config) # Continue the thread

        # User-friendly output
        if current_state:
            if current_state.get("error"):
                 print(f"\n>> Tutor Error: {current_state['error']}")
                 # Optionally break or decide how to handle errors continuing
            if action == "assess" and current_state.get("feedback"):
                 print(f"\n>> Tutor Feedback: {current_state['feedback']}")
            elif action == "hint" and current_state.get("hint"):
                 print(f"\n>> {current_state['hint']}") # Hint text already formatted
            # Check if there was output other than error
            elif not current_state.get("error"):
                 print("\n>> Tutor: Processed your input.") # Generic fallback if no specific feedback/hint

        else:
            print("\n>> Tutor: Failed to process your request.")
            break # Exit loop if invoke fails entirely

    print("\n--- Simulation Complete ---")

    # --- Restore Environment Variables (Optional) ---
    # if original_trace_v2 is not None:
    #     os.environ["LANGCHAIN_TRACING_V2"] = original_trace_v2
    #     print("Restored LANGCHAIN_TRACING_V2.")
    # Keep API key restoration commented unless explicitly needed/removed
    # if original_api_key is not None:
    #      os.environ["LANGCHAIN_API_KEY"] = original_api_key
