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

# Load environment variables
load_dotenv()

# --- State Definition ---
Messages = List[BaseMessage]

class SimpleTutorState(TypedDict):
    """
    Represents the state of our simple tutor graph.
    """
    # Inputs
    problem: str
    user_answer: Optional[str]
    chat_history: Messages

    # Graph Control & Intermediate
    complexity: Optional[str] = None           # Added: 'simple', 'medium', 'complex'
    plan: Optional[List[str]] = None           # Added: High-level steps from planning node

    # Generated Content
    solution_steps: Optional[List[str]] = None # CoT steps (generated directly or from plan)
    correct_answer: Optional[str] = None
    hint: Optional[str] = None
    error: Optional[str] = None

    # Assessment Results
    verification_result: Optional[Dict[str, str]] = None
    is_correct: Optional[bool] = None
    feedback: Optional[str] = None

# --- Pydantic Model for Structured Output ---
class TutorSolution(BaseModel):
    """Structured format for the step-by-step solution and final answer."""
    steps: List[str] = Field(description="List of strings, each representing a step in the solution process.")
    final_answer: str = Field(description="The final numerical or symbolic answer.")

# --- Initialize LLM ---
# Use a cost-effective model for now
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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
    # Remove LaTeX delimiters \(\), \[\]
    cleaned = cleaned.replace('\\(', '').replace('\\)', '').replace('\\[', '').replace('\\\]', '')
    # Remove any remaining leading/trailing whitespace
    cleaned = cleaned.strip()
    return cleaned

# --- Node Functions (Placeholders) ---
def generate_cot(state: SimpleTutorState) -> dict:
    """Generates CoT based on complexity, optionally using a plan."""
    print("--- Node: generate_cot ---")
    problem = state["problem"]
    complexity = state.get("complexity", "medium") # Default if missing
    plan = state.get("plan")
    # Use .get() with a default empty list to avoid KeyError
    current_messages = list(state.get("chat_history", [])[-5:])

    # Define prompts based on complexity
    if complexity == 'simple':
        print("--- Using SIMPLE prompt for CoT generation ---")
        system_prompt = """You are a helpful math tutor. Solve the user's problem step-by-step (Chain-of-Thought).
Keep the steps concise for this simple problem.
Use the available tools (like the calculator) if necessary.
Think step by step before outputting the final structured response."""
        prompt_messages = [SystemMessage(content=system_prompt)] + current_messages + [HumanMessage(content=f"Solve this simple problem: {problem}")]
    else: # medium or complex
        print("--- Using PLAN-BASED prompt for CoT generation ---")
        if not plan:
            print("Error: Plan-based generation requested but no plan found in state.")
            return {"error": "Cannot generate CoT without a plan for medium/complex problems."}
        plan_str = "\n".join(f"- {p}" for p in plan)
        system_prompt = """You are a helpful math tutor. Solve the user's problem step-by-step (Chain-of-Thought), following the provided plan.
Use the available tools (like the calculator) if necessary for calculations within your steps.
Think step by step, elaborating on the plan, before outputting the final structured response."""
        prompt_messages = [
            SystemMessage(content=system_prompt),
            AIMessage(content=f"Here is the plan to follow:\n{plan_str}"), # Inject plan as AI message
            # HumanMessage(content=f"Now, solve the problem following that plan: {problem}") # Reiterate problem?
        ] + current_messages # Include history for context

    max_iterations = 5
    llm_call_count = 0

    try:
        for i in range(max_iterations):
            llm_call_count += 1
            print(f"--- LLM Invocation (Iteration {i+1}) ---")
            invokable_llm = llm_with_tools
            ai_response = invokable_llm.invoke(prompt_messages)
            print(f"--- AI Response Object (Iteration {i+1}): {repr(ai_response)} ---")

            if ai_response.tool_calls:
                print("--- Detected Tool Calls --- ")
                tool_messages = []
                prompt_messages.append(ai_response) # Add AI request to history
                for tool_call in ai_response.tool_calls:
                    tool_name = tool_call.get('name')
                    tool_args = tool_call.get('args')
                    tool_id = tool_call.get('id')
                    print(f"Tool Call: {tool_name}, Args: {tool_args}, ID: {tool_id}")
                    tool_result = "Error: Tool not found"
                    if tool_name == 'calculator':
                        tool_result = calculator.invoke(tool_args)
                    else:
                        print(f"Warning: Unknown tool requested: {tool_name}")
                    tool_messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_id))
                
                prompt_messages.extend(tool_messages) # Add tool results to history
                if i == max_iterations - 1:
                     print("Error: Max iterations reached during tool calls.")
                     return {"error": "Max tool call iterations reached."}
                continue # Continue loop for next LLM call
            
            elif hasattr(ai_response, 'content') and ai_response.content:
                print("--- Final Reasoning Received, formatting with Structured Output --- ")
                structured_llm = llm.with_structured_output(TutorSolution)
                final_structured_response = structured_llm.invoke(prompt_messages + [ai_response])
                print(f"--- Structured Output Object: {repr(final_structured_response)} ---")
                
                if isinstance(final_structured_response, TutorSolution):
                    return {
                        "solution_steps": final_structured_response.steps,
                        "correct_answer": final_structured_response.final_answer,
                        "error": None
                    }
                else:
                     print("Error: Structured output call did not return TutorSolution object.")
                     return {"error": f"Failed to get structured output. Object: {repr(final_structured_response)}"}
            else:
                print("Error: AI Response has no content or tool calls.")
                return {"error": f"AI Response missing content/tool_calls. Object: {repr(ai_response)}"}
        
        print("Error: Loop completed unexpectedly.")
        return {"error": "Unexpected end of generation loop."}

    except Exception as e:
        print(f"Error in generate_cot: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Error generating CoT: {str(e)}"}

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
    # TODO: Implement LLM call for hint generation
    problem = state["problem"]
    steps = state.get("solution_steps")
    history = state["chat_history"]
    hint_text = f"Hint for '{problem}': Consider the overall goal."
    if steps:
        hint_text = f"Hint for '{problem}': The first step is '{steps[0]}'. What comes next?"

    current_history = list(history) if history else []
    new_history = current_history + [AIMessage(content=hint_text)]
    return {"hint": hint_text, "chat_history": new_history}

def generate_solution_response(state: SimpleTutorState) -> dict:
    print("--- Node: generate_solution_response ---")
    correct_ans = state.get("correct_answer", "No solution generated.")
    steps = state.get("solution_steps")
    response = f"The correct answer is: {correct_ans}\n"
    if steps:
        response += "Steps:\n" + "\n".join(f"- {s}" for s in steps)

    current_history = list(state["chat_history"]) if state.get("chat_history") else []
    new_history = current_history + [AIMessage(content=response)]
    return {"feedback": response, "chat_history": new_history} # Use feedback field for final output

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

# --- Graph Definition ---
builder = StateGraph(SimpleTutorState)

# Add nodes
builder.add_node("assess_complexity", assess_complexity)
builder.add_node("generate_plan", generate_plan)
builder.add_node("generate_cot", generate_cot)
builder.add_node("assess_answer", assess_answer)
# Keep hint/response nodes for potential future use, but disconnected for now
builder.add_node("generate_hint", generate_hint) 
builder.add_node("generate_solution_response", generate_solution_response) 

# --- Define Edges and Routing ---

# Start by assessing complexity
builder.set_entry_point("assess_complexity")

# Routing based on complexity
def route_after_complexity(state: SimpleTutorState):
    complexity = state.get("complexity", "medium") # Default if somehow missing
    print(f"--- Routing based on complexity: {complexity} ---")
    if complexity == 'simple':
        return "generate_cot" # Go directly to CoT generation
    else:
        return "generate_plan" # Go to planning step

builder.add_conditional_edges(
    "assess_complexity",
    route_after_complexity,
    {
        "generate_plan": "generate_plan",
        "generate_cot": "generate_cot"
    }
)

# After planning (if done), generate the detailed CoT
builder.add_edge("generate_plan", "generate_cot")

# After generating CoT (either path), assess the user's answer
builder.add_edge("generate_cot", "assess_answer")

# Conditional logic after assessment (remains the same for now)
def decide_after_assessment(state: SimpleTutorState):
    if state.get("is_correct") is None:
        print("--- Branch: Assessment could not be performed (e.g., no user answer), ending. ---")
        return "__end__"
    elif state.get("is_correct") is True:
        print("--- Branch: Correct answer, ending. ---")
        return "__end__"
    else:
        print("--- Branch: Incorrect answer, ending (will interrupt here later). ---")
        return "__end__"

builder.add_conditional_edges(
    "assess_answer",
    decide_after_assessment,
    {"__end__": END} # Map return value to END
)

# Compile the graph
app = builder.compile()
print("Simple Tutor Graph Compiled with Complexity Routing.")

# Example of how to run (will be interactive later)
if __name__ == "__main__":
    import asyncio
    import pprint

    async def run_interactive_example():
        # problem = "x + 5 = 10"
        problem = "Solve for y: 3*y - 5 = 7" # Slightly more complex
        print(f"Problem: {problem}")
        user_input = input(f"Enter your answer (e.g., y = 4): ")

        initial_state = {
            "problem": problem,
            "user_answer": user_input.strip() if user_input else None,
            "chat_history": [HumanMessage(content=f"Solve: {problem}")],
            # Ensure all keys exist, even if None initially
            "solution_steps": None,
            "correct_answer": None,
            "verification_result": None,
            "is_correct": None,
            "feedback": None,
            "hint": None,
            "error": None,
        }

        print("\n--- Running Graph with User Input ---")
        final_state = None
        try:
            # Use ainvoke to get the final state directly
            final_state = await app.ainvoke(initial_state, {"recursion_limit": 10}) # Add recursion limit
            print("\n--- Graph Execution Complete ---")
        except Exception as e:
            print("\n--- Graph Execution Error ---")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        print("\n--- Final State ---")
        if final_state:
            pprint.pprint(final_state)
        else:
            print("Execution did not complete successfully or final state not captured.")

    asyncio.run(run_interactive_example())
