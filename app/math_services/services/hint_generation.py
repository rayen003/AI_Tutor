from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def generate_hint(state):
    """Generate a helpful hint based on the problem and solution"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    problem = state.problem
    solution_steps = state.solution_steps
    subject_area = state.subject_area or "mathematics"
    
    hint_prompt = ChatPromptTemplate.from_template("""
    You are creating a series of hints for a {subject} problem. These hints should guide the student
    toward the solution without giving away the full answer.
    
    Problem: {problem}
    
    Solution approach:
    {solution}
    
    Create three increasingly detailed hints:
    1. A subtle hint that points toward the general approach
    2. A more specific hint that identifies a key concept or formula
    3. A detailed hint that almost reveals the solution method
    
    Format each hint as a JSON object with "level" and "content" fields.
    """)
    
    response = llm.invoke(
        hint_prompt.format(
            subject=subject_area,
            problem=problem,
            solution="\n".join([f"Step {i+1}: {step}" for i, step in enumerate(solution_steps)])
        )
    )
    
    # Parse hints - with simple error handling
    try:
        import json
        import re
        
        # Extract JSON-like structures from the text
        json_pattern = r'\{\s*"level"\s*:.*?"content"\s*:.*?\}'
        hint_matches = re.findall(json_pattern, response.content, re.DOTALL)
        
        hints = []
        for hint_text in hint_matches:
            try:
                hint = json.loads(hint_text)
                hints.append(hint)
            except:
                # If parsing fails, create a simple hint object
                hints.append({
                    "level": len(hints) + 1, 
                    "content": hint_text.replace('{', '').replace('}', '')
                })
        
        # If no hints were found or parsed correctly, create a fallback hint
        if not hints:
            hints = [
                {"level": 1, "content": "Think about the key concepts related to this problem."},
                {"level": 2, "content": "Consider what formulas might be applicable here."},
                {"level": 3, "content": "Try breaking the problem into smaller steps."}
            ]
            
        new_state = state.model_copy()
        new_state.hints = hints
        return new_state
    
    except Exception as e:
        # Fallback for any parsing issues
        print(f"Error parsing hints: {e}")
        new_state = state.model_copy()
        new_state.hints = [
            {"level": 1, "content": "Think about the key concepts related to this problem."},
            {"level": 2, "content": "Consider what formulas might be applicable here."},
            {"level": 3, "content": "Try breaking the problem into smaller steps."}
        ]
        return new_state








