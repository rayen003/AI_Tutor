"""
Interactive command-line demo for the simplified MathTutor implementation.
"""
import os
from dotenv import load_dotenv
from app.math_tutor_simple import process_problem

# Load environment variables
load_dotenv()

def check_api_key():
    """Check if the API key is configured"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("\n" + "!" * 80)
        print("⚠️  OPENAI_API_KEY not configured properly!")
        print("Please edit the .env file and add your OpenAI API key.")
        print("!" * 80 + "\n")
        return False
    return True

def run_demo():
    """Run the interactive demo"""
    if not check_api_key():
        return
        
    print("\n" + "=" * 80)
    print("🧮 MathTutor - AI-Powered Learning Platform for BBA Students")
    print("=" * 80)
    print("This demo showcases the simplified MathTutor implementation.")
    print("You can ask math problems or general business questions.")
    print("Enter 'exit' to quit the demo.")
    print("=" * 80 + "\n")
    
    while True:
        # Get user input
        problem = input("\nEnter your math problem or business question: ")
        if problem.lower() in ['exit', 'quit', 'q']:
            print("\nThank you for using MathTutor! Goodbye.")
            break
            
        # Process the question
        print("\nProcessing your question...\n")
        
        try:
            # First, get a solution without showing it
            result = process_problem(problem)
            
            # Show whether it's a math problem or general question
            problem_type = "Math problem" if result["is_math_problem"] else "General question"
            print(f"Question type: {problem_type}")
            
            if result["is_math_problem"]:
                # For math problems, offer options
                while True:
                    print("\nOptions:")
                    print("1. Show hint (level 1)")
                    print("2. Show hint (level 2)")
                    print("3. Show hint (level 3)")
                    print("4. Show complete solution")
                    print("5. Check my answer")
                    print("6. New question")
                    
                    choice = input("\nWhat would you like to do? (1-6): ")
                    
                    if choice == '1':
                        # Show level 1 hint
                        hint_result = process_problem(problem, hint_level=1)
                        print("\n" + "-" * 40)
                        print(f"Hint (Level 1): {hint_result['hint']}")
                        print("-" * 40)
                    elif choice == '2':
                        # Show level 2 hint
                        hint_result = process_problem(problem, hint_level=2)
                        print("\n" + "-" * 40)
                        print(f"Hint (Level 2): {hint_result['hint']}")
                        print("-" * 40)
                    elif choice == '3':
                        # Show level 3 hint
                        hint_result = process_problem(problem, hint_level=3)
                        print("\n" + "-" * 40)
                        print(f"Hint (Level 3): {hint_result['hint']}")
                        print("-" * 40)
                    elif choice == '4':
                        # Show complete solution
                        print("\n" + "=" * 80)
                        print("COMPLETE SOLUTION:")
                        print("=" * 80)
                        
                        if result["solution_steps"]:
                            for i, step in enumerate(result["solution_steps"]):
                                print(f"\nStep {i+1}: {step}")
                            
                            if result["final_answer"]:
                                print("\n" + "-" * 40)
                                print(f"Final Answer: {result['final_answer']}")
                                print("-" * 40)
                        else:
                            print("No solution steps available.")
                        
                        break  # Back to main loop after showing solution
                    elif choice == '5':
                        # Check user's answer
                        user_answer = input("\nEnter your answer: ")
                        check_result = process_problem(
                            problem, 
                            user_answer=user_answer,
                        )
                        
                        print("\n" + "-" * 40)
                        print(f"Feedback: {check_result['feedback']}")
                        print("-" * 40)
                    elif choice == '6':
                        # New question
                        break  # Back to main loop
                    else:
                        print("Invalid choice. Please enter a number from 1-6.")
            else:
                # For general questions, just show the response
                print("\n" + "=" * 80)
                print("ANSWER:")
                print("=" * 80)
                print(result["response"])
                print("=" * 80)
        
        except Exception as e:
            print(f"Error processing your question: {str(e)}")

if __name__ == "__main__":
    run_demo() 