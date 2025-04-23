import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

print("Testing basic OpenAI API call...")

# Check if API key is loaded (optional but good practice)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY not found in environment variables.")
    exit()
# else:
    # print("OPENAI_API_KEY found.") # Don't print the key itself

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Simple prompt
prompt = [HumanMessage(content="What is 2 + 2?")]

try:
    print("Invoking LLM...")
    response = llm.invoke(prompt)
    print("\n--- LLM Response ---")
    print(response)
    print("--------------------")

    # Check content specifically
    if hasattr(response, 'content') and response.content:
        print("\nContent successfully received:")
        print(response.content)
    else:
        print("\nWarning: Response object received, but content might be missing or empty.")

except Exception as e:
    print(f"\n--- Error during LLM invocation --- ")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\nTest finished.")
