"""
Demo script to demonstrate the use of custom interactive tools.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

from app.math_services.tools.tool_integration import MathToolkit

# Load environment variables
load_dotenv()

def create_interactive_agent():
    """
    Create an agent with mathematical tools for interactive learning.
    """
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    
    # Get all math tools
    tools = MathToolkit.get_all_tools()
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert math tutor for business students. You have access to various mathematical tools to help explain concepts visually and interactively."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Initialize memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create the agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        memory=memory,
        verbose=True
    )
    
    return agent_executor

def demo_present_value_calculation():
    """Demonstrate a present value calculation with visualization."""
    agent = create_interactive_agent()
    
    prompt = """
    I'm trying to understand present value. 
    If I'm going to receive $10,000 in 5 years, what is the present value of this amount with an 8% interest rate?
    Can you show me the calculation step by step and create a visualization to explain this concept?
    """
    
    response = agent.invoke({"input": prompt})
    print(response["output"])

def demo_function_visualization():
    """Demonstrate plotting mathematical functions."""
    agent = create_interactive_agent()
    
    prompt = """
    I'm studying the relationship between cost, revenue, and profit. 
    Can you help me visualize this by plotting these functions:
    - Cost function: C(x) = 50 + 5*x
    - Revenue function: R(x) = 15*x
    - Profit function: P(x) = R(x) - C(x)
    
    Where x is the quantity of units. Plot these for quantities from 0 to 20 units.
    Also, can you identify the break-even point where profit becomes positive?
    """
    
    response = agent.invoke({"input": prompt})
    print(response["output"])

def demo_equation_solver():
    """Demonstrate the equation solving tool with step-by-step explanation."""
    agent = create_interactive_agent()
    
    prompt = """
    I'm working on the following equation from my business algebra class:
    5x + 3 = 2x - 9
    
    Can you solve this equation step by step and explain each step?
    """
    
    response = agent.invoke({"input": prompt})
    print(response["output"])

def demo_practice_problem_generation():
    """Demonstrate generating practice problems based on difficulty level."""
    agent = create_interactive_agent()
    
    prompt = """
    I'm preparing for a finance test that will include questions on future value.
    Can you generate 3 medium difficulty practice problems about future value that I can work on?
    """
    
    response = agent.invoke({"input": prompt})
    print(response["output"])
    
    # Let's solve one of these problems
    follow_up = """
    Great! Now can you solve the first practice problem for me and show me the step-by-step approach?
    """
    
    response = agent.invoke({"input": follow_up})
    print(response["output"])

def demo_investment_comparison():
    """Demonstrate comparing different investment options."""
    agent = create_interactive_agent()
    
    prompt = """
    I'm trying to decide between three investment options:
    1. Investment A: Initial amount $5,000 with 7% annual return
    2. Investment B: Initial amount $8,000 with 5% annual return
    3. Investment C: Initial amount $3,000 with 9% annual return
    
    Can you compare these investments over a 10-year period and help me understand which one would be the best choice?
    Please create a visualization to show the growth of each investment over time.
    """
    
    response = agent.invoke({"input": prompt})
    print(response["output"])

if __name__ == "__main__":
    print("\n===== DEMONSTRATING PRESENT VALUE CALCULATION =====\n")
    demo_present_value_calculation()
    
    print("\n===== DEMONSTRATING FUNCTION VISUALIZATION =====\n")
    demo_function_visualization()
    
    print("\n===== DEMONSTRATING EQUATION SOLVER =====\n")
    demo_equation_solver()
    
    print("\n===== DEMONSTRATING PRACTICE PROBLEM GENERATION =====\n")
    demo_practice_problem_generation()
    
    print("\n===== DEMONSTRATING INVESTMENT COMPARISON =====\n")
    demo_investment_comparison() 