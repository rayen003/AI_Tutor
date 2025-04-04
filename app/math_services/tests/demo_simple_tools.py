"""
Simplified demo script to demonstrate basic math tools.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool

# Import basic math calculations directly
from app.math_services.tools.math_tools import (
    present_value as pv_calc,
    future_value as fv_calc,
    pmt_calculation as pmt_calc,
    mean as mean_calc,
    median as median_calc,
    standard_deviation as stddev_calc
)

# Load environment variables
load_dotenv()

# Create LangChain tools
@tool
def present_value(future_value: float, rate: float, periods: int) -> float:
    """Calculate the present value of a future amount.
    
    Args:
        future_value: Future amount to be received
        rate: Interest rate in decimal form (e.g., 0.08 for 8%)
        periods: Number of time periods
        
    Returns:
        Present value amount
    """
    return pv_calc(future_value, rate, periods)

@tool
def future_value(present_value: float, rate: float, periods: int) -> float:
    """Calculate the future value of a present amount.
    
    Args:
        present_value: Present amount to be invested
        rate: Interest rate in decimal form (e.g., 0.06 for 6%)
        periods: Number of time periods
        
    Returns:
        Future value amount
    """
    return fv_calc(present_value, rate, periods)

@tool
def pmt_calculation(principal: float, rate: float, periods: int) -> float:
    """Calculate the payment for a loan.
    
    Args:
        principal: Loan amount
        rate: Interest rate in decimal form (e.g., 0.05 for 5%)
        periods: Number of payment periods
        
    Returns:
        Payment amount
    """
    return pmt_calc(principal, rate, periods)

@tool
def mean(data: list) -> float:
    """Calculate the mean (average) of a list of numbers.
    
    Args:
        data: List of numeric values
        
    Returns:
        Mean value
    """
    return mean_calc(data)

@tool
def median(data: list) -> float:
    """Calculate the median of a list of numbers.
    
    Args:
        data: List of numeric values
        
    Returns:
        Median value
    """
    return median_calc(data)

@tool
def standard_deviation(data: list) -> float:
    """Calculate the standard deviation of a list of numbers.
    
    Args:
        data: List of numeric values
        
    Returns:
        Standard deviation value
    """
    return stddev_calc(data)

def create_interactive_agent():
    """
    Create an agent with basic mathematical tools for demonstration.
    """
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    
    # Get all tools
    tools = [
        present_value,
        future_value,
        pmt_calculation,
        mean,
        median,
        standard_deviation
    ]
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert math tutor for business students. You have access to various mathematical tools to help solve financial and statistical problems."),
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
    """Demonstrate a present value calculation."""
    agent = create_interactive_agent()
    
    prompt = """
    I'm trying to understand present value. 
    If I'm going to receive $10,000 in 5 years, what is the present value of this amount with an 8% interest rate?
    Can you show me the calculation step by step?
    """
    
    response = agent.invoke({"input": prompt})
    print(response["output"])

def demo_future_value_calculation():
    """Demonstrate a future value calculation."""
    agent = create_interactive_agent()
    
    prompt = """
    I have $5,000 to invest today. If I can earn a 6% annual return, how much will my investment be worth in 3 years?
    Please show the calculation.
    """
    
    response = agent.invoke({"input": prompt})
    print(response["output"])

def demo_statistical_calculations():
    """Demonstrate statistical calculations."""
    agent = create_interactive_agent()
    
    prompt = """
    I have the following exam scores: 78, 85, 90, 72, 88, 95
    Can you calculate the mean, median, and standard deviation of these scores?
    """
    
    response = agent.invoke({"input": prompt})
    print(response["output"])

if __name__ == "__main__":
    print("\n===== DEMONSTRATING PRESENT VALUE CALCULATION =====\n")
    demo_present_value_calculation()
    
    print("\n===== DEMONSTRATING FUTURE VALUE CALCULATION =====\n")
    demo_future_value_calculation()
    
    print("\n===== DEMONSTRATING STATISTICAL CALCULATIONS =====\n")
    demo_statistical_calculations() 