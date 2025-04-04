"""
Integrates the MathTutor's interactive tools with LangChain.

This module provides the interface for the language model to interact
with our custom tools, allowing for enhanced mathematical instruction.
"""

from typing import Dict, Any, List, Optional, Type, Union, Callable, Tuple
from pydantic import BaseModel, create_model, Field
import inspect
from langchain.tools import BaseTool, Tool

from app.math_services.tools.interactive_tools import InteractiveTools
from app.math_services.tools.math_tools import (
    present_value, future_value, pmt_calculation,
    mean, median, standard_deviation
)

class MathToolkit:
    """
    Toolkit for integrating interactive math tools with LangChain.
    
    This class converts our custom tools into LangChain-compatible tools
    that can be used in agent workflows.
    """
    
    @staticmethod
    def _create_tool_from_function(
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> BaseTool:
        """
        Create a LangChain tool from a function.
        
        Args:
            func: The function to convert to a tool
            name: Optional name for the tool (defaults to function name)
            description: Optional description (defaults to function docstring)
            
        Returns:
            A LangChain BaseTool instance
        """
        func_name = name or func.__name__
        func_description = description or func.__doc__ or f"Tool to {func_name.replace('_', ' ')}"
        
        # Inspect the function signature to get parameters
        signature = inspect.signature(func)
        parameters = {}
        
        for param_name, param in signature.parameters.items():
            if param_name == 'self':
                continue
                
            # Get annotation or default to Any
            annotation = param.annotation if param.annotation != inspect.Parameter.empty else Any
            
            # Get default value if available
            default = None if param.default == inspect.Parameter.empty else param.default
            
            # Handle Tuple types specifically for OpenAI schema compatibility
            if getattr(annotation, "__origin__", None) == tuple:
                # For x_range and similar tuple parameters, create a special field with items schema
                if param_name == "x_range":
                    parameters[param_name] = (List[float], Field(
                        default=default,
                        description="Min and max values for x axis as [min_x, max_x]",
                    ))
                elif param_name == "point_of_interest":
                    # Using List[float] with explicit schema for OpenAI
                    parameters[param_name] = (List[float], Field(
                        default=None,
                        description="Optional point to highlight on the graph as [x, y]",
                    ))
                else:
                    # Generic tuple handling
                    parameters[param_name] = (List[Any], Field(default=default))
            else:
                # Regular parameter handling
                parameters[param_name] = (annotation, Field(default=default))
        
        # Create a Pydantic model for the function parameters
        param_model = create_model(f"{func_name.capitalize()}Parameters", **parameters)
        
        # Create the tool function
        def tool_func(**kwargs):
            try:
                # Convert list parameters back to tuples for functions expecting tuples
                for param_name, param in signature.parameters.items():
                    if param_name in kwargs and getattr(param.annotation, "__origin__", None) == tuple:
                        if isinstance(kwargs[param_name], list):
                            kwargs[param_name] = tuple(kwargs[param_name])
                
                result = func(**kwargs)
                return result
            except Exception as e:
                return {"error": str(e)}
        
        # Create the tool with the appropriate syntax
        return Tool(
            name=func_name,
            description=func_description,
            func=tool_func,
            args_schema=param_model
        )
    
    @classmethod
    def get_math_calculation_tools(cls) -> List[BaseTool]:
        """
        Get LangChain tools for math calculations.
        
        Returns:
            List of LangChain tools for basic math calculations
        """
        functions = [
            present_value,
            future_value,
            pmt_calculation,
            mean,
            median,
            standard_deviation
        ]
        
        return [cls._create_tool_from_function(func) for func in functions]
    
    @classmethod
    def get_interactive_tools(cls) -> List[BaseTool]:
        """
        Get LangChain tools for interactive visualizations and step-by-step solutions.
        
        Returns:
            List of LangChain tools for interactive math learning
        """
        # Get InteractiveTools methods (excluding internal and special methods)
        methods = [
            getattr(InteractiveTools, name) for name in dir(InteractiveTools)
            if not name.startswith('_') and callable(getattr(InteractiveTools, name))
        ]
        
        return [cls._create_tool_from_function(method) for method in methods]
    
    @classmethod
    def get_all_tools(cls) -> List[BaseTool]:
        """
        Get all available math tools.
        
        Returns:
            Combined list of all available math tools
        """
        return cls.get_math_calculation_tools() + cls.get_interactive_tools()
    
    @staticmethod
    def add_tools_to_agent(agent_executor, tools: Optional[List[BaseTool]] = None):
        """
        Add math tools to an existing agent executor.
        
        Args:
            agent_executor: The LangChain agent executor to add tools to
            tools: Optional list of specific tools to add (defaults to all tools)
        
        Returns:
            Updated agent executor with math tools
        """
        if tools is None:
            tools = MathToolkit.get_all_tools()
            
        # Add tools to the agent
        agent_executor.tools.extend(tools)
        
        return agent_executor
    
    @staticmethod
    def format_tool_response(response: Any) -> str:
        """
        Format a tool response for display to the user.
        
        Args:
            response: The response from a tool call
            
        Returns:
            Formatted string for display
        """
        if isinstance(response, dict) and "visualization" in response:
            # Handle responses with visualizations
            result = f"## Results\n\n"
            
            if "steps" in response and response["steps"]:
                result += "### Steps\n\n"
                for i, step in enumerate(response["steps"]):
                    result += f"{step}\n\n"
            
            if "result" in response and response["result"] is not None:
                result += f"### Final Result\n\n{response['result']}\n\n"
                
            result += "### Visualization\n\n"
            result += response["visualization"]
            
            return result
            
        elif isinstance(response, dict) and "steps" in response:
            # Handle step-by-step solutions
            result = "## Step-by-Step Solution\n\n"
            for step in response["steps"]:
                result += f"{step}\n\n"
                
            if "solution" in response:
                result += f"## Solution\n\n{response['solution']}\n\n"
                
            return result
            
        elif isinstance(response, list) and all(isinstance(item, dict) and "statement" in item for item in response):
            # Handle practice problems
            result = "## Practice Problems\n\n"
            for i, problem in enumerate(response):
                result += f"### Problem {i+1}\n\n"
                result += f"{problem['statement']}\n\n"
                
            return result
            
        elif isinstance(response, str) and response.startswith("!["):
            # Handle image responses (plots)
            return f"## Visualization\n\n{response}"
            
        else:
            # Default handling
            return f"## Result\n\n{response}" 