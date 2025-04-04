"""
Interactive tools for the MathTutor platform.
These tools provide interactive capabilities that enhance the learning experience.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from sympy import symbols, sympify, solve, Eq, simplify, diff, integrate, limit, oo
from typing import List, Dict, Any, Optional, Union, Tuple
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import the math tools for calculations
from app.math_services.tools.math_tools import (
    present_value, future_value, pmt_calculation,
    mean, median, standard_deviation
)

class InteractiveTools:
    """
    Interactive tools that can generate visual aids, interactive elements,
    and perform symbolic manipulation to enhance the learning experience.
    """
    
    @staticmethod
    def plot_function(
        function_str: str,
        x_range: Tuple[float, float],
        title: str = "Function Plot",
        x_label: str = "x",
        y_label: str = "f(x)",
        point_of_interest: Optional[Tuple[float, float]] = None
    ) -> str:
        """
        Plot a mathematical function and return the plot as a base64 encoded string.
        
        Args:
            function_str: String representation of the function (e.g., "x**2 + 2*x + 1")
            x_range: Tuple of (min_x, max_x) values for the plot
            title: Plot title
            x_label: Label for x axis
            y_label: Label for y axis
            point_of_interest: Optional point to highlight on the graph
            
        Returns:
            Base64 encoded string of the plot
        """
        try:
            # Create the figure
            plt.figure(figsize=(10, 6))
            
            # Generate x values
            x = np.linspace(x_range[0], x_range[1], 1000)
            
            # Create a function from the string
            def f(x):
                # Replace x in the function string with the current value
                return eval(function_str, {"x": x, "np": np, "math": __import__("math")})
            
            # Generate y values
            try:
                y = f(x)
                
                # Plot the function
                plt.plot(x, y, 'b-', linewidth=2)
                
                # Add point of interest if provided
                if point_of_interest:
                    plt.plot(point_of_interest[0], point_of_interest[1], 'ro', markersize=8)
                    plt.annotate(
                        f'({point_of_interest[0]:.2f}, {point_of_interest[1]:.2f})',
                        xy=point_of_interest,
                        xytext=(10, 10),
                        textcoords='offset points'
                    )
                
                # Add labels and title
                plt.title(title)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.grid(True)
                
                # Save the plot to a bytes buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                
                # Encode the image to base64
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()
                
                # Return the image as base64 string with markdown
                return f"![{title}](data:image/png;base64,{img_base64})"
            except Exception as e:
                plt.close()
                return f"Error evaluating function: {str(e)}"
        except Exception as e:
            plt.close()
            return f"Error creating plot: {str(e)}"
    
    @staticmethod
    def calculate_steps(problem_type: str, params: Dict[str, float]) -> Dict[str, Any]:
        """
        Perform step-by-step calculation for a specific type of problem.
        
        Args:
            problem_type: Type of financial calculation (e.g., "present_value", "future_value")
            params: Dictionary of parameters needed for the calculation
            
        Returns:
            Dictionary with steps, result, and optional visualization
        """
        result = {"steps": [], "result": None, "visualization": None}
        
        if problem_type == "present_value":
            # Extract parameters
            fv = params.get("future_value")
            rate = params.get("rate")
            periods = params.get("periods")
            
            if None in [fv, rate, periods]:
                result["steps"].append("Error: Missing parameters")
                return result
            
            # Step 1: Identify the formula
            result["steps"].append(
                "Step 1: Use the present value formula: PV = FV / (1 + r)^n"
            )
            
            # Step 2: Substitute values
            result["steps"].append(
                f"Step 2: Substitute values: PV = {fv} / (1 + {rate})^{periods}"
            )
            
            # Step 3: Calculate denominator
            denominator = (1 + rate) ** periods
            result["steps"].append(
                f"Step 3: Calculate (1 + {rate})^{periods} = {denominator:.4f}"
            )
            
            # Step 4: Calculate result
            pv = fv / denominator
            result["steps"].append(
                f"Step 4: Calculate PV = {fv} / {denominator:.4f} = {pv:.2f}"
            )
            
            result["result"] = pv
            
            # Create visualization
            years = list(range(periods + 1))
            values = [pv]
            
            for i in range(1, periods + 1):
                values.append(pv * (1 + rate) ** i)
            
            plt.figure(figsize=(10, 6))
            plt.plot(years, values, 'bo-', linewidth=2)
            plt.title("Growth of Investment Over Time")
            plt.xlabel("Years")
            plt.ylabel("Value ($)")
            plt.grid(True)
            
            # Add annotations for start and end points
            plt.annotate(
                f'PV: ${pv:.2f}',
                xy=(0, pv),
                xytext=(10, 10),
                textcoords='offset points'
            )
            plt.annotate(
                f'FV: ${fv:.2f}',
                xy=(periods, fv),
                xytext=(10, 10),
                textcoords='offset points'
            )
            
            # Save the plot to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Encode the image to base64
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            result["visualization"] = f"![Growth over time](data:image/png;base64,{img_base64})"
            
        elif problem_type == "future_value":
            # Similar implementation for future value calculation
            pv = params.get("present_value")
            rate = params.get("rate")
            periods = params.get("periods")
            
            if None in [pv, rate, periods]:
                result["steps"].append("Error: Missing parameters")
                return result
            
            # Step 1: Identify the formula
            result["steps"].append(
                "Step 1: Use the future value formula: FV = PV × (1 + r)^n"
            )
            
            # Step 2: Substitute values
            result["steps"].append(
                f"Step 2: Substitute values: FV = {pv} × (1 + {rate})^{periods}"
            )
            
            # Step 3: Calculate (1 + r)^n
            growth_factor = (1 + rate) ** periods
            result["steps"].append(
                f"Step 3: Calculate (1 + {rate})^{periods} = {growth_factor:.4f}"
            )
            
            # Step 4: Calculate result
            fv = pv * growth_factor
            result["steps"].append(
                f"Step 4: Calculate FV = {pv} × {growth_factor:.4f} = {fv:.2f}"
            )
            
            result["result"] = fv
            
            # Create visualization
            years = list(range(periods + 1))
            values = [pv]
            
            for i in range(1, periods + 1):
                values.append(pv * (1 + rate) ** i)
            
            plt.figure(figsize=(10, 6))
            plt.plot(years, values, 'ro-', linewidth=2)
            plt.title("Growth of Investment Over Time")
            plt.xlabel("Years")
            plt.ylabel("Value ($)")
            plt.grid(True)
            
            # Add annotations for start and end points
            plt.annotate(
                f'PV: ${pv:.2f}',
                xy=(0, pv),
                xytext=(10, 10),
                textcoords='offset points'
            )
            plt.annotate(
                f'FV: ${fv:.2f}',
                xy=(periods, fv),
                xytext=(10, 10),
                textcoords='offset points'
            )
            
            # Save the plot to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Encode the image to base64
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            result["visualization"] = f"![Growth over time](data:image/png;base64,{img_base64})"
            
        return result

    @staticmethod
    def solve_equation(equation_str: str, variable: str = 'x') -> Dict[str, Any]:
        """
        Solve an algebraic equation and show steps.
        
        Args:
            equation_str: String representation of the equation (e.g., "2*x + 3 = 7")
            variable: Variable to solve for
            
        Returns:
            Dictionary with steps and solution
        """
        result = {"steps": [], "solution": None}
        
        try:
            # Parse the equation
            result["steps"].append(f"Step 1: Parse the equation: {equation_str}")
            
            # Check if equation contains '='
            if '=' in equation_str:
                left_str, right_str = equation_str.split('=')
                left_expr = sympify(left_str.strip())
                right_expr = sympify(right_str.strip())
                
                # Move all terms to left side
                equation = left_expr - right_expr
            else:
                # Assume it's already in the form f(x) = 0
                equation = sympify(equation_str)
            
            result["steps"].append(f"Step 2: Standard form (setting equation equal to zero): {equation} = 0")
            
            # Create symbol
            x = symbols(variable)
            
            # Solve equation
            solutions = solve(equation, x)
            result["steps"].append(f"Step 3: Solve for {variable}")
            
            if solutions:
                result["steps"].append(f"Step 4: Solution(s): {variable} = {', '.join([str(sol) for sol in solutions])}")
                result["solution"] = solutions
            else:
                result["steps"].append("Step 4: No solutions found")
                result["solution"] = []
            
            return result
        except Exception as e:
            result["steps"].append(f"Error solving equation: {str(e)}")
            return result

    @staticmethod
    def generate_practice_problems(topic: str, difficulty: str, count: int = 3) -> List[Dict[str, Any]]:
        """
        Generate practice problems based on topic and difficulty.
        
        Args:
            topic: Mathematical topic (e.g., "present_value", "quadratic_equations")
            difficulty: Difficulty level ("easy", "medium", "hard")
            count: Number of problems to generate
            
        Returns:
            List of dictionaries with problem statements and solutions
        """
        problems = []
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        if topic == "present_value":
            for i in range(count):
                if difficulty == "easy":
                    fv = np.random.randint(1000, 5000)
                    rate = np.round(np.random.uniform(0.05, 0.1), 2)
                    periods = np.random.randint(1, 5)
                elif difficulty == "medium":
                    fv = np.random.randint(5000, 20000)
                    rate = np.round(np.random.uniform(0.02, 0.2), 2)
                    periods = np.random.randint(5, 10)
                else:  # hard
                    fv = np.random.randint(20000, 100000)
                    rate = np.round(np.random.uniform(0.01, 0.25), 2)
                    periods = np.random.randint(10, 30)
                    
                pv = present_value(fv, rate, periods)
                
                problem = {
                    "id": f"pv_{i+1}",
                    "statement": f"What is the present value of ${fv:,} to be received in {periods} years at a {rate*100:.1f}% interest rate?",
                    "solution": {
                        "steps": [
                            f"Using the present value formula: PV = FV / (1 + r)^n",
                            f"Substituting: PV = {fv} / (1 + {rate})^{periods}",
                            f"Calculating: PV = {fv} / {(1 + rate) ** periods:.4f}",
                            f"Final answer: PV = ${pv:.2f}"
                        ],
                        "answer": f"${pv:.2f}"
                    }
                }
                problems.append(problem)
                
        elif topic == "future_value":
            for i in range(count):
                if difficulty == "easy":
                    pv = np.random.randint(1000, 5000)
                    rate = np.round(np.random.uniform(0.05, 0.1), 2)
                    periods = np.random.randint(1, 5)
                elif difficulty == "medium":
                    pv = np.random.randint(5000, 20000)
                    rate = np.round(np.random.uniform(0.02, 0.2), 2)
                    periods = np.random.randint(5, 10)
                else:  # hard
                    pv = np.random.randint(20000, 100000)
                    rate = np.round(np.random.uniform(0.01, 0.25), 2)
                    periods = np.random.randint(10, 30)
                    
                fv = future_value(pv, rate, periods)
                
                problem = {
                    "id": f"fv_{i+1}",
                    "statement": f"If you invest ${pv:,} at {rate*100:.1f}% interest for {periods} years, what will be the future value?",
                    "solution": {
                        "steps": [
                            f"Using the future value formula: FV = PV × (1 + r)^n",
                            f"Substituting: FV = {pv} × (1 + {rate})^{periods}",
                            f"Calculating: FV = {pv} × {(1 + rate) ** periods:.4f}",
                            f"Final answer: FV = ${fv:.2f}"
                        ],
                        "answer": f"${fv:.2f}"
                    }
                }
                problems.append(problem)
                
        elif topic == "linear_equations":
            for i in range(count):
                # Generate coefficients based on difficulty
                if difficulty == "easy":
                    a = np.random.randint(1, 5)
                    b = np.random.randint(1, 10)
                    c = np.random.randint(1, 20)
                elif difficulty == "medium":
                    a = np.random.randint(2, 10)
                    b = np.random.randint(5, 20)
                    c = np.random.randint(10, 50)
                else:  # hard
                    a = np.random.randint(5, 20)
                    b = np.random.randint(10, 50)
                    c = np.random.randint(20, 100)
                
                # Solution: x = (c - b) / a
                solution = (c - b) / a
                
                problem = {
                    "id": f"lin_{i+1}",
                    "statement": f"Solve for x: {a}x + {b} = {c}",
                    "solution": {
                        "steps": [
                            f"Original equation: {a}x + {b} = {c}",
                            f"Subtract {b} from both sides: {a}x = {c} - {b} = {c-b}",
                            f"Divide both sides by {a}: x = {c-b} / {a} = {solution}"
                        ],
                        "answer": f"{solution:.2f}"
                    }
                }
                problems.append(problem)
                
        return problems
    
    @staticmethod
    def compare_investments(
        investment_data: List[Dict[str, Any]],
        years: int
    ) -> Dict[str, Any]:
        """
        Compare multiple investment options over time.
        
        Args:
            investment_data: List of dictionaries with investment details
            years: Number of years for projection
            
        Returns:
            Dictionary with comparison details and visualization
        """
        result = {"comparison": [], "best_option": None, "visualization": None}
        
        try:
            # Create a DataFrame for storing year-by-year values
            df = pd.DataFrame()
            df['Year'] = list(range(years + 1))
            
            for inv in investment_data:
                name = inv.get("name", "Unknown")
                initial = inv.get("initial", 0)
                rate = inv.get("rate", 0)
                
                # Calculate values for each year
                values = [initial]
                for i in range(1, years + 1):
                    values.append(initial * (1 + rate) ** i)
                
                # Add to DataFrame
                df[name] = values
                
                # Calculate final value
                final_value = values[-1]
                
                # Add to comparison
                result["comparison"].append({
                    "name": name,
                    "initial": initial,
                    "rate": rate,
                    "final_value": final_value,
                    "gain": final_value - initial,
                    "gain_pct": (final_value - initial) / initial * 100
                })
            
            # Determine best option
            best = max(result["comparison"], key=lambda x: x["final_value"])
            result["best_option"] = best["name"]
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            for inv in investment_data:
                name = inv.get("name", "Unknown")
                plt.plot(df['Year'], df[name], marker='o', linewidth=2, label=name)
            
            plt.title("Investment Growth Comparison")
            plt.xlabel("Years")
            plt.ylabel("Value ($)")
            plt.grid(True)
            plt.legend()
            
            # Save the plot to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Encode the image to base64
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            result["visualization"] = f"![Investment Comparison](data:image/png;base64,{img_base64})"
            
            return result
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def get_tool_descriptions() -> List[Dict[str, Any]]:
        """
        Return a list of available interactive tools in a format 
        that can be used with LangChain's tool calling capabilities.
        """
        return [
            {
                "name": "plot_function",
                "description": "Plot a mathematical function",
                "parameters": {
                    "function_str": "String representation of the function (e.g., 'x**2 + 2*x + 1')",
                    "x_range": "Tuple of (min_x, max_x) values for the plot",
                    "title": "Plot title (optional)",
                    "x_label": "Label for x axis (optional)",
                    "y_label": "Label for y axis (optional)",
                    "point_of_interest": "Optional point to highlight on the graph (x, y)"
                }
            },
            {
                "name": "calculate_steps",
                "description": "Perform step-by-step calculation for a specific type of problem",
                "parameters": {
                    "problem_type": "Type of calculation (e.g., 'present_value', 'future_value')",
                    "params": "Dictionary of parameters needed for the calculation"
                }
            },
            {
                "name": "solve_equation",
                "description": "Solve an algebraic equation and show steps",
                "parameters": {
                    "equation_str": "String representation of the equation (e.g., '2*x + 3 = 7')",
                    "variable": "Variable to solve for (optional, defaults to 'x')"
                }
            },
            {
                "name": "generate_practice_problems",
                "description": "Generate practice problems based on topic and difficulty",
                "parameters": {
                    "topic": "Mathematical topic (e.g., 'present_value', 'quadratic_equations')",
                    "difficulty": "Difficulty level ('easy', 'medium', 'hard')",
                    "count": "Number of problems to generate (optional, defaults to 3)"
                }
            },
            {
                "name": "compare_investments",
                "description": "Compare multiple investment options over time",
                "parameters": {
                    "investment_data": "List of dictionaries with investment details",
                    "years": "Number of years for projection"
                }
            }
        ] 