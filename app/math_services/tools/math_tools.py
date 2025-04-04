"""
Mathematical tools for the MathTutor platform.
These tools provide specialized functions for financial, statistical, 
and other business-oriented mathematical operations.
"""
import math
import numpy as np
from typing import List, Dict, Any, Optional, Union

# Financial calculators
def present_value(future_value: float, rate: float, periods: int) -> float:
    """
    Calculate the present value of a future sum.
    
    Args:
        future_value: Future value amount
        rate: Interest rate per period (as a decimal)
        periods: Number of periods
        
    Returns:
        Present value
    """
    return future_value / ((1 + rate) ** periods)

def future_value(present_value: float, rate: float, periods: int) -> float:
    """
    Calculate the future value of a present sum.
    
    Args:
        present_value: Present value amount
        rate: Interest rate per period (as a decimal)
        periods: Number of periods
        
    Returns:
        Future value
    """
    return present_value * ((1 + rate) ** periods)

def pmt_calculation(principal: float, rate: float, periods: int) -> float:
    """
    Calculate the payment for a loan based on constant payments and a constant interest rate.
    
    Args:
        principal: Loan principal
        rate: Interest rate per period (as a decimal)
        periods: Number of periods
        
    Returns:
        Payment amount per period
    """
    if rate == 0:
        return principal / periods
    return (principal * rate * (1 + rate) ** periods) / ((1 + rate) ** periods - 1)

# Statistical tools
def mean(values: List[float]) -> float:
    """Calculate the mean of a list of values."""
    return sum(values) / len(values)

def median(values: List[float]) -> float:
    """Calculate the median of a list of values."""
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n % 2 == 0:
        return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
    else:
        return sorted_values[n//2]

def standard_deviation(values: List[float], sample: bool = True) -> float:
    """
    Calculate the standard deviation of a list of values.
    
    Args:
        values: List of numerical values
        sample: If True, calculate sample standard deviation; if False, calculate population standard deviation
        
    Returns:
        Standard deviation
    """
    avg = mean(values)
    variance = sum((x - avg) ** 2 for x in values) / (len(values) - 1 if sample else len(values))
    return math.sqrt(variance)

def confidence_interval(values: List[float], confidence: float = 0.95) -> Dict[str, float]:
    """
    Calculate the confidence interval for a list of values.
    
    Args:
        values: List of numerical values
        confidence: Confidence level (default: 0.95 for 95% confidence)
        
    Returns:
        Dictionary containing mean, margin of error, lower bound, and upper bound
    """
    import scipy.stats as stats
    
    avg = mean(values)
    std_dev = standard_deviation(values)
    n = len(values)
    
    # Calculate margin of error
    z_score = stats.norm.ppf((1 + confidence) / 2)
    margin_of_error = z_score * (std_dev / math.sqrt(n))
    
    return {
        "mean": avg,
        "margin_of_error": margin_of_error,
        "lower_bound": avg - margin_of_error,
        "upper_bound": avg + margin_of_error
    }

# Accounting tools
def straight_line_depreciation(cost: float, salvage_value: float, useful_life: int) -> Dict[str, Union[float, List[float]]]:
    """
    Calculate straight-line depreciation schedule.
    
    Args:
        cost: Initial cost of the asset
        salvage_value: Estimated salvage value at the end of useful life
        useful_life: Useful life in years
        
    Returns:
        Dictionary with annual depreciation and year-by-year value
    """
    annual_depreciation = (cost - salvage_value) / useful_life
    yearly_values = []
    
    current_value = cost
    for _ in range(useful_life):
        current_value -= annual_depreciation
        yearly_values.append(current_value)
    
    return {
        "annual_depreciation": annual_depreciation,
        "yearly_values": yearly_values
    }

# Economics tools
def price_elasticity(initial_price: float, final_price: float, 
                    initial_quantity: float, final_quantity: float) -> float:
    """
    Calculate price elasticity of demand.
    
    Args:
        initial_price: Initial price
        final_price: Final price
        initial_quantity: Initial quantity demanded
        final_quantity: Final quantity demanded
        
    Returns:
        Price elasticity of demand
    """
    price_change_pct = (final_price - initial_price) / initial_price
    quantity_change_pct = (final_quantity - initial_quantity) / initial_quantity
    
    return quantity_change_pct / price_change_pct if price_change_pct != 0 else float('inf')

# Function to expose tools to the LangChain tools framework
def get_available_tools() -> List[Dict[str, Any]]:
    """
    Returns a list of available mathematical tools in a format
    that can be used with LangChain's tool calling capabilities.
    """
    return [
        {
            "name": "present_value",
            "description": "Calculate the present value of a future sum",
            "parameters": {
                "future_value": "Future value amount",
                "rate": "Interest rate per period (as a decimal)",
                "periods": "Number of periods"
            }
        },
        {
            "name": "future_value",
            "description": "Calculate the future value of a present sum",
            "parameters": {
                "present_value": "Present value amount",
                "rate": "Interest rate per period (as a decimal)",
                "periods": "Number of periods"
            }
        },
        {
            "name": "pmt_calculation",
            "description": "Calculate the payment for a loan based on constant payments and interest",
            "parameters": {
                "principal": "Loan principal",
                "rate": "Interest rate per period (as a decimal)",
                "periods": "Number of periods"
            }
        },
        {
            "name": "mean",
            "description": "Calculate the mean of a list of values",
            "parameters": {
                "values": "List of numerical values"
            }
        },
        {
            "name": "median",
            "description": "Calculate the median of a list of values",
            "parameters": {
                "values": "List of numerical values"
            }
        },
        {
            "name": "standard_deviation",
            "description": "Calculate the standard deviation of a list of values",
            "parameters": {
                "values": "List of numerical values",
                "sample": "If True, calculate sample standard deviation; if False, calculate population standard deviation"
            }
        }
    ] 