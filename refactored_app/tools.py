import sympy
import numpy as np
import matplotlib.pyplot as plt
import math
from langchain.tools import tool

# --- Simple Calculator Tool ---
@tool
def calculator(expression: str) -> str:
    """Evaluates a simple mathematical expression (e.g., '3*y - 5 = 7', '2 + 2'). Solves basic equations for 'x' or 'y'."""
    try:
        # Use sympy for more robust equation solving
        if '=' in expression and ('x' in expression or 'y' in expression):
             # Try to parse and solve
             try:
                 # Define potential variables
                 x = sympy.symbols('x')
                 y = sympy.symbols('y')

                 # Try to identify the variable in the expression
                 if 'y' in expression:
                     var = y
                 elif 'x' in expression:
                     var = x
                 else:
                     # Should not happen if '=' and 'x' or 'y' are present, but as fallback:
                     return f"Error: Equation '{expression}' detected but variable 'x' or 'y' not found."

                 # Properly format for sympy.solve: expr = 0
                 lhs, rhs = expression.split('=', 1)
                 equation = sympy.sympify(lhs.strip()) - sympy.sympify(rhs.strip())

                 # Solve for the identified variable
                 solution = sympy.solve(equation, var)

                 if isinstance(solution, list):
                     if len(solution) == 1:
                         return str(solution[0])
                     else:
                         # Handle multiple solutions if necessary, maybe return list as string
                         return str(solution)
                 else: # Handle non-list results if sympy returns them
                    return str(solution)

             except (SyntaxError, TypeError, NameError, sympy.SympifyError, ValueError) as e:
                 return f"Error: Could not solve equation '{expression}'. Error: {e}"
        else:
            # Simple arithmetic - use a safer eval approach
            try:
                # Allow basic math operations, restrict environment
                safe_dict = {'__builtins__': None}
                # You might add safe functions here if needed, e.g. from math module
                result = eval(expression, {"__builtins__": None}, safe_dict)
                return str(result)
            except Exception as eval_e:
                 return f"Error: Could not evaluate expression '{expression}'. Error: {eval_e}"

    except Exception as e:
        # Catch-all for unexpected errors during processing
        return f"Error processing expression '{expression}': {e}"

# --- Finance Calculation Tools ---

@tool
def present_value(rate: float, n_periods: float, future_value: float) -> str:
    """Calculates the Present Value (PV) of a single future sum.
    Args:
        rate (float): The interest rate per period (e.g., 0.05 for 5%).
        n_periods (float): The number of periods.
        future_value (float): The future value amount.
    Returns:
        str: The calculated present value or an error message.
    """
    try:
        if rate <= -1:
            return "Error: Rate cannot be -100% or lower."
        pv = future_value / ((1 + rate) ** n_periods)
        return f"{pv:.2f}"
    except Exception as e:
        return f"Error calculating present value: {e}"

@tool
def future_value(rate: float, n_periods: float, present_value: float) -> str:
    """Calculates the Future Value (FV) of a single present sum.
    Args:
        rate (float): The interest rate per period (e.g., 0.05 for 5%).
        n_periods (float): The number of periods.
        present_value (float): The present value amount.
    Returns:
        str: The calculated future value or an error message.
    """
    try:
        if rate <= -1:
             return "Error: Rate cannot be -100% or lower."
        fv = present_value * ((1 + rate) ** n_periods)
        return f"{fv:.2f}"
    except Exception as e:
        return f"Error calculating future value: {e}"

@tool
def annuity_present_value(rate: float, n_periods: float, payment: float) -> str:
    """Calculates the Present Value (PV) of an ordinary annuity.
    Args:
        rate (float): The interest rate per period (e.g., 0.05 for 5%).
        n_periods (float): The number of periods.
        payment (float): The amount of each payment.
    Returns:
        str: The calculated present value of the annuity or an error message.
    """
    try:
        if rate == 0:
            pv = payment * n_periods
        elif rate <= -1:
             return "Error: Rate cannot be -100% or lower."
        else:
            pv = payment * ((1 - (1 + rate) ** -n_periods) / rate)
        return f"{pv:.2f}"
    except Exception as e:
        return f"Error calculating annuity present value: {e}"

@tool
def loan_payment_pmt(rate: float, n_periods: float, present_value: float) -> str:
    """Calculates the periodic payment (PMT) for a loan.
    Args:
        rate (float): The interest rate per period (e.g., 0.05 for 5%).
        n_periods (float): The total number of payment periods.
        present_value (float): The principal loan amount (PV).
    Returns:
        str: The calculated periodic payment or an error message.
    """
    try:
        if rate == 0:
            if n_periods == 0:
                 return "Error: Cannot calculate payment with zero rate and zero periods."
            pmt = present_value / n_periods
        elif rate <= -1:
             return "Error: Rate cannot be -100% or lower."
        elif n_periods <= 0:
            return "Error: Number of periods must be positive."
        else:
            pmt = (rate * present_value) / (1 - (1 + rate) ** -n_periods)
        return f"{pmt:.2f}"
    except Exception as e:
        return f"Error calculating loan payment (PMT): {e}"

# --- Economics Plotting Tool ---

@tool
def plot_supply_demand(supply_intercept: float, supply_slope: float, demand_intercept: float, demand_slope: float, filename: str = "sd_plot.png") -> str:
    """Plots simple linear supply and demand curves and saves the plot to a file.
    Assumes standard equations: Qs = supply_intercept + supply_slope*P and Qd = demand_intercept + demand_slope*P.
    Args:
        supply_intercept (float): The intercept of the supply curve (quantity when price is 0).
        supply_slope (float): The slope of the supply curve (change in Qs / change in P). Should be positive.
        demand_intercept (float): The intercept of the demand curve (quantity when price is 0).
        demand_slope (float): The slope of the demand curve (change in Qd / change in P). Should be negative.
        filename (str): The name of the file to save the plot to. Defaults to 'sd_plot.png'.
    Returns:
        str: Confirmation message indicating the file save location or an error message.
    """
    try:
        if supply_slope <= 0:
            return "Error: Supply slope must be positive."
        if demand_slope >= 0:
            return "Error: Demand slope must be negative."
        if supply_slope == demand_slope:
            return "Error: Supply and demand slopes cannot be equal."

        # Calculate equilibrium
        # P_eq: supply_intercept + supply_slope*P = demand_intercept + demand_slope*P
        # P_eq = (demand_intercept - supply_intercept) / (supply_slope - demand_slope)
        equilibrium_price = (demand_intercept - supply_intercept) / (supply_slope - demand_slope)
        if equilibrium_price < 0:
             # Although possible mathematically, doesn't make economic sense for basic model
             # Let's still plot but maybe add a note or find a reasonable plot range
             print("Warning: Calculated equilibrium price is negative.")
             # Avoid negative prices in plot range
             max_price_for_plot = max(abs(supply_intercept/supply_slope), abs(demand_intercept/demand_slope)) * 1.5
             equilibrium_price = 0 # Plot equilibrium at 0 if calculated negative for range purposes

        equilibrium_quantity = supply_intercept + supply_slope * equilibrium_price
        if equilibrium_quantity < 0:
            print("Warning: Calculated equilibrium quantity is negative.")
            equilibrium_quantity = 0 # Plot equilibrium at 0 if calculated negative

        # Determine plot range (make it look reasonable)
        # Find P where Qd=0: P = -demand_intercept / demand_slope
        # Find P where Qs=0 (if intercept is negative): P = -supply_intercept / supply_slope
        max_price = equilibrium_price * 2 # Start with a basic range
        if demand_slope != 0:
            price_at_zero_demand = -demand_intercept / demand_slope
            max_price = max(max_price, price_at_zero_demand * 1.2)
        if supply_intercept < 0 and supply_slope != 0:
             price_at_zero_supply = -supply_intercept / supply_slope
             max_price = max(max_price, price_at_zero_supply * 1.2)
        if max_price <= 0: max_price = 10 # Fallback max price

        max_quantity = max(demand_intercept, supply_intercept + supply_slope * max_price) * 1.2
        if max_quantity <= 0: max_quantity = 10 # Fallback max quantity

        prices = np.linspace(0, max_price, 100)
        quantity_supplied = supply_intercept + supply_slope * prices
        quantity_demanded = demand_intercept + demand_slope * prices

        # Ensure non-negative quantities for plotting
        quantity_supplied[quantity_supplied < 0] = 0
        quantity_demanded[quantity_demanded < 0] = 0

        plt.figure(figsize=(8, 6))
        plt.plot(quantity_supplied, prices, label='Supply', color='blue')
        plt.plot(quantity_demanded, prices, label='Demand', color='red')

        # Plot equilibrium point
        if equilibrium_price >= 0 and equilibrium_quantity >= 0:
             plt.plot(equilibrium_quantity, equilibrium_price, 'go', label=f'Equilibrium (Q={equilibrium_quantity:.2f}, P={equilibrium_price:.2f})')
             plt.hlines(equilibrium_price, 0, equilibrium_quantity, colors='grey', linestyles='dashed')
             plt.vlines(equilibrium_quantity, 0, equilibrium_price, colors='grey', linestyles='dashed')

        plt.title('Supply and Demand')
        plt.xlabel('Quantity')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, max_quantity)
        plt.ylim(0, max_price)

        # Save the plot
        plt.savefig(filename)
        plt.close() # Close the plot to free memory

        return f"Supply and demand plot saved to {filename}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error plotting supply and demand: {e}"

# Export the list of ALL tools
tools = [
    calculator,
    present_value,
    future_value,
    annuity_present_value,
    loan_payment_pmt,
    plot_supply_demand
]

# Example Usage (for testing this file directly)
if __name__ == '__main__':
    print("--- Testing Tools ---")
    print(f"Calculator '2+2': {calculator.invoke('2+2')}")
    print(f"PV(rate=0.05, n=10, fv=1000): {present_value.invoke({'rate': 0.05, 'n_periods': 10, 'future_value': 1000})}")
    print(f"FV(rate=0.03, n=5, pv=500): {future_value.invoke({'rate': 0.03, 'n_periods': 5, 'present_value': 500})}")
    print(f"Annuity PV(rate=0.07, n=20, pmt=100): {annuity_present_value.invoke({'rate': 0.07, 'n_periods': 20, 'payment': 100})}")
    print(f"Loan PMT(rate=0.01, n=36, pv=10000): {loan_payment_pmt.invoke({'rate': 0.01, 'n_periods': 36, 'present_value': 10000})}")
    print(f"Plot S&D (Qs = -10 + 2P, Qd = 50 - 1P): {plot_supply_demand.invoke({'supply_intercept': -10, 'supply_slope': 2, 'demand_intercept': 50, 'demand_slope': -1})}")
    print("Check for sd_plot.png")