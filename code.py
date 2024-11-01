import numpy as np
import time
import sympy as sp
from sympy import symbols, Poly
import matplotlib.pyplot as plt
import math as m

# Define the variable and polynomial function
x = symbols('x')
function = 2*x**4 - 3*x**2 + 3*x - 4  # Example polynomial
poly = Poly(function, x)
coefficients = poly.all_coeffs()  
coefficients = [float(c) for c in coefficients]  # Convert coefficients to float

def horner_method(x0):
    """Evaluate polynomial and its derivative using Horner's method."""
    y = coefficients[0]  # Initialize P(x)
    z = coefficients[0]  # Initialize P'(x) for the derivative
    # Loop to calculate P(x) and P'(x)
    for j in range(1, len(coefficients)):
        y = x0 * y + coefficients[j]  # Compute the value of the polynomial
        z = x0 * z + y if j < len(coefficients) - 1 else z  # Compute derivative up to n-1
    return y, z  # Return P(x) and P'(x)

def newton_method(f, f_prime, root, m, tolerance, max_iterations=100):
    """Estimate root using Newton's Method."""
    num_iterations = 0
    iteration_data = []

    start_time = time.time()
    while True:
        f_val, f_prime_val = f(root), f_prime(root)

        if f_prime_val == 0:
            print("The derivative is zero, cannot proceed with Newton's Method :(")
            break

        if m == 1:
            # Standard Newton's Method
            next_root = root - f_val / f_prime_val
        else:
            # Modified Newton's Method
            next_root = root - (m * f_val) / f_prime_val

        error = abs(next_root - root)
        iteration_data.append((num_iterations, root, f_val, f_prime_val, error))

        if error < tolerance:
            root = next_root
            break

        root = next_root
        num_iterations += 1

        if num_iterations > max_iterations:
            print("Maximum iterations exceeded! The method did not converge :(")
            break
    cpu_time = time.time() - start_time
    return root, num_iterations, iteration_data

def order_of_convergence(iteration_data, root):
    """Calculate the order of convergence based on the last few iterations."""
    if len(iteration_data) < 3:
        return "insufficient data"

    num = m.log(abs(iteration_data[-1][1] - root))         # assuming root is not the last value in the array
    denom = m.log(abs(iteration_data[-2][1] - root))        # assuming root is not the last in the array

    order = m.log(abs(iteration_data[-2][1] - iteration_data[-3][1]))  # using values from the last few iterations

    alpha = num / (denom * order)

    return alpha

def main():
    m = int(input("Enter the multiplicity of the root: "))
    x0 = float(input("Enter the initial guess x0: "))
    tolerance = float(input("Enter the tolerance ε: "))
    is_polynomial = True  # This can be set based on the function being analyzed

    # Define the polynomial function and its derivative
    if is_polynomial:
        f = lambda x: horner_method(x)[0]  # Polynomial function value
        f_prime = lambda x: horner_method(x)[1]  # Polynomial derivative value
    else:
        f = lambda x: np.exp(x) - x - 1  # Example non-polynomial function
        f_prime = lambda x: np.exp(x) - 1  # Its derivative

    # Run Newton's Method
    root, iterations, iteration_data = newton_method(f, f_prime, x0, m, tolerance)

    # Print results
    print(f"Root: {root}, Iterations: {iterations}, CPU Time: {cpu_time:.6f} seconds")

    # Calculate order of convergence
    alpha = order_of_convergence(iteration_data, root)
    print(f"Order of Convergence: {alpha}")

    # Plotting the polynomial function
    x_vals = np.linspace(-2, 2, 400)
    y_vals = [f(val) for val in x_vals]
    plt.plot(x_vals, y_vals, label='Polynomial Function')
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(root, color='red', linestyle='--', label='Estimated Root')
    plt.title('Function and Successive Approximations')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()
