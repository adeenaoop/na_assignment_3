import numpy as np
import time
import sympy as sp
import matplotlib.pyplot as plt
def NewtonsMethod(f,p0,tolerance,multi,is_poly):
    x = sp.symbols('x')
    f_prime = sp.diff(f, x)
    

    #if is_poly: 




    iteration_data = []
    start_time = time.time()
    num_iterations = 0
    root = p0    #current guess of the root



    #iterative loop for the actual method
    









    #calculating CPU time
    cpu_time = time.time() - start_time



    #calculating order of convergence






    #saving iteration data to a text file







    # plotting the function and successive approximations
    
