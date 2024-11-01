import numpy as np
import time
import sympy as sp
import math as m
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

def order_of_convergence():
    if len(iteration_data)<3:
        return "insufficient data"

    num=m.log(abs(iteration_data[len(iteration_data)-1)]-root))         #assuming root is not the last value in the array. if the root last value then -2 
    denom=m.log(abs(iteration_data[len(iteration_data)-2)]-root))        #assuming root is not the last in the array. if root is the last value then -3

    order=m.log(x)                            # where x is calculated by hand using the limit formula

    alpha= num/(denom*order)

    return alpha
    




    #saving iteration data to a text file







    # plotting the function and successive approximations
    
