#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 14:57:03 2017

@author: wuqianfan
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# initial setups

def Euler_PriceSimulation(T,M,S0):

    mu = 0
    
    sigma = 0.17
    lamda = 2.
    a = -0.05
    b = 0.03
       
    deltaT = 1/M
    
    n = 0
    t = 0
    St = S0
    At = 0
    S = np.random.exponential()

    S_path = []
    jump_time = 0
    
    while t<T:
        A_temp  = At + lamda*((n+1)*deltaT - t)
        
        if A_temp >= S:
            jump_time += 1
            
            Tk = S/lamda
            z1 = np.random.normal(loc=0,scale=np.sqrt(Tk - t))
            S_Tk_m = St + St*mu*(Tk - t) + St*sigma*z1
            
            jump_temp = S_Tk_m * (np.exp(np.random.normal(loc = a,scale =b))-1)
            S_Tk = S_Tk_m + jump_temp
            
            
            St = S_Tk
            S_path += [St]
            t = Tk
            At = S
            S = S + np.random.exponential()
        else:
            z2 = np.random.normal(loc = 0.0, scale = np.sqrt((n+1)* deltaT - t))
            S_n = St + St * mu*((n+1)*deltaT - t) + St * sigma * z2
            
            St = S_n
            S_path += [St]
            t = (n+1)*deltaT
            At = A_temp
            n = n + 1
    return St,jump_time,S_path





def generate_matrix (T,M,S0,n_estimators, N):
    Price_matrix = np.zeros((n_estimators, N))
    Jump_matrix = np.zeros((n_estimators, N))
    for i in range(n_estimators):
        for j in range(N):
         price,jp,path = Euler_PriceSimulation(T,M,S0)
         Price_matrix[i,j] = price
         Jump_matrix[i,j] = jp
    return Price_matrix, Jump_matrix
            
    

T = 3 # Need to change here
M = 100
S0 = 2000

N = 1000
n_estimators = 100


price_matrix, jump_matrix = generate_matrix(T,M,S0,n_estimators,N)

X = 2500 # this have to change



def simulate_payoff (e):
    simulated_payoff = np.zeros(n_estimators)
    for i in range(n_estimators):
        jump_ub = kj * deltaT**(-e)
        jump_flg = (jump_matrix[i] < jump_ub)
        payoff_vector = price_matrix[i] - X
        payoff_vector[payoff_vector<0] = 0
        simulated_payoff[i] = np.mean( jump_flg * payoff_vector )
    return simulated_payoff
        


mu = 0
sigma = 0.17
lamda = 2.
a = -0.05
b = 0.03




deltaT = 1/100
kj = 2



true_option_price =  14.5419

#####
epsilon = np.arange(0,0.5,0.0001)

'''
mse_vector = np.zeros(len(epsilon))
for i in range(len(epsilon)):
    optionprice = simulate_payoff(epsilon[i])
    mse_vector[i] = np.mean((optionprice - true_option_price) **2)
    

 
min_epsilon = epsilon[np.argmin(mse_vector)]

plt.plot(epsilon, mse_vector)
plt.xlabel("Epsilon")
plt.ylabel("Mean Sqaure Error")

plt.show
'''

######



'''
variance = np.zeros((len(epsilon)))
for i in range(len(epsilon)):
    optionprice  = simulate_payoff(epsilon[i])
    variance[i] = np.var(optionprice)
 
min_epsilon = epsilon[np.argmin(variance)]

plt.plot(epsilon, variance)
plt.xlabel("Epsilon")
plt.ylabel("Variance")

plt.show
'''

######


bias = np.zeros((len(epsilon)))
for i in range(len(epsilon)):
    optionprice  = simulate_payoff(epsilon[i])
    bias[i] = np.mean(optionprice - true_option_price)
 
min_epsilon = epsilon[np.argmin(bias)]

plt.plot(epsilon, bias)
plt.xlabel("Epsilon")
plt.ylabel("Bias")

plt.show



 

    
    
    
    
    
    
