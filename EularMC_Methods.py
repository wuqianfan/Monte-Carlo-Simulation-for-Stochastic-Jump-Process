# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 14:53:26 2017


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
    
#    M = 100
#    T = 1    
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
            S_n = St + St * mu*(n+1)* (deltaT - t) + St * sigma * z2
            
            St = S_n
            S_path += [St]
            t = (n+1)*deltaT
            At = A_temp
            n = n + 1
    return St,jump_time,S_path
    
price,jp,path= priceSimulation(T,M,S0)


S0 = 2000
T = 1
M = 100

N = 2000
price_vec = np.zeros(N)
jump_vec = np.zeros(N)
for i in range(N):
    price,jp,path= priceSimulation(T,M,S0)
    price_vec[i] = price
    jump_vec[i] = jp
    

plt.hist(price_vec)
plt.hist(jump_vec)

kappa_J = 2
epsilon = 0.01
jump_ub = kappa_J * deltaT**(-epsilon)
jump_flg = (jump_vec < jump_ub)
payoff_vec = price_vec - X
payoff_vec[payoff_vec<0] = 0
option_price1 = np.mean(jump_flg*payoff_vec)



#############################################
mu = 0

sigma = 0.17
lamda = 2.
a = -0.05
b = 0.03

X = 2000



def preciseSimulate(T,S0,N1):

    jump_n = np.random.poisson(lam=lamda,size=N1)
    mean_S = np.log(S0) + (mu - sigma**2/2)*T + a*jump_n
    sigma_S = sigma**2*T + b**2*jump_n
    price_n = np.random.lognormal(mean = mean_S, sigma = np.sqrt(sigma_S))
    return price_n
    
stockprice_p = preciseSimulate(T,S0,50000000)
payoff_simulate = stockprice_p - X
aaa= np.maximum(payoff_simulate,np.zeros(np.shape(payoff_simulate)))
np.mean(aaa)


jump_n = np.random.poisson(lam=lamda,size=N1)
mean_S = np.log(S0) + (mu - sigma**2/2)*T + a*jump_n
sigma_S = sigma**2*T + b**2*jump_n
price_n = np.random.lognormal(mean = mean_S, sigma = np.sqrt(sigma_S))
