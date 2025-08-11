# This file is a training ground for STAR and LSTAR models

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize
#General form of a STAR model follows tre following structure:
# y_t = \phi_1 * y(t) *(1 - G(s_t, \gamma, c)) + \phi_2 * y(t) *G(s_t, \gamma, c) + \epsilon_t
# \phi_1 = phi_1_0 + phi_1_1L + phi_1_2L^2 + ... + phi_1_pL^p
# \phi_2 = phi_2_0 + phi_2_1L + phi_2_2L^2 + ... + phi_2_pL^p
# where G(s_t, \gamma, c) is a smooth transition function, 
# such as logistic : (1 + exp(-\gamma(s_t - C)))^(-1)           ->LSTAR
# or exponential : (1 + exp(-\gamma(s_t - C))^2)                ->ESTAR
# or second order logistic: (1 + exp(-\gamma(s_t - C_1)(s_t - C_2)))^(-1)    -> SETAR with THREE regimes
# s_t is the trransition variable. Can be exogenous or lagged endogeneous.
# c - threshold parameter switching between two regimes
# \gamma - smoothness parameter reflecting the speed of transition between regimes
# large \gamma - instanteneous change of the regimes

def logistic_transition_function(s, gamma, c):
    return 1 / (1 + np.exp(-gamma * (s - c)))

def star1_model(parameters, trigger, data):
    """A simple STAR1 model with the logistic transition function.
    The idea is to isolate gamma and c parameters from the model, leaving
    the remaining parameters linear.

    Args:
        phi1 (_type_): Normal state AR parameter
        phi2 (_type_): Transitioned state AR parameter
        c (_type_): threshold parameter
        gamma (_type_): smoothness parameter
        trigger (_type_): transition variable (exogeneous or lagged endogeneous)
        data (_type_): data
    """
    gamma, c = parameters
    trigger = data[:-1].copy()
    transition = logistic_transition_function(trigger, gamma, c)
    
    # y = phi1 y_t-1 + phi2 y_t-1 * transition + epsilon_t
    
    # params = np.array([phi1, phi2])
    X = np.column_stack([data[:-1], data[:-1]*transition])
    
    beta = np.linalg.inv(X.T @ X)@X.T @ data[1:]
    y_fit = X @ beta
    
    residuals = data[1:] - y_fit
    # sigma = np.std(residuals)
    
    return beta, X, residuals

def star1_mle(params, trigger, data):
    
    beta, X, residuals = star1_model(params, trigger, data)
    
    sigma = np.std(residuals)
    residuals /= sigma
    n = len(residuals)
    llf = -n / 2 * np.log(2 * np.pi * sigma**2) - 0.5 * np.sum(residuals**2)
    return -llf


    
    
    
    
    


    





if __name__ == "__main__":
    # simulation
    np.random.seed(42)
    phi1 = 0.5
    phi2 = -1
    
    noise = stats.t.rvs(df=30, loc = 0, scale=1, size = 200)
    data = noise.copy()
    transition = np.zeros(data.shape)
    
    for i in range(1, len(data)):
        transition[i] = logistic_transition_function(data[i-1], 2, 1)
        # data[i] = phi1 * data[i-1] + (phi2 - phi1) *data[i-1]* transition[i] + noise[i]
        data[i] = phi1 * data[i-1] + phi2*data[i-1]* transition[i] + noise[i]
    
    

    # 3. Initial guess (unconstrained)
    initial_guess = [1, 0]
    trigger = data[:-1].copy()
    # 4. Unconstrained optimization (no bounds, no constraints)
    result = minimize(star1_mle, initial_guess, args=(trigger, data,), method='BFGS')
    est_params = result.x
    beta,X,_ = star1_model(est_params, trigger, data)
    transition_fit = 1/logistic_transition_function(trigger, est_params[0], est_params[1])
    X_ = np.insert(X,0,[0,0],axis=0)
    data_fit = X_ @ beta
    
    # visualize the data
    plt.figure(figsize=(10, 5))
    plt.plot(data, label='Simulated Data')
    plt.plot(data_fit, label='fitted Data',color='red')
    plt.plot(transition, label='Transition Variable', linestyle='--')
    plt.title('Simulated STAR1 Model Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    print("Simulation complete.")