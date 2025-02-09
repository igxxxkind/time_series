import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def simplest_ar_1(par, N):
    """
       A simplest AR process with one root and without a constant.
        Inputs:
        pars - an autoregressive parameter
        N - length of the process
        Returns:
        process - simulated AR process
    """
    noise = np.random.randn(N)
    simulated = noise.copy()
    for item in range(1, N):
        simulated[item] = par*simulated[item-1] + noise[item]
    return simulated


def simple_ar_1(pars, N):
    """
       Simple AR process with one root and a constant.
        Inputs:
        pars: List[float] - constant and an autoregressive parameter
        N - length of the process
        Returns:
        process - simulated AR process
    """
    const = pars.pop(0)
    noise = np.random.randn(N)
    simulated = const + noise.copy()
    for item in range(1, N):
        simulated[item] = const + pars*simulated[item-1] + noise[item]
    return simulated

def simulate_ar(pars, N, sigma = 1):
    """Function to simulate an AR(p) process given the inputs. 
    Returns an array,

   Args:
        pars (List[float]): a parameter list with a constant
        N (int): length of the simulated process
        sigma (int, optional): variance of the noise. Defaults to 1.

    Returns:
        Array: a simulated AR process.
    """
    const = pars.pop(0)
    M = len(pars)
    noise = np.random.randn(N)*abs(sigma)
    simulated = const + noise.copy()
    for item in range(1,N):
        if item < M:
            simulated[item] = const + np.dot(pars[:item], simulated[:item]) + noise[item]
        else:
            simulated[item] = const + np.dot(pars, simulated[(item-M):item]) + noise[item]
    return simulated
    
def simulate_ma(pars, N, sigma=1):
    """Function to simulate an MA(q) process given the inputs. 
    Returns an array,

    Args:
        pars (List[float]): a parameter list with a constant
        N (int): length of the simulated process
        sigma (int, optional): variance of the noise. Defaults to 1.

    Returns:
        Array: a simulated MA process.
    """
    const = pars.pop(0)
    M = len(pars)
    noise = np.random.randn(N)*abs(sigma)
    simulated = const + noise.copy()
    for item in range(1,N):
        if item < M:
            simulated[item] = const + np.dot(pars[:item], noise[:item]) + noise[item]
        else:
            simulated[item] = const + np.dot(pars, noise[(item-M):item]) + noise[item]
    return simulated

