import pandas as pd
import numpy as np
from math import gamma

def gaussian_log_likelihood(params, X, y):
    sigma = abs(params[-1])
    beta = params[:-1]
    fitted = pd.DataFrame(X @ beta)
    residuals = (y - fitted) / sigma
    n = len(residuals)

    llf = -n / 2 * np.log(2 * np.pi * sigma**2) - 0.5 * np.sum(residuals**2)
    return -llf

def t_causal_log_likelihood(params, X, y):
    # Placeholder for centralized t-distribution log likelihood
    # 
    nu = abs(params[-1])
    sigma = abs(params[-2])
    beta = params[:-2]
    fitted = pd.DataFrame(X @ beta)
    residuals = (y - fitted) / sigma
    n = len(residuals)

    part_1 = np.log(gamma((nu+1)/2)) - np.log(gamma(nu/2)) - 0.5*np.log(np.pi*nu*sigma**2)
    part_2 = -(nu+1)/2 * np.log(1 + residuals**2/(nu*sigma**2))
    llf = n * (part_1 + part_2).sum()
    
    return -llf



if __name__ == "__main__":
    pass