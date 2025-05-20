import pandas as pd
import numpy as np

def gaussian_log_likelihood(params, X, y):
    sigma = abs(params[-1])
    beta = params[:-1]
    fitted = pd.DataFrame(X @ beta)
    residuals = (y - fitted) / sigma
    n = len(residuals)

    llf = -n / 2 * np.log(2 * np.pi * sigma**2) - 0.5 * np.sum(residuals**2)
    return -llf
        
if __name__ == "__main__":
    pass