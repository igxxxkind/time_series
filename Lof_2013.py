# this script simulates the results from Lof(2013) paper

import numpy as np
import pandas as pd

import scipy.stats as stats

# we start with a model for omitted variables
# 
#

eta_x = stats.t.rvs(df=3, loc = 1, scale=1, size = 200)
eta_y = stats.t.rvs(df=3, loc = 1, scale=1, size = 200)
residuals = np.array([eta_x, eta_y]).T

A = np.array([[0.8,10],[0,0.8]])

Z = np.ndarray(shape = residuals.shape, dtype=float)
Z[0] = residuals[0]
for i in range(1,residuals.shape[0]):
    Z[i] = A @ Z[i-1] + residuals[i]


