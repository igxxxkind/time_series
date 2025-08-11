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


###### STAR from CHATGPT

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

np.random.seed(42)

def G(s, gamma, c):
    return 1 / (1 + np.exp(-gamma * (s - c)))

def simulate_star(T=200, phi1=0.5, phi2=-1, gamma=2, c=1.0, sigma=1.0):
    y = np.zeros(T)
    for t in range(1, T):
        g = G(y[t-1], gamma, c)
        y[t] = (phi1 + (phi2)*g) * y[t-1] + sigma * np.random.randn()
    return y

# Simulate time series
y = simulate_star()

# Negative log-likelihood function
def neg_loglike(params):
    phi1, phi2, gamma, c, sigma = params
    if sigma <= 0:  # avoid invalid likelihood
        return 1e10
    T = len(y)
    resid = np.zeros(T-1)
    for t in range(1, T):
        g = G(y[t-1], gamma, c)
        mu = (phi1 + (phi2 - phi1) * g) * y[t-1]
        resid[t-1] = y[t] - mu
    ll = -0.5 * np.sum(np.log(2 * np.pi * sigma**2) + (resid**2) / (sigma**2))
    return -ll  # minimize negative LL

# Initial guess (arbitrary, no constraints)
init = np.array([0.1, 1.0, 2.0, 0.0, 1.0])

# --- Unconstrained optimization ---
res = minimize(neg_loglike, init, method="BFGS")

phi1_hat, phi2_hat, gamma_hat, c_hat, sigma_hat = res.x

print("True:      ", [0.5, 1.5, 10.0, 0.0, 1.0])
print("Estimated: ", [phi1_hat, phi2_hat, gamma_hat, c_hat, sigma_hat])

# Plotting
plt.plot(y, label='Simulated y')
plt.title('Simulated STAR process')
plt.legend()
plt.show()


