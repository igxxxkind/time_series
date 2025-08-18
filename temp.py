import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate synthetic data with smooth transition
np.random.seed(42)
n = 300
t = np.arange(1, n+1)

# Define a smooth transition variable
transition = 1 / (1 + np.exp(-0.1 * (t - 150)))

# Generate two different regimes for the time series
y1 = 10 + 0.5 * t[:150] + np.random.normal(0, 1, size=150)
y2 = 20 + 0.1 * t[150:] + np.random.normal(0, 1, size=150)

# Create a smooth transition between regimes
y = np.concatenate([y1, y2 * transition[150:]])

# Plot the synthetic time series with smooth transition
plt.plot(t, y, label='Time Series with Smooth Transition')
plt.xlabel('Time')
plt.ylabel('y')
plt.title('Synthetic Time Series with Smooth Transition')
plt.legend()
plt.show()



import statsmodels.api as sm

# Define the transition variable (lagged value of the time series)
lagged_y = np.roll(y, 1)

# Define a logistic transition function
def logistic_transition(c, gamma, z):
    return 1 / (1 + np.exp(-gamma * (z - c)))

# Define the STAR model
def star_model(params, y, lagged_y, z):
    c, gamma, phi0, phi1, theta0, theta1 = params
    G = logistic_transition(c, gamma, z)
    y_hat = (phi0 + phi1 * lagged_y) + G * (theta0 + theta1 * lagged_y)
    return y_hat

# Fit the model using nonlinear least squares
from scipy.optimize import minimize

# Initial guess for the parameters
initial_params = [150, 0.1, 1, 0.5, 1, 0.5]

# Minimize the sum of squared errors (SSE)
res = minimize(lambda params: np.sum((star_model(params, y, lagged_y, lagged_y) - y) ** 2), initial_params)

# Print the estimated parameters
print("Estimated parameters:", res.x)

# Forecast using the STAR model
y_pred = star_model(res.x, y, lagged_y, lagged_y)

# Plot the original time series and predicted values
plt.plot(t, y, label='Original Time Series')
plt.plot(t, y_pred, '--', label='Fitted STAR Model')
plt.xlabel('Time')
plt.ylabel('y')
plt.title('STAR Model Fit to Time Series')
plt.legend()
plt.show()