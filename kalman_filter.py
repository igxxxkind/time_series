import sklearn as sk
import pandas as pd
import numpy as np
import scipy.optimize as solver
from pydantic import BaseModel, model_validator, Field
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union


class KalmanFilter():
    """Simple Kalman Filter for time_invariant RW processes.
    This class implements a simple Kalman filter for time-invariant random walk processes.
    y_t = Z_t * \alpha_t + d_t + \epsilon_t \\ state equation
    \alpha_t = T_t * \alpha_{t-1} + c_t + R_t * \eta_t \\transition equation
    """
    
    def __init__(self, data: pd.Series):
        self.data = data
        self.n = len(data)
        self.a = np.zeros(self.n)
        self.mu = np.zeros(self.n)
        self.y = self.data.values
        self.ahat = np.zeros(self.n)
        
        self.H = np.ones(self.n)
        self.d = np.zeros(self.n)
        self.c = np.zeros(self.n)
        self.P = np.ones(self.n) * np.var(self.y)
        self.Phat = np.ones(self.n)
        self.F = np.ones(self.n)
        
        self.ahat[0] = self.mu[0]
        self.Phat[0] = self.P[0]
        self.Q = 1
        self.Z = 1
        self.T = 1
        self.R = 1
        

    def prediction_step(self, t):
        """y_t = mu_t + e_t
        mu_t = mu_{t-1} + w_t
        
        e_t ~ N(0, sigma_e)
        w_t ~ N(0, sigma_w)
        
        alpha_t = mu_t
        Starting positions are: mu_0 = 0, sigma_e = 1, sigma_w = 1
        a_0 = 0, P[0] = np.var(self.y)
    """

        # prediction equations - First step of the Filter
        self.a[t] = self.T*self.a[t-1] + self.c[t]
        self.P[t] = self.T*self.P[t-1]*self.T + self.R * self.Q * self.R 
    
    def update_step(self, t):
        # updating equations - Third step of the Filter
        self.F[t] = self.Z * self.P[t] * self.Z + self.H[t]
        self.mu[t] = self.a[t] + self.P[t] * self.Z / self.F[t] * (self.y[t] - self.Z * self.a[t] - self.d[t])
        self.P[t] = self.P[t] - self.P[t] * self.Z / self.F[t] * self.Z * self.P[t]
        self.a[t] = self.mu[t] + self.P[t] * (self.y[t] - self.Z * self.mu[t] - self.d[t]) / self.F[t]
        
    def forecast_step(self, t):
        # forecasting step  - Second step of the Filter
        K = self.T * self.P[t] * self.Z / self.F[t]
        self.ahat[2] = (self.T - K*self.Z)*self.a[t] + K*self.y[t] + (self.c[t] - K * self.d[t])
        self.Phat[2]  = self.T * (self.P[t] - self.P[t] * self.Z / self.F[t] * self.Z * self.P[t]) * self.T +  self.R * self.Q * self.R
        
    def fit(self):
        for t in range(1, self.n):
            self.prediction_step(t)
            self.update_step(t)
            self.forecast_step(t)
        return self.ahat, self.Phat, self.mu, self.P 
    
    def plot(self):
        self.fit()
        plt.plot(self.data, label="Data", alpha=0.25, color="black")
        plt.plot(self.mu, label="Filtered")
        # plt.plot(self.mu, label="Smoothed", alpha=0.15)
        plt.legend()
        plt.show()
        plt.close()
    