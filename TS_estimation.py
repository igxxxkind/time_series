import sklearn as sk
import numpy as np
import scipy.optimize as solver
from pydantic import BaseModel, model_validator, Field
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from TS_simulation import modelParameters
from likelihoods import gaussian_log_likelihood
from matplotlib import pyplot as plt

class Estimate:
    def __init__(self, endog: pd.DataFrame, exog: pd.DataFrame):
        self.endog = endog
        self.exog = exog

    def OLS(self, const: bool = True):
        """
        Ordinary Least Squares (OLS) estimation method.

        This method performs OLS regression to estimate the coefficients of the linear model
        using the endogenous and exogenous variables provided.

        Args:
            const (bool): If True, includes a constant term in the regression model. Default is True.

        Returns:
            tuple: A tuple containing the following elements:
                - beta (pd.DataFrame): Estimated coefficients of the regression model.
                - beta_se (pd.DataFrame): Standard errors of the estimated coefficients.
                - fitted (pd.DataFrame): Fitted values of the endogenous variable.
                - residuals (pd.DataFrame): Residuals of the regression model.
        """
        if const:
            X = np.column_stack((np.ones(len(self.endog)), self.exog))
        else:
            X = self.exog
        beta = np.linalg.inv(X.T @ X) @ X.T @ self.endog

        fitted = X @ beta
        residuals = self.endog - fitted

        sigma_hat = np.sqrt(
            residuals.T @ residuals / (len(self.endog) - len(beta))
        ).values[0]
        cov_matrix = np.linalg.inv(X.T @ X) * sigma_hat**2
        beta_se = pd.DataFrame(np.sqrt(cov_matrix.diagonal()))

        return {
            "beta": beta,
            "beta_se": beta_se,
            "fitted": fitted,
            "residuals": residuals,
        }

    def MLE(self, const: bool = True, gaussian=True) -> dict:
        """
        Gaussian Maximum Likelihood Estimation (MLE) method.

        Args:
            const (bool, optional): boolean to indicate a presence of a constant. Defaults to True.
            gaussian (bool, optional): indicate if the gaussian MLE should be used. Defaults to True

        Returns:
            Dictionary: contains parameter estimations, standard errors, fitted values, and residuals.
        """
        if  gaussian:
            loglikelihood = gaussian_log_likelihood
        else:
            pass
        if const:
            X = np.column_stack((np.ones(len(self.endog)), self.exog))
        else:
            X = self.exog
        y = self.endog
        params = (
            np.ones(X.shape[1] + 1) * 0.7
        )  # parameters bor beta and sigma within the unit circle
        result = solver.minimize(
            fun=loglikelihood,
            x0=params,
            args=(X, y),
            method="BFGS",
            options={"maxiter": 10000, "disp": True},
        )
        fitted = pd.DataFrame(X @ result.x[:-1])
        residuals = y - fitted
        return {
            "beta": result.x,
            "beta_se": np.sqrt(np.diag(result.hess_inv)),
            "fitted": fitted,
            "residuals": residuals,
        }
        
    
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
        self.Q = np.ones(self.n)
        self.R = np.ones(self.n)
        self.P = np.ones(self.n) * np.var(self.y)
        self.Phat = np.ones(self.n)
        self.F = np.ones(self.n)
        
        self.ahat[0] = self.mu[0]
        self.Phat[0] = self.P[0]
        self.Z = 1
        self.T = 1
        self.d = 0.0
        self.c = 0.0

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
        self.a[t] = self.T*self.a[t-1] + self.c
        self.P[t] = self.T*self.P[t-1]*self.T + self.R[t] * self.Q[t] * self.R[t] 
    
    def update_step(self, t):
        # updating equations - Third step of the Filter
        self.F[t] = self.Z * self.P[t] * self.Z + self.H[t]
        self.mu[t] = self.a[t] + self.P[t] * self.Z / self.F[t] * (self.y[t] - self.Z * self.a[t] - self.d)
        self.P[t] = self.P[t] - self.P[t] * self.Z / self.F[t] * self.Z * self.P[t]
        self.a[t] = self.mu[t] + self.P[t] * (self.y[t] - self.Z * self.mu[t] - self.d) / self.F[t]
        
    def forecast_step(self, t):
        # forecasting step  - Second step of the Filter
        K = self.T * self.P[t] * self.Z / self.F[t]
        self.ahat[2] = (self.T - K*self.Z)*self.a[t] + K*self.y[t] + (self.c - K * self.d)
        self.Phat[2]  = self.T * (self.P[t] - self.P[t] * self.Z / self.F[t] * self.Z * self.P[t]) * self.T +  self.R[t] * self.Q[t] * self.R[t]
        
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
    

    
# if __name__ == "__main__":
    
    
    