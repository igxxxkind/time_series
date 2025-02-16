import sklearn as sk
import numpy as np
import scipy.optimize as solver
from pydantic import BaseModel, model_validator, Field
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from .TS_estimation import modelParameters


class estimate():
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
        
        sigma_hat = np.sqrt(residuals.T @ residuals / (len(self.endog) - len(beta))).values[0]
        cov_matrix = np.linalg.inv(X.T @ X) * sigma_hat**2
        beta_se = pd.DataFrame(np.sqrt(cov_matrix.diagonal()))
        return beta, beta_se, fitted, residuals
    
    def gaussianMLE(self, const: bool = True):
        
        if const:
            X = np.column_stack((np.ones(len(self.endog)), self.exog))
        else:
            X = self.exog
        y = self.endog
        params = np.ones(X.shape[1]+1)
        
        def log_likelihood(params, X, y):
            sigma = params[-1]
            beta = pd.DataFrame(params[:-1])
            fitted = X @ beta
            residuals = y - fitted
            
            info = -0.5 * residuals.T @ residuals / sigma**2
            llf = - np.log(sigma) - info
            return -llf
        
        solver.minimize(log_likelihood, params, args=(X, endog), method='Nelder-Mead', options = {'maxiter':10000})
            
            
        
        



class estimateTSM(BaseModel):
    
    parameters: modelParameters
    data: np.ndarray
    
    class Config:
        arbitrary_types_allowed = True
    
    