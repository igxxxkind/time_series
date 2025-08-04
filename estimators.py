import sklearn as sk
import numpy as np
import scipy.optimize as solver
from pydantic import BaseModel, model_validator, Field
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from TS_simulation import modelParameters
from time_series.likelihoods import gaussian_log_likelihood
from matplotlib import pyplot as plt

class Estimators():
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
        
    


    
# if __name__ == "__main__":
    
    
    