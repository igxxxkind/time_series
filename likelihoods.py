from pydantic import BaseModel, Field, model_validator
import pandas as pd
import numpy as np
from math import gamma
from typing import List, Union, Any

class Likelihood(BaseModel):
    params: List[float] = Field(..., description="Model Parameters")
    X: pd.DataFrame  = Field(..., description='Design Matrix')
    y: np.ndarray = Field(..., description="Data Series")
    class Config:
        arbitrary_types_allowed = True
        
    @model_validator(mode='before')
    @classmethod
    def check_params(cls, data: Any) -> Any:
        if len(data['params'])>(data["X"].shape[1]+2):
            raise ValueError('Number of parameters is too large for the available likelihoods or the design matrix is too small')

    def gaussian(self)-> float:
        if len(self.params)>(self.X.shape[1]+1):
            raise ValueError("Too many paramaeters for the Gaussian likelihood")
        sigma = abs(self.params[-1])
        beta = self.params[:-1]
        
        fitted = pd.DataFrame(self.X @ beta)
        residuals = (self.y - fitted) / sigma
        n = len(residuals)

        llf = -n / 2 * np.log(2 * np.pi * sigma**2) - 0.5 * np.sum(residuals**2)
        return -llf

    def t_causal(self):
        # Placeholder for centralized t-distribution log likelihood
        # 
        nu = abs(self.params[-1])
        sigma = abs(self.params[-2])
        beta = self.params[:-2]
        fitted = pd.DataFrame(self.X @ beta)
        residuals = (self.y - fitted) / sigma
        n = len(residuals)

        part_1 = np.log(gamma((nu+1)/2)) - np.log(gamma(nu/2)) - 0.5*np.log(np.pi*nu*sigma**2)
        part_2 = -(nu+1)/2 * np.log(1 + residuals**2/(nu*sigma**2))
        llf = n * (part_1 + part_2).sum()
        
        return -llf



if __name__ == "__main__":
    pass