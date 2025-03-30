import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, model_validator


class modelParameters(BaseModel):
    """
    Model parameters for time series models.
    Attributes:
        phi (Optional[List[float]]): Coefficients for the AR part of the model. Default is [0].
        theta (Optional[List[float]]): Coefficients for the MA part of the model. Default is [0].
        const (Optional[float]): Constant term in the model. Default is 0.
        sigma (Optional[float]): Standard deviation of the error term. Default is 1.
    Methods:
        check_types(cls, data: Any) -> Any:
            Validates and converts the types of 'phi' and 'theta' attributes if they are provided as floats.
    """
    phi: Optional[List[Union[float, int]]] = [0]
    theta: Optional[List[Union[float, int]]] = [0]
    const: Optional[Union[float, int]] = 0
    sigma: Optional[Union[float, int]] = 1
    ksi: Optional[List[Union[float, int]]] = [0] # for ARX model
    
    @model_validator(mode='before')
    @classmethod
    def check_types(cls, data: Any) -> Any:
        if isinstance(data['phi'], (float, int)):
            data['phi'] = [data['phi']]
        if isinstance(data['theta'], (float, int)):
            data['theta'] = [data['theta']]
        if isinstance(data['ksi'], (float, int)):
            data['ksi'] = [data['ksi']]
        return data
        

class simulateTSM(BaseModel):
    
    parameters: modelParameters
    length: int

    def AR1(self):
        """
        Simulates a univariate AR(1) time series model.

        This method generates a time series based on the AR(1) model using the parameters
        provided in the `modelParameters` instance.

        Raises:
            TypeError: If more than one phi parameter is provided.

        Returns:
            np.ndarray: Simulated AR(1) time series data.
        """ 
        if len(self.parameters.phi)>1:
            raise TypeError('Cannot build simple model with complicated parameters')
        phi = self.parameters.phi
        if isinstance(phi, list):
            phi = phi[0]
        if self.parameters.const is None:
            const = 0
        else:
            const = self.parameters.const
        sigma = abs(self.parameters.sigma)
        noise = np.random.random(self.length) * sigma
        simulated = noise.copy()
        for item in range(1,self.length):
            simulated[item] = const + phi*simulated[item-1] + noise[item]
        return simulated
    
    def ARp(self):
        """
        Simulates a univariate AR(p) time series model.

        This method generates a time series based on the AR(p) model using the parameters provided in the `modelParameters` instance.

        Raises:
            TypeError: If the phi parameter is not a list of floats or ints.

        Returns:
            np.ndarray: Simulated AR(p) time series data.
        """
        if not isinstance(self.parameters.phi, list):
            raise TypeError('Cannot build complicated model with simple parameters')
        phi = self.parameters.phi
        if self.parameters.const is None:
            const = 0
        else:
            const = self.parameters.const
        sigma = abs(self.parameters.sigma)
        M = len(phi)
        noise = np.random.random(self.length) * sigma
        simulated = noise.copy()
        for item in range(1,self.length):
            if item < M:
                simulated[item] = const + np.dot(phi[:item], simulated[:item]) + noise[item]
            else:
                simulated[item] = const + np.dot(phi, simulated[(item-M):item]) + noise[item]
        return simulated
        
    def MAq(self):
        """
        Simulates a univariate MA(q) time series model.

        This method generates a time series based on the MA(q) model using the parameters
        provided in the `modelParameters` instance.

        Raises:
            TypeError: If the theta parameter is not a list of floats or ints.

        Returns:
            np.ndarray: Simulated MA(q) time series data.
        """
        phi = self.parameters.phi
        theta = self.parameters.theta
        if self.parameters.const is None:
            const = 0
        else:
            const = self.parameters.const
        sigma = self.parameters.sigma
        if not isinstance(theta, list):
            theta = list(theta)
        M = len(theta)
        noise = np.random.randn(self.length)*abs(sigma)
        simulated = const + noise.copy()
        for item in range(1,self.length):
            if item < M:
                simulated[item] = const + np.dot(theta[:item], noise[:item]) + noise[item]
            else:
                simulated[item] = const + np.dot(theta, noise[(item-M):item]) + noise[item]
        return simulated
    
    def ARMA(self):
        """
        Simulates a univariate ARMA(p, q) time series model.

        This method generates a time series based on the ARMA(p, q) model using the parameters
        provided in the `modelParameters` instance.

        Raises:
            ValueError: If either the phi or theta parameters are not provided.

        Returns:
            np.ndarray: Simulated ARMA(p, q) time series data.
        """
        phi = self.parameters.phi
        theta = self.parameters.theta
        if self.parameters.const is None:
            const = 0
        else:
            const = self.parameters.const
        sigma = self.parameters.sigma
        if not isinstance(theta, list):
            theta = list(theta)
        if not isinstance(phi, list):
            phi = list(phi)
        if (phi is None) or (theta is None):
            raise ValueError('Both AR and MA parameters should be provided')
        P = len(phi)   
        Q = len(theta)
        M = min(P,Q)
        Mp = P-Q if P>Q else 0 # if number of AR parameters is greater than number of MA parameters
        Mq = Q-P if Q>P else 0 # if number of MA parameters is greater than number of AR parameters

        noise = np.random.randn(self.length)*abs(sigma)
        simulated = const + noise.copy()
        for item in range(1,self.length):
            if item < M:
                simulated[item] = const + np.dot(phi[:item], simulated[:item]) + np.dot(theta[:item], noise[:item]) + noise[item]
            elif Mp > 0 and item <(Mp+M):
                simulated[item] = const + np.dot(phi[:item], simulated[:item]) + np.dot(theta, noise[(item-Q):item]) + noise[item]
            elif Mq > 0 and item <(Mq+M):
                simulated[item] = const + np.dot(phi, simulated[(item-P):item]) + np.dot(theta[:item], noise[:item]) + noise[item]
            else:
                simulated[item] = const + np.dot(phi, simulated[(item-P):item]) + np.dot(theta, noise[(item-Q):item]) + noise[item]
        return simulated

    def ARpX(self, exog: pd.DataFrame):
        """
        Simulates a univariate AR(p) time series model with exogenous control variables (ARpX).

        This method generates a time series based on the AR(p) model with exogenous variables
        using the parameters provided in the `modelParameters` instance.

        Args:
            exog (np.array): Exogenous variables to be included in the model. The array should have
                            the same number of columns as the number of exogenous parameters (ksi).

        Raises:
            ValueError: If the number of columns in the exogenous variables array does not match
                        the number of exogenous parameters.

        Returns:
            np.ndarray: Simulated AR(p) time series data with exogenous variables.
        """
        if not isinstance(exog, pd.DataFrame):
            exog = pd.DataFrame(exog)
        ksi = self.parameters.ksi
        phi = self.parameters.phi
        if self.parameters.const is None:
            const = 0
        else:
            const = self.parameters.const
        sigma = abs(self.parameters.sigma)
        M = len(phi)
        if exog.shape[1] != len(ksi):
            raise ValueError('Exogenous variables should have the same number of columns as the number of exogenous parameters')
        noise = np.random.random(self.length) * sigma
        simulated = noise.copy()
        external = np.dot(ksi, exog.T) # because external part does not change with time
        for item in range(1,self.length):
            if item < M:
                simulated[item] = const + np.dot(phi[:item], simulated[:item]) + external[item] + noise[item]
            else:
                simulated[item] = const + np.dot(phi, simulated[(item-M):item]) + external[item] + noise[item]
        return simulated

    def RW(self, drift: bool = False):
        """
        Simulates a random walk (RW) time series model.

        This method generates a time series based on the random walk model. Optionally, a drift term
        can be included in the model.

        Args:
            drift (bool): If True, includes a drift term in the random walk model. Default is False.

        Returns:
            np.ndarray: Simulated random walk time series data.
        """
        self.parameters.theta = [1]
        if not drift:
            self.parameters.const = 0
        simulated = self.MAq()
        return simulated

parameters_ar1 = {'phi': 0.5, 'theta': None, 'const': 0, 'sigma': 1, 'ksi': None}
parameters_arp = {'phi': [0.5,0.3,-0.4], 'theta': None, 'const': 0, 'sigma': 1, 'ksi': None}
parameters_maq = {'phi': None, 'theta': [0.5,0.3,-0.4], 'const': 0, 'sigma': 1, 'ksi': None}
parameters_arma = {'phi': [-0.5,0.4,-0.3], 'theta': [0.5,0.3,-0.4], 'const': 0, 'sigma': 1, 'ksi': None}
parameters_ma_rw = {'phi': None, 'theta': [0.5,0.3,-0.4], 'const': 0, 'sigma': 1, 'ksi': None}

parameters_arX = {'phi': 0.5, 'ksi': 1, 'const': 0, 'sigma': 1, 'theta': 0.2}

modelParameters(**parameters_arX) 


inputs_maq = {'parameters': parameters_maq, 'length': 20}    
inputs_rw = {'parameters': parameters_ma_rw, 'length': 20}
inputs_ar = {'parameters': parameters_ar1, 'length': 20}

simulateTSM(**inputs_rw).RW()
simulateTSM(**inputs_maq).RW()
simulateTSM(**inputs_ar).ARp()

    
aux_x = pd.Series(np.random.randn(20))
aux_x2 = pd.Series(np.random.randn(20))
aux_2x = pd.concat([aux_x, aux_x2], axis = 1)

parameters_arx = {'phi': 0.5, 'ksi': [1,2], 'const': None, 'sigma': 1, 'theta': 0.2}
inputs_arx = {'parameters': parameters_arx, 'length': 20}



test = simulateTSM(**inputs_ar).ARp()

Estimate(endog = pd.DataFrame(test[1:]), exog = pd.DataFrame(test[:-1])).OLS(const = False)


# simulateTSM(**inputs).AR1()
# simulateTSM(**inputs).ARMA()
# simulateTSM(**inputs).MAq()

