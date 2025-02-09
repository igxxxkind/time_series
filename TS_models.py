import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel


class modelParameters(BaseModel):
    phi: Optional[List[float]] = 0
    psi: Optional[List[float]] = 0
    const: Optional[float] = 0
    sigma: Optional[float] = 1


class selectedTSM(BaseModel):
    
    parameters: modelParameters
    length: int

    def AR1(self):
        if len(self.parameters.phi)>1:
            raise TypeError('Cannot build simple model with complicated parameters')
        phi = self.parameters.phi
        const = self.parameters.const
        sigma = abs(self.parameters.sigma)
        noise = np.random.random(self.length) * sigma
        simulated = noise.copy()
        for item in range(1,self.length):
            simulated[item] = const + phi*simulated[item-1] + noise[item]
        return simulated
    
    def ARp(self):
        if not isinstance(self.parameters.phi, list):
            raise TypeError('Cannot build complicated model with simple parameters')
        phi = self.parameters.phi
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
        phi = self.parameters.phi
        psi = self.parameters.psi
        const = self.parameters.const
        sigma = self.parameters.sigma
        if not isinstance(psi, list):
            psi = list(psi)
        M = len(psi)
        noise = np.random.randn(self.length)*abs(sigma)
        simulated = const + noise.copy()
        for item in range(1,self.length):
            if item < M:
                simulated[item] = const + np.dot(psi[:item], noise[:item]) + noise[item]
            else:
                simulated[item] = const + np.dot(psi, noise[(item-M):item]) + noise[item]
        return simulated
    
    def ARMA(self):
        phi = self.parameters.phi
        psi = self.parameters.psi
        const = self.parameters.const
        sigma = self.parameters.sigma
        if not isinstance(psi, list):
            psi = list(psi)
        if not isinstance(phi, list):
            phi = list(phi)
        P = len(phi)   
        Q = len(psi)
        M = min(P,Q)
        Mp = P-Q if P>Q else 0 # if number of AR parameters is greater than number of MA parameters
        Mq = Q-P if Q>P else 0 # if number of MA parameters is greater than number of AR parameters

        noise = np.random.randn(self.length)*abs(sigma)
        simulated = const + noise.copy()
        for item in range(1,self.length):
            if item < M:
                simulated[item] = const + np.dot(phi[:item], simulated[:item]) + np.dot(psi[:item], noise[:item]) + noise[item]
            elif Mp > 0 and item <(Mp+M):
                simulated[item] = const + np.dot(phi[:item], simulated[:item]) + np.dot(psi, noise[(item-Q):item]) + noise[item]
            elif Mq > 0 and item <(Mq+M):
                simulated[item] = const + np.dot(phi, simulated[(item-P):item]) + np.dot(psi[:item], noise[:item]) + noise[item]
            else:
                simulated[item] = const + np.dot(phi, simulated[(item-P):item]) + np.dot(psi, noise[(item-Q):item]) + noise[item]
        return simulated




parameters_ar1 = {'phi': 0.5, 'psi': None, 'const': 0, 'sigma': 1}
parameters_arp = {'phi': [0.5,0.3,-0.4], 'psi': None, 'const': 0, 'sigma': 1}
parameters_maq = {'phi': None, 'psi': [0.5,0.3,-0.4], 'const': 0, 'sigma': 1}
parameters_arma = {'phi': [-0.5,0.4,-0.3], 'psi': [0.5,0.3,-0.4], 'const': 0, 'sigma': 1}
arma_pq = modelParameters(**parameters_arma)

    
inputs = {'parameters': parameters_arma, 'length': 20}
selectedTSM(**inputs).ARp()
selectedTSM(**inputs).ARMA()
selectedTSM(**inputs).MAq()
