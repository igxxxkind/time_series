# This file is a training ground for STAR and LSTAR models
from pydantic import BaseModel, Field, model_validator
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import List, Union, Any, Optional, Dict
from time_series import likelihoods
#General form of a STAR model follows tre following structure:
# y_t = C + \phi_1 * y(t) *(1 - G(s_t, \gamma, c)) + \phi_2 * y(t) *G(s_t, \gamma, c) + \epsilon_t
# \phi_1 = phi_1_0 + phi_1_1L + phi_1_2L^2 + ... + phi_1_pL^p
# \phi_2 = phi_2_0 + phi_2_1L + phi_2_2L^2 + ... + phi_2_pL^p
# where G(s_t, \gamma, c) is a smooth transition function, 
# such as logistic : (1 + exp(-\gamma(s_t - C)))^(-1)           ->LSTAR
# or exponential : (1 + exp(-\gamma(s_t - C))^2)                ->ESTAR
# or second order logistic: (1 + exp(-\gamma(s_t - C_1)(s_t - C_2)))^(-1)    -> SETAR with THREE regimes
# s_t is the trransition variable. Can be exogenous or lagged endogeneous.
# c - threshold parameter switching between two regimes
# \gamma - smoothness parameter reflecting the speed of transition between regimes
# large \gamma - instanteneous change of the regimes

class Transition_function(BaseModel):
    shift: np.ndarray
    gamma: float
    threshold: float
    def logistic(self):
        return 1 / (1 + np.exp(-self.gamma * (self.shift - self.threshold)))
    def exponential(self):
        return 1 / (1 + np.exp(-self.gamma * (self.shift - self.threshold))**2)
    def second_order_logistic(self, threshold2):
        return 1 / (1 + np.exp(-self.gamma * (self.shift - self.threshold)*(self.shift - self.threshold2)))
    class Config:
        arbitrary_types_allowed = True

class Design_matrix_ar(BaseModel):
    y: np.ndarray
    n_params: int = Field(..., description="Number of AR parameters")
    const: bool = Field(..., description="Include constant term in the design matrix")
    
    class Config:
        arbitrary_types_allowed = True
    
    def create(self) -> pd.DataFrame:
        X = pd.DataFrame(self.y, columns=['y'])
        if not self.const:
            for item in range(self.n_params):
                X[f'lag_{item+1}'] = np.roll(self.y, item+1)
            X=X.iloc[len(self.n_params):,].drop('y',axis=1)
        else:
            X['const']=1
            for item in range(self.n_params-1):
                X[f'lag_{item+1}'] = np.roll(self.y, item+1)
            X=X.iloc[(self.n_params-1):,].drop('y',axis=1)

        return X

    
class STARmodel(BaseModel):
    params: List[float] = Field(..., description="Model Parameters")
    shift: np.ndarray = Field(..., description="Transition data")
    data: np.ndarray = Field(..., description="Actual data")
    
    class Config:
        arbitrary_types_allowed = True

    ll_parameters=params[:-2]
    
    def fit(self, type: str = "logistic", noise: str='gaussian', const=True) -> Dict[Any, Any]:
        transition_params={"shift": self.shift,
                           "gamma": self.params[-2],
                           "threshold": self.params[-1]}
        
        trigger = getattr(Transition_function(**transition_params), type)()
        # tail section of params vector contains: [..., sigma, gamma, threshold]]
        X = Design_matrix_ar(y=self.data, n_params=len(self.params)-3, const=const).create()
        temp = len(trigger) - X.shape[0]
        X_g = X.copy().mul(trigger[temp:], axis='rows').drop('const',axis=1)
        X_g.columns = [f'{col}_g' for col in X_g.columns]
        X_full = X.join(X_g)
        init_params = self.params[:-2]
        likelihood = likelihoods.Likelihood(params=init_params, X=X_full, y=self.data)
        if noise != 'gaissian':
            pass
        
        # to be finalized
        result = minimize(likelihood.gaussian, init_params, method='Nelder-Mead')
        
    def simulate(self, type: str = "logistic", noise: str='normal'):
        if noise == "normal":
            noise = stats.norm.rvs(size=len(self.data))
        else:
            ValueError("Unsupported noise distribution")
        transition_params={"shift": self.shift,
                           "gamma": self.gamma,
                           "threshold": self.threshold}
        pass       
        
        
def star1_model(parameters, trigger, data):
    """A simple STAR1 model with the logistic transition function.
    The idea is to isolate gamma and c parameters from the model, leaving
    the remaining parameters linear.

    Args:
        phi1 (_type_): Normal state AR parameter
        phi2 (_type_): Transitioned state AR parameter
        c (_type_): threshold parameter
        gamma (_type_): smoothness parameter
        trigger (_type_): transition variable (exogeneous or lagged endogeneous)
        data (_type_): data
    """
    gamma, c = parameters
    
    transition = logistic_transition_function(trigger, gamma, c)
    
    # y = phi1 y_t-1 + phi2 y_t-1 * transition + epsilon_t
    
    # params = np.array([phi1, phi2])
    X = np.column_stack([(data*(1-transition))[:-1], (data*transition)[:-1]])
    
    beta = np.linalg.inv(X.T @ X)@X.T @ data[1:]
    y_fit = X @ beta
    
    residuals = data[1:] - y_fit
    # sigma = np.std(residuals)
    
    
    return beta, X, residuals

def star1_mle(parameters, trigger, data):
    
    _, _, residuals = star1_model(parameters, trigger, data)
    
    # sigma = np.std(residuals)
    # # residuals /= sigma
    # n = len(residuals)
    # llf = -n / 2 * np.log(2 * np.pi * sigma**2) - 0.5 * np.sum(residuals**2)
          
    return sum(residuals**2)





if __name__ == "__main__":
    
    
    transition_params= {"shift": np.arange(0,20), "gamma": 10, "threshold":9}
    a=transition_function(**params)

    # simulation
    np.random.seed(42)
    phi1 = 0.4
    phi2 = -0.4
    
    noise = stats.t.rvs(df=30, loc = 0, scale=1, size = 200)
    data = noise.copy()
    
    shift = np.arange(0,200,1)
    transition = logistic_transition_function(shift, 50, 100)
    
    for i in range(1, len(data)):
        data[i] = phi1 * data[i-1] * (1-transition[i]) + phi2*data[i-1]* transition[i] + noise[i]
    
    

    # 3. Initial guess (unconstrained)
    initial_guess = [0.1, 100]
    trigger = shift.copy()
    # 4. Unconstrained optimization (no bounds, no constraints)
    result = minimize(star1_mle, initial_guess, args=(trigger, data,), method='BFGS')
    print(result.fun)
    # multimodal optimization problem that requires a grtid of starting values to find the global minimum
    est_params = result.x
    beta,X,_ = star1_model(est_params, trigger, data)
    # transition_fit = 1/logistic_transition_function(trigger, est_params[0], est_params[1])
    X_ = np.insert(X,0,[0,0],axis=0)
    data_fit = X_ @ beta
    # 5. Iterate over a grid of starting values
    x=np.arange(1,101)
    y=np.arange(1,200,2)
    iterations= pd.DataFrame(columns=["X", "Y", "Z"])
    iterations.X=np.repeat(x,100)
    iterations.Y=np.tile(y,100)
    for i in range(0,10000):
        initial_guess = [iterations.iloc[i,0], iterations.iloc[i,1]]
        try:
            result = minimize(star1_mle, initial_guess, args=(trigger, data,), method='Nelder-Mead')
            iterations.iloc[i,2] = result.fun.copy()
        except:
            iterations.iloc[i,2] = np.nan
            continue    
    
    
    # visualize the data
    plt.figure(figsize=(10, 5))
    plt.plot(data, label='Simulated Data')
    plt.plot(data_fit, label='fitted Data',color='red')
    plt.plot(transition, label='Transition Variable', linestyle='--')
    plt.title('Simulated STAR1 Model Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    print("Simulation complete.")
    
    
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Pivot to wide format
    pivot = iterations.pivot(index="Y", columns="X", values="Z")

    X, Y = np.meshgrid(pivot.columns, pivot.index)
    Z = pivot.values

    # Plot
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis")

    fig.colorbar(surf)
    plt.show()
    
    
    plt.figure(figsize=(10, 5))
    plt.plot(pivot.loc[:,50].dropna(), label='Simulated Data')
    # plt.plot(data_fit, label='fitted Data',color='red')
    # plt.plot(transition, label='Transition Variable', linestyle='--')
    plt.title('Simulated STAR1 Model Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    print("Simulation complete.")
    
    
    
    
    