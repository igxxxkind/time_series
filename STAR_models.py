# This file is a training ground for STAR and LSTAR models

import numpy as np
import pandas as pd
import scipy.stats as stats

#General form of a STAR model follows tre following structure:
# y_t = \phi_1 * y(t) *(1 - G(s_t, \gamma, c)) + \phi_2 * y(t) *G(s_t, \gamma, c) + \epsilon_t
# \phi_1 = phi_1_0 + phi_1_1L + phi_1_2L^2 + ... + phi_1_pL^p
# \phi_2 = phi_2_0 + phi_2_1L + phi_2_2L^2 + ... + phi_2_pL^p
# where G(s_t, \gamma, c) is a smooth transition function, 
# such as logistic : (1 + exp(-\gamma(s_t - C)))^(-1)
# s_t is the trransition variable. Can be exogenous or lagged endogeneous.
# c - threshold parameter switching between two regimes
# \gamma - smoothness parameter reflecting the speed of transition between regimes
# large \gamma - instanteneous change of the regimes

