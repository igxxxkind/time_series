from time_series.estimators import Estimate
from time_series.likelihoods import gaussian_log_likelihood
from time_series.TS_simulation import modelParameters, simulateTSM
if __name__ == "__main__":
# Example usage
    parameters_ar1 = {"phi": 0.5, "theta": None, "const": 0, "sigma": 1, "ksi": None}
    parameters_arp = {
        "phi": [0.5, 0.3, -0.4],
        "theta": None,
        "const": 0,
        "sigma": 1,
        "ksi": None,
    }
    parameters_maq = {
        "phi": None,
        "theta": [0.5, 0.3, -0.4],
        "const": 0,
        "sigma": 1,
        "ksi": None,
    }
    parameters_arma = {
        "phi": [-0.5, 0.4, -0.3],
        "theta": [0.5, 0.3, -0.4],
        "const": 0,
        "sigma": 1,
        "ksi": None,
    }
    parameters_ma_rw = {
        "phi": None,
        "theta": [0.5, 0.3, -0.4],
        "const": 0,
        "sigma": 1,
        "ksi": None,
    }

    parameters_ar1 = {"phi": 0.5, "theta": None, "const": 0, "sigma": 1, "ksi": None}
    parameters_arp = {
        "phi": [0.5, 0.3, -0.4],
        "theta": None,
        "const": 0,
        "sigma": 1,
        "ksi": None,
    }
    parameters_maq = {
        "phi": None,
        "theta": [0.5, 0.3, -0.4],
        "const": 0,
        "sigma": 1,
        "ksi": None,
    }
    parameters_arma = {
        "phi": [-0.5, 0.4, -0.3],
        "theta": [0.5, 0.3, -0.4],
        "const": 0,
        "sigma": 1,
        "ksi": None,
    }
    parameters_ma_rw = {
        "phi": None,
        "theta": [0.5, 0.3, -0.4],
        "const": 0,
        "sigma": 1,
        "ksi": None,
    }

    parameters_arX = {"phi": 0.5, "ksi": 1, "const": 0, "sigma": 1, "theta": 0.2}

    modelParameters(**parameters_arX)


    inputs_maq = {"parameters": parameters_maq, "length": 20}
    inputs_rw = {"parameters": parameters_ma_rw, "length": 20}
    inputs_ar = {"parameters": parameters_ar1, "length": 20}

    test = simulateTSM(**inputs_rw).RW()
    simulateTSM(**inputs_maq).RW()
    simulateTSM(**inputs_ar).ARp()


    aux_x = pd.Series(np.random.randn(20))
    aux_x2 = pd.Series(np.random.randn(20))
    aux_2x = pd.concat([aux_x, aux_x2], axis=1)

    parameters_arx = {"phi": 0.5, "ksi": [1, 2], "const": None, "sigma": 1, "theta": 0.2}
    inputs_arx = {"parameters": parameters_arx, "length": 20}


    parameters_ar1 = {"phi": 0.2, "theta": None, "const": 1, "sigma": 1, "ksi": None}
    inputs_ar = {"parameters": parameters_ar1, "length": 200}
    test = simulateTSM(**inputs_ar).ARp()

    Estimate(endog=pd.DataFrame(test[1:]), exog=pd.DataFrame(test[:-1])).OLS(const=True)[
        "beta"
    ]

    Estimate(endog=pd.DataFrame(test[1:]), exog=pd.DataFrame(test[:-1])).MLE(
        const=True
    )["beta"]

    res.values[1][0]


    # simulateTSM(**inputs).AR1()
    # simulateTSM(**inputs).ARMA()
    # simulateTSM(**inputs).MAq()

    sm.tsa.AutoReg(test, lags=1, trend="c").fit().summary()

    sm.tsa.AutoReg(test, lags=1, trend="n").fit().summary()
