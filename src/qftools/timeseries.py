"""
Module for time series analysis
"""

import numpy as np


class LinearRegression:
    """
    Class for linear regression
    """

    def __init__(self, X, y, add_constant=True):
        if add_constant:
            self.X = np.column_stack((np.ones(len(X)), X))
        elif X.ndim == 1:
            self.X = np.array(X).reshape(-1, 1)
        else:
            self.X = X
        self.y = y
        self.coeff = None
        self.residuals = None
        self.regression_se = None
        self.coeff_est = None
        self.coeff_se = None

    def fit(self, model="OLS"):
        if model == "OLS":
            self.coeff = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
            self.residuals = self.y - self.X @ self.coeff
            self.regression_se = np.sqrt(
                np.sum(self.residuals**2) / (len(self.y) - len(self.X.T))
            )
        else:
            # TODO: other models
            raise ValueError("Model not implemented")
        return self

    def tstatistics(self, r=0, estimator="default"):
        if estimator == "default":
            self.coeff_est = estimator
            self.coeff_se = np.sqrt(
                (self.regression_se**2) * np.diag(np.linalg.inv(self.X.T @ self.X))
            )
            t_stat = (self.coeff - r) / self.coeff_se
            return t_stat
        elif estimator == "robust":
            # TODO : robust estiamtor
            raise ValueError("Model not implemented")

    def AIC(self):
        log_likelihood = (-len(self.y) / 2) * (
            np.log(self.regression_se**2) + np.log(2 * np.pi)
        ) - (self.residuals @ self.residuals.T / (2 * (self.regression_se**2)))
        return 2 * len(self.X.T) - 2 * log_likelihood

    def BIC(self):
        log_likelihood = (-len(self.y) / 2) * (
            np.log(self.regression_se**2) + np.log(2 * np.pi)
        ) - (self.residuals @ self.residuals.T / (2 * (self.regression_se**2)))
        return len(self.X.T) * np.log(len(self.y)) - 2 * log_likelihood

    def __str__(self):
        # TODO str
        s = ""
        return s


class AugmentedDickeyFuller:
    """
    Class for Augmented Dickey-Fuller test
    """

    def __init__(self, y, intercept=True, trend=False, maxlag=0, autolag="AIC"):
        self.y = y
        self.intercept = intercept
        self.trend = trend
        self.maxlag = maxlag
        self.autolag = autolag
        self.coeff = None

    def fit(self):
        dependent = self.y.diff()[1:]
        independent = self.y.shift(1)[1:]
        if self.intercept:
            independent = np.column_stack((independent, np.ones(len(independent))))
        if self.trend:
            independent = np.column_stack(
                (independent, np.arange(1, len(independent) + 1))
            )
        if self.autolag is None:
            if self.maxlag != 0:
                lags = np.column_stack(
                    [dependent.shift(i) for i in range(1, self.maxlag + 1)]
                )
                independent = np.column_stack((independent, lags))
                independent = independent[self.maxlag :]
                dependent = dependent[self.maxlag :]
        elif self.autolag == "AIC":
            if self.maxlag != 0:
                AICs = []
                independent_aic = independent.copy()
                dependent_aic = dependent.copy()
                regression = LinearRegression(
                    independent_aic, dependent_aic, add_constant=False
                )
                regression.fit()
                AICs.append(regression.AIC())
                lags = np.column_stack(
                    [dependent.shift(i) for i in range(1, self.maxlag + 1)]
                )
                for i in range(1, self.maxlag + 1):
                    independent_aic = np.column_stack((independent, lags[:, :i]))[i:]
                    dependent_aic = dependent[i:]
                    regression = LinearRegression(
                        independent_aic, dependent_aic, add_constant=False
                    )
                    regression.fit()
                    AICs.append(regression.AIC())
                optimal_lag = np.argmin(AICs)
                independent = np.column_stack((independent, lags[:, :optimal_lag]))[
                    optimal_lag:
                ]
                dependent = dependent[optimal_lag:]
        elif self.autolag == "BIC":
            if self.maxlag != 0:
                BICs = []
                independent_bic = independent.copy()
                dependent_bic = dependent.copy()
                regression = LinearRegression(
                    independent_bic, dependent_bic, add_constant=False
                )
                regression.fit()
                BICs.append(regression.BIC())
                lags = np.column_stack(
                    [dependent.shift(i) for i in range(1, self.maxlag + 1)]
                )
                for i in range(1, self.maxlag + 1):
                    independent_bic = np.column_stack((independent, lags[:, :i]))[i:]
                    dependent_bic = dependent[i:]
                    regression = LinearRegression(
                        independent_bic, dependent_bic, add_constant=False
                    )
                    regression.fit()
                    BICs.append(regression.BIC())
                optimal_lag = np.argmin(BICs)
                independent = np.column_stack((independent, lags[:, :optimal_lag]))[
                    optimal_lag:
                ]
                dependent = dependent[optimal_lag:]
        else:
            print("Unknown autolag method")
        regression = LinearRegression(independent, dependent, add_constant=False)
        regression.fit()
        self.coeff = regression.tstatistics()[0]
        return self

    def __str__(self):
        # TODO str
        s = ""
        return s


class EngleGranger:
    """
    Class for Engle-Granger cointegration test
    """

    def __init__(self, y, x, intercept=True, trend=False, maxlag=0, autolag="AIC"):
        self.y = y
        self.x = x
        self.intercept = intercept
        self.trend = trend
        self.maxlag = maxlag
        self.autolag = autolag
        self.coeff = None
        self.residuals = None
        self.ecm_coeff = None
        self.ecm_tstat = None
        self.ADFstat = None

    def fit(self):
        regression = LinearRegression(self.x, self.y, add_constant=True)
        regression.fit()
        self.residuals = regression.residuals
        self.coeff = regression.coeff
        ADF = AugmentedDickeyFuller(
            self.residuals,
            intercept=self.intercept,
            trend=self.trend,
            maxlag=self.maxlag,
            autolag=self.autolag,
        )
        ADF.fit()
        self.ADFstat = ADF.coeff
        ecm_x = np.column_stack((self.x.diff()[1:], self.residuals.shift(1)[1:]))
        ecm_y = self.y.diff()[1:]
        ecm_regression = LinearRegression(ecm_x, ecm_y, add_constant=False)
        ecm_regression.fit()
        self.ecm_coeff = ecm_regression.coeff[1]
        self.ecm_tstat = ecm_regression.tstatistics()[1]
        return self

    def __str__(self):

        s = (
            f"Engle-Granger Cointegration Test\n"
            f"--------------------------------\n"
            f"Beta from naive regression: {self.coeff[1]:.2f}\n"
            f"ADF Statistic for residuals: {self.ADFstat:.2f}\n"
            f"ECM Coefficient: {self.ecm_coeff:.4f}\n"
            f"ECM t-statistic: {self.ecm_tstat:.2f}"
        )
        return s


def fit_uo_params(x):
    """
    Estimation of parameters for UO process with AR(1) model
    """
    AR_regression = LinearRegression(x[:-1], x[1:])
    AR_regression.fit()
    theta = -np.log(AR_regression.coeff[1])
    mu = AR_regression.coeff[0] / (1 - AR_regression.coeff[1])
    sigma_eq = (
        np.sqrt(2 * theta)
        * np.std(AR_regression.residuals)
        / np.sqrt((1 - AR_regression.coeff[1] ** 2))
    )
    hl = np.log(2) / theta
    return theta, mu, sigma_eq, hl


def mle_uo_params(x, T):
    """
    Maximum likelihood estimation of parameters for UO process
    """
    Sx = np.sum(x[:-1])
    Sy = np.sum(x[1:])
    Sxx = x[:-1] @ x[:-1]
    Sxy = x[:-1] @ x[1:]
    Syy = x[1:] @ x[1:]
    N = len(x) - 1
    dt = T / N

    mu_mle = (Sy * Sxx - Sx * Sxy) / (N * (Sxx - Sxy) - (Sx**2 - Sx * Sy))
    theta_mle = -(1 / dt) * np.log(
        (Sxy - mu_mle * Sx - mu_mle * Sy + N * mu_mle**2)
        / (Sxx - 2 * mu_mle * Sx + N * mu_mle**2)
    )
    sigma2_hat = (
        Syy
        - 2 * np.exp(-theta_mle * dt) * Sxy
        + np.exp(-2 * theta_mle * dt) * Sxx
        - 2
        * mu_mle
        * (1 - np.exp(-theta_mle * dt))
        * (Sy - np.exp(-theta_mle * dt) * Sx)
        + N * mu_mle**2 * (1 - np.exp(-theta_mle * dt)) ** 2
    ) / N
    sigma_mle = np.sqrt(sigma2_hat * 2 * theta_mle / (1 - np.exp(-2 * theta_mle * dt)))
    hl_mle = np.log(2) / theta_mle
    return theta_mle, mu_mle, sigma_mle, hl_mle
