"""
Module for estimating volatility using different methods
"""

from typing import Union

import numpy as np
import pandas as pd


def ctc_vol(
    price_data: Union[pd.DataFrame, pd.Series],
    window: int = 30,
    day_count: int = 252,
    ddof: int = 0,
    zero_mean: bool = False,
) -> pd.Series:
    """
    Realized volatility using the Close-to-Close method
    """
    if type(price_data) == pd.DataFrame:
        log_returns = np.log(price_data["Adj Close"] / price_data["Adj Close"].shift(1))
    elif type(price_data) == pd.Series:
        log_returns = np.log(price_data / price_data.shift(1))
    else:
        raise ValueError("price_data must be a pandas DataFrame or Series")
    if zero_mean:
        return np.sqrt(
            day_count
            * (
                log_returns.rolling(window=window).apply(
                    lambda x: np.sum(np.square(x)), raw=True
                )
                / window
            )
        )
    else:
        return log_returns.rolling(window=window).std(ddof=ddof) * np.sqrt(day_count)


def parkinson(
    price_data: pd.DataFrame, window: int = 30, day_count: int = 252
) -> pd.Series:
    """
    Realized volatility using the Parkinson method
    """
    log_hl = np.log(price_data["High"] / price_data["Low"])
    sum_log_hl_sq = log_hl.rolling(window).apply(
        lambda x: np.sum(np.square(x)), raw=True
    )
    volatility = np.sqrt(day_count) * np.sqrt(
        (1 / (4 * np.log(2))) * (1 / window) * sum_log_hl_sq
    )
    return volatility


def rodgers_satchell(
    price_data: pd.DataFrame, window: int = 30, day_count: int = 252
) -> pd.Series:
    """
    Realized volatility using the Rodgers-Satchell method
    """
    log_hc = np.log(price_data["High"] / price_data["Close"])
    log_ho = np.log(price_data["High"] / price_data["Open"])
    log_lc = np.log(price_data["Low"] / price_data["Close"])
    log_lo = np.log(price_data["Low"] / price_data["Open"])
    sum_prod = (log_hc * log_ho + log_lc * log_lo).rolling(window).sum()
    volatility = np.sqrt(day_count) * np.sqrt((1 / (window)) * sum_prod)
    return volatility


def garman_klass(
    price_data: pd.DataFrame, window: int = 30, day_count: int = 252
) -> pd.Series:
    """
    Realized volatility using the Garman-Klass method
    """
    log_hl = np.log(price_data["High"] / price_data["Low"])
    log_ctc = np.log(price_data["Adj Close"] / price_data["Adj Close"].shift(1))
    sum_log_hl_sq = log_hl.rolling(window).apply(
        lambda x: np.sum(np.square(x)), raw=True
    )
    sum_log_ctc_sq = log_ctc.rolling(window).apply(
        lambda x: np.sum(np.square(x)), raw=True
    )
    volatility = np.sqrt(day_count) * np.sqrt(
        (1 / (window)) * (0.5 * sum_log_hl_sq - (2 * np.log(2) - 1) * sum_log_ctc_sq)
    )
    return volatility


def yang_zhang(
    price_data: pd.DataFrame, window: int = 30, day_count: int = 252
) -> pd.Series:
    """
    Realized volatility using the Yang-Zhang method
    """
    log_olc = np.log(price_data["Open"] / price_data["Close"].shift(1))
    log_clo = np.log(price_data["Close"] / price_data["Open"].shift(1))
    sigma_o_sq = (1 / (window - 1)) * log_olc.rolling(window).apply(
        lambda x: np.sum(np.square(x)), raw=True
    )
    sigma_c_sq = (1 / (window - 1)) * log_clo.rolling(window).apply(
        lambda x: np.sum(np.square(x)), raw=True
    )
    sigma_rs_sq = (window / (day_count * (window - 1))) * rodgers_satchell(
        price_data, window=window, day_count=day_count
    ) ** 2
    k = 0.34 / (1 + (window + 1) / (window - 1))
    volatility = np.sqrt(day_count) * np.sqrt(
        sigma_o_sq + k * sigma_c_sq + (1 - k) * sigma_rs_sq
    )
    return volatility
