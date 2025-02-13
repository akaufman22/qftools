"""
Module for simulating different types of stochastic processes
"""
import numpy as np

def normal_sample(N: int, k: int =1, random_state: int =None) -> np.ndarray:
    """
    Generate k samples of size N from a standard normal distribution
    """
    np.random.seed(random_state)
    return np.random.normal(0, 1, size=(k, N))

def uniform_sample(N: int, k: int = 1, random_state: int = None) -> np.ndarray:
    """
    Generate k samples of size N from a uniform distribution
    """
    np.random.seed(random_state)
    return np.random.uniform(0, 1, size=(k, N))

def generate_bm(T: float, N: int , k: int = 1, random_state: int = None) -> np.ndarray:
    """
    Gerenate k simulations of a Brownian motion with N steps
    """
    dW = normal_sample(N, k, random_state=random_state) * np.sqrt (T / N)
    return np.insert(dW.cumsum(axis=1), 0, 0, 1)

def generate_gbm(
        T: float,
        N: int, mu:
        float, sigma:
        float, S_0:
        float, k:
        int = 1,
        random_state:
        int = None) -> np.ndarray:
    """
    Generate k simulations of a geometric Brownian motion with N steps
    """
    time = np.linspace(0, T, N+1)
    W = generate_bm(T, N, k, random_state=random_state)
    S = S_0 * np.exp((mu - (sigma ** 2) / 2) * time + sigma * W)
    return S

def generate_uo(T: float,
                N: int,
                alpha: float,
                gamma: float,
                sigma: float,
                S_0: float,
                k: int = 1,
                random_state: int = None
                ) -> np.ndarray:
    """
    Generate k simulations of an Ornstein-Uhlenbeck process with N steps
    """
    time = np.linspace(0, T, N+1)
    dW = normal_sample(N, k, random_state=random_state) * np.sqrt (T / N)
    integral = np.insert(np.cumsum(np.exp(alpha * time[1:]) * dW, axis=1), 0, 0, 1)
    S = S_0 * np.exp(-alpha * time) + gamma * (1 - np.exp(-alpha * time)) + \
    sigma * np.exp(-alpha * time) * integral
    return S