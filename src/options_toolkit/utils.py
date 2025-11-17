"""
Utility functions for the options toolkit.

Provides:
- Normal distribution CDF and PDF
- Input validation helpers
- Common calculations
"""

import numpy as np
from scipy.stats import norm
from typing import Union


def normal_cdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Standard normal cumulative distribution function.
    
    Parameters
    ----------
    x : float or np.ndarray
        Input value(s)
    
    Returns
    -------
    float or np.ndarray
        N(x), the cumulative probability up to x under N(0,1)
    """
    return norm.cdf(x)


def normal_pdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Standard normal probability density function.
    
    Parameters
    ----------
    x : float or np.ndarray
        Input value(s)
    
    Returns
    -------
    float or np.ndarray
        φ(x), the probability density at x under N(0,1)
    """
    return norm.pdf(x)


def validate_option_inputs(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
) -> None:
    """
    Validate common option pricing inputs.
    
    Parameters
    ----------
    S : float
        Current stock price (must be > 0)
    K : float
        Strike price (must be > 0)
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (must be > 0, annualized)
    T : float
        Time to expiration in years (must be >= 0)
    
    Raises
    ------
    ValueError
        If any input is invalid
    """
    if S <= 0:
        raise ValueError(f"Stock price S must be positive, got {S}")
    if K <= 0:
        raise ValueError(f"Strike price K must be positive, got {K}")
    if sigma <= 0:
        raise ValueError(f"Volatility sigma must be positive, got {sigma}")
    if T < 0:
        raise ValueError(f"Time to expiration T must be non-negative, got {T}")


def intrinsic_value(
    S: float,
    K: float,
    option_type: str = "call",
) -> float:
    """
    Calculate the intrinsic value of an option at expiration.
    
    Parameters
    ----------
    S : float
        Current stock price
    K : float
        Strike price
    option_type : str, optional
        "call" or "put", default "call"
    
    Returns
    -------
    float
        Intrinsic value: max(S-K, 0) for call, max(K-S, 0) for put
    """
    if option_type.lower() == "call":
        return max(S - K, 0.0)
    elif option_type.lower() == "put":
        return max(K - S, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

