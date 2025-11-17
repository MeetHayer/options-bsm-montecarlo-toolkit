"""
Monte Carlo option pricing under Geometric Brownian Motion.

Provides:
- GBM path simulation
- Monte Carlo pricing for European options
- Numerical Greeks via finite differences
"""

import numpy as np
from typing import Optional
from options_toolkit.utils import validate_option_inputs, intrinsic_value


def simulate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate stock price paths under Geometric Brownian Motion (GBM).
    
    The GBM model assumes:
        dS_t = r·S_t·dt + σ·S_t·dW_t
    
    Using the exact solution over discrete time steps:
        S_{t+Δt} = S_t · exp[(r - 0.5σ²)Δt + σ√Δt·Z]
    where Z ~ N(0,1).
    
    Parameters
    ----------
    S0 : float
        Initial stock price
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    T : float
        Time horizon in years
    n_steps : int
        Number of time steps
    n_paths : int
        Number of paths to simulate
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    np.ndarray
        Array of shape (n_paths, n_steps + 1) containing simulated prices.
        paths[:, 0] = S0, paths[:, -1] = terminal prices at time T.
    
    Examples
    --------
    >>> paths = simulate_gbm_paths(100, 0.05, 0.2, 1.0, 252, 10000, seed=42)
    >>> paths.shape
    (10000, 253)
    >>> np.allclose(paths[:, 0], 100.0)
    True
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / n_steps
    
    # Initialize price paths
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    # Generate all random increments at once for efficiency
    Z = np.random.standard_normal((n_paths, n_steps))
    
    # Calculate drift and diffusion components
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    # Simulate paths using cumulative product
    for t in range(n_steps):
        paths[:, t + 1] = paths[:, t] * np.exp(drift + diffusion * Z[:, t])
    
    return paths


def monte_carlo_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    option_type: str = "call",
    seed: Optional[int] = None,
) -> float:
    """
    Price a European option using Monte Carlo simulation.
    
    The Monte Carlo method:
    1. Simulates many possible future price paths under GBM
    2. Calculates the payoff at expiration for each path
    3. Averages the payoffs and discounts to present value
    
    This is more flexible than closed-form solutions (can handle path-dependent
    options, exotic payoffs, etc.) but is computationally more expensive.
    
    Parameters
    ----------
    S0 : float
        Current stock price
    K : float
        Strike price
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    T : float
        Time to expiration in years
    n_steps : int
        Number of time steps per path
    n_paths : int
        Number of Monte Carlo paths
    option_type : str, optional
        "call" or "put", default "call"
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    float
        Estimated option price
    
    Examples
    --------
    >>> # Should be close to BSM with enough paths
    >>> mc_price = monte_carlo_price(100, 100, 0.05, 0.2, 1.0, 252, 100000, "call", seed=42)
    >>> # Compare to BSM: bsm_call_price(100, 100, 0.05, 0.2, 1.0) ≈ 10.45
    >>> 10.0 < mc_price < 11.0
    True
    
    Notes
    -----
    Standard error decreases as 1/√n_paths. For accurate pricing, use at least
    10,000 paths. For n_steps, 50-252 is typically sufficient for European options.
    """
    validate_option_inputs(S0, K, r, sigma, T)
    
    option_type = option_type.lower()
    if option_type not in ["call", "put"]:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")
    
    if T == 0:
        return intrinsic_value(S0, K, option_type)
    
    # Simulate paths
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed)
    
    # Extract terminal prices
    S_T = paths[:, -1]
    
    # Calculate payoffs at expiration
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0)
    else:  # put
        payoffs = np.maximum(K - S_T, 0)
    
    # Average and discount to present value
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    return option_price


def mc_delta(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    option_type: str = "call",
    h: float = 0.01,
    seed: Optional[int] = None,
) -> float:
    """
    Estimate Delta using finite differences and Monte Carlo.
    
    Delta ≈ [V(S+h) - V(S-h)] / (2h)
    
    This uses central difference for better accuracy. Note that numerical
    Greeks from MC have estimation error from both finite difference
    approximation and Monte Carlo sampling.
    
    Parameters
    ----------
    S0 : float
        Current stock price
    K : float
        Strike price
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    T : float
        Time to expiration in years
    n_steps : int
        Number of time steps per path
    n_paths : int
        Number of Monte Carlo paths
    option_type : str, optional
        "call" or "put", default "call"
    h : float, optional
        Bump size for finite difference (default 0.01, i.e., $0.01)
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    float
        Estimated Delta
    
    Notes
    -----
    For more accurate Greeks, increase n_paths and decrease h (but not too small
    to avoid numerical instability). Typically h = 0.01 to 0.1 works well.
    """
    # Price at S0 + h
    price_up = monte_carlo_price(
        S0 + h, K, r, sigma, T, n_steps, n_paths, option_type, seed
    )
    
    # Price at S0 - h
    price_down = monte_carlo_price(
        S0 - h, K, r, sigma, T, n_steps, n_paths, option_type, seed
    )
    
    # Central difference
    delta = (price_up - price_down) / (2 * h)
    
    return delta


def mc_vega(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    option_type: str = "call",
    h: float = 0.01,
    seed: Optional[int] = None,
) -> float:
    """
    Estimate Vega using finite differences and Monte Carlo.
    
    Vega ≈ [V(σ+h) - V(σ-h)] / (2h)
    
    Measures sensitivity of option price to changes in volatility.
    
    Parameters
    ----------
    S0 : float
        Current stock price
    K : float
        Strike price
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    T : float
        Time to expiration in years
    n_steps : int
        Number of time steps per path
    n_paths : int
        Number of Monte Carlo paths
    option_type : str, optional
        "call" or "put", default "call"
    h : float, optional
        Bump size for volatility (default 0.01, i.e., 1% absolute change)
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    float
        Estimated Vega (per 1% change in volatility)
    
    Notes
    -----
    The returned vega represents the change in option price for a 0.01 (1%)
    change in volatility, matching the convention in bsm_call_greeks.
    """
    # Ensure sigma + h and sigma - h are positive
    sigma_up = sigma + h
    sigma_down = max(sigma - h, 1e-4)
    
    # Price at σ + h
    price_up = monte_carlo_price(
        S0, K, r, sigma_up, T, n_steps, n_paths, option_type, seed
    )
    
    # Price at σ - h
    price_down = monte_carlo_price(
        S0, K, r, sigma_down, T, n_steps, n_paths, option_type, seed
    )
    
    # Central difference
    vega = (price_up - price_down) / (2 * h)
    
    return vega

