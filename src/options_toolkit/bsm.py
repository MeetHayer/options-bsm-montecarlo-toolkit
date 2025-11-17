"""
Black-Scholes-Merton option pricing model.

Provides:
- Closed-form pricing for European calls and puts
- Greeks (Delta, Gamma, Vega, Theta, Rho) via closed-form formulas
- Implied volatility solver via Newton-Raphson method
"""

import numpy as np
from typing import Dict
from options_toolkit.utils import normal_cdf, normal_pdf, validate_option_inputs, intrinsic_value


def _calculate_d1_d2(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
) -> tuple[float, float]:
    """
    Calculate d1 and d2 for Black-Scholes formula.
    
    d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    d2 = d1 - σ√T
    
    Parameters
    ----------
    S : float
        Current stock price
    K : float
        Strike price
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    T : float
        Time to expiration in years
    
    Returns
    -------
    tuple[float, float]
        d1 and d2 values
    """
    if T == 0:
        # At expiration, return values that give intrinsic value
        return (np.inf if S > K else -np.inf, np.inf if S > K else -np.inf)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bsm_call_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
) -> float:
    """
    Calculate Black-Scholes-Merton price for a European call option.
    
    The BSM formula for a call is:
        C = S·N(d1) - K·e^(-rT)·N(d2)
    
    where N(·) is the standard normal CDF.
    
    Parameters
    ----------
    S : float
        Current stock price
    K : float
        Strike price
    r : float
        Risk-free interest rate (annualized, e.g., 0.05 for 5%)
    sigma : float
        Volatility (annualized, e.g., 0.20 for 20%)
    T : float
        Time to expiration in years
    
    Returns
    -------
    float
        Call option price
    
    Examples
    --------
    >>> bsm_call_price(100, 100, 0.05, 0.2, 1.0)
    10.45...
    """
    validate_option_inputs(S, K, r, sigma, T)
    
    # Handle T=0 case: return intrinsic value
    if T == 0:
        return intrinsic_value(S, K, "call")
    
    d1, d2 = _calculate_d1_d2(S, K, r, sigma, T)
    
    call_price = S * normal_cdf(d1) - K * np.exp(-r * T) * normal_cdf(d2)
    return call_price


def bsm_put_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
) -> float:
    """
    Calculate Black-Scholes-Merton price for a European put option.
    
    The BSM formula for a put is:
        P = K·e^(-rT)·N(-d2) - S·N(-d1)
    
    Alternatively, can use put-call parity: P = C - S + K·e^(-rT)
    
    Parameters
    ----------
    S : float
        Current stock price
    K : float
        Strike price
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility (annualized)
    T : float
        Time to expiration in years
    
    Returns
    -------
    float
        Put option price
    
    Examples
    --------
    >>> bsm_put_price(100, 100, 0.05, 0.2, 1.0)
    5.57...
    """
    validate_option_inputs(S, K, r, sigma, T)
    
    # Handle T=0 case: return intrinsic value
    if T == 0:
        return intrinsic_value(S, K, "put")
    
    d1, d2 = _calculate_d1_d2(S, K, r, sigma, T)
    
    put_price = K * np.exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1)
    return put_price


def bsm_call_greeks(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
) -> Dict[str, float]:
    """
    Calculate Greeks for a European call option using BSM closed-form formulas.
    
    Greeks measure the sensitivity of option price to various parameters:
    - Delta: sensitivity to stock price S (∂C/∂S). Ranges from 0 to 1 for calls.
              Represents hedge ratio: how many shares to hold per option.
    - Gamma: rate of change of Delta (∂²C/∂S²). Measures convexity of option value.
              High gamma means Delta changes rapidly with S.
    - Vega: sensitivity to volatility σ (∂C/∂σ). Always positive for long options.
             Measures exposure to changes in implied volatility.
    - Theta: sensitivity to time decay (∂C/∂T). Usually negative for long calls.
              Measures how much value the option loses per day, all else equal.
    - Rho: sensitivity to interest rate r (∂C/∂r). Usually positive for calls.
    
    Parameters
    ----------
    S : float
        Current stock price
    K : float
        Strike price
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    T : float
        Time to expiration in years
    
    Returns
    -------
    dict
        Dictionary containing 'delta', 'gamma', 'vega', 'theta', 'rho'
    
    Notes
    -----
    Vega is returned in terms of 1% change in volatility (multiply by 0.01).
    Theta is annualized; divide by 365 to get per-day theta.
    """
    validate_option_inputs(S, K, r, sigma, T)
    
    if T == 0:
        # At expiration, Greeks are undefined or trivial
        delta = 1.0 if S > K else 0.0
        return {
            "delta": delta,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0,
        }
    
    d1, d2 = _calculate_d1_d2(S, K, r, sigma, T)
    sqrt_T = np.sqrt(T)
    
    # Delta: ∂C/∂S = N(d1)
    delta = normal_cdf(d1)
    
    # Gamma: ∂²C/∂S² = φ(d1) / (S·σ·√T)
    # Same for calls and puts
    gamma = normal_pdf(d1) / (S * sigma * sqrt_T)
    
    # Vega: ∂C/∂σ = S·φ(d1)·√T
    # Reported per 1% change in volatility
    vega = S * normal_pdf(d1) * sqrt_T * 0.01
    
    # Theta: ∂C/∂T (annualized)
    # Theta = -[S·φ(d1)·σ/(2√T)] - r·K·e^(-rT)·N(d2)
    theta = (
        -(S * normal_pdf(d1) * sigma) / (2 * sqrt_T)
        - r * K * np.exp(-r * T) * normal_cdf(d2)
    )
    
    # Rho: ∂C/∂r = K·T·e^(-rT)·N(d2)
    # Reported per 1% change in interest rate
    rho = K * T * np.exp(-r * T) * normal_cdf(d2) * 0.01
    
    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


def bsm_put_greeks(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
) -> Dict[str, float]:
    """
    Calculate Greeks for a European put option using BSM closed-form formulas.
    
    Greeks for puts:
    - Delta: ranges from -1 to 0 for puts. Negative because put value decreases as S increases.
    - Gamma: same as call (always positive, measures convexity)
    - Vega: same as call (always positive for long options)
    - Theta: usually negative for long puts (time decay)
    - Rho: usually negative for puts (put value decreases as r increases)
    
    Parameters
    ----------
    S : float
        Current stock price
    K : float
        Strike price
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    T : float
        Time to expiration in years
    
    Returns
    -------
    dict
        Dictionary containing 'delta', 'gamma', 'vega', 'theta', 'rho'
    """
    validate_option_inputs(S, K, r, sigma, T)
    
    if T == 0:
        # At expiration
        delta = -1.0 if S < K else 0.0
        return {
            "delta": delta,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0,
        }
    
    d1, d2 = _calculate_d1_d2(S, K, r, sigma, T)
    sqrt_T = np.sqrt(T)
    
    # Delta: ∂P/∂S = N(d1) - 1 = -N(-d1)
    delta = normal_cdf(d1) - 1.0
    
    # Gamma: same as call
    gamma = normal_pdf(d1) / (S * sigma * sqrt_T)
    
    # Vega: same as call
    vega = S * normal_pdf(d1) * sqrt_T * 0.01
    
    # Theta: ∂P/∂T
    # Theta = -[S·φ(d1)·σ/(2√T)] + r·K·e^(-rT)·N(-d2)
    theta = (
        -(S * normal_pdf(d1) * sigma) / (2 * sqrt_T)
        + r * K * np.exp(-r * T) * normal_cdf(-d2)
    )
    
    # Rho: ∂P/∂r = -K·T·e^(-rT)·N(-d2)
    rho = -K * T * np.exp(-r * T) * normal_cdf(-d2) * 0.01
    
    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: str = "call",
    initial_guess: float = 0.2,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Solve for implied volatility given a market price.
    
    Implied volatility is the volatility parameter σ that, when plugged into BSM,
    produces the observed market price. It represents "the market's implied
    annualized volatility" rather than historical or realized volatility.
    
    Uses Newton-Raphson method: σ_new = σ_old - f(σ)/f'(σ)
    where f(σ) = BSM_price(σ) - market_price and f'(σ) = Vega.
    
    Parameters
    ----------
    market_price : float
        Observed market price of the option
    S : float
        Current stock price
    K : float
        Strike price
    r : float
        Risk-free rate (annualized)
    T : float
        Time to expiration in years
    option_type : str, optional
        "call" or "put", default "call"
    initial_guess : float, optional
        Starting guess for σ, default 0.2 (20%)
    tol : float, optional
        Convergence tolerance, default 1e-6
    max_iter : int, optional
        Maximum number of iterations, default 100
    
    Returns
    -------
    float
        Implied volatility σ
    
    Raises
    ------
    ValueError
        If Newton-Raphson fails to converge or inputs are invalid
    
    Examples
    --------
    >>> # Price a call, then recover the volatility
    >>> true_price = bsm_call_price(100, 100, 0.05, 0.25, 1.0)
    >>> iv = implied_volatility(true_price, 100, 100, 0.05, 1.0, "call")
    >>> abs(iv - 0.25) < 1e-6
    True
    
    Notes
    -----
    For deep out-of-the-money or near-expiration options, implied volatility
    can be numerically challenging to solve. The algorithm uses sensible bounds
    and will raise an exception if it cannot converge.
    """
    validate_option_inputs(S, K, r, 0.1, T)  # Validate other inputs
    
    if market_price <= 0:
        raise ValueError(f"Market price must be positive, got {market_price}")
    
    # Check intrinsic value
    intrinsic = intrinsic_value(S, K, option_type)
    if market_price < intrinsic:
        raise ValueError(
            f"Market price ({market_price:.4f}) is below intrinsic value ({intrinsic:.4f}). "
            "This violates no-arbitrage."
        )
    
    if T == 0:
        raise ValueError("Cannot solve for implied volatility at expiration (T=0)")
    
    option_type = option_type.lower()
    if option_type not in ["call", "put"]:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")
    
    # Pricing function
    price_func = bsm_call_price if option_type == "call" else bsm_put_price
    
    # Newton-Raphson iteration
    sigma = initial_guess
    
    for iteration in range(max_iter):
        # Calculate model price and vega
        try:
            model_price = price_func(S, K, r, sigma, T)
            
            # Vega (not scaled by 0.01 here, we need raw vega)
            d1, _ = _calculate_d1_d2(S, K, r, sigma, T)
            vega_raw = S * normal_pdf(d1) * np.sqrt(T)
            
            # Check for zero vega (shouldn't happen in practice)
            if vega_raw < 1e-10:
                raise ValueError("Vega is too small; cannot use Newton-Raphson")
            
            # Newton-Raphson update
            diff = model_price - market_price
            
            if abs(diff) < tol:
                return sigma
            
            sigma_new = sigma - diff / vega_raw
            
            # Keep sigma in reasonable bounds
            sigma_new = max(1e-4, min(sigma_new, 5.0))
            
            sigma = sigma_new
            
        except Exception as e:
            raise ValueError(
                f"Failed to compute implied volatility at iteration {iteration}: {e}"
            )
    
    raise ValueError(
        f"Implied volatility did not converge after {max_iter} iterations. "
        f"Last sigma: {sigma:.6f}, diff: {diff:.6f}"
    )

