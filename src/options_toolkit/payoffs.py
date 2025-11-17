"""
Payoff and P&L calculations for option positions.

Provides:
- Payoff functions for calls and puts (long and short)
- P&L functions incorporating premiums
- Straddle payoffs
"""

import numpy as np
from typing import Union


def call_payoff(
    S: Union[float, np.ndarray],
    K: float,
    long: bool = True,
) -> np.ndarray:
    """
    Calculate payoff at expiration for a call option.
    
    Payoff for long call: max(S - K, 0)
    Payoff for short call: -max(S - K, 0) = min(K - S, 0)
    
    Parameters
    ----------
    S : float or np.ndarray
        Stock price(s) at expiration
    K : float
        Strike price
    long : bool, optional
        True for long position, False for short position (default True)
    
    Returns
    -------
    np.ndarray
        Payoff values
    
    Examples
    --------
    >>> call_payoff(np.array([90, 100, 110]), 100, long=True)
    array([ 0.,  0., 10.])
    >>> call_payoff(np.array([90, 100, 110]), 100, long=False)
    array([ 0.,  0., -10.])
    """
    S = np.atleast_1d(S)
    payoff = np.maximum(S - K, 0)
    
    if not long:
        payoff = -payoff
    
    return payoff


def put_payoff(
    S: Union[float, np.ndarray],
    K: float,
    long: bool = True,
) -> np.ndarray:
    """
    Calculate payoff at expiration for a put option.
    
    Payoff for long put: max(K - S, 0)
    Payoff for short put: -max(K - S, 0) = min(S - K, 0)
    
    Parameters
    ----------
    S : float or np.ndarray
        Stock price(s) at expiration
    K : float
        Strike price
    long : bool, optional
        True for long position, False for short position (default True)
    
    Returns
    -------
    np.ndarray
        Payoff values
    
    Examples
    --------
    >>> put_payoff(np.array([90, 100, 110]), 100, long=True)
    array([10.,  0.,  0.])
    >>> put_payoff(np.array([90, 100, 110]), 100, long=False)
    array([-10.,  0.,  0.])
    """
    S = np.atleast_1d(S)
    payoff = np.maximum(K - S, 0)
    
    if not long:
        payoff = -payoff
    
    return payoff


def call_pnl(
    S: Union[float, np.ndarray],
    K: float,
    premium: float,
    long: bool = True,
) -> np.ndarray:
    """
    Calculate P&L (profit & loss) for a call option position.
    
    P&L accounts for both the payoff at expiration and the premium paid/received:
    - Long call P&L: max(S - K, 0) - premium
    - Short call P&L: premium - max(S - K, 0)
    
    Parameters
    ----------
    S : float or np.ndarray
        Stock price(s) at expiration
    K : float
        Strike price
    premium : float
        Option premium (price paid for long, received for short)
    long : bool, optional
        True for long position, False for short position (default True)
    
    Returns
    -------
    np.ndarray
        P&L values
    
    Examples
    --------
    >>> # Long call with premium $5: breaks even at S = 105
    >>> call_pnl(np.array([95, 100, 105, 110]), 100, 5, long=True)
    array([-5., -5.,  0.,  5.])
    """
    payoff = call_payoff(S, K, long=True)
    
    if long:
        pnl = payoff - premium
    else:
        pnl = premium - payoff
    
    return pnl


def put_pnl(
    S: Union[float, np.ndarray],
    K: float,
    premium: float,
    long: bool = True,
) -> np.ndarray:
    """
    Calculate P&L (profit & loss) for a put option position.
    
    P&L accounts for both the payoff at expiration and the premium paid/received:
    - Long put P&L: max(K - S, 0) - premium
    - Short put P&L: premium - max(K - S, 0)
    
    Parameters
    ----------
    S : float or np.ndarray
        Stock price(s) at expiration
    K : float
        Strike price
    premium : float
        Option premium (price paid for long, received for short)
    long : bool, optional
        True for long position, False for short position (default True)
    
    Returns
    -------
    np.ndarray
        P&L values
    
    Examples
    --------
    >>> # Long put with premium $4: breaks even at S = 96
    >>> put_pnl(np.array([90, 96, 100, 110]), 100, 4, long=True)
    array([ 6.,  0., -4., -4.])
    """
    payoff = put_payoff(S, K, long=True)
    
    if long:
        pnl = payoff - premium
    else:
        pnl = premium - payoff
    
    return pnl


def straddle_pnl(
    S: Union[float, np.ndarray],
    K: float,
    call_premium: float,
    put_premium: float,
    long: bool = True,
) -> np.ndarray:
    """
    Calculate P&L for a straddle position.
    
    A straddle consists of a call and a put with the same strike and expiration:
    - Long straddle: long call + long put (bet on volatility/big moves)
    - Short straddle: short call + short put (bet on low volatility/range-bound)
    
    Long straddle P&L:
        max(S - K, 0) + max(K - S, 0) - (call_premium + put_premium)
        = |S - K| - total_premium
    
    Long straddle profits when the stock moves significantly in either direction
    (beyond the breakeven points K ± total_premium).
    
    Parameters
    ----------
    S : float or np.ndarray
        Stock price(s) at expiration
    K : float
        Strike price (same for both call and put)
    call_premium : float
        Call option premium
    put_premium : float
        Put option premium
    long : bool, optional
        True for long straddle, False for short straddle (default True)
    
    Returns
    -------
    np.ndarray
        P&L values
    
    Examples
    --------
    >>> # Long straddle with K=100, premiums total $10
    >>> # Breakeven at 90 and 110
    >>> straddle_pnl(np.array([80, 90, 100, 110, 120]), 100, 5, 5, long=True)
    array([10.,  0., -10.,  0., 10.])
    
    Notes
    -----
    A long straddle has a V-shaped P&L profile: maximum loss is the total
    premium paid (when S = K at expiration), and unlimited upside potential.
    
    A short straddle has an inverted V-shape: maximum profit is the total
    premium collected (when S = K), and substantial downside risk.
    """
    S = np.atleast_1d(S)
    
    call_pnl_val = call_pnl(S, K, call_premium, long=long)
    put_pnl_val = put_pnl(S, K, put_premium, long=long)
    
    total_pnl = call_pnl_val + put_pnl_val
    
    return total_pnl

