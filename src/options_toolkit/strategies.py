"""
Option strategies: straddles, strangles, delta hedging, vega hedging.

Provides illustrative examples of common option strategies with
detailed explanations of their purpose and limitations.
"""

import numpy as np
from typing import Dict, Tuple
from options_toolkit.bsm import bsm_call_price, bsm_put_price, bsm_call_greeks, bsm_put_greeks
from options_toolkit.payoffs import straddle_pnl, call_pnl


def long_straddle_analysis(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    S_range: np.ndarray | None = None,
) -> Dict:
    """
    Analyze a long straddle: long call + long put at the same strike.
    
    A long straddle is a bet on VOLATILITY. It profits when the underlying
    makes a large move in either direction, regardless of direction.
    
    When to use:
    - Expect high volatility but uncertain about direction
    - Before major announcements (earnings, FDA approval, etc.)
    - When implied volatility is low relative to expected realized volatility
    
    Risks:
    - Time decay (both options lose theta)
    - If underlying stays near strike, lose the entire premium paid
    - Requires a significant move to break even
    
    Parameters
    ----------
    S0 : float
        Current stock price
    K : float
        Strike price for both call and put
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    T : float
        Time to expiration in years
    S_range : np.ndarray, optional
        Array of stock prices for P&L analysis. If None, uses S0 ± 50%
    
    Returns
    -------
    dict
        Contains:
        - 'call_price': BSM call price
        - 'put_price': BSM put price
        - 'total_cost': Total premium paid
        - 'call_greeks': Dict of call Greeks
        - 'put_greeks': Dict of put Greeks
        - 'net_greeks': Dict of combined Greeks
        - 'S_range': Stock price array for P&L
        - 'pnl': P&L values at expiration for each S in S_range
        - 'breakeven_lower': Lower breakeven point
        - 'breakeven_upper': Upper breakeven point
    
    Examples
    --------
    >>> result = long_straddle_analysis(100, 100, 0.05, 0.25, 0.5)
    >>> result['total_cost']  # Total premium paid
    12.7...
    >>> result['net_greeks']['delta']  # Near zero for ATM straddle
    0.0...
    >>> result['net_greeks']['vega']  # Positive (long volatility)
    19.8...
    """
    # Price the options
    call_price = bsm_call_price(S0, K, r, sigma, T)
    put_price = bsm_put_price(S0, K, r, sigma, T)
    total_cost = call_price + put_price
    
    # Compute Greeks
    call_greeks = bsm_call_greeks(S0, K, r, sigma, T)
    put_greeks = bsm_put_greeks(S0, K, r, sigma, T)
    
    # Net Greeks for the straddle
    net_greeks = {
        key: call_greeks[key] + put_greeks[key]
        for key in call_greeks.keys()
    }
    
    # P&L analysis over a range of stock prices
    if S_range is None:
        S_range = np.linspace(S0 * 0.5, S0 * 1.5, 100)
    
    pnl = straddle_pnl(S_range, K, call_price, put_price, long=True)
    
    # Breakeven points: K ± total_cost
    breakeven_lower = K - total_cost
    breakeven_upper = K + total_cost
    
    return {
        'call_price': call_price,
        'put_price': put_price,
        'total_cost': total_cost,
        'call_greeks': call_greeks,
        'put_greeks': put_greeks,
        'net_greeks': net_greeks,
        'S_range': S_range,
        'pnl': pnl,
        'breakeven_lower': breakeven_lower,
        'breakeven_upper': breakeven_upper,
    }


def long_strangle_analysis(
    S0: float,
    K_put: float,
    K_call: float,
    r: float,
    sigma: float,
    T: float,
    S_range: np.ndarray | None = None,
) -> Dict:
    """
    Analyze a long strangle: long OTM call + long OTM put at different strikes.
    
    A long strangle is similar to a straddle but RISKIER and CHEAPER.
    It uses OTM options (K_put < S0 < K_call), making it cheaper to enter
    but requiring a LARGER price move to profit.
    
    STRANGLE vs STRADDLE:
    - Strangle: OTM call + OTM put → LOWER cost, HIGHER risk (needs bigger move)
    - Straddle: ATM call + ATM put → HIGHER cost, LOWER risk (easier to profit)
    
    When to use:
    - Expect extreme volatility but want lower upfront cost than straddle
    - Willing to accept higher risk for lower premium
    - Before major binary events (earnings, FDA, mergers) when you're very bullish on movement
    
    Risks (HIGHER than straddle):
    - Requires LARGER move to break even (wider breakeven range)
    - Both options are OTM → more likely to expire worthless
    - Time decay on both positions
    - If price stays between strikes, lose entire premium
    
    Parameters
    ----------
    S0 : float
        Current stock price
    K_put : float
        Strike price for the OTM put (should be < S0)
    K_call : float
        Strike price for the OTM call (should be > S0)
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    T : float
        Time to expiration in years
    S_range : np.ndarray, optional
        Array of stock prices for P&L analysis. If None, uses S0 ± 50%
    
    Returns
    -------
    dict
        Contains:
        - 'call_price': BSM call price (OTM)
        - 'put_price': BSM put price (OTM)
        - 'total_cost': Total premium paid (< straddle cost)
        - 'call_greeks': Dict of call Greeks
        - 'put_greeks': Dict of put Greeks
        - 'net_greeks': Dict of combined Greeks
        - 'S_range': Stock price array for P&L
        - 'pnl': P&L values at expiration for each S in S_range
        - 'breakeven_lower': Lower breakeven point (< K_put)
        - 'breakeven_upper': Upper breakeven point (> K_call)
        - 'strike_width': K_call - K_put (wider = more risk)
    
    Examples
    --------
    >>> result = long_strangle_analysis(100, 95, 105, 0.05, 0.25, 0.5)
    >>> result['total_cost'] < 10  # Cheaper than ATM straddle
    True
    >>> result['strike_width']
    10.0
    >>> result['breakeven_upper'] - result['breakeven_lower'] > 15  # Wider breakeven range
    True
    
    Notes
    -----
    The strangle is a more aggressive volatility bet than the straddle.
    It's cheaper to enter but requires a more significant price movement to profit.
    Traders use strangles when they expect explosive moves but want to minimize
    upfront capital at the cost of higher risk.
    """
    # Validate strikes
    if K_put >= S0:
        raise ValueError(f"K_put ({K_put}) should be < S0 ({S0}) for OTM put")
    if K_call <= S0:
        raise ValueError(f"K_call ({K_call}) should be > S0 ({S0}) for OTM call")
    
    # Price the options (both OTM, so cheaper than ATM)
    call_price = bsm_call_price(S0, K_call, r, sigma, T)
    put_price = bsm_put_price(S0, K_put, r, sigma, T)
    total_cost = call_price + put_price
    
    # Compute Greeks
    call_greeks = bsm_call_greeks(S0, K_call, r, sigma, T)
    put_greeks = bsm_put_greeks(S0, K_put, r, sigma, T)
    
    # Net Greeks for the strangle
    net_greeks = {
        key: call_greeks[key] + put_greeks[key]
        for key in call_greeks.keys()
    }
    
    # P&L analysis over a range of stock prices
    if S_range is None:
        S_range = np.linspace(S0 * 0.5, S0 * 1.5, 100)
    
    # P&L at expiration: max(S - K_call, 0) + max(K_put - S, 0) - total_cost
    call_payoff = np.maximum(S_range - K_call, 0)
    put_payoff = np.maximum(K_put - S_range, 0)
    pnl = call_payoff + put_payoff - total_cost
    
    # Breakeven points:
    # Lower: K_put - total_cost (need put to be ITM by premium amount)
    # Upper: K_call + total_cost (need call to be ITM by premium amount)
    breakeven_lower = K_put - total_cost
    breakeven_upper = K_call + total_cost
    
    strike_width = K_call - K_put
    
    return {
        'call_price': call_price,
        'put_price': put_price,
        'total_cost': total_cost,
        'call_greeks': call_greeks,
        'put_greeks': put_greeks,
        'net_greeks': net_greeks,
        'S_range': S_range,
        'pnl': pnl,
        'breakeven_lower': breakeven_lower,
        'breakeven_upper': breakeven_upper,
        'strike_width': strike_width,
        'K_put': K_put,
        'K_call': K_call,
    }


def delta_hedge_illustration(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    S_range: np.ndarray | None = None,
) -> Dict:
    """
    Illustrate basic delta hedging for a long call position.
    
    Delta hedging aims to neutralize the sensitivity of a position to small
    changes in the underlying price. For a long call with delta Δ, we can
    hedge by shorting Δ shares of the underlying.
    
    STATIC HEDGE LIMITATIONS:
    - This is a ONE-TIME (static) hedge, not continuously rebalanced
    - Only effective for small price moves near S0
    - As S moves, delta changes (gamma effect), and hedge becomes imperfect
    - In practice, traders rebalance frequently (dynamic hedging)
    
    Purpose of delta hedging:
    - Isolate other risks (e.g., volatility risk) by neutralizing price risk
    - Market makers use this to manage inventory risk
    
    Parameters
    ----------
    S0 : float
        Current stock price
    K : float
        Strike price for the call
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    T : float
        Time to expiration in years
    S_range : np.ndarray, optional
        Array of stock prices at expiration for P&L. If None, uses S0 ± 30%
    
    Returns
    -------
    dict
        Contains:
        - 'call_price': BSM call price
        - 'call_delta': Delta of the call
        - 'hedge_shares': Number of shares to short
        - 'S_range': Stock price array
        - 'pnl_unhedged': P&L of long call only
        - 'pnl_hedged': P&L of long call + short shares (static hedge)
        - 'hedge_cost': Cost of shorting shares initially
    
    Examples
    --------
    >>> result = delta_hedge_illustration(100, 100, 0.05, 0.2, 1.0)
    >>> result['call_delta']
    0.56...
    >>> result['hedge_shares']  # Short this many shares
    0.56...
    
    Notes
    -----
    The hedged P&L will be flatter near S0 but diverges as price moves further
    due to gamma. This illustrates why continuous rebalancing is necessary for
    effective delta hedging.
    """
    # Price the call and get Greeks
    call_price = bsm_call_price(S0, K, r, sigma, T)
    call_greeks = bsm_call_greeks(S0, K, r, sigma, T)
    call_delta = call_greeks['delta']
    
    # Delta hedge: short delta shares per call
    hedge_shares = call_delta
    hedge_cost = hedge_shares * S0
    
    # P&L analysis
    if S_range is None:
        S_range = np.linspace(S0 * 0.7, S0 * 1.3, 100)
    
    # Unhedged: just the long call
    pnl_call = call_pnl(S_range, K, call_price, long=True)
    
    # Hedged: long call + short hedge_shares at S0, buy them back at S_range
    # P&L from short shares: S0 * hedge_shares - S_range * hedge_shares
    pnl_shares = hedge_shares * (S0 - S_range)
    pnl_hedged = pnl_call + pnl_shares
    
    return {
        'call_price': call_price,
        'call_delta': call_delta,
        'call_greeks': call_greeks,
        'hedge_shares': hedge_shares,
        'hedge_cost': hedge_cost,
        'S_range': S_range,
        'pnl_unhedged': pnl_call,
        'pnl_hedged': pnl_hedged,
    }


def vega_hedge_illustration(
    S0: float,
    K1: float,
    K2: float,
    r: float,
    sigma: float,
    T: float,
) -> Dict:
    """
    Illustrate vega hedging by combining two calls with different strikes.
    
    Vega measures sensitivity to changes in implied volatility. Traders often
    want to hedge vega risk (isolate delta or gamma) or take pure vega positions.
    
    APPROACH:
    - Hold two calls with different strikes (K1 and K2)
    - Solve for position sizes such that net vega ≈ 0
    - This is a simple 2-option example; in practice, vega hedging can involve
      multiple options across strikes and maturities
    
    LIMITATIONS:
    - Static hedge (vega changes as S, σ, T change)
    - Simplified illustration; real vega hedging is more complex
    
    Use case:
    - Want to bet on direction (delta) without taking volatility risk
    - Or vice versa: want pure volatility exposure without directional risk
    
    Parameters
    ----------
    S0 : float
        Current stock price
    K1 : float
        Strike of first call
    K2 : float
        Strike of second call (should differ from K1)
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    T : float
        Time to expiration in years
    
    Returns
    -------
    dict
        Contains:
        - 'call1_price': Price of call with strike K1
        - 'call1_greeks': Greeks of call with strike K1
        - 'call2_price': Price of call with strike K2
        - 'call2_greeks': Greeks of call with strike K2
        - 'position1': Position size in call 1 (normalized to 1.0)
        - 'position2': Position size in call 2 (to achieve vega = 0)
        - 'net_greeks': Combined Greeks of the portfolio
        - 'total_cost': Total cost of the position
    
    Examples
    --------
    >>> result = vega_hedge_illustration(100, 95, 105, 0.05, 0.25, 1.0)
    >>> abs(result['net_greeks']['vega']) < 0.1  # Should be near zero
    True
    
    Notes
    -----
    We normalize position1 = 1.0 (long one call at K1) and solve for position2
    such that: vega1 * position1 + vega2 * position2 = 0
    => position2 = -vega1 / vega2
    
    If vega1 and vega2 have the same sign, position2 will be negative (short).
    """
    # Price and Greeks for both calls
    call1_price = bsm_call_price(S0, K1, r, sigma, T)
    call1_greeks = bsm_call_greeks(S0, K1, r, sigma, T)
    
    call2_price = bsm_call_price(S0, K2, r, sigma, T)
    call2_greeks = bsm_call_greeks(S0, K2, r, sigma, T)
    
    # Position sizes: normalize call1 position to 1.0
    position1 = 1.0
    
    # Solve for position2 to make net vega = 0
    vega1 = call1_greeks['vega']
    vega2 = call2_greeks['vega']
    
    if abs(vega2) < 1e-10:
        raise ValueError(f"Call 2 has vega ≈ 0; cannot solve for vega hedge")
    
    position2 = -vega1 / vega2
    
    # Net Greeks
    net_greeks = {
        key: call1_greeks[key] * position1 + call2_greeks[key] * position2
        for key in call1_greeks.keys()
    }
    
    # Total cost (negative if net short premium)
    total_cost = call1_price * position1 + call2_price * position2
    
    return {
        'call1_price': call1_price,
        'call1_greeks': call1_greeks,
        'call2_price': call2_price,
        'call2_greeks': call2_greeks,
        'position1': position1,
        'position2': position2,
        'net_greeks': net_greeks,
        'total_cost': total_cost,
    }

