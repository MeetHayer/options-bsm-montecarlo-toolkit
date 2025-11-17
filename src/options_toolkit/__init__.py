"""
Options Toolkit: Educational implementation of option pricing models.

This package provides:
- Black-Scholes-Merton pricing and Greeks
- Monte Carlo simulation for option pricing
- Payoff and P&L analysis
- Basic option strategies (straddles, hedging)
- Visualization utilities
"""

from options_toolkit.bsm import (
    bsm_call_price,
    bsm_put_price,
    bsm_call_greeks,
    bsm_put_greeks,
    implied_volatility,
)
from options_toolkit.monte_carlo import (
    simulate_gbm_paths,
    monte_carlo_price,
    mc_delta,
    mc_vega,
)
from options_toolkit.payoffs import (
    call_payoff,
    put_payoff,
    call_pnl,
    put_pnl,
    straddle_pnl,
)

__version__ = "0.1.0"
__all__ = [
    "bsm_call_price",
    "bsm_put_price",
    "bsm_call_greeks",
    "bsm_put_greeks",
    "implied_volatility",
    "simulate_gbm_paths",
    "monte_carlo_price",
    "mc_delta",
    "mc_vega",
    "call_payoff",
    "put_payoff",
    "call_pnl",
    "put_pnl",
    "straddle_pnl",
]

