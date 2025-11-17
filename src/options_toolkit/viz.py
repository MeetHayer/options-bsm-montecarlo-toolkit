"""
Visualization utilities for option pricing and payoffs.

Provides:
- Payoff diagrams for individual options and strategies
- Heatmaps for price sensitivity analysis
- Monte Carlo histogram visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_payoff_diagram(
    S_range: np.ndarray,
    payoff: np.ndarray,
    title: str = "Option Payoff",
    xlabel: str = "Stock Price at Expiration",
    ylabel: str = "Profit / Loss",
    breakeven: Optional[float | list] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot a payoff or P&L diagram for an option position.
    
    Parameters
    ----------
    S_range : np.ndarray
        Array of stock prices
    payoff : np.ndarray
        Corresponding payoff or P&L values
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    breakeven : float or list of float, optional
        Breakeven point(s) to mark on the plot
    figsize : tuple, optional
        Figure size (width, height)
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    
    Examples
    --------
    >>> from options_toolkit.payoffs import call_pnl
    >>> S_range = np.linspace(80, 120, 100)
    >>> pnl = call_pnl(S_range, 100, 5)
    >>> fig = plot_payoff_diagram(S_range, pnl, "Long Call P&L", breakeven=105)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the payoff
    ax.plot(S_range, payoff, 'b-', linewidth=2, label='P&L')
    
    # Add zero line
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.7)
    
    # Mark breakeven point(s)
    if breakeven is not None:
        if not isinstance(breakeven, list):
            breakeven = [breakeven]
        for be in breakeven:
            ax.axvline(x=be, color='r', linestyle=':', linewidth=1.5, 
                      label=f'Breakeven: {be:.2f}')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_straddle_payoff(
    S_range: np.ndarray,
    pnl: np.ndarray,
    K: float,
    breakeven_lower: float,
    breakeven_upper: float,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot P&L for a straddle position with highlighted features.
    
    Parameters
    ----------
    S_range : np.ndarray
        Array of stock prices
    pnl : np.ndarray
        P&L values for the straddle
    K : float
        Strike price
    breakeven_lower : float
        Lower breakeven point
    breakeven_upper : float
        Upper breakeven point
    figsize : tuple, optional
        Figure size
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot P&L
    ax.plot(S_range, pnl, 'b-', linewidth=2.5, label='Straddle P&L')
    
    # Zero line
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.7)
    
    # Mark strike
    ax.axvline(x=K, color='gray', linestyle='-', linewidth=1, alpha=0.5, 
              label=f'Strike: {K:.2f}')
    
    # Mark breakeven points
    ax.axvline(x=breakeven_lower, color='r', linestyle=':', linewidth=1.5, 
              label=f'Lower BE: {breakeven_lower:.2f}')
    ax.axvline(x=breakeven_upper, color='r', linestyle=':', linewidth=1.5, 
              label=f'Upper BE: {breakeven_upper:.2f}')
    
    # Shade profit/loss regions
    profit_mask = pnl > 0
    loss_mask = pnl <= 0
    ax.fill_between(S_range, 0, pnl, where=profit_mask, alpha=0.2, color='green', 
                    label='Profit region')
    ax.fill_between(S_range, 0, pnl, where=loss_mask, alpha=0.2, color='red', 
                    label='Loss region')
    
    ax.set_xlabel('Stock Price at Expiration', fontsize=12)
    ax.set_ylabel('Profit / Loss', fontsize=12)
    ax.set_title('Long Straddle P&L', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    
    plt.tight_layout()
    return fig


def plot_delta_hedge_comparison(
    S_range: np.ndarray,
    pnl_unhedged: np.ndarray,
    pnl_hedged: np.ndarray,
    S0: float,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Compare P&L of unhedged vs delta-hedged call position.
    
    Parameters
    ----------
    S_range : np.ndarray
        Array of stock prices at expiration
    pnl_unhedged : np.ndarray
        P&L of unhedged long call
    pnl_hedged : np.ndarray
        P&L of delta-hedged call
    S0 : float
        Initial stock price where hedge was established
    figsize : tuple, optional
        Figure size
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot both P&Ls
    ax.plot(S_range, pnl_unhedged, 'b-', linewidth=2, label='Unhedged (Long Call)')
    ax.plot(S_range, pnl_hedged, 'g--', linewidth=2, label='Delta Hedged')
    
    # Zero line and initial price
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(x=S0, color='orange', linestyle=':', linewidth=1.5, alpha=0.7,
              label=f'Initial S: {S0:.2f}')
    
    ax.set_xlabel('Stock Price at Expiration', fontsize=12)
    ax.set_ylabel('Profit / Loss', fontsize=12)
    ax.set_title('Delta Hedging: Unhedged vs Hedged Position', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add annotation
    ax.text(0.02, 0.98, 
            'Note: Static hedge (no rebalancing)\nEffective only near S0 due to gamma',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    return fig


def payoff_heatmap(
    S_grid: np.ndarray,
    param_grid: np.ndarray,
    values: np.ndarray,
    xlabel: str = "Stock Price",
    ylabel: str = "Parameter",
    title: str = "Option Price Heatmap",
    cmap: str = "viridis",
    figsize: tuple = (10, 7),
) -> plt.Figure:
    """
    Create a heatmap showing option values across two parameters.
    
    Useful for visualizing how option price changes with:
    - Stock price vs volatility
    - Stock price vs time to expiration
    - Strike vs volatility
    
    Parameters
    ----------
    S_grid : np.ndarray
        1D array of values for x-axis (e.g., stock prices)
    param_grid : np.ndarray
        1D array of values for y-axis (e.g., volatilities or time)
    values : np.ndarray
        2D array of shape (len(param_grid), len(S_grid)) containing values to plot
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    title : str, optional
        Plot title
    cmap : str, optional
        Colormap name (default 'viridis')
    figsize : tuple, optional
        Figure size
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    
    Examples
    --------
    >>> from options_toolkit.bsm import bsm_call_price
    >>> S_vals = np.linspace(80, 120, 50)
    >>> sigma_vals = np.linspace(0.1, 0.5, 30)
    >>> prices = np.zeros((len(sigma_vals), len(S_vals)))
    >>> for i, sigma in enumerate(sigma_vals):
    ...     for j, S in enumerate(S_vals):
    ...         prices[i, j] = bsm_call_price(S, 100, 0.05, sigma, 1.0)
    >>> fig = payoff_heatmap(S_vals, sigma_vals, prices, 
    ...                      xlabel="Stock Price", ylabel="Volatility",
    ...                      title="Call Price vs S and σ")
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the heatmap
    im = ax.imshow(values, aspect='auto', origin='lower', cmap=cmap,
                   extent=[S_grid.min(), S_grid.max(), 
                          param_grid.min(), param_grid.max()])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value', fontsize=11)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_monte_carlo_histogram(
    terminal_prices: np.ndarray,
    S0: float,
    K: float,
    option_type: str = "call",
    bsm_price: Optional[float] = None,
    mc_price: Optional[float] = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Plot histograms of Monte Carlo simulation results.
    
    Creates two subplots:
    1. Histogram of terminal stock prices
    2. Histogram of discounted payoffs (with BSM price line if provided)
    
    Parameters
    ----------
    terminal_prices : np.ndarray
        Array of simulated terminal stock prices
    S0 : float
        Initial stock price
    K : float
        Strike price
    option_type : str, optional
        "call" or "put"
    bsm_price : float, optional
        BSM price to compare against MC
    mc_price : float, optional
        Monte Carlo estimated price (average of discounted payoffs)
    figsize : tuple, optional
        Figure size
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: histogram of terminal prices
    ax1.hist(terminal_prices, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(x=S0, color='green', linestyle='--', linewidth=2, label=f'S0 = {S0:.2f}')
    ax1.axvline(x=K, color='red', linestyle='--', linewidth=2, label=f'Strike = {K:.2f}')
    ax1.set_xlabel('Terminal Stock Price', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Distribution of Terminal Prices', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: histogram of payoffs
    if option_type.lower() == "call":
        payoffs = np.maximum(terminal_prices - K, 0)
    else:
        payoffs = np.maximum(K - terminal_prices, 0)
    
    ax2.hist(payoffs, bins=50, alpha=0.7, color='green', edgecolor='black')
    
    if mc_price is not None:
        ax2.axvline(x=mc_price, color='blue', linestyle='-', linewidth=2,
                   label=f'MC Price = {mc_price:.3f}')
    
    if bsm_price is not None:
        ax2.axvline(x=bsm_price, color='red', linestyle='--', linewidth=2,
                   label=f'BSM Price = {bsm_price:.3f}')
    
    ax2.set_xlabel('Payoff at Expiration', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title(f'Distribution of {option_type.capitalize()} Payoffs', fontsize=12, fontweight='bold')
    if mc_price is not None or bsm_price is not None:
        ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_greeks_vs_spot(
    S_range: np.ndarray,
    greeks_dict: dict,
    greek_names: list[str] = None,
    title: str = "Greeks vs Stock Price",
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Plot multiple Greeks as functions of stock price.
    
    Parameters
    ----------
    S_range : np.ndarray
        Array of stock prices
    greeks_dict : dict
        Dictionary where keys are Greek names and values are arrays of Greek values
    greek_names : list of str, optional
        List of Greeks to plot. If None, plots all Greeks in greeks_dict
    title : str, optional
        Overall plot title
    figsize : tuple, optional
        Figure size
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    
    Examples
    --------
    >>> from options_toolkit.bsm import bsm_call_greeks
    >>> S_vals = np.linspace(80, 120, 50)
    >>> greeks = {name: [] for name in ['delta', 'gamma', 'vega', 'theta']}
    >>> for S in S_vals:
    ...     g = bsm_call_greeks(S, 100, 0.05, 0.2, 1.0)
    ...     for name in greeks:
    ...         greeks[name].append(g[name])
    >>> for name in greeks:
    ...     greeks[name] = np.array(greeks[name])
    >>> fig = plot_greeks_vs_spot(S_vals, greeks)
    """
    if greek_names is None:
        greek_names = list(greeks_dict.keys())
    
    n_greeks = len(greek_names)
    fig, axes = plt.subplots(n_greeks, 1, figsize=figsize, sharex=True)
    
    if n_greeks == 1:
        axes = [axes]
    
    for ax, greek_name in zip(axes, greek_names):
        values = greeks_dict[greek_name]
        ax.plot(S_range, values, 'b-', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_ylabel(greek_name.capitalize(), fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Stock Price', fontsize=12)
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

