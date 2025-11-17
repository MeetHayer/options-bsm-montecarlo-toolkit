# Options BSM & Monte Carlo Toolkit

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An educational Python toolkit for option pricing and analysis, demonstrating understanding of quantitative finance concepts for actuarial science and financial mathematics applications.

## 🎯 Project Overview

This project implements two fundamental option pricing models from scratch:

- **Black-Scholes-Merton (BSM)**: Closed-form pricing with Greeks and implied volatility
- **Monte Carlo Simulation**: Pricing under Geometric Brownian Motion

The toolkit provides comprehensive functionality for:
- Pricing European calls and puts
- Computing Greeks (Delta, Gamma, Vega, Theta, Rho)
- Solving for implied volatility
- Analyzing option strategies (straddles, hedging)
- Visualizing payoffs, P&L, and sensitivity analyses

**Key Design Principles:**
- Clean, readable code with extensive documentation
- Educational focus: explaining concepts, not just implementing formulas
- Type hints and modern Python 3.10+ features
- Minimal dependencies (numpy, scipy, matplotlib, pandas)
- Comprehensive Jupyter notebooks for exploration

---

## 📋 Features

### Core Functionality

✅ **Black-Scholes-Merton Pricing**
- Closed-form European call and put pricing
- Analytical Greeks (Delta, Gamma, Vega, Theta, Rho)
- Implied volatility solver using Newton-Raphson
- Put-call parity verification

✅ **Monte Carlo Simulation**
- GBM path simulation with configurable steps
- European option pricing via simulation
- Numerical Greeks via finite differences
- Convergence analysis and comparison with BSM

✅ **Payoff & P&L Analysis**
- Long and short positions for calls and puts
- Multi-leg strategies (straddles)
- Breakeven calculations
- Profit/loss visualization

✅ **Option Strategies**
- **Long Straddle**: Volatility play (long call + long put)
- **Delta Hedging**: Static hedge illustration
- **Vega Hedging**: Neutralizing volatility exposure

✅ **Visualization Tools**
- Payoff diagrams with breakeven points
- Greeks vs stock price and time to maturity
- Heatmaps (price vs volatility, price vs time)
- Monte Carlo distribution histograms

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup Instructions

1. **Clone or download this repository**

```bash
cd options-bsm-montecarlo-toolkit
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. **Install the package**

```bash
pip install -e .
```

This installs the package in editable mode along with all dependencies.

4. **Verify installation**

```bash
python -c "import options_toolkit; print('Installation successful!')"
```

---

## 📖 Usage

### Quick Start: Pricing an Option

```python
from options_toolkit.bsm import bsm_call_price, bsm_call_greeks

# Parameters
S0 = 100.0      # Current stock price
K = 100.0       # Strike price
r = 0.05        # Risk-free rate (5%)
sigma = 0.20    # Volatility (20%)
T = 1.0         # Time to expiration (1 year)

# Price a call option
call_price = bsm_call_price(S0, K, r, sigma, T)
print(f"Call Price: ${call_price:.4f}")

# Compute Greeks
greeks = bsm_call_greeks(S0, K, r, sigma, T)
print(f"Delta: {greeks['delta']:.4f}")
print(f"Vega:  {greeks['vega']:.4f}")
```

**Output:**
```
Call Price: $10.4506
Delta: 0.5636
Vega:  39.8942
```

### Solving for Implied Volatility

```python
from options_toolkit.bsm import implied_volatility

# Given a market price, solve for implied volatility
market_price = 12.50
iv = implied_volatility(market_price, S0, K, r, T, option_type="call")
print(f"Implied Volatility: {iv*100:.2f}%")
```

### Monte Carlo Pricing

```python
from options_toolkit.monte_carlo import monte_carlo_price

# Price using Monte Carlo simulation
mc_price = monte_carlo_price(
    S0=100, K=100, r=0.05, sigma=0.20, T=1.0,
    n_steps=252,      # Daily time steps
    n_paths=50000,    # Number of simulations
    option_type="call",
    seed=42           # For reproducibility
)
print(f"Monte Carlo Price: ${mc_price:.4f}")
```

### Analyzing a Straddle

```python
from options_toolkit.strategies import long_straddle_analysis
from options_toolkit.viz import plot_straddle_payoff
import matplotlib.pyplot as plt

# Analyze a long straddle
straddle = long_straddle_analysis(S0=100, K=100, r=0.05, sigma=0.25, T=0.5)

print(f"Total Cost: ${straddle['total_cost']:.2f}")
print(f"Breakeven Lower: ${straddle['breakeven_lower']:.2f}")
print(f"Breakeven Upper: ${straddle['breakeven_upper']:.2f}")

# Visualize P&L
fig = plot_straddle_payoff(
    straddle['S_range'],
    straddle['pnl'],
    K=100,
    breakeven_lower=straddle['breakeven_lower'],
    breakeven_upper=straddle['breakeven_upper']
)
plt.show()
```

### Creating Heatmaps

```python
import numpy as np
from options_toolkit.bsm import bsm_call_price
from options_toolkit.viz import payoff_heatmap

# Create a grid of prices
S_vals = np.linspace(80, 120, 50)
sigma_vals = np.linspace(0.10, 0.50, 40)

# Compute prices for each (S, σ) combination
prices = np.zeros((len(sigma_vals), len(S_vals)))
for i, sigma in enumerate(sigma_vals):
    for j, S in enumerate(S_vals):
        prices[i, j] = bsm_call_price(S, K=100, r=0.05, sigma=sigma, T=1.0)

# Plot heatmap
fig = payoff_heatmap(
    S_vals, sigma_vals, prices,
    xlabel="Stock Price",
    ylabel="Volatility",
    title="Call Price: S vs σ"
)
plt.show()
```

---

## 📓 Jupyter Notebooks

Three comprehensive notebooks are provided in the `notebooks/` directory:

### 1. BSM Pricing and Greeks (`01_bsm_pricing_and_greeks.ipynb`)

- Basic option pricing (calls and puts)
- Put-call parity verification
- Payoff diagrams for long/short positions
- Computing and interpreting Greeks
- Greeks vs stock price (moneyness)
- Greeks vs time to maturity

### 2. Implied Volatility (`02_implied_volatility.ipynb`)

- Option price as a function of volatility
- Solving for implied volatility
- Comparing low-IV vs high-IV scenarios
- Volatility surface heatmap
- Recovering IV from multiple market prices

### 3. Monte Carlo and Strategies (`03_monte_carlo_and_strategies.ipynb`)

- Simulating GBM paths
- Monte Carlo option pricing and convergence
- Comparing MC with BSM
- Long straddle analysis
- Delta hedging illustration
- Vega hedging illustration

**To run the notebooks:**

```bash
jupyter notebook notebooks/
```

---

## 📚 Documentation

### METHODS.md

The `METHODS.md` file provides a comprehensive explanation of:

- **Black-Scholes-Merton theory**: Assumptions, formulas, and intuition
- **Monte Carlo simulation**: GBM, discretization, and convergence
- **Implied volatility**: Definition, solving methods, and interpretation
- **Greeks**: Economic meaning and practical usage
- **Chart interpretation**: Payoff diagrams, heatmaps, and histograms
- **Option strategies**: When and how to use straddles and hedging
- **No-arbitrage principle**: Foundation of option pricing

This is written for students and practitioners with basic probability and finance knowledge.

---

## 🧪 Testing

Basic unit tests are provided in the `tests/` directory to verify:

- Put-call parity under BSM
- Symmetry of long/short payoffs
- Monte Carlo convergence to BSM
- Greeks computation accuracy

**Run tests:**

```bash
pytest tests/
```

---

## 📂 Project Structure

```
options-bsm-montecarlo-toolkit/
├── README.md                   # This file
├── METHODS.md                  # Comprehensive theory and methods
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Package configuration
├── src/
│   └── options_toolkit/
│       ├── __init__.py        # Package initialization
│       ├── bsm.py             # Black-Scholes-Merton pricing and Greeks
│       ├── monte_carlo.py     # Monte Carlo simulation
│       ├── payoffs.py         # Payoff and P&L functions
│       ├── strategies.py      # Option strategies (straddle, hedging)
│       ├── viz.py             # Visualization utilities
│       └── utils.py           # Helper functions
├── notebooks/
│   ├── 01_bsm_pricing_and_greeks.ipynb
│   ├── 02_implied_volatility.ipynb
│   └── 03_monte_carlo_and_strategies.ipynb
└── tests/
    ├── __init__.py
    ├── test_bsm.py
    ├── test_monte_carlo.py
    └── test_strategies.py
```

---

## 🎓 Educational Context

This project was created to demonstrate understanding of:

- **Quantitative finance**: Option pricing theory and practice
- **Stochastic processes**: Geometric Brownian Motion and simulation
- **Numerical methods**: Monte Carlo, Newton-Raphson, finite differences
- **Risk management**: Greeks and hedging strategies
- **Software engineering**: Clean code, documentation, testing

**Target audience:**
- Actuarial science students
- Quantitative finance learners
- Anyone interested in derivatives pricing

---

## 🔮 Future Enhancements

Potential extensions (not implemented in this version):

- **American options**: Early exercise using binomial trees or LSM
- **Exotic options**: Asian, barrier, lookback options
- **Dividends**: Discrete and continuous dividend adjustments
- **Multiple underlyings**: Spread options, basket options
- **Advanced Greeks**: Volga, vanna, charm
- **Calibration**: Fitting volatility surfaces to market data
- **Advanced variance reduction**: Antithetic variates, control variates

---

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

---

## 🙏 Acknowledgments

This project was inspired by various open-source options analysis tools and academic resources:

- **optionlab**: For Greeks and BSM implementation ideas
- **Option-Pricing-Model**: For simple BSM structure
- **GraphVega**: For multi-leg payoff visualization concepts
- Academic literature on Monte Carlo methods in finance

**Note:** All code in this project is original and written from scratch for educational purposes. No code was copied from external sources.

---

## 📧 Contact

For questions or suggestions about this project, please open an issue on GitHub.

---

## 🚦 Status

**Current Version:** 0.1.0  
**Status:** Educational/Portfolio Project  
**Python:** 3.10+  
**Last Updated:** November 2025

---

**Happy options pricing! 📈📉**

