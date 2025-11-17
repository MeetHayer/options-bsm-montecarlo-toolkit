> **🚧 Work in Progress**  
> This is an ongoing project as I continue my quantitative finance education. I'll be updating and expanding this toolkit as I learn new concepts and techniques.

# Options BSM & Monte Carlo Toolkit

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MeetHayer/options-bsm-montecarlo-toolkit/HEAD?labpath=notebooks)

An educational Python toolkit for option pricing using Black-Scholes-Merton and Monte Carlo methods.

## Features

- **Black-Scholes-Merton Pricing**: European calls/puts with analytical Greeks (Delta, Gamma, Vega, Theta, Rho)
- **Implied Volatility Solver**: Newton-Raphson method to recover volatility from market prices
- **Monte Carlo Simulation**: GBM path simulation and European option pricing
- **Option Strategies**: Straddles, delta hedging, vega hedging
- **Interactive Notebooks**: Jupyter notebooks with widgets for exploration
- **Visualization**: Payoff diagrams, Greeks charts, volatility surfaces

## Installation

```bash
git clone https://github.com/MeetHayer/options-bsm-montecarlo-toolkit.git
cd options-bsm-montecarlo-toolkit
pip install -e .
```

## Quick Start

```python
from options_toolkit.bsm import bsm_call_price, bsm_call_greeks

# Price a call option
price = bsm_call_price(S0=100, K=100, r=0.05, sigma=0.20, T=1.0)
greeks = bsm_call_greeks(S0=100, K=100, r=0.05, sigma=0.20, T=1.0)

print(f"Call Price: ${price:.4f}")
print(f"Delta: {greeks['delta']:.4f}")
```

## Jupyter Notebooks

Three interactive notebooks in `notebooks/`:
1. **BSM Pricing and Greeks** - Option pricing, Greeks computation, payoff diagrams
2. **Implied Volatility** - IV solver, volatility surfaces, price sensitivity
3. **Monte Carlo & Strategies** - MC simulation, straddles, hedging

**🚀 [Launch Notebooks in Binder](https://mybinder.org/v2/gh/MeetHayer/options-bsm-montecarlo-toolkit/HEAD?labpath=notebooks)** (Click to run interactively in your browser)

Or run locally: `jupyter notebook notebooks/`

## Documentation

See `METHODS.md` for comprehensive theory and explanations of:
- Black-Scholes-Merton model and assumptions
- Monte Carlo simulation under GBM
- Implied volatility interpretation
- Greeks and their meaning
- Option strategies and hedging

## Testing

```bash
pytest tests/
```

## Contact

Email: hayermanmeetsingh@gmail.com

For questions or suggestions, please open an issue on GitHub.

## License

MIT License - see LICENSE file for details.
