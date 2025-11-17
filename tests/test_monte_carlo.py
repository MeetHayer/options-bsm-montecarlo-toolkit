"""
Unit tests for Monte Carlo pricing module.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '../src')

from options_toolkit.monte_carlo import simulate_gbm_paths, monte_carlo_price
from options_toolkit.bsm import bsm_call_price, bsm_put_price


class TestGBMSimulation:
    """Test GBM path simulation."""
    
    def test_initial_price(self):
        """All paths should start at S0."""
        S0 = 100.0
        paths = simulate_gbm_paths(S0, 0.05, 0.20, 1.0, 100, 1000, seed=42)
        
        assert np.allclose(paths[:, 0], S0)
    
    def test_positive_prices(self):
        """All simulated prices should be positive."""
        paths = simulate_gbm_paths(100, 0.05, 0.20, 1.0, 100, 1000, seed=42)
        
        assert np.all(paths > 0)
    
    def test_path_shape(self):
        """Path array should have correct shape."""
        n_paths = 500
        n_steps = 252
        paths = simulate_gbm_paths(100, 0.05, 0.20, 1.0, n_steps, n_paths, seed=42)
        
        assert paths.shape == (n_paths, n_steps + 1)
    
    def test_seed_reproducibility(self):
        """Same seed should give same results."""
        paths1 = simulate_gbm_paths(100, 0.05, 0.20, 1.0, 100, 100, seed=42)
        paths2 = simulate_gbm_paths(100, 0.05, 0.20, 1.0, 100, 100, seed=42)
        
        assert np.allclose(paths1, paths2)
    
    def test_different_seeds_different_paths(self):
        """Different seeds should give different results."""
        paths1 = simulate_gbm_paths(100, 0.05, 0.20, 1.0, 100, 100, seed=42)
        paths2 = simulate_gbm_paths(100, 0.05, 0.20, 1.0, 100, 100, seed=123)
        
        assert not np.allclose(paths1, paths2)


class TestMonteCarlopricing:
    """Test Monte Carlo option pricing."""
    
    def test_call_price_positive(self):
        """MC call price should be positive."""
        price = monte_carlo_price(100, 100, 0.05, 0.20, 1.0, 100, 10000, "call", seed=42)
        
        assert price > 0
    
    def test_put_price_positive(self):
        """MC put price should be positive."""
        price = monte_carlo_price(100, 100, 0.05, 0.20, 1.0, 100, 10000, "put", seed=42)
        
        assert price > 0
    
    def test_convergence_to_bsm_call(self):
        """MC should converge to BSM for calls with enough paths."""
        S, K, r, sigma, T = 100, 100, 0.05, 0.20, 1.0
        
        bsm_price = bsm_call_price(S, K, r, sigma, T)
        mc_price = monte_carlo_price(S, K, r, sigma, T, n_steps=100, n_paths=100000, 
                                     option_type="call", seed=42)
        
        # Should be within 1% of BSM price
        relative_error = abs(mc_price - bsm_price) / bsm_price
        assert relative_error < 0.01, f"MC: {mc_price:.4f}, BSM: {bsm_price:.4f}"
    
    def test_convergence_to_bsm_put(self):
        """MC should converge to BSM for puts with enough paths."""
        S, K, r, sigma, T = 100, 100, 0.05, 0.20, 1.0
        
        bsm_price = bsm_put_price(S, K, r, sigma, T)
        mc_price = monte_carlo_price(S, K, r, sigma, T, n_steps=100, n_paths=100000,
                                     option_type="put", seed=42)
        
        relative_error = abs(mc_price - bsm_price) / bsm_price
        assert relative_error < 0.01
    
    def test_itm_call_greater_than_intrinsic(self):
        """ITM call MC price should be >= intrinsic value."""
        S, K = 110, 100
        price = monte_carlo_price(S, K, 0.05, 0.20, 1.0, 100, 50000, "call", seed=42)
        intrinsic = max(S - K, 0)
        
        assert price >= intrinsic * 0.95  # Allow small MC error
    
    def test_put_call_parity_approximately(self):
        """MC should approximately satisfy put-call parity."""
        S, K, r, sigma, T = 100, 100, 0.05, 0.20, 1.0
        
        # Use same seed for both
        call = monte_carlo_price(S, K, r, sigma, T, 100, 100000, "call", seed=42)
        put = monte_carlo_price(S, K, r, sigma, T, 100, 100000, "put", seed=42)
        
        lhs = call - put
        rhs = S - K * np.exp(-r * T)
        
        # Should be within 2% due to MC error
        relative_error = abs(lhs - rhs) / abs(rhs)
        assert relative_error < 0.02
    
    def test_higher_volatility_higher_price(self):
        """Higher volatility should increase MC option price."""
        S, K, r, T = 100, 100, 0.05, 1.0
        
        price_low_vol = monte_carlo_price(S, K, r, 0.10, T, 100, 50000, "call", seed=42)
        price_high_vol = monte_carlo_price(S, K, r, 0.30, T, 100, 50000, "call", seed=42)
        
        assert price_high_vol > price_low_vol
    
    def test_reproducibility_with_seed(self):
        """Same parameters and seed should give same price."""
        price1 = monte_carlo_price(100, 100, 0.05, 0.20, 1.0, 100, 10000, "call", seed=42)
        price2 = monte_carlo_price(100, 100, 0.05, 0.20, 1.0, 100, 10000, "call", seed=42)
        
        assert abs(price1 - price2) < 1e-10


class TestMonteCarloGreeks:
    """Test numerical Greeks from Monte Carlo."""
    
    def test_mc_delta_range_call(self):
        """MC delta for call should be roughly between 0 and 1."""
        from options_toolkit.monte_carlo import mc_delta
        
        delta = mc_delta(100, 100, 0.05, 0.20, 1.0, 100, 20000, "call", h=1.0, seed=42)
        
        # Allow some tolerance for MC error
        assert -0.1 < delta < 1.1
    
    def test_mc_vega_positive(self):
        """MC vega should be positive for long options."""
        from options_toolkit.monte_carlo import mc_vega
        
        vega = mc_vega(100, 100, 0.05, 0.20, 1.0, 100, 20000, "call", h=0.01, seed=42)
        
        assert vega > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

