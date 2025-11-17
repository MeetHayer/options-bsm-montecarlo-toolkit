"""
Unit tests for Black-Scholes-Merton pricing module.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '../src')

from options_toolkit.bsm import (
    bsm_call_price,
    bsm_put_price,
    bsm_call_greeks,
    bsm_put_greeks,
    implied_volatility,
)


class TestBSMPricing:
    """Test BSM pricing functions."""
    
    def test_call_price_positive(self):
        """Call price should always be positive."""
        price = bsm_call_price(100, 100, 0.05, 0.20, 1.0)
        assert price > 0
    
    def test_put_price_positive(self):
        """Put price should always be positive."""
        price = bsm_put_price(100, 100, 0.05, 0.20, 1.0)
        assert price > 0
    
    def test_put_call_parity(self):
        """Put-call parity must hold: C - P = S - K*e^(-rT)."""
        S, K, r, sigma, T = 100, 100, 0.05, 0.20, 1.0
        
        call = bsm_call_price(S, K, r, sigma, T)
        put = bsm_put_price(S, K, r, sigma, T)
        
        lhs = call - put
        rhs = S - K * np.exp(-r * T)
        
        assert abs(lhs - rhs) < 1e-10, f"Put-call parity violated: {lhs} != {rhs}"
    
    def test_itm_call_greater_than_intrinsic(self):
        """ITM call price should be >= intrinsic value."""
        S, K = 110, 100
        price = bsm_call_price(S, K, 0.05, 0.20, 1.0)
        intrinsic = max(S - K, 0)
        
        assert price >= intrinsic
    
    def test_otm_call_positive_time_value(self):
        """OTM call should have positive time value."""
        S, K = 95, 100
        price = bsm_call_price(S, K, 0.05, 0.20, 1.0)
        
        assert price > 0, "OTM call should have positive time value"
    
    def test_call_price_increases_with_volatility(self):
        """Call price should increase with volatility."""
        S, K, r, T = 100, 100, 0.05, 1.0
        
        price_low_vol = bsm_call_price(S, K, r, 0.10, T)
        price_high_vol = bsm_call_price(S, K, r, 0.30, T)
        
        assert price_high_vol > price_low_vol
    
    def test_at_expiration_equals_intrinsic(self):
        """At T=0, option price should equal intrinsic value."""
        S, K, r, sigma = 110, 100, 0.05, 0.20
        
        call_price = bsm_call_price(S, K, r, sigma, T=0)
        intrinsic = max(S - K, 0)
        
        assert abs(call_price - intrinsic) < 1e-10


class TestBSMGreeks:
    """Test Greeks calculations."""
    
    def test_call_delta_range(self):
        """Call delta should be between 0 and 1."""
        greeks = bsm_call_greeks(100, 100, 0.05, 0.20, 1.0)
        assert 0 <= greeks['delta'] <= 1
    
    def test_put_delta_range(self):
        """Put delta should be between -1 and 0."""
        greeks = bsm_put_greeks(100, 100, 0.05, 0.20, 1.0)
        assert -1 <= greeks['delta'] <= 0
    
    def test_gamma_positive(self):
        """Gamma should always be positive for long options."""
        call_greeks = bsm_call_greeks(100, 100, 0.05, 0.20, 1.0)
        put_greeks = bsm_put_greeks(100, 100, 0.05, 0.20, 1.0)
        
        assert call_greeks['gamma'] > 0
        assert put_greeks['gamma'] > 0
    
    def test_gamma_equality(self):
        """Call and put gamma should be equal."""
        call_greeks = bsm_call_greeks(100, 100, 0.05, 0.20, 1.0)
        put_greeks = bsm_put_greeks(100, 100, 0.05, 0.20, 1.0)
        
        assert abs(call_greeks['gamma'] - put_greeks['gamma']) < 1e-10
    
    def test_vega_positive(self):
        """Vega should be positive for long options."""
        call_greeks = bsm_call_greeks(100, 100, 0.05, 0.20, 1.0)
        put_greeks = bsm_put_greeks(100, 100, 0.05, 0.20, 1.0)
        
        assert call_greeks['vega'] > 0
        assert put_greeks['vega'] > 0
    
    def test_vega_equality(self):
        """Call and put vega should be equal."""
        call_greeks = bsm_call_greeks(100, 100, 0.05, 0.20, 1.0)
        put_greeks = bsm_put_greeks(100, 100, 0.05, 0.20, 1.0)
        
        assert abs(call_greeks['vega'] - put_greeks['vega']) < 1e-10
    
    def test_atm_call_delta_approx_half(self):
        """ATM call delta should be approximately 0.5-0.65 (higher with positive r)."""
        greeks = bsm_call_greeks(100, 100, 0.05, 0.20, 1.0)
        assert 0.4 < greeks['delta'] < 0.7  # Wider range for positive interest rates


class TestImpliedVolatility:
    """Test implied volatility solver."""
    
    def test_iv_recovery(self):
        """Should recover the input volatility."""
        S, K, r, T = 100, 100, 0.05, 1.0
        true_sigma = 0.25
        
        # Price with true_sigma
        market_price = bsm_call_price(S, K, r, true_sigma, T)
        
        # Recover sigma
        recovered_sigma = implied_volatility(market_price, S, K, r, T, "call")
        
        assert abs(recovered_sigma - true_sigma) < 1e-6
    
    def test_iv_for_put(self):
        """IV solver should work for puts."""
        S, K, r, T = 100, 100, 0.05, 1.0
        true_sigma = 0.30
        
        market_price = bsm_put_price(S, K, r, true_sigma, T)
        recovered_sigma = implied_volatility(market_price, S, K, r, T, "put")
        
        assert abs(recovered_sigma - true_sigma) < 1e-6
    
    def test_iv_itm_option(self):
        """IV solver should work for ITM options."""
        S, K, r, T = 110, 100, 0.05, 1.0
        true_sigma = 0.20
        
        market_price = bsm_call_price(S, K, r, true_sigma, T)
        recovered_sigma = implied_volatility(market_price, S, K, r, T, "call")
        
        assert abs(recovered_sigma - true_sigma) < 1e-6
    
    def test_iv_otm_option(self):
        """IV solver should work for OTM options."""
        S, K, r, T = 95, 100, 0.05, 1.0
        true_sigma = 0.20
        
        market_price = bsm_call_price(S, K, r, true_sigma, T)
        recovered_sigma = implied_volatility(market_price, S, K, r, T, "call")
        
        assert abs(recovered_sigma - true_sigma) < 1e-6
    
    def test_iv_below_intrinsic_raises_error(self):
        """Should raise error if market price < intrinsic value."""
        S, K, r, T = 110, 100, 0.05, 1.0
        market_price = 5.0  # Less than intrinsic value of 10
        
        with pytest.raises(ValueError, match="below intrinsic value"):
            implied_volatility(market_price, S, K, r, T, "call")


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_time_to_expiration(self):
        """At T=0, should return intrinsic value."""
        call = bsm_call_price(110, 100, 0.05, 0.20, 0.0)
        assert abs(call - 10.0) < 1e-10
        
        put = bsm_put_price(95, 100, 0.05, 0.20, 0.0)
        assert abs(put - 5.0) < 1e-10
    
    def test_deep_itm_call(self):
        """Deep ITM call should behave like stock."""
        call = bsm_call_price(200, 100, 0.05, 0.20, 1.0)
        # Should be approximately S - K*e^(-rT)
        expected = 200 - 100 * np.exp(-0.05)
        assert abs(call - expected) < 1.0  # Within $1
    
    def test_deep_otm_call(self):
        """Deep OTM call should be near zero."""
        call = bsm_call_price(50, 100, 0.05, 0.20, 1.0)
        assert call < 0.01  # Very small value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

