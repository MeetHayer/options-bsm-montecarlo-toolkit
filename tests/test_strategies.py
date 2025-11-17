"""
Unit tests for option strategies module.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '../src')

from options_toolkit.payoffs import call_payoff, put_payoff, call_pnl, put_pnl, straddle_pnl
from options_toolkit.strategies import long_straddle_analysis, delta_hedge_illustration, vega_hedge_illustration


class TestPayoffs:
    """Test payoff functions."""
    
    def test_long_call_payoff(self):
        """Long call payoff = max(S - K, 0)."""
        S_range = np.array([90, 100, 110])
        K = 100
        payoff = call_payoff(S_range, K, long=True)
        
        expected = np.array([0, 0, 10])
        assert np.allclose(payoff, expected)
    
    def test_short_call_payoff(self):
        """Short call payoff = -max(S - K, 0)."""
        S_range = np.array([90, 100, 110])
        K = 100
        payoff = call_payoff(S_range, K, long=False)
        
        expected = np.array([0, 0, -10])
        assert np.allclose(payoff, expected)
    
    def test_long_put_payoff(self):
        """Long put payoff = max(K - S, 0)."""
        S_range = np.array([90, 100, 110])
        K = 100
        payoff = put_payoff(S_range, K, long=True)
        
        expected = np.array([10, 0, 0])
        assert np.allclose(payoff, expected)
    
    def test_short_put_payoff(self):
        """Short put payoff = -max(K - S, 0)."""
        S_range = np.array([90, 100, 110])
        K = 100
        payoff = put_payoff(S_range, K, long=False)
        
        expected = np.array([-10, 0, 0])
        assert np.allclose(payoff, expected)
    
    def test_call_pnl_with_premium(self):
        """Long call P&L should account for premium."""
        S_range = np.array([90, 100, 110])
        K = 100
        premium = 5
        pnl = call_pnl(S_range, K, premium, long=True)
        
        expected = np.array([-5, -5, 5])  # max(S-K,0) - premium
        assert np.allclose(pnl, expected)
    
    def test_long_short_symmetry(self):
        """Long and short positions should be mirror images."""
        S_range = np.linspace(80, 120, 100)
        K = 100
        premium = 5
        
        long_pnl = call_pnl(S_range, K, premium, long=True)
        short_pnl = call_pnl(S_range, K, premium, long=False)
        
        assert np.allclose(long_pnl, -short_pnl)


class TestStraddlePayoff:
    """Test straddle payoff calculations."""
    
    def test_straddle_shape(self):
        """Straddle P&L should be V-shaped."""
        S_range = np.array([80, 90, 100, 110, 120])
        K = 100
        call_premium = 5
        put_premium = 5
        
        pnl = straddle_pnl(S_range, K, call_premium, put_premium, long=True)
        
        # Should be symmetric around K with minimum at K
        assert pnl[2] < pnl[1]  # Lower at K than away from K
        assert pnl[2] < pnl[3]
        assert pnl[0] > pnl[2]  # Higher at extremes
        assert pnl[4] > pnl[2]
    
    def test_straddle_max_loss(self):
        """Max loss for long straddle is total premium at K."""
        K = 100
        call_premium = 5
        put_premium = 5
        
        pnl_at_strike = straddle_pnl(K, K, call_premium, put_premium, long=True)
        
        expected_loss = -(call_premium + put_premium)
        assert abs(pnl_at_strike - expected_loss) < 1e-10
    
    def test_straddle_breakeven(self):
        """Straddle breaks even at K ± total premium."""
        K = 100
        call_premium = 5
        put_premium = 5
        total_premium = call_premium + put_premium
        
        # Test lower breakeven
        pnl_lower = straddle_pnl(K - total_premium, K, call_premium, put_premium, long=True)
        assert abs(pnl_lower) < 1e-10
        
        # Test upper breakeven
        pnl_upper = straddle_pnl(K + total_premium, K, call_premium, put_premium, long=True)
        assert abs(pnl_upper) < 1e-10


class TestStrategies:
    """Test option strategy analysis functions."""
    
    def test_long_straddle_analysis(self):
        """Long straddle analysis should return expected structure."""
        result = long_straddle_analysis(S0=100, K=100, r=0.05, sigma=0.25, T=1.0)
        
        # Check structure
        assert 'call_price' in result
        assert 'put_price' in result
        assert 'total_cost' in result
        assert 'net_greeks' in result
        assert 'breakeven_lower' in result
        assert 'breakeven_upper' in result
        
        # Check values
        assert result['total_cost'] > 0
        assert result['breakeven_lower'] < 100
        assert result['breakeven_upper'] > 100
        
        # Check Greeks
        assert abs(result['net_greeks']['delta']) < 0.3  # Reasonably close to zero for ATM
        assert result['net_greeks']['vega'] > 0  # Positive vega
        assert result['net_greeks']['theta'] < 0  # Negative theta
    
    def test_delta_hedge_illustration(self):
        """Delta hedge should return proper structure."""
        result = delta_hedge_illustration(S0=100, K=100, r=0.05, sigma=0.25, T=1.0)
        
        # Check structure
        assert 'call_price' in result
        assert 'call_delta' in result
        assert 'hedge_shares' in result
        assert 'pnl_unhedged' in result
        assert 'pnl_hedged' in result
        
        # Check delta
        assert 0 < result['call_delta'] < 1
        assert abs(result['hedge_shares'] - result['call_delta']) < 1e-10
        
        # Hedged position should be flatter near S0
        # (variance of hedged P&L should be lower near S0)
    
    def test_vega_hedge_illustration(self):
        """Vega hedge should produce near-zero net vega."""
        result = vega_hedge_illustration(S0=100, K1=95, K2=105, r=0.05, sigma=0.25, T=1.0)
        
        # Check structure
        assert 'call1_price' in result
        assert 'call2_price' in result
        assert 'position1' in result
        assert 'position2' in result
        assert 'net_greeks' in result
        
        # Net vega should be near zero
        assert abs(result['net_greeks']['vega']) < 0.1
        
        # Position 1 is normalized to 1
        assert abs(result['position1'] - 1.0) < 1e-10
    
    def test_delta_hedge_reduces_price_sensitivity(self):
        """Hedged position should be less sensitive to price near S0."""
        result = delta_hedge_illustration(S0=100, K=100, r=0.05, sigma=0.20, T=1.0)
        
        # Find index closest to S0 = 100
        S_range = result['S_range']
        idx_S0 = np.argmin(np.abs(S_range - 100))
        
        # Compare variances around S0
        window = 10
        unhedged_var = np.var(result['pnl_unhedged'][idx_S0-window:idx_S0+window])
        hedged_var = np.var(result['pnl_hedged'][idx_S0-window:idx_S0+window])
        
        # Hedged should have lower variance near S0
        assert hedged_var < unhedged_var


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

