# Methods: Option Pricing Theory and Implementation

This document provides a comprehensive explanation of the methods implemented in this toolkit. It is written for students and practitioners with a basic understanding of probability and finance, particularly those studying actuarial science or quantitative finance.

---

## Table of Contents

1. [Black-Scholes-Merton Model](#1-black-scholes-merton-model)
2. [Monte Carlo Simulation](#2-monte-carlo-simulation)
3. [Implied Volatility](#3-implied-volatility)
4. [Understanding the Greeks](#4-understanding-the-greeks)
5. [Reading the Charts](#5-reading-the-charts)
6. [Option Strategies](#6-option-strategies)
7. [No-Arbitrage Principle](#7-no-arbitrage-principle)

---

## 1. Black-Scholes-Merton Model

### 1.1 What is BSM?

The **Black-Scholes-Merton (BSM) model** is a mathematical model for pricing European-style options. Developed by Fischer Black, Myron Scholes, and Robert Merton in the early 1970s, it revolutionized financial markets by providing a closed-form formula for option prices.

### 1.2 Inputs to BSM

The BSM formula requires five inputs:

- **S₀**: Current price of the underlying stock
- **K**: Strike price (the price at which the option can be exercised)
- **r**: Risk-free interest rate (annualized, continuously compounded)
- **σ** (sigma): Volatility of the underlying stock (annualized standard deviation of returns)
- **T**: Time to expiration (in years)

### 1.3 Key Assumptions

The BSM model makes several simplifying assumptions:

1. **Efficient markets**: No transaction costs, no taxes, and all securities are perfectly divisible
2. **Continuous trading**: Can buy/sell at any time
3. **Constant volatility**: σ remains constant over the life of the option
4. **Constant interest rate**: r is known and constant
5. **No dividends**: The underlying stock pays no dividends
6. **Log-normal stock prices**: Stock prices follow geometric Brownian motion
7. **European exercise**: Option can only be exercised at expiration
8. **No arbitrage**: There are no risk-free profit opportunities

In reality, many of these assumptions are violated, but BSM still provides a useful baseline for option pricing.

### 1.4 The BSM Formulas

For a **European call option**:

```
C = S₀·N(d₁) - K·e^(-rT)·N(d₂)
```

For a **European put option**:

```
P = K·e^(-rT)·N(-d₂) - S₀·N(-d₁)
```

Where:

```
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

And N(·) is the cumulative distribution function of the standard normal distribution.

### 1.5 Intuition Behind BSM

The BSM formula can be understood intuitively:

- **S₀·N(d₁)**: Expected present value of the stock at expiration, conditional on the option being in-the-money. N(d₁) is approximately the delta-adjusted probability of exercise.

- **K·e^(-rT)·N(d₂)**: Discounted strike price, weighted by N(d₂), which approximates the risk-neutral probability that the option finishes in-the-money.

The call price is essentially: (probability-weighted stock value) - (probability-weighted strike value), discounted to present value.

### 1.6 Put-Call Parity

An important relationship between call and put prices:

```
C - P = S₀ - K·e^(-rT)
```

This relationship must hold to prevent arbitrage. Our implementation verifies this relationship.

---

## 2. Monte Carlo Simulation

### 2.1 What is Monte Carlo?

**Monte Carlo simulation** is a computational technique that uses random sampling to estimate numerical results. For option pricing, we simulate many possible future paths of the stock price and average the option payoffs across all paths.

### 2.2 Geometric Brownian Motion (GBM)

Under the BSM assumptions, stock prices follow GBM:

```
dS_t = μ·S_t·dt + σ·S_t·dW_t
```

Where:
- **μ**: Expected return (drift)
- **σ**: Volatility
- **dW_t**: Wiener process (Brownian motion) increment

For pricing, we use the **risk-neutral measure**, replacing μ with r (risk-free rate).

### 2.3 Simulating GBM Paths

We discretize time into small steps and use the exact solution of GBM:

```
S_{t+Δt} = S_t · exp[(r - 0.5σ²)Δt + σ√Δt·Z]
```

Where Z ~ N(0,1) is a standard normal random variable.

**Why the 0.5σ² term?** This is the Itô correction, arising from Jensen's inequality. It ensures the expected value of S_t equals S₀·e^(rt) under the risk-neutral measure.

### 2.4 Monte Carlo Option Pricing Algorithm

1. **Simulate**: Generate N paths of the stock price from S₀ to time T
2. **Payoff**: For each path, calculate the option payoff at expiration:
   - Call: max(S_T - K, 0)
   - Put: max(K - S_T, 0)
3. **Average**: Take the mean of all payoffs
4. **Discount**: Multiply by e^(-rT) to get present value

**Mathematical formula**:

```
Option Price ≈ e^(-rT) · (1/N) · Σ payoff_i
```

### 2.5 Why Use Monte Carlo?

**Advantages:**
- Flexible: Can handle complex payoffs (Asian, barrier, lookback options)
- Natural: Directly simulates the underlying stochastic process
- Scalable: Works for multi-asset options

**Disadvantages:**
- Slow: Requires many paths for accuracy (standard error ~ 1/√N)
- Computationally intensive: Not ideal for European options where closed-form solutions exist
- Slow convergence for Greeks (require finite differences)

For European options, BSM is faster and more accurate. Monte Carlo shines for exotic options without closed-form solutions.

### 2.6 Convergence and Accuracy

The **standard error** of the Monte Carlo estimate is:

```
SE ≈ σ_payoff / √N
```

Where σ_payoff is the standard deviation of payoffs. To halve the error, you need 4× more paths.

---

## 3. Implied Volatility

### 3.1 What is Implied Volatility?

**Implied volatility (IV)** is the volatility parameter that, when input into the BSM formula, produces the observed market price of an option.

Unlike historical volatility (calculated from past price data), IV represents the market's expectation of future volatility over the life of the option.

### 3.2 Why is IV Important?

1. **Comparing options**: Different options have different strikes and maturities. IV provides a standardized metric to compare "expensiveness"
2. **Market sentiment**: High IV often indicates uncertainty (before earnings, elections, FDA approvals)
3. **Trading strategies**: Many strategies bet on changes in IV rather than stock price direction
4. **Volatility surface**: IV varies with strike (smile/skew) and maturity (term structure)

### 3.3 Solving for Implied Volatility

Since BSM is monotonic in σ, we can invert it numerically. We use **Newton-Raphson**:

```
σ_new = σ_old - [BSM(σ_old) - Market_Price] / Vega(σ_old)
```

Where Vega = ∂BSM/∂σ (the sensitivity to volatility).

**Algorithm:**
1. Start with an initial guess (e.g., σ = 0.20)
2. Compute BSM price and Vega at current σ
3. Update σ using Newton-Raphson formula
4. Repeat until |BSM(σ) - Market_Price| < tolerance

**Challenges:**
- May fail for deep out-of-the-money options
- Requires market price > intrinsic value (else arbitrage)
- Numerically unstable near expiration

### 3.4 Interpreting Implied Volatility

- **IV = 15%**: Low volatility expected (calm market)
- **IV = 30%**: Moderate volatility (typical for large-cap stocks)
- **IV = 60%+**: High volatility expected (uncertain event upcoming)

**Important**: IV is forward-looking (market expectation) whereas historical volatility is backward-looking (realized).

---

## 4. Understanding the Greeks

**Greeks** measure the sensitivity of an option's price to various factors. They are essential for risk management and strategy construction.

### 4.1 Delta (Δ)

**Definition**: ∂V/∂S (sensitivity to stock price)

**Range**: 
- Call: 0 to 1
- Put: -1 to 0

**Interpretation**:
- Delta = 0.60 means the option gains ~$0.60 for each $1 increase in stock price
- Also approximates the probability of finishing in-the-money (risk-neutral)
- Used as a **hedge ratio**: to hedge 100 calls, short 60 shares (if delta = 0.60)

**Behavior**:
- Increases from 0 (OTM) to 1 (ITM) for calls
- ATM calls have delta ≈ 0.5
- Approaches 1.0 as S >> K (deep ITM)

### 4.2 Gamma (Γ)

**Definition**: ∂²V/∂S² = ∂Δ/∂S (rate of change of delta)

**Range**: Always ≥ 0 for long options

**Interpretation**:
- Measures **convexity** of the option value
- High gamma means delta changes rapidly as stock moves
- **Risk**: Long gamma is good (convex payoff), short gamma is dangerous (convex losses)

**Behavior**:
- Peaks at ATM
- Decreases as you move OTM or ITM
- Increases as expiration approaches (for ATM options)

**Why it matters**: If you delta-hedge, gamma tells you how quickly your hedge becomes stale. High gamma requires frequent rebalancing.

### 4.3 Vega (ν)

**Definition**: ∂V/∂σ (sensitivity to volatility)

**Range**: Always > 0 for long options

**Interpretation**:
- Vega = 20 means the option gains $20 for a 1% increase in volatility (e.g., 20% → 21%)
- **Long options have positive vega**: benefit from volatility increases
- **Short options have negative vega**: benefit from volatility decreases

**Behavior**:
- Peaks at ATM
- Increases with time to expiration (longer options more sensitive to vol)
- Zero at expiration

**Trading implications**: Buying options before expected volatility spikes (earnings, events) can be profitable if IV increases more than time decay.

### 4.4 Theta (Θ)

**Definition**: ∂V/∂t (time decay, note: often negative)

**Range**: Usually < 0 for long options

**Interpretation**:
- Theta = -10 means the option loses $10 in value per year (divide by 365 for daily)
- **Time decay** accelerates as expiration approaches
- **Long options suffer theta**, short options earn it

**Behavior**:
- Most negative for ATM options near expiration
- Less negative for deep ITM/OTM
- Competes with vega: holding options costs theta but gains from vega if vol rises

### 4.5 Rho (ρ)

**Definition**: ∂V/∂r (sensitivity to interest rate)

**Interpretation**:
- Rho = 50 means option gains $50 for a 1% increase in interest rate (e.g., 5% → 6%)
- **Calls**: positive rho (benefit from higher rates)
- **Puts**: negative rho

**Practical importance**: Usually the least important Greek for short-dated options, but matters for LEAPs (long-term options) and in varying rate environments.

### 4.6 Greeks Summary Table

| Greek | Measures | Call Sign | Put Sign | Peaks At |
|-------|----------|-----------|----------|----------|
| Delta | Price sensitivity | + | - | ITM |
| Gamma | Delta sensitivity | + | + | ATM |
| Vega | Vol sensitivity | + | + | ATM |
| Theta | Time decay | - | - | ATM (near expiry) |
| Rho | Rate sensitivity | + | - | ITM |

---

## 5. Reading the Charts

### 5.1 Payoff Diagrams

**X-axis**: Stock price at expiration  
**Y-axis**: Profit/Loss (P&L)

**Key features**:
- **Breakeven point(s)**: Where P&L = 0
- **Maximum loss**: Lowest point on the curve
- **Maximum gain**: Highest point (may be unbounded)
- **Kinks**: Occur at strike prices

**Example**: Long call with K=100, premium=$5
- Max loss: $5 (if S ≤ 100)
- Breakeven: $105
- Max gain: Unlimited (as S → ∞)

### 5.2 Heatmaps

Heatmaps show how option price varies with **two parameters simultaneously**.

**Common types**:

1. **Price vs Volatility**: Shows how option value increases with both S and σ
   - Bright (high value): ITM and/or high vol
   - Dark (low value): OTM and low vol

2. **Price vs Time**: Shows time decay across different stock prices
   - Shows how theta erodes value as T decreases

**How to read**:
- X-axis: typically stock price or strike
- Y-axis: typically volatility or time
- Color intensity: option price or P&L

### 5.3 Monte Carlo Histograms

**Left plot - Terminal Stock Prices**:
- Shows distribution of simulated final prices
- Should be log-normally distributed under GBM
- Vertical lines mark S₀ (initial) and K (strike)

**Right plot - Option Payoffs**:
- Shows distribution of payoffs at expiration
- Often skewed (many zeros for OTM, large payoffs for ITM)
- Vertical lines mark BSM price and MC estimate

**What to look for**:
- MC estimate should be close to BSM for European options
- Wider distribution → higher volatility → higher option value

---

## 6. Option Strategies

### 6.1 Long Straddle

**Construction**: Long call + long put (same strike K, same expiration T)

**Payoff at expiration**:
```
P&L = |S_T - K| - (C + P)
```

Where C and P are the premiums paid.

**When to use**:
- Expect **large move** in either direction
- Before earnings, FDA approvals, elections
- When current IV is low relative to expected realized volatility

**Risk profile**:
- **Max loss**: Total premium paid (C + P), occurs when S_T = K
- **Breakeven points**: K ± (C + P)
- **Max gain**: Unlimited as S moves far from K

**Greeks**:
- **Delta ≈ 0**: Neutral to small price moves (delta_call + delta_put ≈ 0 at ATM)
- **Gamma > 0**: Benefits from large moves
- **Vega > 0**: Benefits from volatility increases (vega_call + vega_put)
- **Theta < 0**: Suffers time decay

**Key insight**: You're betting on **volatility**, not direction. Need a big move to overcome the premium paid.

### 6.2 Delta Hedging

**Purpose**: Neutralize sensitivity to small stock price movements

**Method**: For a long call with delta Δ, short Δ shares of stock

**Example**: 
- Long 1 call, delta = 0.60
- Short 0.60 shares
- Small changes in S have minimal impact on portfolio value

**Static vs Dynamic**:
- **Static hedge** (our implementation): Set once, not rebalanced
  - Only works for small moves near S₀
  - Gamma causes hedge to deteriorate as S moves
  
- **Dynamic hedge** (real trading): Continuously rebalanced
  - Maintains delta ≈ 0 as market moves
  - Costly due to transaction costs and frequent trading

**Why delta hedge?**:
- Market makers: Manage inventory risk while providing liquidity
- Traders: Isolate other exposures (e.g., want pure volatility play)
- Risk management: Reduce directional exposure

**Limitations**:
- Only hedges local (small) price moves
- Gamma and theta remain unhedged
- Requires continuous monitoring and rebalancing

### 6.3 Vega Hedging

**Purpose**: Neutralize sensitivity to volatility changes

**Method**: Combine options with offsetting vegas

**Example**: 
- Long 1 call at K₁, vega = +20
- Short 0.8 calls at K₂, vega = +25 each
- Net vega = 20 - 0.8(25) = 0

**When to use**:
- Want directional exposure (delta) without volatility risk
- Want gamma exposure without vega risk
- Managing multiple option positions

**Complexity**:
- Requires at least 2 options to solve for
- More complex than delta hedging (delta can be hedged with stock)
- Vegas change with S, σ, and T (need rebalancing)

**Real-world application**:
- Options traders often target specific vega exposures
- Volatility arbitrage: bet on realized vol vs implied vol
- Correlation trading in multi-asset portfolios

---

## 7. No-Arbitrage Principle

### 7.1 What is Arbitrage?

**Arbitrage** is a risk-free profit opportunity: buy low in one market, sell high in another simultaneously, with no net investment.

**Example**: If gold costs $1800/oz in New York and $1850/oz in London, buy in NY, sell in London, pocket $50 risk-free.

### 7.2 No-Arbitrage in Options

The **no-arbitrage principle** states that in efficient markets, arbitrage opportunities should not persist. This principle underlies all option pricing.

**Implications**:

1. **Put-call parity must hold**:
   ```
   C - P = S - K·e^(-rT)
   ```
   If violated, arbitrageurs will exploit it until prices adjust.

2. **Option price ≥ Intrinsic value**:
   - For calls: C ≥ max(S - K, 0)
   - For puts: P ≥ max(K - S, 0)
   - Otherwise, buy the option and exercise immediately for profit.

3. **Volatility smile consistency**: IVs across strikes must be consistent (no arbitrage across strikes).

### 7.3 Risk-Neutral Pricing

The BSM formula uses the **risk-neutral measure**: we price as if investors are risk-neutral (indifferent to risk).

**Key idea**: 
- In a no-arbitrage world, expected return on all assets equals r
- We can replicate option payoffs with stock + bond
- Option price = expected payoff under risk-neutral measure, discounted at r

**Important**: This doesn't mean investors are actually risk-neutral; it's a mathematical convenience that works because we can hedge away risk.

### 7.4 Practical Implications

- Market prices should approximately follow BSM (or other no-arbitrage models)
- Large deviations often indicate:
  - Market frictions (transaction costs, liquidity)
  - Model misspecification (e.g., volatility not constant)
  - Arbitrage opportunities (quickly exploited)

---

## 8. Putting It All Together

### 8.1 Workflow for Analysis

1. **Price options**: Use BSM for quick calculations, MC for exotic features
2. **Compute Greeks**: Understand risk exposures
3. **Solve for IV**: Compare options fairly across strikes/maturities
4. **Analyze strategies**: Combine options to achieve desired risk profile
5. **Hedge risks**: Use delta, gamma, vega hedging as appropriate
6. **Monitor**: Greeks change over time; rebalance as needed

### 8.2 Practical Considerations

**BSM limitations in practice**:
- Volatility is not constant (volatility smile/skew exists)
- Real markets have jumps (not continuous)
- Transaction costs and liquidity matter
- Interest rates vary
- Dividends exist

**Despite limitations**, BSM remains the industry standard baseline.

### 8.3 Further Learning

To deepen your understanding:
- **Stochastic calculus**: Rigorous treatment of SDEs and Itô's lemma
- **Numerical methods**: Advanced MC techniques (variance reduction, quasi-MC)
- **Market microstructure**: How options are actually traded
- **Volatility modeling**: GARCH, stochastic volatility (Heston), local volatility
- **Jump models**: Merton jump-diffusion, Lévy processes

---

## References and Further Reading

- **Hull, J.C.** (2018). *Options, Futures, and Other Derivatives* (10th ed.). Pearson.
- **Black, F. & Scholes, M.** (1973). "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*, 81(3), 637-654.
- **Glasserman, P.** (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
- **Wilmott, P.** (2006). *Paul Wilmott on Quantitative Finance* (2nd ed.). Wiley.

---

*This document provides a conceptual foundation for the methods implemented in this toolkit. For implementation details, see the source code and notebooks.*

