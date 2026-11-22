# Dynamic Risk Parity Research

A reproducible study of dynamic **equal risk contribution (ERC)** portfolios versus equal-weight, inverse-volatility, and long-only minimum-variance portfolios.

## Research question
Does a rolling covariance-based risk-parity allocation improve risk-adjusted, cost-adjusted portfolio behavior compared with simpler baselines in changing correlation and volatility regimes?

## Installation and tests
```bash
python -m pip install -e '.[test]'
python -m pytest -q
risk-parity --days 850 --output results
```

Run on daily split-adjusted close prices in a CSV with a date column and at least two asset columns:
```bash
risk-parity --csv my_prices.csv --lookback 126 --rebalance-every 21 --cost-bps 5 --output real-results
```

Outputs: returns.csv, weights.csv, metrics.json, equity_curves.png.

## Methods
- Equal-weight benchmark
- Inverse-volatility weights proportional to 1/sigma
- Equal risk contribution: RC_i = w_i(Sigma w)_i / sqrt(w^T Sigma w)
- Long-only minimum variance
- Rolling covariance shrunk to diagonal
- Periodic rebalancing with drifting weights and proportional transaction costs

## Evaluation
Allocations at close t are based on observations no later than t, applied to the next close-to-close return. This is a **research approximation** because trading at the same close after observing it is not executable. Execution prices, spreads, market impact, financing, taxes, survivorship bias and stale pricing require further treatment. Synthetic returns are not empirical proof of profitability.

All strategies share the same out-of-sample interval. The 12 automated tests exercise risk budgeting, covariance regularization, constraints, costs, portfolio drift, data validation and prefix invariance.

MIT License. See LICENSE.
