"""Interactive Marimo Notebook for AR and EWMA Estimators.

- Load SPX returns and compute volatility-adjusted returns
- Construct AR signals (PACF-based)
- Construct EWMA signals
- Perform regression and penalized LASSO-style estimation
- Compute positions and cumulative PnL
- Interactive Plotly dashboards
"""

import marimo

__generated_with = "0.17.8"
app = marimo.App()

with app.setup:
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    import statsmodels.tsa.stattools as sts
    from numpy.linalg import lstsq
    from plotly.subplots import make_subplots

    from epfl.signals import ar, convolution, exp_weights

    frame = pl.read_csv(mo.notebook_location() / "public" / "SPX_Index.csv")
    frame = frame.with_columns([pl.col("price").pct_change().alias("ret")]).drop_nulls(["ret"])

    from regress import LinearRegression

    lp = LinearRegression(frame.select(pl.col("ret")).to_numpy())


# --- Introduction ---
@app.cell
def _():
    mo.md(
        r"""
# AR and EWMA Estimators for SPX Returns

This notebook demonstrates autoregressive (AR) models and exponentially weighted moving averages (EWMA)
for constructing trading signals.

**Key ideas:**
- AR prediction using PACF
- EWMA signals with varying decay periods
- Regression and penalized regression (LASSO-style)
- Volatility-adjusted returns and cumulative PnL
        """
    )
    return


# --- Volatility-adjusted returns ---
@app.cell
def _():
    r = frame["ret"].to_numpy()
    r_vol = r / np.std(r)
    return r, r_vol, frame["date"].to_list()


# --- PACF-based AR signal and cumulative PnL ---
@app.cell
def _(r_vol, dates):
    nlags = 100
    pacf_weights = sts.pacf(r_vol, nlags=nlags)[1:]
    _pos = convolution(r_vol, pacf_weights)

    # Shift to avoid lookahead
    pos_shift = np.roll(_pos, 1)
    pos_shift[0] = 0.0

    # Cumulative PnL
    cum_profit = np.cumsum(r_vol * pos_shift)

    # Interactive Plotly dashboard
    _fig = make_subplots(rows=2, cols=1, subplot_titles=("PACF Weights", "Cumulative PnL"))

    _fig.add_trace(go.Bar(x=list(range(1, nlags + 1)), y=pacf_weights, name="PACF"), row=1, col=1)

    _fig.add_trace(go.Scatter(x=dates, y=cum_profit, mode="lines", name="Cumulative PnL"), row=2, col=1)

    _fig.update_layout(height=700, width=900, template="plotly_white")
    _fig.show()

    return pacf_weights, pos_shift, cum_profit


# --- EWMA weight matrix for multiple periods ---
@app.cell
def _():
    periods = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 192]
    n_lags = 200
    w_matrix = np.column_stack([exp_weights(m=p, n=n_lags) for p in periods])
    return periods, w_matrix


# --- Convolve returns with EWMA signals ---
@app.cell
def _(periods, w_matrix, r, frame):
    dates = frame["date"].to_list()

    # Convolve each weight vector
    a_full = np.column_stack([convolution(r, w_matrix[:, j]) for j in range(len(periods))])

    # Shift by 1 to avoid lookahead
    a_shift = np.roll(a_full, 1, axis=0)
    a_shift[0, :] = np.nan

    # Remove NaN rows
    mask = ~np.any(np.isnan(a_shift), axis=1)
    a = a_shift[mask]
    r_filtered = r[mask]
    dates_filtered = [d for d, m in zip(dates, mask) if m]

    # Plot a few representative periods
    sample_periods = [2, 16, 64]
    _fig = go.Figure()
    for sp in sample_periods:
        if sp in periods:
            _j = periods.index(sp)
            _fig.add_trace(
                go.Scatter(
                    x=dates_filtered,
                    y=a[:, _j],
                    mode="lines",
                    name=f"Period {sp}",
                    hovertemplate="Date %{x}<br>Value %{y:.4f}",
                )
            )
    _fig.update_layout(
        title="Convoluted Return Time Series (Shifted by 1)",
        xaxis_title="Date",
        yaxis_title="Convolved Signal",
        template="plotly_white",
        height=500,
        width=900,
    )
    _fig.show()

    return a, r_filtered, dates_filtered


# --- Naive regression to find period weights ---
@app.cell
def _(periods, a, r_filtered, W_matrix):
    weights = lstsq(a, r_filtered, rcond=None)[0]
    combined_weights = W_matrix @ weights

    # Plot bar chart of period weights
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=[str(p) for p in periods], y=weights))
    fig1.update_layout(
        title="Regression Weights by Period", xaxis_title="Period", yaxis_title="Weight", template="plotly_white"
    )
    fig1.show()

    # Plot combined lag weights
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=list(range(1, len(combined_weights) + 1)), y=combined_weights, mode="lines"))
    fig2.update_layout(title="Combined Lag Weights", xaxis_title="Index", yaxis_title="Weight", template="plotly_white")
    fig2.show()

    return weights, combined_weights


# --- Penalized regression with total variation ---
@app.cell
def _(W_matrix, A, r_filtered):
    lambdas = [0.0, 5.0, 15.0]
    t_weights = {_lamb: W_matrix @ ar(A, r_filtered, lamb=_lamb) for _lamb in lambdas}

    # Plot weight distributions
    _fig = go.Figure()
    for _lamb in lambdas:
        _fig.add_trace(
            go.Scatter(
                x=list(range(1, len(t_weights[_lamb]) + 1)), y=t_weights[_lamb], mode="lines", name=f"Lambda {_lamb}"
            )
        )
    _fig.update_layout(
        title="Weight Distribution for Different Lambda Values",
        xaxis_title="Index",
        yaxis_title="Weight",
        width=1200,
        height=400,
        template="plotly_white",
    )
    _fig.show()

    return t_weights


# --- Compute positions and cumulative PnL for each lambda ---
@app.cell
def _(r, t_weights, dates):
    _pos = {_lamb: convolution(r, t_weights[_lamb]) for _lamb in t_weights}
    _pos = {_lamb: 1e6 * (v / np.std(v)) for _lamb, v in _pos.items()}

    profit = {}
    for _lamb, v in _pos.items():
        v_shift = np.roll(v, 1)
        v_shift[0] = 0.0
        profit[_lamb] = np.cumsum(r * v_shift)

    # Plot cumulative profit
    _fig = go.Figure()
    for _lamb in [0.0, 5.0, 15.0]:
        _fig.add_trace(go.Scatter(x=dates, y=profit[_lamb], mode="lines", name=f"Lambda {_lamb}"))
    _fig.update_layout(
        title="Cumulative Profit for Different Lambda Values",
        xaxis_title="Date",
        yaxis_title="Profit",
        template="plotly_white",
        height=500,
        width=900,
    )
    _fig.show()

    return _pos, profit


# --- Summary markdown ---
@app.cell
def _():
    mo.md(
        r"""
## Summary

- AR and EWMA models allow constructing predictive signals from historical returns.
- PACF gives insight into optimal lags for AR models.
- EWMA signals can be combined via regression, with optional penalization (total variation / LASSO).
- Volatility-adjusted returns and shifting positions prevent lookahead bias.
- Interactive dashboards allow exploring weights, convolutions, and cumulative PnL.
        """
    )
    return


if __name__ == "__main__":
    app.run()
