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

    from epfl.signals import convolution, exp_weights

    frame = pl.read_csv(mo.notebook_location() / "public" / "SPX_Index.csv")
    frame = frame.with_columns([pl.col("price").pct_change().alias("ret")]).drop_nulls(["ret"])

    r = frame.select(pl.col("ret")).to_numpy().flatten()

    from regress import AR, LinearRegression

    lp = LinearRegression(frame.select(pl.col("ret")).to_numpy())
    ar = AR()

    periods = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 192]
    n_lags = 200


# --- Convolve returns with EWMA signals ---
@app.cell
def _():
    for i, p in enumerate(periods):
        _weights = exp_weights(m=p, n=n_lags)
        ar[p] = _weights
        lp[p] = convolution(r, _weights)


# --- Penalized regression with total variation ---
@app.cell
def _():
    lambdas = [0.0, 5.0, 15.0]

    lp_shift = lp.shift(lag=1)

    # Plot weight distributions
    _fig1 = go.Figure()
    _fig1.update_layout(title="Coefficients for Lambda")
    _fig2 = go.Figure()
    _fig2.update_layout(title="Effective weights for convolution")
    _fig3 = go.Figure()
    _fig3.update_layout(title="Expected returns for convolution")
    _fig4 = go.Figure()
    _fig4.update_layout(title="Expected returns for AR")

    for lamb in lambdas:
        x = lp_shift.fit(lamb=lamb)
        t = ar.matrix @ np.array(list(x.values()))

        _p = 1e6 * ar.convolve(x=r, coeff=t)
        _p = np.roll(_p, 1)
        _p[0] = 0.0
        _profit = np.cumsum(r * _p)

        _fig1.add_trace(go.Bar(x=list(x.keys()), y=list(x.values()), name=f"Lamb={lamb}"))
        _fig2.add_trace(go.Scatter(y=t, mode="lines", name=f"Lamb={lamb}"))
        _fig3.add_trace(go.Scatter(x=frame["date"], y=_p, mode="lines", name=f"Lamb={lamb}"))
        _fig4.add_trace(go.Scatter(x=frame["date"], y=_profit, mode="lines", name=f"Lamb={lamb}"))

    _fig4.show()
    _fig3.show()
    _fig2.show()
    _fig1.show()


if __name__ == "__main__":
    app.run()
