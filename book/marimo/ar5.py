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
    import plotly.graph_objects as go
    import polars as pl

    from epfl.signals import convolution, exp_weights

    frame = pl.read_csv(mo.notebook_location() / "public" / "SPX_Index.csv")
    frame = frame.with_columns([pl.col("price").pct_change().alias("ret")]).drop_nulls(["ret"])
    r = frame.select(pl.col("ret")).to_numpy().flatten()

    from regress import AR, LinearRegression

    lp = LinearRegression(r)
    arcorr = AR()

    periods = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 192]
    n_lags = 200
    dates = frame["date"].to_list()

    for i, p in enumerate(periods):
        _weights = exp_weights(m=p, n=n_lags)
        # generate a column for the AR matrix
        arcorr[p] = _weights
        # generate a column for the LP matrix
        lp[p] = convolution(r, _weights)


# --- Penalized regression with total variation ---
@app.cell
def _():
    lambdas = [0.0, 1.0, 5.0, 15.0, 20, 30]

    _lp_shift = lp.shift(lag=1)

    fig3 = go.Figure()
    fig3.update_layout(title="Expected returns for convolution")

    for lamb in lambdas:
        weights = arcorr.matrix @ _lp_shift.fit(lamb=lamb)
        profit = arcorr.profit(r, weights)
        fig3.add_trace(go.Scatter(x=frame["date"], y=profit, mode="lines", name=f"Lambda {lamb}"))

    fig3.show()


if __name__ == "__main__":
    app.run()
