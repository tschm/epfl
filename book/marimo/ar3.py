"""Module for AR and EWMA Estimators as Functions."""

import marimo

__generated_with = "0.17.8"
app = marimo.App()

with app.setup:
    import marimo as mo
    import plotly.graph_objects as go
    import polars as pl

    from epfl.signals import (
        build_ewma_matrix,
        compute_positions_profit,
        convolve_returns,
        get_vol_adjusted_returns,
        pacf_signal,
        penalized_weights,
    )

    # Load SPX returns
    frame = pl.read_csv(mo.notebook_location() / "public" / "SPX_Index.csv")
    frame = frame.with_columns([pl.col("price").pct_change().alias("ret")]).drop_nulls(["ret"])


# -----------------------------
# Marimo Cells
# -----------------------------
@app.cell
def _():
    mo.md(
        r"""
# AR and EWMA Estimators
This notebook has been refactored to use functions for clarity and reusability.
        """
    )
    return


@app.cell
def _():
    _frame = get_vol_adjusted_returns(frame)
    r_vol = _frame.select(pl.col("ret_vola")).to_numpy().flatten()
    pacf_weights, pos_shift, cum_profit = pacf_signal(r_vol)
    print(pacf_weights, pos_shift, cum_profit)


@app.cell
def _():
    periods = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 192]
    w_matrix = build_ewma_matrix(periods)
    print(w_matrix)

    return periods, w_matrix


@app.cell
def _(periods, w_matrix):
    r = frame.select(pl.col("ret")).to_numpy().flatten()
    a, r_filtered, dates_filtered = convolve_returns(r, w_matrix, periods, frame)
    return a, r_filtered


@app.cell
def _(periods, a, r_filtered, w_matrix):
    # Compute regression weights and combined lag weights
    t_weights = penalized_weights(a, r_filtered, w_matrix, lambdas=[0.0, 5.0, 15.0])
    print(t_weights)

    # Print weights mapped to periods for readads
    _fig = go.Figure()
    _fig.add_trace(go.Bar(x=[str(p) for p in periods], y=t_weights[0.0]))
    _fig.update_layout(
        title="Sparse Regression Weights by Period",
        xaxis_title="Period",
        yaxis_title="Weight",
        template="plotly_white",
    )
    _fig.show()

    return t_weights


@app.cell
def _(r, t_weights):
    return compute_positions_profit(r, t_weights)


@app.cell
def _():
    mo.md(
        r"""
## Summary
- Functions modularize computations for AR and EWMA estimators.
- Clean separation between signal generation, regression, and profit calculation.
- Easy to extend for new datasets or parameters.
        """
    )
    return


if __name__ == "__main__":
    app.run()
