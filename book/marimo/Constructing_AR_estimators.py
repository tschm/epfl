"""Module for constructing and analyzing autoregressive (AR) estimators.

This module demonstrates how to build autoregressive models for financial time series,
including techniques for parameter selection, convolution operations, and optimization
approaches for constructing robust estimators with regularization.
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cvxpy-base==1.7.5",
#     "marimo==0.18.4",
#     "numpy==2.3.5",
#     "pandas==2.3.3",
#     "plotly==6.5.0",
#     "statsmodels==0.14.6",
#     "clarabel==0.11.1",
# ]
# ///

import marimo

__generated_with = "0.13.15"
app = marimo.App()

with app.setup:
    import marimo as mo
    import pandas as pd
    import plotly.graph_objects as go
    import statsmodels.tsa.stattools as sts


@app.cell
def _():
    mo.md(
        r"""
    # Constructing estimators

    https://en.wikipedia.org/wiki/Autoregressive_model

    **Thomas Schmelzer**
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    a very common estimator is based on AR models (autoregressive)

    $$R_T = \sum_{i=1}^n w_i r_{T-i}$$

    Predict the (unknown) return $R_T$ using the last $n$ previous returns.
    **Attention**: You may want to use volatility adjusted returns, apply filters etc.

    How to pick the $n$ free parameters in $\mathbf{w}$? (Partial) autocorrelations?
    """
    )
    return


@app.function
def convolution(ts, weights):
    """Apply a convolution filter to a time series using specified weights.

    This function performs a one-sided convolution operation on a time series,
    which is useful for creating autoregressive models and moving averages.

    Args:
        ts: The time series data to filter (pandas Series or array-like).
        weights: The weights to use in the convolution filter.

    Returns:
        a filtered time series resulting from the convolution operation.
    """
    from statsmodels.tsa.filters.filtertools import convolution_filter

    return convolution_filter(ts, weights, nsides=1)


@app.cell
def _():
    _r = pd.Series([1.0, -2.0, 1.0, 1.0, 1.5, 0.0, 2.0])
    _weights = [2.0, 1.0]
    # trendfollowing == positive weights
    _x = pd.DataFrame()
    _x["r"] = _r
    _x["pred"] = convolution(_r, _weights)
    _x["before"] = _x["pred"].shift(1)
    print(_x)
    print(_x.corr())
    return


@app.cell
def _():
    # mean-reversion == negative weights
    _r = pd.Series([1.0, -2.0, 1.0, 1.0, 1.5, 0.0, 2.0])
    _weights = [-2.0, -1.0]
    _x = pd.DataFrame()
    _x["r"] = _r
    _x["pred"] = convolution(_r, _weights)
    _x["before"] = _x["pred"].shift(1)
    print(_x)
    print(_x.corr())
    return


@app.cell
def _():
    mo.md(
        r"""
    ## Looking only at the last two returns might be a bit ...

    Is it a good idea to have $n=200$ free parameters?
    """
    )
    return


@app.cell
def _():
    # generate random returns
    r = (
        pd.read_csv(
            mo.notebook_location() / "public" / "SPX_Index.csv",
            index_col=0,
            header=None,
            parse_dates=True,
        )
        .pct_change()
        .dropna()[1]
    )
    # let's compute the optimal convolution!
    weights = sts.pacf(r, nlags=200)

    # Create a bar chart with plotly
    _fig = go.Figure()
    _fig.add_trace(go.Bar(x=list(range(1, len(weights))), y=weights[1:]))
    _fig.update_layout(title="Partial Autocorrelation", xaxis_title="Lag", yaxis_title="PACF")
    _fig
    return r, weights


@app.cell
def _(r, weights):
    # The trading system!
    _pos = convolution(r, weights[1:])
    pos = 1e6 * (_pos / _pos.std())
    # profit = return[today] * position[yesterday]

    # Create a line chart with plotly
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=r.index, y=(r * pos.shift(1)).cumsum(), mode="lines"))
    _fig.update_layout(title="Cumulative Profit", xaxis_title="Time", yaxis_title="Profit")
    _fig

    # Create a line chart with plotly
    # derive simple per-lambda weights from PACF weights to visualize
    lambdas = [0.0, 5.0, 15.0]
    base = weights[1:]
    # simple scaling per lambda to keep notebook self-contained
    t_weights = {lam: base / (1.0 + lam) for lam in lambdas}

    _fig = go.Figure()
    for _lamb in lambdas:
        _fig.add_trace(
            go.Scatter(
                x=list(range(1, len(t_weights[_lamb]) + 1)),
                y=t_weights[_lamb],
                mode="lines",
                name=f"Lambda {_lamb}",
            )
        )
    _fig.update_layout(
        title="Weight Distribution for Different Lambda Values",
        xaxis_title="Index",
        yaxis_title="Weight",
        width=1200,
        height=400,
    )
    _fig.show()

    return t_weights


@app.cell
def _(r, t_weights):
    # for lamb in sorted(_t_weight.keys()):

    _pos = pd.DataFrame({_lamb: convolution(r, t_weights[_lamb]) for _lamb in t_weights})
    _pos = 1e6 * (_pos / _pos.std())

    _profit = pd.DataFrame({lamb: (r * _pos[lamb].shift(1)).cumsum() for lamb in _pos})

    # Create a line chart with plotly
    _fig = go.Figure()
    for _lamb in [0.0, 5.0, 15.0]:
        _fig.add_trace(go.Scatter(x=_profit.index, y=_profit[_lamb], mode="lines", name=f"Lambda {_lamb}"))
    _fig.update_layout(
        title="Cumulative Profit for Different Lambda Values",
        xaxis_title="Date",
        yaxis_title="Profit",
    )
    _fig.show()

    return _pos, _profit


@app.cell
def _():
    mo.md(
        r"""
    ## Summary

    - The problem of constructing an estimator is corresponds to tracking an index.
      The index is here a historic return time series.
      The **assets** are standard estimators.


    - Using the (mean) total variation of the signals can help to prefer slower signals rather than
      expensive fast signals.


    - Using a penalty induced by the $1$-norm (see LARS, LASSO) it is possible to establish a ranking
      amongst the indicators and construct them robustly.


    - It is possible to (vertical) stack the resulting systems to find optimal weights across a group of assets.
    """
    )
    return


if __name__ == "__main__":
    app.run()
