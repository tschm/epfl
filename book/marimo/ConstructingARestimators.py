import marimo

__generated_with = "0.13.15"
app = marimo.App()

with app.setup:
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import cvxpy as cvx
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
    A very common estimator is based on AR models (autoregressive)

    $$R_T = \sum_{i=1}^n w_i r_{T-i}$$

    Predict the (unknown) return $R_T$ using the last $n$ previous returns. **Attention**: You may want to use volatility adjusted returns, apply filters etc.

    How to pick the $n$ free parameters in $\mathbf{w}$? (Partial) autocorrelations?
    """
    )
    return


@app.function
def convolution(ts, weights):
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
    _fig.update_layout(
        title="Partial Autocorrelation", xaxis_title="Lag", yaxis_title="PACF"
    )
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
    _fig.update_layout(
        title="Cumulative Profit", xaxis_title="Time", yaxis_title="Profit"
    )
    _fig

    return pos


@app.cell
def _():
    mo.md(
        r"""
    ## Bias

    We assume the weights are exponentially decaying, e.g.

    $$w_i = \frac{1}{S}\lambda^i$$

    where $S$ is a suitable scaling constant and $\lambda = 1-1/N$. Note that $N \neq n$.

    **Everything** that is **not** an exponentially weighted moving average is **wrong**.
    """
    )
    return


@app.cell
def _():
    def exp_weights(m, n=100):
        x = np.power(1.0 - 1.0 / m, range(1, n + 1))
        S = np.linalg.norm(x)
        return x / S

    # Create a bar chart with plotly
    _weights = exp_weights(m=16, n=40)
    _fig = go.Figure()
    _fig.add_trace(go.Bar(x=list(range(1, len(_weights) + 1)), y=_weights))
    _fig.update_layout(
        title="Exponential Weights (m=16, n=40)",
        xaxis_title="Index",
        yaxis_title="Weight",
    )
    _fig

    return exp_weights


@app.cell
def _(exp_weights):
    _periods = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 192]
    # matrix of weights
    _W = pd.DataFrame({_period: exp_weights(m=_period, n=200) for _period in _periods})

    # Create a line chart with plotly
    _fig = go.Figure()
    for _period in _periods:
        _fig.add_trace(
            go.Scatter(
                x=list(range(1, 201)),
                y=_W[_period],
                mode="lines",
                name=f"Period {_period}",
            )
        )
    _fig.update_layout(
        title="Exponential Weights for Different Periods",
        xaxis_title="Index",
        yaxis_title="Weight",
    )
    _fig

    return _periods, _W


@app.cell
def _(_r, _periods, _W):
    # each column of A is a convoluted return time series
    _A = pd.DataFrame(
        {_period: convolution(_r, _W[_period]).shift(1) for _period in _periods}
    )

    _A = _A.dropna(axis=0)
    _r_filtered = _r[_A.index].dropna()

    # Create a line chart with plotly
    _fig = go.Figure()
    for _period in [2, 16, 64]:
        _fig.add_trace(
            go.Scatter(
                x=_A.index, y=_A[_period], mode="lines", name=f"Period {_period}"
            )
        )
    _fig.update_layout(
        title="Convoluted Return Time Series", xaxis_title="Date", yaxis_title="Value"
    )
    _fig

    return _A, _r_filtered


@app.cell
def _():
    mo.md(
        r"""
    ## (Naive) regression

    \begin{align}
    \mathbf{w}^{*}=\arg\min_{\mathbf{w} \in \mathbb{R}^m}& \rVert{\mathbf{A}\mathbf{w} - \mathbf{r}}\lVert_2
    \end{align}
    """
    )
    return


@app.cell
def _(_periods, _A, _r_filtered, _W):
    from numpy.linalg import lstsq

    # sometimes you don't need to use MOSEK :-)
    _weights = pd.Series(index=_periods, data=lstsq(_A.values, _r_filtered.values)[0])
    print(_weights)

    # Create bar chart
    _fig1 = go.Figure()
    _fig1.add_trace(go.Bar(x=_weights.index.astype(str), y=(_W * _weights).sum(axis=1)))
    _fig1.update_layout(
        title="Weights Distribution (Bar Chart)",
        xaxis_title="Period",
        yaxis_title="Weight",
    )
    _fig1.show()

    # Create line chart
    _fig2 = go.Figure()
    _fig2.add_trace(
        go.Scatter(
            x=list(range(1, len((_W * _weights).sum(axis=1)) + 1)),
            y=(_W * _weights).sum(axis=1),
            mode="lines",
        )
    )
    _fig2.update_layout(
        title="Weights Distribution (Line Chart)",
        xaxis_title="Index",
        yaxis_title="Weight",
    )
    _fig2.show()

    return _weights


@app.cell
def _():
    mo.md(
        r"""
    ## Mean variation

    We provide a few indicators. Avoid fast indicators. Prefer slower indicators as they induce less trading costs.
    Use the mean variation of the signal (convoluted returns here)

    $$f(\mathbf{x}) = \frac{1}{n}\sum{\lvert x_i - x_{i-1}\rvert}=\frac{1}{n}\rVert{\Delta \mathbf{x}}\lVert_1$$

    The $i$th column of $\mathbf{A}$ has a mean variation $d_i$. We introduce the diagonal penalty matrix $\mathbf{D}$ with $D_{i,i}=d_i$.

    $$\mathbf{w}^{*}=\arg\min_{\mathbf{w} \in \mathbb{R}^m} \lVert{\mathbf{Aw}-\mathbf{r}}\lVert_2 + \lambda \rVert{\mathbf{Dw}}\lVert_1$$
    """
    )
    return


@app.function
def minimize(objective, constraints=None):
    return cvx.Problem(cvx.Minimize(objective), constraints).solve()


@app.function
def mean_variation(ts):
    return ts.diff().abs().mean()


@app.function
def ar(A, r, lamb=0.0):
    # introduce the variable for the var
    x = cvx.Variable(A.shape[1])
    D = np.diag(A.apply(mean_variation))
    minimize(objective=cvx.norm(A.values @ x - r, 2) + lamb * cvx.norm(D @ x, 1))
    return pd.Series(index=A.keys(), data=x.value)


@app.cell
def _(_W, _A, _r_filtered):
    _t_weight = pd.DataFrame(
        {
            _lamb: (_W * ar(_A, _r_filtered.values, lamb=_lamb)).sum(axis=1)
            for _lamb in [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 12.0, 15.0]
        }
    )

    # Create a line chart with plotly
    _fig = go.Figure()
    for _lamb in [0.0, 5.0, 15.0]:
        _fig.add_trace(
            go.Scatter(
                x=list(range(1, len(_t_weight) + 1)),
                y=_t_weight[_lamb],
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

    return _t_weight


@app.cell
def _(_r, _t_weight):
    # for lamb in sorted(_t_weight.keys()):

    _pos = pd.DataFrame(
        {_lamb: convolution(_r, _t_weight[_lamb]) for _lamb in _t_weight.keys()}
    )
    _pos = 1e6 * (_pos / _pos.std())

    _profit = pd.DataFrame(
        {lamb: (_r * _pos[lamb].shift(1)).cumsum() for lamb in _pos.keys()}
    )

    # Create a line chart with plotly
    _fig = go.Figure()
    for _lamb in [0.0, 5.0, 15.0]:
        _fig.add_trace(
            go.Scatter(
                x=_profit.index, y=_profit[_lamb], mode="lines", name=f"Lambda {_lamb}"
            )
        )
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

    - The problem of constructing an estimator is corresponds to tracking an index. The index is here a historic return time series. The **assets** are standard estimators.


    - Using the (mean) total variation of the signals can help to prefer slower signals rather than expensive fast signals.


    - Using a penalty induced by the $1$-norm (see LARS, LASSO) it is possible to establish a ranking amongst the indicators and construct them robustly.


    - It is possible to (vertical) stack the resulting systems to find optimal weights across a group of assets.
    """
    )
    return


if __name__ == "__main__":
    app.run()
