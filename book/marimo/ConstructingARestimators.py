import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _():
    import matplotlib

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    matplotlib.style.use("ggplot")
    return np, pd, plt


@app.cell
def _(mo):
    mo.md(
        r"""
    # Constructing estimators

    https://en.wikipedia.org/wiki/Autoregressive_model

    **Thomas Schmelzer**
    """
    )
    return


@app.cell
def _(mo):
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
def _(pd):
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
def _(pd):
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
def _(mo):
    mo.md(
        r"""
    ## Looking only at the last two returns might be a bit ...

    Is it a good idea to have $n=200$ free parameters?
    """
    )
    return


@app.cell
def _(mo, pd, plt):
    import statsmodels.tsa.stattools as sts

    # generate random returns
    r = (
        pd.read_csv(
            mo.notebook_location() / "data" / "SPX_Index.csv",
            index_col=0,
            header=None,
            parse_dates=True,
        )
        .pct_change()
        .dropna()[1]
    )
    # let's compute the optimal convolution!
    _weights = sts.pacf(r, nlags=200)
    pd.Series(data=_weights[1:]).plot(kind="bar")
    plt.show()
    print(r)
    return (r,)


@app.cell
def _(plt, r, weights):
    # The trading system!
    _pos = convolution(r, weights[1:])
    _pos = 1e6 * (_pos / _pos.std())
    # profit = return[today] * position[yesterday]
    (r * _pos.shift(1)).cumsum().plot()
    plt.xlabel("Time"), plt.ylabel("Profit")
    plt.show()
    return


@app.cell
def _(mo):
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
def _(np, pd, plt):
    def exp_weights(m, n=100):
        x = np.power(1.0 - 1.0 / m, range(1, n + 1))
        S = np.linalg.norm(x)
        return x / S

    pd.Series(exp_weights(m=16, n=40)).plot(kind="bar")
    plt.show()
    return (exp_weights,)


@app.cell
def _(exp_weights, pd, plt):
    periods = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 192]
    # matrix of weights
    W = pd.DataFrame({period: exp_weights(m=period, n=200) for period in periods})
    W.plot()
    plt.show()
    return W, periods


@app.cell
def _(W, pd, periods, plt, r):
    # each column of A is a convoluted return time series
    A = pd.DataFrame({period: convolution(r, W[period]).shift(1) for period in periods})

    A = A.dropna(axis=0)
    r_filtered = r[A.index].dropna()

    A[[2, 16, 64]].plot()
    plt.show()
    return A, r_filtered


@app.cell
def _(mo):
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
def _(A, W, pd, periods, plt, r_filtered):
    from numpy.linalg import lstsq

    # sometimes you don't need to use MOSEK :-)
    weights = pd.Series(index=periods, data=lstsq(A.values, r_filtered.values)[0])
    print(weights)
    (W * weights).sum(axis=1).plot(kind="bar")
    (W * weights).sum(axis=1).plot()
    plt.show()
    return (weights,)


@app.cell
def _(mo):
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


@app.cell
def _(np, pd):
    from cvx.util import cvx, minimize

    def mean_variation(ts):
        return ts.diff().abs().mean()

    def ar(A, r, lamb=0.0):
        # introduce the variable for the var
        x = cvx.Variable(A.shape[1])
        D = np.diag(A.apply(mean_variation))
        minimize(objective=cvx.norm(A.values @ x - r, 2) + lamb * cvx.norm(D @ x, 1))
        return pd.Series(index=A.keys(), data=x.value)

    return (ar,)


@app.cell
def _(A, W, ar, pd, plt, r_filtered):
    t_weight = pd.DataFrame(
        {
            lamb: (W * ar(A, r_filtered.values, lamb=lamb)).sum(axis=1)
            for lamb in [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 12.0, 15.0]
        }
    )
    t_weight[[0.0, 5.0, 15.0]].plot(figsize=(30, 10))
    plt.show()
    return (t_weight,)


@app.cell
def _(pd, plt, r, t_weight):
    # for lamb in sorted(t_weight.keys()):

    pos = pd.DataFrame(
        {lamb: convolution(r, t_weight[lamb]) for lamb in t_weight.keys()}
    )
    pos = 1e6 * (pos / pos.std())

    profit = pd.DataFrame(
        {lamb: (r * pos[lamb].shift(1)).cumsum() for lamb in pos.keys()}
    )
    profit[[0.0, 5.0, 15.0]].plot()
    plt.show()
    return


@app.cell
def _(mo):
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


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
