import marimo

__generated_with = "0.13.15"
app = marimo.App()


with app.setup:
    import marimo as mo
    import cvxpy as cvx
    import numpy as np


@app.cell
def _():
    mo.md(
        r"""
    # Leveraged Portfolios

    https://en.wikipedia.org/wiki/130%E2%80%9330_fund

    **Thomas Schmelzer**
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    ## A 130/30 Equity Portfolio

    - Allocate capital $C=1$. Sell short at most $c = 0.3$ to finance a long position of $1 + c$.
    - Universe of $n$ assets.
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    \begin{align}\mathbf{x}^{*}=\arg\max_{\mathbf{x} \in \mathbb{R}^n}& \mu^{T}\mathbf{x}\\
    \text{s.t. } &\Sigma\,x_i=1\\
                 &\Sigma\,\lvert x_i\rvert \leq 1 + 2c\\
                 &\sqrt{\mathbf{x}^T\mathbf{C}\mathbf{x}} \leq \sigma_{\max}
    \end{align}
    """
    )
    return


@app.function
def maximize(objective, constraints=None):
    return cvx.Problem(cvx.Maximize(objective), constraints).solve()


@app.cell
def _():
    # make some random data, e.g. cov-matrix and expected returns
    _n = 100
    _c = 0.9
    _C = _c * np.ones((_n, _n)) + (1 - _c) * np.eye(_n)
    _mu = 0.05 * np.sin(range(0, _n))
    # maximal volatility and leverage...
    _sigma_max = 1.0
    _excess = 0.3

    _x = cvx.Variable(_n)
    _constraints = [
        cvx.sum(_x) == 1,
        cvx.norm(_x, 1) <= 1 + 2 * _excess,
        cvx.quad_form(_x, _C) <= _sigma_max * _sigma_max,
    ]
    maximize(objective=_x.T @ _mu, constraints=_constraints)
    _f = _x.value

    print("Sum of positive weights: {}".format(np.sum(_f[_f > 0])))
    print("Sum of negative weights: {}".format(np.sum(_f[_f < 0])))
    print("Sum of all weights:      {}".format(np.sum(_f)))
    return


@app.cell
def _():
    mo.md(
        r"""
    ## Summary

    - Leverage is here a constraint for the $1$-norm of the weight vector.

    - Note that we do not solve two problems for the short and long part of the portfolio.
    """
    )
    return


if __name__ == "__main__":
    app.run()
