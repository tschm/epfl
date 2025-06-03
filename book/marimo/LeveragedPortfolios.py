import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _():
    import numpy as np

    return (np,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Leveraged Portfolios

    https://en.wikipedia.org/wiki/130%E2%80%9330_fund

    **Thomas Schmelzer**
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## A 130/30 Equity Portfolio

    - Allocate capital $C=1$. Sell short at most $c = 0.3$ to finance a long position of $1 + c$.
    - Universe of $n$ assets.
    """
    )
    return


@app.cell
def _(mo):
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


@app.cell
def _(np):
    from cvx.util import cvx, maximize

    # make some random data, e.g. cov-matrix and expected returns
    n = 100
    c = 0.9
    C = c * np.ones((n, n)) + (1 - c) * np.eye(n)
    mu = 0.05 * np.sin(range(0, n))
    # maximal volatility and leverage...
    sigma_max = 1.0
    excess = 0.3

    x = cvx.Variable(n)
    constraints = [
        cvx.sum(x) == 1,
        cvx.norm(x, 1) <= 1 + 2 * excess,
        cvx.quad_form(x, C) <= sigma_max * sigma_max,
    ]
    maximize(objective=x.T @ mu, constraints=constraints)
    f = x.value

    print("Sum of positive weights: {}".format(np.sum(f[f > 0])))
    print("Sum of negative weights: {}".format(np.sum(f[f < 0])))
    print("Sum of all weights:      {}".format(np.sum(f)))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Summary

    - Leverage is here a constraint for the $1$-norm of the weight vector.

    - Note that we do not solve two problems for the short and long part of the portfolio.
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
