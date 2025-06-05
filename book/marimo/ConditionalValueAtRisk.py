import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
    # The Conditional Value at Risk

    https://en.wikipedia.org/wiki/Expected_shortfall

    **Thomas Schmelzer**
    """
    )
    return


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    import cvxpy as cvx

    return np, cvx, go


@app.cell
def _(mo):
    mo.md(
        r"""
    The $\alpha=0.99$ tail of a loss distribution
    -----------------------------------------------
    <img src="talk/tail.jpg" style="margin-left:auto; margin-right:auto; display:block">
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    * In this talk we assume losses are positive. Larger losses, more pain... We want negative losses!

    * The value at risk $\mathtt{VaR}_{\alpha}$ at level $\alpha$ is (the smallest) loss such that $\alpha \%$ of losses are smaller than $\mathtt{VaR}_{\alpha}$.

    * This does not say anything about the magnitude of the losses larger than the $\mathtt{VaR}_{\alpha}$. We can only make statements about their number: $n(1 - \alpha)$

    * The $\mathtt{VaR}_{\alpha}$ has some sever mathematical flaws. It's not sub-additive, it's not convex. It's broken! However, the regulator embraced it.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    * We compute the mean of the largest $n(1-\alpha)$ entries of a vector (or a optimal linear combination of vectors) without ever sorting the entries of any vector.

    * The resulting convex program is linear.

    * This mean is called Conditional Value at Risk $\mathtt{CVaR}_{\alpha}$ and is an upper bound for the Value at Risk $\mathtt{VaR}_{\alpha}$.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Given a vector $\mathbf{r}$ we introduce a free variable $\gamma$ and define the function $f$ as:
    \begin{eqnarray}
    f(\gamma) &=& \gamma + \frac{1}{n\,(1-\alpha)}\sum (r_i - \gamma)^{+}
    \end{eqnarray}
    This is a continuous and convex function (in $\gamma$). The first derivative is:
    $$
    f^{'}(\gamma) = 1 - \frac{\#\left\{r_i \geq \gamma\right\}}{n\,(1-\alpha)}
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    If $\gamma$ such that $\#\{r_i \geq \gamma\}=n\,(1-\alpha)$:
    - $\gamma$ is a minimizer of $f$.
    - $f(\gamma) =\mathtt{CVaR}_\alpha(\mathbf{r})$.

    In particular:

    * $f(\mathtt{VaR}_\alpha(\mathbf{r})) = \mathtt{CVaR}_\alpha(\mathbf{r})$.
    """
    )
    return


@app.cell
def _(np):
    def f(gamma, returns, alpha=0.99):
        excess = returns - gamma
        return gamma + 1.0 / (len(returns) * (1 - alpha)) * excess[excess > 0].sum()

    # note that cvar = (3+4)/2  and var = ? ... depends on your definition. 2?, 3?, 2.5?
    _r = np.array([-1.0, 2.0, 3.0, 2.0, 4.0, 2.0, 0.0, 1.0, -2.0, -2.0])
    _x = np.linspace(start=-1.0, stop=5.0, num=1000)
    _v = np.array([f(gamma=g, returns=_r, alpha=0.80) for g in _x])

    # Uncomment to show the plot
    # _fig = go.Figure()
    # _fig.add_trace(go.Scatter(x=_x, y=_v, mode='lines'))
    # _fig.update_layout(
    #     title='Conditional value at risk as global minimum of a function f',
    #     xaxis_title='$\gamma$',
    #     yaxis_title='$f$',
    #     xaxis_range=[0, 5],
    #     yaxis_range=[3, 6],
    #     grid=dict(rows=1, columns=1, pattern='independent')
    # )
    # _fig.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Before (using conic reformulation of the $x^+$ function):

    * \begin{align}\mathtt{CVaR}(\mathbf{r})=\min_{\gamma \in \mathbb{R}, \mathbf{t} \in \mathbb{R}^n} \,&\, \gamma + \frac{1}{n\,(1-\alpha)}\sum t_i\\
    \text{s.t. }&t_i \geq r_i - \gamma \\
                &\mathbf{t}\geq 0
    \end{align}

    Now

    * http://www.cvxpy.org/en/latest/tutorial/functions/, in particular the $x^{+} = \max\{0,x\}$
    """
    )
    return


@app.cell
def _(cvx, minimize):
    _R = [-1.0, 2.0, 3.0, 2.0, 4.0, 2.0, 0.0, 1.0, -2.0, -2.0]

    _n = len(_R)
    # We are interested in CVaR for alpha=0.80, e.g. what's the mean of the 20% of the biggest losses
    _alpha = 0.80

    # introduce the variable for the var
    _gamma = cvx.Variable(1)
    _cvar = minimize(
        objective=_gamma + 1.0 / int(_n * (1 - _alpha)) * cvx.sum(cvx.pos(_R - _gamma))
    )

    print(1.0 / (_n * (1 - _alpha)))
    print(f"A minimizer of f (<= VaR):  {_gamma.value}")
    print(f"Minimum of f (== CVaR):     {_cvar}")

    _x = cvx.sum_largest(_R, k=int(_n * (1 - _alpha)))
    print(_x.value)
    return


@app.cell
def _(np, cvx):
    def minimize(objective, constraints=None):
        return cvx.Problem(cvx.Minimize(objective), constraints).solve()

    # take some random return data
    _R = np.random.randn(2500, 100)
    _n, _m = _R.shape

    # We are interested in CVaR for alpha=0.95, e.g. what's the mean of the 5% of the biggest losses
    _alpha = 0.95
    _k = int(_n * (1 - _alpha))

    _gamma, _w = (cvx.Variable(1), cvx.Variable(_m))
    _constraints = [0 <= _w, cvx.sum(_w) == 1]

    _obj = cvx.Minimize(_gamma + cvx.sum(cvx.pos(_R @ _w - _gamma)) / _k)
    _cvar = cvx.Problem(objective=_obj, constraints=_constraints).solve()
    print(f"CVaR: {_cvar}")

    _obj2 = cvx.Minimize(cvx.sum_largest(_R @ _w, k=_k) / _k)
    _cvar2 = cvx.Problem(objective=_obj2, constraints=_constraints).solve()
    print(f"CVaR 2: {_cvar2}")

    # Uncomment to show the plot
    # _fig = go.Figure()
    # _fig.add_trace(go.Histogram(x=_R @ _w.value, nbinsx=100))
    # _fig.update_layout(
    #     title=f"CVaR {_cvar}",
    #     xaxis_title="Value",
    #     yaxis_title="Frequency",
    #     xaxis_range=[-0.4, 0.4],
    #     yaxis_range=[0, 150]
    # )
    # _fig.show()
    return cvx, minimize


@app.cell
def _(mo):
    mo.md(
        r"""
    Summary
    -------

    * We could compute the $\mathtt{CVaR}$ for a vector of length $n$ by solving a linear program.

    * We do not need to sort the elements nor do we need to know the Value at Risk $\mathtt{VaR}$.

    In practice the vector $\mathbf{r}$ is not given. Rather we have $m$ assets and try to find a linear combination of their corresponding return vectors such that the resulting portfolio has minimal Conditional Value at Risk.
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
