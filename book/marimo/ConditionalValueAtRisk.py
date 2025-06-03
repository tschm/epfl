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

    return (np,)


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
    r = np.array([-1.0, 2.0, 3.0, 2.0, 4.0, 2.0, 0.0, 1.0, -2.0, -2.0])
    _x = np.linspace(start=-1.0, stop=5.0, num=1000)
    _v = np.array([f(gamma=g, returns=r, alpha=0.80) for g in _x])

    # Uncomment to show the plot
    # plt.plot(_x, _v), plt.grid(True), plt.xlabel('$\gamma$'), plt.ylabel('$f$')
    # plt.title('Conditional value at risk as global minimum of a function f')
    # plt.axis([0, 5, 3, 6])
    # plt.show()
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
    R = [-1.0, 2.0, 3.0, 2.0, 4.0, 2.0, 0.0, 1.0, -2.0, -2.0]

    n = len(R)
    # We are interested in CVaR for alpha=0.80, e.g. what's the mean of the 20% of the biggest losses
    alpha = 0.80

    # introduce the variable for the var
    gamma = cvx.Variable(1)
    cvar = minimize(
        objective=gamma + 1.0 / int(n * (1 - alpha)) * cvx.sum(cvx.pos(R - gamma))
    )

    print(1.0 / (n * (1 - alpha)))
    print(f"A minimizer of f (<= VaR):  {gamma.value}")
    print(f"Minimum of f (== CVaR):     {cvar}")

    x = cvx.sum_largest(R, k=int(n * (1 - alpha)))
    print(x.value)
    return


@app.cell
def _(np):
    from cvx.util import minimize, cvx

    # take some random return data
    _R = np.random.randn(2500, 100)
    _n, _m = _R.shape

    # We are interested in CVaR for alpha=0.95, e.g. what's the mean of the 5% of the biggest losses
    _alpha = 0.95
    _k = int(_n * (1 - _alpha))

    _gamma, _w = (cvx.Variable(1), cvx.Variable(_m))
    constraints = [0 <= _w, cvx.sum(_w) == 1]

    obj = cvx.Minimize(_gamma + cvx.sum(cvx.pos(_R @ _w - _gamma)) / _k)
    _cvar = cvx.Problem(objective=obj, constraints=constraints).solve()
    print(f"CVaR: {_cvar}")

    obj = cvx.Minimize(cvx.sum_largest(_R @ _w, k=_k) / _k)
    cvar2 = cvx.Problem(objective=obj, constraints=constraints).solve()
    print(f"CVaR 2: {cvar2}")

    # Uncomment to show the plot
    # plt.hist(R @ weights, bins=100)
    # plt.axis([-0.4, 0.4, 0, 150])
    # plt.title("CVaR {0}".format(cvar))
    # plt.show()
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
