import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _():
    import matplotlib

    import numpy as np
    import matplotlib.pyplot as plt

    matplotlib.style.use("ggplot")
    return np, plt


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Location problem

    We want to find the smallest circle such that $n$ points are all contained in it.
    """
    )
    return


@app.cell
def _(np, plt):
    # pick a bunch of random points
    pos = np.random.randn(1000, 2)

    plt.scatter(pos[:, 0], pos[:, 1], s=50, marker="x")
    plt.xlabel("x"), plt.ylabel("y")
    plt.axis("equal")
    plt.show()
    return (pos,)


@app.cell
def _():
    # solution with cvxpy
    from cvx.util import cvx, minimize

    def location(pos):
        R, x = cvx.Variable(1), cvx.Variable(2)
        minimize(objective=R, constraints=[cvx.norm(row - x, 2) <= R for row in pos])
        return R.value, x.value

    return (location,)


@app.cell
def _(location, pos):
    print(location(pos))
    return


@app.cell
def _(location, np, plt, pos):
    import math

    plt.scatter(pos[:, 0], pos[:, 1], s=50, marker="x")
    plt.xlabel("x"), plt.ylabel("y")
    plt.axis("equal")

    radius, midpoint = location(pos)

    c = np.array(
        [
            [radius * np.cos(a) + midpoint[0], radius * np.sin(a) + midpoint[1]]
            for a in np.linspace(0, 2 * math.pi, 1000)
        ]
    )
    plt.plot(c[:, 0], c[:, 1], "r")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Summary

    - Each constraint $\rVert{\mathbf{x}-\mathbf{c}}\lVert_2 < R$ represents a cone. Feasible domain is the intersection of all cones.

    - It is trivial to generalize (but not to plot) for points in higher dimensional spaces.

    - However, all of this fails once we can construct multiple circles.
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
