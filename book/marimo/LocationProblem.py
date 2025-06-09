import marimo

__generated_with = "0.13.15"
app = marimo.App()

with app.setup:
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import math
    import cvxpy as cvx


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
def _():
    # pick a bunch of random points
    pos = np.random.randn(1000, 2)

    # Create a scatter plot with plotly
    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=pos[:, 0], y=pos[:, 1], mode="markers", marker=dict(symbol="x", size=10)
        )
    )
    _fig.update_layout(
        title="Random Points",
        xaxis_title="x",
        yaxis_title="y",
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )

    _fig

    return (pos,)


@app.function
def minimize(objective, constraints=None):
    return cvx.Problem(cvx.Minimize(objective), constraints).solve()


@app.function
def location(pos):
    R, x = cvx.Variable(1), cvx.Variable(2)
    minimize(objective=R, constraints=[cvx.norm(row - x, 2) <= R for row in pos])
    return R.value, x.value


@app.cell
def _(pos):
    print(location(pos))
    return


@app.cell
def _(pos):
    # Create a scatter plot with a circle overlay using plotly
    _radius, _midpoint = location(pos)

    # Generate points for the circle
    _theta = np.linspace(0, 2 * math.pi, 1000)
    _circle_x = _radius * np.cos(_theta) + _midpoint[0]
    _circle_y = _radius * np.sin(_theta) + _midpoint[1]

    # Create the figure
    _fig = go.Figure()

    # Add the scatter points
    _fig.add_trace(
        go.Scatter(
            x=pos[:, 0],
            y=pos[:, 1],
            mode="markers",
            marker=dict(symbol="x", size=10),
            name="Points",
        )
    )

    # Add the circle
    _fig.add_trace(
        go.Scatter(
            x=_circle_x,
            y=_circle_y,
            mode="lines",
            line=dict(color="red", width=2),
            name="Minimum Circle",
        )
    )

    # Update layout
    _fig.update_layout(
        title="Minimum Circle Containing All Points",
        xaxis_title="x",
        yaxis_title="y",
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )

    _fig

    return


@app.cell
def _():
    mo.md(
        r"""
    # Summary

    - Each constraint $\rVert{\mathbf{x}-\mathbf{c}}\lVert_2 < R$ represents a cone. Feasible domain is the intersection of all cones.

    - It is trivial to generalize (but not to plot) for points in higher dimensional spaces.

    - However, all of this fails once we can construct multiple circles.
    """
    )
    return


if __name__ == "__main__":
    app.run()
