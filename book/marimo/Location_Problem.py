"""Module for solving and visualizing the minimum enclosing circle problem.

This module demonstrates how to find the smallest circle that contains a set of points
in a plane, using convex optimization techniques. It includes visualization of the
points and the resulting minimum enclosing circle.
"""

import marimo

__generated_with = "0.13.15"
app = marimo.App()

with app.setup:
    import math

    import cvxpy as cvx
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    # pick a bunch of random points
    pos = np.random.randn(1000, 2)

    print(f"cvxpy-base version: {cvx.__version__}")
    print(f"numpy version: {np.__version__}")


@app.cell
def _():
    mo.md(
        r"""
    ### Location problem

    We want to find the smallest circle such that $n$ points are all contained in it.
    """
    )
    return


@app.cell
def _():
    # Create a scatter plot with plotly
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=pos[:, 0], y=pos[:, 1], mode="markers", marker={"symbol": "x", "size": 10}))

    _fig.update_layout(
        title="Random Points",
        xaxis_title="x",
        yaxis_title="y",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
    )

    _fig

    return


@app.function
def minimize(objective, constraints=None):
    """Minimizes a given objective function subject to optional constraints.

    This function takes an objective function and an optional list of constraints,
    creates a convex optimization problem, and solves it to obtain the minimum
    value of the objective.

    :param objective: The objective function to be minimized.
    :type objective: cvx.Expression
    :param constraints: a list of constraints applied to the optimization problem,
                        or None if there are no constraints.
    :type constraints: Optional[List[cvx.Constraint]]
    :return: The minimum value of the objective function achieved under the given constraints.
    :rtype: float
    """
    return cvx.Problem(cvx.Minimize(objective), constraints).solve()


@app.function
def location(pos):
    """Computes the minimum enclosing circle (or sphere in higher dimensions) for a given set of points.

    The function minimizes the radius while ensuring that all points lie within
    or on the boundary of the circle.

    :param pos: a 2D array-like structure containing coordinates of points
        (e.g., list of tuples or numpy array). Each row represents a point
        in an n-dimensional Euclidean space, and the number of columns must
        be consistent across all rows.
    :type pos: array-like
    :return: a tuple containing two elements: the radius of the minimum
        enclosing circle (as a float) and the center coordinates of the circle
        (as a 1D array of floats).
    :rtype: tuple[float, numpy.ndarray]
    """
    r, x = cvx.Variable(1), cvx.Variable(2)
    minimize(objective=r, constraints=[cvx.norm(row - x, 2) <= r for row in pos])
    return r.value, x.value


@app.cell
def _():
    print(location(pos))
    return


@app.cell
def _():
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
            marker={"symbol": "x", "size": 10},
            name="Points",
        )
    )

    # Add the circle
    _fig.add_trace(
        go.Scatter(
            x=_circle_x,
            y=_circle_y,
            mode="lines",
            line={"color": "red", "width": 2},
            name="Minimum Circle",
        )
    )

    # Update layout
    _fig.update_layout(
        title="Minimum Circle Containing All Points",
        xaxis_title="x",
        yaxis_title="y",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
    )

    _fig

    return


@app.cell
def _():
    mo.md(
        r"""
    # Summary

    - Each constraint $\rVert{\mathbf{x}-\mathbf{c}}\lVert_2 < R$ represents a cone.
      Feasible domain is the intersection of all cones.

    - It is trivial to generalize (but not to plot) for points in higher dimensional spaces.

    - However, all of this fails once we can construct multiple circles.
    """
    )
    return


if __name__ == "__main__":
    app.run()
