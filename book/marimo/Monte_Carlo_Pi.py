# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo==0.13.15",
#     "numpy==2.3.0",
#     "plotly==6.1.2",
# ]
# ///

"""Module for estimating the value of π using Monte Carlo simulation.

This module demonstrates a probabilistic approach to approximating π by generating
random points in a square and determining the ratio of points that fall within a
circle. The interactive visualization shows how the accuracy improves with more points.
"""

import marimo

__generated_with = "0.13.15"
app = marimo.App()

with app.setup:
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go


@app.cell
def _():
    mo.md(
        r"""
    # Estimating π using Monte Carlo

    This notebook demonstrates how to estimate the value of π (pi) using a Monte Carlo method.

    **How it works:**
    1. We generate random points within a square with side length 2, centered at the origin.
    2. We count how many points fall within a circle of radius 1, also centered at the origin.
    3. The ratio of points inside the circle to the total number of points, multiplied by 4, approximates π.

    This works because the ratio of the area of the circle (πr²) to the area of the square (4r²) is π/4.
    """
    )
    return


@app.cell
def _():
    # Create a slider to control the number of points
    num_points = mo.ui.slider(3, 7, step=1, value=3, label="Number of points 10^{n}")

    num_points

    return (num_points,)


@app.cell
def _(num_points):
    # Generate random points in a 2x2 square centered at the origin
    np.random.seed(42)  # For reproducibility
    n = 10**num_points.value
    print(n)
    points = np.random.uniform(-1, 1, (n, 2))

    # Calculate distances from origin
    distances = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)

    # Determine which points are inside the circle
    inside_circle = distances <= 1

    # Count points inside the circle
    count_inside = np.sum(inside_circle)

    # Estimate pi
    pi_estimate = 4 * count_inside / n

    # Create points dataframe for plotting
    points_inside = points[inside_circle]
    points_outside = points[~inside_circle]

    return pi_estimate, points_inside, points_outside


@app.cell
def _(pi_estimate, points_inside, points_outside):
    # Create a scatter plot with plotly
    fig = go.Figure()

    # Add points inside the circle
    fig.add_trace(
        go.Scatter(
            x=points_inside[:, 0],
            y=points_inside[:, 1],
            mode="markers",
            marker={"color": "blue", "size": 5},
            name="Inside Circle",
        )
    )

    # Add points outside the circle
    fig.add_trace(
        go.Scatter(
            x=points_outside[:, 0],
            y=points_outside[:, 1],
            mode="markers",
            marker={"color": "red", "size": 5},
            name="Outside Circle",
        )
    )

    # Draw the circle
    theta = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)

    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line={"color": "black", "width": 2}, name="Circle"))

    # Update layout
    fig.update_layout(
        title=f"Monte Carlo Estimation of π = {pi_estimate:.6f} (True value: {np.pi:.6f})",
        xaxis_title="x",
        yaxis_title="y",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        width=700,
        height=700,
        showlegend=True,
    )

    # Add square boundary
    fig.add_shape(
        type="rect",
        x0=-1,
        y0=-1,
        x1=1,
        y1=1,
        line={"color": "black", "width": 2},
        fillcolor="rgba(0,0,0,0)",
    )

    return


@app.cell
def _(pi_estimate):
    mo.md(
        f"""
    ## Results

    - Estimated value of π: **{pi_estimate:.6f}**
    - True value of π: **{np.pi:.6f}**
    - Absolute error: **{abs(pi_estimate - np.pi):.6f}**
    - Relative error: **{100 * abs(pi_estimate - np.pi) / np.pi:.4f}%**

    ### How to improve the estimate?

    The accuracy of the Monte Carlo method improves as the number of points increases.
    Try adjusting the slider to see how the estimate changes with more points.

    The error in this method decreases proportionally to 1/√n, where n is the number of points.
    This means you need to quadruple the number of points to halve the error.
    """
    )
    return


if __name__ == "__main__":
    app.run()
