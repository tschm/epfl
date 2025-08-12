# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo==0.13.15"
# ]
# ///
"""Introduction to portfolio optimization, regression, and conic programming.

This module provides an overview of quantitative finance concepts, focusing on
portfolio optimization techniques, challenges in the field, and practical applications.
It includes background information, literature references, and key concepts in the domain.
"""

import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
    # Portfolio Optimization, Regression and Conic Programming

    **Thomas Schmelzer**



    **Thalesians**, Zurich, November 26
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Jobs you might be interested in (from LinkedIn):

    **Quantitative Trader**

    ... - Zurich Area, Switzerland


    Candidates should possess:

    - Excellent knowledge in at least one object oriented language e.g. Java, C# or C++
    - Knowledge of Linux/Unix shells and scripting languages
    - **Knowledge of optimization solvers (SOCP) and experience with an optimization toolbox e.g. Mosek**
    - Solid experience in statistical analysis and software (e.g. R)
    - more wishful thinking...
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Most utterly humble brief personal history within portfolio optimization

    - 2007: Joined **Winton Capital**. Risk measurement (covariance matrices, volatilities, etc.)
      and portfolio optimization. Projects with **Raphael Hauser**.

    - 2008: Started to cooperate with **Mosek** (Danish company providing mathematical software).

    - 2010: Return to Switzerland via **IMC Zug**.

    - 2013: Gardening leave at **Maui**. Two publications with Raphael Hauser.

    - since Feb 2014: Head of Research for **Lobnek Wealth Management** in Geneva.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Warning

    Be **careful** when you mention Optimization... the term is just too ambiguous.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Today

    - we render problems arising in quantitative finance as conic programs.

    - we solve such programs using 3rd party software (Mosek).

    - we illustrate common mistakes made in practice.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Challenges

    - underestimated?


    - modelling (implicit constraints, reverse engineering, politics etc).


    - complex maths, flexibility to formulate problems
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## User feedback

    - It works!
    - It's **broken**.
    - It's **not relevant**. It's all about getting the estimators correct.
    - Our problems are far too complicated for this. We have developed a **proprietary** method far superior.
    - Some are rediscovering **familiar concepts**: *(The solvers) overuse statistically estimated information and
      magnify the impact of estimation errors. It is not simply a matter of garbage in, garbage out, but, rather,
      a molehill of garbage in, a mountain of garbage out* (Michaud 1998)
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## *If the answer is highly sensitive to perturbations, you have probably asked the wrong question.*

    **Lloyd N. Trefethen**, FRS

    MAXIMS ABOUT NUMERICAL MATHEMATICS, SCIENCE, COMPUTERS, AND LIFE ON EARTH.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Literature

    - Stephen Boyd, Convex Optimization, http://stanford.edu/~boyd/cvxbook/

    - Mosek Modeling Manual, http://docs.mosek.com/generic/modeling-letter.pdf

    - Mosek Tutorials, https://github.com/MOSEK/Tutorials

    - Thomas Schmelzer and Raphael Hauser, Seven Sins in Portfolio Optimization, http://arxiv.org/abs/1310.3396

    - Thomas Schmelzer et al., Regression techniques for Portfolio Optimization using MOSEK, http://arxiv.org/abs/1310.3397

    - Gerard Cornuejols, Reha Tutuncu, Optimization Methods in Finance


    This talk is available online:

    https://github.com/tschm/thalesians
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
