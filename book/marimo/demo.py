from marimo import App

__generated_with = "0.13.15"
app = App()


with app.setup:
    import cvxsimulator as sim
    import marimo as mo


@app.cell
def _():
    print(mo.notebook_location())
    return


@app.cell
def _():
    print(sim.__version__)
    return


if __name__ == "__main__":
    app.run()
