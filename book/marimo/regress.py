"""Small utilities for autoregressive signals and linear regression.

This module contains an `AR` helper and a lightweight `LinearRegression`
class used in notebooks. It is intentionally minimal and relies on NumPy
and CVXPY.
"""

from dataclasses import dataclass, field

import cvxpy as cvx
import numpy as np
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.filters.filtertools import convolution_filter


@dataclass(frozen=True)
class AR:
    """Container for AR weight vectors with basic operations.

    Stores named weight arrays and provides helpers to stack them into a
    matrix, compute linear combinations, and convolve returns.
    """

    W: dict[str, np.ndarray] = field(default_factory=dict)

    def __setitem__(self, key, value):
        """Set a weight vector for a given key (column name)."""
        self.W[key] = value

    def __getitem__(self, key):
        """Return the stored weight vector for the given key."""
        return self.W[key]

    @property
    def matrix(self):
        """Stack all weight vectors column-wise into a 2D matrix."""
        return np.column_stack([self.W[col] for col in self.W.keys()])

    def prod(self, w):
        """Compute the linear combination `matrix @ w`."""
        return self.matrix @ w

    def convolve(self, returns, coeff):
        """One-sided convolution of `returns` with coefficients `coeff`."""
        x = convolution_filter(returns, coeff, nsides=1)

        # fill the leading NaNs with zeros
        x[np.isnan(x)] = 0.0
        return x

    def profit(self, returns, coeff):
        """Compute cumulative profit using volatility-scaled positions.

        Positions are shifted by one step to avoid lookahead.
        """
        pos = self.convolve(returns, coeff)
        _v = pos / np.std(pos)
        _v_shift = np.roll(_v, 1)
        _v_shift[0] = 0.0
        _profit = np.cumsum(returns * _v_shift)
        return _profit


@dataclass(frozen=True)
class LinearRegression:
    """Linear regression utilities.

    Linear Regression with:
    - a dictionary of features (column name -> numpy array)
    - Target vector b fixed at initialization
    No intercept.
    """

    b: np.ndarray  # target vector fixed at initialization
    A: dict[str, np.ndarray] = field(default_factory=dict)
    _keys: list[str] = field(default_factory=list)

    @property
    def n(self):
        """Number of samples."""
        return len(self.b)

    @property
    def n_features(self):
        """Number of registered feature columns."""
        return len(self._keys)

    @property
    def matrix(self):
        """Design matrix built by stacking features in `_keys` order."""
        return np.column_stack([self.A[col] for col in self._keys])

    @property
    def b_adjusted(self):
        """Target scaled to unit standard deviation."""
        return self.b / self.b.std()

    @property
    def penalty(self):
        """Penalty weights (mean absolute variation for each feature)."""
        return np.array(list(self.mean_variation.values()))

    def keys(self):
        """Return the list of feature names in insertion order."""
        return self._keys

    def __setitem__(self, key, value):
        """Add a feature column by name, asserting matching length."""
        assert len(value) == self.n
        self.A[key] = value
        self._keys.append(key)

    def __getitem__(self, key):
        """Access a stored feature column by name."""
        return self.A[key]

    def __len__(self):
        """Number of stored features."""
        return len(self.A)

    def __iter__(self):
        """Iterate over the feature mapping items."""
        return iter(self.A)

    def fit(self, lamb=0.0):
        """Fit coefficients with optional L1-penalization.

        Args:
            lamb: Non-negative penalty multiplier applied to a weighted L1 norm.

        Returns:
            A NumPy array with fitted coefficients ordered as in `keys()`.
        """
        x = cvx.Variable(self.n_features)

        # compute the mask of feasible rows
        mask = ~np.any(np.isnan(self.matrix), axis=1)
        a_matrix = self.matrix[mask]
        b = self.b[mask]

        objective = cvx.norm(a_matrix @ x - b, p=2) + lamb * cvx.norm(np.diag(self.penalty) @ x, p=1)
        cvx.Problem(cvx.Minimize(objective)).solve(solver=cvx.CLARABEL)
        # Return coefficients as a dict
        # coef_ = dict(zip(self.keys(), x.value))
        return x.value

        # return coef_

    def predict(self, coef_):
        """Compute predictions for the given coefficient vector."""
        # assert coef_.keys() == self.a.keys()
        return self.matrix @ coef_

    def pacf(self, nlags=100):
        """Partial autocorrelation of the adjusted target (without lag 0)."""
        return sts.pacf(self.b_adjusted, nlags=nlags)[1:]

    @property
    def mean_variation(self):
        """Mean absolute variation for each feature time series."""

        def f(ts):
            x = np.asarray(ts, dtype=float)
            # drop all NaNs?
            x = x[~np.isnan(x)]
            if x.size <= 1:
                return 0.0
            return float(np.mean(np.abs(np.diff(x))))

        return {name: f(ts) for name, ts in self.items()}

    def shift(self, lag=1):
        """Create a shifted copy of the design by `lag` steps (with NaNs)."""
        assert lag >= 1
        lp = LinearRegression(b=self.b)

        for name, ts in self.items():
            lp[name] = np.r_[np.full(lag, np.nan), self[name][:-lag]]

        return lp

    def items(self):
        """Yield pairs of (feature name, feature array)."""
        yield from self.A.items()


if __name__ == "__main__":
    b = np.array([1, 2, 3, 4, 5])
    lp = LinearRegression(b=b)

    lp["x1"] = np.array([1, 0, 3, 4, 3])
    lp["x2"] = np.array([6, 5, 6, 1, 0])

    a = lp.fit()
    # print(a)

    # print(lp.predict(a))

    # print(lp.b)
    # print(lp.b_adjusted)
    print(lp.pacf(nlags=2))
    # print(lp.mean_variation)
    # print(lp.fit(0.0))
    # print(lp.fit2())
    print(lp.penalty)
    # make a bar chart of the coefficients
    # fig = go.Figure(data=[go.Bar(x=list(a.keys()), y=list(a.values()))])
    # fig.show()
    print(lp.matrix)
    lp.roll(lag=1)
    print(lp.matrix)
