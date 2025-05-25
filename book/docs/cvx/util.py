import numpy as np
import cvxpy as cvx
from typing import List, Optional


def minimize(
    objective: cvx.Expression,
    constraints: Optional[List[cvx.constraints.constraint.Constraint]] = None,
) -> float:
    return cvx.Problem(cvx.Minimize(objective), constraints).solve()


def maximize(
    objective: cvx.Expression,
    constraints: Optional[List[cvx.constraints.constraint.Constraint]] = None,
) -> float:
    return cvx.Problem(cvx.Maximize(objective), constraints).solve()


def std(vector: cvx.Expression) -> cvx.Expression:
    # we assume here that the vector is centered
    return cvx.norm(vector, 2) / np.sqrt(vector.size[0] - 1)
