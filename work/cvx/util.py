import numpy as np
import cvxpy as cvx

def minimize(objective, constraints=None):
    return cvx.Problem(cvx.Minimize(objective), constraints).solve()

def maximize(objective, constraints=None):
    return cvx.Problem(cvx.Maximize(objective), constraints).solve()

def std(vector):
    # we assume here that the vector is centered
    return cvx.norm(vector, 2)/np.sqrt(vector.size[0]-1)