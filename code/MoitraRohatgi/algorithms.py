import numpy as np
import gurobipy as gp
from gurobipy import GRB
import scipy.linalg
import itertools


'''
ORDINARY LEAST SQUARES estimator

Input:
- X: n x d (covariate matrix)
- y: n (response vector)
- w: n (weight vector)

Output:
- beta: d (minimum-norm OLS estimator of samples (X_i,y_i) with weights w_i)
'''
def ols(X, y, w):
    W = np.diag(w)
    Sigma = X.T @ W @ X
    return np.linalg.pinv(Sigma) @ (X.T @ W @ y)

'''
COMPUTE SAMPLE INFLUENCES

Input:
- X: n x d (covariate matrix)
- y: n (response vector)

Output:
- psi: n (psi_i is influence of sample i on first coordinate of OLS, i.e.
partial derivative of OLS(X,y,w)[0] with respect to w_i at w=1

Assumes that covariates are full-rank
'''

def influences(X, y):
    n = X.shape[0]
    Sigma = X.T @ X
    v = np.linalg.inv(Sigma)[0]
    beta = ols(X, y, np.ones(n))
    psi = np.zeros((n))
    for j in range(n):
        psi[j] = (v @ X[j]) * (y[j] - X[j] @ beta)
    return psi

'''
GREEDY UPPER BOUND for stability [Kuschnig et al., 21]

Input:
- X: n x d (covariate matrix)
- y: n (response vector)
- threshold (real number, default 0)

Output:
- Number of samples removed by greedy algorithm

'''
def sensitivity(X, y, threshold = 0):
    n = X.shape[0]
    w = np.ones(n)
    beta = ols(X, y, w)
    if beta[0] < threshold:
        y = -y
    for j in range(n):
        # sort samples by influence
        psi = influences(X,y)
        inf_order = np.argsort(psi)[::-1]
        # zero out weight of most influential sample
        w = np.ones(X.shape[0],dtype=bool)
        w[inf_order[0]] = 0
        if ols(X, y, w)[0] <= threshold:
            return j + 1
        X = X[w]
        y = y[w]

'''
COMPUTE EXTREMAL VALUES IN POLYHEDRON

Input:
- x: d
- y: 1
- A_ub: m x d
- b_ub: d

Output:
- loval: 1 (minimum of <x,lambda> - y subject to A_ub lambda <= y)
- hival: 1 (maximum of <x,lambda> - y subject to A_ub lambda <= y)
'''
def compute_single_extremal_vals(x,y,A_ub,b_ub):
    m = gp.Model("extr-lp")
    m.Params.OutputFlag = 0
    m.Params.DualReductions = 0
    lam = m.addMVar(shape=x.shape[0],vtype=GRB.CONTINUOUS,name="lambda",lb=-np.inf,ub=np.inf)
    m.addConstr(A_ub @ lam <= b_ub.ravel())
    m.setObjective(x @ lam - y, GRB.MAXIMIZE)
    m.optimize()
    if m.Status == GRB.UNBOUNDED:
        hival = np.inf
    else:
        hival = m.ObjVal
    m.setObjective(x @ lam - y, GRB.MINIMIZE)
    m.optimize()
    if m.Status == GRB.UNBOUNDED:
        loval = -np.inf
    else:
        loval = m.ObjVal
    return loval,hival

'''
YIELD ALL REGIONS DEMARCATED BY HYPERPLANES

Input:
- X: n x d
- y: n

Yields:
- A_ub
- b_ub
describing a region by system of inequalities A_ub lambda <= b_ub
'''
def generate_bucket_assignments(X,y,L,j=0,A_ub=None,b_ub=None,assignment=[]):
    # have already constrained first j residuals
    if j == X.shape[0]:
        yield A_ub,b_ub
        return
    if j == 0:
        A_ub = np.zeros((0, X.shape[1]))
        b_ub = np.zeros((0, 1))
        lower_bound,upper_bound = -np.inf,np.inf
    else:
        lower_bound, upper_bound = compute_single_extremal_vals(X[j],y[j],A_ub,b_ub)
    for i in range(len(L)):
        if lower_bound <= L[i]:
            if i == 0:
                if upper_bound > L[i]:
                    yield from generate_bucket_assignments(X,y,L,j+1,np.vstack([A_ub, X[j]]), np.vstack([b_ub, y[j] + L[i]]), assignment+[i])
                else:
                    yield from generate_bucket_assignments(X,y,L,j+1,A_ub, b_ub, assignment+[i])
            else:
                if upper_bound > L[i] or lower_bound < L[i-1]:
                    yield from generate_bucket_assignments(X,y,L,j+1,np.vstack([A_ub, X[j], -X[j]]), np.vstack([b_ub, y[j] + L[i], -y[j] - L[i-1]]),assignment + [i])
                else:
                    yield from generate_bucket_assignments(X,y,L,j+1,A_ub, b_ub, assignment+[i])
        if upper_bound <= L[i]:
            break
    if upper_bound > L[-1]:
        if lower_bound < L[-1]:
            yield from generate_bucket_assignments(X,y,L,j+1,np.vstack([A_ub, -X[j]]),np.vstack([b_ub,-y[j]-L[-1]]), assignment + [len(L)])
        else:
            yield from generate_bucket_assignments(X,y,L,j+1,A_ub, b_ub, assignment+[i])

'''
COMPUTE EXTREMAL VALUES IN A POLYHEDRON

Input:
- XP: n x d
- y: n
- A_ub: m x d
- b_ub: m

Output:
- lovals: n (lovals[i] is minimum of XP[i] @ lambda - y[i] over A_ub @ lambda <= b_ub)
- hivals: n (hivals[i] is maximum of XP[i] @ lambda - y[i] over A_ub @ lambda <= b_ub)
'''
def compute_extremal_vals(XP,y,A_ub,b_ub):
    n = XP.shape[0]
    lovals = []
    hivals = []
    m = gp.Model("extr-lp")
    m.Params.OutputFlag = 0
    m.Params.DualReductions = 0
    lam = m.addMVar(shape=XP.shape[1],vtype=GRB.CONTINUOUS,name="lambda",lb=-np.inf,ub=np.inf)
    #print(A_ub.shape,b_ub.shape)
    #print(x.shape[0])
    m.addConstr(A_ub @ lam <= b_ub.ravel())
    for j in range(n):
        m.setObjective(XP[j] @ lam - y[j], GRB.MAXIMIZE)
        m.optimize()
        if m.Status == GRB.UNBOUNDED:
            hival = np.inf
        else:
            hival = m.ObjVal
        m.setObjective(XP[j] @ lam - y[j], GRB.MINIMIZE)
        m.optimize()
        if m.Status == GRB.UNBOUNDED:
            loval = -np.inf
        else:
            loval = m.ObjVal
        lovals.append(loval)
        hivals.append(hival)
    return lovals,hivals


'''
Helper function for LP lower bound and Baseline lower bound:
 given interval in which each residual is constrained,
 and given the original squared loss,
 find lower bound on number of samples which need to be removed
 so that the squared loss will be at most original
'''
def compute_residual_lb(lovals,hivals,original_error):
    n = len(lovals)
    error_list = []
    for j in range(n):
        if (lovals[j]>0 and hivals[j]>0) or (lovals[j]<0 and hivals[j]<0):
            error_list.append(min(lovals[j]**2, hivals[j]**2))
    m = len(error_list)
    error_list.sort()
    total_error = sum(error_list)
    number_removed = 0
    while(1):
        if total_error <= original_error:
            return number_removed-1
        if number_removed >= m:
            print(total_error, original_error, m, number_removed)
        assert(number_removed < m)
        total_error -= error_list[m - number_removed - 1]
        number_removed += 1
    return number_removed

'''
LP LOWER BOUND

Input:
- X: n x d (covariate matrix)
- y: n (response vector)
- L: list (list of thresholds)
- sample_size: integer (number of subsamples to take)
- baseline_only: Boolean

Output:
- if baseline_only is False, then LP lower bound on Stability(X,y)
- if baseline_only is True, then Baseline lower bound on Stability(X,y)
'''
def lp_algorithm(X,y,L,sample_size, baseline_only = False):
    n = X.shape[0]
    d = X.shape[1]
    beta0 = ols(X,y,np.ones(n))
    original_error = np.linalg.norm(X@beta0 - y) ** 2

    XP = np.zeros((d-1,n))
    for i in range(d-1):
        XP[i] = (X.T)[i+1]
    XP = XP.T
    all_rows = range(n)
    sample_rows = np.random.choice(all_rows,size = sample_size, replace = False)
    XP_subset = XP[sample_rows]
    y_subset = y[sample_rows]
    min_removal = n
    for A_ub_lam,b_ub_lam in generate_bucket_assignments(XP_subset,y_subset,L):
        m = gp.Model("approx-lp")
        m.Params.OutputFlag = 0
        lam = m.addMVar(shape=d-1, vtype=GRB.CONTINUOUS,name="lambda",lb=-np.inf,ub=np.inf)
        g = m.addMVar(shape=n, vtype=GRB.CONTINUOUS,name="g",lb=-np.inf,ub=np.inf)
        m.addConstr(A_ub_lam @ lam <= b_ub_lam.ravel())
        lam_eqs = b_ub_lam.shape[0]
        lovals,hivals = compute_extremal_vals(XP,y,A_ub_lam,b_ub_lam)
        residual_lb = compute_residual_lb(lovals,hivals,original_error)
        if baseline_only:
            min_removal = min(min_removal, residual_lb)
            continue
        ngood = 0
        c = np.zeros((n))
        for j in range(n):
            if (lovals[j]>0 and hivals[j]>0):
                c[j] = 1.0/hivals[j]
                ngood += 1
            elif (lovals[j]<0 and hivals[j]<0):
                c[j] = 1.0/lovals[j]
                ngood += 1
        A_ub_lam = np.vstack([np.zeros((ngood, d-1)), A_ub_lam])
        A_ub_g = np.zeros((ngood + lam_eqs, n))
        b_ub = np.vstack([np.zeros((ngood,1)), b_ub_lam])
        bounds = []
        ix=0
        for j in range(n):
            if lovals[j] > 0 and hivals[j] > 0:
                #0 <= g[j] <= X[j][1]*lambda - y[j]
                m.addConstr(0 <= g[j])
                m.addConstr(0 <= XP[j] @ lam - y[j] - g[j])
            elif lovals[j] < 0 and hivals[j] < 0:
                #X[j][1]*lambda - y[j] <= g[j] <= 0
                m.addConstr(XP[j] @ lam - y[j] - g[j] <= 0)
                m.addConstr(g[j]<=0)
            else:
                #lb <= g[j] <= ub
                m.addConstr(lovals[j] <= g[j])
                m.addConstr(g[j] <= hivals[j])
        m.addConstr((X.T @ XP) @ lam - X.T @ y - X.T @ g == 0)
        m.setObjective(c @ g, GRB.MINIMIZE)
        m.optimize()
        try:
            min_removal = min(min_removal, max(residual_lb,m.ObjVal))
        except:
            m.Params.DualReductions = 0
            m.optimize()
            if m.Status == 3: # infeasible
                min_removal = min_removal
            elif m.Status == 4: # unbounded
                min_removal = max(min_removal, residual_lb)
    return min_removal

'''
COMPUTE EXTREMAL VALUES IN AN INTERVAL

Input:
- X: n x 2
- y: n
- A_ub: m x d
- b_ub: m

Output:
- lovals: n (lovals[i] is minimum of X[i][1] @ lambda - y[i] over lo <= lambda <= hi)
- hivals: n (hivals[i] is maximum of XP[i][1] @ lambda - y[i] over lo <= lambda <= hi)
'''

def compute_extremal_vals_2d(X,y,lo,hi):
    n = X.shape[0]
    lovals = []
    hivals = []
    for j in range(n):
        if lo == -np.inf:
            if X[j][1] >= 0:
                lo_val = -np.inf
            else:
                lo_val = np.inf
        else:
            lo_val = X[j][1]*lo - y[j]
        if hi == np.inf:
            if X[j][1] >= 0:
                hi_val = np.inf
            else:
                hi_val = -np.inf
        else:
            hi_val = X[j][1]*hi - y[j]
        if abs(lo_val) > abs(hi_val):
            lo_val,hi_val = hi_val,lo_val
        lovals.append(lo_val)
        hivals.append(hi_val)
    return lovals, hivals

'''
BASELINE LOWER BOUND for 2D data

Input:
- X: n x 2 (covariate matrix)
- y: n (response vector)
- L: list (list of thresholds)
- sample_size: integer (number of subsamples to take)

Output:
- min_removal: float (Baseline lower bound on Stability(X,y))
'''

def certify_by_residual_2d(X,y,L,sample_size):
    n = X.shape[0]
    d = X.shape[1]
    beta0 = ols(X,y,np.ones(n))
    original_error = np.linalg.norm(X@beta0 - y) ** 2
    XP = np.zeros((d-1,n))
    for i in range(d-1):
        XP[i] = (X.T)[i+1]
    XP = XP.T
    all_rows = range(n)
    sample_rows = np.random.choice(all_rows,size = sample_size)
    eqs = itertools.product(sample_rows, L)
    thresholds = []
    for row,val in eqs:
        if abs(X[row][1]) > 1e-8:
            threshold = (val + y[row]) / X[row][1]
            thresholds.append(threshold)
    thresholds.append(np.inf)
    thresholds.append(-np.inf)
    thresholds.sort()
    residual_lbs = []
    for i in range(1, len(thresholds)):
        lovals,hivals = compute_extremal_vals_2d(X,y,thresholds[i-1],thresholds[i])
        residual_lbs.append(compute_residual_lb(lovals,hivals,original_error))
    return min(residual_lbs)

'''
LP LOWER BOUND for 2D data

Input:
- X: n x 2 (covariate matrix)
- y: n (response vector)
- L: list (list of thresholds)
- sample_size: integer (number of subsamples to take)
- baseline_only: Boolean

Output:
- min_removal: float (LP lower bound on Stability(X,y))
'''

def lp_algorithm_2d(X,y,L,sample_size):
    n = X.shape[0]
    d = X.shape[1]
    beta0 = ols(X,y,np.ones(n))
    original_error = np.linalg.norm(X@beta0 - y) ** 2
    XP = np.zeros((d-1,n))
    for i in range(d-1):
        XP[i] = (X.T)[i+1]
    XP = XP.T
    all_rows = range(n)
    sample_rows = np.random.choice(all_rows,size = sample_size)
    eqs = itertools.product(sample_rows, L)
    thresholds = []
    for row,val in eqs:
        if abs(X[row][1]) > 1e-8:
            threshold = (val + y[row]) / X[row][1]
            thresholds.append(threshold)
    thresholds.append(np.inf)
    thresholds.append(-np.inf)
    thresholds.sort()
    residual_lbs = []
    for i in range(1, len(thresholds)):
        lovals,hivals = compute_extremal_vals_2d(X,y,thresholds[i-1],thresholds[i])
        residual_lbs.append((i, compute_residual_lb(lovals,hivals,original_error)))
    residual_lbs.sort(key = lambda x:x[1])
    
    min_removal = n
    for (i,residual_lb) in residual_lbs:
        if residual_lb > min_removal:
            break
        lo = thresholds[i-1]
        hi = thresholds[i]
        num_bad = 0
        lovals, hivals = compute_extremal_vals_2d(X,y,lo,hi)
        c = np.zeros((n+1)) #g_0 ... g_{n-1} lambda
        ngood = 0
        for j in range(n):
            if (lovals[j]>0 and hivals[j]>0) or (lovals[j]<0 and hivals[j]<0):
                c[j] = 1.0/hivals[j]
                ngood += 1
        A_eq = np.zeros((2,n+1))
        for j in range(n):
            A_eq[0][j] = X[j][0]
            A_eq[1][j] = X[j][1]
        A_eq[0][n] = -X.T[0] @ X.T[1]
        A_eq[1][n] = -X.T[1] @ X.T[1]
        b_eq = np.zeros((2))
        b_eq[0] = - X.T[0] @ y
        b_eq[1] = -X.T[1] @ y
        A_ub = np.zeros((ngood,n+1))
        b_ub = np.zeros((ngood))
        ix = 0
        bounds = []
        for j in range(n):
            if lovals[j] > 0 and hivals[j] > 0:
                #0 <= g[j] <= X[j][1]*lambda - y[j]
                bounds.append((0, None))
                A_ub[ix][j] = 1
                A_ub[ix][n] = -X[j][1]
                b_ub[ix] = -y[j]
                ix += 1
            elif lovals[j] < 0 and hivals[j] < 0:
                #X[j][1]*lambda - y[j] <= g[j] <= 0
                bounds.append((None,0))
                A_ub[ix][j] = -1
                A_ub[ix][n] = X[j][1]
                b_ub[ix] = y[j]
                ix += 1
            else:
                lb = min(lovals[j],hivals[j])
                ub = max(lovals[j],hivals[j])
                #lb <= g[j] <= ub
                bounds.append((lb,ub))
        bounds.append((lo,hi))
        m = gp.Model("gur-lp")
        m.Params.OutputFlag = 0
        x = m.addMVar(shape=n+1,vtype=GRB.CONTINUOUS, name = "x", lb=-np.inf, ub=np.inf)
        m.setObjective(c @ x, GRB.MINIMIZE)
        m.addConstr(A_ub @ x <= b_ub, name="ub")
        m.addConstr(A_eq @ x <= b_eq, name="eq1")
        m.addConstr(A_eq @ x >= b_eq, name="eq2")
        lbounds = np.array([r[0] if r[0]!=None else -np.inf for r in bounds])
        rbounds = np.array([r[1] if r[1]!=None else np.inf for r in bounds])
        m.addConstr(lbounds <= x, name="lbounds")
        m.addConstr(x <= rbounds, name="rbounds")
        m.optimize()
        min_removal = min(min_removal, max(residual_lb, m.ObjVal))
    return min_removal



'''
For fixed lambda, compute maximum weight of any weight vector w
that has lambda in OLS(X,y,w)
'''
def solve_fixed_lambda(X,XR,lam):
    n = X.shape[0]
    d = X.shape[1]
    c = -np.ones((n))
    A_eq = np.zeros((d,n))
    b_eq = np.zeros((d))
    for i in range(d):
        A_eq[i] = (X.T)[i] * (XR@lam)
    m = gp.Model("net-model")
    m.Params.OutputFlag = 0
    w = m.addMVar(shape=n,vtype=GRB.CONTINUOUS,name="w",lb=0,ub=1)
    m.setObjective(c @ w, GRB.MINIMIZE)
    m.addConstr(A_eq @ w == b_eq)
    m.optimize()
    return w.X

'''
NET UPPER BOUND

Input:
- X: n x d (covariate matrix)
- y: n (response vector)
- trials: integer

Output:
- Net upper bound on Stability(X,y)
'''
def net_algorithm(X,y,trials):
    n = X.shape[0]
    d = X.shape[1]
    XR = np.zeros((d,n))
    for i in range(d-1):
        XR[i] = (X.T)[i+1]
    XR[d-1] = y
    XR = XR.T
    assert(np.linalg.matrix_rank(XR)==d)
    U,s,Vh = scipy.linalg.svd(XR, full_matrices=False)
    A = Vh.T @ np.diag(1.0/s) @ np.eye(d)
    wstar = np.zeros((n))
    for i in range(trials):
        v = np.random.multivariate_normal(np.zeros((d)),np.eye(d))
        v = v / np.linalg.norm(v)
        lam = A@v
        w = solve_fixed_lambda(X,XR,lam)
        if sum(w) > sum(wstar):
            wstar = w
    return n-sum(wstar)
