# A toolkit for regression auditing
# Authors: Daniel Freund, Sam Hopkins

import numpy as np
import scipy as sp
import copy
import scipy.linalg
import gurobipy as gp
from gurobipy import GRB
import sys
sys.path.append('MoitraRohatgi/')
import algorithms

def solve_regression_integral(X,y, intercept=True,time_limit=30, warm_start=None, verbose=True, #beta_sign=1,
                             warm_start_ub=None, pairs = None):
    """
    X: is the input vector of independent features
    y: is the input vector of observations
    intercept: if False, then the intercept of the regression is forced to be 0
    (and we drop the constraint that the derivative with respect to the intercept be 0 at the identified solution)
    warm_start: can be used as a vector/list of input-weights that produce a feasible solution
    verbose: set to False to surpress outputs
    beta_sign: set to -1 if the sign of the last feature's coefficient should be non-positive; to 1 for non-negative
    """
    if not intercept: 
        negative = check_negative_OLS(X,y,verbose=False)
    if intercept: 
        negative = check_negative_OLS(np.vstack([np.ones(len(X)),X]).T,y,verbose=False)
    if negative: y=-y
    n = len(X)
    try: m = X.shape[1]
    except: m=1

    model = gp.Model("bilinear")
    if not verbose: model.Params.LogToConsole = 0
    if warm_start is None: W = [model.addVar(name="w" + str(i), lb=0,ub=1, vtype=GRB.BINARY) for i in range(n)]
    else:
        W=[]
        for i in range(n):
            w = model.addVar(name="w" + str(i), lb=0,ub=1,vtype=GRB.BINARY)
            w.Start = warm_start[i]
            W.append(w)
    betas = [model.addVar(name="beta" + str(j), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
             for j in range(m)]
    if intercept: alpha = model.addVar(name="alpha", lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS)
    else: alpha = 0
    model.update()

    model.setObjective(gp.quicksum(W), GRB.MAXIMIZE)
    
    if not pairs is None:
        for pair in pairs:
            model.addConstr(W[pair[0]]==W[pair[1]])
    
    # these guys ensure that the gradient of the least-squares objective is 0
    # when the coeff. for index1 is 0 and the coeff for index2 is beta
    if m>1:
        residuals = [ gp.quicksum([X[i,j] * betas[j] for j in range(m)]) +alpha-y[i] for i in range(n)]
    else:
        residuals = [ (X[i] * betas[0] +alpha-y[i]) for i in range(n)]
    for j in range(m):
        if m>1:
            model.addConstr(gp.quicksum([ W[i] * X[i,j] * (residuals[i]) for i in range(n)]) == 0)
        else:
            model.addConstr(gp.quicksum([ W[i] * X[i] * (residuals[i]) for i in range(n)]) == 0)
    
    if intercept: model.addConstr(gp.quicksum([ W[i] * residuals[i] for i in range(n)]) == 0)
    model.addConstr(betas[-1]<=0)
    model.Params.NonConvex = 2
    model.Params.TimeLimit = time_limit
    if not (warm_start_ub is None): model.addConstr(gp.quicksum(W)<=warm_start_ub)
    
    model.update()

    model.optimize()
    model.update()
    return model.ObjBound, model.ObjVal, W, model

def solve_regression_fractional(X,y, intercept=True,time_limit=30, warm_start=None, verbose=True,
                               greater_one_constraint=False, pairs=None):
    """
    X: is the input vector of independent features; requires X to have at least 2 dimensions, can stack first
    one with 0s
    y: is the input vector of observations
    intercept: if False, then the intercept of the regression is forced to be 0
    (and we drop the constraint that the derivative with respect to the intercept be 0 at the identified solution)
    warm_start: can be used as a vector/list of input-weights that produce a feasible solution
    verbose: set to False to surpress outputs
    """
    n = len(X)
    m = X.shape[1]-1

    model = gp.Model("bilinear")
    if not verbose: model.Params.LogToConsole = 0
    if warm_start is None: W = [model.addVar(name="w" + str(i)) for i in range(n)]
    else:
        W=[]
        for i in range(n):
            w = model.addVar(name="w" + str(i), lb=0,ub=1)
            w.Start = warm_start[i]
            W.append(w)
    for i in range(n):
        model.addConstr( 0 <= W[i] )
        model.addConstr( W[i] <= 1 )
    
    betas = [model.addVar(name="beta", vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY)
             for j in range(m)]
    if intercept: alpha = model.addVar(name="alpha", lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS)
    else: alpha = 0 # model.addVar(name="alpha", lb=0, ub=0, vtype=GRB.BINARY)
    model.update()
    if greater_one_constraint: model.addConstr(sum(W)>=1)
    model.setObjective(gp.quicksum(W), GRB.MAXIMIZE)
    
    if not pairs is None:
        for pair in pairs:
            model.addConstr(W[pair[0]]==W[pair[1]])

    # these guys ensure that the gradient of the least-squares objective is 0
    # when the coeff. for index1 is 0 and the coeff for index2 is beta
    residuals = [ gp.quicksum([X[i,j] * betas[j] for j in range(m)]) +alpha-y[i] for i in range(n)]
    if verbose: print('set residual constraints')
    for j in range(m+1):
        model.addConstr(gp.quicksum([ W[i] * X[i,j] * (residuals[i]) for i in range(n)]) == 0)
    
    if intercept: model.addConstr(gp.quicksum([ W[i] * residuals[i] for i in range(n)]) == 0)

    model.Params.NonConvex = 2
    model.Params.TimeLimit = time_limit
    
    model.update()
    if verbose: print('start solving')
    model.optimize()
    model.update()
    if model.ObjVal<1:
        return solve_regression_fractional(X,y, intercept,time_limit, warm_start, verbose,
                               greater_one_constraint=True)
    return model.ObjBound, model.ObjVal, W, model



def get_influences(X,Y):
    '''
    # returns a list of influences for the last coordinate of the minimum-norm OLS solution for X,Y (without intercept)
    '''
    n = X.shape[0]
    ols_regressor = algorithms.ols(X,Y, np.ones(len(Y)))
    inverse_covariance = np.linalg.pinv(np.dot(X.T,X))

    residuals = [ Y[i] - np.dot(ols_regressor, X[i]) for i in range(n) ]

    gradients = [ np.dot(inverse_covariance, X[i]) * residuals[i] for i in range(n)]
    influences = [gradient[-1] for gradient in gradients]
    return influences

def ZAMinfluence_resolving(X,Y, verbose=False):
    '''
    drop high-influence samples until last OLS coordinate changes sign.
    assumes that last coordinate of OLS on X,Y is unique (though this may not persist after dropping samples from X,Y)
    '''
    X = copy.deepcopy(X)
    Y = copy.deepcopy(Y)
    d = X.shape[1]
    n=len(Y)
    ols_regressor = algorithms.ols(X,Y,np.ones(len(Y)))
    sign = ols_regressor[-1]
    if sign < 0:
        Y=-Y
    
    t = 0
    removed=[]
    while not check_negative_OLS(X,Y,verbose):
        if verbose: print(t)
        t += 1
        influences = get_influences(X,Y)
        imax = np.argmax(influences)
        X[imax]=np.zeros(X.shape[1])
        Y[imax]=0
        if verbose: print("dropping sample " + str(imax))
        removed.append(imax)
    
    weights = np.ones(len(Y))
    for i in removed:
        weights[i]=0
    
    return n-t, weights


def ZAMinfluence_upper_bound(X,Y, verbose=False):
    '''
    drop high-influence samples until last OLS coordinate changes sign
    '''
    n=len(Y)
    d = X.shape[1]
    ols_regressor = algorithms.ols(X,Y,np.ones(len(Y)))
    sign = ols_regressor[-1]
    if sign < 0:
        Y=-Y
    
    influences =  get_influences(X,Y)
    indices = np.argsort(influences)
    X_sorted = X[indices]
    Y_sorted = Y[indices]

    t = 0
    
    while not check_negative_OLS(X_sorted,Y_sorted,verbose) and t<n:
        if verbose: 
            print(t)
            print('index removed:',indices[-t-1])
        t += 1
        X_sorted = X_sorted[:-1]
        Y_sorted = Y_sorted[:-1]
    
    weights = np.ones(len(Y))
    for i in range(t):
        weights[indices[-i-1]]=0
    
    return n-t, weights

def check_negative_OLS(X,Y,verbose=False):
    """
    Returns True if there exists an OLS solution with negative last coefficient
    Returns False otherwise
    Assumes no intercept
    """
    cov = (X.T@X)

    if np.linalg.matrix_rank(cov)!=cov.shape[0]:
        if verbose: print('singular matrix? at length %d'%len(Y))
        null_space = scipy.linalg.null_space(cov)
        for j in range(null_space.shape[1]):
            if not np.isclose(null_space[-1][j],0): # TODO: return to whether this is the right cutoff
                if verbose: print("WARNING: possibly after removing some samples, found an instance where the sample covariance is singular, and furthermore the last coordinate of the OLS solution is not unique! There is probably now an OLS solution with negative last coordinate, but we will continue dropping samples until the last coordinate of the minimum-norm OLS solution has negative last coordinate.")
#                if verbose: print("found nullspace with nonzero last coordinate")
#                if algorithms.ols(X,Y,np.ones(len(Y)))[-1] >= 0:
#                    if verbose: print("furthermore, min norm ols solution has NONnegative last coordinate!")
#                return True

    if verbose: print("last coordinate of OLS is unique")
    ols_value = algorithms.ols(X,Y,np.ones(len(Y)))[-1]
    if verbose: print(ols_value)
    return ols_value<0

def get_negative_OLS(X,Y):
    """
    Returns an OLS regressor with non-positive last coefficient, or None if it does not exist.
    If Cov(X) is nonsingular, this will just be the unique OLS line.
    Otherwise, if Cov(X) is singular, we look in the affine space of OLS optimizers for a vector with non-positive last coordinate.
    We (heuristically) look for a vector whose coordinates are not too large.
    (But we do not guarantee anything about this.)

    TODO: a lot of this logic is duplicated in check_negative_OLS; should probably be merged into one function
    """

    print("WARNING: do not call this method -- it is untested code.")
    cov = (X.T@X)

    # nonsingular covariance case
    if np.linalg.matrix_rank(cov) == cov.shape[0]:
        beta = algorithms.ols(X,Y,np.ones(len(Y)))
        if beta[-1] <= 0:
            return beta
        else:
            return None

    # now we are in the singular covariance case

    # get the least-norm ols regressor
    beta = algorithms.ols(X,Y,np.ones(len(Y)))
    if beta[-1] <= 0:
        return beta

    null_space = scipy.linalg.null_space(cov)
    last_coords_abs = [np.abs(null_space[-1][j]) for j in range(null_space.shape[1])]
    i = np.argmax(last_coords_abs)

    if np.is_close(last_coords_abs[i], 0): # nullspace has all zeros in the last coordinate, and beta[-1] is > 0
        return None

    coeff = - beta[-1] / null_space[-1][i]

    beta_shifted = beta + (coeff + 1) * null_space[:,i]

    assert beta_shifted[-1] < 0 

    beta_err = sum( [ (X[i,:] @ beta - Y[i])**2 for i in range(len(Y))] )
    beta_shifted_err = sum( [ (X[i,:] @ beta_shifted - Y[i])**2 for i in range(len(Y)) ] )

    assert np.is_close(beta_err, beta_shifted_err) # beta and beta_shifted should both be OLS optimizers

    return beta_shifted

 
    
def spectral_certify(X,Y,i=None,intercept=False,verbose=False):
    '''
    X = a n x d data matrix (numpy array)
    Y = an n x 1 list of outcomes (numpy array)
    i = an integer from 0 to d-1

    returns a real number from 0 to n representing a
    lower bound on the fraction of samples which must be
    removed from the regression instance X,Y to change
    the sign of the i-th coordinate
    '''
    
    n = len(Y)

    if len(X.shape) == 1 and intercept == False:
        print("ERROR: spectral auditor not available for this regression problem")
        print("Explanation: It seems you are trying to do a spectral robustness audit for regression with one treatment variable and no intercept term/fixed effects. The spectral auditor is not implemented for this setting. However, you can compute stability of this regression exactly via a simple greedy algorithm so you should not be using the spectral auditor anyway.")
        return

    if intercept:
        if verbose:
            print("adding extra column to account for intercept")
        if len(X.shape) > 1:
            ones_column = np.ones((n,1))
            X = np.hstack((X,ones_column))
        else:
            ones_column = np.ones(n)
            X = np.stack((X,ones_column),axis=1)

    if i == None:
        if intercept:
            if verbose:
                print("coordinate i not specified -- setting i to be last column of X (except for added intercept column)")
            i = X.shape[1]-2
            if verbose:
                print("set i equal to " + str(i))
        else:
            if verbose:
                print("coordinate i not specified -- setting i to be last column of X")
            i = X.shape[1]-1
            if verbose:
                print("set i equal to " + str(i))


    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    
    C1 = get_C1(X,Y,beta)
    C2 = get_C2(X)

    cov = np.dot(X.T,X) / n
    scale = np.sqrt(np.linalg.inv(cov)[i,i])

    beta_coord_abs = np.abs(beta[i])

    eps = beta_coord_abs**2 / ( C1 * scale + beta_coord_abs * C2 )**2

    return eps * n
    

def get_C1(X,Y,beta):
    '''
    X = an n x d data matrix (numpy array)
    Y = a length n vector of outcomes (numpy array)
    beta = a length d vector of coefficients (numpy array)
    '''
    n = X.shape[0]
    res_weighted_matrix = np.array([X[i] * (np.dot(X[i],beta) - Y[i]) for i in range(n)])
    cov_unnormalized = np.dot(X.T,X)
    preconditioner = sp.linalg.sqrtm(np.linalg.inv(cov_unnormalized))
    res_weighted_matrix_preconditioned = np.dot(preconditioner, res_weighted_matrix.T)
    sing_vals = np.linalg.svd(res_weighted_matrix_preconditioned,compute_uv=False)
    return(sing_vals[0])


def get_C2(X):
    '''
    X = an n x d data matrix (numpy array)
    beta = a length d vector of coefficients (numpy array)
    '''
    n = X.shape[0]
    d = X.shape[1]
    cov = np.dot(X.T,X) / n

    Xpreconditioned = np.dot(sp.linalg.sqrtm(np.linalg.inv(cov)), X.T).T
   
    WM2 = np.zeros((n, d*d))
    for i in range(n):
        phi_coeff = np.sqrt(3 / (2+d)) - np.sqrt(3/2)
        identity_coeff = np.sqrt(3/2)
        phi = np.reshape(np.identity(d),d*d)
        vec_square =  np.kron(Xpreconditioned[i],Xpreconditioned[i]) 
        WM2[i,:] = identity_coeff * vec_square + phi_coeff * np.dot(phi,vec_square) * phi / d

    WM2 = 1/np.sqrt(n) * WM2
    
    sing_vals = np.linalg.svd(WM2, compute_uv=False)

    return(sing_vals[0])

def solve_1D_binary_feature(x,y):
    pos_coeff=np.linalg.lstsq(np.vstack([x, 
                    np.ones(len(x))]).T, y, rcond=None)[0][0]
    n = len(y)
    if pos_coeff>0:
        y0s = [y[i] for i in range(n) if x[i]==0]
        y1s = [y[i] for i in range(n) if x[i]==1]
    else:
        y0s = [y[i] for i in range(n) if x[i]==1]
        y1s = [y[i] for i in range(n) if x[i]==0]
    n0, n1 = len(y0s), len(y1s)
    y0s.sort(reverse=True) # maximize y0s by having the high values first
    y1s.sort() # minimize y1s by having small values first
    sum0s = np.cumsum(y0s) # compute the
    sum1s = np.cumsum(y1s)
    lower, upper = 0, n # lower is too small, upper is sufficient
    k = (lower+upper)//2
    while True:
        if lower==upper-1 and Flag: return (k+1, None, (zeros_to_keep, ones_to_keep))
        if lower==upper-1 and not Flag: return (k, None, (zeros_to_keep, ones_to_keep))
        try:
            if Flag: lower = k
            if not Flag: upper =k
        except: pass
        k = (lower+upper)//2
        Flag = True
        # k shall be the points that we remove
        # j iterates over how many to keep/drop from X_i=0 and X_i=1
        for j in range(max(0,n-k-n1), min(n0+1,n-k+1)):
            slope_sign = - (n-k-j) * sum0s[j-1] + j * sum1s[n-k-j-1]
            if slope_sign <= 0:
                Flag = False
                zeros_to_keep = y0s[:j] 
                ones_to_keep = y1s[:n-k-j]
                break

        
        
        
def solve_diff_in_diff(delta_pa, delta_nj):
    """
    Input: 
    delta_pa is the set of deltas among non-treated observations
    delta_nj is the set of deltas among non-treated observations
    ASSUMPTION: ATE is positive with no samples dropped 
    (else, flip sign of observations)
    """
    delta_pa.sort(reverse=True) # ordered in decreasing order; untreated
    delta_nj.sort(reverse=False) # ordered in increasing order; treated
    nj_sums = np.cumsum(delta_nj)
    pa_sums = np.cumsum(delta_pa)
    no_treated, no_not = len(nj_sums), len(pa_sums)
    n = no_treated+no_not
    lower = 0
    upper = n
    current_sign = 0
    while True:
        current_sign, slope_sign = None, 1
        k = (lower+upper)//2
        Flag = True
        #print(lower,k,upper)
        for j in range(k):#max(0,n-k-no_not),min(no_treated+1,n-k+1)):
            #print(j)
            if (j<= no_treated and no_treated-j<=len(nj_sums) and n-no_treated-(k-j)-1<=len(pa_sums) 
                and n-no_treated-(k-j)-1>=0):
                slope_sign = 1/(no_treated-j) * nj_sums[no_treated-j-1] - 1/(n-k)*(nj_sums[no_treated-j-1]
                                                            + pa_sums[n-no_treated-(k-j)-1])
                if slope_sign <= 0:
                    Flag=False
                    current_sign=slope_sign
                    number_treated = j
                    #print('this runs with: j=', j,' k=', k)
                    break
        if lower==upper:
            print('Number of observation pairs to remove: ',k)#, current_sign)
            print('treated removed: ', delta_nj[-number_treated-1:])
            print('untreated removed: ', delta_pa[-(k-number_treated)-1:])
            return k
        if k==upper-1 and Flag:
            print('Number of observation pairs to remove: ', k+1)#, current_sign)
            print('treated removed: ', delta_nj[-number_treated if number_treated>0 else len(delta_nj):])
            print('untreated removed: ', delta_pa[-(k-number_treated-1):])
            return k+1
        if not Flag: upper=k
        else: lower=k
