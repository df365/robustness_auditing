import numpy as np
import pandas
import algorithms

def heterogeneous_data(n,k,noise):
    X = np.zeros((n,2))
    y = np.zeros((n))
    Cov = np.eye((2))
    Cov[0][0] = 0.01
    for i in range(k):
        X[i] = np.random.multivariate_normal(np.array([-1,0]),Cov)
        y[i] = X[i][0]
    for i in range(k+1,n):
        X[i][0] = 0
        X[i][1] = np.random.normal()
        y[i] = np.random.normal()
    return X,y

def isotropic_gaussian_data(n,d,noise):
    X = np.zeros((n,d))
    y = np.zeros((n))
    for i in range(n):
        X[i] = np.random.multivariate_normal(np.zeros((d)),np.eye(d))
        y[i]=X[i]@np.ones((d)) + noise*np.random.normal()
    return X,y

def covariance_shift_data(n=1000,k=30,c=0.2,C=300):
    X=np.zeros((n + k + 1,2))
    y=np.zeros((n + k + 1))
    Sigma = np.array([[1,-1],[-1,2]])
    for i in range(n):
        X[i] = np.random.multivariate_normal(np.zeros((2)), Sigma)
        y[i] = X[i][1] - X[i][0]
    for i in range(n,n+k):
        X[i] = np.array([1,-3]) * c
        y[i] = -C
    w=np.ones((n+k+1))
    for i in range(n+k,n+k+1):
        X[i] = np.array([1, 1]) * (n**.5)
        w[i] = 0
    for i in range(n+k,n+k+1):
        y[i] = X[i] @ algorithms.ols(X,y,w)
    return X,y

def boston_housing_data(feature_list):
    df = pandas.read_csv("../../data/BostonHousing.csv")
    X = np.array(df.drop(columns=["medv",'Unnamed: 0']))
    y = np.array(df["medv"])
    XL = (X.T[feature_list]).T
    return XL, y

def boston_housing_features():
    df = pandas.read_csv("BostonHousing.csv")
    df = df.drop(columns=["medv",'Unnamed: 0'])
    return list(df.columns)
