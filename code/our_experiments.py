# Functions that run the following experiments and create corresponding plots:
# 1. Find the optimal solution to the microcredit studies in Broderick et al.
# 2. Use Gurobi to compute upper/lower bounds for the Boston Housing data in Moitra & Rohatgi
# 3. Use Gurobi to compute a stronger upper bound for Eubank & Fresh
# 4. Apply spectral certifier to obtain non-trivial bounds for synthetic data
# 5. 
# Authors: Daniel Freund, Sam Hopkins

import sys
sys.path.append('MoitraRohatgi/')
import examples
import algorithms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def LoadMicroCreditData():
    attanasio_path = "../../data/Meager_simplified/attanasio.csv"
    angelucci_path = "../../data/Meager_simplified/angelucci.csv"
    augsberg_path = "../../data/Meager_simplified/augsberg.csv"
    banerjee_path = "../../data/Meager_simplified/banerjee.csv"
    crepon_path = "../../data/Meager_simplified/crepon.csv"
    karlan_path = "../../data/Meager_simplified/karlan.csv"
    tarozzi_path = "../../data/Meager_simplified/tarozzi.csv"


    attanasio = pd.read_csv(attanasio_path)  # mongolia
    angelucci = pd.read_csv(angelucci_path)  # mexico
    augsberg = pd.read_csv(augsberg_path)    # bosnia
    banerjee = pd.read_csv(banerjee_path)    # india
    crepon = pd.read_csv(crepon_path)        # morocco
    karlan = pd.read_csv(karlan_path)        # philippines
    tarozzi = pd.read_csv(tarozzi_path)      # ethiopia

    locations = ["mongolia","mexico","bosnia","india","morocco","philippines","ethiopia"]

    datasets = [attanasio,angelucci,augsberg,banerjee,crepon,karlan,tarozzi]
    datasets = [df.dropna() for df in datasets]
    Xs = [np.array(df)[:,2] for df in datasets]
    Ys = [np.array(df)[:,1] for df in datasets]
    return locations, Xs, Ys

def LoadEubankFreshData():
    eubank_path = "../../data/Eubank_black_perc.csv"
    eubank = pd.read_csv(eubank_path)
    coordinates = {}
    i = 0
    for key in eubank["state"].unique():
        coordinates[key] = i
        i += 1

    for key in eubank["year"].unique():
        coordinates[key] = i
        i += 1
        
    n = eubank.shape[0]

    def build_sample(j):
        dim = eubank["state"].unique().shape[0] + eubank["year"].unique().shape[0] + 2
        X = np.zeros(dim)
        
        state_index = coordinates[eubank["state"][j]]
        X[state_index] = 1
        
        year_index = coordinates[eubank["year"][j]]
        X[year_index] = 1
        X[dim-2] = eubank["st_census_urban_perc"][j]
        X[dim-1] = eubank["vra2_x_post1965"][j]
        
        
        return X

    X = np.array([build_sample(i) for i in range(n)])
    Y = np.array(eubank["st_icpsr_MI_rate_black"])
    dim = eubank["state"].unique().shape[0] + eubank["year"].unique().shape[0] + 2
    return X, Y, dim
            
def LoadMartinezData():
    martinez_path = "../../data/martinez.csv"
    martinez = pd.read_csv(martinez_path)
    # get dependent var
    Y = martinez["lngdp14"].to_numpy()

    # grab only the columns we care about, and
    # reorder columns so that lndn13_fiw is last since
    # this is the coefficient whose sign we care about.
    # we are following Martinez, equation 6
    keys = martinez.columns.to_list()
    keys.remove("lndn13_fiw")
    X = martinez[keys[4:] + ["lndn13_fiw"]].to_numpy()
    return X,Y

def LoadSyntheticData():
    # load 2d synthetic dataset
    synthetic_data_path = '../../data/synthetic2d.csv'
    df = pd.read_csv(synthetic_data_path)
    X2 = np.array([[df["X"][i],1] for i in range(len(df["X"]))])
    Y2 = np.array(df["Y"])
    n2 = X2.shape[0]
    # load 4d synthetic dataset
    synthetic_data_path = '../../data/synthetic4d.csv'
    df = pd.read_csv(synthetic_data_path)
    X4 = np.array([[df["X1"][i],df["X2"][i],df["X3"][i],df["X4"][i],1] for i in range(len(df["X1"]))])
    Y4 = np.array(df["Y"])
    n4 = X4.shape[0]
    return X2, Y2, n2, X4, Y4, n4
    
def LoadCardKruegerData():
    df2 = pd.read_csv('../../data/minwage.csv')
    print('NJ mean numbers')
    print(df2[df2.d_nj==1].mean())
    print('PA mean numbers')
    print(df2[df2.d_pa==1].mean())
    df2 = df2[['d_pa','y_ft_employment_before', 'y_ft_employment_after']]
    df2 = df2.dropna()
    df2['delta']=df2.y_ft_employment_after-df2.y_ft_employment_before
    # We sort our dataframe by having first all NJ stores, then all PA stores, 
    # and sorting within these two sets by decreasing delta
    df2 = df2.sort_values(['d_pa', 'delta'],
                  ascending = [True, False])
    print('Head of dataframe after editing')
    print(df2.head())
    # We first compute the OLS on this data by creating 
    # numpy arrays with the appropriate data
    data_X = []
    data_Y = []
    for x in df2.index:
        # Dummy for whether in NJ
        NJ = 0 if df2.d_pa[x] else 1
        # 1 for intercept, dummy for NJ, 
        # dummy for treatment and dummy for treatment*NJ
        data_X.append([1,NJ, 0, 0])
        data_Y.append(df2.y_ft_employment_before[x])
        data_X.append([1,NJ, 1, NJ])
        data_Y.append(df2.y_ft_employment_after[x])
    X=np.array(data_X)
    Y=np.array(data_Y)
    print('OLS before removing samples', algorithms.ols(X,Y,np.ones(len(X)) ) )
    ## we first create lists of deltas in PA and NJ, sort them, and 
    delta_pa, delta_nj = [], []
    for x in df2.index:
        if df2.d_pa[x]:
            delta_pa.append(df2.delta[x])
        else:
            delta_nj.append(df2.delta[x])
    return delta_pa, delta_nj, df2
    
def LoadKruegerDataWith10PAStoresRemoved():
    df2 = pd.read_csv('../../data/minwage.csv')
    df2 = df2[['d_pa','y_ft_employment_before', 'y_ft_employment_after']]
    df2 = df2.dropna()
    df2['delta']=df2.y_ft_employment_after-df2.y_ft_employment_before
    # We sort our dataframe by having first all NJ stores, then all PA stores, 
    # and sorting within these two sets by decreasing delta
    df2 = df2.sort_values(['d_pa', 'delta'],
                  ascending = [True, False])
    data_X = []
    data_Y = []
    for x in df2.index[:-10]: 
        # Dummy for whether in NJ
        NJ = 0 if df2.d_pa[x] else 1 #x_northeast_philadelphia[x]+df2.x_easton_philadelphia[x] else 1
        # 1 for intercept, dummy for NJ, dummy for treatment and dummy for treatment*NJ
        data_X.append([1,NJ, 0, 0])
        data_Y.append(df2.y_ft_employment_before[x])
        data_X.append([1,NJ, 1, NJ])
        data_Y.append(df2.y_ft_employment_after[x])
    X=np.array(data_X)
    Y=np.array(data_Y)
    return X,Y
        
     