This code was written in Python 3.9 with the following dependencies:
- numpy 1.21.1
- scipy 1.7.0
- pandas 1.3.0
- gurobipy 9.5.1 (GUROBI Optimizer under Academic License; see https://www.gurobi.com/academia/academic-program-and-licenses/)

To replicate the experiments in the paper:
- open a Python command line
- switch directory to this folder
- run the following commands

-------
>>> import experiments
>>> experiments.heterogeneous_experiment(10) # parameter is number of independent trials per choice of subpopulation size k
>>> experiments.iso_experiment_2d(10) # parameter is number of independent trials per choice of noise level
>>> experiments.iso_experiment_3d()
>>> experiments.covariance_shift_experiment()
>>> experiments.boston_housing_experiment(0,1,2) # can select which subset of the boston housing experiments to run: 0 is 3-feature experiment; 1 is zn/crim experiment; 2 is all-pairs experiment