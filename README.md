# robustness_auditing
Contains algorithms to identify influential sets of minimum size for OLS regression problems

Authors: Daniel Freund, Sam Hopkins
Date: June, 2023

Our codebase is tested with the following versions of standard software packages:

Python 3.9
Numpy 1.21.5
Gurobi 9.5.2
Gurobipy 9.5.2
IPython 8.2

Files in this repository:

code:

auditor_tools.py: contains our robustness auditing codebase, implementations of all algorithms we use except those of Moitra-Rohatgi
    
MoitraRohatgi: local copy of replication files released by Moitra and Rohatgi, downloaded from https://openreview.net/forum?id=DlpCotqdTy
    
notebooks: contains one jupyter notebook for each dataset we study. Notebooks should be run from top to bottom to reproduce all results and figures in our paper.

data: contains csvs for all datasets used in our paper


Contact: if you wish to get in touch with the authors please email Daniel Freund (dfreund@mit.edu) and Sam Hopkins (samhop@mit.edu).
