# LifeSim_childhood
Latest publicly available version of LifeSim Childhood and code for replication of papers

Running this code requires Millenium Cohort Study (MCS) data which can be downloaded after registration from UK data service.

This code requires use of Python, R and STATA. However, only python is used for running the simulations. Stata is used for multiple imputation of the MCS data and R is used to run regressions on multiply imputed data to paramenterise the simulations.

datagen (Python)       - clean raw MCS data files and create files for imputation/simulation
mcsimput (STATA)       - multiple imputation of raw data to fill missing observations
miregs (R)             - regressions of multiply imputed data to parameterise simulation
lifesim2_main (Python) - run simulations and create simulated datasets, results tables and figures  
person (Python)        - function called in lifesim2_main to run simulation
