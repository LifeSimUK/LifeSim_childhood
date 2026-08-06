

				#ReadMe for LifeSim Childhood
				A microsimulation model of childhood for the UK


Latest publicly available version of LifeSim Childhood and code for replication of papers

This code is for use by researchers and policy analysts



#Using LifeSim Childhood
Running this code requires Millenium Cohort Study (MCS) data which can be downloaded after registration from UK data service. The latest publicly available version of the MCS may no longer include some of the variables used in this model.

This code requires use of Python, R and STATA. However, only python is used for running the simulations. Stata is used for multiple imputation of the MCS data and R is used to run regressions on multiply imputed data to paramenterise the simulations.

1. datagen (Python)       - clean raw MCS data files and create files for imputation/simulation
2. mcsimput (STATA)       - multiple imputation of raw data to fill missing observations
3. miregs (R)             - regressions of multiply imputed data to parameterise simulation
4. lifesim2_main (Python) - run simulations and create simulated datasets, results tables and figures  
5. person (Python)        - function called in lifesim2_main to run simulation


#Replication code
The folder IJM_2026 contains the code to replicate LifeSim Childhood: Extrapolating Intervention Effects and Public Cost Savings from Birth to Adolescence in the UK .

#Contact
Shrathinth Venkatesh - shrathinth.venkatesh@york.ac.uk
Ieva Skarda - ieva.skarda@york.ac.uk
Richard Cookson - richard.cookson@york.ac.uk
#License
