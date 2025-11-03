lfsm = '//lifesim2-main/IJM2026'


# Save copies of all packages required.
import os
os.chdir(lfsm)

%pip install --no-index --find-links="{lfsm}/pypkgs" tqdm
%pip install --no-index --find-links="{lfsm}/pypkgs" joblib
%pip install --no-index --find-links="{lfsm}/pypkgs" statsmodels
%pip install --no-index --find-links="{lfsm}/pypkgs" itertools
%pip install --no-index --find-links="{lfsm}/pypkgs" seaborn
%pip install --no-index --find-links="{lfsm}/pypkgs" uuid
%pip install --no-index --find-links="{lfsm}/pypkgs" pyarrow
%pip install --upgrade pip


import pandas as pd
import numpy as np
import re
import time
import os.path
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy import stats
from statsmodels.stats.weightstats import DescrStatsW
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels as sms
import itertools
from itertools import chain, zip_longest
import matplotlib.pyplot as plt
import logging
import sys
import inspect
import seaborn as sns
#To svae csvs faster
import uuid
import pyarrow as pa
import pyarrow.csv as csv
# import forestplot as fp
# from pylatex import Document, Section, Subsection, Table, Tabular, NoEscape, Center, Command, Figure, Package
# from pylatex.utils import italic, bold, NoEscape

%pip download -d pypkgs numpy
%pip download -d pypkgs numpy
%pip download -d pypkgs pandas
%pip download -d pypkgs tqdm
%pip download -d pypkgs joblib
# %pip download -d pypkgs scipi
%pip download -d pypkgs statsmodels
# %pip download -d pypkgs itertools
%pip download -d pypkgs matplotlib
%pip download -d pypkgs seaborn
%pip download -d pypkgs uuid
%pip download -d pypkgs pyarrow




####################################################################################################
#################### Functions for use in rest of file
####################################################################################################
# Function to generate unweighted descriptive stats for simulation output
def descsim(data, varl, ppt=False):
    stats = ["mean", "std"]
    d1 = data[varl].groupby(data['simulation']).mean()
    desc = sms.stats.descriptivestats.describe(d1, stats=stats, numeric=True).T
    # desc = desc.rename(index=var_dict)
    if ppt:         #If percentage point
        desc['mean'] = desc['mean']*100
        desc['std'] = desc['std']*100
    logging.info(desc)
    return desc

# Function to print cost tables with totals by sweep
def cstsm(desc, r, ctp="CT"):
    #Choose results format to report (Default - CT)
    # A - Annual average cost per child 
    # T - Total cost per child
    # CA - Cohort average annual cost
    # CT - Cohort total cost
    # RT - Total cost per recipient child
    # I suffix represents using SDQ internalising instead of SDQ emotion for wellbeing
    d = desc.copy()
    # d = d.filter(regex=r'^(?!.*se$)')
    d['sum'] = d.sum(numeric_only=True, axis=1)
    logging.info(d)    
    n = 15 #number of years 3 to 17 incluseive = 15
    # if dr:
    #     r = pd.to_numeric(dr)      
    # else: 
    #     r = 0.035 #discount rate
    #     print(r)
    ##### Adding costs (Ages)
    #Wave 1 - 3 Years (9 months -> 0, 1, 2)
    #Wave 2 - 2 Years ( 3 ->  3,  4)
    #Wave 3 - 2 Years ( 5 ->  5,  6)
    #Wave 4 - 3 Years ( 7 ->  7,  8,  9)
    #Wave 5 - 3 Years (11 -> 10, 11, 12)
    #Wave 6 - 3 Years (14 -> 13, 14, 15)
    #Wave 7 - 2 Years (17 -> 16, 17)
    # weights = {'2': 2, '3': 2, '4': 3, '5': 3, '6': 3, '7': 2}
    weights = {'2': pow((1/(1+r)), 3)+pow((1/(1+r)), 4),
               '3': pow((1/(1+r)), 5)+pow((1/(1+r)), 6),
               '4': pow((1/(1+r)), 7)+pow((1/(1+r)), 8)+pow((1/(1+r)), 9),
               '5': pow((1/(1+r)), 10)+pow((1/(1+r)), 11)+pow((1/(1+r)), 12),
               '6': pow((1/(1+r)), 13)+pow((1/(1+r)), 14)+pow((1/(1+r)), 15),
               '7': pow((1/(1+r)), 16)+pow((1/(1+r)), 17)}
    ## Annual average between 3 and 17
    # Calculate the weighted sum of 'sum' values based on the sweep
    weighted_sums = d.apply(lambda row: row['sum'] * weights[row['Sweep']], axis=1)
    ws = weighted_sums.sum()
    # Calculate t as the total of weighted sums divided by the total weight
    # total_weight = sum(weights.values())
    t = weighted_sums.sum() / n
    ct = t*700000
    cws = ws*700000
    logging.info(f"Annual cost 3-17         = £ {t}")
    logging.info(f"Total cost 3-17          = £ {ws}")
    logging.info(f"Cohort Annual cost 3-17  = £ {ct}")
    logging.info(f"Cohort total cost 3-17   = £ {cws}")
    

    if ctp in ("A", "AI"):
        d['ttl'] = t
    elif ctp in ("T", "TI", "RT", "RTI"):
        d['ttl'] = ws
    elif ctp in ("CA", "CAI"):   
        d['ttl'] = ct
    elif ctp in ("CT", "CTI"):    
        d['ttl'] = cws
    else : {}  
    
    columns = ['hc', 'll', 'cd', 'se', 'pt', 'pe']
    for col in columns:
        weighted_sums = d.apply(lambda row: row[col] * weights[row['Sweep']], axis=1)
        ws = weighted_sums.sum()
        # total_weight = sum(weights.values())
        t = weighted_sums.sum() / n
        ct = t*700000
        cws = ws*700000
        logging.info(f"Annual {col} cost 3-17         = £ {t}")
        logging.info(f"Total {col} cost 3-17          = £ {ws}")
        logging.info(f"Cohort {col} Annual cost 3-17  = £ {ct}")
        logging.info(f"Cohort {col} total cost 3-17   = £ {cws}")
        if ctp in ("A", "AI"):
            d[f'{col}'] = t
        elif ctp in ("T", "TI", "RT", "RTI"):
            d[f'{col}'] = ws
        elif ctp in ("CA", "CAI"):   
            d[f'{col}'] = ct
        elif ctp in ("CT", "CTI"): 
            d[f'{col}'] = cws
        else : {}  
     
    d = d.iloc[[0]].drop(columns=['Sweep', 'sum'])   

    return d.T

# Function to calculate totals across sweeps weighted by years
def aswpwtot(data, varl, r, s, e):
    ##### Adding costs (Ages)
    #Wave 1 - 3 Years (9 months -> 0, 1, 2)
    #Wave 2 - 2 Years ( 3 ->  3,  4)
    #Wave 3 - 2 Years ( 5 ->  5,  6)
    #Wave 4 - 3 Years ( 7 ->  7,  8,  9)
    #Wave 5 - 3 Years (11 -> 10, 11, 12)
    #Wave 6 - 3 Years (14 -> 13, 14, 15)
    #Wave 7 - 2 Years (17 -> 16, 17)
    # weights = {'2': 2, '3': 2, '4': 3, '5': 3, '6': 3, '7': 2}
    
    weights = {'1': pow((1/(1+r)), 1)+pow((1/(1+r)), 2),
               '2': pow((1/(1+r)), 3)+pow((1/(1+r)), 4),
               '3': pow((1/(1+r)), 5)+pow((1/(1+r)), 6),
               '4': pow((1/(1+r)), 7)+pow((1/(1+r)), 8)+pow((1/(1+r)), 9),
               '5': pow((1/(1+r)), 10)+pow((1/(1+r)), 11)+pow((1/(1+r)), 12),
               '6': pow((1/(1+r)), 13)+pow((1/(1+r)), 14)+pow((1/(1+r)), 15),
               '7': pow((1/(1+r)), 16)+pow((1/(1+r)), 17)}
    
    for var in varl:
        weighted_sums = data.apply(lambda row: sum(row[f"{var}{i}"] * weights[str(i)] for i in range(s+1, e+1)), axis=1)
        data[f"{var}_ttl"] = weighted_sums
    return data
    
# Function to generate bar plots for baseline or individual interventions
def pltsim(desc, lbl, plt_path, ya=None, xa=None, tt=None, typ=None):
    r = np.arange(len(lbl))
    w = 0.2
    # if typ == "bar":
    #     plt.figure(figsize=(15, 9))
    #     plt.rcParams.update({'font.size': 18})  # Set the font size to 14
    #     plt.bar(lbl, desc['mean'], yerr=desc['std'], capsize=5, color='#585869', zorder=2)
    #     plt.bar(lbl, desc['mean'], capsize=5, color='#585869', zorder=2)
    #     plt.grid(axis='y', which='both', linestyle="--", alpha=0.5, zorder=1)
    # else:
    if typ == "frst":
        plt.figure(figsize=(15, 9))
        plt.errorbar(desc['mean'], lbl, xerr=desc['std'], fmt='o', color='#585869', ecolor='#585869', capsize=5, zorder=2)
        plt.grid(axis='x', which='both', linestyle="--", alpha=0.5, zorder=1)
        plt.tight_layout()
        # Prepare data for forest plot
        # data = {
        #     'mean': desc['mean'].values,
        #     'lower': desc['mean'].values - 1.96 * desc['se'].values,
        #     'upper': desc['mean'].values + 1.96 * desc['se'].values
        # }
        # # Create a DataFrame for forest plot
        # df = pd.DataFrame(data, index=desc.index)
        # # Plot the forest plot
        # fig, ax = plt.subplots(figsize=(15, 9))
        # fp.forestplot(df, ax=ax)
        # plt.show()
        #plt.errorbar(lbl, desc['mean'], yerr=desc['std'], fmt='o', color='#585869', ecolor='#585869', capsize=5, zorder=2)
        #plt.grid(axis='y', which='both', linestyle="--", alpha=0.5, zorder=1)
    else:
        plt.figure(figsize=(15, 9))
        plt.rcParams.update({'font.size': 18})  # Set the font size to 14
        plt.bar(lbl, desc['mean'], yerr=1.96*desc['std'], capsize=5, color='#585869', zorder=2)
        plt.bar(lbl, desc['mean'], capsize=5, color='#585869', zorder=2)
        plt.grid(axis='y', which='both', linestyle="--", alpha=0.5, zorder=1)
    #    pass
    if xa:
        plt.xlabel(xa)
    if ya:
        plt.ylabel(ya)
    if tt:
        plt.ylim(top=tt)
    # os.remove(plt_path)
    plt.savefig(plt_path, bbox_inches='tight')
    plt.show()
    

# Function to generate descriptive statistics of the differences
def diffdescuw(varl, num_interventions, neg=False, pct=False, ppt=False, rec=False):
    stats = ["mean", "std"]
    d1 = histint0[varl].groupby(histint0['simulation']).mean()
    desc = sms.stats.descriptivestats.describe(d1[varl], stats=stats, numeric=True).T
    # desc = desc.rename(index=var_dict)
    logging.info('Baseline stats')
    logging.info(desc)

    for i in range(1, num_interventions + 1):
        histint_i = globals()[f'histint{i}']
        d = histint_i[varl] - histint0[varl]
        d['simulation'] = histint0['simulation']
        #If limiting to recipient population
        if rec:
            d['recip'] = histint_i['recip']
            d = d.loc[d.recip == 1]
        d1 = d[varl].groupby(d['simulation']).mean()
        desc_i = sms.stats.descriptivestats.describe(d1, stats=stats, numeric=True).T
        # desc_i = desc_i.rename(index=var_dict)
        if neg:         #If negative outcomes
            desc_i['mean'] = -desc_i['mean']
        if ppt:         #If percentage point
            desc_i['mean'] = desc_i['mean']*100
            desc_i['std'] = desc_i['std']*100
        if pct:         #If percent of baseline
            desc_i['mean'] = desc_i['mean']*100/desc['mean']
            desc_i['std'] = desc_i['std']*100/desc['mean']
        logging.info(f'Intervention {i} effects')
        logging.info(desc_i)
        desc[f'int{i}'] = desc_i['mean']
        desc[f'int{i}se'] = desc_i['std']

    return desc

# Function to generate descriptive statistics of the differences by quantiles
def diffdescuwq(varl, num_interventions, neg=False, pct=False, ppt=False, rec=False):
    stats = ["mean", "std"]
    d1 = histint0[varl].groupby(histint0['simulation', 'incqnt123']).mean()
    desc = sms.stats.descriptivestats.describe(d1[varl], stats=stats, numeric=True).T
    # desc = desc.rename(index=var_dict)
    logging.info('Baseline stats')
    logging.info(desc)

    for i in range(1, num_interventions + 1):
        histint_i = globals()[f'histint{i}']
        d = histint_i[varl] - histint0[varl]
        d['simulation'] = histint0['simulation']
        #If limiting to recipient population
        if rec:
            d['recip'] = histint_i['recip']
            d = d.loc[d.recip == 1]
        d1 = d[varl].groupby(d['simulation', 'incqnt123']).mean()
        desc_i = sms.stats.descriptivestats.describe(d1, stats=stats, numeric=True).T
        # desc_i = desc_i.rename(index=var_dict)
        if neg:         #If negative outcomes
            desc_i['mean'] = -desc_i['mean']
        if ppt:         #If percentage point
            desc_i['mean'] = desc_i['mean']*100
            desc_i['std'] = desc_i['std']*100
        if pct:         #If percent of baseline
            desc_i['mean'] = desc_i['mean']*100/desc['mean']
            desc_i['std'] = desc_i['std']*100/desc['mean']
        logging.info(f'Intervention {i} effects')
        logging.info(desc_i)
        desc[f'int{i}'] = desc_i['mean']
        desc[f'int{i}se'] = desc_i['std']

    return desc

# Function to print lifesatisfaction tables with totals by sweep       
def lfstfm(desc, r, ctp="CT"):
     #Choose results format to report (Default - CT)
     # A - Annual average cost per child 
     # T - Total cost per child
     # CA - Cohort average annual cost
     # CT - Cohort total cost
     # RT - Total cost per recipient child
     # I suffix represents using SDQ internalising instead of SDQ emotion for wellbeing
     d = desc.copy()
     d = d.loc[:, ~d.columns.str.endswith('se')] 
     n = 15 #number of years 3 to 17 incluseive = 15
     ##### Adding costs (Ages)
     #Wave 1 - 3 Years (9 months -> 0, 1, 2)
     #Wave 2 - 2 Years ( 3 ->  3,  4)
     #Wave 3 - 2 Years ( 5 ->  5,  6)
     #Wave 4 - 3 Years ( 7 ->  7,  8,  9)
     #Wave 5 - 3 Years (11 -> 10, 11, 12)
     #Wave 6 - 3 Years (14 -> 13, 14, 15)
     #Wave 7 - 2 Years (17 -> 16, 17)
     # weights = {'2': 2, '3': 2, '4': 3, '5': 3, '6': 3, '7': 2}
     # if dr:
     #    r = dr      
     # else: 
     #    r = 0.035 #discount rate
     weights = {'2': pow((1/(1+r)), 3)+pow((1/(1+r)), 4),
                '3': pow((1/(1+r)), 5)+pow((1/(1+r)), 6),
                '4': pow((1/(1+r)), 7)+pow((1/(1+r)), 8)+pow((1/(1+r)), 9),
                '5': pow((1/(1+r)), 10)+pow((1/(1+r)), 11)+pow((1/(1+r)), 12),
                '6': pow((1/(1+r)), 13)+pow((1/(1+r)), 14)+pow((1/(1+r)), 15),
                '7': pow((1/(1+r)), 16)+pow((1/(1+r)), 17)}
     
     d1 = pd.DataFrame()
     for i in range(len(d.columns)-1):
         weighted_sums = d.apply(lambda row: row[f'int{i+1}'] * weights[row['sweep']], axis=1)
         ws_i = weighted_sums.sum()
         # total_weight = sum(weights.values())
         t_i = weighted_sums.sum() / n
         ct_i = t_i*700000
         cws_i = ws_i*700000
         tv_i = t_i*13000
         wsv_i = ws_i*13000
         ctv_i = ct_i*13000
         cwv_i = cws_i*13000
         logging.info(f"Annual WELLBYs 3-17 scenario {i+1}  = {t_i}")
         logging.info(f"Total WELLBYs 3-17 scenario {i+1}   = {ws_i}")
         logging.info(f"Cohort total Annual WELLBYs 3-17 scenario {i+1}  = {ct_i}")
         logging.info(f"Cohort total WELLBYs 3-17 scenario {i+1}   = {cws_i}")
         logging.info(f"Cohort total Annual WELLBYs value scenario {i+1}  = {ctv_i}")
         logging.info(f"Cohort total WELLBYs value scenario {i+1}   = {cwv_i}")
         if ctp in ("A", "AI"):
            d1.at['wlby', f'int{i+1}'] = t_i
            d1.at['wbvl', f'int{i+1}'] = tv_i
         elif ctp in ("T", "TI", "RT", "RTI"):
            d1.at['wlby', f'int{i+1}'] = ws_i
            d1.at['wbvl', f'int{i+1}'] = wsv_i
         elif ctp in ("CA", "CAI"):    
            d1.at['wlby', f'int{i+1}'] = ct_i
            d1.at['wbvl', f'int{i+1}'] = ctv_i
         elif ctp in ("CT", "CTI"):     
            d1.at['wlby', f'int{i+1}'] = cws_i
            d1.at['wbvl', f'int{i+1}'] = cwv_i
         else : {}  
         
     
     return d1

# Function to generate bar plots for intervention comparison
def diffplt(desc, lbl, plt_path, nint, ya=None, xa=None):
    r = np.arange(len(lbl))
    w = 0.2
    plt.figure(figsize=(15, 9))
    plt.rcParams.update({'font.size': 18})  # Set the font size to 14
    # colors = ['#DDCC77', '#117733', '#882255', '#332288']  # Define the colors for each intervention
    colors = ['#dadae4', '#a8a8af', '#585869', '#28282b']  # Define the colors for each intervention
    plt.grid(axis='y', which='both', linestyle="--", alpha=0.5, zorder=1)
    for i in range(nint):
        plt.bar(r + w * i, desc[f'int{i+1}'], 
                yerr=1.96*desc[f'int{i+1}se'], 
                capsize=5, color=colors[i],
                width=w, edgecolor='black', zorder=2,
                label=f'Scenario {i+1}')
    if nint > 1: 
        plt.xticks(r + w * (nint-1) / 2, lbl)
        plt.legend(loc='upper left')
    if xa:
        plt.xlabel(xa)
    if ya:
        plt.ylabel(ya)
    # os.remove(plt_path)
    plt.savefig(plt_path, bbox_inches='tight')
    plt.show()


########## Var dictionaries

var_dict = np.load(os.path.join(lfsm,'varlabel.npy'),allow_pickle='TRUE').item()

swpag_dict = np.load(os.path.join(lfsm,'swplabel.npy'),allow_pickle='TRUE').item()


### GDP deflators
gdpdeflator = np.load(os.path.join(lfsm,'gdpdef.npy'),allow_pickle='TRUE').item()


####################################################################################################
#################### Data setup and descriptive statistics
####################################################################################################

#Set sample size
num_people = 20000
#Set nmber of universes 
num_universes = 100
#Choose to save files (0 - No files saved, 1 - working file saved, 2 - permanent copy saved)
save = 0
### Read all data
##Multiply imputed MCS data from stata
#mcs = pd.read_stata(os.path.join(lfsm,'mcs1ic.dta'))
mcs = pd.read_stata(os.path.join(lfsm,'mcs1ic.dta'))
#Keep only imputed data sets
mcsi = mcs[mcs['imputn'] > 0]
# add variables for regression
mcsi['intercept'] = 1

# Number of MCS waves
nwvs = 7

rskfctrs = ["incqnt"]

#### Costs and wellbeing value
#Year for cost calculations 
cyr = 2023
### Costs
#create dataframe to store costs
costs = pd.DataFrame(index=['cd1', 'cd2', 'cd3', 'tru', 'exc', 'sen', 'dis', 'hos'],
                        columns =['ttl', 'nhs', 'ded', 'ss', 'cj', 'vs'])
##Conduct disorder
gdfl = gdpdeflator[cyr]/gdpdeflator[2008]
# Ages 5 - 10
costs.loc['cd1', 'ttl'] = 2152 * gdfl
costs.loc['cd1', 'nhs'] = 1113 * gdfl
costs.loc['cd1', 'ss'] = 157 * gdfl
# costs.loc['cd1', 'ded'] = 882 * gdfl
costs.loc['cd1', 'ded'] = 0 * gdfl
costs.loc['cd1', 'vs'] = 23 * gdfl
# Ages 11 - 16 
costs.loc['cd2', 'ttl'] = 1366 * gdfl
costs.loc['cd2', 'nhs'] = 101 * gdfl
costs.loc['cd2', 'ss'] = 63 * gdfl
# costs.loc['cd2', 'ded'] = 1202 * gdfl
costs.loc['cd2', 'ded'] = 0 * gdfl
costs.loc['cd2', 'vs'] = 23 * gdfl
# Ages 17+
costs.loc['cd3', 'ttl'] = 164 * gdfl
costs.loc['cd3', 'nhs'] = 101 * gdfl
costs.loc['cd3', 'ss'] = 63 * gdfl
costs.loc['cd3', 'ded'] = 0 * gdfl
costs.loc['cd3', 'vs'] = 23 * gdfl
## Truancy
gdfl = gdpdeflator[cyr]/gdpdeflator[2005]
costs.loc['tru', 'ttl'] = 604 * gdfl
costs.loc['tru', 'ded'] = costs.loc['tru', 'ttl']
## Exclusion
gdfl = gdpdeflator[cyr]/gdpdeflator[2018]
costs.loc['exc', 'ttl'] = 18000 * gdfl
costs.loc['exc', 'ded'] = costs.loc['exc', 'ttl']
## Special Education Needs
gdfl = gdpdeflator[cyr]/gdpdeflator[2023]
costs.loc['sen', 'ttl'] = 25500 * gdfl
costs.loc['sen', 'ded'] = costs.loc['sen', 'ttl']
## Disability
gdfl = gdpdeflator[cyr]/gdpdeflator[2016]
costs.loc['dis', 'ttl'] = 800 * gdfl
costs.loc['dis', 'nhs'] = costs.loc['dis', 'ttl'] 
## Inpatient hospital visit
gdfl = gdpdeflator[cyr]/gdpdeflator[2020]
costs.loc['hos', 'ttl'] = 1405 * gdfl
costs.loc['hos', 'nhs'] = costs.loc['hos', 'ttl']

### Wellbeing
#create dataframe to store costs
wbval = pd.DataFrame(index=['wbnv', 'wblv', 'wbuv', 'wbsv'],
                        columns =['ttl'])
## Values
gdfl = gdpdeflator[cyr]/gdpdeflator[2019]
# Normal value
wbval.loc['wbnv', 'ttl'] = 13000 * gdfl
# Lower bound
wbval.loc['wblv', 'ttl'] = 10000 * gdfl
# Upper bound
wbval.loc['wbuv', 'ttl'] = 16000 * gdfl
# Supply side value
# Based on
# page 55 of https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1005388/Wellbeing_guidance_for_appraisal_-_supplementary_Green_Book_guidance.pdf
# Marginal productivity value from https://www.sciencedirect.com/science/article/pii/S1098301519321503
gdfl = gdpdeflator[cyr]/gdpdeflator[2012]
wbval.loc['wbsv', 'ttl'] = 14410 * gdfl/(8-1)


del gdfl

# ##### Plots describing MCS data

# ### Life Satisfaction measure comparison
# ##All outcomes and exposures data set
# varl = ['sdqinternal2', 'sdqinternal3', 'sdqinternal4', 'sdqinternal5', 'sdqinternal6', 'sdqinternal7', 'sdqinternalsr7', 
#         'genwelb5', 'genwelb6',
#         'swemwbs7',
#         'lifesat2', 'lifesat3', 'lifesat4', 'lifesat5', 'lifesat6', 'lifesat7', 'lifesatsr7',
#         'lifesatwb5', 'lifesatwb6', 'lifesatwb7'
#         ]
# varl = ['lifesat2', 
#         'lifesat3', 
#         'lifesat4', 
#         'lifesat5', 'lifesatwb5',
#         'lifesat6', 'lifesatwb6',
#         'lifesat7', 'lifesatsr7',
#         'lifesatwb7'
#         ]

# print(mcs['lifesatwb7'].value_counts(dropna=False))
# ##Unweighted 
# stats = ["nobs", "missing", "mean", "std", "percentiles"]
# # stats = ["nobs", "missing", "mean", "std"]
# # stats = ["mean", "std_err"]
# #mean_values = history[varl].mean()
# #logging.info(mean_values)
# descf = sms.stats.descriptivestats.describe(mcsi[varl], stats=stats, numeric=True, percentiles=(5, 25, 50, 75, 95) ).T
# print(descf)

# fig, ax = plt.subplots(figsize=(12, 6))
# sns.kdeplot(data=mcsi['lifesat5'],
#             color='crimson', label='SDQ Internalising', fill=True, ax=ax)
# sns.kdeplot(data=mcsi['lifesatwb5'],
#             color='limegreen', label='General wellbeing', fill=True, ax=ax)
# ax.legend()
# plt.xlabel('Life Satisfaction at age 11')
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(figsize=(12, 6))
# sns.kdeplot(data=mcsi['lifesat6'],
#             color='crimson', label='SDQ Internalising', fill=True, ax=ax)
# sns.kdeplot(data=mcsi['lifesatwb6'],
#             color='limegreen', label='General wellbeing', fill=True, ax=ax)
# ax.legend()
# plt.xlabel('Life Satisfaction at age 14')
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(figsize=(12, 6))
# sns.kdeplot(data=mcsi['lifesat7'],
#             color='crimson', label='SDQ Internalising', fill=True, ax=ax)
# sns.kdeplot(data=mcsi['lifesatsr7'],
#             color='blue', label='Self reported SDQ', fill=True, ax=ax)
# sns.kdeplot(data=mcs['lifesatwb7'],
#             color='limegreen', label='SWEMWBS', fill=True, ax=ax)
# ax.legend()
# plt.xlabel('Life Satisfaction at age 17')
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(figsize=(12, 6))
# sns.kdeplot(data=mcsi['sdqpeersr7'],
#             color='crimson', label='SDQ Peer', fill=True, ax=ax)
# sns.kdeplot(data=mcsi['sdqemotionsr7'],
#             color='blue', label='SDQ Emotion', fill=True, ax=ax)
# sns.kdeplot(data=mcsi['sdqinternalsr7'],
#             color='limegreen', label='SDQ Internalising', fill=True, ax=ax)
# ax.legend()
# plt.xlabel('Life Satisfaction at age 17')
# plt.tight_layout()
# plt.show()


# mcso = mcs[mcs['imputn'] == 0]
# varl = ['sdqinternal2', 'sdqinternal3', 'sdqinternal4', 'sdqinternal5', 'sdqinternal6', 'sdqinternal7', 'sdqinternalsr7', 
#         'genwelb5', 'genwelb6',
#         'swemwbs7',
#         'lifesat2', 'lifesat3', 'lifesat4', 'lifesat5', 'lifesat6', 'lifesat7', 'lifesatsr7',
#         'lifesatwb5', 'lifesatwb6', 'lifesatwb7'
#         ]
# varl = ['lifesat2', 
#         'lifesat3', 
#         'lifesat4', 
#         'lifesat5', 'lifesatwb5',
#         'lifesat6', 'lifesatwb6',
#         'lifesat7', 'lifesatsr7',
#         'lifesatwb7'
#         ]
# sns.pairplot(mcsi[varl], kind = "reg")
# sns.pairplot(mcso[varl], kind = "reg")

# varl = ['zcog2', 
#         'zcog3', 
#         'zcog4', 
#         'zcog5',
#         'zcog6',
#         'zcog7'
#         ]
# sns.pairplot(mcsi[varl], kind = "reg")
# sns.pairplot(mcso[varl], kind = "reg")

# ### Income distribution with quintiles


# # Establish figure
# plt.figure()
# wt = mcsi['wt_uk2']
# # Draw the KDE
# sns.kdeplot(data=mcsi, x='income123', linewidth=1, fill=False, color="black", weights='wt_uk2', cut=0)

x = mcsi.loc[mcsi["male"] == 1, "sdqemotion7"]
y = mcsi.loc[mcsi["male"] == 0, "sdqemotion7"]

bins = np.linspace(0, 10, 10)

# plt.hist(mcsi.loc[mcsi["male"] == 1, "sdqemotion7"], bins, alpha=0.5, label='Male')
# plt.hist(mcsi.loc[mcsi["male"] == 0, "sdqemotion7"], bins, alpha=0.5, label='Female')
plt.hist([x, y], bins, label=['Male', 'Female'])
plt.legend(loc='upper right')
plt.show()

x = mcsi.loc[mcsi["male"] == 1, "sdqexternal7"]
y = mcsi.loc[mcsi["male"] == 0, "sdqexternal7"]

bins = np.linspace(0, 20, 20)

# plt.hist(mcsi.loc[mcsi["male"] == 1, "sdqexternal7"], bins, alpha=0.5, label='Male')
# plt.hist(mcsi.loc[mcsi["male"] == 0, "sdqexternal7"], bins, alpha=0.5, label='Female')
plt.hist([x, y], bins, label=['Male', 'Female'])
plt.legend(loc='upper right')
plt.show()

mcsi["swemwbs7"] = np.clip(mcsi["swemwbs7"], a_max=28, a_min=0)
x = mcsi.loc[mcsi["male"] == 1, "swemwbs7"]
y = mcsi.loc[mcsi["male"] == 0, "swemwbs7"]

bins = np.linspace(0, 28, 28)

# plt.hist(mcsi.loc[mcsi["male"] == 1, "sdqexternal7"], bins, alpha=0.5, label='Male')
# plt.hist(mcsi.loc[mcsi["male"] == 0, "sdqexternal7"], bins, alpha=0.5, label='Female')
plt.hist([x, y], bins, label=['Male', 'Female'])
plt.legend(loc='upper right')
plt.show()

# plt_path = os.path.join(lfsm, 'output/figures', 'inc123distq.png')
# # Get percentiles
# d1 = DescrStatsW(mcsi['income123'], weights=mcsi['wt_uk2'])
# q = d1.quantile(probs=(0, 0.2, 0.4, 0.6, 0.8, 1)) 
# # Create ax object to reference
# plt.figure(figsize=(15, 9))
# plt.rcParams.update({'font.size': 18})
# ax = sns.kdeplot(data=mcsi, x='income123', linewidth=1, fill=False, color="black", weights='wt_uk2', cut=0)
# xt = ax.get_xticks() 
# # Select the lines that are drawing the KDE
# kde_line = ax.get_lines()[-1]
# # Select all data bounded within the shape of the graph
# x, y = kde_line.get_data()

# # To shade the region of interest the color green
# qi = np.round(q, decimals=0).astype(int)
# # plt.fill_between(x, y, where = (x < q.iloc[1]) & (x >= q.iloc[0]), color='#dadae4',  label=f"Poorest fifth ({qi.iloc[0]}     - {qi.iloc[1]}  £s)")
# # plt.fill_between(x, y, where = (x < q.iloc[2]) & (x >= q.iloc[1]), color='#a8a8af',  label=f"Second fifth ({qi.iloc[1]} - {qi.iloc[2]}  £s)")
# # plt.fill_between(x, y, where = (x < q.iloc[3]) & (x >= q.iloc[2]), color='#585869', label=f"Middle fifth  ({qi.iloc[2]} - {qi.iloc[3]}  £s)")
# # plt.fill_between(x, y, where = (x < q.iloc[4]) & (x >= q.iloc[3]), color='#28282b', label=f"Fourth fifth  ({qi.iloc[3]} - {qi.iloc[4]}  £s)")
# # plt.fill_between(x, y, where = (x <= q.iloc[5]) & (x >= q.iloc[4]), color='#111111', label=f"Richest fifth ({qi.iloc[4]} - {qi.iloc[5]}  £s)")
# plt.fill_between(x, y, where = (x < q.iloc[1]) & (x >= q.iloc[0]), color='#dadae4',  label="Poorest quintile")
# plt.fill_between(x, y, where = (x < q.iloc[2]) & (x >= q.iloc[1]), color='#a8a8af',  label="Second quintile")
# plt.fill_between(x, y, where = (x < q.iloc[3]) & (x >= q.iloc[2]), color='#585869', label="Middle quintile")
# plt.fill_between(x, y, where = (x < q.iloc[4]) & (x >= q.iloc[3]), color='#28282b', label="Fourth quintile")
# plt.fill_between(x, y, where = (x <= q.iloc[5]) & (x >= q.iloc[4]), color='#111111', label="Richest quintile")
# # mask = x < q.iloc[1]
# # x, y = x[mask], y[mask]
# # ax.fill_between(x, y1=y, color="b", alpha=0.5)
# # mask = (x < q.iloc[2]) & (x > q.iloc[1])
# # x, y = x[mask], y[mask]
# # ax.fill_between(x, y1=y, color="g", alpha=0.5)

# # #Add xticks at quantile cut-offs

# # xt = np.round(xt, decimals=0)
# # xt = np.delete(xt, 0)
# # xt=np.append(xt, q.iloc[1])
# # xt=np.append(xt, q.iloc[2])
# # xt=np.append(xt, q.iloc[3])
# # xt=np.append(xt, q.iloc[4])
# # xt=np.append(xt, q.iloc[5])

# # ax.set_xticks(xt)

# plt.yticks(visible=False)
# plt.ylabel("")
# # plt.gca().set_yticklabels([])
# # Title & labels
# # plt.title("Height", pad=20)
# plt.xlabel("Early years OECD equivalised weekly household income")
# # plt.ylabel("Probability", labelpad=20)
# plt.legend()
# # Area of the shaded region when x < 67.5
# # area = np.trapz(y, x)
# # print(f"Probability: {area.round(4)}")
# # os.remove(plt_path)
# plt.savefig(plt_path, bbox_inches='tight')
# # Show plot
# plt.show()

# print(mcsi.groupby('incqnt123')['income123'].mean())

##### Loop simulation across all risk factors

for rskfctr in rskfctrs:
    
    ####################################################################################################
    #################### Log all console output
    ####################################################################################################
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    ltm = time.strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename= os.path.join(lfsm, f'output/{rskfctr}/log', f'lifesim2log_{rskfctr}_{ltm}.log'),
                        filemode='a+')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)

    # Add a separator line to indicate a new log
    logging.info('----------------------------------------------------------------------------------------------------------------------------------------------------------------')
    logging.info('----------------------------------------------------------------------------------------------------------------------------------------------------------------')
    logging.info('----------------------------------------------------------------------------------------------------------------------------------------------------------------')
    logging.info('----------------------------------------------------------------------------------------------------------------------------------------------------------------')
    logging.info('----------------------------------------------------------------------------------------------------------------------------------------------------------------')

    ## Option to display all data
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    # pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', None)
    # pd.set_option("display.max_rows", 1000)
    # pd.set_option("display.max_columns", 1000)
    # pd.set_option("display.max_colwidth", 1000)
    pd.set_option("display.expand_frame_repr", True)
    pd.set_option('display.width', 1000)

    #Log begining marker
    logging.info('Log start - ' + time.strftime("%Y-%m-%d %H:%M:%S"))
    #Number of universe description
    logging.info(f"Number of universes - {num_universes}")
    #Mark code start time 
    ostm = time.time()
    
    ##Coeffcients, errors and residual distributions from R
    beta = pd.read_csv(os.path.join(lfsm ,'regout' ,f'{rskfctr}_beta.csv'))
    se = pd.read_csv(os.path.join(lfsm ,'regout' ,f'{rskfctr}_se.csv'))
    res = pd.read_csv(os.path.join(lfsm ,'regout' ,f'{rskfctr}_res.csv'))
    #Clean column names
    beta = beta.rename(columns={'(Intercept)': 'intercept'})
    se = se.rename(columns={'(Intercept)': 'intercept'})
    #Set index as outcome variable 
    beta = beta.set_index('outcome')
    se = se.set_index('outcome')
    res = res.set_index('Unnamed: 0')
    #Temporary rename badgcseme
    beta = beta.rename(index={'bdgcseme': 'bdgcseme7'})
    se = se.rename(index={'bdgcseme': 'bdgcseme7'})
    res = res.rename(index={'bdgcseme': 'bdgcseme7'})
    #Drop unnecessary columns
    beta = beta.drop('Unnamed: 0', axis=1)
    se = se.drop('Unnamed: 0', axis=1)
    #Transpose data
    beta = beta.T
    se = se.T
    res = res.T
    
    
    
    ### Set Sweep start number 
    try:
        last_digit = int(rskfctr[-1:])
        if last_digit < 1:
            swpstn = 1
        else :    
            swpstn = int(rskfctr[-1:])    
    except ValueError:
        # This block executes if the last character cannot be converted to an integer
        swpstn = 1
    # if int(rskfctr[-1:]) < 1:
    #     swpstn = 1
    # else :    
    #     swpstn = int(rskfctr[-1:])
    
    ##### Outcomes list
    #Continuous outcomes
    coutcomes = [outcome for outcome in [
        "zcog2", "zcog3", "zcog4", "zcog5", "zcog6", "zcog7",
        "zbasnv2", "zbsrar2", "zbasnv3", "zbaspc3", "zbasps3", "zbaswr4", "zbaspc4", "znferpm4",
        "sdqconduct2", "sdqconduct3", "sdqconduct4", "sdqconduct5", "sdqconduct6", "sdqconduct7", #"sdqconductsr7",
        "sdqemotion2", "sdqemotion3", "sdqemotion4", "sdqemotion5", "sdqemotion6", "sdqemotion7", #"sdqemotionsr7",
        "sdqpeer2", "sdqpeer3", "sdqpeer4", "sdqpeer5", "sdqpeer6", "sdqpeer7", #"sdqpeersr7",
        "sdqprosoc2", "sdqprosoc3", "sdqprosoc4", "sdqprosoc5", "sdqprosoc6", "sdqprosoc7", #"sdqprosocsr7",
        "sdqhyper2", "sdqhyper3", "sdqhyper4", "sdqhyper5", "sdqhyper6", "sdqhyper7", #"sdqhypersr7",
        "sdqinternal2", "sdqinternal3", "sdqinternal4", "sdqinternal5", "sdqinternal6", "sdqinternal7", #"sdqinternalsr7", 
        "sdqexternal2", "sdqexternal3", "sdqexternal4", "sdqexternal5", "sdqexternal6", "sdqexternal7", #"sdqexternalsr7",
        #"genwelb5", "welbgrid5", "genwelb6", "welbgrid6", "swemwbs7", 
        "lifesat2", "lifesat3", "lifesat4", "lifesat5", "lifesat6", "lifesat7",
        #"zlifesat2", "zlifesat3", "zlifesat4", "zlifesat5", "zlifesat6", "zlifesat7",
        #"internal2", "internal3", "internal4", "internal5", "internal6", "internal7", #"internalsr7", 
        "kessler7"
    ] if int(outcome[-1:]) > swpstn]

    # Binary outcomes:
    boutcomes = [outcome for outcome in [
        "hosp2", "hosp3", "hosp4", "hosp5", "hosp6", "hosp7",
        #"obesity2", "obesity3", "obesity4", "obesity5", "obesity6",  
        "alc2", "alc3", "alc4", "alc5", "alc6", "alc7", 
        #"pccc2", "pccc3", "pccc4", "pccc23", "pccc34", "pccc234",
        "sen4", "sen5", "sen6", 
        "truancy5", "truancy6", 
        "excl5", "excl6",
        "smkreg6", 
        "smkreg7", "bdgcseme7", "obesity7", "prfrhlth7"
    ] if int(outcome[-1:]) > swpstn]
    
    
    ### Outcomes from regressions:
    outlst = beta.columns.tolist()
    # Continuous outcomes:
    coutlst = [outcome for outcome in outlst if outcome in coutcomes]
    # Binary outcomes:
    boutlst = [outcome for outcome in outlst if outcome in boutcomes]
    
    ###Descriptive Stats of full information sample
    ##All outcomes and exposures data set
    varl = ["income1", "income2", "income3", "income123", 
            "teenbrth", 
            "pretrm_1", "pretrm_2", "pretrm_3", 
            "bthwt_1", "bthwt_2", 
            "lwht3", 
            "alc23", 
            "deldev3",
            "wt_uk2"
            ]
    l = [x for x in itertools.chain(coutcomes, boutcomes, varl)]
    mcsb = mcsi[l]
    mcsb['distress7'] = np.where((mcsb['kessler7'] >= 13), 1, 0)
    mcsb.loc[mcs['kessler7'].isna(), 'distress7'] = np.nan
    
    print(min(mcs['zcog7']))
    ##Unweighted 
    # stats = ["nobs", "missing", "mean", "std", "percentiles"]
    # stats = ["nobs", "missing", "mean", "std"]
    stats = ["mean", "std_err"]
    #mean_values = history[varl].mean()
    #logging.info(mean_values)
    descf = sms.stats.descriptivestats.describe(mcsb, stats=stats, numeric=True, percentiles=(5, 25, 50, 75, 95) ).T
    descf = descf.rename(index=var_dict)
    logging.info('Unweighted outcome stats - full imputed sample')
    logging.info(descf)
    
    # Export to LaTeX with specified formatters and index
    tex_path = os.path.join(lfsm, f'output/{rskfctr}/tables', 'descuw.tex')  # Define your output file path here
    descf.to_latex(tex_path,
                   index=True,
                   formatters={"name": str.upper},
                   float_format="{:.3f}".format,
            )
    
    
    # ### Pre-imputation descriptive stats
    # #Keep only imputed data sets
    # mcsr = mcs[mcs['imputn'] == 0]
    # ### All 
    # ##All confounders and exposures data set
    # l = beta.index.tolist()[1:] + ['income123', 'country1_1', 'region1_7', 'ethnicity_1', 'imdqnt123_5', 'incqnt123_5', 'meduc1_5', 'male', 'incqnt123']
    # l.sort()
    # l = [item for item in l if item not in ['incqnt123_1', 'incqnt123_2', 'incqnt123_3', 'incqnt123_4', 'incqnt123_5']]
    # mcst = mcsr[l]
    # #Drop columns only included for descriptive stats 
    # mcsr = mcsr.drop(columns=['income123', 'country1_1', 'region1_7', 'ethnicity_1', 'imdqnt123_5', 'meduc1_5', 'male'])
    
    # ##Unweighted 
    # # stats = ["nobs", "missing", "mean", "std", "percentiles"]
    # # stats = ["nobs", "missing", "mean", "std"]
    # stats = ["nobs", "missing", "mean", "std_err"]
    # #mean_values = history[varl].mean()
    # #logging.info(mean_values)
    # descf = sms.stats.descriptivestats.describe(mcst[l], stats=stats, numeric=True, percentiles=(5, 25, 50, 75, 95)).T
    # descf['miss'] = descf['missing']*100/descf['nobs']
    # del descf['nobs']
    # del descf['missing']
    # descf = descf.rename(index=var_dict)
    # logging.info('Descriptive stats - random sample')
    # logging.info(descf)
    
    # # Export to LaTeX with specified formatters and index
    # tex_path = os.path.join(lfsm, 'output/tables', 'descrw.tex')  # Define your output file path here
    # descf.to_latex(tex_path,
    #                index=True,
    #                formatters={"name": str.upper},
    #                float_format="{:.3f}".format,
    #         )
    
    # del l
    # del mcst
    # del stats
    # del descf
    # del tex_path
    
    # ##All outcomes
    # l = boutcomes + coutcomes
    # mcst = mcsr[l]
    
    # ##Unweighted 
    # # stats = ["nobs", "missing", "mean", "std", "percentiles"]
    # # stats = ["nobs", "missing", "mean", "std"]
    # stats = ["nobs", "missing", "mean", "std_err"]
    # #mean_values = history[varl].mean()
    # #logging.info(mean_values)
    # descf = sms.stats.descriptivestats.describe(mcst[l], stats=stats, numeric=True, percentiles=(5, 25, 50, 75, 95)).T
    # descf['miss'] = descf['missing']*100/descf['nobs']
    # del descf['nobs']
    # del descf['missing']
    # descf = descf.rename(index=var_dict)
    # logging.info('Descriptive stats - random sample')
    # logging.info(descf)
    
    # # Export to LaTeX with specified formatters and index
    # tex_path = os.path.join(lfsm, 'output/tables', 'desorw.tex')  # Define your output file path here
    # descf.to_latex(tex_path,
    #                index=True,
    #                formatters={"name": str.upper},
    #                float_format="{:.3f}".format,
    #         )
    
    # del l
    # del mcst
    # del stats
    # del descf
    # del tex_path
    
    # del mcsr
    
    
    
    ####################################################################################################
    #################### Simulation set up
    ####################################################################################################
    ##### set the seed to allow us to reproduce stochastic results
    rsd = 986437
    np.random.seed(rsd)
    logging.info(f"Random seed - {rsd}")
    
    
    ## Keep only RHS and linkage
    # List of variables that aren't in used in regressions but may be needed later
    msc = ['income123', 'incqnt123', 'aincome123', 'laincome123',
           'country1_1', 'region1_7', 'ethnicity_1', 
           'imdqnt123_5', 'imdqnt1_5', 'imdqnt2_5', 'imdqnt3_5', 'incqnt1_5', 
           'meduc1_5', 
           'hheduc1_5', 
           'gpeduc7_5', 
           'incqnt123_1', 'incqnt123_2', 'incqnt123_3', 'incqnt123_4', 'incqnt123_5',
           'male',
           'income1', 'income2', 'income3', 'income6',
           'aincome1', 'incqnt1',
           'wealth5', 'wealth6',
           'imddec6', 'imdqnt6', 'imdqnt123',
           'numchld1', 'numchld2', 'numchld3',
           'teenbrth',
           'pretrm_1', 'pretrm_2', 'pretrm_3',
           'bthwt_1', 'bthwt_2',
           'lwht3',
           'alc3',
           'deldev3'
            ]
    if rskfctr in ("linctb0", "linctbim0", "lincnctb0"):
        msc = ['ils_dispy_uk_2023_std_wt', 'ils_dispy_uk_2023_1_std_wt', 'ils_dispy_uk_2023_2_std_wt', 'ils_dispy_uk_2023_3_std_wt'] + msc
        
    
    l = ['mcsid', 'wt_uk2'] + beta.index.tolist() + [item for item in msc if item not in beta.index.tolist()]
    mcsm = mcsi[l]
    mcsm = mcsm.dropna()
    mcsm = mcsm.reset_index() 
    print(mcsm.columns[mcsm.columns.str.startswith('incqnt123')].tolist())
    del l
    
    ##### Pick random subsample from imputed dataset
    logging.info(f"Cohort size - {num_people}")
    mcsr = mcsm.sample(n = num_people, replace = True, weights = mcsm['wt_uk2'], axis = 0, ignore_index = True)
    
    
    # # calculate the total number of individuals we will simulate
    # num_people = mcsr.shape[0]
    
    ###Descriptive Stats of random sample
    ##All confounders and exposures data set
    l = beta.index.tolist()[1:] + ['country1_1', 'region1_7', 'ethnicity_1', 'imdqnt123_5', 'incqnt123_5', 'meduc1_5', 'male', 'aincome123']
    l.sort()
    l = [item for item in l if item not in ['incqnt123_1', 'incqnt123_2', 'incqnt123_3', 'incqnt123_4', 'incqnt123_5']]
    mcst = mcsr[l]
    # # Drop columns only included for descriptive stats 
    # mcsr = mcsr.drop(columns=['country1_1', 'region1_7', 'ethnicity_1', 'imdqnt123_5', 'meduc1_5'])
    
    ##Unweighted 
    # stats = ["nobs", "missing", "mean", "std", "percentiles"]
    # stats = ["nobs", "missing", "mean", "std"]
    stats = ["mean", "std_err"]
    #mean_values = history[varl].mean()
    #logging.info(mean_values)
    descf = sms.stats.descriptivestats.describe(mcst, stats=stats, numeric=True, percentiles=(5, 25, 50, 75, 95)).T
    descf = descf.rename(index=var_dict)
    logging.info('Descriptive stats - random sample')
    logging.info(descf)
    
    # Export to LaTeX with specified formatters and index
    tex_path = os.path.join(lfsm, f'output/{rskfctr}/tables', 'descrs.tex')  # Define your output file path here
    descf.to_latex(tex_path,
                   index=True,
                   formatters={"name": str.upper},
                   float_format="{:.3f}".format,
            )
    
    del l
    del mcst
    del stats
    del descf
    del tex_path
    
    ##### Simulated Betas
    # create the simulated betas combining the means and standard errors to create 
    # alternative parameter universes that we will use to capture the uncertainty
    # in the estimated equation coefficients
    sim_betas = []
    if num_universes > 1:
      for i in tqdm(range(num_universes)):
          sim_mean = beta + np.random.normal(scale=se)
          # sim_mean = beta + np.random.normal(beta, se) - beta
          # sim_mean = beta
          sim_betas.append(sim_mean)
    else:
        sim_mean = beta + np.random.normal(scale=se)
        sim_betas.append(beta)
      
    del sim_mean
    ##### Simulated Residuals
    # create the simulated residuals based on standard deviation dof residuals from the regressions
    # We use pearson standard errors for logit residuals so they approximate a normal distribution
    # First extract the standard deviation from the raw residuals
    sd_res = res.mean()
    #
    sim_res = []
    for i in tqdm(range(num_universes)):
        tmp = pd.DataFrame()
        for column in sd_res.index:
            sim_grs = np.random.normal(0, sd_res[0], mcsr.shape[0])
            tmp[column] = sim_grs
        sim_res.append(tmp)
        
    del column 
    del tmp    
    del sd_res
    del sim_grs
      
    ##### Simulated Probabilities 
    # add these binary columns as probability draws for each person in MCS
    # these probabilities will be used to simulate if binary outcomes occur
    sim_probs = []
    bop = boutlst + [f"condis{i}" for i in range(swpstn + 1, nwvs + 1)]
    
    
    for i in range(num_universes):
        df = pd.DataFrame(data=np.random.rand(num_people, len(bop)), columns=bop)
        sim_probs.append(df)
    
    del df
    del bop
    del i
    #


    ##########Post intervention population change
    #####Income
    #### Income quintiles
    if rskfctr in ("incqnt", "incqntim", "incqntnc"):
        logging.info(mcsr['incqnt123_1'].value_counts())
        logging.info(mcsr['incqnt123_2'].value_counts())
        logging.info(mcsr['incqnt123_3'].value_counts())
        logging.info(mcsr['incqnt123_4'].value_counts())
        logging.info(mcsr['incqnt123_5'].value_counts())
        a = mcs.groupby("incqnt123")["aincome123"].min().reset_index(drop=True)
        interventions = [
            {'id': 'int0', 'name': 'qincb', 'code': "mcst['recip'] = 0\n"
                                                    "mcst['nwmninc123'] = mcst['aincome123']"},
            {'id': 'int1', 'name': 'qinc1', 'code': "mcst['incqnt123_1'] = mcst['incqnt123_1']\n"
                                                    "mcst['incqnt123_2'] = mcst['incqnt123_2']\n"
                                                    "mcst['incqnt123_3'] = mcst['incqnt123_3']\n"
                                                    "mcst['incqnt123_4'] = mcst['incqnt123_4']\n"
                                                    "mcst['recip'] = 0\n"
                                                    "mcst['recip'] = np.where(mcst['incqnt123_1'] == 1, 1, mcst['recip'])\n"
                                                    "mcst['incqnt123_2'] = np.where(mcst['incqnt123_1'] == 1, 1, mcst['incqnt123_2'])\n"
                                                    "mcst['incqnt123_1'] = np.where(mcst['incqnt123_1'] == 1, 0, mcst['incqnt123_1'])\n"
                                                    "mcst['nwmninc123'] = np.where(mcst['incqnt123'] == 1, a.iloc[1], mcst['aincome123'])\n"
                                                    "logging.info(mcst['incqnt123_1'].value_counts())\n"
                                                    "logging.info(mcst['incqnt123_2'].value_counts())\n"
                                                    "logging.info(mcst['incqnt123_3'].value_counts())\n"
                                                    "logging.info(mcst['incqnt123_4'].value_counts())"},
            {'id': 'int2', 'name': 'qinc2', 'code': "mcst['incqnt123_1'] = mcst['incqnt123_1']\n"
                                                    "mcst['incqnt123_2'] = mcst['incqnt123_2']\n"
                                                    "mcst['incqnt123_3'] = mcst['incqnt123_3']\n"
                                                    "mcst['incqnt123_4'] = mcst['incqnt123_4']\n"
                                                    "mcst['recip'] = 0\n"
                                                    "mcst['recip'] = np.where(mcst['incqnt123_1'] == 1, 1, mcst['recip'])\n"
                                                    "mcst['recip'] = np.where(mcst['incqnt123_2'] == 1, 1, mcst['recip'])\n"
                                                    "mcst['incqnt123_3'] = np.where(mcst['incqnt123_1'] == 1, 1, mcst['incqnt123_3'])\n"
                                                    "mcst['incqnt123_3'] = np.where(mcst['incqnt123_2'] == 1, 1, mcst['incqnt123_3'])\n"
                                                    "mcst['incqnt123_1'] = np.where(mcst['incqnt123_1'] == 1, 0, mcst['incqnt123_1'])\n"
                                                    "mcst['incqnt123_2'] = np.where(mcst['incqnt123_2'] == 1, 0, mcst['incqnt123_2'])\n"
                                                    "mcst['nwmninc123'] = np.where(mcst['incqnt123'] <= 2, a.iloc[2], mcst['aincome123'])\n"
                                                    "logging.info(mcst['incqnt123_1'].value_counts())\n"
                                                    "logging.info(mcst['incqnt123_2'].value_counts())\n"
                                                    "logging.info(mcst['incqnt123_3'].value_counts())\n"
                                                    "logging.info(mcst['incqnt123_4'].value_counts())"},
            {'id': 'int3', 'name': 'qinc3', 'code': "mcst['incqnt123_1'] = mcst['incqnt123_1']\n"
                                                    "mcst['incqnt123_2'] = mcst['incqnt123_2']\n"
                                                    "mcst['incqnt123_3'] = mcst['incqnt123_3']\n"
                                                    "mcst['incqnt123_4'] = mcst['incqnt123_4']\n"
                                                    "mcst['recip'] = 0\n"
                                                    "mcst['recip'] = np.where(mcst['incqnt123_1'] == 1, 1, mcst['recip'])\n"
                                                    "mcst['incqnt123_1'] = np.where(mcst['incqnt123_1'] == 1, 0, mcst['incqnt123_1'])\n"
                                                    "mcst['nwmninc123'] = np.where(mcst['incqnt123'] == 1, a.iloc[4], mcst['aincome123'])\n"
                                                    "logging.info(mcst['incqnt123_1'].value_counts())\n"
                                                    "logging.info(mcst['incqnt123_2'].value_counts())\n"
                                                    "logging.info(mcst['incqnt123_3'].value_counts())\n"
                                                    "logging.info(mcst['incqnt123_4'].value_counts())"},
            {'id': 'int4', 'name': 'qinc4', 'code': "mcst['incqnt123_1'] = mcst['incqnt123_1']\n"
                                                    "mcst['incqnt123_2'] = mcst['incqnt123_2']\n"
                                                    "mcst['incqnt123_3'] = mcst['incqnt123_3']\n"
                                                    "mcst['incqnt123_4'] = mcst['incqnt123_4']\n"
                                                    "mcst['recip'] = 0\n"
                                                    "mcst['recip'] = np.where(mcst['incqnt123_1'] == 1, 1, mcst['recip'])\n"
                                                    "mcst['recip'] = np.where(mcst['incqnt123_2'] == 1, 1, mcst['recip'])\n"
                                                    "mcst['recip'] = np.where(mcst['incqnt123_3'] == 1, 1, mcst['recip'])\n"
                                                    "mcst['recip'] = np.where(mcst['incqnt123_4'] == 1, 1, mcst['recip'])\n"
                                                    "mcst['incqnt123_1'] = np.where(mcst['incqnt123_1'] == 1, 0, mcst['incqnt123_1'])\n"
                                                    "mcst['incqnt123_2'] = np.where(mcst['incqnt123_2'] == 1, 0, mcst['incqnt123_2'])\n"
                                                    "mcst['incqnt123_3'] = np.where(mcst['incqnt123_3'] == 1, 0, mcst['incqnt123_3'])\n"
                                                    "mcst['incqnt123_4'] = np.where(mcst['incqnt123_4'] == 1, 0, mcst['incqnt123_4'])\n"
                                                    "mcst['nwmninc123'] = np.where(mcst['incqnt123'] < 5, a.iloc[4], mcst['aincome123'])\n"
                                                    "logging.info(mcst['incqnt123_1'].value_counts())\n"
                                                    "logging.info(mcst['incqnt123_2'].value_counts())\n"
                                                    "logging.info(mcst['incqnt123_3'].value_counts())\n"
                                                    "logging.info(mcst['incqnt123_4'].value_counts())"}
                        ]
    #### Income quintiles
    elif rskfctr in ("incqnt1", "incqntim1", "incqntnc1"):
        logging.info(mcsr['incqnt1_1'].value_counts())
        logging.info(mcsr['incqnt1_2'].value_counts())
        logging.info(mcsr['incqnt1_3'].value_counts())
        logging.info(mcsr['incqnt1_4'].value_counts())
        logging.info(mcsr['incqnt1_5'].value_counts())
        a = mcs.groupby("incqnt1")["aincome1"].min().reset_index(drop=True)
        interventions = [
            {'id': 'int0', 'name': 'qincb', 'code': "mcst['recip'] = 0\n"
                                                    "mcst['nwmninc1'] = mcst['aincome1']"},
            {'id': 'int1', 'name': 'qinc1', 'code': "mcst['incqnt1_1'] = mcst['incqnt1_1']\n"
                                                    "mcst['incqnt1_2'] = mcst['incqnt1_2']\n"
                                                    "mcst['incqnt1_3'] = mcst['incqnt1_3']\n"
                                                    "mcst['incqnt1_4'] = mcst['incqnt1_4']\n"
                                                    "mcst['recip'] = 0\n"
                                                    "mcst['recip'] = np.where(mcst['incqnt1_1'] == 1, 1, mcst['recip'])\n"
                                                    "mcst['incqnt1_2'] = np.where(mcst['incqnt1_1'] == 1, 1, mcst['incqnt1_2'])\n"
                                                    "mcst['incqnt1_1'] = np.where(mcst['incqnt1_1'] == 1, 0, mcst['incqnt1_1'])\n"
                                                    "mcst['nwmninc1'] = np.where(mcst['incqnt1'] == 1, a.iloc[1], mcst['aincome1'])\n"
                                                    "logging.info(mcst['incqnt1_1'].value_counts())\n"
                                                    "logging.info(mcst['incqnt1_2'].value_counts())\n"
                                                    "logging.info(mcst['incqnt1_3'].value_counts())\n"
                                                    "logging.info(mcst['incqnt1_4'].value_counts())"},
            {'id': 'int2', 'name': 'qinc2', 'code': "mcst['incqnt1_1'] = mcst['incqnt1_1']\n"
                                                    "mcst['incqnt1_2'] = mcst['incqnt1_2']\n"
                                                    "mcst['incqnt1_3'] = mcst['incqnt1_3']\n"
                                                    "mcst['incqnt1_4'] = mcst['incqnt1_4']\n"
                                                    "mcst['recip'] = 0\n"
                                                    "mcst['recip'] = np.where(mcst['incqnt1_1'] == 1, 1, mcst['recip'])\n"
                                                    "mcst['recip'] = np.where(mcst['incqnt1_2'] == 1, 1, mcst['recip'])\n"
                                                    "mcst['incqnt1_3'] = np.where(mcst['incqnt1_1'] == 1, 1, mcst['incqnt1_3'])\n"
                                                    "mcst['incqnt1_3'] = np.where(mcst['incqnt1_2'] == 1, 1, mcst['incqnt1_3'])\n"
                                                    "mcst['incqnt1_1'] = np.where(mcst['incqnt1_1'] == 1, 0, mcst['incqnt1_1'])\n"
                                                    "mcst['incqnt1_2'] = np.where(mcst['incqnt1_2'] == 1, 0, mcst['incqnt1_2'])\n"
                                                    "mcst['nwmninc1'] = np.where(mcst['incqnt1'] <= 2, a.iloc[2], mcst['aincome1'])\n"
                                                    "logging.info(mcst['incqnt1_1'].value_counts())\n"
                                                    "logging.info(mcst['incqnt1_2'].value_counts())\n"
                                                    "logging.info(mcst['incqnt1_3'].value_counts())\n"
                                                    "logging.info(mcst['incqnt1_4'].value_counts())"},
            {'id': 'int3', 'name': 'qinc3', 'code': "mcst['incqnt1_1'] = mcst['incqnt1_1']\n"
                                                    "mcst['incqnt1_2'] = mcst['incqnt1_2']\n"
                                                    "mcst['incqnt1_3'] = mcst['incqnt1_3']\n"
                                                    "mcst['incqnt1_4'] = mcst['incqnt1_4']\n"
                                                    "mcst['recip'] = 0\n"
                                                    "mcst['recip'] = np.where(mcst['incqnt1_1'] == 1, 1, mcst['recip'])\n"
                                                    "mcst['incqnt1_1'] = np.where(mcst['incqnt1_1'] == 1, 0, mcst['incqnt1_1'])\n"
                                                    "mcst['nwmninc1'] = np.where(mcst['incqnt1'] == 1, a.iloc[4], mcst['aincome1'])\n"
                                                    "logging.info(mcst['incqnt1_1'].value_counts())\n"
                                                    "logging.info(mcst['incqnt1_2'].value_counts())\n"
                                                    "logging.info(mcst['incqnt1_3'].value_counts())\n"
                                                    "logging.info(mcst['incqnt1_4'].value_counts())"},
            {'id': 'int4', 'name': 'qinc4', 'code': "mcst['incqnt1_1'] = mcst['incqnt1_1']\n"
                                                    "mcst['incqnt1_2'] = mcst['incqnt1_2']\n"
                                                    "mcst['incqnt1_3'] = mcst['incqnt1_3']\n"
                                                    "mcst['incqnt1_4'] = mcst['incqnt1_4']\n"
                                                    "mcst['recip'] = 0\n"
                                                    "mcst['recip'] = np.where(mcst['incqnt1_1'] == 1, 1, mcst['recip'])\n"
                                                    "mcst['recip'] = np.where(mcst['incqnt1_2'] == 1, 1, mcst['recip'])\n"
                                                    "mcst['recip'] = np.where(mcst['incqnt1_3'] == 1, 1, mcst['recip'])\n"
                                                    "mcst['recip'] = np.where(mcst['incqnt1_4'] == 1, 1, mcst['recip'])\n"
                                                    "mcst['incqnt1_1'] = np.where(mcst['incqnt1_1'] == 1, 0, mcst['incqnt1_1'])\n"
                                                    "mcst['incqnt1_2'] = np.where(mcst['incqnt1_2'] == 1, 0, mcst['incqnt1_2'])\n"
                                                    "mcst['incqnt1_3'] = np.where(mcst['incqnt1_3'] == 1, 0, mcst['incqnt1_3'])\n"
                                                    "mcst['incqnt1_4'] = np.where(mcst['incqnt1_4'] == 1, 0, mcst['incqnt1_4'])\n"
                                                    "mcst['nwmninc1'] = np.where(mcst['incqnt1'] < 5, a.iloc[4], mcst['aincome1'])\n"
                                                    "logging.info(mcst['incqnt1_1'].value_counts())\n"
                                                    "logging.info(mcst['incqnt1_2'].value_counts())\n"
                                                    "logging.info(mcst['incqnt1_3'].value_counts())\n"
                                                    "logging.info(mcst['incqnt1_4'].value_counts())"}
                        ]
    #### Log Income
    elif rskfctr in ("linc0", "lincim0", "lincnc0"):    
        interventions = [
            {'id': 'int0', 'name': 'lincb', 'code': "mcst['recip'] = 0"},
            {'id': 'int1', 'name': 'linc1', 'code': "mcst['recip'] = 0\n"
                                                    "mcst['recip'] = np.where(mcst['incqnt123'] == 1, 1, mcst['recip'])\n"
                                                    "mcst['aincome123'] = np.where(mcst['incqnt123'] == 1, mcst['aincome123'] + 667, mcst['aincome123'])\n"
                                                    "mcst['laincome123'] = np.log(mcst['aincome123'])"},
            {'id': 'int2', 'name': 'linc2', 'code': "mcst['recip'] = 0\n"
                                                    "mcst['recip'] = np.where(mcst['incqnt123'] <= 2, 1, mcst['recip'])\n"
                                                    "mcst['aincome123'] = np.where(mcst['incqnt123'] <= 2, mcst['aincome123'] + 667, mcst['aincome123'])\n"
                                                    "mcst['laincome123'] = np.log(mcst['aincome123'])"},
            {'id': 'int3', 'name': 'linc3', 'code': "mcst['recip'] = np.where(mcst['incqnt123'] <= 5, 1, 0)\n"
                                                    "mcst['aincome123'] = np.where(mcst['incqnt123'] <= 5, mcst['aincome123'] + 667, mcst['aincome123'])\n"
                                                    "mcst['laincome123'] = np.log(mcst['aincome123'])"}
                        ]
        # interventions = [
        #     {'id': 'int0', 'name': 'lincb', 'code': "mcst['recip'] = 0"},
        #     {'id': 'int1', 'name': 'linc1', 'code': "mcst['recip'] = 0\n"
        #                                             "mcst['recip'] = np.where(mcst['incqnt123'] == 1, 1, mcst['recip'])\n"
        #                                             "mcst['aincome123'] = np.where(mcst['incqnt123'] == 1, mcst['aincome123'] + 737, mcst['aincome123'])\n"
        #                                             "mcst['laincome123'] = np.log(mcst['aincome123'])"},
        #     {'id': 'int2', 'name': 'linc2', 'code': "mcst['recip'] = 0\n"
        #                                             "mcst['recip'] = np.where(mcst['incqnt123'] <= 2, 1, mcst['recip'])\n"
        #                                             "mcst['aincome123'] = np.where(mcst['incqnt123'] <= 2, mcst['aincome123'] + 737, mcst['aincome123'])\n"
        #                                             "mcst['laincome123'] = np.log(mcst['aincome123'])"},
        #     {'id': 'int3', 'name': 'linc3', 'code': "mcst['recip'] = np.where(mcst['incqnt123'] <= 5, 1, 0)\n"
        #                                             "mcst['aincome123'] = np.where(mcst['incqnt123'] <= 5, mcst['aincome123'] + 737, mcst['aincome123'])\n"
        #                                             "mcst['laincome123'] = np.log(mcst['aincome123'])"}
        #                 ]
    #### Log Income (tax benefit)
    elif rskfctr in ("linctb0", "linctbim0", "lincnctb0"):    
        interventions = [
            {'id': 'int0', 'name': 'linctbb', 'code': "mcst['recip'] = 0\n"
                                                      "mcst['aincome123'] = mcst['ils_dispy_uk_2023_std_wt']*12\n"
                                                      "mcst['laincome123'] = np.log(mcst['aincome123'])\n"
                                                      "mcst['nwmninc123'] = mcst['ils_dispy_uk_2023_std_wt']"},
            {'id': 'int1', 'name': 'linctb1', 'code': "mcst['recip'] = 0\n"
                                                      "mcst['recip'] = np.where(mcst['ils_dispy_uk_2023_std_wt'] == mcst['ils_dispy_uk_2023_1_std_wt'], mcst['recip'], 1)\n"
                                                      "mcst['aincome123'] = mcst['ils_dispy_uk_2023_1_std_wt']*12\n"
                                                      "mcst['laincome123'] = np.log(mcst['aincome123'])\n"
                                                      "mcst['nwmninc123'] = mcst['ils_dispy_uk_2023_std_wt']"},
            {'id': 'int2', 'name': 'linctb2', 'code': "mcst['recip'] = 0\n"
                                                      "mcst['recip'] = np.where(mcst['ils_dispy_uk_2023_std_wt'] == mcst['ils_dispy_uk_2023_2_std_wt'], mcst['recip'], 1)\n"
                                                      "mcst['aincome123'] = mcst['ils_dispy_uk_2023_2_std_wt']*12\n"
                                                      "mcst['laincome123'] = np.log(mcst['aincome123'])\n"
                                                      "mcst['nwmninc123'] = mcst['ils_dispy_uk_2023_std_wt']"},
            {'id': 'int3', 'name': 'linctb3', 'code': "mcst['recip'] = 0\n"
                                                      "mcst['recip'] = np.where(mcst['ils_dispy_uk_2023_std_wt'] == mcst['ils_dispy_uk_2023_3_std_wt'], mcst['recip'], 1)\n"
                                                      "mcst['aincome123'] = mcst['ils_dispy_uk_2023_3_std_wt']*12\n"
                                                      "mcst['laincome123'] = np.log(mcst['aincome123'])\n"
                                                      "mcst['nwmninc123'] = mcst['ils_dispy_uk_2023_std_wt']"}
                        ]      
    #### Income
    elif rskfctr in ("inc", "incim", "incnc"):
        interventions = [
            {'id': 'int0', 'name': 'incb', 'code': "mcst['recip'] = 0"},
            {'id': 'int1', 'name': 'inc1', 'code': "mcst['recip'] = 0\n"
                                                   "mcst['recip'] = np.where(mcst['incqnt123'] == 1, 1, mcst['recip'])\n"
                                                   "mcst['aincome123'] = np.where(mcst['incqnt123'] == 1, mcst['aincome123'] + 667, mcst['aincome123'])"},
            {'id': 'int2', 'name': 'inc2', 'code': "mcst['recip'] = 0\n"
                                                   "mcst['recip'] = np.where(mcst['incqnt123'] <= 2, 1, mcst['recip'])\n"
                                                   "mcst['aincome123'] = np.where(mcst['incqnt123'] <= 2, mcst['aincome123'] + 667, mcst['aincome123'])"},
            {'id': 'int3', 'name': 'inc3', 'code': "mcst['recip'] = np.where(mcst['incqnt123'] <= 5, 1, 0)\n"
                                                   "mcst['aincome123'] = np.where(mcst['incqnt123'] <= 5, mcst['aincome123'] + 667, mcst['aincome123'])"}
                        ]
    #### Have a teenage mother
    elif rskfctr in ("teenbrth0", "teenbrthim0", "teenbrthnc0"):
        interventions = [
            {'id': 'int0', 'name': 'tnbb', 'code': "mcst['recip'] = 0"},
            {'id': 'int1', 'name': 'tnb1', 'code': "mcst['recip'] = np.where(mcst['teenbrth'] == 1, 1, 0)\n"
                                                   "mcst['teenbrth'] = np.where(mcst['teenbrth'] == 1, 0, mcst['teenbrth'])"}
                        ]
    #### Pre-term birth
    elif rskfctr in ("pretrmbrth0", "pretrmbrthim0", "pretrmbrthnc0"):
        interventions = [
            {'id': 'int0', 'name': 'ptbb', 'code': "mcst['recip'] = 0"},
            {'id': 'int1', 'name': 'ptb1', 'code': "mcst['recip'] = 0\n"
                                                   "mcst['recip'] = np.where(mcst['pretrm_3'] == 1, 1, mcst['recip'])\n"
                                                   "mcst['recip'] = np.where(mcst['pretrm_2'] == 1, 1, mcst['recip'])\n"
                                                   "mcst['recip'] = np.where(mcst['pretrm_1'] == 1, 1, mcst['recip'])\n"
                                                   "mcst['pretrm_3'] = np.where(mcst['pretrm_3'] == 1, 0, mcst['pretrm_3'])\n"
                                                   "mcst['pretrm_2'] = np.where(mcst['pretrm_2'] == 1, 0, mcst['pretrm_2'])\n"
                                                   "mcst['pretrm_1'] = np.where(mcst['pretrm_1'] == 1, 0, mcst['pretrm_1'])"},
            {'id': 'int2', 'name': 'ptb2', 'code': "mcst['recip'] = 0\n"
                                                    "mcst['recip'] = np.where(mcst['pretrm_3'] == 1, 1, mcst['recip'])\n"
                                                    "mcst['recip'] = np.where(mcst['pretrm_2'] == 1, 1, mcst['recip'])\n"
                                                    "mcst['pretrm_3'] = np.where(mcst['pretrm_3'] == 1, 0, mcst['pretrm_3'])\n"
                                                    "mcst['pretrm_2'] = np.where(mcst['pretrm_2'] == 1, 0, mcst['pretrm_2'])"},
            {'id': 'int3', 'name': 'ptb3', 'code': "mcst['recip'] = 0\n"
                                                    "mcst['recip'] = np.where(mcst['pretrm_3'] == 1, 1, mcst['recip'])\n"
                                                    "mcst['pretrm_3'] = np.where(mcst['pretrm_3'] == 1, 0, mcst['pretrm_3'])"} 
            ]
    #### Low birth weight
    elif rskfctr in ("lwbrthwt0", "lwbrthwtim0", "lwbrthwtnc0"):
        interventions = [
            {'id': 'int0', 'name': 'lbwb', 'code': "mcst['recip'] = 0"},
            {'id': 'int1', 'name': 'lbw1', 'code': "mcst['recip'] = 0\n"
                                                    "mcst['recip'] = np.where(mcst['bthwt_1'] == 1, 1, mcst['recip'])\n"
                                                    "mcst['bthwt_1'] = np.where(mcst['bthwt_1'] == 1, 0, mcst['bthwt_1'])"},
            {'id': 'int2', 'name': 'lbw2', 'code': "mcst['recip'] = 0\n"
                                                   "mcst['recip'] = np.where(mcst['bthwt_1'] == 1, 1, mcst['recip'])\n"
                                                   "mcst['recip'] = np.where(mcst['bthwt_2'] == 1, 1, mcst['recip'])\n"
                                                   "mcst['bthwt_1'] = np.where(mcst['bthwt_1'] == 1, 0, mcst['bthwt_1'])\n"
                                                   "mcst['bthwt_2'] = np.where(mcst['bthwt_2'] == 1, 0, mcst['bthwt_2'])"}
                        ]
    #### Low height at age 5
    elif rskfctr in ("lwht3", "lwhtim3", "lwhtnc3"):
        interventions = [
            {'id': 'int0', 'name': 'lhtb', 'code': "mcst['recip'] = 0"},
            {'id': 'int1', 'name': 'lht1', 'code': "mcst['recip'] = np.where(mcst['lwht3'] == 1, 1, 0)\n"
                                                   "mcst['lwht3'] = np.where(mcst['lwht3'] == 1, 0, mcst['lwht3'])"}
                        ]
    #### Disability at age 5
    elif rskfctr in ("dsblty3", "dsbltyim3", "dsbltync3"):
        interventions = [
            {'id': 'int0', 'name': 'dabb', 'code': "mcst['recip'] = 0"},
            {'id': 'int1', 'name': 'dab1', 'code': "mcst['recip'] = np.where(mcst['alc3'] == 1, 1, 0)\n"
                                                   "mcst['alc3'] = np.where(mcst['alc3'] == 1, 0, mcst['alc3'])"}
                        ]
    #### Have a teenage mother
    elif rskfctr in ("deldev3", "deldevim3", "deldevnc3"):
        interventions = [
            {'id': 'int0', 'name': 'ddvb', 'code': "mcst['recip'] = 0"},
            {'id': 'int1', 'name': 'ddv1', 'code': "mcst['recip'] = np.where(mcst['deldev3'] == 1, 1, 0)\n"
                                                   "mcst['deldev3'] = np.where(mcst['deldev3'] == 1, 0, mcst['deldev3'])"}
                        ]
    # interventions = [
    #     {'id': 'int0', 'name': 'incb', 'code': ''},
    #                 ]
    # # 1st intervention - Shift bottom quintile up to 2nd quintile
    # mcsr['incqnt123_2'] = np.where(mcsr['incqnt123_1'] == 1, 1, mcsr['incqnt123_2'])
    # mcsr['incqnt123_1'] = np.where(mcsr['incqnt123_1'] == 1, 0, mcsr['incqnt123_1'])
    
    # # 2nd intervention - Shift bottom 2 quintles up to 3rd quintile
    # mcsr['incqnt123_3'] = np.where(mcsr['incqnt123_1'] == 1, 1, mcsr['incqnt123_3'])
    # mcsr['incqnt123_3'] = np.where(mcsr['incqnt123_2'] == 1, 1, mcsr['incqnt123_3'])
    # mcsr['incqnt123_1'] = np.where(mcsr['incqnt123_1'] == 1, 0, mcsr['incqnt123_1'])
    # mcsr['incqnt123_2'] = np.where(mcsr['incqnt123_2'] == 1, 0, mcsr['incqnt123_2'])
    
    # # 3rd intervention - Shift all quintiles up to top quintile
    # mcsr['incqnt123_1'] = np.where(mcsr['incqnt123_1'] == 1, 0, mcsr['incqnt123_1'])
    # mcsr['incqnt123_2'] = np.where(mcsr['incqnt123_2'] == 1, 0, mcsr['incqnt123_2'])
    # mcsr['incqnt123_3'] = np.where(mcsr['incqnt123_3'] == 1, 0, mcsr['incqnt123_3'])
    # mcsr['incqnt123_4'] = np.where(mcsr['incqnt123_4'] == 1, 0, mcsr['incqnt123_4'])
    
    # # 4th intervention - Shift bottom quintile up to top quintile
    # mcsr['incqnt123_1'] = np.where(mcsr['incqnt123_1'] == 1, 0, mcsr['incqnt123_1'])
    
    
    # ##### Teenage Pregnancy 
    # # 1st intervention
    # mcsr['teenbrth'] = np.where(mcsr['teenbrth'] == 1, 0, mcsr['teenbrth'])
    # mcsr['pretrm3'] = 0
    # mcsr['pretrm2'] = 0
    # mcsr['pretrm1'] = 0
    # mcsr['bthwt_1'] = 0
    # mcsr['bthwt_2'] = 0
    
    ####################################################################################################        
    #################### Run Simulations
    ####################################################################################################
    
    
    sys.path.append(lfsm)      
    #Define ducntion for simulation
    from Person_ijm25 import Person1 # type: ignore
    print(inspect.getsource(Person1))
    # for a given person in MCS create an instance of the Person class and simulate
    # their life from ages 0 to 17 returning key events in their life history
    def simulate_person(mcs, beta, res, probs, co, bo, sw):
        person1 = Person1(mcs, beta, res, probs, co, bo, sw)
        return person1.history  
        #return person1.c
        #return person1.b
    
    
        
    ### Begin Simulation
    logging.info('Simulation start time - ' + time.strftime("%H:%M:%S"))
    sstm = time.time()
    for intervention in interventions:
        
    
        #Create temporary datafile to change for interventions
        mcst = mcsr.copy()
        #Run intervention code
        exec(intervention['code'])
        # the output file for working copy of simulation data
        wcp_pth = lfsm + f"/output/{rskfctr}/" + "wrkcp_nu" + str(num_universes) + intervention['id'] + ".csv"
        # the output file where we will save the simualtion history to
        hst_pth = lfsm + f"/output/{rskfctr}/" + time.strftime("%Y%m%d-%H%M%S") + "_history_nu" + str(num_universes) + intervention['name'] + ".csv"
        # run the simulation for each parameter universe and intervention
        for n in tqdm(range(num_universes)):
            start_time = time.time()
            
            # use joblib to run the function in parallel for each individual in the MCS
            #est = ducntion for simulatiosim_betas[n], sim_res[n], sim_probs[n], coutcomes, boutcomes)
            est = simulate_person(mcst, sim_betas[n], sim_res[n], sim_probs[n], coutlst, boutlst, swpstn)
            est.insert(0, 'simulation', n)
            
            # Append mcsr['wt_uk2'] to the end of est
            est['wt_uk2'] = mcst['wt_uk2']
            est['incqnt123_1'] = mcst['incqnt123_1']
            est['incqnt123_2'] = mcst['incqnt123_2']
            est['incqnt123_3'] = mcst['incqnt123_3']
            est['incqnt123_4'] = mcst['incqnt123_4']
            est['incqnt123_5'] = mcst['incqnt123_5']
            est['incqnt123'] = mcst['incqnt123']
            est['teenbrth'] = mcst['teenbrth']
            est['pretrm_1'] = mcst['pretrm_1']
            est['pretrm_2'] = mcst['pretrm_2']
            est['pretrm_3'] = mcst['pretrm_3']
            est['bthwt_1'] = mcst['bthwt_1']
            est['bthwt_2'] = mcst['bthwt_2']
            est['aincome123'] = mcst['aincome123']
            est['laincome123'] = mcst['laincome123']
            if rskfctr in ("incqnt0", "incqntim0", "incqntnc0", "linctb0"):
                est['nwmninc123'] = mcst['nwmninc123']
            #For merge with LifeSim 1
            est['income123'] = mcst['income123']
            est['income1'] = mcst['income1']
            est['income2'] = mcst['income2']
            est['income3'] = mcst['income3']
            est['income6'] = mcst['income6']
            est['wealth5'] = mcst['wealth5']
            est['wealth6'] = mcst['wealth6']
            est['imddec6'] = mcst['imddec6']
            est['male'] = mcst['male']
            est['numchld1'] = mcst['numchld1']
            est['numchld2'] = mcst['numchld2']
            est['numchld3'] = mcst['numchld3']
            est['recip'] = mcst['recip']
            
            if n == 0:
            
                history = est 
                
            else:   
            
                history = pd.concat([history, est])
            
            
            # append the overall history for all simulated MCS individuals for this parameter universe
            # out to output_file adding a header row if file does not already exist 
        
        
            elapsed_time = time.time() - start_time
            
            # logging.info the elapsed time
            logging.info(f"Simulation {n+1} of {intervention['id']} for risk factor {rskfctr} took {elapsed_time:.2f} seconds to run.")
            
        del est 
        del elapsed_time
        del start_time
        del n
        # logging.info(mcst['mcsid'].isnull().sum()) 
        # logging.info(history['mcsid'].isnull().sum())
        # logging.info(mcst['wt_uk2'].isnull().sum())  
        # logging.info(history['wt_uk2'].isnull().sum()) 
        ##### Adding costs (Ages)
        #Wave 1 - 3 Years (9 months -> 0, 1, 2)
        #Wave 2 - 2 Years ( 3 ->  3,  4)
        #Wave 3 - 2 Years ( 5 ->  5,  6)
        #Wave 4 - 3 Years ( 7 ->  7,  8,  9)
        #Wave 5 - 3 Years (11 -> 10, 11, 12)
        #Wave 6 - 3 Years (14 -> 13, 14, 15)
        #Wave 7 - 3 Years (17 -> 16, 17, 18)
        
        
        ### Calculating costs for simulation
        #Conduct Disorder
        #"conditions2", "hosp2", "conditions3", "hosp3", "conditions4", "hosp4", "senstate4", "conditions5", "hosp5", "senstate5", "truancy5", "excl5", "antisoc5b", "conditions6", "hosp6", "senstate6", "truancy6", "excl6", "antisoc6b", "cigregular6", "badgcse_me", "conditions7", "hosp7", "antisoc7b", "cigregular7", "anxdepcurrent7", "mhconditions7", "physicalcond7"
        if swpstn < 7:
            history['cd_cst7'] = history['condis7'] * (costs.loc['cd2', 'ttl'] + costs.loc['cd3', 'ttl'] * 2) / 3  
            if swpstn < 6:
                history['cd_cst6'] = history['condis6'] * costs.loc['cd2', 'ttl']
                if swpstn < 5:
                    history['cd_cst5'] = history['condis5'] * (costs.loc['cd1', 'ttl'] + costs.loc['cd2', 'ttl'] * 2) / 3 
                    if swpstn < 4:
                        history['cd_cst4'] = history['condis4'] * costs.loc['cd1', 'ttl']
                        if swpstn < 3:
                            history['cd_cst3'] = history['condis3'] * costs.loc['cd1', 'ttl']
                            #if swpstn < 2:
                                # history['cd_cst2'] = history['condis2']*2943
     
        # Persistent Truancy
        ptp = 175 / 946
        if swpstn < 6:
            history['pt_cst6'] = history['truancy6'] * costs.loc['tru', 'ttl'] * ptp
            history['pt_cst7'] = history['truancy6'] * costs.loc['tru', 'ttl'] * ptp
            if swpstn < 5:
                history['pt_cst5'] = history['truancy5'] * costs.loc['tru', 'ttl'] * ptp
        # School Exclusion
        pep5 = 20 / 193
        pep6 = 49 / 636
        if swpstn < 6:
            history['pe_cst6'] = history['excl6'] * costs.loc['exc', 'ttl'] * pep6
            history['pe_cst7'] = history['excl6'] * costs.loc['exc', 'ttl'] * pep6
            if swpstn < 5:
                history['pe_cst5'] = history['excl5'] * costs.loc['exc', 'ttl'] * pep5
        

        # Special Education Needs
        if swpstn < 6:
            history['se_cst6'] = history['sen6'] * costs.loc['sen', 'ttl']
            history['se_cst7'] = history['sen6'] * costs.loc['sen', 'ttl']
            if swpstn < 5:
                history['se_cst4'] = history['sen4'] * costs.loc['sen', 'ttl']
                if swpstn < 4:
                    history['se_cst5'] = history['sen5'] * costs.loc['sen', 'ttl']
        

        # Disability
        for i in range(swpstn + 1, nwvs+1):
            history[f'll_cst{i}'] = history[f'alc{i}'] * costs.loc['dis', 'ttl']

        # Hospitalisation
        for i in range(swpstn + 1, nwvs+1):
            history[f'hc_cst{i}'] = history[f'hosp{i}'] * costs.loc['hos', 'ttl']
        
        # Full childhood costs
        if swpstn < 7:
            history['tot_cst7'] = history['cd_cst7'] +  history['ll_cst7'] + history['hc_cst7'] + history['se_cst6'] + history['pe_cst6'] + history['pt_cst6'] 
            if swpstn < 6:
                history['tot_cst6'] = history['cd_cst6'] +  history['ll_cst6'] + history['hc_cst6'] + history['se_cst6'] + history['pe_cst6'] + history['pt_cst6']
                if swpstn < 5:
                    history['tot_cst5'] = history['cd_cst5'] +  history['ll_cst5'] + history['hc_cst5'] + history['se_cst5'] + history['pe_cst5'] + history['pt_cst5']
                    if swpstn < 4:
                        history['tot_cst4'] = history['cd_cst4'] +  history['ll_cst4'] + history['hc_cst4'] + history['se_cst4']
                        if swpstn < 3:
                            history['tot_cst3'] = history['cd_cst3'] +  history['ll_cst3'] + history['hc_cst3']
                            if swpstn < 2:
                                history['tot_cst2'] = history['ll_cst2'] + history['hc_cst2']

        
        
        # Costs by sector
        # NHS
        if swpstn < 7:
            history['nhs_cst7'] = history['condis7'] * ((costs.loc['cd2', 'nhs'] + costs.loc['cd3', 'nhs'] * 2) / 3)   + history['hosp7'] * costs.loc['hos', 'nhs'] + history['alc7'] * costs.loc['dis', 'nhs']
            if swpstn < 6:
                history['nhs_cst6'] = history['condis6'] * costs.loc['cd2', 'nhs'] + history['hosp6'] * costs.loc['hos', 'nhs'] + history['alc6'] * costs.loc['dis', 'nhs']
                if swpstn < 5:
                    history['nhs_cst5'] = history['condis5'] * ((costs.loc['cd1', 'nhs'] + costs.loc['cd2', 'nhs'] * 2) / 3) + history['hosp5'] * costs.loc['hos', 'nhs'] + history['alc5'] * costs.loc['dis', 'nhs']
                    if swpstn < 4:
                        history['nhs_cst4'] = history['condis4'] * costs.loc['cd1', 'nhs'] + history['hosp4'] * costs.loc['hos', 'nhs'] + history['alc4'] * costs.loc['dis', 'nhs']
                        if swpstn < 3:
                            history['nhs_cst3'] = history['condis3'] * costs.loc['cd1', 'nhs'] + history['hosp3'] * costs.loc['hos', 'nhs'] + history['alc3'] * costs.loc['dis', 'nhs']
                            if swpstn < 2:
                                history['nhs_cst2'] = history['hosp2'] * costs.loc['hos', 'nhs']


        # Social Care
        if swpstn < 7:
            history['ssc_cst7'] = history['condis7'] * ((costs.loc['cd2', 'ss'] + costs.loc['cd3', 'ss'] * 2) / 3)
            if swpstn < 6:
                history['ssc_cst6'] = history['condis6'] * costs.loc['cd2', 'ss']
                if swpstn < 5:
                    history['ssc_cst5'] = history['condis5'] * ((costs.loc['cd1', 'ss'] + costs.loc['cd2', 'ss'] * 2) / 3)
                    if swpstn < 4:
                        history['ssc_cst4'] = history['condis4'] * costs.loc['cd1', 'ss']
                        if swpstn < 3:
                            history['ssc_cst3'] = history['condis3'] * costs.loc['cd1', 'ss']
        

        # Education
        if swpstn < 7:
            history['ded_cst7'] = history['condis7'] * ((costs.loc['cd2', 'ded'] + costs.loc['cd3', 'ded'] * 2) / 3) + history['sen6'] * costs.loc['sen', 'ded'] + history['truancy6'] * costs.loc['tru', 'ded'] * ptp + history['excl6'] * costs.loc['exc', 'ded'] * pep6
            if swpstn < 6:
                history['ded_cst6'] = history['condis6'] * costs.loc['cd2', 'ded'] + history['sen6'] * costs.loc['sen', 'ded'] + history['truancy6'] * costs.loc['tru', 'ded'] * ptp + history['excl6'] * costs.loc['exc', 'ded'] * pep6
                if swpstn < 5:
                    history['ded_cst5'] = history['condis5'] * ((costs.loc['cd1', 'ded'] + costs.loc['cd2', 'ded'] * 2) / 3) + history['sen5'] * costs.loc['sen', 'ded'] + history['truancy5'] * costs.loc['tru', 'ded'] * ptp + history['excl5'] * costs.loc['exc', 'ded'] * pep5
                    if swpstn < 4:
                        history['ded_cst4'] = history['condis4'] * costs.loc['cd1', 'ded'] + history['sen4'] * costs.loc['sen', 'ded']
                        if swpstn < 3:
                            history['ded_cst3'] = history['condis3'] * costs.loc['cd1', 'ded']

        ### Adding up all costs and wellebing across sweeps to calculate totals
        
        # Adding up from sweep 2
        # varl = ['cd_cst', 'pt_cst', 'pe_cst', 'se_cst', 'll_cst', 'hc_cst', 'nhs_cst', 'ssc_cst', 'ded_cst', 'tot_cst', 'lifesat']
        # varl = ['tot_cst', 'lifesat']
        varl = ['ll_cst', 'hc_cst', 'nhs_cst', 'tot_cst', 'lifesat']
        history = aswpwtot(history, varl, 0.035, swpstn, nwvs)
        
        # Adding up from sweep 3
        varl = ['cd_cst', 'ssc_cst', 'ded_cst']
        if swpstn < 3:
            history = aswpwtot(history, varl, 0.035, 3, nwvs)
        else :
            history = aswpwtot(history, varl, 0.035, swpstn, nwvs)
        
        # Adding up from sweep 4
        varl = ['se_cst']
        if swpstn < 4:
            history = aswpwtot(history, varl, 0.035, 4, nwvs)
        else :
            history = aswpwtot(history, varl, 0.035, swpstn, nwvs)    
        
        # Adding up from sweep 5
        varl = ['pt_cst', 'pe_cst']
        if swpstn < 5:
            history = aswpwtot(history, varl, 0.035, 5, nwvs)
        else :
            history = aswpwtot(history, varl, 0.035, swpstn, nwvs)      
        
        #Psychological distress indicator
        history['distress7'] = np.where((history['kessler7'] >= 13), 1, 0)
        
        #Change life satisfaction to SDQ emotion
        for i in range(swpstn+1, nwvs+1):
            history[f'lfstint{i}'] = (history[f'lifesat{i}'])
            history[f'lifesat{i}'] = 2 + ((10 - history[f'sdqemotion{i}']) * 8 / 11)
            
        #Income transferred
        if rskfctr in ("incqnt0", "incqntim0", "incqntnc0", "linctb0"):
            history['prgcst123'] = history['nwmninc123'] - history['aincome123']
    
        ###Descriptive Stats of simulation results
        ##All outcomes
        varl = [var for var in ["bdgcseme7", "distress7", "obesity7", "smkreg7", "prfrhlth7",
                    "lifesat2", "lifesat3", "lifesat4", "lifesat5", "lifesat6", "lifesat7", 
                    "zcog2", "zcog3", "zcog4", "zcog5", "zcog6", "zcog7",
                    "zbasnv2", "zbsrar2", "zbasnv3", "zbaspc3", "zbasps3", "zbaswr4", "zbaspc4", "znferpm4",
                    "sdqinternal2", "sdqinternal3", "sdqinternal4", "sdqinternal5", "sdqinternal6", "sdqinternal7",
                    "sdqemotion2", "sdqemotion3", "sdqemotion4", "sdqemotion5", "sdqemotion6", "sdqemotion7",
                    "sdqconduct2", "sdqconduct3", "sdqconduct4", "sdqconduct5", "sdqconduct6", "sdqconduct7",
                    "sdqexternal2", "sdqexternal3", "sdqexternal4", "sdqexternal5", "sdqexternal6", "sdqexternal7",
                    "hosp2", "hosp3", "hosp4", "hosp5", "hosp6", "hosp7",  
                    "alc2", "alc3", "alc4", "alc5", "alc6", "alc7", 
                    "condis3", "condis4", "condis5", "condis6", "condis7",
                    "sen4", "sen5", "sen6", 
                    "truancy5", "truancy6", 
                    "excl5", "excl6"] 
            if int(var[-1]) > swpstn and int(var[-1]) <= nwvs]
        varl.append('tot_cst_ttl')
        varl.append('lifesat_ttl')
        desc = descsim(history, varl)
        # #code to export descriptive stats to latex (long)
        # desc = desc.rename(index=var_dict)
        # desc.columns = ['mean', 'standard error']
        # logging.info(f'simulated outcome stats - {intervention["name"]}')
        # logging.info(desc)
        # tex_path = os.path.join(lfsm, f'output/{rskfctr}/tables', f'otw{intervention["name"]}.tex')  # define your output file path here
        # desc.to_latex(tex_path,
        #             index=True,
        #             formatters={"name": str.upper},
        #             float_format="{:.3f}".format,
        #         )
        
        #Code to export descriptive stats to latex (wide)
        #Reshape the dataframe to create table of results wide
        desc = desc.reset_index()
        desc.columns = ['variable', 'mean', 'se']
        # desc.rename(columns={'index': 'variable'}, inplace=True)
        # Split variable names to outcome and sweep
        desc['sweep'] = desc['variable'].str.extract('(\d+)$')[0]
        desc['outcome'] = desc['variable'].str.extract('([a-z]+)')[0]
        #desc.loc[desc['outcome'] == 'bdgcseme', 'sweep'] = '7'
        # Reshape the mean and errors to long
        desc = desc[['sweep', 'outcome', 'mean', 'se']]
        # desc = desc.sort_values(by=['outcome', 'sweep']).reset_index(drop=True)
        desc = desc.melt(id_vars = ['sweep', 'outcome'])
        # Reshape everything to wide
        # desc = desc.pivot(index=['sweep', 'variable'], columns='outcome', values='value') #For outcome columns and sweep rows
        desc = desc.pivot(index=['outcome', 'variable'], columns='sweep', values='value') #For sweep columns and outcome rows
        #Order outcomes as needed
        order = ['bdgcseme', 'distress', 'obesity', 'smkreg', 'prfrhlth', 
                 'zcog', 
                 'lifesat', 'internal', 'sdqexternal', 
                 'hosp', 'alc', 'condis', 'sen', 'truancy', 'excl']
        # desc = desc[order] #For outcome columns and sweep rows
        desc = desc.reindex(order, level = 'outcome') #For sweep columns and outcome rows
        desc.reset_index(inplace=True)
        desc.set_index(['outcome', 'variable'], drop=False)
        #Print stats
        logging.info(f'Simulated outcome stats - {intervention["name"]}')
        logging.info(desc)
        
        
        # Format data to output to latex
        #Keep only variables needed
        lst = ['bdgcseme', 'distress', 'obesity', 'smkreg', 'prfrhlth', 
               'lifesat',
               'hosp', 'alc', 'condis', 'sen', 'truancy', 'excl']
        desc = desc[desc['outcome'].isin(lst)]  #For sweep columns and outcome rows
        #Rename columns and sweeps
        # desc.rename(columns=var_dict, inplace=True) #For outcome columns and sweep rows
        desc.rename(columns=swpag_dict, inplace=True) #For sweep columns and outcome rows
        desc.replace({'outcome': var_dict}, inplace=True) #For sweep columns and outcome rows
        # Functions to format data for latex
        #Mean
        def mn_fmt(x):
            return '' if pd.isna(x) else f"{x:.2f}"
        # Errors
        def se_fmt(x):
            return '' if pd.isna(x) else f"({x:.3f})"
        # Set 'Age (years)' to an empty string for 'se'
        # desc.loc[desc['variable'] == 'se', 'sweep'] = '' #For outcome columns and sweep rows
        desc.loc[desc['variable'] == 'se', 'outcome'] = '' #For sweep columns and outcome rows
        # Apply formatting to DataFrame excluding 'variable' column
        for col in desc.columns:
            # if col not in ['sweep', 'variable']: #For outcome columns and sweep rows
            if col not in ['outcome', 'variable']: #For sweep columns and outcome rows
                desc[col] = desc.apply(lambda row: se_fmt(row[col]) if row['variable'] == 'se' else mn_fmt(row[col]), axis=1)
        # Exclude 'variable' column and ename columns and sweeps
        desc.drop('variable', axis=1, inplace=True)
        # desc.replace({'sweep': swpag_dict}, inplace=True) #For outcome columns and sweep rows
        # desc.rename(columns={'sweep': 'Age (years)'}, inplace=True) #For outcome columns and sweep rows
        desc.rename(columns={'outcome': 'Outcome'}, inplace=True) #For sweep columns and outcome rows    
        # Dynamically generate column format string (left-align the first column, center-align others)
        col_fmt = 'l' + 'c' * (len(desc.columns) - 1)
        tex_path = os.path.join(lfsm, f'output/{rskfctr}/tables', f'otw_{rskfctr}_{intervention["name"]}.tex')  # Define your output file path here
        desc.to_latex(tex_path,
                    index=False,
                    formatters={"name": str.upper},
                    na_rep='',
                    # float_format="{:.3f}".format,
                    column_format=col_fmt
                )
    
        del tex_path
        del desc
        del col_fmt
    
        #Distribution of means by universe
        umeans = history[varl].groupby(history['simulation']).mean()
        stats = ["nobs", "mean", "std", "percentiles"]
        stats = ["mean", "std_err"]
        #mean_values = history[varl].mean()
        #logging.info(mean_values)
        udesc = sms.stats.descriptivestats.describe(umeans, stats=stats, numeric=True).T
        udesc = udesc.rename(index=var_dict)
        logging.info(f'Simulated outcome stats (across universes) - {intervention["name"]}')
        logging.info(udesc)
        
        del umeans
        del udesc
        del stats
    
        ##All Costs
        #Costs by source
        varl = [var for var in ['hc_cst2', 'hc_cst3', 'hc_cst4', 'hc_cst5', 'hc_cst6', 'hc_cst7',
                'll_cst2', 'll_cst3', 'll_cst4', 'll_cst5', 'll_cst6', 'll_cst7',
                'cd_cst3', 'cd_cst4', 'cd_cst5', 'cd_cst6', 'cd_cst7',
                'se_cst4', 'se_cst5', 'se_cst6',
                'pe_cst5', 'pe_cst6',
                'pt_cst5', 'pt_cst6'] 
            if int(var[-1]) > swpstn and int(var[-1]) <= nwvs]
        # costdesc = descsim(history, varl)
        # costdesc = costdesc.rename(index=var_dict)
        # costdesc.columns = ['Mean', 'Standard Error']
        # logging.info(f'Simulated costs by source - {intervention["name"]}')
        # logging.info(costdesc)
        # tex_path = os.path.join(lfsm, 'output', f'cstsw{intervention["name"]}.tex')  # Define your output file path here
        # costdesc.to_latex(tex_path,
        #             index=True,
        #             formatters={"name": str.upper},
        #             float_format="{:.3f}".format,
        #         )
        
        #Code to export descriptive stats to latex (wide)
        #Reshape the dataframe to create table of results wide
        desc = descsim(history, varl)
        desc.index = desc.index.str.replace('_cst', '')
        #Change values in 7th sweep with values in 6th sweep for education variables
        desc.loc['pe7'] = desc.loc['pe6']
        desc.loc['pt7'] = desc.loc['pt6']
        desc.loc['se7'] = desc.loc['se6']
        desc = desc.reset_index()
        desc.columns = ['variable', 'mean', 'se']
        # desc.rename(columns={'index': 'variable'}, inplace=True)
        # Split variable names to outcome and sweep
        desc['sweep'] = desc['variable'].str.extract('(\d+)$')[0]
        desc['outcome'] = desc['variable'].str.extract('([a-z]+)')[0]
        # Reshape the mean and errors to long
        desc = desc[['sweep', 'outcome', 'mean', 'se']]
        # desc = desc.sort_values(by=['outcome', 'sweep']).reset_index(drop=True)
        desc = desc.melt(id_vars = ['sweep', 'outcome'])
        # Reshape everything to wide
        # desc = desc.pivot(index=['sweep', 'variable'], columns='outcome', values='value') #For outcome columns and sweep rows
        desc = desc.pivot(index=['outcome', 'variable'], columns='sweep', values='value') #For sweep columns and outcome rows
        #Order outcomes as needed
        order = ['hc', 'll', 'cd', 'se', 'pe', 'pt']
        # desc = desc[order] #For outcome columns and sweep rows
        desc = desc.reindex(order, level = 'outcome') #For sweep columns and outcome rows
        desc.reset_index(inplace=True)
        desc.set_index(['outcome', 'variable'], drop=False)
        #Rename columns and sweeps
        # desc.rename(columns=var_dict, inplace=True) #For outcome columns and sweep rows
        desc.rename(columns=swpag_dict, inplace=True) #For sweep columns and outcome rows
        desc.replace({'outcome': var_dict}, inplace=True) #For sweep columns and outcome rows
        #Print stats
        logging.info(f'Simulated costs by source - {intervention["name"]}')
        logging.info(desc)    
        
        # Format data to output to latex
        # Functions to format data for latex
        #Mean
        def mn_fmt(x):
            return '' if pd.isna(x) else f"{x:.2f}"
        # Errors
        def se_fmt(x):
            return '' if pd.isna(x) else f"({x:.2f})"
        # Set 'Age (years)' to an empty string for 'se'
        # desc.loc[desc['variable'] == 'se', 'sweep'] = '' #For outcome columns and sweep rows
        desc.loc[desc['variable'] == 'se', 'outcome'] = '' #For sweep columns and outcome rows
        # Apply formatting to DataFrame excluding 'variable' column
        for col in desc.columns:
            # if col not in ['sweep', 'variable']: #For outcome columns and sweep rows
            if col not in ['outcome', 'variable']: #For sweep columns and outcome rows
                desc[col] = desc.apply(lambda row: se_fmt(row[col]) if row['variable'] == 'se' else mn_fmt(row[col]), axis=1)
        # Exclude 'variable' column and ename columns and sweeps
        desc.drop('variable', axis=1, inplace=True)
        # desc.replace({'sweep': swpag_dict}, inplace=True) #For outcome columns and sweep rows
        # desc.rename(columns={'sweep': 'Age (years)'}, inplace=True) #For outcome columns and sweep rows
        desc.rename(columns={'outcome': 'Outcome'}, inplace=True) #For sweep columns and outcome rows 
        # Dynamically generate column format string (left-align the first column, center-align others)
        col_fmt = 'l' + 'c' * (len(desc.columns) - 1)
        tex_path = os.path.join(lfsm, f'output/{rskfctr}/tables', f'cstsw_{rskfctr}_{intervention["name"]}.tex')  # Define your output file path here
        desc.to_latex(tex_path,
                    index=False,
                    formatters={"name": str.upper},
                    na_rep='',
                    # float_format="{:.3f}".format,
                    column_format=col_fmt
                )
    
        del tex_path
        del desc
        del col_fmt
        
        
        
    
        #Costs by sector
        varl = [var for var in ['nhs_cst2', 'nhs_cst3', 'nhs_cst4', 'nhs_cst5', 'nhs_cst6', 'nhs_cst7', 
                'ssc_cst3', 'ssc_cst4', 'ssc_cst5', 'ssc_cst6', 'ssc_cst7',
                'ded_cst3', 'ded_cst4', 'ded_cst5', 'ded_cst6', 'ded_cst7'] 
            if int(var[-1]) > swpstn and int(var[-1]) <= nwvs]
        costdesc = descsim(history, varl)
        costdesc = costdesc.rename(index=var_dict)
        costdesc.columns = ['Mean', 'Standard Error']
        logging.info(f'Simulated costs by sector - {intervention["name"]}')
        logging.info(costdesc)
        tex_path = os.path.join(lfsm, f'output/{rskfctr}/tables', f'cstow_{rskfctr}_{intervention["name"]}.tex')  # Define your output file path here
        costdesc.to_latex(tex_path,
                    index=True,
                    formatters={"name": str.upper},
                    float_format="{:.3f}".format,
                )
    
        del tex_path
        del costdesc
        del varl
    
        #Distribution of means by universe
        varl = [var for var in ['hc_cst2', 'hc_cst3', 'hc_cst4', 'hc_cst5', 'hc_cst6', 'hc_cst7',
                'll_cst2', 'll_cst3', 'll_cst4', 'll_cst5', 'll_cst6', 'll_cst7',
                'cd_cst3', 'cd_cst4', 'cd_cst5', 'cd_cst6', 'cd_cst7',
                'se_cst4', 'se_cst5', 'se_cst6',
                'pe_cst5', 'pe_cst6',
                'pt_cst5', 'pt_cst6',
                'tot_cst2', 'tot_cst3', 'tot_cst4', 'tot_cst5', 'tot_cst6', 'tot_cst7'] 
            if int(var[-1]) > swpstn and int(var[-1]) <= nwvs]
        varl.append('tot_cst_ttl')
        ucostm = history[varl].groupby(history['simulation']).mean()
        stats = ["nobs", "mean", "std", "percentiles"]
        stats = ["mean", "std"]
        #mean_values = history[varl].mean()
        #logging.info(mean_values)
        ucostd = sms.stats.descriptivestats.describe(ucostm, stats=stats, numeric=True, percentiles=(5, 25, 50, 75, 95) ).T
        ucostd = ucostd.rename(index=var_dict)
        logging.info(f'Simulated costs (across universes) - {intervention["name"]}')
        logging.info(ucostd)
        
        del ucostm
        del ucostd
        del stats
        
        
        
        
        ########### Save simulation output to file (remove to speed up)
        if save > 0:
        
            # rows = [row for row in zip(*history)]
            # t0 = time.time()
            # with open(hst_pth, 'a') as csvfile:
            #     csvfile.writelines(rows)
            # tdelta = time.time() - t0
            # print(tdelta)
            
            # t0 = time.time()
            # csv.write_csv(history, 'hst_pth')
            # tdelta = time.time() - t0
            # print(tdelta)
            
            history.to_csv(wcp_pth)
            
            if save > 1:
                t0 = time.time()
                history.to_csv(hst_pth)
                tdelta = time.time() - t0
                logging.info(tdelta)
            else:
                logging.info("Only working file saved")
                
                
        else:       
            logging.info("No files saved")
              
        del hst_pth
        del wcp_pth
        ## Rename and keep simulation output to compare with other interventions
        globals()[f'hist{intervention["id"]}'] = history.copy()
        
        
        
        
        
        if intervention["id"] == "int0":
            
            ##### Plot baseline outputs
            ### Bad Age 17 outcomes
            varl = ["bdgcseme7", "distress7", "obesity7", "smkreg7", "prfrhlth7"]
            desc = descsim(history, varl, ppt = True)
            logging.info(f'{intervention["name"]} levels - adverse outcomes')
            logging.info(desc)
            # Plotting means with error bars
            plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'b17o_{rskfctr}_{intervention["name"]}.png')
            lbl = ["Poor GCSEs", "Psych. distress", "Obesity", "Regular smoker", "Poor health"]
            xt = 'Adverse outcome'
            yt = 'Percent of cohort with adverse outcome'
            pltsim(desc, lbl, plt_path, ya=yt)
            
            ## Bad Age 17 outcomes by income quintile
            varl = ["bdgcseme7", "distress7", "obesity7", "smkreg7", "prfrhlth7"]
            desc = history.groupby(['incqnt123'])[varl].mean()
            desc *= 100
            desc = desc.T
            desc.rename(columns={1.0: 'q1', 2.0: 'q2', 3.0: 'q3', 4.0: 'q4', 5.0: 'q5'}, inplace=True)
            logging.info(f'{intervention["name"]} levels of adverse outcomes by income quintile')
            logging.info(desc)
            # logging.info(desc[['int1', 'int2', 'int3', 'int4']])
            # Plotting bar plots for comparison
            plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'b17oiq_{rskfctr}_{intervention["name"]}.png')
            lbl = ["Poor GCSEs", "Psych. distress", "Obesity", "Regular smoker", "Poor health"]
            yt = 'Percent with adverse outcome'
            xt = 'Adverse outcome'
            r = np.arange(len(lbl))
            w = 0.15
            plt.figure(figsize=(15, 9))
            plt.rcParams.update({'font.size': 18})  # Set the font size to 14
            # colors = ['#DDCC77', '#117733', '#882255', '#332288']  # Define the colors for each intervention
            colors = ['#dadae4', '#a8a8af', '#585869', '#28282b' , '#111111']  # Define the colors for each intervention
            plt.grid(axis='y', which='both', linestyle="--", alpha=0.5, zorder=1)
            for i in range(5):
                plt.bar(r + w * i, desc[f'q{i+1}'], 
                        # yerr=desc[f'int{i+1}se'], 
                        capsize=5, color=colors[i],
                        width=w, edgecolor='black', zorder=2,
                        label=f'Income quintile {i+1}')
            plt.xticks(r + w * (5-1) / 2, lbl)
            plt.legend(loc='upper right')
            # plt.xlabel(xt)
            plt.ylabel(yt)
            # os.remove(plt_path)
            plt.savefig(plt_path, bbox_inches='tight')
            plt.show()
            
    
            ### Life Satisfaction
            varl = [f"lifesat{i}" for i in range(swpstn + 1, nwvs + 1)]
            desc = descsim(history, varl)
            logging.info(f'{intervention["name"]} levels - life satisfaction')
            logging.info(desc)
            # Plotting means with error bars
            plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'lifesat_{rskfctr}_{intervention["name"]}.png')
            # lbl = ["2", "3", "4", "5", "6", "7"]
            lbl = ["Age 3", "Age 5", "Age 7", "Age 11", "Age 14", "Age 17"][(swpstn + 1 - 2):]
            xt = 'Age'
            yt = 'Mean life satisfaction'
            pltsim(desc, lbl, plt_path, ya=yt, tt=10)
            
            ## Life Satisfaction by income quintile
            varl = [f"lifesat{i}" for i in range(swpstn + 1, nwvs + 1)]
            desc = history.groupby(['incqnt123'])[varl].mean()
            desc = desc.T
            desc.rename(columns={1.0: 'q1', 2.0: 'q2', 3.0: 'q3', 4.0: 'q4', 5.0: 'q5'}, inplace=True)
            logging.info(f'{intervention["name"]} levels of life satisfaction by income quintile')
            logging.info(desc)
            # logging.info(desc[['int1', 'int2', 'int3', 'int4']])
            # Plotting bar plots for comparison
            plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'lifesatiq_{rskfctr}_{intervention["name"]}.png')
            lbl = ["Age 3", "Age 5", "Age 7", "Age 11", "Age 14", "Age 17"][(swpstn + 1 - 2):]
            yt = 'Mean life satisfaction'
            xt = 'Age'
            r = np.arange(len(lbl))
            w = 0.15
            plt.figure(figsize=(15, 9))
            plt.rcParams.update({'font.size': 18})  # Set the font size to 14
            # colors = ['#DDCC77', '#117733', '#882255', '#332288']  # Define the colors for each intervention
            colors = ['#dadae4', '#a8a8af', '#585869', '#28282b' , '#111111']  # Define the colors for each intervention
            plt.grid(axis='y', which='both', linestyle="--", alpha=0.5, zorder=1)
            for i in range(5):
                plt.bar(r + w * i, desc[f'q{i+1}'], 
                        # yerr=desc[f'int{i+1}se'], 
                        capsize=5, color=colors[i],
                        width=w, edgecolor='black', zorder=2,
                        label=f'Income quintile {i+1}')
            plt.xticks(r + w * (5-1) / 2, lbl)
            plt.legend(loc='lower left')
            # plt.xlabel(xt)
            plt.ylim(top=10)
            plt.ylabel(yt)
            # os.remove(plt_path)
            plt.savefig(plt_path, bbox_inches='tight')
            plt.show()
            
            del colors
    
    
            ### Cognitive Ability
            varl = [f"zcog{i}" for i in range(swpstn + 1, nwvs + 1)]
            desc = descsim(history, varl)
            logging.info(f'{intervention["name"]} levels - Cognitive ability')
            logging.info(desc)
            # Plotting means with error bars
            plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'zcog_{rskfctr}_{intervention["name"]}.png')
            # lbl = ["2", "3", "4", "5", "6", "7"]
            lbl = ["Age 3", "Age 5", "Age 7", "Age 11", "Age 14", "Age 17"][(swpstn + 1 - 2):]
            xt = 'Age'
            yt = 'Mean cognitive ability'
            pltsim(desc, lbl, plt_path, ya=yt)        
    
    
            ### SDQ Internalising
            varl = [f"sdqinternal{i}" for i in range(swpstn + 1, nwvs + 1)]
            desc = descsim(history, varl)
            logging.info(f'{intervention["name"]} levels - SDQ internalising')
            logging.info(desc)
            # Plotting means with error bars
            plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'intrnl_{rskfctr}_{intervention["name"]}.png')
            # lbl = ["2", "3", "4", "5", "6", "7"]
            lbl = ["Age 3", "Age 5", "Age 7", "Age 11", "Age 14", "Age 17"][(swpstn + 1 - 2):]
            xt = 'Age'
            yt = 'Mean internalising'
            pltsim(desc, lbl, plt_path, ya=yt)
    
            
            ### SDQ Externalising
            varl = [f"sdqexternal{i}" for i in range(swpstn + 1, nwvs + 1)]
            desc = descsim(history, varl)
            logging.info(f'{intervention["name"]} levels - SDQ externalising')
            logging.info(desc)
            # Plotting means with error bars
            plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'sdqext_{rskfctr}_{intervention["name"]}.png')
            # lbl = ["2", "3", "4", "5", "6", "7"]
            lbl = ["Age 3", "Age 5", "Age 7", "Age 11", "Age 14", "Age 17"][(swpstn + 1 - 2):]
            xt = 'Age'
            yt = 'Mean SDQ externalising'
            pltsim(desc, lbl, plt_path, ya=yt)
            
                    
            ### Hospitilisation
            varl = [f"hosp{i}" for i in range(swpstn + 1, nwvs + 1)]
            desc = descsim(history, varl, ppt = True)
            logging.info(f'{intervention["name"]} levels - hospitilisation')
            logging.info(desc)
            # Plotting means with error bars
            plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'hosp_{rskfctr}_{intervention["name"]}.png')
            # lbl = ["2", "3", "4", "5", "6", "7"]
            # lbl = ["3", "5", "7", "11", "14", "17"]
            lbl = ["Age 3", "Age 5", "Age 7", "Age 11", "Age 14", "Age 17"][(swpstn + 1 - 2):]
            xt = 'Age'
            yt = 'Percentage of cohort with any hospitilisation'
            pltsim(desc, lbl, plt_path, ya=yt)
    
    
            ### Disability
            varl = [f"alc{i}" for i in range(swpstn + 1, nwvs + 1)]
            desc = descsim(history, varl, ppt = True)
            logging.info(f'{intervention["name"]} levels - disability')
            logging.info(desc)
            # Plotting means with error bars
            plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'dis_{rskfctr}_{intervention["name"]}.png')
            # lbl = ["2", "3", "4", "5", "6", "7"]
            # lbl = ["3", "5", "7", "11", "14", "17"]
            lbl = ["Age 3", "Age 5", "Age 7", "Age 11", "Age 14", "Age 17"][(swpstn + 1 - 2):]
            xt = 'Age'
            yt = 'Percentage of cohort with disability'
            pltsim(desc, lbl, plt_path, ya=yt)
    
    
            ### Conduct disorder
            varl = [f"condis{i}" for i in range(swpstn + 1, nwvs + 1)]
            desc = descsim(history, varl, ppt = True)
            logging.info(f'{intervention["name"]} levels - conduct disorder')
            logging.info(desc)
            # Plotting means with error bars
            plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'condis_{rskfctr}_{intervention["name"]}.png')
            # lbl = ["3", "4", "5", "6", "7"]
            # lbl = ["3", "5", "7", "11", "14", "17"]
            lbl = ["Age 5", "Age 7", "Age 11", "Age 14", "Age 17"][(swpstn + 1 - 3):]
            xt = 'Age'
            yt = 'Percentage of cohort with conduct disorder'
            pltsim(desc, lbl, plt_path, ya=yt)
    
    
            ### Education
            ### SEN
            varl = ["sen4", "sen5", "sen6"]
            desc = descsim(history, varl, ppt = True)
            logging.info(f'{intervention["name"]} levels - special education needs')
            logging.info(desc)
            # Plotting means with error bars
            plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'sen_{rskfctr}_{intervention["name"]}.png')
            # lbl = ["4", "5", "6"]
            # lbl = ["7", "11", "14"]
            lbl = ["Age 7", "Age 11", "Age 14"]
            xt = 'Age'
            yt = 'Percentage of cohort with Special Education Needs'
            pltsim(desc, lbl, plt_path, ya=yt)
    
    
            ###Any Truancy
            varl = ["truancy5", "truancy6"]
            desc = descsim(history, varl, ppt = True)
            logging.info(f'{intervention["name"]} levels - truancy')
            logging.info(desc)
            # Plotting means with error bars
            plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'truanc_{rskfctr}_{intervention["name"]}.png')
            # lbl = ["5", "6"]
            # lbl = ["11", "14"]
            lbl = ["Age 11", "Age 14"]
            xt = 'Age'
            yt = 'Percentage of cohort with any truancy'
            pltsim(desc, lbl, plt_path, ya=yt)
    
    
            ###Any Exclusion
            varl = ["excl5", "excl6"]
            desc = descsim(history, varl, ppt = True)
            logging.info(f'{intervention["name"]} levels - exclusion')
            logging.info(desc)
            # Plotting means with error bars
            plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'exclus_{rskfctr}_{intervention["name"]}.png')
            # lbl = ["5", "6"]
            # lbl = ["11", "14"]
            lbl = ["Age 11", "Age 14"]
            xt = 'Age'
            yt = 'Percentage of cohort with any exclusion'
            pltsim(desc, lbl, plt_path, ya=yt)
            
            
            ### Costs by source
            varl = [var for var in ['hc_cst2', 'hc_cst3', 'hc_cst4', 'hc_cst5', 'hc_cst6', 'hc_cst7',
                    'll_cst2', 'll_cst3', 'll_cst4', 'll_cst5', 'll_cst6', 'll_cst7',
                    'cd_cst3', 'cd_cst4', 'cd_cst5', 'cd_cst6', 'cd_cst7',
                    'se_cst4', 'se_cst5', 'se_cst6',
                    'pe_cst5', 'pe_cst6',
                    'pt_cst5', 'pt_cst6'] 
            if int(var[-1]) > swpstn and int(var[-1]) <= nwvs]
            desc = descsim(history, varl)
            logging.info(f'{intervention["name"]} levels - costs')
            logging.info(desc)
            
            ##Reshape dataframe to panel
            d = pd.DataFrame(desc['mean'].copy()).T
            d.columns = d.columns.str.replace('_cst', '')
            #Change values in 7th sweep with values in 6th sweep for education variables
            d['pe7'] = d['pe6']
            d['pt7'] = d['pt6']
            d['se7'] = d['se6']
            
            # Reshape the means to a data frame
            d = d.melt(var_name='variable', value_name='value')
    
            # Extract the measurement type and sweep number
            d['Sweep'] = d['variable'].str.extract('(\d+)$')[0]
            d['Type'] = d['variable'].str.extract('([a-z]+)')[0]
            
            # Rearrange and sort the DataFrame
            d = d[['Sweep', 'Type', 'value']]
            d = d.sort_values(by=['Sweep', 'Type']).reset_index(drop=True)
            
            # Pivot the table to get desired format
            d = d.pivot(index='Sweep', columns='Type', values='value')
            d.reset_index(inplace=True)
            logging.info(f'{intervention["name"]} levels - costs')
            logging.info(d)
            
            logging.info(f'{intervention["name"]} levels - costs with total')
            cstsm(d, 0.035)
            logging.info(f'{intervention["name"]} levels - costs with undiscounted total')
            cstsm(d, 0)
            
            ## Plotting means
            plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'cstsrc_{rskfctr}_{intervention["name"]}.png')  # Define your output file path here
            # lbl = ["2", "3", "4", "5", "6", "7"]
            # lbl = ["3", "5", "7", "11", "14", "17"]
            lbl = ["Age 3", "Age 5", "Age 7", "Age 11", "Age 14", "Age 17"][(swpstn + 1 - 2):]
            plt.figure(figsize=(15, 9))
            plt.rcParams.update({'font.size': 14})  # Set the font size to 14
            # d.plot(x='Sweep', kind='bar', stacked=True)
            plt.grid(axis='y', which='both', linestyle="--", alpha=0.5, zorder=1)
            plt.bar(lbl, d['hc'], color='#44AA99', zorder=2)
            plt.bar(lbl, d['ll'], bottom=d['hc'], color='#882255', zorder=2)
            plt.bar(lbl, d['cd'], bottom=d['hc']+d['ll'], color='#88CCEE', zorder=2)
            plt.bar(lbl, d['se'], bottom=d['hc']+d['cd']+d['ll'], color='#117733', zorder=2)
            plt.bar(lbl, d['pt'], bottom=d['hc']+d['cd']+d['ll']+d['se'], color='#CC6677', zorder=2)
            plt.bar(lbl, d['pe'], bottom=d['hc']+d['cd']+d['ll']+d['se']+d['pt'], color='#332288', zorder=2)
            # plt.xlabel('MCS Sweep')
            plt.ylabel('Annual cost per child in £s')
            # plt.title('Simulated baseline cost at each age')
            plt.legend(["Hospitalisation", "Disability", "Conduct disorder", "SEN", "Truancy", "Exclusion"], loc='upper left', reverse=True)     
            # os.remove(plt_path)
            plt.savefig(plt_path, bbox_inches='tight')
            plt.show()
            
            
            del d
            del varl
            del lbl
            del plt_path
            
            ##### Costs by source
            varl = [var for var in ['hc_cst2', 'hc_cst3', 'hc_cst4', 'hc_cst5', 'hc_cst6', 'hc_cst7',
                    'll_cst2', 'll_cst3', 'll_cst4', 'll_cst5', 'll_cst6', 'll_cst7',
                    'cd_cst3', 'cd_cst4', 'cd_cst5', 'cd_cst6', 'cd_cst7',
                    'se_cst4', 'se_cst5', 'se_cst6',
                    'pe_cst5', 'pe_cst6',
                    'pt_cst5', 'pt_cst6'] 
            if int(var[-1]) > swpstn and int(var[-1]) <= nwvs]
            desc = history.groupby(['incqnt123'])[varl].mean()
            desc = desc.T
            desc.rename(columns={1.0: 'q1', 2.0: 'q2', 3.0: 'q3', 4.0: 'q4', 5.0: 'q5'}, inplace=True)
            logging.info(f'{intervention["name"]} costs by income quintile group')
            logging.info(desc)
    
            for i in range(1, 6):
                desc_i = pd.DataFrame(desc[f'q{i}'].copy())
                
                d_i = desc_i.T
                d_i.columns = d_i.columns.str.replace('_cst', '')
                #Change values in 7th sweep with values in 6th sweep for education variables
                d_i['pe7'] = d_i['pe6']
                d_i['pt7'] = d_i['pt6']
                d_i['se7'] = d_i['se6']
    
                # Reshape the means to a data frame
                d_i = d_i.melt(var_name='variable', value_name='value')
    
                # Extract the measurement type and sweep number
                d_i['Sweep'] = d_i['variable'].str.extract('(\d+)$')[0]
                d_i['Type'] = d_i['variable'].str.extract('([a-z]+)')[0]
    
                # Rearrange and sort the DataFrame
                d_i = d_i[['Sweep', 'Type', 'value']]
                d_i = d_i.sort_values(by=['Sweep', 'Type']).reset_index(drop=True)
    
                # Pivot the table to get desired format
                d_i = d_i.pivot(index='Sweep', columns='Type', values='value')
                d_i.reset_index(inplace=True)
                logging.info(f'{intervention["name"]} costs for quintile {i}')
                logging.info(d_i)
                d_i.fillna(0, inplace=True)
                d_i.fillna(0, inplace=True)
                globals()[f"d{i}"] = d_i.copy()
                
            del desc_i    

    
            for i in range(1, 6):
                d_i = globals()[f"d{i}"]
                logging.info(f'{intervention["name"]} costs for quintile {i} with sum')
                cstsm(d_i, 0.035)
                logging.info(f'{intervention["name"]} costs for quintile {i} with undiscounted sum')
                cstsm(d_i, 0)
                
            del d_i    
                
            # Plotting means with error bars
            # lbl = ["2", "3", "4", "5", "6", "7"]
            # lbl = ["3", "5", "7", "11", "14", "17"]
            lbl = ["Age 3", "Age 5", "Age 7", "Age 11", "Age 14", "Age 17"][(swpstn + 1 - 2):]
            # lbl1 = ['S1', 'S2', 'S3', 'S4']
            r = np.arange(len(lbl))
            w = 0.15
            plt.figure(figsize=(15, 9))
            plt.rcParams.update({'font.size': 14})  # Set the font size to 14
            # d.plot(x='Sweep', kind='bar', stacked=True)
            plt.grid(axis='y', which='both', linestyle="--", alpha=0.5, zorder=1)
            for i in range(5):
                dp = globals()[f'd{i+1}']
                plt.bar(r + w * i, dp['hc'], capsize=5, width = w, edgecolor = 'black', color='#44AA99', zorder=2)
                plt.bar(r + w * i, dp['ll'], capsize=5, width = w, edgecolor = 'black', bottom=dp['hc'], color='#882255', zorder=2)
                plt.bar(r + w * i, dp['cd'], capsize=5, width = w, edgecolor = 'black', bottom=dp['hc']+dp['ll'], color='#88CCEE', zorder=2)
                plt.bar(r + w * i, dp['se'], capsize=5, width = w, edgecolor = 'black', bottom=dp['hc']+dp['ll']+dp['cd'], color='#117733', zorder=2)
                plt.bar(r + w * i, dp['pt'], capsize=5, width = w, edgecolor = 'black', bottom=dp['hc']+dp['ll']+dp['cd']+dp['se'], color='#CC6677', zorder=2)
                plt.bar(r + w * i, dp['pe'], capsize=5, width = w, edgecolor = 'black', bottom=dp['hc']+dp['ll']+dp['cd']+dp['se']+dp['pt'], color='#332288', zorder=2)
            # plt.xlabel('MCS Sweep')
            plt.ylabel('Annual cost per child in £s')
            # plt.xticks(r + w * (np.arange(nint)-1) / 2, lbl1 * r)
            plt.xticks(r + w*(5-1)/2, lbl)
            # plt.title('Simulated baseline cost at each age')
            plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
            plt.legend(["Hospitalisation", "Disability", "Conduct disorder", "SEN", "Truancy", "Exclusion"], loc='upper left')
            plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'cstsrciq_{rskfctr}_{intervention["name"]}.png')  # Define your output file path here
            # os.remove(plt_path)
            plt.savefig(plt_path, bbox_inches='tight')
            plt.show()
            
            del d1
            del d2
            del d3
            del d4
            del d5
            del dp
            del varl
            del lbl
            del plt_path
            del i
            del w 
            del xt
            del yt
            del r
    
            
        else:
            logging.info("No plots for this intervention")
    
        del mcst
    
    setm = (time.time() - sstm)/60
    logging.info('Simulation end time - ' + time.strftime("%H:%M:%S"))
    logging.info(f"Total simulation time - {setm:.2f} minutes")
    
    del setm
    del sstm
    del intervention
    del history
    
    ####################################################################################################        
    #################### Post simulation intervention comparison
    ####################################################################################################
    # for intervention in interventions:
    #     if intervention["id"] == "int0":
    # #####Plot baseline outputs
    # # Bad Age 17 outcomes
    # varl = ["smkreg7", "bdgcseme7", "obesity7", "prfrhlth7", "distress7"]
    # lbl = ["Regular smoker", "Bad GCSEs", "Obesity", "Poor Health", "Psychological distress"]
    # dtmp = DescrStatsW(histint0[varl], weights=histint0['wt_uk2'])
    # logging.info(varl)
    # logging.info(dtmp.mean)
    # logging.info(dtmp.std)
    # logging.info(dtmp.nobs)
    
    # # Plotting means with error bars
    # plt.figure(figsize=(20, 12))
    # plt.bar(lbl, dtmp.mean, yerr=dtmp.std_mean, capsize=5)
    # plt.xlabel('Outcome')
    # plt.ylabel('Proportion')
    # plt.title('Simulated baseline proportions of Bad age 17 outcomes')
    # #plt.xticks(rotation=90)
    # plt_path = os.path.join(lfsm, 'output', 'b17oint0.png')
    # os.remove(plt_path)
    # plt.savefig(plt_path, bbox_inches='tight')
    # plt.show()
    
    
    
    

    ##### Difference in between intervention and baseline
    #Number of interventions 
    nint = len(interventions)-1
    #Code to remove error highlights
    
    #Checking correct intervention implementation
    if rskfctr in ("incqnt0", "incqntim0", "incqntnc0"):
        logging.info('Baseline')
        logging.info(histint0['incqnt123_1'].value_counts()) 
        logging.info(histint0['incqnt123_2'].value_counts()) 
        logging.info(histint0['incqnt123_3'].value_counts()) 
        logging.info(histint0['incqnt123_4'].value_counts()) 
        logging.info('Intervention 1')
        logging.info(histint1['incqnt123_1'].value_counts()) 
        logging.info(histint1['incqnt123_2'].value_counts()) 
        logging.info(histint1['incqnt123_3'].value_counts()) 
        logging.info(histint1['incqnt123_4'].value_counts()) 
        logging.info('Intervention 2')
        logging.info(histint2['incqnt123_1'].value_counts()) 
        logging.info(histint2['incqnt123_2'].value_counts()) 
        logging.info(histint2['incqnt123_3'].value_counts()) 
        logging.info(histint2['incqnt123_4'].value_counts())
        logging.info('Intervention 3') 
        logging.info(histint3['incqnt123_1'].value_counts()) 
        logging.info(histint3['incqnt123_2'].value_counts()) 
        logging.info(histint3['incqnt123_3'].value_counts()) 
        logging.info(histint3['incqnt123_4'].value_counts()) 
        logging.info('Intervention 4')
        logging.info(histint4['incqnt123_1'].value_counts()) 
        logging.info(histint4['incqnt123_2'].value_counts()) 
        logging.info(histint4['incqnt123_3'].value_counts())
        logging.info(histint4['incqnt123_4'].value_counts())
    elif rskfctr in ("linc0", "lincim0", "lincnc0"):    
        logging.info(' ')
    elif rskfctr in ("inc0", "incim0", "incnc0"):
        logging.info(' ')
    elif rskfctr in ("teenbrth0", "teenbrthnm", "teenbrthnc0"):
        logging.info('Baseline')
        logging.info(histint0['teenbrth'].value_counts()) 
        logging.info('Intervention 1')
        logging.info(histint1['teenbrth'].value_counts()) 
    #### Pre-term birth
    elif rskfctr in ("pretrmbrth0", "pretrmbrthim0", "pretrmbrthnc0"):
        logging.info('Baseline')
        logging.info(histint0['pretrm_3'].value_counts()) 
        logging.info(histint0['pretrm_2'].value_counts()) 
        logging.info(histint0['pretrm_1'].value_counts()) 
        logging.info('Intervention 1')
        logging.info(histint1['pretrm_3'].value_counts()) 
        logging.info(histint1['pretrm_2'].value_counts()) 
        logging.info(histint1['pretrm_1'].value_counts()) 
        logging.info('Intervention 2')
        logging.info(histint2['pretrm_3'].value_counts()) 
        logging.info(histint2['pretrm_2'].value_counts()) 
        logging.info(histint2['pretrm_1'].value_counts()) 
        logging.info('Intervention 3')
        logging.info(histint3['pretrm_3'].value_counts()) 
        logging.info(histint3['pretrm_2'].value_counts()) 
        logging.info(histint3['pretrm_1'].value_counts()) 
    #### Low birth weight
    elif rskfctr in ("lwbrthwt0", "lwbrthwtim0", "lwbrthwtnc0"):
        logging.info('Baseline')
        logging.info(histint0['bthwt_2'].value_counts()) 
        logging.info(histint0['bthwt_1'].value_counts()) 
        logging.info('Intervention 1')
        logging.info(histint1['bthwt_2'].value_counts()) 
        logging.info(histint1['bthwt_1'].value_counts()) 
        logging.info('Intervention 2')
        logging.info(histint2['bthwt_2'].value_counts()) 
        logging.info(histint2['bthwt_1'].value_counts()) 
        #### Low birth weight
    elif rskfctr in ("linctb0", "linctbim0", "linctbnc0"):
        logging.info('Baseline')
        logging.info(histint0['recip'].value_counts()) 
        logging.info('Intervention 1')
        logging.info(histint1['recip'].value_counts()) 
        logging.info('Intervention 2')
        logging.info(histint2['recip'].value_counts()) 
        logging.info('Intervention 3') 
        logging.info(histint3['recip'].value_counts()) 


    ###Descriptive Stats of simulation results
    ##All outcomes
    varl = [var for var in ["bdgcseme7", "distress7", "obesity7", "smkreg7", "prfrhlth7",
            "lifesat2", "lifesat3", "lifesat4", "lifesat5", "lifesat6", "lifesat7", 
            "zcog2", "zcog3", "zcog4", "zcog5", "zcog6", "zcog7", 
            "zbasnv2", "zbsrar2", "zbasnv3", "zbaspc3", "zbasps3", "zbaswr4", "zbaspc4", "znferpm4",
            "sdqinternal2", "sdqinternal3", "sdqinternal4", "sdqinternal5", "sdqinternal6", "sdqinternal7",
            "sdqemotion2", "sdqemotion3", "sdqemotion4", "sdqemotion5", "sdqemotion6", "sdqemotion7",
            "sdqconduct2", "sdqconduct3", "sdqconduct4", "sdqconduct5", "sdqconduct6", "sdqconduct7",
            "sdqexternal2", "sdqexternal3", "sdqexternal4", "sdqexternal5", "sdqexternal6", "sdqexternal7",
            "hosp2", "hosp3", "hosp4", "hosp5", "hosp6", "hosp7",  
            "alc2", "alc3", "alc4", "alc5", "alc6", "alc7", 
            "condis3", "condis4", "condis5", "condis6", "condis7",
            "sen4", "sen5", "sen6", 
            "truancy5", "truancy6", 
            "excl5", "excl6",
            "kessler7"] 
            if int(var[-1]) > swpstn and int(var[-1]) <= nwvs]
    varl.append('tot_cst_ttl')
    varl.append('lifesat_ttl')
    desc = diffdescuw(varl, nint)
    desc = desc.rename(index=var_dict)
    del desc['mean']
    del desc['std']
    logging.info('Intervention effects')
    logging.info(desc)
    
    

    # Create a new DataFrame with interleaved values and standard errors
    idt = []
    for idx in desc.index:
        row = [f"{desc.loc[idx, f'int{i}']:.2f}" for i in range(1, nint + 1)]
        se_row = [f"({desc.loc[idx, f'int{i}se']:.3f})" for i in range(1, nint + 1)]
        idt.append(row)
        idt.append(se_row)

    idf = pd.DataFrame(idt, index=[f"{idx}" if i % 2 == 0 else "" for i, idx in enumerate(desc.index.repeat(2))], columns=[f'int{i}' for i in range(1, nint + 1)])
    
    for i in range(1, nint+1):
        idf = idf.rename(columns={f"int{i}": f"Scenario {i}"})
    
    tex_path = os.path.join(lfsm, f'output/{rskfctr}/tables', f'otwdif_{rskfctr}.tex')  # Define your output file path here
    idf.to_latex(tex_path,
                index=True,
                formatters={"name": str.upper},
                float_format="{:.3f}".format,
            )
    
    
    del desc
    del tex_path
    del idt
    del idf
    
    # ###Descriptive Stats of simulation results
    # ##All outcomes
    # varl = [var for var in ["bdgcseme7", "distress7", "obesity7", "smkreg7", "prfrhlth7",
    #         "lifesat2", "lifesat3", "lifesat4", "lifesat5", "lifesat6", "lifesat7", 
    #         "zcog2", "zcog3", "zcog4", "zcog5", "zcog6", "zcog7", 
    #         "zbasnv2", "zbsrar2", "zbasnv3", "zbaspc3", "zbasps3", "zbaswr4", "zbaspc4", "znferpm4",
    #         "sdqinternal2", "sdqinternal3", "sdqinternal4", "sdqinternal5", "sdqinternal6", "sdqinternal7",
    #         "sdqemotion2", "sdqemotion3", "sdqemotion4", "sdqemotion5", "sdqemotion6", "sdqemotion7",
    #         "sdqconduct2", "sdqconduct3", "sdqconduct4", "sdqconduct5", "sdqconduct6", "sdqconduct7",
    #         "sdqexternal2", "sdqexternal3", "sdqexternal4", "sdqexternal5", "sdqexternal6", "sdqexternal7",
    #         "hosp2", "hosp3", "hosp4", "hosp5", "hosp6", "hosp7",  
    #         "alc2", "alc3", "alc4", "alc5", "alc6", "alc7", 
    #         "condis3", "condis4", "condis5", "condis6", "condis7",
    #         "sen4", "sen5", "sen6", 
    #         "truancy5", "truancy6", 
    #         "excl5", "excl6",
    #         "kessler7"] 
    #         if int(var[-1]) > swpstn and int(var[-1]) <= nwvs]
    # varl.append('tot_cst_ttl')
    # varl.append('lifesat_ttl')
    # desc = diffdescuw(varl, nint)
    # desc = desc.rename(index=var_dict)
    # del desc['mean']
    # del desc['std']
    # logging.info('Intervention effects')
    # logging.info(desc)
    
    

    # # Create a new DataFrame with interleaved values and standard errors
    # idt = []
    # for idx in desc.index:
    #     row = [f"{desc.loc[idx, f'int{i}']:.2f}" for i in range(1, nint + 1)]
    #     se_row = [f"({desc.loc[idx, f'int{i}se']:.3f})" for i in range(1, nint + 1)]
    #     idt.append(row)
    #     idt.append(se_row)

    # idf = pd.DataFrame(idt, index=[f"{idx}" if i % 2 == 0 else "" for i, idx in enumerate(desc.index.repeat(2))], columns=[f'int{i}' for i in range(1, nint + 1)])
    
    # for i in range(1, nint+1):
    #     idf = idf.rename(columns={f"int{i}": f"Scenario {i}"})
    
    # tex_path = os.path.join(lfsm, f'output/{rskfctr}/tables', f'otwdif_{rskfctr}.tex')  # Define your output file path here
    # idf.to_latex(tex_path,
    #             index=True,
    #             formatters={"name": str.upper},
    #             float_format="{:.3f}".format,
    #         )
    
    
    # del desc
    # del tex_path
    # del idt
    # del idf
    
    # ##All outcomes
    # varl = [var for var in ["bdgcseme7", "distress7", "obesity7", "smkreg7", "prfrhlth7",
    #         "lifesat2", "lifesat3", "lifesat4", "lifesat5", "lifesat6", "lifesat7", 
    #         "zcog2", "zcog3", "zcog4", "zcog5", "zcog6", "zcog7", 
    #         "zbasnv2", "zbsrar2", "zbasnv3", "zbaspc3", "zbasps3", "zbaswr4", "zbaspc4", "znferpm4",
    #         "sdqinternal2", "sdqinternal3", "sdqinternal4", "sdqinternal5", "sdqinternal6", "sdqinternal7",
    #         "sdqemotion2", "sdqemotion3", "sdqemotion4", "sdqemotion5", "sdqemotion6", "sdqemotion7",
    #         "sdqconduct2", "sdqconduct3", "sdqconduct4", "sdqconduct5", "sdqconduct6", "sdqconduct7",
    #         "sdqexternal2", "sdqexternal3", "sdqexternal4", "sdqexternal5", "sdqexternal6", "sdqexternal7",
    #         "hosp2", "hosp3", "hosp4", "hosp5", "hosp6", "hosp7",  
    #         "alc2", "alc3", "alc4", "alc5", "alc6", "alc7", 
    #         "condis3", "condis4", "condis5", "condis6", "condis7",
    #         "sen4", "sen5", "sen6", 
    #         "truancy5", "truancy6", 
    #         "excl5", "excl6",
    #         "kessler7"] 
    #         if int(var[-1]) > swpstn and int(var[-1]) <= nwvs]
    # varl.append('tot_cst_ttl')
    # varl.append('lifesat_ttl')
    # desc = diffdescuw(varl, nint, rec = True)
    # desc = desc.rename(index=var_dict)
    # del desc['mean']
    # del desc['std']
    # logging.info('Intervention effects')
    # logging.info(desc)
    
    

    # # Create a new DataFrame with interleaved values and standard errors
    # idt = []
    # for idx in desc.index:
    #     row = [f"{desc.loc[idx, f'int{i}']:.2f}" for i in range(1, nint + 1)]
    #     se_row = [f"({desc.loc[idx, f'int{i}se']:.3f})" for i in range(1, nint + 1)]
    #     idt.append(row)
    #     idt.append(se_row)

    # idf = pd.DataFrame(idt, index=[f"{idx}" if i % 2 == 0 else "" for i, idx in enumerate(desc.index.repeat(2))], columns=[f'int{i}' for i in range(1, nint + 1)])
    
    # for i in range(1, nint+1):
    #     idf = idf.rename(columns={f"int{i}": f"Scenario {i}"})
    
    # tex_path = os.path.join(lfsm, f'output/{rskfctr}/tables', f'otwrdif_{rskfctr}.tex')  # Define your output file path here
    # idf.to_latex(tex_path,
    #             index=True,
    #             formatters={"name": str.upper},
    #             float_format="{:.3f}".format,
    #         )
    
    
    # del desc
    # del tex_path
    # del idt
    # del idf
    
    ###Adverse outcomes
    varl = ["bdgcseme7", "distress7", "obesity7", "smkreg7", "prfrhlth7"]
    desc = diffdescuw(varl, nint, neg = True)
    desc = desc.rename(index=var_dict)
    del desc['mean']
    del desc['std']
    ad = desc[['int' + str(i) for i in range(1, nint+1)]]
    # if nint == 1:
    #     d = desc[['int1']]
    # elif nint == 2:
    #     d = desc[['int1', 'int2']]
    # elif nint == 3:
    #     d = desc[['int1', 'int2', 'int3']]
    # elif nint == 4:
    #     d = desc[['int1', 'int2', 'int3', 'int4']]
    for i in range(1, nint+1):
        ad[f'int{i}'] = ad[f'int{i}']*700000
    logging.info(ad)
    
    
    
    ###Wellbeing 
    ##All outcomes
    varl = [f"lifesat{i}" for i in range(swpstn + 1, nwvs + 1)]
    varl.append("lifesat_ttl")
    desc = diffdescuw(varl, nint)
    del desc['mean']
    del desc['std']
    desc = desc.drop('lifesat_ttl')
    wb = desc[['int' + str(i) for i in range(1, nint+1)]]
    wb['var'] = wb.index
    wb['sweep'] = wb['var'].str.extract('(\d+)$')[0]
    del wb['var']
    logging.info('Intervention effects on wellbeing')
    lfstfm(wb, 0.035)
    
    logging.info('Intervention effects on wellbeing (undiscounted)')
    lfstfm(wb, 0)
    
             

    
    ##All Costs
    varl = [x for x in histint0.columns if x.endswith('_cst')]
    
    ##Costs by source
    varl = [var for var in ['hc_cst2', 'hc_cst3', 'hc_cst4', 'hc_cst5', 'hc_cst6', 'hc_cst7',
            'll_cst2', 'll_cst3', 'll_cst4', 'll_cst5', 'll_cst6', 'll_cst7',
            'cd_cst3', 'cd_cst4', 'cd_cst5', 'cd_cst6', 'cd_cst7',
            'se_cst4', 'se_cst5', 'se_cst6',
            'pe_cst5', 'pe_cst6',
            'pt_cst5', 'pt_cst6',
            'tot_cst2', 'tot_cst3', 'tot_cst4', 'tot_cst5', 'tot_cst6', 'tot_cst7'] 
            if int(var[-1]) > swpstn and int(var[-1]) <= nwvs]
    varl.append('tot_cst_ttl')
    desc = diffdescuw(varl, nint)
    desc = desc.rename(index=var_dict)
    del desc['mean']
    del desc['std']
    logging.info('Intervention effects - costs by source')
    logging.info(desc)
    desc = desc.drop('tot_cst_ttl')    
    # Create a new DataFrame with interleaved values and standard errors
    idt = []
    for idx in desc.index:
        row = [f"{desc.loc[idx, f'int{i}']:.2f}" for i in range(1, nint + 1)]
        se_row = [f"({desc.loc[idx, f'int{i}se']:.3f})" for i in range(1, nint + 1)]
        idt.append(row)
        idt.append(se_row)

    idf = pd.DataFrame(idt, index=[f"{idx}" if i % 2 == 0 else "" for i, idx in enumerate(desc.index.repeat(2))], columns=[f'int{i}' for i in range(1, nint + 1)])
    
    for i in range(1, nint+1):
        idf = idf.rename(columns={f"int{i}": f"Scenario {i}"})
    
    tex_path = os.path.join(lfsm, f'output/{rskfctr}/tables', f'cstswdif_{rskfctr}.tex')  # Define your output file path here
    idf.to_latex(tex_path,
                index=True,
                formatters={"name": str.upper},
                float_format="{:.3f}".format,
            )
    
    del desc
    del tex_path
    del idt
    del idf
    
  
    
    
    ##Costs by sector
    varl = [var for var in ['nhs_cst2', 'nhs_cst3', 'nhs_cst4', 'nhs_cst5', 'nhs_cst6', 'nhs_cst7', 
            'ssc_cst3', 'ssc_cst4', 'ssc_cst5', 'ssc_cst6', 'ssc_cst7',
            'ded_cst3', 'ded_cst4', 'ded_cst5', 'ded_cst6', 'ded_cst7'] 
            if int(var[-1]) > swpstn and int(var[-1]) <= nwvs]
    varl.append('tot_cst_ttl')
    desc = diffdescuw(varl, nint)
    desc = desc.rename(index=var_dict)
    del desc['mean']
    del desc['std']
    logging.info('Intervention effects - costs by sector')
    logging.info(desc)
    desc = desc.drop('tot_cst_ttl')    
    # Create a new DataFrame with interleaved values and standard errors
    idt = []
    for idx in desc.index:
        row = [f"{desc.loc[idx, f'int{i}']:.2f}" for i in range(1, nint + 1)]
        se_row = [f"({desc.loc[idx, f'int{i}se']:.3f})" for i in range(1, nint + 1)]
        idt.append(row)
        idt.append(se_row)

    idf = pd.DataFrame(idt, index=[f"{idx}" if i % 2 == 0 else "" for i, idx in enumerate(desc.index.repeat(2))], columns=[f'int{i}' for i in range(1, nint + 1)])
    
    for i in range(1, nint+1):
        idf = idf.rename(columns={f"int{i}": f"Scenario {i}"})
    
    tex_path = os.path.join(lfsm, f'output/{rskfctr}/tables', f'cstowdif_{rskfctr}.tex')  # Define your output file path here
    idf.to_latex(tex_path,
                index=True,
                formatters={"name": str.upper},
                float_format="{:.3f}".format,
            )
    
    del desc
    del tex_path
    del idt
    del idf
    
    
    ######## Plots
    
    ##### Bad Age 17 outcomes
    varl = ["bdgcseme7", "distress7", "obesity7", "smkreg7", "prfrhlth7"]
    desc = diffdescuw(varl, nint, neg = True, ppt = True)
    del desc['mean']
    del desc['std']
    logging.info('Intervention effects on adverse outcomes - percentage point')
    logging.info(desc)
    # logging.info(desc[['int1', 'int2', 'int3', 'int4']])
    # Plotting bar plots for comparison
    plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures' f'inc17ocdif_{rskfctr}.png')
    lbl = ["Poor GCSEs", "Psych. distress", "Obesity", "Regular smoker", "Poor health"]
    yt = 'Percentage point reduction in adverse outcome'
    xt = 'Adverse outcome'
    diffplt(desc, lbl, plt_path, nint, ya=yt)
    
    ### Number of children
    desc = diffdescuw(varl, nint, neg = True)
    del desc['mean']
    del desc['std']
    logging.info(desc)
    for i in range(1, nint + 1):
        desc[f'int{i}'] = desc[f'int{i}'] * 100000
        desc[f'int{i}se'] = desc[f'int{i}se'] * 100000
    logging.info('Intervention effects on adverse outcomes - number per 100000')
    logging.info(desc)
    # Plotting bar plots for comparison
    plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'inc17ncdif_{rskfctr}.png')
    lbl = ["Poor GCSEs", "Psych. distress", "Obesity", "Regular smoker", "Poor health"]
    yt = 'Number of cases avoided per 100000'
    xt = 'Adverse outcome'
    diffplt(desc, lbl, plt_path, nint, ya=yt)
    
    ### Percentage change
    desc = diffdescuw(varl, nint, neg = True, pct = True)
    logging.info('Intervention effects on adverse outcomes - percent')
    logging.info(desc)
    # logging.info(desc[['int1', 'int2', 'int3', 'int4']])
    # Plotting bar plots for comparison
    plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'inc17pcdif_{rskfctr}.png')
    lbl = ["Poor GCSEs", "Psych. distress", "Obesity", "Regular smoker", "Poor health"]
    yt = 'Percent reduction in adverse outcome'
    xt = 'Adverse outcome'
    diffplt(desc, lbl, plt_path, nint, ya=yt)
    
    
    ##### Life Satisfaction
    varl = [f"lifesat{i}" for i in range(swpstn + 1, nwvs + 1)]
    desc = diffdescuw(varl, nint)
    del desc['mean']
    del desc['std']
    logging.info('Intervention effects on life satisfaction')
    logging.info(desc)
    # logging.info(desc[['int1', 'int2', 'int3', 'int4']])
    # Plotting means
    plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'lifesatdif_{rskfctr}.png')
    # lbl = ["2", "3", "4", "5", "6", "7"]
    # lbl = ["3", "5", "7", "11", "14", "17"]
    lbl = ["Age 3", "Age 5", "Age 7", "Age 11", "Age 14", "Age 17"][(swpstn + 1 - 2):]
    yt = 'Change in life satisfaction'
    xt = 'Age'
    diffplt(desc, lbl, plt_path, nint, ya=yt)
    
    
    
    ##### Cognitive ability
    varl = [f"zcog{i}" for i in range(swpstn + 1, nwvs + 1)]
    desc = diffdescuw(varl, nint)
    del desc['mean']
    del desc['std']
    logging.info('Intervention effects on Cognitive ability')
    logging.info(desc)
    # logging.info(desc[['int1', 'int2', 'int3', 'int4']])
    # Plotting means
    plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'zcogdif_{rskfctr}.png')
    # lbl = ["2", "3", "4", "5", "6", "7"]
    # lbl = ["3", "5", "7", "11", "14", "17"]
    lbl = ["Age 3", "Age 5", "Age 7", "Age 11", "Age 14", "Age 17"][(swpstn + 1 - 2):]
    yt = 'Change in cognitive ability'
    xt = 'Age'
    diffplt(desc, lbl, plt_path, nint, ya=yt)
         
    
    ##### SDQ internalising
    varl = [f"sdqinternal{i}" for i in range(swpstn + 1, nwvs + 1)]
    desc = diffdescuw(varl, nint)
    del desc['mean']
    del desc['std']
    logging.info('Intervention effects on SDQ internalising')
    logging.info(desc)
    # logging.info(desc[['int1', 'int2', 'int3', 'int4']])
    # Plotting means
    plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'intrnldif_{rskfctr}.png')
    # lbl = ["2", "3", "4", "5", "6", "7"]
    # lbl = ["3", "5", "7", "11", "14", "17"]
    lbl = ["Age 3", "Age 5", "Age 7", "Age 11", "Age 14", "Age 17"][(swpstn + 1 - 2):]
    yt = 'Change in SDQ internalising'
    xt = 'Age'
    diffplt(desc, lbl, plt_path, nint, ya=yt)
    
    
    ##### SDQ Externalising
    varl = [f"sdqexternal{i}" for i in range(swpstn + 1, nwvs + 1)]
    desc = diffdescuw(varl, nint, neg = True)
    del desc['mean']
    del desc['std']
    logging.info('Intervention effects on SDQ externalising')
    logging.info(desc)
    # logging.info(desc[['int1', 'int2', 'int3', 'int4']])
    # Plotting means
    plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'sdqextdif_{rskfctr}.png')
    # lbl = ["2", "3", "4", "5", "6", "7"]
    # lbl = ["3", "5", "7", "11", "14", "17"]
    lbl = ["Age 3", "Age 5", "Age 7", "Age 11", "Age 14", "Age 17"][(swpstn + 1 - 2):]
    yt = 'Change in SDQ externalising'
    xt = 'Age'
    diffplt(desc, lbl, plt_path, nint, ya=yt)
    
    
    ##### Binary outcomes
    ##### Hospitilisation
    varl = [f"hosp{i}" for i in range(swpstn + 1, nwvs + 1)]
    desc = diffdescuw(varl, nint, neg = True, ppt = True)
    del desc['mean']
    del desc['std']
    logging.info('Intervention effects on hospitilisation')
    logging.info(desc)
    # Plotting means with error bars
    plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'hospdif_{rskfctr}.png')
    # lbl = ["2", "3", "4", "5", "6", "7"]
    # lbl = ["3", "5", "7", "11", "14", "17"]
    lbl = ["Age 3", "Age 5", "Age 7", "Age 11", "Age 14", "Age 17"][(swpstn + 1 - 2):]
    yt = 'Percentage reduction in hospitilisation'
    xt = 'Age'
    diffplt(desc, lbl, plt_path, nint, ya=yt)
    
    
    ##### Disability
    varl = [f"alc{i}" for i in range(swpstn + 1, nwvs + 1)]
    desc = diffdescuw(varl, nint, neg = True, ppt = True)
    del desc['mean']
    del desc['std']
    logging.info('Intervention effects on disability')
    logging.info(desc)
    # Plotting means with error bars
    plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'disdif_{rskfctr}.png')
    # lbl = ["2", "3", "4", "5", "6", "7"]
    # lbl = ["3", "5", "7", "11", "14", "17"]
    lbl = ["Age 3", "Age 5", "Age 7", "Age 11", "Age 14", "Age 17"][(swpstn + 1 - 2):]
    yt = 'Percentage reduction in disability'
    xt = 'Age'
    diffplt(desc, lbl, plt_path, nint, ya=yt)
    
    
    ##### Conduct disorder
    varl = [f"condis{i}" for i in range(swpstn + 1, nwvs + 1)]
    desc = diffdescuw(varl, nint, neg = True, ppt = True)
    del desc['mean']
    del desc['std']
    logging.info('Intervention effects on conduct disorder')
    logging.info(desc)
    # Plotting means with error bars
    plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'condisdif_{rskfctr}.png')
    # lbl = ["3", "4", "5", "6", "7"]
    # lbl = ["5", "7", "11", "14", "17"]
    lbl = ["Age 5", "Age 7", "Age 11", "Age 14", "Age 17"][(swpstn + 1 - 3):]
    yt = 'Percentage reduction in conduct disorder'
    xt = 'Age'
    diffplt(desc, lbl, plt_path, nint, ya=yt)
    
    
    ##### Education
    ##### SEN
    varl = ["sen4", "sen5", "sen6"]
    desc = diffdescuw(varl, nint, neg = True, ppt = True)
    del desc['mean']
    del desc['std']
    logging.info('Intervention effects on special education needs')
    logging.info(desc)
    # Plotting means with error bars
    plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'sendif_{rskfctr}.png')
    # lbl = ["4", "5", "6"]
    # lbl = ["7", "11", "14"]
    lbl = ["Age 7", "Age 11", "Age 14"]
    yt = 'Percentage reduction in SEN'
    xt = 'Age'
    diffplt(desc, lbl, plt_path, nint, ya=yt)
    
    
    ##### Any Truancy
    varl = ["truancy5", "truancy6"]
    desc = diffdescuw(varl, nint, neg = True, ppt = True)
    del desc['mean']
    del desc['std']
    logging.info('Intervention effects on truancy')
    logging.info(desc)
    # Plotting means with error bars
    plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'truancdif_{rskfctr}.png')
    # lbl = ["5", "6"]
    # lbl = ["11", "14"]
    lbl = ["Age 11", "Age 14"]
    yt = 'Percentage reduction in truancy'
    xt = 'Age'
    diffplt(desc, lbl, plt_path, nint, ya=yt)
    
    
    ##### Any Exclusion
    varl = ["excl5", "excl6"]
    desc = diffdescuw(varl, nint, neg = True, ppt = True)
    del desc['mean']
    del desc['std']
    logging.info('Intervention effects on exclusion')
    logging.info(desc)
    # Plotting means with error bars
    plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'exclusdif_{rskfctr}.png') 
    # lbl = ["5", "6"]
    # lbl = ["11", "14"]
    lbl = ["Age 11", "Age 14"]
    yt = 'Percentage reduction in exclusion'
    xt = 'Age'
    diffplt(desc, lbl, plt_path, nint, ya=yt)
    
    
    ##### Costs by source
    varl = [var for var in ['hc_cst2', 'hc_cst3', 'hc_cst4', 'hc_cst5', 'hc_cst6', 'hc_cst7',
            'll_cst2', 'll_cst3', 'll_cst4', 'll_cst5', 'll_cst6', 'll_cst7',
            'cd_cst3', 'cd_cst4', 'cd_cst5', 'cd_cst6', 'cd_cst7',
            'se_cst4', 'se_cst5', 'se_cst6',
            'pe_cst5', 'pe_cst6',
            'pt_cst5', 'pt_cst6'] 
            if int(var[-1]) > swpstn and int(var[-1]) <= nwvs]
    desc = diffdescuw(varl, nint, neg = True)
    del desc['mean']
    del desc['std']
    logging.info('Intervention effects on costs')
    logging.info(desc)
    
    for i in range(1, nint+1):
        desc_i = pd.DataFrame(desc[f'int{i}'].copy())
        
        d_i = desc_i.T
        d_i.columns = d_i.columns.str.replace('_cst', '')
        #Change values in 7th sweep with values in 6th sweep for education variables
        d_i['pe7'] = d_i['pe6']
        d_i['pt7'] = d_i['pt6']
        d_i['se7'] = d_i['se6']
    
        # Reshape the means to a data frame
        d_i = d_i.melt(var_name='variable', value_name='value')
    
        # Extract the measurement type and sweep number
        d_i['Sweep'] = d_i['variable'].str.extract('(\d+)$')[0]
        d_i['Type'] = d_i['variable'].str.extract('([a-z]+)')[0]
    
        # Rearrange and sort the DataFrame
        d_i = d_i[['Sweep', 'Type', 'value']]
        d_i = d_i.sort_values(by=['Sweep', 'Type']).reset_index(drop=True)
    
        # Pivot the table to get desired format
        d_i = d_i.pivot(index='Sweep', columns='Type', values='value')
        d_i.reset_index(inplace=True)
        logging.info(f'Intervention {i} effects on costs')
        logging.info(d_i)
        d_i.fillna(0, inplace=True)
        d_i.fillna(0, inplace=True)
        globals()[f"d{i}"] = d_i.copy()
        # d_i = d_i.apply(pd.to_numeric, errors='coerce')
        # globals()[f"d{i}p"] = d_i[d_i > 0].copy()
        # globals()[f"d{i}n"] = d_i[d_i < 0].copy()
    
    for i in range(1, nint + 1):
        d_i = globals()[f"d{i}"]
        logging.info(f'Intervention {i} effects on costs')
        logging.info(d_i)
        
        d_ip = d_i[d_i.apply(pd.to_numeric, errors='coerce') > 0].copy()
        d_in = d_i[d_i.apply(pd.to_numeric, errors='coerce') < 0].copy()
        d_in['Sweep'] = d_i['Sweep']
        d_ip.fillna(0, inplace=True)
        d_in.fillna(0, inplace=True)
        
        logging.info(f'Intervention {i} effects on costs - reduction')
        logging.info(d_ip)
        logging.info(f'Intervention {i} effects on costs - increase')
        logging.info(d_in)
        
        globals()[f"d{i}p"] = d_ip
        globals()[f"d{i}n"] = d_in
    
    # Calculate total and average costs
    for i in range(1, nint+1):
        d_i = globals()[f"d{i}"]
        # dp = globals()[f"d{i}p"]
        # dn = globals()[f"d{i}n"]
        logging.info(f'Intervention {i} effects on costs with sum')
        cstsm(d_i, 0.035)
        logging.info(f'Intervention {i} effects on costs with undiscounted sum')
        cstsm(d_i, 0)
        # logging.info(f'Intervention {i} effects on costs with sum - positive')
        # cstsm(dp, 0.035)
        # logging.info(f'Intervention {i} effects on costs with undiscounted sum - positive')
        # cstsm(dpu, 0)
        # logging.info(f'Intervention {i} effects on costs with sum - negative')
        # cstsm(dn, 0.035)
        # logging.info(f'Intervention {i} effects on costs with undiscounted sum - positive')
        # cstsm(dnu, 0)
        
        
    # Create dataframes to store moximum and minimum by sweep for y axis limits of graph
    aps = pd.DataFrame()
    ans = pd.DataFrame()
    for i in range(1, nint+1):    
        globals()[f"d{i}p"]['sum'] = globals()[f"d{i}p"].sum(numeric_only=True, axis=1)
        globals()[f"d{i}n"]['sum'] = globals()[f"d{i}n"].sum(numeric_only=True, axis=1)
        aps[f'int{i}'] = globals()[f"d{i}p"]['sum']
        ans[f'int{i}'] = globals()[f"d{i}n"]['sum']

    # Plotting means with error bars
    # lbl = ["2", "3", "4", "5", "6", "7"]
    # lbl = ["3", "5", "7", "11", "14", "17"]
    lbl = ["Age 3", "Age 5", "Age 7", "Age 11", "Age 14", "Age 17"][(swpstn + 1 - 2):]
    lbl1 = ['S1', 'S2', 'S3', 'S4']
    r = np.arange(len(lbl))
    w = 0.2
    plt.figure(figsize=(15, 9))
    plt.rcParams.update({'font.size': 14})  # Set the font size to 14
    # d.plot(x='Sweep', kind='bar', stacked=True)
    plt.grid(axis='y', which='both', linestyle="--", alpha=0.5, zorder=1)
    for i in range(nint):
        dp = globals()[f'd{i+1}p']
        dn = globals()[f'd{i+1}n']
        plt.bar(r + w * i, dp['hc'], capsize=5, width = w, edgecolor = 'black', color='#44AA99', zorder=2)
        plt.bar(r + w * i, dp['ll'], capsize=5, width = w, edgecolor = 'black', bottom=dp['hc'], color='#882255', zorder=2)
        plt.bar(r + w * i, dp['cd'], capsize=5, width = w, edgecolor = 'black', bottom=dp['hc']+dp['ll'], color='#88CCEE', zorder=2)
        plt.bar(r + w * i, dp['se'], capsize=5, width = w, edgecolor = 'black', bottom=dp['hc']+dp['ll']+dp['cd'], color='#117733', zorder=2)
        plt.bar(r + w * i, dp['pt'], capsize=5, width = w, edgecolor = 'black', bottom=dp['hc']+dp['ll']+dp['cd']+dp['se'], color='#CC6677', zorder=2)
        plt.bar(r + w * i, dp['pe'], capsize=5, width = w, edgecolor = 'black', bottom=dp['hc']+dp['ll']+dp['cd']+dp['se']+dp['pt'], color='#332288', zorder=2)
        plt.bar(r + w * i, dn['hc'], capsize=5, width = w, edgecolor = 'black', color='#44AA99', zorder=2)
        plt.bar(r + w * i, dn['ll'], capsize=5, width = w, edgecolor = 'black', bottom=dn['hc'], color='#882255', zorder=2)
        plt.bar(r + w * i, dn['cd'], capsize=5, width = w, edgecolor = 'black', bottom=dn['hc']+dn['ll'], color='#88CCEE', zorder=2)
        plt.bar(r + w * i, dn['se'], capsize=5, width = w, edgecolor = 'black', bottom=dn['hc']+dn['ll']+dn['cd'], color='#117733', zorder=2)
        plt.bar(r + w * i, dn['pt'], capsize=5, width = w, edgecolor = 'black', bottom=dn['hc']+dn['ll']+dn['cd']+dn['se'], color='#CC6677', zorder=2)
        plt.bar(r + w * i, dn['pe'], capsize=5, width = w, edgecolor = 'black', bottom=dn['hc']+dn['ll']+dn['cd']+dn['se']+dn['pt'], color='#332288', zorder=2)
    # plt.xlabel('MCS Sweep')
    plt.ylabel('Reduction in annual cost per child in £s')
    # plt.xticks(r + w * (np.arange(nint)-1) / 2, lbl1 * r)
    plt.xticks(r + w*(nint-1)/2, lbl)
    # plt.title('Simulated baseline cost at each age')
    if aps.max().max()>100 : 
        plt.ylim(top=(aps.max().max() // 10 + 2) * 10, bottom=(ans.min().min() // 10) * 10)
    else :
        plt.ylim(top=(aps.max().max() // 5 + 2) * 5, bottom=(ans.min().min() // 5) * 5)
    #plt.ylim(top=(aps.max().max() // 10 + 2) * 10, bottom=(ans.min().min() // 10) * 10)
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    plt.legend(["Hospitalisation", "Disability", "Conduct disorder", "SEN", "Truancy", "Exclusion"], loc='upper left')
    plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'cstsrcdif_{rskfctr}.png')  # Define your output file path here
    # os.remove(plt_path)
    plt.savefig(plt_path, bbox_inches='tight')
    plt.show()
    
    print(aps.max().max())
    
    del lbl
    del plt_path
    del w
    del r
    
    
    del varl
    del i
    del d_i
    del desc_i
    del desc
    del dp
    del dn
    del aps
    del ans
    for i in range(1, nint+1):
        del globals()[f"d{i}"]
        del globals()[f"d{i}n"]
        del globals()[f"d{i}p"]

    for rtp in ["CT", "CTI", "T", "RT", "RTI"]: 
        # CT- Cohort total
        # T- Total per child
        # CTI- Cohort total (SDQ Internalising for life satisfaction)
        # RT- Total per recipient child
        #####Genrate final results table
        #Panel 1 - Age 17 outcomes
        varl = ["bdgcseme7", "distress7", "obesity7", "smkreg7", "prfrhlth7"]
        if rtp in ("CT", "CTI", "T"):
            daoc = diffdescuw(varl, nint, neg = True)
            # daoq = diffdescuwq(varl, nint, neg = True)
        if rtp in ("RT", "RTI"):
            daoc = diffdescuw(varl, nint, neg = True, rec = True)
            # daocq = diffdescuw(varl, nint, neg = True, rec = True)
        del daoc['mean']
        del daoc['std']
        daoc = daoc.loc[:, ~daoc.columns.str.endswith('se')] 
        # daocq = daocq.loc[:, ~daocq.columns.str.endswith('se')] 
        logging.info(daoc)
        if rtp in ("CT", "CTI"):
            for i in range(1, nint + 1):
                daoc[f'int{i}'] = daoc[f'int{i}'] * 700000
                # daoc[f'int{i}se'] = daoc[f'int{i}se'] * 700000
        elif rtp == ("T", "RT", "RTI"):        
            for i in range(1, nint + 1):
                daoc[f'int{i}'] = daoc[f'int{i}'] * 100
                # daocq[f'int{i}'] = daocq[f'int{i}'] * 100
        daoc = daoc.rename(index=var_dict) 
        for i in range(1, nint+1):
            daoc = daoc.rename(columns={f"int{i}": f"Scenario {i}"})
            # daocq = daocq.rename(columns={f"int{i}": f"Scenario {i}"})
           
        #Panel 2 - Costs
        varl = [var for var in ['hc_cst2', 'hc_cst3', 'hc_cst4', 'hc_cst5', 'hc_cst6', 'hc_cst7',
                'll_cst2', 'll_cst3', 'll_cst4', 'll_cst5', 'll_cst6', 'll_cst7',
                'cd_cst3', 'cd_cst4', 'cd_cst5', 'cd_cst6', 'cd_cst7',
                'se_cst4', 'se_cst5', 'se_cst6',
                'pe_cst5', 'pe_cst6',
                'pt_cst5', 'pt_cst6'] 
            if int(var[-1]) > swpstn and int(var[-1]) <= nwvs]
        if rtp in ("CT", "CTI", "T"):
            desc = diffdescuw(varl, nint, neg = True)
        if rtp in ("RT", "RTI"):
            desc = diffdescuw(varl, nint, neg = True, rec = True)
        del desc['mean']
        del desc['std']
        logging.info('Intervention effects on costs')
        logging.info(desc)
    
        for i in range(1, nint+1):
            desc_i = pd.DataFrame(desc[f'int{i}'].copy())
            
            d_i = desc_i.T
            d_i.columns = d_i.columns.str.replace('_cst', '')
            #Change values in 7th sweep with values in 6th sweep for education variables
            d_i['pe7'] = d_i['pe6']
            d_i['pt7'] = d_i['pt6']
            d_i['se7'] = d_i['se6']
    
            # Reshape the means to a data frame
            d_i = d_i.melt(var_name='variable', value_name='value')
    
            # Extract the measurement type and sweep number
            d_i['Sweep'] = d_i['variable'].str.extract('(\d+)$')[0]
            d_i['Type'] = d_i['variable'].str.extract('([a-z]+)')[0]
    
            # Rearrange and sort the DataFrame
            d_i = d_i[['Sweep', 'Type', 'value']]
            d_i = d_i.sort_values(by=['Sweep', 'Type']).reset_index(drop=True)
    
            # Pivot the table to get desired format
            d_i = d_i.pivot(index='Sweep', columns='Type', values='value')
            d_i.reset_index(inplace=True)
            logging.info(f'Intervention {i} effects on costs')
            logging.info(d_i)
            d_i.fillna(0, inplace=True)
            d_i.fillna(0, inplace=True)
            globals()[f"d{i}"] = d_i.copy()
            # d_i = d_i.apply(pd.to_numeric, errors='coerce')
            # globals()[f"d{i}p"] = d_i[d_i > 0].copy()
            # globals()[f"d{i}n"] = d_i[d_i < 0].copy()
        dtwc = pd.DataFrame()
        dtuc = pd.DataFrame()
        for i in range(1, nint+1):
            d_i = globals()[f"d{i}"]
            # logging.info(f'Intervention {i} effects on costs with sum')
            # dtwc = cstsm(d_i, 0.035)
            # logging.info(f'Intervention {i} effects on costs with undiscounted sum')
            # dtuc  = cstsm(d_i, 0)
            # Initialize dtwc and dtuc if they don't exist
            # if 'dtwc' not in globals():
            #     dtwc = pd.DataFrame()
            # if 'dtuc' not in globals():
            #     dtuc = pd.DataFrame()
    
            # Add each output of cstsm as a column in dtwc and dtuc
            dtwc[f'int{i}'] = cstsm(d_i, 0.035, ctp=rtp)
            dtuc[f'int{i}'] = cstsm(d_i, 0, ctp=rtp)
        if rtp in ("CT", "CTI"):    
            #Convert values to millions    
            dtwc = dtwc / 1000000 
            dtuc = dtuc / 1000000   
           
        #Correctly label rows and columns
        dtwc = dtwc.reindex(['ttl', 'hc', 'll', 'cd', 'se', 'pt', 'pe'])
        dtuc = dtuc.reindex(['ttl', 'hc', 'll', 'cd', 'se', 'pt', 'pe'])
        dtwc = dtwc.rename(index=var_dict) 
        dtuc = dtuc.rename(index=var_dict) 
        for i in range(1, nint+1):
            dtwc = dtwc.rename(columns={f"int{i}": f"Scenario {i}"})
            dtuc = dtuc.rename(columns={f"int{i}": f"Scenario {i}"})
            
        #Panel 3 - Wellbeing
        varl = [f"lifesat{i}" for i in range(swpstn + 1, nwvs + 1)]
        if rtp == ("CTI", "RTI"):
            varl = [f"lfstint{i}" for i in range(swpstn + 1, nwvs + 1)]
        if rtp in ("CT", "CTI", "T"):
            desc = diffdescuw(varl, nint)
        if rtp in ("RT", "RTI"):
            desc = diffdescuw(varl, nint, rec = True)
        del desc['mean']
        del desc['std']
        wb = desc[['int' + str(i) for i in range(1, nint+1)]]
        for i in range(1, nwvs+1):
            wb = wb.rename(index={f"lfstint{i}": f"lifesat{i}"})
        wb['var'] = wb.index
        wb['sweep'] = wb['var'].str.extract('(\d+)$')[0]
        del wb['var']
        logging.info('Intervention effects on wellbeing')
        dtww = lfstfm(wb, 0.035, ctp=rtp)
        
        logging.info('Intervention effects on wellbeing (undiscounted)')
        dtuw = lfstfm(wb, 0, ctp=rtp)
        if rtp in ("CT", "CTI"):
            #Convert values to millions    
            dtww.loc['wbvl'] = dtww.loc['wbvl'] / 1000000 
            dtuw.loc['wbvl'] = dtuw.loc['wbvl'] / 1000000   
        
        #Correctly label rows and columns
        dtww = dtww.rename(index=var_dict) 
        dtuw = dtuw.rename(index=var_dict) 
        for i in range(1, nint+1):
            dtww = dtww.rename(columns={f"int{i}": f"Scenario {i}"})
            dtuw = dtuw.rename(columns={f"int{i}": f"Scenario {i}"})
            
        #Panel 4 : program cost
        if rskfctr in ("incqnt0", "incqntim0", "incqntnc0", "linctb0"):   
            varl = ["prgcst123"]
            if rtp in ("CT", "CTI", "T"):
                dapc = diffdescuw(varl, nint, neg = True)
            if rtp in ("RT", "RTI"):
                dapc = diffdescuw(varl, nint, rec = True, neg = True)
            del dapc['mean']
            del dapc['std']
            dapc = dapc.loc[:, ~dapc.columns.str.endswith('se')]
            dapu = dapc.copy()
            logging.info(dapc)
            for i in range(1, nint + 1):
                dapc[f'int{i}'] = dapc[f'int{i}'] * (pow((1/(1+0.035)), 1)+pow((1/(1+0.035)), 2)+pow((1/(1+0.035)), 3)+pow((1/(1+0.035)), 4)+pow((1/(1+0.035)), 5))
                dapu[f'int{i}'] = dapu[f'int{i}'] * 5  
            # if rtp == "CT":
            #     for i in range(1, nint + 1):
            #         dapc[f'int{i}'] = dapc[f'int{i}'] * 700000
            #         dapu[f'int{i}'] = dapu[f'int{i}'] * 700000    
            dapc = dapc.rename(index=var_dict) 
            dapu = dapu.rename(index=var_dict) 
            for i in range(1, nint+1):
                dapc = dapc.rename(columns={f"int{i}": f"Scenario {i}"})
                dapu = dapu.rename(columns={f"int{i}": f"Scenario {i}"})
            if rtp in ("CT", "CTI"):       
                #Convert values to millions    
                dapc = dapc * 700000 / 1000000 
                dapu = dapu * 700000 / 1000000   
            
            #Adjust for 2023 £s  
            # GDP deflator 2000 - 56.6000
            # GDP deflator 2022 - 92.3699
            # GDP deflator 2023 - 98.9782
            # GDP deflator 2024 - 101.9699
            dapc = dapc * (98.9782/56.6) 
            dapu = dapu * (98.9782/56.6) 
        # Combine rows from daoc, dtwc, and dtww into a single DataFrame
        # dc = pd.concat([daoc, dtwc, dtww], keys=['Adverse Outcomes', 'Total Weighted Costs', 'Total WELLBYs'])
        if rskfctr in ("incqnt0", "incqntim0", "incqntnc0", "linctb0"): 
            dc = pd.concat([daoc, dtwc, dtww, dapc])
            dcu = pd.concat([daoc, dtuc, dtuw, dapu])
        else :   
            dc = pd.concat([daoc, dtwc, dtww])
            dcu = pd.concat([daoc, dtuc, dtuw])
    
        # Reset index to have a clean DataFrame
        # dc.reset_index(level=0, inplace=True)
        # dc.rename(columns={'level_0': 'Category'}, inplace=True)
    
        # Export the combined DataFrame to LaTeX
        # tex_path = os.path.join(lfsm, f'output/{rskfctr}/tables', f'combined_results_{rskfctr}.tex')
        # dc.to_latex(tex_path,
        #                      index=True,
        #                      formatters={"name": str.upper},
        #                      float_format="{:.0f}".format,
        #                      column_format='l' + 'c' * (len(dc.columns) - 1))
        
    
        with open(os.path.join(lfsm, f'output/{rskfctr}/tables', f'combined_results_{rskfctr}_{rtp}.tex'), 'w', encoding='utf-8') as f:
                f.write(r'\begin{tabular}{' + '\n')
                f.write(r'l' + '\n')
                f.write(r'l' + '\n')
                for _ in range(nint):
                    f.write(r'S[table-format = 4, group-separator={,}, group-minimum-digits=4, table-number-alignment = center]' + '\n')
                f.write(r'}' + '\n')
                f.write(r'\hline' + '\n')
                f.write(r'\multicolumn{2}{c}{\textbf{Outcome}}')
                for i in range(1, nint + 1):
                    f.write(r'&\multicolumn{1}{c}{\textbf{Scenario ' + str(i) + r'}}')
                f.write(r'\\' + '\n')
                f.write(r'\hline' + '\n')
                f.write(r'\multicolumn{' + str(nint + 2) + r'}{l}{}\\' + '\n')                
                if rtp in ("T", "RT"):
                    f.write(r'\multicolumn{' + str(nint + 2) + r'}{l}{\textbf{Adverse outcomes} \textit{(Percentage point reduction by age 17)}}\\' + '\n')
                else:
                    f.write(r'\multicolumn{' + str(nint + 2) + r'}{l}{\textbf{Adverse outcomes} \textit{(Number of cases prevented by age 17)}}\\' + '\n')
                for outcome in ["Poor GCSEs age 17", "Psychological distress age 17", "Obesity age 17", "Regular smoker age 17", "Poor health age 17"]:
                    f.write(r'              &    ' + outcome.replace(' age 17', '') )
                    if rtp in ("T", "RT"):
                        for i in range(1, nint + 1):
                            f.write(r' & ' + str(round(daoc.loc[outcome, f"Scenario {i}"], 2)))
                    else:    
                        for i in range(1, nint + 1):
                            f.write(r' & ' + str(round(daoc.loc[outcome, f"Scenario {i}"])))
                    f.write(r'\\' + '\n')
                f.write(r'\multicolumn{' + str(nint + 2) + r'}{l}{}\\' + '\n')
                if rtp in ("T", "RT"):
                    f.write(r'\multicolumn{' + str(nint + 2) + r'}{l}{\textbf{Public cost savings} \textit{(Cost savings between ages 3 and 17, in 2023 £s)}}\\' + '\n')
                else:
                    f.write(r'\multicolumn{' + str(nint + 2) + r'}{l}{\textbf{Public cost savings} \textit{(Cost savings between ages 3 and 17, in millions of 2023 £s)}}\\' + '\n')
                f.write(r'              &    Total savings			 ')
                for i in range(1, nint + 1):
                    f.write(r' & ' + str(round(dtwc.loc["Total", f"Scenario {i}"])))
                f.write(r'\\' + '\n')
                for source in ["Any hospitalisation", "Disability", "Conduct disorder", "Special education needs", "Persistent truancy", "Permanent exclusion"]:
                    if source == "Any hospitalisation":
                        f.write(r'By source     &    ' + source)
                    else:
                        f.write(r'              &    ' + source)
                    for i in range(1, nint + 1):
                        f.write(r' & ' + str(round(dtwc.loc[source, f"Scenario {i}"])))
                    f.write(r'\\' + '\n')
                f.write(r'\multicolumn{' + str(nint + 2) + r'}{l}{}\\' + '\n')
                f.write(r'\multicolumn{' + str(nint + 2) + r'}{l}{\textbf{Wellbeing improvement} \textit{(WELLBYs gained between ages 3 and 17)}}\\' + '\n')
                f.write(r'              &    WELLBYs					 ')
                if rtp in ("T", "RT"):
                    for i in range(1, nint + 1):
                        f.write(r' & ' + str(round(dtww.loc["WELLBYs", f"Scenario {i}"], 2)))
                else:
                    for i in range(1, nint + 1):
                        f.write(r' & ' + str(round(dtww.loc["WELLBYs", f"Scenario {i}"])))
                f.write(r'\\' + '\n')
                if rtp in ("T", "RT"):
                    f.write(r'              &    Value (in 2023 £s)  ')
                else:
                    f.write(r'              &    Value (in millions of 2023 £s)  ')
                for i in range(1, nint + 1):
                    f.write(r' & ' + str(round(dtww.loc["Value of WELLBYs", f"Scenario {i}"])))
                f.write(r'\\' + '\n')
                if rskfctr in ("incqnt0", "incqntim0", "incqntnc0", "linctb0"):
                    f.write(r'\multicolumn{' + str(nint + 2) + r'}{l}{}\\' + '\n')
                    if rtp == "T":
                        f.write(r'\multicolumn{' + str(nint + 2) + r'}{l}{\textbf{Scenario cost} \textit{(Cost to increase income to quintile minimum, in 2023 £s)}}\\' + '\n')
                    else:
                        f.write(r'\multicolumn{' + str(nint + 2) + r'}{l}{\textbf{Scenario cost} \textit{(Cost to increase income to quintile minimum, in millions of 2023 £s)}}\\' + '\n')
                    f.write(r'              &    Cost estimate				 ')
                    for i in range(1, nint + 1):
                        f.write(r' & ' + str(round(dapc.loc["prgcst123", f"Scenario {i}"])))
                    f.write(r'\\' + '\n')
                f.write(r'\bottomrule' + '\n')
                f.write(r'\end{tabular}')
          
        with open(os.path.join(lfsm, f'output/{rskfctr}/tables', f'combined_results_undis_{rskfctr}_{rtp}.tex'), 'w', encoding='utf-8') as f:
                f.write(r'\begin{tabular}{' + '\n')
                f.write(r'l' + '\n')
                f.write(r'l' + '\n')
                for _ in range(nint):
                    f.write(r'S[table-format = 4, group-separator={,}, group-minimum-digits=4, table-number-alignment = center]' + '\n')
                f.write(r'}' + '\n')
                f.write(r'\multicolumn{2}{c}{\textbf{Outcome}}')
                for i in range(1, nint + 1):
                    f.write(r'&\multicolumn{1}{c}{\textbf{Scenario ' + str(i) + r'}}')
                f.write(r'\\' + '\n')
                f.write(r'\hline' + '\n')
                f.write(r'\multicolumn{' + str(nint + 2) + r'}{l}{}\\' + '\n')
                if rtp == "T":
                    f.write(r'\multicolumn{' + str(nint + 2) + r'}{l}{\textbf{Adverse outcomes} \textit{(Percentage point reduction by age 17)}}\\' + '\n')
                else:
                    f.write(r'\multicolumn{' + str(nint + 2) + r'}{l}{\textbf{Adverse outcomes} \textit{(Number of cases prevented by age 17)}}\\' + '\n')
                for outcome in ["Poor GCSEs age 17", "Psychological distress age 17", "Obesity age 17", "Regular smoker age 17", "Poor health age 17"]:
                    f.write(r'              &    ' + outcome.replace(' age 17', '') )
                    if rtp in ("T", "RT"):
                        for i in range(1, nint + 1):
                            f.write(r' & ' + str(round(daoc.loc[outcome, f"Scenario {i}"], 2)))
                    else:    
                        for i in range(1, nint + 1):
                            f.write(r' & ' + str(round(daoc.loc[outcome, f"Scenario {i}"])))
                    f.write(r'\\' + '\n')
                f.write(r'\multicolumn{' + str(nint + 2) + r'}{l}{}\\' + '\n')
                if rtp in ("T", "RT"):
                    f.write(r'\multicolumn{' + str(nint + 2) + r'}{l}{\textbf{Public cost savings} \textit{(Cost savings between ages 3 and 17, in 2023 £s)}}\\' + '\n')
                else:
                    f.write(r'\multicolumn{' + str(nint + 2) + r'}{l}{\textbf{Public cost savings} \textit{(Cost savings between ages 3 and 17, in millions of 2023 £s)}}\\' + '\n')
                f.write(r'              &    Total savings			 ')
                for i in range(1, nint + 1):
                    f.write(r' & ' + str(round(dtuc.loc["Total", f"Scenario {i}"])))
                f.write(r'\\' + '\n')
                for source in ["Any hospitalisation", "Disability", "Conduct disorder", "Special education needs", "Persistent truancy", "Permanent exclusion"]:
                    if source == "Any hospitalisation":
                        f.write(r'By source     &    ' + source)
                    else:
                        f.write(r'              &    ' + source)
                    for i in range(1, nint + 1):
                        f.write(r' & ' + str(round(dtuc.loc[source, f"Scenario {i}"])))
                    f.write(r'\\' + '\n')
                f.write(r'\multicolumn{' + str(nint + 2) + r'}{l}{}\\' + '\n')
                f.write(r'\multicolumn{' + str(nint + 2) + r'}{l}{\textbf{Wellbeing improvement} \textit{(WELLBYs gained between ages 3 and 17)}}\\' + '\n')
                f.write(r'              &    WELLBYs					 ')
                if rtp in ("T", "RT"):
                    for i in range(1, nint + 1):
                        f.write(r' & ' + str(round(dtuw.loc["WELLBYs", f"Scenario {i}"], 2)))
                else:
                    for i in range(1, nint + 1):
                        f.write(r' & ' + str(round(dtuw.loc["WELLBYs", f"Scenario {i}"])))
                f.write(r'\\' + '\n')
                if rtp in ("T", "RT"):
                    f.write(r'              &    Value (in 2023 £s)  ')
                else:
                    f.write(r'              &    Value (in millions of 2023 £s)  ')
                for i in range(1, nint + 1):
                    f.write(r' & ' + str(round(dtuw.loc["Value of WELLBYs", f"Scenario {i}"])))
                f.write(r'\\' + '\n')
                if rskfctr in ("incqnt0", "incqntim0", "incqntnc0", "linctb0"):
                    f.write(r'\multicolumn{' + str(nint + 2) + r'}{l}{}\\' + '\n')
                    if rtp in ("T", "RT"):
                        f.write(r'\multicolumn{' + str(nint + 2) + r'}{l}{\textbf{Scenario cost} \textit{(Cost to increase income to quintile minimum, in 2023 £s)}}\\' + '\n')
                    else:
                        f.write(r'\multicolumn{' + str(nint + 2) + r'}{l}{\textbf{Scenario cost} \textit{(Cost to increase income to quintile minimum, in millions of 2023 £s)}}\\' + '\n')
                    f.write(r'              &    Cost estimate				 ')
                    for i in range(1, nint + 1):
                        f.write(r' & ' + str(round(dapu.loc["prgcst123", f"Scenario {i}"])))
                    f.write(r'\\' + '\n')
                f.write(r'\bottomrule' + '\n')
                f.write(r'\end{tabular}') 
            
        with open(os.path.join(lfsm, f'output/{rskfctr}/tables', f'wellbyvalcomp_{rskfctr}_{rtp}.tex'), 'w', encoding='utf-8') as f:
                f.write(r'\begin{tabular}{' + '\n')
                f.write(r'l' + '\n')
                f.write(r'l' + '\n')
                for _ in range(nint):
                    f.write(r'S[table-format = 4, group-separator={,}, group-minimum-digits=4, table-number-alignment = center]' + '\n')
                f.write(r'}' + '\n')
                f.write(r'\hline' + '\n')
                f.write(r'\multicolumn{2}{c}{\textbf{Outcome}}')
                for i in range(1, nint + 1):
                    f.write(r'&\multicolumn{1}{c}{\textbf{Scenario ' + str(i) + r'}}')
                f.write(r'\\' + '\n')
                f.write(r'\hline' + '\n')
                f.write(r'\multicolumn{' + str(nint + 2) + r'}{l}{}\\' + '\n')                
                f.write(r'\multicolumn{' + str(nint + 2) + r'}{l}{\textbf{Wellbeing improvement} \textit{(WELLBYs gained between ages 3 and 17)}}\\' + '\n')
                f.write(r'              &    WELLBYs					 ')
                if rtp in ("T", "RT"):
                    for i in range(1, nint + 1):
                        f.write(r' & ' + str(round(dtww.loc["WELLBYs", f"Scenario {i}"], 2)))
                else:
                    for i in range(1, nint + 1):
                        f.write(r' & ' + str(round(dtww.loc["WELLBYs", f"Scenario {i}"])))
                f.write(r'\\' + '\n')
                if rtp in ("T", "RT"):
                    f.write(r'              &    Value (in 2023 £s)  ')
                else:
                    f.write(r'              &    Value (in millions of 2023 £s)  ')
                for i in range(1, nint + 1):
                    f.write(r' & ' + str(round(dtww.loc["Value of WELLBYs", f"Scenario {i}"])))
                f.write(r'\\' + '\n')
                if rtp in ("T", "RT"):
                    f.write(r'              &  LB Value (in 2023 £s)  ')
                else:
                    f.write(r'              &  LB Value (in millions of 2023 £s)  ')
                for i in range(1, nint + 1):
                    f.write(r' & ' + str(round(dtww.loc["Value of WELLBYs", f"Scenario {i}"])))
                f.write(r'\\' + '\n')
                if rtp in ("T", "RT"):
                    f.write(r'              &  UB Value (in 2023 £s)  ')
                else:
                    f.write(r'              &  UB Value (in millions of 2023 £s)  ')
                for i in range(1, nint + 1):
                    f.write(r' & ' + str(round(dtww.loc["Value of WELLBYs", f"Scenario {i}"])))
                f.write(r'\\' + '\n')
                if rtp in ("T", "RT"):
                    f.write(r'              &  SS Value (in 2023 £s)  ')
                else:
                    f.write(r'              &  SS Value (in millions of 2023 £s)  ')
                for i in range(1, nint + 1):
                    f.write(r' & ' + str(round(dtww.loc["Value of WELLBYs", f"Scenario {i}"])))
                       
                f.write(r'\\' + '\n')
                f.write(r'\bottomrule' + '\n')
                f.write(r'\end{tabular}')
            



    #Log end marker
    oetm = ( time.time()- ostm)/60
    logging.info('Log end - ' + time.strftime("%Y-%m-%d %H:%M:%S"))
    logging.info(f"Total run time - {oetm:.2f} minutes")

varl = ['aincome123','tot_cst_ttl', 'lifesat_ttl']
print(histint0.groupby(['incqnt123'])[varl].mean())
qn1 = histint1.groupby(['incqnt123'])[varl].mean()
qn2 = histint2.groupby(['incqnt123'])[varl].mean()
qn3 = histint3.groupby(['incqnt123'])[varl].mean()