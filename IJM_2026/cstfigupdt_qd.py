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
    print(desc)
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
    print(d)    
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
    print(f"Annual cost 3-17         = £ {t}")
    print(f"Total cost 3-17          = £ {ws}")
    print(f"Cohort Annual cost 3-17  = £ {ct}")
    print(f"Cohort total cost 3-17   = £ {cws}")
    

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
        print(f"Annual {col} cost 3-17         = £ {t}")
        print(f"Total {col} cost 3-17          = £ {ws}")
        print(f"Cohort {col} Annual cost 3-17  = £ {ct}")
        print(f"Cohort {col} total cost 3-17   = £ {cws}")
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
    print('Baseline stats')
    print(desc)

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
        print(f'Intervention {i} effects')
        print(desc_i)
        desc[f'int{i}'] = desc_i['mean']
        desc[f'int{i}se'] = desc_i['std']

    return desc

# Function to generate descriptive statistics of the differences by quantiles
def diffdescuwq(varl, num_interventions, neg=False, pct=False, ppt=False, rec=False):
    stats = ["mean", "std"]
    d1 = histint0[varl].groupby(histint0['simulation', 'incqnt123']).mean()
    desc = sms.stats.descriptivestats.describe(d1[varl], stats=stats, numeric=True).T
    # desc = desc.rename(index=var_dict)
    print('Baseline stats')
    print(desc)

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
        print(f'Intervention {i} effects')
        print(desc_i)
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
         print(f"Annual WELLBYs 3-17 scenario {i+1}  = {t_i}")
         print(f"Total WELLBYs 3-17 scenario {i+1}   = {ws_i}")
         print(f"Cohort total Annual WELLBYs 3-17 scenario {i+1}  = {ct_i}")
         print(f"Cohort total WELLBYs 3-17 scenario {i+1}   = {cws_i}")
         print(f"Cohort total Annual WELLBYs value scenario {i+1}  = {ctv_i}")
         print(f"Cohort total WELLBYs value scenario {i+1}   = {cwv_i}")
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

########################################################################################################################################################################################################
########################################################################################################################################################################################################
########################################################################################################################################################################################################

rskfctrs = ["incqnt"]

for rskfctr in rskfctrs:
    #history = pd.read_csv(os.path.join(lfsm,f'output/{rskfctr}/20241122-153052_history_nu100qincb.csv'))  
    history = pd.read_csv(os.path.join(lfsm,f'output/{rskfctr}/wrkcp_nu100int0.csv'))  
    
    interventions = [
        {'id': 'int0', 'name': 'qincb', 'code': "mcst['recip'] = 0\n"
                                                "mcst['nwmninc1'] = mcst['aincome1']"}]
    for intervention in interventions:
        ### Costs by source
        varl = ['hc2_cst', 'hc3_cst', 'hc4_cst', 'hc5_cst', 'hc6_cst', 'hc7_cst',
                'll2_cst', 'll3_cst', 'll4_cst', 'll5_cst', 'll6_cst', 'll7_cst',
                'cd3_cst', 'cd4_cst', 'cd5_cst', 'cd6_cst', 'cd7_cst',
                'se4_cst', 'se5_cst', 'se6_cst',
                'pe5_cst', 'pe6_cst',
                'pt5_cst', 'pt6_cst']
        desc = descsim(history, varl)
        print(f'{intervention["name"]} levels - costs')
        print(desc)
        
        ##Reshape dataframe to panel
        d = pd.DataFrame(desc['mean'].copy()).T
        d.columns = d.columns.str.rstrip('_cst')
        
        #Update special education costs
        d['se4'] = d['se4']*25500/6000
        d['se5'] = d['se5']*25500/6000
        d['se6'] = d['se6']*25500/6000
        
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
        print(f'{intervention["name"]} levels - costs')
        print(d)
        
        print(f'{intervention["name"]} levels - costs with total')
        cstsm(d, 0.035)
        print(f'{intervention["name"]} levels - costs with undiscounted total')
        cstsm(d, 0)
        
        ## Plotting means
        plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'cstsrc_{rskfctr}_{intervention["name"]}.png')  # Define your output file path here
        # lbl = ["2", "3", "4", "5", "6", "7"]
        # lbl = ["3", "5", "7", "11", "14", "17"]
        lbl = ["Age 3", "Age 5", "Age 7", "Age 11", "Age 14", "Age 17"]
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
        varl = ['hc2_cst', 'hc3_cst', 'hc4_cst', 'hc5_cst', 'hc6_cst', 'hc7_cst',
                'll2_cst', 'll3_cst', 'll4_cst', 'll5_cst', 'll6_cst', 'll7_cst',
                'cd3_cst', 'cd4_cst', 'cd5_cst', 'cd6_cst', 'cd7_cst',
                'se4_cst', 'se5_cst', 'se6_cst',
                'pe5_cst', 'pe6_cst',
                'pt5_cst', 'pt6_cst']
        desc = history.groupby(['incqnt123'])[varl].mean()
        desc = desc.T
        desc.rename(columns={1.0: 'q1', 2.0: 'q2', 3.0: 'q3', 4.0: 'q4', 5.0: 'q5'}, inplace=True)
        print(f'{intervention["name"]} costs by income quintile group')
        print(desc)
        
        for i in range(1, 6):
            desc_i = pd.DataFrame(desc[f'q{i}'].copy())
            
            d_i = desc_i.T
            d_i.columns = d_i.columns.str.rstrip('_cst')
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
            print(f'{intervention["name"]} costs for quintile {i}')
            print(d_i)
            d_i.fillna(0, inplace=True)
            d_i.fillna(0, inplace=True)
            globals()[f"d{i}"] = d_i.copy()
            
        del desc_i    
        
        d1 = d1 # type: ignore
        d2 = d2 # type: ignore
        d3 = d3 # type: ignore
        d4 = d4 # type: ignore
        d5 = d5 # type: ignore
        
        for i in range(1, 6):
            d_i = globals()[f"d{i}"]
            print(f'{intervention["name"]} costs for quintile {i} with sum')
            cstsm(d_i, 0.035)
            print(f'{intervention["name"]} costs for quintile {i} with undiscounted sum')
            cstsm(d_i, 0)
            
        del d_i    
            
        # Plotting means with error bars
        # lbl = ["2", "3", "4", "5", "6", "7"]
        # lbl = ["3", "5", "7", "11", "14", "17"]
        lbl = ["Age 3", "Age 5", "Age 7", "Age 11", "Age 14", "Age 17"]
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


nint = 4
rskfctrs = ["incqntnc", "incqntim", "incqnt"]

for rskfctr in rskfctrs:
    ##### Read in old data
    histint0 = pd.read_csv(os.path.join(lfsm,f'output/{rskfctr}/wrkcp_nu100int0.csv'))
    histint1 = pd.read_csv(os.path.join(lfsm,f'output/{rskfctr}/wrkcp_nu100int1.csv'))
    histint2 = pd.read_csv(os.path.join(lfsm,f'output/{rskfctr}/wrkcp_nu100int2.csv'))
    histint3 = pd.read_csv(os.path.join(lfsm,f'output/{rskfctr}/wrkcp_nu100int3.csv'))
    histint4 = pd.read_csv(os.path.join(lfsm,f'output/{rskfctr}/wrkcp_nu100int4.csv'))
    
    
    # Full childhood costs
    dataframes = [histint0, histint1, histint2, histint3, histint4]
    
    for df in dataframes:
        #Change values in 7th sweep with values in 6th sweep for education variables
        df['se4_cst'] = df['se4_cst']*25500/6000
        df['se5_cst'] = df['se5_cst']*25500/6000
        df['se6_cst'] = df['se6_cst']*25500/6000
    
    for df in dataframes:
        df['tot_cst7'] = df['cd7_cst'] + df['ll7_cst'] + df['hc7_cst'] + df['se6_cst'] + df['pe6_cst'] + df['pt6_cst']
        df['tot_cst6'] = df['cd6_cst'] + df['ll6_cst'] + df['hc6_cst'] + df['se6_cst'] + df['pe6_cst'] + df['pt6_cst']
        df['tot_cst5'] = df['cd5_cst'] + df['ll5_cst'] + df['hc5_cst'] + df['se5_cst'] + df['pe5_cst'] + df['pt5_cst']
        df['tot_cst4'] = df['cd4_cst'] + df['ll4_cst'] + df['hc4_cst'] + df['se4_cst']
        df['tot_cst3'] = df['cd3_cst'] + df['ll3_cst'] + df['hc3_cst']
        df['tot_cst2'] = df['ll2_cst'] + df['hc2_cst']
    
    
    varl = ['tot_cst', 'lifesat']
    for df in dataframes:
        df = aswpwtot(df, varl, 0.035, swpstn, nwvs)
    ##All Costs
    varl = [x for x in histint0.columns if x.endswith('_cst')]
    
    ##Costs by source
    varl = ['hc2_cst', 'hc3_cst', 'hc4_cst', 'hc5_cst', 'hc6_cst', 'hc7_cst',
            'll2_cst', 'll3_cst', 'll4_cst', 'll5_cst', 'll6_cst', 'll7_cst',
            'cd3_cst', 'cd4_cst', 'cd5_cst', 'cd6_cst', 'cd7_cst',
            'se4_cst', 'se5_cst', 'se6_cst',
            'pe5_cst', 'pe6_cst',
            'pt5_cst', 'pt6_cst',
            'tot_cst2', 'tot_cst3', 'tot_cst4', 'tot_cst5', 'tot_cst6', 'tot_cst7']
    varl.append('tot_cst_ttl')
    desc = diffdescuw(varl, nint)
    desc = desc.rename(index=var_dict)
    del desc['mean']
    del desc['std']
    logging.info('Intervention effects - costs by source')
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
    
    ##### Life Satisfaction
    for rtp in ["CT", "RT"]:
        varl = ["lifesat2", "lifesat3", "lifesat4", "lifesat5", "lifesat6", "lifesat7"]
        varl.append('lifesat_ttl')
        if rtp in ("CT"):
            desc = diffdescuw(varl, nint)
        elif rtp in ("RT"):
            desc = diffdescuw(varl, nint, rec = True)
        del desc['mean']
        del desc['std']
        logging.info('Intervention effects on life satisfaction')
        logging.info(desc)
    
    ###Wellbeing 
    for rtp in ["CT", "RT"]:
        varl = ["lifesat2", "lifesat3", "lifesat4", "lifesat5", "lifesat6", "lifesat7"]
        if rtp in ("CT"):
            desc = diffdescuw(varl, nint)
        elif rtp in ("RT"):
            desc = diffdescuw(varl, nint, rec = True)
        del desc['mean']
        del desc['std']
        wb = desc[['int' + str(i) for i in range(1, nint+1)]]
        wb['var'] = wb.index
        wb['sweep'] = wb['var'].str.extract('(\d+)$')[0]
        del wb['var']
        logging.info('Intervention effects on wellbeing')
        lfstfm(wb, 0.035)
    
        logging.info('Intervention effects on wellbeing (undiscounted)')
        lfstfm(wb, 0)
        
        
     for rtp in ["CT", "RT"]:
        varl = ["lifesat2", "lifesat3", "lifesat4", "lifesat5", "lifesat6", "lifesat7"]
        if rtp in ("CT"):
            desc = diffdescuw(varl, nint)
        elif rtp in ("RT"):
            desc = diffdescuw(varl, nint, rec = True)
        del desc['mean']
        del desc['std']
        logging.info('Intervention effects on life satisfaction')
        logging.info(desc)
        # logging.info(desc[['int1', 'int2', 'int3', 'int4']])
        # Plotting means
        plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'lifesatdif_{rskfctr}_{rtp}.png')
        # lbl = ["2", "3", "4", "5", "6", "7"]
        # lbl = ["3", "5", "7", "11", "14", "17"]
        lbl = ["Age 3", "Age 5", "Age 7", "Age 11", "Age 14", "Age 17"]
        yt = 'Change in life satisfaction'
        xt = 'Age'
        diffplt(desc, lbl, plt_path, nint, ya=yt)
        
    
    ##### Costs by source
    varl = ['hc2_cst', 'hc3_cst', 'hc4_cst', 'hc5_cst', 'hc6_cst', 'hc7_cst',
            'll2_cst', 'll3_cst', 'll4_cst', 'll5_cst', 'll6_cst', 'll7_cst',
            'cd3_cst', 'cd4_cst', 'cd5_cst', 'cd6_cst', 'cd7_cst',
            'se4_cst', 'se5_cst', 'se6_cst',
            'pe5_cst', 'pe6_cst',
            'pt5_cst', 'pt6_cst']
    desc = diffdescuw(varl, nint, neg = True)
    del desc['mean']
    del desc['std']
    logging.info('Intervention effects on costs')
    logging.info(desc)
    
    for i in range(1, nint+1):
        desc_i = pd.DataFrame(desc[f'int{i}'].copy())
        
        d_i = desc_i.T
        d_i.columns = d_i.columns.str.rstrip('_cst')
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
        d_ip.fillna(0, inplace=True)
        d_in.fillna(0, inplace=True)
        
        logging.info(f'Intervention {i} effects on costs - reduction')
        logging.info(d_ip)
        logging.info(f'Intervention {i} effects on costs - increase')
        logging.info(d_in)
        
        globals()[f"d{i}p"] = d_ip
        globals()[f"d{i}n"] = d_in
    
    
    for i in range(1, nint+1):
        d_i = globals()[f"d{i}"]
        dp = globals()[f"d{i}p"]
        dn = globals()[f"d{i}n"]
        logging.info(f'Intervention {i} effects on costs with sum')
        cstsm(d_i, 0.035)
        logging.info(f'Intervention {i} effects on costs with undiscounted sum')
        cstsm(d_i, 0)
        # logging.info(f'Intervention {i} effects on costs with sum - positive')
        # cstsm(dp)
        # logging.info(f'Intervention {i} effects on costs with undiscounted sum - positive')
        # cstsm(dp)
        # logging.info(f'Intervention {i} effects on costs with sum - negative')
        # cstsm(dn)
        
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
    lbl = ["Age 3", "Age 5", "Age 7", "Age 11", "Age 14", "Age 17"]
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
    plt.ylim(top=(aps.max().max() // 10 + 2) * 10, bottom=(ans.min().min() // 10) * 10)
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    plt.legend(["Hospitalisation", "Disability", "Conduct disorder", "SEN", "Truancy", "Exclusion"], loc='upper left')
    plt_path = os.path.join(lfsm, f'output/{rskfctr}/figures', f'cstsrcdif_{rskfctr}.png')  # Define your output file path here
    # os.remove(plt_path)
    plt.savefig(plt_path, bbox_inches='tight')
    plt.show()
    
    del lbl
    del plt_path
    del w
    del r
        
    for rtp in ["CT", "T", "RT"]: 
        #####Genrate final results table
        #Panel 1 - Age 17 outcomes
        varl = ["bdgcseme", "distress7", "obesity7", "smkreg7", "prfrhlth7"] 
        if rtp in ("CT", "CTI", "T"):
            daoc = diffdescuw(varl, nint, neg = True)
            # daoq = diffdescuwq(varl, nint, neg = True)
        if rtp in ("RT", "RTI"):
            daoc = diffdescuw(varl, nint, neg = True, rec = True)
            # daocq = diffdescuw(varl, nint, neg = True, rec = True)
        del daoc['mean']
        del daoc['std']
        daoc = daoc.loc[:, ~daoc.columns.str.endswith('se')] 
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
            
        #Panel 2 - Costs
        varl = ['hc2_cst', 'hc3_cst', 'hc4_cst', 'hc5_cst', 'hc6_cst', 'hc7_cst',
                'll2_cst', 'll3_cst', 'll4_cst', 'll5_cst', 'll6_cst', 'll7_cst',
                'cd3_cst', 'cd4_cst', 'cd5_cst', 'cd6_cst', 'cd7_cst',
                'se4_cst', 'se5_cst', 'se6_cst',
                'pe5_cst', 'pe6_cst',
                'pt5_cst', 'pt6_cst']
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
            d_i.columns = d_i.columns.str.rstrip('_cst')
            
            
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
        varl = ["lifesat2", "lifesat3", "lifesat4", "lifesat5", "lifesat6", "lifesat7"]
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
        if rskfctr in ("incqnt", "incqntnm", "incqntnc"):   
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
        if rskfctr in ("incqnt", "incqntnm", "incqntnc"): 
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
                for outcome in ["Poor GCSEs", "Psychological distress age 17", "Obesity age 17", "Regular smoker age 17", "Poor health age 17"]:
                    f.write(r'              &    ' + outcome.replace(' age 17', '') )
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
                for outcome in ["Poor GCSEs", "Psychological distress age 17", "Obesity age 17", "Regular smoker age 17", "Poor health age 17"]:
                    f.write(r'              &    ' + outcome.replace(' age 17', '') )
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
                
            
            
for rtp in ["CT", "T", "CTI"]: 
    #####Genrate final results table       
    #Panel 3 - Wellbeing
    varl = ["lifesat2", "lifesat3", "lifesat4", "lifesat5", "lifesat6", "lifesat7"]
    if rtp == "CTI":
        varl = ["lfstint2", "lfstint3", "lfstint4", "lfstint5", "lfstint6", "lfstint7"]
    desc = diffdescuw(varl, nint)
    del desc['mean']
    del desc['std']
    wb = desc[['int' + str(i) for i in range(1, nint+1)]]
    for i in range(1, nwvs+1):
        wb = wb.rename(index={f"lfstint{i}": f"lifesat{i}"})
    wb['var'] = wb.index
    wb['sweep'] = wb['var'].str.extract('(\d+)$')[0]
    del wb['var']
    logging.info('Intervention effects on wellbeing')
    dtww = lfstfm(wb, 0.035, cohsz, wbval.loc['wbnv', 'ttl'], ctp=rtp)
    
    logging.info('Intervention effects on wellbeing (undiscounted)')
    dtuw = lfstfm(wb, 0, cohsz, wbval.loc['wbnv', 'ttl'], ctp=rtp)
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