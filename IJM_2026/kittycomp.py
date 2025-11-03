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
import itertools
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels as sms
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

lfsm = '//lifesim2-main/IJM2026'

num_universes = 100
# rskfctr = 'linc'
# rskfctr = 'incqnt'

####################################################################################################
#################### Log all console output
####################################################################################################
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
ltm = time.strftime("%Y%m%d-%H%M%S")
logging.basicConfig(level=logging.DEBUG,
                    format='%(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename= os.path.join(lfsm, 'output/kittycomp/log', f'kittycomplog_{ltm}.log'),
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
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)
pd.set_option("display.max_colwidth", 1000)
pd.set_option("display.expand_frame_repr", True)
pd.set_option('display.width', 1000)

####################################################################################################
#################### Functions for use in rest of file
####################################################################################################
# Standard normalise
def stdnrmkc(data0, data1, var):
    """
    Standardize a column in a DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the column to be standardized.
    column_name (str): The name of the column to be standardized.

    Returns:
    pd.Series: A new Series containing the standardized values.
    """
    # Check if the column exists in the DataFrame
    if var in data0:
        # Calculate the mean and standard deviation
        mean = data0[var].mean()
        std = data0[var].std()
        
        # Apply standardization formula to the column
        std_var = (data1[var] - mean) / std
        
        return std_var
    else:
        print(f"Column '{var}' not found in the DataFrame.") # type: ignore
        return None
    

# Function to generate descriptive statistics of the differences
def diffdescuwkc(varl, num_interventions, rskfctr, pop="FP", neg=False, pct=False, ppt=False):
    stats = ["mean", "std"]
    dt0 = globals()[f'{rskfctr}0']
    if pop == "B1Q":
        dt = dt0[dt0['incqnt123'] == 1]
    elif pop == "B2Q":    
        dt = dt0[dt0['incqnt123'] <= 2]
    elif pop == "FP":     
        dt = dt0
    else : {}  
    d1 = dt[varl].groupby(dt['simulation']).mean()
    desc = sms.stats.descriptivestats.describe(d1[varl], stats=stats, numeric=True).T
    # desc = desc.rename(index=var_dict)
    # logging.info('Baseline stats')
    # logging.info(desc)

    for i in range(num_interventions, num_interventions + 1):
        histint_i = globals()[f'{rskfctr}{i}']
        if pop == "B1Q":
            dt_i = histint_i[histint_i['incqnt123'] == 1]
        elif pop == "B2Q":    
            dt_i = histint_i[histint_i['incqnt123'] <= 2]
        elif pop == "FP":     
            dt_i = histint_i
        else : {}  
        d = dt_i[varl] - dt[varl]
        d['simulation'] = dt['simulation']
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
        # logging.info(f'Intervention {i} effects')
        # logging.info(desc_i)
        desc[f'int{i}'] = desc_i['mean']
        desc[f'int{i}se'] = desc_i['std']

    return desc


########## Var dictionaries

var_dict = np.load(os.path.join(lfsm,'varlabel.npy'),allow_pickle='TRUE').item()

swpag_dict = np.load(os.path.join(lfsm,'swplabel.npy'),allow_pickle='TRUE').item()

########## Papers
##### Poor households
### Fernald et al 2008
# zcog3
# zbasnv2 
# zbasnv3 


### Milligan and Stabile 2011
# zcog3
# zbasnv3 
# znferpm4
# sdqhyper3
# sdqhyper4
# sdqhyper5 
# sdqcond3
# sdqcond4
# sdqcond5
# condis3
# condis4
# condis5  
### Dahl and Lochner 2012
# zbaswr4
# znferpm4
# zcog5 
# zcog6
### Gennetian and miller 2002
# sdqinternal3
# sdqinternal4
# sdqinternal5
# sdqinternal6
# sdqexternal3
# sdqexternal4
# sdqexternal5
# sdqexternal6
### Dearing et al. 2006
# sdqinternal2
# sdqinternal3
# sdqexternal2
# sdqexternal3

##### All households
### Blau 1999
# zbaswr4
# znferpm4
### Votruba-Drzal 2006
# zbaswr4
# znferpm4
### Zachrisson and Dearing 2015
# sdqinternal2

####################################################################################################
#################### Data setup and descriptive statistics
####################################################################################################
##### Initalise parameters
rskfctrs = ["linc", "lincim", "lincnc"]
# rskfctr = "linc"
nmod = 1
nint = 1
nuni = 100

##### Read all data
for rskfctr in rskfctrs:
    # for i in range(nint+1):
    #     globals()[f"{rskfctr}{i}"]  = pd.read_csv(os.path.join(lfsm,f'output/{rskfctr}/wrkcp_nu{nuni}int{i}.csv'))
    globals()[f"{rskfctr}0"]  = pd.read_csv(os.path.join(lfsm,f'output/{rskfctr}/wrkcp_nu{nuni}int0.csv'))
    globals()[f"{rskfctr}1"]  = pd.read_csv(os.path.join(lfsm,f'output/{rskfctr}/wrkcp_nu{nuni}int3.csv'))

# for rskfctr in rskfctrs:
#     for i in range(nint+1):
#         print([x for x in globals()[f"{rskfctr}{i}"].columns if x.startswith('sdq')])
 

for rskfctr in rskfctrs:        
    globals()[f"d{rskfctr}"] = pd.DataFrame()  # Define dlinc as an empty DataFrame or load it with appropriate data
    
    # y = [x for x in linc0.columns if x.startswith('sdq')]  
        
    
    vars = ["sdqinternal2", "sdqinternal3", "sdqinternal4", "sdqinternal5", "sdqinternal6", "sdqinternal7",
            "sdqexternal2", "sdqexternal3", "sdqexternal4", "sdqexternal5", "sdqexternal6", "sdqexternal7",
            "sdqhyper2", "sdqhyper3", "sdqhyper4", "sdqhyper5", "sdqhyper6", "sdqhyper7",
            "sdqpeer2", "sdqpeer3", "sdqpeer4", "sdqpeer5", "sdqpeer6", "sdqpeer7",
            "sdqprosoc2", "sdqprosoc3", "sdqprosoc4", "sdqprosoc5", "sdqprosoc6", "sdqprosoc7",
            "sdqconduct2", "sdqconduct3", "sdqconduct4", "sdqconduct5", "sdqconduct6", "sdqconduct7",
            "sdqemotion2", "sdqemotion3", "sdqemotion4", "sdqemotion5", "sdqemotion6", "sdqemotion7"]
    
    for var in vars:
        for i in range(nint+1):
            globals()[f"{rskfctr}{i}"][f'z{var}'] = stdnrmkc(globals()[f"{rskfctr}0"], globals()[f"{rskfctr}{i}"], var)
    
    # y = [x for x in linc0.columns if x.startswith('zsdq')]  
    
    # Create average measure for age range by paper
    ########## Papers
    
    ##### All households
    ### Blau 1999
    # PPVT 3+
    for i in range(nint+1):
        df = globals()[f"{rskfctr}{i}"]
        df['zbasnv23'] = df[['zbasnv2', 'zbasnv3']].mean(axis=1)
    ### Votruba-Drzal 2006
    # 
    ### Zachrisson and Dearing 2015
    # 
    
    ##### Poor households
    ### Fernald et al 2008
    #
    
    ### Milligan and Stabile 2011
    # SDQ Scores
    for i in range(nint+1):
        df = globals()[f"{rskfctr}{i}"]
        df['zsdqconduct35'] = df[['zsdqconduct3', 'zsdqconduct4', 'zsdqconduct5']].mean(axis=1)
        df['zsdqemotion35'] = df[['zsdqemotion3', 'zsdqemotion4', 'zsdqemotion5']].mean(axis=1)
        df['zsdqpeer35'] = df[['zsdqpeer3', 'zsdqpeer4', 'zsdqpeer5']].mean(axis=1)
        df['zsdqprosoc35'] = df[['zsdqprosoc3', 'zsdqprosoc4', 'zsdqprosoc5']].mean(axis=1)
        df['zsdqhyper35'] = df[['zsdqhyper3', 'zsdqhyper4', 'zsdqhyper5']].mean(axis=1)
         
    ### Dahl and Lochner 2012
    # 
    
    ### Gennetian and miller 2002
    # SDQ Scores
    for i in range(nint+1):
        df = globals()[f"{rskfctr}{i}"]
        df['zsdqinternal36'] = df[['zsdqinternal3', 'zsdqinternal4', 'zsdqinternal5', 'zsdqinternal6']].mean(axis=1)
        df['zsdqexternal36'] = df[['zsdqexternal3', 'zsdqexternal4', 'zsdqexternal5', 'zsdqexternal6']].mean(axis=1)
        
    ### Dearing et al. 2006
    for i in range(nint+1):
        df = globals()[f"{rskfctr}{i}"]
        df['zsdqinternal23'] = df[['zsdqinternal2', 'zsdqinternal3']].mean(axis=1)
        df['zsdqexternal23'] = df[['zsdqexternal2', 'zsdqexternal3']].mean(axis=1)
    
    
    
    ###Simulation results
    ##All full population comparison
    varl = ["zcog2", "zcog3", "zcog4", "zcog5", "zcog6", "zcog7", 
            "zbasnv2", "zbsrar2", "zbasnv3", "zbaspc3", "zbasps3", "zbaswr4", "zbaspc4", "znferpm4",
            "zsdqinternal23", "zsdqinternal36", "zsdqinternal2", "zsdqinternal3", "zsdqinternal4", "zsdqinternal5", "zsdqinternal6", "zsdqinternal7",
            "zsdqexternal36", "zsdqexternal23", "zsdqexternal2", "zsdqexternal3", "zsdqexternal4", "zsdqexternal5", "zsdqexternal6", "zsdqexternal7",
            "zsdqhyper35", "zsdqhyper2", "zsdqhyper3", "zsdqhyper4", "zsdqhyper5", "zsdqhyper6", "zsdqhyper7",
            "zsdqpeer35", "zsdqpeer2", "zsdqpeer3", "zsdqpeer4", "zsdqpeer5", "zsdqpeer6", "zsdqpeer7",
            "zsdqprosoc35", "zsdqprosoc2", "zsdqprosoc3", "zsdqprosoc4", "zsdqprosoc5", "zsdqprosoc6", "zsdqprosoc7",
            "zsdqconduct35", "zsdqconduct2", "zsdqconduct3", "zsdqconduct4", "zsdqconduct5", "zsdqconduct6", "zsdqconduct7",
            "zsdqemotion35", "zsdqemotion2", "zsdqemotion3", "zsdqemotion4", "zsdqemotion5", "zsdqemotion6", "zsdqemotion7"]
    # varl = ["zcog3", "zcog4", "zcog5", "zcog6",
    #         "zbasnv23", "zbasnv2", "zbasnv3", "zbaswr4", "znferpm4",
    #         "zinternal23", "zinternal36", "zinternal2", "zinternal3", "zinternal4", "zinternal5", "zinternal6",
    #         "zsdqexternal36", "zsdqexternal23", "zsdqexternal2", "zsdqexternal3", "zsdqexternal4", "zsdqexternal5", "zsdqexternal6",
    #         "zsdqconduct35", "zsdqconduct2", "zsdqconduct3", "zsdqconduct4", "zsdqconduct5",
    #         "zsdqemotion35", "zsdqemotion2", "zsdqemotion3", "zsdqemotion4", "zsdqemotion5"]
    varl = ["zbasnv23", "zbaswr4", "znferpm4",
            "zsdqinternal2", "zsdqinternal23", "zsdqinternal36",
            "zsdqexternal23", "zsdqexternal36",
            "zsdqhyper35",
            "zsdqpeer35",
            "zsdqprosoc35",
            "zsdqconduct35",
            "zsdqemotion35"]
    
    # varl = ["zbasnv23","zbaswr4", "znferpm4",
    #         "zinternal2"]
    desc1 = diffdescuwkc(varl, nint, rskfctr)
    desc1 = desc1.rename(index=var_dict)
    del desc1['mean']
    del desc1['std']
    for i in range(1, nint+1):
        desc1 = desc1.rename(columns={f"int{i}": f"Scenario {i}", f"int{i}se": f"Scenario {i}se"})
    logging.info('Intervention effects')
    logging.info(desc1)
    
    # tex_path = os.path.join(lfsm, f'output/{rskfctr}/tables', f'otwdif_{rskfctr}.tex')  # Define your output file path here
    # desc.to_latex(tex_path,
    #             index=True,
    #             formatters={"name": str.upper},
    #             float_format="{:.3f}".format,
    #         )
    
    
    # del desc
    # del tex_path
    
    ## Bottom fifth comparison
    # varl = ["zbasnv23", "zbaswr4", "znferpm4",
    #         "zinternal23", "zinternal36",
    #         "zsdqexternal23", "zsdqexternal36",
    #         "zsdqconduct35",
    #         "zsdqemotion35"]
    desc2 = diffdescuwkc(varl, nint, rskfctr, pop="B1Q")
    desc2 = desc2.rename(index=var_dict)
    del desc2['mean']
    del desc2['std']
    for i in range(1, nint+1):
        desc2 = desc2.rename(columns={f"int{i}": f"Scenario {i}", f"int{i}se": f"Scenario {i}se"})
    logging.info('Intervention effects')
    logging.info(desc2)
    
    ## Bottom two-fifth comparison
    desc3 = diffdescuwkc(varl, nint, rskfctr, pop="B2Q")
    desc3 = desc3.rename(index=var_dict)
    del desc3['mean']
    del desc3['std']
    for i in range(1, nint+1):
        desc3 = desc3.rename(columns={f"int{i}": f"Scenario {i}", f"int{i}se": f"Scenario {i}se"})
    logging.info('Intervention effects')
    logging.info(desc3)
    
    globals()[f"d{rskfctr}"] = pd.merge(desc1, desc2, left_index=True, right_index=True,suffixes=('_1', '_2'))
    
    #####Genrate final results table
    # Export the final results table to LaTeX
    with open(os.path.join(lfsm, f'output/kittycomp/tables/kittycomp_{rskfctr}.tex'), 'w', encoding='utf-8') as f:
        f.write(r'\begin{tabular}{lllcllc}' + '\n')
        f.write(r'\hline' + '\n')
        f.write(r'  \multicolumn{1}{c}{\textbf{}}&\multicolumn{3}{c}{\textbf{\cite{cooper2021does}}}&\multicolumn{3}{c}{\textbf{LifeSim Childhood}}\\' + '\n')
        f.write(r' \cmidrule(lr){2-4} \cmidrule(lr){5-7}' + '\n')
        f.write(r' \multicolumn{1}{c}{\textbf{Paper}}&\multicolumn{1}{c}{\textbf{Outcome}}&\multicolumn{1}{c}{\textbf{Age}}&\multicolumn{1}{c}{\textbf{Effect}}&\multicolumn{1}{c}{\textbf{Outcome}}&\multicolumn{1}{c}{\textbf{Age}}&\multicolumn{1}{c}{\textbf{Effect}}\\ ' + '\n')
        f.write(r' \hline' + '\n')
        f.write(r'\multicolumn{7}{l}{}\\' + '\n')
        f.write(r'\multicolumn{7}{l}{\textbf{\cite{blau1999effect}} \textit{(All families in US)}}\\' + '\n')
        f.write(r'&    PIAT Maths			& 5+	& 0.01 & NFER Progress in maths	& 7 	& ' + str(round(desc1.loc["NFER Progress in Maths age 7", "Scenario 1"], 2)) + r'	\\' + '\n')
        f.write(r'&    PIAT Reading 		& 5+	& 0.01 & BAS Word reading		& 7 	& ' + str(round(desc1.loc["BAS Word Reading age 7", "Scenario 1"], 2)) + r'	\\' + '\n')
        f.write(r'&    PPVT 				& 3+	& 0.01 & BAS Naming vocabulary	& 3-5 	& ' + str(round(desc1.loc["BAS Naming vocabulary age 3-5", "Scenario 1"], 2)) + r'	\\' + '\n')
        f.write(r'\multicolumn{7}{l}{}\\' + '\n')
        f.write(r'\multicolumn{7}{l}{\textbf{\cite{votruba2006economic}} \textit{(All families in US)}}\\' + '\n')
        f.write(r'&    PIAT Maths 		    & 5+	& 0.02 & NFER Progress in maths	& 7 	& ' + str(round(desc1.loc["NFER Progress in Maths age 7", "Scenario 1"], 2)) + r'	\\' + '\n')
        f.write(r'&    PIAT Reading 		& 5+	& 0.02 & BAS Word reading		& 7 	& ' + str(round(desc1.loc["BAS Word Reading age 7", "Scenario 1"], 2)) + r'	\\' + '\n')
        f.write(r'\multicolumn{7}{l}{}\\' + '\n')
        f.write(r'\multicolumn{7}{l}{\textbf{\cite{fernald2008role}} \textit{(Poor households in Mexico)}}\\' + '\n')
        f.write(r'&    PPVT				    & 4-6	& 0.21 & BAS Naming vocabulary	& 3-5 	& ' + str(round(desc2.loc["BAS Naming vocabulary age 3-5", "Scenario 1"], 2)) + r'  \\' + '\n')
        f.write(r'\multicolumn{7}{l}{}\\' + '\n')
        f.write(r'\multicolumn{7}{l}{\textbf{\cite{milligan2011child}} \textit{(Low education households in Canada)}}\\' + '\n')
        f.write(r'&    PIAT Maths			& 6-10	& 0.07 & NFER Progress in maths	& 7 	& ' + str(round(desc2.loc["NFER Progress in Maths age 7", "Scenario 1"], 2)) + r'  \\' + '\n')
        f.write(r'&    Hyperactivity 		& 4-10	& 0.07 & SDQ Hyperactivity		& 5-11 	& ' + str(round(desc2.loc["SDQ Hyperactivity age 5-11", "Scenario 1"], 2)) + r'	\\' + '\n')
        f.write(r'&    Conduct disorder 	& 4-10	& 0.10 & SDQ Conduct problems 	& 5-11 	& ' + str(round(-desc2.loc["SDQ Conduct age 5-11", "Scenario 1"], 2)) + r'	\\' + '\n')
        f.write(r'\multicolumn{7}{l}{}\\' + '\n')
        f.write(r'\multicolumn{7}{l}{\textbf{\cite{dahl2012impact}} \textit{(Poor families in US)}}\\' + '\n')
        f.write(r'&    Maths 				& 8-14	& 0.21 & NFER Progress in maths	& 7 	& ' + str(round(desc2.loc["NFER Progress in Maths age 7", "Scenario 1"], 2)) + r'  \\' + '\n')
        f.write(r'&    Reading 			    & 8-14	& 0.21 & BAS Word reading		& 7 	& ' + str(round(desc2.loc["BAS Word Reading age 7", "Scenario 1"], 2)) + r'  \\' + '\n')
        f.write(r'\multicolumn{7}{l}{}\\' + '\n')
        f.write(r'\multicolumn{7}{l}{\textbf{\cite{gennetian2002children}} \textit{(Poor families in Minnesota)}}\\' + '\n')
        f.write(r'&    BPI Internalising 	& 5-13	& 0.12 & SDQ Internalising		& 5-14 	& ' + str(round(-desc2.loc["SDQ Internalising age 5-14", "Scenario 1"], 2)) + r'  \\' + '\n')
        f.write(r'&    BPI Externalising 	& 5-13	& 0.11 & SDQ Externalising		& 5-14 	& ' + str(round(-desc2.loc["SDQ Externalising age 5-14", "Scenario 1"], 2)) + r'  \\' + '\n')
        f.write(r'\multicolumn{7}{l}{}\\' + '\n')
        f.write(r'\multicolumn{7}{l}{\textbf{\cite{dearing2006within}} \textit{(Poor households in US)}}\\' + '\n')
        f.write(r'&    CBCL Internalising   & 2-5	& 0.02 & SDQ Internalising		& 3-5 	& ' + str(round(-desc2.loc["SDQ Internalising age 3-5", "Scenario 1"], 2)) + r'	\\' + '\n')
        f.write(r'&    CBCL Externalising   & 2-5	& 0.03 & SDQ Externalising		& 3-5 	& ' + str(round(-desc2.loc["SDQ Externalising age 3-5", "Scenario 1"], 2)) + r'	\\' + '\n')
        f.write(r'\multicolumn{7}{l}{}\\' + '\n')
        f.write(r'\multicolumn{7}{l}{\textbf{\cite{zachrisson2015family}} \textit{(All families in Norway)}}\\' + '\n')
        f.write(r'&    CBCL Internalising	& 2-3	& 0.02 & SDQ Internalising		& 3		& ' + str(round(-desc1.loc["SDQ internalising age 3", "Scenario 1"], 2)) + r'	\\' + '\n')
        f.write(r'\bottomrule' + '\n')
        f.write(r' \end{tabular}')



with open(os.path.join(lfsm, f'output/kittycomp/tables/kittycomp_mcmp.tex'), 'w', encoding='utf-8') as f:
    f.write(r'\begin{tabular}{llclccc}' + '\n')
    f.write(r'\hline' + '\n')
    f.write(r' \multicolumn{3}{c}{\textbf{\cite{cooper2021does}}}&\multicolumn{4}{c}{\textbf{LifeSim Childhood}}\\' + '\n')
    f.write(r' \cmidrule(lr){1-3} \cmidrule(lr){4-7}' + '\n')
    f.write(r' \multicolumn{1}{c}{\textbf{Paper}}&\multicolumn{1}{c}{\textbf{Outcome}}&\multicolumn{1}{c}{\textbf{Effect}}&\multicolumn{1}{c}{\textbf{Outcome}}&\multicolumn{1}{c}{\textbf{Preferred}}&\multicolumn{1}{c}{\textbf{Extra}}&\multicolumn{1}{c}{\textbf{Unadjusted}}\\ ' + '\n')
    f.write(r' \multicolumn{1}{c}{\textbf{}}&\multicolumn{1}{c}{\textbf{}}&\multicolumn{1}{c}{\textbf{}}&\multicolumn{1}{c}{\textbf{}}&\multicolumn{1}{c}{\textbf{Model}}&\multicolumn{1}{c}{\textbf{Conservative}}&\multicolumn{1}{c}{\textbf{}}\\ ' + '\n')
    f.write(r' \hline' + '\n')
    f.write(r'\multicolumn{7}{l}{}\\' + '\n')
    f.write(r'\multicolumn{7}{l}{\textbf{\cite{blau1999effect}} \textit{(All families in US)}}\\' + '\n')
    f.write(r'&    PIAT Maths			& 0.01 & NFER Progress in maths	& ' 
            + str(round(dlinc.loc["NFER Progress in Maths age 7", "Scenario 1_1"], 2)) + r' & ' 
            + str(round(dlincim.loc["NFER Progress in Maths age 7", "Scenario 1_1"], 2)) + r' & ' 
            + str(round(dlincnc.loc["NFER Progress in Maths age 7", "Scenario 1_1"], 2)) + r'	\\' + '\n')
    f.write(r'&    PIAT Reading 		& 0.01 & BAS Word reading		& ' 
            + str(round(dlinc.loc["BAS Word Reading age 7", "Scenario 1_1"], 2))  + r' & ' 
            + str(round(dlincim.loc["BAS Word Reading age 7", "Scenario 1_1"], 2)) + r' & ' 
            + str(round(dlincnc.loc["BAS Word Reading age 7", "Scenario 1_1"], 2)) + r'	\\' + '\n')
    f.write(r'&    PPVT 				& 0.01 & BAS Naming vocabulary	& ' 
            + str(round(dlinc.loc["BAS Naming vocabulary age 3-5", "Scenario 1_1"], 2)) + r' & ' 
            + str(round(dlincim.loc["BAS Naming vocabulary age 3-5", "Scenario 1_1"], 2)) + r' & ' 
            + str(round(dlincnc.loc["BAS Naming vocabulary age 3-5", "Scenario 1_1"], 2)) + r'	\\' + '\n')
    f.write(r'\multicolumn{7}{l}{}\\' + '\n')
    f.write(r'\multicolumn{7}{l}{\textbf{\cite{votruba2006economic}} \textit{(All families in US)}}\\' + '\n')
    f.write(r'&    PIAT Maths 		    & 0.02 & NFER Progress in maths	& ' 
            + str(round(dlinc.loc["NFER Progress in Maths age 7", "Scenario 1_1"], 2)) + r' & ' 
            + str(round(dlincim.loc["NFER Progress in Maths age 7", "Scenario 1_1"], 2)) + r' & ' 
            + str(round(dlincnc.loc["NFER Progress in Maths age 7", "Scenario 1_1"], 2)) + r'	\\' + '\n')
    f.write(r'&    PIAT Reading 		& 0.02 & BAS Word reading		& ' 
            + str(round(dlinc.loc["BAS Word Reading age 7", "Scenario 1_1"], 2)) + r' & ' 
            + str(round(dlincim.loc["BAS Word Reading age 7", "Scenario 1_1"], 2)) + r' & ' 
            + str(round(dlincnc.loc["BAS Word Reading age 7", "Scenario 1_1"], 2)) + r'	\\' + '\n')
    f.write(r'\multicolumn{7}{l}{}\\' + '\n')
    f.write(r'\multicolumn{7}{l}{\textbf{\cite{fernald2008role}} \textit{(Poor households in Mexico)}}\\' + '\n')
    f.write(r'&    PPVT				    & 0.21 & BAS Naming vocabulary	& ' 
            + str(round(dlinc.loc["BAS Naming vocabulary age 3-5", "Scenario 1_1"], 2)) + r' & ' 
            + str(round(dlincim.loc["BAS Naming vocabulary age 3-5", "Scenario 1_1"], 2)) + r' & ' 
            + str(round(dlincnc.loc["BAS Naming vocabulary age 3-5", "Scenario 1_1"], 2)) + r'	\\' + '\n')
    f.write(r'\multicolumn{7}{l}{}\\' + '\n')
    f.write(r'\multicolumn{7}{l}{\textbf{\cite{milligan2011child}} \textit{(Low education households in Canada)}}\\' + '\n')
    f.write(r'&    PIAT Maths			& 0.07 & NFER Progress in maths	& ' 
            + str(round(dlinc.loc["NFER Progress in Maths age 7", "Scenario 1_2"], 2)) + r' & ' 
            + str(round(dlincim.loc["NFER Progress in Maths age 7", "Scenario 1_2"], 2)) + r' & ' 
            + str(round(dlincnc.loc["NFER Progress in Maths age 7", "Scenario 1_2"], 2)) + r'	\\' + '\n')
    f.write(r'&    Hyperactivity 		& 0.07 & SDQ Hyperactivity		& ' 
            + str(round(dlinc.loc["SDQ Hyperactivity age 5-11", "Scenario 1_2"], 2)) + r' & ' 
            + str(round(dlincim.loc["SDQ Hyperactivity age 5-11", "Scenario 1_2"], 2)) + r' & ' 
            + str(round(dlincnc.loc["SDQ Hyperactivity age 5-11", "Scenario 1_2"], 2)) + r'	\\' + '\n')
    f.write(r'&    Conduct disorder 	& 0.10 & SDQ Conduct problems 	& ' 
            + str(round(-dlinc.loc["SDQ Conduct age 5-11", "Scenario 1_2"], 2)) + r' & ' 
            + str(round(-dlincim.loc["SDQ Conduct age 5-11", "Scenario 1_2"], 2)) + r' & ' 
            + str(round(-dlincnc.loc["SDQ Conduct age 5-11", "Scenario 1_2"], 2)) + r'	\\' + '\n')
    f.write(r'\multicolumn{7}{l}{}\\' + '\n')
    f.write(r'\multicolumn{7}{l}{\textbf{\cite{dahl2012impact}} \textit{(Poor families in US)}}\\' + '\n')
    f.write(r'&    Maths 				& 0.21 & NFER Progress in maths	& ' 
            + str(round(dlinc.loc["NFER Progress in Maths age 7", "Scenario 1_2"], 2)) + r' & ' 
            + str(round(dlincim.loc["NFER Progress in Maths age 7", "Scenario 1_2"], 2)) + r' & ' 
            + str(round(dlincnc.loc["NFER Progress in Maths age 7", "Scenario 1_2"], 2)) + r'	\\' + '\n')
    f.write(r'&    Reading 			    & 0.21 & BAS Word reading		& ' 
            + str(round(dlinc.loc["BAS Word Reading age 7", "Scenario 1_2"], 2)) + r' & ' 
            + str(round(dlincim.loc["BAS Word Reading age 7", "Scenario 1_2"], 2)) + r' & ' 
            + str(round(dlincnc.loc["BAS Word Reading age 7", "Scenario 1_2"], 2)) + r'	\\' + '\n')
    f.write(r'\multicolumn{7}{l}{}\\' + '\n')
    f.write(r'\multicolumn{7}{l}{\textbf{\cite{gennetian2002children}} \textit{(Poor families in Minnesota)}}\\' + '\n')
    f.write(r'&    BPI Internalising 	& 0.12 & SDQ Internalising		& ' 
            + str(round(-dlinc.loc["SDQ Internalising age 5-14", "Scenario 1_2"], 2)) + r' & ' 
            + str(round(-dlincim.loc["SDQ Internalising age 5-14", "Scenario 1_2"], 2)) + r' & ' 
            + str(round(-dlincnc.loc["SDQ Internalising age 5-14", "Scenario 1_2"], 2)) + r'	\\' + '\n')
    f.write(r'&    BPI Externalising 	& 0.11 & SDQ Externalising		& ' 
            + str(round(-dlinc.loc["SDQ Externalising age 5-14", "Scenario 1_2"], 2)) + r' & ' 
            + str(round(-dlincim.loc["SDQ Externalising age 5-14", "Scenario 1_2"], 2)) + r' & ' 
            + str(round(-dlincnc.loc["SDQ Externalising age 5-14", "Scenario 1_2"], 2)) + r'	\\' + '\n')
    f.write(r'\multicolumn{7}{l}{}\\' + '\n')
    f.write(r'\multicolumn{7}{l}{\textbf{\cite{dearing2006within}} \textit{(Poor households in US)}}\\' + '\n')
    f.write(r'&    CBCL Internalising   & 0.02 & SDQ Internalising		& ' 
            + str(round(-dlinc.loc["SDQ Internalising age 3-5", "Scenario 1_2"], 2)) + r' & ' 
            + str(round(-dlincim.loc["SDQ Internalising age 3-5", "Scenario 1_2"], 2)) + r' & ' 
            + str(round(-dlincnc.loc["SDQ Internalising age 3-5", "Scenario 1_2"], 2)) + r'	\\' + '\n')
    f.write(r'&    CBCL Externalising   & 0.03 & SDQ Externalising		& ' 
            + str(round(-dlinc.loc["SDQ Externalising age 3-5", "Scenario 1_2"], 2)) + r' & ' 
            + str(round(-dlincim.loc["SDQ Externalising age 3-5", "Scenario 1_2"], 2)) + r' & ' 
            + str(round(-dlincnc.loc["SDQ Externalising age 3-5", "Scenario 1_2"], 2)) + r'	\\' + '\n')
    f.write(r'\multicolumn{7}{l}{}\\' + '\n')
    f.write(r'\multicolumn{7}{l}{\textbf{\cite{zachrisson2015family}} \textit{(All families in Norway)}}\\' + '\n')
    f.write(r'&    CBCL Internalising	& 0.02 & SDQ Internalising		& ' 
            + str(round(-dlinc.loc["SDQ internalising age 3", "Scenario 1_1"], 2)) + r' & ' 
            + str(round(-dlincim.loc["SDQ internalising age 3", "Scenario 1_2"], 2)) + r' & ' 
            + str(round(-dlincnc.loc["SDQ internalising age 3", "Scenario 1_2"], 2)) + r'	\\' + '\n')
    f.write(r'\bottomrule' + '\n')
    f.write(r' \end{tabular}')

#Log end marker
oetm = ( time.time()- ostm)/60
logging.info('Log end - ' + time.strftime("%Y-%m-%d %H:%M:%S"))
logging.info(f"Total run time - {oetm:.2f} minutes")

