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
# import forestplot as fp
# from pylatex import Document, Section, Subsection, Table, Tabular, NoEscape, Center, Command, Figure, Package
# from pylatex.utils import italic, bold, NoEscape

lfsm = '//lifesim2-main/IJM2025'



####################################################################################################
#################### Log all console output
####################################################################################################
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
ltm = time.strftime("%Y%m%d-%H%M%S")
logging.basicConfig(level=logging.DEBUG,
                    format='%(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename= os.path.join(lfsm, 'output/incqntcomp/log', 'lifesim2inccomplog_{ltm}.log'),
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
# Function to print cost tables with totals by sweep
def cstsm(desc, r, ctp="CT"):
    #Choose results format to report (Default - CT)
    # A - Annual average cost per child 
    # T - Total cost per child
    # CA - Cohort average annual cost
    # CT - Cohort total cost
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
    

    if ctp == "A":
        d['ttl'] = t
    elif ctp == "T":
        d['ttl'] = ws
    elif ctp == "CA":    
        d['ttl'] = ct
    elif ctp == "CT":     
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
        if ctp == "A":
            d[f'{col}'] = t
        elif ctp == "T":
            d[f'{col}'] = ws
        elif ctp == "CA":    
            d[f'{col}'] = ct
        elif ctp == "CT":     
            d[f'{col}'] = cws
        else : {}  
     
    d = d.iloc[[0]].drop(columns=['Sweep', 'sum'])   

    return d.T
# Function to generate descriptive statistics of the differences between models
def compdiffmd(varl, num_interventions, rskfctr = "incqnt", neg=False, pct=False, ppt=False):
    stats = ["mean", "std"]
    b_1 = globals()[f"{rskfctr}0"]
    b_2 = globals()[f"{rskfctr}im0"]
    b_3 = globals()[f"{rskfctr}nc0"]
    
    # desc = pd.DataFrame()
    # for i, b in enumerate([b1, b2, b3], start=1):
    #     d = b[varl].groupby(b['simulation']).mean()
    #     ds = sms.stats.descriptivestats.describe(d[varl], stats=stats, numeric=True).T
    #     desc = pd.concat([desc, ds], axis=1)
        
    for i in range(1, 4):
        d_i = locals()[f"b_{i}"][varl].groupby(locals()[f"b_{i}"]['simulation']).mean()
        locals()[f"ds_{i}"] = sms.stats.descriptivestats.describe(d_i, stats=stats, numeric=True).T
    
    desc = locals()["ds_1"]
    
    # d1 = b1[varl].groupby(b1['simulation']).mean()
    # ds1 = sms.stats.descriptivestats.describe(d1[varl], stats=stats, numeric=True).T
    # d2 = b2[varl].groupby(b2['simulation']).mean()
    # ds2 = sms.stats.descriptivestats.describe(d2[varl], stats=stats, numeric=True).T
    # d3 = b3[varl].groupby(b3['simulation']).mean()
    # ds3 = sms.stats.descriptivestats.describe(d3[varl], stats=stats, numeric=True).T
    # desc = desc.rename(index=var_dict)
    # logging.info('Baseline stats')
    # logging.info(ds1)
    # logging.info(ds2)
    # logging.info(ds3)

    for i in range(1, num_interventions + 1):
        int_1_i = globals()[f"{rskfctr}{i}"]
        int_2_i = globals()[f"{rskfctr}im{i}"]
        int_3_i = globals()[f"{rskfctr}nc{i}"]
        
        for j in range(1, 4):
            df_j = locals()[f"int_{j}_i"][varl] - locals()[f"b_{j}"][varl]
            df_j['simulation'] = locals()[f"b_{j}"]['simulation']
            dfc_j = df_j.groupby(df_j['simulation']).mean()
            dfds_j_i = sms.stats.descriptivestats.describe(dfc_j, stats=stats, numeric=True).T
        
        # df1 = int1_i[varl] - b1[varl]
        # df1['simulation'] = b1['simulation']
        # dfc1 = df1[varl].groupby(df1['simulation']).mean()
        # dfds1_i = sms.stats.descriptivestats.describe(dfc1, stats=stats, numeric=True).T
            if neg:         #If negative outcomes
                dfds_j_i['mean'] = -dfds_j_i['mean']
            if ppt:         #If percentage point
                dfds_j_i['mean'] = dfds_j_i['mean']*100
                dfds_j_i['std'] = dfds_j_i['std']*100
            if pct:         #If percent of baseline
                dfds_j_i['mean'] = dfds_j_i['mean']*100/desc['mean']
                dfds_j_i['std'] = dfds_j_i['std']*100/desc['mean']
            locals()[f"dfds_{j}_{i}"] = dfds_j_i.rename(index=var_dict)    
            logging.info(f'Intervention {i} effects')
            logging.info(dfds_j_i)
        
            if num_interventions == 1:
                desc[f'mod{j}'] = dfds_j_i['mean']
                desc[f'mod{j}se'] = dfds_j_i['std']
            elif num_interventions > 1:
                desc[f'mod{j}int{i}'] = dfds_j_i['mean']
                desc[f'mod{j}int{i}se'] = dfds_j_i['std']
        # desc[f'intim{i}'] = dfds_2_i['mean']
        # desc[f'intim{i}se'] = dfds_2_i['std']
        # desc[f'intnc{i}'] = dfds_3_i['mean']
        # desc[f'intnc{i}se'] = dfds_3_i['std']

    return desc

# Function to print lifesatisfaction tables with totals by sweep       
def lfstfmmd(desc, r):
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
         weighted_sums = d.apply(lambda row: row[f'mod{i+1}'] * weights[row['sweep']], axis=1)
         ws_i = weighted_sums.sum()
         # total_weight = sum(weights.values())
         t_i = weighted_sums.sum() / n
         ct_i = t_i*700000
         cws_i = ws_i*700000
         ctv_i = ct_i*13000
         cwv_i = cws_i*13000
         logging.info(f"Annual WELLBYs 3-17 scenario {i+1}  = {t_i}")
         logging.info(f"Total WELLBYs 3-17 scenario {i+1}   = {ws_i}")
         logging.info(f"Cohort total Annual WELLBYs 3-17 scenario {i+1}  = {ct_i}")
         logging.info(f"Cohort total WELLBYs 3-17 scenario {i+1}   = {cws_i}")
         logging.info(f"Cohort total Annual WELLBYs value scenario {i+1}  = {ctv_i}")
         logging.info(f"Cohort total WELLBYs value scenario {i+1}   = {cwv_i}")
         d1.at['wlby', f'mod{i+1}'] = cws_i
         d1.at['wbvl', f'mod{i+1}'] = cwv_i
     
     return d1

# Function to generate bar plots for intervention comparison
def diffpltmd(desc, lbl, plt_path, nint, nmod, ya=None, xa=None):
    r = np.arange(len(lbl))
    w = 0.2
    plt.figure(figsize=(15, 9))
    plt.rcParams.update({'font.size': 18})  # Set the font size to 14
    # colors = ['#DDCC77', '#117733', '#882255', '#332288']  # Define the colors for each intervention
    colors = ['#dadae4', '#a8a8af', '#585869', '#28282b']  # Define the colors for each intervention
    plt.grid(axis='y', which='both', linestyle="--", alpha=0.5, zorder=1)
    for i in range(nmod):
        plt.bar(r + w * i, desc[f'mod{i+1}'], 
            yerr=desc[f'mod{i+1}se'], 
            capsize=5, color=colors[i],
            width=w, edgecolor='black', zorder=2,
            label=['Baseline', 'Additional confounders', 'Pure correlation'][i])
    if nmod > 1: 
        plt.xticks(r + w * (nmod-1) / 2, lbl)
        plt.legend(loc='upper right')
    if xa:
        plt.xlabel(xa)
    if ya:
        plt.ylabel(ya)
    # os.remove(plt_path)
    plt.savefig(plt_path, bbox_inches='tight')
    plt.show()


########## Load variable dictionaries
var_dict = np.load(os.path.join(lfsm,'varlabel.npy'),allow_pickle='TRUE').item()

swpag_dict = np.load(os.path.join(lfsm,'swplabel.npy'),allow_pickle='TRUE').item()

mdl_dict = np.load(os.path.join(lfsm,'mdllabel.npy'),allow_pickle='TRUE').item()


#####################################################
#Log begining marker
logging.info('Log start - ' + time.strftime("%Y-%m-%d %H:%M:%S"))
#Mark code start time 
ostm = time.time()


####################################################################################################
#################### Data setup and descriptive statistics
####################################################################################################
##### Initalise parameters
rskfctrs = ["incqnt"]
rskfctr = "incqnt"
nmod = 3
nint = 1
nuni = 100

##### Read all data
for rskfctr in rskfctrs:
    for i in range(nint+1):
        globals()[f"{rskfctr}{i}"]  = pd.read_csv(os.path.join(lfsm,f'output/{rskfctr}/wrkcp_nu{nuni}int{i}.csv'))
        globals()[f"{rskfctr}im{i}"]  = pd.read_csv(os.path.join(lfsm,f'output/{rskfctr}im/wrkcp_nu{nuni}int{i}.csv'))
        globals()[f"{rskfctr}nc{i}"]  = pd.read_csv(os.path.join(lfsm,f'output/{rskfctr}nc/wrkcp_nu{nuni}int{i}.csv'))


# ### Read all data
# incqntb = pd.read_csv(os.path.join(lfsm,'output/incqnt/wrkcp_nu100qincb.csv'))
# incqnt1 = pd.read_csv(os.path.join(lfsm,'output/incqnt/wrkcp_nu100qinc1.csv'))
# # incqnt2 = pd.read_csv(os.path.join(lfsm,'output/incqnt/history_nu1000qinc2.dta'))
# # incqnt3 = pd.read_csv(os.path.join(lfsm,'output/incqnt/history_nu1000qinc3.dta'))
# # incqnt4 = pd.read_csv(os.path.join(lfsm,'output/incqnt/history_nu1000qinc4.dta'))
# incqntimb = pd.read_csv(os.path.join(lfsm,'output/incqntim/wrkcp_nu100qincb.csv'))
# incqntim1 = pd.read_csv(os.path.join(lfsm,'output/incqntim/wrkcp_nu100qinc1.csv'))
# # incqntim2 = pd.read_csv(os.path.join(lfsm,'output/incqntim/history_nu1000qinc2.dta'))
# # incqntim3 = pd.read_csv(os.path.join(lfsm,'output/incqntim/history_nu1000qinc3.dta'))
# # incqntim4 = pd.read_csv(os.path.join(lfsm,'output/incqntim/history_nu1000qinc4.dta'))
# incqntncb = pd.read_csv(os.path.join(lfsm,'output/incqntnc/wrkcp_nu100qincb.csv'))
# incqntnc1 = pd.read_csv(os.path.join(lfsm,'output/incqntnc/wrkcp_nu100qinc1.csv'))
# # incqntnc2 = pd.read_csv(os.path.join(lfsm,'output/incqntnc/history_nu1000qinc2.dta'))
# # incqntnc3 = pd.read_csv(os.path.join(lfsm,'output/incqntnc/history_nu1000qinc3.dta'))
# # incqntnc4 = pd.read_csv(os.path.join(lfsm,'output/incqntnc/history_nu1000qinc4.dta'))






###Descriptive Stats of simulation results
##All outcomes
varl = ["bdgcseme", "distress7", "obesity7", "smkreg7", "prfrhlth7",
        "lifesat2", "lifesat3", "lifesat4", "lifesat5", "lifesat6", "lifesat7", 
        # "zcog2", "zcog3", "zcog4", "zcog5", "zcog6", "zcog7",
        # "internal2", "internal3", "internal4", "internal5", "internal6", "internal7",
        # "sdqexternal2", "sdqexternal3", "sdqexternal4", "sdqexternal5", "sdqexternal6", "sdqexternal7",
        "hosp2", "hosp3", "hosp4", "hosp5", "hosp6", "hosp7",  
        "alc2", "alc3", "alc4", "alc5", "alc6", "alc7", 
        "condis3", "condis4", "condis5", "condis6", "condis7",
        "sen4", "sen5", "sen6", 
        "truancy5", "truancy6", 
        "excl5", "excl6"]
desc = compdiffmd(varl, nint)
desc = desc.rename(index=var_dict)
del desc['mean']
del desc['std']
desc = desc.rename(columns=mdl_dict)
logging.info('Intervention effects')
logging.info(desc)

tex_path = os.path.join(lfsm, f'output/{rskfctr}comp/tables', f'otwdif_{rskfctr}.tex')  # Define your output file path here
desc.to_latex(tex_path,
            index=True,
            formatters={"name": str.upper},
            float_format="{:.3f}".format,
        )


del desc
del tex_path

###Adverse outcomes
varl = ["bdgcseme7", "distress7", "obesity7", "smkreg7", "prfrhlth7"]
desc = compdiffmd(varl, nint, neg = True)
desc = desc.rename(index=var_dict)
del desc['mean']
del desc['std']
ad = desc[['mod' + str(i) for i in range(1, nint+1)]]
# if nint == 1:
#     d = desc[['int1']]
# elif nint == 2:
#     d = desc[['int1', 'int2']]
# elif nint == 3:
#     d = desc[['int1', 'int2', 'int3']]
# elif nint == 4:
#     d = desc[['int1', 'int2', 'int3', 'int4']]
for i in range(1, nint+1):
    ad[f'mod{i}'] = ad[f'mod{i}']*700000
logging.info(ad)



###Wellbeing 
##All outcomes
varl = ["lifesat2", "lifesat3", "lifesat4", "lifesat5", "lifesat6", "lifesat7"]
desc = compdiffmd(varl, nint)
del desc['mean']
del desc['std']
wb = desc[['mod' + str(i) for i in range(1, nint+1)]]
wb['var'] = wb.index
wb['sweep'] = wb['var'].str.extract('(\d+)$')[0]
del wb['var']
logging.info('Intervention effects on wellbeing')
lfstfmmd(wb, 0.035)

logging.info('Intervention effects on wellbeing (undiscounted)')
lfstfmmd(wb, 0)


##All Costs


##Costs by source
varl = ['hc_cst2', 'hc_cst3', 'hc_cst4', 'hc_cst5', 'hc_cst6', 'hc_cst7',
        'll_cst2', 'll_cst3', 'll_cst4', 'll_cst5', 'll_cst6', 'll_cst7',
        'cd_cst3', 'cd_cst4', 'cd_cst5', 'cd_cst6', 'cd_cst7',
        'se_cst4', 'se_cst5', 'se_cst6',
        'pe_cst5', 'pe_cst6',
        'pt_cst5', 'pt_cst6']

desc = compdiffmd(varl, nint)
desc = desc.rename(index=var_dict)
del desc['mean']
del desc['std']
desc = desc.rename(columns=mdl_dict)
logging.info('Intervention effects - costs by source')
logging.info(desc)

tex_path = os.path.join(lfsm, f'output/{rskfctr}comp/tables', 'cstswdif_{rskfctr}.tex')  # Define your output file path here
desc.to_latex(tex_path,
            index=True,
            formatters={"name": str.upper},
            float_format="{:.3f}".format,
        )

del desc
del tex_path


# ##Costs by sector
# varl = ['nhs2_cst', 'nhs3_cst', 'nhs4_cst', 'nhs5_cst', 'nhs6_cst', 'nhs7_cst', 
#         'ssc3_cst', 'ssc4_cst', 'ssc5_cst', 'ssc6_cst', 'ssc7_cst',
#         'ded3_cst', 'ded4_cst', 'ded5_cst', 'ded6_cst', 'ded7_cst']

# desc = diffdescuw(varl, nint)
# desc = desc.rename(index=var_dict)
# del desc['mean']
# del desc['std']
# for i in range(1, nint+1):
#     desc = desc.rename(columns={f"int{i}": f"Scenario {i}", f"int{i}se": f"Scenario {i}se"})
# logging.info('Intervention effects - costs by sector')
# logging.info(desc)

# tex_path = os.path.join(lfsm, f'output/{rskfctr}comp/tables', f'cstowdif_{rskfctr}.tex')  # Define your output file path here
# desc.to_latex(tex_path,
#             index=True,
#             formatters={"name": str.upper},
#             float_format="{:.3f}".format,
#         )


# del desc
# del tex_path


######## Plots

##### Bad Age 17 outcomes
varl = ["bdgcseme", "prfrhlth7", "distress7", "obesity7", "smkreg7"]
desc = compdiffmd(varl, nint, neg = True, ppt = True)
del desc['mean']
del desc['std']
logging.info('Intervention effects on adverse outcomes - percentage point')
logging.info(desc)
# logging.info(desc[['int1', 'int2', 'int3', 'int4']])
# Plotting bar plots for comparison
plt_path = os.path.join(lfsm, f'output/{rskfctr}comp/figures' f'inc17ocdif_{rskfctr}.png')
lbl = ["Poor GCSEs", "Poor Health", "Psychological distress", "Obesity", "Regular smoker"]
yt = 'Percentage point reduction in adverse outcome'
xt = 'Adverse outcome'
diffpltmd(desc, lbl, plt_path, nint, nmod, ya=yt)

### Number of children
desc = compdiffmd(varl, nint, neg = True)
del desc['mean']
del desc['std']
logging.info(desc)
for i in range(1, nmod + 1):
    desc[f'mod{i}'] = desc[f'mod{i}'] * 100000
    desc[f'mod{i}se'] = desc[f'mod{i}se'] * 100000
logging.info('Intervention effects on adverse outcomes - number per 100000')
logging.info(desc)
# Plotting bar plots for comparison
plt_path = os.path.join(lfsm, f'output/{rskfctr}comp/figures', f'inc17ncdif_{rskfctr}.png')
lbl = ["Poor GCSEs", "Poor Health", "Psychological distress", "Obesity", "Regular smoker"]
yt = 'Number of cases avoided per 100000'
xt = 'Adverse outcome'
diffpltmd(desc, lbl, plt_path, nint, nmod, ya=yt)
    
    
### Percentage change
desc = compdiffmd(varl, nint, neg = True, pct = True)
logging.info('Intervention effects on adverse outcomes - percent')
logging.info(desc)
# logging.info(desc[['int1', 'int2', 'int3', 'int4']])
# Plotting bar plots for comparison
plt_path = os.path.join(lfsm, f'output/{rskfctr}comp/figures', f'inc17pcdif_{rskfctr}.png')
lbl = ["Poor GCSEs", "Poor Health", "Psychological distress", "Obesity", "Regular smoker"]
yt = 'Percent reduction in adverse outcome'
xt = 'Adverse outcome'
diffpltmd(desc, lbl, plt_path, nint, nmod, ya=yt)


##### Life Satisfaction
varl = ["lifesat2", "lifesat3", "lifesat4", "lifesat5", "lifesat6", "lifesat7"]
desc = compdiffmd(varl, nint)
del desc['mean']
del desc['std']
logging.info('Intervention effects on life satisfaction')
logging.info(desc)
# logging.info(desc[['int1', 'int2', 'int3', 'int4']])
# Plotting means
plt_path = os.path.join(lfsm, f'output/{rskfctr}comp/figures', f'lifesatdif_{rskfctr}.png')
# lbl = ["2", "3", "4", "5", "6", "7"]
# lbl = ["3", "5", "7", "11", "14", "17"]
lbl = ["Age 3", "Age 5", "Age 7", "Age 11", "Age 14", "Age 17"]
yt = 'Change in life satisfaction'
xt = 'Age'
diffpltmd(desc, lbl, plt_path, nint, nmod, ya=yt)



##### Cognitive ability
varl = ["zcog2", "zcog3", "zcog4", "zcog5", "zcog6", "zcog7"]
desc = compdiffmd(varl, nint)
del desc['mean']
del desc['std']
logging.info('Intervention effects on Cognitive ability')
logging.info(desc)
# logging.info(desc[['int1', 'int2', 'int3', 'int4']])
# Plotting means
plt_path = os.path.join(lfsm, f'output/{rskfctr}comp/figures', f'zcogdif_{rskfctr}.png')
# lbl = ["2", "3", "4", "5", "6", "7"]
# lbl = ["3", "5", "7", "11", "14", "17"]
lbl = ["Age 3", "Age 5", "Age 7", "Age 11", "Age 14", "Age 17"]
yt = 'Change in cognitive ability'
xt = 'Age'
diffpltmd(desc, lbl, plt_path, nint, nmod, ya=yt)
     

##### SDQ internalising
varl = ["sdqinternal2", "sdqinternal3", "sdqinternal4", "sdqinternal5", "sdqinternal6", "sdqinternal7"]
desc = compdiffmd(varl, nint)
del desc['mean']
del desc['std']
logging.info('Intervention effects on SDQ internalising')
logging.info(desc)
# logging.info(desc[['int1', 'int2', 'int3', 'int4']])
# Plotting means
plt_path = os.path.join(lfsm, f'output/{rskfctr}comp/figures', f'intrnldif_{rskfctr}.png')
# lbl = ["2", "3", "4", "5", "6", "7"]
# lbl = ["3", "5", "7", "11", "14", "17"]
lbl = ["Age 3", "Age 5", "Age 7", "Age 11", "Age 14", "Age 17"]
yt = 'Change in SDQ internalising'
xt = 'Age'
diffpltmd(desc, lbl, plt_path, nint, nmod, ya=yt)


##### SDQ Externalising
varl = ["sdqexternal2", "sdqexternal3", "sdqexternal4", "sdqexternal5", "sdqexternal6", "sdqexternal7"]
desc = compdiffmd(varl, nint, neg = True)
del desc['mean']
del desc['std']
logging.info('Intervention effects on SDQ externalising')
logging.info(desc)
# logging.info(desc[['int1', 'int2', 'int3', 'int4']])
# Plotting means
plt_path = os.path.join(lfsm, f'output/{rskfctr}comp/figures', f'sdqextdif_{rskfctr}.png')
# lbl = ["2", "3", "4", "5", "6", "7"]
# lbl = ["3", "5", "7", "11", "14", "17"]
lbl = ["Age 3", "Age 5", "Age 7", "Age 11", "Age 14", "Age 17"]
yt = 'Change in SDQ externalising'
xt = 'Age'
diffpltmd(desc, lbl, plt_path, nint, nmod, ya=yt)


##### Binary outcomes
##### Hospitilisation
varl = ["hosp2", "hosp3", "hosp4", "hosp5", "hosp6", "hosp7"]
desc = compdiffmd(varl, nint, neg = True, ppt = True)
del desc['mean']
del desc['std']
logging.info('Intervention effects on hospitilisation')
logging.info(desc)
# Plotting means with error bars
plt_path = os.path.join(lfsm, f'output/{rskfctr}comp/figures', f'hospdif_{rskfctr}.png')
# lbl = ["2", "3", "4", "5", "6", "7"]
# lbl = ["3", "5", "7", "11", "14", "17"]
lbl = ["Age 3", "Age 5", "Age 7", "Age 11", "Age 14", "Age 17"]
yt = 'Percentage reduction in hospitilisation'
xt = 'Age'
diffpltmd(desc, lbl, plt_path, nint, nmod, ya=yt)


##### Disability
varl = ["alc2", "alc3", "alc4", "alc5", "alc6", "alc7"]
desc = compdiffmd(varl, nint, neg = True, ppt = True)
del desc['mean']
del desc['std']
logging.info('Intervention effects on disability')
logging.info(desc)
# Plotting means with error bars
plt_path = os.path.join(lfsm, f'output/{rskfctr}comp/figures', f'disdif_{rskfctr}.png')
# lbl = ["2", "3", "4", "5", "6", "7"]
# lbl = ["3", "5", "7", "11", "14", "17"]
lbl = ["Age 3", "Age 5", "Age 7", "Age 11", "Age 14", "Age 17"]
yt = 'Percentage reduction in disability'
xt = 'Age'
diffpltmd(desc, lbl, plt_path, nint, nmod, ya=yt)


##### Conduct disorder
varl = ["condis3", "condis4", "condis5", "condis6", "condis7"]
desc = compdiffmd(varl, nint, neg = True, ppt = True)
del desc['mean']
del desc['std']
logging.info('Intervention effects on conduct disorder')
logging.info(desc)
# Plotting means with error bars
plt_path = os.path.join(lfsm, f'output/{rskfctr}comp/figures', f'condisdif_{rskfctr}.png')
# lbl = ["3", "4", "5", "6", "7"]
# lbl = ["5", "7", "11", "14", "17"]
lbl = ["Age 5", "Age 7", "Age 11", "Age 14", "Age 17"]
yt = 'Percentage reduction in conduct disorder'
xt = 'Age'
diffpltmd(desc, lbl, plt_path, nint, nmod, ya=yt)


##### Education
##### SEN
varl = ["sen4", "sen5", "sen6"]
desc = compdiffmd(varl, nint, neg = True, ppt = True)
del desc['mean']
del desc['std']
logging.info('Intervention effects on special education needs')
logging.info(desc)
# Plotting means with error bars
plt_path = os.path.join(lfsm, f'output/{rskfctr}comp/figures', f'sendif_{rskfctr}.png')
# lbl = ["4", "5", "6"]
# lbl = ["7", "11", "14"]
lbl = ["Age 7", "Age 11", "Age 14"]
yt = 'Percentage reduction in SEN'
xt = 'Age'
diffpltmd(desc, lbl, plt_path, nint, nmod, ya=yt)


##### Any Truancy
varl = ["truancy5", "truancy6"]
desc = compdiffmd(varl, nint, neg = True, ppt = True)
del desc['mean']
del desc['std']
logging.info('Intervention effects on truancy')
logging.info(desc)
# Plotting means with error bars
plt_path = os.path.join(lfsm, f'output/{rskfctr}comp/figures', f'truancdif_{rskfctr}.png')
# lbl = ["5", "6"]
# lbl = ["11", "14"]
lbl = ["Age 11", "Age 14"]
yt = 'Percentage reduction in truancy'
xt = 'Age'
diffpltmd(desc, lbl, plt_path, nint, nmod, ya=yt)


##### Any Exclusion
varl = ["excl5", "excl6"]
desc = compdiffmd(varl, nint, neg = True, ppt = True)
del desc['mean']
del desc['std']
logging.info('Intervention effects on exclusion')
logging.info(desc)
# Plotting means with error bars
plt_path = os.path.join(lfsm, f'output/{rskfctr}comp/figures', f'exclusdif_{rskfctr}.png') 
# lbl = ["5", "6"]
# lbl = ["11", "14"]
lbl = ["Age 11", "Age 14"]
yt = 'Percentage reduction in exclusion'
xt = 'Age'
diffpltmd(desc, lbl, plt_path, nint, nmod, ya=yt)


##### Costs by source
varl = ['hc_cst2', 'hc_cst3', 'hc_cst4', 'hc_cst5', 'hc_cst6', 'hc_cst7',
        'll_cst2', 'll_cst3', 'll_cst4', 'll_cst5', 'll_cst6', 'll_cst7',
        'cd_cst3', 'cd_cst4', 'cd_cst5', 'cd_cst6', 'cd_cst7',
        'se_cst4', 'se_cst5', 'se_cst6',
        'pe_cst5', 'pe_cst6',
        'pt_cst5', 'pt_cst6']
desc = compdiffmd(varl, nint, neg = True)
del desc['mean']
del desc['std']
logging.info('Intervention effects on costs')
logging.info(desc)

for i in range(1, nmod+1):
    desc_i = pd.DataFrame(desc[f'mod{i}'].copy())
    
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
    logging.info(f'Intervention 1 effects on costs based on model {i}')
    logging.info(d_i)
    d_i.fillna(0, inplace=True)
    d_i.fillna(0, inplace=True)
    globals()[f"d{i}"] = d_i.copy()
    # d_i = d_i.apply(pd.to_numeric, errors='coerce')
    # globals()[f"d{i}p"] = d_i[d_i > 0].copy()
    # globals()[f"d{i}n"] = d_i[d_i < 0].copy()

for i in range(1, nmod + 1):
    d_i = globals()[f"d{i}"]
    logging.info(f'Intervention 1 effects on costs based on model {i}')
    logging.info(d_i)
    
    d_ip = d_i[d_i.apply(pd.to_numeric, errors='coerce') > 0].copy()
    d_in = d_i[d_i.apply(pd.to_numeric, errors='coerce') < 0].copy()
    d_ip.fillna(0, inplace=True)
    d_in.fillna(0, inplace=True)
    
    logging.info(f'Intervention 1 effects on costs based on model {i} - reduction')
    logging.info(d_ip)
    logging.info(f'Intervention 1 effects on costs based on model {i} - increase')
    logging.info(d_in)
    
    globals()[f"d{i}p"] = d_ip
    globals()[f"d{i}n"] = d_in


for i in range(1, nmod+1):
    d_i = globals()[f"d{i}"]
    dp = globals()[f"d{i}p"]
    dn = globals()[f"d{i}n"]
    logging.info(f'Intervention 1 effects on costs based on model {i} with sum')
    cstsm(d_i, 0.035)
    logging.info(f'Intervention 1 effects on costs based on model {i} with undiscounted sum')
    cstsm(d_i, 0)
    # logging.info(f'Intervention {i} effects on costs with sum - positive')
    # cstsm(dp)
    # logging.info(f'Intervention {i} effects on costs with undiscounted sum - positive')
    # cstsm(dp)
    # logging.info(f'Intervention {i} effects on costs with sum - negative')
    # cstsm(dn)

    
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
for i in range(nmod):
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
# plt.ylim(-30, 60)
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
plt.legend(["Hospitalisation", "Disability", "Conduct disorder", "SEN", "Truancy", "Exclusion"], loc='upper left')
plt_path = os.path.join(lfsm, f'output/{rskfctr}comp/figures', f'cstsrcdif_{rskfctr}.png')  # Define your output file path here
# os.remove(plt_path)
plt.savefig(plt_path, bbox_inches='tight')
plt.show()


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
for i in range(1, nmod+1):
    del globals()[f"d{i}"]
    del globals()[f"d{i}n"]
    del globals()[f"d{i}p"]
    
#####Genrate final results table
#Panel 1 - Age 17 outcomes
varl = ["bdgcseme", "distress7", "obesity7", "smkreg7", "prfrhlth7"] 
daoc = compdiffmd(varl, nint, neg = True)
del daoc['mean']
del daoc['std']
daoc = daoc.loc[:, ~daoc.columns.str.endswith('se')]
logging.info(daoc)
for i in range(1, nmod + 1):
    daoc[f'mod{i}'] = daoc[f'mod{i}'] * 700000
    # daoc[f'int{i}se'] = daoc[f'int{i}se'] * 700000
 
  
daoc = daoc.rename(index=var_dict) 
daoc = daoc.rename(columns=mdl_dict)
#Panel 2 - Costs
varl = ['hc_cst2', 'hc_cst3', 'hc_cst4', 'hc_cst5', 'hc_cst6', 'hc_cst7',
        'll_cst2', 'll_cst3', 'll_cst4', 'll_cst5', 'll_cst6', 'll_cst7',
        'cd_cst3', 'cd_cst4', 'cd_cst5', 'cd_cst6', 'cd_cst7',
        'se_cst4', 'se_cst5', 'se_cst6',
        'pe_cst5', 'pe_cst6',
        'pt_cst5', 'pt_cst6']
desc = compdiffmd(varl, nint, neg = True)
del desc['mean']
del desc['std']
logging.info('Intervention effects on costs')
logging.info(desc)

for i in range(1, nmod+1):
    desc_i = pd.DataFrame(desc[f'mod{i}'].copy())
    
    d_i = desc_i.T
    d_i.columns = d_i.columns.str.replace('_cst', '')
    
    #Change values in 7th sweep with values in 6th sweep for education variables
    d_i['se4'] = d_i['se4']*25500/6000
    d_i['se5'] = d_i['se5']*25500/6000
    d_i['se6'] = d_i['se6']*25500/6000
    
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
for i in range(1, nmod+1):
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
    dtwc[f'mod{i}'] = cstsm(d_i, 0.035)
    dtuc[f'mod{i}'] = cstsm(d_i, 0)
    
#Convert values to millions    
dtwc = dtwc / 1000000 
dtuc = dtuc / 1000000   
   
#Correctly label rows and columns
dtwc = dtwc.reindex(['ttl', 'hc', 'll', 'cd', 'se', 'pt', 'pe'])
dtuc = dtuc.reindex(['ttl', 'hc', 'll', 'cd', 'se', 'pt', 'pe'])
dtwc = dtwc.rename(index=var_dict) 
dtuc = dtuc.rename(index=var_dict) 
dtwc = dtwc.rename(columns=mdl_dict)
dtuc = dtuc.rename(columns=mdl_dict)
    
#Panel 3 - Wellbeing
varl = ["lifesat2", "lifesat3", "lifesat4", "lifesat5", "lifesat6", "lifesat7"]
desc = compdiffmd(varl, nint)
del desc['mean']
del desc['std']
wb = desc[['mod' + str(i) for i in range(1, nmod+1)]]
wb['var'] = wb.index
wb['sweep'] = wb['var'].str.extract('(\d+)$')[0]
del wb['var']
logging.info('Intervention effects on wellbeing')
dtww = lfstfmmd(wb, 0.035)

logging.info('Intervention effects on wellbeing (undiscounted)')
dtuw = lfstfmmd(wb, 0)

#Convert values to millions    
dtww.loc['wbvl'] = dtww.loc['wbvl'] / 1000000 
dtuw.loc['wbvl'] = dtuw.loc['wbvl'] / 1000000   
   
#Correctly label rows and columns
dtww = dtww.rename(index=var_dict) 
dtuw = dtuw.rename(index=var_dict) 
dtww = dtww.rename(columns=mdl_dict)
dtuw = dtuw.rename(columns=mdl_dict)


# Combine rows from daoc, dtwc, and dtww into a single DataFrame
# dc = pd.concat([daoc, dtwc, dtww], keys=['Adverse Outcomes', 'Total Weighted Costs', 'Total WELLBYs'])
dc = pd.concat([daoc, dtwc, dtww])
dcu = pd.concat([daoc, dtuc, dtuw])

# Reset index to have a clean DataFrame
# dc.reset_index(level=0, inplace=True)
# dc.rename(columns={'level_0': 'Category'}, inplace=True)

# Export the combined DataFrame to LaTeX
# tex_path = os.path.join(lfsm, f'output/{rskfctr}comp/tables', f'combcomp_results_{rskfctr}.tex')
# dc.to_latex(tex_path,
#                      index=True,
#                      formatters={"name": str.upper},
#                      float_format="{:.0f}".format,
#                      column_format='l' + 'c' * (len(dc.columns) - 1))  

tex_path = os.path.join(lfsm, f'output/{rskfctr}comp/tables', f'combcomp_results_undis_{rskfctr}.tex')
dcu.to_latex(tex_path,
                     index=True,
                     formatters={"name": str.upper},
                     float_format="{:.0f}".format,
                     column_format='l' + 'c' * (len(dcu.columns) - 1))    
        
with open(os.path.join(lfsm, f'output/{rskfctr}comp/tables', f'combcomp_results_{rskfctr}.tex'), 'w', encoding='utf-8') as f:
     f.write(r'\begin{tabular}{' + '\n')
     f.write(r'l' + '\n')
     f.write(r'l' + '\n')
     f.write(r'S[table-format = 4, group-separator={,}, group-minimum-digits=4, table-number-alignment = center]' + '\n')
     f.write(r'S[table-format = 4, group-separator={,}, group-minimum-digits=4, table-number-alignment = center]' + '\n')
     f.write(r'S[table-format = 4, group-separator={,}, group-minimum-digits=4, table-number-alignment = center]' + '\n')
     f.write(r'}' + '\n')
     f.write(r'\hline' + '\n')
     f.write(r'\multicolumn{2}{c}{\textbf{Outcome}}&\multicolumn{1}{c}{\textbf{Preferred}}&\multicolumn{1}{c}{\textbf{Extra}}&\multicolumn{1}{c}{\textbf{Unadjusted}}\\'     + '\n')            
     f.write(r'\multicolumn{2}{c}{\textbf{}}&\multicolumn{1}{c}{\textbf{Model}}&\multicolumn{1}{c}{\textbf{Conservative}}&\multicolumn{1}{c}{\textbf{}}\\'     + '\n')   
     f.write(r'\hline' + '\n')
     f.write(r'\multicolumn{5}{l}{}\\' + '\n')
     f.write(r'\multicolumn{5}{l}{\textbf{Adverse outcomes} \textit{(Number of cases prevented by age 17)}}\\' + '\n')
     f.write(r'              &    Poor GCSEs				& ' 
                             + str(round(daoc.loc["Poor GCSEs age 17", "Baseline"])) + r' & ' 
                             + str(round(daoc.loc["Poor GCSEs age 17", "Baseline+"])) + r' & ' 
                             + str(round(daoc.loc["Poor GCSEs age 17", "Correlation"])) + r' \\' + '\n')
     f.write(r'              &    Psychological distress 	& ' 
                             + str(round(daoc.loc["Psychological distress age 17", "Baseline"])) + r' & ' 
                             + str(round(daoc.loc["Psychological distress age 17", "Baseline+"])) + r' & ' 
                             + str(round(daoc.loc["Psychological distress age 17", "Correlation"])) + r' \\' + '\n')
     f.write(r'              &    Obesity					& ' 
                             + str(round(daoc.loc["Obesity age 17", "Baseline"])) + r' & ' 
                             + str(round(daoc.loc["Obesity age 17", "Baseline+"])) + r' & ' 
                             + str(round(daoc.loc["Obesity age 17", "Correlation"])) + r' \\' + '\n')
     f.write(r'              &    Regular smoking			& ' 
                             + str(round(daoc.loc["Regular smoker age 17", "Baseline"])) + r' & ' 
                             + str(round(daoc.loc["Regular smoker age 17", "Baseline+"])) + r' & ' 
                             + str(round(daoc.loc["Regular smoker age 17", "Correlation"])) + r' \\' + '\n')
     f.write(r'              &    Poor Health				& ' 
                             + str(round(daoc.loc["Poor health age 17", "Baseline"])) + r' & ' 
                             + str(round(daoc.loc["Poor health age 17", "Baseline+"])) + r' & ' 
                             + str(round(daoc.loc["Poor health age 17", "Correlation"])) + r' \\' + '\n')
     f.write(r'\multicolumn{5}{l}{}\\'  + '\n')       	
     f.write(r'\multicolumn{5}{l}{\textbf{Public cost savings} \textit{(Cost savings between ages 3 and 17, in millions of 2023 £s)}}\\' + '\n')
     f.write(r'              &    Total cost				& '
                             + str(round(dtwc.loc["Total", "Baseline"])) + r' & ' 
                             + str(round(dtwc.loc["Total", "Baseline+"])) + r' & ' 
                             + str(round(dtwc.loc["Total", "Correlation"])) + r' \\' + '\n')
     f.write(r'By source     &    Hospitalisation			& '
                             + str(round(dtwc.loc["Any hospitalisation", "Baseline"])) + r' & ' 
                             + str(round(dtwc.loc["Any hospitalisation", "Baseline+"])) + r' & ' 
                             + str(round(dtwc.loc["Any hospitalisation", "Correlation"])) + r' \\' + '\n')
     f.write(r'              &    Disability				& ' 
                             + str(round(dtwc.loc["Disability", "Baseline"])) + r' & ' 
                             + str(round(dtwc.loc["Disability", "Baseline+"])) + r' & ' 
                             + str(round(dtwc.loc["Disability", "Correlation"])) + r' \\'	 + '\n')
     f.write(r'              &    Conduct disorder			& '
                             + str(round(dtwc.loc["Conduct disorder", "Baseline"])) + r' & ' 
                             + str(round(dtwc.loc["Conduct disorder", "Baseline+"])) + r' & ' 
                             + str(round(dtwc.loc["Conduct disorder", "Correlation"])) + r' \\' + '\n')
     f.write(r'              &    Special education needs	& '
                             + str(round(dtwc.loc["Special education needs", "Baseline"])) + r' & ' 
                             + str(round(dtwc.loc["Special education needs", "Baseline+"])) + r' & ' 
                             + str(round(dtwc.loc["Special education needs", "Correlation"])) + r' \\' + '\n')
     f.write(r'              &    Truancy					& '
                             + str(round(dtwc.loc["Persistent truancy", "Baseline"])) + r' & ' 
                             + str(round(dtwc.loc["Persistent truancy", "Baseline+"])) + r' & ' 
                             + str(round(dtwc.loc["Persistent truancy", "Correlation"])) + r' \\' + '\n')
     f.write(r'              &    Exclusion					& '
                             + str(round(dtwc.loc["Permanent exclusion", "Baseline"])) + r' & ' 
                             + str(round(dtwc.loc["Permanent exclusion", "Baseline+"])) + r' & ' 
                             + str(round(dtwc.loc["Permanent exclusion", "Correlation"])) + r' \\' + '\n')
     f.write(r'\multicolumn{5}{l}{}\\'     + '\n')     	
     f.write(r'\multicolumn{5}{l}{\textbf{Wellbeing improvement} \textit{(WELLBYs gained between ages 3 and 17)}}\\' + '\n')
     f.write(r'              &    WELLBYs					& '
                             + str(round(dtww.loc["WELLBYs", "Baseline"])) + r' & ' 
                             + str(round(dtww.loc["WELLBYs", "Baseline+"])) + r' & ' 
                             + str(round(dtww.loc["WELLBYs", "Correlation"])) + r' \\' + '\n')
     f.write(r'              &    Value (in millions of £s) & '
                             + str(round(dtww.loc["Value of WELLBYs", "Baseline"])) + r' & ' 
                             + str(round(dtww.loc["Value of WELLBYs", "Baseline+"])) + r' & '
                             + str(round(dtww.loc["Value of WELLBYs", "Correlation"])) + r' \\' + '\n')
     f.write(r'\bottomrule' + '\n')
     f.write(r'\end{tabular}')
   

#Log end marker
oetm = ( time.time()- ostm)/60
logging.info('Log end - ' + time.strftime("%Y-%m-%d %H:%M:%S"))
logging.info(f"Total run time - {oetm:.2f} minutes")
