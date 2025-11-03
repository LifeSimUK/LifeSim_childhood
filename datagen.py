


import pandas as pd
import numpy as np
import logging
import os.path
from pathlib import Path
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels as sms
import matplotlib.pyplot as plt
from warnings import simplefilter
import seaborn as sns
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)



#Define paths
data = '//MCS' #Location of MCS data
lfsm = '//lifesim2-main' #Location of lifesim main directory


##### Functions required while processing the raw data
#Fuction to calculate family count by age categories
def hhcount(data, cage_col, page_col, rel_col, swp):

    data['cage'] = data[cage_col]
    data['page'] = data[page_col]
    data['rel'] = data[rel_col]
    
    # Check dataset name to determine 'cage' calculation
    if swp in [1, 2, 3, 4, 5]:
        data['cage'] = data['cage'] / 365
    

    
    #Relation groups we want to include in count
    # -9 - Refusal
    # -8 - Dont Know
    # -2 - Not Known
    # -1 - Not applicable
    # 1  - Husband/Wife
    # 2  - Partner/Cohabitee
    # 3  - Natural son/daughter
    # 6  - Step-son-daughter
    # 7  - Natural parent
    # 8  - Adoptive Parent
    # 9  - Foster Parent
    # 10 - Step-parent/partner of parent
    # 11 - Natural brother/Natural sister
    # 12 - Half-brother/Half-sister
    # 13 - Step-brother/Step-sister
    # 14 - Adopted brother/Adopted sister
    # 15 - Foster brother/Foster sister
    # 16 - Grandchild
    # 17 - Grandparent
    # 18 - Nanny/au pair
    # 19 - Other relative
    # 20 - Other non-relative
    # 69 - Self
    
    ### Household Counts
    hhrelgrp = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 96]
    
    # Define age categories
    acat = {
        'hhag1ch': (0, 4),
        'hhag2ch': (5, 9),
        'hhag3ch': (10, 17),
        'hhag4ch': (18, np.inf)
    }
    # Create age category columns
    for category, (lower, upper) in acat.items():
        data[category] = ((data['cage'].between(lower, upper) | data['page'].between(lower, upper))
                         & (~data['cage'].isna() | ~data['page'].isna()) 
                         & (data['rel'].isin(hhrelgrp))).astype(int)

    # Group by 'id' and calculate the family total for each age category
    ftotcol = [f'{category}{swp}' for category in acat.keys()]
    data[ftotcol] = data.groupby('mcsid')[list(acat.keys())].transform('sum')

    data[ftotcol] = data.groupby('mcsid')[list(acat.keys())].transform('sum')
    # Drop unnecessary columns
    data.drop(columns=acat.keys(), inplace=True)
    
    ### Household Counts
    
    burelgrp = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 96]
    
    # Define age categories
    acat = {
        'buag1ch': (0, 4),
        'buag2ch': (5, 9),
        'buag3ch': (10, 17),
        'buag4ch': (18, np.inf)
    }
    # Create age category columns
    for category, (lower, upper) in acat.items():
        data[category] = ((data['cage'].between(lower, upper) | data['page'].between(lower, upper))
                         & (~data['cage'].isna() | ~data['page'].isna()) 
                         & (data['rel'].isin(burelgrp))).astype(int)

    # Group by 'id' and calculate the family total for each age category
    ftotcol = [f'{category}{swp}' for category in acat.keys()]
    data[ftotcol] = data.groupby('mcsid')[list(acat.keys())].transform('sum')

    data[ftotcol] = data.groupby('mcsid')[list(acat.keys())].transform('sum')
    # Drop unnecessary columns
    data.drop(columns=['cage', 'page', 'rel'], inplace=True)
    data.drop(columns=acat.keys(), inplace=True)

    return data

#################### Create Initial Data Set ####################
########## Wave 1 ##########
######### Read Data
mcs1_ci = pd.read_stata(os.path.join(data, 'Sweep1/stata/stata13/mcs1_cm_interview.dta'), convert_categoricals=False)
mcs1_cd = pd.read_stata(os.path.join(data, 'Sweep1/stata/stata13/mcs1_cm_derived.dta'), convert_categoricals=False)
mcs1_pi = pd.read_stata(os.path.join(data, 'Sweep1/stata/stata13/mcs1_parent_interview.dta'), convert_categoricals=False)
mcs1_fd = pd.read_stata(os.path.join(data, 'Sweep1/stata/stata13/mcs1_family_derived.dta'), convert_categoricals=False)
mcs1_pd = pd.read_stata(os.path.join(data, 'Sweep1/stata/stata13/mcs1_parent_derived.dta'), convert_categoricals=False)
mcs1_gld = pd.read_stata(os.path.join(data, 'Sweep1/stata/stata13/mcs1_geographically_linked_data.dta'), convert_categoricals=False)
mcs1_ppi = pd.read_stata(os.path.join(data, 'Sweep1/stata/stata13/mcs1_proxy_partner_interview.dta'), convert_categoricals=False)
mcs1_hh = pd.read_stata(os.path.join(data, 'Sweep1/stata/stata13/mcs1_hhgrid.dta'), convert_categoricals=False)
mcs1_pci = pd.read_stata(os.path.join(data, 'Sweep1/stata/stata13/mcs1_parent_cm_interview.dta'), convert_categoricals=False)

# Change variable names to lowecase
mcs1_ci.columns= mcs1_ci.columns.str.lower()
mcs1_cd.columns= mcs1_cd.columns.str.lower()
mcs1_pi.columns= mcs1_pi.columns.str.lower()
mcs1_fd.columns= mcs1_fd.columns.str.lower()
mcs1_pd.columns= mcs1_pd.columns.str.lower()
mcs1_gld.columns= mcs1_gld.columns.str.lower()
mcs1_ppi.columns= mcs1_ppi.columns.str.lower()
mcs1_hh.columns= mcs1_hh.columns.str.lower()
mcs1_pci.columns= mcs1_pci.columns.str.lower()
###Number of unique IDs in each file
print(mcs1_pi['mcsid'].nunique())
print(mcs1_fd['mcsid'].nunique())
print(mcs1_pd['mcsid'].nunique())
print(mcs1_gld['mcsid'].nunique())
print(mcs1_ppi['mcsid'].nunique())
print(mcs1_hh['mcsid'].nunique())
print(mcs1_pci['mcsid'].nunique())
#Number of unique IDs in each country
print(mcs1_fd.iloc[:, 0:2].groupby('aactry00').agg(['nunique']).stack())
      
### Parent interview
##Main parent
# Parent types interviewed
print(mcs1_pi['aelig00'].value_counts(dropna=False))

# #Keep main parent only
# mcs1_pim = mcs1_pi[mcs1_pi['aelig00'] == 1]
#Keep main parent only if available, keep partner if main parent unavailable
# mcs1_pim = (
#     mcs1_pi
#     .sort_values(['mcsid', 'aelig00'])
#     .groupby(['mcsid'])
#     .first()
# )
mcs1_pim = mcs1_pi.sort_values(['mcsid', 'aelig00']).drop_duplicates(subset=['mcsid'], keep='first')
# Parent types remaining
print(mcs1_pim['aelig00'].value_counts(dropna=False))
##Partner
# Parent types interviewed
print(mcs1_pi['aelig00'].value_counts(dropna=False))
# Filter out only partners
mcs1_pip = mcs1_pi[mcs1_pi['aelig00'] == 2]
# Keep keep partners only if main parent available
mp = mcs1_pi[mcs1_pi['aelig00'] == 1]['mcsid'].unique()
mcs1_pip = mcs1_pip[mcs1_pip['mcsid'].isin(mp)]
# Parent types remaining
print(mcs1_pip['aelig00'].value_counts(dropna=False))
#Rename variables 
mcs1_pip = mcs1_pip.rename(columns={col: col + '_p' if col != 'mcsid' else col for col in mcs1_pip.columns})


### Parent CM interview
# # Parent types interviewed
# print(mcs1_pci['aelig00'].value_counts(dropna=False))

# Keep main parent only if available, keep partner if main parent unavailable
mcs1_pci = (
    mcs1_pci
    .sort_values(['mcsid', 'acnum00', 'aelig00'])
    .groupby(['mcsid', 'acnum00'])
    .first()
)
# Parent types remaining
print(mcs1_pci['aelig00'].value_counts(dropna=False))

### Parent derived variables
# Parent types interviewed
print(mcs1_pd['aelig00'].value_counts(dropna=False))

# # Keep main parent only if available, keep partner if main parent unavailable
# mcs1_pdm = (
#     mcs1_pd
#     .sort_values(['mcsid', 'aelig00'])
#     .groupby(['mcsid'])
#     .first()
# )
# Parent types remaining
mcs1_pdm = mcs1_pd.sort_values(['mcsid', 'aelig00']).drop_duplicates(subset=['mcsid'], keep='first')
# Parent types remaining
print(mcs1_pdm['aelig00'].value_counts(dropna=False))
##Partner
# Parent types interviewed
print(mcs1_pd['aelig00'].value_counts(dropna=False))
# Filter out only partners
mcs1_pdp = mcs1_pd[mcs1_pd['aelig00'] == 2]
# Keep keep partners only if main parent available
mp = mcs1_pd[mcs1_pd['aelig00'] == 1]['mcsid'].unique()
mcs1_pdp = mcs1_pdp[mcs1_pdp['mcsid'].isin(mp)]
# Parent types remaining
print(mcs1_pdp['aelig00'].value_counts(dropna=False))
#Rename variables 
mcs1_pdp = mcs1_pdp.rename(columns={col: col + '_p' if col != 'mcsid' else col for col in mcs1_pdp.columns})


#Household grid variables
#Drop all Aupair/nannys and other non-relatives
#mcs1_hh = mcs1_hh[mcs1_hh['rel']!=18]
#mcs1_hh = mcs1_hh[mcs1_hh['rel']!=20]
#Replace negatives values as missing (have to skip the first column as it is MCSID)
for col in mcs1_hh.columns[1:]:
    mcs1_hh[col] = mcs1_hh[col].apply(lambda x: np.nan if x < 0 else x)
#Use function defined above
mcs1_hh=hhcount(mcs1_hh, 'ahcage00', 'ahpage00', 'ahcrel00', 1)

del col

#Creating separate hh grid files for CM, main parent and partner so benefit unit characteristics can be included
# mcs1_hhmp = mcs1_hh[mcs1_hh['aelig00'] == 1]
mcs1_hhmp = pd.merge(mcs1_hh, mcs1_pim[['mcsid', 'aelig00']], on=['mcsid', 'aelig00'], how='inner')
print(mcs1_hhmp['aelig00'].value_counts(dropna=False))
mcs1_hhmp = mcs1_hhmp.drop(columns=['aresp00', 'acnum00', 'ahintm00', 'ahinty00', 'ahcsex00',
                                    'ahcdbm00', 'ahcdby00', 'ahcage00', 'ahcprs00',
                                    'hhag1ch1', 'hhag2ch1', 'hhag3ch1', 'hhag4ch1',
                                    'buag1ch1', 'buag2ch1', 'buag3ch1', 'buag4ch1'])
# mcs1_hhpp = mcs1_hh[mcs1_hh['aelig00'] == 2]
mcs1_hhpp = pd.merge(mcs1_hh, mcs1_pip[['mcsid', 'aelig00_p']], left_on=['mcsid', 'aelig00'], right_on=['mcsid', 'aelig00_p'], how='inner')
mcs1_hhpp = mcs1_hhpp.drop(columns=['aresp00', 'acnum00', 'ahintm00', 'ahinty00', 'ahcsex00',
                                    'ahcdbm00', 'ahcdby00', 'ahcage00', 'ahcprs00',
                                    'hhag1ch1', 'hhag2ch1', 'hhag3ch1', 'hhag4ch1',
                                    'buag1ch1', 'buag2ch1', 'buag3ch1', 'buag4ch1'])
mcs1_hhpp = mcs1_hhpp.rename(columns={col: col + '_p' if col != 'mcsid' else col for col in mcs1_hhpp.columns})
mcs1_hhcm = mcs1_hh[(mcs1_hh['acnum00'] == 1) | (mcs1_hh['acnum00'] == 2) | (mcs1_hh['acnum00'] == 3)]
mcs1_hhcm = mcs1_hhcm.drop(columns=['apnum00', 'aelig00', 'aresp00', 'ahpsex00',
                                    'ahpdbm00', 'ahpdby00', 'ahpres00', 'ahpage00', 'ahcrel00',
                                    'ahprela0', 'ahprelb0', 'ahprelc0', 'ahpreld0', 'ahprele0',
                                    'ahprelf0', 'ahprelg0', 'ahprelh0', 'ahpreli0', 'ahprelj0',
                                    'ahprelk0', 'ahpjob00', 'ahptpc00', 'ahaway00', 'ahinca00'])
### Combine all files in wave
#Cohort member interview and derived
mcs1 = pd.merge(mcs1_ci, mcs1_cd, on = ['mcsid', 'acnum00'])
#Main file and geographic data
mcs1 = pd.merge(mcs1, mcs1_gld, how='left', on = ['mcsid'])
mcs1.isnull().any()
#Main file and main parent interview
mcs1 = pd.merge(mcs1, mcs1_pim, how='left', on = ['mcsid'])
#Main file and partner parent interview
mcs1 = pd.merge(mcs1, mcs1_pip, how='left', on = ['mcsid'])
#Main file and main parent derived
mcs1 = pd.merge(mcs1, mcs1_pdm, how='left', on = ['mcsid', 'apnum00', 'aelig00', 'aresp00'])
#Main file and partner derived
mcs1 = pd.merge(mcs1, mcs1_pdp, how='left', on = ['mcsid', 'apnum00_p', 'aelig00_p', 'aresp00_p'])
#Main file and family derived
mcs1 = pd.merge(mcs1, mcs1_fd, how='left', on = ['mcsid'])
#Main file and parent CM interview
mcs1 = pd.merge(mcs1, mcs1_pci, how='left', on = ['mcsid', 'acnum00', 'apnum00', 'aelig00', 'aresp00'])
#Main file and household grid (CM)
mcs1 = pd.merge(mcs1, mcs1_hhcm, how='left', on = ['mcsid', 'acnum00'])
#Main file and household grid (MP)
mcs1 = pd.merge(mcs1, mcs1_hhmp, how='left', on = ['mcsid', 'apnum00', 'aelig00'])
#Main file and household grid (P)
mcs1 = pd.merge(mcs1, mcs1_hhpp, how='left', on = ['mcsid', 'apnum00_p', 'aelig00_p'])


del mcs1_cd
del mcs1_ci
del mcs1_fd
del mcs1_gld
del mcs1_pd
del mcs1_pdm
del mcs1_pdp
del mcs1_pi
del mcs1_pim
del mcs1_pip
del mp
del mcs1_ppi
del mcs1_hh
del mcs1_pci
del mcs1_hhmp
del mcs1_hhpp
del mcs1_hhcm

print(mcs1['acnum00'].value_counts(dropna=False))

mcs1['cnum'] = mcs1['acnum00']

# print([x for x in mcs1.columns if x.startswith('a')])
# print([x for x in mcs1_hhmp.columns if x.endswith('resp00')])

########## Wave 2 ##########
######### Read Data
mcs2_ca = pd.read_stata(os.path.join(data, 'Sweep2/stata/stata13/mcs2_cm_cognitive_assessment.dta'), convert_categoricals=False)
mcs2_ci = pd.read_stata(os.path.join(data, 'Sweep2/stata/stata13/mcs2_cm_interview.dta'), convert_categoricals=False)
mcs2_cd = pd.read_stata(os.path.join(data, 'Sweep2/stata/stata13/mcs2_cm_derived.dta'), convert_categoricals=False)
mcs2_pi = pd.read_stata(os.path.join(data, 'Sweep2/stata/stata13/mcs2_parent_interview.dta'), convert_categoricals=False)
mcs2_pd = pd.read_stata(os.path.join(data, 'Sweep2/stata/stata13/mcs2_parent_derived.dta'), convert_categoricals=False)
mcs2_fi = pd.read_stata(os.path.join(data, 'Sweep2/stata/stata13/mcs2_family_interview.dta'), convert_categoricals=False)
mcs2_fd = pd.read_stata(os.path.join(data, 'Sweep2/stata/stata13/mcs2_family_derived.dta'), convert_categoricals=False)
mcs2_pd = pd.read_stata(os.path.join(data, 'Sweep2/stata/stata13/mcs2_parent_derived.dta'), convert_categoricals=False)
mcs2_gld = pd.read_stata(os.path.join(data, 'Sweep2/stata/stata13/mcs2_geographically_linked_data.dta'), convert_categoricals=False)
mcs2_ppi = pd.read_stata(os.path.join(data, 'Sweep2/stata/stata13/mcs2_proxy_partner_interview.dta'), convert_categoricals=False)
mcs2_hh = pd.read_stata(os.path.join(data, 'Sweep2/stata/stata13/mcs2_hhgrid.dta'), convert_categoricals=False)
mcs2_pci = pd.read_stata(os.path.join(data, 'Sweep2/stata/stata13/mcs2_parent_cm_interview.dta'), convert_categoricals=False)


# Change variable names to lowecase
mcs2_ca.columns= mcs2_ca.columns.str.lower()
mcs2_ci.columns= mcs2_ci.columns.str.lower()
mcs2_cd.columns= mcs2_cd.columns.str.lower()
mcs2_pi.columns= mcs2_pi.columns.str.lower()
mcs2_fi.columns= mcs2_fi.columns.str.lower()
mcs2_fd.columns= mcs2_fd.columns.str.lower()
mcs2_pd.columns= mcs2_pd.columns.str.lower()
mcs2_gld.columns= mcs2_gld.columns.str.lower()
mcs2_ppi.columns= mcs2_ppi.columns.str.lower()
mcs2_hh.columns= mcs2_hh.columns.str.lower()
mcs2_pci.columns= mcs2_pci.columns.str.lower()
###Number of unique IDs in each file
print(mcs2_pi['mcsid'].nunique())
print(mcs2_fd['mcsid'].nunique())
print(mcs2_pd['mcsid'].nunique())
print(mcs2_gld['mcsid'].nunique())
print(mcs2_ppi['mcsid'].nunique())
print(mcs2_hh['mcsid'].nunique())
#Number of unique IDs in each country
print(mcs2_fd.iloc[:, 0:2].groupby('bactry00').agg(['nunique']).stack())
      


### Parent interview
# Parent types interviewed
print(mcs2_pi['belig00'].value_counts(dropna=False))
# #Keep main parent only
# mcs2_pim = mcs2_pi[mcs2_pi['belig00'] == 1]
# Keep main parent only if available, keep partner if main parent unavailable
# mcs2_pim = (
#     mcs2_pi
#     .sort_values(['mcsid', 'belig00'])
#     .groupby(['mcsid'])
#     .first()
# )
mcs2_pim = mcs2_pi.sort_values(['mcsid', 'belig00']).drop_duplicates(subset=['mcsid'], keep='first')
# Parent types remaining
print(mcs2_pim['belig00'].value_counts(dropna=False))

##Partner
# Parent types interviewed
print(mcs2_pi['belig00'].value_counts(dropna=False))
# Filter out only partners
mcs2_pip = mcs2_pi[mcs2_pi['belig00'] == 2]
# Keep keep partners only if main parent available
mp = mcs2_pi[mcs2_pi['belig00'] == 1]['mcsid'].unique()
mcs2_pip = mcs2_pip[mcs2_pip['mcsid'].isin(mp)]
# Parent types remaining
print(mcs2_pip['belig00'].value_counts(dropna=False))
#Rename variables 
mcs2_pip = mcs2_pip.rename(columns={col: col + '_p' if col != 'mcsid' else col for col in mcs2_pip.columns})

# Keep main parent only if available, keep partner if main parent unavailable
mcs2_pci = (
    mcs2_pci
    .sort_values(['mcsid', 'bcnum00', 'belig00'])
    .groupby(['mcsid', 'bcnum00'])
    .first()
)
# Parent types remaining
print(mcs2_pci['belig00'].value_counts(dropna=False))

### Parent derived variables
# Parent types interviewed
print(mcs2_pd['belig00'].value_counts(dropna=False))
# #Keep main parent only
# mcs2_pdm = mcs2_pd[mcs2_pd['belig00'] == 1]
# Keep main parent only if available, keep partner if main parent unavailable
# mcs2_pdm = (
#     mcs2_pd
#     .sort_values(['mcsid', 'belig00'])
#     .groupby(['mcsid'])
#     .first()
# )
mcs2_pdm = mcs2_pd.sort_values(['mcsid', 'belig00']).drop_duplicates(subset=['mcsid'], keep='first')
# Parent types remaining
print(mcs2_pdm['belig00'].value_counts(dropna=False))

##Partner
# Parent types interviewed
print(mcs2_pd['belig00'].value_counts(dropna=False))
# Filter out only partners
mcs2_pdp = mcs2_pd[mcs2_pd['belig00'] == 2]
# Keep keep partners only if main parent available
mp = mcs2_pd[mcs2_pd['belig00'] == 1]['mcsid'].unique()
mcs2_pdp = mcs2_pdp[mcs2_pdp['mcsid'].isin(mp)]
# Parent types remaining
print(mcs2_pdp['belig00'].value_counts(dropna=False))
#Rename variables 
mcs2_pdp = mcs2_pdp.rename(columns={col: col + '_p' if col != 'mcsid' else col for col in mcs2_pdp.columns})



#Household grid variables
#Drop all Aupair/nannys and other non-relatives
#mcs2_hh = mcs2_hh[mcs2_hh['rel']!=18]
#mcs2_hh = mcs2_hh[mcs2_hh['rel']!=20]
#Replace negatives values as missing (have to skip the first column as it is MCSID)
for col in mcs2_hh.columns[1:]:
    mcs2_hh[col] = mcs2_hh[col].apply(lambda x: np.nan if x < 0 else x)
#Use function defined above
mcs2_hh=hhcount(mcs2_hh, 'bhcage00', 'bhpage00', 'bhcrel00', 2)

del col

#Creating separate hh grid files for CM and main parent so both characteristics can be included
# mcs2_hhmp = mcs2_hh[mcs2_hh['belig00'] == 1]
mcs2_hhmp = pd.merge(mcs2_hh, mcs2_pim[['mcsid', 'belig00']], on=['mcsid', 'belig00'], how='inner')
mcs2_hhmp = mcs2_hhmp.drop(columns=['bresp00', 'bcnum00', 'bhintm00', 'bhinty00', 'bhcsex00',
                                    'bhcdbm00', 'bhcdby00', 'bhcage00', 'bhcprs00', 'bhcass00',
                                    'bhmesu00', 'hhag1ch2', 'hhag2ch2', 'hhag3ch2', 'hhag4ch2',
                                    'buag1ch2', 'buag2ch2', 'buag3ch2', 'buag4ch2'])
# mcs2_hhpp = mcs2_hh[mcs2_hh['belig00'] == 2]
mcs2_hhpp = pd.merge(mcs2_hh, mcs2_pip[['mcsid', 'belig00_p']], left_on=['mcsid', 'belig00'], right_on=['mcsid', 'belig00_p'], how='inner')
mcs2_hhpp = mcs2_hhpp.drop(columns=['bresp00', 'bcnum00', 'bhintm00', 'bhinty00', 'bhcsex00',
                                    'bhcdbm00', 'bhcdby00', 'bhcage00', 'bhcprs00', 'bhcass00',
                                    'bhmesu00', 'hhag1ch2', 'hhag2ch2', 'hhag3ch2', 'hhag4ch2',
                                    'buag1ch2', 'buag2ch2', 'buag3ch2', 'buag4ch2'])
mcs2_hhpp = mcs2_hhpp.rename(columns={col: col + '_p' if col != 'mcsid' else col for col in mcs2_hhpp.columns})
mcs2_hhcm = mcs2_hh[(mcs2_hh['bcnum00'] == 1) | (mcs2_hh['bcnum00'] == 2) | (mcs2_hh['bcnum00'] == 3)]
mcs2_hhcm = mcs2_hhcm.drop(columns=['bpnum00', 'belig00', 'bresp00', 'bhpsex00',
                                    'bhpdbm00', 'bhpdby00', 'bhpres00', 'bhpage00', 'bhcrel00',
                                    'bhprela0', 'bhprelb0', 'bhprelc0', 'bhpreld0', 'bhprele0',
                                    'bhprelf0', 'bhprelg0', 'bhprelh0', 'bhpreli0', 'bhprelj0',
                                    'bhprelk0', 'bhprell0', 'bhprelm0', 'bhpreln0', 'bhprelo0',
                                    'bhprelp0', 'bhprelq0', 'bhprelr0', 'bhpjob00', 'bhptpc00',
                                    'bhpjob00', 'bhptpc00'])
### Combine all files in wave
#Cohort member interview and derived
mcs2 = pd.merge(mcs2_ci, mcs2_cd, on = ['mcsid', 'bcnum00'])
#Main file and cognitive scores
mcs2 = pd.merge(mcs2, mcs2_ca, how='left', on = ['mcsid', 'bcnum00'])
mcs2.isnull().any()
#Main file and geographic data
mcs2 = pd.merge(mcs2, mcs2_gld, how='left', on = ['mcsid'])
mcs2.isnull().any()
#Main file and main parent interview
mcs2 = pd.merge(mcs2, mcs2_pim, how='left', on = ['mcsid'])
#Main file and partner interview
mcs2 = pd.merge(mcs2, mcs2_pip, how='left', on = ['mcsid'])
#Main file and main parent derived
mcs2 = pd.merge(mcs2, mcs2_pdm, how='left', on = ['mcsid', 'bpnum00', 'belig00', 'bresp00'])
#Main file and partner derived
mcs2 = pd.merge(mcs2, mcs2_pdp, how='left', on = ['mcsid', 'bpnum00_p', 'belig00_p', 'bresp00_p'])
#Main file and family interview
mcs2 = pd.merge(mcs2, mcs2_fi, how='left', on = ['mcsid'])
#Main file and family derived
mcs2 = pd.merge(mcs2, mcs2_fd, how='left', on = ['mcsid'])
#Main file and parent CM interview
mcs2 = pd.merge(mcs2, mcs2_pci, how='left', on = ['mcsid', 'bcnum00', 'bpnum00', 'belig00', 'bresp00'])
#Main file and household grid (CM)
mcs2 = pd.merge(mcs2, mcs2_hhcm, how='left', on = ['mcsid', 'bcnum00'])
#Main file and household grid (MP)
mcs2 = pd.merge(mcs2, mcs2_hhmp, how='left', on = ['mcsid', 'bpnum00', 'belig00'])
#Main file and household grid (P)
mcs2 = pd.merge(mcs2, mcs2_hhpp, how='left', on = ['mcsid', 'bpnum00_p', 'belig00_p'])


del mcs2_ca
del mcs2_cd
del mcs2_ci
del mcs2_fi
del mcs2_fd
del mcs2_gld
del mcs2_pd
del mcs2_pdm
del mcs2_pdp
del mcs2_pi
del mcs2_pim
del mcs2_pip
del mp
del mcs2_ppi
del mcs2_hh
del mcs2_pci
del mcs2_hhmp
del mcs2_hhpp
del mcs2_hhcm


print(mcs2['bcnum00'].value_counts(dropna=False))

mcs2['cnum'] = mcs2['bcnum00']

########## Wave 3 ##########
######### Read Data
mcs3_ca = pd.read_stata(os.path.join(data, 'Sweep3/stata/stata13/mcs3_cm_cognitive_assessment.dta'), convert_categoricals=False)
mcs3_ci = pd.read_stata(os.path.join(data, 'Sweep3/stata/stata13/mcs3_cm_interview.dta'), convert_categoricals=False)
mcs3_cd = pd.read_stata(os.path.join(data, 'Sweep3/stata/stata13/mcs3_cm_derived.dta'), convert_categoricals=False)
mcs3_pi = pd.read_stata(os.path.join(data, 'Sweep3/stata/stata13/mcs3_parent_interview.dta'), convert_categoricals=False)
mcs3_pd = pd.read_stata(os.path.join(data, 'Sweep3/stata/stata13/mcs3_parent_derived.dta'), convert_categoricals=False)
mcs3_fi = pd.read_stata(os.path.join(data, 'Sweep3/stata/stata13/mcs3_family_interview.dta'), convert_categoricals=False)
mcs3_fd = pd.read_stata(os.path.join(data, 'Sweep3/stata/stata13/mcs3_family_derived.dta'), convert_categoricals=False)
mcs3_pd = pd.read_stata(os.path.join(data, 'Sweep3/stata/stata13/mcs3_parent_derived.dta'), convert_categoricals=False)
mcs3_ts = pd.read_stata(os.path.join(data, 'Sweep3/stata/stata13/mcs3_cm_teacher_survey.dta'), convert_categoricals=False)
mcs3_gld = pd.read_stata(os.path.join(data, 'Sweep3/stata/stata13/mcs3_geographically_linked_data.dta'), convert_categoricals=False)
mcs3_ppi = pd.read_stata(os.path.join(data, 'Sweep3/stata/stata13/mcs3_proxy_partner_interview.dta'), convert_categoricals=False)
mcs3_hh = pd.read_stata(os.path.join(data, 'Sweep3/stata/stata13/mcs3_hhgrid.dta'), convert_categoricals=False)
mcs3_pci = pd.read_stata(os.path.join(data, 'Sweep3/stata/stata13/mcs3_parent_cm_interview.dta'), convert_categoricals=False)

# Change variable names to lowecase
mcs3_ca.columns= mcs3_ca.columns.str.lower()
mcs3_ci.columns= mcs3_ci.columns.str.lower()
mcs3_cd.columns= mcs3_cd.columns.str.lower()
mcs3_pi.columns= mcs3_pi.columns.str.lower()
mcs3_pd.columns= mcs3_pd.columns.str.lower()
mcs3_fi.columns= mcs3_fi.columns.str.lower()
mcs3_fd.columns= mcs3_fd.columns.str.lower()
mcs3_ts.columns= mcs3_ts.columns.str.lower()
mcs3_gld.columns= mcs3_gld.columns.str.lower()
mcs3_ppi.columns= mcs3_ppi.columns.str.lower()
mcs3_hh.columns= mcs3_hh.columns.str.lower()
mcs3_pci.columns= mcs3_pci.columns.str.lower()
###Number of unique IDs in each file
print(mcs3_pi['mcsid'].nunique())
print(mcs3_pd['mcsid'].nunique())
print(mcs3_fi['mcsid'].nunique())
print(mcs3_fd['mcsid'].nunique())
print(mcs3_gld['mcsid'].nunique())
print(mcs3_ppi['mcsid'].nunique())
print(mcs3_hh['mcsid'].nunique())
#Number of unique IDs in each country
print(mcs3_fd.iloc[:, 0:2].groupby('cactry00').agg(['nunique']).stack())

### Parent interview
# Parent types interviewed
print(mcs3_pi['celig00'].value_counts(dropna=False))
# Keep main parent only if available, keep partner if main parent unavailable
mcs3_pim = mcs3_pi.sort_values(['mcsid', 'celig00']).drop_duplicates(subset=['mcsid'], keep='first')
# Parent types remaining
print(mcs3_pim['celig00'].value_counts(dropna=False))
## Partner
# Parent types interviewed
print(mcs3_pi['celig00'].value_counts(dropna=False))
# Filter out only partners
mcs3_pip = mcs3_pi[mcs3_pi['celig00'] == 2]
# Keep partners only if main parent available
mp = mcs3_pi[mcs3_pi['celig00'] == 1]['mcsid'].unique()
mcs3_pip = mcs3_pip[mcs3_pip['mcsid'].isin(mp)]
# Parent types remaining
print(mcs3_pip['celig00'].value_counts(dropna=False))
# Rename variables
mcs3_pip = mcs3_pip.rename(columns={col: col + '_p' if col != 'mcsid' else col for col in mcs3_pip.columns})


### Parent CM interview
# Parent types interviewed
print(mcs3_pci['celig00'].value_counts(dropna=False))

# Keep main parent only if available, keep partner if main parent unavailable
mcs3_pci = (
    mcs3_pci
    .sort_values(['mcsid', 'ccnum00', 'celig00'])
    .groupby(['mcsid', 'ccnum00'])
    .first()
)
# Parent types remaining
print(mcs3_pci['celig00'].value_counts(dropna=False))

### Parent derived variables
# Parent types interviewed
print(mcs3_pd['celig00'].value_counts(dropna=False))
# Keep main parent only if available, keep partner if main parent unavailable
mcs3_pdm = mcs3_pd.sort_values(['mcsid', 'celig00']).drop_duplicates(subset=['mcsid'], keep='first')
# Parent types remaining
print(mcs3_pdm['celig00'].value_counts(dropna=False))

## Partner
# Parent types interviewed
print(mcs3_pd['celig00'].value_counts(dropna=False))
# Filter out only partners
mcs3_pdp = mcs3_pd[mcs3_pd['celig00'] == 2]
# Keep partners only if main parent available
mp = mcs3_pd[mcs3_pd['celig00'] == 1]['mcsid'].unique()
mcs3_pdp = mcs3_pdp[mcs3_pdp['mcsid'].isin(mp)]
# Parent types remaining
print(mcs3_pdp['celig00'].value_counts(dropna=False))
# Rename variables
mcs3_pdp = mcs3_pdp.rename(columns={col: col + '_p' if col != 'mcsid' else col for col in mcs3_pdp.columns})


#Household grid variables
#Drop all Aupair/nannys and other non-relatives
#mcs3_hh = mcs3_hh[mcs3_hh['rel']!=18]
#mcs3_hh = mcs3_hh[mcs3_hh['rel']!=20]
#Replace negatives values as missing (have to skip the first column as it is MCSID)
for col in mcs3_hh.columns[1:]:
    mcs3_hh[col] = mcs3_hh[col].apply(lambda x: np.nan if x < 0 else x)
#Use function defined above
mcs3_hh=hhcount(mcs3_hh, 'chcage00', 'chpage00', 'chcrel00', 3)

del col
#Creating separate hh grid files for CM and main parent so both characteristics can be included
mcs3_hhmp = mcs3_hh[mcs3_hh['celig00'] == 1]
mcs3_hhmp = pd.merge(mcs3_hh, mcs3_pim[['mcsid', 'celig00']], on=['mcsid', 'celig00'], how='inner')
mcs3_hhmp = mcs3_hhmp.drop(columns=['cresp00', 'ccnum00', 'chintm00', 'chinty00', 'chcsex00',
                                    'chcdbm00', 'chcdby00', 'chcage00', 'chcprs00', 'chsaoc00',
                                    'chpsoc00', 'chnvoc00', 'chpcoc00', 'chhtoc00', 'chwtoc00',
                                    'chwsoc00', 'hhag1ch3', 'hhag2ch3', 'hhag3ch3', 'hhag4ch3',
                                    'buag1ch3', 'buag2ch3', 'buag3ch3', 'buag4ch3'])
# mcs3_hhpp = mcs3_hh[mcs3_hh['celig00'] == 2]
mcs3_hhpp = pd.merge(mcs3_hh, mcs3_pip[['mcsid', 'celig00_p']], left_on=['mcsid', 'celig00'], right_on=['mcsid', 'celig00_p'], how='inner')
mcs3_hhpp = mcs3_hhpp.drop(columns=['cresp00', 'ccnum00', 'chintm00', 'chinty00', 'chcsex00',
                                    'chcdbm00', 'chcdby00', 'chcage00', 'chcprs00', 'chsaoc00',
                                    'chpsoc00', 'chnvoc00', 'chpcoc00', 'chhtoc00', 'chwtoc00',
                                    'chwsoc00', 'hhag1ch3', 'hhag2ch3', 'hhag3ch3', 'hhag4ch3',
                                    'buag1ch3', 'buag2ch3', 'buag3ch3', 'buag4ch3'])
mcs3_hhpp = mcs3_hhpp.rename(columns={col: col + '_p' if col != 'mcsid' else col for col in mcs3_hhpp.columns})
mcs3_hhcm = mcs3_hh[(mcs3_hh['ccnum00'] == 1) | (mcs3_hh['ccnum00'] == 2) | (mcs3_hh['ccnum00'] == 3)]
mcs3_hhcm = mcs3_hhcm.drop(columns=['cpnum00', 'celig00', 'cresp00', 'chpsex00',
                                    'chpdbm00', 'chpdby00', 'chpres00', 'chpage00', 'chcrel00',
                                    'chprela0', 'chprelb0', 'chprelc0', 'chpreld0', 'chprele0',
                                    'chprelf0', 'chprelg0', 'chprelh0', 'chpreli0', 'chprelj0',
                                    'chprelk0', 'chprell0', 'chprelm0', 'chpreln0', 'chprelo0',
                                    'chprelp0', 'chprelq0', 'chprelr0', 'chprels0', 'chprelt0',
                                    'chprelu0', 'chprelv0', 'chprelw0', 'chprelx0', 'chprely0',
                                    'chpjob00', 'chptpc00', 'chpsty00', 'chpstm00', 'chpdcy00',
                                    'chpdcm00', 'chpspy00', 'chpspm00'])

### Combine all files in wave
#Cohort member interview and derived
mcs3 = pd.merge(mcs3_ci, mcs3_cd, on = ['mcsid', 'ccnum00'])
#Main file and cognitive scores
mcs3 = pd.merge(mcs3, mcs3_ca, how='left', on = ['mcsid', 'ccnum00'])
mcs3.isnull().any()
#Main file and teacher survey
mcs3 = pd.merge(mcs3, mcs3_ts, how='left', on = ['mcsid', 'ccnum00'])
#Main file and geographic data
mcs3 = pd.merge(mcs3, mcs3_gld, how='left', on = ['mcsid'])
mcs3.isnull().any()
#Main file and parent interview
mcs3 = pd.merge(mcs3, mcs3_pim, how='left', on = ['mcsid'])
#Main file and partner interview
mcs3 = pd.merge(mcs3, mcs3_pip, how='left', on = ['mcsid'])
#Main file and main parent derived
mcs3 = pd.merge(mcs3, mcs3_pdm, how='left', on = ['mcsid', 'cpnum00', 'celig00', 'cresp00'])
#Main file and partner derived
mcs3 = pd.merge(mcs3, mcs3_pdp, how='left', on = ['mcsid', 'cpnum00_p', 'celig00_p', 'cresp00_p'])
#Main file and family interview
mcs3 = pd.merge(mcs3, mcs3_fi, how='left', on = ['mcsid'])
#Main file and family derived
mcs3 = pd.merge(mcs3, mcs3_fd, how='left', on = ['mcsid'])
#Main file and parent CM interview
mcs3 = pd.merge(mcs3, mcs3_pci, how='left', on = ['mcsid', 'ccnum00', 'cpnum00', 'celig00', 'cresp00'])
#Main file and household grid (CM)
mcs3 = pd.merge(mcs3, mcs3_hhcm, how='left', on = ['mcsid', 'ccnum00'])
#Main file and household grid (MP)
mcs3 = pd.merge(mcs3, mcs3_hhmp, how='left', on = ['mcsid', 'cpnum00', 'celig00'])
#Main file and household grid (P)
mcs3 = pd.merge(mcs3, mcs3_hhpp, how='left', on = ['mcsid', 'cpnum00_p', 'celig00_p'])

del mcs3_ca
del mcs3_cd
del mcs3_ci
del mcs3_fi
del mcs3_fd
del mcs3_ts
del mcs3_gld
del mcs3_pd
del mcs3_pdm
del mcs3_pdp
del mcs3_pi
del mcs3_pim
del mcs3_pip
del mp
del mcs3_ppi
del mcs3_hh
del mcs3_hhcm
del mcs3_hhmp
del mcs3_hhpp
del mcs3_pci

print(mcs3['ccnum00'].value_counts(dropna=False))

mcs3['cnum'] = mcs3['ccnum00']

########## Wave 4 ##########
######### Read Data
mcs4_ca = pd.read_stata(os.path.join(data, 'Sweep4/stata/stata13/mcs4_cm_cognitive_assessment.dta'), convert_categoricals=False)
mcs4_ci = pd.read_stata(os.path.join(data, 'Sweep4/stata/stata13/mcs4_cm_interview.dta'), convert_categoricals=False)
mcs4_cd = pd.read_stata(os.path.join(data, 'Sweep4/stata/stata13/mcs4_cm_derived.dta'), convert_categoricals=False)
mcs4_pi = pd.read_stata(os.path.join(data, 'Sweep4/stata/stata13/mcs4_parent_interview.dta'), convert_categoricals=False)
mcs4_pd = pd.read_stata(os.path.join(data, 'Sweep4/stata/stata13/mcs4_parent_derived.dta'), convert_categoricals=False)
mcs4_fi = pd.read_stata(os.path.join(data, 'Sweep4/stata/stata13/mcs4_family_interview.dta'), convert_categoricals=False)
mcs4_fd = pd.read_stata(os.path.join(data, 'Sweep4/stata/stata13/mcs4_family_derived.dta'), convert_categoricals=False)
mcs4_pd = pd.read_stata(os.path.join(data, 'Sweep4/stata/stata13/mcs4_parent_derived.dta'), convert_categoricals=False)
mcs4_ts = pd.read_stata(os.path.join(data, 'Sweep4/stata/stata13/mcs4_cm_teacher_survey.dta'), convert_categoricals=False)
mcs4_gld = pd.read_stata(os.path.join(data, 'Sweep4/stata/stata13/mcs4_geographically_linked_data.dta'), convert_categoricals=False)
mcs4_ppi = pd.read_stata(os.path.join(data, 'Sweep4/stata/stata13/mcs4_proxy_partner_interview.dta'), convert_categoricals=False)
mcs4_hh = pd.read_stata(os.path.join(data, 'Sweep4/stata/stata13/mcs4_hhgrid.dta'), convert_categoricals=False)
mcs4_pci = pd.read_stata(os.path.join(data, 'Sweep4/stata/stata13/mcs4_parent_cm_interview.dta'), convert_categoricals=False)


# Change variable names to lowecase
mcs4_ca.columns= mcs4_ca.columns.str.lower()
mcs4_ci.columns= mcs4_ci.columns.str.lower()
mcs4_cd.columns= mcs4_cd.columns.str.lower()
mcs4_pi.columns= mcs4_pi.columns.str.lower()
mcs4_pd.columns= mcs4_pd.columns.str.lower()
mcs4_fi.columns= mcs4_fi.columns.str.lower()
mcs4_fd.columns= mcs4_fd.columns.str.lower()
mcs4_ts.columns= mcs4_ts.columns.str.lower()
mcs4_gld.columns= mcs4_gld.columns.str.lower()
mcs4_ppi.columns= mcs4_ppi.columns.str.lower()
mcs4_hh.columns= mcs4_hh.columns.str.lower()
mcs4_pci.columns= mcs4_pci.columns.str.lower()
###Number of unique IDs in each file
print(mcs4_pi['mcsid'].nunique())
print(mcs4_pd['mcsid'].nunique())
print(mcs4_fi['mcsid'].nunique())
print(mcs4_fd['mcsid'].nunique())
print(mcs4_gld['mcsid'].nunique())
print(mcs4_ppi['mcsid'].nunique())
print(mcs4_hh['mcsid'].nunique())
#Number of unique IDs in each country
print(mcs4_fd.iloc[:, 0:2].groupby('dactry00').agg(['nunique']).stack())

### Parent interview
# Parent types interviewed
print(mcs4_pi['delig00'].value_counts(dropna=False))
# #Keep main parent only
# mcs4_pim = mcs4_pi[mcs4_pi['delig00'] == 1]
# Keep main parent only if available, keep partner if main parent unavailable
# mcs4_pim = (
#     mcs4_pi
#     .sort_values(['mcsid', 'delig00'])
#     .groupby(['mcsid'])
#     .first()
# )
mcs4_pim = mcs4_pi.sort_values(['mcsid', 'delig00']).drop_duplicates(subset=['mcsid'], keep='first')
# Parent types remaining
print(mcs4_pim['delig00'].value_counts(dropna=False))

##Partner
# Parent types interviewed
print(mcs4_pi['delig00'].value_counts(dropna=False))
# Filter out only partners
mcs4_pip = mcs4_pi[mcs4_pi['delig00'] == 2]
# Keep keep partners only if main parent available
mp = mcs4_pi[mcs4_pi['delig00'] == 1]['mcsid'].unique()
mcs4_pip = mcs4_pip[mcs4_pip['mcsid'].isin(mp)]
# Parent types remaining
print(mcs4_pip['delig00'].value_counts(dropna=False))
#Rename variables 
mcs4_pip = mcs4_pip.rename(columns={col: col + '_p' if col != 'mcsid' else col for col in mcs4_pip.columns})


### Parent CM interview
# Parent types interviewed
print(mcs4_pci['delig00'].value_counts(dropna=False))

# Keep main parent only if available, keep partner if main parent unavailable
mcs4_pci = (
    mcs4_pci
    .sort_values(['mcsid', 'dcnum00', 'delig00'])
    .groupby(['mcsid', 'dcnum00'])
    .first()
)
# Parent types remaining
print(mcs4_pci['delig00'].value_counts(dropna=False))


### Parent derived variables
# Parent types interviewed
print(mcs4_pd['delig00'].value_counts(dropna=False))
# #Keep main parent only
# mcs4_pdm = mcs4_pd[mcs4_pd['delig00'] == 1]
# Keep main parent only if available, keep partner if main parent unavailable
# mcs4_pdm = (
#     mcs4_pd
#     .sort_values(['mcsid', 'delig00'])
#     .groupby(['mcsid'])
#     .first()
# )
mcs4_pdm = mcs4_pd.sort_values(['mcsid', 'delig00']).drop_duplicates(subset=['mcsid'], keep='first')
# Parent types remaining
print(mcs4_pdm['delig00'].value_counts(dropna=False))

### Parent CM interview
# Parent types interviewed
print(mcs4_pci['delig00'].value_counts(dropna=False))
##Partner
# Parent types interviewed
print(mcs4_pd['delig00'].value_counts(dropna=False))
# Filter out only partners
mcs4_pdp = mcs4_pd[mcs4_pd['delig00'] == 2]
# Keep keep partners only if main parent available
mp = mcs4_pd[mcs4_pd['delig00'] == 1]['mcsid'].unique()
mcs4_pdp = mcs4_pdp[mcs4_pdp['mcsid'].isin(mp)]
# Parent types remaining
print(mcs4_pdp['delig00'].value_counts(dropna=False))
#Rename variables 
mcs4_pdp = mcs4_pdp.rename(columns={col: col + '_p' if col != 'mcsid' else col for col in mcs4_pdp.columns})

#Household grid variables
#Drop all Aupair/nannys and other non-relatives
#mcs4_hh = mcs4_hh[mcs4_hh['rel']!=18]
#mcs4_hh = mcs4_hh[mcs4_hh['rel']!=20]
#Replace negatives values as missing (have to skip the first column as it is MCSID)
for col in mcs4_hh.columns[1:]:
    mcs4_hh[col] = mcs4_hh[col].apply(lambda x: np.nan if x < 0 else x)
#Use function defined above
mcs4_hh=hhcount(mcs4_hh, 'dhcage00', 'dhpage00', 'dhcrel00', 4)

del col

#Creating separate hh grid files for CM and main parent so both characteristics can be included
# mcs4_hhmp = mcs4_hh[mcs4_hh['delig00'] == 1]
mcs4_hhmp = pd.merge(mcs4_hh, mcs4_pim[['mcsid', 'delig00']], on=['mcsid', 'delig00'], how='inner')
mcs4_hhmp = mcs4_hhmp.drop(columns=['dresp00', 'dcnum00', 'dhintm00', 'dhinty00', 'dhcsex00',
                                    'dhcdbm00', 'dhcdby00', 'dhcage00', 'dhcprs00', 'dhsaoc00',
                                    'dhoaoc00', 'dhwroc00', 'dhoc00', 'dhpcoc00', 'dhhtoc00',
                                    'dhwtoc00', 'dhwsoc00', 'hhag1ch4', 'hhag2ch4', 'hhag3ch4', 'hhag4ch4',
                                    'buag1ch4', 'buag2ch4', 'buag3ch4', 'buag4ch4'])
# mcs4_hhpp = mcs4_hh[mcs4_hh['delig00'] == 2]
mcs4_hhpp = pd.merge(mcs4_hh, mcs4_pip[['mcsid', 'delig00_p']], left_on=['mcsid', 'delig00'], right_on=['mcsid', 'delig00_p'], how='inner')
mcs4_hhpp = mcs4_hhpp.drop(columns=['dresp00', 'dcnum00', 'dhintm00', 'dhinty00', 'dhcsex00',
                                    'dhcdbm00', 'dhcdby00', 'dhcage00', 'dhcprs00', 'dhsaoc00',
                                    'dhoaoc00', 'dhwroc00', 'dhoc00', 'dhpcoc00', 'dhhtoc00',
                                    'dhwtoc00', 'dhwsoc00', 'hhag1ch4', 'hhag2ch4', 'hhag3ch4', 'hhag4ch4',
                                    'buag1ch4', 'buag2ch4', 'buag3ch4', 'buag4ch4'])
mcs4_hhpp = mcs4_hhpp.rename(columns={col: col + '_p' if col != 'mcsid' else col for col in mcs4_hhpp.columns})
mcs4_hhcm = mcs4_hh[(mcs4_hh['dcnum00'] == 1) | (mcs4_hh['dcnum00'] == 2) | (mcs4_hh['dcnum00'] == 3)]
mcs4_hhcm = mcs4_hhcm.drop(columns=['dpnum00', 'delig00', 'dresp00', 'dhpsex00',
                                    'dhpdbm00', 'dhpdby00', 'dhpres00', 'dhpage00', 'dhcrel00',
                                    'dhprela0', 'dhprelb0', 'dhprelc0', 'dhpreld0', 'dhprele0',
                                    'dhprelf0', 'dhprelg0', 'dhprelh0', 'dhpreli0', 'dhprelj0',
                                    'dhprelk0', 'dhprell0', 'dhprelm0', 'dhpreln0', 'dhprelo0',
                                    'dhprelp0', 'dhprelq0', 'dhprelr0', 'dhprels0', 'dhprelt0',
                                    'dhprelu0', 'dhprelv0', 'dhprelw0', 'dhprelx0', 'dhprely0',
                                    'dhpjob00', 'dhptpc00', 'dhpsty00', 'dhpstm00', 'dhpdcy00',
                                    'dhpspy00', 'dhpspm00'])
#'dhpdcm00',

### Combine all files in wave
#Cohort member interview and derived
mcs4 = pd.merge(mcs4_ci, mcs4_cd, on = ['mcsid', 'dcnum00'])
#Main file and cognitive scores
mcs4 = pd.merge(mcs4, mcs4_ca, how='left', on = ['mcsid', 'dcnum00'])
mcs4.isnull().any()
#Main file and teacher survey
mcs4 = pd.merge(mcs4, mcs4_ts, how='left', on = ['mcsid', 'dcnum00'])
#Main file and geographic data
mcs4 = pd.merge(mcs4, mcs4_gld, how='left', on = ['mcsid'])
mcs4.isnull().any()
#Main file and parent interview
mcs4 = pd.merge(mcs4, mcs4_pim, how='left', on = ['mcsid'])
#Main file and partner interview
mcs4 = pd.merge(mcs4, mcs4_pip, how='left', on = ['mcsid'])
#Main file and main parent derived
mcs4 = pd.merge(mcs4, mcs4_pdm, how='left', on = ['mcsid', 'dpnum00', 'delig00', 'dresp00'])
#Main file and partner derived
mcs4 = pd.merge(mcs4, mcs4_pdp, how='left', on = ['mcsid', 'dpnum00_p', 'delig00_p', 'dresp00_p'])
#Main file and family interview
mcs4 = pd.merge(mcs4, mcs4_fi, how='left', on = ['mcsid'])
#Main file and family derived
mcs4 = pd.merge(mcs4, mcs4_fd, how='left', on = ['mcsid'])
#Main file and parent CM interview
mcs4 = pd.merge(mcs4, mcs4_pci, how='left', on = ['mcsid', 'dcnum00', 'dpnum00', 'delig00', 'dresp00'])
#Main file and household grid (CM)
mcs4 = pd.merge(mcs4, mcs4_hhcm, how='left', on = ['mcsid', 'dcnum00'])
#Main file and household grid (MP)
mcs4 = pd.merge(mcs4, mcs4_hhmp, how='left', on = ['mcsid', 'dpnum00', 'delig00'])
#Main file and household grid (P)
mcs4 = pd.merge(mcs4, mcs4_hhpp, how='left', on = ['mcsid', 'dpnum00_p', 'delig00_p'])


del mcs4_ca
del mcs4_cd
del mcs4_ci
del mcs4_fi
del mcs4_fd
del mcs4_ts
del mcs4_gld
del mcs4_pd
del mcs4_pdm
del mcs4_pdp
del mcs4_pi
del mcs4_pim
del mcs4_pip
del mp
del mcs4_ppi
del mcs4_hh
del mcs4_hhcm
del mcs4_hhmp
del mcs4_hhpp
del mcs4_pci


print(mcs4['dcnum00'].value_counts(dropna=False))

mcs4['cnum'] = mcs4['dcnum00']

########## Wave 5 ##########
######### Read Data
mcs5_ca = pd.read_stata(os.path.join(data, 'Sweep5/stata/stata13/mcs5_cm_cognitive_assessment.dta'), convert_categoricals=False)
mcs5_ci = pd.read_stata(os.path.join(data, 'Sweep5/stata/stata13/mcs5_cm_interview.dta'), convert_categoricals=False)
mcs5_cd = pd.read_stata(os.path.join(data, 'Sweep5/stata/stata13/mcs5_cm_derived.dta'), convert_categoricals=False)
mcs5_pi = pd.read_stata(os.path.join(data, 'Sweep5/stata/stata13/mcs5_parent_interview.dta'), convert_categoricals=False)
mcs5_pd = pd.read_stata(os.path.join(data, 'Sweep5/stata/stata13/mcs5_parent_derived.dta'), convert_categoricals=False)
mcs5_fi = pd.read_stata(os.path.join(data, 'Sweep5/stata/stata13/mcs5_family_interview.dta'), convert_categoricals=False)
mcs5_fd = pd.read_stata(os.path.join(data, 'Sweep5/stata/stata13/mcs5_family_derived.dta'), convert_categoricals=False)
mcs5_pd = pd.read_stata(os.path.join(data, 'Sweep5/stata/stata13/mcs5_parent_derived.dta'), convert_categoricals=False)
mcs5_ts = pd.read_stata(os.path.join(data, 'Sweep5/stata/stata13/mcs5_cm_teacher_survey.dta'), convert_categoricals=False)
mcs5_gld = pd.read_stata(os.path.join(data, 'Sweep5/stata/stata13/mcs5_geographically_linked_data.dta'), convert_categoricals=False)
mcs5_ppi = pd.read_stata(os.path.join(data, 'Sweep5/stata/stata13/mcs5_proxy_partner_interview.dta'), convert_categoricals=False)
mcs5_hh = pd.read_stata(os.path.join(data, 'Sweep5/stata/stata13/mcs5_hhgrid.dta'), convert_categoricals=False)
mcs5_pci = pd.read_stata(os.path.join(data, 'Sweep5/stata/stata13/mcs5_parent_cm_interview.dta'), convert_categoricals=False)


# Change variable names to lowecase
mcs5_ca.columns= mcs5_ca.columns.str.lower()
mcs5_ci.columns= mcs5_ci.columns.str.lower()
mcs5_cd.columns= mcs5_cd.columns.str.lower()
mcs5_pi.columns= mcs5_pi.columns.str.lower()
mcs5_pd.columns= mcs5_pd.columns.str.lower()
mcs5_fi.columns= mcs5_fi.columns.str.lower()
mcs5_fd.columns= mcs5_fd.columns.str.lower()
mcs5_ts.columns= mcs5_ts.columns.str.lower()
mcs5_gld.columns= mcs5_gld.columns.str.lower()
mcs5_ppi.columns= mcs5_ppi.columns.str.lower()
mcs5_hh.columns= mcs5_hh.columns.str.lower()
mcs5_pci.columns= mcs5_pci.columns.str.lower()
###Number of unique IDs in each file
print(mcs5_pi['mcsid'].nunique())
print(mcs5_pd['mcsid'].nunique())
print(mcs5_fi['mcsid'].nunique())
print(mcs5_fd['mcsid'].nunique())
print(mcs5_gld['mcsid'].nunique())
print(mcs5_ppi['mcsid'].nunique())
print(mcs5_hh['mcsid'].nunique())
#Number of unique IDs in each country
print(mcs5_fd.iloc[:, 0:2].groupby('eactry00').agg(['nunique']).stack())

### Parent interview
# Parent types interviewed
print(mcs5_pi['eelig00'].value_counts(dropna=False))
# #Keep main parent only
# mcs5_pim = mcs5_pi[mcs5_pi['eelig00'] == 1]
# Keep main parent only if available, keep partner if main parent unavailable
# mcs5_pim = (
#     mcs5_pi
#     .sort_values(['mcsid', 'eelig00'])
#     .groupby(['mcsid'])
#     .first()
# )
mcs5_pim = mcs5_pi.sort_values(['mcsid', 'eelig00']).drop_duplicates(subset=['mcsid'], keep='first')
# Parent types remaining
print(mcs5_pim['eelig00'].value_counts(dropna=False))

##Partner
# Parent types interviewed
print(mcs5_pi['eelig00'].value_counts(dropna=False))
# Filter out only partners
mcs5_pip = mcs5_pi[mcs5_pi['eelig00'] == 2]
# Keep keep partners only if main parent available
mp = mcs5_pi[mcs5_pi['eelig00'] == 1]['mcsid'].unique()
mcs5_pip = mcs5_pip[mcs5_pip['mcsid'].isin(mp)]
# Parent types remaining
print(mcs5_pip['eelig00'].value_counts(dropna=False))
#Rename variables 
mcs5_pip = mcs5_pip.rename(columns={col: col + '_p' if col != 'mcsid' else col for col in mcs5_pip.columns})

### Parent CM interview
# Parent types interviewed
print(mcs5_pci['eelig00'].value_counts(dropna=False))

# Keep main parent only if available, keep partner if main parent unavailable
mcs5_pci = (
    mcs5_pci
    .sort_values(['mcsid', 'ecnum00', 'eelig00'])
    .groupby(['mcsid', 'ecnum00'])
    .first()
)
# Parent types remaining
print(mcs5_pci['eelig00'].value_counts(dropna=False))

### Parent derived variables
# Parent types interviewed
print(mcs5_pd['eelig00'].value_counts(dropna=False))
# #Keep main parent only
# mcs5_pdm = mcs5_pd[mcs5_pd['eelig00'] == 1]
# Keep main parent only if available, keep partner if main parent unavailable
# mcs5_pdm = (
#     mcs5_pd
#     .sort_values(['mcsid', 'eelig00'])
#     .groupby(['mcsid'])
#     .first()
# )
mcs5_pdm = mcs5_pd.sort_values(['mcsid', 'eelig00']).drop_duplicates(subset=['mcsid'], keep='first')
# Parent types remaining
print(mcs5_pdm['eelig00'].value_counts(dropna=False))

##Partner
# Parent types interviewed
print(mcs5_pd['eelig00'].value_counts(dropna=False))
# Filter out only partners
mcs5_pdp = mcs5_pd[mcs5_pd['eelig00'] == 2]
# Keep keep partners only if main parent available
mp = mcs5_pd[mcs5_pd['eelig00'] == 1]['mcsid'].unique()
mcs5_pdp = mcs5_pdp[mcs5_pdp['mcsid'].isin(mp)]
# Parent types remaining
print(mcs5_pdp['eelig00'].value_counts(dropna=False))
#Rename variables 
mcs5_pdp = mcs5_pdp.rename(columns={col: col + '_p' if col != 'mcsid' else col for col in mcs5_pdp.columns})

#Household grid variables
#Drop all Aupair/nannys and other non-relatives
#mcs5_hh = mcs5_hh[mcs5_hh['rel']!=18]
#mcs5_hh = mcs5_hh[mcs5_hh['rel']!=20]
#Replace negatives values as missing (have to skip the first column as it is MCSID)
for col in mcs5_hh.columns[1:]:
    mcs5_hh[col] = mcs5_hh[col].apply(lambda x: np.nan if x < 0 else x)
#Use function defined above
mcs5_hh=hhcount(mcs5_hh, 'ecage0000', 'epage0000', 'ecrel0000', 5)

del col

#Creating separate hh grid files for CM and main parent so both characteristics can be included
# mcs5_hhmp = mcs5_hh[mcs5_hh['eelig00'] == 1]
mcs5_hhmp = pd.merge(mcs5_hh, mcs5_pim[['mcsid', 'eelig00']], on=['mcsid', 'eelig00'], how='inner')
mcs5_hhmp = mcs5_hhmp.drop(columns=['eresp00', 'ecnum00', 'eintm0000', 'einty0000', 'ecsex0000',
                                    'ecdbm0000', 'ecdby0000', 'ecage0000', 'ecprs0000', 'eswmoc000',
                                    'ecgtoc000', 'evsoc000', 'escoc0000', 'ehtoc0000', 'ewtoc0000', 
                                    'ebfoc0000', 'hhag1ch5', 'hhag2ch5', 'hhag3ch5', 'hhag4ch5',
                                    'buag1ch5', 'buag2ch5', 'buag3ch5', 'buag4ch5'])
# mcs5_hhpp = mcs5_hh[mcs5_hh['eelig00'] == 2]
mcs5_hhpp = pd.merge(mcs5_hh, mcs5_pip[['mcsid', 'eelig00_p']], left_on=['mcsid', 'eelig00'], right_on=['mcsid', 'eelig00_p'], how='inner')
mcs5_hhpp = mcs5_hhpp.drop(columns=['eresp00', 'ecnum00', 'eintm0000', 'einty0000', 'ecsex0000',
                                    'ecdbm0000', 'ecdby0000', 'ecage0000', 'ecprs0000', 'eswmoc000',
                                    'ecgtoc000', 'evsoc000', 'escoc0000', 'ehtoc0000', 'ewtoc0000', 
                                    'ebfoc0000', 'hhag1ch5', 'hhag2ch5', 'hhag3ch5', 'hhag4ch5',
                                    'buag1ch5', 'buag2ch5', 'buag3ch5', 'buag4ch5'])
mcs5_hhpp = mcs5_hhpp.rename(columns={col: col + '_p' if col != 'mcsid' else col for col in mcs5_hhpp.columns})
mcs5_hhcm = mcs5_hh[(mcs5_hh['ecnum00'] == 1) | (mcs5_hh['ecnum00'] == 2) | (mcs5_hh['ecnum00'] == 3)]
mcs5_hhcm = mcs5_hhcm.drop(columns=['epnum00', 'eelig00', 'eresp00', 'epsex0000',
                                    'epdbm0000', 'epdby0000', 'epres0000', 'epage0000', 'ecrel0000',
                                    'eprel0a00', 'eprel0b00', 'eprel0c00', 'eprel0d00', 'eprel0e00',
                                    'eprel0f00', 'eprel0g00', 'eprel0h00', 'eprel0i00', 'eprel0j00',
                                    'eprel0k00', 'eprel0l00', 'eprel0m00', 'eprel0n00', 'eprel0o00',
                                    'eprel0p00', 'eprel0q00', 'eprel0r00', 'eprel0s00', 'eprel0t00',
                                    'eprel0u00', 'eprel0v00', 'eprel0w00', 'eprel0x00', 'epjob0000',
                                    'epsty0000', 'epstm0000', 'epdcy0000', 'epspy0000', 'epspm0000',
                                    'bchk0000', 'ebwhp0000'])
# 'epdcm0000',
### Combine all files in wave
#Cohort member interview and derived
mcs5 = pd.merge(mcs5_ci, mcs5_cd, on = ['mcsid', 'ecnum00'])
#Main file and cognitive scores
mcs5 = pd.merge(mcs5, mcs5_ca, how='left', on = ['mcsid', 'ecnum00'])
mcs5.isnull().any()
#Main file and teacher survey
mcs5 = pd.merge(mcs5, mcs5_ts, how='left', on = ['mcsid', 'ecnum00'])
#Main file and geographic data
mcs5 = pd.merge(mcs5, mcs5_gld, how='left', on = ['mcsid'])
mcs5.isnull().any()
#Main file and main parent interview
mcs5 = pd.merge(mcs5, mcs5_pim, how='left', on = ['mcsid'])
#Main file and partner interview
mcs5 = pd.merge(mcs5, mcs5_pip, how='left', on = ['mcsid'])
#Main file and main parent derived
mcs5 = pd.merge(mcs5, mcs5_pdm, how='left', on = ['mcsid', 'epnum00', 'eelig00', 'eresp00'])
#Main file and partner derived
mcs5 = pd.merge(mcs5, mcs5_pdp, how='left', on = ['mcsid', 'epnum00_p', 'eelig00_p', 'eresp00_p'])
#Main file and family interview
mcs5 = pd.merge(mcs5, mcs5_fi, how='left', on = ['mcsid'])
#Main file and family derived
mcs5 = pd.merge(mcs5, mcs5_fd, how='left', on = ['mcsid'])
#Main file and parent CM interview
mcs5 = pd.merge(mcs5, mcs5_pci, how='left', on = ['mcsid', 'ecnum00', 'epnum00', 'eelig00', 'eresp00'])
#Main file and household grid (CM)
mcs5 = pd.merge(mcs5, mcs5_hhcm, how='left', on = ['mcsid', 'ecnum00'])
#Main file and household grid (MP)
mcs5 = pd.merge(mcs5, mcs5_hhmp, how='left', on = ['mcsid', 'epnum00', 'eelig00'])
#Main file and household grid (P)
mcs5 = pd.merge(mcs5, mcs5_hhpp, how='left', on = ['mcsid', 'epnum00_p', 'eelig00_p'])

del mcs5_ca
del mcs5_cd
del mcs5_ci
del mcs5_fi
del mcs5_fd
del mcs5_ts
del mcs5_gld
del mcs5_pd
del mcs5_pdm
del mcs5_pdp
del mcs5_pi
del mcs5_pim
del mcs5_pip
del mp
del mcs5_ppi
del mcs5_hh
del mcs5_hhcm
del mcs5_hhmp
del mcs5_hhpp
del mcs5_pci

print(mcs5['ecnum00'].value_counts(dropna=False))

mcs5['cnum'] = mcs5['ecnum00']

########## Wave 6 ##########
######### Read Data
mcs6_ca = pd.read_stata(os.path.join(data, 'Sweep6/stata/stata13/mcs6_cm_cognitive_assessment.dta'), convert_categoricals=False)
mcs6_ci = pd.read_stata(os.path.join(data, 'Sweep6/stata/stata13/mcs6_cm_interview.dta'), convert_categoricals=False)
mcs6_cd = pd.read_stata(os.path.join(data, 'Sweep6/stata/stata13/mcs6_cm_derived.dta'), convert_categoricals=False)
mcs6_pi = pd.read_stata(os.path.join(data, 'Sweep6/stata/stata13/mcs6_parent_interview.dta'), convert_categoricals=False)
mcs6_pd = pd.read_stata(os.path.join(data, 'Sweep6/stata/stata13/mcs6_parent_derived.dta'), convert_categoricals=False)
mcs6_fd = pd.read_stata(os.path.join(data, 'Sweep6/stata/stata13/mcs6_family_derived.dta'), convert_categoricals=False)
mcs6_pd = pd.read_stata(os.path.join(data, 'Sweep6/stata/stata13/mcs6_parent_derived.dta'), convert_categoricals=False)
mcs6_ppi = pd.read_stata(os.path.join(data, 'Sweep6/stata/stata13/mcs6_proxy_partner_interview.dta'), convert_categoricals=False)
mcs6_hh = pd.read_stata(os.path.join(data, 'Sweep6/stata/stata13/mcs6_hhgrid.dta'), convert_categoricals=False)
mcs6_pa = pd.read_stata(os.path.join(data, 'Sweep6/stata/stata13/mcs6_parent_assessment.dta'), convert_categoricals=False)
mcs6_pci = pd.read_stata(os.path.join(data, 'Sweep6/stata/stata13/mcs6_parent_cm_interview.dta'), convert_categoricals=False)
mcs6_imde = pd.read_stata(os.path.join(data, 'Sweep6/stata/stata13/mcs_sweep6_imd_e_2004.dta'), convert_categoricals=False)
mcs6_imdw = pd.read_stata(os.path.join(data, 'Sweep6/stata/stata13/mcs_sweep6_imd_w_2004.dta'), convert_categoricals=False)
mcs6_imds = pd.read_stata(os.path.join(data, 'Sweep6/stata/stata13/mcs_sweep6_imd_s_2004.dta'), convert_categoricals=False)
mcs6_imdn = pd.read_stata(os.path.join(data, 'Sweep6/stata/stata13/mcs_sweep6_imd_n_2004.dta'), convert_categoricals=False)


# Change variable names to lowecase
mcs6_ca.columns= mcs6_ca.columns.str.lower()
mcs6_ci.columns= mcs6_ci.columns.str.lower()
mcs6_cd.columns= mcs6_cd.columns.str.lower()
mcs6_pi.columns= mcs6_pi.columns.str.lower()
mcs6_pd.columns= mcs6_pd.columns.str.lower()
mcs6_fd.columns= mcs6_fd.columns.str.lower()
mcs6_pa.columns= mcs6_pa.columns.str.lower()
mcs6_hh.columns= mcs6_hh.columns.str.lower()
mcs6_ppi.columns= mcs6_ppi.columns.str.lower()
mcs6_pci.columns= mcs6_pci.columns.str.lower()
mcs6_imde.columns= mcs6_imde.columns.str.lower()
mcs6_imdw.columns= mcs6_imdw.columns.str.lower()
mcs6_imds.columns= mcs6_imds.columns.str.lower()
mcs6_imdn.columns= mcs6_imdn.columns.str.lower()
###Number of unique IDs in each file
print(mcs6_pi['mcsid'].nunique())
print(mcs6_pd['mcsid'].nunique())
print(mcs6_fd['mcsid'].nunique())
print(mcs6_hh['mcsid'].nunique())
print(mcs6_pa['mcsid'].nunique())
#Number of unique IDs in each country
print(mcs6_fd.iloc[:, 0:2].groupby('factry00').agg(['nunique']).stack())

### Parent interview
# Parent types interviewed
print(mcs6_pi['felig00'].value_counts(dropna=False))
# #Keep main parent only
# mcs6_pim = mcs6_pi[mcs6_pi['felig00'] == 1]
# Keep main parent only if available, keep partner if main parent unavailable
# mcs6_pim = (
#     mcs6_pi
#     .sort_values(['mcsid', 'felig00'])
#     .groupby(['mcsid'])
#     .first()
# )
mcs6_pim = mcs6_pi.sort_values(['mcsid', 'felig00']).drop_duplicates(subset=['mcsid'], keep='first')
# Parent types remaining
print(mcs6_pim['felig00'].value_counts(dropna=False))

##Partner
# Parent types interviewed
print(mcs6_pi['felig00'].value_counts(dropna=False))
# Filter out only partners
mcs6_pip = mcs6_pi[mcs6_pi['felig00'] == 2]
# Keep keep partners only if main parent available
mp = mcs6_pi[mcs6_pi['felig00'] == 1]['mcsid'].unique()
mcs6_pip = mcs6_pip[mcs6_pip['mcsid'].isin(mp)]
# Parent types remaining
print(mcs6_pip['felig00'].value_counts(dropna=False))
#Rename variables 
mcs6_pip = mcs6_pip.rename(columns={col: col + '_p' if col != 'mcsid' else col for col in mcs6_pip.columns})

### Parent derived variables
# Parent types interviewed
print(mcs6_pd['felig00'].value_counts(dropna=False))
# #Keep main parent only
# mcs6_pdm = mcs6_pd[mcs6_pd['felig00'] == 1]
# Keep main parent only if available, keep partner if main parent unavailable
# mcs6_pdm = (
#     mcs6_pd
#     .sort_values(['mcsid', 'felig00'])
#     .groupby(['mcsid'])
#     .first()
# )
mcs6_pdm = mcs6_pd.sort_values(['mcsid', 'felig00']).drop_duplicates(subset=['mcsid'], keep='first')
# Parent types remaining
print(mcs6_pdm['felig00'].value_counts(dropna=False))

##Partner
# Parent types interviewed
print(mcs6_pd['felig00'].value_counts(dropna=False))
# Filter out only partners
mcs6_pdp = mcs6_pd[mcs6_pd['felig00'] == 2]
# Keep keep partners only if main parent available
mp = mcs6_pd[mcs6_pd['felig00'] == 1]['mcsid'].unique()
mcs6_pdp = mcs6_pdp[mcs6_pdp['mcsid'].isin(mp)]
# Parent types remaining
print(mcs6_pdp['felig00'].value_counts(dropna=False))
#Rename variables 
mcs6_pdp = mcs6_pdp.rename(columns={col: col + '_p' if col != 'mcsid' else col for col in mcs6_pdp.columns})

### Parent CM interview
# Parent types interviewed
print(mcs6_pci['felig00'].value_counts(dropna=False))

# Keep main parent only if available, keep partner if main parent unavailable
mcs6_pci = (
    mcs6_pci
    .sort_values(['mcsid', 'fcnum00', 'felig00'])
    .groupby(['mcsid', 'fcnum00'])
    .first()
)
# Parent types remaining
print(mcs6_pci['felig00'].value_counts(dropna=False))

### Parent assessment
# Parent types interviewed
print(mcs6_pa['felig00'].value_counts(dropna=False))

# Keep main parent only if available, keep partner if main parent unavailable
mcs6_pa = (
    mcs6_pa
    .sort_values(['mcsid', 'felig00'])
    .groupby(['mcsid'])
    .first()
)
# Parent types remaining
print(mcs6_pa['felig00'].value_counts(dropna=False))

#Household grid variables
#Drop all Aupair/nannys and other non-relatives
#mcs6_hh = mcs6_hh[mcs6_hh['rel']!=18]
#mcs6_hh = mcs6_hh[mcs6_hh['rel']!=20]
#Replace negatives values as missing (have to skip the first column as it is MCSID)
for col in mcs6_hh.columns[1:]:
    mcs6_hh[col] = mcs6_hh[col].apply(lambda x: np.nan if x < 0 else x)
#Use function defined above
mcs6_hh=hhcount(mcs6_hh, 'fhcage00', 'fhpage00', 'fhcrel00', 6)

del col
#Creating separate hh grid files for CM and main parent so both characteristics can be included
# mcs6_hhmp = mcs6_hh[mcs6_hh['felig00'] == 1]
mcs6_hhmp = pd.merge(mcs6_hh, mcs6_pim[['mcsid', 'felig00']], on=['mcsid', 'felig00'], how='inner')
mcs6_hhmp = mcs6_hhmp.drop(columns=['fresp00', 'fcnum00', 'fhintm00', 'fhinty00', 'fhcsex00',
                                    'fhcdbm00', 'fhcdby00', 'fhcage00', 'fhcprs00', 'fhsaliw',
                                    'hhag1ch6', 'hhag2ch6', 'hhag3ch6', 'hhag4ch6',
                                    'buag1ch6', 'buag2ch6', 'buag3ch6', 'buag4ch6'])
# mcs6_hhpp = mcs6_hh[mcs6_hh['felig00'] == 2]
mcs6_hhpp = pd.merge(mcs6_hh, mcs6_pip[['mcsid', 'felig00_p']], left_on=['mcsid', 'felig00'], right_on=['mcsid', 'felig00_p'], how='inner')
mcs6_hhpp = mcs6_hhpp.drop(columns=['fresp00', 'fcnum00', 'fhintm00', 'fhinty00', 'fhcsex00',
                                    'fhcdbm00', 'fhcdby00', 'fhcage00', 'fhcprs00', 'fhsaliw',
                                    'hhag1ch6', 'hhag2ch6', 'hhag3ch6', 'hhag4ch6',
                                    'buag1ch6', 'buag2ch6', 'buag3ch6', 'buag4ch6'])
mcs6_hhpp = mcs6_hhpp.rename(columns={col: col + '_p' if col != 'mcsid' else col for col in mcs6_hhpp.columns})
mcs6_hhcm = mcs6_hh[(mcs6_hh['fcnum00'] == 1) | (mcs6_hh['fcnum00'] == 2) | (mcs6_hh['fcnum00'] == 3)]
mcs6_hhcm = mcs6_hhcm.drop(columns=['fpnum00', 'felig00', 'fresp00', 'fhpsex00',
                                    'fhpdbm00', 'fhpdby00', 'fhpres00', 'fhpage00', 'fhcrel00',
                                    'fhprel0a', 'fhprel0b', 'fhprel0c', 'fhprel0d', 'fhprel0e',
                                    'fhprel0f', 'fhprel0g', 'fhprel0h', 'fhprel0i', 'fhprel0j',
                                    'fhprel0k', 'fhprel0l', 'fhprel0m', 'fhprel0n', 'fhprel0o',
                                    'fhprel0p', 'fhprel0q', 'fhprel0r', 'fhprel0s', 'fhprel0t',
                                    'fhprel0u', 'fhprel0v', 'fhprel0w', 'fhprel0x',
                                    'fhpjob00', 'fhpdcm00', 'fhpsty00', 'fhpstm00', 'fhpdcy00',
                                    'fhpspy00', 'fhpspm00'])

### Combine all files in wave
#Cohort member interview and derived
mcs6 = pd.merge(mcs6_ci, mcs6_cd, on = ['mcsid', 'fcnum00'])
#Main file and cognitive scores
mcs6 = pd.merge(mcs6, mcs6_ca, how='left', on = ['mcsid', 'fcnum00'])
mcs6.isnull().any()
#Main file and parent assessment
mcs6 = pd.merge(mcs6, mcs6_pa, how='left', on = ['mcsid'])
#Main file and main parent interview
mcs6 = pd.merge(mcs6, mcs6_pim, how='left', on = ['mcsid', 'fpnum00', 'felig00', 'fresp00'])
#Main file and partner interview
mcs6 = pd.merge(mcs6, mcs6_pip, how='left', on = ['mcsid'])
#Main file and main parent derived
mcs6 = pd.merge(mcs6, mcs6_pdm, how='left', on = ['mcsid', 'fpnum00', 'felig00', 'fresp00'])
#Main file and partner derived
mcs6 = pd.merge(mcs6, mcs6_pdp, how='left', on = ['mcsid', 'fpnum00_p', 'felig00_p', 'fresp00_p'])
#Main file and family derived
mcs6 = pd.merge(mcs6, mcs6_fd, how='left', on = ['mcsid'])
#Main file and parent CM interview
mcs6_pci = mcs6_pci.drop(columns=['fccage00', 'fccdbm00', 'fccsex00', 'fccdby00'])
mcs6 = pd.merge(mcs6, mcs6_pci, how='left', on = ['mcsid', 'fcnum00', 'fpnum00', 'felig00', 'fresp00'])
#Main file and household grid (CM)
mcs6 = pd.merge(mcs6, mcs6_hhcm, how='left', on = ['mcsid', 'fcnum00'])
#Main file and household grid (MP)
mcs6 = pd.merge(mcs6, mcs6_hhmp, how='left', on = ['mcsid', 'fpnum00', 'felig00'])
#Main file and household grid (P)
mcs6 = pd.merge(mcs6, mcs6_hhpp, how='left', on = ['mcsid', 'fpnum00_p', 'felig00_p'])
#Main file and IMD England
mcs6 = pd.merge(mcs6, mcs6_imde, how='left', on = ['mcsid', 'factry00'])
#Main file and IMD Wales
mcs6 = pd.merge(mcs6, mcs6_imdw, how='left', on = ['mcsid', 'factry00'])
#Main file and IMD Scotland
mcs6 = pd.merge(mcs6, mcs6_imds, how='left', on = ['mcsid', 'factry00'])
#Main file and IMD NI
mcs6 = pd.merge(mcs6, mcs6_imdn, how='left', on = ['mcsid', 'factry00'])


del mcs6_ca
del mcs6_cd
del mcs6_ci
del mcs6_fd
del mcs6_hh
del mcs6_pa
del mcs6_pd
del mcs6_pdm
del mcs6_pdp
del mcs6_pi
del mcs6_pim
del mcs6_pip
del mp
del mcs6_ppi
del mcs6_hhcm
del mcs6_hhmp
del mcs6_hhpp
del mcs6_pci
del mcs6_imde
del mcs6_imdw
del mcs6_imds
del mcs6_imdn

print(mcs6['fcnum00'].value_counts(dropna=False))

mcs6['cnum'] = mcs6['fcnum00']

########## Wave 7 ##########
######### Read Data
mcs7_ca = pd.read_stata(os.path.join(data, 'Sweep7/stata/stata13/mcs7_cm_cognitive_assessment.dta'), convert_categoricals=False)
mcs7_ci = pd.read_stata(os.path.join(data, 'Sweep7/stata/stata13/mcs7_cm_interview.dta'), convert_categoricals=False)
mcs7_cd = pd.read_stata(os.path.join(data, 'Sweep7/stata/stata13/mcs7_cm_derived.dta'), convert_categoricals=False)
mcs7_pi = pd.read_stata(os.path.join(data, 'Sweep7/stata/stata13/mcs7_parent_interview.dta'), convert_categoricals=False)
mcs7_pd = pd.read_stata(os.path.join(data, 'Sweep7/stata/stata13/mcs7_parent_derived.dta'), convert_categoricals=False)
mcs7_fi = pd.read_stata(os.path.join(data, 'Sweep7/stata/stata13/mcs7_family_interview.dta'), convert_categoricals=False)
mcs7_fd = pd.read_stata(os.path.join(data, 'Sweep7/stata/stata13/mcs7_family_derived.dta'), convert_categoricals=False)
mcs7_pd = pd.read_stata(os.path.join(data, 'Sweep7/stata/stata13/mcs7_parent_derived.dta'), convert_categoricals=False)
mcs7_cq = pd.read_stata(os.path.join(data, 'Sweep7/stata/stata13/mcs7_cm_qualifications.dta'), convert_categoricals=False)
mcs7_hh = pd.read_stata(os.path.join(data, 'Sweep7/stata/stata13/mcs7_hhgrid.dta'), convert_categoricals=False)
mcs7_pci = pd.read_stata(os.path.join(data, 'Sweep7/stata/stata13/mcs7_parent_cm_interview.dta'), convert_categoricals=False)

# Change variable names to lowecase
mcs7_ca.columns= mcs7_ca.columns.str.lower()
mcs7_ci.columns= mcs7_ci.columns.str.lower()
mcs7_cd.columns= mcs7_cd.columns.str.lower()
mcs7_pi.columns= mcs7_pi.columns.str.lower()
mcs7_pd.columns= mcs7_pd.columns.str.lower()
mcs7_fi.columns= mcs7_fi.columns.str.lower()
mcs7_fd.columns= mcs7_fd.columns.str.lower()
mcs7_cq.columns= mcs7_cq.columns.str.lower()
mcs7_hh.columns= mcs7_hh.columns.str.lower()
mcs7_pci.columns= mcs7_pci.columns.str.lower()
###Number of unique IDs in each file
print(mcs7_hh['mcsid'].nunique())
print(mcs7_pi['mcsid'].nunique())
print(mcs7_pd['mcsid'].nunique())
print(mcs7_fi['mcsid'].nunique())
print(mcs7_fd['mcsid'].nunique())
#Number of unique IDs in each country
print(mcs7_fi.iloc[:, 0:2].groupby('g_country').agg(['nunique']).stack())
#Reshape qualifications file
# Explicitly specify the columns to select
col_s = [col for col in mcs7_cq.columns if '_s_' in col]
col_s = ['mcsid', 'gcnum00', 'gc_rowid'] + col_s
col_l = [col for col in mcs7_cq.columns if '_l_' in col]
col_l = ['mcsid', 'gcnum00', 'gc_rowid'] + col_l
# Filter only the single columns
df_s = mcs7_cq[col_s].groupby(['mcsid', 'gcnum00']).first().reset_index()
df_s.columns = df_s.columns.str.replace('_s_', '_')
del df_s['gc_rowid']
# Filter only the looped columns
df_l = mcs7_cq[col_l]
# Pivot the looped DataFrame
df_l = df_l.pivot(index=['mcsid', 'gcnum00'], columns=['gc_rowid'],
                  values=[col for col in mcs7_cq.columns if '_l_' in col]).reset_index()
df_l.columns = [f'{col[0]}_{col[1]}' if col[0] != ''  else col[1] for col in df_l.columns]
df_l.columns = df_l.columns.str.replace('mcsid_', 'mcsid')
df_l.columns = df_l.columns.str.replace('gcnum00_', 'gcnum00')
df_l.columns = df_l.columns.str.replace('_l_', '_')
# Merge the "_S_" and "_L_" DataFrames
mcs7_pcq = pd.merge(df_s, df_l, on=['mcsid', 'gcnum00'])
del col_l
del col_s
del df_s
del df_l

### Parent interview
print(mcs7_hh['gelig00'].value_counts(dropna=False))
mcs7_hh.loc[mcs7_hh['gelig00'] < 0, 'gelig00'] = np.nan
print(mcs7_hh['gelig00'].value_counts(dropna=False))
mcs7_hhm = mcs7_hh[['mcsid', 'gpnum00', 'gelig00']]
mcs7_pi = pd.merge(mcs7_pi, mcs7_hhm, how='left', on = ['mcsid', 'gpnum00'])

# Parent types interviewed
print(mcs7_pi['gelig00'].value_counts(dropna=False))
# #Keep main parent only
# mcs7_pim = mcs7_pi[mcs7_pi['gelig00'] == 1]
# Keep main parent only if available, keep partner if main parent unavailable
# mcs7_pim = (
#     mcs7_pi
#     .sort_values(['mcsid', 'gelig00'])
#     .groupby(['mcsid'])
#     .first()
# )
mcs7_pim = mcs7_pi.sort_values(['mcsid', 'gelig00']).drop_duplicates(subset=['mcsid'], keep='first')
# Parent types remaining
print(mcs7_pim['gelig00'].value_counts(dropna=False))

### Parent CM interview
##Partner
# Parent types interviewed
print(mcs7_pi['gelig00'].value_counts(dropna=False))
# Filter out only partners
mcs7_pip = mcs7_pi[mcs7_pi['gelig00'] == 2]
# Keep keep partners only if main parent available
mp = mcs7_pi[mcs7_pi['gelig00'] == 1]['mcsid'].unique()
mcs7_pip = mcs7_pip[mcs7_pip['mcsid'].isin(mp)]
# Parent types remaining
print(mcs7_pip['gelig00'].value_counts(dropna=False))
#Rename variables 
mcs7_pip = mcs7_pip.rename(columns={col: col + '_p' if col != 'mcsid' else col for col in mcs7_pip.columns})

### Parent CM interview
mcs7_hhm = mcs7_hh[['mcsid', 'gpnum00', 'gelig00']]
mcs7_pci = pd.merge(mcs7_pci, mcs7_hhm, how='left', on = ['mcsid', 'gpnum00'])
# Parent types interviewed
print(mcs7_pci['gelig00'].value_counts(dropna=False))

# Keep main parent only if available, keep partner if main parent unavailable
mcs7_pci = (
    mcs7_pci
    .sort_values(['mcsid', 'gcnum00', 'gelig00'])
    .groupby(['mcsid', 'gcnum00'])
    .first()
)
# Parent types remaining
print(mcs7_pci['gelig00'].value_counts(dropna=False))

del mcs7_hhm

### Parent derived variables
mcs7_hhm = mcs7_hh[['mcsid', 'gpnum00', 'gelig00']]
mcs7_pd = pd.merge(mcs7_pd, mcs7_hhm, how='left', on = ['mcsid', 'gpnum00'])
# Parent types interviewed
print(mcs7_pd['gelig00'].value_counts(dropna=False))
# #Keep main parent only
# mcs7_pdm = mcs7_pd[mcs7_pd['gelig00'] == 1]
# Keep main parent only if available, keep partner if main parent unavailable
# mcs7_pdm = (
#     mcs7_pd
#     .sort_values(['mcsid', 'gelig00'])
#     .groupby(['mcsid'])
#     .first()
# )
# mcs7_pdm = pd.merge(mcs7_pd, mcs7_pim, how='left', on = ['mcsid', 'gpnum00'])
mcs7_pdm = mcs7_pd.sort_values(['mcsid', 'gelig00']).drop_duplicates(subset=['mcsid'], keep='first')
# Parent types remaining
print(mcs7_pdm['gelig00'].value_counts(dropna=False))

##Partner
# Parent types interviewed
print(mcs7_pd['gelig00'].value_counts(dropna=False))
# Filter out only partners
# mcs7_pdp = pd.merge(mcs7_pdp, mcs7_pip, how='left', on = ['mcsid', 'gpnum00_p'])
# Filter out only partners
mcs7_pdp = mcs7_pd[mcs7_pd['gelig00'] == 2]
# Keep keep partners only if main parent available
mp = mcs7_pd[mcs7_pd['gelig00'] == 1]['mcsid'].unique()
mcs7_pdp = mcs7_pdp[mcs7_pdp['mcsid'].isin(mp)]
# Parent types remaining
print(mcs7_pdp['gelig00'].value_counts(dropna=False))
#Rename variables 
mcs7_pdp = mcs7_pd.rename(columns={col: col + '_p' if col != 'mcsid' else col for col in mcs7_pd.columns})

#Household grid variables
#Drop all Aupair/nannys and other non-relatives
#mcs7_hh = mcs7_hh[mcs7_hh['rel']!=18]
#mcs7_hh = mcs7_hh[mcs7_hh['rel']!=20]
#Replace negatives values as missing (have to skip the first column as it is MCSID)
for col in mcs7_hh.columns[1:]:
    mcs7_hh[col] = mcs7_hh[col].apply(lambda x: np.nan if x < 0 else x)
#Use function defined above
mcs7_hh=hhcount(mcs7_hh, 'ghcage00', 'ghpage00', 'ghcrel00', 7)

del col
#Creating separate hh grid files for CM and main parent so both characteristics can be included
# mcs7_hhmp = mcs7_hh[mcs7_hh['gelig00'] == 1]
mcs7_hhmp = pd.merge(mcs7_hh, mcs7_pim[['mcsid', 'gelig00']], on=['mcsid', 'gelig00'], how='inner')
mcs7_hhmp = mcs7_hhmp.drop(columns=['gcnum00', 'ghintm00', 'ghinty00', 'ghcsex00',
                                    'ghcdbm00', 'ghcdby00', 'ghcage00', 'ghcprs00',
                                    'hhag1ch7', 'hhag2ch7', 'hhag3ch7', 'hhag4ch7',
                                    'buag1ch7', 'buag2ch7', 'buag3ch7', 'buag4ch7'])
# mcs7_hhpp = mcs7_hh[mcs7_hh['gelig00'] == 2]
mcs7_hhpp = pd.merge(mcs7_hh, mcs7_pip[['mcsid', 'gelig00_p']], left_on=['mcsid', 'gelig00'], right_on=['mcsid', 'gelig00_p'], how='inner')
mcs7_hhpp = mcs7_hhpp.drop(columns=['gcnum00', 'ghintm00', 'ghinty00', 'ghcsex00',
                                    'ghcdbm00', 'ghcdby00', 'ghcage00', 'ghcprs00',
                                    'hhag1ch7', 'hhag2ch7', 'hhag3ch7', 'hhag4ch7',
                                    'buag1ch7', 'buag2ch7', 'buag3ch7', 'buag4ch7'])
mcs7_hhpp = mcs7_hhpp.rename(columns={col: col + '_p' if col != 'mcsid' else col for col in mcs7_hhpp.columns})
mcs7_hhcm = mcs7_hh[(mcs7_hh['gcnum00'] == 1) | (mcs7_hh['gcnum00'] == 2) | (mcs7_hh['gcnum00'] == 3)]
mcs7_hhcm = mcs7_hhcm.drop(columns=['gpnum00', 'gelig00', 'ghpsex00',
                                    'ghpdbm00', 'ghpdby00', 'ghpres00', 'ghpage00', 'ghcrel00',
                                    'ghprel0a', 'ghprel0b', 'ghprel0c', 'ghprel0d', 'ghprel0e',
                                    'ghprel0f', 'ghprel0g', 'ghprel0h', 'ghprel0i', 'ghprel0j',
                                    'ghprel0k', 'ghprel0l', 'ghprel0m', 'ghprel0n', 'ghprel0o',
                                    'ghprel0p', 'ghprel0q', 'ghprel0r', 'ghprel0s', 'ghprel0t',
                                    'ghprel0u', 'ghprel0v', 'ghprel0w', 'ghprel0x', 'ghprel0y',
                                    'ghpjob00', 'ghpdcm00', 'ghpsty00', 'ghpstm00', 'ghpdcy00',
                                    'ghpspy00', 'ghpspm00'])


### Combine all files in wave
#Cohort member interview and derived
mcs7 = pd.merge(mcs7_cd, mcs7_ci, on = ['mcsid', 'gcnum00'])
#Main file and cognitive scores
mcs7 = pd.merge(mcs7, mcs7_ca, how='left', on = ['mcsid', 'gcnum00'])
mcs7.isnull().any()
#Main file and cohort member qualifications
mcs7 = pd.merge(mcs7, mcs7_pcq, how='left', on = ['mcsid', 'gcnum00'])
#Main file and main parent interview
mcs7 = pd.merge(mcs7, mcs7_pim, how='left', on = ['mcsid'])
#Main file and partner interview
mcs7 = pd.merge(mcs7, mcs7_pip, how='left', on = ['mcsid'])
#Main file and main parent derived
mcs7 = pd.merge(mcs7, mcs7_pdm, how='left', on = ['mcsid', 'gpnum00', 'gelig00'])
#Main file and partner derived
mcs7 = pd.merge(mcs7, mcs7_pdp, how='left', on = ['mcsid', 'gpnum00_p', 'gelig00_p'])
#Main file and family interview
mcs7 = pd.merge(mcs7, mcs7_fi, how='left', on = ['mcsid'])
#Main file and family derived
mcs7 = pd.merge(mcs7, mcs7_fd, how='left', on = ['mcsid'])
#Main file and parent CM interview
mcs7 = pd.merge(mcs7, mcs7_pci, how='left', on = ['mcsid', 'gcnum00', 'gelig00', 'gpnum00'])
#Main file and household grid (CM)
mcs7 = pd.merge(mcs7, mcs7_hhcm, how='left', on = ['mcsid', 'gcnum00'])
#Main file and household grid (MP)
mcs7 = pd.merge(mcs7, mcs7_hhmp, how='left', on = ['mcsid', 'gpnum00', 'gelig00'])
#Main file and household grid (P)
mcs7 = pd.merge(mcs7, mcs7_hhpp, how='left', on = ['mcsid', 'gpnum00_p', 'gelig00_p'])
del mcs7_ca
del mcs7_cd
del mcs7_ci
del mcs7_fi
del mcs7_fd
del mcs7_cq
del mcs7_pcq
del mcs7_hh
del mcs7_pd
del mcs7_pdm
del mcs7_pdp
del mcs7_pi
del mcs7_pim
del mcs7_pip
del mp
del mcs7_hhcm
del mcs7_hhmp
del mcs7_hhpp
del mcs7_pci
del mcs7_hhm

print(mcs7['gcnum00'].value_counts(dropna=False))

mcs7['cnum'] = mcs7['gcnum00']

# print([x for x in mcs7.columns if x.startswith('gar')])
# print([x for x in mcs7_hhmp.columns if x.endswith('resp00')])

########## Cleaning the data and defining variables

##### Functions for cleaning
#Calculate age in months at time of interview
def agemth(intm, inty, brtm, brty):

    # Check if the column exists in the DataFrame
    if pd.notna(inty) and pd.notna(intm) and pd.notna(brty) and pd.notna(brtm):
        # Calculate months since birth
        mnthsb = brty*12 + brtm
        mnthsi = inty*12 + intm
        mnth = mnthsi - mnthsb
        
        return mnth
    else:
        
        return np.nan
    
#Standard normalise
def stdnrm(data, var):
    """
    Standardize a column in a DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the column to be standardized.
    column_name (str): The name of the column to be standardized.

    Returns:
    pd.Series: A new Series containing the standardized values.
    """
    # Check if the column exists in the DataFrame
    if var in data:
        # Calculate the mean and standard deviation
        mean = data[var].mean()
        std = data[var].std()
        
        # Apply standardization formula to the column
        std_var = (data[var] - mean) / std
        
        return std_var
    else:
        print(f"Column '{column_name}' not found in the DataFrame.") # type: ignore
        return None

#Age Adjust
def ageadj(data, var, age):
    """
    Standardize a column in a DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the column to be standardized.
    column_name (str): The name of the column to be standardized.

    Returns:
    pd.Series: A new Series containing the standardized values.
    """
    # Check if the column exists in the DataFrame
    if var in data:
        # Calculate the mean and standard deviation
        formula = "var ~ C(age)"
        m = smf.ols(formula, data=data).fit()
        adj = m.resid + m.params.Intercept
        
        return adj
        print(f"Column '{column_name}' not found in the DataFrame.")
        return None
    


    
##### Basic demographic variables
#dt1 = mcs1.dtypes
##Cohort member age in months at time of interview ("cmagem`X'")
# Sweep 1
mcs1['cmagem1'] = mcs1.apply(lambda row: 
                             agemth(row['ahintm00'], row['ahinty00'], row['ahcdbm00'], row['ahcdby00']),
                             axis=1)
print(mcs1['cmagem1'].value_counts(dropna=False))
print(mcs1['cmagem1'].value_counts().sum())
print(mcs1[mcs1['cnum']== 1]['cmagem1'].value_counts(dropna=False))
print(mcs1[mcs1['cnum']== 1]['cmagem1'].value_counts().sum())
mcs1['cmaged1'] = mcs1['ahcage00']

# Sweep 2
mcs2['cmagem2'] = mcs2.apply(lambda row: 
                             agemth(row['bhintm00'], row['bhinty00'], row['bhcdbm00'], row['bhcdby00']),
                             axis=1)
print(mcs2['cmagem2'].value_counts(dropna=False))
print(mcs2['cmagem2'].value_counts().sum())
print(mcs2[mcs2['cnum']== 1]['cmagem2'].value_counts(dropna=False))
print(mcs2[mcs2['cnum']== 1]['cmagem2'].value_counts().sum())
mcs2['cmaged2'] = mcs2['bhcage00']
# Sweep 3
mcs3['cmagem3'] = mcs3.apply(lambda row: 
                             agemth(row['chintm00'], row['chinty00'], row['chcdbm00'], row['chcdby00']),
                             axis=1)
print(mcs3['cmagem3'].value_counts(dropna=False))
print(mcs3['cmagem3'].value_counts().sum())
print(mcs3[mcs3['cnum']== 1]['cmagem3'].value_counts(dropna=False))
print(mcs3[mcs3['cnum']== 1]['cmagem3'].value_counts().sum())
mcs3['cmaged3'] = mcs3['chcage00']

# Sweep 4
mcs4['cmagem4'] = mcs4.apply(lambda row: 
                             agemth(row['dhintm00'], row['dhinty00'], row['dhcdbm00'], row['dhcdby00']),
                             axis=1)
print(mcs4['cmagem4'].value_counts(dropna=False))
print(mcs4['cmagem4'].value_counts().sum())
print(mcs4[mcs4['cnum']== 1]['cmagem4'].value_counts(dropna=False))
print(mcs4[mcs4['cnum']== 1]['cmagem4'].value_counts().sum())
mcs4['cmaged4'] = mcs4['dhcage00']

# Sweep 5
mcs5['cmagem5'] = mcs5.apply(lambda row: 
                             agemth(row['eintm0000'], row['einty0000'], row['ecdbm0000'], row['ecdby0000']),
                             axis=1)
print(mcs5['cmagem5'].value_counts(dropna=False))
print(mcs5['cmagem5'].value_counts().sum())
print(mcs5[mcs5['cnum']== 1]['cmagem5'].value_counts(dropna=False))
print(mcs5[mcs5['cnum']== 1]['cmagem5'].value_counts().sum())
mcs5['cmagey5'] = mcs5['ecage0000']
print(mcs5['cmagey5'].value_counts(dropna=False))
print(mcs5['cmagey5'].value_counts().sum())
# Sweep 6
mcs6['cmagem6'] = mcs6.apply(lambda row: 
                             agemth(row['fhintm00'], row['fhinty00'], row['fhcdbm00'], row['fhcdby00']),
                             axis=1)
print(mcs6['cmagem6'].value_counts(dropna=False))
print(mcs6['cmagem6'].value_counts().sum())
print(mcs6[mcs6['cnum']== 1]['cmagem6'].value_counts(dropna=False))
print(mcs6[mcs6['cnum']== 1]['cmagem6'].value_counts().sum())
mcs6['cmagey6'] = mcs6['fhcage00']
print(mcs6['cmagey6'].value_counts(dropna=False))
print(mcs6['cmagey6'].value_counts().sum())
# Sweep 7
mcs7['cmagem7'] = mcs7.apply(lambda row: 
                             agemth(row['ghintm00'], row['ghinty00'], row['ghcdbm00'], row['ghcdby00']),
                             axis=1)
print(mcs7['cmagem7'].value_counts(dropna=False))
print(mcs7['cmagem7'].value_counts().sum())
print(mcs7[mcs7['cnum']== 1]['cmagem7'].value_counts(dropna=False))
print(mcs7[mcs7['cnum']== 1]['cmagem7'].value_counts().sum())
mcs7['cmagey7'] = mcs7['ghcage00']
print(mcs7['cmagey7'].value_counts(dropna=False))
print(mcs7['cmagey7'].value_counts().sum())

##Sex
print([x for x in mcs5.columns if x.startswith('ehcs')])
# Sweep 1
mcs1['sex1'] = mcs1['ahcsex00']
print(mcs1['sex1'].value_counts(dropna=False))
print(mcs1['sex1'].value_counts().sum())
# Sweep 2
mcs2['sex2'] = mcs2['bhcsex00']
print(mcs2['sex2'].value_counts(dropna=False))
print(mcs2['sex2'].value_counts().sum())
# Sweep 3
mcs3['sex3'] = mcs3['chcsex00_y']
print(mcs3['sex3'].value_counts(dropna=False))
print(mcs3['sex3'].value_counts().sum())
# Sweep 4
mcs4['sex4'] = mcs4['dhcsex00']
print(mcs4['sex4'].value_counts(dropna=False))
print(mcs4['sex4'].value_counts().sum())
# Sweep 5
mcs5['sex5'] = mcs5['ecsex0000']
print(mcs5['sex5'].value_counts(dropna=False))
print(mcs5['sex5'].value_counts().sum())
# Sweep 6
mcs6['sex6'] = mcs6['fhcsex00']
print(mcs6['sex6'].value_counts(dropna=False))
print(mcs6['sex6'].value_counts().sum())
# Sweep 7
mcs7['sex7'] = mcs7['ghcsex00']
print(mcs7['sex7'].value_counts(dropna=False))
print(mcs7['sex7'].value_counts().sum())

##Country
# Sweep 1
mcs1['country1'] = mcs1['aactry00_y']
print(mcs1['country1'].value_counts(dropna=False))
print(mcs1['country1'].value_counts().sum())
# Sweep 2
mcs2['country2'] = mcs2['bactry00_y']
print(mcs2['country2'].value_counts(dropna=False))
print(mcs2['country2'].value_counts().sum())
# Sweep 3
mcs3['country3'] = mcs3['cactry00_y']
print(mcs3['country3'].value_counts(dropna=False))
print(mcs3['country3'].value_counts().sum())
# Sweep 4
mcs4['country4'] = mcs4['dactry00_y']
print(mcs4['country4'].value_counts(dropna=False))
print(mcs4['country4'].value_counts().sum())
# Sweep 5
mcs5['country5'] = mcs5['eactry00_y']
print(mcs5['country5'].value_counts(dropna=False))
print(mcs5['country5'].value_counts().sum())
# Sweep 6
mcs6['country6'] = mcs6['factry00']
print(mcs6['country6'].value_counts(dropna=False))
print(mcs6['country6'].value_counts().sum())
# Sweep 7
mcs7['country7'] = mcs7['gactry00_y']
print(mcs7['country7'].value_counts(dropna=False))
print(mcs7['country7'].value_counts().sum())

##Region
# Sweep 1
mcs1['region1'] = mcs1['aaregn00']
print(mcs1['region1'].value_counts(dropna=False))
print(mcs1['region1'].value_counts().sum())
# Sweep 2
mcs2['region2'] = mcs2['baregn00']
print(mcs2['region2'].value_counts(dropna=False))
print(mcs2['region2'].value_counts().sum())
# Sweep 3
mcs3['region3'] = mcs3['caregn00']
print(mcs3['region3'].value_counts(dropna=False))
print(mcs3['region3'].value_counts().sum())
# Sweep 4
mcs4['region4'] = mcs4['daregn00']
print(mcs4['region4'].value_counts(dropna=False))
print(mcs4['region4'].value_counts().sum())
# Sweep 5
mcs5['region5'] = mcs5['earegn00']
print(mcs5['region5'].value_counts(dropna=False))
print(mcs5['region5'].value_counts().sum())
# Sweep 6
mcs6['region6'] = mcs6['faregn00']
print(mcs6['region6'].value_counts(dropna=False))
print(mcs6['region6'].value_counts().sum())
# Sweep 7
mcs7['region7'] = mcs7['garegn00_x']
print(mcs7['region7'].value_counts(dropna=False))
print(mcs7['region7'].value_counts().sum())


##Ethnicity
# Sweep 1
#Ethnicity 30 category
mcs1['eth30_eng1'] = mcs1['adceea00_r30']
mcs1['eth30_wal1'] = mcs1['adcewa00_r30']
mcs1['eth30_sco1'] = mcs1['adcesa00_r30']
mcs1['eth30_nir1'] = mcs1['adcena00_r30']
mcs1['ethnicity301'] = mcs1['eth30_eng1']
mcs1['ethnicity301'] = np.where((mcs1['country1'] == 2) & (mcs1['ethnicity301'] < 0)
                                , mcs1['eth30_wal1'], mcs1['ethnicity301'])
mcs1['ethnicity301'] = np.where((mcs1['country1'] == 3) & (mcs1['ethnicity301'] < 0)
                                , mcs1['eth30_sco1'], mcs1['ethnicity301'])
mcs1['ethnicity301'] = np.where((mcs1['country1'] == 4) & (mcs1['ethnicity301'] < 0)
                                , mcs1['eth30_nir1'], mcs1['ethnicity301'])
mcs1.loc[mcs1['ethnicity301'] < 0, 'ethnicity301'] = np.nan
print(mcs1['ethnicity301'].value_counts(dropna=False))
print(mcs1['ethnicity301'].value_counts().sum())

#Ethnicity 6 category
# 1                                 White
# 2                                 Mixed
# 3                                Indian
# 4             Pakistani and Bangladeshi
# 5                Black or Black British
# 6 Other Ethnic group (inc Chinese,Other)
mcs1['ethnicity1'] = mcs1['adc06e00']
mcs1.loc[mcs1['ethnicity1'] < 0, 'ethnicity1'] = np.nan
print(mcs1['ethnicity1'].value_counts(dropna=False))
print(mcs1['ethnicity1'].value_counts().sum())
# Sweep 2
#Ethnicity 30 category
mcs2['eth30_eng2'] = mcs2['bdceea00_r30']
mcs2['eth30_wal2'] = mcs2['bdcewa00_r30']
mcs2['eth30_sco2'] = mcs2['bdcesa00_r30']
# mcs2['eth30_nir2'] = mcs2['bdcena00_r30']
mcs2['ethnicity302'] = mcs2['eth30_eng2']
mcs2['ethnicity302'] = np.where((mcs2['country2'] == 2) & (mcs2['ethnicity302'] < 0)
                                , mcs2['eth30_wal2'], mcs2['ethnicity302'])
mcs2['ethnicity302'] = np.where((mcs2['country2'] == 3) & (mcs2['ethnicity302'] < 0)
                                , mcs2['eth30_sco2'], mcs2['ethnicity302'])
# mcs2['ethnicity302'] = np.where((mcs2['country2'] == 4) & (mcs2['ethnicity302'] < 0)
#                                 , mcs2['eth30_nir2'], mcs2['ethnicity302'])
mcs2.loc[mcs2['ethnicity302'] < 0, 'ethnicity302'] = np.nan
print(mcs2['ethnicity302'].value_counts(dropna=False))
print(mcs2['ethnicity302'].value_counts().sum())
#Ethnicity 6 category
mcs2['ethnicity2'] = mcs2['bdc06e00']
mcs2.loc[mcs2['ethnicity2'] < 0, 'ethnicity2'] = np.nan
print(mcs2['ethnicity2'].value_counts(dropna=False))
print(mcs2['ethnicity2'].value_counts().sum())






##### Birth characteristics
##Birth date
# Sweep 1
mcs1['birthm1'] = mcs1['ahcdbm00']
print(mcs1['birthm1'].value_counts(dropna=False))
print(mcs1['birthm1'].value_counts().sum())
mcs1['birthy1'] = mcs1['ahcdby00']
print(mcs1['birthy1'].value_counts(dropna=False))
print(mcs1['birthy1'].value_counts().sum())
# Sweep 2
mcs2['birthm2'] = mcs2['bhcdbm00']
print(mcs2['birthm2'].value_counts(dropna=False))
print(mcs2['birthm2'].value_counts().sum())
mcs2['birthy2'] = mcs2['bhcdby00']
print(mcs2['birthy2'].value_counts(dropna=False))
print(mcs2['birthy2'].value_counts().sum())
# Sweep 3
mcs3['birthm3'] = mcs3['chcdbm00']
print(mcs3['birthm3'].value_counts(dropna=False))
print(mcs3['birthm3'].value_counts().sum())
mcs3['birthy3'] = mcs3['chcdby00']
print(mcs3['birthy3'].value_counts(dropna=False))
print(mcs3['birthy3'].value_counts().sum())
# Sweep 4
mcs4['birthm4'] = mcs4['dhcdbm00']
print(mcs4['birthm4'].value_counts(dropna=False))
print(mcs4['birthm4'].value_counts().sum())
mcs4['birthy4'] = mcs4['dhcdby00']
print(mcs4['birthy4'].value_counts(dropna=False))
print(mcs4['birthy4'].value_counts().sum())
# Sweep 5
mcs5['birthm5'] = mcs5['ecdbm0000']
print(mcs5['birthm5'].value_counts(dropna=False))
print(mcs5['birthm5'].value_counts().sum())
mcs5['birthy5'] = mcs5['ecdby0000']
print(mcs5['birthy5'].value_counts(dropna=False))
print(mcs5['birthy5'].value_counts().sum())
# Sweep 6
mcs6['birthm6'] = mcs6['fhcdbm00']
print(mcs6['birthm6'].value_counts(dropna=False))
print(mcs6['birthm6'].value_counts().sum())
mcs6['birthy6'] = mcs6['fhcdby00']
print(mcs6['birthy6'].value_counts(dropna=False))
print(mcs6['birthy6'].value_counts().sum())
# Sweep 7
mcs7['birthm7'] = mcs7['ghcdbm00']
print(mcs7['birthm7'].value_counts(dropna=False))
print(mcs7['birthm7'].value_counts().sum())
mcs7['birthy7'] = mcs7['ghcdby00']
print(mcs7['birthy7'].value_counts(dropna=False))
print(mcs7['birthy7'].value_counts().sum())

##Mothers age at birth
mcs1['agemb1'] = mcs1['addagb00']
print(mcs1['agemb1'].value_counts(dropna=False))
print(mcs1['agemb1'].value_counts().sum())
#Only include age of biological female parent
mcs1.loc[((mcs1['ahcrel00'] != 7) | (mcs1['ahpsex00'] != 2)) , 'agemb1'] = np.nan
mcs1.loc[mcs1['agemb1'] < 0, 'agemb1'] = np.nan
print(mcs1['agemb1'].value_counts(dropna=False))
print(mcs1['agemb1'].value_counts().sum())
# Sweep 2
mcs2['agemb2'] = mcs2['bddagb00']
print(mcs2['agemb2'].value_counts(dropna=False))
print(mcs2['agemb2'].value_counts().sum())
#Only include age of biological female parent
mcs2.loc[((mcs2['bhcrel00'] != 7) | (mcs2['bhpsex00'] != 2)) , 'agemb2'] = np.nan
mcs2.loc[mcs2['agemb2'] < 0, 'agemb2'] = np.nan
print(mcs2['agemb2'].value_counts(dropna=False))
print(mcs2['agemb2'].value_counts().sum())

##Birth weight
mcs1['bthwt1'] = mcs1['adbwgt00'] #Derived birth weight in kgs and grams
# mcs1['bthwt1kg'] = mcs1['apwtkg00'] #Non dervied/parent reported weight in kgs and grams
# mcs1.loc[mcs1['bthwt1kg'] < 0, 'bthwt1kg'] = np.nan
# mcs1['bthwt1lb'] = mcs1['apwtlb00']*0.453592 #Non dervied/parent reported weight in pounds
# mcs1.loc[mcs1['bthwt1lb'] < 0, 'bthwt1lb'] = np.nan
# mcs1['bthwt1oz'] = mcs1['apwtou00']*0.0283495 #Non dervied/parent reported weight in ounces
# mcs1.loc[mcs1['bthwt1oz'] < 0, 'bthwt1oz'] = np.nan
# mcs1['bthwt1'] = mcs1['bthwt1kg'].add(mcs1['bthwt1lb'], fill_value=0).add(mcs1['bthwt1oz'], fill_value=0)
# mcs1.loc[mcs1['bthwt1'] < 0, 'bthwt1'] = np.nan
print(mcs1['bthwt1'].value_counts(dropna=False))
print(mcs1['bthwt1'].value_counts().sum())
print(mcs1['bthwt1'].value_counts(dropna=False).sum())
# Sweep 2
mcs2['bthwt2kg'] = mcs2['bpwtkg00'] #Non dervied/parent reported weight in kgs and grams
mcs2.loc[mcs2['bthwt2kg'] < 0, 'bthwt2kg'] = np.nan
print(mcs2['bthwt2kg'].value_counts().sum())
mcs2.loc[mcs2['bthwt2kg'] > 9, 'bthwt2kg'] = np.nan #Birth weights of 9 kgs and above discarded
print(mcs2['bthwt2kg'].value_counts(dropna=False))
print(mcs2['bthwt2kg'].value_counts().sum())
mcs2['bthwt2lb'] = mcs2['bpwtlb00']*0.453592 #Non dervied/parent reported weight in pounds
mcs2.loc[mcs2['bthwt2lb'] < 0, 'bthwt2lb'] = np.nan
mcs2['bthwt2oz'] = mcs2['bpwtou00']*0.0283495 #Non dervied/parent reported weight in ounces
mcs2.loc[mcs2['bthwt2oz'] < 0, 'bthwt2oz'] = np.nan
mcs2['bthwt2'] = mcs2['bthwt2kg'].add(mcs2['bthwt2lb'], fill_value=0).add(mcs2['bthwt2oz'], fill_value=0)
mcs2.loc[mcs2['bthwt2'] < 0, 'bthwt2'] = np.nan
mcs2['bthwt2'] = (mcs2['bthwt2'].round(2))
print(mcs2['bthwt2'].value_counts(dropna=False))
print(mcs2['bthwt2'].value_counts().sum())
print(mcs2['bthwt2'].value_counts(dropna=False).sum())

##Weeks of gestation
mcs1['gestweeks1'] = mcs1['adgest00']/7
mcs1.loc[mcs1['gestweeks1'] < 0, 'gestweeks1'] = np.nan
mcs1['gestweeks1'] = round(mcs1['gestweeks1'])
print(mcs1['gestweeks1'].value_counts(dropna=False))
print(mcs1['gestweeks1'].value_counts().sum())


#####Perinatal characterisitcs
##Smoke during pregnancy (Only taking response 1 into account to identify non-smokers)
# Do you smoke?
    # 1 No, does not smoke [exclusive code]
    # 2 Yes, cigarettes
    # 3 Yes, roll-ups
    # 4 Yes, cigars
    # 5 Yes, a pipe
    # 95 Yes, other tobacco product
mcs1['smkm1'] = mcs1['apsmus0a']
mcs1.loc[mcs1['smkm1'] == 1, 'smkm1'] = 0
mcs1.loc[mcs1['smkm1'] > 1 , 'smkm1'] = 1
mcs1.loc[mcs1['smkm1'] < 0, 'smkm1'] = np.nan
print(mcs1['smkm1'].value_counts(dropna=False))
print(mcs1['smkm1'].value_counts().sum())
#Smoke last 2 years (if not current smoker, but change to general)
    # 1 Yes
    # 2 No
mcs1['smkl2yr1'] = mcs1['apsmty00']
# mcs1['smkl2yr1'] = np.where((mcs1['smkl2yr1'].isna()),                                             
                               # 1, mcs1['smkl2yr1'])
mcs1.loc[mcs1['smkl2yr1'] == 2, 'smkl2yr1'] = 0
mcs1.loc[mcs1['smkm1'] == 1, 'smkl2yr1'] = 1
# mcs1.loc[(mcs1['smkm1'] == 1 & mcs1['smkl2yr1'] < 0), 'smkl2yr1'] = 1
mcs1.loc[mcs1['smkl2yr1'] < 0, 'smkl2yr1'] = np.nan
print(mcs1['smkl2yr1'].value_counts(dropna=False))
print(mcs1['smkl2yr1'].value_counts().sum())
#Smoked regularly last 2 years if smoked last two years (1 or more a day for 12 months or more)
    # 1 Yes
    # 2 No
mcs1['smkrgl2yr1'] = mcs1['apsmev00']
mcs1.loc[mcs1['smkrgl2yr1'] == 2, 'smkrgl2yr1'] = 0
mcs1.loc[mcs1['smkl2yr1'] == 0, 'smkrgl2yr1'] = 0
mcs1.loc[mcs1['smkrgl2yr1'] < 0, 'smkrgl2yr1'] = np.nan
print(mcs1['smkrgl2yr1'].value_counts(dropna=False))
print(mcs1['smkrgl2yr1'].value_counts().sum())
#Number smoked before prgnancy (If smoked in last two years)
mcs1['nsmkbprg'] = mcs1['apcipr00']
mcs1.loc[mcs1['smkl2yr1'] == 0, 'nsmkbprg'] = 0
mcs1.loc[mcs1['nsmkbprg'] < 0, 'nsmkbprg'] = np.nan
print(mcs1['nsmkbprg'].value_counts(dropna=False))
print(mcs1['nsmkbprg'].value_counts().sum())
#Changed number smoked during pregnancy 
    # 1 Yes
    # 2 No
    # 3 Can't remember 
mcs1['smkchpr'] = mcs1['apsmch00']
mcs1.loc[mcs1['smkchpr'] == 2, 'smkchpr'] = 0
mcs1.loc[mcs1['smkchpr'] == 3, 'smkchpr'] = np.nan
mcs1.loc[mcs1['smkchpr'] < 0, 'smkchpr'] = np.nan
# mcs1.loc[mcs1['smkchpr'] == 3, 'smkchpr'] = np.nan
print(mcs1['smkchpr'].value_counts(dropna=False))
print(mcs1['smkchpr'].value_counts().sum())
#Number smoked per day after change  (If changed smoking during pregnancy)
    # IF GAVE UP, CODE 00
    # IF LESS THAN ONE A DAY, CODE 96
    # IF CAN'T REMEMBER, CODE 97
    # IF SMOKED ROLL-UPS, ASK FOR BEST ESTIMATE
    # IF VARIED, TAKE AVERAGE
mcs1['smkchprnum'] = mcs1['apcich00']
mcs1.loc[mcs1['smkchprnum'] == 96, 'smkchprnum'] = np.nan
mcs1.loc[mcs1['smkchprnum'] == 97, 'smkchprnum'] = np.nan
mcs1.loc[mcs1['smkchprnum'] < 0, 'smkchprnum'] = np.nan
print(mcs1['smkchprnum'].value_counts(dropna=False))
print(mcs1['smkchprnum'].value_counts().sum())
#When changed smoking habits
    # 1 First
    # 2 Second
    # 3 Third
    # 4 Fourth
    # 5 Fifth
    # 6 Sixth
    # 7 Seventh
    # 8 Eighth
    # 9 Ninth
    # 10 Can't remember
mcs1['msmkch'] = mcs1['apwhch00']
mcs1.loc[mcs1['msmkch'] == 10, 'msmkch'] = np.nan
mcs1.loc[mcs1['msmkch'] < 0, 'msmkch'] = np.nan
print(mcs1['msmkch'].value_counts(dropna=False))
print(mcs1['msmkch'].value_counts().sum())
#Smoking indicator (Smoked 1 or more after 1st trimester)
# Is smoker
#   smoked in the last two years
#       did not change smoking habits during pregancy
#       changed smoking habits during pregancy
#           but did not give up smoking
#           but gave up smoking after 1st trimester      
# mcs1['smokepreg1'] = np.where(   ((mcs1['smkl2yr1'] == 1) & (mcs1['nsmkbprg']>=1) & (mcs1['smkchpr'] != 1))    
#                                | ((mcs1['smkl2yr1'] == 1) & (mcs1['nsmkbprg']>=1) & (mcs1['smkchpr'] == 1) & (mcs1['smkchprnum'] > 0))                       
#                                | ((mcs1['smkl2yr1'] == 1) & (mcs1['nsmkbprg']>=1) & (mcs1['smkchpr'] == 1) & (mcs1['smkchprnum'] == 0) & (mcs1['msmkch'] > 3)),                                             
#                                1, 0)
mcs1['smokepreg1'] = np.where(   ((mcs1['nsmkbprg']>=1) & (mcs1['smkchpr'] == 0))    
                               | ((mcs1['nsmkbprg']>=1) & (mcs1['smkchpr'] == 1) & (mcs1['smkchprnum'] > 0))                       
                               | ((mcs1['nsmkbprg']>=1) & (mcs1['smkchpr'] == 1) & (mcs1['smkchprnum'] == 0) & (mcs1['msmkch'] > 3)),                                             
                               1, 0)
mcs1.loc[mcs1['smkl2yr1'].isna(), 'smokepreg1'] = np.nan
print(mcs1['smokepreg1'].value_counts(dropna=False))
print(mcs1['smokepreg1'].value_counts().sum())

##Drink during pregnancy
#Frequency of alcohol consumption durinf pregnancy
# 1 Every day
# 2 5-6 times per week
# 3 3-4 times per week
# 4 1-2 times per week
# 5 1-2 times per month
# 6 Less than once a month
# 7 Never
mcs1['drnkfrqprg1'] = mcs1['apdrof00']
mcs1.loc[mcs1['drnkfrqprg1'] < 0, 'drnkfrqprg1'] = np.nan
print(mcs1['drnkfrqprg1'].value_counts(dropna=False))
print(mcs1['drnkfrqprg1'].value_counts().sum())
#Ever drink during pregnancy
mcs1['drinkpreg1'] = np.nan
mcs1.loc[((mcs1['drnkfrqprg1'] != 7) & (~mcs1['drnkfrqprg1'].isna())) , 'drinkpreg1'] = 1
mcs1.loc[mcs1['drnkfrqprg1'] == 7, 'drinkpreg1'] = 0
print(mcs1['drinkpreg1'].value_counts(dropna=False))
print(mcs1['drinkpreg1'].value_counts().sum())


###Post-partum characteristics 
##Breast feeding
mcs1['brstfdevr1'] = mcs1['acbfev00']
mcs1.loc[mcs1['brstfdevr1'] == 2, 'brstfdevr1'] = 0
mcs1.loc[mcs1['brstfdevr1'] < 0, 'brstfdevr1'] = np.nan
print(mcs1['brstfdevr1'].value_counts(dropna=False))
print(mcs1['brstfdevr1'].value_counts().sum())

##Maternal malaise
#Rutter
mcs1['rimm_tir'] = mcs1['aptire00']
mcs1['rimm_dep'] = mcs1['apdepr00']
mcs1['rimm_wor'] = mcs1['apworr00']
mcs1['rimm_rag'] = mcs1['aprage00']
mcs1['rimm_scr'] = mcs1['apscar00']
mcs1['rimm_ups'] = mcs1['apupse00']
mcs1['rimm_jit'] = mcs1['apkeyd00']
mcs1['rimm_ner'] = mcs1['apnerv00']
mcs1['rimm_her'] = mcs1['aphera00']
mcs1.loc[mcs1['rimm_tir'] < 0, 'rimm_tir'] = np.nan
mcs1.loc[mcs1['rimm_dep'] < 0, 'rimm_dep'] = np.nan
mcs1.loc[mcs1['rimm_wor'] < 0, 'rimm_wor'] = np.nan
mcs1.loc[mcs1['rimm_rag'] < 0, 'rimm_rag'] = np.nan
mcs1.loc[mcs1['rimm_scr'] < 0, 'rimm_scr'] = np.nan
mcs1.loc[mcs1['rimm_ups'] < 0, 'rimm_ups'] = np.nan
mcs1.loc[mcs1['rimm_jit'] < 0, 'rimm_jit'] = np.nan
mcs1.loc[mcs1['rimm_ner'] < 0, 'rimm_ner'] = np.nan
mcs1.loc[mcs1['rimm_her'] < 0, 'rimm_her'] = np.nan
mcs1['rimm1'] = (mcs1['rimm_tir'] + mcs1['rimm_dep'] + mcs1['rimm_wor'] + mcs1['rimm_rag'] 
                + mcs1['rimm_scr'] + mcs1['rimm_ups'] + mcs1['rimm_jit'] + mcs1['rimm_ner'] 
                + mcs1['rimm_her'])
print(mcs1['rimm1'].value_counts(dropna=False))
print(mcs1['rimm1'].value_counts().sum())
mcs1['malm1'] = 18 - mcs1['rimm1']
print(mcs1['malm1'].value_counts(dropna=False))
print(mcs1['malm1'].value_counts().sum())
#Maternal malaise clinical indicator
mcs1['malm_clin1'] = np.where((mcs1['malm1'] >= 4), 1, 0)
mcs1.loc[mcs1['rimm1'].isna(), 'malm_clin1'] = np.nan
print(mcs1['malm_clin1'].value_counts(dropna=False))
print(mcs1['malm_clin1'].value_counts().sum())


##Maternal attachment - Condon Maternal Attachment Questionnaire
mcs1['cma_ann'] = mcs1['apanno00']
mcs1['cma_thn'] = mcs1['apthnk00']
mcs1['cma_lev'] = mcs1['apleav00']
mcs1['cma_cmp'] = mcs1['apcomp00']
mcs1['cma_pat'] = mcs1['appati00']
mcs1['cma_gip'] = mcs1['apgiup00']
mcs1.loc[mcs1['cma_ann'] < 0, 'cma_ann'] = np.nan
mcs1.loc[mcs1['cma_thn'] < 0, 'cma_thn'] = np.nan
mcs1.loc[mcs1['cma_lev'] < 0, 'cma_lev'] = np.nan
mcs1.loc[mcs1['cma_cmp'] < 0, 'cma_cmp'] = np.nan
mcs1.loc[mcs1['cma_pat'] < 0, 'cma_pat'] = np.nan
mcs1.loc[mcs1['cma_gip'] < 0, 'cma_gip'] = np.nan
mcs1['cma1'] = (mcs1['cma_ann'] + mcs1['cma_thn'] + mcs1['cma_lev'] + mcs1['cma_cmp'] 
                + mcs1['cma_pat'] + mcs1['cma_gip'])
print(mcs1['cma1'].value_counts(dropna=False))
print(mcs1['cma1'].value_counts().sum())

## Infant temperment
#aphapn00 apunfa00 apbrus00 apfeed00 apinju00 ##Pleasent and content
#apbath00 apwary00 apbshy00 apfret00 apslee00 ##Fretful bothered/shy
#apmilk00 apslti00 apnaps00 apsofo00 ##Consistent with feeding and sleeping

###Sure Start (Main parent only)
#Heard of sure start
mcs1['hsrstrt1'] = mcs1['apsust00']
mcs1.loc[mcs1['hsrstrt1'] < 0, 'hsrstrt1'] = np.nan
print(mcs1['hsrstrt1'].value_counts(dropna=False))
print(mcs1['hsrstrt1'].value_counts().sum())
mcs1['hsrstrt1'] = np.where((mcs1['hsrstrt1'] == 2) | (mcs1['hsrstrt1'] == 3), 0, mcs1['hsrstrt1'])
print(mcs1['hsrstrt1'].value_counts(dropna=False))
print(mcs1['hsrstrt1'].value_counts().sum())
#Used sure start
mcs1['usrstrt1'] = mcs1['apusst00']
print(mcs1['usrstrt1'].value_counts(dropna=False))
print(mcs1['usrstrt1'].value_counts().sum())
mcs1['usrstrt1'] = np.where((mcs1['usrstrt1'] == 1), 1, 0)
mcs1['usrstrt1'] = np.where((mcs1['hsrstrt1'].isna()), np.nan, mcs1['usrstrt1'])
print(mcs1['usrstrt1'].value_counts(dropna=False))
print(mcs1['usrstrt1'].value_counts().sum())

##### Family/Household characteristics
##Biological parents
#Sweep 1
#Main parent
mcs1['biom1'] = np.where(mcs1['ahcrel00'] == 7, 1, 0)
print(mcs1['biom1'].value_counts(dropna=False))
print(mcs1['biom1'].value_counts().sum())
#Partner
mcs1['biop1'] = np.where(mcs1['ahcrel00_p'] == 7, 1, 0)
print(mcs1['biop1'].value_counts(dropna=False))
print(mcs1['biop1'].value_counts().sum())
# Sweep 2
#Main parent
mcs2['biom2'] = np.where(mcs2['bhcrel00'] == 7, 1, 0)
print(mcs2['biom2'].value_counts(dropna=False))
print(mcs2['biom2'].value_counts().sum())
#Partner
mcs2['biop2'] = np.where(mcs2['bhcrel00_p'] == 7, 1, 0)
print(mcs2['biop2'].value_counts(dropna=False))
print(mcs2['biop2'].value_counts().sum())


#Parent's Ethnicity - 6 category 
#Sweep1
#Main parent's
mcs1['methnicity1'] = mcs1['add06e00']
mcs1.loc[mcs1['methnicity1'] < 0, 'methnicity1'] = np.nan
print(mcs1['methnicity1'].value_counts(dropna=False))
print(mcs1['methnicity1'].value_counts().sum())
#Partner's
mcs1['pethnicity1'] = mcs1['add06e00_p']
mcs1.loc[mcs1['pethnicity1'] < 0, 'pethnicity1'] = np.nan
print(mcs1['pethnicity1'].value_counts(dropna=False))
print(mcs1['pethnicity1'].value_counts().sum())
#Sweep 2
#Main parent's
mcs2['methnicity2'] = mcs2['bdd06e00']
mcs2.loc[mcs2['methnicity2'] < 0, 'methnicity2'] = np.nan
print(mcs2['methnicity2'].value_counts(dropna=False))
print(mcs2['methnicity2'].value_counts().sum())
#Partner's
mcs2['pethnicity2'] = mcs2['bdd06e00_p']
mcs2.loc[mcs2['pethnicity2'] < 0, 'pethnicity2'] = np.nan
print(mcs2['pethnicity2'].value_counts(dropna=False))
print(mcs2['pethnicity2'].value_counts().sum())
#Convert from 8 category to 6
# 1                                 White
# 2                                 Mixed
# 3                                Indian
# 4                             Pakistani
# 5                           Bangladeshi
# 6                       Black Caribbean
# 7                         Black African
# 8 Other Ethnic group (inc Chinese,Other)


##Pianta scale Conflicts
#Sweep 2
#Main parent
mcs2['psconm2'] = mcs2['bmpnta0000']
mcs2.loc[mcs2['psconm2'] < 0, 'psconm2'] = np.nan
print(mcs2['psconm2'].value_counts(dropna=False))
print(mcs2['psconm2'].value_counts().sum())
#Partner
mcs2['psconp2'] = mcs2['bppnta0000']
mcs2.loc[mcs2['psconp2'] < 0, 'psconp2'] = np.nan
print(mcs2['psconp2'].value_counts(dropna=False))
print(mcs2['psconp2'].value_counts().sum())

##Pianta scale closeness
#Sweep 2
#Main parent
mcs2['psclom2'] = mcs2['bmpntc0000']
mcs2.loc[mcs2['psclom2'] < 0, 'psclom2'] = np.nan
print(mcs2['psclom2'].value_counts(dropna=False))
print(mcs2['psclom2'].value_counts().sum())
#Partner
mcs2['psclop2'] = mcs2['bppntc0000']
mcs2.loc[mcs2['psclop2'] < 0, 'psclop2'] = np.nan
print(mcs2['psclop2'].value_counts(dropna=False))
print(mcs2['psclop2'].value_counts().sum())

## Home learning environment
#https://cls.ucl.ac.uk/wp-content/uploads/2017/07/FINAL-The-home-learning-environment-as-measured-in-the-MCS-at-age-3-MCS-data-note-1.pdf
#Sweep 2
#Read
#Frequecy parents read
mcs2['hlemfrd2'] = mcs2['bpofre00']
mcs2.loc[mcs2['hlemfrd2'] < 0, 'hlemfrd2'] = np.nan
print(mcs2['hlemfrd2'].value_counts(dropna=False))
print(mcs2['hlemfrd2'].value_counts().sum())
#Frequecy anyone read
mcs2['hleafrd2'] = mcs2['bpreof00']
mcs2.loc[mcs2['hleafrd2'] < 0, 'hleafrd2'] = np.nan
print(mcs2['hleafrd2'].value_counts(dropna=False))
print(mcs2['hleafrd2'].value_counts().sum())
#Composite read measure
mcs2['hlerd2'] = mcs2['hlemfrd2']
mcs2.loc[(mcs2['hleafrd2'] < mcs2['hlemfrd2']) & (mcs2['hleafrd2'].notna()) & (mcs2['hlemfrd2'].notna()), 'hlerd2'] = mcs2['hleafrd2']
# additional replacements with specific conditions
mcs2.loc[(mcs2['hlemfrd2'] == 2) & (mcs2['hleafrd2'] == 2) & (mcs2['hleafrd2'].notna()) & (mcs2['hlemfrd2'].notna()), 'hlerd2'] = 1
mcs2.loc[(mcs2['hlemfrd2'] == 3) & (mcs2['hleafrd2'] == 3) & (mcs2['hleafrd2'].notna()) & (mcs2['hlemfrd2'].notna()), 'hlerd2'] = 2
print(mcs2['hlerd2'].value_counts(dropna=False))
print(mcs2['hlerd2'].value_counts().sum())
# Recode values using a dictionary mapping old values to new ones
rdfrq = {
    6: 0,
    5: 1,
    4: 3,
    3: 5,
    2: 6,
    1: 7,
    np.nan: np.nan
}
# Apply the recode map to create a new column with the recoded values
mcs2['hlerd2'] = mcs2['hlerd2'].map(lambda x: rdfrq.get(x))
del rdfrq
#Library
#Anyone take
mcs2['hlealib2'] = mcs2['bptoli00']
mcs2.loc[mcs2['hlealib2'] < 0, 'hlealib2'] = np.nan
print(mcs2['hlealib2'].value_counts(dropna=False))
print(mcs2['hlealib2'].value_counts().sum())
#Frequency
mcs2['hleflib2'] = mcs2['bpofli00']
mcs2.loc[mcs2['hleflib2'] < 0, 'hleflib2'] = np.nan
print(mcs2['hleflib2'].value_counts(dropna=False))
print(mcs2['hleflib2'].value_counts().sum())
#Actual measure
mcs2['hlelib2'] = mcs2['hleflib2']
mcs2.loc[mcs2['hlealib2'] == 2, 'hlelib2'] = 0
print(mcs2['hlelib2'].value_counts(dropna=False))
print(mcs2['hlelib2'].value_counts().sum())
#Alphabet
#Anyone take
mcs2['hleaalp2'] = mcs2['bpalph00']
mcs2.loc[mcs2['hleaalp2'] < 0, 'hleaalp2'] = np.nan
print(mcs2['hleaalp2'].value_counts(dropna=False))
print(mcs2['hleaalp2'].value_counts().sum())
#Frequency
mcs2['hlefalp2'] = mcs2['bpofab00']
mcs2.loc[mcs2['hlefalp2'] < 0, 'hlefalp2'] = np.nan
print(mcs2['hlefalp2'].value_counts(dropna=False))
print(mcs2['hlefalp2'].value_counts().sum())
#Actual measure
mcs2['hlealp2'] = mcs2['hlefalp2']
mcs2.loc[mcs2['hleaalp2'] == 2, 'hlealp2'] = 0
print(mcs2['hlealp2'].value_counts(dropna=False))
print(mcs2['hlealp2'].value_counts().sum())
#Counting
#Anyone take
mcs2['hleacnt2'] = mcs2['bpnumb00']
mcs2.loc[mcs2['hleacnt2'] < 0, 'hleacnt2'] = np.nan
print(mcs2['hleacnt2'].value_counts(dropna=False))
print(mcs2['hleacnt2'].value_counts().sum())
#Frequency
mcs2['hlefcnt2'] = mcs2['bpofco00']
mcs2.loc[mcs2['hlefcnt2'] < 0, 'hlefcnt2'] = np.nan
print(mcs2['hlefcnt2'].value_counts(dropna=False))
print(mcs2['hlefcnt2'].value_counts().sum())
#Actual measure
mcs2['hlecnt2'] = mcs2['hlefcnt2']
mcs2.loc[mcs2['hleacnt2'] == 2, 'hlecnt2'] = 0
print(mcs2['hlecnt2'].value_counts(dropna=False))
print(mcs2['hlecnt2'].value_counts().sum())
#Songs
#Anyone take
mcs2['hleasng2'] = mcs2['bpsong00']
mcs2.loc[mcs2['hleasng2'] < 0, 'hleasng2'] = np.nan
print(mcs2['hleasng2'].value_counts(dropna=False))
print(mcs2['hleasng2'].value_counts().sum())
#Frequency
mcs2['hlefsng2'] = mcs2['bpofso00']
mcs2.loc[mcs2['hlefsng2'] < 0, 'hlefsng2'] = np.nan
print(mcs2['hlefsng2'].value_counts(dropna=False))
print(mcs2['hlefsng2'].value_counts().sum())
#Actual measure
mcs2['hlesng2'] = mcs2['hlefsng2']
mcs2.loc[mcs2['hleasng2'] == 2, 'hlesng2'] = 0
print(mcs2['hlesng2'].value_counts(dropna=False))
print(mcs2['hlesng2'].value_counts().sum())
#Drawing
#Anyone take
mcs2['hleadrw2'] = mcs2['bpdraw00']
mcs2.loc[mcs2['hleadrw2'] < 0, 'hleadrw2'] = np.nan
print(mcs2['hleadrw2'].value_counts(dropna=False))
print(mcs2['hleadrw2'].value_counts().sum())
#Frequency
mcs2['hlefdrw2'] = mcs2['bppama00']
mcs2.loc[mcs2['hlefdrw2'] < 0, 'hlefdrw2'] = np.nan
print(mcs2['hlefdrw2'].value_counts(dropna=False))
print(mcs2['hlefdrw2'].value_counts().sum())
#Actual measure
mcs2['hledrw2'] = mcs2['hlefdrw2']
mcs2.loc[mcs2['hleadrw2'] == 2, 'hledrw2'] = 0
print(mcs2['hledrw2'].value_counts(dropna=False))
print(mcs2['hledrw2'].value_counts().sum())
#Combined measure
mcs2['hle2'] = mcs2['hlerd2'] + mcs2['hlelib2'] + mcs2['hlealp2'] + mcs2['hlecnt2'] + mcs2['hlesng2'] + mcs2['hledrw2']
print(mcs2['hle2'].value_counts(dropna=False))
print(mcs2['hle2'].value_counts().sum())
#Sweep 3
hleval = {
        6: 0,
        5: 1,
        4: 2,
        3: 3,
        2: 4,
        1: 5
    }
#Reading
#Anyone help
mcs3['hleard3'] = mcs3['cpalrd00']
mcs3.loc[mcs3['hleard3'] < 0, 'hleard3'] = np.nan
print(mcs3['hleard3'].value_counts(dropna=False))
print(mcs3['hleard3'].value_counts().sum())
#Frequency
mcs3['hlefrd3'] = mcs3['cpalwh00']
mcs3.loc[mcs3['hlefrd3'] < 0, 'hlefrd3'] = np.nan
print(mcs3['hlefrd3'].value_counts(dropna=False))
print(mcs3['hlefrd3'].value_counts().sum())
#Actual measure
mcs3['hlerd3'] = mcs3['hlefrd3'].map(lambda x: hleval.get(x))
mcs3.loc[mcs3['hleard3'] == 2, 'hlerd3'] = 0
print(mcs3['hlerd3'].value_counts(dropna=False))
print(mcs3['hlerd3'].value_counts().sum())
#Writing
#Anyone help
mcs3['hleawrt3'] = mcs3['cphlwr00']
mcs3.loc[mcs3['hleawrt3'] < 0, 'hleawrt3'] = np.nan
print(mcs3['hleawrt3'].value_counts(dropna=False))
print(mcs3['hleawrt3'].value_counts().sum())
#Frequency
mcs3['hlefwrt3'] = mcs3['cphlwx00']
mcs3.loc[mcs3['hlefwrt3'] < 0, 'hlefwrt3'] = np.nan
print(mcs3['hlefwrt3'].value_counts(dropna=False))
print(mcs3['hlefwrt3'].value_counts().sum())
#Actual measure
mcs3['hlewrt3'] = mcs3['hlefwrt3'].map(lambda x: hleval.get(x))
mcs3.loc[mcs3['hleawrt3'] == 2, 'hlewrt3'] = 0
print(mcs3['hlewrt3'].value_counts(dropna=False))
print(mcs3['hlewrt3'].value_counts().sum())
#Maths
#Anyone help
mcs3['hleamth3'] = mcs3['cphlco00']
mcs3.loc[mcs3['hleamth3'] < 0, 'hleamth3'] = np.nan
print(mcs3['hleamth3'].value_counts(dropna=False))
print(mcs3['hleamth3'].value_counts().sum())
#Frequency
mcs3['hlefmth3'] = mcs3['cphlnc00']
mcs3.loc[mcs3['hlefmth3'] < 0, 'hlefmth3'] = np.nan
print(mcs3['hlefmth3'].value_counts(dropna=False))
print(mcs3['hlefmth3'].value_counts().sum())
#Actual measure
mcs3['hlemth3'] = mcs3['hlefmth3'].map(lambda x: hleval.get(x))
mcs3.loc[mcs3['hleamth3'] == 2, 'hlemth3'] = 0
print(mcs3['hlemth3'].value_counts(dropna=False))
print(mcs3['hlemth3'].value_counts().sum())
#Combined measure
#Combined measure
mcs3['hle3'] = mcs3['hlerd3'] + mcs3['hlewrt3'] + mcs3['hlemth3']
print(mcs3['hle3'].value_counts(dropna=False))
print(mcs3['hle3'].value_counts().sum())

del hleval

## Number of siblings in household including CM
# Sweep 1
mcs1['numchld1'] = mcs1['adtots00']
mcs1.loc[mcs1['numchld1'] < 0, 'numchld1'] = np.nan
print(mcs1['numchld1'].value_counts(dropna=False))
print(mcs1['numchld1'].value_counts().sum())
# Sweep 2
mcs2['numchld2'] = mcs2['bdtots00']
mcs2.loc[mcs2['numchld2'] < 0, 'numchld2'] = np.nan
print(mcs2['numchld2'].value_counts(dropna=False))
print(mcs2['numchld2'].value_counts().sum())
# Sweep 3
mcs3['numchld3'] = mcs3['cdtots00']
mcs3.loc[mcs3['numchld3'] < 0, 'numchld3'] = np.nan
print(mcs3['numchld3'].value_counts(dropna=False))
print(mcs3['numchld3'].value_counts().sum())
# Sweep 4
mcs4['numchld4'] = mcs4['ddtots00']
mcs4.loc[mcs4['numchld4'] < 0, 'numchld4'] = np.nan
print(mcs4['numchld4'].value_counts(dropna=False))
print(mcs4['numchld4'].value_counts().sum())
# Sweep 5
mcs5['numchld5'] = mcs5['etots00']
mcs5.loc[mcs5['numchld5'] < 0, 'numchld5'] = np.nan
print(mcs5['numchld5'].value_counts(dropna=False))
print(mcs5['numchld5'].value_counts().sum())
# Sweep 6
mcs6['numchld6'] = mcs6['fdtots00']
mcs6.loc[mcs6['numchld6'] < 0, 'numchld6'] = np.nan
print(mcs6['numchld6'].value_counts(dropna=False))
print(mcs6['numchld6'].value_counts().sum())
# # Sweep 7
mcs7['numchld7'] = mcs7['gdtots00']
mcs7.loc[mcs7['numchld7'] < 0, 'numchld7'] = np.nan
print(mcs7['numchld7'].value_counts(dropna=False))
print(mcs7['numchld7'].value_counts().sum())


##Number of people in household including CM
# Sweep 1
mcs1['numppl1'] = mcs1['adtotp00']
mcs1.loc[mcs1['numppl1'] < 0, 'numppl1'] = np.nan
print(mcs1['numppl1'].value_counts(dropna=False))
print(mcs1['numppl1'].value_counts().sum())
# Sweep 2
mcs2['numppl2'] = mcs2['bdtotp00']
mcs2.loc[mcs2['numppl2'] < 0, 'numppl2'] = np.nan
print(mcs2['numppl2'].value_counts(dropna=False))
print(mcs2['numppl2'].value_counts().sum())
# Sweep 3
mcs3['numppl3'] = mcs3['cdtotp00']
mcs3.loc[mcs3['numppl3'] < 0, 'numppl3'] = np.nan
print(mcs3['numppl3'].value_counts(dropna=False))
print(mcs3['numppl3'].value_counts().sum())
# Sweep 4
mcs4['numppl4'] = mcs4['ddtotp00']
mcs4.loc[mcs4['numppl4'] < 0, 'numppl4'] = np.nan
print(mcs4['numppl4'].value_counts(dropna=False))
print(mcs4['numppl4'].value_counts().sum())
# Sweep 5
mcs5['numppl5'] = mcs5['etotp00']
mcs5.loc[mcs5['numppl5'] < 0, 'numppl5'] = np.nan
print(mcs5['numppl5'].value_counts(dropna=False))
print(mcs5['numppl5'].value_counts().sum())
# Sweep 6
mcs6['numppl6'] = mcs6['fdtotp00']
mcs6.loc[mcs6['numppl6'] < 0, 'numppl6'] = np.nan
print(mcs6['numppl6'].value_counts(dropna=False))
print(mcs6['numppl6'].value_counts().sum())
# # Sweep 7
mcs7['numppl7'] = mcs7['gdtotp00']
mcs7.loc[mcs7['numppl7'] < 0, 'numppl7'] = np.nan
print(mcs7['numppl7'].value_counts(dropna=False))
print(mcs7['numppl7'].value_counts().sum())

## Number of parent/carers
# Sweep 1
mcs1['snglprnt1'] = np.where(mcs1['adhtys00'] == 2, 1, 0)
mcs1.loc[mcs1['adhtys00'] < 0, 'snglprnt1'] = np.nan
print(mcs1['snglprnt1'].value_counts(dropna=False))
print(mcs1['snglprnt1'].value_counts().sum())
# Sweep 2
mcs2['snglprnt2'] = np.where(mcs2['bdhtys00'] == 2, 1, 0)
mcs2.loc[mcs2['bdhtys00'] < 0, 'snglprnt2'] = np.nan
mcs2.loc[mcs2['bdhtys00'].isnull(), 'snglprnt2'] = np.nan
print(mcs2['snglprnt2'].value_counts(dropna=False))
print(mcs2['snglprnt2'].value_counts().sum())
# Sweep 3
mcs3['snglprnt3'] = np.where(mcs3['cdhtys00'] == 2, 1, 0)
mcs3.loc[mcs3['cdhtys00'] < 0, 'snglprnt3'] = np.nan
print(mcs3['snglprnt3'].value_counts(dropna=False))
print(mcs3['snglprnt3'].value_counts().sum())
# Sweep 4
mcs4['snglprnt4'] = np.where(mcs4['ddhtys00'] == 2, 1, 0)
mcs4.loc[mcs4['ddhtys00'] < 0, 'snglprnt4'] = np.nan
print(mcs4['snglprnt4'].value_counts(dropna=False))
print(mcs4['snglprnt4'].value_counts().sum())
# Sweep 5
mcs5['snglprnt5'] = np.where(mcs5['ehtys00'] == 2, 1, 0)
mcs5.loc[mcs5['ehtys00'] < 0, 'snglprnt5'] = np.nan
print(mcs5['snglprnt5'].value_counts(dropna=False))
print(mcs5['snglprnt5'].value_counts().sum())
# Sweep 6
mcs6['snglprnt6'] = np.where(mcs6['fdhtys00'] == 2, 1, 0)
mcs6.loc[mcs6['fdhtys00'] < 0, 'snglprnt6'] = np.nan
print(mcs6['snglprnt6'].value_counts(dropna=False))
print(mcs6['snglprnt6'].value_counts().sum())
# Sweep 7
mcs7['snglprnt7'] = np.where(mcs7['gdhtys00'] == 2, 1, 0)
mcs7.loc[mcs7['gdhtys00'] < 0, 'snglprnt7'] = np.nan
print(mcs7['snglprnt7'].value_counts(dropna=False))
print(mcs7['snglprnt7'].value_counts().sum())

## Time Spent with child
# Sweep 1
#                       Refusal   -9
#                     Dont Know   -8
#                Not applicable   -1    
# ... plenty of time with baby     1
#                  just enough     2
#             not quite enough     3
#      or, nowhere near enough     4
#                    (Not sure)    5
#Main parent
mcs1['tmspntchldm1'] = mcs1['apchti00'] # Usual hours
mcs1.loc[mcs1['tmspntchldm1'] < 0, 'tmspntchldm1'] = np.nan
mcs1.loc[mcs1['tmspntchldm1'] == 5, 'tmspntchldm1'] = np.nan
print(mcs1['tmspntchldm1'].value_counts(dropna=False))
print(mcs1['tmspntchldm1'].value_counts().sum())
#Partner
mcs1['tmspntchldp1'] = mcs1['apchti00_p'] # Usual hours
mcs1.loc[mcs1['tmspntchldp1'] < 0, 'tmspntchldp1'] = np.nan
mcs1.loc[mcs1['tmspntchldp1'] == 5, 'tmspntchldp1'] = np.nan
print(mcs1['tmspntchldp1'].value_counts(dropna=False))
print(mcs1['tmspntchldp1'].value_counts().sum())
# Sweep 2
#Main parent
mcs2['tmspntchldm2'] = mcs2['bpchti00'] # Usual hours
mcs2.loc[mcs2['tmspntchldm2'] < 0, 'tmspntchldm2'] = np.nan
mcs2.loc[mcs2['tmspntchldm2'] == 5, 'tmspntchldm2'] = np.nan
print(mcs2['tmspntchldm2'].value_counts(dropna=False))
print(mcs2['tmspntchldm2'].value_counts().sum())
#Partner
mcs2['tmspntchldp2'] = mcs2['bpchti00_p'] # Usual hours
mcs2.loc[mcs2['tmspntchldp2'] < 0, 'tmspntchldp2'] = np.nan
mcs2.loc[mcs2['tmspntchldp2'] == 5, 'tmspntchldp2'] = np.nan
print(mcs2['tmspntchldp2'].value_counts(dropna=False))
print(mcs2['tmspntchldp2'].value_counts().sum())
# Sweep 3
#                  Refusal   -9
#               Don't know   -8
#           Not applicable   -1
#            Too much time    1
#    More than enough time    2
#         Just enough time    3
#    Not quite enough time    4
# Nowhere near enough time    5
#               (Not sure)    6
#Main parent
mcs3['tmspntchldm3'] = mcs3['cpchti00'] # Usual hours
mcs3.loc[mcs3['tmspntchldm3'] < 0, 'tmspntchldm3'] = np.nan
mcs3.loc[mcs3['tmspntchldm3'] == 5, 'tmspntchldm3'] = np.nan
print(mcs3['tmspntchldm3'].value_counts(dropna=False))
print(mcs3['tmspntchldm3'].value_counts().sum())
#Partner
mcs3['tmspntchldp3'] = mcs3['cpchti00_p'] # Usual hours
mcs3.loc[mcs3['tmspntchldp3'] < 0, 'tmspntchldp3'] = np.nan
mcs3.loc[mcs3['tmspntchldp3'] == 5, 'tmspntchldp3'] = np.nan
print(mcs3['tmspntchldp3'].value_counts(dropna=False))
print(mcs3['tmspntchldp3'].value_counts().sum())
# Sweep 4
#Main parent
mcs4['tmspntchldm4'] = mcs4['dpchti00'] # Usual hours
mcs4.loc[mcs4['tmspntchldm4'] < 0, 'tmspntchldm4'] = np.nan
mcs4.loc[mcs4['tmspntchldm4'] == 5, 'tmspntchldm4'] = np.nan
print(mcs4['tmspntchldm4'].value_counts(dropna=False))
print(mcs4['tmspntchldm4'].value_counts().sum())
#Partner
mcs4['tmspntchldp4'] = mcs4['dpchti00_p'] # Usual hours
mcs4.loc[mcs4['tmspntchldp4'] < 0, 'tmspntchldp4'] = np.nan
mcs4.loc[mcs4['tmspntchldp4'] == 5, 'tmspntchldp4'] = np.nan
print(mcs4['tmspntchldp4'].value_counts(dropna=False))
print(mcs4['tmspntchldp4'].value_counts().sum())
# Sweep 5
#Main parent
mcs5['tmspntchldm5'] = mcs5['epchti00'] # Usual hours
mcs5.loc[mcs5['tmspntchldm5'] < 0, 'tmspntchldm5'] = np.nan
mcs5.loc[mcs5['tmspntchldm5'] == 6, 'tmspntchldm5'] = np.nan
print(mcs5['tmspntchldm5'].value_counts(dropna=False))
print(mcs5['tmspntchldm5'].value_counts().sum())
#Partner
mcs5['tmspntchldp5'] = mcs5['epchti00_p'] # Usual hours
mcs5.loc[mcs5['tmspntchldp5'] < 0, 'tmspntchldp5'] = np.nan
mcs5.loc[mcs5['tmspntchldp5'] == 6, 'tmspntchldp5'] = np.nan
print(mcs5['tmspntchldp5'].value_counts(dropna=False))
print(mcs5['tmspntchldp5'].value_counts().sum())
# # Sweep 6
# #Main parent
# mcs6['tmspntchldm6'] = mcs6['fpchti00'] # Usual hours
# print(mcs6['tmspntchldm6'].value_counts(dropna=False))
# print(mcs6['tmspntchldm6'].value_counts().sum())
# #Partner
# mcs6['tmspntchldp6'] = mcs6['fpchti00_p'] # Usual hours
# print(mcs6['tmspntchldp6'].value_counts(dropna=False))
# print(mcs6['tmspntchldp6'].value_counts().sum())
# # Sweep 7
# #Main parent
# mcs7['tmspntchldm7'] = mcs7['gpchti00'] # Usual hours
# print(mcs7['tmspntchldm7'].value_counts(dropna=False))
# print(mcs7['tmspntchldm7'].value_counts().sum())
# #Partner
# mcs7['tmspntchldp7'] = mcs7['gpchti00_p'] # Usual hours
# print(mcs7['tmspntchldp7'].value_counts(dropna=False))
# print(mcs7['tmspntchldp7'].value_counts().sum())

##Parental Labour force status
    # -1 - Not applicable
    # 1  - Both in work
    # 2  - Main in work, partner not
    # 3  - Partner in work, main not
    # 4  - Both not in work
    # 5  - Main in work or on leave, no partner
    # 6  - Main not on work nor on leave, no partner
    # 9  - Main in work, partner status unknown
    # 10 - Main not in work, partner status unknown
    
# Any labour force participation
lmsgrp1 = [1, 2, 3, 5, 9]
#No known Labour force paticipation
lmsgrp2 = [-1, 4, 6, 10]
# Create age category columns
# Sweep 1
mcs1['nplfp1'] = np.where(mcs1['adcwrk00'].isin(lmsgrp2), 1, 0)
mcs1.loc[mcs1['adcwrk00'] < 0, 'nplfp1'] = np.nan
mcs1.loc[mcs1['adcwrk00'].isnull(), 'nplfp1'] = np.nan
print(mcs1['nplfp1'].value_counts(dropna=False))
mcs1['aplfp1'] = np.where(mcs1['adcwrk00'].isin(lmsgrp1), 1, 0)
mcs1.loc[mcs1['adcwrk00'] < 0, 'aplfp1'] = np.nan
mcs1.loc[mcs1['adcwrk00'].isnull(), 'aplfp1'] = np.nan
print(mcs1['aplfp1'].value_counts(dropna=False))
# Sweep 2
mcs2['nplfp2'] = np.where(mcs2['bdcwrk00'].isin(lmsgrp2), 1, 0)
mcs2.loc[mcs2['bdcwrk00'] < 0, 'nplfp2'] = np.nan
mcs2.loc[mcs2['bdcwrk00'].isnull(), 'nplfp2'] = np.nan
print(mcs2['nplfp2'].value_counts(dropna=False))
mcs2['aplfp2'] = np.where(mcs2['bdcwrk00'].isin(lmsgrp1), 1, 0)
mcs2.loc[mcs2['bdcwrk00'] < 0, 'aplfp2'] = np.nan
mcs2.loc[mcs2['bdcwrk00'].isnull(), 'aplfp2'] = np.nan
print(mcs2['aplfp2'].value_counts(dropna=False))
# Sweep 3
mcs3['nplfp3'] = np.where(mcs3['cdcwrk00'].isin(lmsgrp2), 1, 0)
mcs3.loc[mcs3['cdcwrk00'] < 0, 'nplfp3'] = np.nan
mcs3.loc[mcs3['cdcwrk00'].isnull(), 'nplfp3'] = np.nan
print(mcs3['nplfp3'].value_counts(dropna=False))
mcs3['aplfp3'] = np.where(mcs3['cdcwrk00'].isin(lmsgrp1), 1, 0)
mcs3.loc[mcs3['cdcwrk00'] < 0, 'aplfp3'] = np.nan
mcs3.loc[mcs3['cdcwrk00'].isnull(), 'aplfp3'] = np.nan
print(mcs3['aplfp3'].value_counts(dropna=False))
# Sweep 4
mcs4['nplfp4'] = np.where(mcs4['ddcwrk00'].isin(lmsgrp2), 1, 0)
mcs4.loc[mcs4['ddcwrk00'] < 0, 'nplfp4'] = np.nan
mcs4.loc[mcs4['ddcwrk00'].isnull(), 'nplfp4'] = np.nan
print(mcs4['nplfp4'].value_counts(dropna=False))
mcs4['aplfp4'] = np.where(mcs4['ddcwrk00'].isin(lmsgrp1), 1, 0)
mcs4.loc[mcs4['ddcwrk00'] < 0, 'aplfp4'] = np.nan
mcs4.loc[mcs4['ddcwrk00'].isnull(), 'aplfp4'] = np.nan
print(mcs4['aplfp4'].value_counts(dropna=False))
# Sweep 5
mcs5['nplfp5'] = np.where(mcs5['ecwrk00'].isin(lmsgrp2), 1, 0)
mcs5.loc[mcs5['ecwrk00'] < 0, 'nplfp5'] = np.nan
mcs5.loc[mcs5['ecwrk00'].isnull(), 'nplfp5'] = np.nan
print(mcs5['nplfp5'].value_counts(dropna=False))
mcs5['aplfp5'] = np.where(mcs5['ecwrk00'].isin(lmsgrp1), 1, 0)
mcs5.loc[mcs5['ecwrk00'] < 0, 'aplfp5'] = np.nan
mcs5.loc[mcs5['ecwrk00'].isnull(), 'aplfp5'] = np.nan
print(mcs5['aplfp5'].value_counts(dropna=False))
# Sweep 6
mcs6['nplfp6'] = np.where(mcs6['fdcwrk00'].isin(lmsgrp2), 1, 0)
mcs6.loc[mcs6['fdcwrk00'] < 0, 'nplfp6'] = np.nan
mcs6.loc[mcs6['fdcwrk00'].isnull(), 'nplfp6'] = np.nan
print(mcs6['nplfp6'].value_counts(dropna=False))
mcs6['aplfp6'] = np.where(mcs6['fdcwrk00'].isin(lmsgrp1), 1, 0)
mcs6.loc[mcs6['fdcwrk00'] < 0, 'aplfp6'] = np.nan
mcs6.loc[mcs6['fdcwrk00'].isnull(), 'aplfp6'] = np.nan
print(mcs6['aplfp6'].value_counts(dropna=False))
# # Sweep 7
# mcs7['nplfp7'] = np.where(mcs7['gdcwrk00'].isin(lmsgrp2), 1, 0)
# mcs7.loc[mcs7['gdcwrk00'] < 0, 'nplfp7'] = np.nan
# mcs7.loc[mcs7['gdcwrk00'].isnull(), 'nplfp7'] = np.nan
# print(mcs7['nplfp7'].value_counts(dropna=False))
# mcs7['aplfp7'] = np.where(mcs7['gdcwrk00'].isin(lmsgrp1), 1, 0)
# mcs7.loc[mcs7['gdcwrk00'] < 0, 'aplfp7'] = np.nan
# mcs7.loc[mcs7['gdcwrk00'].isnull(), 'aplfp7'] = np.nan
# print(mcs7['aplfp7'].value_counts(dropna=False))

del lmsgrp1
del lmsgrp2

##Parental disability living allowance or disabled person's tax credit
# Sweep 1
    # -1 - Not applicable
    # 4  - Disabled Persons Tax Credit
    # 10 - Invalid Care Allowance
    # 12 - Disability Living Allowance
    # 13 - Incapacity Benefit
    # 51 - Other/Severe disablement allowance
    # 52 - Other/Industrial injuries benefit
    # 53 - Other/Statutory sick pay
    # 85 - Other answer (not codeable 1-17, 51-53) 
# Any Disability
dislst = [4, 12, 51]
varlst =['apstwm0a', 'apstwm0b', 'apstwm0c', 'apstwm0d', 'apstwm0e', 'apstwm0f', 'apstwm0g', 'apstwm0h']
mcs1['hhdis1'] = 0
for var in varlst:
    mcs1['hhdis1'] = np.where(mcs1[var].isin(dislst), 1, mcs1['hhdis1'])
    # mcs1.loc[mcs1[var].isnull(), 'hhdis1'] = np.nan
print(mcs1['hhdis1'].value_counts(dropna=False))
# Sweep 2
varlst =['bpstbm0e', 'bpstbm0m', 'bpstbm0s']
mcs2['hhdis2'] = 0
for var in varlst:
    mcs2['hhdis2'] = np.where(mcs2[var] == 1, 1, mcs2['hhdis2'])
print(mcs2['hhdis2'].value_counts(dropna=False))
# Sweep 3
    # -9 - Refusal
    # -8 - Don't Know
    # -1 - Not applicable
    # 6  - War Disablement Pension or War Widow's
    # 7  - Severe Disablement Allowance (SDA)
    # 8  - Disability Living/Attendance Allowance
    # 14 - Incapacity Benefit
    # 18 - Statutory Sick Pay (SSP)
    # 51 - Industrial Injuries Disablement Benefit
    # 85 - Other answer (not codeable 1-17, 51-53) 
    # 86 - Vague / irrelevant answer
    # 96 - None of these
# Any Disability
dislst = [7, 8]
varlst =['cpstyz0a', 'cpstyz0b', 'cpstyz0c', 'cpstyz0d', 'cpstyz0e', 'cpstyz0f', 'cpstyz0g', 'cpstyz0h']
mcs3['hhdis3'] = 0
for var in varlst:
    mcs3['hhdis3'] = np.where(mcs3[var].isin(dislst), 1, mcs3['hhdis3'])
    # mcs3.loc[mcs3[var] < 0, 'hhdis3'] = np.nan
print(mcs3['hhdis3'].value_counts(dropna=False))
# Sweep 4
dislst = [7, 8]
varlst =['dpstyz0a', 'dpstyz0b', 'dpstyz0c', 'dpstyz0d', 'dpstyz0e', 'dpstyz0f', 'dpstyz0g', 'dpstyz0h']
mcs4['hhdis4'] = 0
for var in varlst:
    mcs4['hhdis4'] = np.where(mcs4[var].isin(dislst), 1, mcs4['hhdis4'])
    # mcs4.loc[mcs4[var] < 0, 'hhdis4'] = np.nan
print(mcs4['hhdis4'].value_counts(dropna=False))
# Sweep 5
varlst =['epbenz0a', 'epbenz0m', 'epbenz0o']
mcs5['hhdis5'] = 0
for var in varlst:
    mcs5['hhdis5'] = np.where(mcs5[var] == 1, 1, mcs5['hhdis5'])
print(mcs5['hhdis5'].value_counts(dropna=False))
# Sweep 6
varlst =['fpbenh0a', 'fpbenh0n', 'fpbenh0p']
mcs6['hhdis6'] = 0
for var in varlst:
    mcs6['hhdis6'] = np.where(mcs6[var] == 1, 1, mcs6['hhdis6'])
print(mcs6['hhdis6'].value_counts(dropna=False))
# # Sweep 7
# varlst =['apstwm0a', 'apstwm0b', 'apstwm0c', 'apstwm0d', 'apstwm0e', 'apstwm0f', 'apstwm0g', 'apstwm0h']
# mcs7['hhdis1'] = 0
# for var in varlst:
#     mcs7['hhdis1'] = np.where(mcs7[var].isin(dislst), 1, mcs7['hhdis1'])
#     mcs7.loc[mcs7[var].isnull(), 'hhdis1'] = np.nan
# print(mcs7['hhdis1'].value_counts(dropna=False))

del dislst
del varlst
del var

#Parental disability (Activity limiting long standing illness)
# lsc_m - long standing illness/health condition  mother/main parent
# alc_m - activity limiting condition mother/main parent
# lsc_p - long standing illness/health condition  partner
# alc_p - activity limiting condition partner
# Sweep 1.
mcs1['lsc_m1'] = mcs1['aploil00']
mcs1.loc[mcs1['lsc_m1'] == 2, 'lsc_m1'] = 0
mcs1.loc[mcs1['lsc_m1'] < 0, 'lsc_m1'] = np.nan
print(mcs1['lsc_m1'].value_counts(dropna=False))
print(mcs1['lsc_m1'].value_counts().sum())
mcs1['alc_m1'] = mcs1['lsc_m1']
mcs1.loc[mcs1['aplolm00'] == 2, 'alc_m1'] = 0
print(mcs1['alc_m1'].value_counts(dropna=False))
print(mcs1['alc_m1'].value_counts().sum())
mcs1['lsc_p1'] = mcs1['aploil00_p']
mcs1.loc[mcs1['lsc_p1'] == 2, 'lsc_p1'] = 0
mcs1.loc[mcs1['lsc_p1'] < 0, 'lsc_p1'] = np.nan
print(mcs1['lsc_p1'].value_counts(dropna=False))
print(mcs1['lsc_p1'].value_counts().sum())
mcs1['alc_p1'] = mcs1['lsc_p1']
mcs1.loc[mcs1['aplolm00_p'] == 2, 'alc_p1'] = 0
print(mcs1['alc_p1'].value_counts(dropna=False))
print(mcs1['alc_p1'].value_counts().sum())
# Sweep 2.
# Respondents not asked if long term affects ability to perform daily activities.
mcs2['lsc_m2'] = mcs2['bploil00']
mcs2.loc[mcs2['lsc_m2'] == 2, 'lsc_m2'] = 0
mcs2.loc[mcs2['lsc_m2'] < 0, 'lsc_m2'] = np.nan
print(mcs2['lsc_m2'].value_counts(dropna=False))
print(mcs2['lsc_m2'].value_counts().sum())
# mcs2['alc_m2'] = mcs2['lsc_m2']
# mcs2.loc[mcs2['bplolm00'] == 2, 'alc_m2'] = 0
# print(mcs2['alc_m2'].value_counts(dropna=False))
# print(mcs2['alc_m2'].value_counts().sum())
mcs2['lsc_p2'] = mcs2['bploil00_p']
mcs2.loc[mcs2['lsc_p2'] == 2, 'lsc_p2'] = 0
mcs2.loc[mcs2['lsc_p2'] < 0, 'lsc_p2'] = np.nan
print(mcs2['lsc_p2'].value_counts(dropna=False))
print(mcs2['lsc_p2'].value_counts().sum())
# mcs2['alc_p2'] = mcs2['lsc_p2']
# mcs2.loc[mcs2['bplolm00_p'] == 2, 'alc_p2'] = 0
# print(mcs2['alc_p2'].value_counts(dropna=False))
# print(mcs2['alc_p2'].value_counts().sum())
# Sweep 3
mcs3['lsc_m3'] = mcs3['cploil00']
mcs3.loc[mcs3['lsc_m3'] == 2, 'lsc_m3'] = 0
mcs3.loc[mcs3['lsc_m3'] < 0, 'lsc_m3'] = np.nan
print(mcs3['lsc_m3'].value_counts(dropna=False))
print(mcs3['lsc_m3'].value_counts().sum())
mcs3['alc_m3'] = mcs3['lsc_m3']
mcs3.loc[mcs3['cplolm00'] == 2, 'alc_m3'] = 0
print(mcs3['alc_m3'].value_counts(dropna=False))
print(mcs3['alc_m3'].value_counts().sum())
mcs3['lsc_p3'] = mcs3['cploil00_p']
mcs3.loc[mcs3['lsc_p3'] == 2, 'lsc_p3'] = 0
mcs3.loc[mcs3['lsc_p3'] < 0, 'lsc_p3'] = np.nan
print(mcs3['lsc_p3'].value_counts(dropna=False))
print(mcs3['lsc_p3'].value_counts().sum())
mcs3['alc_p3'] = mcs3['lsc_p3']
mcs3.loc[mcs3['cplolm00_p'] == 2, 'alc_p3'] = 0
print(mcs3['alc_p3'].value_counts(dropna=False))
print(mcs3['alc_p3'].value_counts().sum())
# Sweep 4
mcs4['lsc_m4'] = mcs4['dploil00']
mcs4.loc[mcs4['lsc_m4'] == 2, 'lsc_m4'] = 0
mcs4.loc[mcs4['lsc_m4'] < 0, 'lsc_m4'] = np.nan
print(mcs4['lsc_m4'].value_counts(dropna=False))
print(mcs4['lsc_m4'].value_counts().sum())
mcs4['alc_m4'] = mcs4['lsc_m4']
mcs4.loc[mcs4['dplolm00'] == 2, 'alc_m4'] = 0
print(mcs4['alc_m4'].value_counts(dropna=False))
print(mcs4['alc_m4'].value_counts().sum())
mcs4['lsc_p4'] = mcs4['dploil00_p']
mcs4.loc[mcs4['lsc_p4'] == 2, 'lsc_p4'] = 0
mcs4.loc[mcs4['lsc_p4'] < 0, 'lsc_p4'] = np.nan
print(mcs4['lsc_p4'].value_counts(dropna=False))
print(mcs4['lsc_p4'].value_counts().sum())
mcs4['alc_p4'] = mcs4['lsc_p4']
mcs4.loc[mcs4['dplolm00_p'] == 2, 'alc_p4'] = 0
print(mcs4['alc_p4'].value_counts(dropna=False))
print(mcs4['alc_p4'].value_counts().sum())
# Sweep 5
mcs5['lsc_m5'] = mcs5['eploil00']
mcs5.loc[mcs5['lsc_m5'] == 2, 'lsc_m5'] = 0
mcs5.loc[mcs5['lsc_m5'] < 0, 'lsc_m5'] = np.nan
print(mcs5['lsc_m5'].value_counts(dropna=False))
print(mcs5['lsc_m5'].value_counts().sum())
mcs5['alc_m5'] = mcs5['lsc_m5']
mcs5.loc[mcs5['eplolm00'] == 2, 'alc_m5'] = 0
print(mcs5['alc_m5'].value_counts(dropna=False))
print(mcs5['alc_m5'].value_counts().sum())
mcs5['lsc_p5'] = mcs5['eploil00_p']
mcs5.loc[mcs5['lsc_p5'] == 2, 'lsc_p5'] = 0
mcs5.loc[mcs5['lsc_p5'] < 0, 'lsc_p5'] = np.nan
print(mcs5['lsc_p5'].value_counts(dropna=False))
print(mcs5['lsc_p5'].value_counts().sum())
mcs5['alc_p5'] = mcs5['lsc_p5']
mcs5.loc[mcs5['eplolm00_p'] == 2, 'alc_p5'] = 0
print(mcs5['alc_p5'].value_counts(dropna=False))
print(mcs5['alc_p5'].value_counts().sum())
# Sweep 6
mcs6['lsc_m6'] = mcs6['fploil00']
mcs6.loc[mcs6['lsc_m6'] == 2, 'lsc_m6'] = 0
mcs6.loc[mcs6['lsc_m6'] < 0, 'lsc_m6'] = np.nan
print(mcs6['lsc_m6'].value_counts(dropna=False))
print(mcs6['lsc_m6'].value_counts().sum())
mcs6['alc_m6'] = mcs6['lsc_m6']
mcs6.loc[mcs6['fplolm00'] == 2, 'alc_m6'] = 0
print(mcs6['alc_m6'].value_counts(dropna=False))
print(mcs6['alc_m6'].value_counts().sum())
mcs6['lsc_p6'] = mcs6['fploil00_p']
mcs6.loc[mcs6['lsc_p6'] == 2, 'lsc_p6'] = 0
mcs6.loc[mcs6['lsc_p6'] < 0, 'lsc_p6'] = np.nan
print(mcs6['lsc_p6'].value_counts(dropna=False))
print(mcs6['lsc_p6'].value_counts().sum())
mcs6['alc_p6'] = mcs6['lsc_p6']
mcs6.loc[mcs6['fplolm00_p'] == 2, 'alc_p6'] = 0
print(mcs6['alc_p6'].value_counts(dropna=False))
print(mcs6['alc_p6'].value_counts().sum())
# Sweep 7



## Hours Worked/ week (usual)
# Sweep 1
#Main parent
mcs1['hrswrkdm1'] = mcs1['apwohr00'] # Usual hours
print(mcs1['hrswrkdm1'].value_counts(dropna=False))
print(mcs1['hrswrkdm1'].value_counts().sum())
#Partner
mcs1['hrswrkdp1'] = mcs1['apwohr00_p'] # Usual hours
print(mcs1['hrswrkdp1'].value_counts(dropna=False))
print(mcs1['hrswrkdp1'].value_counts().sum())

## Hours Worked/ week (Total)
# Sweep 1
#Main parent
# mcs1['thrswrkdm1'] = mcs1['aptohr00'] # Total Hours
mcs1['thrswrkdm1'] = mcs1['apwkhr00'] # Hours including over time, excluding lunch etc.
mcs1.loc[mcs1['thrswrkdm1'] < 0, 'thrswrkdm1'] = np.nan
print(mcs1['thrswrkdm1'].value_counts(dropna=False))
print(mcs1['thrswrkdm1'].value_counts().sum())
#Partner
mcs1['thrswrkdp1'] = mcs1['aptohr00_p'] # Total Hours
# mcs1['thrswrkdp1'] = mcs1['apwkhr00_p'] # Hours including over time, excluding lunch etc.
mcs1.loc[mcs1['thrswrkdp1'] < 0, 'thrswrkdp1'] = np.nan
print(mcs1['thrswrkdp1'].value_counts(dropna=False))
print(mcs1['thrswrkdp1'].value_counts().sum())
# Sweep 2
#Main parent
# mcs2['thrswrkdm2'] = mcs2['bptohr00'] # Total Hours
mcs2['thrswrkdm2'] = mcs2['bpwkhr00'] # Hours including over time, excluding lunch etc.
mcs2.loc[mcs2['thrswrkdm2'] < 0, 'thrswrkdm2'] = np.nan
print(mcs2['thrswrkdm2'].value_counts(dropna=False))
print(mcs2['thrswrkdm2'].value_counts().sum())
#Partner
# mcs2['thrswrkdp2'] = mcs2['bptohr00_p'] # Total Hours
mcs2['thrswrkdp2'] = mcs2['bpwkhr00_p'] # Hours including over time, excluding lunch etc.
mcs2.loc[mcs2['thrswrkdp2'] < 0, 'thrswrkdp2'] = np.nan
print(mcs2['thrswrkdp2'].value_counts(dropna=False))
print(mcs2['thrswrkdp2'].value_counts().sum())
# Sweep 3
#Main parent
# mcs3['thrswrkdm3'] = mcs3['cptohr00'] # Total Hours
mcs3['thrswrkdm3'] = mcs3['cpwkhr00'] # Hours including over time, excluding lunch etc.
mcs3.loc[mcs3['thrswrkdm3'] < 0, 'thrswrkdm3'] = np.nan
print(mcs3['thrswrkdm3'].value_counts(dropna=False))
print(mcs3['thrswrkdm3'].value_counts().sum())
#Partner
# mcs3['thrswrkdp3'] = mcs3['cptohr00_p'] # Total Hours
mcs3['thrswrkdp3'] = mcs3['cpwkhr00_p'] # Hours including over time, excluding lunch etc.
mcs3.loc[mcs3['thrswrkdp3'] < 0, 'thrswrkdp3'] = np.nan
print(mcs3['thrswrkdp3'].value_counts(dropna=False))
print(mcs3['thrswrkdp3'].value_counts().sum())
# Sweep 4
#Main parent
# mcs4['thrswrkdm4'] = mcs4['dptohr00'] # Total Hours
mcs4['thrswrkdm4'] = mcs4['dpwkhr00'] # Hours including over time, excluding lunch etc.
mcs4.loc[mcs4['thrswrkdm4'] < 0, 'thrswrkdm4'] = np.nan
print(mcs4['thrswrkdm4'].value_counts(dropna=False))
print(mcs4['thrswrkdm4'].value_counts().sum())
#Partner
# mcs4['thrswrkdp4'] = mcs4['dptohr00_p'] # Total Hours
mcs4['thrswrkdp4'] = mcs4['dpwkhr00_p'] # Hours including over time, excluding lunch etc.
mcs4.loc[mcs4['thrswrkdp4'] < 0, 'thrswrkdp4'] = np.nan
print(mcs4['thrswrkdp4'].value_counts(dropna=False))
print(mcs4['thrswrkdp4'].value_counts().sum())
# Sweep 5
#Main parent
# mcs5['thrswrkdm5'] = mcs5['eptohr00'] # Total Hours
mcs5['thrswrkdm5'] = mcs5['epwkhr00'] # Hours including over time, excluding lunch etc.
mcs5.loc[mcs5['thrswrkdm5'] < 0, 'thrswrkdm5'] = np.nan
print(mcs5['thrswrkdm5'].value_counts(dropna=False))
print(mcs5['thrswrkdm5'].value_counts().sum())
#Partner
# mcs5['thrswrkdp5'] = mcs5['eptohr00_p'] # Total Hours
mcs5['thrswrkdp5'] = mcs5['epwkhr00_p'] # Hours including over time, excluding lunch etc.
mcs5.loc[mcs5['thrswrkdp5'] < 0, 'thrswrkdp5'] = np.nan
print(mcs5['thrswrkdp5'].value_counts(dropna=False))
print(mcs5['thrswrkdp5'].value_counts().sum())
# Sweep 6
#Main parent
# mcs6['thrswrkdm6'] = mcs6['fptohr00'] # Total Hours
mcs6['thrswrkdm6'] = mcs6['fpwkhr00'] # Hours including over time, excluding lunch etc.
mcs6.loc[mcs6['thrswrkdm6'] < 0, 'thrswrkdm6'] = np.nan
print(mcs6['thrswrkdm6'].value_counts(dropna=False))
print(mcs6['thrswrkdm6'].value_counts().sum())
#Partner
# mcs6['thrswrkdp6'] = mcs6['fptohr00_p'] # Total Hours
mcs6['thrswrkdp6'] = mcs6['fpwkhr00_p'] # Hours including over time, excluding lunch etc.
mcs6.loc[mcs6['thrswrkdp6'] < 0, 'thrswrkdp6'] = np.nan
print(mcs6['thrswrkdp6'].value_counts(dropna=False))
print(mcs6['thrswrkdp6'].value_counts().sum())
# Sweep 7
#Main parent
# mcs7['thrswrkdm7'] = mcs7['gptohr00'] # Total Hours
mcs7['thrswrkdm7'] = mcs7['gpwkhr00'] # Hours including over time, excluding lunch etc.
mcs7.loc[mcs7['thrswrkdm7'] < 0, 'thrswrkdm7'] = np.nan
print(mcs7['thrswrkdm7'].value_counts(dropna=False))
print(mcs7['thrswrkdm7'].value_counts().sum())
#Partner
# mcs7['thrswrkdp7'] = mcs7['gptohr00_p'] # Total Hours
mcs7['thrswrkdp7'] = mcs7['gpwkhr00_p'] # Hours including over time, excluding lunch etc.
mcs7.loc[mcs7['thrswrkdp7'] < 0, 'thrswrkdp7'] = np.nan
print(mcs7['thrswrkdp7'].value_counts(dropna=False))
print(mcs7['thrswrkdp7'].value_counts().sum())

## Days worked/ week
# Sweep 1
#Main parent
mcs1['dyswrkdm1'] = mcs1['apdayw00'] 
print(mcs1['dyswrkdm1'].value_counts(dropna=False))
print(mcs1['dyswrkdm1'].value_counts().sum())
#Partner
mcs1['dyswrkdp1'] = mcs1['apdayw00_p']
print(mcs1['dyswrkdp1'].value_counts(dropna=False))
print(mcs1['dyswrkdp1'].value_counts().sum())
# Sweep 2
#Main parent
mcs2['dyswrkdm2'] = mcs2['bpdayw00'] 
print(mcs2['dyswrkdm2'].value_counts(dropna=False))
print(mcs2['dyswrkdm2'].value_counts().sum())
#Partner
mcs2['dyswrkdp2'] = mcs2['bpdayw00_p']
print(mcs2['dyswrkdp2'].value_counts(dropna=False))
print(mcs2['dyswrkdp2'].value_counts().sum())
# Sweep 3
#Main parent
mcs3['dyswrkdm3'] = mcs3['cpdayw00'] 
print(mcs3['dyswrkdm3'].value_counts(dropna=False))
print(mcs3['dyswrkdm3'].value_counts().sum())
#Partner
mcs3['dyswrkdp3'] = mcs3['cpdayw00_p']
print(mcs3['dyswrkdp3'].value_counts(dropna=False))
print(mcs3['dyswrkdp3'].value_counts().sum())
# Sweep 4
#Main parent
mcs4['dyswrkdm4'] = mcs4['dpdayw00'] 
print(mcs4['dyswrkdm4'].value_counts(dropna=False))
print(mcs4['dyswrkdm4'].value_counts().sum())
#Partner
mcs4['dyswrkdp4'] = mcs4['dpdayw00_p']
print(mcs4['dyswrkdp4'].value_counts(dropna=False))
print(mcs4['dyswrkdp4'].value_counts().sum())
# # Sweep 5
# #Main parent
# mcs5['dyswrkdm5'] = mcs5['epdayw00'] 
# print(mcs5['dyswrkdm5'].value_counts(dropna=False))
# print(mcs5['dyswrkdm5'].value_counts().sum())
# #Partner
# mcs5['dyswrkdp5'] = mcs5['epdayw00_p']
# print(mcs5['dyswrkdp5'].value_counts(dropna=False))
# print(mcs5['dyswrkdp5'].value_counts().sum())
# # Sweep 6
# #Main parent
# mcs6['dyswrkdm6'] = mcs6['fpdayw00'] 
# print(mcs6['dyswrkdm6'].value_counts(dropna=False))
# print(mcs6['dyswrkdm6'].value_counts().sum())
# #Partner
# mcs6['dyswrkdp6'] = mcs6['fpdayw00_p']
# print(mcs6['dyswrkdp6'].value_counts(dropna=False))
# print(mcs6['dyswrkdp6'].value_counts().sum())
# # Sweep 7
# #Main parent
# mcs7['dyswrkdm7'] = mcs7['gpdayw00'] 
# print(mcs7['dyswrkdm7'].value_counts(dropna=False))
# print(mcs7['dyswrkdm7'].value_counts().sum())
# #Partner
# mcs7['dyswrkdp7'] = mcs7['gpdayw00_p']
# print(mcs7['dyswrkdp7'].value_counts(dropna=False))
# print(mcs7['dyswrkdp7'].value_counts().sum())


##OECD equivalised weekly income
# Sweep 1
mcs1['income1'] = mcs1['adoede00']
mcs1.loc[mcs1['income1'] < 0, 'income1'] = np.nan
print(mcs1['income1'].value_counts(dropna=False))
print(mcs1['income1'].value_counts().sum())
# Sweep 2
mcs2['income2'] = mcs2['bdoede00']
mcs2.loc[mcs2['income2'] < 0, 'income2'] = np.nan
print(mcs2['income2'].value_counts(dropna=False))
print(mcs2['income2'].value_counts().sum())
# Sweep 3
mcs3['income3'] = mcs3['cdoede00']
mcs3.loc[mcs3['income3'] < 0, 'income3'] = np.nan
print(mcs3['income3'].value_counts(dropna=False))
print(mcs3['income3'].value_counts().sum())
# Sweep 4
mcs4['income4'] = mcs4['ddoede00']
mcs4.loc[mcs4['income4'] < 0, 'income4'] = np.nan
print(mcs4['income4'].value_counts(dropna=False))
print(mcs4['income4'].value_counts().sum())
# Sweep 5
mcs5['income5'] = mcs5['eoede000']
mcs5.loc[mcs5['income5'] < 0, 'income5'] = np.nan
print(mcs5['income5'].value_counts(dropna=False))
print(mcs5['income5'].value_counts().sum())
# Sweep 6
mcs6['income6'] = mcs6['foede000']
mcs6.loc[mcs6['income6'] < 0, 'income6'] = np.nan
print(mcs6['income6'].value_counts(dropna=False))
print(mcs6['income6'].value_counts().sum())
# # Sweep 7
# mcs7['income7'] = mcs7['gdoede00']
# mcs7.loc[mcs7['income7'] < 0, 'income7'] = np.nan
# print(mcs7['income7'].value_counts(dropna=False))
# print(mcs7['income7'].value_counts().sum())

##Net weeklyfamily income (predicted)
# Sweep 1
mcs1['netinc1'] = mcs1['aoedex00']
mcs1.loc[mcs1['netinc1'] < 0, 'netinc1'] = np.nan
print(mcs1['netinc1'].value_counts(dropna=False))
print(mcs1['netinc1'].value_counts().sum())
# Sweep 2
mcs2['netinc2'] = mcs2['boedex00']
mcs2.loc[mcs2['netinc2'] < 0, 'netinc2'] = np.nan
print(mcs2['netinc2'].value_counts(dropna=False))
print(mcs2['netinc2'].value_counts().sum())
# Sweep 3
mcs3['netinc3'] = mcs3['coedex00']
mcs3.loc[mcs3['netinc3'] < 0, 'netinc3'] = np.nan
print(mcs3['netinc3'].value_counts(dropna=False))
print(mcs3['netinc3'].value_counts().sum())
# Sweep 4
mcs4['netinc4'] = mcs4['ddoedex00']
mcs4.loc[mcs4['netinc4'] < 0, 'netinc4'] = np.nan
print(mcs4['netinc4'].value_counts(dropna=False))
print(mcs4['netinc4'].value_counts().sum())
# # Sweep 5
# mcs5['netinc5'] = mcs5['eeoedex00']
# mcs5.loc[mcs5['netinc5'] < 0, 'netinc5'] = np.nan
# print(mcs5['netinc5'].value_counts(dropna=False))
# print(mcs5['netinc5'].value_counts().sum())
# # Sweep 6
# mcs6['netinc6'] = mcs6['foedex00']
# mcs6.loc[mcs6['netinc6'] < 0, 'netinc6'] = np.nan
# print(mcs6['netinc6'].value_counts(dropna=False))
# print(mcs6['netinc6'].value_counts().sum())
# # Sweep 7
# mcs7['netinc7'] = mcs7['goedex00']
# mcs7.loc[mcs7['netinc7'] < 0, 'netinc7'] = np.nan
# print(mcs7['netinc7'].value_counts(dropna=False))
# print(mcs7['netinc7'].value_counts().sum())

# ##Annual net take home income
#     # -8 - Don't Know
#     # -1 - Not applicable
#     # 1  - 1 week
#     # 2  - Fortnight
#     # 3  - Four weeks
#     # 4 - Calendar month
#     # 5 - Year
#     # 51 - Other/Three weeks
#     # 52 - Other/Five weeks
#     # 53 - Other/Six weeks
#     # 54 - Other/Seven weeks
#     # 56 - Other/Two calendar months
#     # 60 - Other/Three months/13 weeks
#     # 61 - Other/Six months/26 weeks
#     # 62 - Other/Hourly
#     # 63 - Other/Daily
#     # 64 - Other/One off lump sum
#     # 85 - Other answer (not codeable 1-5,51-64)
#     # 86 - Irrelevant response
        
# def annual_pay(pay, period_code):
#     if period_code == 1:  # 1 week
#         return pay * 52
#     elif period_code == 2:  # Fortnight
#         return pay * 26
#     elif period_code == 3:  # Four weeks
#         return pay * 13
#     elif period_code == 4:  # Calendar month
#         return pay * 12
#     elif period_code == 5:  # Year
#         return pay
#     elif period_code == 51:  # Other/Three weeks
#         return pay * (52/3)  # Convert weeks to annua
#     elif period_code in [52, 53, 54]:  # Other/Five to Seven weeks
#         return pay * (52 / (period_code - 47))  # Convert weeks to annual
#     elif period_code == 56:  # Other/Two calendar months
#         return pay * 6
#     elif period_code == 60:  # Other/Three months/13 weeks
#         return pay * 4
#     elif period_code == 61:  # Other/Six months/26 weeks
#         return pay * 2
#     elif period_code == 62:  # Other/Hourly (assume 1,866 annually based on UK fulltime average )
#         return pay * 1866
#     elif period_code == 63:  # Other/Daily (Usual number ok work days UK/year)
#         return pay * 252
#     elif period_code == 64:  # One off lump sum
#         return pay
#     elif period_code in [85, 86]:  # Other answer / Irrelevant response
#         return pay  # Could be NaN or left as is, depending on how you want to handle these responses
#     else:
#         return pay  # Default case if none of the above matches

# # Apply the function
# df['annual_netpay1'] = df.apply(annual_pay, axis=1)
# # Sweep 1
# mcs1['netpay1'] = mcs1['apneta00']
# mcs1.loc[mcs1['netpay1'] < 0, 'netpay1'] = np.nan
# mcs1['netpap1'] = mcs1['apnetp00'] #Pay period
# mcs1.loc[mcs1['netpap1'] < 0, 'netpap1'] = np.nan
# mcs1.loc[mcs1['netpap1'] == 85, 'netpap1'] = np.nan
# mcs1.loc[mcs1['netpap1'] == 86, 'netpap1'] = np.nan
# #Annualising income

# mcs1['netsep1'] = mcs1['apsepa00'] #self employed take home income
# mcs1.loc[mcs1['netsep1'] < 0, 'netsep1'] = np.nan
# print(mcs1['netpay1'].value_counts(dropna=False))
# print(mcs1['netpay1'].value_counts().sum())
# # Sweep 2
# mcs2['income2'] = mcs2['bdoede00']
# mcs2.loc[mcs2['income2'] < 0, 'income2'] = np.nan
# print(mcs2['income2'].value_counts(dropna=False))
# print(mcs2['income2'].value_counts().sum())
# # Sweep 3
# mcs3['income3'] = mcs3['cdoede00']
# mcs3.loc[mcs3['income3'] < 0, 'income3'] = np.nan
# print(mcs3['income3'].value_counts(dropna=False))
# print(mcs3['income3'].value_counts().sum())
# # Sweep 4
# mcs4['income4'] = mcs4['ddoede00']
# mcs4.loc[mcs4['income4'] < 0, 'income4'] = np.nan
# print(mcs4['income4'].value_counts(dropna=False))
# print(mcs4['income4'].value_counts().sum())
# # Sweep 5
# mcs5['income5'] = mcs5['eoede000']
# mcs5.loc[mcs5['income5'] < 0, 'income5'] = np.nan
# print(mcs5['income5'].value_counts(dropna=False))
# print(mcs5['income5'].value_counts().sum())
# # Sweep 6
# mcs6['income6'] = mcs6['foede000']
# mcs6.loc[mcs6['income6'] < 0, 'income6'] = np.nan
# print(mcs6['income6'].value_counts(dropna=False))
# print(mcs6['income6'].value_counts().sum())
# # # Sweep 7
# # mcs7['income7'] = mcs7['gdoede00']
# # mcs7.loc[mcs7['income7'] < 0, 'income7'] = np.nan
# # print(mcs7['income7'].value_counts(dropna=False))
# # print(mcs7['income7'].value_counts().sum())

##Poverty
# Sweep 1
mcs1['poverty1'] = mcs1['adoedp00']
mcs1.loc[mcs1['poverty1'] < 0, 'poverty1'] = np.nan
print(mcs1['poverty1'].value_counts(dropna=False))
print(mcs1['poverty1'].value_counts().sum())
# Sweep 2
mcs2['poverty2'] = mcs2['bdoedp00']
mcs2.loc[mcs2['poverty2'] < 0, 'poverty2'] = np.nan
print(mcs2['poverty2'].value_counts(dropna=False))
print(mcs2['poverty2'].value_counts().sum())
# Sweep 3
mcs3['poverty3'] = mcs3['cdoedp00']
mcs3.loc[mcs3['poverty3'] < 0, 'poverty3'] = np.nan
print(mcs3['poverty3'].value_counts(dropna=False))
print(mcs3['poverty3'].value_counts().sum())
# Sweep 4
mcs4['poverty4'] = mcs4['ddoedp00']
mcs4.loc[mcs4['poverty4'] < 0, 'poverty4'] = np.nan
print(mcs4['poverty4'].value_counts(dropna=False))
print(mcs4['poverty4'].value_counts().sum())
# Sweep 5
mcs5['poverty5'] = mcs5['eoedp000']
mcs5.loc[mcs5['poverty5'] < 0, 'poverty5'] = np.nan
print(mcs5['poverty5'].value_counts(dropna=False))
print(mcs5['poverty5'].value_counts().sum())
# Sweep 6
mcs6['poverty6'] = mcs6['foedp000']
mcs6.loc[mcs6['poverty6'] < 0, 'poverty6'] = np.nan
print(mcs6['poverty6'].value_counts(dropna=False))
print(mcs6['poverty6'].value_counts().sum())
# # Sweep 7
# mcs7['poverty7'] = mcs7['gdoedp00']
# mcs7.loc[mcs7['poverty7'] < 0, 'poverty7'] = np.nan
# print(mcs7['poverty7'].value_counts(dropna=False))
# print(mcs7['poverty7'].value_counts().sum())

##Parental wealth (Assets and savings (range)) (Main parent)
# Sweep 5
mcs5['wealth5'] = mcs5['epinvt00']
mcs5.loc[mcs5['wealth5'] < 0, 'wealth5'] = np.nan
print(mcs5['wealth5'].value_counts(dropna=False))
print(mcs5['wealth5'].value_counts().sum())
# Sweep 6
mcs6['wealth6'] = mcs6['fpinvt00']
mcs6.loc[mcs6['wealth6'] < 0, 'wealth6'] = np.nan
print(mcs6['wealth6'].value_counts(dropna=False))
print(mcs6['wealth6'].value_counts().sum())

##Parental wealth (Assets and savings (range)) (Partner)
# Sweep 5
mcs5['wealthp5'] = mcs5['epinvt00_p']
mcs5.loc[mcs5['wealthp5'] < 0, 'wealthp5'] = np.nan
print(mcs5['wealthp5'].value_counts(dropna=False))
print(mcs5['wealthp5'].value_counts().sum())
# Sweep 6
mcs6['wealthp6'] = mcs6['fpinvt00_p']
mcs6.loc[mcs6['wealthp6'] < 0, 'wealthp6'] = np.nan
print(mcs6['wealthp6'].value_counts(dropna=False))
print(mcs6['wealthp6'].value_counts().sum())


##IMD
#Decile
# Sweep 1
mcs1['imddec1'] = mcs1['aimdscoe']
mcs1['imddec1'] = np.where((mcs1['country1'] == 2) & (mcs1['imddec1'].isnull() == 1)
                                , mcs1['aiwimdsc'], mcs1['imddec1'])
mcs1['imddec1'] = np.where((mcs1['country1'] == 3) & (mcs1['imddec1'].isnull() == 1)
                                , mcs1['aisimdsc'], mcs1['imddec1'])
mcs1['imddec1'] = np.where((mcs1['country1'] == 4) & (mcs1['imddec1'].isnull() == 1)
                                , mcs1['aimdscon'], mcs1['imddec1'])
print(mcs1['imddec1'].value_counts(dropna=False))
print(mcs1['imddec1'].value_counts().sum())
# Sweep 2
mcs2['imddec2'] = mcs2['bimdscoe']
mcs2['imddec2'] = np.where((mcs2['country2'] == 2) & (mcs2['imddec2'].isnull() == 1)
                                , mcs2['biwimdsc'], mcs2['imddec2'])
mcs2['imddec2'] = np.where((mcs2['country2'] == 3) & (mcs2['imddec2'].isnull() == 1)
                                , mcs2['bisimdsc'], mcs2['imddec2'])
mcs2['imddec2'] = np.where((mcs2['country2'] == 4) & (mcs2['imddec2'].isnull() == 1)
                                , mcs2['bimdscon'], mcs2['imddec2'])
print(mcs2['imddec2'].value_counts(dropna=False))
print(mcs2['imddec2'].value_counts().sum())
# Sweep 3
mcs3['imddec3'] = mcs3['cimdscoe']
mcs3['imddec3'] = np.where((mcs3['country3'] == 2) & (mcs3['imddec3'].isnull() == 1)
                                , mcs3['ciwimdsc'], mcs3['imddec3'])
mcs3['imddec3'] = np.where((mcs3['country3'] == 3) & (mcs3['imddec3'].isnull() == 1)
                                , mcs3['cisimdsc'], mcs3['imddec3'])
mcs3['imddec3'] = np.where((mcs3['country3'] == 4) & (mcs3['imddec3'].isnull() == 1)
                                , mcs3['cimdscon'], mcs3['imddec3'])
print(mcs3['imddec3'].value_counts(dropna=False))
print(mcs3['imddec3'].value_counts().sum())
# Sweep 4
mcs4['imddec4'] = mcs4['dimdscoe']
mcs4['imddec4'] = np.where((mcs4['country4'] == 2) & (mcs4['imddec4'].isnull() == 1)
                                , mcs4['diwimdsc'], mcs4['imddec4'])
mcs4['imddec4'] = np.where((mcs4['country4'] == 3) & (mcs4['imddec4'].isnull() == 1)
                                , mcs4['disimdsc'], mcs4['imddec4'])
mcs4['imddec4'] = np.where((mcs4['country4'] == 4) & (mcs4['imddec4'].isnull() == 1)
                                , mcs4['dimdscon'], mcs4['imddec4'])
print(mcs4['imddec4'].value_counts(dropna=False))
print(mcs4['imddec4'].value_counts().sum())
# Sweep 5
mcs5['imddec5'] = mcs5['eimdscoe']
mcs5['imddec5'] = np.where((mcs5['country5'] == 2) & (mcs5['imddec5'].isnull() == 1)
                                , mcs5['eiwimdsc'], mcs5['imddec5'])
mcs5['imddec5'] = np.where((mcs5['country5'] == 3) & (mcs5['imddec5'].isnull() == 1)
                                , mcs5['eisimdsc'], mcs5['imddec5'])
mcs5['imddec5'] = np.where((mcs5['country5'] == 4) & (mcs5['imddec5'].isnull() == 1)
                                , mcs5['eimdscon'], mcs5['imddec5'])
print(mcs5['imddec5'].value_counts(dropna=False))
print(mcs5['imddec5'].value_counts().sum())
# Sweep 6
mcs6['imddec6'] = mcs6['fimdscoe']
mcs6['imddec6'] = np.where((mcs6['country6'] == 2) & (mcs6['imddec6'].isnull() == 1)
                                , mcs6['fiwimdsc'], mcs6['imddec6'])
mcs6['imddec6'] = np.where((mcs6['country6'] == 3) & (mcs6['imddec6'].isnull() == 1)
                                , mcs6['fisimdsc'], mcs6['imddec6'])
mcs6['imddec6'] = np.where((mcs6['country6'] == 4) & (mcs6['imddec6'].isnull() == 1)
                                , mcs6['fimdscon'], mcs6['imddec6'])
print(mcs6['imddec6'].value_counts(dropna=False))
print(mcs6['imddec6'].value_counts().sum())
# # Sweep 7
# mcs7['imddec7'] = mcs7['gimdscoe']
# mcs7['imddec7'] = np.where((mcs7['country7'] == 2) & (mcs7['imddec7'].isnull() == 1)
#                                 , mcs7['giwimdsc'], mcs7['imddec7'])
# mcs7['imddec7'] = np.where((mcs7['country7'] == 3) & (mcs7['imddec7'].isnull() == 1)
#                                 , mcs7['gisimdsc'], mcs7['imddec7'])
# mcs7['imddec7'] = np.where((mcs7['country7'] == 4) & (mcs7['imddec7'].isnull() == 1)
#                                 , mcs7['gimdscon'], mcs7['imddec7'])
# print(mcs7['imddec7'].value_counts(dropna=False))
# print(mcs7['imddec7'].value_counts().sum())
#Quintile
# Sweep 1
mcs1['imdqnt1'] = mcs1['imddec1']
mcs1['imdqnt1'] = np.where((mcs1['imddec1'] == 1) | (mcs1['imddec1'] == 2)
                                , 1, mcs1['imdqnt1'])
mcs1['imdqnt1'] = np.where((mcs1['imddec1'] == 3) | (mcs1['imddec1'] == 4)
                                , 2, mcs1['imdqnt1'])
mcs1['imdqnt1'] = np.where((mcs1['imddec1'] == 5) | (mcs1['imddec1'] == 6)
                                , 3, mcs1['imdqnt1'])
mcs1['imdqnt1'] = np.where((mcs1['imddec1'] == 7) | (mcs1['imddec1'] == 8)
                                , 4, mcs1['imdqnt1'])
mcs1['imdqnt1'] = np.where((mcs1['imddec1'] == 9) | (mcs1['imddec1'] == 10)
                                , 5, mcs1['imdqnt1'])
print(mcs1['imdqnt1'].value_counts(dropna=False))
print(mcs1['imdqnt1'].value_counts().sum())
# Sweep 2
mcs2['imdqnt2'] = mcs2['imddec2']
mcs2['imdqnt2'] = np.where((mcs2['imddec2'] == 1) | (mcs2['imddec2'] == 2)
                                , 1, mcs2['imdqnt2'])
mcs2['imdqnt2'] = np.where((mcs2['imddec2'] == 3) | (mcs2['imddec2'] == 4)
                                , 2, mcs2['imdqnt2'])
mcs2['imdqnt2'] = np.where((mcs2['imddec2'] == 5) | (mcs2['imddec2'] == 6)
                                , 3, mcs2['imdqnt2'])
mcs2['imdqnt2'] = np.where((mcs2['imddec2'] == 7) | (mcs2['imddec2'] == 8)
                                , 4, mcs2['imdqnt2'])
mcs2['imdqnt2'] = np.where((mcs2['imddec2'] == 9) | (mcs2['imddec2'] == 10)
                                , 5, mcs2['imdqnt2'])
print(mcs2['imdqnt2'].value_counts(dropna=False))
print(mcs2['imdqnt2'].value_counts().sum())
# Sweep 3
mcs3['imdqnt3'] = mcs3['imddec3']
mcs3['imdqnt3'] = np.where((mcs3['imddec3'] == 1) | (mcs3['imddec3'] == 2)
                                , 1, mcs3['imdqnt3'])
mcs3['imdqnt3'] = np.where((mcs3['imddec3'] == 3) | (mcs3['imddec3'] == 4)
                                , 2, mcs3['imdqnt3'])
mcs3['imdqnt3'] = np.where((mcs3['imddec3'] == 5) | (mcs3['imddec3'] == 6)
                                , 3, mcs3['imdqnt3'])
mcs3['imdqnt3'] = np.where((mcs3['imddec3'] == 7) | (mcs3['imddec3'] == 8)
                                , 4, mcs3['imdqnt3'])
mcs3['imdqnt3'] = np.where((mcs3['imddec3'] == 9) | (mcs3['imddec3'] == 10)
                                , 5, mcs3['imdqnt3'])
print(mcs3['imdqnt3'].value_counts(dropna=False))
print(mcs3['imdqnt3'].value_counts().sum())
# Sweep 4
mcs4['imdqnt4'] = mcs4['imddec4']
mcs4['imdqnt4'] = np.where((mcs4['imddec4'] == 1) | (mcs4['imddec4'] == 2)
                                , 1, mcs4['imdqnt4'])
mcs4['imdqnt4'] = np.where((mcs4['imddec4'] == 3) | (mcs4['imddec4'] == 4)
                                , 2, mcs4['imdqnt4'])
mcs4['imdqnt4'] = np.where((mcs4['imddec4'] == 5) | (mcs4['imddec4'] == 6)
                                , 3, mcs4['imdqnt4'])
mcs4['imdqnt4'] = np.where((mcs4['imddec4'] == 7) | (mcs4['imddec4'] == 8)
                                , 4, mcs4['imdqnt4'])
mcs4['imdqnt4'] = np.where((mcs4['imddec4'] == 9) | (mcs4['imddec4'] == 10)
                                , 5, mcs4['imdqnt4'])
print(mcs4['imdqnt4'].value_counts(dropna=False))
print(mcs4['imdqnt4'].value_counts().sum())
# Sweep 5
mcs5['imdqnt5'] = mcs5['imddec5']
mcs5['imdqnt5'] = np.where((mcs5['imddec5'] == 1) | (mcs5['imddec5'] == 2)
                                , 1, mcs5['imdqnt5'])
mcs5['imdqnt5'] = np.where((mcs5['imddec5'] == 3) | (mcs5['imddec5'] == 4)
                                , 2, mcs5['imdqnt5'])
mcs5['imdqnt5'] = np.where((mcs5['imddec5'] == 5) | (mcs5['imddec5'] == 6)
                                , 3, mcs5['imdqnt5'])
mcs5['imdqnt5'] = np.where((mcs5['imddec5'] == 7) | (mcs5['imddec5'] == 8)
                                , 4, mcs5['imdqnt5'])
mcs5['imdqnt5'] = np.where((mcs5['imddec5'] == 9) | (mcs5['imddec5'] == 10)
                                , 5, mcs5['imdqnt5'])
print(mcs5['imdqnt5'].value_counts(dropna=False))
print(mcs5['imdqnt5'].value_counts().sum())
# Sweep 6
mcs6['imdqnt6'] = mcs6['imddec6']
mcs6['imdqnt6'] = np.where((mcs6['imddec6'] == 1) | (mcs6['imddec6'] == 2)
                                , 1, mcs6['imdqnt6'])
mcs6['imdqnt6'] = np.where((mcs6['imddec6'] == 3) | (mcs6['imddec6'] == 4)
                                , 2, mcs6['imdqnt6'])
mcs6['imdqnt6'] = np.where((mcs6['imddec6'] == 5) | (mcs6['imddec6'] == 6)
                                , 3, mcs6['imdqnt6'])
mcs6['imdqnt6'] = np.where((mcs6['imddec6'] == 7) | (mcs6['imddec6'] == 8)
                                , 4, mcs6['imdqnt6'])
mcs6['imdqnt6'] = np.where((mcs6['imddec6'] == 9) | (mcs6['imddec6'] == 10)
                                , 5, mcs6['imdqnt6'])
print(mcs6['imdqnt6'].value_counts(dropna=False))
print(mcs6['imdqnt6'].value_counts().sum())
# # Sweep 7
# mcs7['imddec7'] = mcs7['gimdscoe']
# mcs7['imddec7'] = np.where((mcs7['country7'] == 2) & (mcs7['imddec7'].isnull() == 1)
#                                 , mcs7['giwimdsc'], mcs7['imddec7'])
# mcs7['imddec7'] = np.where((mcs7['country7'] == 3) & (mcs7['imddec7'].isnull() == 1)
#                                 , mcs7['gisimdsc'], mcs7['imddec7'])
# mcs7['imddec7'] = np.where((mcs7['country7'] == 4) & (mcs7['imddec7'].isnull() == 1)
#                                 , mcs7['gimdscon'], mcs7['imddec7'])
# print(mcs7['imddec7'].value_counts(dropna=False))
# print(mcs7['imddec7'].value_counts().sum())

##Marital status - Single (parent)
# Sweep 1
mcs1['mrtlsts1'] = mcs1['apfcin00']
mcs1.loc[mcs1['mrtlsts1'] < 0, 'mrtlsts1'] = np.nan
mcs1['mssngl1'] = np.where((mcs1['mrtlsts1'] == 1) | (mcs1['mrtlsts1'] == 4) | (mcs1['mrtlsts1'] == 5) | (mcs1['mrtlsts1'] == 6)
                                , 1, 0)
mcs1['mssngl1'] = np.where((mcs1['mrtlsts1'].isnull() == 1)
                                , np.nan, mcs1['mssngl1'])
print(mcs1['mssngl1'].value_counts(dropna=False))
print(mcs1['mssngl1'].value_counts().sum())
# Sweep 2
mcs2['mrtlsts2'] = mcs2['bpfcin00']
mcs2.loc[mcs2['mrtlsts2'] < 0, 'mrtlsts2'] = np.nan
mcs2['mssngl2'] = np.where((mcs2['mrtlsts2'] == 1) | (mcs2['mrtlsts2'] == 4) | (mcs2['mrtlsts2'] == 5) | (mcs2['mrtlsts2'] == 6)
                                , 1, 0)
mcs2['mssngl2'] = np.where((mcs2['mrtlsts2'].isnull() == 1)
                                , np.nan, mcs2['mssngl2'])
print(mcs2['mssngl2'].value_counts(dropna=False))
print(mcs2['mssngl2'].value_counts().sum())
# Sweep 3
mcs3['mrtlsts3'] = mcs3['cpfcin00']
mcs3.loc[mcs3['mrtlsts3'] < 0, 'mrtlsts3'] = np.nan
mcs3['mssngl3'] = np.where((mcs3['mrtlsts3'] == 1) | (mcs3['mrtlsts3'] == 4) | (mcs3['mrtlsts3'] == 5) | (mcs3['mrtlsts3'] == 6)
                                , 1, 0)
mcs3['mssngl3'] = np.where((mcs3['mrtlsts3'].isnull() == 1)
                                , np.nan, mcs3['mssngl3'])
print(mcs3['mssngl3'].value_counts(dropna=False))
print(mcs3['mssngl3'].value_counts().sum())
# Sweep 4
mcs4['mrtlsts4'] = mcs4['dpfcin00']
mcs4.loc[mcs4['mrtlsts4'] < 0, 'mrtlsts4'] = np.nan
mcs4['mssngl4'] = np.where((mcs4['mrtlsts4'] == 1) | (mcs4['mrtlsts4'] == 4) | (mcs4['mrtlsts4'] == 5) | (mcs4['mrtlsts4'] == 6)
                                , 1, 0)
mcs4['mssngl4'] = np.where((mcs4['mrtlsts4'].isnull() == 1)
                                , np.nan, mcs4['mssngl4'])
print(mcs4['mssngl4'].value_counts(dropna=False))
print(mcs4['mssngl4'].value_counts().sum())
# Sweep 5
mcs5['mrtlsts5'] = mcs5['epfcin00']
mcs5.loc[mcs5['mrtlsts5'] < 0, 'mrtlsts5'] = np.nan
mcs5['mssngl5'] = np.where((mcs5['mrtlsts5'] == 1) | (mcs5['mrtlsts5'] == 4) | (mcs5['mrtlsts5'] == 5) | (mcs5['mrtlsts5'] == 6)
                                , 1, 0)
mcs5['mssngl5'] = np.where((mcs5['mrtlsts5'].isnull() == 1)
                                , np.nan, mcs5['mssngl5'])
print(mcs5['mssngl5'].value_counts(dropna=False))
print(mcs5['mssngl5'].value_counts().sum())
# Sweep 6
mcs6['mrtlsts6'] = mcs6['fpfcin00']
mcs6.loc[mcs6['mrtlsts6'] < 0, 'mrtlsts6'] = np.nan
mcs6['mssngl6'] = np.where((mcs6['mrtlsts6'] == 1) | (mcs6['mrtlsts6'] == 4) | (mcs6['mrtlsts6'] == 5) | (mcs6['mrtlsts6'] == 6)
                                , 1, 0)
mcs6['mssngl6'] = np.where((mcs6['mrtlsts6'].isnull() == 1)
                                , np.nan, mcs6['mssngl6'])
print(mcs6['mssngl6'].value_counts(dropna=False))
print(mcs6['mssngl6'].value_counts().sum())
# Sweep 7
mcs7['mrtlsts7'] = mcs7['gpfcin00']
mcs7.loc[mcs7['mrtlsts7'] < 0, 'mrtlsts7'] = np.nan
mcs7['mssngl7'] = np.where((mcs7['mrtlsts7'] == 1) | (mcs7['mrtlsts7'] == 4) | (mcs7['mrtlsts7'] == 5) | (mcs7['mrtlsts7'] == 6)
                                , 1, 0)
mcs7['mssngl7'] = np.where((mcs7['mrtlsts7'].isnull() == 1)
                                , np.nan, mcs7['mssngl7'])
print(mcs7['mssngl7'].value_counts(dropna=False))
print(mcs7['mssngl7'].value_counts().sum())

##Rental housing
# Sweep 1
mcs1['hsngtnr1'] = mcs1['adroow00_x']
mcs1.loc[mcs1['hsngtnr1'] < 0, 'hsngtnr1'] = np.nan
mcs1['htrnt1'] = np.where((mcs1['hsngtnr1'] == 4) | (mcs1['hsngtnr1'] == 5) | (mcs1['hsngtnr1'] == 6)
                                , 1, 0)
mcs1['htrnt1'] = np.where((mcs1['hsngtnr1'].isnull() == 1)
                                , np.nan, mcs1['htrnt1'])
print(mcs1['htrnt1'].value_counts(dropna=False))
print(mcs1['htrnt1'].value_counts().sum())
mcs1['htown1'] = np.where((mcs1['hsngtnr1'] == 1) | (mcs1['hsngtnr1'] == 2) | (mcs1['hsngtnr1'] == 3)
                                , 1, 0)
mcs1['htown1'] = np.where((mcs1['hsngtnr1'].isnull() == 1)
                                , np.nan, mcs1['htown1'])
print(mcs1['htown1'].value_counts(dropna=False))
print(mcs1['htown1'].value_counts().sum())
# Sweep 2
mcs2['hsngtnr2'] = mcs2['bdroow00_x']
mcs2.loc[mcs2['hsngtnr2'] < 0, 'hsngtnr2'] = np.nan
mcs2['htrnt2'] = np.where((mcs2['hsngtnr2'] == 4) | (mcs2['hsngtnr2'] == 5) | (mcs2['hsngtnr2'] == 6)
                                , 1, 0)
mcs2['htrnt2'] = np.where((mcs2['hsngtnr2'].isnull() == 1)
                                , np.nan, mcs2['htrnt2'])
print(mcs2['htrnt2'].value_counts(dropna=False))
print(mcs2['htrnt2'].value_counts().sum())
mcs2['htown2'] = np.where((mcs2['hsngtnr2'] == 1) | (mcs2['hsngtnr2'] == 2) | (mcs2['hsngtnr2'] == 3)
                                , 1, 0)
mcs2['htown2'] = np.where((mcs2['hsngtnr2'].isnull() == 1)
                                , np.nan, mcs2['htown2'])
print(mcs2['htown2'].value_counts(dropna=False))
print(mcs2['htown2'].value_counts().sum())
# Sweep 3
mcs3['hsngtnr3'] = mcs3['cdroow00_x']
mcs3.loc[mcs3['hsngtnr3'] < 0, 'hsngtnr3'] = np.nan
mcs3['htrnt3'] = np.where((mcs3['hsngtnr3'] == 4) | (mcs3['hsngtnr3'] == 5) | (mcs3['hsngtnr3'] == 6)
                                , 1, 0)
mcs3['htrnt3'] = np.where((mcs3['hsngtnr3'].isnull() == 1)
                                , np.nan, mcs3['htrnt3'])
print(mcs3['htrnt3'].value_counts(dropna=False))
print(mcs3['htrnt3'].value_counts().sum())
mcs3['htown3'] = np.where((mcs3['hsngtnr3'] == 1) | (mcs3['hsngtnr3'] == 2) | (mcs3['hsngtnr3'] == 3)
                                , 1, 0)
mcs3['htown3'] = np.where((mcs3['hsngtnr3'].isnull() == 1)
                                , np.nan, mcs3['htown3'])
print(mcs3['htown3'].value_counts(dropna=False))
print(mcs3['htown3'].value_counts().sum())
# Sweep 4
mcs4['hsngtnr4'] = mcs4['ddroow00']
mcs4.loc[mcs4['hsngtnr4'] < 0, 'hsngtnr4'] = np.nan
mcs4['htrnt4'] = np.where((mcs4['hsngtnr4'] == 4) | (mcs4['hsngtnr4'] == 5) | (mcs4['hsngtnr4'] == 6)
                                , 1, 0)
mcs4['htrnt4'] = np.where((mcs4['hsngtnr4'].isnull() == 1)
                                , np.nan, mcs4['htrnt4'])
print(mcs4['htrnt4'].value_counts(dropna=False))
print(mcs4['htrnt4'].value_counts().sum())
mcs4['htown4'] = np.where((mcs4['hsngtnr4'] == 1) | (mcs4['hsngtnr4'] == 2) | (mcs4['hsngtnr4'] == 3)
                                , 1, 0)
mcs4['htown4'] = np.where((mcs4['hsngtnr4'].isnull() == 1)
                                , np.nan, mcs4['htown4'])
print(mcs4['htown4'].value_counts(dropna=False))
print(mcs4['htown4'].value_counts().sum())
# Sweep 5
mcs5['hsngtnr5'] = mcs5['eroow00']
mcs5.loc[mcs5['hsngtnr5'] < 0, 'hsngtnr5'] = np.nan
mcs5['htrnt5'] = np.where((mcs5['hsngtnr5'] == 4) | (mcs5['hsngtnr5'] == 5) | (mcs5['hsngtnr5'] == 6)
                                , 1, 0)
mcs5['htrnt5'] = np.where((mcs5['hsngtnr5'].isnull() == 1)
                                , np.nan, mcs5['htrnt5'])
print(mcs5['htrnt5'].value_counts(dropna=False))
print(mcs5['htrnt5'].value_counts().sum())
mcs5['htown5'] = np.where((mcs5['hsngtnr5'] == 1) | (mcs5['hsngtnr5'] == 2) | (mcs5['hsngtnr5'] == 3)
                                , 1, 0)
mcs5['htown5'] = np.where((mcs5['hsngtnr5'].isnull() == 1)
                                , np.nan, mcs5['htown5'])
print(mcs5['htown5'].value_counts(dropna=False))
print(mcs5['htown5'].value_counts().sum())
# Sweep 6
mcs6['hsngtnr6'] = mcs6['fdroow00']
mcs6.loc[mcs6['hsngtnr6'] < 0, 'hsngtnr6'] = np.nan
mcs6['htrnt6'] = np.where((mcs6['hsngtnr6'] == 4) | (mcs6['hsngtnr6'] == 5) | (mcs6['hsngtnr6'] == 6)
                                , 1, 0)
mcs6['htrnt6'] = np.where((mcs6['hsngtnr6'].isnull() == 1)
                                , np.nan, mcs6['htrnt6'])
print(mcs6['htrnt6'].value_counts(dropna=False))
print(mcs6['htrnt6'].value_counts().sum())
mcs6['htown6'] = np.where((mcs6['hsngtnr6'] == 1) | (mcs6['hsngtnr6'] == 2) | (mcs6['hsngtnr6'] == 3)
                                , 1, 0)
mcs6['htown6'] = np.where((mcs6['hsngtnr6'].isnull() == 1)
                                , np.nan, mcs6['htown6'])
print(mcs6['htown6'].value_counts(dropna=False))
print(mcs6['htown6'].value_counts().sum())
# # Sweep 7
# mcs7['hsngtnr7'] = mcs7['gdroow00_x']
# mcs7.loc[mcs7['hsngtnr7'] < 0, 'hsngtnr7'] = np.nan
# mcs7['htrnt7'] = np.where((mcs7['hsngtnr7'] == 4) | (mcs7['hsngtnr7'] == 5) | (mcs7['hsngtnr7'] == 6)
#                                 , 1, 0)
# mcs7['htrnt7'] = np.where((mcs7['hsngtnr7'].isnull() == 1)
#                                 , np.nan, mcs7['htrnt7'])
# print(mcs7['htrnt7'].value_counts(dropna=False))
# print(mcs7['htrnt7'].value_counts().sum())
# mcs7['htown7'] = np.where((mcs7['hsngtnr7'] == 1) | (mcs7['hsngtnr7'] == 2) | (mcs7['hsngtnr7'] == 3)
#                                 , 1, 0)
# mcs7['htown7'] = np.where((mcs7['hsngtnr7'].isnull() == 1)
#                                 , np.nan, mcs7['htown7'])
# print(mcs7['htown7'].value_counts(dropna=False))
# print(mcs7['htown7'].value_counts().sum())


##Smoking
#Currently smoke/use tobacco products
#Sweep 1
#Main parent
mcs1['smkm1'] = mcs1['apsmus0a']
print(mcs1['smkm1'].value_counts(dropna=False))
print(mcs1['smkm1'].value_counts().sum())
mcs1.loc[mcs1['smkm1'] < 0, 'smkm1'] = np.nan
mcs1['anysmkm1'] = np.where((mcs1['smkm1'] > 1), 1, 0)
mcs1['anysmkm1'] = np.where(mcs1['smkm1'].isnull(), np.nan, mcs1['anysmkm1'])
print(mcs1['anysmkm1'].value_counts(dropna=False))
print(mcs1['anysmkm1'].value_counts().sum())
#Partner
mcs1['smkp1'] = mcs1['apsmus0a_p']
print(mcs1['smkp1'].value_counts(dropna=False))
print(mcs1['smkp1'].value_counts().sum())
mcs1.loc[mcs1['smkp1'] < 0, 'smkp1'] = np.nan
mcs1['anysmkp1'] = np.where((mcs1['smkp1'] > 1), 1, 0)
mcs1['anysmkp1'] = np.where(mcs1['smkp1'].isnull(), np.nan, mcs1['anysmkp1'])
print(mcs1['anysmkp1'].value_counts(dropna=False))
print(mcs1['anysmkp1'].value_counts().sum())
#Sweep 2
#Main parent
mcs2['smkm2'] = mcs2['bpsmus0a']
print(mcs2['smkm2'].value_counts(dropna=False))
print(mcs2['smkm2'].value_counts().sum())
mcs2.loc[mcs2['smkm2'] < 0, 'smkm2'] = np.nan
mcs2['anysmkm2'] = np.where((mcs2['smkm2'] == 1), 0, 1)
mcs2['anysmkm2'] = np.where(mcs2['smkm2'].isnull(), np.nan, mcs2['anysmkm2'])
print(mcs2['anysmkm2'].value_counts(dropna=False))
print(mcs2['anysmkm2'].value_counts().sum())
#Partner
mcs2['smkp2'] = mcs2['bpsmus0a_p']
print(mcs2['smkp2'].value_counts(dropna=False))
print(mcs2['smkp2'].value_counts().sum())
mcs2.loc[mcs2['smkp2'] < 0, 'smkp2'] = np.nan
mcs2['anysmkp2'] = np.where((mcs2['smkp2'] == 1), 0, 1)
mcs2['anysmkp2'] = np.where(mcs2['smkp2'].isnull(), np.nan, mcs2['anysmkp2'])
print(mcs2['anysmkp2'].value_counts(dropna=False))
print(mcs2['anysmkp2'].value_counts().sum())
#Sweep 3
#Main parent
mcs3['smkm3'] = mcs3['cpsmus0a']
print(mcs3['smkm3'].value_counts(dropna=False))
print(mcs3['smkm3'].value_counts().sum())
mcs3.loc[mcs3['smkm3'] < 0, 'smkm3'] = np.nan
mcs3['anysmkm3'] = np.where((mcs3['smkm3'] > 1), 1, 0)
mcs3['anysmkm3'] = np.where(mcs3['smkm3'].isnull(), np.nan, mcs3['anysmkm3'])
print(mcs3['anysmkm3'].value_counts(dropna=False))
print(mcs3['anysmkm3'].value_counts().sum())
#Partner
mcs3['smkp3'] = mcs3['cpsmus0a_p']
print(mcs3['smkp3'].value_counts(dropna=False))
print(mcs3['smkp3'].value_counts().sum())
mcs3.loc[mcs3['smkp3'] < 0, 'smkp3'] = np.nan
mcs3['anysmkp3'] = np.where((mcs3['smkp3'] > 1), 1, 0)
mcs3['anysmkp3'] = np.where(mcs3['smkp3'].isnull(), np.nan, mcs3['anysmkp3'])
print(mcs3['anysmkp3'].value_counts(dropna=False))
print(mcs3['anysmkp3'].value_counts().sum())
#Sweep 4
#Main parent
mcs4['smkm4'] = mcs4['dpsmus0a']
print(mcs4['smkm4'].value_counts(dropna=False))
print(mcs4['smkm4'].value_counts().sum())
mcs4.loc[mcs4['smkm4'] < 0, 'smkm4'] = np.nan
mcs4['anysmkm4'] = np.where((mcs4['smkm4'] > 1), 1, 0)
mcs4['anysmkm4'] = np.where(mcs4['smkm4'].isnull(), np.nan, mcs4['anysmkm4'])
print(mcs4['anysmkm4'].value_counts(dropna=False))
print(mcs4['anysmkm4'].value_counts().sum())
#Partner
mcs4['smkp4'] = mcs4['dpsmus0a_p']
print(mcs4['smkp4'].value_counts(dropna=False))
print(mcs4['smkp4'].value_counts().sum())
mcs4.loc[mcs4['smkp4'] < 0, 'smkp4'] = np.nan
mcs4['anysmkp4'] = np.where((mcs4['smkp4'] > 1), 1, 0)
mcs4['anysmkp4'] = np.where(mcs4['smkp4'].isnull(), np.nan, mcs4['anysmkp4'])
print(mcs4['anysmkp4'].value_counts(dropna=False))
print(mcs4['anysmkp4'].value_counts().sum())
#Sweep 5
#Main parent
mcs5['smkm5'] = mcs5['epsmus0a']
print(mcs5['smkm5'].value_counts(dropna=False))
print(mcs5['smkm5'].value_counts().sum())
mcs5.loc[mcs5['smkm5'] < 0, 'smkm5'] = np.nan
mcs5['anysmkm5'] = np.where((mcs5['smkm5'] == 1), 0, 1)
mcs5['anysmkm5'] = np.where(mcs5['smkm5'].isnull(), np.nan, mcs5['anysmkm5'])
print(mcs5['anysmkm5'].value_counts(dropna=False))
print(mcs5['anysmkm5'].value_counts().sum())
#Partner
mcs5['smkp5'] = mcs5['epsmus0a_p']
print(mcs5['smkp5'].value_counts(dropna=False))
print(mcs5['smkp5'].value_counts().sum())
mcs5.loc[mcs5['smkp5'] < 0, 'smkp5'] = np.nan
mcs5['anysmkp5'] = np.where((mcs5['smkp5'] == 1), 0, 1)
mcs5['anysmkp5'] = np.where(mcs5['smkp5'].isnull(), np.nan, mcs5['anysmkp5'])
print(mcs5['anysmkp5'].value_counts(dropna=False))
print(mcs5['anysmkp5'].value_counts().sum())
#Sweep 6
#Main parent
mcs6['smkm6'] = mcs6['fpsmus0a']
print(mcs6['smkm6'].value_counts(dropna=False))
print(mcs6['smkm6'].value_counts().sum())
mcs6.loc[mcs6['smkm6'] < 0, 'smkm6'] = np.nan
mcs6['anysmkm6'] = np.where((mcs6['smkm6'] == 1), 0, 1)
mcs6['anysmkm6'] = np.where(mcs6['smkm6'].isnull(), np.nan, mcs6['anysmkm6'])
print(mcs6['anysmkm6'].value_counts(dropna=False))
print(mcs6['anysmkm6'].value_counts().sum())
#Partner
mcs6['smkp6'] = mcs6['fpsmus0a_p']
print(mcs6['smkp6'].value_counts(dropna=False))
print(mcs6['smkp6'].value_counts().sum())
mcs6.loc[mcs6['smkp6'] < 0, 'smkp6'] = np.nan
mcs6['anysmkp6'] = np.where((mcs6['smkp6'] == 1), 0, 1)
mcs6['anysmkp6'] = np.where(mcs6['smkp6'].isnull(), np.nan, mcs6['anysmkp6'])
print(mcs6['anysmkp6'].value_counts(dropna=False))
print(mcs6['anysmkp6'].value_counts().sum())

##Drinking
#Any Drinking (Never) and regular drinking (>1-2 times a week)
#                Refusal     -9
#             Don't Know     -8
#         Not applicable     -1
#              Every day      1
#     5-6 times per week      2
#     3-4 times per week      3
#     1-2 times per week      4
#    1-2 times per month      5
# Less than once a month      6
#                  Never      7

#Sweep 1
#Main parent
mcs1['drnkm1'] = mcs1['apaldr00']
print(mcs1['drnkm1'].value_counts(dropna=False))
print(mcs1['drnkm1'].value_counts().sum())
mcs1.loc[mcs1['drnkm1'] < 0, 'drnkm1'] = np.nan
mcs1['anydrnkm1'] = np.where((mcs1['drnkm1'] == 7), 0, 1)
mcs1['anydrnkm1'] = np.where(mcs1['drnkm1'].isnull(), np.nan, mcs1['anydrnkm1'])
print(mcs1['anydrnkm1'].value_counts(dropna=False))
print(mcs1['anydrnkm1'].value_counts().sum())
mcs1['regdrnkm1'] = np.where((mcs1['drnkm1'] == 1) | (mcs1['drnkm1'] == 2) | (mcs1['drnkm1'] == 3)
                                , 1, 0)
mcs1['regdrnkm1'] = np.where(mcs1['drnkm1'].isnull(), np.nan, mcs1['regdrnkm1'])
print(mcs1['regdrnkm1'].value_counts(dropna=False))
print(mcs1['regdrnkm1'].value_counts().sum())
#Partner
mcs1['drnkp1'] = mcs1['apaldr00_p']
print(mcs1['drnkp1'].value_counts(dropna=False))
print(mcs1['drnkp1'].value_counts().sum())
mcs1.loc[mcs1['drnkp1'] < 0, 'drnkp1'] = np.nan
mcs1['anydrnkp1'] = np.where((mcs1['drnkp1'] == 7), 0, 1)
mcs1['anydrnkp1'] = np.where(mcs1['drnkp1'].isnull(), np.nan, mcs1['anydrnkp1'])
print(mcs1['anydrnkp1'].value_counts(dropna=False))
print(mcs1['anydrnkp1'].value_counts().sum())
mcs1['regdrnkp1'] = np.where((mcs1['drnkp1'] == 1) | (mcs1['drnkp1'] == 2) | (mcs1['drnkp1'] == 3)
                                , 1, 0)
mcs1['regdrnkp1'] = np.where(mcs1['drnkp1'].isnull(), np.nan, mcs1['regdrnkp1'])
print(mcs1['regdrnkp1'].value_counts(dropna=False))
print(mcs1['regdrnkp1'].value_counts().sum())
#Sweep 2
#Main parent
mcs2['drnkm2'] = mcs2['bpaldr00']
print(mcs2['drnkm2'].value_counts(dropna=False))
print(mcs2['drnkm2'].value_counts().sum())
mcs2.loc[mcs2['drnkm2'] < 0, 'drnkm2'] = np.nan
mcs2['anydrnkm2'] = np.where((mcs2['drnkm2'] == 7), 0, 1)
mcs2['anydrnkm2'] = np.where(mcs2['drnkm2'].isnull(), np.nan, mcs2['anydrnkm2'])
print(mcs2['anydrnkm2'].value_counts(dropna=False))
print(mcs2['anydrnkm2'].value_counts().sum())
mcs2['regdrnkm2'] = np.where((mcs2['drnkm2'] == 1) | (mcs2['drnkm2'] == 2) | (mcs2['drnkm2'] == 3)
                                , 1, 0)
mcs2['regdrnkm2'] = np.where(mcs2['drnkm2'].isnull(), np.nan, mcs2['regdrnkm2'])
print(mcs2['regdrnkm2'].value_counts(dropna=False))
print(mcs2['regdrnkm2'].value_counts().sum())
#Partner
mcs2['drnkp2'] = mcs2['bpaldr00_p']
print(mcs2['drnkp2'].value_counts(dropna=False))
print(mcs2['drnkp2'].value_counts().sum())
mcs2.loc[mcs2['drnkp2'] < 0, 'drnkp2'] = np.nan
mcs2['anydrnkp2'] = np.where((mcs2['drnkp2'] == 7), 0, 1)
mcs2['anydrnkp2'] = np.where(mcs2['drnkp2'].isnull(), np.nan, mcs2['anydrnkp2'])
print(mcs2['anydrnkp2'].value_counts(dropna=False))
print(mcs2['anydrnkp2'].value_counts().sum())
mcs2['regdrnkp2'] = np.where((mcs2['drnkp2'] == 1) | (mcs2['drnkp2'] == 2) | (mcs2['drnkp2'] == 3)
                                , 1, 0)
mcs2['regdrnkp2'] = np.where(mcs2['drnkp2'].isnull(), np.nan, mcs2['regdrnkp2'])
print(mcs2['regdrnkp2'].value_counts(dropna=False))
print(mcs2['regdrnkp2'].value_counts().sum())
#Sweep 3
#Main parent
mcs3['drnkm3'] = mcs3['cpaldr00']
print(mcs3['drnkm3'].value_counts(dropna=False))
print(mcs3['drnkm3'].value_counts().sum())
mcs3.loc[mcs3['drnkm3'] < 0, 'drnkm3'] = np.nan
mcs3['anydrnkm3'] = np.where((mcs3['drnkm3'] == 7), 0, 1)
mcs3['anydrnkm3'] = np.where(mcs3['drnkm3'].isnull(), np.nan, mcs3['anydrnkm3'])
print(mcs3['anydrnkm3'].value_counts(dropna=False))
print(mcs3['anydrnkm3'].value_counts().sum())
mcs3['regdrnkm3'] = np.where((mcs3['drnkm3'] == 1) | (mcs3['drnkm3'] == 2) | (mcs3['drnkm3'] == 3)
                                , 1, 0)
mcs3['regdrnkm3'] = np.where(mcs3['drnkm3'].isnull(), np.nan, mcs3['regdrnkm3'])
print(mcs3['regdrnkm3'].value_counts(dropna=False))
print(mcs3['regdrnkm3'].value_counts().sum())
#Partner
mcs3['drnkp3'] = mcs3['cpaldr00_p']
print(mcs3['drnkp3'].value_counts(dropna=False))
print(mcs3['drnkp3'].value_counts().sum())
mcs3.loc[mcs3['drnkp3'] < 0, 'drnkp3'] = np.nan
mcs3['anydrnkp3'] = np.where((mcs3['drnkp3'] == 7), 0, 1)
mcs3['anydrnkp3'] = np.where(mcs3['drnkp3'].isnull(), np.nan, mcs3['anydrnkp3'])
print(mcs3['anydrnkp3'].value_counts(dropna=False))
print(mcs3['anydrnkp3'].value_counts().sum())
mcs3['regdrnkp3'] = np.where((mcs3['drnkp3'] == 1) | (mcs3['drnkp3'] == 2) | (mcs3['drnkp3'] == 3)
                                , 1, 0)
mcs3['regdrnkp3'] = np.where(mcs3['drnkp3'].isnull(), np.nan, mcs3['regdrnkp3'])
print(mcs3['regdrnkp3'].value_counts(dropna=False))
print(mcs3['regdrnkp3'].value_counts().sum())
#Sweep 4
#Main parent
mcs4['drnkm4'] = mcs4['dpaldr00']
print(mcs4['drnkm4'].value_counts(dropna=False))
print(mcs4['drnkm4'].value_counts().sum())
mcs4.loc[mcs4['drnkm4'] < 0, 'drnkm4'] = np.nan
mcs4['anydrnkm4'] = np.where((mcs4['drnkm4'] == 7), 0, 1)
mcs4['anydrnkm4'] = np.where(mcs4['drnkm4'].isnull(), np.nan, mcs4['anydrnkm4'])
print(mcs4['anydrnkm4'].value_counts(dropna=False))
print(mcs4['anydrnkm4'].value_counts().sum())
mcs4['regdrnkm4'] = np.where((mcs4['drnkm4'] == 1) | (mcs4['drnkm4'] == 2) | (mcs4['drnkm4'] == 3)
                                , 1, 0)
mcs4['regdrnkm4'] = np.where(mcs4['drnkm4'].isnull(), np.nan, mcs4['regdrnkm4'])
print(mcs4['regdrnkm4'].value_counts(dropna=False))
print(mcs4['regdrnkm4'].value_counts().sum())
#Partner
mcs4['drnkp4'] = mcs4['dpaldr00_p']
print(mcs4['drnkp4'].value_counts(dropna=False))
print(mcs4['drnkp4'].value_counts().sum())
mcs4.loc[mcs4['drnkp4'] < 0, 'drnkp4'] = np.nan
mcs4['anydrnkp4'] = np.where((mcs4['drnkp4'] == 7), 0, 1)
mcs4['anydrnkp4'] = np.where(mcs4['drnkp4'].isnull(), np.nan, mcs4['anydrnkp4'])
print(mcs4['anydrnkp4'].value_counts(dropna=False))
print(mcs4['anydrnkp4'].value_counts().sum())
mcs4['regdrnkp4'] = np.where((mcs4['drnkp4'] == 1) | (mcs4['drnkp4'] == 2) | (mcs4['drnkp4'] == 3)
                                , 1, 0)
mcs4['regdrnkp4'] = np.where(mcs4['drnkp4'].isnull(), np.nan, mcs4['regdrnkp4'])
print(mcs4['regdrnkp4'].value_counts(dropna=False))
print(mcs4['regdrnkp4'].value_counts().sum())
#Sweep 5
#                  Not applicable     -1
#          4 or more times a week      1
#                2-3 times a week      2
#             2-4 times per month      3
#                 Monthly or less      4
#                           Never      5
# Don't know/don't wish to answer      6
#Main parent
mcs5['drnkm5'] = mcs5['epaldr00']
print(mcs5['drnkm5'].value_counts(dropna=False))
print(mcs5['drnkm5'].value_counts().sum())
mcs5.loc[mcs5['drnkm5'] < 0, 'drnkm5'] = np.nan
mcs5['anydrnkm5'] = np.where((mcs5['drnkm5'] == 5), 0, 1)
mcs5['anydrnkm5'] = np.where(mcs5['drnkm5'].isnull(), np.nan, mcs5['anydrnkm5'])
print(mcs5['anydrnkm5'].value_counts(dropna=False))
print(mcs5['anydrnkm5'].value_counts().sum())
mcs5['regdrnkm5'] = np.where((mcs5['drnkm5'] == 1) | (mcs5['drnkm5'] == 2)
                                , 1, 0)
mcs5['regdrnkm5'] = np.where(mcs5['drnkm5'].isnull(), np.nan, mcs5['regdrnkm5'])
print(mcs5['regdrnkm5'].value_counts(dropna=False))
print(mcs5['regdrnkm5'].value_counts().sum())
#Partner
mcs5['drnkp5'] = mcs5['epaldr00_p']
print(mcs5['drnkp5'].value_counts(dropna=False))
print(mcs5['drnkp5'].value_counts().sum())
mcs5.loc[mcs5['drnkp5'] < 0, 'drnkp5'] = np.nan
mcs5['anydrnkp5'] = np.where((mcs5['drnkp5'] == 5), 0, 1)
mcs5['anydrnkp5'] = np.where(mcs5['drnkp5'].isnull(), np.nan, mcs5['anydrnkp5'])
print(mcs5['anydrnkp5'].value_counts(dropna=False))
print(mcs5['anydrnkp5'].value_counts().sum())
mcs5['regdrnkp5'] = np.where((mcs5['drnkp5'] == 1) | (mcs5['drnkp5'] == 2)
                                , 1, 0)
mcs5['regdrnkp5'] = np.where(mcs5['drnkp5'].isnull(), np.nan, mcs5['regdrnkp5'])
print(mcs5['regdrnkp5'].value_counts(dropna=False))
print(mcs5['regdrnkp5'].value_counts().sum())
#Sweep 6
#Main parent
mcs6['drnkm6'] = mcs6['fpaldr00']
print(mcs6['drnkm6'].value_counts(dropna=False))
print(mcs6['drnkm6'].value_counts().sum())
mcs6.loc[mcs6['drnkm6'] < 0, 'drnkm6'] = np.nan
mcs6['anydrnkm6'] = np.where((mcs6['drnkm6'] == 5), 0, 1)
mcs6['anydrnkm6'] = np.where(mcs6['drnkm6'].isnull(), np.nan, mcs6['anydrnkm6'])
print(mcs6['anydrnkm6'].value_counts(dropna=False))
print(mcs6['anydrnkm6'].value_counts().sum())
mcs6['regdrnkm6'] = np.where((mcs6['drnkm6'] == 1) | (mcs6['drnkm6'] == 2)
                                , 1, 0)
mcs6['regdrnkm6'] = np.where(mcs6['drnkm6'].isnull(), np.nan, mcs6['regdrnkm6'])
print(mcs6['regdrnkm6'].value_counts(dropna=False))
print(mcs6['regdrnkm6'].value_counts().sum())
#Partner
mcs6['drnkp6'] = mcs6['fpaldr00_p']
print(mcs6['drnkp6'].value_counts(dropna=False))
print(mcs6['drnkp6'].value_counts().sum())
mcs6.loc[mcs6['drnkp6'] < 0, 'drnkp6'] = np.nan
mcs6['anydrnkp6'] = np.where((mcs6['drnkp6'] == 5), 0, 1)
mcs6['anydrnkp6'] = np.where(mcs6['drnkp6'].isnull(), np.nan, mcs6['anydrnkp6'])
print(mcs6['anydrnkp6'].value_counts(dropna=False))
print(mcs6['anydrnkp6'].value_counts().sum())
mcs6['regdrnkp6'] = np.where((mcs6['drnkp6'] == 1) | (mcs6['drnkp6'] == 2)
                                , 1, 0)
mcs6['regdrnkp6'] = np.where(mcs6['drnkp6'].isnull(), np.nan, mcs6['regdrnkp6'])
print(mcs6['regdrnkp6'].value_counts(dropna=False))
print(mcs6['regdrnkp6'].value_counts().sum())

##Drug use
#Used recreational drugs (not never)

#       Refusal     -9
#    Don't Know     -8
#Not applicable     -1
#  Occasionally      1
#     Regularly      2
#         Never      3
#     Can't say      4

#Sweep 2
#Main parent
mcs2['drgm2'] = mcs2['bpdrug00']
print(mcs2['drgm2'].value_counts(dropna=False))
print(mcs2['drgm2'].value_counts().sum())
mcs2.loc[mcs2['drgm2'] < 0, 'drgm2'] = np.nan
mcs2['anydrgm2'] = np.where((mcs2['drgm2'] == 3), 0, 1)
mcs2['anydrgm2'] = np.where(mcs2['drgm2'].isnull(), np.nan, mcs2['anydrgm2'])
print(mcs2['anydrgm2'].value_counts(dropna=False))
print(mcs2['anydrgm2'].value_counts().sum())
#Partner
mcs2['drgp2'] = mcs2['bpdrug00_p']
print(mcs2['drgp2'].value_counts(dropna=False))
print(mcs2['drgp2'].value_counts().sum())
mcs2.loc[mcs2['drgp2'] < 0, 'drgp2'] = np.nan
mcs2['anydrgp2'] = np.where((mcs2['drgp2'] == 3), 0, 1)
mcs2['anydrgp2'] = np.where(mcs2['drgp2'].isnull(), np.nan, mcs2['anydrgp2'])
print(mcs2['anydrgp2'].value_counts(dropna=False))
print(mcs2['anydrgp2'].value_counts().sum())
#Sweep 3
#Main parent
mcs3['drgm3'] = mcs3['cpdrug00']
print(mcs3['drgm3'].value_counts(dropna=False))
print(mcs3['drgm3'].value_counts().sum())
mcs3.loc[mcs3['drgm3'] < 0, 'drgm3'] = np.nan
mcs3['anydrgm3'] = np.where((mcs3['drgm3'] == 3), 0, 1)
mcs3['anydrgm3'] = np.where(mcs3['drgm3'].isnull(), np.nan, mcs3['anydrgm3'])
print(mcs3['anydrgm3'].value_counts(dropna=False))
print(mcs3['anydrgm3'].value_counts().sum())
#Partner
mcs3['drgp3'] = mcs3['cpdrug00_p']
print(mcs3['drgp3'].value_counts(dropna=False))
print(mcs3['drgp3'].value_counts().sum())
mcs3.loc[mcs3['drgp3'] < 0, 'drgp3'] = np.nan
mcs3['anydrgp3'] = np.where((mcs3['drgp3'] == 3), 0, 1)
mcs3['anydrgp3'] = np.where(mcs3['drgp3'].isnull(), np.nan, mcs3['anydrgp3'])
print(mcs3['anydrgp3'].value_counts(dropna=False))
print(mcs3['anydrgp3'].value_counts().sum())
#Sweep 6
#Main parent
mcs6['drgm6'] = mcs6['fpdrug00']
print(mcs6['drgm6'].value_counts(dropna=False))
print(mcs6['drgm6'].value_counts().sum())
mcs6.loc[mcs6['drgm6'] < 0, 'drgm6'] = np.nan
mcs6['anydrgm6'] = np.where((mcs6['drgm6'] == 3), 0, 1)
mcs6['anydrgm6'] = np.where(mcs6['drgm6'].isnull(), np.nan, mcs6['anydrgm6'])
print(mcs6['anydrgm6'].value_counts(dropna=False))
print(mcs6['anydrgm6'].value_counts().sum())
#Partner
mcs6['drgp6'] = mcs6['fpdrug00_p']
print(mcs6['drgp6'].value_counts(dropna=False))
print(mcs6['drgp6'].value_counts().sum())
mcs6.loc[mcs6['drgp6'] < 0, 'drgp6'] = np.nan
mcs6['anydrgp6'] = np.where((mcs6['drgp6'] == 3), 0, 1)
mcs6['anydrgp6'] = np.where(mcs6['drgp6'].isnull(), np.nan, mcs6['anydrgp6'])
print(mcs6['anydrgp6'].value_counts(dropna=False))
print(mcs6['anydrgp6'].value_counts().sum())


##Main parent's Mental health
#Kessler score
# Sweep 2
mcs2['mmhlth2'] = mcs2['bdkess00']
print(mcs2['mmhlth2'].value_counts(dropna=False))
print(mcs2['mmhlth2'].value_counts().sum())
mcs2.loc[mcs2['mmhlth2'] < 0, 'mmhlth2'] = np.nan
print(mcs2['mmhlth2'].value_counts(dropna=False))
print(mcs2['mmhlth2'].value_counts().sum())
# Sweep 3
mcs3['mmhlth3'] = mcs3['cdkess00']
print(mcs3['mmhlth3'].value_counts(dropna=False))
print(mcs3['mmhlth3'].value_counts().sum())
mcs3.loc[mcs3['mmhlth3'] < 0, 'mmhlth3'] = np.nan
print(mcs3['mmhlth3'].value_counts(dropna=False))
print(mcs3['mmhlth3'].value_counts().sum())
# Sweep 4
mcs4['mmhlth4'] = mcs4['ddkessler']
print(mcs4['mmhlth4'].value_counts(dropna=False))
print(mcs4['mmhlth4'].value_counts().sum())
mcs4.loc[mcs4['mmhlth4'] < 0, 'mmhlth4'] = np.nan
print(mcs4['mmhlth4'].value_counts(dropna=False))
print(mcs4['mmhlth4'].value_counts().sum())
# Sweep 5
mcs5['mmhlth5'] = mcs5['edkessl']
print(mcs5['mmhlth5'].value_counts(dropna=False))
print(mcs5['mmhlth5'].value_counts().sum())
mcs5.loc[mcs5['mmhlth5'] < 0, 'mmhlth5'] = np.nan
print(mcs5['mmhlth5'].value_counts(dropna=False))
print(mcs5['mmhlth5'].value_counts().sum())
# Sweep 6
mcs6['mmhlth6'] = mcs6['fdkessl']
print(mcs6['mmhlth6'].value_counts(dropna=False))
print(mcs6['mmhlth6'].value_counts().sum())
mcs6.loc[mcs6['mmhlth6'] < 0, 'mmhlth6'] = np.nan
print(mcs6['mmhlth6'].value_counts(dropna=False))
print(mcs6['mmhlth6'].value_counts().sum())



##Partner's Mental health
#Kessler score
# Sweep 2
mcs2['pmhlth2'] = mcs2['bdkess00_p']
print(mcs2['pmhlth2'].value_counts(dropna=False))
print(mcs2['pmhlth2'].value_counts().sum())
mcs2.loc[mcs2['pmhlth2'] < 0, 'pmhlth2'] = np.nan
print(mcs2['pmhlth2'].value_counts(dropna=False))
print(mcs2['pmhlth2'].value_counts().sum())
# Sweep 3
mcs3['pmhlth3'] = mcs3['cdkess00_p']
print(mcs3['pmhlth3'].value_counts(dropna=False))
print(mcs3['pmhlth3'].value_counts().sum())
mcs3.loc[mcs3['pmhlth3'] < 0, 'pmhlth3'] = np.nan
print(mcs3['pmhlth3'].value_counts(dropna=False))
print(mcs3['pmhlth3'].value_counts().sum())
# Sweep 4
mcs4['pmhlth4'] = mcs4['ddkessler_p']
print(mcs4['pmhlth4'].value_counts(dropna=False))
print(mcs4['pmhlth4'].value_counts().sum())
mcs4.loc[mcs4['pmhlth4'] < 0, 'pmhlth4'] = np.nan
print(mcs4['pmhlth4'].value_counts(dropna=False))
print(mcs4['pmhlth4'].value_counts().sum())
# Sweep 5
mcs5['pmhlth5'] = mcs5['edkessl_p']
print(mcs5['pmhlth5'].value_counts(dropna=False))
print(mcs5['pmhlth5'].value_counts().sum())
mcs5.loc[mcs5['pmhlth5'] < 0, 'pmhlth5'] = np.nan
print(mcs5['pmhlth5'].value_counts(dropna=False))
print(mcs5['pmhlth5'].value_counts().sum())
# Sweep 6
mcs6['pmhlth6'] = mcs6['fdkessl_p']
print(mcs6['pmhlth6'].value_counts(dropna=False))
print(mcs6['pmhlth6'].value_counts().sum())
mcs6.loc[mcs6['pmhlth6'] < 0, 'pmhlth6'] = np.nan
print(mcs6['pmhlth6'].value_counts(dropna=False))
print(mcs6['pmhlth6'].value_counts().sum())




##Main parent's Highest degree
#NVQ education levels
# Sweep 1
mcs1['meduc1'] = mcs1['adacaq00']
print(mcs1['meduc1'].value_counts(dropna=False))
print(mcs1['meduc1'].value_counts().sum())
mcs1.loc[mcs1['meduc1'] < 0, 'meduc1'] = np.nan
mcs1['meduc1'] = np.where((mcs1['meduc1'] == 96)
                                , 0, mcs1['meduc1'])
mcs1['meduc1'] = np.where((mcs1['meduc1'] == 95)
                                , np.nan, mcs1['meduc1'])
print(mcs1['meduc1'].value_counts(dropna=False))
print(mcs1['meduc1'].value_counts().sum())
# Sweep 2
mcs2['meduc2'] = mcs2['bdacaq00']
print(mcs2['meduc2'].value_counts(dropna=False))
print(mcs2['meduc2'].value_counts().sum())
mcs2.loc[mcs2['meduc2'] < 0, 'meduc2'] = np.nan
mcs2['meduc2'] = np.where((mcs2['meduc2'] == 96)
                                , 0, mcs2['meduc2'])
mcs2['meduc2'] = np.where((mcs2['meduc2'] == 95)
                                , np.nan, mcs2['meduc2'])
print(mcs2['meduc2'].value_counts(dropna=False))
print(mcs2['meduc2'].value_counts().sum())
# Sweep 3
mcs3['meduc3'] = mcs3['cdacaq00']
print(mcs3['meduc3'].value_counts(dropna=False))
print(mcs3['meduc3'].value_counts().sum())
mcs3.loc[mcs3['meduc3'] < 0, 'meduc3'] = np.nan
mcs3['meduc3'] = np.where((mcs3['meduc3'] == 96)
                                , 0, mcs3['meduc3'])
mcs3['meduc3'] = np.where((mcs3['meduc3'] == 95)
                                , np.nan, mcs3['meduc3'])
print(mcs3['meduc3'].value_counts(dropna=False))
print(mcs3['meduc3'].value_counts().sum())
# Sweep 4
mcs4['meduc4'] = mcs4['ddacaq00']
print(mcs4['meduc4'].value_counts(dropna=False))
print(mcs4['meduc4'].value_counts().sum())
mcs4.loc[mcs4['meduc4'] < 0, 'meduc4'] = np.nan
mcs4['meduc4'] = np.where((mcs4['meduc4'] == 96)
                                , 0, mcs4['meduc4'])
mcs4['meduc4'] = np.where((mcs4['meduc4'] == 95)
                                , np.nan, mcs4['meduc4'])
print(mcs4['meduc4'].value_counts(dropna=False))
print(mcs4['meduc4'].value_counts().sum())
# Sweep 5
mcs5['meduc5'] = mcs5['eacaq00']
print(mcs5['meduc5'].value_counts(dropna=False))
print(mcs5['meduc5'].value_counts().sum())
mcs5.loc[mcs5['meduc5'] < 0, 'meduc5'] = np.nan
mcs5['meduc5'] = np.where((mcs5['meduc5'] == 96)
                                , 0, mcs5['meduc5'])
mcs5['meduc5'] = np.where((mcs5['meduc5'] == 95)
                                , np.nan, mcs5['meduc5'])
print(mcs5['meduc5'].value_counts(dropna=False))
print(mcs5['meduc5'].value_counts().sum())
# Sweep 6
mcs6['meduc6'] = mcs6['fdacaq00']
print(mcs6['meduc6'].value_counts(dropna=False))
print(mcs6['meduc6'].value_counts().sum())
mcs6.loc[mcs6['meduc6'] < 0, 'meduc6'] = np.nan
mcs6['meduc6'] = np.where((mcs6['meduc6'] == 96)
                                , 0, mcs6['meduc6'])
mcs6['meduc6'] = np.where((mcs6['meduc6'] == 95)
                                , np.nan, mcs6['meduc6'])
print(mcs6['meduc6'].value_counts(dropna=False))
print(mcs6['meduc6'].value_counts().sum())

##Partner's Highest degree
#NVQ education levels
# Sweep 1
mcs1['peduc1'] = mcs1['adacaq00_p']
print(mcs1['peduc1'].value_counts(dropna=False))
print(mcs1['peduc1'].value_counts().sum())
mcs1.loc[mcs1['peduc1'] < 0, 'peduc1'] = np.nan
mcs1['peduc1'] = np.where((mcs1['peduc1'] == 96)
                                , 0, mcs1['peduc1'])
mcs1['peduc1'] = np.where((mcs1['peduc1'] == 95)
                                , np.nan, mcs1['peduc1'])
print(mcs1['peduc1'].value_counts(dropna=False))
print(mcs1['peduc1'].value_counts().sum())
# Sweep 2
mcs2['peduc2'] = mcs2['bdacaq00_p']
print(mcs2['peduc2'].value_counts(dropna=False))
print(mcs2['peduc2'].value_counts().sum())
mcs2.loc[mcs2['peduc2'] < 0, 'peduc2'] = np.nan
mcs2['peduc2'] = np.where((mcs2['peduc2'] == 96)
                                , 0, mcs2['peduc2'])
mcs2['peduc2'] = np.where((mcs2['peduc2'] == 95)
                                , np.nan, mcs2['peduc2'])
print(mcs2['peduc2'].value_counts(dropna=False))
print(mcs2['peduc2'].value_counts().sum())
# Sweep 3
mcs3['peduc3'] = mcs3['cdacaq00_p']
print(mcs3['peduc3'].value_counts(dropna=False))
print(mcs3['peduc3'].value_counts().sum())
mcs3.loc[mcs3['peduc3'] < 0, 'peduc3'] = np.nan
mcs3['peduc3'] = np.where((mcs3['peduc3'] == 96)
                                , 0, mcs3['peduc3'])
mcs3['peduc3'] = np.where((mcs3['peduc3'] == 95)
                                , np.nan, mcs3['peduc3'])
print(mcs3['peduc3'].value_counts(dropna=False))
print(mcs3['peduc3'].value_counts().sum())
# Sweep 4
mcs4['peduc4'] = mcs4['ddacaq00_p']
print(mcs4['peduc4'].value_counts(dropna=False))
print(mcs4['peduc4'].value_counts().sum())
mcs4.loc[mcs4['peduc4'] < 0, 'peduc4'] = np.nan
mcs4['peduc4'] = np.where((mcs4['peduc4'] == 96)
                                , 0, mcs4['peduc4'])
mcs4['peduc4'] = np.where((mcs4['peduc4'] == 95)
                                , np.nan, mcs4['peduc4'])
print(mcs4['peduc4'].value_counts(dropna=False))
print(mcs4['peduc4'].value_counts().sum())
# Sweep 5
mcs5['peduc5'] = mcs5['eacaq00_p']
print(mcs5['peduc5'].value_counts(dropna=False))
print(mcs5['peduc5'].value_counts().sum())
mcs5.loc[mcs5['peduc5'] < 0, 'peduc5'] = np.nan
mcs5['peduc5'] = np.where((mcs5['peduc5'] == 96)
                                , 0, mcs5['peduc5'])
mcs5['peduc5'] = np.where((mcs5['peduc5'] == 95)
                                , np.nan, mcs5['peduc5'])
print(mcs5['peduc5'].value_counts(dropna=False))
print(mcs5['peduc5'].value_counts().sum())
# Sweep 6
mcs6['peduc6'] = mcs6['fdacaq00_p']
print(mcs6['peduc6'].value_counts(dropna=False))
print(mcs6['peduc6'].value_counts().sum())
mcs6.loc[mcs6['peduc6'] < 0, 'peduc6'] = np.nan
mcs6['peduc6'] = np.where((mcs6['peduc6'] == 96)
                                , 0, mcs6['peduc6'])
mcs6['peduc6'] = np.where((mcs6['peduc6'] == 95)
                                , np.nan, mcs6['peduc6'])
print(mcs6['peduc6'].value_counts(dropna=False))
print(mcs6['peduc6'].value_counts().sum())

### Grand parent's highest degrees
#Maternal grandparents
mcs7['gmeduc7'] = mcs7['gpmacq00']
print(mcs7['gmeduc7'].value_counts(dropna=False))
print(mcs7['gmeduc7'].value_counts().sum())
mcs7.loc[mcs7['gmeduc7'] < 0, 'gmeduc7'] = np.nan
mcs7['gmeduc7'] = np.where((mcs7['gmeduc7'] == 7) | (mcs7['gmeduc7'] == 8) | (mcs7['gmeduc7'] == 9)
                                , np.nan, mcs7['gmeduc7'])
print(mcs7['gmeduc7'].value_counts(dropna=False))
print(mcs7['gmeduc7'].value_counts().sum())
mcs7['gfeduc7'] = mcs7['gpfacq00']
print(mcs7['gfeduc7'].value_counts(dropna=False))
print(mcs7['gfeduc7'].value_counts().sum())
mcs7.loc[mcs7['gfeduc7'] < 0, 'gfeduc7'] = np.nan
mcs7['gfeduc7'] = np.where((mcs7['gfeduc7'] == 7) | (mcs7['gfeduc7'] == 8) | (mcs7['gfeduc7'] == 9)
                                , np.nan, mcs7['gfeduc7'])
print(mcs7['gfeduc7'].value_counts(dropna=False))
print(mcs7['gfeduc7'].value_counts().sum())



#Fraternal grandparents
mcs7['pgmeduc7'] = mcs7['gpmacq00_p']
print(mcs7['pgmeduc7'].value_counts(dropna=False))
print(mcs7['pgmeduc7'].value_counts().sum())
mcs7.loc[mcs7['pgmeduc7'] < 0, 'pgmeduc7'] = np.nan
mcs7['pgmeduc7'] = np.where((mcs7['pgmeduc7'] == 7) | (mcs7['pgmeduc7'] == 8) | (mcs7['pgmeduc7'] == 9)
                                , np.nan, mcs7['pgmeduc7'])
print(mcs7['pgmeduc7'].value_counts(dropna=False))
print(mcs7['pgmeduc7'].value_counts().sum())
mcs7['pgfeduc7'] = mcs7['gpfacq00_p']
print(mcs7['pgfeduc7'].value_counts(dropna=False))
print(mcs7['pgfeduc7'].value_counts().sum())
mcs7.loc[mcs7['pgfeduc7'] < 0, 'pgfeduc7'] = np.nan
mcs7['pgfeduc7'] = np.where((mcs7['pgfeduc7'] == 7) | (mcs7['pgfeduc7'] == 8) | (mcs7['pgfeduc7'] == 9)
                                , np.nan, mcs7['pgfeduc7'])
print(mcs7['pgfeduc7'].value_counts(dropna=False))
print(mcs7['pgfeduc7'].value_counts().sum())


##### Health
##Cohort member height in cm ("height`X'")
# Sweep 1
#NA
# Sweep 2.
mcs2['height2'] = np.where((mcs2['bchcmc00'] >= 0) & (mcs2['bchmmc00'] >= 0), 
                            mcs2['bchcmc00'] + (mcs2['bchmmc00'] / 10), 
                            mcs2['bchcmc00'])
mcs2.loc[mcs2['height2'] < 0, 'height2'] = np.nan
print(mcs2['height2'].value_counts(dropna=False))
print(mcs2['height2'].value_counts().sum())
# Sweep 3
mcs3['height3'] = mcs3['cchtcm00']
mcs3.loc[mcs3['height3'] < 0, 'height3'] = np.nan
print(mcs3['height3'].value_counts(dropna=False))
print(mcs3['height3'].value_counts().sum())
# Sweep 4
mcs4['height4'] = mcs4['dchtdv00']
mcs4.loc[mcs4['height4'] < 0, 'height4'] = np.nan
print(mcs4['height4'].value_counts(dropna=False))
print(mcs4['height4'].value_counts().sum())
# Sweep 5
mcs5['height5'] = mcs5['echtcmb0']
mcs5.loc[mcs5['height5'] < 0, 'height5'] = np.nan
print(mcs5['height5'].value_counts(dropna=False))
print(mcs5['height5'].value_counts().sum())
# Sweep 6
mcs6['height6'] = np.where((mcs6['fchtcm00'] >= 0) & (mcs6['fchtcm1d'] >= 0), 
                            mcs6['fchtcm00'] + (mcs6['fchtcm1d'] / 10), 
                            mcs6['fchtcm00'])
mcs6.loc[mcs6['height6'] < 0, 'height6'] = np.nan
print(mcs6['height6'].value_counts(dropna=False))
print(mcs6['height6'].value_counts().sum())
# Sweep 7
mcs7['height7'] = mcs7['gchtcm00']
mcs7.loc[mcs7['height7'] < 0, 'height7'] = np.nan
print(mcs7['height7'].value_counts(dropna=False))
print(mcs7['height7'].value_counts().sum())

##Cohort member weight in kg ("weight`X'")
# Sweep 1
mcs1['weight1'] = mcs1['aclawe00']
mcs1.loc[mcs1['weight1'] < 0, 'weight1'] = np.nan
print(mcs1['weight1'].value_counts(dropna=False))
print(mcs1['weight1'].value_counts().sum())
# Sweep 2.
mcs2['weight2'] = np.where((mcs2['bcwtkc00'] >= 0) & (mcs2['bcwtgc00'] >= 0), 
                            mcs2['bcwtkc00'] + (mcs2['bchmmc00'] / 10), 
                            mcs2['bcwtkc00'])
mcs2.loc[mcs2['weight2'] < 0, 'weight2'] = np.nan
print(mcs2['weight2'].value_counts(dropna=False))
print(mcs2['weight2'].value_counts().sum())
# Sweep 3
mcs3['weight3'] = mcs3['ccwtcm00']
mcs3.loc[mcs3['weight3'] < 0, 'weight3'] = np.nan
print(mcs3['weight3'].value_counts(dropna=False))
print(mcs3['weight3'].value_counts().sum())
# Sweep 4
mcs4['weight4'] = mcs4['dcwtdv00']
mcs4.loc[mcs4['weight4'] < 0, 'weight4'] = np.nan
print(mcs4['weight4'].value_counts(dropna=False))
print(mcs4['weight4'].value_counts().sum())
# Sweep 5
mcs5['weight5'] = mcs5['ecwtcmb0']
mcs5.loc[mcs5['weight5'] < 0, 'weight5'] = np.nan
print(mcs5['weight5'].value_counts(dropna=False))
print(mcs5['weight5'].value_counts().sum())
# Sweep 6
mcs6['weight6'] = np.where((mcs6['fcwtcm00'] >= 0) & (mcs6['fcwtcm1d'] >= 0), 
                            mcs6['fcwtcm00'] + (mcs6['fcwtcm1d'] / 10), 
                            mcs6['fcwtcm00'])
mcs6.loc[mcs6['weight6'] < 0, 'weight6'] = np.nan
print(mcs6['weight6'].value_counts(dropna=False))
print(mcs6['weight6'].value_counts().sum())
# Sweep 7
mcs7['weight7'] = mcs7['gcwtcm00']
mcs7.loc[mcs7['weight7'] < 0, 'weight7'] = np.nan
print(mcs7['weight7'].value_counts(dropna=False))
print(mcs7['weight7'].value_counts().sum())

##Cohort member BMI
# Sweep 2.
mcs2['bmi2'] = mcs2['bcbmin00']
mcs2.loc[mcs2['bmi2'] < 0, 'bmi2'] = np.nan
print(mcs2['bmi2'].value_counts(dropna=False))
print(mcs2['bmi2'].value_counts().sum())
# Sweep 3
mcs3['bmi3'] = mcs3['bmin3']
mcs3.loc[mcs3['bmi3'] < 0, 'bmi3'] = np.nan
print(mcs3['bmi3'].value_counts(dropna=False))
print(mcs3['bmi3'].value_counts().sum())
# Sweep 4
mcs4['bmi4'] = mcs4['dcbmin4']
mcs4.loc[mcs4['bmi4'] < 0, 'bmi4'] = np.nan
print(mcs4['bmi4'].value_counts(dropna=False))
print(mcs4['bmi4'].value_counts().sum())
# Sweep 5
mcs5['bmi5'] = mcs5['ebmin5']
mcs5.loc[mcs5['bmi5'] < 0, 'bmi5'] = np.nan
print(mcs5['bmi5'].value_counts(dropna=False))
print(mcs5['bmi5'].value_counts().sum())
# Sweep 6
mcs6['bmi6'] = mcs6['fcbmin6']
mcs6.loc[mcs6['bmi6'] < 0, 'bmi6'] = np.nan
print(mcs6['bmi6'].value_counts(dropna=False))
print(mcs6['bmi6'].value_counts().sum())
# Sweep 7
mcs7['bmi7'] = mcs7['gcbmin7']
mcs7.loc[mcs7['bmi7'] < 0, 'bmi7'] = np.nan
print(mcs7['bmi7'].value_counts(dropna=False))
print(mcs7['bmi7'].value_counts().sum())

##Cohort member obesity
# Sweep 2.
mcs2['obflag2'] = mcs2['bobflag2']
mcs2.loc[mcs2['obflag2'] < 0, 'obflag2'] = np.nan
mcs2['obesity2'] = np.where((mcs2['obflag2'] == 2)
                                , 1, 0)
mcs2['obesity2'] = np.where((mcs2['obflag2'].isnull() == 1)
                                , np.nan, mcs2['obesity2'])
print(mcs2['obesity2'].value_counts(dropna=False))
print(mcs2['obesity2'].value_counts().sum())
# Sweep 3
mcs3['obflag3'] = mcs3['ccobflag3']
mcs3.loc[mcs3['obflag3'] < 0, 'obflag3'] = np.nan
mcs3['obesity3'] = np.where((mcs3['obflag3'] == 2)
                                , 1, 0)
mcs3['obesity3'] = np.where((mcs3['obflag3'].isnull() == 1)
                                , np.nan, mcs3['obesity3'])
print(mcs3['obesity3'].value_counts(dropna=False))
print(mcs3['obesity3'].value_counts().sum())
# Sweep 4
mcs4['obflag4'] = mcs4['dcobflag']
mcs4.loc[mcs4['obflag4'] < 0, 'obflag4'] = np.nan
mcs4['obesity4'] = np.where((mcs4['obflag4'] == 2)
                                , 1, 0)
mcs4['obesity4'] = np.where((mcs4['obflag4'].isnull() == 1)
                                , np.nan, mcs4['obesity4'])
print(mcs4['obesity4'].value_counts(dropna=False))
print(mcs4['obesity4'].value_counts().sum())
# Sweep 5
mcs5['obflag5'] = mcs5['eobflag5']
mcs5.loc[mcs5['obflag5'] < 0, 'obflag5'] = np.nan
mcs5['obesity5'] = np.where((mcs5['obflag5'] == 2)
                                , 1, 0)
mcs5['obesity5'] = np.where((mcs5['obflag5'].isnull() == 1)
                                , np.nan, mcs5['obesity5'])
print(mcs5['obesity5'].value_counts(dropna=False))
print(mcs5['obesity5'].value_counts().sum())
# Sweep 6
mcs6['obflag6'] = mcs6['fcuk90o6']
mcs6.loc[mcs6['obflag6'] < 0, 'obflag6'] = np.nan
mcs6['obesity6'] = np.where((mcs6['obflag6'] == 4)
                                , 1, 0)
mcs6['obesity6'] = np.where((mcs6['obflag6'].isnull() == 1)
                                , np.nan, mcs6['obesity6'])
print(mcs6['obesity6'].value_counts(dropna=False))
print(mcs6['obesity6'].value_counts().sum())
# Sweep 7
mcs7['obflag7'] = mcs7['gcuk90o7']
mcs7.loc[mcs7['obflag7'] < 0, 'obflag7'] = np.nan
mcs7['obesity7'] = np.where((mcs7['obflag7'] == 4)
                                , 1, 0)
mcs7['obesity7'] = np.where((mcs7['obflag7'].isnull() == 1)
                                , np.nan, mcs7['obesity7'])
print(mcs7['obesity7'].value_counts(dropna=False))
print(mcs7['obesity7'].value_counts().sum())

##Activity limiting conditions
# lsc - long standing illness/health condition
# alc - activity limiting condition
# Sweep 2.
mcs2['lsc2'] = mcs2['bpclsi00']
mcs2.loc[mcs2['lsc2'] == 2, 'lsc2'] = 0
mcs2.loc[mcs2['lsc2'] < 0, 'lsc2'] = np.nan
print(mcs2['lsc2'].value_counts(dropna=False))
print(mcs2['lsc2'].value_counts().sum())
mcs2['alc2'] = mcs2['lsc2']
mcs2.loc[mcs2['bpclsl00'] == 2, 'alc2'] = 0
print(mcs2['alc2'].value_counts(dropna=False))
print(mcs2['alc2'].value_counts().sum())
# Sweep 3
mcs3['lsc3'] = mcs3['cpclsi00']
mcs3.loc[mcs3['lsc3'] == 2, 'lsc3'] = 0
mcs3.loc[mcs3['lsc3'] < 0, 'lsc3'] = np.nan
print(mcs3['lsc3'].value_counts(dropna=False))
print(mcs3['lsc3'].value_counts().sum())
mcs3['alc3'] = mcs3['lsc3']
mcs3.loc[mcs3['cpclsl00'] == 2, 'alc3'] = 0
print(mcs3['alc3'].value_counts(dropna=False))
print(mcs3['alc3'].value_counts().sum())
# Sweep 4
mcs4['lsc4'] = mcs4['dpclsi00']
mcs4.loc[mcs4['lsc4'] == 2, 'lsc4'] = 0
mcs4.loc[mcs4['lsc4'] < 0, 'lsc4'] = np.nan
print(mcs4['lsc4'].value_counts(dropna=False))
print(mcs4['lsc4'].value_counts().sum())
mcs4['alc4'] = mcs4['lsc4']
mcs4.loc[mcs4['dpclsl00'] == 2, 'alc4'] = 0
print(mcs4['alc4'].value_counts(dropna=False))
print(mcs4['alc4'].value_counts().sum())
# Sweep 5
mcs5['lsc5'] = mcs5['epclsi00']
mcs5.loc[mcs5['lsc5'] == 2, 'lsc5'] = 0
mcs5.loc[mcs5['lsc5'] < 0, 'lsc5'] = np.nan
print(mcs5['lsc5'].value_counts(dropna=False))
print(mcs5['lsc5'].value_counts().sum())
mcs5['alc5'] = mcs5['lsc5']
mcs5.loc[mcs5['epclsl00'] == 2, 'alc5'] = 0
print(mcs5['alc5'].value_counts(dropna=False))
print(mcs5['alc5'].value_counts().sum())
# Sweep 6
mcs6['lsc6'] = mcs6['fpclsi00']
mcs6.loc[mcs6['lsc6'] == 2, 'lsc6'] = 0
mcs6.loc[mcs6['lsc6'] < 0, 'lsc6'] = np.nan
print(mcs6['lsc6'].value_counts(dropna=False))
print(mcs6['lsc6'].value_counts().sum())
mcs6['alc6'] = mcs6['lsc6']
mcs6.loc[mcs6['fpclsl00'] == 2, 'alc6'] = 0
print(mcs6['alc6'].value_counts(dropna=False))
print(mcs6['alc6'].value_counts().sum())
# Sweep 7
mcs7['lsc7'] = mcs7['gcclsi00']
mcs7.loc[mcs7['lsc7'] == 2, 'lsc7'] = 0
mcs7.loc[mcs7['lsc7'] > 2, 'lsc7'] = np.nan
print(mcs7['lsc7'].value_counts(dropna=False))
print(mcs7['lsc7'].value_counts().sum())
mcs7['alc7'] = mcs7['lsc7']
mcs7.loc[((mcs7['gcclsl00'] == 3) | (mcs7['gcclsl00'] == 4) | (mcs7['gcclsl00'] == 5)), 'alc7'] = 0
print(mcs7['alc7'].value_counts(dropna=False))
print(mcs7['alc7'].value_counts().sum())

##Hospitalisation
# hospac - hospitalisation due to accident
# hospilna - number of hospitalisations
# hosp - any hospitalisation
# Sweep 1
mcs1['hospac1'] = mcs1['acaccha0']
mcs1.loc[mcs1['hospac1'] == -1, 'hospac1'] = 0
mcs1.loc[mcs1['hospac1'] == 1, 'hospac1'] = 0
mcs1.loc[mcs1['hospac1'] == 3, 'hospac1'] = 1
mcs1.loc[mcs1['hospac1'] < 0, 'hospac1'] = np.nan
print(mcs1['hospac1'].value_counts(dropna=False))
print(mcs1['hospac1'].value_counts().sum())
mcs1['hospilna1'] = mcs1['acadma00']
mcs1.loc[mcs1['hospilna1'] < 0, 'hospilna1'] = np.nan
print(mcs1['hospilna1'].value_counts(dropna=False))
print(mcs1['hospilna1'].value_counts().sum())
mcs1['hosp1'] = np.where(((mcs1['hospilna1'] > 0) | (mcs1['hospac1'] == 1))
                                , 1, 0)
mcs1.loc[mcs1['hospilna1'] < 0, 'hosp1'] = np.nan
mcs1.loc[mcs1['hospac1'] < 0, 'hosp1'] = np.nan
print(mcs1['hosp1'].value_counts(dropna=False))
print(mcs1['hosp1'].value_counts().sum())
# Sweep 2
mcs2['hospac2'] = mcs2['bpacch00']
mcs2.loc[mcs2['hospac2'] == -1, 'hospac2'] = 0
mcs2.loc[mcs2['hospac2'] == 1, 'hospac2'] = 0
mcs2.loc[mcs2['hospac2'] == 3, 'hospac2'] = 1
mcs2.loc[mcs2['hospac2'] < 0, 'hospac2'] = np.nan
print(mcs2['hospac2'].value_counts(dropna=False))
print(mcs2['hospac2'].value_counts().sum())
mcs2['hospilna2'] = mcs2['bpadma00']
mcs2.loc[mcs2['hospilna2'] < 0, 'hospilna2'] = np.nan
print(mcs2['hospilna2'].value_counts(dropna=False))
print(mcs2['hospilna2'].value_counts().sum())
mcs2['hosp2'] = np.where(((mcs2['hospilna2'] > 0) | (mcs2['hospac2'] == 1))
                                , 1, 0)
mcs2.loc[mcs2['hospilna2'] < 0, 'hosp2'] = np.nan
mcs2.loc[mcs2['hospac2'] < 0, 'hosp2'] = np.nan
print(mcs2['hosp2'].value_counts(dropna=False))
print(mcs2['hosp2'].value_counts().sum())
# Sweep 3
mcs3['hospac3'] = mcs3['cpacch00']
mcs3.loc[mcs3['hospac3'] == -1, 'hospac3'] = 0
mcs3.loc[mcs3['hospac3'] == 1, 'hospac3'] = 0
mcs3.loc[mcs3['hospac3'] == 3, 'hospac3'] = 1
mcs3.loc[mcs3['hospac3'] < 0, 'hospac3'] = np.nan
print(mcs3['hospac3'].value_counts(dropna=False))
print(mcs3['hospac3'].value_counts().sum())
mcs3['hospilna3'] = mcs3['cpadma00']
mcs3.loc[mcs3['hospilna3'] < 0, 'hospilna3'] = np.nan
print(mcs3['hospilna3'].value_counts(dropna=False))
print(mcs3['hospilna3'].value_counts().sum())
mcs3['hosp3'] = np.where(((mcs3['hospilna3'] > 0) | (mcs3['hospac3'] == 1))
                                , 1, 0)
mcs3.loc[mcs3['hospilna3'] < 0, 'hosp3'] = np.nan
mcs3.loc[mcs3['hospac3'] < 0, 'hosp3'] = np.nan
print(mcs3['hosp3'].value_counts(dropna=False))
print(mcs3['hosp3'].value_counts().sum())
# Sweep 4
mcs4['hospac4'] = mcs4['dpacch00']
mcs4.loc[mcs4['hospac4'] == -1, 'hospac4'] = 0
mcs4.loc[mcs4['hospac4'] == 1, 'hospac4'] = 0
mcs4.loc[mcs4['hospac4'] == 3, 'hospac4'] = 1
mcs4.loc[mcs4['hospac4'] < 0, 'hospac4'] = np.nan
print(mcs4['hospac4'].value_counts(dropna=False))
print(mcs4['hospac4'].value_counts().sum())
mcs4['hospilna4'] = mcs4['dpadma00']
mcs4.loc[mcs4['hospilna4'] < 0, 'hospilna4'] = np.nan
print(mcs4['hospilna4'].value_counts(dropna=False))
print(mcs4['hospilna4'].value_counts().sum())
mcs4['hosp4'] = np.where(((mcs4['hospilna4'] > 0) | (mcs4['hospac4'] == 1))
                                , 1, 0)
mcs4.loc[mcs4['hospilna4'] < 0, 'hosp4'] = np.nan
mcs4.loc[mcs4['hospac4'] < 0, 'hosp4'] = np.nan
print(mcs4['hosp4'].value_counts(dropna=False))
print(mcs4['hosp4'].value_counts().sum())
# Sweep 5
mcs5['hospac5'] = mcs5['epacch00']
mcs5.loc[mcs5['hospac5'] == -1, 'hospac5'] = 0
mcs5.loc[mcs5['hospac5'] == 1, 'hospac5'] = 0
mcs5.loc[mcs5['hospac5'] == 3, 'hospac5'] = 1
mcs5.loc[mcs5['hospac5'] < 0, 'hospac5'] = np.nan
print(mcs5['hospac5'].value_counts(dropna=False))
print(mcs5['hospac5'].value_counts().sum())
mcs5['hospilna5'] = mcs5['epadma00']
mcs5.loc[mcs5['hospilna5'] < 0, 'hospilna5'] = np.nan
print(mcs5['hospilna5'].value_counts(dropna=False))
print(mcs5['hospilna5'].value_counts().sum())
mcs5['hosp5'] = np.where(((mcs5['hospilna5'] > 0) | (mcs5['hospac5'] == 1))
                                , 1, 0)
mcs5.loc[mcs5['hospilna5'] < 0, 'hosp5'] = np.nan
mcs5.loc[mcs5['hospac5'] < 0, 'hosp5'] = np.nan
print(mcs5['hosp5'].value_counts(dropna=False))
print(mcs5['hosp5'].value_counts().sum())
# Sweep 6
mcs6['hospac6'] = mcs6['fpacch00']
mcs6.loc[mcs6['hospac6'] == -1, 'hospac6'] = 0
mcs6.loc[mcs6['hospac6'] == 1, 'hospac6'] = 0
mcs6.loc[mcs6['hospac6'] == 3, 'hospac6'] = 1
mcs6.loc[mcs6['hospac6'] < 0, 'hospac6'] = np.nan
print(mcs6['hospac6'].value_counts(dropna=False))
print(mcs6['hospac6'].value_counts().sum())
mcs6['hospilna6'] = mcs6['fpadma00']
mcs6.loc[mcs6['hospilna6'] < 0, 'hospilna6'] = np.nan
print(mcs6['hospilna6'].value_counts(dropna=False))
print(mcs6['hospilna6'].value_counts().sum())
mcs6['hosp6'] = np.where(((mcs6['hospilna6'] > 0) | (mcs6['hospac6'] == 1))
                                , 1, 0)
mcs6.loc[mcs6['hospilna6'] < 0, 'hosp6'] = np.nan
mcs6.loc[mcs6['hospac6'] < 0, 'hosp6'] = np.nan
print(mcs6['hosp6'].value_counts(dropna=False))
print(mcs6['hosp6'].value_counts().sum())
# Sweep 7
mcs7['hospilna7'] = mcs7['gcadma00']
mcs7.loc[mcs7['hospilna7'] < 0, 'hospilna7'] = np.nan
print(mcs7['hospilna7'].value_counts(dropna=False))
print(mcs7['hospilna7'].value_counts().sum())
mcs7['hosp7'] = np.where((mcs7['hospilna7'] > 0)
                                , 1, 0)
mcs7.loc[mcs7['hospilna7'] < 0, 'hosp7'] = np.nan
print(mcs7['hosp7'].value_counts(dropna=False))
print(mcs7['hosp7'].value_counts().sum())


###Smoking
# Sweep 6
mcs6['smkfreq6'] = mcs6['fcsmok00']
mcs6.loc[mcs6['smkfreq6'] == -1, 'smkfreq6'] = 0
mcs6.loc[mcs6['smkfreq6'] < 0, 'smkfreq6'] = np.nan
print(mcs6['smkfreq6'].value_counts(dropna=False))
print(mcs6['smkfreq6'].value_counts().sum())
mcs6['smkevr6'] = np.where((mcs6['smkfreq6'] > 1)
                                , 1, 0)
mcs6.loc[mcs6['smkfreq6'] < 0, 'smkevr6'] = np.nan
print(mcs6['smkevr6'].value_counts(dropna=False))
print(mcs6['smkevr6'].value_counts().sum())
mcs6['smkreg6'] = np.where((mcs6['smkfreq6'] == 6)
                                , 1, 0)
mcs6.loc[mcs6['smkfreq6'] < 0, 'smkreg6'] = np.nan
print(mcs6['smkreg6'].value_counts(dropna=False))
print(mcs6['smkreg6'].value_counts().sum())
# Sweep 7
mcs7['smkfreq7'] = mcs7['gcsmok00']
mcs7.loc[mcs7['smkfreq7'] == -1, 'smkfreq7'] = 0
mcs7.loc[mcs7['smkfreq7'] < 0, 'smkfreq7'] = np.nan
print(mcs7['smkfreq7'].value_counts(dropna=False))
print(mcs7['smkfreq7'].value_counts().sum())
mcs7['smkevr7'] = np.where((mcs7['smkfreq7'] > 1)
                                , 1, 0)
mcs7.loc[mcs7['smkfreq7'] < 0, 'smkevr7'] = np.nan
print(mcs7['smkevr7'].value_counts(dropna=False))
print(mcs7['smkevr7'].value_counts().sum())
mcs7['smkreg7'] = np.where((mcs7['smkfreq7'] == 6)
                                , 1, 0)
mcs7.loc[mcs7['smkfreq7'] < 0, 'smkreg7'] = np.nan
print(mcs7['smkreg7'].value_counts(dropna=False))
print(mcs7['smkreg7'].value_counts().sum())


###Drinking
# Sweep 6
mcs6['drnkfreq6'] = mcs6['fcalnf00']
mcs6.loc[mcs6['drnkfreq6'] == -1, 'drnkfreq6'] = 0
mcs6.loc[mcs6['drnkfreq6'] < 0, 'drnkfreq6'] = np.nan
print(mcs6['drnkfreq6'].value_counts(dropna=False))
print(mcs6['drnkfreq6'].value_counts().sum())
mcs6['drnkevr6'] = mcs6['fcalcd00']
np.where((mcs6['drnkfreq6'] > 1)
                                , 1, 0)
mcs6.loc[mcs6['drnkfreq6'] < 0, 'drnkevr6'] = np.nan
print(mcs6['drnkevr6'].value_counts(dropna=False))
print(mcs6['drnkevr6'].value_counts().sum())
mcs6['drnkreg6'] = np.where((mcs6['drnkfreq6'] == 6)
                                , 1, 0)
mcs6.loc[mcs6['drnkfreq6'] < 0, 'drnkreg6'] = np.nan
print(mcs6['drnkreg6'].value_counts(dropna=False))
print(mcs6['drnkreg6'].value_counts().sum())
# Sweep 7
mcs7['drnkfreq7'] = mcs7['gcalnf00']
mcs7.loc[mcs7['drnkfreq7'] == -1, 'drnkfreq7'] = 0
mcs7.loc[mcs7['drnkfreq7'] < 0, 'drnkfreq7'] = np.nan
print(mcs7['drnkfreq7'].value_counts(dropna=False))
print(mcs7['drnkfreq7'].value_counts().sum())
mcs7['drnkevr7'] = np.where((mcs7['drnkfreq7'] > 1)
                                , 1, 0)
mcs7.loc[mcs7['drnkfreq7'] < 0, 'drnkevr7'] = np.nan
print(mcs7['drnkevr7'].value_counts(dropna=False))
print(mcs7['drnkevr7'].value_counts().sum())
mcs7['drnkreg7'] = np.where((mcs7['drnkfreq7'] == 6)
                                , 1, 0)
mcs7.loc[mcs7['drnkfreq7'] < 0, 'drnkreg7'] = np.nan
print(mcs7['drnkreg7'].value_counts(dropna=False))
print(mcs7['drnkreg7'].value_counts().sum())


#Health poor/Fair
# Sweep 7
mcs7['hlthstate7'] = mcs7['gccghe00']
mcs7.loc[mcs7['hlthstate7'] > 5, 'hlthstate7'] = np.nan
print(mcs7['hlthstate7'].value_counts(dropna=False))
print(mcs7['hlthstate7'].value_counts().sum())
mcs7['prfrhlth7'] = np.where((mcs7['hlthstate7'] > 3)
                                , 1, 0)
mcs7.loc[mcs7['hlthstate7'].isnull(), 'prfrhlth7'] = np.nan
print(mcs7['prfrhlth7'].value_counts(dropna=False))
print(mcs7['prfrhlth7'].value_counts().sum())

##### School variables
###Exclusion
# Sweep 5
mcs5['texcl5'] = mcs5['eptsus00']
mcs5.loc[mcs5['texcl5'] == 2, 'texcl5'] = 0
mcs5.loc[mcs5['texcl5'] < 0, 'texcl5'] = np.nan
print(mcs5['texcl5'].value_counts(dropna=False))
print(mcs5['texcl5'].value_counts().sum())
mcs5['pexcl5'] = mcs5['eptexc00']
mcs5.loc[mcs5['pexcl5'] == 2, 'pexcl5'] = 0
mcs5.loc[mcs5['pexcl5'] < 0, 'pexcl5'] = np.nan
print(mcs5['pexcl5'].value_counts(dropna=False))
print(mcs5['pexcl5'].value_counts().sum())
mcs5['excl5'] = np.where(((mcs5['texcl5'] == 1) | (mcs5['pexcl5'] == 1))
                                , 1, 0)
mcs5['excl5'] = np.where(((mcs5['texcl5'].isnull() == 1) | (mcs5['pexcl5'].isnull() == 1))
                                , np.nan, mcs5['excl5'])
print(mcs5['excl5'].value_counts(dropna=False))
print(mcs5['excl5'].value_counts().sum())
# Sweep 6
mcs6['texcl6'] = mcs6['fptsus00']
mcs6.loc[mcs6['texcl6'] == 2, 'texcl6'] = 0
mcs6.loc[mcs6['texcl6'] < 0, 'texcl6'] = np.nan
print(mcs6['texcl6'].value_counts(dropna=False))
print(mcs6['texcl6'].value_counts().sum())
mcs6['pexcl6'] = mcs6['fptexc00']
mcs6.loc[mcs6['pexcl6'] == 2, 'pexcl6'] = 0
mcs6.loc[mcs6['pexcl6'] < 0, 'pexcl6'] = np.nan
print(mcs6['pexcl6'].value_counts(dropna=False))
print(mcs6['pexcl6'].value_counts().sum())
mcs6['excl6'] = np.where(((mcs6['texcl6'] == 1) | (mcs6['pexcl6'] == 1))
                                , 1, 0)
mcs6['excl6'] = np.where(((mcs6['texcl6'].isnull() == 1) | (mcs6['pexcl6'].isnull() == 1))
                                , np.nan, mcs6['excl6'])
mcs6.loc[mcs6['texcl6'] < 0, 'excl6'] = np.nan
mcs6.loc[mcs6['pexcl6'] < 0, 'excl6'] = np.nan
print(mcs6['excl6'].value_counts(dropna=False))
print(mcs6['excl6'].value_counts().sum())


###Truancy
# Sweep 5
mcs5['truancy5'] = mcs5['epsabs00']
mcs5.loc[mcs5['truancy5'] == 2, 'truancy5'] = 0
mcs5.loc[mcs5['truancy5'] < 0, 'truancy5'] = np.nan
print(mcs5['truancy5'].value_counts(dropna=False))
print(mcs5['truancy5'].value_counts().sum())
mcs5['nwoffscl5'] = mcs5['epwabs00']
mcs5.loc[mcs5['nwoffscl5'] == -1, 'nwoffscl5'] = 0
mcs5.loc[mcs5['nwoffscl5'] < 0, 'nwoffscl5'] = np.nan
print(mcs5['nwoffscl5'].value_counts(dropna=False))
print(mcs5['nwoffscl5'].value_counts().sum())
mcs5['regtruancy5'] = np.where(((mcs5['truancy5'] == 1) & (mcs5['nwoffscl5'] > 4))
                                , 1, 0)
mcs5.loc[mcs5['truancy5'].isnull(), 'regtruancy5'] = np.nan
mcs5.loc[mcs5['nwoffscl5'].isnull(), 'regtruancy5'] = np.nan
print(mcs5['regtruancy5'].value_counts(dropna=False))
print(mcs5['regtruancy5'].value_counts().sum())
# Sweep 6
mcs6['truancy6'] = mcs6['fpsabs00']
mcs6.loc[mcs6['truancy6'] == 2, 'truancy6'] = 0
mcs6.loc[mcs6['truancy6'] < 0, 'truancy6'] = np.nan
print(mcs6['truancy6'].value_counts(dropna=False))
print(mcs6['truancy6'].value_counts().sum())
mcs6['nwoffscl6'] = mcs6['fpwabs00']
mcs6.loc[mcs6['nwoffscl6'] == -1, 'nwoffscl6'] = 0
mcs6.loc[mcs6['nwoffscl6'] < 0, 'nwoffscl6'] = np.nan
print(mcs6['nwoffscl6'].value_counts(dropna=False))
print(mcs6['nwoffscl6'].value_counts().sum())
mcs6['regtruancy6'] = np.where(((mcs6['truancy6'] == 1) & (mcs6['nwoffscl6'] > 4))
                                , 1, 0)
mcs6.loc[mcs6['truancy6'].isnull(), 'regtruancy6'] = np.nan
mcs6.loc[mcs6['nwoffscl6'].isnull(), 'regtruancy6'] = np.nan
print(mcs6['regtruancy6'].value_counts(dropna=False))
print(mcs6['regtruancy6'].value_counts().sum())

###Special education needs statement
# Sweep 4
mcs4['sen4'] = mcs4['dpsens00']
mcs4.loc[mcs4['sen4'] == -1, 'sen4'] = 0
mcs4.loc[mcs4['sen4'] == 2, 'sen4'] = 0
mcs4.loc[mcs4['sen4'] == 97, 'sen4'] = 0
mcs4.loc[mcs4['sen4'] < 0, 'sen4'] = np.nan
print(mcs4['sen4'].value_counts(dropna=False))
print(mcs4['sen4'].value_counts().sum())
# Sweep 5
mcs5['sen5'] = mcs5['epsens00']
mcs5.loc[mcs5['sen5'] == -1, 'sen5'] = 0
mcs5.loc[mcs5['sen5'] == 2, 'sen5'] = 0
mcs5.loc[mcs5['sen5'] == 3, 'sen5'] = 0
mcs5.loc[mcs5['sen5'] < 0, 'sen5'] = np.nan
print(mcs5['sen5'].value_counts(dropna=False))
print(mcs5['sen5'].value_counts().sum())
# Sweep 6
mcs6['sen6'] = mcs6['fpsens00']
mcs6.loc[mcs6['sen6'] == -1, 'sen6'] = 0
mcs6.loc[mcs6['sen6'] == 2, 'sen6'] = 0
mcs6.loc[mcs6['sen6'] == 3, 'sen6'] = 0
mcs6.loc[mcs6['sen6'] < 0, 'sen6'] = np.nan
print(mcs6['sen6'].value_counts(dropna=False))
print(mcs6['sen6'].value_counts().sum())

### Bad GCSEs
#GCSESs
print([x for x in mcs7.columns if x.startswith('gc_gcgd')])
print(mcs7['gc_gcgd_1'].value_counts(dropna=False))
gcmincs = [4, 5, 6, 7, 8, 9, 11, 12, 13, 14]
gcengn = [89, 92, 95, 96]
gcmthn = [170, 171, 172]
print(mcs7['gc_qual_gcsn_r20'].value_counts(dropna=False))
mcs7['numgdgcs'] = mcs7['gc_qual_gcsn_r20']
mcs7.loc[mcs7['numgdgcs'] < 0, 'numgdgcs'] = np.nan
print(mcs7['numgdgcs'].value_counts(dropna=False))
print(mcs7['numgdgcs'].value_counts().sum())
print([x for x in mcs7.columns if x.startswith('gc_gcsb')])
for i in range(1, 21):  # Loop from 1 to 20
    gcgrade_col = f'gc_gcgd_{i}'  # grade columns
    gcsubj_col = f'gc_gcsb_name_r40_{i}'  # subject name columns
    gcgdgrd_col = f'gcgdgd_{i}'  # good grade columns
    gceng_col = f'gceng{i}'  # eng column name
    gcmath_col = f'gcmath{i}'  # math column name
    
    mcs7.loc[mcs7[gcgrade_col] < 0, gcgrade_col] = np.nan
    # Apply conditions to create good grades column
    mcs7[gcgdgrd_col] = np.where(mcs7[gcgrade_col].isna(), np.nan,
                                 ((mcs7[gcgrade_col].isin(gcmincs))).astype(int))
    # Apply conditions to create eng column
    mcs7[gceng_col] = np.where(mcs7[gcgrade_col].isna(), np.nan,
                                 ((mcs7[gcgrade_col].isin(gcmincs)) & (mcs7[gcsubj_col].isin(gcengn))).astype(int))
    # Apply conditions to create math column
    mcs7[gcmath_col] = np.where(mcs7[gcgrade_col].isna(), np.nan,
                                ((mcs7[gcgrade_col].isin(gcmincs)) & (mcs7[gcsubj_col].isin(gcmthn))).astype(int))
print([x for x in mcs7.columns if x.startswith('gcgdgd')])
print([x for x in mcs7.columns if x.startswith('gceng')])
print([x for x in mcs7.columns if x.startswith('gcmath')])
mcs7['gcgdgd'] = mcs7.filter(like='gcgdgd_').sum(axis=1)
mcs7.loc[mcs7['gc_gcgd_1'].isna(), 'gcgdgd'] = np.nan
print(mcs7['gcgdgd'].value_counts(dropna=False))
mcs7['gceng'] = mcs7.filter(like='gceng').sum(axis=1)
mcs7.loc[mcs7['gc_gcgd_1'].isna(), 'gceng'] = np.nan
print(mcs7['gceng'].value_counts(dropna=False))
mcs7['gcmath'] = mcs7.filter(like='gcmath').sum(axis=1)
mcs7.loc[mcs7['gc_gcgd_1'].isna(), 'gcmath'] = np.nan
print(mcs7['gcmath'].value_counts(dropna=False))
mcs7['gdgc'] = np.where(
    (mcs7['gcgdgd'] >= 5) & (mcs7['gcgdgd'].notna()), 
    1, 
    np.where(
        ((mcs7['gcgdgd'] < 5)) & (mcs7['gcgdgd'].notna()), 
        0, 
        np.nan
    )
)
print(mcs7['gdgc'].value_counts(dropna=False))
mcs7['gdgcme'] = np.where(
    (mcs7['gcgdgd'] >= 5) & (mcs7['gceng'] >= 1) & (mcs7['gcmath'] >= 1) & (mcs7['gcgdgd'].notna()), 
    1, 
    np.where(
        ((mcs7['gcgdgd'] < 5) | (mcs7['gceng'] < 1) | (mcs7['gcmath'] < 1)) & (mcs7['gcgdgd'].notna()), 
        0, 
        np.nan
    )
)
print(mcs7['gdgcme'].value_counts(dropna=False))
del gcmincs
del gcengn
del gcmthn
del gcgrade_col
del gcsubj_col
del gcgdgrd_col
del gceng_col
del gcmath_col
del i
#iGCSESs
print([x for x in mcs7.columns if x.startswith('gc_iggd')])
print(mcs7['gc_iggd_1'].value_counts(dropna=False))
igmincs = [1, 2, 3, 4, 5, 6, 10, 11, 12, 14]
igengn = [16, 19]
igmthn = [33]
print(mcs7['gc_qual_igsn_rec'].value_counts(dropna=False))
mcs7['numgdigs'] = mcs7['gc_qual_igsn_rec']
mcs7.loc[mcs7['numgdigs'] < 0, 'numgdigs'] = np.nan
print(mcs7['numgdigs'].value_counts(dropna=False))
print(mcs7['numgdigs'].value_counts().sum())
print([x for x in mcs7.columns if x.startswith('gc_igsb')])
for i in range(1, 21):  # Loop from 1 to 20

    iggrade_col = f'gc_iggd_{i}'  # grade columns
    igsubj_col = f'gc_igsb_name_r30_{i}'  # subject name columns
    iggdgrd_col = f'iggdgd_{i}'  # good grade columns
    igeng_col = f'igeng{i}'  # eng column name
    igmath_col = f'igmath{i}'  # math column name
    
    mcs7.loc[mcs7[iggrade_col] < 0, iggrade_col] = np.nan
    # Apply conditions to create good grades column
    mcs7[iggdgrd_col] = np.where(mcs7[iggrade_col].isna(), np.nan,
                                 ((mcs7[iggrade_col].isin(igmincs))).astype(int))
    # Apply conditions to create eng column
    mcs7[igeng_col] = np.where(mcs7[iggrade_col].isna(), np.nan,
                                 ((mcs7[iggrade_col].isin(igmincs)) & (mcs7[igsubj_col].isin(igengn))).astype(int))
    # Apply conditions to create math column
    mcs7[igmath_col] = np.where(mcs7[iggrade_col].isna(), np.nan,
                                ((mcs7[iggrade_col].isin(igmincs)) & (mcs7[igsubj_col].isin(igmthn))).astype(int))
print([x for x in mcs7.columns if x.startswith('iggdgd')])
print([x for x in mcs7.columns if x.startswith('igeng')])
print([x for x in mcs7.columns if x.startswith('igmath')])
mcs7['iggdgd'] = mcs7.filter(like='iggdgd_').sum(axis=1)
mcs7.loc[mcs7['gc_iggd_1'].isna(), 'iggdgd'] = np.nan
print(mcs7['iggdgd'].value_counts(dropna=False))
mcs7['igeng'] = mcs7.filter(like='igeng').sum(axis=1)
mcs7.loc[mcs7['gc_iggd_1'].isna(), 'igeng'] = np.nan
print(mcs7['igeng'].value_counts(dropna=False))
mcs7['igmath'] = mcs7.filter(like='igmath').sum(axis=1)
mcs7.loc[mcs7['gc_iggd_1'].isna(), 'igmath'] = np.nan
print(mcs7['igmath'].value_counts(dropna=False))
mcs7['gdig'] = np.where(
    (mcs7['iggdgd'] >= 5) & (mcs7['iggdgd'].notna()), 
    1, 
    np.where(
        ((mcs7['iggdgd'] < 5)) & (mcs7['iggdgd'].notna()), 
        0, 
        np.nan
    )
)
print(mcs7['gdig'].value_counts(dropna=False))
mcs7['gdigme'] = np.where(
    (mcs7['iggdgd'] >= 5) & (mcs7['igeng'] >= 1) & (mcs7['igmath'] >= 1) & (mcs7['iggdgd'].notna()), 
    1, 
    np.where(
        ((mcs7['iggdgd'] < 5) | (mcs7['igeng'] < 1) | (mcs7['igmath'] < 1)) & (mcs7['iggdgd'].notna()), 
        0, 
        np.nan
    )
)
print(mcs7['gdigme'].value_counts(dropna=False))
del igmincs
del igengn
del igmthn
del iggrade_col
del igsubj_col
del iggdgrd_col
del igeng_col
del igmath_col
del i
#National 5 (Scotland)
print([x for x in mcs7.columns if x.startswith('gc_fvgd')])
print(mcs7['gc_fvgd_1'].value_counts(dropna=False))
n5mincs = [1, 2, 3, 4]
n5engn = [10]
n5mthn = [35]
print(mcs7['gc_qual_nfir_r20'].value_counts(dropna=False))
mcs7['numgdn5s'] = mcs7['gc_qual_nfir_r20']
mcs7.loc[mcs7['numgdn5s'] < 0, 'numgdigs'] = np.nan
print(mcs7['numgdn5s'].value_counts(dropna=False))
print(mcs7['numgdn5s'].value_counts().sum())
print([x for x in mcs7.columns if x.startswith('gc_fvsb')])
for i in range(1, 21):  # Loop from 1 to 20

    n5grade_col = f'gc_fvgd_{i}'  # grade columns
    n5subj_col = f'gc_fvsb_name_r30_{i}'  # subject name columns
    n5gdgrd_col = f'n5gdgd_{i}'  # good grade columns
    n5eng_col = f'n5eng{i}'  # eng column name
    n5math_col = f'n5math{i}'  # math column name
    
    mcs7.loc[mcs7[n5grade_col] < 0, n5grade_col] = np.nan
    # Apply conditions to create good grades column
    mcs7[n5gdgrd_col] = np.where(mcs7[n5grade_col].isna(), np.nan,
                                 ((mcs7[n5grade_col].isin(n5mincs))).astype(int))
    # Apply conditions to create eng column
    mcs7[n5eng_col] = np.where(mcs7[n5grade_col].isna(), np.nan,
                                 ((mcs7[n5grade_col].isin(n5mincs)) & (mcs7[n5subj_col].isin(n5engn))).astype(int))
    # Apply conditions to create math column
    mcs7[n5math_col] = np.where(mcs7[n5grade_col].isna(), np.nan,
                                ((mcs7[n5grade_col].isin(n5mincs)) & (mcs7[n5subj_col].isin(n5mthn))).astype(int))
print([x for x in mcs7.columns if x.startswith('n5eng')])
print([x for x in mcs7.columns if x.startswith('n5math')])
mcs7['n5gdgd'] = mcs7.filter(like='n5gdgd_').sum(axis=1)
mcs7.loc[mcs7['gc_fvgd_1'].isna(), 'n5gdgd'] = np.nan
print(mcs7['n5gdgd'].value_counts(dropna=False))
mcs7['n5eng'] = mcs7.filter(like='n5eng').sum(axis=1)
mcs7.loc[mcs7['gc_fvgd_1'].isna(), 'n5eng'] = np.nan
print(mcs7['n5eng'].value_counts(dropna=False))
mcs7['n5math'] = mcs7.filter(like='n5math').sum(axis=1)
mcs7.loc[mcs7['gc_fvgd_1'].isna(), 'n5math'] = np.nan
print(mcs7['n5math'].value_counts(dropna=False))
mcs7['gdn5'] = np.where(
    (mcs7['n5gdgd'] >= 5) & (mcs7['n5gdgd'].notna()), 
    1, 
    np.where(
        ((mcs7['n5gdgd'] < 5)) & (mcs7['n5gdgd'].notna()), 
        0, 
        np.nan
    )
)
print(mcs7['gdn5'].value_counts(dropna=False))
mcs7['gdn5me'] = np.where(
    (mcs7['n5gdgd'] >= 5) & (mcs7['n5eng'] >= 1) & (mcs7['n5math'] >= 1) & (mcs7['n5gdgd'].notna()), 
    1, 
    np.where(
        ((mcs7['n5gdgd'] < 5) | (mcs7['n5eng'] < 1) | (mcs7['n5math'] < 1)) & (mcs7['n5gdgd'].notna()), 
        0, 
        np.nan
    )
)
print(mcs7['gdn5me'].value_counts(dropna=False))
del n5mincs
del n5engn
del n5mthn
del n5grade_col
del n5subj_col
del n5gdgrd_col
del n5eng_col
del n5math_col
del i
#Bad GCSE - combines indicator
mcs7['gdgcse7'] = np.where(((mcs7['gdgc'] == 1) | (mcs7['gdig'] == 1) | (mcs7['gdn5'] == 1)), 1, 0)
mcs7.loc[(mcs7['gdgc'].isna() & mcs7['gdig'].isna() & mcs7['gdn5'].isna()), 'gdgcse7'] = np.nan
print(mcs7['gdgcse7'].value_counts(dropna=False))
mcs7['gdgcseme7']= np.where(((mcs7['gdgcme'] == 1) | (mcs7['gdigme'] == 1) | (mcs7['gdn5me'] == 1)), 1, 0)
mcs7.loc[(mcs7['gdgcme'].isna() & mcs7['gdigme'].isna() & mcs7['gdn5me'].isna()), 'gdgcseme7'] = np.nan
print(mcs7['gdgcseme7'].value_counts(dropna=False))

mcs7['bdgcse7'] = np.where(mcs7['gdgcse7'] == 0, 1, 0)
mcs7.loc[(mcs7['gdgcse7'].isna()), 'bdgcse7'] = np.nan
print(mcs7['bdgcse7'].value_counts(dropna=False))
mcs7['bdgcseme7'] = np.where(mcs7['gdgcseme7'] == 0, 1, 0)
mcs7.loc[(mcs7['gdgcseme7'].isna()), 'bdgcseme7'] = np.nan
print(mcs7['bdgcseme7'].value_counts(dropna=False))


##### Scores
# Development scores
#gmotor9mths= amsitua0 + amstana0 + ammovea0 + amwalka0
#fmotor9mths=amptoya0 + amgraba0 + ampicka0 + amhanda0
#comms9mths= amsmila0+amgivea0+amwavea0+amarmsa0+amnodsa0-4
#gmotor9mths fmotor9mths tmotor9mths comms9mths develop9mths
#gmotor9mths_adj fmotor9mths_adj tmotor9mths_adj comms9mths_adj develop9mths_adj
#tmotor9mths_dec develop9mths_dec
mcs1['gmotor1'] = np.sum(mcs1[['acsitu00', 'acstan00', 'acmove00', 'acwalk00']], axis = 1)
mcs1['fmotor1'] = np.sum(mcs1[['acptoy00', 'acgrab00', 'acpick00', 'achand00']], axis = 1)
mcs1['comms1'] = np.sum(mcs1[['acsmil00', 'acgive00', 'acwave00', 'acarms00', 'acnods00']], axis = 1)

#Changing all scores so higher is better and minimum is 1
#Gross motor skills - between 1 and 9
print(mcs1['gmotor1'].value_counts(dropna=False))
mcs1.loc[mcs1['gmotor1'] < 4, 'gmotor1'] = np.nan
print(mcs1['gmotor1'].value_counts(dropna=False))
#mcs1['gmotoradj1'] =ageadj(mcs1, 'gmotor1', 'cmagem1')
mcs1['gmotor1'] = 13 - mcs1['gmotor1']
print(mcs1['gmotor1'].value_counts(dropna=False))
mcs1['zgmotor1'] = stdnrm(mcs1, 'gmotor1')
#Fine motor skills - between 1 and 9
print(mcs1['fmotor1'].value_counts(dropna=False))
mcs1.loc[mcs1['fmotor1'] < 4, 'fmotor1'] = np.nan
print(mcs1['fmotor1'].value_counts(dropna=False))
mcs1['fmotor1'] = 13 - mcs1['fmotor1']
print(mcs1['fmotor1'].value_counts(dropna=False))
mcs1['zfmotor1'] = stdnrm(mcs1, 'fmotor1')
#Communication Devolopment - between 1 and 11
print(mcs1['comms1'].value_counts(dropna=False))
mcs1.loc[mcs1['comms1'] < 5, 'comms1'] = np.nan
print(mcs1['comms1'].value_counts(dropna=False))
mcs1['comms1'] = mcs1['comms1'] - 4
print(mcs1['comms1'].value_counts(dropna=False))
mcs1['zcomms1'] = stdnrm(mcs1, 'comms1')
#Combined development measure - between 1 and 27
mcs1['develop1'] = np.sum(mcs1[['gmotor1', 'fmotor1', 'comms1']], axis = 1)
print(mcs1['develop1'].value_counts(dropna=False))
mcs1.loc[mcs1['develop1'] < 3, 'develop1'] = np.nan
print(mcs1['develop1'].value_counts(dropna=False))
mcs1['develop1'] = mcs1['develop1'] - 2
print(mcs1['develop1'].value_counts(dropna=False))
mcs1['zdevelop1'] = np.sum(mcs1[['zgmotor1', 'zfmotor1', 'zcomms1']], axis = 1)
mcs1['zdevelop1'] = stdnrm(mcs1, 'zdevelop1')
###Cognitive ability
## Sweep 2
#British Ability Scales II (Naming vocabulary)
mcs2['basnv2'] = mcs2['bdbast00']
mcs2.loc[mcs2['basnv2'] < 0, 'basnv2'] = np.nan
mcs2['zbasnv2'] = stdnrm(mcs2, 'basnv2')
#Bracken School Readiness Assessment-Revised (BSRA-R)
mcs2['bsrar2'] = mcs2['bdsrcs00']
mcs2.loc[mcs2['bsrar2'] < 0, 'bsrar2'] = np.nan
mcs2['zbsrar2'] = stdnrm(mcs2, 'bsrar2')
#Combined measure
mcs2['zcm2'] = mcs2['zbasnv2'] + mcs2['zbsrar2']
mcs2['zcog2'] = stdnrm(mcs2, 'zcm2')
#Bracken composite School readiness measure (Normative)
mcs2['sclrdn2'] = mcs2['bdsrcn00']
mcs2.loc[mcs2['sclrdn2'] < 0, 'sclrdn2'] = np.nan
print(mcs2['sclrdn2'].value_counts(dropna=False))
#Delayed development based on normative bracken composite measure (delayed or extremely delayed)
mcs2['deldevscr2'] = np.where(((mcs2['sclrdn2'] == 1) | (mcs2['sclrdn2'] == 2)), 1, 0)
mcs2.loc[mcs2['sclrdn2'].isnull(), 'deldevscr2'] = np.nan
print(mcs2['deldevscr2'].value_counts(dropna=False))
## Sweep 3
#Age and ability adjusted scores are between 20 and 80, there is a large number of oberveations at 20 for each of the scores so I am setting them to missing
# Find out why scores are oddly clustered.
# British Ability Scales II Naming Vocabulary
mcs3['basnv3'] = mcs3['ccnvtscore']
mcs3.loc[mcs3['basnv3'] <= 20, 'basnv3'] = np.nan
mcs3['zbasnv3'] = stdnrm(mcs3, 'basnv3')
print(mcs3['basnv3'].value_counts(dropna=False))
# British Ability Scales II Pattern Construction
mcs3['baspc3'] = mcs3['ccpctscore']
mcs3.loc[mcs3['baspc3'] <= 20, 'baspc3'] = np.nan
mcs3['zbaspc3'] = stdnrm(mcs3, 'baspc3')
print(mcs3['baspc3'].value_counts(dropna=False))
# British Ability Scales II Picture Similarities
mcs3['basps3'] = mcs3['ccpstscore']
mcs3.loc[mcs3['basps3'] <= 20, 'basps3'] = np.nan
mcs3['zbasps3'] = stdnrm(mcs3, 'basps3')
print(mcs3['basps3'].value_counts(dropna=False))
#Combined measure
mcs3['zcm3'] = mcs3['zbasnv3'] + mcs3['zbaspc3'] + mcs3['zbasps3']
mcs3['zcog3'] = stdnrm(mcs3, 'zcm3')
print(mcs3['zcm3'].value_counts(dropna=False))
## Sweep 4
# British Ability Scales II Word Reading
mcs4['baswr4'] = mcs4['dcwrsd00']
mcs4.loc[mcs4['baswr4'] < 0, 'baswr4'] = np.nan
mcs4['zbaswr4'] = stdnrm(mcs4, 'baswr4')
# British Ability Scales II Pattern Construction
mcs4['baspc4'] = mcs4['dcpcts00']
mcs4.loc[mcs4['baspc4'] < 0, 'baspc4'] = np.nan
mcs4['zbaspc4'] = stdnrm(mcs4, 'baspc4')
# National Foundation for Educational Research (NFER) Progress in Maths (adapted)
mcs4['nferpm4'] = mcs4['dcmaths7sa']
mcs4.loc[mcs4['nferpm4'] < 0, 'nferpm4'] = np.nan
mcs4['znferpm4'] = stdnrm(mcs4, 'nferpm4')
#Combined measure
mcs4['zcm4'] = mcs4['zbaswr4'] + mcs4['zbaspc4'] + mcs4['znferpm4']
mcs4['zcog4'] = stdnrm(mcs4, 'zcm4')
## Sweep 5
# British Ability Scales II (Verbal Similarities)
mcs5['basvs5'] = mcs5['evstsco']
mcs5.loc[mcs5['basvs5'] < 0, 'basvs5'] = np.nan
mcs5['zcog5'] = stdnrm(mcs5, 'basvs5')
## Sweep 6
# Applied Psychology Unit - Vocabulary Test
print([x for x in mcs6.columns if x.startswith('fcwrdsc')])
mcs6['apuvt6'] = mcs6['fcwrdsc_x']
mcs6.loc[mcs6['apuvt6'] < 0, 'apuvt6'] = np.nan
mcs6['zcog6'] = stdnrm(mcs6, 'apuvt6')
## Sweep 7
# Number Analogies (GL Assessment)
mcs7['nglaa'] = np.where(mcs7['gcnaas0a'] == 5, 1, 0)
mcs7['nglaa'] = np.where((mcs7['gcnaas0a'] == -1) | (mcs7['gcnaas0a'].isna()), np.nan, mcs7['nglaa'])
mcs7['nglab'] = np.where(mcs7['gcnaas0b'] == 1, 1, 0)
mcs7['nglab'] = np.where((mcs7['gcnaas0b'] == -1) | (mcs7['gcnaas0b'].isna()), np.nan, mcs7['nglab'])
mcs7['nglac'] = np.where(mcs7['gcnaas0c'] == 3, 1, 0)
mcs7['nglac'] = np.where((mcs7['gcnaas0c'] == -1) | (mcs7['gcnaas0c'].isna()), np.nan, mcs7['nglac'])
mcs7['nglad'] = np.where(mcs7['gcnaas0d'] == 4, 1, 0)
mcs7['nglad'] = np.where((mcs7['gcnaas0d'] == -1) | (mcs7['gcnaas0d'].isna()), np.nan, mcs7['nglad'])
mcs7['nglae'] = np.where(mcs7['gcnaas0e'] == 1, 1, 0)
mcs7['nglae'] = np.where((mcs7['gcnaas0e'] == -1) | (mcs7['gcnaas0e'].isna()), np.nan, mcs7['nglae'])
mcs7['nglaf'] = np.where(mcs7['gcnaas0f'] == 5, 1, 0)
mcs7['nglaf'] = np.where((mcs7['gcnaas0f'] == -1) | (mcs7['gcnaas0f'].isna()), np.nan, mcs7['nglaf'])
mcs7['nglag'] = np.where(mcs7['gcnaas0g'] == 4, 1, 0)
mcs7['nglag'] = np.where((mcs7['gcnaas0g'] == -1) | (mcs7['gcnaas0g'].isna()), np.nan, mcs7['nglag'])
mcs7['nglah'] = np.where(mcs7['gcnaas0h'] == 4, 1, 0)
mcs7['nglah'] = np.where((mcs7['gcnaas0h'] == -1) | (mcs7['gcnaas0h'].isna()), np.nan, mcs7['nglah'])
mcs7['nglai'] = np.where(mcs7['gcnaas0i'] == 2, 1, 0)
mcs7['nglai'] = np.where((mcs7['gcnaas0i'] == -1) | (mcs7['gcnaas0i'].isna()), np.nan, mcs7['nglai'])
mcs7['nglaj'] = np.where(mcs7['gcnaas0j'] == 5, 1, 0)
mcs7['nglaj'] = np.where((mcs7['gcnaas0j'] == -1) | (mcs7['gcnaas0j'].isna()), np.nan, mcs7['nglaj'])
mcs7['nagla7'] = mcs7['nglaa'] + mcs7['nglab'] + mcs7['nglac'] + mcs7['nglad'] + mcs7['nglae'] + mcs7['nglaf'] + mcs7['nglag'] + mcs7['nglah'] + mcs7['nglai'] + mcs7['nglaj']
mcs7.loc[mcs7['nagla7'] < 0, 'nagla7'] = np.nan
mcs7['zcog7'] = stdnrm(mcs7, 'nagla7')
print(mcs7['nagla7'].value_counts(dropna=False))
print(mcs7['nagla7'].value_counts().sum())

#Bad GCSEs


#SDQ Conduct problems
# Sweep 2.
mcs2['sdqconduct2'] = mcs2['bconduct']
mcs2.loc[mcs2['sdqconduct2'] == 97, 'sdqconduct2'] = np.nan
mcs2.loc[mcs2['sdqconduct2'] < 0, 'sdqconduct2'] = np.nan
print(mcs2['sdqconduct2'].value_counts(dropna=False))
print(mcs2['sdqconduct2'].value_counts().sum())
# Sweep 3
mcs3['sdqconduct3'] = mcs3['cconduct']
mcs3.loc[mcs3['sdqconduct3'] == 97, 'sdqconduct3'] = np.nan
mcs3.loc[mcs3['sdqconduct3'] < 0, 'sdqconduct3'] = np.nan
print(mcs3['sdqconduct3'].value_counts(dropna=False))
print(mcs3['sdqconduct3'].value_counts().sum())
# Sweep 4
mcs4['sdqconduct4'] = mcs4['ddconduct']
mcs4.loc[mcs4['sdqconduct4'] == 97, 'sdqconduct4'] = np.nan
mcs4.loc[mcs4['sdqconduct4'] < 0, 'sdqconduct4'] = np.nan
print(mcs4['sdqconduct4'].value_counts(dropna=False))
print(mcs4['sdqconduct4'].value_counts().sum())
# Sweep 5
mcs5['sdqconduct5'] = mcs5['econduct']
mcs5.loc[mcs5['sdqconduct5'] == 97, 'sdqconduct5'] = np.nan
mcs5.loc[mcs5['sdqconduct5'] < 0, 'sdqconduct5'] = np.nan
print(mcs5['sdqconduct5'].value_counts(dropna=False))
print(mcs5['sdqconduct5'].value_counts().sum())
# Sweep 6
mcs6['sdqconduct6'] = mcs6['fconduct']
mcs6.loc[mcs6['sdqconduct6'] == 97, 'sdqconduct6'] = np.nan
mcs6.loc[mcs6['sdqconduct6'] < 0, 'sdqconduct6'] = np.nan
print(mcs6['sdqconduct6'].value_counts(dropna=False))
print(mcs6['sdqconduct6'].value_counts().sum())
# Sweep 7
mcs7['sdqconduct7'] = mcs7['gconduct']
mcs7.loc[mcs7['sdqconduct7'] == 97, 'sdqconduct7'] = np.nan
mcs7.loc[mcs7['sdqconduct7'] < 0, 'sdqconduct7'] = np.nan
print(mcs7['sdqconduct7'].value_counts(dropna=False))
print(mcs7['sdqconduct7'].value_counts().sum())
mcs7['sdqconductsr7'] = mcs7['gconduct_c']
mcs7.loc[mcs7['sdqconductsr7'] == 97, 'sdqconductsr7'] = np.nan
mcs7.loc[mcs7['sdqconductsr7'] < 0, 'sdqconductsr7'] = np.nan
print(mcs7['sdqconductsr7'].value_counts(dropna=False))
print(mcs7['sdqconductsr7'].value_counts().sum())

#SDQ Emotional symptoms
# Sweep 2.
mcs2['sdqemotion2'] = mcs2['bemotion']
mcs2.loc[mcs2['sdqemotion2'] == 97, 'sdqemotion2'] = np.nan
mcs2.loc[mcs2['sdqemotion2'] < 0, 'sdqemotion2'] = np.nan
print(mcs2['sdqemotion2'].value_counts(dropna=False))
print(mcs2['sdqemotion2'].value_counts().sum())
# Sweep 3
mcs3['sdqemotion3'] = mcs3['cemotion']
mcs3.loc[mcs3['sdqemotion3'] == 97, 'sdqemotion3'] = np.nan
mcs3.loc[mcs3['sdqemotion3'] < 0, 'sdqemotion3'] = np.nan
print(mcs3['sdqemotion3'].value_counts(dropna=False))
print(mcs3['sdqemotion3'].value_counts().sum())
# Sweep 4
mcs4['sdqemotion4'] = mcs4['ddemotion']
mcs4.loc[mcs4['sdqemotion4'] == 97, 'sdqemotion4'] = np.nan
mcs4.loc[mcs4['sdqemotion4'] < 0, 'sdqemotion4'] = np.nan
print(mcs4['sdqemotion4'].value_counts(dropna=False))
print(mcs4['sdqemotion4'].value_counts().sum())
# Sweep 5
mcs5['sdqemotion5'] = mcs5['eemotion']
mcs5.loc[mcs5['sdqemotion5'] == 97, 'sdqemotion5'] = np.nan
mcs5.loc[mcs5['sdqemotion5'] < 0, 'sdqemotion5'] = np.nan
print(mcs5['sdqemotion5'].value_counts(dropna=False))
print(mcs5['sdqemotion5'].value_counts().sum())
# Sweep 6
mcs6['sdqemotion6'] = mcs6['femotion']
mcs6.loc[mcs6['sdqemotion6'] == 97, 'sdqemotion6'] = np.nan
mcs6.loc[mcs6['sdqemotion6'] < 0, 'sdqemotion6'] = np.nan
print(mcs6['sdqemotion6'].value_counts(dropna=False))
print(mcs6['sdqemotion6'].value_counts().sum())
# Sweep 7
mcs7['sdqemotion7'] = mcs7['gemotion']
mcs7.loc[mcs7['sdqemotion7'] == 97, 'sdqemotion7'] = np.nan
mcs7.loc[mcs7['sdqemotion7'] < 0, 'sdqemotion7'] = np.nan
print(mcs7['sdqemotion7'].value_counts(dropna=False))
print(mcs7['sdqemotion7'].value_counts().sum())
mcs7['sdqemotionsr7'] = mcs7['gemotion_c']
mcs7.loc[mcs7['sdqemotionsr7'] == 97, 'sdqemotionsr7'] = np.nan
mcs7.loc[mcs7['sdqemotionsr7'] < 0, 'sdqemotionsr7'] = np.nan
print(mcs7['sdqemotionsr7'].value_counts(dropna=False))
print(mcs7['sdqemotionsr7'].value_counts().sum())

#SDQ Peer relationship problems
# Sweep 2.
mcs2['sdqpeer2'] = mcs2['bpeer']
mcs2.loc[mcs2['sdqpeer2'] == 97, 'sdqpeer2'] = np.nan
mcs2.loc[mcs2['sdqpeer2'] < 0, 'sdqpeer2'] = np.nan
print(mcs2['sdqpeer2'].value_counts(dropna=False))
print(mcs2['sdqpeer2'].value_counts().sum())
# Sweep 3
mcs3['sdqpeer3'] = mcs3['cpeer']
mcs3.loc[mcs3['sdqpeer3'] == 97, 'sdqpeer3'] = np.nan
mcs3.loc[mcs3['sdqpeer3'] < 0, 'sdqpeer3'] = np.nan
print(mcs3['sdqpeer3'].value_counts(dropna=False))
print(mcs3['sdqpeer3'].value_counts().sum())
# Sweep 4
mcs4['sdqpeer4'] = mcs4['ddpeer']
mcs4.loc[mcs4['sdqpeer4'] == 97, 'sdqpeer4'] = np.nan
mcs4.loc[mcs4['sdqpeer4'] < 0, 'sdqpeer4'] = np.nan
print(mcs4['sdqpeer4'].value_counts(dropna=False))
print(mcs4['sdqpeer4'].value_counts().sum())
# Sweep 5
mcs5['sdqpeer5'] = mcs5['epeer']
mcs5.loc[mcs5['sdqpeer5'] == 97, 'sdqpeer5'] = np.nan
mcs5.loc[mcs5['sdqpeer5'] < 0, 'sdqpeer5'] = np.nan
print(mcs5['sdqpeer5'].value_counts(dropna=False))
print(mcs5['sdqpeer5'].value_counts().sum())
# Sweep 6
mcs6['sdqpeer6'] = mcs6['fpeer']
mcs6.loc[mcs6['sdqpeer6'] == 97, 'sdqpeer6'] = np.nan
mcs6.loc[mcs6['sdqpeer6'] < 0, 'sdqpeer6'] = np.nan
print(mcs6['sdqpeer6'].value_counts(dropna=False))
print(mcs6['sdqpeer6'].value_counts().sum())
# Sweep 7
mcs7['sdqpeer7'] = mcs7['gpeer']
mcs7.loc[mcs7['sdqpeer7'] == 97, 'sdqpeer7'] = np.nan
mcs7.loc[mcs7['sdqpeer7'] < 0, 'sdqpeer7'] = np.nan
print(mcs7['sdqpeer7'].value_counts(dropna=False))
print(mcs7['sdqpeer7'].value_counts().sum())
mcs7['sdqpeersr7'] = mcs7['gpeer_c']
mcs7.loc[mcs7['sdqpeersr7'] == 97, 'sdqpeersr7'] = np.nan
mcs7.loc[mcs7['sdqpeersr7'] < 0, 'sdqpeersr7'] = np.nan
print(mcs7['sdqpeersr7'].value_counts(dropna=False))
print(mcs7['sdqpeersr7'].value_counts().sum())

#SDQ Prosocial behaviour
# Sweep 2.
mcs2['sdqprosoc2'] = mcs2['bprosoc']
mcs2.loc[mcs2['sdqprosoc2'] == 97, 'sdqprosoc2'] = np.nan
mcs2.loc[mcs2['sdqprosoc2'] < 0, 'sdqprosoc2'] = np.nan
print(mcs2['sdqprosoc2'].value_counts(dropna=False))
print(mcs2['sdqprosoc2'].value_counts().sum())
# Sweep 3
mcs3['sdqprosoc3'] = mcs3['cprosoc']
mcs3.loc[mcs3['sdqprosoc3'] == 97, 'sdqprosoc3'] = np.nan
mcs3.loc[mcs3['sdqprosoc3'] < 0, 'sdqprosoc3'] = np.nan
print(mcs3['sdqprosoc3'].value_counts(dropna=False))
print(mcs3['sdqprosoc3'].value_counts().sum())
# Sweep 4
mcs4['sdqprosoc4'] = mcs4['ddprosoc']
mcs4.loc[mcs4['sdqprosoc4'] == 97, 'sdqprosoc4'] = np.nan
mcs4.loc[mcs4['sdqprosoc4'] < 0, 'sdqprosoc4'] = np.nan
print(mcs4['sdqprosoc4'].value_counts(dropna=False))
print(mcs4['sdqprosoc4'].value_counts().sum())
# Sweep 5
mcs5['sdqprosoc5'] = mcs5['eprosoc']
mcs5.loc[mcs5['sdqprosoc5'] == 97, 'sdqprosoc5'] = np.nan
mcs5.loc[mcs5['sdqprosoc5'] < 0, 'sdqprosoc5'] = np.nan
print(mcs5['sdqprosoc5'].value_counts(dropna=False))
print(mcs5['sdqprosoc5'].value_counts().sum())
# Sweep 6
mcs6['sdqprosoc6'] = mcs6['fprosoc']
mcs6.loc[mcs6['sdqprosoc6'] == 97, 'sdqprosoc6'] = np.nan
mcs6.loc[mcs6['sdqprosoc6'] < 0, 'sdqprosoc6'] = np.nan
print(mcs6['sdqprosoc6'].value_counts(dropna=False))
print(mcs6['sdqprosoc6'].value_counts().sum())
# Sweep 7
mcs7['sdqprosoc7'] = mcs7['gprosoc']
mcs7.loc[mcs7['sdqprosoc7'] == 97, 'sdqprosoc7'] = np.nan
mcs7.loc[mcs7['sdqprosoc7'] < 0, 'sdqprosoc7'] = np.nan
print(mcs7['sdqprosoc7'].value_counts(dropna=False))
print(mcs7['sdqprosoc7'].value_counts().sum())
mcs7['sdqprosocsr7'] = mcs7['gprosoc_c']
mcs7.loc[mcs7['sdqprosocsr7'] == 97, 'sdqprosocsr7'] = np.nan
mcs7.loc[mcs7['sdqprosocsr7'] < 0, 'sdqprosocsr7'] = np.nan
print(mcs7['sdqprosocsr7'].value_counts(dropna=False))
print(mcs7['sdqprosocsr7'].value_counts().sum())

#SDQ Hyperactivity/inattention
# Sweep 2.
mcs2['sdqhyper2'] = mcs2['bhyper']
mcs2.loc[mcs2['sdqhyper2'] == 97, 'sdqhyper2'] = np.nan
mcs2.loc[mcs2['sdqhyper2'] < 0, 'sdqhyper2'] = np.nan
print(mcs2['sdqhyper2'].value_counts(dropna=False))
print(mcs2['sdqhyper2'].value_counts().sum())
# Sweep 3
mcs3['sdqhyper3'] = mcs3['chyper']
mcs3.loc[mcs3['sdqhyper3'] == 97, 'sdqhyper3'] = np.nan
mcs3.loc[mcs3['sdqhyper3'] < 0, 'sdqhyper3'] = np.nan
print(mcs3['sdqhyper3'].value_counts(dropna=False))
print(mcs3['sdqhyper3'].value_counts().sum())
# Sweep 4
mcs4['sdqhyper4'] = mcs4['ddhyper']
mcs4.loc[mcs4['sdqhyper4'] == 97, 'sdqhyper4'] = np.nan
mcs4.loc[mcs4['sdqhyper4'] < 0, 'sdqhyper4'] = np.nan
print(mcs4['sdqhyper4'].value_counts(dropna=False))
print(mcs4['sdqhyper4'].value_counts().sum())
# Sweep 5
mcs5['sdqhyper5'] = mcs5['ehyper']
mcs5.loc[mcs5['sdqhyper5'] == 97, 'sdqhyper5'] = np.nan
mcs5.loc[mcs5['sdqhyper5'] < 0, 'sdqhyper5'] = np.nan
print(mcs5['sdqhyper5'].value_counts(dropna=False))
print(mcs5['sdqhyper5'].value_counts().sum())
# Sweep 6
mcs6['sdqhyper6'] = mcs6['fhyper']
mcs6.loc[mcs6['sdqhyper6'] == 97, 'sdqhyper6'] = np.nan
mcs6.loc[mcs6['sdqhyper6'] < 0, 'sdqhyper6'] = np.nan
print(mcs6['sdqhyper6'].value_counts(dropna=False))
print(mcs6['sdqhyper6'].value_counts().sum())
# Sweep 7
mcs7['sdqhyper7'] = mcs7['ghyper']
mcs7.loc[mcs7['sdqhyper7'] == 97, 'sdqhyper7'] = np.nan
mcs7.loc[mcs7['sdqhyper7'] < 0, 'sdqhyper7'] = np.nan
print(mcs7['sdqhyper7'].value_counts(dropna=False))
print(mcs7['sdqhyper7'].value_counts().sum())
mcs7['sdqhypersr7'] = mcs7['ghyper_c']
mcs7.loc[mcs7['sdqhypersr7'] == 97, 'sdqhypersr7'] = np.nan
mcs7.loc[mcs7['sdqhypersr7'] < 0, 'sdqhypersr7'] = np.nan
print(mcs7['sdqhypersr7'].value_counts(dropna=False))
print(mcs7['sdqhypersr7'].value_counts().sum())

#SDQ Impact
# Sweep 2.
mcs2['sdqimpact2'] = mcs2['bimpact']
mcs2.loc[mcs2['sdqimpact2'] == 97, 'sdqimpact2'] = np.nan
mcs2.loc[mcs2['sdqimpact2'] < 0, 'sdqimpact2'] = np.nan
print(mcs2['sdqimpact2'].value_counts(dropna=False))
print(mcs2['sdqimpact2'].value_counts().sum())
# Sweep 3
mcs3['sdqimpact3'] = mcs3['cimpact']
mcs3.loc[mcs3['sdqimpact3'] == 97, 'sdqimpact3'] = np.nan
mcs3.loc[mcs3['sdqimpact3'] < 0, 'sdqimpact3'] = np.nan
print(mcs3['sdqimpact3'].value_counts(dropna=False))
print(mcs3['sdqimpact3'].value_counts().sum())
# Sweep 4
mcs4['sdqimpact4'] = mcs4['ddimpact']
mcs4.loc[mcs4['sdqimpact4'] == 97, 'sdqimpact4'] = np.nan
mcs4.loc[mcs4['sdqimpact4'] < 0, 'sdqimpact4'] = np.nan
print(mcs4['sdqimpact4'].value_counts(dropna=False))
print(mcs4['sdqimpact4'].value_counts().sum())

#SDQ Internalising
# Sweep 2.
mcs2['sdqinternal2'] = mcs2['sdqemotion2'] + mcs2['sdqpeer2']
print(mcs2['sdqinternal2'].value_counts(dropna=False))
print(mcs2['sdqinternal2'].value_counts().sum())
# Sweep 3
mcs3['sdqinternal3'] = mcs3['sdqemotion3'] + mcs3['sdqpeer3']
print(mcs3['sdqinternal3'].value_counts(dropna=False))
print(mcs3['sdqinternal3'].value_counts().sum())
# Sweep 4
mcs4['sdqinternal4'] = mcs4['sdqemotion4'] + mcs4['sdqpeer4']
print(mcs4['sdqinternal4'].value_counts(dropna=False))
print(mcs4['sdqinternal4'].value_counts().sum())
# Sweep 5
mcs5['sdqinternal5'] = mcs5['sdqemotion5'] + mcs5['sdqpeer5']
print(mcs5['sdqinternal5'].value_counts(dropna=False))
print(mcs5['sdqinternal5'].value_counts().sum())
# Sweep 6
mcs6['sdqinternal6'] = mcs6['sdqemotion6'] + mcs6['sdqpeer6']
print(mcs6['sdqinternal6'].value_counts(dropna=False))
print(mcs6['sdqinternal6'].value_counts().sum())
# Sweep 7
mcs7['sdqinternal7'] = mcs7['sdqemotion7'] + mcs7['sdqpeer7']
print(mcs7['sdqinternal7'].value_counts(dropna=False))
print(mcs7['sdqinternal7'].value_counts().sum())
mcs7['sdqinternalsr7'] = mcs7['sdqemotionsr7'] + mcs7['sdqpeersr7']
print(mcs7['sdqinternalsr7'].value_counts(dropna=False))
print(mcs7['sdqinternalsr7'].value_counts().sum())


#SDQ Externalising
mcs2['sdqexternal2'] = mcs2['sdqconduct2'] + mcs2['sdqhyper2']
print(mcs2['sdqexternal2'].value_counts(dropna=False))
print(mcs2['sdqexternal2'].value_counts().sum())
# Sweep 3
mcs3['sdqexternal3'] = mcs3['sdqconduct3'] + mcs3['sdqhyper3']
print(mcs3['sdqexternal3'].value_counts(dropna=False))
print(mcs3['sdqexternal3'].value_counts().sum())
# Sweep 4
mcs4['sdqexternal4'] = mcs4['sdqconduct4'] + mcs4['sdqhyper4']
print(mcs4['sdqexternal4'].value_counts(dropna=False))
print(mcs4['sdqexternal4'].value_counts().sum())
# Sweep 5
mcs5['sdqexternal5'] = mcs5['sdqconduct5'] + mcs5['sdqhyper5']
print(mcs5['sdqexternal5'].value_counts(dropna=False))
print(mcs5['sdqexternal5'].value_counts().sum())
# Sweep 6
mcs6['sdqexternal6'] = mcs6['sdqconduct6'] + mcs6['sdqhyper6']
print(mcs6['sdqexternal6'].value_counts(dropna=False))
print(mcs6['sdqexternal6'].value_counts().sum())
# Sweep 7
mcs7['sdqexternal7'] = mcs7['sdqconduct7'] + mcs7['sdqhyper7']
print(mcs7['sdqexternal7'].value_counts(dropna=False))
print(mcs7['sdqexternal7'].value_counts().sum())
mcs7['sdqexternalsr7'] = mcs7['sdqconductsr7'] + mcs7['sdqhypersr7']
print(mcs7['sdqexternalsr7'].value_counts(dropna=False))
print(mcs7['sdqexternalsr7'].value_counts().sum())




#Kessler
# Sweep 7
mcs7['kessler7'] = mcs7['gdckessl']
mcs7.loc[mcs7['kessler7'] < 0, 'kessler7'] = np.nan
print(mcs7['kessler7'].value_counts(dropna=False))
print(mcs7['kessler7'].value_counts().sum())


#Big 5
# Sweep 7
mcs7['bfpopen7'] = mcs7['gdcopen']
mcs7.loc[mcs7['bfpopen7'] < 0, 'bfpopen7'] = np.nan
print(mcs7['bfpopen7'].value_counts(dropna=False))
print(mcs7['bfpopen7'].value_counts().sum())
mcs7['bfpcons7'] = mcs7['gdcconsc']
mcs7.loc[mcs7['bfpcons7'] < 0, 'bfpcons7'] = np.nan
print(mcs7['bfpcons7'].value_counts(dropna=False))
print(mcs7['bfpcons7'].value_counts().sum())
mcs7['bfpextr7'] = mcs7['gdcextrav']
mcs7.loc[mcs7['bfpextr7'] < 0, 'bfpextr7'] = np.nan
print(mcs7['bfpextr7'].value_counts(dropna=False))
print(mcs7['bfpextr7'].value_counts().sum())
mcs7['bfpagre7'] = mcs7['gdcagree']
mcs7.loc[mcs7['bfpagre7'] < 0, 'bfpagre7'] = np.nan
print(mcs7['bfpagre7'].value_counts(dropna=False))
print(mcs7['bfpagre7'].value_counts().sum())
mcs7['bfpneur7'] = mcs7['gdcneurot']
mcs7.loc[mcs7['bfpneur7'] < 0, 'bfpneur7'] = np.nan
print(mcs7['bfpneur7'].value_counts(dropna=False))
print(mcs7['bfpneur7'].value_counts().sum())



### Wellbeing measures
#General wellbeing & Wellbeing grid
# Sweep 5
mcs5['genwelb5'] = mcs5['ecq10f00']
mcs5.loc[mcs5['genwelb5'] < 0, 'genwelb5'] = np.nan
print(mcs5['genwelb5'].value_counts(dropna=False))
print(mcs5['genwelb5'].value_counts().sum())
mcs5['welbgrid5'] = mcs5['ecq10a00'] + mcs5['ecq10b00'] + mcs5['ecq10c00'] + mcs5['ecq10d00'] + mcs5['ecq10e00'] + mcs5['ecq10f00']
mcs5.loc[mcs5['ecq10a00'] < 0, 'welbgrid5'] = np.nan
mcs5.loc[mcs5['ecq10b00'] < 0, 'welbgrid5'] = np.nan
mcs5.loc[mcs5['ecq10c00'] < 0, 'welbgrid5'] = np.nan
mcs5.loc[mcs5['ecq10d00'] < 0, 'welbgrid5'] = np.nan
mcs5.loc[mcs5['ecq10e00'] < 0, 'welbgrid5'] = np.nan
mcs5.loc[mcs5['ecq10f00'] < 0, 'welbgrid5'] = np.nan
print(mcs5['welbgrid5'].value_counts(dropna=False))
print(mcs5['welbgrid5'].value_counts().sum())
# Sweep 6
mcs6['genwelb6'] = mcs6['fclife00']
mcs6.loc[mcs6['genwelb6'] < 0, 'genwelb6'] = np.nan
print(mcs6['genwelb6'].value_counts(dropna=False))
print(mcs6['genwelb6'].value_counts().sum())
mcs6['welbgrid6'] = mcs6['fcscwk00'] + mcs6['fcwylk00'] + mcs6['fcfmly00'] + mcs6['fcfrns00'] + mcs6['fcschl00'] + mcs6['fclife00']
mcs6.loc[mcs6['fcscwk00'] < 0, 'welbgrid6'] = np.nan
mcs6.loc[mcs6['fcwylk00'] < 0, 'welbgrid6'] = np.nan
mcs6.loc[mcs6['fcfmly00'] < 0, 'welbgrid6'] = np.nan
mcs6.loc[mcs6['fcfrns00'] < 0, 'welbgrid6'] = np.nan
mcs6.loc[mcs6['fcschl00'] < 0, 'welbgrid6'] = np.nan
mcs6.loc[mcs6['fclife00'] < 0, 'welbgrid6'] = np.nan
print(mcs6['welbgrid6'].value_counts(dropna=False))
print(mcs6['welbgrid6'].value_counts().sum())

#SWEMWBS
# Sweep 7
mcs7['swemwbs7'] = mcs7['gdwemwbs']
mcs7.loc[mcs7['swemwbs7'] < 0, 'swemwbs7'] = np.nan
print(mcs7['swemwbs7'].value_counts(dropna=False))
print(mcs7['swemwbs7'].value_counts().sum())



###### Creating a combined dataset
#Variable list by sweep
#Sweep 1
vl1 = [
       "mcsid", "cnum",
       "cmagem1", "cmaged1", "sex1", "ethnicity301", "ethnicity1", "country1", "region1",
       "birthm1", "birthy1", "agemb1", "gestweeks1", "bthwt1", 
       "smokepreg1", "drinkpreg1", "brstfdevr1", "malm1", "malm_clin1", "cma1",
       "income1", "poverty1", "imddec1", "imdqnt1", "mrtlsts1", "mssngl1", "meduc1", "peduc1", "htrnt1", "htown1",
       "anysmkm1", "anysmkp1", "anydrnkm1", "anydrnkp1", "regdrnkm1", "regdrnkp1",
       "lsc_m1", "alc_m1", "lsc_p1", "alc_p1",
       "hhdis1", "nplfp1", "aplfp1", "snglprnt1", 'methnicity1', 'pethnicity1', 'biom1', 'biop1',
       "numppl1", "numchld1", "tmspntchldm1", "tmspntchldp1", 'thrswrkdm1', 'thrswrkdp1', 
       "hhag1ch1", "hhag2ch1", "hhag3ch1", "hhag4ch1", "buag1ch1", "buag2ch1", "buag3ch1", "buag4ch1",
       "weight1", "hosp1",
       "gmotor1", "fmotor1", "comms1", "develop1",
       "rimm_tir", "rimm_dep", "rimm_wor", "rimm_rag", "rimm_scr", "rimm_ups", "rimm_jit", "rimm_ner", "rimm_her",
       "cma_ann", "cma_thn", "cma_lev", "cma_cmp", "cma_pat", "cma_gip",
       "hsngtnr1", "hospac1", "hospilna1"
       ]

#Sweep 2
vl2 = [
       "mcsid", "cnum",
       "cmagem2", "cmaged2", "sex2", "ethnicity302", "ethnicity2", "country2", "region2",
       "birthm2", "birthy2", "agemb2", "bthwt2", 
       "income2", "poverty2", "imddec2", "imdqnt2", "mrtlsts2", "mssngl2", "htrnt2", "htown2", "meduc2", "peduc2",
       "hhdis2", "nplfp2", "aplfp2", "snglprnt2", "mmhlth2", "pmhlth2", "methnicity2", "pethnicity2", "biom2", "biop2",
       "numppl2", "numchld2", "tmspntchldm2", "tmspntchldp2", "hle2", "thrswrkdm2", "thrswrkdp2", 
       "anysmkm2", "anysmkp2", "anydrnkm2", "anydrnkp2", "regdrnkm2", "regdrnkp2", "anydrgm2", "anydrgp2",
       "lsc_m2", "lsc_p2",
       "psconm2", "psconp2", "psclom2", "psclop2",
       "hhag1ch2", "hhag2ch2", "hhag3ch2", "hhag4ch2", "buag1ch2", "buag2ch2", "buag3ch2", "buag4ch2",
       "height2", "weight2", "bmi2", "obesity2", "lsc2", "alc2", "hosp2",
       "zcog2","sclrdn2", "deldevscr2",
       "sdqconduct2", "sdqemotion2", "sdqpeer2", "sdqprosoc2", "sdqhyper2", "sdqinternal2", "sdqexternal2", "sdqimpact2",
       "hsngtnr2", "hospac2", "hospilna2", "basnv2", "zbasnv2", "bsrar2", "zbsrar2", "zcm2"
       ]
#Sweep 3
vl3 = [
       "mcsid", "cnum", 
       "cmagem3", "cmaged3", "sex3", "country3", "region3",
       "birthm3", "birthy3",
       "income3", "poverty3", "imddec3", "imdqnt3", "mrtlsts3", "mssngl3", "htrnt3", "htown3", "meduc3", "peduc3",
       "hhdis3", "nplfp3", "aplfp3", "snglprnt3", "thrswrkdm3", "thrswrkdp3", "mmhlth3", "pmhlth3",
       "numppl3", "numchld3", "tmspntchldm3", "tmspntchldp3", "hle3",
       "anysmkm3", "anysmkp3", "anydrnkm3", "anydrnkp3", "regdrnkm3", "regdrnkp3", "anydrgm3", "anydrgp3",
       "lsc_m3", "alc_m3", "lsc_p3", "alc_p3",
       "hhag1ch3", "hhag2ch3", "hhag3ch3", "hhag4ch3", "buag1ch3", "buag2ch3", "buag3ch3", "buag4ch3",
       "height3", "weight3", "bmi3", "obesity3", "lsc3", "alc3", "hosp3", 
       "zcog3", 
       "sdqconduct3", "sdqemotion3", "sdqpeer3", "sdqprosoc3", "sdqhyper3", "sdqinternal3", "sdqexternal3", "sdqimpact3",
       "hsngtnr3", "hospac3", "hospilna3", "basnv3", "zbasnv3", "baspc3", "zbaspc3", "basps3", "zbasps3", "zcm3" 
       ]
#Sweep 4
vl4 = [
       "mcsid", "cnum", 
       "cmagem4", "cmaged4", "sex4", "country4", "region4",
       "birthm4", "birthy4",
       "income4", "poverty4", "imddec4", "imdqnt4", "mrtlsts4", "mssngl4", "htrnt4", "htown4", "meduc4", "peduc4",
       "hhdis4", "nplfp4", "aplfp4", "snglprnt4", "thrswrkdm4", "thrswrkdp4", "mmhlth4", "pmhlth4",
       "numppl4", "numchld4", "tmspntchldm4", "tmspntchldp4",
       "anysmkm4", "anysmkp4", "anydrnkm4", "anydrnkp4", "regdrnkm4", "regdrnkp4",
       "lsc_m4", "alc_m4", "lsc_p4", "alc_p4",
       "hhag1ch4", "hhag2ch4", "hhag3ch4", "hhag4ch4", "buag1ch4", "buag2ch4", "buag3ch4", "buag4ch4",
       "height4", "weight4", "bmi4", "obesity4", "lsc4", "alc4", "hosp4", 
       "sen4", 
       "zcog4", 
       "sdqconduct4", "sdqemotion4", "sdqpeer4", "sdqprosoc4", "sdqhyper4", "sdqinternal4", "sdqexternal4", "sdqimpact4",
       "hsngtnr4", "hospac4", "hospilna4", "baswr4", "zbaswr4", "baspc4", "zbaspc4", "nferpm4", "znferpm4", "zcm4"
       ]
#Sweep 5
vl5 = [
       "mcsid", "cnum", 
       "cmagem5", "cmagey5", "sex5", "country5", "region5", 
       "birthm5", "birthy5",
       "income5", "wealth5", "poverty5", "imddec5", "imdqnt5", "mrtlsts5", "mssngl5", "htrnt5", "htown5", "meduc5", "peduc5", 
       "hhdis5", "nplfp5", "aplfp5", "snglprnt5", "thrswrkdm5", "thrswrkdp5", "mmhlth5", "pmhlth5",
       "numppl5", "numchld5", "tmspntchldm5", "tmspntchldp5",
       "anysmkm5", "anysmkp5", "anydrnkm5", "anydrnkp5", "regdrnkm5", "regdrnkp5",
       "lsc_m5", "alc_m5", "lsc_p5", "alc_p5",
       "hhag1ch5", "hhag2ch5", "hhag3ch5", "hhag4ch5", "buag1ch5", "buag2ch5", "buag3ch5", "buag4ch5",
       "height5", "weight5", "bmi5", "obesity5", "lsc5", "alc5", "hosp5", 
       "excl5", "truancy5", "regtruancy5", "sen5", 
       "zcog5", 
       "sdqconduct5", "sdqemotion5", "sdqpeer5", "sdqprosoc5", "sdqhyper5", "sdqinternal5", "sdqexternal5", 
       "genwelb5", "welbgrid5",
       "hsngtnr5", "hospac5", "hospilna5", "texcl5", "pexcl5", "nwoffscl5", "basvs5"
       ]
#Sweep 6
vl6 = [
       "mcsid", "cnum", 
       "cmagem6", "cmagey6", "sex6", "country6", "region6", 
       "birthm6", "birthy6",
       "income6", "wealth6", "poverty6", "imddec6", "imdqnt6", "mrtlsts6", "mssngl6", "htrnt6", "htown6", "meduc6", "peduc6", 
       "hhdis6", "nplfp6", "aplfp6", "snglprnt6", "thrswrkdm6", "thrswrkdp6", "mmhlth6", "pmhlth6",
       "numppl6", "numchld6",
       "anysmkm6", "anysmkp6", "anydrnkm6", "anydrnkp6", "regdrnkm6", "regdrnkp6", "anydrgm6", "anydrgp6",
       "lsc_m6", "alc_m6", "lsc_p6", "alc_p6",
       "hhag1ch6", "hhag2ch6", "hhag3ch6", "hhag4ch6", "buag1ch6", "buag2ch6", "buag3ch6", "buag4ch6",
       "height6", "weight6", "bmi6", "obesity6", "lsc6", "alc6", "hosp6", 
       "smkevr6", "smkreg6", "drnkevr6", "drnkfreq6", "drnkreg6", 
       "excl6", "truancy6", "regtruancy6", "sen6", 
       "zcog6", 
       "sdqconduct6", "sdqemotion6", "sdqpeer6", "sdqprosoc6", "sdqhyper6", "sdqinternal6", "sdqexternal6", 
       "genwelb6", "welbgrid6",
       "hsngtnr6", "hospac6", "hospilna6", "smkfreq6", "texcl6", "pexcl6", "nwoffscl6", "apuvt6"
       ]
#Sweep 7
vl7 = [
       "mcsid", "cnum", 
       "cmagem7", "cmagey7", "sex7", "country7", "region7", 
       "birthm7", "birthy7",
       "mrtlsts7", "mssngl7", "thrswrkdm7", "thrswrkdp7", 
       "snglprnt7", "numppl7", "numchld7", "gmeduc7", "gfeduc7", "pgmeduc7", "pgfeduc7",
       "hhag1ch7", "hhag2ch7", "hhag3ch7", "hhag4ch7", "buag1ch7", "buag2ch7", "buag3ch7", "buag4ch7",
       "height7", "weight7", "bmi7", "obesity7", "lsc7", "alc7", "hosp7", 
       "smkevr7", "smkreg7", "drnkevr7", "drnkfreq7", "drnkreg7", 
       "prfrhlth7", 
       "gdgcse7", "bdgcse7", "gdgcseme7", "bdgcseme7",
       "zcog7", 
       "sdqconduct7", "sdqconductsr7", "sdqemotion7", "sdqemotionsr7", "sdqpeer7", "sdqpeersr7", "sdqprosoc7", "sdqprosocsr7", "sdqhyper7", "sdqhypersr7", "sdqinternal7", "sdqinternalsr7", "sdqexternal7", "sdqexternalsr7", 
       "kessler7", 
       "bfpopen7", "bfpcons7", "bfpextr7", "bfpagre7", "bfpneur7", 
       "swemwbs7",
       "hospilna7", "smkfreq7", "nagla7"
       ]

#Create combined file

mcs = pd.merge(mcs2[vl2], mcs1[vl1], on=['mcsid', 'cnum'], how='outer')
mcs = pd.merge(mcs, mcs3[vl3], on=['mcsid', 'cnum'], how='outer')
mcs = pd.merge(mcs, mcs4[vl4], on=['mcsid', 'cnum'], how='outer')
mcs = pd.merge(mcs, mcs5[vl5], on=['mcsid', 'cnum'], how='outer')
mcs = pd.merge(mcs, mcs6[vl6], on=['mcsid', 'cnum'], how='outer')
mcs = pd.merge(mcs, mcs7[vl7], on=['mcsid', 'cnum'], how='outer')

del mcs1
del mcs2
del mcs3
del mcs4
del mcs5
del mcs6
del mcs7
del vl1
del vl2
del vl3
del vl4
del vl5
del vl6
del vl7


##### Sample weights

lff = pd.read_stata(os.path.join(data, 'lff/stata/stata13/mcs_longitudinal_family_file.dta'), convert_categoricals=False)
lff.columns= lff.columns.str.lower()
print(lff['mcsid'].nunique())
## Overall MCS weight
lff['owt_cs'] = lff['weight1']  #Country specific
lff['owt_uk'] = lff['weight2']  #UK weight
lff.loc[lff['owt_cs'] < 0, 'owt_cs'] = np.nan
lff.loc[lff['owt_uk'] < 0, 'owt_uk'] = np.nan
## Weight including non-response
# Sweep 1
lff['wt_cs1'] = lff['aovwt1']  #Country specific
lff['wt_uk1'] = lff['aovwt2']  #UK weight
lff.loc[lff['wt_cs1'] < 0, 'wt_cs1'] = np.nan
lff.loc[lff['wt_uk1'] < 0, 'wt_uk1'] = np.nan
# Sweep 2
lff['wt_cs2'] = lff['bovwt1']  #Country specific
lff['wt_uk2'] = lff['bovwt2']  #UK weight
lff.loc[lff['wt_cs2'] < 0, 'wt_cs2'] = np.nan
lff.loc[lff['wt_uk2'] < 0, 'wt_uk2'] = np.nan
# Sweep 3
lff['wt_cs3'] = lff['covwt1']  #Country specific
lff['wt_uk3'] = lff['covwt2']  #UK weight
lff.loc[lff['wt_cs3'] < 0, 'wt_cs3'] = np.nan
lff.loc[lff['wt_uk3'] < 0, 'wt_uk3'] = np.nan
# Sweep 4
lff['wt_cs4'] = lff['dovwt1']  #Country specific
lff['wt_uk4'] = lff['dovwt2']  #UK weight
lff.loc[lff['wt_cs4'] < 0, 'wt_cs4'] = np.nan
lff.loc[lff['wt_uk4'] < 0, 'wt_uk4'] = np.nan
# Sweep 5
lff['wt_cs5'] = lff['eovwt1']  #Country specific
lff['wt_uk5'] = lff['eovwt2']  #UK weight
lff.loc[lff['wt_cs5'] < 0, 'wt_cs5'] = np.nan
lff.loc[lff['wt_uk5'] < 0, 'wt_uk5'] = np.nan
# Sweep 6
lff['wt_cs6'] = lff['fovwt1']  #Country specific
lff['wt_uk6'] = lff['fovwt2']  #UK weight
lff.loc[lff['wt_cs6'] < 0, 'wt_cs6'] = np.nan
lff.loc[lff['wt_uk6'] < 0, 'wt_uk6'] = np.nan
# Sweep 7
lff['wt_cs7'] = lff['govwt1']  #Country specific
lff['wt_uk7'] = lff['govwt2']  #UK weight 
lff.loc[lff['wt_cs7'] < 0, 'wt_cs7'] = np.nan
lff.loc[lff['wt_uk7'] < 0, 'wt_uk7'] = np.nan

vll = [
       "mcsid",
       "owt_cs", "owt_uk",
       "wt_cs1", "wt_uk1",
       "wt_cs2", "wt_uk2",
       "wt_cs3", "wt_uk3",
       "wt_cs4", "wt_uk4",
       "wt_cs5", "wt_uk5",
       "wt_cs6", "wt_uk6",
       "wt_cs7", "wt_uk7",
       "nocmhh", "sptn00",
       "pttype2", "nh2"
       ]

    
mcs = pd.merge(mcs, lff[vll], how='left', on = ['mcsid'])

del vll
del lff

##### Other Basic variables
##### Birth characteristics
##Birth date
mcs['birthm'] = np.where(mcs['birthm1'].isna(), mcs['birthm2'], mcs['birthm1'])
print(mcs['birthm'].value_counts(dropna=False))
print(mcs['birthm'].value_counts().sum())
print(mcs['birthm1'].value_counts().sum())
mcs['birthy'] = np.where(mcs['birthy1'].isna(), mcs['birthy2'], mcs['birthy1'])
print(mcs['birthy'].value_counts(dropna=False))
print(mcs['birthy'].value_counts().sum())
print(mcs['birthy1'].value_counts().sum())

##Birth month advantage
mcs['bmadv1'] = np.nan
mcs.loc[mcs['birthm'] == 9, 'bmadv1'] = 1
mcs.loc[mcs['birthm'] == 10, 'bmadv1'] = 2
mcs.loc[mcs['birthm'] == 11, 'bmadv1'] = 3
mcs.loc[mcs['birthm'] == 12, 'bmadv1'] = 4
mcs.loc[mcs['birthm'] == 1, 'bmadv1'] = 5
mcs.loc[mcs['birthm'] == 2, 'bmadv1'] = 6
mcs.loc[mcs['birthm'] == 3, 'bmadv1'] = 7
mcs.loc[mcs['birthm'] == 4, 'bmadv1'] = 8
mcs.loc[mcs['birthm'] == 5, 'bmadv1'] = 9
mcs.loc[mcs['birthm'] == 6, 'bmadv1'] = 10
mcs.loc[mcs['birthm'] == 7, 'bmadv1'] = 11
mcs.loc[mcs['birthm'] == 8, 'bmadv1'] = 12
print(mcs['bmadv1'].value_counts(dropna=False))
print(mcs['bmadv1'].value_counts().sum())

##Mothers age at birth
mcs['agemb'] = np.where(mcs['agemb1'].isna(), mcs['agemb2'], mcs['agemb1'])
print(mcs['agemb'].value_counts(dropna=False))
print(mcs['agemb'].value_counts().sum())
print(mcs['agemb1'].value_counts().sum())

##Birth weight
mcs['bthwt'] = np.where(mcs['bthwt1'].isna(), mcs['bthwt2'], mcs['bthwt1'])
print(mcs['bthwt'].value_counts(dropna=False))
print(mcs['bthwt'].value_counts().sum())
print(mcs['bthwt1'].value_counts().sum())

##Sex
mcs['sex'] = np.where(mcs['sex1'].isna(), mcs['sex2'], mcs['sex1'])
print(mcs['sex'].value_counts(dropna=False))
print(mcs['sex'].value_counts().sum())
print(mcs['sex1'].value_counts(dropna=False))
print(mcs['sex1'].value_counts().sum())
mcs['male'] = np.where(mcs['sex'] == 1, 1, 0)
mcs['male'] = np.where(mcs['sex'].isna(), np.nan, mcs['male'])
print(mcs['male'].value_counts(dropna=False))
print(mcs['male'].value_counts().sum())
# print(mcs['sex1'].value_counts(dropna=False))
# print(mcs['sex2'].value_counts(dropna=False))
# mcs['male1'] = np.where(mcs['sex1'] == 1, 1, 0)
# mcs['male1'] = np.where(mcs['sex1'].isna(), np.nan, mcs['male1'])
# mcs['male2'] = np.where(mcs['sex2'] == 1, 1, 0)
# mcs['male2'] = np.where(mcs['sex2'].isna(), np.nan, mcs['male2'])
# print(mcs['male1'].value_counts(dropna=False))
# print(mcs['male2'].value_counts(dropna=False))
# mcs['male'] = np.where(mcs['sex1'].isna(), mcs['male2'], mcs['male1'])
# print(mcs['male'].value_counts(dropna=False))
# print(mcs['male'].value_counts().sum())

##Ethnicity
mcs['ethnicity'] = np.where(mcs['ethnicity1'].isnull() == 1, mcs['ethnicity2'], mcs['ethnicity1'])
print(mcs['ethnicity'].value_counts(dropna=False))
print(mcs['ethnicity'].value_counts().sum())
print(mcs['ethnicity1'].value_counts().sum())
# mcs['ethnicity30'] = np.where(mcs['ethnicity301'].isnull() == 1, mcs['ethnicity302'], mcs['ethnicity301'])
# print(mcs['ethnicity30'].value_counts(dropna=False))
# print(mcs['ethnicity30'].value_counts().sum())
# print(mcs['ethnicity301'].value_counts().sum())

# Code to replace ethnicity with bio parent ethnicity where missing
print(mcs['methnicity1'].value_counts(dropna=False))
print(mcs['pethnicity1'].value_counts(dropna=False))
print(mcs['methnicity2'].value_counts(dropna=False))
print(mcs['pethnicity2'].value_counts(dropna=False))
mcs['methnicity1'] = np.where(mcs['biom1'] == 1, mcs['methnicity1'], np.nan)
mcs['pethnicity1'] = np.where(mcs['biop1'] == 1, mcs['pethnicity1'], np.nan)
mcs['methnicity2'] = np.where(mcs['biom2'] == 1, mcs['methnicity2'], np.nan)
mcs['pethnicity2'] = np.where(mcs['biop2'] == 1, mcs['pethnicity2'], np.nan)
mcs['methnicity'] = np.where(mcs['methnicity1'].isna(), mcs['methnicity2'], mcs['methnicity1'])
mcs['pethnicity'] = np.where(mcs['pethnicity1'].isna(), mcs['pethnicity2'], mcs['pethnicity1'])
print(mcs['methnicity'].value_counts(dropna=False))
print(mcs['pethnicity'].value_counts(dropna=False))
mcs['bopeth'] = np.where(mcs['methnicity'].isna(), mcs['pethnicity'], mcs['methnicity'])
print(mcs['bopeth'].value_counts(dropna=False))
mcs['ethnicity'] = np.where(mcs['ethnicity'].isna(), mcs['bopeth'], mcs['ethnicity'])
print(mcs['ethnicity'].value_counts(dropna=False))
print(mcs['ethnicity'].value_counts().sum())

##Country
mcs['country'] = np.where(mcs['country1'].isnull() == 1, mcs['country2'], mcs['country1'])
print(mcs['country'].value_counts(dropna=False))
print(mcs['country'].value_counts().sum())
print(mcs['country1'].value_counts().sum())

##Region
mcs['region'] = np.where(mcs['region1'].isnull() == 1, mcs['region2'], mcs['region1'])
print(mcs['region'].value_counts(dropna=False))
print(mcs['region'].value_counts().sum())
print(mcs['region1'].value_counts().sum())

## Hours worked by parents
# Loop through sweeps
for i in range(1, 8):
    # Total hours worked
    print(mcs[f'thrswrkdm{i}'].value_counts(dropna=False))
    print(mcs[f'thrswrkdm{i}'].value_counts().sum())
    print(mcs[f'thrswrkdp{i}'].value_counts(dropna=False))
    print(mcs[f'thrswrkdp{i}'].value_counts().sum())
    mcs[f'thrswrkdhh{i}'] = np.where((mcs[f'thrswrkdm{i}'].isna() == 0) | (mcs[f'thrswrkdp{i}'].isna() == 0), 
                                    np.nansum([mcs[f'thrswrkdm{i}'], mcs[f'thrswrkdp{i}']], axis=0), np.nan)
    print(mcs[f'thrswrkdhh{i}'].value_counts(dropna=False))
    print(mcs[f'thrswrkdhh{i}'].value_counts().sum())
    # Average hours worked
    print(mcs[f'snglprnt{i}'].value_counts(dropna=False))
    print(mcs[f'snglprnt{i}'].value_counts().sum())
    mcs[f'ahrswrkdhh{i}'] = np.where(mcs[f'snglprnt{i}'] == 1, mcs[f'thrswrkdhh{i}'], mcs[f'thrswrkdhh{i}']/2)
    print(mcs[f'ahrswrkdhh{i}'].value_counts(dropna=False))
    print(mcs[f'ahrswrkdhh{i}'].value_counts().sum())


## Highest parental Education
for i in range(1, 7):
    print(mcs[f'meduc{i}'].value_counts(dropna=False))
    print(mcs[f'meduc{i}'].value_counts().sum())
    print(mcs[f'peduc{i}'].value_counts(dropna=False))
    print(mcs[f'peduc{i}'].value_counts().sum())
    mcs[f'hheduc{i}'] = np.where(mcs[f'meduc{i}'] < mcs[f'peduc{i}'], 
                                    mcs[f'peduc{i}'], mcs[f'meduc{i}'])
    print(mcs[f'hheduc{i}'].value_counts(dropna=False))
    print(mcs[f'hheduc{i}'].value_counts().sum())
    
## Highest grandparent Education
#Maternal grandparents
print(mcs['gmeduc7'].value_counts(dropna=False))
print(mcs['gmeduc7'].value_counts().sum())
print(mcs['gfeduc7'].value_counts(dropna=False))
print(mcs['gfeduc7'].value_counts().sum())
mcs['gpeduc7'] = np.where(mcs['gmeduc7'] < mcs['gfeduc7'], 
                                mcs['gfeduc7'], mcs['gmeduc7'])
print(mcs['gpeduc7'].value_counts(dropna=False))
print(mcs['gpeduc7'].value_counts().sum())

#Fraternal grandparents
print(mcs['pgmeduc7'].value_counts(dropna=False))
print(mcs['pgmeduc7'].value_counts().sum())
print(mcs['pgfeduc7'].value_counts(dropna=False))
print(mcs['pgfeduc7'].value_counts().sum())
mcs['fgpeduc7'] = np.where(mcs['pgmeduc7'] < mcs['pgfeduc7'], 
                                mcs['pgfeduc7'], mcs['pgmeduc7'])
print(mcs['fgpeduc7'].value_counts(dropna=False))
print(mcs['fgpeduc7'].value_counts().sum())

#####Act Early - List of birth disadvantages

####1#Lifelimiting conditions at 0
###Using "lifelimiting2" as there is no age 0 measure
print(mcs['alc2'].value_counts(dropna=False))
print(mcs['alc2'].value_counts().sum())

###Pediatric compex chronic conditions age 3 or age 5
# List of three-digit codes
icdcodes = ["978", "B20", "B21", "B22", "B23", "B24", "CXX", "D00", "D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08", "D09", "D37", "D38", "D39", "D40", "D41", "D42", "D43", "D44", "D45", "D46", "D47", "D48", "D49", "D55", "D56", "D57", "D58", "D60", "D61", "D66", "D68", "D69", "D70", "D71", "D72", "D76", "D80", "D81", "D82", "D83", "D84", "D85", "D86", "D87", "D88", "D3A", "E00", "E22", "E23", "E24", "E25", "E26", "E34", "E70", "E71", "E72", "E74", "E75", "E76", "E77", "E78", "E79", "E80", "E83", "E84", "E85", "E88", "F71", "F72", "F73", "F84", "G10", "G11", "G12", "G20", "G21", "G23", "G24", "G25", "G31", "G37", "G40", "G47", "G71", "G72", "G80", "G81", "G82", "G83", "G90", "G91", "G93", "G94", "G95", "G97", "H49", "I27", "I34", "I36", "I37", "I42", "I43", "I44", "I45", "I47", "I48", "I49", "I50", "I51", "I63", "I82", "J95", "J96", "K44", "K50", "K51", "K55", "K56", "K59", "K73", "K74", "K75", "K76", "K94", "M30", "M31", "M32", "M33", "M34", "M35", "M41", "M43", "M96", "N18", "N31", "P05", "P07", "P10", "P11", "P21", "P25", "P27", "P28", "P35", "P52", "P56", "P57", "P61", "P77", "P83", "P84", "P91", "Q00", "Q01", "Q02", "Q03", "Q04", "Q05", "Q06", "Q07", "Q20", "Q21", "Q22", "Q23", "Q24", "Q25", "Q26", "Q28", "Q30", "Q31", "Q32", "Q33", "Q34", "Q39", "Q41", "Q42", "Q43", "Q44", "Q45", "Q60", "Q61", "Q62", "Q63", "Q64", "Q72", "Q75", "Q76", "Q77", "Q78", "Q79", "Q81", "Q85", "Q87", "Q89", "Q90", "Q91", "Q92", "Q93", "Q95", "Q96", "Q97", "Q98", "Q99", "R00", "R40", "T82", "T84", "T85", "T86", "T87", "Y83", "Z21", "Z43", "Z44", "Z45", "Z46", "Z49", "Z79", "Z90", "Z91", "Z93", "Z94", "Z95", "Z96", "Z98", "Z99"]

#bmclscaa bmclscab bmclscac bmclscad cmclsxaa cmclsxab dmclscab dmclxmab dmclscac dmclxmac dmclscaa dmclxmaa
##Sweep 2
icdcols2 = ['MCSID', 'bmclscaa', 'bmclscab', 'bmclscac', 'bmclscad', 'bmclscba', 'bmclscbb', 'bmclscbc', 'bmclscbd', 'bmclscca', 'bmclsccb', 'bmclsccc', 'bmclsccd']
icd2 = pd.read_stata(os.path.join(data, 'MCS with ICD codes/mcs2_parent_interview.dta'), columns = icdcols2)
icd2.columns= icd2.columns.str.lower()
# Step 1: Rename columns
icd2_dict = {
    'bmclscaa': 'lsicd1_1',
    'bmclscab': 'lsicd2_1',
    'bmclscac': 'lsicd3_1',
    'bmclscad': 'lsicd4_1',
    'bmclscba': 'lsicd1_2',
    'bmclscbb': 'lsicd2_2',
    'bmclscbc': 'lsicd3_2',
    'bmclscbd': 'lsicd4_2',
    'bmclscca': 'lsicd1_3',
    'bmclsccb': 'lsicd2_3',
    'bmclsccc': 'lsicd3_3',
    'bmclsccd': 'lsicd4_3',
}
icd2.rename(columns=icd2_dict, inplace=True)
# Step 2: Melt the DataFrame
icd2 = icd2.melt(id_vars='mcsid', var_name='lsicd_cm', value_name='value')
# Step 3: Extract 'cm' and 'lsicdX'
icd2['cnum'] = icd2['lsicd_cm'].str.split('_').str[1]
icd2['lsicd'] = icd2['lsicd_cm'].str.split('_').str[0]
# Step 4: Pivot back to wide format
icd2 = icd2.pivot(index=['mcsid', 'cnum'], columns='lsicd', values='value').reset_index()
# Rename 'cm' column to clarify meaning and ensure it's integer type
icd2['cnum'] = icd2['cnum'].astype(int)
# Reorder and rename columns as necessary
icd2 = icd2[['mcsid', 'cnum', 'lsicd1', 'lsicd2', 'lsicd3', 'lsicd4']]
# Create an indicator for each observation
icdcols = ['lsicd1', 'lsicd2', 'lsicd3', 'lsicd4']
icd2['pccc2'] = icd2.apply(lambda row: any(row[col] in icdcodes for col in icdcols if pd.notna(row[col])), axis=1)
icd2['pccc2'] = icd2['pccc2'].astype(int)
icd2 = icd2.drop(['lsicd1', 'lsicd2', 'lsicd3', 'lsicd4'], axis=1)
mcs = pd.merge(mcs, icd2, how='left', on=['mcsid', 'cnum'])
print(mcs['pccc2'].value_counts(dropna=False))
print(mcs['pccc2'].value_counts().sum())
mcs.loc[mcs['lsc2'] == 0, 'pccc2'] = 0
mcs.loc[mcs['lsc2'].isna(), 'pccc2'] = np.nan
print(mcs['pccc2'].value_counts(dropna=False))
print(mcs['pccc2'].value_counts().sum())
del icd2
del icd2_dict
del icdcols2
##Sweep 3
icdcols3 = ['MCSID','cmclsxaa', 'cmclsxab', 'cmclsxac', 'cmclsxba', 'cmclsxbb', 'cmclsxbc']
icd3 = pd.read_stata(os.path.join(data, 'MCS with ICD codes/mcs3_parent_interview.dta'), columns = icdcols3)
icd3.columns= icd3.columns.str.lower()
# Step 1: Rename columns
icd3_dict = {
    'cmclsxaa': 'lsicd1_1',
    'cmclsxab': 'lsicd2_1',
    'cmclsxac': 'lsicd3_1',
    'cmclsxba': 'lsicd1_2',
    'cmclsxbb': 'lsicd2_2',
    'cmclsxbc': 'lsicd3_2',
}
icd3.rename(columns=icd3_dict, inplace=True)
# Step 2: Melt the DataFrame
icd3 = icd3.melt(id_vars='mcsid', var_name='lsicd_cm', value_name='value')
# Step 3: Extract 'cm' and 'lsicdX'
icd3['cnum'] = icd3['lsicd_cm'].str.split('_').str[1]
icd3['lsicd'] = icd3['lsicd_cm'].str.split('_').str[0]
# Step 4: Pivot back to wide format
icd3 = icd3.pivot(index=['mcsid', 'cnum'], columns='lsicd', values='value').reset_index()
# Rename 'cm' column to clarify meaning and ensure it's integer type
icd3['cnum'] = icd3['cnum'].astype(int)
# Reorder and rename columns as necessary
icd3 = icd3[['mcsid', 'cnum', 'lsicd1', 'lsicd2', 'lsicd3']]
# Create an indicator for each observation
icdcols = ['lsicd1', 'lsicd2', 'lsicd3']
icd3['pccc3'] = icd3.apply(lambda row: any(row[col] in icdcodes for col in icdcols if pd.notna(row[col])), axis=1)
icd3['pccc3'] = icd3['pccc3'].astype(int)
icd3 = icd3.drop(['lsicd1', 'lsicd2', 'lsicd3'], axis=1)
mcs = pd.merge(mcs, icd3, how='left', on=['mcsid', 'cnum'])
print(mcs['pccc3'].value_counts(dropna=False))
print(mcs['pccc3'].value_counts().sum())
mcs.loc[mcs['lsc3'] == 0, 'pccc3'] = 0
mcs.loc[mcs['lsc3'].isna(), 'pccc3'] = np.nan
print(mcs['pccc3'].value_counts(dropna=False))
print(mcs['pccc3'].value_counts().sum())
del icd3
del icd3_dict
del icdcols3
##Sweep 4
icdcols4 = ['MCSID', 'dmclscaa', 'dmclscab', 'dmclscac', 'dmclscba', 'dmclscbb', 'dmclscbc', 'dmclscca', 'dmclsccb', 'dmclsccc']
icd4 = pd.read_stata(os.path.join(data, 'MCS with ICD codes/mcs4_parent_interview.dta'), columns = icdcols4)
icd4.columns= icd4.columns.str.lower()
# Step 1: Rename columns
icd4_dict = {
    'dmclscaa': 'lsicd1_1',
    'dmclscab': 'lsicd2_1',
    'dmclscac': 'lsicd3_1',
    'dmclscba': 'lsicd1_2',
    'dmclscbb': 'lsicd2_2',
    'dmclscbc': 'lsicd3_2',
    'dmclscca': 'lsicd1_3',
    'dmclsccb': 'lsicd2_3',
    'dmclsccc': 'lsicd3_3',
}
icd4.rename(columns=icd4_dict, inplace=True)
# Step 2: Melt the DataFrame
icd4 = icd4.melt(id_vars='mcsid', var_name='lsicd_cm', value_name='value')
# Step 3: Extract 'cm' and 'lsicdX'
icd4['cnum'] = icd4['lsicd_cm'].str.split('_').str[1]
icd4['lsicd'] = icd4['lsicd_cm'].str.split('_').str[0]
# Step 4: Pivot back to wide format
icd4 = icd4.pivot(index=['mcsid', 'cnum'], columns='lsicd', values='value').reset_index()
# Rename 'cm' column to clarify meaning and ensure it's integer type
icd4['cnum'] = icd4['cnum'].astype(int)
# Reorder and rename columns as necessary
icd4 = icd4[['mcsid', 'cnum', 'lsicd1', 'lsicd2', 'lsicd3']]
# Create an indicator for each observation
icdcols = ['lsicd1', 'lsicd2', 'lsicd3']
icd4['pccc4'] = icd4.apply(lambda row: any(row[col] in icdcodes for col in icdcols if pd.notna(row[col])), axis=1)
icd4['pccc4'] = icd4['pccc4'].astype(int)
icd4 = icd4.drop(['lsicd1', 'lsicd2', 'lsicd3'], axis=1)
mcs = pd.merge(mcs, icd4, how='left', on=['mcsid', 'cnum'])
print(mcs['pccc4'].value_counts(dropna=False))
print(mcs['pccc4'].value_counts().sum())
mcs.loc[mcs['lsc4'] == 0, 'pccc4'] = 0
mcs.loc[mcs['lsc4'].isna(), 'pccc4'] = np.nan
print(mcs['pccc4'].value_counts(dropna=False))
print(mcs['pccc4'].value_counts().sum())
del icd4
del icd4_dict
del icdcols4
# #Sweep 5
# icdcols5 = ['MCSID', ]
# icd5 = pd.read_stata(os.path.join(data, 'MCS with ICD codes/mcs5_cm_capi.dta'), columns = icdcols5)
# icd5.columns= icd5.columns.str.lower()
# #Sweep 6
# icdcols6 = ['MCSID',]
# icd6 = pd.read_stata(os.path.join(data, 'MCS with ICD codes/mcs6_parent_cm_interview.dta'), columns = icdcols6)
# icd6.columns= icd6.columns.str.lower()
# #Sweep 7
# icdcols7 = ['MCSID', ]
# icd7 = pd.read_stata(os.path.join(data, 'MCS with ICD codes/mcs4_parent_interview.dta'), columns = icdcols7)
# icd7.columns= icd7.columns.str.lower()
del icdcols
del icdcodes


# Generate Combined observations
##Sweep 2 or 3
# Define conditions
conditions = [
    ((mcs['pccc2'] == 1) | (mcs['pccc3'] == 1)),  # Either m or n is 1
    ((mcs['pccc2'].isna()) & (mcs['pccc3'].isna()))  # Both m and n are np.nan
]
# Define choices
choices = [
    1,  # Corresponding to the first condition
    np.nan  # Corresponding to the second condition
]
# Create the new variable x with default=0 for cases not matching the conditions above
mcs['pccc23'] = np.select(conditions, choices, default=0)
print(mcs['pccc23'].value_counts(dropna=False))
print(mcs['pccc23'].value_counts().sum())
##Sweep 3 or 4
conditions = [
    ((mcs['pccc3'] == 1) | (mcs['pccc4'] == 1)),  # Either m or n is 1
    ((mcs['pccc3'].isna()) & (mcs['pccc4'].isna()))  # Both m and n are np.nan
]
# Define choices
choices = [
    1,  # Corresponding to the first condition
    np.nan  # Corresponding to the second condition
]
# Create the new variable x with default=0 for cases not matching the conditions above
mcs['pccc34'] = np.select(conditions, choices, default=0)
print(mcs['pccc34'].value_counts(dropna=False))
print(mcs['pccc34'].value_counts().sum())
##Sweeps 2, 3 or 4
conditions = [
    ((mcs['pccc2'] == 1) | (mcs['pccc3'] == 1) | (mcs['pccc4'] == 1)),  # Either m or n is 1
    ((mcs['pccc2'].isna()) & (mcs['pccc3'].isna()) & (mcs['pccc4'].isna()))  # Both m and n are np.nan
]
# Define choices
choices = [
    1,  # Corresponding to the first condition
    np.nan  # Corresponding to the second condition
]
# Create the new variable x with default=0 for cases not matching the conditions above
mcs['pccc234'] = np.select(conditions, choices, default=0)
print(mcs['pccc234'].value_counts(dropna=False))
print(mcs['pccc234'].value_counts().sum())

del choices
del conditions

####2#Teenage Parent
mcs['teenbrth'] = np.nan
mcs.loc[mcs['agemb'] < 20, 'teenbrth'] = 1
mcs.loc[mcs['agemb'] >= 20, 'teenbrth'] = 0
print(mcs['teenbrth'].value_counts(dropna=False))
print(mcs['teenbrth'].value_counts().sum())

####3#Preterm Birth
mcs['pretrm'] = np.nan
mcs.loc[mcs['gestweeks1'] < 28, 'pretrm'] = 3
mcs.loc[(mcs['gestweeks1'] >= 28) & (mcs['gestweeks1'] < 32), 'pretrm'] = 2
mcs.loc[(mcs['gestweeks1'] >= 32) & (mcs['gestweeks1'] < 37), 'pretrm'] = 1
mcs.loc[mcs['gestweeks1'] >= 37, 'pretrm'] = 0
print(mcs['pretrm'].value_counts(dropna=False))
print(mcs['pretrm'].value_counts().sum())
# Generate 'bthwt_c1' and 'bthwt_c2' based on 'bthwt_c'
#mcs = pd.concat([mcs, pd.get_dummies(mcs['pretrm'], prefix = 'pretrm')], axis=1)
mcs['pretrm_1'] = np.where(mcs['pretrm'] == 1, 1, 0)
mcs['pretrm_1'] = np.where(mcs['pretrm'].isnull(), np.nan, mcs['pretrm_1'])
mcs['pretrm_2'] = np.where(mcs['pretrm'] == 2, 1, 0)
mcs['pretrm_2'] = np.where(mcs['pretrm'].isnull(), np.nan, mcs['pretrm_2'])
mcs['pretrm_3'] = np.where(mcs['pretrm'] == 3, 1, 0)
mcs['pretrm_3'] = np.where(mcs['pretrm'].isnull(), np.nan, mcs['pretrm_3'])
print(mcs['pretrm_1'].value_counts(dropna=False))
print(mcs['pretrm_2'].value_counts(dropna=False))
print(mcs['pretrm_3'].value_counts(dropna=False))
####4#Birth weight for gestational age
# adbwgta0 - birth weight
#Weight for age at 9 months
# adlstwa0 - most recent weight (9 months)
# Load the weight chart dataset
weight_chart = pd.read_stata(os.path.join(lfsm, 'data/weightchart/wfga.dta'), convert_categoricals=False)
#pd.read_stata('M:/Desktop/LifeSim/Python Code/lifesim2-main/data/weightchart/wfga.dta')
# Merge the datasets on 'male' and 'gestwks'
#Round down to nearest integer
mcs['gestwks'] = np.floor(mcs['gestweeks1'])
#truncate at 43 weeks
mcs['gestwks'] = mcs['gestwks'].clip(upper=43)
#merge
mcs = mcs.merge(weight_chart, on=['male', 'gestwks'], how='left', indicator=True)
# Drop rows that only exist in the weight chart dataset
mcs = mcs[mcs['_merge'] != 'right_only']
# Drop the '_merge' column
mcs.drop(columns='_merge', inplace=True)

# Generate 'bthwt_c' based on conditions
mcs['bthwt_c'] = np.where(
    (mcs['bthwt']*1000 > mcs['p10']) & 
    (mcs['bthwt']*1000 < mcs['p90']) & 
    (mcs['bthwt'].notnull()) & (mcs['bthwt'] >= 0),
    0, np.nan
)
mcs['bthwt_c'] = np.where(
    (mcs['bthwt']*1000 <= mcs['p10']) & 
    (mcs['bthwt'].notnull()) & (mcs['bthwt'] >= 0),
    1, mcs['bthwt_c']
)
mcs['bthwt_c'] = np.where(
    (mcs['bthwt']*1000 >= mcs['p90']) & 
    (mcs['bthwt'].notnull()) & (mcs['bthwt'] >= 0),
    2, mcs['bthwt_c']
)

# Tabulate 'bthwt_c' (this prints the frequency table to the console)
print(mcs['bthwt_c'].value_counts(dropna=False))

# Drop 'p10' and 'p90' columns
mcs.drop(columns=['p10', 'p90'], inplace=True)

# Generate 'bthwt_1' and 'bthwt_2' based on 'bthwt_c'
mcs['bthwt_1'] = np.where(mcs['bthwt_c'] == 1, 1, 0)
mcs['bthwt_1'] = np.where(mcs['bthwt_c'].isnull(), np.nan, mcs['bthwt_1'])
mcs['bthwt_2'] = np.where(mcs['bthwt_c'] == 2, 1, 0)
mcs['bthwt_2'] = np.where(mcs['bthwt_c'].isnull(), np.nan, mcs['bthwt_2'])
print(mcs['bthwt_1'].value_counts(dropna=False))
print(mcs['bthwt_2'].value_counts(dropna=False))
del weight_chart

# Weight at 3 years
# wtm = pd.read_excel('M:/Desktop/LifeSim/Python Code/lifesim2-main/data/weightchart/tab_wfa_boys_p_0_5.xlsx')
# wtm.columns= wtm.columns.str.lower()
# wtm = wtm[['month', 'p10', 'p90']]
# wtm = wtm.rename(columns={"p10":"p10_wtm", "p90":"p90_wtm"})
# wtf = pd.read_excel('M:/Desktop/LifeSim/Python Code/lifesim2-main/data/weightchart/tab_wfa_girls_p_0_5.xlsx')
# wtf.columns= wtf.columns.str.lower()
# wtf = wtf[['month', 'p10', 'p90']]
# wtf = wtf.rename(columns={"p10":"p10_wtf", "p90":"p90_wtf"})

# mcs['month'] = mcs['cmagem1']
# mcs = pd.merge(mcs, wtm, how='left', on='month', suffixes=('', '_m'))
# mcs = pd.merge(mcs, wtf, how='left', on='month', suffixes=('', '_m'))

# mcs['bthwt_c'] = np.nan
# mcs.loc[(mcs['male'] == 1) & (mcs['adlstwa0'] > mcs['p10_wtm']) & (mcs['adlstwa0'] < mcs['p90_wtm']), 'bthwt_c'] = 0
# mcs.loc[(mcs['male'] == 1) & (mcs['adlstwa0'] <= mcs['p10_wtm']), 'bthwt_c'] = 1
# mcs.loc[(mcs['male'] == 1) & (mcs['adlstwa0'] >= mcs['p90_wtm']), 'bthwt_c'] = 2
# mcs.loc[(mcs['male'] == 0) & (mcs['adlstwa0'] > mcs['p10_wtf']) & (mcs['adlstwa0'] < mcs['p90_wtf']), 'bthwt_c'] = 0
# mcs.loc[(mcs['male'] == 0) & (mcs['adlstwa0'] <= mcs['p10_wtf']), 'bthwt_c'] = 1
# mcs.loc[(mcs['male'] == 0) & (mcs['adlstwa0'] >= mcs['p90_wtf']), 'bthwt_c'] = 2
# print(mcs['bthwt_c'].value_counts(dropna=False))
# print(mcs['bthwt_c'].value_counts().sum())
# mcs = pd.concat([mcs, pd.get_dummies(mcs['bthwt_c'], prefix = 'bthwt')], axis=1)

# mcs = mcs.drop(['month', 'p10_wtm', 'p90_wtm', 'p10_wtf', 'p90_wtf'], axis=1)

# del wtf
# del wtm

#5#Low length for gestational age
#Based on height at age 3 - bchcmc00
htm = pd.read_excel(os.path.join(lfsm, 'data/heightchart/tab_lhfa_boys_p_2_5.xlsx'))
#pd.read_excel('M:/Desktop/LifeSim/Python Code/lifesim2-main/data/heightchart/tab_lhfa_boys_p_2_5.xlsx')
htm.columns = htm.columns.str.lower()
htm = htm[['month', 'p10', 'p25']]
htm = htm.rename(columns={"p10":"p10_htm", "p25":"p25_htm"})
htf = pd.read_excel(os.path.join(lfsm, 'data/heightchart/tab_lhfa_girls_p_2_5.xlsx'))
#pd.read_excel('M:/Desktop/LifeSim/Python Code/lifesim2-main/data/heightchart/tab_lhfa_girls_p_2_5.xlsx')
htf.columns = htf.columns.str.lower()
htf = htf[['month', 'p10', 'p25']]
htf = htf.rename(columns={"p10":"p10_htf", "p25":"p25_htf"})

print(htm['month'].value_counts(dropna=False))
print(htf['month'].value_counts(dropna=False))

print(mcs['cmagem2'].value_counts(dropna=False))
print(mcs['cmagem2'].value_counts().sum())
mcs['month'] = mcs['cmagem2']
mcs = pd.merge(mcs, htm, how='left', on='month', suffixes=('', '_m'))
mcs = pd.merge(mcs, htf, how='left', on='month', suffixes=('', '_m'))

print(mcs['height2'].value_counts().sum())
print(mcs['p10_htm'].value_counts().sum())
print(mcs['p10_htf'].value_counts().sum())
mcs['lwht2'] = np.nan
mcs.loc[(mcs['male'] == 1) & (mcs['height2'] > mcs['p10_htm']), 'lwht2'] = 0
mcs.loc[(mcs['male'] == 1) & (mcs['height2'] <= mcs['p10_htm']), 'lwht2'] = 1
mcs.loc[(mcs['male'] == 0) & (mcs['height2'] > mcs['p10_htf']), 'lwht2'] = 0
mcs.loc[(mcs['male'] == 0) & (mcs['height2'] <= mcs['p10_htf']), 'lwht2'] = 1
print(mcs['lwht2'].value_counts(dropna=False))
print(mcs['lwht2'].value_counts().sum())

mcs = mcs.drop(['month', 'p10_htm', 'p25_htm', 'p10_htf', 'p25_htf'], axis=1)

del htf
del htm

#Based on height at age 5
ht = pd.read_stata(os.path.join(lfsm, 'data/heightchart/hmmf.dta'), convert_categoricals=False)
#pd.read_stata('M:/Desktop/LifeSim/Python Code/lifesim2-main/data/heightchart/hmmf.dta')
ht.columns = ht.columns.str.lower()

print(ht['months'].value_counts(dropna=False))

print(mcs['cmagem3'].value_counts(dropna=False))
print(mcs['cmagem3'].value_counts().sum())
mcs['months'] = mcs['cmagem3']
mcs = pd.merge(mcs, ht, how='left', on=['male','months'])

print(mcs['height3'].value_counts().sum())
print(mcs['p10'].value_counts().sum())
mcs['lwht3'] = np.nan
mcs.loc[(mcs['height3'] > mcs['p10']), 'lwht3'] = 0
mcs.loc[(mcs['height3'] <= mcs['p10']), 'lwht3'] = 1
print(mcs['lwht3'].value_counts(dropna=False))
print(mcs['lwht3'].value_counts().sum())

mcs = mcs.drop(['months', 'p10'], axis=1)

del ht

#6#Delayed development
#Age in months at time of interview (9 Months)
print(mcs['cmagem1'].value_counts(dropna=False))

##Age Adjusting the measures
formula = "gmotor1 ~ C(cmagem1)"
m = smf.ols(formula, data=mcs).fit()
adj = m.resid + m.params.Intercept
mcs['gmotor1_adj'] = adj
plt.hist(mcs['gmotor1_adj'], alpha=0.5, label='Adjusted', bins=10)
plt.hist(mcs['gmotor1'], alpha=0.5, label='Unadjusted', bins=10)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of Gross Motor Skills')
plt.legend()
plt.show()
    
formula = "fmotor1 ~ C(cmagem1)"
m = smf.ols(formula, data=mcs).fit()
adj = m.resid + m.params.Intercept
mcs['fmotor1_adj'] = adj
plt.hist(mcs['fmotor1_adj'], alpha=0.5, label='Adjusted', bins=10)
plt.hist(mcs['fmotor1'], alpha=0.5, label='Unadjusted', bins=10)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of Fine Motor Skills')
plt.legend()
plt.show()

formula = "comms1 ~ C(cmagem1)"
m = smf.ols(formula, data=mcs).fit()
adj = m.resid + m.params.Intercept
mcs['comms1_adj'] = adj
plt.hist(mcs['comms1_adj'], alpha=0.5, label='Adjusted', bins=10)
plt.hist(mcs['comms1'], alpha=0.5, label='Unadjusted', bins=10)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of Communication')
plt.legend()
plt.show()

formula = "develop1 ~ C(cmagem1)"
m = smf.ols(formula, data=mcs).fit()
adj = m.resid + m.params.Intercept
mcs['develop1_adj'] = adj
plt.hist(mcs['develop1_adj'], alpha=0.5, label='Adjusted', bins=10)
plt.hist(mcs['develop1'], alpha=0.5, label='Unadjusted', bins=10)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of Development Score')
plt.legend()
plt.show()

del formula
del m
del adj

mcs['motor1'] = mcs['fmotor1'] + mcs['gmotor1']
print(mcs['motor1'].value_counts(dropna=False))
print(mcs['motor1'].value_counts().sum())
mcs['motor1_adj'] = mcs['fmotor1_adj'] + mcs['gmotor1_adj']
plt.hist(mcs['motor1_adj'], alpha=0.5, label='Adjusted', bins=16)
plt.hist(mcs['motor1'], alpha=0.5, label='Unadjusted', bins=16)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of Motor Development Score')
plt.legend()
plt.show()
motor_p10 = np.nanquantile(mcs['motor1_adj'],0.1)
mcs['motor_b10'] = mcs['motor1_adj'].apply(lambda x: np.nan if pd.isna(x) else (1 if x <= motor_p10 else 0))
print(mcs['motor_b10'].value_counts(dropna=False))
print(mcs['motor_b10'].value_counts().sum())

del motor_p10
#Delayed developemnt at age 3
plt.hist(mcs['zcog2'], alpha=1, bins=60)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of Cognitive score (Age 3)')
plt.show()
zcog_p10 = np.nanquantile(mcs['zcog2'],0.1)
mcs['zcog2_b10'] = mcs['zcog2'].apply(lambda x: np.nan if pd.isna(x) else (1 if x <= -1.282 else 0))
print(mcs['zcog2_b10'].value_counts(dropna=False))
print(mcs['zcog2_b10'].value_counts().sum())
mcs['deldev2'] = mcs['zcog2_b10']

#Delayed developemnt at age 3
plt.hist(mcs['zcog2'], alpha=1, bins=60)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of Cognitive score (Age 3)')
plt.show()
zcog_p10 = np.nanquantile(mcs['zcog2'],0.1)
mcs['zcog2_b10'] = mcs['zcog2'].apply(lambda x: np.nan if pd.isna(x) else (1 if x <= -1.282 else 0))
print(mcs['zcog2_b10'].value_counts(dropna=False))
print(mcs['zcog2_b10'].value_counts().sum())
mcs['deldev2'] = mcs['zcog2_b10']

#Delayed developemnt at age 5
plt.hist(mcs['zcog3'], alpha=1, bins=60)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of Cognitive score (Age5)')
plt.show()
zcog_p10 = np.nanquantile(mcs['zcog3'],0.1)
mcs['zcog3_b10'] = mcs['zcog3'].apply(lambda x: np.nan if pd.isna(x) else (1 if x <= -1.282 else 0))
print(mcs['zcog3_b10'].value_counts(dropna=False))
print(mcs['zcog3_b10'].value_counts().sum())
mcs['deldev3'] = mcs['zcog3_b10']

del zcog_p10

#Age Adjusting raw cognitive ability zscores
#Age 14
formula = "zcog6 ~ C(cmagem6)"
m = smf.ols(formula, data=mcs).fit()
adj = m.resid + m.params.Intercept
mcs['zcog6'] = adj
#Age 17
formula = "zcog7 ~ C(cmagem7)"
m = smf.ols(formula, data=mcs).fit()
adj = m.resid + m.params.Intercept
mcs['zcog7'] = adj

#Life satisfaction/Wellbeing measures
#Sweep 2
be = np.arange(start=0, stop=11, step=1)
plt.hist(mcs['sdqemotion2'], alpha=1, bins=be)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of SDQ Emotion at age 3')
plt.show()
mcs['internal2'] = 20 - mcs['sdqinternal2']
be = np.arange(start=0, stop=21, step=1)
plt.hist(mcs['internal2'], alpha=1, bins=be)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of SDQ Internalising at age 3')
plt.show()
mcs['lifesat2'] = 2 + (mcs['internal2'] * 8 / 21)
mcs['zlifesat2'] = (mcs['internal2']-mcs['internal2'].mean())/mcs['internal2'].std()
#Sweep 3
be = np.arange(start=0, stop=11, step=1)
plt.hist(mcs['sdqemotion3'], alpha=1, bins=be)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of SDQ Emotion at age 5')
plt.show()
mcs['internal3'] = 20 - mcs['sdqinternal3']
be = np.arange(start=0, stop=21, step=1)
plt.hist(mcs['internal3'], alpha=1, bins=be)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of SDQ Internalising at age 5')
plt.show()
mcs['lifesat3'] = 2 + (mcs['internal3'] * 8 / 21)
mcs['zlifesat3'] = (mcs['internal3']-mcs['internal3'].mean())/mcs['internal3'].std()
#Sweep 4
be = np.arange(start=0, stop=11, step=1)
plt.hist(mcs['sdqemotion4'], alpha=1, bins=be)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of SDQ Emotion at age 7')
plt.show()
mcs['internal4'] = 20 - mcs['sdqinternal4']
be = np.arange(start=0, stop=21, step=1)
plt.hist(mcs['internal4'], alpha=1, bins=be)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of SDQ Internalising at age 7')
plt.show()
mcs['lifesat4'] = 2 + (mcs['internal4'] * 8 / 21)
mcs['zlifesat4'] = (mcs['internal4']-mcs['internal4'].mean())/mcs['internal4'].std()
#Sweep 5
be = np.arange(start=0, stop=11, step=1)
plt.hist(mcs['sdqemotion5'], alpha=1, bins=be)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of SDQ Emotion at age 11')
plt.show()
mcs['internal5'] = 20 - mcs['sdqinternal5']
be = np.arange(start=0, stop=21, step=1)
plt.hist(mcs['internal5'], alpha=1, bins=be)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of SDQ Internalising at age 11')
plt.show()
print(mcs['genwelb5'].value_counts(dropna=False))
print(mcs['genwelb5'].value_counts().sum())
mcs.loc[mcs['genwelb5'] < 0, 'genwelb5'] = np.nan
mcs['genwelb5'] = 8 - mcs['genwelb5']
be = np.arange(start=1, stop=8, step=1)
plt.hist(mcs['genwelb5'], alpha=1, bins=be)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of happiness with life at age 11')
plt.show()
print(mcs['genwelb5'].value_counts(dropna=False))
print(mcs['genwelb5'].value_counts().sum())
mcs['lifesatwb5'] = 2 + (mcs['genwelb5'] * 8 / 7)
mcs['zlifesatwb5'] = (mcs['genwelb5']-mcs['genwelb5'].mean())/mcs['genwelb5'].std()
mcs['lifesat5'] = 2 + (mcs['internal5'] * 8 / 21)
mcs['zlifesat5'] = (mcs['internal5']-mcs['internal5'].mean())/mcs['internal5'].std()
#Sweep 6
be = np.arange(start=0, stop=11, step=1)
plt.hist(mcs['sdqemotion6'], alpha=1, bins=be)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of SDQ Emotion at age 14')
plt.show()
mcs['internal6'] = 20 - mcs['sdqinternal6']
be = np.arange(start=0, stop=21, step=1)
plt.hist(mcs['internal6'], alpha=1, bins=be)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of SDQ Internalising at age 14')
plt.show()
print(mcs['genwelb6'].value_counts(dropna=False))
print(mcs['genwelb6'].value_counts().sum())
mcs.loc[mcs['genwelb6'] < 0, 'genwelb6'] = np.nan
mcs['genwelb6'] = 8 - mcs['genwelb6']
be = np.arange(start=1, stop=8, step=1)
plt.hist(mcs['genwelb6'], alpha=1, bins=be)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of happiness with life at age 14')
plt.show()
print(mcs['genwelb6'].value_counts(dropna=False))
print(mcs['genwelb6'].value_counts().sum())
mcs['lifesatwb6'] = 2 + (mcs['genwelb6'] * 8 / 7)
mcs['zlifesatwb6'] = (mcs['genwelb6']-mcs['genwelb6'].mean())/mcs['genwelb6'].std()
mcs['lifesat6'] = 2 + (mcs['internal6'] * 8 / 21)
mcs['zlifesat6'] = (mcs['internal6']-mcs['internal6'].mean())/mcs['internal6'].std()
#Sweep 7
## SDQ internalising score for wellbeing equations
be = np.arange(start=0, stop=11, step=1)
plt.hist(mcs['sdqemotion7'], alpha=1, bins=be)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of SDQ Emotion at age 5')
plt.show()
be = np.arange(start=0, stop=11, step=1)
plt.hist(mcs['sdqemotionsr7'], alpha=1, bins=be)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of SDQ Emotion at age 5')
plt.show()
mcs['internal7'] = 20 - mcs['sdqinternal7']
mcs['internalsr7'] = 20 - mcs['sdqinternalsr7']
be = np.arange(start=0, stop=21, step=1)
plt.hist(mcs['internal7'], alpha=1, bins=be)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of SDQ Internalising at age 17')
plt.show()
print(mcs['swemwbs7'].value_counts(dropna=False))
print(mcs['swemwbs7'].value_counts().sum())
mcs.loc[mcs['swemwbs7'] < 0, 'swemwbs7'] = np.nan
be = np.arange(start=7, stop=36, step=1)
plt.hist(mcs['swemwbs7'], alpha=1, bins=be)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of SWEMWBS at age 14')
plt.show()
print(mcs['swemwbs7'].value_counts(dropna=False))
print(mcs['swemwbs7'].value_counts().sum())
mcs['wemws7'] = mcs['swemwbs7'] - 7
mcs['wemws7'] = 28 - mcs['wemws7']
print(mcs['wemws7'].value_counts(dropna=False))
print(mcs['wemws7'].value_counts().sum())
mcs['lifesatwb7'] = 2 + (mcs['wemws7'] * 8 / 29)
mcs['zlifesatwb7'] = (mcs['wemws7']-mcs['wemws7'].mean())/mcs['wemws7'].std()
mcs['lifesat7'] = 2 + (mcs['internal7'] * 8 / 21)
mcs['zlifesat7'] = (mcs['internal7']-mcs['internal7'].mean())/mcs['internal7'].std()
mcs['lifesatsr7'] = 2 + (mcs['internalsr7'] * 8 / 21)
mcs['zlifesatsr7'] = (mcs['internalsr7']-mcs['internalsr7'].mean())/mcs['internalsr7'].std()
del be


#Income
#Sweep 1
mcs['lincome1'] = np.log(mcs['income1'])
#Sweep 2
mcs['lincome2'] = np.log(mcs['income2'])
#Sweep 3
mcs['lincome3'] = np.log(mcs['income3'])
#Sweep 4
mcs['lincome4'] = np.log(mcs['income4'])
#Sweep 5
mcs['lincome5'] = np.log(mcs['income5'])
#Sweep 6
mcs['lincome6'] = np.log(mcs['income6'])
# #Sweep 7
# mcs['lincome7'] = np.log(mcs['income7'])
#Early years permanent income
mcs['income123'] = (mcs['income1'] + mcs['income2'] + mcs['income3'])/3
mcs['lincome123'] = np.log(mcs['income123'])
#Permanent years quintiles
varl = ["income123"]
stats = ["nobs", "missing", "mean", "std", "median"]
desc = sms.stats.descriptivestats.describe(mcs[varl], stats=stats, numeric=True).T
print(desc)
varl = ["income123"]
stats = ["percentiles"]
desc = sms.stats.descriptivestats.describe(mcs[varl], stats=stats, numeric=True, percentiles=(20, 40, 60, 80) ).T
print(desc)
mcs['income123_q1'] = np.where(
    (mcs['income123'] <= desc.loc['income123', '20%']) & 
    (mcs['income123'].notnull()) & (mcs['income123'] >= 0),
    1, np.nan
)
mcs.loc[mcs['income123'].isnull(), 'income123_q1'] = np.nan
print(mcs['income123_q1'].value_counts(dropna=False))
mcs['income123_q2'] = np.where(
    (mcs['income123'] > desc.loc['income123', '20%']) & 
    (mcs['income123'] <= desc.loc['income123', '40%']) & 
    (mcs['income123'].notnull()) & (mcs['income123'] >= 0),
    1, 0
)
mcs.loc[mcs['income123'].isnull(), 'income123_q2'] = np.nan
print(mcs['income123_q2'].value_counts(dropna=False))
mcs['income123_q3'] = np.where(
    (mcs['income123'] > desc.loc['income123', '40%']) & 
    (mcs['income123'] <= desc.loc['income123', '60%']) & 
    (mcs['income123'].notnull()) & (mcs['income123'] >= 0),
    1, 0
)
mcs.loc[mcs['income123'].isnull(), 'income123_q3'] = np.nan
print(mcs['income123_q3'].value_counts(dropna=False))
mcs['income123_q4'] = np.where(
    (mcs['income123'] > desc.loc['income123', '60%']) & 
    (mcs['income123'] <= desc.loc['income123', '80%']) & 
    (mcs['income123'].notnull()) & (mcs['income123'] >= 0),
    1, 0
)
mcs.loc[mcs['income123'].isnull(), 'income123_q4'] = np.nan
print(mcs['income123_q4'].value_counts(dropna=False))
mcs['income123_q5'] = np.where(
    (mcs['income123'] > desc.loc['income123', '80%']) & 
    (mcs['income123'].notnull()) & (mcs['income123'] >= 0),
    1, 0
)
mcs.loc[mcs['income123'].isnull(), 'income123_q5'] = np.nan
print(mcs['income123_q5'].value_counts(dropna=False))
varl = ["income123"]
stats = ["nobs", "missing", "mean", "std", "median"]
desc = sms.stats.descriptivestats.describe(mcs[mcs['income123_q1'] == 1][varl], stats=stats, numeric=True).T
print(desc)
desc = sms.stats.descriptivestats.describe(mcs[mcs['income123_q2'] == 1][varl], stats=stats, numeric=True).T
print(desc)
desc = sms.stats.descriptivestats.describe(mcs[mcs['income123_q3'] == 1][varl], stats=stats, numeric=True).T
print(desc)
desc = sms.stats.descriptivestats.describe(mcs[mcs['income123_q4'] == 1][varl], stats=stats, numeric=True).T
print(desc)
desc = sms.stats.descriptivestats.describe(mcs[mcs['income123_q5'] == 1][varl], stats=stats, numeric=True).T
print(desc)

del varl
del stats
del desc

#Poverty
#Early years poverty
mcs['poverty123'] = (mcs['poverty1'] + mcs['poverty2'] + mcs['poverty3'])/3
print(mcs['poverty123'].value_counts(dropna=False))
print(mcs['poverty123'].value_counts().sum())

#Country dummies
mcs['country1_1'] = np.where(mcs['country'] == 1, 1, 0)
mcs['country1_1'] = np.where(mcs['country'].isnull() == 1, np.nan, mcs['country1_1'])
mcs['country1_2'] = np.where(mcs['country'] == 2, 1, 0)
mcs['country1_2'] = np.where(mcs['country'].isnull() == 1, np.nan, mcs['country1_2'])
mcs['country1_3'] = np.where(mcs['country'] == 3, 1, 0)
mcs['country1_3'] = np.where(mcs['country'].isnull() == 1, np.nan, mcs['country1_3'])
mcs['country1_4'] = np.where(mcs['country'] == 4, 1, 0)
mcs['country1_4'] = np.where(mcs['country'].isnull() == 1, np.nan, mcs['country1_4'])
print(mcs['country1_1'].value_counts(dropna=False))
print(mcs['country1_2'].value_counts(dropna=False))
print(mcs['country1_3'].value_counts(dropna=False))
print(mcs['country1_4'].value_counts(dropna=False))

for i in range(2, 4):
    mcs[f'country{i}_1'] = np.where(mcs[f'country{i}'] == 1, 1, 0)
    mcs[f'country{i}_1'] = np.where(mcs[f'country{i}'].isnull() == 1, np.nan, mcs[f'country{i}_1'])
    mcs[f'country{i}_2'] = np.where(mcs[f'country{i}'] == 2, 1, 0)
    mcs[f'country{i}_2'] = np.where(mcs[f'country{i}'].isnull() == 1, np.nan, mcs[f'country{i}_2'])
    mcs[f'country{i}_3'] = np.where(mcs[f'country{i}'] == 3, 1, 0)
    mcs[f'country{i}_3'] = np.where(mcs[f'country{i}'].isnull() == 1, np.nan, mcs[f'country{i}_3'])
    mcs[f'country{i}_4'] = np.where(mcs[f'country{i}'] == 4, 1, 0)
    mcs[f'country{i}_4'] = np.where(mcs[f'country{i}'].isnull() == 1, np.nan, mcs[f'country{i}_4'])
    print(mcs[f'country{i}_1'].value_counts(dropna=False))
    print(mcs[f'country{i}_2'].value_counts(dropna=False))
    print(mcs[f'country{i}_3'].value_counts(dropna=False))
    print(mcs[f'country{i}_4'].value_counts(dropna=False))


#Region dummies
mcs['region1_1'] = np.where(mcs['region'] == 1, 1, 0)
mcs['region1_1'] = np.where(mcs['region'].isnull() == 1, np.nan, mcs['region1_1'])
mcs['region1_2'] = np.where(mcs['region'] == 2, 1, 0)
mcs['region1_2'] = np.where(mcs['region'].isnull() == 1, np.nan, mcs['region1_2'])
mcs['region1_3'] = np.where(mcs['region'] == 3, 1, 0)
mcs['region1_3'] = np.where(mcs['region'].isnull() == 1, np.nan, mcs['region1_3'])
mcs['region1_4'] = np.where(mcs['region'] == 4, 1, 0)
mcs['region1_4'] = np.where(mcs['region'].isnull() == 1, np.nan, mcs['region1_4'])
mcs['region1_5'] = np.where(mcs['region'] == 5, 1, 0)
mcs['region1_5'] = np.where(mcs['region'].isnull() == 1, np.nan, mcs['region1_5'])
mcs['region1_6'] = np.where(mcs['region'] == 6, 1, 0)
mcs['region1_6'] = np.where(mcs['region'].isnull() == 1, np.nan, mcs['region1_6'])
mcs['region1_7'] = np.where(mcs['region'] == 7, 1, 0)
mcs['region1_7'] = np.where(mcs['region'].isnull() == 1, np.nan, mcs['region1_7'])
mcs['region1_8'] = np.where(mcs['region'] == 8, 1, 0)
mcs['region1_8'] = np.where(mcs['region'].isnull() == 1, np.nan, mcs['region1_8'])
mcs['region1_9'] = np.where(mcs['region'] == 9, 1, 0)
mcs['region1_9'] = np.where(mcs['region'].isnull() == 1, np.nan, mcs['region1_9'])
mcs['region1_10'] = np.where(mcs['region'] == 10, 1, 0)
mcs['region1_10'] = np.where(mcs['region'].isnull() == 1, np.nan, mcs['region1_10'])
mcs['region1_11'] = np.where(mcs['region'] == 11, 1, 0)
mcs['region1_11'] = np.where(mcs['region'].isnull() == 1, np.nan, mcs['region1_11'])
mcs['region1_12'] = np.where(mcs['region'] == 12, 1, 0)
mcs['region1_12'] = np.where(mcs['region'].isnull() == 1, np.nan, mcs['region1_12'])
print(mcs['region1_1'].value_counts(dropna=False))
print(mcs['region1_2'].value_counts(dropna=False))
print(mcs['region1_3'].value_counts(dropna=False))
print(mcs['region1_4'].value_counts(dropna=False))
print(mcs['region1_5'].value_counts(dropna=False))
print(mcs['region1_6'].value_counts(dropna=False))
print(mcs['region1_7'].value_counts(dropna=False))
print(mcs['region1_8'].value_counts(dropna=False))
print(mcs['region1_9'].value_counts(dropna=False))
print(mcs['region1_10'].value_counts(dropna=False))
print(mcs['region1_11'].value_counts(dropna=False))
print(mcs['region1_12'].value_counts(dropna=False))
for i in range(2, 4):
    mcs[f'region{i}_1'] = np.where(mcs[f'region{i}'] == 1, 1, 0)
    mcs[f'region{i}_1'] = np.where(mcs[f'region{i}'].isnull() == 1, np.nan, mcs[f'region{i}_1'])
    mcs[f'region{i}_2'] = np.where(mcs[f'region{i}'] == 2, 1, 0)
    mcs[f'region{i}_2'] = np.where(mcs[f'region{i}'].isnull() == 1, np.nan, mcs[f'region{i}_2'])
    mcs[f'region{i}_3'] = np.where(mcs[f'region{i}'] == 3, 1, 0)
    mcs[f'region{i}_3'] = np.where(mcs[f'region{i}'].isnull() == 1, np.nan, mcs[f'region{i}_3'])
    mcs[f'region{i}_4'] = np.where(mcs[f'region{i}'] == 4, 1, 0)
    mcs[f'region{i}_4'] = np.where(mcs[f'region{i}'].isnull() == 1, np.nan, mcs[f'region{i}_4'])
    mcs[f'region{i}_5'] = np.where(mcs[f'region{i}'] == 5, 1, 0)
    mcs[f'region{i}_5'] = np.where(mcs[f'region{i}'].isnull() == 1, np.nan, mcs[f'region{i}_5'])
    mcs[f'region{i}_6'] = np.where(mcs[f'region{i}'] == 6, 1, 0)
    mcs[f'region{i}_6'] = np.where(mcs[f'region{i}'].isnull() == 1, np.nan, mcs[f'region{i}_6'])
    mcs[f'region{i}_7'] = np.where(mcs[f'region{i}'] == 7, 1, 0)
    mcs[f'region{i}_7'] = np.where(mcs[f'region{i}'].isnull() == 1, np.nan, mcs[f'region{i}_7'])
    mcs[f'region{i}_8'] = np.where(mcs[f'region{i}'] == 8, 1, 0)
    mcs[f'region{i}_8'] = np.where(mcs[f'region{i}'].isnull() == 1, np.nan, mcs[f'region{i}_8'])
    mcs[f'region{i}_9'] = np.where(mcs[f'region{i}'] == 9, 1, 0)
    mcs[f'region{i}_9'] = np.where(mcs[f'region{i}'].isnull() == 1, np.nan, mcs[f'region{i}_9'])
    mcs[f'region{i}_10'] = np.where(mcs[f'region{i}'] == 10, 1, 0)
    mcs[f'region{i}_10'] = np.where(mcs[f'region{i}'].isnull() == 1, np.nan, mcs[f'region{i}_10'])
    mcs[f'region{i}_11'] = np.where(mcs[f'region{i}'] == 11, 1, 0)
    mcs[f'region{i}_11'] = np.where(mcs[f'region{i}'].isnull() == 1, np.nan, mcs[f'region{i}_11'])
    mcs[f'region{i}_12'] = np.where(mcs[f'region{i}'] == 12, 1, 0)
    mcs[f'region{i}_12'] = np.where(mcs[f'region{i}'].isnull() == 1, np.nan, mcs[f'region{i}_12'])
    print(mcs[f'region{i}_1'].value_counts(dropna=False))
    print(mcs[f'region{i}_2'].value_counts(dropna=False))
    print(mcs[f'region{i}_3'].value_counts(dropna=False))
    print(mcs[f'region{i}_4'].value_counts(dropna=False))
    print(mcs[f'region{i}_5'].value_counts(dropna=False))
    print(mcs[f'region{i}_6'].value_counts(dropna=False))
    print(mcs[f'region{i}_7'].value_counts(dropna=False))
    print(mcs[f'region{i}_8'].value_counts(dropna=False))
    print(mcs[f'region{i}_9'].value_counts(dropna=False))
    print(mcs[f'region{i}_10'].value_counts(dropna=False))
    print(mcs[f'region{i}_11'].value_counts(dropna=False))
    print(mcs[f'region{i}_12'].value_counts(dropna=False))


#IMD dummies
for i in range(1, 4):
    mcs[f'imdqnt{i}_1'] = np.where(mcs[f'imdqnt{i}'] == 1, 1, 0)
    mcs[f'imdqnt{i}_1'] = np.where(mcs[f'imdqnt{i}'].isnull(), np.nan, mcs[f'imdqnt{i}_1'])
    mcs[f'imdqnt{i}_2'] = np.where(mcs[f'imdqnt{i}'] == 2, 1, 0)
    mcs[f'imdqnt{i}_2'] = np.where(mcs[f'imdqnt{i}'].isnull(), np.nan, mcs[f'imdqnt{i}_2'])
    mcs[f'imdqnt{i}_3'] = np.where(mcs[f'imdqnt{i}'] == 3, 1, 0)
    mcs[f'imdqnt{i}_3'] = np.where(mcs[f'imdqnt{i}'].isnull(), np.nan, mcs[f'imdqnt{i}_3'])
    mcs[f'imdqnt{i}_4'] = np.where(mcs[f'imdqnt{i}'] == 4, 1, 0)
    mcs[f'imdqnt{i}_4'] = np.where(mcs[f'imdqnt{i}'].isnull(), np.nan, mcs[f'imdqnt{i}_4'])
    mcs[f'imdqnt{i}_5'] = np.where(mcs[f'imdqnt{i}'] == 5, 1, 0)
    mcs[f'imdqnt{i}_5'] = np.where(mcs[f'imdqnt{i}'].isnull(), np.nan, mcs[f'imdqnt{i}_5'])
    print(mcs[f'imdqnt{i}_1'].value_counts(dropna=False))
    print(mcs[f'imdqnt{i}_2'].value_counts(dropna=False))
    print(mcs[f'imdqnt{i}_3'].value_counts(dropna=False))
    print(mcs[f'imdqnt{i}_4'].value_counts(dropna=False))
    print(mcs[f'imdqnt{i}_5'].value_counts(dropna=False))


#Ethnicity dummies
mcs['ethnicity_1'] = np.where(mcs['ethnicity'] == 1, 1, 0)  # White
mcs['ethnicity_2'] = np.where(mcs['ethnicity'] == 2, 1, 0)  # Mixed
mcs['ethnicity_3'] = np.where(mcs['ethnicity'] == 3, 1, 0)  # Indian
mcs['ethnicity_4'] = np.where(mcs['ethnicity'] == 4, 1, 0)  # Pakistani and Bangladeshi
mcs['ethnicity_5'] = np.where(mcs['ethnicity'] == 5, 1, 0)  # Black or Black British
mcs['ethnicity_6'] = np.where(mcs['ethnicity'] == 6, 1, 0)  # Other Ethnic group (inc Chinese,Other) 
# # Comment out below lines to assume missing ethnicity same as omitted category
# mcs['ethnicity_1'] = np.where(mcs['ethnicity'].isnull(), np.nan, mcs['ethnicity_1'])
# mcs['ethnicity_2'] = np.where(mcs['ethnicity'].isnull(), np.nan, mcs['ethnicity_2'])
# mcs['ethnicity_3'] = np.where(mcs['ethnicity'].isnull(), np.nan, mcs['ethnicity_3'])
# mcs['ethnicity_4'] = np.where(mcs['ethnicity'].isnull(), np.nan, mcs['ethnicity_4'])
# mcs['ethnicity_5'] = np.where(mcs['ethnicity'].isnull(), np.nan, mcs['ethnicity_5'])
# mcs['ethnicity_6'] = np.where(mcs['ethnicity'].isnull(), np.nan, mcs['ethnicity_6'])
# #Uncomment line below to include missing/non-response as other
# mcs['ethnicity_6'] = np.where(mcs['ethnicity'].isna(), 1, mcs['ethnicity_6'])
print(mcs['ethnicity_1'].value_counts(dropna=False))
print(mcs['ethnicity_2'].value_counts(dropna=False))
print(mcs['ethnicity_3'].value_counts(dropna=False))
print(mcs['ethnicity_4'].value_counts(dropna=False))
print(mcs['ethnicity_5'].value_counts(dropna=False))
print(mcs['ethnicity_6'].value_counts(dropna=False))
                              
x = mcs.loc[mcs["male"] == 1, "sdqemotion7"]
y = mcs.loc[mcs["male"] == 0, "sdqemotion7"]

bins = np.linspace(0, 10, 10)

# plt.hist(mcs.loc[mcs["male"] == 1, "sdqemotion7"], bins, alpha=0.5, label='Male')
# plt.hist(mcs.loc[mcs["male"] == 0, "sdqemotion7"], bins, alpha=0.5, label='Female')
plt.hist([x, y], bins, label=['Male', 'Female'])
plt.legend(loc='upper right')
plt.show()

x = mcs.loc[mcs["male"] == 1, "sdqexternal7"]
y = mcs.loc[mcs["male"] == 0, "sdqexternal7"]

bins = np.linspace(0, 20, 20)

# plt.hist(mcs.loc[mcs["male"] == 1, "sdqexternal7"], bins, alpha=0.5, label='Male')
# plt.hist(mcs.loc[mcs["male"] == 0, "sdqexternal7"], bins, alpha=0.5, label='Female')
plt.hist([x, y], bins, label=['Male', 'Female'])
plt.legend(loc='upper right')
plt.show()

x = mcs.loc[mcs["male"] == 1, "swemwbs7"]
y = mcs.loc[mcs["male"] == 0, "swemwbs7"]

bins = np.linspace(0, 35, 35)

# plt.hist(mcs.loc[mcs["male"] == 1, "sdqexternal7"], bins, alpha=0.5, label='Male')
# plt.hist(mcs.loc[mcs["male"] == 0, "sdqexternal7"], bins, alpha=0.5, label='Female')
plt.hist([x, y], bins, label=['Male', 'Female'])
plt.legend(loc='upper right')
plt.show()          

mcs.to_csv(os.path.join(lfsm, 'mcs1.csv'))                     
del data

# #Minimum data set for Paul

# print(mcs.columns.tolist())
# lst = ['mcsid', 
#        'cmagem1', 'country1', 'region1', 'bthwt1','income1', 'poverty1', 'imdqnt1',
#        'smokepreg', 'drinkpreg', 'brstfdevr', 'malm', 'malm_clin', 
#        'htrnt1', 'meduc1', 'hheduc1', 'hhdis1', 'nplfp1', 'snglprnt1', 'numppl1', 'numchld1', 'thrswrkdm1', 'thrswrkdhh1',
#        'weight1', 'hosp1',
       
#        'cmagem2', 'country2', 'region2', 'bthwt2', 'income2', 'poverty2', 'imdqnt2', 
#        'htrnt2', 'meduc2', 'hheduc2', 'hhdis2', 'nplfp2', 'snglprnt2', 'numppl2', 'numchld2', 'thrswrkdm2', 'thrswrkdhh2',
#        'height2', 'weight2', 'bmi2', 'obesity2', 'lsc2', 'alc2', 'hosp2',
#        'zcog2',
#        'sdqconduct2', 'sdqemotion2', 'sdqpeer2', 'sdqprosoc2', 'sdqhyper2', 'sdqinternal2', 'sdqexternal2', 'sdqimpact2', 
       
#        'cmagem3', 'country3', 'region3', 'income3', 'poverty3', 'imdqnt3',  
#        'htrnt3', 'meduc3', 'hheduc3', 'hhdis3', 'nplfp3', 'snglprnt3', 'numppl3', 'numchld3', 'thrswrkdm3', 'thrswrkdhh3', 
#        'height3', 'weight3', 'bmi3', 'obesity3', 'lsc3', 'alc3', 'hosp3', 
#        'zcog3', 
#        'sdqconduct3', 'sdqemotion3', 'sdqpeer3', 'sdqprosoc3', 'sdqhyper3', 'sdqinternal3', 'sdqexternal3', 'sdqimpact3', 
       
#        'cmagem4', 'country4', 'region4', 'income4', 'poverty4', 'imdqnt4',
#        'htrnt4', 'meduc4', 'hheduc4', 'hhdis4', 'nplfp4', 'snglprnt4', 'numppl4', 'numchld4', 'thrswrkdm4', 'thrswrkdhh4', 
#        'height4', 'weight4', 'bmi4', 'obesity4', 'lsc4', 'alc4', 'hosp4', 
#        'zcog4', 'sen4',
#        'sdqconduct4', 'sdqemotion4', 'sdqpeer4', 'sdqprosoc4', 'sdqhyper4', 'sdqinternal4', 'sdqexternal4', 'sdqimpact4',
       
#        'cmagem5', 'country5', 'region5', 'income5', 'poverty5', 'imdqnt5', 
#        'htrnt5', 'meduc5', 'hheduc5', 'hhdis5', 'nplfp5', 'snglprnt5', 'numppl5', 'numchld5', 'thrswrkdm5', 'thrswrkdhh5',
#        'height5', 'weight5', 'bmi5', 'obesity5', 'lsc5', 'alc5', 'hosp5',
#        'zcog5', 'excl5', 'truancy5', 'sen5', 
#        'sdqconduct5', 'sdqemotion5', 'sdqpeer5', 'sdqprosoc5', 'sdqhyper5', 'sdqinternal5', 'sdqexternal5',
       
#        'cmagem6', 'country6', 'region6', 'income6', 'poverty6', 'imdqnt6', 
#        'htrnt6', 'meduc6', 'hheduc6', 'hhdis6', 'nplfp6', 'snglprnt6', 'numppl6', 'numchld6', 'thrswrkdm6', 'thrswrkdhh6', 
#        'height6', 'weight6', 'bmi6', 'obesity6', 'lsc6', 'alc6', 'hosp6', 'smkreg6',
#        'zcog6', 'excl6', 'truancy6', 'regtruancy6', 'sen6', 
#        'sdqconduct6', 'sdqemotion6', 'sdqpeer6', 'sdqprosoc6', 'sdqhyper6', 'sdqinternal6', 'sdqexternal6',
       
#        'cmagem7', 'country7', 'region7',
#        'snglprnt7', 'numppl7', 'numchld7', 'thrswrkdm7', 'thrswrkdhh7', 'thrswrkdhh7', 'gpeduc7',  
#        'height7', 'weight7', 'bmi7', 'obesity7', 'lsc7', 'alc7', 'hosp7', 'smkreg7', 'prfrhlth7', 
#        'zcog7', 'bdgcse_me', 
#        'sdqconduct7', 'sdqemotion7', 'sdqpeer7', 'sdqprosoc7', 'sdqhyper7', 'sdqinternal7', 'sdqexternal7', 
#        'sdqconductsr7', 'sdqemotionsr7', 'sdqpeersr7', 'sdqprosocsr7', 'sdqhypersr7', 'sdqinternalsr7', 'sdqexternalsr7',
#        'kessler7', 'bfpopen7', 'bfpcons7', 'bfpextr7', 'bfpagre7', 'bfpneur7', 
#        'swemwbs7', 
       
#        'owt_cs', 'wt_cs1', 'wt_cs2', 'wt_cs3', 'wt_cs4', 'wt_cs5', 'wt_cs6', 'wt_cs7', 
#        'owt_uk', 'wt_uk1', 'wt_uk2', 'wt_uk3', 'wt_uk4', 'wt_uk5', 'wt_uk6', 'wt_uk7', 
       
#        'male', 'ethnicity', 'country', 'region', 
#        'teenbrth', 'pretrm', 'bthwt_c', 'lwht3', 'deldev3', 
#        'income123', 'lincome123', 'income123_q1', 'income123_q2', 'income123_q3', 'income123_q4', 'income123_q5', 
#        'country1_1', 'country1_2', 'country1_3', 'country1_4', 
#        'region1_1', 'region1_2', 'region1_3', 'region1_4', 'region1_5', 'region1_6', 'region1_7', 'region1_8', 'region1_9', 'region1_10', 'region1_11', 'region1_12', 
#        'imdqnt1_1', 'imdqnt1_2', 'imdqnt1_3', 'imdqnt1_4', 'imdqnt1_5', 
#        'ethnicity_1', 'ethnicity_2', 'ethnicity_3', 'ethnicity_4', 'ethnicity_5', 'ethnicity_6']

# lstm = ['mcsid', 
#        'cmagem1', 'country1', 'region1', 'income1', 'poverty1', 'imdqnt1',
#        'smokepreg', 'drinkpreg', 'brstfdevr', 'malm', 'malm_clin', 
#        'htrnt1', 'meduc1', 'hheduc1', 'hhdis1', 'nplfp1', 'snglprnt1', 'numppl1', 'numchld1', 'thrswrkdm1', 'thrswrkdhh1',
#        'hosp1',
#        'cmagem2', 'country2', 'region2', 'income2', 'poverty2', 'imdqnt2', 
#        'htrnt2', 'meduc2', 'hheduc2', 'hhdis2', 'nplfp2', 'snglprnt2', 'numppl2', 'numchld2', 'thrswrkdm2', 'thrswrkdhh2',
#        'obesity2', 'alc2', 'hosp2',
#        'zcog2',
#        'sdqconduct2', 'sdqemotion2', 'sdqpeer2', 'sdqprosoc2', 'sdqhyper2', 'sdqinternal2', 'sdqexternal2', 'sdqimpact2', 
#        'cmagem3', 'country3', 'region3', 'income3', 'poverty3', 'imdqnt3',  
#        'htrnt3', 'meduc3', 'hheduc3', 'hhdis3', 'nplfp3', 'snglprnt3', 'numppl3', 'numchld3', 'thrswrkdm3', 'thrswrkdhh3', 
#        'obesity3', 'alc3', 'hosp3', 
#        'zcog3', 
#        'sdqconduct3', 'sdqemotion3', 'sdqpeer3', 'sdqprosoc3', 'sdqhyper3', 'sdqinternal3', 'sdqexternal3', 'sdqimpact3', 
#        'cmagem4', 'country4', 'region4', 'income4', 'poverty4', 'imdqnt4',
#        'htrnt4', 'meduc4', 'hheduc4', 'hhdis4', 'nplfp4', 'snglprnt4', 'numppl4', 'numchld4', 'thrswrkdm4', 'thrswrkdhh4', 
#        'obesity4', 'alc4', 'hosp4', 
#        'zcog4', 'sen4',
#        'sdqconduct4', 'sdqemotion4', 'sdqpeer4', 'sdqprosoc4', 'sdqhyper4', 'sdqinternal4', 'sdqexternal4', 'sdqimpact4',
#        'cmagem5', 'country5', 'region5', 'income5', 'poverty5', 'imdqnt5', 
#        'htrnt5', 'meduc5', 'hheduc5', 'hhdis5', 'nplfp5', 'snglprnt5', 'numppl5', 'numchld5', 'thrswrkdm5', 'thrswrkdhh5',
#        'obesity5', 'alc5', 'hosp5',
#        'zcog5', 'excl5', 'truancy5', 'sen5', 
#        'sdqconduct5', 'sdqemotion5', 'sdqpeer5', 'sdqprosoc5', 'sdqhyper5', 'sdqinternal5', 'sdqexternal5',
#        'cmagem6', 'country6', 'region6', 'income6', 'poverty6', 'imdqnt6', 
#        'htrnt6', 'meduc6', 'hheduc6', 'hhdis6', 'nplfp6', 'snglprnt6', 'numppl6', 'numchld6', 'thrswrkdm6', 'thrswrkdhh6',
#        'obesity6', 'alc6', 'hosp6', 
#        'zcog6', 'excl6', 'truancy6', 'regtruancy6', 'sen6', 
#        'sdqconduct6', 'sdqemotion6', 'sdqpeer6', 'sdqprosoc6', 'sdqhyper6', 'sdqinternal6', 'sdqexternal6',
#        'cmagem7', 'country7', 'region7',
#        'snglprnt7', 'numppl7', 'numchld7', 'thrswrkdm7', 'thrswrkdhh7', 'gpeduc7', 
#        'obesity7', 'alc7', 'hosp7', 'smkreg7', 'prfrhlth7', 
#        'zcog7', 'bdgcse_me', 
#        'sdqconduct7', 'sdqemotion7', 'sdqpeer7', 'sdqprosoc7', 'sdqhyper7', 'sdqinternal7', 'sdqexternal7', 
#        'sdqconductsr7', 'sdqemotionsr7', 'sdqpeersr7', 'sdqprosocsr7', 'sdqhypersr7', 'sdqinternalsr7', 'sdqexternalsr7',
#        'kessler7',
#        'owt_uk', 'wt_uk1', 'wt_uk2', 'wt_uk3', 'wt_uk4', 'wt_uk5', 'wt_uk6', 'wt_uk7', 
#        'male', 'ethnicity', 'country', 'region', 
#        'teenbrth', 'pretrm', 'bthwt_c', 'lwht3', 'deldev3', 
#        'income123', 'lincome123', 'income123_q1', 'income123_q2', 'income123_q3', 'income123_q4', 'income123_q5', 
#        'country1_1', 'country1_2', 'country1_3', 'country1_4', 
#        'region1_1', 'region1_2', 'region1_3', 'region1_4', 'region1_5', 'region1_6', 'region1_7', 'region1_8', 'region1_9', 'region1_10', 'region1_11', 'region1_12', 
#        'imdqnt1_1', 'imdqnt1_2', 'imdqnt1_3', 'imdqnt1_4', 'imdqnt1_5', 
#        'ethnicity_1', 'ethnicity_2', 'ethnicity_3', 'ethnicity_4', 'ethnicity_5', 'ethnicity_6']

# mcsm = mcs[lstm]
# mcsm.to_stata(os.path.join(lfsm, 'mcsmin.dta'))  


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
# descf = sms.stats.descriptivestats.describe(mcs[varl], stats=stats, numeric=True, percentiles=(5, 25, 50, 75, 95) ).T
# print(descf)

# fig, ax = plt.subplots(figsize=(12, 6))
# sns.kdeplot(data=mcs['lifesat5'],
#             color='crimson', label='SDQ Internalising', fill=True, ax=ax)
# sns.kdeplot(data=mcs['lifesatwb5'],
#             color='limegreen', label='General wellbeing', fill=True, ax=ax)
# ax.legend()
# plt.xlabel('Life Satisfaction at age 11')
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(figsize=(12, 6))
# sns.kdeplot(data=mcs['lifesat6'],
#             color='crimson', label='SDQ Internalising', fill=True, ax=ax)
# sns.kdeplot(data=mcs['lifesatwb6'],
#             color='limegreen', label='General wellbeing', fill=True, ax=ax)
# ax.legend()
# plt.xlabel('Life Satisfaction at age 14')
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(figsize=(12, 6))
# sns.kdeplot(data=mcs['lifesat7'],
#             color='crimson', label='SDQ Internalising', fill=True, ax=ax)
# sns.kdeplot(data=mcs['lifesatsr7'],
#             color='blue', label='Self reported SDQ', fill=True, ax=ax)
# sns.kdeplot(data=mcs['lifesatwb7'],
#             color='limegreen', label='SWEMWBS', fill=True, ax=ax)
# ax.legend()
# plt.xlabel('Life Satisfaction at age 17')
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(figsize=(12, 6))
# sns.kdeplot(data=mcs['sdqpeersr7'],
#             color='crimson', label='SDQ Peer', fill=True, ax=ax)
# sns.kdeplot(data=mcs['sdqemotionsr7'],
#             color='blue', label='SDQ Emotion', fill=True, ax=ax)
# sns.kdeplot(data=mcs['sdqinternalsr7'],
#             color='limegreen', label='SDQ Internalising', fill=True, ax=ax)
# ax.legend()
# plt.xlabel('Life Satisfaction at age 17')
# plt.tight_layout()
# plt.show()