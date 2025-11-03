import pandas as pd
import numpy as np
import re

class Person1:
    """
    A class used to represent and simulate an MCS individual from birth to age 17

    ...

    Attributes
    ----------
    mcsid: str
        contains the unique MCSID of the individual to allow linking back to the raw data
    x : Series
        Contains all the person specific characteristics
    betas : DataFrame
        Contains the regression coefficients for the lifecourse trajectory equations
    probs : Series
        Contains random draws to compare the results of the binary equations 
        with in order to decide whether a binary outcome occurred
    history : DataFrame
        Keeps track of the lifecourse trajectory by capturing the evolution of key variables
    social_care : bool
        Records whether the person is currently in social care

    Methods
    -------
    simulate_sweep(sweep_num, sweep_age, sweep_age_prev)
        Simulates an MCS sweep updating person characteristics x and adding to history
    simulate_all_sweeps()
        Simulates all 7 sweeps of MCS taking an individual from birth to age 17
    """


    def __init__(self, x, betas, res, probs, co, bo, sw):
        """
        Parameters
        ----------
        x : Series
            Contains all person specific characteristics including placeholder
            for characteristics that will be simulated for MCS sweeps 2-7
        betas : DataFrame
            Contains the regression coefficients for the lifecourse trajectory 
            equations with rows representing independent variables coefficients 
            and columns representing dependent (outcome) variables
        probs : Series
            Contains random draws to compare the results of the binary equations 
            with in order to decide whether a binary outcome occurred
        """
        self.mcsid = x.mcsid
        #self.wt = x.wt_uk2
        #self.x = x.drop('mcsid')
        self.betas = betas
        self.res = res
        self.probs = probs.reset_index() 
        self.co = co
        self.bo = bo
        self.sw = sw
        self.cno = ["zcog2", "zcog3", "zcog4", "zcog5", "zcog6", "zcog7",
                    "zbasnv2", "zbsrar2", "zbasnv3", "zbaspc3", "zbasps3", "zbaswr4", "zbaspc4", "znferpm4",
                    "lifesat2", "lifesat3", "lifesat4", "lifesat5", "lifesat6", "lifesat7", 
                    "zlifesat2", "zlifesat3", "zlifesat4", "zlifesat5", "zlifesat6", "zlifesat7"]
        self.cto = [x for x in self.co if x not in self.cno]
        ln = 8
        #Align matrices for multiplication (Based on order of coefficients from regression)
        self.x = x[list(self.betas.index.values)]
        
        # Identify the starting and ending wave for risk factor
        if self.sw < 2:
            sn = 2
        else:
            sn = self.sw+1
        ln = 8    
        #####################################################
        # calculate continuous outcome equations
        #####################################################
        print(self.x.columns[self.x.columns.str.startswith('')].tolist()) 
        print(self.betas.index[self.betas.index.str.startswith('')].tolist())   
        # compute dot product of betas with x
        #import pdb; pdb.set_trace()
        continuous_outcomes = np.dot(self.betas[self.co].values.T, self.x.values.T)
        resc = self.res[self.co].values.T
        #import pdb; pdb.set_trace()
        continuous_outcomes = continuous_outcomes + resc
        #continuous_outcomes = continuous_outcomes
        continuous_outcomes = pd.DataFrame(continuous_outcomes, index=self.co).T
        
        def exp(x):
            return np.exp(x)
        continuous_outcomes[self.cto] = continuous_outcomes[self.cto].applymap(exp)

        sdq1 = ['sdqconduct', 'sdqemotion', 'sdqpeer', 'sdqhyper', 'sdqprosoc']
        sdq2 = ['sdqinternal', 'sdqexternal', 'internal']
        
        for c in range(sn, ln):
            for score in sdq1:
                column_name = score + str(c)
                if column_name in self.cto:
                    continuous_outcomes[column_name] = np.clip(continuous_outcomes[column_name], a_max=10, a_min=0)
             
        for c in range(sn, ln):
           for score in sdq2:
               column_name = score + str(c)
               if column_name in self.cto:
                      continuous_outcomes[column_name] = np.clip(continuous_outcomes[column_name], a_max=20, a_min=0)

        #continuous_outcomes['welbgrid5'] = np.clip(continuous_outcomes['welbgrid5'], a_max=26, a_min=0)        
        # continuous_outcomes['welbgrid6'] = np.clip(continuous_outcomes['welbgrid6'], a_max=26, a_min=0)
        continuous_outcomes['kessler7'] = np.clip(continuous_outcomes['kessler7'], a_max=24, a_min=0)
        #continuous_outcomes['swemwbs7'] = np.clip(continuous_outcomes['swemwbs7'], a_max=28, a_min=0)

        self.c = continuous_outcomes
        
        #####################################################
        # calculate binary outcome equations
        #####################################################
                

        # compute dot product of betas with x to get the log odds ration
        xb = np.dot(self.betas[self.bo].values.T, self.x.values.T).astype(float)
        resb = self.res[self.bo].values.T
      
        # compute threshold probabilities thershold_p using the sigmoid function and the odds ratio
        threshold_p = np.transpose(1 / (1 + np.exp(-xb-resb)))
        # threshold_p = np.transpose(1 / (1 + np.exp(-xb)))
        threshold_p = pd.DataFrame(threshold_p, columns=self.bo)
        
        #Adding Conduct disorder indicator
        ac = [f"condis{i}" for i in range(sn, ln)]
        bop = bo + ac
        def cdprob(row):
            x_i = []
            for i in range(sn, ln):
                y_i_col = f'sdqconduct{i}'
                y_i = row[y_i_col]
                if y_i < 4:
                    x_i.append(0.06)
                elif  y_i >= 4:
                    x_i.append(0.31)
                #elif y_i >= 5:
                #    x_i.append(0.61)
                else:
                    x_i.append(np.nan)
            return x_i
        print("Shape of threshold_p:", threshold_p.shape)
        print("Shape of apply result:", continuous_outcomes.apply(cdprob, axis=1, result_type='expand').shape)

        threshold_p[ac] = continuous_outcomes.apply(cdprob, axis=1, result_type='expand')




                
        # create boolean mask of outcomes based on probability thresholds and random probabilities from probs
        # binary_outcomes = (self.probs[bop] < threshold_p).astype(int)
        # # create boolean mask of outcomes based on probability thresholds and random probabilities from probs
        #import pdb; pdb.set_trace()
        binary_outcomes = np.where((probs < threshold_p) & ~probs.isna() & ~threshold_p.isna(), 1, 0)
        
        cmm = (self.probs[bop] < threshold_p) & ~self.probs[bop].isna() & ~threshold_p.isna()
        mm = self.probs[bop].isna() | threshold_p.isna()
        
        # Create binary outcomes based on masks
        binary_outcomes = np.zeros_like(cmm, dtype=float)  # Initialize with 0.0
        binary_outcomes[cmm] = 1.0  # Set values to 1.0 where the condition is met
        binary_outcomes[mm] = np.nan  # Set missing values to NaN
        binary_outcomes = pd.DataFrame(binary_outcomes, columns=self.probs[bop].columns)
        
        self.b = binary_outcomes
        
        #self.mcsid = self.mcsid.reset_index()
        #self.wt = self.wt.reset_index()
        # add continuous outcomes in x
        self.history = continuous_outcomes.join(binary_outcomes) 
        self.history = self.history.join(self.mcsid)
        #self.history = self.history.join(self.wt)
    

# # #################### Simulation redo to check dataframes
#         #mcsid = mcsm.mcsid
#         betas = beta
#         tres = sim_res[0]
#         probs = sim_probs[0]
#         co = coutcomes
#         bo = boutcomes
        
        
#         #Align matrices for multiplication (Based on order of coefficients from regression)
#         x = mcsm[list(beta.index.values)]
        

#         #####################################################
#         # calculate continuous outcome equations
#         #####################################################
             
#         # compute dot product of betas with x
#         continuous_outcomes = np.dot(betas[co].values.T, x.values.T)
#         resc = tres[co].values.T
#         continuous_outcomes = continuous_outcomes + resc
#         continuous_outcomes = pd.DataFrame(continuous_outcomes, index=co).T

#         for c in range(3, 8):
#             continuous_outcomes['con' + str(c)] = np.clip(continuous_outcomes['con' + str(c)], a_max=10, a_min=0)
#             continuous_outcomes['emo' + str(c)] = np.clip(continuous_outcomes['emo' + str(c)], a_max=10, a_min=0)
#             continuous_outcomes['internal' + str(c)] = np.clip(continuous_outcomes['internal' + str(c)], a_max=20, a_min=0)

#         for c in range(3, 5):
#             continuous_outcomes['sdqimp' + str(c)] = np.clip(continuous_outcomes['sdqimp' + str(c)], a_max=3, a_min=0)
#         for c in range(5, 8):
#             continuous_outcomes['sdqimp' + str(c)] = np.clip(continuous_outcomes['sdqimp4'], a_max=3, a_min=0)

#         c = continuous_outcomes
        
#         #####################################################
#         # calculate binary outcome equations
#         #####################################################
                

#         # compute dot product of betas with x to get the log odds ration
#         xb = np.dot(betas[bo].values.T, x.values.T).astype(float)
#         resb = tres[bo].values.T
                        
#         # compute threshold probabilities thershold_p using the sigmoid function and the odds ratio
#         threshold_p = np.transpose(1 / (1 + np.exp(-xb-resb)))
#         threshold_p = pd.DataFrame(threshold_p, columns=bo)
#         #Adding Conduct disorder indicator
#         ac = ["condis3", "condis4", "condis5", "condis6", "condis7"]
#         bop = bo + ac
#         def cdprob(row):
#             x_i = []
#             for i in range(3, 8):
#                 y_i_col = f'con{i}'
#                 y_i = row[y_i_col]
#                 if y_i < 4:
#                     x_i.append(0.06)
#                 elif  y_i >= 4:
#                     x_i.append(0.31)
#                 elif y_i >= 5:
#                     x_i.append(0.61)
#                 else:
#                     x_i.append(np.NaN)
#             return x_i
#         threshold_p[ac] = continuous_outcomes.apply(cdprob, axis=1, result_type='expand')


                
#         # create boolean mask of outcomes based on probability thresholds and random probabilities from probs
#         binary_outcomes = np.where((probs < threshold_p) & ~probs.isna() & ~threshold_p.isna(), 1, 0)
        
        
#         condition_met_mask = (probs[bop] < threshold_p) & ~probs[bop].isna() & ~threshold_p.isna()
#         missing_mask = probs[bop].isna() | threshold_p.isna()

#         # Create binary outcomes based on masks
#         binary_outcomes = np.zeros_like(condition_met_mask, dtype=float)  # Initialize with 0.0
#         binary_outcomes[condition_met_mask] = 1.0  # Set values to 1.0 where the condition is met
#         binary_outcomes[missing_mask] = np.nan  # Set missing values to NaN
#         binary_outcomes = pd.DataFrame(binary_outcomes, columns=probs[bop].columns)
        
#         b = binary_outcomes
        
#         # add continuous outcomes in x
#         historytmp = continuous_outcomes.join(binary_outcomes) 