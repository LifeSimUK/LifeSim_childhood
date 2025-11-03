

global path "E:\Shrathinth\LifeSim\Python Code\lifesim2-main\data\heightchart"

clear all
set maxvar 120000
set more off

import excel "E:\Shrathinth\LifeSim\Python Code\lifesim2-main\data\heightchart\uk90htchrtedt.xlsx", sheet("Sheet1") firstrow clear

keep male months sd2 sd15 p10 p90

save "E:\Shrathinth\LifeSim\Python Code\lifesim2-main\data\heightchart\hmmf.dta", replace