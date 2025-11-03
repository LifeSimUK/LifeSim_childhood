########## Code to run regressions on multiply imputed data and apply Rubins rules to extract coefficients, std errors and residual distributions ##########

setwd("//lifesim2/IJM2026")


library(dplyr)
library(haven)
library(purrr)
library(tidyverse)
library(gtools)
library(reticulate)
library(json)

library(MASS)
library(estimatr)
library(logisticRR)
library(marginaleffects)
library(mfx)

library(huxtable)
library(flextable)
library(xtable)

library(foreach)
library(parallel)
library(doParallel)

library(mice)
library(childsds)

########## Functions for calculations
# Define the path to the folder containing your R scripts
folder_path <- "r_functions"

# Get a list of all R script files in the folder
r_func <- list.files(path = folder_path, pattern = "\\.R$|\\.r$", full.names = TRUE)

# Source each file
for (file in r_func) {
  source(file)
}

# variable dictionary
d <- rjson::fromJSON(file = "varlabel.json")
var_dict <- unlist(d)

# # Parallel processing
# num_cores <- detectCores()
# cl <- makeCluster(6) #Choose number of cores to use
# registerDoParallel(cl)

###Read Multiple imputed data from stata
mcs <- read_stata("mcs1ic.dta")
#Rename variables that may cause breakdown in regressions
table(mcs[mcs$imputn > 0, "bdgcseme7"])

#Number of imputations
n_imp <- max(mcs$imputn)
# #Create a dataset with only the imputed data
# mcsi <- mcs %>% 
#   filter(imputn > 0)

#Output tables
tb <- "NA"
# tb <- "dydx"
# tb <- "rrr"

# Outcome groups
#From 9 months/ birth
cog1 <- c("zcog2", "zcog3", "zcog4", "zcog5", "zcog6", "zcog7", "zbasnv2", "zbsrar2", "zbasnv3", "zbaspc3", "zbasps3", "zbaswr4", "zbaspc4", "znferpm4")
# zls <- c("zlifesat2", "zlifesat3", "zlifesat4", "zlifesat5", "zlifesat6", "zlifesat7")
lfs1<- c("lifesat2", "lifesat3", "lifesat4", "lifesat5", "lifesat6", "lifesat7")

con1<- c("sdqconduct2", "sdqconduct3", "sdqconduct4", "sdqconduct5", "sdqconduct6", "sdqconduct7", "sdqconductsr7")
emo1<- c("sdqemotion2", "sdqemotion3", "sdqemotion4", "sdqemotion5", "sdqemotion6", "sdqemotion7", "sdqemotionsr7")  
per1<- c("sdqpeer2", "sdqpeer3", "sdqpeer4", "sdqpeer5", "sdqpeer6", "sdqpeer7", "sdqpeersr7") 
prs1<- c("sdqprosoc2", "sdqprosoc3", "sdqprosoc4", "sdqprosoc5", "sdqprosoc6", "sdqprosoc7", "sdqprosocsr7") 
hyp1<- c("sdqhyper2", "sdqhyper3", "sdqhyper4", "sdqhyper5", "sdqhyper6", "sdqhyper7", "sdqhypersr7") 
int1<- c("sdqinternal2", "sdqinternal3", "sdqinternal4", "sdqinternal5", "sdqinternal6", "sdqinternal7", "sdqinternalsr7") 
ext1<- c("sdqexternal2", "sdqexternal3", "sdqexternal4", "sdqexternal5", "sdqexternal6", "sdqexternal7", "sdqexternalsr7")
wbs4<- c("genwelb5", "welbgrid5", "genwelb6", "welbgrid6", "swemwbs7")
kes6<- c("kessler7")

hsp1<- c("hosp2", "hosp3", "hosp4", "hosp5", "hosp6", "hosp7")
alc1<- c("alc2", "alc3", "alc4", "alc5", "alc6", "alc7")
obs1<- c("obese2", "obese3", "obese4", "obese5", "obese6", "obese7")
sen3<- c("sen4", "sen5", "sen6")
tru4<- c("truancy5", "truancy6")
exc4<- c("excl5", "excl6")
bgc6<- c("bdgcseme7")
smk5<- c("smkreg6", "smkreg7")
obs6<- c("obesity7")
hpf6<- c("prfrhlth7")

out1 <- c(cog1, lfs1, con1, emo1, per1, prs1, hyp1, int1, ext1, kes6, hsp1, alc1, obs6, sen3, tru4, exc4, bgc6, smk5, hpf6)
cout1 <- c(cog1, lfs1)
sout1 <- c(con1, emo1, per1, prs1, hyp1, int1, ext1, kes6)
bout1 <- c(hsp1, alc1, obs6, sen3, tru4, exc4, bgc6, smk5, hpf6)

out0 <- c(cog1, lfs1, con1, emo1, per1, prs1, hyp1, int1, ext1, kes6, hsp1, alc1, obs6, sen3, tru4, exc4, bgc6, smk5, hpf6)
cout0 <- c(cog1, lfs1)
sout0 <- c(con1, emo1, per1, prs1, hyp1, int1, ext1, kes6) 
bout0 <- c(hsp1, alc1, obs6, sen3, tru4, exc4, bgc6, smk5, hpf6)


#From age 3
cog2<- c("zcog3", "zcog4", "zcog5", "zcog6", "zcog7", "zbasnv3", "zbaspc3", "zbasps3", "zbaswr4", "zbaspc4", "znferpm4")
# zls3<- c("zlifesat4", "zlifesat5", "zlifesat6", "zlifesat7")
lfs2<- c("lifesat3", "lifesat4", "lifesat5", "lifesat6", "lifesat7")

con2<- c("sdqconduct3", "sdqconduct4", "sdqconduct5", "sdqconduct6", "sdqconduct7", "sdqconductsr7")
emo2<- c("sdqemotion3", "sdqemotion4", "sdqemotion5", "sdqemotion6", "sdqemotion7", "sdqemotionsr7")  
per2<- c("sdqpeer3", "sdqpeer4", "sdqpeer5", "sdqpeer6", "sdqpeer7", "sdqpeersr7") 
prs2<- c("sdqprosoc3", "sdqprosoc4", "sdqprosoc5", "sdqprosoc6", "sdqprosoc7", "sdqprosocsr7") 
hyp2<- c("sdqhyper3", "sdqhyper4", "sdqhyper5", "sdqhyper6", "sdqhyper7", "sdqhypersr7") 
int2<- c("sdqinternal3", "sdqinternal4", "sdqinternal5", "sdqinternal6", "sdqinternal7", "sdqinternalsr7") 
ext2<- c("sdqexternal3", "sdqexternal4", "sdqexternal5", "sdqexternal6", "sdqexternal7", "sdqexternalsr7")

hsp2<- c("hosp3", "hosp4", "hosp5", "hosp6", "hosp7")
alc2<- c("alc3", "alc4", "alc5", "alc6", "alc7")
obs2<- c("obese3", "obese4", "obese5", "obese6", "obese7")
# ccc3<- c("pccc4")

out2 <- c(cog2, lfs2, con2, emo2, per2, prs2, hyp2, int2, ext2, kes6, hsp2, alc2, obs6, sen3, tru4, exc4, bgc6, smk5, hpf6)
cout2 <- c(cog2, lfs2)
sout2 <- c(con2, emo2, per2, prs2, hyp2, int2, ext2, kes6)
bout2 <- c(hsp2, alc2, obs6, sen3, tru4, exc4, bgc6, smk5, hpf6)


#From age 5
cog3<- c("zcog4", "zcog5", "zcog6", "zcog7", "zbaswr4", "zbaspc4", "znferpm4")
# zls3<- c("zlifesat4", "zlifesat5", "zlifesat6", "zlifesat7")
lfs3<- c("lifesat4", "lifesat5", "lifesat6", "lifesat7")

con3<- c("sdqconduct4", "sdqconduct5", "sdqconduct6", "sdqconduct7", "sdqconductsr7")
emo3<- c("sdqemotion4", "sdqemotion5", "sdqemotion6", "sdqemotion7", "sdqemotionsr7")
per3<- c("sdqpeer4", "sdqpeer5", "sdqpeer6", "sdqpeer7", "sdqpeersr7") 
prs3<- c("sdqprosoc4", "sdqprosoc5", "sdqprosoc6", "sdqprosoc7", "sdqprosocsr7") 
hyp3<- c("sdqhyper4", "sdqhyper5", "sdqhyper6", "sdqhyper7", "sdqhypersr7") 
int3<- c("sdqinternal4", "sdqinternal5", "sdqinternal6", "sdqinternal7", "sdqinternalsr7") 
ext3<- c("sdqexternal4", "sdqexternal5", "sdqexternal6", "sdqexternal7", "sdqexternalsr7")

hsp3<- c("hosp4", "hosp5", "hosp6", "hosp7")
alc3<- c("alc4", "alc5", "alc6", "alc7")
obs3<- c("obesity4", "obesity5", "obesity6", "obesity7")
# ccc3<- c("pccc4")

out3 <- c(cog3, lfs3, con3, emo3, per3, prs3, hyp3, int3, ext3, kes6, hsp3, alc3, obs6, sen3, tru4, exc4, bgc6, smk5, hpf6)
cout3 <- c(cog3, lfs3)
sout3 <- c(con3, emo3, per3, prs3, hyp3, int3, ext3, kes6)
bout3 <- c(hsp3, alc3, obs6, sen3, tru4, exc4, bgc6, smk5, hpf6)


##### Confounder lists
### Geography
# Country
cntry1 <- c("country1_2", "country1_3", "country1_4")
cntry2 <- c("country2_2", "country2_3", "country2_4")
cntry2 <- c("country3_2", "country3_3", "country3_4")
# Region
rgn1 <- c("region1_1", "region1_2", "region1_3", "region1_4", "region1_5", "region1_6", "region1_8", "region1_9")
rgn2 <- c("region2_1", "region2_2", "region2_3", "region2_4", "region2_5", "region2_6", "region2_8", "region2_9")
rgn3 <- c("region3_1", "region3_2", "region3_3", "region3_4", "region3_5", "region3_6", "region3_8", "region3_9")
# IMD quintile
imd1 <- c("imdqnt1_2", "imdqnt1_3", "imdqnt1_4", "imdqnt1_5")
imd2 <- c("imdqnt2_2", "imdqnt2_3", "imdqnt2_4", "imdqnt2_5")
imd3 <- c("imdqnt3_2", "imdqnt3_3", "imdqnt3_4", "imdqnt3_5")

### Child demographics
#Ethnicity
ethn <- c("ethnicity_2", "ethnicity_3", "ethnicity_4", "ethnicity_5", "ethnicity_6")
# Sex
sex <- c("male")

### Maternal/birth characteristics
# Smoking during pregnancy
psmk <- c("smokepreg")
# Drinking during pregnancy
pdrk <- c("drinkpreg")
# Perinatal malaise
pmmm <- c("malm_clin")
# Age of mother at birth
agmb <- c("agemb")
bagi <- c("teenbrth + latrbrth")


### Household and family characteristics
# Highest education in household
hedu1 <- c("hheduc1_0", "hheduc1_1", "hheduc1_2", "hheduc1_3", "hheduc1_4")
linc123 <- c("lincome123")
plfp1 <- c("nplfp12")
htnr1 <- c("htrnt1")
hhdis1 <- c("hhdis1")
sngpr1 <- c("snglprnt1")
nchld1 <- c("numchld1")
nadlt1 <- c("numadlt1")
# Grand parent's education
gpedu <- c("gpeduc7_1", "gpeduc7_2", "gpeduc7_3", "gpeduc7_4", "gpeduc7_5")

### Child characteristics
# Disability
alc3 <- c("alc23")


##### Choose Risk Factor variable
# rskfctrs <- c("teenbrth0", "teenbrthim0", "teenbrthnc0")
# rskfctrs <- c("pretrmbrth0", "pretrmbrthim0", "pretrmbrthnc0")
# rskfctrs <- c("lwbrthwt0", "lwbrthwtim0", "lwbrthwtnc0")
# rskfctrs <- c("lwht3", "lwhtim3", "lwhtnc3")
# rskfctrs <- c("dsblty2", "dsbltyim2", "dsbltync2")
# rskfctrs <- c("dsblty3", "dsbltyim3", "dsbltync3")
# rskfctrs <- c("deldev3", "deldevim3", "deldevc3")
# 
# rskfctrs <- c("linc0", "lincim0", "lincnc0", "incqnt0", "incqntim0", "incqntnc0")
# rskfctrs <- c("inc0", "incim0", "incnc0")
# rskfctrs <- c("linc0", "lincim0", "lincnc0")
# rskfctrs <- c("incqnt0", "incqntim0", "incqntnc0")
# rskfctrs <- c("incqnt1", "incqntim1", "incqntnc1")
rskfctrs <- c("incqnt", "incqntim", "incqntnc", "linc", "incqnt1", "incqntim1", "incqntnc1")
# rskfctrs <- c("teenbrth0","teenbrthim0", "teenbrthnc0",
#               "pretrmbrth0", "pretrmbrthim0", "pretrmbrthnc0",
#               "lwbrthwt0", "lwbrthwtim0", "lwbrthwtnc0",
#               "deldev3", "deldevim3", "deldevnc3",
#               "dsblty3", "dsbltyim3", "dsbltync3",
#               "lwht3", "lwhtim3", "lwhtnc3",
#               "linc0", "lincim0", "lincnc0",
#               "incqnt0", "incqntim0", "incqntnc0")
# rskfctrs <- c("deldev3", "deldevim3", "deldevnc3")
# rskfctrs <- c("pretrmbrth0",
#               "lwbrthwt0")

# rskfctrs <- c("lwht3",
#               "disab3",
#               "deldev3")

# rskfctrs <- c("dsblty3", "dsbltyim3", "dsbltync3")

# rskfctrs <- c("lincim0", "lincnc0")
##### Choose type of coefficients output
## Risk factors that can be combined for regression structures
birthdis0 <- c("teenbrth0", "teenbrthim0", "teenbrthnc0",
               "pretrmbrth0", "pretrmbrthim0", "pretrmbrthnc0",
               "lwbrthwt0", "lwbrthwtim0", "lwbrthwtnc0")
# birthdis2 <- c("lwht2", "lwhtim2", "lwhtnc2",
#               "dsblty2", "dsbltyim2", "dsbltync2",
#               "deldev2", "deldevim2", "deldevc2")
birthdis3 <- c("lwht3", "lwhtim3", "lwhtnc3",
               "dsblty3", "dsbltyim3", "dsbltync3",
               "deldev3", "deldevim3", "deldevc3")
income0 <- c("linc0", "lincim0", "lincnc0",
             "incqnt1", "incqntim1", "incqntnc1",
             "incqnt", "incqntim", "incqntnc",
             "inc0", "incim0", "incnc0",
             "inc", "incim", "incnc")
mdlincdis <- c("lincim0", "incqntim0", "incim0", "lincim", "incqntim", "incim", "lincim1", "incqntim1", "incim1", "deldevim3")

#Loop over list of selected risk factors
for (rskfctr in rskfctrs) {
  # foreach (rskfctr in rskfctrs) %dopar% {
  
  ##### Choose outcomes
  if (substr(rskfctr, nchar(rskfctr), nchar(rskfctr)) == "0") {
    out <- out0
    cout <- cout0
    sout <- sout0
    bout <- bout0
  } else if (substr(rskfctr, nchar(rskfctr), nchar(rskfctr)) == "1") {
    out <- out1
    cout <- cout1
    sout <- sout1
    bout <- bout1
  } else if (substr(rskfctr, nchar(rskfctr), nchar(rskfctr)) == "2") {
    out <- out2
    cout <- cout2
    sout <- sout2
    bout <- bout2
  } else if (substr(rskfctr, nchar(rskfctr), nchar(rskfctr)) == "3") {
    out <- out3
    cout <- cout3
    sout <- sout3
    bout <- bout3
  }
  ##### Specify formulas
  if (rskfctr == "linc") {
    frml <- paste("~ laincome123",
                  "+ country1_2 + country1_3 + country1_4",
                  "+ region1_1 + region1_2 + region1_3 + region1_4 + region1_5 + region1_6 + region1_8 + region1_9 ",
                  "+ ethnicity_2 + ethnicity_3 + ethnicity_4 + ethnicity_5 + ethnicity_6 ", 
                  "+ smokepreg",
                  "+ agemb ",
                  "+ hheduc1_0 + hheduc1_1 + hheduc1_2 + hheduc1_3 + hheduc1_4 ", 
                  "+ hhdis1",
                  "+ snglprnt1",
                  sep = "")
    ## Including mediators
  } else if (rskfctr == "lincim") {
    frml <- paste("~ laincome123",
                  "+ country1_2 + country1_3 + country1_4 ", 
                  "+ region1_1 + region1_2 + region1_3 + region1_4 + region1_5 + region1_6 + region1_8 + region1_9",
                  "+ imdqnt3_2 + imdqnt3_3 + imdqnt3_4 + imdqnt3_5 ", 
                  "+ ethnicity_2 + ethnicity_3 + ethnicity_4 + ethnicity_5 + ethnicity_6 ", 
                  "+ smokepreg",
                  "+ malm_clin ", 
                  "+ agemb ",
                  "+ hheduc1_0 + hheduc1_1 + hheduc1_2 + hheduc1_3 + hheduc1_4 ", 
                  "+ nplfp12",
                  "+ hhdis1",
                  "+ snglprnt1",
                  "+ gpeduc7_1 + gpeduc7_2 + gpeduc7_3 + gpeduc7_4 + gpeduc7_5 ", 
                  "+ alc23",
                  sep = "")
    tfrml <- paste("~ laincome123",
                   "+ country1_2 + country1_3 + country1_4",
                   "+ region1_1 + region1_2 + region1_3 + region1_4 + region1_5 + region1_6 + region1_8 + region1_9 ",
                   "+ imdqnt123_1 + imdqnt123_2 + imdqnt123_3 + imdqnt123_4 ",
                   "+ ethnicity_2 + ethnicity_3 + ethnicity_4 + ethnicity_5 + ethnicity_6 ", 
                   "+ smokepreg",
                   "+ malm_clin ", 
                   "+ agemb ",
                   "+ hheduc1_0 + hheduc1_1 + hheduc1_2 + hheduc1_3 + hheduc1_4 ", 
                   "+ nplfp12",
                   "+ hhdis1",
                   "+ snglprnt1",
                   "+ gpeduc7_1 + gpeduc7_2 + gpeduc7_3 + gpeduc7_4 + gpeduc7_5 ", 
                   sep = "")
    ## Simple correlation only
  } else if (rskfctr == "lincnc") {
    frml <- paste("~ laincome123",
                  sep = "")
    ### Income quintiles
    ## Base model
  } else if (rskfctr == "incqnt") {
    frml <- paste("~ incqnt123_1 + incqnt123_2 + incqnt123_3 + incqnt123_4",
                  "+ country1_2 + country1_3 + country1_4",
                  "+ region1_1 + region1_2 + region1_3 + region1_4 + region1_5 + region1_6 + region1_8 + region1_9 ",
                  "+ ethnicity_2 + ethnicity_3 + ethnicity_4 + ethnicity_5 + ethnicity_6 ", 
                  "+ smokepreg",
                  "+ agemb ",
                  "+ hheduc1_0 + hheduc1_1 + hheduc1_2 + hheduc1_3 + hheduc1_4 ",
                  "+ hhdis1",
                  "+ snglprnt1",
                  sep = "")
    ## Including mediators
  } else if (rskfctr == "incqntim") {
    frml <- paste("~ incqnt123_1 + incqnt123_2 + incqnt123_3 + incqnt123_4",
                  "+ country1_2 + country1_3 + country1_4 ", 
                  "+ region1_1 + region1_2 + region1_3 + region1_4 + region1_5 + region1_6 + region1_8 + region1_9",
                  "+ imdqnt123_1 + imdqnt123_2 + imdqnt123_3 + imdqnt123_4 ",
                  "+ ethnicity_2 + ethnicity_3 + ethnicity_4 + ethnicity_5 + ethnicity_6 ", 
                  "+ smokepreg",
                  "+ malm_clin ", 
                  "+ agemb ",
                  "+ hheduc1_0 + hheduc1_1 + hheduc1_2 + hheduc1_3 + hheduc1_4 ", 
                  "+ nplfp12",
                  "+ hhdis1",
                  "+ snglprnt1",
                  "+ gpeduc7_1 + gpeduc7_2 + gpeduc7_3 + gpeduc7_4 + gpeduc7_5 ", 
                  "+ alc23",
                  sep = "")
    tfrml <- paste("~ incqnt123_1 + incqnt123_2 + incqnt123_3 + incqnt123_4",
                   "+ country1_2 + country1_3 + country1_4 ", 
                   "+ region1_1 + region1_2 + region1_3 + region1_4 + region1_5 + region1_6 + region1_8 + region1_9",
                   "+ imdqnt123_1 + imdqnt123_2 + imdqnt123_3 + imdqnt123_4 ",
                   "+ ethnicity_2 + ethnicity_3 + ethnicity_4 + ethnicity_5 + ethnicity_6 ", 
                   "+ smokepreg",
                   "+ malm_clin ", 
                   "+ agemb ",
                   "+ hheduc1_0 + hheduc1_1 + hheduc1_2 + hheduc1_3 + hheduc1_4 ", 
                   "+ nplfp12",
                   "+ hhdis1",
                   "+ snglprnt1",
                   "+ gpeduc7_1 + gpeduc7_2 + gpeduc7_3 + gpeduc7_4 + gpeduc7_5 ", 
                   sep = "")
    ## Simple correlation only
  } else if (rskfctr == "incqntnc") {
    frml <- paste("~ incqnt123_1 + incqnt123_2 + incqnt123_3 + incqnt123_4",
                  sep = "")
    ### Income quintiles (9 months)
    ## Base model
  } else if (rskfctr == "incqnt1") {
    frml <- paste("~ incqnt1_1 + incqnt1_2 + incqnt1_3 + incqnt1_4",
                  "+ country1_2 + country1_3 + country1_4",
                  "+ region1_1 + region1_2 + region1_3 + region1_4 + region1_5 + region1_6 + region1_8 + region1_9 ",
                  "+ ethnicity_2 + ethnicity_3 + ethnicity_4 + ethnicity_5 + ethnicity_6 ", 
                  "+ smokepreg",
                  "+ agemb ",
                  "+ hheduc1_0 + hheduc1_1 + hheduc1_2 + hheduc1_3 + hheduc1_4 ",
                  "+ hhdis1",
                  "+ snglprnt1",
                  sep = "")
    ## Including mediators
  } else if (rskfctr == "incqntim1") {
    frml <- paste("~ incqnt1_1 + incqnt1_2 + incqnt1_3 + incqnt1_4",
                  "+ country1_2 + country1_3 + country1_4 ", 
                  "+ region1_1 + region1_2 + region1_3 + region1_4 + region1_5 + region1_6 + region1_8 + region1_9",
                  "+ imdqnt1_2 + imdqnt1_3 + imdqnt1_4 + imdqnt1_5 ", 
                  "+ ethnicity_2 + ethnicity_3 + ethnicity_4 + ethnicity_5 + ethnicity_6 ", 
                  "+ smokepreg",
                  "+ malm_clin ", 
                  "+ agemb ",
                  "+ hheduc1_0 + hheduc1_1 + hheduc1_2 + hheduc1_3 + hheduc1_4 ", 
                  "+ nplfp1",
                  "+ hhdis1",
                  "+ snglprnt1",
                  "+ gpeduc7_1 + gpeduc7_2 + gpeduc7_3 + gpeduc7_4 + gpeduc7_5 ",
                  sep = "")
    ## Simple correlation only
  } else if (rskfctr == "incqntnc1") {
    frml <- paste("~ incqnt1_1 + incqnt1_2 + incqnt1_3 + incqnt1_4",
                  sep = "")
  } else{}
  
  
  ##### Loop to run correct set of regressions based on exposure variable and outcome list chosen above
  # Initializing Lists for Results
  init_results(out, n_imp, tab = tb, bout)
  
  # Run regressions for each imputation
  for (imp in 1:n_imp) {
    #foreach(imp = 1:30) %dopar% {
    # Subset data for current imputn
    data_subset <- mcs %>% 
      filter(imputn == imp)
    ##Continuous outcomes
    for (outcome in cout) {
      formula <- as.formula(paste(outcome, frml, sep = ""))
      # model <- lm_robust(formula, data = data_subset, weights = data_subset$wt_uk2, se_type = "stata")
      results <- extract_results(formula, data_subset, outcome, wt = 'wt_uk2', reg = "linear", tab = tb) 
      
      # Store results by outcome and imputn
      m_beta[[outcome]][[imp]] <- results$beta
      m_se[[outcome]][[imp]] <- results$se
      m_res[[outcome]][[imp]] <- results$residuals
      if (tb == "dydx"){
        d_beta[[outcome]][[imp]] <- results$b
        d_se[[outcome]][[imp]] <- results$s
      } else{}
    }
    
    ##Scores
    for (outcome in sout) {
      formula <- as.formula(paste(outcome, frml, sep = ""))
      results <- extract_results(formula, data_subset, outcome, wt = 'wt_uk2', reg = "negbinom", tab = tb) 
      
      # Store results by outcome and imputn
      m_beta[[outcome]][[imp]] <- results$beta
      m_se[[outcome]][[imp]] <- results$se
      m_res[[outcome]][[imp]] <- results$residuals
      if (tb == "dydx"){
        d_beta[[outcome]][[imp]] <- results$b
        d_se[[outcome]][[imp]] <- results$s
      } else{}
    }
    
    ##Binary outcomes
    for (outcome in bout) {
      #remove disability as confounder when it is outcome
      if (rskfctr %in% mdlincdis && outcome %in% alc1) {
        formula <- as.formula(paste(outcome, tfrml, sep = ""))
      } else {
        formula <- as.formula(paste(outcome, frml, sep = ""))
      }
      results <- extract_results(formula, data_subset, outcome, wt = 'wt_uk2', reg = "logit", tab = tb) 
      
      # Store results by outcome and imputn
      m_beta[[outcome]][[imp]] <- results$beta
      m_se[[outcome]][[imp]] <- results$se
      m_res[[outcome]][[imp]] <- results$residuals
      if (tb == "dydx"){
        d_beta[[outcome]][[imp]] <- results$b
        d_se[[outcome]][[imp]] <- results$s
      } else{}
    }
  }

  
  ####### Regular estimates for simulation
  ##### Rubins rules
  rr <- rubins_rules(m_beta, m_se)
  
  beta <- as.data.frame(rr$beta)
  se <- as.data.frame(rr$se)
  
  ###Residuals
  
  std_devs <- map_dbl(m_res, ~sd(unlist(.x)))
  res <- as.data.frame(std_devs)
  
  ###### Replace missing coefficients and errors with 0 
  # This is to prevent simulation from breaking
  
  beta <- beta %>% na.replace(0)
  se <- se %>% na.replace(0)
  
  
  ###### Export final files
  write.csv(beta, file = file.path('regout', paste0(rskfctr, "_beta.csv")))
  write.csv(se, file = file.path('regout', paste0(rskfctr, "_se.csv")))
  write.csv(res, file = file.path('regout', paste0(rskfctr, "_res.csv")))
  
  # # Load the xtable library for generating LaTeX tables
  # library(xtable)
  # 
  # # Filter the coefficients for the specific outcomes
  # bbeta <- beta %>%
  #   filter(outcome %in% c(hsp, alc, sen, tru, exc, bgc, smk, obs, hpf)) %>%
  #   mutate(across(-outcome, base::exp))
  # #
  # 
  # # Create a LaTeX table for the coefficients
  # coefficients_table <- xtable(bbeta, caption = "Coefficients")
  # # Create a LaTeX table for the standard errors
  # errors_table <- xtable(se, caption = "Standard Errors")
  # 
  # # Print the LaTeX code for the tables
  # print(coefficients_table, include.rownames = FALSE)
  # print(errors_table, include.rownames = FALSE)
  
  ###### Estimates for tables
  ### Marginal effects
  if (tb == "dydx"){
    ##### Rubins rules
    rrd <- rubins_rules(d_beta, d_se)
    
    bd <- as.data.frame(rrd$beta)
    sd <- as.data.frame(rrd$se)
    
    ###### Replace missing coefficients and errors with 0 
    # This is to prevent simulation from breaking
    bd <- bd %>% na.replace(0)
    sd <- sd %>% na.replace(0)
    
    
    ###### Export final files
    write.csv(bd, file = file.path('regout', paste0(rskfctr, "_betam.csv")))
    write.csv(sd, file = file.path('regout', paste0(rskfctr, "_sem.csv")))
    
    
    
    # Choose only Coefficients of interest for the table
    
    tnb <- c("teenbrth0", "teenbrthim", "teenbrthnc0")
    ptb <- c("pretrmbrth0", "pretrmbrthim0", "pretrmbrthnc0")
    lwt <- c("lwbrthwt0", "lwbrthwtim0", "lwbrthwtnc0")
    lht <- c("lwht3", "lwhtim3", "lwhtnc3")
    dsb <- c("dsblty3", "dsbltyim3", "dsbltync3")
    ddv <- c("deldev3", "deldevim3", "deldevc3")
    ain <- c("inc0", "incim", "incnc")
    lai <- c("linc0", "lincim0", "lincnc0")
    iqn <- c("incqnt0", "incqntim0", "incqntnc0")
    
    if (rskfctr %in% tnb) {
      l <- c("teenbrth")
    } else if (rskfctr %in% ptb) {
      l <- c("pretrm_1", "pretrm_2", "pretrm_3")
    } else if (rskfctr %in% lwt) {
      l <- c("bthwt_1", "bthwt_2")
    } else if (rskfctr %in% lht) {
      l <- c("lwht3")    
    } else if (rskfctr %in% dsb) {
      l <- c("alc3") 
    } else if (rskfctr %in% ddv) {
      l <- c("deldev3")     
    } else if (rskfctr %in% ain) {
      l <- c("aincome123") 
    } else if (rskfctr %in% lai) {
      l <- c("laincome123") 
    } else if (rskfctr %in% iqn) {
      l <- c("incqnt123_1", "incqnt123_2", "incqnt123_3", "incqnt123_4") 
    } else{}    
    
    b <- bd[, c("outcome", l)]
    s <- sd[, c("outcome", l)]
    
    # b$outcome <- var_dict[b$outcome]
    # s$outcome <- var_dict[s$outcome]
    
    # b <- b %>% rename(any_of(var_dict))
    # s <- s %>% rename(any_of(var_dict))
    
    # Combine the beta and se data frames
    combined_df <- interleave_rows(b, s)
    
    # combined_df$outcome <- str_replace_all(string = combined_df$outcome, pattern = var_dict)
    
    # Replace the strings in the "outcome" column
    combined_df$outcome <- var_dict[combined_df$outcome]
    combined_df <- rbind(colnames(combined_df), combined_df)
    combined_df <- apply(combined_df, 2, function(x) {
      x[1] <- var_dict[x[1]]
      x
    })
    
    # Create a LaTeX table for the combined data frame
    combined_table <- xtable(combined_df,
                             include.rownames = FALSE)
    print(combined_table,
          hline.after = c(0, 1, nrow(combined_table)),
          align = c("l", paste0("c", rep("", ncol(combined_table) - 1))),
          include.rownames = FALSE,
          include.colnames = FALSE,
          file = file.path("regout", paste0(rskfctr, "_coefs.tex")))
    
    
    # Print the LaTeX code for the combined table
    out <- list(hsp1, alc1, sen3, tru4, exc4, bgc6, smk5, obs6, hpf6)
    # print(combined_table, include.rownames = FALSE, include.columnnames = FALSE)
    for (outlst in out) {
      tcomtbl <- combined_table[, c(1, which(colnames(combined_table) %in% outlst))]
      table_name <- paste0(rskfctr, "_", substr(outlst[1], 1, 3), "_coefs.tex")
      print(tcomtbl,
            hline.after = c(0, 1, nrow(tcomtbl)),
            align = c("l", paste0("c", rep("", ncol(tcomtbl) - 1))),
            # include.rownames = FALSE,
            include.colnames = FALSE,
            file = file.path("regout", table_name))
    }
    
    # # Create a LaTeX table for the coefficients
    # coefficients_table <- xtable(beta, caption = "Coefficients")
    # # Create a LaTeX table for the standard errors
    # errors_table <- xtable(se, caption = "Standard Errors")
    # 
    # # Print the LaTeX code for the tables
    # print(coefficients_table, include.rownames = FALSE, file = file.path('regout', paste0(rskfctr, "_beta.tex")))
    # print(errors_table, include.rownames = FALSE, file = file.path('regout', paste0(rskfctr, "_se.tex")))
  } else {}
}

# stopCluster(cl)