

global path "\\lifesim2-main"

clear all
set maxvar 120000
set more off


import delimited "$path\mcs1.csv", clear

*Drop vunnecessary variables/variables that have to be regerated with imputed data							
drop v1 pccc23 pccc34 pccc234 teenbrth pretrm pretrm_1 pretrm_2 pretrm_3 bthwt_c bthwt_1 bthwt_2 lwht2 lwht3 gmotor1_adj fmotor1_adj comms1_adj develop1_adj develop1 motor1 motor1_adj motor_b10 zcog2_b10 deldev2 zcog3_b10 deldev3 internal2 lifesat2 zlifesat2 internal3 lifesat3 zlifesat3 internal4 lifesat4 zlifesat4 internal5 lifesat5 zlifesat5 internal6 lifesat6 zlifesat6 internal7 internalsr7 lifesat7 zlifesat7 lifesatsr7 lifesatwb5 lifesatwb6 lifesatwb7 lincome1 lincome2 lincome3 lincome4 lincome5 lincome6 income123 lincome123 income123_q1 income123_q2 income123_q3 income123_q4 income123_q5 poverty123 sdqinternal2 sdqinternal3 sdqinternal4 sdqinternal5 sdqinternal6 sdqinternal7 sdqinternalsr7 sdqexternal2 sdqexternal3 sdqexternal4 sdqexternal5 sdqexternal6 sdqexternal7 sdqexternalsr7 zcog2 zbasnv2 zbsrar2 zcm2 zcog3 zbasnv3 zbaspc3 zbasps3 zcm3 zcog4 zbaswr4 zbaspc4 znferpm4 zcm4 zcog5 zcog6 zcog7 imdqnt1 imdqnt2 imdqnt3 imdqnt4 imdqnt5 imdqnt6 imdqnt1_1 imdqnt1_2 imdqnt1_3 imdqnt1_4 imdqnt1_5 gestwks imdqnt2 imdqnt1 imdqnt3 imdqnt4 imdqnt5 imdqnt6 imdqnt*_* country2_* region2_* country3_* region3_* poverty*


//use "$modelling\ActEarly data file (handover to Shrathinth) TIDY VERSION v2", clear //come back and do a new tidy version after imputations
/*
mcsid cnum birthm birthy bmadv agemb bthwt smokepreg drinkpreg brstfdevr malm malm_clin rimm_tir rimm_dep rimm_wor rimm_rag rimm_ups rimm_jit rimm_ner rimm_her

cmagem1 cmaged1 sex1 ethnicity301 ethnicity1 country1 income1 poverty1 imddec1 imdqnt1 mrtlsts1 mssngl1 meduc1 htrnt1 htown1 hhdis1 nplfp1 aplfp1 snglprnt1 numppl1 numchld1 hhag1ch1 hhag2ch1 hhag3ch1 hhag4ch1 buag1ch1 buag2ch1 buag3ch1 buag4ch1 weight1 hosp1 gmotor1 fmotor1 comms1 develop1 hsngtnr1 hospac1 hospilna1 

cmagem2 cmaged2 sex2 ethnicity302 ethnicity2 country2 income2 poverty2 imddec2 imdqnt2 mrtlsts2 mssngl2 htrnt2 htown2 meduc2 hhdis2 nplfp2 aplfp2 snglprnt2 numppl2 numchld2 hhag1ch2 hhag2ch2 hhag3ch2 hhag4ch2 buag1ch2 buag2ch2 buag3ch2 buag4ch2 height2 weight2 bmi2 obesity2 lsc2 alc2 hosp2 zcog2 sdqconduct2 sdqemotion2 sdqpeer2 sdqprosoc2 sdqhyper2 sdqinternal2 sdqexternal2 sdqimpact2 hsngtnr2 hospac2 hospilna2 basnv2 zbasnv2 bsrar2 zbsrar2 zcm2 

cmagem3 cmaged3 sex3 country3 income3 poverty3 imddec3 imdqnt3 mrtlsts3 mssngl3 htrnt3 htown3 meduc3 hhdis3 nplfp3 aplfp3 snglprnt3 numppl3 numchld3 hhag1ch3 hhag2ch3 hhag3ch3 hhag4ch3 buag1ch3 buag2ch3 buag3ch3 buag4ch3 height3 weight3 bmi3 obesity3 lsc3 alc3 hosp3 zcog3 sdqconduct3 sdqemotion3 sdqpeer3 sdqprosoc3 sdqhyper3 sdqinternal3 sdqexternal3 sdqimpact3 hsngtnr3 hospac3 hospilna3 basnv3 zbasnv3 baspc3 zbaspc3 basps3 zbasps3 zcm3 

cmagem4 cmaged4 sex4 country4 income4 poverty4 imddec4 imdqnt4 mrtlsts4 mssngl4 htrnt4 htown4 meduc4 hhdis4 nplfp4 aplfp4 snglprnt4 numppl4 numchld4 hhag1ch4 hhag2ch4 hhag3ch4 hhag4ch4 buag1ch4 buag2ch4 buag3ch4 buag4ch4 height4 weight4 bmi4 obesity4 lsc4 alc4 hosp4 sen4 zcog4 sdqconduct4 sdqemotion4 sdqpeer4 sdqprosoc4 sdqhyper4 sdqinternal4 sdqexternal4 sdqimpact4 hsngtnr4 hospac4 hospilna4 baswr4 zbaswr4 baspc4 zbaspc4 nferpm4 znferpm4 zcm4 

cmagem5 cmagey5 sex5 country5 income5 poverty5 imddec5 imdqnt5 mrtlsts5 mssngl5 htrnt5 htown5 meduc5 hhdis5 nplfp5 aplfp5 snglprnt5 numppl5 numchld5 hhag1ch5 hhag2ch5 hhag3ch5 hhag4ch5 buag1ch5 buag2ch5 buag3ch5 buag4ch5 height5 weight5 bmi5 obesity5 lsc5 alc5 hosp5 excl5 truancy5 regtruancy5 sen5 zcog5 sdqconduct5 sdqemotion5 sdqpeer5 sdqhyper5 sdqprosoc5 sdqinternal5 sdqexternal5 genwelb5 welbgrid5 hsngtnr5 hospac5 hospilna5 texcl5 pexcl5 nwoffscl5 basvs5 

cmagem6 cmagey6 sex6 country6 income6 poverty6 imddec6 imdqnt6 mrtlsts6 mssngl6 htrnt6 htown6 meduc6 hhdis6 nplfp6 aplfp6 numppl6 numchld6 hhag1ch6 hhag2ch6 hhag3ch6 hhag4ch6 buag1ch6 buag2ch6 buag3ch6 buag4ch6 height6 weight6 bmi6 obesity6 lsc6 alc6 hosp6 smkevr6 smkreg6 drnkevr6 drnkfreq6 drnkreg6 excl6 truancy6 regtruancy6 sen6 zcog6 sdqconduct6 sdqpeer6 sdqprosoc6 sdqhyper6 sdqinternal6 sdqexternal6 genwelb6 hsngtnr6 welbgrid6 hospac6 hospilna6 smkfreq6 texcl6 pexcl6 nwoffscl6 apuvt6 

cmagem7 cmagey7 sex7 country7 mrtlsts7 mssngl7 snglprnt7 numppl7 numchld7 hhag1ch7 hhag2ch7 hhag3ch7 hhag4ch7 buag1ch7 buag2ch7 buag3ch7 buag4ch7 height7 weight7 bmi7 obesity7 lsc7 alc7 hosp7 smkevr7 smkreg7 drnkevr7 drnkfreq7 drnkreg7 prfrhlth7 gdgcse bdgcse gdgcse_me bdgcse_me zcog7 sdqconduct7 sdqconductsr7 sdqemotion7 sdqemotionsr7 sdqpeer7 sdqpeersr7 sdqprosoc7 sdqprosocsr7 sdqhyper7 sdqhypersr7 sdqinternal7 sdqinternalsr7 sdqexternal7 sdqexternalsr7 kessler7 bfpopen7 bfpcons7 bfpextr7 bfpagre7 bfpneur7 swemwbs7 hospilna7 smkfreq7 nagla7 

owt_cs owt_uk wt_cs1 wt_uk1 wt_cs2 wt_uk2 wt_cs3 wt_uk3 wt_cs4 wt_uk4 wt_cs5 wt_uk5 wt_cs6 wt_uk6 wt_uk6 wt_cs7 wt_uk7 
pccc2 pccc3 pccc4 pccc23 pccc34 pccc234 
pretrm pretrm_1 pretrm_2 pretrm_3 
male gestwks bthwt_c bthwt_1 bthwt_2 lwht2 
gmotor1_adj fmotor1_adj comms1_adj develop1_adj motor1 motor1_adj motor_b10 
zcog2_b10 deldev2 zcog3_b10 deldev3 
internal2 lifesat2 zlifesat2 internal3 lifesat3 zlifesat3 internal4 lifesat4 zlifesat4 internal5 lifesat5 zlifesat5 internal6 lifesat6 zlifesat6 internal7 internalsr7 wemws7 lifesat7 zlifesat7 
lincome1 lincome2 lincome3 lincome4 lincome5 lincome6 income123 lincome123 income123_q1 income123_q2 income123_q3 income123_q4 income123_q5 poverty123 
country1_1 country1_2 country1_3 country1_4 imdqnt1_1 imdqnt1_2 imdqnt1_3 imdqnt1_4 imdqnt1_5 
ethnicity1_1 ethnicity1_2 ethnicity1_3 ethnicity1_4 ethnicity1_5 ethnicity1_6
*/
/*
mcsid cnum bmadv

cmaged1 sex1 ethnicity301
imdqnt1 
mrtlsts1 mssngl1 htown1 aplfp1 
hsngtnr1  
hhag1ch1 hhag2ch1 hhag3ch1 hhag4ch1 buag1ch1 buag2ch1 buag3ch1 buag4ch1

cmaged2 sex2 ethnicity302 ethnicity2 country2 
imdqnt2 
mrtlsts2 mssngl2 htown2 aplfp2 
hsngtnr2
hhag1ch2 hhag2ch2 hhag3ch2 hhag4ch2 buag1ch2 buag2ch2 buag3ch2 buag4ch2

cmaged3 sex3 country3 
imdqnt3
mrtlsts3 mssngl3 htown3 aplfp3 
hsngtnr3 
hhag1ch3 hhag2ch3 hhag3ch3 hhag4ch3 buag1ch3 buag2ch3 buag3ch3 buag4ch3

cmaged4 sex4 country4 
imdqnt4 
mrtlsts4 mssngl4 htown4 aplfp4 
hsngtnr4 
hhag1ch4 hhag2ch4 hhag3ch4 hhag4ch4 buag1ch4 buag2ch4 buag3ch4 buag4ch4

cmagey5 sex5 country5 
imdqnt5 
mrtlsts5 mssngl5 htown5 aplfp5 nwoffscl5 
hsngtnr5 texcl5 pexcl5 

hhag1ch5 hhag2ch5 hhag3ch5 hhag4ch5 buag1ch5 buag2ch5 buag3ch5 buag4ch5

cmagey6 sex6 country6 
imdqnt6 
mrtlsts6 mssngl6 htown6 aplfp6
smkevr6 drnkevr6 drnkfreq6 smkfreq6 texcl6 pexcl6 nwoffscl6 hsngtnr6  
hhag1ch6 hhag2ch6 hhag3ch6 hhag4ch6 buag1ch6 buag2ch6 buag3ch6 buag4ch6 


cmagey7 sex7 country7 
mrtlsts7 mssngl7
smkevr7 smkfreq7 drnkevr7 drnkfreq7 gdgcse bdgcse gdgcse_me
hhag1ch7 hhag2ch7 hhag3ch7 hhag4ch7 buag1ch7 buag2ch7 buag3ch7 buag4ch7 
 

owt_cs owt_uk wt_cs1 wt_uk1 wt_cs2 wt_uk2 wt_cs3 wt_uk3 wt_cs4 wt_uk4 wt_cs5 wt_uk5 wt_cs6 wt_uk6 wt_uk6 wt_cs7 wt_uk7 
pccc23 pccc34 pccc234 
pretrm pretrm_1 pretrm_2 pretrm_3 
bthwt_c bthwt_1 bthwt_2 lwht2 
gmotor1_adj fmotor1_adj comms1_adj develop1_adj motor1 motor1_adj motor_b10 
zcog2_b10 deldev2 zcog3_b10 deldev3 
internal2 lifesat2 zlifesat2 internal3 lifesat3 zlifesat3 internal4 lifesat4 zlifesat4 internal5 lifesat5 zlifesat5 internal6 lifesat6 zlifesat6 internal7 internalsr7 wemws7 lifesat7 zlifesat7 
lincome1 lincome2 lincome3 lincome4 lincome5 lincome6 income123 lincome123 income123_q1 income123_q2 income123_q3 income123_q4 income123_q5 poverty123 
country1_1 country1_2 country1_3 country1_4 imdqnt1_1 imdqnt1_2 imdqnt1_3 imdqnt1_4 imdqnt1_5 
ethnicity1_1 ethnicity1_2 ethnicity1_3 ethnicity1_4 ethnicity1_5 ethnicity1_6
*/

*keeping sample at age 3 (these have the missed elegible babies missed in the first sweep)
keep if wt_uk2!=.
*keeping just singletons, meaning that all twins and triplet are dropped
keep if nocmhh == 1
codebook mcsid //N=15,381

*dropping those with missing ethnicity (N=14), child months of birth  (N=1) as numbers are so low and including in ologit will most likely cause imputation problems
*replace ethnicity1 = ethnicity2 if ethnicity1 == . & ethnicity2 != .
*drop if ethnicity == .
drop if birthm==.

**** IMPUTATIONS ****


*setting mi to wide
*******************
*mi set wide
*mi set mlong
mi set flong

*registering of the variables to be imputed
*******************************************
mi register imputed ///
/// ****************** VARIABLES ****************** 
gmotor1 fmotor1 comms1  /// Motor and communication development scores	
basnv2 bsrar2 basnv3 baspc3 basps3 baswr4 baspc4 nferpm4 basvs5 apuvt6 nagla7 /// cognitive measures named by sweep   
bdgcse_me /// end of school results
sdqconduct2 sdqconduct3 sdqconduct4 sdqconduct5 sdqconduct6 sdqconduct7 sdqconductsr7 /// conduct problems raw score
sdqemotion2 sdqemotion3 sdqemotion4 sdqemotion5 sdqemotion6 sdqemotion7 sdqemotionsr7 /// emotional problems raw score 
sdqpeer2 sdqpeer3 sdqpeer4 sdqpeer5 sdqpeer6 sdqpeer7 sdqpeersr7  /// peer effects raw score
sdqprosoc2 sdqprosoc3 sdqprosoc4 sdqprosoc5 sdqprosoc6 sdqprosoc7 sdqprosocsr7  /// pro-scoial raw score
sdqhyper2 sdqhyper3 sdqhyper4 sdqhyper5 sdqhyper6 sdqhyper7 sdqhypersr7  /// hyperactivity raw score
sdqimpact2 sdqimpact3 sdqimpact4 /// impact raw score
///sdqinternal2 sdqinternal3 sdqinternal4 sdqinternal5 sdqinternal6 sdqinternal7 sdqinternalsr7  /// Internalising raw score
///sdqexternal2 sdqexternal3 sdqexternal4 sdqexternal5 sdqexternal6 sdqexternal7 sdqexternalsr7  /// Externalising raw score
weight1 weight2 weight3 weight4 weight5 weight6 weight7 /// Weight
height2 height3 height4 height5 height6 height7 /// Height
bmi2 bmi3 bmi4 bmi5 bmi6 bmi7 /// BMI continuous
obesity2 obesity3 obesity4 obesity5 obesity6 obesity7 /// UK90 based obesity indicator /// Removed: obesity3 obesity4
lsc2 lsc3 lsc4 lsc5 lsc6 lsc7 /// longstanding health conditions
alc2 alc3 alc4 alc5 alc6 alc7 /// activity limiting conditions
pccc2 pccc3 pccc4 /// Pediatric complex chronic conditions
hosp1 hosp2 hosp3 hosp4 hosp5 hosp6 hosp7  /// hospitalisation //drop specific reason if imputation difficuties 
*hospac1 hospac2 hospac3 hospac4 hospac5 hospac6 /// hospitalisation due to accidents
hospilna1 hospilna2 hospilna3 hospilna4 hospilna5 hospilna6 hospilna7 /// number of hospitalisations
sen4 sen5 sen6 /// Special educational needs
truancy5 truancy6 /// truancy
regtruancy5 regtruancy6  /// regular truancy
excl5 excl6 /// school exclusion
/// antisocAge11b antisocAge14b antisocAge17b /// antisocial behaviour binary
smkreg6 smkreg7 /// cigarette smoking
drnkreg6 drnkreg7 /// drinking
kessler7 /// mental health ///removed: MHconditions17 anxdepcurrent17 
prfrhlth7  /// physical health ///removed: Physicalcond17
genwelb5 genwelb6  /// General wellbeing
welbgrid5 welbgrid6 /// Wellbeing grid
swemwbs7 /// Warwick edinburgh mental wellbeing scale
///
/// ********* FAMILY CHARECTERISTICS*******************
///*EDUHdegree abuse1 smokebiMP alcbiMP drugsbiMP ObeseMP  /// initial family variables 
agemb smokepreg drinkpreg malm   /// mother's perinatal characteristics
country2 country3 country4 country5 country6 country7 /// Country
region2 region3 region4 region5 region6 region7 /// Region
imddec1 imddec2 imddec3 imddec4 imddec5 imddec6 /// IMD 
income1 income2 income3 income4 income5 income6 /// continuous equivalised income in each sweep
wealth5 wealth6 /// assets and savings
htrnt1 htrnt2 htrnt3 htrnt4 htrnt5 htrnt6 /// live in rental housing
///*poverty1 poverty2 poverty3 poverty4 poverty5 poverty6 /// poverty indicator in each sweep
nplfp1 nplfp2 nplfp4 nplfp5 nplfp6 /// Indicator for for no labour force participation in household
hhdis1 hhdis2 hhdis3 hhdis4 hhdis5 hhdis6 /// Disability in household
numppl1 numppl2 numppl3 numppl4 numppl5 numppl6 numppl7 /// Number of people in hosehold
numchld1 numchld2 numchld3 numchld4 numchld5 numchld6 numchld7 /// Number of children in hosehold
tmspntchldm1 tmspntchldm2 tmspntchldm3 /// Time speant with child Main parent
tmspntchldp1 tmspntchldp2 tmspntchldp3 /// Time speant with child partner
///
/// **************** CHILD CHARACTERISTICS ****************
bthwt  /// child initial characteristics  ///removed: ITtotal
brstfdevr /// breastfed ever 
gestweeks /// gestation in weeks
cmagem1 cmagem2 cmagem3 cmagem4 cmagem5 cmagem6 cmagem7 /// child age in months at each outcome
///
/// ******** NEW VARIABELS ADDED IN MAY 2023 ************
/// note: these have bot been integrated into any of the analyses so far, so not in kitchensink model 
/// smackAge3_b smackAge5_b smackAge7_b /// smacking of child
cma  /// attachment raw
psconm2 psconp2 /// Parental conflict /// parent-child relationship conflict and closeness subscales
psclom2 psclop2 /// Parental closeness
hle2 hle3  /// home learning environment raw 
/// pbBMI /// maternal prebirth BMI
///
/// ******** BIG 5 PERSONALITY ************ 
/// these variables have not been or renamed 
/// DMNEUR00 DMEXTR00 DPNEUR00 DPEXTR00 /// sweep 4 (age 7) main and parter ///
/// FMDOPEN FMDCONSC FMDEXTRAV FMDAGREE FMDNEUROT FPDOPEN FPDCONSC FPDEXTRAV FPDAGREE FPDNEUROT /// sweep 6 (age 14) main and partner (FM is main and FP is partner) 
bfpopen7 bfpcons7 bfpextr7 bfpagre7 bfpneur7 /// sweep 7 (age 17) cohort member
///
/// ***** SOME POTENTIAL AUXILLARY VARIABLES TO AIDE IN IMPUTATIONS *****
meduc1 meduc2 meduc3 meduc4 meduc5 meduc6 /// Main parent's education
peduc1 peduc2 peduc3 peduc4 peduc5 peduc6 /// Partner's education
hheduc1 hheduc2 hheduc3 hheduc4 hheduc5 hheduc6 /// Household education
gpeduc7 /// Maternal grandparent's education
fgpeduc7 /// Fraternal grandparent's education
/// EDUP /// parental education
snglprnt1 snglprnt2 snglprnt3 snglprnt4 snglprnt5 snglprnt6 snglprnt7 /// single parent status
///  MKESS00_S7 /// maternal mental health
mmhlth2 mmhlth3 mmhlth4 mmhlth5 mmhlth6 /// maternal mental health
///pmhlth2 pmhlth3 pmhlth4 pmhlth5 pmhlth6 /// partner's mental health
///SMFQ /// cm short moods and feelings questionnaire
///academint4 academint5 /// cm academic interest age 7 and 11
///academself5 academself6 /// cm academin self concept age 11 and 14 


*register variables with no missing variables
************************************
mi register regular male ethnicity_2 ethnicity_3 ethnicity_4 ethnicity_5 ethnicity_6 birthm birthy country1_2 country1_3 country1_4 region1_1 region1_2 region1_3 region1_4 region1_5 region1_6 region1_8 region1_9 wt_uk2 sptn00 pttype2 nh2



*imputation modelling
*********************
mi impute chained ///
/// ****************** OUTCOMES ****************** 
(regress) gmotor1 fmotor1 comms1  /// Motor and communication development scores																							///  removed: 
(regress) basnv2 bsrar2 basnv3 baspc3 basps3 baswr4 baspc4 nferpm4 basvs5 apuvt6 nagla7 /// original cognitive measures named by sweep										///  removed: 
(logit, augment) bdgcse_me /// end of school results																														///  removed: 
(truncreg) sdqconduct2 sdqconduct3 sdqconduct4 sdqconduct5 sdqconduct6 sdqconduct7 sdqconductsr7  /// conduct problems raw score							### Poisson		///  removed: 
(truncreg) sdqemotion2 sdqemotion3 sdqemotion4 sdqemotion5 sdqemotion6 sdqemotion7 sdqemotionsr7 /// emotional problems raw score							### Poisson		///  removed: 
(truncreg) sdqpeer2 sdqpeer3 sdqpeer4 sdqpeer5 sdqpeer6 sdqpeer7 sdqpeersr7  /// peer effects raw score														### Poisson		///  removed: 
(truncreg) sdqprosoc2 sdqprosoc3 sdqprosoc4 sdqprosoc5 sdqprosoc6 sdqprosoc7 sdqprosocsr7  /// pro-scoial raw score											### Poisson		///  removed: 
(truncreg) sdqhyper2 sdqhyper3 sdqhyper4 sdqhyper5 sdqhyper6 sdqhyper7 sdqhypersr7  /// hyperactivity raw score												### Poisson		///  removed:
///(regress) sdqinternal2 sdqinternal3 sdqinternal4 sdqinternal5 sdqinternal6 sdqinternal7 sdqinternalsr7  /// Internalising raw score										///  removed: 
///(regress) sdqexternal2 sdqexternal3 sdqexternal4 sdqexternal5 sdqexternal6 sdqexternal7 sdqexternalsr7  /// Externalising raw score										///  removed: 
///(poisson) sdqimpact2 sdqimpact3 sdqimpact4 /// impact raw score																											///  removed: 
(truncreg) weight1 weight2 weight3  /// Weight																								///  removed: 
(truncreg) height2 height3 /// Height																										///  removed: 
///(truncreg) bmi2 bmi3 bmi4 bmi5 bmi6 bmi7 /// BMI continuous																												///  removed:
(regress) obesity2 obesity3 obesity4 obesity5 obesity6 obesity7 /// UK90 based obesity indicator																			///  removed: 
///(regress) lsc2 lsc3 lsc4 lsc5 lsc6 lsc7 /// longstanding health conditions																								///  removed: 
(regress) alc2 alc3 alc4 alc5 alc6 alc7 /// activity limiting conditions 																					### Logit		///  removed: 
(regress) pccc2 pccc3 pccc4 /// Pediatric complex chronic conditions																						### Logit		///  removed: 
(logit, augment) hosp1 hosp2 hosp3 hosp4 hosp5 hosp6 hosp7 /// hospitalisation //drop specific reason if imputation difficuties												///  removed: 
///(nbreg) hospilna1 hospilna2 hospilna3 hospilna4 hospilna5 hospilna6 hospilna7 /// number of hospitalisations																///  removed: 
(regress) sen4 sen5 sen6 /// Special educational needs																										### Logit		///  removed: 
(regress) truancy5 truancy6 /// truancy																														### Logit		///  removed: 
///(logit, augment) regtruancy5 regtruancy6  /// regular truancy																											///  removed: 
(logit, augment) excl5 excl6 /// school exclusion																															///  removed:  
(logit, augment) smkreg6 smkreg7 /// cigarette smoking																														///  removed:
///(logit, augment) drnkreg7 /// drinking																																	///  removed:  drnkreg6
(regress) kessler7 /// mental health ///removed: MHconditions17 anxdepcurrent17 																							///  removed: 
(logit, augment) prfrhlth7 /// physical health ///removed: Physicalcond17																									///  removed: 
///(regress) genwelb5 genwelb6 /// General wellbeing																										### Poisson		///  removed: 
(regress) welbgrid5 welbgrid6 /// Wellbeing grid																											### Poisson		///  removed: 
(regress) swemwbs7 /// Warwick edinburgh mental wellbeing scale																								### Poisson		///  removed: 
///
/// ********* INITIAL FAMILY CHARECTERISTICS*******************
(regress) agemb ///																																			### Poisson		///  removed: 
(logit, augment) smokepreg drinkpreg ///																																	///  removed: 
(regress) malm   /// initial family variables 																												### Poisson		///  removed: 
///(mlogit) country3 /// Country																																			///  removed:  country2 country4 country5 country6 country7
///(mlogit) region3 /// Region																																				///  removed:  region2 region4 region5 region6 region7
(ologit, augment) imddec1 imddec2 imddec3 imddec6 /// IMD 																													///  removed:  imddec4 imddec5
(truncreg) income1 income2 income3 income6 /// continuous equivalised income in each sweep																					///  removed:  income4 income5
(truncreg) wealth5 wealth6 /// Assets and savings																															///  removed: 
(regress) htrnt1 htrnt2 htrnt3 /// live in rental housing																									### Logit		///  removed:  htrnt4 htrnt5 htrnt6
///(logit, augment) poverty1 poverty2 poverty3 poverty4 poverty5 poverty6 /// poverty indicator in each sweep																///  removed: 
(logit, augment) nplfp1 nplfp2 /// Indicator for for no labour force participation in household 																			///  removed:  nplfp3 nplfp4 nplfp5 nplfp6
(regress) hhdis1 hhdis2 hhdis3 /// Disability in household 																									### Logit		///  removed:  hhdis4 hhdis5 hhdis6
(truncreg) numppl1 numppl2 numppl3 /// Number of people in hosehold																							### Poisson		///  removed:  numppl4 numppl5 numppl6 numppl7
(truncreg) numchld1 numchld2 numchld3  /// Number of children in hosehold																					### Poisson		///  removed:  numchld4 numchld5 numchld6 numchld7
(ologit, augment) meduc2 /// Main parent's education																														///  removed:  meduc2 meduc3 meduc4 meduc5 meduc6
///(ologit, augment) peduc2  /// Main parent's education																													///  removed:  peduc1 peduc3 peduc4 peduc5 peduc6
(ologit, augment) hheduc2  /// Main parent's education																														///  removed:  hheduc1 hheduc3 hheduc4 hheduc5 hheduc6
(ologit, augment) gpeduc7  /// Grandparent's education																														///  removed:  
(logit, augment) snglprnt1 /// single parent status																															///  removed:  snglprnt2 snglprnt3 snglprnt4 snglprnt5 snglprnt6 snglprnt7
///(regress) cma /// Parental attachment																													### Poisson		///  removed: 
///(regress) psconm2 /// Parental conflict																													### Poisson		///  removed:  psconp2
///(regress) psclom2 /// Parental closeness																													### Poisson		///  removed:  psclop2 
(ologit) tmspntchldm1  /// Time speant with child Main parent																								### Poisson		///  removed:  tmspntchldm2 tmspntchldm3
///(regress) tmspntchldp1 tmspntchldp2 tmspntchldp3 /// Time speant with child partner																		### Poisson		///  removed: 
(regress) hle2 hle3 /// Home learning environment																											### Poisson		///  removed: 
///
/// **************** INITIAL CHILD CHARACTERISTICS ****************
(regress) bthwt gestweeks /// child initial characteristics  ///removed: ITtotal
///(logit, augment) brstfdevr /// breastfed ever 
(truncreg) cmagem1 cmagem2 cmagem3 cmagem4 cmagem5 cmagem6 cmagem7 /// child age in months at each outcome													### Poisson		///  removed:
/// 
/// ******** NEW VARIABELS ADDED IN MAY 2023 ************
/// note: these have not been integrated into any of the analyses so far, so not in Aase's kitchensink model 
///
/// ******** BIG 5 PERSONALITY ************ 
/// these variables have not yet been renamed 
(regress) bfpopen7 bfpcons7 bfpextr7 bfpagre7 bfpneur7 /// sweep 7 (age 17) cohort member																					///  removed: 
///
/// ***** SOME POTENTIAL AUXILLARY VARIABLES TO AIDE IN IMPUTATIONS *****
(regress) mmhlth2 mmhlth3 mmhlth4 mmhlth5 mmhlth6 /// maternal mental health
= male ethnicity_2 ethnicity_3 ethnicity_4 ethnicity_5 ethnicity_6 country1_2 country1_3 country1_4 region1_1 region1_2 region1_3 region1_4 region1_5 region1_6 region1_8 region1_9, /// not included for imputation as cases are complate but helps imputation
add(30) force rseed (4000) dots noisily augment

//NOTE: when the right imputation methods did not work (eg. ologit or logit), I have used another methods, usually regress or truncreg. A little flag ### had been added to noting what the right method would have been, but which did not work.

save "$path\mcs1i", replace

							******** END OF IMPUTATION *********
							
use "$path\mcs1i", clear
***** Clean data for regressions
* Age of mother at birth
mi passive: gen agemb_r = round(agemb)
mi passive: replace agemb_r = 14 if agemb_r < 14
drop agemb
mi rename agemb_r agemb


mi update
*mi register imputed agemb


*Age in months
forval i = 1/7{
	mi passive: gen cmagem_`i' = round(cmagem`i')
	drop cmagem`i'
	mi rename cmagem_`i' cmagem`i'
}

mi update
*****Cognitive ability
***Development
*Age adjust scores
/*
mi passive: gen msng = missing(gmotor1, cmagem1)
mi estimate, post: reg gmotor1 i.cmagem1
mi predict pred
mi passive: gen gmotoradj1 = gmotor1 - pred
drop pred msng
mi passive: gen msng = missing(fmotor1, cmagem1)
mi estimate, post: reg fmotor1 i.cmagem1 if msng == 0
mi predict pred
mi passive: gen fmotoradj1 = fmotor1 - pred
drop pred msng
mi passive: gen msng = missing(comms1, cmagem1)
mi estimate, post: reg comms1 i.cmagem1 if msng == 0
mi predict pred
mi passive: gen commsadj1 = comms1 - pred
drop pred msng
*/
mi xeq: reg gmotor1 i.cmagem1
mi xeq: predict pred
mi passive: gen gmotoradj1 = gmotor1 - pred
drop pred
mi xeq: reg fmotor1 i.cmagem1
mi xeq: predict pred
mi passive: gen fmotoradj1 = fmotor1 - pred
drop pred
mi xeq: reg comms1 i.cmagem1
mi xeq: predict pred
mi passive: gen commsadj1 = comms1 - pred
drop pred
*Standard normalise
mi passive: egen zgmotor1 = std(gmotoradj1)
mi passive: egen zfmotor1  = std(fmotoradj1)
mi passive: egen zcomms1 = std(commsadj1)
*Motor development
mi passive: gen develop1 = zgmotor1 + zfmotor1
mi passive: egen zdevelop1 = std(develop1)
mi passive: gen deldev1 = zdevelop1 <= -1.282
mi passive: replace deldev1 = . if zdevelop1 == .
**Sweep 2
/*Age adjust scores
mi xeq: reg basnv2 i.cmagem2
mi xeq: predict pred
mi passive: gen basnvadj2 = basnv2 - pred
drop pred
mi xeq: reg bsrar2 i.cmagem2
mi xeq: predict pred
mi passive: gen bsraradj2 = bsrar2 - pred
drop pred*/
*Standard normalise
mi passive: egen zbasnv2 = std(basnv2)
mi passive: egen zbsrar2 = std(bsrar2)
mi passive: gen cog2 = zbasnv2 + zbsrar2
mi passive: egen zcog2 = std(cog2)
*Delayed development
mi passive: gen deldev2 = zcog2 <= -1.282
mi passive: replace deldev2 = . if zcog2 == .
**Sweep 3
/*Age adjust scores
mi xeq: reg basnv3 i.cmagem3
mi xeq: predict pred
mi passive: gen basnvadj3 = basnv3 - pred
drop pred
reg baspc3 i.cmagem3
mi xeq: predict pred
mi passive: gen baspcadj3 = baspc3 - pred
drop pred
reg basps3 i.cmagem3
mi xeq: predict pred
mi passive: gen baspsadj3 = basps3 - pred
drop pred */
*Standard normalise
mi passive: egen zbasnv3 = std(basnv3)
mi passive: egen zbaspc3 = std(baspc3)
mi passive: egen zbasps3 = std(basps3)
mi passive: gen cog3 = zbasnv3 + zbaspc3 + zbasps3
mi passive: egen zcog3 = std(cog3)
*Delayed development
mi passive: gen deldev3 = zcog3 <= -1.282
mi passive: replace deldev3 = . if zcog3 == .
**Sweep 4
/*Age adjust scores
mi xeq: reg baswr4 i.cmagem4
mi xeq: predict pred
mi passive: gen baswradj4 = baswr4 - pred
drop pred
reg baspc4 i.cmagem4
mi xeq: predict pred
mi passive: gen baspcadj4 = baspc4 - pred
drop pred 
mi xeq: reg nferpm4 i.cmagem4
mi xeq: predict pred
mi passive: gen nferpmadj4 = nferpm4 - pred
drop pred */
*Standard normalise
mi passive: egen zbaswr4 = std(baswr4)
mi passive: egen zbaspc4 = std(baspc4)
mi passive: egen znferpm4 = std(nferpm4)
mi passive: gen cog4 = zbaswr4 + zbaspc4 + znferpm4
mi passive: egen zcog4 = std(cog4)
**Sweep 5
*Age adjust scores
mi xeq: reg basvs5 i.cmagem5
mi xeq: predict pred
mi passive: gen basvsadj5 = basvs5 - pred
drop pred
*Standard normalise
mi passive: egen zcog5 = std(basvsadj5)
**Sweep 6
*Age adjust scores
mi xeq: reg apuvt6 i.cmagem6
mi xeq: predict pred
mi passive: gen apuvtadj6 = apuvt6 - pred
drop pred
*Standard normalise
mi passive: egen zcog6 = std(apuvtadj6)
**Sweep 7
*Age adjust scores
mi xeq: reg nagla7 i.cmagem7
mi xeq: predict pred
mi passive: gen naglaadj7 = nagla7 - pred
drop pred
*Standard normalise
mi passive: egen zcog7 = std(naglaadj7)

mi update
***** SDQ Scores *****
*** SDQ Scores
*Round and truncate
local var1 "sdqconduct sdqemotion sdqpeer sdqprosoc sdqhyper"
foreach X of local var1{
	forval i = 2/7{
		mi passive: gen `X'_`i' = `X'`i'
		mi passive: replace `X'_`i' = round(`X'_`i')
		mi passive: replace `X'_`i' = 0 if `X'_`i' < 0
		mi passive: replace `X'_`i' = 10 if `X'_`i' > 10 & `X'_`i' != .
		drop `X'`i'
		mi rename `X'_`i' `X'`i'
	}
		mi passive: gen `X'sr_7 = `X'sr7
		mi passive: replace `X'sr_7 = round(`X'sr_7)
		mi passive: replace `X'sr_7 = 0 if `X'sr_7 < 0
		mi passive: replace `X'sr_7 = 10 if `X'sr_7 > 10 & `X'sr_7 != .
		drop `X'sr7
		mi rename `X'sr_7 `X'sr7
}
mi update
***SDQ Externalising
forval i = 2/7{
mi passive: gen sdqexternal`i' = sdqconduct`i' + sdqhyper`i'
mi passive: gen external`i' = 20 - sdqexternal`i'
}
mi passive: gen sdqexternalsr7 = sdqconductsr7 + sdqhypersr7
mi passive: gen externalsr7 = 20 - sdqexternalsr7

***Kessler
mi passive: gen kessler7_r = round(kessler7)
mi passive: replace kessler7_r = 0 if kessler7_r < 0
mi passive: replace kessler7_r = 24 if kessler7_r > 24 & kessler7_r != .
drop kessler7
mi rename kessler7_r kessler7
mi update

***SWEMWBS Warwick edinburgh mental wellbeing scale	
mi passive: gen swemwbs7_r = round(swemwbs7)
mi passive: replace swemwbs7_r = 7 if swemwbs7_r < 7
mi passive: replace swemwbs7_r = 35 if swemwbs7_r > 35 & swemwbs7_r != .
drop swemwbs7
mi rename swemwbs7_r swemwbs7
mi update

***SDQ Internalising
forval i = 2/7{
mi passive: gen sdqinternal`i' = sdqemotion`i' + sdqpeer`i'
mi passive: gen internal`i' = 20 - sdqinternal`i'
}
mi passive: gen sdqinternalsr7 = sdqemotionsr7 + sdqpeersr7
mi passive: gen internalsr7 = 20 - sdqinternalsr7

*** Life satisfaction
local var1 "welbgrid5 welbgrid6"
foreach X of local var1{
		mi passive: gen `X'_r = round(`X')
		mi passive: replace `X'_r = 6 if `X'_r < 6
		mi passive: replace `X'_r = 42 if `X'_r > 42 & `X'_r != .
		drop `X'
		mi rename `X'_r `X'
}
local var1 "swemwbs7"
foreach X of local var1{
		mi passive: gen `X'_r = round(`X')
		mi passive: replace `X'_r = 7 if `X'_r < 7
		mi passive: replace `X'_r = 35 if `X'_r > 35 & `X'_r != .
		drop `X'
		mi rename `X'_r `X'
}
forval i = 2/7{
mi passive: gen lifesat`i' = 2 + (internal`i' *8/21)
}
mi passive: gen lifesatsr7 = 2 + (internalsr7 *8/21)
forval i = 5/6{
mi passive: gen lifesatwb`i' = 2 + ((36-welbgrid`i'+6)*8/36)
}
mi passive: gen lifesatwb7 = 2 + ((swemwbs7-7)*8/28)


***Maternal perinatal mental health - Rutter
mi passive: gen malm_r = round(malm)
mi passive: replace malm_r = 0 if malm_r < 0
mi passive: replace malm_r = 24 if malm_r > 24 & malm_r != .
drop malm
mi rename malm_r malm

drop malm_clin
mi passive: gen malm_clin = malm >= 4 & malm != .
mi update

***Number of people in household
forval i = 1/3{
mi passive: gen numppl`i'_r = round(numppl`i')
drop numppl`i'
mi rename numppl`i'_r numppl`i'
mi passive: replace numppl`i' = 2 if numppl`i' < 2
}
mi update

***Number of children in household
forval i = 1/3{
mi passive: gen numchld`i'_r = round(numchld`i')
drop numchld`i'
mi rename numchld`i'_r numchld`i'
mi passive: replace numchld`i' = 1 if numchld`i' < 1
}
mi update

***Number of adults in household
forval i = 1/3{
mi passive: gen numadlt`i' = numppl`i' - numchld`i'
mi passive: replace numadlt`i' = 1 if numadlt`i' < 1
}
mi update
/*
***Parental attachment	
mi passive: gen cma_r = round(cma)
mi passive: replace cma_r = 6 if cma_r < 6
mi passive: replace cma_r = 35 if cma_r > 35 & cma_r != .
drop cma
mi rename cma_r cma
mi update

*** Parental conflict	
mi passive: gen psconm_r = round(psconm)
mi passive: replace psconm_r = 8 if psconm_r < 8
mi passive: replace psconm_r = 40 if psconm_r > 40 & psconm_r != .
drop psconm
mi rename psconm_r psconm

mi passive: gen psconp_r = round(psconp)
mi passive: replace psconp_r = 8 if psconp_r < 8
mi passive: replace psconp_r = 40 if psconp_r > 40 & psconp_r != .
drop psconp
mi rename psconp_r psconp
mi update

*** Parental closeness
mi passive: gen psclom_r = round(psclom)
mi passive: replace psclom_r = 7 if psclom_r < 7
mi passive: replace psclom_r = 35 if psclom_r > 35 & psclom_r != .
drop psclom
mi rename psclom_r psclom

mi passive: gen psclop_r = round(psclop)
mi passive: replace psclop_r = 7 if psclop_r < 7
mi passive: replace psclop_r = 35 if psclop_r > 35 & psclop_r != .
drop psclop
mi rename psclop_r psclop
mi update
*/
*** Big 5 personlaity
local var1 "bfpopen7 bfpcons7 bfpextr7 bfpagre7 bfpneur7"
foreach X of local var1{
		mi passive: gen `X'_r = `X'
		mi passive: replace `X'_r = 0 if `X'_r < 0
		mi passive: replace `X'_r = 21 if `X'_r > 21 & `X'_r != .
		drop `X'
		mi rename `X'_r `X'
}
/*
*** Home learning environment
mi passive: gen hle2_r = round(hle2)
mi passive: replace hle2_r = 0 if hle2_r < 0
mi passive: replace hle2_r = 39 if hle2_r > 39 & hle2_r != .
drop hle2
mi rename hle2_r hle2

mi passive: gen hle3_r = round(hle3)
mi passive: replace hle3_r = 0 if hle3_r < 0
mi passive: replace hle3_r = 15 if hle3_r > 15 & hle3_r != .
drop hle3
mi rename hle3_r hle3
mi update	
*/
*** Maternal mental health - Kessler
local var1 "mmhlth2 mmhlth3 mmhlth4 mmhlth5 mmhlth6"
foreach X of local var1{
		mi passive: gen `X'_r = round(`X')
		mi passive: replace `X'_r = 0 if `X'_r < 0
		mi passive: replace `X'_r = 24 if `X'_r > 24 & `X'_r != .
		drop `X'
		mi rename `X'_r `X'
}
							
*** Activity limiting condition
tab alc2
local var1 "alc2 alc3 alc4 alc5 alc6 alc7"
foreach X of local var1{
		mi passive: gen `X'_r = round(`X')
		mi passive: replace `X'_r = 0 if `X'_r < 1
		mi passive: replace `X'_r = 1 if `X'_r >= 1 & `X'_r != .
		drop `X'
		mi rename `X'_r `X'
}

*** Pediatric complex chronic conditions
local var1 "pccc2 pccc3 pccc4"
foreach X of local var1{
		mi passive: gen `X'_r = round(`X')
		mi passive: replace `X'_r = 0 if `X'_r < 1
		mi passive: replace `X'_r = 1 if `X'_r >= 1 & `X'_r != .
		drop `X'
		mi rename `X'_r `X'
}

*** Special educational needs
local var1 "sen4 sen5 sen6"
foreach X of local var1{
		mi passive: gen `X'_r = round(`X')
		mi passive: replace `X'_r = 0 if `X'_r < 1
		mi passive: replace `X'_r = 1 if `X'_r >= 1 & `X'_r != .
		drop `X'
		mi rename `X'_r `X'
}

*** Truancy	
local var1 "truancy5 truancy6"
foreach X of local var1{
		mi passive: gen `X'_r = round(`X')
		mi passive: replace `X'_r = 0 if `X'_r < 1
		mi passive: replace `X'_r = 1 if `X'_r >= 1 & `X'_r != .
		drop `X'
		mi rename `X'_r `X'
}
*** Live in rental housing	
local var1 "htrnt1 htrnt2 htrnt3"
foreach X of local var1{
		mi passive: gen `X'_r = round(`X')
		mi passive: replace `X'_r = 0 if `X'_r < 1
		mi passive: replace `X'_r = 1 if `X'_r >= 1 & `X'_r != .
		drop `X'
		mi rename `X'_r `X'
}
*** Disability in household 	
local var1 "hhdis1 hhdis2 hhdis3"
foreach X of local var1{
		mi passive: gen `X'_r = round(`X')
		mi passive: replace `X'_r = 0 if `X'_r < 1
		mi passive: replace `X'_r = 1 if `X'_r >= 1 & `X'_r != .
		drop `X'
		mi rename `X'_r `X'
}

****** Early years variables  (at sweeps 1,2 and 3)
*** Income
forval i = 1/3{
replace income`i' = 0 if income`i' < 0
}
mi passive: egen income123 = rowmean(income1 income2 income3)

*Annual income
mi passive: gen aincome123 = 52 * income123
mi passive: gen aincome1 = 52 * income1

*** IMD (Minimum)
mi passive: egen imddec123 = rowmin(imddec1 imddec2 imddec3)

*** Number of children (Maximum)
mi passive: egen numchld123 = rowmax(numchld1 numchld2 numchld3)

*** Number of people (Minimum)
mi passive: egen numppl123 = rowmin(numppl1 numppl2 numppl3)

*** Number of adults (Minimum)
mi passive: egen numadlt123 = rowmin(numadlt1 numadlt2 numadlt3)

*** Household disability (Any)
mi passive: egen hhdis123 = rowmax(hhdis1 hhdis2 hhdis3)

*** Parental labour force participation (Any)
mi passive: egen nplfp12 = rowmax(nplfp1 nplfp2)

*** Disability
mi passive: egen alc23 = rowmax(alc2 alc3)


******************** Creating indicators
*** Mother's education
forval j = 1/4 {
	forval i = 0/5{
		gen meduc`j'_`i' = (meduc`j' == `i')
		replace meduc`j'_`i' = . if missing(meduc`j')
	}
}

*** Household education
forval j = 1/4 {
	forval i = 0/5{
		gen hheduc`j'_`i' = (hheduc`j' == `i')
		replace hheduc`j'_`i' = . if missing(hheduc`j')
	}
}

*** Grandparents education
forval j = 7/7 {
	forval i = 1/6{
		gen gpeduc`j'_`i' = (gpeduc`j' == `i')
		replace gpeduc`j'_`i' = . if missing(gpeduc`j')
	}
}


mi update
*** IMD Quintile
local var1 "imddec1 imddec2 imddec3"
forval i = 1/3{
mi passive: gen imdqnt`i' = imddec`i'
mi passive: replace imdqnt`i' = 1 if imddec`i' == 1 | imddec`i' == 2
mi passive: replace imdqnt`i' = 2 if imddec`i' == 3 | imddec`i' == 4
mi passive: replace imdqnt`i' = 3 if imddec`i' == 5 | imddec`i' == 6
mi passive: replace imdqnt`i' = 4 if imddec`i' == 7 | imddec`i' == 8
mi passive: replace imdqnt`i' = 5 if imddec`i' == 9 | imddec`i' == 10
}
forval i = 6/6{
mi passive: gen imdqnt`i' = imddec`i'
mi passive: replace imdqnt`i' = 1 if imddec`i' == 1 | imddec`i' == 2
mi passive: replace imdqnt`i' = 2 if imddec`i' == 3 | imddec`i' == 4
mi passive: replace imdqnt`i' = 3 if imddec`i' == 5 | imddec`i' == 6
mi passive: replace imdqnt`i' = 4 if imddec`i' == 7 | imddec`i' == 8
mi passive: replace imdqnt`i' = 5 if imddec`i' == 9 | imddec`i' == 10
}
forval i = 1/3{
mi passive: gen imdqnt`i'_1 = imdqnt`i' == 1 if imdqnt`i' != .
mi passive: gen imdqnt`i'_2 = imdqnt`i' == 2 if imdqnt`i' != .
mi passive: gen imdqnt`i'_3 = imdqnt`i' == 3 if imdqnt`i' != .
mi passive: gen imdqnt`i'_4 = imdqnt`i' == 4 if imdqnt`i' != .
mi passive: gen imdqnt`i'_5 = imdqnt`i' == 5 if imdqnt`i' != .
}

*Early years IMD
mi passive: gen imdqnt123 = imddec123
mi passive: replace imdqnt123 = 1 if imddec123 == 1 | imddec123 == 2
mi passive: replace imdqnt123 = 2 if imddec123 == 3 | imddec123 == 4
mi passive: replace imdqnt123 = 3 if imddec123 == 5 | imddec123 == 6
mi passive: replace imdqnt123 = 4 if imddec123 == 7 | imddec123 == 8
mi passive: replace imdqnt123 = 5 if imddec123 == 9 | imddec123 == 10

mi passive: gen imdqnt123_1 = imdqnt123 == 1 if imdqnt123 != .
mi passive: gen imdqnt123_2 = imdqnt123 == 2 if imdqnt123 != .
mi passive: gen imdqnt123_3 = imdqnt123 == 3 if imdqnt123 != .
mi passive: gen imdqnt123_4 = imdqnt123 == 4 if imdqnt123 != .
mi passive: gen imdqnt123_5 = imdqnt123 == 5 if imdqnt123 != .

mi update

* Country dummies
forval j = 2/3{
	forval i = 1/4{
		gen country`j'_`i' = (country`j' == `i')
		replace country`j'_`i' = . if missing(country`j')
	}
}
* Region dummies
forval j = 2/3{
	forval i = 1/12{
		gen region`j'_`i' = (region`j' == `i')
		replace region`j'_`i' = . if missing(region`j')
	}
}

*** Income Quintle

*local var1 "income1 income2 income3" /// continuous equivalised income in each sweep	
forval i = 1/3{
	mi xeq: xtile incqnt`i' = income`i' [pweight=wt_uk2], nq(5)
}

forval j = 1/3{
	forval i = 1/5{
		gen incqnt`j'_`i' = (incqnt`j' == `i')
		replace incqnt`j'_`i' = . if missing(incqnt`j')
	}
}

* Early years income
*mi xeq: xtile incqnt123 = income123 [pweight=wt_uk2], nq(5)
xtile incqnt123 = income123 [pweight=wt_uk2], nq(5)

tab incqnt123
tab incqnt123 if _mi_m == 0
tab incqnt123 if _mi_m != 0
forval i = 1/5{
		gen incqnt123_`i' = (incqnt123 == `i')
		replace incqnt123_`i' = . if missing(incqnt123)
}

/*
***Income quintiles based on 2014 UK data (https://www.ons.gov.uk/peoplepopulationandcommunity/personalandhouseholdfinances/incomeandwealth/compendium/familyspending/2015/chapter3equivalisedincome)
** CPI from ONS (https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/l522/mm23)
* UK CPI 2001 - 74.6
* UK CPI 2003 - 76.7
* UK CPI 2014 - 99.6
* UK CPI 2019 - 107.8
* UK CPI 2021 - 111.6
* UK CPI 2022 - 120.5
* UK CPI 2023 - 128.6
* Adjusting for 2014 £s
mi passive: replace income123 = income123*(99.6/76.7) 
mi passive: gen incqnt123 = .
mi passive: replace incqnt123 = 1 if income123 <= 200 & income123 != .
mi passive: replace  incqnt123 = 2 if (income123 >= 201 & income123 <= 287) & income123 != .
mi passive: replace  incqnt123 = 3 if (income123 >= 288 & income123 <= 388) & income123 != .
mi passive: replace  incqnt123 = 4 if (income123 >= 389 & income123 <= 540) & income123 != .
mi passive: replace  incqnt123 = 5 if income123 >= 698 & income123 != .

tab incqnt123
tab incqnt123 if _mi_m == 0
tab incqnt123 if _mi_m != 0

mi passive: gen incqnt123_1 = income123 <= 200 if income123 != .
mi passive: gen incqnt123_2 = (income123 >= 201 & income123 <= 287) if income123 != .
mi passive: gen incqnt123_3 = (income123 >= 288 & income123 <= 388) if income123 != .
mi passive: gen incqnt123_4 = (income123 >= 389 & income123 <= 540) if income123 != .
mi passive: gen incqnt123_5 = income123 >= 698 if income123 != .
*/

*** Poverty
mi passive: gen poverty1 = income1 < 187 if income1 != .
mi passive: gen poverty2 = income2 < 200 if income2 != .
mi passive: gen poverty3 = income3 < 217 if income3 != .
mi passive: egen poverty123 = rowmean(poverty1 poverty2 poverty3)


*** Obesity
local var1 "obesity2 obesity3 obesity4 obesity5 obesity6 obesity7"
foreach X of local var1{
		mi passive: gen `X'_r = round(`X')
		mi passive: replace `X'_r = 0 if `X'_r < 1
		mi passive: replace `X'_r = 1 if `X'_r >= 1 & `X'_r != .
		drop `X'
		mi rename `X'_r `X'
}
tab obesity7
/*
forval i = 2/7{
gen months = cmagem`i'
*Add height thresholds
merge m:1 male months using "$path\data\bmichart\bmimin.dta"
tab _merge
drop if _merge == 2
drop _merge
*create indicators
mi passive: gen obese`i' = bmi`i' >= p85 if bmi`i' != . & p85 != .
mi update
drop p90 p85 months
}
*/
*** log income
local var1 "income1 income2 income3 income123 aincome123"
foreach X of local var1{
		mi passive: gen l`X' = log(`X')
}





******* ActEarly variables
*** Teenage parent
mi passive: gen teenbrth = agemb<19 if agemb != . 
*Mother over 35
mi passive: gen latrbrth = agemb>=35 if agemb != . 
*** Preterm birth
/*
mi passive: gen pretrm = .
mi passive: replace pretrm = 3 if gestweeks < 28
mi passive: replace pretrm = 2 if gestweeks >= 28 & gestweeks < 32
mi passive: replace pretrm = 1 if gestweeks >= 32 & gestweeks < 37
mi passive: replace pretrm = 0 if gestweeks >= 37
*/
mi passive: gen pretrm_3 = gestweeks < 28 & gestweeks != .
mi passive: gen pretrm_2 = gestweeks >= 28 & gestweeks < 32 & gestweeks != .
mi passive: gen pretrm_1 = gestweeks >= 32 & gestweeks < 37 & gestweeks != .

mi update
*** Low birthweight
*Weeks of gestation
mi passive: gen gestwks = round(gestweeks)
mi passive: replace gestwks = 43 if gestwks > 43 & gestwks != .
*Add weight thresholds
merge m:1 male gestwks using "$path\data\weightchart\wfga.dta"
tab _merge
drop if _merge == 2
drop _merge
*create indicators
/*
mi passive: gen bthwt_c = .
mi passive: replace bthwt_c = 0 if bthwt*1000 > p10 & bthwt*1000 < p90 & bthwt != . & p10 != . & p90 != .
mi passive: replace bthwt_c = 2 if bthwt*1000 >= p90 & bthwt != . & p10 != . & p90 != .
mi passive: replace bthwt_c = 1 if bthwt*1000 <= p10 & bthwt != . & p10 != . & p90 != .
*/
mi passive: gen bthwt_2 = bthwt*1000 >= p90 if bthwt != . & p90 != .
mi passive: gen bthwt_1 = bthwt*1000 <= p10 if bthwt != . & p10 != .
mi update

drop p10 p90
*** Low height
gen months = cmagem3
*Add height thresholds
merge m:1 male months using "$path\data\heightchart\hmmf.dta"
tab _merge
drop if _merge == 2
drop _merge
*create indicators
mi passive: gen lwht3 = height3 <= p10 if height3 != . & p10 != .
mi update

drop p10 months
*** Paedriatic Chronic condition
mi passive: gen pccc23 = pccc3
mi passive: replace pccc23 = 1 if pccc2 == 1 & pccc23 != 1
mi update

*** School Readiness
tab deldev3

*** Imputation number duplicate
gen imputn = _mi_m

save "$path\mcs1ic", replace


use "$path\mcs1ic", clear

*setting the combined sampling and attrition weight for sweep 2 (age 3) as this was the sample I imputed back to in this study. This is to be used in all analyses.
mi svyset sptn00 [pweight=wt_uk2], strata (pttype2) fpc (nh2)

*I usually use esampvaryok in script as number of some datasets may vary in some analyses because of imputation difficulties. 

*simple descriptives example
mi estimate, post esampvaryok: svy: mean con_age5
mi estimate, post esampvaryok: svy: proportion badgcse_me

*simple linear regression example
reg zcog2 teenbrth country1_2 country1_3 country1_4 ethnicity_2 ethnicity_3 ethnicity_4 ethnicity_5 ethnicity_6 imdqnt1_2 imdqnt1_3 imdqnt1_4 imdqnt1_5 smokepreg malm_clin snglprnt1 meduc1_1 meduc1_2 meduc1_3 meduc1_4 meduc1_5 lincome1 [pweight=wt_uk2] if imputn == 1
reg zcog2 teenbrth country1_2 country1_3 country1_4 ethnicity_2 ethnicity_3 ethnicity_4 ethnicity_5 ethnicity_6 imdqnt1_2 imdqnt1_3 imdqnt1_4 imdqnt1_5 smokepreg malm_clin snglprnt1 meduc1_1 meduc1_2 meduc1_3 meduc1_4 meduc1_5 lincome1 [pweight=wt_uk2] if imputn == 2
mi estimate, post esampvaryok ni(2): reg zcog2 teenbrth country1_2 country1_3 country1_4 ethnicity_2 ethnicity_3 ethnicity_4 ethnicity_5 ethnicity_6 imdqnt1_2 imdqnt1_3 imdqnt1_4 imdqnt1_5 smokepreg malm_clin snglprnt1 meduc1_1 meduc1_2 meduc1_3 meduc1_4 meduc1_5 lincome1 [pweight=wt_uk2]

mi estimate, post esampvaryok: reg zcog2 teenbrth country1_2 country1_3 country1_4 ethnicity_2 ethnicity_3 ethnicity_4 ethnicity_5 ethnicity_6 imdqnt1_2 imdqnt1_3 imdqnt1_4 imdqnt1_5 smokepreg malm_clin snglprnt1 meduc1_1 meduc1_2 meduc1_3 meduc1_4 meduc1_5 lincome1 [pweight=wt_uk2]

*logistic regression example
logit bdgcse_me incqnt123_1 incqnt123_2 incqnt123_3 incqnt123_4 country1_2 country1_3 country1_4 region1_1 region1_2 region1_3 region1_4 region1_5 region1_6 region1_8 region1_9 ethnicity_2 ethnicity_3 ethnicity_4 ethnicity_5 ethnicity_6 teenbrth latrbrth smokepreg snglprnt1 meduc1_0 meduc1_1 meduc1_2 meduc1_3 meduc1_4 numchld1 numadlt1 hhdis123 nplfp12 alc23 [pweight=wt_uk2] if imputn == 1
margins,  dydx(*)   
logit bdgcse_me incqnt123_1 incqnt123_2 incqnt123_3 incqnt123_4 country1_2 country1_3 country1_4 region1_1 region1_2 region1_3 region1_4 region1_5 region1_6 region1_8 region1_9 ethnicity_2 ethnicity_3 ethnicity_4 ethnicity_5 ethnicity_6 teenbrth latrbrth smokepreg snglprnt1 meduc1_0 meduc1_1 meduc1_2 meduc1_3 meduc1_4 numchld1 numadlt1 hhdis123 nplfp12 alc23 [pweight=wt_uk2] if imputn == 2
margins,  dydx(*) 
mi estimate, post esampvaryok ni(2): logit bdgcse_me incqnt123_1 incqnt123_2 incqnt123_3 incqnt123_4 country1_2 country1_3 country1_4 region1_1 region1_2 region1_3 region1_4 region1_5 region1_6 region1_8 region1_9 ethnicity_2 ethnicity_3 ethnicity_4 ethnicity_5 ethnicity_6 teenbrth latrbrth smokepreg snglprnt1 meduc1_0 meduc1_1 meduc1_2 meduc1_3 meduc1_4 numchld1 numadlt1 hhdis123 nplfp12 alc23 [pweight=wt_uk2]

program mimargins, eclass properties(mi)
	version 12
	logit bdgcse_me incqnt123_1 incqnt123_2 incqnt123_3 incqnt123_4 country1_2 country1_3 country1_4 region1_1 region1_2 region1_3 region1_4 region1_5 region1_6 region1_8 region1_9 ethnicity_2 ethnicity_3 ethnicity_4 ethnicity_5 ethnicity_6 teenbrth latrbrth smokepreg snglprnt1 meduc1_0 meduc1_1 meduc1_2 meduc1_3 meduc1_4 numchld1 numadlt1 hhdis123 nplfp12 alc23 [pweight=wt_uk2]
	margins,  dydx(*) 
end
mi estimate, cmdok: mimargins 1 

pr drop mimargins

mi estimate, post esampvaryok: reg zcog2 teenbrth country1_2 country1_3 country1_4 ethnicity_2 ethnicity_3 ethnicity_4 ethnicity_5 ethnicity_6 imdqnt1_2 imdqnt1_3 imdqnt1_4 imdqnt1_5 smokepreg malm_clin snglprnt1 meduc1_1 meduc1_2 meduc1_3 meduc1_4 meduc1_5 lincome1 [pweight=wt_uk2]
mi estimate, post or esampvaryok: svy: logistic badGCSE_ME i. EDUHdegree INCEQ1


mi estimate, post esampvaryok: reg zcog4 incqnt123_1 incqnt123_2 incqnt123_3 incqnt123_4 country1_2 country1_3 country1_4 region1_1 region1_2 region1_3 region1_4 region1_5 region1_6 region1_8 region1_9 ethnicity_2 ethnicity_3 ethnicity_4 ethnicity_5 ethnicity_6 smokepreg agemb hheduc1_0 hheduc1_1 hheduc1_2 hheduc1_3 hheduc1_4 hhdis1 snglprnt1 [pweight=wt_uk2]
mi estimate, post esampvaryok: reg zcog4 incqnt123_1 incqnt123_2 incqnt123_3 incqnt123_4 country1_2 country1_3 country1_4 region1_1 region1_2 region1_3 region1_4 region1_5 region1_6 region1_8 region1_9 ethnicity_2 ethnicity_3 ethnicity_4 ethnicity_5 ethnicity_6 smokepreg agemb hheduc1_0 hheduc1_1 hheduc1_2 hheduc1_3 hheduc1_4 hhdis1 snglprnt1 nplfp12 [pweight=wt_uk2]

mi estimate, post esampvaryok: logit bdgcse_me incqnt123_1 incqnt123_2 incqnt123_3 incqnt123_4 country1_2 country1_3 country1_4 region1_1 region1_2 region1_3 region1_4 region1_5 region1_6 region1_8 region1_9 ethnicity_2 ethnicity_3 ethnicity_4 ethnicity_5 ethnicity_6 smokepreg agemb hheduc1_0 hheduc1_1 hheduc1_2 hheduc1_3 hheduc1_4 hhdis1 snglprnt1 [pweight=wt_uk2]
mi estimate, post esampvaryok: logit bdgcse_me incqnt123_1 incqnt123_2 incqnt123_3 incqnt123_4 country1_2 country1_3 country1_4 region1_1 region1_2 region1_3 region1_4 region1_5 region1_6 region1_8 region1_9 ethnicity_2 ethnicity_3 ethnicity_4 ethnicity_5 ethnicity_6 smokepreg agemb hheduc1_0 hheduc1_1 hheduc1_2 hheduc1_3 hheduc1_4 hhdis1 snglprnt1 nplfp12 [pweight=wt_uk2]

mi estimate, post esampvaryok: nbreg kessler7 incqnt123_1 incqnt123_2 incqnt123_3 incqnt123_4 country1_2 country1_3 country1_4 region1_1 region1_2 region1_3 region1_4 region1_5 region1_6 region1_8 region1_9 ethnicity_2 ethnicity_3 ethnicity_4 ethnicity_5 ethnicity_6 smokepreg agemb hheduc1_0 hheduc1_1 hheduc1_2 hheduc1_3 hheduc1_4 hhdis1 snglprnt1 [pweight=wt_uk2], irr
mi estimate, post esampvaryok: nbreg kessler7 incqnt123_1 incqnt123_2 incqnt123_3 incqnt123_4 country1_2 country1_3 country1_4 region1_1 region1_2 region1_3 region1_4 region1_5 region1_6 region1_8 region1_9 ethnicity_2 ethnicity_3 ethnicity_4 ethnicity_5 ethnicity_6 smokepreg agemb hheduc1_0 hheduc1_1 hheduc1_2 hheduc1_3 hheduc1_4 hhdis1 snglprnt1 nplfp12 [pweight=wt_uk2], irr

/*
//2) *** usual macros and loops and outreg works with mi data as exampled below using a different dataset***

*family environment predictors
global model1 "zconfM zconfP zcloseM zcloseP zpprelcom forcecom singpa2 sibs2 zpeer2"
*individual predictors
global model2 "$model1 male cmagem6 i.pubcomp bwght gest oldest BAS_decile zBracken"
*socioeconomic, demographic and other controls
global model3 "$model2 agemb agepb zkesm2 zkesp2 zBhomesafe i.ethcm6 i.eduh i.sech5 inc100 i.rent2 i.country"
*sweep 6 predictors
//global model4 "$model3 i.S6CMcloM i.S6CMcloF i.S6CMargM i.S6CMargF i.FPSCHC00M i.FPQARP00M i.FPSCHC00P i.FPQARP00P S6hapcom i.forcecom6 zkesm6 zkesp6 i.singpa6 sibs6 peer6"
global model4 "$model3 zS6CMcloM zS6CMcloF zS6CMargM zS6CMargF zFPSCHC00M zFPQARP00M zFPSCHC00P zFPQARP00P S6hapcom i.forcecom6 zkesm6 zkesp6 i.singpa6 sibs6 peer6"

mi svyset sptn00 [pweight=bovwt2], strata (pttype2) fpc (nh2)

foreach X of varlist zSMFQ zSMFQ selfharm zwellbe zselfest zemotion6 zASB zconduct6 {
mi estimate, post: svy: reg `X' $model1 
outreg2 using "$pathafc\`X'.xls", stats(coef se) bdec(2) sdec(2) side alpha(.001, .01, .05, .10) symbol(***, **, *, +) nor2 label replace
mi estimate, post: svy: reg `X' $model2
outreg2 using "$pathafc\`X'.xls", stats(coef se) bdec(2) sdec(2) side alpha(.001, .01, .05, .10) symbol(***, **, *, +) nor2 label append
mi estimate, post: svy: reg `X' $model3
outreg2 using "$pathafc\`X'.xls", stats(coef se) bdec(2) sdec(2) side alpha(.001, .01, .05, .10) symbol(***, **, *, +) nor2 label append
mi estimate, post: svy: reg `X' $model4
outreg2 using "$pathafc\SMFQ.xls", stats(coef se) bdec(2) sdec(2) side alpha(.001, .01, .05, .10) symbol(***, **, *, +) nor2 label append
}
*/

cd "E:\Shrathinth\LifeSim\Python Code\lifesim2-main"

keep mcsid cmagem1 cmagem2 cmagem3 cmagem4 cmagem5 cmagem6 cmagem7 smokepreg brstfdevr teenbrth latrbrth malm income123 malm_clin lwht3 deldev3 pretrm_1 pretrm_2 pretrm_3 bthwt_1 bthwt_2 htrnt1 htrnt2 htrnt3 snglprnt1 snglprnt2 snglprnt3 numppl1 numppl2 numppl3 numchld1 numchld2 numchld3 hosp1 hosp2 hosp3 hosp4 hosp5 hosp6 alc2 alc3 alc4 alc5 alc6 alc7 sen4 sen5 sen6 truancy5 truancy6 excl5 excl6 sdqconduct2 sdqconduct3 sdqconduct4 sdqconduct5 sdqconduct6 sdqconduct7 internal2 internal3 internal4 internal5 internal6 internal7 internalsr7 obesity7 bdgcse_me smkreg7 prfrhlth7 kessler7 develop1 zcog2 zcog3 zcog4 zcog5 zcog6 zcog7 external2 external3 external4 external5 external6 external7 externalsr7 meduc1_0 meduc1_1 meduc1_2 meduc1_3 meduc1_4 meduc1_5 country1_1 country1_2 country1_3 country1_4 region1_1 region1_2 region1_3 region1_4 region1_5 region1_6 region1_7 region1_8 region1_9 region1_10 region1_11 region1_12 ethnicity_1 ethnicity_2 ethnicity_3 ethnicity_4 ethnicity_5 ethnicity_6 male incqnt123_1 incqnt123_2 incqnt123_3 incqnt123_4 incqnt123_5 _mi_id _mi_miss _mi_m
mi convert flongsep mcs_imp, clear
							