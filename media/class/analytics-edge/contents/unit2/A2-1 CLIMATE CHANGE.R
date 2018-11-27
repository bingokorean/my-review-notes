### CLIMATE CHANGE

# 지난 15년동안 average global temperature가 꾸준히 증가하고 있음.
# 이로 인해, rising sea levels, increased frequency of extreme weather 등 인간에게 부정적인 영향을 끼침.
# 이번 분석은 average global temperature와 다양한 factor들간의 관계를 분석함.

# Data from May 1983 to December 2008
# Year: the observation year
# Month: the observation month
# Temp: the difference in degrees Celsious between the average global temporature in that period and a reference value. This data comes from the Climatic Research Unit at the University of East Anglia
# CO2, N2O, CH4, CFC.11, CFC.12: atmospheric concentrations of carbon dioxide (CO2), nitrous oxide (N2O), methane (CH4), trichlorofluoromethane (CCI3F; commonly referred to as CFC-11) and dichlorodifluoromethane (CCI2F2; commonly referred to as CFC-12), respectively. This data comes from the ESRL/NOAA Global Monitoring Division
   # CO2, N2O and CH4 are expressed in ppmv (parts of per million by volume)
   # CFC.11 and CFC.12 are expressed in ppbv (parts per billion by volume)
# Aerosols: the mean stratospheric aerosol optical depth at 550 nm. This variable is linked to volcanoes, as volcanic eruptions result in new particles being added to the atmosphere, which affect how much of the sun's energy is reflected back into space. This data from the Godard Institute for Space Studies at NASA.
# TSI: the total solar irradiance (TSI) in W/m2 (the rate where the sun's energy is deposited per unit area). Due to sunsopts and other solar phenomena, the amount of energy that is given off by the sun varies substantially with time. This data is from the SOLARIS-HEPPA project website.
# MEI: multivariate EI Nino Southern Oscillation index (MEI), a measure of the strength of the EI Nino/La Nina-Southern Oscillation (a weather effect in the Pacific Ocean that affects global temperatures). This data comes from the ESRL/NOAA Physical Sciences Division




### PROBLEM 1.1 - CREATING OUR FIRST MODEL
CLCH = read.csv("climate_change.csv")
CLCH_Train = subset(CLCH, Year < 2007)
CLCH_Test = subset(CLCH, Year > 2006)

Model1 = lm(Temp ~ MEI + CO2 + CH4 + N2O + CFC.11 + CFC.12 + TSI + Aerosols, data=CLCH_Train)
summary(Model1)


### PROBLEM 2.1 - UNDERSTANDING THE MODEL
## Current scientific opinion is that nitrous oxide and CFC-11 are greenhouse gases: gases that are able to trap heat from the sun and contribute to the heating of the Earth. However, the regression coefficients of both the N2O and CFC-11 variables are negative, indicating that increasing atmospheric concentrations of either of these two compounds is associated with lower global temperatures.
## Which of the following is the simplest correct explanation for this contradiction?
## All of the gas concentration variables reflect human development - N2O and CFC.11 are correlated with other variables in the data set.


### PROBLEM 2.2 - UNDERSTANDING THE MODEL
cor(CLCH_Train)


### PROBLEM 3 - SIMPLIFYING THE MODEL
Model2 = lm(Temp ~ MEI + TSI + Aerosols + N2O, data=CLCH_Train)
summary(Model2) # Enter the coefficient of N2O in this reduced model: (How does this compare to the coefficient in the previous model with all of the variables?), Enter the model R2:



# STEP함수를 사용하여 변수들의 최적의 조합을 찾자.
# AIC(Akaike information criterion)를 기준으로 model simplicty와 R2의 타협점을 찾아 최적의 조합을 찾는다.
# AIC는 모델의 질을 측정하는 척도로 변수들의 개수도 참고함.
# STEP함수를 사용하면 최소한 insignificant한 변수들은 삭제할 수 있음.

### PROBLEM 4 - AUTOMATICALLY BUILDING THE MODEL
## We have many variables in this problem, and as we have seen above, dropping some from the model does not decrease model quality. R provides a function, step, that will automate the procedure of trying different combinations of variables to find a good compromise of model simplicity and R2. This trade-off is formalized by the Akaike information criterion (AIC) - it can be informally thought of as the quality of the model with a penalty for the number of variables in the model.
## The step function has one argument - the name of the initial model. It returns a simplified model. Use the step function in R to derive a new model, with the full model as the initial model (HINT: If your initial full model was called "climateLM", you could create a new model with the step function by typing step(climateLM). Be sure to save your new model to a variable name so that you can look at the summary. For more information about the step function, type ?step in your R console.)
STEP_Model1 = step(Model1)
summary(STEP_Model1) # Which of the following variable(s) were eliminated from the full model by the step function? Select all that apply.
## It is interesting to note that the step function does not address the collinearity of the variables, except that adding highly correlated variables will not improve the R2 significantly. The consequence of this is that the step function will not necessarily produce a very interpretable model - just a model that has balanced quality and simplicity for a particular weighting of quality and simplicity (AIC).


### PROBLEM 5 - TESTING ON UNSEEN DATA
## We have developed an understanding of how well we can fit a linear regression to the training data, but does the model quality hold when applied to unseen data? Using the model produced from the step function, calculate temperature predictions for the testing data set, using the predict function.
predictTest = predict(STEP_Model1, newdata=CLCH_Test)
summary(predictTest)

# compute R^2
SSE = sum((CLCH_Test$Temp - predictTest)^2)
SST = sum((CLCH_Test$Temp - mean(CLCH_Train$Temp))^2)
1 - SSE/SST




