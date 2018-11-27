# 검색엔진 쿼리 데이터를 활용한 독감(flu) 전염병 탐지
# CDC (U.S Center for Disease Control and Prevention) and European Influenza Srveillance Scheme (EISS)
### Influenza-like Illness (ILI) physician visits (published 1-2 week lag)
# Google Flu Trends project
### flu-related online search queries를 활용하여 report를 매우 빠르게 작성.

# Week
# ILI
# Quries




### PROBLEM 1.1 - UNDERSTANDING THE DATA
FluTrain = read.csv("FluTrain.csv")
summary(FluTrain)

# Looking at the time period 2004-2011, which week corresponds to the highest percentage of ILI-related physician visits? Select the day of the month corresponding to the start of this week.
subset(FluTrain, ILI == max(ILI))
## We can limit FluTrain to the observations that obtain the maximum ILI value with subset(FluTrain, ILI == max(ILI)). From here, we can read information about the week at which the maximum was obtained. Alternatively, you can use which.max(FluTrain$ILI) to find the row number corresponding to the observation with the maximum value of ILI, which is 303. Then, you can output the corresponding week using FluTrain$Week[303].
which.max(FluTrain$ILI)
FluTrain$Week[303]
# Which week corresponds to the highest percentage of ILI-related query fraction?
subset(FluTrain, Queries == max(Queries))
which.max(FluTrain$Queries)
FluTrain$Week[303]

# 데이터 분포가 skewed 하다.

### PROBLEM 1.2 - UNDERSTANDING THE DATA
hist(FluTrain$ILI)



# 만약 skewed한 분포의 독립 변수가 있다면, logarithm을 취하는 게 좋다.
# 왜냐면 데이터가 많은 쪽의 SSE가 상대적으로 상당히 크기 때문이다.

### PROBLEM 1.3 - UNDERSTANDING THE DATA
## When handling a skewed dependent variable, it is often useful to predict the logarithm of the dependent variable instead of the dependent variable itself -- this prevents the small number of unusually large or small observations from having an undue influence on the sum of squared errors of predictive models. In this problem, we will predict the natural log of the ILI variable, which can be computed in R using the log() function.
plot(FluTrain$Queries, (FluTrain$ILI))
plot(FluTrain$Queries, log(FluTrain$ILI))


### PROBLEM 2.1 - LINEAR REGRESSION MODEL
## Based on the plot we just made, it seems that a linear regression model could be a good modeling choice. Based on our understanding of the data from the previous subproblem, which model best describes our estimation problem?
## -> log(ILI) = intercept + coefficient x Queries, where the coefficient is positive


### PROBLEM 2.2 - LINEAR REGRESSION MODEL
FluTrend1 = lm(log(ILI) ~ Queries ,data=FluTrain) # log.
summary(FluTrend1)

# log(ILI)와 Queries 사이의 correlation 제곱은 위 모델에서 Multiple R2과 같다.

## For a single variable linear regression model, there is a direct relationship between the R-squared and the correlation between the independent and the dependent variables. What is the relationship we infer from our problem? (Don't forget that you can use the cor function to compute the correlation between two variables.)
Correlation = cor(FluTrain$Queries, log(FluTrain$ILI))
Correlation
##To test these hypotheses, we first need to compute the correlation between the independent variable used in the model (Queries) and the dependent variable (log(ILI)). This can be done with
#Correlation = cor(FluTrain$Queries, log(FluTrain$ILI))
# The values of the three expressions are then:
# Correlation^2 = 0.7090201
# log(1/Correlation) = 0.1719357
# exp(-0.5*Correlation) = 0.6563792
# It appears that Correlation^2 is equal to the R-squared value. It can be proved that this is always the case.


### PROBLEM 3.1 - PERFORMANCE ON THE TEST SET
FluTest = read.csv("FluTest.csv")
## The csv file FluTest.csv provides the 2012 weekly data of the ILI-related search queries and the observed weekly percentage of ILI-related physician visits. Load this data into a data frame called FluTest.
## Normally, we would obtain test-set predictions from the model FluTrend1 using the code
## PredTest1 = predict(FluTrend1, newdata=FluTest)
## However, the dependent variable in our model is log(ILI), so PredTest1 would contain predictions of the log(ILI) value. We are instead interested in obtaining predictions of the ILI value. We can convert from predictions of log(ILI) to predictions of ILI via exponentiation, or the exp() function. The new code, which predicts the ILI value, is
PredTest1 = exp(predict(FluTrend1, newdata=FluTest))
# What is our estimate for the percentage of ILI-related physician visits for the week of March 11, 2012? (HINT: You can either just output FluTest$Week to find which element corresponds to March 11, 2012, or you can use the "which" function in R. To learn more about the which function, type ?which in your R console.)
which(FluTest$Week == "2012-03-11 - 2012-03-17")
PredTest1[11]

## What is the relative error betweeen the estimate (our prediction) and the observed value for the week of March 11, 2012? Note that the relative error is calculated as
## (Observed ILI - Estimated ILI)/Observed ILI
(FluTest$ILI[11] - PredTest1[11])/FluTest$ILI[11]

SSE = sum((PredTest1 - FluTest$ILI)^2)
RMSE = sqrt(SSE/nrow(FluTest))
RMSE


### PROBLEM 4.1 - TRAINING A TIME SERIES MODEL
## The observations in this dataset are consecutive weekly measurements of the dependent and independent variables. This sort of dataset is called a "time series." Often, statistical models can be improved by predicting the current value of the dependent variable using the value of the dependent variable from earlier weeks. In our models, this means we will predict the ILI variable in the current week using values of the ILI variable from previous weeks.
## First, we need to decide the amount of time to lag the observations. Because the ILI variable is reported with a 1- or 2-week lag, a decision maker cannot rely on the previous week's ILI value to predict the current week's value. Instead, the decision maker will only have data available from 2 or more weeks ago. We will build a variable called ILILag2 that contains the ILI value from 2 weeks before the current observation.
## To do so, we will use the "zoo" package, which provides a number of helpful methods for time series models. While many functions are built into R, you need to add new packages to use some functions. New packages can be installed and loaded easily in R, and we will do this many times in this class. Run the following two commands to install and load the zoo package. In the first command, you will be prompted to select a CRAN mirror to use for your download. Select a mirror near you geographically.
install.packages("zoo")
library(zoo)
## After installing and loading the zoo package, run the following commands to create the ILILag2 variable in the training set:
ILILag2 = lag(zoo(FluTrain$ILI), -2, na.pad=TRUE)
## In these commands, the value of -2 passed to lag means to return 2 observations before the current one; a positive value would have returned future observations. The parameter na.pad=TRUE means to add missing values for the first two weeks of our dataset, where we can't compute the data from 2 weeks earlier.
summary(ILILag2)

plot(FluTrain$ILI, ILILag2)

FluTrend2 = lm(log(ILI) ~ Queries + log(ILILag2), data=FluTrain)
summary(FluTrend2)


### PROBLEM 5.1 - EVALUATING THE TIME SERIES MODEL IN THE TEST SET
## So far, we have only added the ILILag2 variable to the FluTrain data frame. To make predictions with our FluTrend2 model, we will also need to add ILILag2 to the FluTest data frame (note that adding variables before splitting into a training and testing set can prevent this duplication of effort).
# Modify the code from the previous subproblem to add an ILILag2 variable to the FluTest data frame. How many missing values are there in this new variable?
ILILag2 = lag(zoo(FluTest$ILI), -2, na.pad=TRUE)
FluTest$ILILag2 = coredata(ILILag2)
summary(FluTest$ILILag2)

## In this problem, the training and testing sets are split sequentially -- the training set contains all observations from 2004-2011 and the testing set contains all observations from 2012. There is no time gap between the two datasets, meaning the first observation in FluTest was recorded one week after the last observation in FluTrain. From this, we can identify how to fill in the missing values for the ILILag2 variable in FluTest.
# Which value should be used to fill in the ILILag2 variable for the first observation in FluTest?
# The ILI value of the second-to-last observation in the FluTrain data frame. The ILI value of the second-to-last observation in the FluTrain data frame. - correct
# Which value should be used to fill in the ILILag2 variable for the second observation in FluTest?
# The ILI value of the last observation in the FluTrain data frame.

## Fill in the missing values for ILILag2 in FluTest. In terms of syntax, you could set the value of ILILag2 in row "x" of the FluTest data frame to the value of ILI in row "y" of the FluTrain data frame with "FluTest$ILILag2[x] = FluTrain$ILI[y]". Use the answer to the previous questions to determine the appropriate values of "x" and "y". It may be helpful to check the total number of rows in FluTrain using str(FluTrain) or nrow(FluTrain).
# From nrow(FluTrain), we see that there are 417 observations in the training set. Therefore, we need to run the following two commands:
FluTest$ILILag2[1] = FluTrain$ILI[416]
FluTest$ILILag2[2] = FluTrain$ILI[417]
FluTest$ILILag2[1] # What is the new value of the ILILag2 variable in the first row of FluTest?
FluTest$ILILag2[2] # What is the new value of the ILILag2 variable in the second row of FluTest?

## Obtain test set predictions of the ILI variable from the FluTrend2 model, again remembering to call the exp() function on the result of the predict() function to obtain predictions for ILI instead of log(ILI).
# What is the test-set RMSE of the FluTrend2 model?
PredTest2 = exp(predict(FluTrend2, newdata=FluTest))
SSE = sum((PredTest2-FluTest$ILI)^2)
RMSE = sqrt(SSE / nrow(FluTest))
RMSE
sqrt(mean((PredTest2-FluTest$ILI)^2))

## The test-set RMSE of FluTrend2 is 0.294, as opposed to the 0.749 value obtained by the FluTrend1 model.
## In this problem, we used a simple time series model with a single lag term. ARIMA models are a more general form of the model we built, which can include multiple lag terms as well as more complicated combinations of previous values of the dependent variable. If you're interested in learning more, check out ?arima or the available online tutorials for these sorts of models.






