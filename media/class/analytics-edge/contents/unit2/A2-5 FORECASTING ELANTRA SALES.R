# 미국에서 현대 엘란트라 차를 판매율 높이기 위한 분석
# 일정 기간에 회사가 상품을 효율적으로 판매하기 위해서는
# 고객이 구입하는 상품들 보다, 공장에서 생산되는 상품이 많으면, 회사에서 손실 발생함
# 공장에서 생산되는 상품보다 고객이 원하는 구입량이 더 많으면, 회사에서 손실 발생함
# 따라서, 공장에서 생산이 들어가기 전에 고객의 구입량을 미리 예측하는 것이 중요함
# 이 문제에서는 미국에 있는 현대 엘란트라의 monthly sales를 예측하고자 함.
# 데이터로 economic indicators of US와 Google search queries를 사용함.

## Variable Description
# Month = the month of the year for the observation (1 = January, 2 = February, 3 = March, ...).
# Year = the year of the observation.
# ElantraSales = the number of units of the Hyundai Elantra sold in the United States in the given month.
# Unemployment = the estimated unemployment percentage in the United States in the given month.
# Queries = a (normalized) approximation of the number of Google searches for "hyundai elantra" in the given month.
# CPI_energy = the monthly consumer price index (CPI) for energy for the given month.
# CPI_all = the consumer price index (CPI) for all products for the given month; this is a measure of the magnitude of the prices paid by consumer households for goods and services (e.g., food, clothing, electricity, etc.).





### PROBLEM 1 - LOADING THE DATA
###
Elantra = read.csv("elantra.csv")
ElantraTrain = subset(Elantra, Year <= 2012)
ElantraTest = subset(Elantra, Year > 2012)


### PROBLEM 2.1 - A LINEAR REGRESSION MODEL 
###
# Mutiple과 adjusted R2의 점수들이 낮은 것으로 보아 MODEL을 improve할 필요성이 있다.
ElantraLM = lm(ElantraSales ~ Unemployment + Queries + CPI_energy + CPI_all, data=ElantraTrain)
summary(ElantraLM)


### PROBLEM 3.1 - MODELING SEASONALITY  
###

# 새로운 변수를 추가해보자.
# seasonality를 반영하는 새로운 변수 month를 추가.

# 오히려 adjusted R2 점수가 감소했음.
## Our model R-Squared is relatively low, so we would now like to improve our model. In modeling demand and sales, it is often useful to model seasonality. Seasonality refers to the fact that demand is often cyclical/periodic in time. For example, in countries with different seasons, demand for warm outerwear (like jackets and coats) is higher in fall/autumn and winter (due to the colder weather) than in spring and summer. (In contrast, demand for swimsuits and sunscreen is higher in the summer than in the other seasons.) Another example is the "back to school" period in North America: demand for stationary (pencils, notebooks and so on) in late July and all of August is higher than the rest of the year due to the start of the school year in September.
## In our problem, since our data includes the month of the year in which the units were sold, it is feasible for us to incorporate monthly seasonality. From a modeling point of view, it may be reasonable that the month plays an effect in how many Elantra units are sold.
## To incorporate the seasonal effect due to the month, build a new linear regression model that predicts monthly Elantra sales using Month as well as Unemployment, CPI_all, CPI_energy and Queries. Do not modify the training and testing data frames before building the model.
ElantraLM = lm(ElantraSales ~ Unemployment + Queries + CPI_energy + CPI_all + Month, data=ElantraTrain)
summary(ElantraLM)


### PROBLEM 3.2 - EFFECT OF ADDING A NEW VARIABLE 
###

# Multiple R2 점수만 바라보면, overfitting되기 쉽다. 

# Which of the following best describes the effect of adding Month?
# 2. The model is not better because the adjusted R-squared has gone down and none of the variables (including the new one) are very significant. 
#@ The second option is correct: the adjusted R-Squared is the R-Squared but adjusted to take into account the number of variables. If the adjusted R-Squared is lower, then this indicates that our model is not better and in fact may be worse. Furthermore, if none of the variables have become significant, then this also indicates that the model is not better.
# Incorrect examples
# 1. The model is better because the R-squared has increased. -> The first option is incorrect because (ordinary) R-Squared always increases (or at least stays the same) when you add new variables. This does not make the model better, and in fact, may hurt the ability of the model to generalize to new, unseen data (overfitting).
# 3. The model is better because the p-values of the four previous variables have decreased (they have become more significant). -> The third option is not correct because as stated above, the adjusted R-Squared has become worse. Although the variables have come closer to being significant, this doesn't make it a better model.
# 4. The model is not better because it has more variables. -> The fourth option is not correct. Although it is desirable to have models that are parsimonious (fewer variables), we are ultimately interested in models that have high explanatory power (as measured in training R-Squared) and out of sample predictive power (as measured in testing R-Squared). Adding a key variable may significantly improve the predictive power of the model, and we should thus not dismiss the model simply because it has more variables.


### PROBLEM 3.3 - UNDERSTANDING THE MODEL  
# In the new model, given two monthly periods that are otherwise identical in Unemployment, CPI_all, CPI_energy and Queries, what is the absolute difference in predicted Elantra sales given that one period is in January and one is in March?
110.69 * (3 - 1)  # 다른 변수들은 constant라 가정

# In the new model, given two monthly periods that are otherwise identical in Unemployment, CPI_all, CPI_energy and Queries, what is the absolute difference in predicted Elantra sales given that one period is in January and one is in May?
110.69 * (5 - 1)


### PROBLEM 3.4 - NUMERIC VS. FACTORS
###
## You may be experiencing an uneasy feeling that there is something not quite right in how we have modeled the effect of the calendar month on the monthly sales of Elantras. If so, you are right. In particular, we added Month as a variable, but Month is an ordinary numeric variable. In fact, we must convert Month to a factor variable before adding it to the model.
# What is the best explanation for why we must do this?
# By modeling Month as a factor variable, the effect of each calendar month is not restricted to be linear in the numerical coding of the month. 
#@ The second choice is the correct answer. The previous subproblem essentially showed that for every month that we move into the future (e.g, from January to February, from February to March, etc.), our predicted sales go up by 110.69. This isn't right, because the effect of the month should not be affected by the numerical coding, and by modeling Month as a numeric variable, we cannot capture more complex effects. For example, suppose that when the other variables are fixed, an additional 500 units are sold from June to December, relative to the other months. This type of relationship between the boost to the sales and the Month variable would look like a step function at Month = 6, which cannot be modeled as a linear function of Month.
#@ 1~12 고유의 month의미를 여기서는 1~12라는 value값으로 인식하여 model에 적용하기 때문에 month변수는 factor변수로 변환할 필요가 있다.


### PROBLEM 4.1 - A NEW MODEL  
###
# month변ㅅ를 factor로 만들어 binary화 하였다.
# R2 점수들이 높아진 것을 확인할 수 있다.
ElantraTrain$MonthFactor = as.factor(ElantraTrain$Month)
ElantraTest$MonthFactor = as.factor(ElantraTest$Month)
ElantraLM = lm(ElantraSales ~ Unemployment + Queries + CPI_energy + CPI_all + MonthFactor, data=ElantraTrain)
summary(ElantraLM)

summary(ElantraTrain)
str(ElantraTrain)


### PROBLEM 5.1 - MULTICOLINEARITY
###
## Another peculiar observation about the regression is that the sign of the Queries variable has changed. In particular, when we naively modeled Month as a numeric variable, Queries had a positive coefficient. Now, Queries has a negative coefficient. Furthermore, CPI_energy has a positive coefficient -- as the overall price of energy increases, we expect Elantra sales to increase, which seems counter-intuitive (if the price of energy increases, we'd expect consumers to have less funds to purchase automobiles, leading to lower Elantra sales).
## Model의 어느 2개의 coefficient 부호(+/-)가 counter-intuitive한다면, multicolinearity를 의심해볼 수 있다.
## As we have seen before, changes in coefficient signs and signs that are counter to our intuition may be due to a multicolinearity problem. To check, compute the correlations of the variables in the training set.
# Which of the following variables is CPI_energy highly correlated with? Select all that apply. (Include only variables where the absolute value of the correlation exceeds 0.6. For the purpose of this question, treat Month as a numeric variable, not a factor variable.)
cor(ElantraTrain[c("Unemployment","Month","Queries","CPI_energy","CPI_all")])


# multiplinearity를 의심하여 independent 변수들 사이의 correlation을 비교해보니,
# 의외로 많은 변수들이 서로 highly correlated된 상태였다.
# 따라서, 한 번에 하나의 변수를 제거하면서 모델을 simplify할 필요가 있다.

### PROBLEM 5.2 - CORRELATIONS 
### 
# Which of the following variables is Queries highly correlated with? Again, compute the correlations on the training set. Select all that apply. (Include only variables where the absolute value of the correlation exceeds 0.6. For the purpose of this question, treat Month as a numeric variable, not a factor variable.)
cor(ElantraTrain[c("Unemployment","Month","Queries","CPI_energy","CPI_all")])
## Based on these results, we can see that (somewhat surprisingly) there are many variables highly correlated with each other; as a result, the sign change of Queries is likely to be due to multicolinearity.


### PROBLEM 6.1 - A REDUCED MODEL  
## Let us now simplify our model (the model using the factor version of the Month variable). We will do this by iteratively removing variables, one at a time. Remove the variable with the highest p-value (i.e., the least statistically significant variable) from the model. Repeat this until there are no variables that are insignificant or variables for which all of the factor levels are insignificant. Use a threshold of 0.10 to determine whether a variable is significant.
#@ The variable with the highest p-value is "Queries". After removing it and looking at the model summary again, we can see that there are no variables that are insignificant, at the 0.10 p-level. Note that Month has a few values that are insignificant, but we don't want to remove it because many values are very significant.
ElantraLM = lm(ElantraSales ~ Unemployment + CPI_energy + CPI_all + MonthFactor, data=ElantraTrain)
summary(ElantraLM)


### PROBLEM 6.2 - TEST SET PREDICTIONS  
PredictTest = predict(ElantraLM, newdata=ElantraTest)
SSE = sum((PredictTest - ElantraTest$ElantraSales)^2)
SSE


### PROBLEM 6.3 - COMPARING TO A BASELINE  
## What would the baseline method predict for all observations in the test set? Remember that the baseline method we use predicts the average outcome of all observations in the training set.
mean(ElantraTrain$ElantraSales)
#@ The baseline method that is used in the R-Squared calculation (to compute SST, the total sum of squares) simply predicts the mean of ElantraSales in the training set for every observation (i.e., without regard to any of the independent variables).


### PROBLEM 6.4 - TEST SET R-SQUARED  
SST = sum((mean(ElantraTrain$ElantraSales) - ElantraTest$ElantraSales)^2)
1-(SSE/SST)


### PROBLEM 6.5 - ABSOLUTE ERRORS  
max(abs(PredictTest - ElantraTest$ElantraSales))

### PROBLEM 6.6 - MONTH OF LARGEST ERROR  
# In which period (Month,Year pair) do we make the largest absolute error in our prediction?
which.max(abs(PredictTest - ElantraTest$ElantraSales))
#@ This returns 5, which is the row number in ElantraTest corresponding to the period for which we make the largest absolute error.

