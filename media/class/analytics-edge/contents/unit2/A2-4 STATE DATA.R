# "state" dataset from the 1970s on all fifty US states
# for each state, dataset includes the population, per capita income, illiteracy rate, murder rate, high schoold grauation rate, average number of frost days, area, latitude and longitude, division the state belongs to, region the state belongs to, and two-letter abbreviation


# This dataset has 50 observations (one for each US state) and the following 15 variables:
  
# Population - the population estimate of the state in 1975
# Income - per capita income in 1974
# Illiteracy - illiteracy rates in 1970, as a percent of the population
# Life.Exp - the life expectancy in years of residents of the state in 1970
# Murder - the murder and non-negligent manslaughter rate per 100,000 population in 1976 
# HS.Grad - percent of high-school graduates in 1970
# Frost - the mean number of days with minimum temperature below freezing from 1931???1960 in the capital or a large city of the state
# Area - the land area (in square miles) of the state
# state.abb - a 2-letter abreviation for each state
# state.area - the area of each state, in square miles
# x - the longitude of the center of the state
# y - the latitude of the center of the state
# state.division - the division each state belongs to (New England, Middle Atlantic, South Atlantic, East South Central, West South Central, East North Central, West North Central, Mountain, or Pacific)
# state.name - the full names of each state
# state.region - the region each state belong to (Northeast, South, North Central, or West)



### PROBLEM 1.1 - DATA EXPLORATION
statedata = read.csv("statedata.csv")
str(statedata)
plot( statedata$x, statedata$y)
# Using the tapply command, determine which region of the US (West, North Central, South, or Northeast) has the highest average high school graduation rate of all the states in the region:
tapply(statedata$HS.Grad, statedata$state.region, mean) # West
# Now, make a boxplot of the murder rate by region (for more information about creating boxplots in R, type ?boxplot in your console).
boxplot(statedata$Murder ~ statedata$state.region) # South 
# You should see that there is an outlier in the Northeast region of the boxplot you just generated. Which state does this correspond to? (Hint: There are many ways to find the answer to this question, but one way is to use the subset command to only look at the Northeast data.)
NortheastData = subset(statedata, state.region == "Northeast")
boxplot(NortheastData$Murder ~ NortheastData$state.abb)


### PROBLEM 2.1 - PREDICTING LIFE EXPECTANCY - AN INITIAL MODEL  
LinReg = lm(Life.Exp ~ Population + Income + Illiteracy + Murder + HS.Grad + Frost + Area, data=statedata)
summary(LinReg)
plot(statedata$Income, statedata$Life.Exp) #  Life expectancy is somewhat positively correlated with income. 
## The model we built does not display the relationship we saw from the plot of life expectancy vs. income. Which of the following explanations seems the most reasonable?
# Multicollinearity
# Although income is an insignificant variable in the model, this does not mean that there is no association between income and life expectancy. However, in the presence of all of the other variables, income does not add statistically significant explanatory power to the model. This means that multicollinearity is probably the issue.
# 즉, Life.Exp와 income은 correlation관계이지만, income이 insignificant하게 나왔다. income이 다른 변수에 의해서 confounding됬기 떄문에..  


### PROBLEM 3.1 - PREDICTING LIFE EXPECTANCY - REFINING THE MODEL AND ANALYZING PREDICTIONS
## Recall that we discussed the principle of simplicity: that is, a model with fewer variables is preferable to a model with many unnnecessary variables. Experiment with removing independent variables from the original model. Remember to use the significance of the coefficients to decide which variables to remove (remove the one with the largest "p-value" first, or the one with the "t value" closest to zero), and to remove them one at a time (this is called "backwards variable selection"). This is important due to multicollinearity issues - removing one insignificant variable may make another previously insignificant variable become significant.
LinReg = lm(Life.Exp ~ Population + Income + Illiteracy + Murder + HS.Grad + Frost, data=statedata)
summary(LinReg)
LinReg = lm(Life.Exp ~ Population + Income + Murder + HS.Grad + Frost, data=statedata)
summary(LinReg)
LinReg = lm(Life.Exp ~ Population + Murder + HS.Grad + Frost, data=statedata)
summary(LinReg)
# Life.Exp와 income이 correlation하지만, 결국 제거했다. 다른 변수에 의해 income의 correlation은 충분히 설명되기 때문..
## This model with 4 variables is a good model. However, we can see that the variable "Population" is not quite significant. In practice, it would be up to you whether or not to keep the variable "Population" or eliminate it for a 3-variable model. Population does not add much statistical significance in the presence of murder, high school graduation rate, and frost days. However, for the remainder of this question, we will analyze the 4-variable model.


### PROBLEM 3.2 - PREDICTING LIFE EXPECTANCY - REFINING THE MODEL AND ANALYZING PREDICTIONS
## Removing insignificant variables changes the Multiple R-squared value of the model. By looking at the summary output for both the initial model (all independent variables) and the simplified model (only 4 independent variables) and using what you learned in class,
#  which of the following correctly explains the change in the Multiple R-squared value?
#  We expect the "Multiple R-squared" value of the simplified model to be slightly worse than that of the initial model. It can't be better than the "Multiple R-squared" value of the initial model.
## When we remove insignificant variables, the "Multiple R-squared" will always be worse, but only slightly worse. This is due to the nature of a linear regression model. It is always possible for the regression model to make a coefficient zero, which would be the same as removing the variable from the model. The fact that the coefficient is not zero in the intial model means it must be helping the R-squared value, even if it is only a very small improvement. So when we force the variable to be removed, it will decrease the R-squared a little bit. However, this small decrease is worth it to have a simpler model. 
## On the contrary, when we remove insignificant variables, the "Adjusted R-squred" will frequently be better. This value accounts for the complexity of the model, and thus tends to increase as insignificant variables are removed, and decrease as insignificant variables are added.


### PROBLEM 3.3 - PREDICTING LIFE EXPECTANCY - REFINING THE MODEL AND ANALYZING PREDICTIONS
sort(predict(LinReg))
which.min(statedata$Life.Exp)
statedata$state.name[40]


### PROBLEM 3.4 - PREDICTING LIFE EXPECTANCY - REFINING THE MODEL AND ANALYZING PREDICTIONS
sort(predict(LinReg))
which.max(statedata$Life.Exp)
statedata$state.name[11]


## PROBLEM 3.5 - PREDICTING LIFE EXPECTANCY - REFINING THE MODEL AND ANALYZING PREDICTIONS
# Take a look at the vector of residuals (the difference between the predicted and actual values). For which state do we make the smallest absolute error?
sort(abs(LinReg$residuals))
sort(abs(statedata$Life.Exp - predict(LinReg)))





