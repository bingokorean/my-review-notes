# VIDEO 4

# Read in data
wine = read.csv("wine.csv")

# price: 종속변수 
str(wine)
summary(wine)

# Linear Regression (one variable)
model1 = lm(Price ~ AGST, data=wine)

#model33 = lm(Price ~ HarvestRain + WinterRain, data=wine)

# Estimate Std. : estimates of the beta values of our model
# Multiple R-sqaured : will always increase if you add more independent variables
# Adjusted R-squared : will decrease if you add an independent variable, that doesn't help the model
# Adjusted R-squared value adjusts the R-squared value to account for the number of independent variables used relative to the number of data points
# Adjusted R-squared is a good way to determine if an additional variable should even be included in the model
summary(model1)

# Sum of Squared Errors
model1$residuals
SSE = sum(model1$residuals^2)
SSE

# Linear Regression (two variables)
model2 = lm(Price ~ AGST + HarvestRain, data=wine)

# 1개의 변수를 추가하니까 Multiple R-squared와 Adjusted R-quared가 모두 증가함.
# 이는 추가된 변수가 우리의 모델에 큰 도움을 줬다는 뜻임.
summary(model2)

# Sum of Squared Errors
# SSE 또한 지난 모델보다 줄어들어 좋아졌음.
SSE = sum(model2$residuals^2)
SSE

# Linear Regression (all variables)
# 모든 변수를 사용하니, 또 증가함.
# 그러나 summary를 보아하니 insignificant 변수들이 보임 -> Age, FrancePop
model3 = lm(Price ~ AGST + HarvestRain + WinterRain + Age + FrancePop, data=wine)
summary(model3)

# Sum of Squared Errors
SSE = sum(model3$residuals^2)
SSE


# VIDEO 5

# 별로 중요하지 않은 변수를 삭제해보자.
# Remove FrancePop 
model4 = lm(Price ~ AGST + HarvestRain + WinterRain + Age, data=wine)
# 주목할 점은 FrancePop 변수를 삭제하기 전까지 Age도 insignificant했지만, FrancePop을 삭제하니까 Age가 **로 very significantr=가 되었다.
# 이러한 현상은 "multicollinearity" 때문이다. 즉, Age와 FrancePop은 서로 highly correlated된 상태임.
# 중요하지 않은 FrancePop 변수를 삭제함으로써, Multiple R-squared는 조금 감소 (어찌되었든 변수1개를 삭제했으므로) 그리고 Adjusted R-squared가 조금 증가 (이제 중요한 알짜베기 변수들만 남아서) 하였다.
summary(model4)


# VIDEO 6

# Correlations
cor(wine$WinterRain, wine$Price)
cor(wine$Age, wine$FrancePop)

# "Multicollinearity Problem"
# the situation when two independent variables are highly correlated
# due to this problem, we need to remove the insignificant variables one at a time (한 번에 한 개씩 삭제 요망!)

# covariance matrix를 통해 이러한 multicollinerity 문제를 확인할 수 있음.
# high correlation의 기준은 특별히 없지만 보통 절대값 0.7이상인 경우를 말한다.
# 중요한 점은 종속 변수와 독립 변수 사이의 높은 상관성은 좋은 것이다. 반면, 독립 변수들 사이의 높은 상관성은 좋지 않다.
cor(wine)


# Age와 FrancePop에 따른 성능 변화
#                      둘다보존 / Age삭제 / FrancePop삭제 / 둘다삭제
# Multiple R-squared   0.8294   / 0.8294  / 0.8286        / 0.7537
# Adjusted R-squared   0.7845   / 0.7952  / 0.7943        / 0.7185

# 위의 결과로 봤을 때 Age만 삭제하는 것이 제일 큰 효용성을 가짐.
# FrancePop 변수만 삭제했을 때보다 둘다삭제했을 때 R-qaured가 감소했다 -> Age가 significant하다는 증거
# 위와 같이 모든 경우의 수를 다 실험해보기 어려운 상황이면, 외적인 정보가 들어가서 가정에 의한 변수 선택을 하기도 함.
# 예) older wines are typically more expensive, so Age makes more intuitive sense in our model.

# Remove Age and FrancePop
model5 = lm(Price ~ AGST + HarvestRain + WinterRain, data=wine)
summary(model5)

model55 = lm(Price ~ AGST + HarvestRain + WinterRain + FrancePop, data=wine)
summary(model55)
# VIDEO 7

# Read in test set
wineTest = read.csv("wine_test.csv")
str(wineTest)

# Make test set predictions
predictTest = predict(model4, newdata=wineTest)
predictTest

# Compute R-squared
SSE = sum((wineTest$Price - predictTest)^2)
SST = sum((wineTest$Price - mean(wine$Price))^2)
# 높은 점수이지만, new data에 대해 신뢰성을 높이려면 test set의 size를 늘려야 함.
1 - SSE/SST

