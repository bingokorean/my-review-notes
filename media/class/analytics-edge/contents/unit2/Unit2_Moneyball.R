# VIDEO 2

# Read in data
baseball = read.csv("baseball.csv")
str(baseball)

# Subset to only include moneyball years
moneyball = subset(baseball, Year < 2002)
str(moneyball)

# Compute Run Difference가
moneyball$RD = moneyball$RS - moneyball$RA # difference를 이용한 새로운 변수 추가
str(moneyball)

# Scatterplot to check for linear relationship
# scatter plot의 결과로 보아 두 변수는 서로 strong relationship을 가짐.
plot(moneyball$RD, moneyball$W)

# Regression model to predict wins

# 모델이 학습이 잘 되었는지 어느 정도 평가를 Multiple R-sqaured와 Adjusted R-squared를 통해 할 수 있음.
# 여기서는 각각 0.8808과 0.8807로 비교적 높은 점수를 얻었으므로 strong model이라고 평가할 수 있음.
WinsReg = lm(W ~ RD, data=moneyball)
summary(WinsReg)





# VIDEO 3

str(moneyball)

# Regression model to predict runs scored
RunsReg = lm(RS ~ OBP + SLG + BA, data=moneyball)
# 아래 coefficient를 보면 BA가 음수의 큰 수를 가지는데, 이는 기존의 상식과 맞지 않은 결과이다.
# Batting Average인데, Run scored와 negative ? -> counterintuitive함.
# 이처럼 기존 상식과 맞지 않은 경우 multicollinearity를 의심해볼 필요가 있다.
summary(RunsReg)

# BA를 삭제해보자.
RunsReg = lm(RS ~ OBP + SLG, data=moneyball)
# 독립 변수들의 중요도와 두 가지 r-squared 점수가 이전과 비슷하다
# 하지만, 변수가 3개에서 2개로 줄었기 때문에 이전보다 더 좋은 모델이라 볼 수 있다
# SLG보다는 OBP가 coefficient 수치가 더 크므로 더 중요한 변수라 할 수 있다
summary(RunsReg)


# By using linear regression,
# we're able to verify the claims made in Moneyball
# (1). Batting Average (BA) is overvalued (과대평가)
# (2). OBP is the most important
# (3). SLG is important for predicting run scored
