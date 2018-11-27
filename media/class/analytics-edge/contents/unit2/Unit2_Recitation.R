### Moneyball for NBA
# the goal of basketball team is similar to that of baseball team, making the playoffs.

### Variable Description
# PTS: points scored during the regular season
# FG: the number of successful field goals (2,3점 모두 포함)
# FGA: the number of field goal attempts
# FT: the number of successful free throws
# ORB: offensive rebounds
# DRB: deffensive rebounds
# AST: assists
# STL: steels
# BLK: blocks
# TOV: turn overs

# (참고) 원본 엑셀 파일에서 2P, 2PA, 3P와 같이 숫자로 시작하는 변수들은 R로 로딩하면서 자동적으로 맨 앞에 X를 붙여줌.


# VIDEO 1

# Read in the data
NBA = read.csv("NBA_train.csv")
str(NBA)


# VIDEO 2

# How many wins to make the playoffs?
# 아래의 테이블을 보아하니, playoff에 가기 위해서는 대략저으로 42 게임 정도는 이겨야 되는 듯 하다.
table(NBA$W, NBA$Playoffs)

# Compute Points Difference
NBA$PTSdiff = NBA$PTS - NBA$oppPTS

# Check for linear relationship
# 아래와 같이 W와 PTFdiff와의 관계는 strong linear 관계를 가짐.
plot(NBA$PTSdiff, NBA$W)

# Linear regression model for wins
WinsReg = lm(W ~ PTSdiff, data=NBA)

# 변수들이 모두 sinificant하고, Multiple R-squared와 Adjusted R-squared 점수가 모두 매우 높은 것을 보아 모델이 very strong하다는 걸 알 수 있음.
# 아래의 학습된 coefficient들의 수치들을 토대로 대략적으로 다음과 같은 결론을 내릴 수 있다
# W = 41 + 0.0326 * PTSdiff
# 우리가 42 게임을 이기기 위해서는, 대략적으로 31점을 더 얻어야 한다.
# -> PTFdiff >= (42-41)/0.0326 = 30.67
summary(WinsReg)


# VIDEO 3

# PTS가 W에 지대한 영향을 끼치므로, 이제부터는 PTS를 종속 변수로 두고 분석해보자.
# 예측을 위한 것이 아닌, 변수들 사이 상관성을 이해하기 위한 분석이다 (통계학과 기계학습의 차이점)

# Linear regression model for points scored
PointsReg = lm(PTS ~ X2PA + X3PA + FTA + AST + ORB + DRB + TOV + STL + BLK, data=NBA)
summary(PointsReg)

# Sum of Squared Errors
PointsReg$residuals
SSE = sum(PointsReg$residuals^2)
SSE

# 수치상으로 해석/이해할 때는 SSE보다는 RMSE가 더 좋음.
# SSE는 quite a lot하기 때문에 그다지 interpretable하지 않음.
# RMSE는 average error로서 훨씬 더 interpretable하다.

# Root mean squared error
RMSE = sqrt(SSE/nrow(NBA))
RMSE

# 평균 8370 에서 평균 184 error면, 그다지 나쁘지 않은 모델이다.

# Average number of points in a season
mean(NBA$PTS)

# 자 이제, 중요하지 않은 변수들을 삭제해나가보자.
# TOV의 P-value가 제일 높기 때문에 1순위로 삭제해야 한다.

# Remove insignifcant variables
summary(PointsReg)

PointsReg2 = lm(PTS ~ X2PA + X3PA + FTA + AST + ORB + DRB + STL + BLK, data=NBA) # TOV를 삭제
# TOV를 삭제하기 전과 후의 Multiple R-square와 adjusted R-squared 점수가 거의 같다 -> 그만큼 TOV 변수가 쓸모없다는 뜻.
summary(PointsReg2)

PointsReg3 = lm(PTS ~ X2PA + X3PA + FTA + AST + ORB + STL + BLK, data=NBA) # TOV와 DRB를 삭제
# TOV와 DRB를 삭제해도 여전히 multiple, adjusted r-sqaured 점수가 같다.
summary(PointsReg3)

PointsReg4 = lm(PTS ~ X2PA + X3PA + FTA + AST + ORB + STL, data=NBA)
# 이제 모든 변수가 significant하다.
# 여전히 multiple, adjusted r-sqaured 점수가 같다.
# 그러나 쓸데없는 독립변수들을 삭제하였다 -> simple할수록 모델은 좋아진다.
# it seems like we've narrowed down on a much better model because it's simpler, it's more interpretable, and it's got just about the same amount of error.
summary(PointsReg4)

# Compute SSE and RMSE for new model
# 변수들을 삭제한 이후로 RMSE가 아주 조금 증가하였다 (거의 똑같다)
SSE_4 = sum(PointsReg4$residuals^2)
RMSE_4 = sqrt(SSE_4/nrow(NBA))
SSE_4
RMSE_4




# VIDEO 4

# training set : 1980 ~ 2011
# testing set : 2012, 2013

# Read in test set
NBA_test = read.csv("NBA_test.csv")

# Make predictions on test set
PointsPredictions = predict(PointsReg4, newdata=NBA_test)

# R2 (out-of-sample = test set) : 0.8127
# Multiple R2 (insample = training set) : 0.8991
# 위의 둘은 거의 비슷하다. 따라서, 좋은 모델이라고 볼 수 있다.
# RMSE도 조금 증가했지만, 나쁘지 않다.

# Compute out-of-sample R^2
SSE = sum((PointsPredictions - NBA_test$PTS)^2)
SST = sum((mean(NBA$PTS) - NBA_test$PTS)^2)
R2 = 1 - SSE/SST
R2

# Compute the RMSE
RMSE = sqrt(SSE/nrow(NBA_test))
RMSE
