### By using all the statistical methods or data visualizations, 
### we should figure out the relationship between variables. 

# Data Load
eBayTrain = read.csv("eBayiPadTrain.csv", stringsAsFactors=FALSE)
# Duplications of training data
train0 = eBayTrain

# Identify the type of data
summary(eBayTrain)
str(eBayTrain)


########## Per one variable
# Analysis of continuous variable
hist(eBayiPadTrain$startprice, breaks=2000) # the distribution of continuous variables
hist(exp(-(eBayiPadTrain$startprice/20)))
boxplot(eBayiPadTrain$startprice)
summary(eBayiPadTrain$startprice)
# -> continuous 변수의 분포, statistic을 보면서, 크게 차이나는 구간이 있으면 clustering 여부도 고려해볼수 있다.
# -> distribution을 봐서 특별한 패턴이 있는지 ...  조금 sparse 한것 빼고는...

########## Per Two variables
# 모든 경우의 수를 이용해 변수들끼리 관계를 파악하긴 힘드므로, 직감적으로 가정을 정의하여 의미있을 법한 두가지 변수 선정 또는
# independent 변수와 dependent 변수를 중심적으로 비교해본다.
# 가정 1 : dependent 변수인 sold==1일때와 sold==0일때의 사이 다른 independent 변수에 의해 구별되는 패턴이 존재 할 것이다.
# Especially, examine the relationship between dependent and independent variable

### 모든 independent 변수를 다 사용할 것이 아니라, dependent variable과 correlation이 큰 변수들만 사용하자. (overfitting방지)
##### independent 변수와 dependent 변수와의 상관관계 파악
# sold <-> description
table((eBayTrain$description)=="", eBayTrain$sold) # destription있을 때 아주 조금 더 item이 더 잘 팔린다
(446+516)/(446+516+344+555)
# sold <-> biddable
table(eBayTrain$biddable, eBayTrain$sold)
(804+640)/(804+640+220+197)*100 # biddable과 sold의 연관성 77%, 그리고 sold=1와 sold=0일때 biddable이 구분된다.
# sold <-> condition  
table(eBayTrain$condition, eBayTrain$sold) # 뚜렷한 패턴x
# sold <-> cellular
table(eBayTrain$cellular, eBayTrain$sold) # 뚜렷한 패턴x 
# sold <-> carrier
table(eBayTrain$carrier, eBayTrain$sold) # unknown빼고 뚜렷한 패턴x,  3개 밖에 없는 carrier==other인 example은 지울필요가 있다. 
# sold <-> color
table(eBayTrain$color, eBayTrain$sold) # gold와 white빼고는 뚜렷한 패턴x  
# sold <-> storage
table(eBayTrain$storage, eBayTrain$sold) # 128 기가, unknown
# sold <-> productline
table(eBayTrain$productline, eBayTrain$sold) # 1개밖에 없는 ipad5 example 지울필요
# sold <-> startprice
Train_oSold = subset(eBayiPadTrain, eBayiPadTrain$sold==1) 
summary(Train_oSold$startprice)
hist(Train_oSold$startprice, breaks=2000)
boxplot(Train_oSold$startprice)
tapply(eBayTrain$startprice, eBayTrain$sold, mean)
boxplot(eBayTrain$startprice ~ eBayTrain$sold)

##### dependent 변수간의 관계
## 특히, 
# bidable <-> startprice
tapply(eBayTrain$startprice, eBayTrain$biddable, mean)
boxplot(eBayTrain$startprice ~ eBayTrain$biddable) ###
# productline <-> startprice 
boxplot(eBayTrain$startprice ~ eBayTrain$productline, xlab="productline", ylab="Startprice") #### productline 각각 factor마다 startprice가 많이 다름 -> 유용할듯.
# color <-> startprice
boxplot(eBayTrain$startprice ~ eBayTrain$color) # color 각각 factor 마다 startprice가 조금 다름
# carrier <-> startprice
boxplot(eBayTrain$startprice ~ eBayTrain$carrier) # carrier 각각 factor 마다 startprice가 조금 다름
# cellular <-> startprice
boxplot(eBayTrain$startprice ~ eBayTrain$cellular) ## cellular 각각 factor마다 startprice가 모두 비슷하므로... 쓸모없을듯
# storage <-> startprice
boxplot(eBayTrain$startprice ~ eBayTrain$storage) ###
# condition <-> startprice
boxplot(eBayTrain$startprice ~ eBayTrain$condition) ###

# ---> 다음 관계들이 그나마 뚜렷한 관계, 서로 구분되어 질 수 있음  
# sold <-> biddable
# startprice <-> biddable 
# 여기서는 regression보다는, tree방식의 모델을 선택하는게 유리하므로 어떤 변수든 상관없이 구분되질 수 있는 관계를 찾는게 중요

##### Data Relation Analysis
##### 분간이 가는 변수가 누가 있을까 ? 패턴을 가진 변수들간의 관계가 누가 있을까 ?
## 먼저, sold dependent 변수를 잘 설명할 수 있는 independent 변수를 찾는다. dependent ~ independent간의 관계는 귀찮아도 모든 경우의를 찾아볼만 하다. 
## continuous 변수인 startprice 변수가 sold 변수를 잘 설명할 수 있다. sold=1 or 0일때 startprice 분포가 많이 다르다., 다른 변수들은 biddable 빼고 sold와 특별히 관계가 있지 않다. 
## 하지만, 그냥 그대로 startprice를 feature로 사용한다면, sold를 설명할 수 있는 범위가 크다 모델이 bias되기 쉽다. 따라서 좀 더 세분화 시켜서 잘 설명할 수 있는 새로운 범위의 feature가 필요하다 
## 이런 intuition으로 2차 계층적으로, startprice변수를 잘 설명 할 수 있는 dependent 변수를 찾는다.  
## boxplot을 이용해 startprice변수와 각각 변수들을 비교해본다. 각각의 변수에서 factor들의 startprice median값이 큰 차이가 있으면, 그 dependent 변수는 유용하다.
## 이러한 변수들을 설명?을 모델에 넣기위해서 새로운 feature를 만들 필요가 있다.(data relation analysis의 목적)
## lm(startprice ~. -sold -UniqueId - description -...(불필요한 것들..은 뺀다)) 
## startprice_diff = startprice - prediction 이라는 새로운 startprice_diff feature를 만든다.



## etc
Test = subset(eBayiPadTrain, eBayiPadTrain$sold==1&eBayiPadTrain$biddable==0)
hist(Test$startprice, breaks=2000)


tapply(eBayTrain$startprice, eBayTrain$biddable==1, mean)


tapply(eBayTrain$startprice, eBayTrain$productline, median)
p1 = tapply(eBayTrain$startprice, eBayTrain$productline, mean)
p1[as.character(eBayTrain$productline)]
p1



install.packages("plyr")
library(plyr)
medProduct <- ddply(eBayiPadTrain, "productline", summarise, medProduct = median(startprice))
medCondition <- ddply(eBayiPadTrain, "condition", summarise, medCondition = median(startprice))
## ddply 사용하면 new feature로 사용할 때 모델에 변수를 추가할 때 편리할 듯....
medProduct
medCondition
