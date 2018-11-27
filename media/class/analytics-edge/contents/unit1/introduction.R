### VECTORS AND DATA FRAMES
c(2,3,5,8,13) # this creates a vector of five numbers all stored as the same object 
Country = c("Brazil","China","India","Switzerland","USA")

Country
LifeExpectancy = c(74,76,65,83,79)
LifeExpectancy
Country[1]
LifeExpectancy[3]
seq(0,100,2)  # creates sequence of numbers,  this can be useful if you want to create a unique identifier for observations.
CountryData = data.frame(Country,LifeExpectancy)  # combine our vectors into data frame
CountryData
CountryData$Population = c(199000, 1390000, 1240000, 7997, 31800) # to add another variable to our data frame
CountryData
Country = c("Australia", "Greece")  # to add two new observations
LifeExpectancy = c(82,81)
Population = c(23050,11125)
NewCountryData = data.frame(Country, LifeExpectancy, Population)
# data.frame함수는 여러 개의 (변수) 벡터들을 하나의 매트릭스로 묶어준다. column-wise로 추가해 나간다.
# 변수 벡터들을 추가해나가는 느낌.
NewCountryData
AllCountryData = rbind(CountryData, NewCountryData) # combine two data frame
# rbind함수는 row-wise로 매트릭스들을 서로 덧붙여준다.
# 데이터를 추가해나가는 느낌. (변수들은 고정되어 있음.)
AllCountryData
getwd()
# setwd()  # working directory change

### LOADING DATA FILES
WHO = read.csv("WHO.csv")
str(WHO)  # show us the structure of the data
summary(WHO)  
WHO_Europe = subset(WHO, Region == "Europe") # to subset your data
str(WHO_Europe)
write.csv(WHO_Europe, "WHO_Europe.csv")
ls()
rm(WHO_Europe)  # remove the data frame from our current session.
ls()

### SUMMARY STATISTICS AND SCATTERPLOTS
WHO$Under15
mean(WHO$Under15)
sd(WHO$Under15)
summary(WHO$Under15)  # we can get also the statistical summary of just on variable using summary()
# 1st Qu. = the value for which 25% of the data is less than that value
# 3rd Qu. = the value for which 75% of the data is less than that value
# this output tells us that there's a country with only 13% of the population under 15.
which.min(WHO$Under15)  # which is the row number of the observation with the minimum value of Under15.
WHO$Country[86]  # to see which country is in row 86
which.max(WHO$Under15)
WHO$Country[124]
plot(WHO$GNI, WHO$FertilityRate)
Outliers = subset(WHO, GNI > 10000 & FertilityRate > 2.5)
nrow(Outliers) # how many rows of data are in our subset 
Outliers[c("Country", "GNI", "FertilityRate")]

### PLOTS AND SUMMARY TABLES
hist(WHO$CellularSubscribers)  
boxplot(WHO$LifeExpectancy ~ WHO$Region) 
# Box plot은 어떤 변수의 statistical range를 이해하기 좋음.
# 각각 박스는 25% (=first quartile) ~ 75% (third quartile) 크기/범위(=inter-quartile range (IRQ))를 가지고, 중간에 median value를 기준으로 middle line (50%)이 있음.
# 점선은 whiskers로 outlier를 제외한, 최소값부터 최댓값의 범위를 나타냄.
# outlier는 박스의 높이(=IRQ)에 따라 정의됨: 75% + 1.5*IRQ 보다 이상이면, 또는 25% - 1.5*IRQ 보다 이하이면 outlier라 간주함.
# 아래의 박스 플랏을 보고 region 변수에 따른 life expectancy를 적절히 비교할 수 있음.
boxplot(WHO$LifeExpectancy ~ WHO$Region, xlab="", ylab="Life Expectancy", main="Life Expectancy of Countries by Region")
table(WHO$Region) # similar with summary, but tables work well for variables with only a few possible values, and we'll see more of this in recitation.
tapply(WHO$Over60, WHO$Region, mean)  # by using tapply(), we can see nice information about numerical variables
# tapply splits the data by the second argument, and than applies the third argument function to the variable given as the first argument
tapply(WHO$LiteracyRate, WHO$Region, min)
tapply(WHO$LiteracyRate, WHO$Region, min, na.rm=TRUE)  # to remove missing values




