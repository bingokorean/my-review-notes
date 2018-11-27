# Video 2 - Reading in the Dataset

### Problem Description
# 균형된? 영양 섭취는 건강한 삶의 중요한 요소임.
# 영양 실조 (영양 불균형)는 비만을 초래함. 
# 비만 트랜드는 1990년대부터 2010년까지 미국에서 기하급수적으로 증가.
# 미국인의 35%이상이 비만임.
# USDA Food Database: 미국 농업부에서 제공; 7000개 이상의 food item들의 영양 정보;


# Get the current directory
  getwd()
  
# Read the csv file
  USDA = read.csv("USDA.csv")
# Structure of the dataset
  str(USDA) 
# Statistical summary
  # 이 중에서 sodium을 예를 들면, 하루 권장량은 2300 milligrams인데, 어떤 food는 38758 milligrams인 경우가 있다. 최대한 이런 음식을 피해야 한다.
  summary(USDA)


# Video 3 - Basic Data Analysis

# Vector notation
  USDA$Sodium
# Finding the index of the food with highest sodium levels
  which.max(USDA$Sodium)
# Get names of variables in the dataset
  names(USDA)
# Get the name of the food with highest sodium levels
  USDA$Description[265]
# Create a subset of the foods with sodium content above 10,000mg
  HighSodium = subset(USDA, Sodium>10000)
# Count the number of rows, or observations
nrow(HighSodium)
# Output names of the foods with high sodium content
  HighSodium$Description
# Finding the index of CAVIAR in the dataset
  match("CAVIAR", USDA$Description)
# Find amount of sodium in caviar
  USDA$Sodium[4154]
# Doing it in one command!
  USDA$Sodium[match("CAVIAR", USDA$Description)]
# Summary function over Sodium vector
  # 4154번째 음식인 caviar의 sodium이 다른 음식에 비해 높은 편인지 summary를 통해 알아보자.
  summary(USDA$Sodium)
# Standard deviation
  # 322.1 (mean) + sd (1045.417) = 1367.517 < 1500 (=USDA$Sodium[4154])
  # 아하 꽤 높은 수치구나 라고 인지할 수 있음.
  sd(USDA$Sodium, na.rm = TRUE)
  

  
# Video 4 - Plots

  # 흥미롭게도 삼각형 모양이 나옴. higher protein -> lower fat
  
# Scatter Plots
  plot(USDA$Protein, USDA$TotalFat)
# Add xlabel, ylabel and title
  plot(USDA$Protein, USDA$TotalFat, xlab="Protein", ylab = "Fat", main = "Protein vs Fat", col = "red")

  # 히스토그램은 오직 하나의 변수만 분석함. (y축은 항상 frequency이기 때문)
  # 적절히 breaks 크기를 잘 설정해주는 것이 중요. (breaks크기를 크게할 수록 zoom-in한다고 보면 됨.)
  
# Creating a histogram
  hist(USDA$VitaminC, xlab = "Vitamin C (mg)", main = "Histogram of Vitamin C")
# Add limits to x-axis
  hist(USDA$VitaminC, xlab = "Vitamin C (mg)", main = "Histogram of Vitamin C", xlim = c(0,100))

  # Specify breaks of histogram
  hist(USDA$VitaminC, xlab = "Vitamin C (mg)", main = "Histogram of Vitamin C", xlim = c(0,100), breaks=100)
  hist(USDA$VitaminC, xlab = "Vitamin C (mg)", main = "Histogram of Vitamin C", xlim = c(0,100), breaks=2000)

  
  # sugar의 boxplot은 특이함. average는 굉장히 낮은 반면, outlier가 굉장히 많음 (즉, 조심해야될 food (e.g. candy)가 많음)
  
  # Boxplots
  boxplot(USDA$Sugar, ylab = "Sugar (g)", main = "Boxplot of Sugar")


# Video 5 - Adding a variable

# Creating a variable that takes value 1 if the food has higher sodium than average, 0 otherwise
  HighSodium = as.numeric(USDA$Sodium > mean(USDA$Sodium, na.rm=TRUE))
  str(HighSodium)
# Adding the variable to the dataset
  USDA$HighSodium = as.numeric(USDA$Sodium > mean(USDA$Sodium, na.rm=TRUE))
# Similarly for HighProtein, HigCarbs, HighFat
  USDA$HighCarbs = as.numeric(USDA$Carbohydrate > mean(USDA$Carbohydrate, na.rm=TRUE))
  USDA$HighProtein = as.numeric(USDA$Protein > mean(USDA$Protein, na.rm=TRUE))
  USDA$HighFat = as.numeric(USDA$TotalFat > mean(USDA$TotalFat, na.rm=TRUE))

"
With the table and tapply functions, we can understand our data and the relationships between our variables better.
"
  
  
# Video 6 - Summary Tables

# How many foods have higher sodium level than average?
  table(USDA$HighSodium)
# How many foods have both high sodium and high fat?
  table(USDA$HighSodium, USDA$HighFat)
  
  # (중요) tapply함수
  # continuous 변수가 categorical 변수에 맞게 그룹핑되고, 각 그룹마다 통계수치(e.g. 평균) 계산
  # -> tapply(agr1, arg2, mean/max/summary/..., na.rm=True)
  # -> arg1: continuous variable
  # -> arg2: categorical variable
  
# Average amount of iron sorted by high and low protein?
  tapply(USDA$Iron, USDA$HighProtein, mean, na.rm=TRUE)
# Maximum level of Vitamin C in hfoods with high and low carbs?
  tapply(USDA$VitaminC, USDA$HighCarbs, max, na.rm=TRUE)
# Using summary function with tapply
  tapply(USDA$VitaminC, USDA$HighCarbs, summary, na.rm=TRUE)
