### Problem Description

# 범죄(crime)는 국제적 관심 사항임.
# 미국에선 Federal Bureau of Investigation (FBI)에 의해 violent 범죄 그리고 그의 특징들이 기록됨.
# 또한, 각 도시별로 범죄 비율이 기록됨. 
# 두 가지 종류의 범죄가 있다: violent crimes, property crimes
# 여기서는 property crime 중에서 "motor vehicle theft" 범죄 (e.g. stealing a car)만 다룸
# Chicago 내에 있는 motor vehicle theft를 이해하기 위한 data analysis임.


### Variable Description

# ID: a unique identifier for each observation
# Date: the date the crime occurred
# LocationDescription: the location where the crime occurred
# Arrest: whether or not an arrest was made for the crime
# Domestic: whether or not the crime was a domestic crime, meaning that it was committed against a fmily member (TRUE is domestic)
# Beat: the area, or "beat" where the crime occurred. This is the smallest regional division defined by the Chicago police department.
# District: the police district where the crime occured. Each district is composed of many beats, and are defined by the Chicago Police Department.
# CommunityArea: the community area in which the crime occurred. Since the 1920s, Chicago has been divided into what are called "community areas", of which there are now 77. The community areas were devised in an attempt to create socially homogeneous regions.
# Year: the year in which the crime occurred
# Latitude: the latitude of the location where the crime occurred
# Longitude: the longitude of the location where the crime occurred




### PROBLEM 1 - LOADING THE DATA
getwd()
mvt = read.csv("mvtWeek1.csv")
str(mvt)  # How many rows of data (observations) are in this dataset? # How many variables are in this dataset?
max(mvt$ID) # Using the "max" function, what is the maximum value of the variable "ID"?
min(mvt$Beat) # What is the minimum value of the variable "Beat"?
summary(mvt$Arrest) # How many observations have value TRUE in the Arrest variable (this is the number of crimes for which an arrest was made)?
summary(mvt$LocationDescription) # How many observations have a LocationDescription value of ALLEY?

### PROBLEM 2 - UNDERSTANDING DATES IN R
## In many datasets, like this one, you have a date field. Unfortunately, R does not automatically recognize entries that look like dates. We need to use a function in R to extract the date and time. Take a look at the first entry of Date (remember to use square brackets when looking at a certain entry of a variable).
summary(mvt$Date) # In what format are the entries in the variable Date?
DateConvert = as.Date(strptime(mvt$Date, "%m/%d/%y %H:%M")) # let's convert these characters into a Date object in R
#DateConvert
summary(DateConvert)
## Now, let's extract the month and the day of the week, and add these variables to our data frame mvt. We can do this with two simple functions. 
mvt$Month = months(DateConvert)
mvt$Weekday = weekdays(DateConvert)
mvt$Date = DateConvert # replace the old Date variable with DateConvert
table(mvt$Month) # In which month did the fewest motor vehicle thefts occur?
table(mvt$Weekday)
table(mvt$Arrest, mvt$Month) # Which month has the largest number of motor vehicle thefts for which an arrest was made?

### PROBLEM 3 - VISUALIZING CRIME TRENDS
hist(mvt$Date, breaks=100)
## In a boxplot, the bold horizontal line is the median value of the data, the box shows the range of values between the first quartile and third quartile, and the whiskers (the dotted lines extending outside the box) show the minimum and maximum values, excluding any outliers (which are plotted as circles). Outliers are defined by first computing the difference between the first and third quartile values, or the height of the box. This number is called the Inter-Quartile Range (IQR). Any point that is greater than the third quartile plus the IQR or less than the first quartile minus the IQR is considered an outlier.
boxplot(mvt$Date)
table(mvt$Arrest, mvt$Year)
## Since there may still be open investigations for recent crimes, this could explain the trend we are seeing in the data. There could also be other factors at play, and this trend should be investigated further. However, since we don't know when the arrests were actually made, our detective work in this area has reached a dead end.

### PROBLEM 4 - POPULAR LOCATIONS
## Analyzing this data could be useful to the Chicago Police Department when deciding where to allocate resources. If they want to increase the number of arrests that are made for motor vehicle thefts, where should they focus their efforts?
## We want to find the top five locations where motor vehicle thefts occur. If you create a table of the LocationDescription variable, it is unfortunately very hard to read since there are 78 different locations in the data set. By using the sort function, we can view this same table, but sorted by the number of observations in each category. 
sort(table(mvt$LocationDescription))
Top5 = subset(mvt, LocationDescription=="STREET" | LocationDescription=="PARKING LOT/GARAGE(NON.RESID.)" | LocationDescription=="ALLEY" | LocationDescription=="GAS STATION" | LocationDescription=="DRIVEWAY - RESIDENTIAL")  # How many observations are in Top5?
# Another way of doing this would be to use the %in% operator in R. This operator checks for inclusion in a set
TopLocations = c("STREET", "PARKING LOT/GARAGE(NON.RESID.)", "ALLEY", "GAS STATION", "DRIVEWAY - RESIDENTIAL") 
Top5 = subset(mvt, LocationDescription %in% TopLocations)
## R will remember the other categories of the LocationDescription variable from the original dataset, so running table(Top5$LocationDescription) will have a lot of unnecessary output. To make our tables a bit nicer to read, we can refresh this factor variable. In your R console, type:
Top5$LocationDescription = factor(Top5$LocationDescription)
str(Top5)
table(Top5$Arrest, Top5$LocationDescription) # One of the locations has a much higher arrest rate than the other locations. Which is it? 
table(Top5$LocationDescription, Top5$Weekday) # On which day of the week do the most motor vehicle thefts at gas stations happen?





