### Probem Description

# 미국에서는 unemployment을 Current Population Survey (CPS)로부터 얻음.
# CPS는 매달마다 DEMOGRAPHIC과 EMPLYMENT 정보를 수집함.
# 데이터는 September 2013 CPS에 survey한 사람들의 정보임.
# 원래는 385개 변수들로 구성되어 있지만, 여기서는 간단하게 표현


### Variable Description

# PeopleInHouse: the number of people in the interviewee's household
# Region: the census region where the interviewee lives
# State: the state where the interviewee lives
# MetroAreaCode: a code that identifies the metropolitan area where the interviewee lives
# Age: the age, in years, of the interviewee. 80 represents people aged 80-84, and 85 represents people aged 85 higher
# Married: the marriage status of the interviewee
# Sex: the sex of the interviewee
# Eduacation: the maximum level of education obtained by the interviewee
# Race: the race of the interviewee
# Hispanic: whether the interviewee is of Hispanic ethnicity
# CountryOfBirthCode: a code identifying the country of birth of the interviewee
# Citizenship: the U.S citizenship status of the interviewee
# EmploymentSatatus: the status of employment of the interviewee
# Industy: the industry of employment of the interviewee (only available if they are employed)




### PROBLEM 1.1 - LOADING AND SUMMARIZING THE DATASET
CPS = read.csv("CPSData.csv")
str(CPS) # How many interviewees are in the dataset?
summary(CPS$Industry) # Among the interviewees with a value reported for the Industry variable, what is the most common industry of employment?
sort(table(CPS$State)) # Which state has the fewest interviewees?, Which state has the largest number of interviewees?
sort(table(CPS$Citizenship)) # What proportion of interviewees are citizens of the United States?
table(CPS$Hispanic, CPS$Race ) # For which races are there at least 250 interviewees in the CPS dataset of Hispanic ethnicity? 

### PROBLEM 2.1 - EVALUATING MISSING VALUES
summary(CPS) # Which variables have at least one interviewee with a missing (NA) value? 
## Often when evaluating a new dataset, we try to identify if there is a pattern in the missing values in the dataset. We will try to determine if there is a pattern in the missing values of the Married variable. The function is.na(CPS$Married) returns a vector of TRUE/FALSE values for whether the Married variable is missing. We can see the breakdown of whether Married is missing based on the reported value of the Region variable with the function table(CPS$Region, is.na(CPS$Married)). Which is the most accurate:
table(CPS$Region, is.na(CPS$Married))
table(CPS$Sex, is.na(CPS$Married))
table(CPS$Age, is.na(CPS$Married))
table(CPS$Citizenship, is.na(CPS$Married)) # How many states had all interviewees living in a non-metropolitan area (aka they have a missing MetroAreaCode value)?
table(CPS$State, is.na(CPS$MetroAreaCode)) # How many states had all interviewees living in a metropolitan area? Again, treat the District of Columbia as a state.
table(CPS$Region, is.na(CPS$MetroAreaCode)) # Which region of the United States has the largest proportion of interviewees living in a non-metropolitan area?
## While we were able to use the table() command to compute the proportion of interviewees from each region not living in a metropolitan area, it was somewhat tedious (it involved manually computing the proportion for each region) and isn't something you would want to do if there were a larger number of options. It turns out there is a less tedious way to compute the proportion of values that are TRUE. The mean() function, which takes the average of the values passed to it, will treat TRUE as 1 and FALSE as 0, meaning it returns the proportion of values that are true. For instance, mean(c(TRUE, FALSE, TRUE, TRUE)) returns 0.75. Knowing this, use tapply() with the mean function to answer the following questions:
tapply(is.na(CPS$MetroAreaCode), CPS$State, mean) # argument 위치 주의 , Which state has a proportion of interviewees living in a non-metropolitan area closest to 30%?, 
sort(tapply(is.na(CPS$MetroAreaCode), CPS$State, mean)) # Which state has the largest proportion of non-metropolitan interviewees, ignoring states where all interviewees were non-metropolitan?

### PROBLEM 3.1 - INTEGRATING METROPOLITAN AREA DATA
## Codes like MetroAreaCode and CountryOfBirthCode are a compact way to encode factor variables with text as their possible values, and they are therefore quite common in survey datasets. In fact, all but one of the variables in this dataset were actually stored by a numeric code in the original CPS datafile.
## When analyzing a variable stored by a numeric code, we will often want to convert it into the values the codes represent. To do this, we will use a dictionary, which maps the the code to the actual value of the variable. We have provided dictionaries MetroAreaCodes.csv and CountryCodes.csv, which respectively map MetroAreaCode and CountryOfBirthCode into their true values. Read these two dictionaries into data frames MetroAreaMap and CountryMap.
MetroAreaMap = read.csv("MetroAreaCodes.csv")
CountryMap = read.csv("CountryCodes.csv")
str(MetroAreaMap)
str(CountryMap)
## To merge in the metropolitan areas, we want to connect the field MetroAreaCode from the CPS data frame with the field Code in MetroAreaMap. The following command merges the two data frames on these columns, overwriting the CPS data frame with the result:
CPS = merge(CPS, MetroAreaMap, by.x="MetroAreaCode", by.y="Code", all.x=TRUE)
## The first two arguments determine the data frames to be merged (they are called "x" and "y", respectively, in the subsequent parameters to the merge function). by.x="MetroAreaCode" means we're matching on the MetroAreaCode variable from the "x" data frame (CPS), while by.y="Code" means we're matching on the Code variable from the "y" data frame (MetroAreaMap). Finally, all.x=TRUE means we want to keep all rows from the "x" data frame (CPS), even if some of the rows' MetroAreaCode doesn't match any codes in MetroAreaMap (for those familiar with database terminology, this parameter makes the operation a left outer join instead of an inner join).
summary(CPS)
str(CPS)
summary(CPS$MetroArea)
sort(tapply(CPS$Hispanic, CPS$MetroArea,  mean)) # Which metropolitan area has the highest proportion of interviewees of Hispanic ethnicity?
sort(tapply(CPS$Race == "Asian", CPS$MetroArea,  mean)) # determine the number of metropolitan areas in the United States from which at least 20% of interviewees are Asian.

## Normally, we would look at the sorted proportion of interviewees from each metropolitan area who have not received a high school diploma with the command:
sort(tapply(CPS$Education == "No high school diploma", CPS$MetroArea, mean))
## However, none of the interviewees aged 14 and younger have an education value reported, so the mean value is reported as NA for each metropolitan area. To get mean (and related functions, like sum) to ignore missing values, you can pass the parameter na.rm=TRUE. Passing na.rm=TRUE to the tapply function, determine which metropolitan area has the smallest proportion of interviewees who have received no high school diploma.
sort(tapply(CPS$Education == "No high school diploma", CPS$MetroArea, mean, na.rm=TRUE))

### PROBLEM 4.1 - INTEGRATING COUNTRY OF BIRTH DATA
## Just as we did with the metropolitan area information, merge in the country of birth information from the CountryMap data frame, replacing the CPS data frame with the result. If you accidentally overwrite CPS with the wrong values, remember that you can restore it by re-loading the data frame from CPSData.csv and then merging in the metropolitan area information using the command provided in the previous subproblem.
CPS = merge(CPS, CountryMap, by.x="CountryOfBirthCode", by.y="Code", all.x=TRUE) # What is the name of the variable added to the CPS data frame by this merge operation?, How many interviewees have a missing value for the new country of birth variable?
summary(CPS)
sort(table(CPS$Country)) # Among all interviewees born outside of North America, which country was the most common place of birth?
sort(table(CPS$MetroArea))
table(CPS$MetroArea == "New York-Northern New Jersey-Long Island, NY-NJ-PA", CPS$Country != "United States") # What proportion of the interviewees from the "New York-Northern New Jersey-Long Island, NY-NJ-PA" metropolitan area have a country of birth that is not the United States? For this computation, don't include people from this metropolitan area who have a missing country of birth.

# Which metropolitan area has the largest number (note -- not proportion) of interviewees with a country of birth in India? Hint -- remember to include na.rm=TRUE if you are using tapply() to answer this question.
table(CPS$MetroArea,CPS$Country =="India" )
table(CPS$MetroArea,CPS$Country =="Brazil" )
table(CPS$MetroArea,CPS$Country =="Somalia" )

sort(tapply(CPS$Country == "India", CPS$MetroArea, sum, na.rm=TRUE))
sort(tapply(CPS$Country == "Brazil", CPS$MetroArea, sum, na.rm=TRUE))
sort(tapply(CPS$Country == "Somalia", CPS$MetroArea, sum, na.rm=TRUE))












