# 15세 대상으로 하는 Programme for International Student Assessment (PISA) 시험
# MATHEMATICS, READING, 그리고 SCIENCE 시험으로 나뉨
# 여기서는 미국의 2009 PISA 시험을 치룬 학생들의 READING 점수를 예측하고자 함.
# 2009 PISA Public-Use Data Files distributed by the United States National Center for Education Statistics (NCES)

## Variable Description
# grade: the grade in school of the student (most 15-years-olds in America are in 10th grade)
# male: whether the student is male (1/0)
# raceeth: the race/ethnicity composite of the student
# preschool: whether the student attended preschool (1/0)
# expectBachelors: whether the student expects to obtain a bachelor's degree (1/0)
# motherHS: whether the student's mother completed high school (1/0)
# motherBachelors: whether the student's mother obtained a bachelor's degree (1/0)
# motherWork: whether the studen't mother has part-time or full-time word (1/0)
# fatherHS: whether the student's father complete high school (1/0)
# fatherBachelors: whetehr the student's father obtained a bachelor's degree (1/0)




### PROBLEM 1.1 - DATASET SIZE
pisaTrain = read.csv("pisa2009train.csv")
pisaTest = read.csv("pisa2009test.csv")
str(pisaTrain)

### PROBLEM 1.2 - SUMMARIZING THE DATASET
tapply(pisaTrain$readingScore, pisaTrain$male, mean, na.rm=TRUE)

### PROBLEM 1.3 - LOCATING MISSING VALUES  
summary(pisaTrain)

### PROBLEM 1.4 - REMOVING MISSING VALUES
## Linear regression discards observations with missing data, so we will remove all such observations from the training and testing sets. Later in the course, we will learn about imputation, which deals with missing data by filling in missing values with plausible information.
## Type the following commands into your R console to remove observations with any missing value from pisaTrain and pisaTest:
pisaTrain = na.omit(pisaTrain)
pisaTest = na.omit(pisaTest)
str(pisaTrain)
str(pisaTest)

# Factor 변수는 크게 unordered factor와 ordered factor로 구분됨.
# Factor 변수는 one-hot encoding되므로 numerical value 크기의 의미는 없음.
# unordered factor는 most common을 기준으로 reference level을 설정함. 
# reference level인 factor는 항상 0 벡터로 표현됨.

### PROBLEM 2.1 - FACTOR VARIABLES
## Factor variables are variables that take on a discrete set of values, like the "Region" variable in the WHO dataset from the second lecture of Unit 1. This is an unordered factor because there isn't any natural ordering between the levels. An ordered factor has a natural ordering between the levels (an example would be the classifications "large," "medium," and "small").
str(pisaTrain)
# Which of the following variables is an unordered factor with at least 3 levels? => raceeth
# Which of the following variables is an ordered factor with at least 3 levels? => grade

### PROBLEM 2.2 - UNORDERED FACTORS IN REGRESSION MODELS
## To include unordered factors in a linear regression model, we define one level as the "reference level" and add a binary variable for each of the remaining levels. In this way, a factor with n levels is replaced by n-1 binary variables. The reference level is typically selected to be the most frequently occurring level in the dataset.
## As an example, consider the unordered factor variable "color", with levels "red", "green", and "blue". If "green" were the reference level, then we would add binary variables "colorred" and "colorblue" to a linear regression problem. All red examples would have colorred=1 and colorblue=0. All blue examples would have colorred=0 and colorblue=1. All green examples would have colorred=0 and colorblue=0.
## Now, consider the variable "raceeth" in our problem, which has levels "American Indian/Alaska Native", "Asian", "Black", "Hispanic", "More than one race", "Native Hawaiian/Other Pacific Islander", and "White". Because it is the most common in our population, we will select White as the reference level.
summary(pisaTrain)

### PROBLEM 2.3 - EXAMPLE UNORDERED FACTORS
## Consider again adding our unordered factor race to the regression model with reference level "White".

### PROBLEM 3.1 - BUILDING A MODEL
## Because the race variable takes on text values, it was loaded as a factor variable when we read in the dataset with read.csv() -- you can see this when you run str(pisaTrain) or str(pisaTest). However, by default R selects the first level alphabetically ("American Indian/Alaska Native") as the reference level of our factor instead of the most common level ("White"). Set the reference level of the factor by typing the following two lines in your R console:
pisaTrain$raceeth = relevel(pisaTrain$raceeth, "White")
pisaTest$raceeth = relevel(pisaTest$raceeth, "White")
## Now, build a linear regression model (call it lmScore) using the training set to predict readingScore using all the remaining variables.
## It would be time-consuming to type all the variables, but R provides the shorthand notation "readingScore ~ ." to mean "predict readingScore using all the other variables in the data frame." The period is used to replace listing out all of the independent variables. As an example, if your dependent variable is called "Y", your independent variables are called "X1", "X2", and "X3", and your training data set is called "Train", instead of the regular notation:
## LinReg = lm(Y ~ X1 + X2 + X3, data = Train)
## You would use the following command to build your model:
## LinReg = lm(Y ~ ., data = Train)
LinReg = lm(readingScore ~. ,data=pisaTrain)
summary(LinReg)
## Note that this R-squared is lower than the ones for the models we saw in the lectures and recitation. This does not necessarily imply that the model is of poor quality. More often than not, it simply means that the prediction problem at hand (predicting a student's test score based on demographic and school-related variables) is more difficult than other prediction problems (like predicting a team's number of wins from their runs scored and allowed, or predicting the quality of wine from weather conditions).

### PROBLEM 3.2 - COMPUTING THE ROOT-MEAN SQUARED ERROR OF THE MODEL
SSE = sum(LinReg$residuals^2)
RMSE = sqrt(SSE/nrow(pisaTrain))
RMSE

### PROBLEM 3.3 - COMPARING PREDICTIONS FOR SIMILAR STUDENTS
## Consider two students A and B. They have all variable values the same, except that student A is in grade 11 and student B is in grade 9. What is the predicted reading score of student A minus the predicted reading score of student B?
## The coefficient 29.54 on grade is the difference in reading score between two students who are identical other than having a difference in grade of 1. Because A and B have a difference in grade of 2, the model predicts that student A has a reading score that is 2*29.54 larger.
summary(LinReg)

### PROBLEM 3.4 - INTERPRETING MODEL COEFFICIENTS
# What is the meaning of the coefficient associated with variable raceethAsian?
# Predicted difference in the reading score between an Asian student and a white student who is otherwise identical 
# The only difference between an Asian student and white student with otherwise identical variables is that the former has raceethAsian=1 and the latter has raceethAsian=0. The predicted reading score for these two students will differ by the coefficient on the variable raceethAsian.

### PROBLEM 3.5 - IDENTIFYING VARIABLES LACKING STATISTICAL SIGNIFICANCE
# From summary(lmScore), we can see which variables were significant at the 0.05 level. Because several of the binary variables generated from the race factor variable are significant, we should not remove this variable.

### PROBLEM 4.1 - PREDICTING ON UNSEEN DATA
## Using the "predict" function and supplying the "newdata" argument, use the lmScore model to predict the reading scores of students in pisaTest. Call this vector of predictions "predTest". Do not change the variables in the model (for example, do not remove variables that we found were not significant in the previous part of this problem). Use the summary function to describe the test set predictions.
predTest = predict(LinReg, newdata=pisaTest)
summary(predTest)

SSE = sum((predTest - pisaTest$readingScore)^2)
SSE
SST = sum((mean(pisaTrain$readingScore) - pisaTest$readingScore)^2)
SST

RMSE = sqrt(SSE/nrow(pisaTest))
RMSE


### PROBLEM 4.3 - BASELINE PREDICTION AND TEST-SET SSE
# What is the predicted test score used in the baseline model? Remember to compute this value using the training set and not the test set.
baseline = mean(pisaTrain$readingScore)
baseline
# What is the sum of squared errors of the baseline model on the testing set? HINT: We call the sum of squared errors for the baseline model the total sum of squares (SST).
sum((baseline-pisaTest$readingScore)^2)


### PROBLEM 4.4 - TEST-SET R-SQUARED
R2 = 1 - SSE/SST
R2

