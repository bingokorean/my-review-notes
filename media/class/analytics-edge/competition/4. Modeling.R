### OUTPUT
# ebayTrain
# ebayTest
str(ebayTrain)
str(ebayTest)


################################### MODEL 1 ##################################
### Logistic Regression ###
GLM = glm(Tsold ~. - TUniqueID, data = ebayTrain, family="binomial")
predictTrain = predict(GLM, type="response")
predictTest = predict(GLM, type="response", newdata = ebayTest)



################################### MODEL 2 ##################################
### CART + cross-validation ###
#install.packages("caret")
library(caret)
#install.packages("e1071")
library(e1071)
numFolds = trainControl(method="cv", number=10)
cpGrid = expand.grid( .cp = seq(0.01,0.5,0.01))
train(Tsold ~. -TUniqueID, data = ebayTrain, method = "rpart", trControl = numFolds, tuneGrid = cpGrid)
CART = rpart(Tsold ~. - TUniqueID, data = ebayTrain, method="class", cp=0.01)
predictTrain = predict(CART, type="class")
predictTest = predict(CART, newdata = ebayTest, type="class")

################################### MODEL 3 ##################################
### Random Forest ###
#install.packages("randomForest")
library(randomForest)
RF.Train = randomForest(Tsold ~. - TUniqueID, data = ebayTrain, ntree=200, nodesize=25)
predictTrain = predict(RF.Train)
predictTest = predict(RF.Train, newdata = ebayTest)





################################### FINAL PROCESS ########################################
#install.packages("ROCR")
library(ROCR)
ROCRpred = prediction(predictTrain, ebayTrain$Tsold)
ROCRperf = performance(ROCRpred, "tpr", "fpr")
as.numeric(performance(ROCRpred, "auc")@y.values)
plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1,7))

# Check AUC value
#ROCRpred = prediction(predictTest, )

# Now we can prepare our submission file for Kaggle:
MySubmission = data.frame(UniqueID = ebayTest$TUniqueID, Probability1 = predictTest)
write.csv(MySubmission, "SubmissionDescriptionLog.csv", row.names=FALSE)



#################### Test Result #################### 
# Feature Engineering x + Logistic Regression = 0.85052
# Feature Engineering x + CART (cross validation) = 0.76513
# Feature Engineering x + Random Forest = 0.84835

# 1 + 3 + Logistic Regression = 84774
# 1 + 3 + CART = 0.84547
# 1 + 2 + 3 + Random Forest = 0.84872

# 1 + 3 + repeatedCV = 0.85027
