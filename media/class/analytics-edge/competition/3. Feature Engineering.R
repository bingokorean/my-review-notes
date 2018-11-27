# ebayTrain
# ebayTest
str(ebayTrain)
str(ebayTest)


############################################### STEP 1 ############################################### 
### Assumption 1 : sold <-설명- startprice <-설명- productline, biddable, color, storage, contition
# sold 1 or 0을 startprice로 잘 설명할 수 있다. sold=1일때 startprice의 distribution과 sold=0일때 distribution이 많이 다르기 때문이다.
# 따라서 그냥 startprice를 feature로 사용해도, 모델에 도움이 주긴 한다. 하지만, 우리 모델의 outcome이 binary이고, binary를 decision하기에는 연속적인 변수인 startprice를 그대로 사용하면 1과 0을 설명할 수있는 범위가 커서, 모델이 bias되기가 쉽다.
# 그래서 startprice와 관련된 새로운 feature를 하나 만드는게 더 낫다. startprice를 잘 설명할 수 있는 변수들을 이용해서.. 
# 어차피, startprice를 잘 설명하는 변수들도 sold와 관련있는 변수들이기 때문에, 그대로 startprice를 사용하는 것보다, 그러한 변수들의 정보가 들어가 있는 estimation of startprice를 사용하는게 좋다. (Linear regression사용)
# 따라서 새로운 feature를 Linear model의 prediction으로 사용한다.
# 편의상 (startprice - predictedprice)를 새로운 variable로 지정한다. (그냥 predictedprice를 지정하면 test set을 돌리기 전에 lm을 거쳐서 predictprice를 만들어야함.)
pricemodel = lm(Tstartprice ~. -Tsold - TUniqueID, data=ebayTrain)
predprice = predict(pricemodel)
ebayTrain$adjustedprice = ebayTrain$Tstartprice - predprice # new feature 추가 !

pricemodel_test = lm(Tstartprice ~. - TUniqueID, data=ebayTest)
predprice_test = predict(pricemodel_test)
ebayTest$adjustedprice = ebayTest$Tstartprice - predprice_test


### Assumption 2 : startprice와 제일 관계있는 productline과 condition 변수들에 관한 새로운 feature를 생성하자. ( assumption 1 에서 다뤘던 문제이지만, 좀 더 overfit를 향해서 갈 수 있다. )
#p1 = tapply(ebayTrain$Tstartprice, ebayTrain$productline, median)
#ebayTrain$diffmedian = ebayTrain$Tstartprice - p1[as.character(ebayTrain$productline)] # new feature 추가 !
#p2 = tapply(ebayTest$Tstartprice, ebayTest$productline, median)
#ebayTest$diffmedian = ebayTest$Tstartprice - p2[as.character((ebayTest$productline))]

### Assumption 3 : decription 에 negative한 단어가 있는 경우 사길 꺼려진다.
negativeWords <- c("blemish","crack","damag","dent","scratch","wear","tear","lock")
negative_count <- apply(as.matrix(DescriptionWords[, names(DescriptionWords) %in% negativeWords]), 1, sum )
ebayTrain$negative <- negative_count[1:nrow(eBayiPadTrain)]

ebayTest$negative <- negative_count[(nrow(eBayiPadTrain)+1):(nrow(eBayiPadTrain)+nrow(eBayiPadTest))] 


### ... 등등 가정을 하나 제안하면서 새로운 feature를 만든다... (내 가정으로 인해 모델의 성능이 좋아질 것이라는 기대하에..)


############################################### STEP 2 ###############################################
# ebayTrain
# ebayTest
str(ebayTrain)

# optimal한 variable들만 남기기 (repeatedCV, high correlation 삭제 사용)



# Feature selection
# We don't want to have sold in our feature selection
ebayTrainMinusSold = ebayTrain
ebayTrainMinusSold$Tsold = NULL

# Using Caret Package, run RFE to select a good feature set. This uses Random Forests. Tried SVM here too, but got errors
library(caret)
library(randomForest)
rfFuncs$summary <- twoClassSummary
trainctrl <- trainControl(classProbs= TRUE,summaryFunction = twoClassSummary)
control <- rfeControl(functions=rfFuncs, method="repeatedcv", number=10, repeats = 5, verbose=TRUE)
results <- rfe(ebayTrainMinusSold, as.factor(ebayTrain$Tsold), sizes=c(4,8,10,16,20,22,24,26,28,32,34,38,42,48,50,54,56,60,80), rfeControl=control,metric="ROC", trControl = trainctrl)

# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))*

str(ebayTrain)
str(ebayTest)
# Transform Dataset to remove discarded features
ebayTrain = ebayTrain[c(predictors(results), "Tsold")]
ebayTest = ebayTest[c(predictors(results))]

# Remove highly correlated features
#correlationMatrix <- cor(ebayTrain)
#highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.7)
#ebayTrain = ebayTrain[,-c(highlyCorrelated)]
#ebayTest = ebayTest[,-c(highlyCorrelated)]
