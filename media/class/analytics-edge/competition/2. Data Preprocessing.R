### Data Load
eBayiPadTrain = read.csv("eBayiPadTrain.csv", stringsAsFactors=FALSE)
eBayiPadTest = read.csv("eBayiPadTest.csv", stringsAsFactors=FALSE)

### Remove dirty examples
#eBayiPadTrain = subset(eBayiPadTrain, carrier != "Other") # test set과 factor level 개수 서로 같도록 유지하기 위해...
eBayiPadTrain = subset(eBayiPadTrain, productline != "iPad 5")
eBayiPadTrain = subset(eBayiPadTrain, productline != "iPad mini Retina")

# 원래 Text set도 같이 dirty example을 삭제해줘야 하나, 제출하는데 버그가 걸리므로, 개수 유지를 위해 실행하지 않는다.
#eBayiPadTest = subset(eBayiPadTest, carrier != "Other")
#eBayiPadTest = subset(eBayiPadTest, productline != "iPad 5")
#eBayiPadTest = subset(eBayiPadTest, productline != "iPad mini Retina")


### Factor 변수화 순서와 dirty example 순서 조심. text와 train의 factor변수의 level은 서로 맞춰줘야 한다. dirty example을 삭제하면서 factor level이 달라 질 수도 있다. 
### Convert Factor variable
eBayiPadTrain$condition = as.factor(eBayiPadTrain$condition)
eBayiPadTrain$cellular = as.factor(eBayiPadTrain$cellular)
eBayiPadTrain$carrier = as.factor(eBayiPadTrain$carrier)
eBayiPadTrain$color = as.factor(eBayiPadTrain$color)
eBayiPadTrain$storage = as.factor(eBayiPadTrain$storage)
eBayiPadTrain$productline = as.factor(eBayiPadTrain$productline)

eBayiPadTest$condition = as.factor(eBayiPadTest$condition)
eBayiPadTest$cellular = as.factor(eBayiPadTest$cellular)
eBayiPadTest$carrier = as.factor(eBayiPadTest$carrier)
eBayiPadTest$color = as.factor(eBayiPadTest$color)
eBayiPadTest$storage = as.factor(eBayiPadTest$storage)
eBayiPadTest$productline = as.factor(eBayiPadTest$productline)


### ### word variable과 기존 variable의 중복 방지. Text Analysis할 때 꼭 필요
eBayiPadTrain$Tbiddable = eBayiPadTrain$biddable
eBayiPadTrain$Tstartprice = eBayiPadTrain$startprice
eBayiPadTrain$Tcondition = eBayiPadTrain$condition
eBayiPadTrain$Tcellular = eBayiPadTrain$cellular
eBayiPadTrain$Tcarrier = eBayiPadTrain$carrier
eBayiPadTrain$Tcolor = eBayiPadTrain$color
eBayiPadTrain$Tstorage = eBayiPadTrain$storage
BayiPadTrain$Tproduction = eBayiPadTrain$production
eBayiPadTrain$Tsold = eBayiPadTrain$sold
eBayiPadTrain$TUniqueID = eBayiPadTrain$UniqueID

eBayiPadTrain$biddable = NULL
eBayiPadTrain$startprice = NULL
eBayiPadTrain$condition = NULL  # 얘내들은 categorical variable로 변환 
eBayiPadTrain$cellular = NULL
eBayiPadTrain$carrier = NULL
eBayiPadTrain$color = NULL
eBayiPadTrain$storage = NULL
eBayiPadTrain$production = NULL
eBayiPadTrain$sold = NULL
eBayiPadTrain$UniqueID = NULL

eBayiPadTest$Tbiddable = eBayiPadTest$biddable
eBayiPadTest$Tstartprice = eBayiPadTest$startprice
eBayiPadTest$Tcondition = eBayiPadTest$condition
eBayiPadTest$Tcellular = eBayiPadTest$cellular
eBayiPadTest$Tcarrier = eBayiPadTest$carrier
eBayiPadTest$Tcolor = eBayiPadTest$color
eBayiPadTest$Tstorage = eBayiPadTest$storage
eBayiPadTest$Tproduction = eBayiPadTest$production
eBayiPadTest$Tsold = eBayiPadTest$sold
eBayiPadTest$TUniqueID = eBayiPadTest$UniqueID

eBayiPadTest$biddable = NULL
eBayiPadTest$startprice = NULL
eBayiPadTest$condition = NULL
eBayiPadTest$cellular = NULL
eBayiPadTest$carrier = NULL
eBayiPadTest$color = NULL
eBayiPadTest$storage = NULL
eBayiPadTest$production = NULL
#eBayiPadTest$sold = NULL : Test set에서는 dependent variable sold가 없다.
eBayiPadTest$UniqueID = NULL

### Text Mining - add all kind of frequent words, then do feature selection
library(tm)
# add all frequent words
eBayTrain = eBayiPadTrain
eBayTest = eBayiPadTest
CorpusDescription = Corpus(VectorSource(c(eBayTrain$description, eBayTest$description)))
CorpusDescription = tm_map(CorpusDescription, content_transformer(tolower), lazy=TRUE)
CorpusDescription = tm_map(CorpusDescription, PlainTextDocument, lazy=TRUE)
CorpusDescription = tm_map(CorpusDescription, removePunctuation, lazy=TRUE)
CorpusDescription = tm_map(CorpusDescription, removeWords, stopwords("english"), lazy=TRUE)
# This has to be run multiple times due to some bug i assume. X100 is some Pen for ipad. Removing apple, ipad and item, since they
# provide no information gain
CorpusDescription = tm_map(CorpusDescription, removeWords, c("X100", "x100", "appl", "apple", "ipad", "item"), lazy=TRUE)
CorpusDescription = tm_map(CorpusDescription, stemDocument, lazy=TRUE)
dtm = DocumentTermMatrix(CorpusDescription)
sparse = removeSparseTerms(dtm, 0.990)
DescriptionWords = as.data.frame(as.matrix(sparse))
colnames(DescriptionWords) = make.names(colnames(DescriptionWords))
DescriptionWordsTrain = head(DescriptionWords, nrow(eBayiPadTrain))
DescriptionWordsTest = tail(DescriptionWords, nrow(eBayiPadTest))

# Encoding of categorial features ( cor계산 위해서 + alpha )
#eBayiPadTrain = cbind(eBayiPadTrain[-grep("condition", colnames(eBayiPadTrain))], model.matrix( ~ 0 + condition, eBayiPadTrain))
#eBayiPadTrain = cbind(eBayiPadTrain[-grep("productline", colnames(eBayiPadTrain))], model.matrix( ~ 0 + productline, eBayiPadTrain))
#eBayiPadTrain = cbind(eBayiPadTrain[-grep("carrier", colnames(eBayiPadTrain))], model.matrix( ~ 0 + carrier, eBayiPadTrain))
#eBayiPadTrain = cbind(eBayiPadTrain[-grep("color", colnames(eBayiPadTrain))], model.matrix( ~ 0 + color, eBayiPadTrain))
#eBayiPadTrain = cbind(eBayiPadTrain[-grep("storage", colnames(eBayiPadTrain))], model.matrix( ~ 0 + storage, eBayiPadTrain))
#eBayiPadTrain = cbind(eBayiPadTrain[-grep("cellular", colnames(eBayiPadTrain))], model.matrix( ~ 0 + cellular, eBayiPadTrain))

#eBayiPadTest = cbind(eBayiPadTest[-grep("condition", colnames(eBayiPadTest))], model.matrix( ~ 0 + condition, eBayiPadTest))
#eBayiPadTest = cbind(eBayiPadTest[-grep("productline", colnames(eBayiPadTest))], model.matrix( ~ 0 + productline, eBayiPadTest))
#eBayiPadTest = cbind(eBayiPadTest[-grep("carrier", colnames(eBayiPadTest))], model.matrix( ~ 0 + carrier, eBayiPadTest))
#eBayiPadTest = cbind(eBayiPadTest[-grep("color", colnames(eBayiPadTest))], model.matrix( ~ 0 + color, eBayiPadTest))
#eBayiPadTest = cbind(eBayiPadTest[-grep("storage", colnames(eBayiPadTest))], model.matrix( ~ 0 + storage, eBayiPadTest))
#eBayiPadTest = cbind(eBayiPadTest[-grep("cellular", colnames(eBayiPadTest))], model.matrix( ~ 0 + cellular, eBayiPadTest))

### combine word variables
eBayiPadTrain = cbind(eBayiPadTrain, DescriptionWordsTrain)
eBayiPadTest = cbind(eBayiPadTest, DescriptionWordsTest)

### finish
ebayTrain = eBayiPadTrain
ebayTrain$description = NULL
ebayTrain$UniqueID = NULL
names(ebayTrain) = make.names(names(ebayTrain), unique=TRUE)



### Same process for test data
ebayTest = eBayiPadTest
ebayTest$description = NULL
names(ebayTest) = make.names(names(ebayTest), unique=TRUE)







# 최종 마무리 
# eBayiPadTrain -> ebayTrain
# eBayiPadTest -> ebayTest

