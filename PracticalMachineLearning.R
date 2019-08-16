library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(corrplot)
library(gbm)
library(caret)
library(randomForest)

train_data <- read.csv('./pml-training.csv', header=T)
valid_data <- read.csv('./pml-testing.csv', header=T)
dim(train_data)

trainData<- train_data[, colSums(is.na(train_data)) == 0]
validData <- valid_data[, colSums(is.na(valid_data)) == 0]

trainData <- trainData[, -c(1:7)]
validData <- validData[, -c(1:7)]

set.seed(1234) 
inTrain <- createDataPartition(trainData$classe, p = 0.8, list = FALSE)
trainData <- trainData[inTrain, ]
testData <- trainData[-inTrain, ]

NZV <- nearZeroVar(trainData)
trainData <- trainData[, -NZV]
testData  <- testData[, -NZV]
dim(trainData)
dim(testData)

cormat <- cor(trainData[, -53])
corrplot(cormat, order = "FPC", method = "color", type = "lower", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))

highlyCorrelated = findCorrelation(cormat, cutoff=0.75)
names(trainData)[highlyCorrelated]

set.seed(12345)
decisionTreeMod1 <- rpart(classe ~ ., data=trainData, method="class")
fancyRpartPlot(decisionTreeMod1)

predictTreeMod1 <- predict(decisionTreeMod1, testData, type = "class")
cmtree <- confusionMatrix(predictTreeMod1, testData$classe)
cmtree

cmtree$overall['Accuracy']

controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modRF1 <- train(classe ~ ., data=trainData, method="rf", trControl=controlRF)
modRF1$finalModel

predictRF1 <- predict(modRF1, newdata=testData)
cmrf <- confusionMatrix(predictRF1, testData$classe)
cmrf

cmrf$overall['Accuracy']

plot(modRF1)

set.seed(12345)
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modGBM  <- train(classe ~ ., data=trainData, method = "gbm", trControl = controlGBM, verbose = FALSE)
modGBM$finalModel
print(modGBM)

predictGBM <- predict(modGBM, newdata=testData)
cmGBM <- confusionMatrix(predictGBM, testData$classe)
cmGBM

Results <- predict(modRF1, newdata=validData)
Results