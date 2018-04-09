set.seed(1234567890)
library(neuralnet)

setwd("C:/Users/rakendra/Documents/MyCodes/MachineLearning/dataset/")
load("creditcard.Rdata")

## head(creditcard)
data <- creditcard

## Check no datapoint is missing, else needs to fix.
#apply(data, 2, function(x) sum(is.na(x)))

#create 70 percent of the data
index <- sample(1:nrow(data),round(0.70*nrow(data)))

## normalize the dataset
dataset <- data

#Normalize the numeric values
temp <- dataset[,1:(NCOL(data) - 1)]
scaled <- as.data.frame(scale(temp, center=TRUE, scale=TRUE))

## copy it back to the dataset
dataset[,1:(NCOL(data) - 1)] <- scaled
dataset$Class <- as.numeric(levels(data$Class))[data$Class]

## Create a training and test set for training
train <- dataset[index,]
test <- dataset[-index,]


n <- names(train)
f <- as.formula(paste("Class ~", paste(n[!n %in% "Class"], collapse = " + ")))

# ANN for studying the data
creditnet <- neuralnet(f, data=train, hidden=c(20, 20), lifesign = "minimal", linear.output = FALSE, threshold = 0.1)

## plot NN
plot(creditnet, rep="best")

## test the resulting output
temp_test <- test[,1:(NCOL(test) - 1)]
creditnet.results <- compute(creditnet, temp_test)
result <- data.frame(actual=test$Class, prediction=creditnet.results$net.result)

result$prediction <- round(result$prediction)

library(caret)
xtab <- table(result$prediction, result$actual)
confusionMatrix(xtab)
