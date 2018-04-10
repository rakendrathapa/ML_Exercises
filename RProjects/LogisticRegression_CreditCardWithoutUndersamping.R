#Logistic Regression Without Undersampling.
set.seed(1234567890)

setwd("C:/Users/rakendra/Documents/MyCodes/MachineLearning/ML_Exercises/dataset/")
load("creditcard.Rdata")

## head(creditcard)
data <- creditcard

## Check no datapoint is missing, else needs to fix.
#apply(data, 2, function(x) sum(is.na(x)))

#create 60 percent of the test data
index <- sample(1:nrow(data),round(0.60*nrow(data)))

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

# Traditional Logistic Regression Model.
creditLR <- glm(f, data=train, family = binomial())

# Calculating the ROC curve for the model.
# 1. Predic the reponse from test
creditLR.result<-predict(creditLR,type='response',test)

resultframe <- data.frame(actual=test$Class, prediction=creditLR.result)
resultframe$prediction <- round(resultframe$prediction)

# Confusion matrix
library(caret)
xtab <- table(resultframe$prediction, resultframe$actual)
confusionMatrix(xtab)
confusionMatrix(data=as.factor(resultframe$prediction), reference = as.factor(resultframe$actual), positive = "1")


#create new column
resultframe["newclass"] <- ifelse(resultframe["actual"]==0 & resultframe["prediction"]==0, "TN",
                             ifelse(resultframe["actual"]==0 & resultframe["prediction"]==1, "FP",
                                    ifelse(resultframe["actual"]==1 & resultframe["prediction"]==0, "FN", "TP")))

#Create raw confusion matrix using "table" function
(conf.val <- table(resultframe["newclass"]))


# 2. Store the prediction in results
result <-  prediction(creditLR.result, test$Class)

perf <- performance(result,"tpr","fpr")
plot(perf)

performance(result, "auc")

#get results of terms in regression
g <- predict(creditLR, type='terms', test)

# function to pick top 3 reasons
# works by sorting coefficient terms in equation
# and selecting top 3 in sort for each loan scored
ftopk <- function(x,top=3){
  res=names(x)[order(x, decreasing = TRUE)][1:top]
  paste(res,collapse=";",sep="")
}
# Application of the function using the top 3 rows
topk=apply(g,1,ftopk,top=3)

plot(perf,col='red',lty=1,main='ROC curve for Logistic Regression')
