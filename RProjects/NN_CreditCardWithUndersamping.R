set.seed(1234567890)
library(neuralnet)

setwd("C:/Users/ThapaRak/Documents/MyCodes/ML_Exercises/dataset/")
load("creditcard.Rdata")

## head(creditcard)
data <- creditcard

## Check no datapoint is missing, else needs to fix.
#apply(data, 2, function(x) sum(is.na(x)))

#create 70 percent of the data
index <- sample(1:nrow(data),round(0.60*nrow(data)))

## normalize the dataset. Remove the Time column from calculation.
dataset <- data[,2:(NCOL(data))]

#Normalize the numeric values
temp <- dataset[,1:(NCOL(dataset) - 1)]
scaled <- as.data.frame(scale(temp, center=TRUE, scale=TRUE))

## copy it back to the dataset
dataset[,1:(NCOL(data) - 1)] <- scaled
dataset$Class <- as.numeric(levels(data$Class))[data$Class]

#write.csv(dataset, creditcard.csv, col.names = F, row.names = F)

## Create a training and test set for training
train <- dataset[index,]
test <- dataset[-index,]

n <- names(train)
f <- as.formula(paste("Class ~", paste(n[!n %in% "Class"], collapse = " + ")))

# Data Preprocessing.
# 1. Calculate Beta. Based on Percent.
# 2. Calculate Threshold. Based on original N+/(N).
# 2. Send the data for training.
# 3. Adjust Posterior Probability.
# 4. Adjust Threshold.
# 5. Make Prediction.
library(ROSE)
NPlus <- sum(train$Class == 1)
NMinus <- sum(train$Class == 0)
percent=0.50
Beta = ((1/percent) - 1) * (NPlus/ NMinus)
threshold = NPlus / (NPlus + NMinus)

undersampled_train <- ovun.sample(f, data = train, method = "under", p=percent, seed = 1)$data
table(undersampled_train$Class)

# ANN for studying the data
creditnet <- neuralnet(f, data=undersampled_train, hidden=c(20), lifesign = "minimal", linear.output = FALSE, threshold = 0.1)

## plot NN
# plot(creditnet, rep="best")

## test the resulting output
temp_test <- test[,1:(NCOL(test) - 1)]
creditnet.results <- compute(creditnet, temp_test)
result <- data.frame(actual=test$Class, probability=creditnet.results$net.result)

# Calibrate the Probability
result$calibratedProbab = (Beta * result$probability) / ((Beta * result$probability) - result$probability + 1)
result$prediction[result$calibratedProbab < threshold] <- 0
result$prediction[result$calibratedProbab >= threshold] <- 1

result$probability <- result$calibratedProbab

library(caret)
mat <- confusionMatrix(data=as.factor(result$prediction), reference = as.factor(result$actual), positive = "1")


#create new column
result["newclass"] <- ifelse(result["actual"]==0 & result["prediction"]==0, "TN",
                             ifelse(result["actual"]==0 & result["prediction"]==1, "FP",
                                    ifelse(result["actual"]==1 & result["prediction"]==0, "FN", "TP")))

#Create raw confusion matrix using "table" function
(conf.val <- table(result["newclass"]))

# Accuracy Function
# Accuracy = ( TP + TN ) / (TP + FP + TN + FN)
#Function areguments are 1st: data frame, 2nd: actual class column name, 
#3rd: predicted class column name
my.accuracy <- function(df, actual, predicted){
  y <- as.vector(table(df[,predicted], df[,actual]))
  names(y) <- c("TN", "FP", "FN", "TP")
  acur <- (y["TP"]+y["TN"])/sum(y)
  return(as.numeric(acur))
}
my.accuracy(result, "actual", "prediction")



# Classification Error Rate
#(FP + FN) / (TP + FP + TN + FN)
my.class.err <- function(df, actual.col, predicted.col){
  actual <- df[actual.col]
  pred <- df[predicted.col]
  newvec <-  ifelse(actual==0 & pred==0, "TN",ifelse(actual==0 & pred==1, "FP",
                                                     ifelse(actual==1 & pred==0, "FN", "TP")))
  conf <-table(newvec)
  c.err <- (conf["FP"]+conf["FN"])/sum(conf)
  return(as.numeric(c.err))
}
my.class.err(result, "actual", "prediction")

# Precision
# Precision : What proportion of the +ve identification was actually correct.
# TP / (TP + FP)
#Precision
my.precision <- function(df, actual.col, predicted.col){
  actual <- df[actual.col]
  pred <- df[predicted.col]
  newvec <-  ifelse(actual==0 & pred==0, "TN",ifelse(actual==0 & pred==1, "FP",
                                                     ifelse(actual==1 & pred==0, "FN", "TP")))
  conf <-table(newvec)
  precis <- conf["TP"]/(conf["TP"]+conf["FP"])
  return(as.numeric(precis))
}
my.precision(result, "actual", "prediction")

# Sensitivity Function
# True Postive Rate (TPR)
# TP / (TP + FN)
my.sensitivity <- function(df, actual.col, predicted.col){
  actual <- df[actual.col]
  pred <- df[predicted.col]
  newvec <-  ifelse(actual==0 & pred==0, "TN",ifelse(actual==0 & pred==1, "FP",
                                                     ifelse(actual==1 & pred==0, "FN", "TP")))
  conf <-table(newvec)
  sens <- conf["TP"]/(conf["TP"]+conf["FN"])
  return(as.numeric(sens))
}
my.sensitivity(result, "actual", "prediction")

# Specificity Function
# True Negative Rate
# TN / (TN + FP)
my.specificity <- function(df, actual.col, predicted.col){
  actual <- df[actual.col]
  pred <- df[predicted.col]
  newvec <-  ifelse(actual==0 & pred==0, "TN",ifelse(actual==0 & pred==1, "FP",
                                                     ifelse(actual==1 & pred==0, "FN", "TP")))
  conf <-table(newvec)
  spec <- conf["TN"]/(conf["TN"]+conf["FP"])
  return(as.numeric(spec))
}
my.specificity(result, "actual", "prediction")


# F1-score
# (2 * Precision * Specificity) / (Precision + Sensitivity)
my.f1score <- function(df, actual.col, predicted.col){
  prec <- my.precision(df, actual.col, predicted.col)
  sens <- my.sensitivity(df, actual.col, predicted.col)
  f1s <- 2*prec*sens/(prec+sens)
  return(as.numeric(f1s))
}
my.f1score(result, "actual", "prediction")

# ROC Curve
# Receiver Operating Characteristic curve plot and calculate AUC
# for a dataframe with classification and probablity column.
# ROC Curve - Plots of TPR(Sensitivity) aganist FPR. 
# TRP = Sensitivity = TP / (TP + FN)
# FPR = Total False Positives / (Total False Positive + Total True Negatives) = FP / (FP + TN)
# FPR = ((FP + TN) - TN) / (FP + TN) = 1 - TN/(FP + TN) = 1 - Specificity
# ROC curve = Plot of Sensitivity aganist (1 - Specificity)
# ROC curve and AUC calculation using trapezoid method
my.rocfunc <- function(df, actual, prob){
  thresh <- seq(0.01,1,0.01)
  xval <- vector()
  yval <- vector() #empty vector
  for(x in thresh){
    df["pred"] <- ifelse(df[prob] < x, 0, 1)    
    xval <- c(xval,1-my.specificity(df, actual, "pred"))
    yval <- c(yval, my.sensitivity(df, actual, "pred"))
  }
  xydf <- data.frame(xval, yval)
  xydf <- xydf[complete.cases(xydf),]
  id <- order(xydf$xval)  #find the id of sorted values
  #AUC inside a 1 by 1 square calculated by trapezoid method
  b <- -diff(xydf$xval)  #get the consecutive differences
  my.auc <- min(xydf$xval)*min(xydf$yval)/2 +  #area below the minimum y value
    sum(b*xydf$yval[-length(xydf$yval)])+(1-max(xydf$xval))*max(xydf$yval) +  #area of rectangles and triangles
    ((1-max(xydf$xval))*(1-max(xydf$yval))/2) #area after the max y value
  return(c(plot(yval~xval, xydf, type="l", xlab="1-Specificity", ylab="Sensitivity", main="ROC Curve"), abline(0,1,lty=2), my.auc))
}
my.rocfunc(result, "actual" , "probability")

# Alternate (Statistical) AUC calculation
test_class <- result$actual
test_score <- result$probability
pos <- test_score[test_class == '1']
neg <- test_score[test_class == '0']
#Find the probability
p <- replicate(100000, sample(pos, size=1) > sample(neg, size=1))
#AUC value
(pAUC<-mean(p))

# ROC curve using "pROC" package
library(pROC)
roc.val <- roc(actual~probability, result)
plot(roc.val, main="ROC plot")

# AUC calculation using "pROC" package
## Call:
## roc.formula(formula = actual ~ probability, data = result)
## 
## Data: scored.probability in 124 controls (class 0) < 57 cases (class 1).
## Area under the curve: 0.8503
#AUC
roc.val$auc

library(ROCR)
annmode.prediction = ROCR::prediction(result$probability, result$actual)
annmodel.performance <- ROCR::performance(annmode.prediction, "tpr", "fpr")

# AUC of a ROC
# Probablity that a randomly chosen Postive instance (score) will rank higher than 
# a randomly chosen negative instance (score)
# AUC = P(score(positive) > score(negative))
annmodel.AUC <- ROCR::performance(annmode.prediction, "auc")@y.values[[1]]
