set.seed(1234567890)

library("neuralnet")

setwd("C:/Users/rakendra/Documents/MyCodes/MachineLearning/dataset/")
dataset <- read.csv("creditset.csv")
head(dataset)

rows <- nrow(dataset)
columns <- ncol(dataset)

rowstodivide <- floor(rows * 0.4)

## extract a set to train the NN
trainset <- dataset[1:rowstodivide,]

## select the test set
testset <- dataset[(rowstodivide+1):rows, ]

## build the neural network.
creditnet <- neuralnet(default10yr ~ LTI + age, trainset, hidden = 4, lifesign = "minimal", linear.output = FALSE, threshold = 0.1)


## plot NN
plot(creditnet, rep="best")

## test the resulting output
temp_test <- subset(testset, select = c("LTI", "age"))

creditnet.results <- compute(creditnet, temp_test)

head(temp_test)

result <- data.frame(actual=testset$default10yr, prediction=creditnet.results$net.result)
head(result)

result$prediction <- round(result$prediction)
result[100:115]