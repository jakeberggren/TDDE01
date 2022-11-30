# Assigment 1 of Laboration 2 of course TDDE01,
# Machine Learning at Linkoping University, Sweden

########### Libraries #############
library(dplyr)
library(tidyr)
library(tree)
library(MLmetrics)
library(caret)
library(rpart)
library(rpart.plot)

############### Task 1 #################
# Import the data to R, remove variable "duration" and divide into
# training/validation/test as 40/30/30.
############### Task 1 #################

data <- read.csv("data/bank-full.csv", sep = ";", stringsAsFactors = TRUE)
data <- data %>% select(-duration)

# data partition
n <- dim(data)[1]
set.seed(12345)
id <- sample(1:n, floor(n * 0.4))
train <- data[id, ]
id1 <- setdiff(1:n, id)
set.seed(12345)
id2 <- sample(id1, floor(n * 0.3))
valid <- data[id2, ]
id3 <- setdiff(id1, id2)
test <- data[id3, ]

############### Task 2 #################
# Fit decision trees to the training data so that you can change default
# settings one by one (i.e. not simultaneously):
# a. Decision Tree with default settings.
# b. Decision Tree with smallest allowed node size equal to 7000.
# c. Decision trees minimum deviance to 0.0005.
# and report the misclassification rates for the training and validation data.
# Which model is the best one among these three? Report how changing the
# devience and node size affected the size of the trees and explain why.
############### Task 2 #################

# Computing trees with settings a, b and c
tree_a <- tree(y ~ ., data = train)
tree_b <- tree(y ~ .,
  data = train,
  control = tree.control(nrow(train), minsize = 7000)
)
tree_c <- tree(y ~ .,
  data = train,
  control = tree.control(nrow(train), mindev = 0.0005)
)

# Predictions on validation set
pred_a <- predict(tree_a, newdata = valid, type = "class")
pred_b <- predict(tree_b, newdata = valid, type = "class")
pred_c <- predict(tree_c, newdata = valid, type = "class")

# Summaries to report misclassification rate for trees on train set.
summary(tree_a) # 0.1048
summary(tree_b) # 0.1048
summary(tree_c) # 0.09362

# Calculating misclassification rate for validation data
misclass_valid_a <- mean(pred_a != valid$y)
misclass_valid_b <- mean(pred_b != valid$y)
misclass_valid_c <- mean(pred_c != valid$y)
# Print validation misclass
data.frame(
  a = misclass_valid_a,
  b = misclass_valid_b,
  c = misclass_valid_c
)

############### Task 3 #################
# Use training and validation sets to choose optimal tree depth in the model 2c:
# study the trees up to 50 leaves. Present a graph of the dependence of
# deviances for the training and the validation data on the number of leaves
# and interpret this graph in terms of bias-variance trade off. Report optimal
# amount of leaves and which variables seem to be most important for decision
# making in this tree. Interpret the information provided by the tree structure.
############### Task 3 #################

train_score <- rep(0, 50)
valid_score <- rep(0, 50)

# Find optimal number of leaves.
for (i in 2:50) {
  pruned_tree <- prune.tree(tree_c, best = i)
  valid_pred <- predict(pruned_tree, newdata = valid, type = "tree")
  train_score[i] <- deviance(pruned_tree)
  valid_score[i] <- deviance(valid_pred)
}

# Visualize with plot
plot(2:50, train_score[2:50],
  type = "b", col = "red", ylim = c(8000, 12000),
  main = "Optimal tree depth", ylab = "Deviance", xlab = "Number of leaves"
)
points(2:50, valid_score[2:50], type = "b", col = "blue")
legend("topright", c("train data", "validation data"), fill = c("red", "blue"))

# Optimal number of leaves
opt_train <- which.min(train_score[-1])
opt_valid <- which.min(valid_score[-1])
data.frame(train = opt_train + 1,
           valid = opt_valid + 1)

opt_tree <- prune.tree(tree_c, best = 22) # vad är bästa träddjupet?

# visualization of tree structure with optimal number of leaves
plot(opt_tree)
opt_tree

############### Task 4 #################
# Estimate the confusion matrix, accuracy and F1 score for the test data by
# using the optimal model from step 3. Comment whether the model has a good
# predictive power and which of the measures (accuracy or F1-score) should
# be preferred here.
############### Task 4 #################

pred_test <- predict(opt_tree, newdata = test, type = "class")

# Confusion matrix
confusion_matrix <- table(pred_test, test$y)
conf_matrix <- confusionMatrix(pred_test, test$y)

# Using MLmetrics to estimate accuracy and F1 score.
Accuracy(pred_test, test$y) # 0.891035, accuracy for no
F1_Score(pred_test, test$y) # 0.9414004

# Manual estimations of accuracy and F1 score
tp <- confusion_matrix[4]
fp <- confusion_matrix[2]
fn <- confusion_matrix[3]
p <- confusion_matrix[4] + confusion_matrix[3]
n <- confusion_matrix[1] + confusion_matrix[2] 

accuracy <- (tp + fp)/(p+n) # 0.023665, accuracy for yes
f1 <- 2*tp/(2*tp + fp + fn) # 0.224554, f1 score for yes
  
############### Task 5 #################
# Perform a decision tree classification of the test data with the following
# loss matrix:
# yes 0 5
# no  1 0
# and report the confusion matrix for the test data. Compare the results with
# the results from step 4 and discuss how the rates has changed and why.
############### Task 5 #################

# Alternative 1
prob <- predict(opt_tree, newdata = test)
losses <- prob%*%matrix(c(0,1,5,0), byrow = TRUE, nrow=2)
best_i <- apply(losses, MARGIN = 1, FUN = which.min)
pred <- levels(test$y)[best_i]
table(pred, test$y)
  
# Alternative 2
fit <- rpart(y ~ ., data=test, method="class",
             parms=list(loss=matrix(c(0,1,5,0), byrow = TRUE, nrow=2)))

predict_fit <- predict(fit, test, type = "class")
cm_fit <- table(predict_fit ,test$y)

############### Task 6 #################

############### Task 6 #################
 














