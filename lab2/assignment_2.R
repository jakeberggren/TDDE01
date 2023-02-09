# Assignment 2, Lab 2 in course TDDE01,
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

# Read data
data <- read.csv2("data/bank-full.csv", stringsAsFactors = TRUE)

# Remove "duration" variable
data <- data %>% select(-duration)

# Data partitioning
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

# clean up environment
rm(id, id1, id2, n)

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
tree_b <- tree(y ~ ., data = train, control = tree.control(nrow(train), minsize = 7000))
tree_c <- tree(y ~ ., data = train, control = tree.control(nrow(train), mindev = 0.0005))

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
plot(2:50, train_score[2:50], type = "b", col = "red", ylim = c(8000, 12000),
  main = "Optimal tree depth", ylab = "Deviance", xlab = "Number of leaves")
points(2:50, valid_score[2:50], type = "b", col = "blue")
legend("topright", c("train data", "validation data"), fill = c("red", "blue"))

# Optimal number of leaves
opt_train <- which.min(train_score[-1])
opt_valid <- which.min(valid_score[-1])
data.frame(train = opt_train + 1,
           valid = opt_valid + 1)

# Opitmal tree
opt_tree <- prune.tree(tree_c, best = opt_valid)

# visualization of tree structure with optimal number of leaves
plot(opt_tree)
opt_tree

# Bias variance trade off: 
# Bias is the models ability to capture the true relationship. So a high bias
# means that the model will underfit. However as bias gets lower, the model
# varience gets higher. This means that the model will be very sensitive to
# change and thus not have as good predictive properties. So a model with high
# varience is overfitted to the data.

############### Task 4 #################
# Estimate the confusion matrix, accuracy and F1 score for the test data by
# using the optimal model from step 3. Comment whether the model has a good
# predictive power and which of the measures (accuracy or F1-score) should
# be preferred here.
############### Task 4 #################

# prediction on test data with optimal model
pred.opt <- predict(opt.tree, newdata = test, type = "class")

# confusion matrix of optimal model
cm <- table(pred.opt, test$y)

# computing accuracy and F1 score:
tp <- cm[4]
tn <- cm[1]
fp <- cm[2]
p <- cm[3] + cm[4]
n <- cm[1] + cm[2]
rec <- (tp / p)
prec <- (tp / (tp + fp))

rbind(
  acc = ((tp + tn) / (p + n)),
  f1 = ((2 * prec * rec) / (prec + rec))
)

# judging from the accuracy and the F1 score, the model has poor predictive power.
# The accuracy is high due to many correct "no" classifications. However, we
# should prefer F1 score here since the classes are very imbalanced. The F1 score
# is low which indicates a model with poor predictive power.
  
############### Task 5 #################
# Perform a decision tree classification of the test data with the following
# loss matrix:
# yes 0 5
# no  1 0
# and report the confusion matrix for the test data. Compare the results with
# the results from step 4 and discuss how the rates has changed and why.
############### Task 5 #################

# Decision tree classification with Loss matrix:
loss.matrix <- matrix(c(0,1,5,0), byrow = TRUE, nrow = 2)

prob <- predict(opt.tree, newdata = test)
losses <- prob %*% loss.matrix
best.i <- apply(losses, MARGIN = 1, FUN = which.min)
pred <- levels(test$y)[best.i]
cm2 <- table(pred, test$y)
cm2

# computing acc and f1 to compare:
tp <- cm2[4]
tn <- cm2[1]
fp <- cm2[2]
p <- cm2[3] + cm2[4]
n <- cm2[1] + cm2[2]
rec <- (tp / p)
prec <- (tp / (tp + fp))

rbind(
  acc = ((tp + tn) / (p + n)),
  f1 = ((2 * prec * rec) / (prec + rec))
)

# The model now punishes false negatives more and thus it has become more balanced.
# We can see that the accuracy decreased but the F1 score has increased. So since
# we prefer F1 score due to imbalanced classes this model performs better and has
# better predictive power compared to the previous model.

############### Task 6 #################
# Compute TPR and FPR values and plot corresponding ROC curves. Make conclusions.
# Why would a precision recall curve be a better option in this case?
############### Task 6 #################

prob.tree <- predict(opt.tree, newdata = test, type = "vector")

# Computing ROC curves for FPR and TPR for the optimal tree:
tpr <- numeric()
fpr <- numeric()

for (i in seq(from = 0.05, to = 0.95, by = 0.05)) {
  decision <- ifelse(prob.tree[, 2] > i, "yes", "no")
  cm.temp <- table(decision, test$y)
  
  true.pos <- cm.temp[4]
  false.pos <- cm.temp[2]
  pos <- cm.temp[3] + cm.temp[4]
  neg <- cm.temp[1] + cm.temp[2]
  tpr <- c(tpr, true.pos / pos) 
  fpr <- c(fpr, false.pos / neg)
}

plot(fpr, tpr, pch = 5, type = "b")

# regression model
reg.model <- glm(y ~ ., data = train, family = "binomial")
prob.log <- predict(reg.model, newdata = test, type = "response")

# Computing ROC curves for FPR and TPR for the regression model:
tpr <- numeric()
fpr <- numeric()

for (i in seq(from = 0.05, to = 0.95, by = 0.05)) {
  decision <- ifelse(prob.log > i, "yes", "no")
  cm.temp <- table(decision, test$y)
  
  true.pos <- cm.temp[4]
  false.pos <- cm.temp[2]
  pos <- cm.temp[3] + cm.temp[4]
  neg <- cm.temp[1] + cm.temp[2]
  tpr <- c(tpr, true.pos / pos)
  fpr <- c(fpr, false.pos / neg)
}

points(fpr, tpr, pch = 5, type = "b", col = "blue")

# The plot shows that the tree model is slightly better since it has a greater
# area under the curve (However they are very similar). Once again since the
# classes are imbalanced, a better choice in this case would be to look at
# the precision recall curves instead.
