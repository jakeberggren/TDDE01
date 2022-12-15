# Assigment 1 of Laboration 1 of course TDDE01,
# Machine Learning at Linkoping University, Sweden

###########
# Task 1 - Import data and divide into train, validation and test (50%/25%/25%)
###########

# Read data
data <- read.csv("data/optdigits.csv", header = FALSE)

# Splitting dataset into train(50%), validation(25%) and test data(25%).
# set.seed is used to ensure the same results every time
n <- dim(data)[1]
set.seed(12345)
id <- sample(1:n, floor(n * 0.5))
data_train <- data[id, ]

id1 <- setdiff(1:n, id)
set.seed(12345)
id2 <- sample(id1, floor(n * 0.25))
data_valid <- data[id2, ]

id3 <- setdiff(id1, id2)
data_test <- data[id3, ]


###########
# Task 2 - Use training data to fit 30-nearest neighbor classifier and estimate
# confusion matrices and misclassification rates for training and test data
###########

data_train$X0.26 <- as.factor(data_train$X0.26)
data_test$X0.26 <- as.factor(data_test$X0.26)
library(kknn)
# Using data to fit 30-nearest neighbor on train and test data
model_train <- kknn(X0.26 ~ ., train = data_train, test = data_train, k = 30,
                    kernel = "rectangular")
model_test <- kknn(X0.26 ~ ., train = data_train, test = data_test, k = 30,
                   kernel = "rectangular")

prediction_train <- predict(model_train)
prediction_test <- predict(model_test)

# Confusion matrices
cm_train <- table(prediction_train, data_train$X0.26)
cm_test <- table(prediction_test, data_test$X0.26)

# Function to compute misclassification rate
misclass <- function(x, x1) {
  n <- length(x)
  return(1 - sum(diag(table(x, x1))) / n)
}

# Computing misclassification rates from confusion matrices
misclass_train <- misclass(prediction_train, data_train$X0.26)
misclass_test <- misclass(prediction_test, data_test$X0.26)


###########
# Task 3 - Find cases of digit "8" which were easy/hard to classify.
# Visualize with heatmap().
###########

# Combining dataset of 8s in the training data and the probability of
# being an 8 in the kknn prob table
train8 <- data_train[data_train$X0.26 == "8", ]
prob8 <- model_train[["prob"]][data_train$X0.26 == "8", ]

# Find index of the 3 hardest and the 2 easiest 8s to classify
prob8_hardest <- sort(prob8[, "8"], decreasing = FALSE,
                      index.return = TRUE)$ix[1:3]
prob8_easiest <- sort(prob8[, "8"], decreasing = TRUE,
                      index.return = TRUE)$ix[1:2]

# Remove last column and reshape into 8x8 numeric matrix
prob8_hardest_1 <- matrix(as.numeric(train8[prob8_hardest[1], ][-65]),
                          nrow = 8, ncol = 8)
prob8_hardest_2 <- matrix(as.numeric(train8[prob8_hardest[2], ][-65]),
                          nrow = 8, ncol = 8)
prob8_hardest_3 <- matrix(as.numeric(train8[prob8_hardest[3], ][-65]),
                          nrow = 8, ncol = 8)

# Visualize
heatmap(t(prob8_hardest_1), Colv = "Rowv", Rowv = NA)
heatmap(t(prob8_hardest_2), Colv = "Rowv", Rowv = NA)
heatmap(t(prob8_hardest_3), Colv = "Rowv", Rowv = NA)

# Remove last column and reshape into 8x8 numeric matrix
prob8_easiest_1 <- matrix(as.numeric(train8[prob8_easiest[1], ][-65]),
                          nrow = 8, ncol = 8)
prob8_easiest_2 <- matrix(as.numeric(train8[prob8_easiest[2], ][-65]),
                          nrow = 8, ncol = 8)

# Visualize
heatmap(t(prob8_easiest_1), Colv = "Rowv", Rowv = NA)
heatmap(t(prob8_easiest_2), Colv = "Rowv", Rowv = NA)


###########
# Task 4 - Fit K-nearest neighbor classifiers to training data for values
# of K = 1-30 and plot the dependence of the training and validation
# misclassification errors on the value on the value of K.
###########

multi_misclass_train <- c()
multi_misclass_valid <- c()
for (x in 1:30) {
  model_train_temp <- kknn(X0.26 ~ ., train = data_train, test = data_train,
                           k = x, kernel = "rectangular")
  prediction_train_temp <- predict(model_train_temp)
  multi_misclass_train <- append(multi_misclass_train,
    misclass(prediction_train_temp, data_train$X0.26))

  model_valid_temp <- kknn(X0.26 ~ ., train = data_train, test = data_valid,
                           k = x, kernel = "rectangular")
  prediction_valid_temp <- predict(model_valid_temp)
  multi_misclass_valid <- append(
    multi_misclass_valid,
    misclass(prediction_valid_temp, data_valid$X0.26))
}

best_k <- which.min(multi_misclass_valid)
best_k_model <- kknn(X0.26 ~ ., train = data_train, test = data_test,
                     k = best_k, kernel = "rectangular")
best_k_model_prediction <- best_k_model$fitted.values
best_k_misclass <- misclass(best_k_model_prediction, data_test$X0.26)

plot(multi_misclass_train * 100, col = "blue", ylim = c(0, 6),
     xlab = "K-nearest neighbors", ylab = "Misclassification rate in %",
     pch = 20)
points(multi_misclass_valid * 100, col = "green", pch = 20)
points(7, best_k_misclass * 100, col = "red", pch = 20)
mtext("Difference in misclassification
      dependent on number of neighbours in KNN")
legend("topleft", c("misclass. training data", "misclass. validation data",
                   "missclass. test data with optimal K value"),
       fill = c("blue", "green", "red"))


###########
# Task 5
###########
