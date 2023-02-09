# Assignment 3 of Lab 2 of course TDDE01,
# Machine Learning at Linkoping University, Sweden

########### Libraries #############
library(caret)
library(dplyr)
library(ggplot2)
library(ggfortify)
########### Libraries #############

############### Task 1 #################
# Scale all variables except of ViolentCrimesPerPop and implement PCA by using
# eigen(). Report how many components are needed to obtain at least 95% of
# variance in the data. What is the proportion of variation explained by each
# of the first two principal components?
############### Task 1 #################

# Reading and scaling data
data <- read.csv("data/communities.csv")
scaler <- preProcess(data %>% select(-ViolentCrimesPerPop))
data2 <- predict(scaler, data)

# PCA and eigen values
res <- prcomp(data2)

cov <- cov(data2)
eig <- eigen(cov)

# Components needed to obtain 95 % of variance in data:
cs <- cumsum(eig$values / sum(eig$values)*100)
which(cs >= 95)[1] #35 components

# proportion of variation by the two first principal components?
sum(eig$values[1:2]) # 41.97854

############### Task 2 #################
# Repeat PCA analysis by using princomp() function and make the trace plot of
# the first principle component. Do many features have a notable contribution
# to this component? report which 5 features contribute mostly (by the absolute
# value) to the first principle component. Comment whether these features have
# anything in common and whether they may have a logical relationship to the
# crime level. Also provide a plot of the PC scores in the coordinates (PC1,
# PC2) in which the color of the points is given by ViolentCrimesPerPop.
############### Task 2 #################

# PCA using princomp
res2 <- princomp(data2)

# trace plot of the first component.
plot(res2[["loadings"]][, 1], col = "blue", pch = 5, ylab = "")

# adding the 5 most contributing features by absolute value
top_5 <- head(sort(abs(res2[["loadings"]][, 1]), decreasing = TRUE), n = 5)
index_top_5 <- which(abs(res2[["loadings"]][, 1]) %in% top_5)
points(index_top_5, res2[["loadings"]][index_top_5, 1], col = "red", pch = 5)


# plot of the PC scores. Color of points is given by ViolentCrimesPerPop
autoplot(res2, colour = "ViolentCrimesPerPop") +
  labs(x = "PC1", y = "PC2", color = "Violent crimes per pop.")

############### Task 3 #################
# Split the original data into training and test (50/50) and scale both
# features and response appropriately, and estimate a linear regression model
# from training data in which ViolentCrimesPerPop is target and all other
# data columns are features. Compute training and test errors for these data
# and comment on the quality of model.
############### Task 3 #################

# Split data into train and test
n <- dim(data)[1]
set.seed(12345)
id <- sample(1:n, floor(n * 0.5))
train <- data[id, ]
test <- data[-id, ]

# scaling train and test data
scaler <- preProcess(train)
scaled_train <- predict(scaler, train)
scaled_test <- predict(scaler, test)

model <- lm(ViolentCrimesPerPop ~ ., data = scaled_train)

pred_train <- predict(model, type = "response")
pred_test <- predict(model, newdata = scaled_test, type = "response")

mse_train <- mean((scaled_train$ViolentCrimesPerPop - pred_train)^2)
mse_test <- mean((scaled_test$ViolentCrimesPerPop - pred_test)^2)

plot(scaled_test$ViolentCrimesPerPop, pred_test, xlab = "Actual",
     ylab = "Predictions", pch = 5, col = c("black", "blue"))
abline(0, 1)
legend("topleft", c("predicted", "actual"), col = c("blue", "black"), pch = 5)

############### Task 4 #################
# Implement a function that depends on parameter vector theta and represents
# the cost function for linear regression without intercept on the training
# data set. Afterwards, use BFGS method to optimize this cost with starting
# point theta_zero = 0 and compute training and test errors for every iteration
# number. Present a plot showing dependence of both errors on the iteration
# number and comment which iteration number is optimal according to the early
# stopping
############### Task 4 #################


# Used in function below.
x_train <- as.matrix(scaled_train %>% select(-ViolentCrimesPerPop))
x_test <- as.matrix(scaled_test %>% select(-ViolentCrimesPerPop))
y_train <- scaled_train$ViolentCrimesPerPop
y_test <- scaled_test$ViolentCrimesPerPop

multi_mse_test <- c()
multi_mse_train <- c()

cost <- function(theta) {
  mse_train <- mean((y_train - x_train %*% theta)^2)
  mse_test <- mean((y_test - x_test %*% theta)^2)

  multi_mse_train <<- c(multi_mse_train, mse_train)
  multi_mse_test <<- c(multi_mse_test, mse_test)
  return(mse_train)
}

theta <- rep(0, 100)
opt <- optim(par = theta, fn = cost, method = "BFGS")

plot(multi_mse_train[500:8000],
     ylim = c(0.2, 0.8), ylab = "Mean square error",
     xlab = "# of iterations",
     pch = ".", xlim = c(500, 7000))

points(multi_mse_test[500:8000], col = "blue", pch = ".")
opt_iteration <- which.min(multi_mse_test)

abline(v = opt_iteration - 500, col = "grey", lty = "dashed")
legend("topright", c("test data", "training data"), col = c("blue", "black"), lwd = 1)

# training and test error in the optimal model
multi_mse_train[opt_iteration] # 0.3032999
multi_mse_test[opt_iteration] # 0.4002329
