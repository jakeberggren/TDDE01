# Assignment 2 of Laboration 1 of course TDDE01,
# Machine Learning at Linkoping University, Sweden

########### Libraries #############
library(caret)
library(dplyr)
########### Libraries #############

#########################
######## Task 1 #########
#########################

# read file
data <- read.csv("data/parkinsons.csv")

# divide data into training and test
n <- dim(data)[1]
set.seed(12345)
id <- sample(1:n, floor(n * 0.6))
data_train <- data[id, ]
data_test <- data[-id, ]

# Scale data using library caret
scaler <- preProcess(data_train)
train_scaled <- predict(scaler, data_train)
test_scaled <- predict(scaler, data_test)

#########################
######## Task 2 #########
#########################

model_train <- lm(motor_UPDRS ~ . - age - sex - subject. - test_time - total_UPDRS, data = train_scaled)
model_test <- lm(motor_UPDRS ~ . - age - sex - subject. - test_time - total_UPDRS, data = test_scaled)

summary(model_train)
summary(model_test)

pred_train <- predict(model_train)
pred_test <- predict(model_test)
mse_train <- mean((train_scaled$motor_UPDRS - pred_train)^2)
mse_test <- mean((test_scaled$motor_UPDRS - pred_test)^2)


#########################
######## Task 3 #########
#########################

# Extracting data needed for functions below
n <- nrow(train_scaled)
y <- as.matrix(train_scaled$motor_UPDRS)
x <- as.matrix(train_scaled %>% select(Jitter...:PPE))

Loglikelihood <- function(theta, sigma) {
  return(-n / 2 * log(2 * pi * sigma^2) - (1 / (2 * sigma^2)) * sum((y - x %*% theta)^2))
}

Ridge <- function(theta, lambda) {
  sigma <- tail(theta, n = 1)
  theta <- theta[-17]
  return(lambda * sum(theta^2) - Loglikelihood(theta, sigma))
}

RidgeOpt <- function(lambda) {
  return(optim(rep(1, 17), fn = Ridge, lambda = lambda, method = "BFGS"))
}

I <- as.matrix(diag(16))
DF <- function(lambda) {
  return(sum(diag((x %*% solve((t(x) %*% x + lambda * I))) %*% t(x))))
}


#########################
######## Task 4 #########
#########################

theta_opt_1 <- RidgeOpt(lambda = 1)
theta_opt_100 <- RidgeOpt(lambda = 100)
theta_opt_1000 <- RidgeOpt(lambda = 1000)

x_train <- as.matrix(train_scaled %>% select(Jitter...:PPE))
x_test <- as.matrix(test_scaled %>% select(Jitter...:PPE))
opt_1 <- as.matrix(theta_opt_1$par[-17])
opt_100 <- as.matrix(theta_opt_100$par[-17])
opt_1000 <- as.matrix(theta_opt_1000$par[-17])

# Prediction of motor_UPDRS values for training and test data
pred_train_opt_1 <- x_train %*% opt_1
pred_test_opt_1  <- x_test %*% opt_1
pred_train_opt_100 <- x_train %*% opt_100
pred_test_opt_100  <- x_test %*% opt_100
pred_train_opt_1000 <- x_train %*% opt_1000
pred_test_opt_1000  <- x_test %*% opt_1000

# Training and test Mean Square Error
mse_train_opt_1 <- mean((train_scaled$motor_UPDRS - pred_train_opt_1)^2)
mse_test_opt_1  <- mean((test_scaled$motor_UPDRS  - pred_test_opt_1)^2)
mse_train_opt_100 <- mean((train_scaled$motor_UPDRS - pred_train_opt_100)^2)
mse_test_opt_100  <- mean((test_scaled$motor_UPDRS  - pred_test_opt_100)^2)
mse_train_opt_1000 <- mean((train_scaled$motor_UPDRS - pred_train_opt_1000)^2)
mse_test_opt_1000  <- mean((test_scaled$motor_UPDRS  - pred_test_opt_1000)^2)

# Degrees of freedom
df_1 <- DF(lambda = 1) # 13.860736
df_100 <- DF(lambda = 100) # 9.924871
df_1000 <- DF(lambda = 1000) # 5.643925
