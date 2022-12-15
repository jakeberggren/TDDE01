# Assignment 1 of Lab 2 of course TDDE01,
# Machine Learning at Linkoping University, Sweden

########### Libraries #############
library(dplyr)
library(ggplot2)
library(glmnet)
library(caret)
########## Read and divide data ########
data <- read.csv("data/tecator.csv")

# Divide into test and train data

n <- dim(data)[1]
set.seed(12345)
id <- sample(1:n, floor(n * 0.5))
train <- data[id, ]
test <- data[-id, ]

############### Task 1 #################
# model "fat" as a linear regression with channels as features. Report
# underlying probabilistic model, fit the linear regression to the training
# data and estimate the training and test errors. Comment on the quality of
# fit and prediciton and therefore on the quality of the model.

# fitting model to training data
df_train <- train %>% select(Fat, Channel1:Channel100)
df_test <- test %>% select(Fat, Channel1:Channel100)

model <- lm(Fat ~ ., data = df_train)
summary(model) # summary of underlying probabilistic model

# Predicitons made on both train and test data.
pred_train <- predict(model, type = "response")
pred_test <- predict(model, df_test, type = "response")

mse_train <- mean((df_train$Fat - pred_train)^2) # 0.00570911
mse_test <- mean((df_test$Fat - pred_test)^2) # 722.4294


############### Task 3 #################
# Fit the LASSO regression model to the training data. Present a plot
# illustrating how the regression coefficients depend on the log of the
# penalty factor and intepret this plot. What value of the penalty factor
# can be chosen if we want to select a model with only three features?

x <- as.matrix(train %>% select(-Sample, -Fat, -Protein, -Moisture))
y <- as.matrix(train %>% select(Fat))

lasso <- glmnet(x, y, alpha = 1, family = "gaussian")
plot(lasso, xvar = "lambda", label = TRUE)


############### Task 4 #################
# Repeat step 3 bit fit ridge regression instead of LASSO regression. Compare
# plot from steps 3 and 4.

ridge <- glmnet(x, y, alpha = 0, family = "gaussian")
plot(ridge, xvar = "lambda", label = TRUE)


############### Task 5 #################
# Use cross validation with default number of folds to compute the optimal LASSO
# model. Present a plot showing the dependence of the CV score on log lambda and
# comment on how the CV score changes with log lambda. Report the optimal lambda
# and how many variables were chosen in this model. Does the information
# displayed in the plot suggest that the optimal lambda value results in a
# statistically significantly better prediction than log lambda = -4? Finally,
# create a scatter plot of the original test versus predicted test values for
# the model corresponding to the optimal lambda and comment on whether the model
# predictions are good.

# Using cross validation 
cv <- cv.glmnet(x, y, alpha = 1, family = "gaussian")
plot(cv)
lambda_opt <- cv$lambda.min # 0.004561105
coef(cv, s = "lambda.min")
summary(cv)
cv_pred <- predict(cv, newx = x, s = lambda_opt)

plot(test$Fat, col = "blue", ylim = c(0, 60), ylab = "Fat levels",
     main = "Original test vs model with optimal lamdbda")
points(cv_pred, col = "green")
legend("topright", c("actuals", "predicted using cross validation"),
       fill = c("blue", "green"))

# Still needs to be done
# finally,
# create a scatter plot of the original test versus predicted test values for
# the model corresponding to the optimal lambda and comment on whether the model
# predictions are good.
