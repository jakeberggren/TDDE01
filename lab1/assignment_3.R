# Assigment 3 of Laboration 1 of course TDDE01,
# Machine Learning at Linkoping University, Sweden

# Read data
data <- read.csv("data/pima-indians-diabetes.csv", header = FALSE)

###########
# Task 1 - Make a scatterplot showing a plasma glucose concentration on age
# where abservations are colored by diabetes level.
###########

library(ggplot2)
# extracting Plasma glucose concentration, age and diabetes from data
pgc <- data$V2
age <- data$V8
diabetes <- data$V9

# Plot
ggplot(data, aes(x = age, y = pgc, color = as.factor(diabetes))) +
  geom_point() +
  labs(x = "Age", y = "Plasma Glucose Concentration", color = "Diabetes") +
  scale_color_manual(values = c("black", "deepskyblue2"))

###########
# Task 2 - Train a logistic regression model with y = Diabetes as target
# x1 = plasma glucose concentration and x2 = Age as features and make a
# prediction for all observations by using r = 0.5 as the classification
# threshold. Also compute the training misclassification error and make a
# scatter plot with the predicted values of Diabetes as a color.
###########

logistic_model <- glm(diabetes ~ pgc + age, data = data, family = "binomial")

summary(logistic_model)

prediction <- predict(logistic_model, data, type = "response")
prediction_1 <- ifelse(prediction > 0.5, 1, 0)

# Confusion matrix
cm <- table(prediction_1, diabetes)
misclass <- (1 - sum(diag(cm)) / length(prediction_1)) # 0.2552083 or 0.2630208?

# Plot
ggplot(data, aes(x = age, y = pgc, color = as.factor(prediction_1))) +
  geom_point() +
  labs(x = "Age", y = "Plasma Glucose Concentration",
       color = "Predicted Diabetes") +
  scale_color_manual(values = c("black", "deepskyblue2"))

###########
# Task 3 - Use the model in step 2 to a) report the equation of the decision
# boundry between the two classes and
# b) add a curve showing this boundary to the scatter plot.
###########

# plot including boundary line
ggplot(data, aes(x = age, y = pgc, color = as.factor(prediction_1))) +
  geom_point() +
  scale_color_manual(values = c("black", "deepskyblue2")) +
  geom_abline(slope = coef(logistic_model)[["age"]]
              / (-coef(logistic_model)[["pgc"]]),
              intercept = coef(logistic_model)[["(Intercept)"]]
              / (-coef(logistic_model)[["pgc"]]), color = "darkslategrey") +
  labs(x = "Age", y = "Plasma Glucose Concentration",
       color = "Predicted Diabetes")


###########
# Task 4 - Make same kind of plot as in step 2 but use thresholds r = 0.2
# and r=0.8.
###########

prediction_2 <- ifelse(prediction > 0.2, 1, 0)
prediction_3 <- ifelse(prediction > 0.8, 1, 0)

# plot with treshold r = 0.2
ggplot(data, aes(x = age, y = pgc, color = as.factor(prediction_2))) +
  geom_point() +
  labs(x = "Age", y = "Plasma Glucose Concentration",
       color = "Predicted Diabetes") +
  scale_color_manual(values = c("black", "deepskyblue2"))

# plot with treshold r = 0.8
ggplot(data, aes(x = age, y = pgc, color = as.factor(prediction_3))) +
  geom_point() +
  labs(x = "Age", y = "Plasma Glucose Concentration",
       color = "Predicted Diabetes") +
  scale_color_manual(values = c("black", "deepskyblue2"))


###########
# Task 5 - Perform a basis function expansion trick by computing new features.
# Create a scatterplot of the same kind as in step 2.
###########

data$z1 <- pgc^4
data$z2 <- pgc^3 * age
data$z3 <- pgc^2 * age^2
data$z4 <- pgc * age^3
data$z5 <- age^4
y <- diabetes

model <- glm(y ~ pgc + age + z1 + z2 + z3 + z4 + z5,
             data = data, family = "binomial")
summary(model)

prediction_basis <- predict(model, data, type = "response")
prediction_basis <- ifelse(prediction_basis > 0.5, 1, 0)

# Confusion Matrix
cm_basis <- table(prediction_basis, y)
misclass_basis <- (1 - sum(diag(cm_basis)) / length(prediction_basis)) # 0.2447917

# Plot prediction with basis function expansion
ggplot(data, aes(x = age, y = pgc, color = as.factor(prediction_basis))) +
  geom_point() +
  labs(x = "Age", y = "Plasma Glucose Concentration",
       color = "Predicted Diabetes") +
  scale_color_manual(values = c("black", "deepskyblue2"))

