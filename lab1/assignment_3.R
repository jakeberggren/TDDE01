# Assignment 3, Lab 1 in course TDDE01,
# Machine Learning at Linkoping University, Sweden

########### Libraries #############
library(ggplot2)
########### Libraries #############

# Read data
data <- read.csv("data/pima-indians-diabetes.csv", header = FALSE)


#########################
######## Task 1 #########
#########################

# extracting Plasma glucose concentration, age and diabetes from data
pgc <- data$V2
age <- data$V8
diabetes <- data$V9

# Plot
ggplot(data, aes(x = age, y = pgc, color = as.factor(diabetes))) + geom_point() +
  labs(x = "Age", y = "Plasma Glucose Concentration", color = "Diabetes") +
  scale_color_manual(values = c("black", "deepskyblue2"))


#########################
######## Task 2 #########
#########################

logistic_model <- glm(diabetes ~ pgc + age, data = data, family = "binomial")

summary(logistic_model)

prediction <- predict(logistic_model, data, type = "response")
prediction_1 <- ifelse(prediction > 0.5, 1, 0)

# Confusion matrix
cm <- table(prediction_1, diabetes)
misclass <- (1 - sum(diag(cm)) / length(prediction_1)) # 0.2552083 or 0.2630208?

# Plot
ggplot(data, aes(x = age, y = pgc, color = as.factor(prediction_1))) + geom_point() +
  labs(x = "Age", y = "Plasma Glucose Concentration", color = "Predicted Diabetes") +
  scale_color_manual(values = c("black", "deepskyblue2"))


#########################
######## Task 3 #########
#########################

# plot including boundary line
ggplot(data, aes(x = age, y = pgc, color = as.factor(prediction_1))) + geom_point() +
  scale_color_manual(values = c("black", "deepskyblue2")) +
  geom_abline(slope = coef(logistic_model)[["age"]]
              / (-coef(logistic_model)[["pgc"]]),
              intercept = coef(logistic_model)[["(Intercept)"]]
              / (-coef(logistic_model)[["pgc"]]), color = "darkslategrey") +
  labs(x = "Age", y = "Plasma Glucose Concentration", color = "Predicted Diabetes")


#########################
######## Task 4 #########
#########################

prediction_2 <- ifelse(prediction > 0.2, 1, 0)
prediction_3 <- ifelse(prediction > 0.8, 1, 0)

# plot with treshold r = 0.2
ggplot(data, aes(x = age, y = pgc, color = as.factor(prediction_2))) + geom_point() +
  labs(x = "Age", y = "Plasma Glucose Concentration", color = "Predicted Diabetes") +
  scale_color_manual(values = c("black", "deepskyblue2"))

# plot with treshold r = 0.8
ggplot(data, aes(x = age, y = pgc, color = as.factor(prediction_3))) + geom_point() +
  labs(x = "Age", y = "Plasma Glucose Concentration", color = "Predicted Diabetes") +
  scale_color_manual(values = c("black", "deepskyblue2"))


#########################
######## Task 5 #########
#########################

data$z1 <- pgc^4
data$z2 <- pgc^3 * age
data$z3 <- pgc^2 * age^2
data$z4 <- pgc * age^3
data$z5 <- age^4
y <- diabetes

model <- glm(y ~ pgc + age + z1 + z2 + z3 + z4 + z5, data = data, family = "binomial")
summary(model)

prediction_basis <- predict(model, data, type = "response")
prediction_basis <- ifelse(prediction_basis > 0.5, 1, 0)

# Confusion Matrix and misclasification rate
cm_basis <- table(prediction_basis, y)
misclass_basis <- (1 - sum(diag(cm_basis)) / length(prediction_basis)) # 0.2447917

# Plot prediction with basis function expansion
ggplot(data, aes(x = age, y = pgc, color = as.factor(prediction_basis))) + geom_point() +
  labs(x = "Age", y = "Plasma Glucose Concentration", color = "Predicted Diabetes") +
  scale_color_manual(values = c("black", "deepskyblue2"))
