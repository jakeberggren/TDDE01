# Lab 3 block 1 of 732A99/TDDE01/732A68 Machine Learning
# Author: jose.m.pena@liu.se
# Made for teaching purposes

library(kernlab)
set.seed(1234567890)

data(spam)
foo <- sample(nrow(spam))
spam <- spam[foo, ]
spam[, -58] <- scale(spam[, -58])
tr <- spam[1:3000, ]
va <- spam[3001:3800, ]
trva <- spam[1:3800, ]
te <- spam[3801:4601, ]

by <- 0.3
err_va <- NULL
for (i in seq(by, 5, by)) {
  filter <- ksvm(type ~ ., data = tr, kernel = "rbfdot",
                 kpar = list(sigma = 0.05), C = i, scaled = FALSE)
  mailtype <- predict(filter, va[, -58])
  t <- table(mailtype, va[, 58])
  err_va <- c(err_va, (t[1, 2] + t[2, 1]) / sum(t))
}

filter0 <- ksvm(type ~ ., data = tr, kernel = "rbfdot",
                kpar = list(sigma = 0.05), C = which.min(err_va) * by,
                scaled = FALSE)
mailtype <- predict(filter0, va[, -58])
t <- table(mailtype, va[, 58])
err0 <- (t[1, 2] + t[2, 1]) / sum(t)
err0

filter1 <- ksvm(type ~ ., data = tr, kernel = "rbfdot",
                kpar = list(sigma = 0.05), C = which.min(err_va) * by,
                scaled = FALSE)
mailtype <- predict(filter1, te[, -58])
t <- table(mailtype, te[, 58])
err1 <- (t[1, 2] + t[2, 1]) / sum(t)
err1

filter2 <- ksvm(type ~ ., data = trva, kernel = "rbfdot",
                kpar = list(sigma = 0.05), C = which.min(err_va) * by,
                scaled = FALSE)
mailtype <- predict(filter2, te[, -58])
t <- table(mailtype, te[, 58])
err2 <- (t[1, 2] + t[2, 1]) / sum(t)
err2

filter3 <- ksvm(type ~ ., data = spam, kernel = "rbfdot",
                kpar = list(sigma = 0.05), C = which.min(err_va) * by,
                scaled = FALSE)
mailtype <- predict(filter3, te[, -58])
t <- table(mailtype, te[, 58])
err3 <- (t[1, 2] + t[2, 1]) / sum(t)
err3

# 3. Implementation of SVM predictions. (Students code from here)

library(pracma)

sv <- alphaindex(filter3)[[1]]
support.vectors <- spam[sv, -58]
co <- coef(filter3)[[1]]
inte <- - b(filter3)

kernel.function <- rbfdot(0.05)

k <- NULL
dot.products <- NULL
for (i in 1:10) { # Produce predictions for the first 10 points in the dataset.

  k2 <- NULL

  for (j in seq_along(sv)) {
    k2 <- unlist(support.vectors[j, ])
    sample <- unlist(spam[i, -58])
    dot.prod <- kernel.function(sample, k2)
    dot.products <- c(dot.products, dot.prod)
  }

  start.index <- 1 + length(sv) * (i - 1)
  end.index <- length(sv) * i
  prediction <- co %*% dot.products[start.index:end.index] + inte
  k <- c(k, prediction)
}

k
predict(filter3, spam[1:10, -58], type = "decision")
