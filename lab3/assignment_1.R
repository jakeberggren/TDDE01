# Assignment 1 of Lab 3 of course TDDE01,
# Machine Learning at Linkoping University, Sweden

########### Libraries #############
library(geosphere)
library(dplyr)
########### Libraries #############


# Set seed and read files
set.seed(1234567890)
stations <- read.csv("data/stations.csv", fileEncoding = "ISO-8859-1")
temps <- read.csv("data/temps50k.csv")

# merge data sets
st <- merge(stations, temps, by = "station_number")

# Smoothing coefficients
h.distance <- 100000
h.date <- 20
h.time <- 6

# The point to predict
a <- 58.4274  # longitude
b <- 14.8263  # latitude

p.date <- "2023-11-04" # The date to predict

times <- c("04:00:00", "06:00:00", "08:00:00", "10:00:00",
           "12:00:00", "14:00:00", "16:00:00", "18:00:00",
           "20:00:00", "22:00:00", "24:00:00")

# Filter out times and dates which should not be included in data.
st <- st %>%  filter(time %in% times & date < p.date)

# instantiation of temperature vectors
temp.ksum <- vector(length = length(times))
temp.kprod <- vector(length = length(times))

# Function for a general Gaussian kernel
gaussianKernel <- function(x, h) {
  return(exp(-((x / h)^2)))
}

# Distance kernel
dist.diff <- distHaversine(cbind(st[, 4:5]), cbind(a, b))
k.dist <- gaussianKernel(dist.diff, h.distance)

# Day kernel
day.diff <- as.numeric(difftime(p.date, st$date, units = "days")) %% 365
k.date <- gaussianKernel(day.diff, h.date)

# Hour kernel for each time and fill temp vector with values.
for (i in seq_along(times)) {
  # Hour kernel
  hour.diff <- abs(as.numeric(difftime(strptime(times[i], format = "%H:%M:%S"),
                                       strptime(cbind(st$time),
                                                format = "%H:%M:%S"),
                                       units = "hours")))
  k.hours <- gaussianKernel(hour.diff, h.time)

  # sum of the three kernel (Task 1)
  kernel.sum <- k.dist + k.date + k.hours
  kernel.prod <- k.dist * k.date * k.hours

  # Append temperatures to temperature vector.
  temp.ksum[i] <- sum(kernel.sum %*% st$air_temperature) / sum(kernel.sum)
  temp.kprod[i] <- sum(kernel.prod %*% st$air_temperature) / sum(kernel.prod)
}

# Plot of both kernel models
plot(temp.ksum, type = "b", pch = 5,
     ylim = c(min(temp.ksum[which.min(temp.ksum)],
                  temp.kprod[which.min(temp.kprod)]),
     max(temp.ksum[which.max(temp.ksum)],
                     temp.kprod[which.max(temp.kprod)])),
     xaxt = "n", main = "Predicted Temperatures", xlab = "Time of day",
     ylab = "Temperature", col = "blue")

points(temp.kprod, type = "b", pch = 5, xaxt = "n", col = "red")

axis(1, at = seq_along(times), labels = substring(times, 1, 2))
legend("bottomright", c("Sum. kernel", "Prod. kernel"),
       col = c("blue", "red"), pch = 5, lty = 2, box.lty = 0, cex = 0.8)
