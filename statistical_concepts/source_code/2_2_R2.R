#Loading necessary packages
library(dplyr)
library(ggplot2)
library(pwr)

#Set seed
set.seed(526)

#Number of samples
n_smokers <- 36
n_non_smokers <- 64

#Define the statistics for each group
mean_smokers <- 4.7
mean_non_smokers <- 5.0
sd_smokers <- 0.6
sd_non_smokers <- 0.6
alpha <- 0.05  # Significance level

#Generate synthetic data using rnorm()
FVC_smokers <- rnorm(n_smokers, mean = mean_smokers, sd = sd_smokers)
FVC_non_smokers <- rnorm(n_non_smokers, mean = mean_non_smokers, sd = sd_non_smokers)

###################
#R2 and Adjusted R2
###################

#Combine data
FVC <- c(FVC_smokers, FVC_non_smokers)
SmokingStatus <- rep(c("Smoker", "Non-Smoker"), times = c(n_smokers, n_non_smokers))

#Add an unnecessary predictor (random noise)
RandomPredictor <- rnorm(n_smokers + n_non_smokers)

#Create the data frame
data <- data.frame(FVC = FVC, SmokingStatus = SmokingStatus, RandomPredictor = RandomPredictor)

#Fit model with SmokingStatus only
model1 <- lm(FVC ~ SmokingStatus, data = data)
summary(model1)


#Fit model with SmokingStatus and RandomPredictor
model2 <- lm(FVC ~ SmokingStatus + RandomPredictor, data = data)
summary(model2)
#Extract R-squared and Adjusted R-squared for both models
r_squared_model1 <- summary(model1)$r.squared
adj_r_squared_model1 <- summary(model1)$adj.r.squared

r_squared_model2 <- summary(model2)$r.squared
adj_r_squared_model2 <- summary(model2)$adj.r.squared

#Display the results
cat("Model 1 (SmokingStatus only):\n")
cat("R-squared:", round(r_squared_model1, 3), "\n")
cat("Adjusted R-squared:", round(adj_r_squared_model1, 3), "\n\n")

cat("Model 2 (SmokingStatus + RandomPredictor):\n")
cat("R-squared:", round(r_squared_model2, 3), "\n")
cat("Adjusted R-squared:", round(adj_r_squared_model2, 3), "\n")



