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

#Create a data frame combining both groups
synthetic_data <- data.frame(
  Group = rep(c("Smoker", "Non-Smoker"), times = c(n_smokers, n_non_smokers)),
  FVC = c(FVC_smokers, FVC_non_smokers)
)

#Pooled sd
pooled_sd <- sqrt(((n_smokers - 1) * sd_smokers^2 + (n_non_smokers - 1) * sd_non_smokers^2) / (n_smokers + n_non_smokers - 2))

#Perform t-test
t_statistic <- (mean_smokers - mean_non_smokers) / (pooled_sd * sqrt((1 / n_smokers) + (1 / n_non_smokers)))

#Degree of freedom
df <- n_smokers + n_non_smokers - 2

#Critical value
t_critical <- qt(alpha, df, lower.tail = TRUE)  # One-sided test

#Output the results
cat("Pooled standard deviation:", round(pooled_sd, 3), "\n")
cat("t-statistic:", round(t_statistic, 3), "\n")
cat("Degrees of freedom:", df, "\n")
cat("Critical value:", round(t_critical, 3), "\n")  #this value is used to construct the rejection region

