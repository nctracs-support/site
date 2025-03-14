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
cat("Critical value:", round(t_critical, 3), "\n")  

#Calculate Power
effect_size <- (mean_non_smokers - mean_smokers) / pooled_sd 
power_result <- pwr.t2n.test(n1 = n_smokers, n2 = n_non_smokers, d = effect_size, sig.level = alpha, alternative = "greater")
print(power_result)

#Explore power changes (different factors)
# Increase sample sizes for both groups
n_smokers <- 36 * 2          # Increase from 36 to 72 smokers
n_non_smokers <- 64 * 2      # Increase from 64 to 128 non-smokers

#Calculate power again using the increased sample size
power_result_new <- pwr.t2n.test(
  n1 = n_smokers, 
  n2 = n_non_smokers, 
  d = effect_size, 
  sig.level = alpha, 
  alternative = "greater"
)

#Print the new power result
cat("Power of the test with increased sample size:", round(power_result_new$power, 3), "\n")

#Approach 2 - decrease variability
n_smokers <- 36          # Increase from 36 to 200 smokers
n_non_smokers <- 64      # Increase from 64 to 200 non-smokers
sd_smokers <- 0.4
sd_non_smokers <- 0.4
pooled_sd <- sqrt(((n_smokers - 1) * sd_smokers^2 + (n_non_smokers - 1) * sd_non_smokers^2) / (n_smokers + n_non_smokers - 2))
effect_size <- (mean_non_smokers - mean_smokers) / pooled_sd
power_result_new <- pwr.t2n.test(
  n1 = n_smokers, 
  n2 = n_non_smokers, 
  d = effect_size, 
  sig.level = alpha, 
  alternative = "greater"
)

#Print the new power result
cat("Power of the test with decreased variability:", round(power_result_new$power, 3), "\n")

#Approach 3 - increase effect size
effect_size <- 0.7
sd_smokers <- 0.6
sd_non_smokers <- 0.6
power_result_new <- pwr.t2n.test(
  n1 = n_smokers, 
  n2 = n_non_smokers, 
  d = effect_size, 
  sig.level = alpha, 
  alternative = "greater"
)

#Print the new power result
cat("Power of the test with increased effect size:", round(power_result_new$power, 3), "\n")

#Approach 4 - change significance level
# Change significance level to see its impact on power
alpha <- 0.1 #this number is free to change
effect_size <- 0.5
power_result_new <- pwr.t2n.test(
  n1 = n_smokers, 
  n2 = n_non_smokers, 
  d = effect_size, 
  sig.level = alpha,  # Stricter significance level
  alternative = "greater"
)

#Print the new power result
cat("Power of the test with reduced significance level:", round(power_result_new$power, 3), "\n")

