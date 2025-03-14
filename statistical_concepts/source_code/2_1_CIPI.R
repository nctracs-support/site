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

##########
#PI and CI
##########
#we want to estimate lung function (FVC) in smokers
data.here <- data.frame(FVC_smokers)
#Fit a linear model (simple mean model)
model <- lm(FVC_smokers ~ 1, data = data.here)  #Mean model with no predictors
#we calculate both the confidence interval and the prediction interval around the mean FVC
#Calculate 95% Confidence Interval for Mean FVC
conf_int <- predict(model, interval = "confidence", level = 0.95)
conf_int
#Calculate 95% Prediction Interval for Individual FVC
pred_int <- predict(model, interval = "prediction", level = 0.95)
pred_int

#Create a data frame with mean, confidence interval, and prediction interval
intervals_data <- data.frame(
  FVC = data.here$FVC_smokers,
  Mean = conf_int[, "fit"],
  CI_Lower = conf_int[, "lwr"],
  CI_Upper = conf_int[, "upr"],
  PI_Lower = pred_int[, "lwr"],
  PI_Upper = pred_int[, "upr"]
)

#Generate a summary of the confidence and prediction intervals
mean_FVC <- mean(intervals_data$Mean)
CI_Lower <- mean(intervals_data$CI_Lower); CI_Lower
CI_Upper <- mean(intervals_data$CI_Upper); CI_Upper
PI_Lower <- mean(intervals_data$PI_Lower); PI_Lower
PI_Upper <- mean(intervals_data$PI_Upper); PI_Upper

#Plotting
ggplot(intervals_data, aes(x = 1, y = Mean)) +
  geom_point(aes(y = FVC), color = "grey50", alpha = 0.6) +  # individual data points
  geom_errorbar(aes(ymin = CI_Lower, ymax = CI_Upper), width = 0.1, color = "blue", size = 1) +
  geom_errorbar(aes(ymin = PI_Lower, ymax = PI_Upper), width = 0.2, color = "red", size = 1) +
  geom_point(aes(y = mean_FVC), color = "black", size = 3) +  # Mean point
  labs(
    title = "Confidence Interval vs Prediction Interval",
    y = "Forced Vital Capacity (FVC)"
  ) +
  theme_minimal() +
  scale_x_continuous(labels = NULL, breaks = NULL) +  # Hide x-axis labels and ticks
  annotate("text", x = 1.2, y = mean_FVC, label = "Mean FVC", color = "black") +
  annotate("text", x = 1.2, y = CI_Lower - 0.1, label = "95% CI", color = "blue") +
  annotate("text", x = 1.2, y = PI_Lower - 0.1, label = "95% PI", color = "red")

