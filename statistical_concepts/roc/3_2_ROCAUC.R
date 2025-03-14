#ROCAUC
library(dplyr)
library(ggplot2)
library(pROC)
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
alpha <- 0.05  #Significance level

#Generate synthetic data using rnorm()
FVC_smokers <- rnorm(n_smokers, mean = mean_smokers, sd = sd_smokers)
FVC_non_smokers <- rnorm(n_non_smokers, mean = mean_non_smokers, sd = sd_non_smokers)

#Combine data
FVC <- c(FVC_smokers, FVC_non_smokers)
SmokingStatus <- rep(c("Smoker", "Non-Smoker"), times = c(n_smokers, n_non_smokers))

#Create data frame
data <- data.frame(
  FVC = FVC,
  SmokingStatus = SmokingStatus
)

#Plot boxplot of FVC by SmokingStatus
ggplot(data, aes(x = SmokingStatus, y = FVC, fill = SmokingStatus)) +
  geom_boxplot() +
  theme_minimal() +
  labs(
    title = "Distribution of FVC by Smoking Status",
    x = "Group",
    y = "FVC (liters)"
  )

#Create a Binary Outcome
#Define 'low lung function' as FVC < 4.5
data$FVC_binary <- ifelse(data$FVC < 4.5, 1, 0)

#Add a Predictor: Age (ages between 25 and 29)
set.seed(526)
total_n <- n_smokers + n_non_smokers
data$Age <- round(runif(total_n, min = 25, max = 29))

#Fit the Original Logistic Regression Model using SmokingStatus + Age
original_model <- glm(FVC_binary ~ SmokingStatus + Age, data = data, family = binomial)
data$PredictedProb <- predict(original_model, type = "response")

#Compute ROC and AUC for the original model
roc_original <- roc(data$FVC_binary, data$PredictedProb)
auc_original <- auc(roc_original)
cat("Original Model AUC:", auc_original, "\n")

#Plot the ROC curve for the original model
ggroc(roc_original) +
  ggtitle("ROC Curve for Predicting Low FVC (Original Model)") +
  xlab("False Positive Rate") +
  ylab("True Positive Rate") +
  theme_minimal()

#Further Improve the Model: Add a moderately correlated Biomarker predictor
#Biomarker is generated as FVC plus noise; it is predictive of FVC but not perfect.
set.seed(526)
data$Biomarker <- 1.0 * data$FVC + rnorm(total_n, mean = 0, sd = 0.25)

#Fit an extended logistic model including SmokingStatus, Age, and Biomarker
extended_model <- glm(FVC_binary ~ SmokingStatus + Age + Biomarker, data = data, family = binomial)
data$ExtendedPredProb <- predict(extended_model, type = "response")

#Compute ROC and AUC for the extended model
roc_extended <- roc(data$FVC_binary, data$ExtendedPredProb)
auc_extended <- auc(roc_extended)
cat("Extended Model AUC (with Biomarker):", auc_extended, "\n")

#Plot and compare ROC curves for both models
ggroc(list("Original Model" = roc_original, "Extended Model" = roc_extended)) +
  ggtitle("ROC Curve Comparison: Original vs Extended Model") +
  xlab("False Positive Rate") +
  ylab("True Positive Rate") +
  theme_minimal()


