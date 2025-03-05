#Load necessary library
library(ggplot2)
library(gridExtra)

#Read the data from the CSV files
setwd("") #add your local device path here (where you save the csv files)
train_data_loaded <- read.csv("3_1_train_data.csv")
test_data_loaded <- read.csv("3_1_test_data.csv")

#Fit models
#Simple Model: Linear
simple_model <- lm(LungFunction ~ CigarettesPerDay, data = train_data_loaded)
simple_model

#Complex Model: Quadratic
complex_model <- lm(LungFunction ~ CigarettesPerDay + I(CigarettesPerDay^2), data = train_data_loaded)
complex_model

#Make/add predictions to both datasets
train_data_loaded$SimplePred <- predict(simple_model, newdata = train_data_loaded)
train_data_loaded$ComplexPred <- predict(complex_model, newdata = train_data_loaded)

test_data_loaded$SimplePred <- predict(simple_model, newdata = test_data_loaded)
test_data_loaded$ComplexPred <- predict(complex_model, newdata = test_data_loaded)

#Calculate Mean Squared Error (MSE)
calculate_mse <- function(actual, predicted) {
  mean((actual - predicted)^2)
}

mse_simple_train <- calculate_mse(train_data_loaded$LungFunction, train_data_loaded$SimplePred)
mse_complex_train <- calculate_mse(train_data_loaded$LungFunction, train_data_loaded$ComplexPred)

mse_simple_test <- calculate_mse(test_data_loaded$LungFunction, test_data_loaded$SimplePred)
mse_complex_test <- calculate_mse(test_data_loaded$LungFunction, test_data_loaded$ComplexPred)

#Print MSE results
cat("Mean Squared Errors:\n")
cat("Training MSE:\n")
cat("  Simple Model:", mse_simple_train, "\n")
cat("  Complex Model:", mse_complex_train, "\n\n")
cat("Testing MSE:\n")
cat("  Simple Model:", mse_simple_test, "\n")
cat("  Complex Model:", mse_complex_test, "\n")

#Determine consistent axis ranges
x_range <- range(c(train_data_loaded$CigarettesPerDay, test_data_loaded$CigarettesPerDay))
y_range <- range(c(train_data_loaded$LungFunction, test_data_loaded$LungFunction))

#Create the plots
#Training Data Plot
training_plot <- ggplot() +
  geom_point(data = train_data_loaded, aes(x = CigarettesPerDay, y = LungFunction), color = "blue", alpha = 0.4) +
  geom_line(data = train_data_loaded, aes(x = CigarettesPerDay, y = SimplePred, color = "Simple Model (Linear)"), linetype = "dashed", size = 1.2) +
  geom_line(data = train_data_loaded, aes(x = CigarettesPerDay, y = ComplexPred, color = "Complex Model (Quadratic)"), linetype = "solid", size = 1.2) +
  labs(
    title = "Model Fit on Training Data",
    x = "Cigarettes Smoked Per Day",
    y = "Lung Function (FVC)",
    color = "Model Type"
  ) +
  scale_color_manual(values = c(
    "Simple Model (Linear)" = "red",
    "Complex Model (Quadratic)" = "green"
  )) +
  coord_cartesian(xlim = x_range, ylim = y_range) +
  theme_minimal() +
  theme(legend.position = "right", legend.title = element_text(size = 12))

#Testing Data Plot
testing_plot <- ggplot() +
  geom_point(data = test_data_loaded, aes(x = CigarettesPerDay, y = LungFunction), color = "blue", alpha = 0.4) +
  geom_line(data = test_data_loaded, aes(x = CigarettesPerDay, y = SimplePred, color = "Simple Model (Linear)"), linetype = "dashed", size = 1.2) +
  geom_line(data = test_data_loaded, aes(x = CigarettesPerDay, y = ComplexPred, color = "Complex Model (Quadratic)"), linetype = "solid", size = 1.2) +
  labs(
    title = "Model Predictions on Testing Data",
    x = "Cigarettes Smoked Per Day",
    y = "Lung Function (FVC)",
    color = "Model Type"
  ) +
  scale_color_manual(values = c(
    "Simple Model (Linear)" = "red",
    "Complex Model (Quadratic)" = "green"
  )) +
  coord_cartesian(xlim = x_range, ylim = y_range) +
  theme_minimal() +
  theme(legend.position = "right", legend.title = element_text(size = 12))

#Combine plots into a single figure with a 1x2 layout
combined_plot <- grid.arrange(training_plot, testing_plot, nrow = 2)

#Display the combined plot
print(combined_plot)

