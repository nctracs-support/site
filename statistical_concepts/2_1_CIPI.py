# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t, ttest_ind
from statsmodels.formula.api import ols
from statsmodels.stats.power import TTestIndPower

# Set seed for reproducibility
np.random.seed(1119)

# Number of samples
n_smokers = 36
n_non_smokers = 64

# Define the statistics for each group
mean_smokers = 4.7
mean_non_smokers = 5.0
sd_smokers = 0.6
sd_non_smokers = 0.6
alpha = 0.05  # Significance level

# Generate synthetic data using numpy
FVC_smokers = np.random.normal(mean_smokers, sd_smokers, n_smokers)

# FVC_smokers data. We overwrite the FVC_smokers and FVC_non_smokers data to make sure we are using the same dataset as in the R code file.
FVC_smokers = np.array([
    4.597676, 3.005049, 6.277790, 4.663538, 5.373852, 4.054780, 5.634638, 4.717132, 4.711907,
    3.774985, 5.126801, 4.948480, 5.699916, 4.432831, 4.504127, 4.828612, 5.443604, 4.344972,
    5.236672, 3.424502, 3.959826, 5.065351, 4.959656, 4.205429, 5.465809, 4.520172, 4.161801,
    4.252363, 4.676555, 5.377242, 4.533895, 4.954345, 5.433195, 4.881052, 5.801496, 4.339266
])


##########
#PI and CI
##########
# Define FVC_smokers as a pandas DataFrame
data_here = pd.DataFrame({'FVC_smokers': FVC_smokers})

# Fit a linear model (simple mean model)
model = ols('FVC_smokers ~ 1', data=data_here).fit()

# Calculate 95% Confidence Interval for Mean FVC
conf_int = model.conf_int(alpha=0.05)  # Confidence Interval

# Prediction interval
pred_int = model.get_prediction(data_here).summary_frame(alpha=0.05)

# Create a DataFrame with mean, confidence interval, and prediction interval
intervals_data = pd.DataFrame({
    'FVC': data_here['FVC_smokers'],
    'Mean': pred_int['mean'],
    'CI_Lower': pred_int['mean_ci_lower'],
    'CI_Upper': pred_int['mean_ci_upper'],
    'PI_Lower': pred_int['obs_ci_lower'],
    'PI_Upper': pred_int['obs_ci_upper']
})

# Generate a summary of the confidence and prediction intervals
mean_FVC = intervals_data['Mean'].mean()
CI_Lower = intervals_data['CI_Lower'].mean()
CI_Upper = intervals_data['CI_Upper'].mean()
PI_Lower = intervals_data['PI_Lower'].mean()
PI_Upper = intervals_data['PI_Upper'].mean()

print(f"Mean FVC: {mean_FVC:.3f}")
print(f"95% Confidence Interval: ({CI_Lower:.3f}, {CI_Upper:.3f})")
print(f"95% Prediction Interval: ({PI_Lower:.3f}, {PI_Upper:.3f})")

# Plot Confidence Interval vs Prediction Interval
plt.figure(figsize=(8, 6))

# Scatter individual data points
plt.scatter([1] * len(intervals_data['FVC']), intervals_data['FVC'], color='grey', alpha=0.6, label='Individual Data Points')

# Add Confidence Interval (CI) error bars with increased width
plt.errorbar(
    1, mean_FVC, 
    yerr=[[mean_FVC - CI_Lower], [CI_Upper - mean_FVC]], 
    fmt='o', color='blue', label='95% Confidence Interval', capsize=15, lw=3
)

# Add Prediction Interval (PI) error bars with increased width
plt.errorbar(
    1, mean_FVC, 
    yerr=[[mean_FVC - PI_Lower], [PI_Upper - mean_FVC]], 
    fmt='o', color='red', label='95% Prediction Interval', capsize=20, lw=3
)

# Mean FVC point
plt.scatter(1, mean_FVC, color='black', s=100, label='Mean FVC')

# Add annotations for Mean, CI, and PI
plt.text(1.15, mean_FVC, "Mean FVC", color='black', fontsize=10)
plt.text(1.15, CI_Lower - 0.1, "95% CI", color='blue', fontsize=10)
plt.text(1.15, PI_Lower - 0.1, "95% PI", color='red', fontsize=10)

# Set plot labels and title
plt.title("Confidence Interval vs Prediction Interval")
plt.ylabel("Forced Vital Capacity (FVC)")
plt.xticks([], [])  # Hide x-axis labels and ticks

# Apply minimal theme and display legend
plt.legend()
plt.show()

