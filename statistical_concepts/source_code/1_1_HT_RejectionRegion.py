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
FVC_non_smokers = np.random.normal(mean_non_smokers, sd_non_smokers, n_non_smokers)

# FVC_smokers data. We overwrite the FVC_smokers and FVC_non_smokers data to make sure we are using the same dataset as in the R code file.
FVC_smokers = np.array([
    4.597676, 3.005049, 6.277790, 4.663538, 5.373852, 4.054780, 5.634638, 4.717132, 4.711907,
    3.774985, 5.126801, 4.948480, 5.699916, 4.432831, 4.504127, 4.828612, 5.443604, 4.344972,
    5.236672, 3.424502, 3.959826, 5.065351, 4.959656, 4.205429, 5.465809, 4.520172, 4.161801,
    4.252363, 4.676555, 5.377242, 4.533895, 4.954345, 5.433195, 4.881052, 5.801496, 4.339266
])

# FVC_non_smokers data
FVC_non_smokers = np.array([
    4.214978, 4.024213, 4.615419, 4.737117, 5.962784, 5.635151, 5.277876, 5.050901, 5.446808,
    5.611526, 5.769301, 5.565237, 4.981123, 5.835770, 4.498458, 5.482763, 5.130003, 5.676979,
    3.922558, 5.195120, 4.907686, 6.134220, 5.442895, 5.219639, 4.975167, 4.916785, 5.042427,
    5.784897, 3.933958, 5.277223, 5.007990, 4.901559, 4.574370, 4.463008, 4.983143, 5.549591,
    5.952701, 4.853574, 4.912157, 3.957400, 4.558928, 5.880242, 5.561859, 5.004543, 4.027939,
    4.813787, 5.054393, 3.834674, 4.357586, 3.645311, 5.188090, 4.813484, 5.105440, 5.872425,
    4.577276, 3.870309, 4.417616, 5.300843, 5.079302, 4.787950, 4.918714, 4.871330, 5.361444,
    5.240772
])

# Create a DataFrame combining both groups
synthetic_data = pd.DataFrame({
    'Group': ['Smoker'] * n_smokers + ['Non-Smoker'] * n_non_smokers,
    'FVC': np.concatenate([FVC_smokers, FVC_non_smokers])
})

# Display the first few rows of the synthetic data
print(synthetic_data.head())

# Pooled standard deviation
pooled_sd = np.sqrt(((n_smokers - 1) * sd_smokers**2 + (n_non_smokers - 1) * sd_non_smokers**2) / (n_smokers + n_non_smokers - 2))

# Perform t-test manually
t_statistic = (mean_smokers - mean_non_smokers) / (pooled_sd * np.sqrt((1 / n_smokers) + (1 / n_non_smokers)))

# Degrees of freedom
df = n_smokers + n_non_smokers - 2

# Critical value
t_critical = t.ppf(alpha, df)

# P-value calculation
p_value = t.cdf(t_statistic, df)

print(f"Pooled standard deviation: {pooled_sd:.3f}")
print(f"t-statistic: {t_statistic:.3f}")
print(f"Degrees of freedom: {df}")
print(f"Critical value: {t_critical:.3f}")
print(f"p-value: {p_value:.4f}")
