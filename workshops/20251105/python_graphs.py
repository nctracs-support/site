#!/usr/bin/env python
# coding: utf-8

# # Clinical Data Visaulization in Python
# This notebook provides sample code for producing clincal-related visualizations. The sample dataset is an extract 
# from a synthetic database based upon COVID cases in 2020. 
# 
# 
# ## Setup
# This notebook use a small number of direct dependencies - most notably, Jupyter and [Seaborn](https://seaborn.pydata.org/).  The Seaborn library is built
# on top of matplotlib and provides  a simpler interface and more visually appeal default themes. The library will install pandas, numpy, matplotlib, and 
# other dependencies.
# 
# 
# ```bash
# python3 -m venv venv
# source venv/bin/activate
# pip install jupyter seaborn
# ```
# 
# For the Sankey diagram at the bottom of the notebook, we do use [plotly](https://plotly.com/python/) to make that visualization.
# ```bash
# pip install plotly
# ```

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


# In[2]:


#  Setup 
sns.set(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (10, 6)

# Initialize random number generator for reproducibility, Panda's utilizes NumPy module not "random"
np.random.seed(42)


# In[3]:


# Load data
df = pd.read_csv("sample_dataset.csv")


# ## Exploratory Data Analysis (EDA)
# With any dataset, you'll want to perform an initial exploratory data analysis to help you 
# understand the structure, patterns, and relationships. Initially, we'll perform some of
# these steps using Pandas, but we'll also see how visualizations can assist with this process.
# 
# With EDA, we'll have a few goals:
# 1. **Data Summarization** - gain an quick overview of the dataset
#    - **Shape and size of data:** Number of rows, columns, and unique values.
#    - **Descriptive statistics:** Mean, median, standard deviation, percentiles.
# 2. **Data Cleaning** - ensure data quality
#    - **Handling missing values:** Identify and impute (mean/median/mode) or remove missing entries.
#    - **Removing duplicates:** Eliminate redundant rows or records.
#    - **Correcting data types:** Convert data to appropriate formats (e.g., dates, numbers, categories).
#    - **Dealing with outliers:** Detect and decide whether to remove or transform extreme values.
# 3. **Visualization** - uncover patterns in the data
# 

# In[4]:


# Perform basic exploratory data analysis
# df.head(n)  # top n rows, n defaults to 5
# df.tail(n)  # last n rows
# df.sample(5) # sample x rows
df


# In[5]:


print("Dataframe shape:",df.shape)
print(df.info())


# Looking at the sample records from the dataframe (`df`), we can clearly see that a number of columns are actually dates, but listed as `objects` in the dataframe.
# 
# Let's convert those to actual dates - 

# In[6]:


for c in ["visit_start_date", "visit_end_date", "birth_datetime", "measurement_Date","flu_last_administered","tdap_last_administered","mmr_last_administered","polio_last_administered"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")


# One of the typical steps in EDA is to examine missing values. While `df.info()` does provide the number of records that have a value, let's
# reorganize the output to highlight the potentially problematic fields.

# In[7]:


n_rows = len(df)

missing_table = (   # create a new dataframe
    df.isna()
      .agg(['sum', 'mean'])
      .T
      .rename(columns={'sum': 'missing_count', 'mean': 'missing_percent'})
)

missing_table['missing_percent'] = (missing_table['missing_percent'] * 100).round(2)
missing_table['non_missing_count'] = n_rows - missing_table['missing_count']
missing_table['dtype'] = df.dtypes.astype(str)

missing_table = (
    missing_table
      .reset_index(names='column')
      .sort_values(by=['missing_percent', 'column'], ascending=[False, True])
      .set_index('column')
)

missing_table


# Typically, we'll also create additonal columns to help help visualize the data.
# 
# We're also converting several of the columns that have a limited number of values into a category data type. (less memory, better performance, clearer intent)

# In[8]:


# Create a column for visit length - ignoring visit type 
los = (df["visit_end_date"] - df["visit_start_date"]).dt.days
df["length_of_stay_days"] = los.clip(lower=0)


# Modify labels for deceased column
df["deceased_flag"] = df["deceased"].map({"Y": "Deceased", "N": "Alive"}).fillna("Unknown").astype("category")

# columns for year and month
df["visit_year"] = df["visit_start_date"].dt.year
df["visit_month"] = df["visit_start_date"].dt.to_period("M").astype(str)

df['gender_source_value'] = df['gender_source_value'].astype('category')
df['race_source_value'] = df['race_source_value'].astype('category')
df['ethnicity_source_value'] = df['ethnicity_source_value'].astype('category')


# In[9]:


df.sample(5)


# ## Produce Descriptive Statistics

# In[10]:


df.describe()             # for numeric columns


# In[11]:


df.describe(include=['object','category'])  # for categorical columns


# In[12]:


# Check for duplicates
df.duplicated().sum()


# In[13]:


print(df.info())


# In[14]:


# Produce quick categorical counts
print(df['gender_source_value'].value_counts(dropna=False), end="\n\n")
print(df['ethnicity_source_value'].value_counts(dropna=False), end="\n\n")
print(df['race_source_value'].value_counts(dropna=False), end="\n\n")
print(df['visit_type'].value_counts(dropna=False), end="\n\n")


# The condition field is actually a denormalized field in that it contains multiple values separated by colons.  We'll split this out into two different storage approaches:
# 1. "list" column within our original dataset
# 2. separate dataframe that's in  a "tidy" format.  (Each variable forms a column, each observation is a row, each value is a cell) 

# In[15]:


df['condition']


# In[16]:


import re

# robust split on ":" allowing extra spaces; keep NaN if empty
def split_conditions(s):
    if pd.isna(s) or str(s).strip() == "":
        return []
    # split on ":" with optional surrounding spaces
    parts = re.split(r"\s*:\s*", str(s))
    # normalize: strip, drop empties, lower (or title-case if you prefer)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts

# apply once to create a list-typed column
df["condition_list"] = df["condition"].map(split_conditions)


# In[17]:


cond_long = (
    df[["visit_occurrence_id", "person_id", "visit_start_date"]]
      .assign(condition_item=df["condition_list"])
      .explode("condition_item", ignore_index=True)
)

# drop rows where no condition exists after cleaning
cond_long = cond_long.dropna(subset=["condition_item"])

# (optional) dedupe within visit in case the same condition appears twice
cond_long = cond_long.drop_duplicates(subset=["visit_occurrence_id", "condition_item"])
cond_long


# In[18]:


top_conditions = (
    cond_long["condition_item"]
    .value_counts()
    .head(10)
)


# In[19]:


top_conditions


# ## Distributions
# This section explores how individual variables are spread across the dataset (e.g., ages, vitals, visit durations). Use histograms and boxplots to spot skew, outliers, and multi-modal patterns that may influence modeling choices and summary statistics.

# In[20]:


ax = sns.histplot(data=df, x="age_at_visit_years", bins=30, kde=True)
ax.set_title("Distribution of Age at Visit (years)")
ax.set_xlabel("Age (years)")
plt.show()


# In[21]:


# Slightly alternate syntax for setting the title and x-axis label
sns.histplot(data=df, x="age_at_visit_years", bins=30, kde=True).set(title="Distribution of Age at Visit (years)", xlabel = "Age(years)")
plt.show()


# In[22]:


# Code to save the image to a png file
def save_show(fig, filename):
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")


# In[23]:


fig = plt.figure()
sns.histplot(data=df, x="age_at_visit_years", bins=30, kde=True).set(title="Distribution of Age at Visit (years)")
save_show(fig,"dist_age_at_visit.png")
plt.show()


# In[24]:


# BMI Distribution as a box plot
sns.boxplot(data=df, y="bmi")
sns.stripplot(data=df, y="bmi", size=3, alpha=0.4, color="0.3")  # adds jittering to see values
plt.title("BMI Distribution")
plt.ylabel("BMI")
plt.show()


# In[25]:


# Distribution of the length of stays
sns.histplot(data=df, x="length_of_stay_days", bins=30, kde=True)
plt.title("Distribution of Length of Stay (Days)")
plt.xlabel("Days")
plt.ylabel("Count")


# In[26]:


# Let's see what data could be causing this.
df[df["length_of_stay_days"] > 100]


# In[27]:


# For outpatient visits, assume this is a data issue and the length should be 0
is_outpatient = df['visit_type'].astype(str).str.contains('outpatient', case=False, na=False)

# align dates, then recompute LOS as zero
df.loc[is_outpatient, 'visit_end_date'] = df.loc[is_outpatient, 'visit_start_date']
df.loc[is_outpatient, 'length_of_stay_days'] = 0


# In[28]:


sns.histplot(data=df, x="length_of_stay_days", bins=30, kde=True)
plt.title("Distribution of Length of Stay (Days)")
plt.xlabel("Days")
plt.ylabel("Count")


# In[29]:


df[df["length_of_stay_days"] > 100]


# In[30]:


# remove any records where length_of_stay_days > 100
df = df[df["length_of_stay_days"] <= 100].copy()


# In[31]:


sns.histplot(data=df, x="length_of_stay_days", bins=50, kde=True)
plt.title("Distribution of Length of Stay (Days)")
plt.xlabel("Days")
plt.ylabel("Count")


# Completely dominated by zero days - let's just look at inpatient/emergency room visits

# In[32]:


# filter to non-outpatient visits
non_outpatient = df[~df["visit_type"].astype(str).str.contains("outpatient", case=False, na=False)]

# basic distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(
    data=non_outpatient,
    x="length_of_stay_days",
    bins=45,
    kde=True,
    color="steelblue"
)
plt.title("Distribution of Length of Stay (Inpatient/Emergency Room Visits)")
plt.xlabel("Length of Stay (days)")
plt.ylabel("Number of Visits")
plt.tight_layout()
plt.show()


# In[33]:


# Now, let's look a blood pressure


# In[34]:


# Melt systolic/diastolic into long form
bp_long = df.melt(
    id_vars=["gender_source_value"],
    value_vars=["systolic", "diastolic"],
    var_name="Blood Pressure Type",
    value_name="Value"
)

sns.boxplot(
    data=bp_long,
    x="gender_source_value",
    y="Value",
    hue="Blood Pressure Type",
    palette="Set2"
)

plt.title("Blood Pressure Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Blood Pressure (mmHg)")
plt.legend(title="Type")


# In[35]:


plt.figure(figsize=(10, 6))
sns.barplot(
    data=bp_long,
    x="gender_source_value",
    y="Value",
    hue="Blood Pressure Type",
    errorbar=('ci', 95),
    palette="coolwarm"
)
plt.title("Mean Blood Pressure by Gender (with 95% CI)")
plt.xlabel("Gender")
plt.ylabel("Blood Pressure (mmHg)")
plt.tight_layout()
plt.show()


# In[36]:


g_raw = df["gender_source_value"].astype(str).str.strip().str.lower()
df["gender_clean"] = np.where(
    g_raw.str.startswith("m"), "Male",
    np.where(g_raw.str.startswith("f"), "Female", "Other")
)

# long (tidy) format
bp_long = df.melt(
    id_vars=["gender_clean"],
    value_vars=["systolic", "diastolic"],
    var_name="Blood Pressure Type",
    value_name="Value"
).dropna(subset=["Value"])

# keep only Male/Female as requested
bp_long = bp_long[bp_long["gender_clean"].isin(["Male", "Female"])]

# --- Plot: histograms by gender, split (facet) by BP type ---
palette = {"Male": "#1f77b4", "Female": "#ff69b4"}  # blue / pink

# choose sensible binning; you can tweak the range as needed
bins = np.arange(40, 221, 5)  # 5 mmHg bins from 40 to 220

g = sns.displot(
    data=bp_long,
    x="Value",
    hue="gender_clean",
    col="Blood Pressure Type",
    kind="hist",
    bins=bins,
    element="step",           # overlaid outlines for clarity
    common_bins=True,         # same bins for both facets
    multiple="layer",         # overlay, not stacked
    palette=palette,
    height=5,
    aspect=1.2
)

g.set_axis_labels("Blood Pressure (mmHg)", "Count")
g.set_titles("{col_name}")
plt.tight_layout()
plt.show()


# ## Comparisons
# This section contrasts groups side-by-side to surface differences in level and spread (e.g., inpatient vs. outpatient, deceased vs. non-deceased). Clustered bars and faceted plots make it easy to see rank order, gaps, and effect sizes, guiding where deeper analysis is warranted.

# In[37]:


# Comparison Bar Chart
top = (cond_long["condition_item"]
       .dropna()
       .value_counts()
       .head(10)
       .sort_values(ascending=True))  

print(top)

plt.figure(figsize=(12, 6))
ax = sns.barplot(x=top.values, y=top.index, orient="h")
ax.set_title("Top 10 Conditions")
ax.set_xlabel("Number of occurrences")
ax.set_ylabel("Condition")

# nice value labels at the end of each bar
for i, v in enumerate(top.values):
    ax.text(v, i, f"{v:,}", va="center", ha="left", fontsize=9)

plt.tight_layout()
plt.show()


# In[38]:


# Comparison of visit length by gender (and showing distribution)
inpatient = df[df["visit_type"].astype(str).str.contains("inpatient", case=False, na=False)]


# In[39]:


plt.figure(figsize=(8, 6))
palette = {"Male": "#1f77b4", "Female": "#ff69b4"}
sns.boxplot(
    data=inpatient,
    x="gender_clean",
    y="length_of_stay_days",
    hue="gender_clean",          # ← add hue
    dodge=False,                 # ← keep a single box per category
    palette=palette,
    legend=False                 # ← hide redundant legend
)
plt.title("Length of Stay for Inpatient Visits by Gender")
plt.xlabel("Gender")
plt.ylabel("Length of Stay (days)")
plt.tight_layout()
plt.show()


# In[40]:


plt.figure(figsize=(8, 6))
sns.barplot(
    data=inpatient,
    x="gender_clean",
    y="length_of_stay_days",
    hue="gender_clean",          # same trick
    dodge=False,
    errorbar=('ci', 95),
    palette=palette,
    legend=False
)
plt.title("Average Length of Stay (Inpatient) by Gender")
plt.xlabel("Gender")
plt.ylabel("Length of Stay (days)")
plt.tight_layout()
plt.show()


# In[41]:


sns.violinplot(
    data=inpatient,
    x="gender_clean",
    y="length_of_stay_days",
    palette=palette,
    hue="gender_clean",
    legend=False
)


# In[42]:


# Additional Comparisons: Condition and visit type
# Condition counts by visit type / outcome, etc.
# (assuming df has 'visit_type' and 'deceased')
cond_by_visit_type = (
    cond_long
    .merge(df[["visit_occurrence_id", "visit_type", "deceased"]], on="visit_occurrence_id", how="left")
    .groupby(["visit_type", "condition_item"])
    .size()
    .reset_index(name="count")
    .sort_values(["visit_type","count"], ascending=[True,False])
)
cond_by_visit_type.head(8)


# In[43]:


TOP_K = 10
ranked = (cond_by_visit_type
          .assign(rank=cond_by_visit_type.groupby("visit_type")["count"]
                  .rank(method="first", ascending=False))
          .query("rank <= @TOP_K"))

# percent share within each visit type
ranked = ranked.merge(
    cond_by_visit_type.groupby("visit_type", as_index=False)["count"].sum().rename(columns={"count": "visit_total"}),
    on="visit_type", how="left"
)
ranked["pct"] = (ranked["count"] / ranked["visit_total"] * 100).round(1)

# order bars by count within each facet
ranked["condition_item"] = ranked["condition_item"].astype(str)
ranked["condition_order"] = ranked.groupby("visit_type")["count"].rank(ascending=True, method="first")
ranked = ranked.sort_values(["visit_type", "condition_order"])

# plot
g = sns.catplot(
    data=ranked,
    x="count", y="condition_item",
    col="visit_type", col_wrap=2,  # wrap if many visit types
    kind="bar", orient="h", sharex=False, height=5, aspect=1.2
)
g.set_axis_labels("Count", "Condition")
g.set_titles("{col_name}")


plt.tight_layout()
plt.show()


# In[44]:


# Alternate view as clustered(grouped) horizontal bar
N = 15  # number of top conditions overall to show
H = max(6, N * 0.45)  # figure height that scales with N

# 1) Pick the overall top-N conditions (by total count across visit types)
overall = (cond_by_visit_type.groupby("condition_item", observed=True)["count"]
           .sum()
           .sort_values(ascending=False))
top_conditions = overall.head(N).index

# 2) Filter and order categories so the biggest end up at the bottom
plot_df = cond_by_visit_type[cond_by_visit_type["condition_item"].isin(top_conditions)].copy()

# y-order: ascending total so largest ends up at the top of the chart
y_order = (plot_df.groupby("condition_item", observed=True)["count"]
           .sum()
           .sort_values(ascending=True).index)

# (optional) consistent visit_type order by total volume
hue_order = (plot_df.groupby("visit_type", observed=True)["count"]
             .sum()
             .sort_values(ascending=False).index)

# 3) Plot clustered horizontal bars
plt.figure(figsize=(12, H))
ax = sns.barplot(
    data=plot_df,
    x="count", y="condition_item",
    hue="visit_type", hue_order=hue_order,
    order=y_order, orient="h"
)

ax.set_title(f"Top {N} Conditions by Visit Type (clustered)", pad=12)
ax.set_xlabel("Count")
ax.set_ylabel("Condition")

# Put legend outside and clean up
ax.legend(title="Visit Type", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
sns.despine(left=True, bottom=True)
ax.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()


# ## Relationships
# Here we examine how variables move together—both numerically and clinically relevant pairs (e.g., vitals vs. outcomes, age vs. length of stay).

# In[45]:


# drop missing values
df_rel = df.dropna(subset=["oxygen_saturation_percent", "respiratory_rate_per_minute"]).copy()

# base scatterplot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_rel,
    x="respiratory_rate_per_minute",
    y="oxygen_saturation_percent",
    alpha=0.2,
    color="steelblue"
)
plt.title("Oxygen Saturation vs. Respiratory Rate")
plt.xlabel("Respiratory Rate (breaths per minute)")
plt.ylabel("Oxygen Saturation (%)")
plt.tight_layout()
plt.show()


# In[46]:


# Bins (tweak as needed)
rr_bins = np.arange(10, 50, 1)      # 1-bpm bins
spo2_bins = np.arange(70, 101, 1)  # 1% bins

plt.figure(figsize=(9, 6))
ax = sns.histplot(
    data=df_rel,
    x="respiratory_rate_per_minute",
    y="oxygen_saturation_percent",
    bins=[rr_bins, spo2_bins],
    cbar=True,                   # colorbar on the right
    cbar_kws={"label": "Count"},
    stat="count"                 # use "density" if you want probabilities
)
plt.title("Oxygen Saturation vs Respiratory Rate — Heatmap")
plt.xlabel("Respiratory Rate (breaths/min)")
plt.ylabel("Oxygen Saturation (%)")
plt.tight_layout()
plt.show()


# In[47]:


sns.scatterplot(
    data=df.dropna(subset=["age_at_visit_years", "oxygen_saturation_percent"]),
    x="age_at_visit_years",
    y="oxygen_saturation_percent",
    hue="deceased_flag",
    alpha=0.7
)
plt.title("Age vs Oxygen Saturation by Outcome")
plt.xlabel("Age (years)")
plt.ylabel("Oxygen Saturation (%)")
plt.legend(title="Outcome", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()


# In[48]:


sns.scatterplot(
    data=non_outpatient.dropna(subset=["length_of_stay_days", "age_at_visit_years"]),
    x="age_at_visit_years",
    y="length_of_stay_days",
    hue="gender_source_value",
    alpha=0.7
)
plt.title("Age vs Length of Stay by Gender for In-patient/Emergency Visits")
plt.xlabel("Age (years)")
plt.ylabel("Length of Stay (days)")
plt.legend(title="Gender", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()


# ## Composition
# Composition views show how whole-to-part relationships break down across categories (e.g., visit types by race/ethnicity or gender, condition mix over time). Stacked bars, stacked area, and Sankey diagrams reveal share changes and flows that raw counts can obscure.

# In[49]:


def stacked_bar(df, group, hue, title, x_label=None, y_label="Count",
                normalize=False, palette="Set2"):
    """
    Build stacked bars from categorical counts (or row-wise % if normalize=True).
    """
    # NOTE: observed=False keeps current behavior (includes unused category levels).
    ct = (df.groupby([group, hue], observed=False).size()
            .reset_index(name="count")
            .pivot(index=group, columns=hue, values="count")
            .fillna(0))

    if normalize:
        ct = ct.div(ct.sum(axis=1), axis=0) * 100
        y_label = "Percent"

    group_order = ct.sum(axis=1).sort_values(ascending=False).index
    colors = sns.color_palette(palette, n_colors=ct.shape[1])

    ax = ct.loc[group_order].plot(kind="bar", stacked=True, figsize=(10, 6),
                                  color=colors, edgecolor="none")
    ax.set_title(title, pad=12)
    ax.set_xlabel(x_label or group)
    ax.set_ylabel(y_label)
    ax.legend(title=hue, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.show()


# In[50]:


# 1) Visit types by race (stacked counts)
stacked_bar(
    df,
    group="race_source_value",
    hue="visit_type",
    title="Visit Types by Race",
    x_label="Race",
    palette="Set2"
)

# 2) Gender composition by race (stacked counts)
stacked_bar(
    df,
    group="race_source_value",
    hue="gender_source_value",
    title="Gender Composition by Race",
    x_label="Race",
    palette="Pastel2"
)

# 3) Ethnicity composition by visit type (stacked counts)
stacked_bar(
    df,
    group="visit_type",
    hue="ethnicity_source_value",
    title="Ethnicity Composition by Visit Type",
    x_label="Visit Type",
    palette="Set3"
)

# 4) Deceased status by race (stacked counts)
stacked_bar(
    df,
    group="race_source_value",
    hue="deceased",
    title="Deceased Status by Race",
    x_label="Race",
    palette="Set1"
)

# Optional: percentage versions of any of the above
# stacked_bar(df, "race_source_value", "visit_type", "Visit Types by Race (% within race)", x_label="Race", normalize=True)
# stacked_bar(df, "race_source_value", "gender_source_value", "Gender Composition by Race (%)", x_label="Race", normalize=True)
# stacked_bar(df, "visit_type", "ethnicity_source_value", "Ethnicity Composition by Visit Type (%)", x_label="Visit Type", normalize=True)
# stacked_bar(df, "race_source_value", "deceased", "Deceased Status by Race (%)", x_label="Race", normalize=True)


# ## Time series
# This section examines how variables and conditions change over time. By visualizing trends, cycles, and abrupt shifts, time series plots help uncover seasonal patterns, responses to interventions, or external events that influence visit frequency, vital signs, or condition counts.

# In[51]:


visits_2020 = df[df["visit_start_date"].dt.year == 2020].copy()

# --- Count visits per day ---
daily_visits = (
    visits_2020.groupby(df["visit_start_date"].dt.date)
    .size()
    .reset_index(name="visit_count")
    .rename(columns={"visit_start_date": "date"})
)

# --- Plot time series ---
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=daily_visits,
    x="date",
    y="visit_count",
    marker="o",
    linewidth=1.5
)

plt.title("Daily Visit Counts (2020)")
plt.xlabel("Date")
plt.ylabel("Number of Visits")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[52]:


visits_2020 = df[df["visit_start_date"].dt.year == 2020].copy()

# --- COVID classification from observation_source (case-insensitive) ---
# Works even if observation_source is multivalued text
visits_2020["is_covid"] = (
    visits_2020["observation_source"]
      .astype(str)
      .str.contains("covid", case=False, na=False)
)

# --- Count visits per day by COVID status ---
daily = (
    visits_2020
      .groupby([visits_2020["visit_start_date"].dt.date, "is_covid"])
      .size()
      .reset_index(name="visit_count")
      .rename(columns={"visit_start_date": "date"})
)

# Friendly labels for plotting
daily["Category"] = daily["is_covid"].map({True: "COVID-related", False: "Non-COVID"})

# --- Plot daily counts ---
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=daily,
    x="date",
    y="visit_count",
    hue="Category",                 # hue assigned to avoid palette warning
    marker="o",
    linewidth=1.6,
    legend=True
)
plt.title("Daily Visit Counts in 2020 — COVID vs Non-COVID (by observation_source)")
plt.xlabel("Date")
plt.ylabel("Number of Visits")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Composition / Time Series

# In[53]:


# --- CONFIG ---
TOP_K = 10
START = "2020-02-01"
END   = "2020-04-01"  # inclusive
# If you store numeric weights (e.g., a "count" column), set this to that name; else None for simple row counts.
WEIGHT_COL = None  # e.g., "count"

# --- Pick the date column robustly ---
date_col = "visit_start_date" if "visit_start_date" in cond_long.columns else "date"
cond = cond_long.copy()
cond[date_col] = pd.to_datetime(cond[date_col], errors="coerce")

# --- Filter to the window ---
mask = (cond[date_col] >= pd.Timestamp(START)) & (cond[date_col] <= pd.Timestamp(END))
win = cond.loc[mask].dropna(subset=[date_col, "condition_item"])

# --- Determine the TOP_K conditions within the window ---
if WEIGHT_COL is None:
    top_conditions = (win["condition_item"].value_counts().head(TOP_K).index.tolist())
else:
    top_conditions = (win.groupby("condition_item")[WEIGHT_COL]
                        .sum()
                        .sort_values(ascending=False)
                        .head(TOP_K)
                        .index
                        .tolist())

win = win[win["condition_item"].isin(top_conditions)]

# --- Daily counts for the top conditions ---
if WEIGHT_COL is None:
    daily = (win
             .assign(day=win[date_col].dt.floor("D"))
             .groupby(["day", "condition_item"])
             .size()
             .rename("count")
             .reset_index())
else:
    daily = (win
             .assign(day=win[date_col].dt.floor("D"))
             .groupby(["day", "condition_item"])[WEIGHT_COL]
             .sum()
             .rename("count")
             .reset_index())

# Ensure every condition has every day in the range (fill 0s)
full_idx = pd.MultiIndex.from_product(
    [pd.date_range(START, END, freq="D"), top_conditions],
    names=["day", "condition_item"]
)
daily = (daily
         .set_index(["day", "condition_item"])
         .reindex(full_idx, fill_value=0)
         .reset_index())

# ---------- 1) Multi-line chart (Seaborn lineplot) ----------
plt.figure(figsize=(12, 7))
ax = sns.lineplot(data=daily, x="day", y="count", hue="condition_item")

ax.set(
    title=f"Top {TOP_K} Conditions by Day (Counts) • {START} to {END}",
    xlabel="Date",
    ylabel="Daily Count"
)

# Rotate and space out labels
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))   # one tick per week
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))     # e.g. "Feb 01"
plt.xticks(rotation=45, ha='right')

# Lighten gridlines for clarity
sns.despine()
ax.grid(True, linestyle="--", alpha=0.5)

# Move legend outside for readability
ax.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.show()

# ---------- 2) Stacked AREA (composition over time, % per day) ----------
wide = daily.pivot(index="day", columns="condition_item", values="count").sort_index()
share = wide.div(wide.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100

plt.figure(figsize=(12, 7))
share.plot(kind="area", stacked=True, figsize=(12, 7))
plt.title(f"Composition of Top {TOP_K} Conditions by Day (% of visits) • {START} to {END}")
plt.xlabel("Day")
plt.ylabel("Percent of visits")
plt.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

# ---------- 3) Heatmap (condition x day) ----------
heat = wide.T  # rows=condition, cols=day
plt.figure(figsize=(12, 6))
ax = sns.heatmap(heat, cmap="mako", cbar_kws={"label": "Daily count"})
cols = heat.columns  # should be a DatetimeIndex or array-like of dates
labels = pd.Index(cols).astype(str).str[:10]   # keep 'YYYY-MM-DD'

# choose ~10–12 ticks, skipping the rest
n = len(cols)
step = max(1, n // 12)                         # show ~12 ticks (tweak as needed)
tick_idx = np.arange(0, n, step)

ax.set_xticks(tick_idx)                        # positions correspond to column indices
ax.set_xticklabels(labels[tick_idx], rotation=45, ha='right')

# optional polish
ax.set_xlabel("Day")
ax.set_ylabel("Condition")
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# ## Additional Graphs
# ### Co-occurences

# In[54]:


# what conditions co-occur?
# Co-occurrence (conditions that tend to appear together per visit)

N = 10
topN = cond_long["condition_item"].value_counts().head(N).index

matrix = (
    cond_long[cond_long["condition_item"].isin(topN)]
    .assign(val=1)
    .pivot_table(index="visit_occurrence_id", columns="condition_item", values="val", fill_value=0)
)

# co-occurrence counts (symmetric)
cooc_cnt = matrix.T.dot(matrix)   # diagonal = occurrences of each condition

np.fill_diagonal(cooc_cnt.values, 0)  # hide self-counts on diagonal

mask = np.triu(np.ones_like(cooc_cnt, dtype=bool), k=1)

plt.figure(figsize=(15, 10))
ax = sns.heatmap(
    cooc_cnt,
    mask=mask,  # remove this line to make it symmetric
    cmap="mako_r",
    square=True,
    linewidths=0,
    cbar_kws={"label": "co-occurrence count"}
)
ax.set_title(f"Condition Co-occurrence (Top {len(cooc_cnt)} conditions)")
ax.set_xlabel("Condition")
ax.set_ylabel("Condition")
ax.grid(False)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# ## Sankey 
# A Sankey diagram is a type of flow chart that visualizes how quantities move between categories or stages. Each node represents a category, and the width of the connecting bands (or “flows”) corresponds to the magnitude of that relationship. In other words, thicker lines mean more observations flowing from one category to another. Sankey diagrams are particularly useful for showing proportional relationships and pathways through a process, allowing viewers to see both distribution and direction at once.
# 
# In this analysis, the Sankey diagram traces how conditions lead into visit types, and how those visits ultimately relate to deceased status.

# In[55]:


# Sankey (Condition → Visit Type → Deceased) with robust rendering & diagnostics

import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "notebook_connected"   # works in Jupyter classic/JLab

base = (cond_long[["visit_occurrence_id", "condition_item"]]
        .merge(df[["visit_occurrence_id", "visit_type", "deceased_flag"]],
               on="visit_occurrence_id", how="inner")
        .dropna(subset=["condition_item", "visit_type", "deceased_flag"]))

TOP_K = 12

top_conditions = base["condition_item"].value_counts().head(TOP_K).index
base = base[base["condition_item"].isin(top_conditions)]

# Build node list
conditions   = sorted(base["condition_item"].unique().tolist())
visit_types  = sorted(base["visit_type"].unique().tolist())
deceased_flg = sorted(base["deceased_flag"].unique().tolist())

nodes = conditions + visit_types + deceased_flg
node_idx = {name: i for i, name in enumerate(nodes)}

# Links: condition - visit_type
cv = (base.groupby(["condition_item", "visit_type"], observed=True)
           .size().reset_index(name="count"))
src1 = [node_idx[c] for c in cv["condition_item"]]
tgt1 = [node_idx[v] for v in cv["visit_type"]]
val1 = cv["count"].tolist()

# Links: visit_type - deceased
vd = (base.groupby(["visit_type", "deceased_flag"], observed=True)
           .size().reset_index(name="count"))
src2 = [node_idx[v] for v in vd["visit_type"]]
tgt2 = [node_idx[d] for d in vd["deceased_flag"]]
val2 = vd["count"].tolist()

fig = go.Figure(data=[go.Sankey(
    arrangement="snap",
    node=dict(
        pad=16, thickness=16, line=dict(width=0.5, color="gray"),
        label=nodes
    ),
    link=dict(
        source=src1 + src2,
        target=tgt1 + tgt2,
        value=val1 + val2
    )
)])

fig.update_layout(
    title=f"Sankey: Condition → Visit Type → Deceased (Top {TOP_K} conditions)",
    font=dict(size=12)
)
fig.show()



# In[ ]:




