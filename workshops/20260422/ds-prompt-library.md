# Data Science Prompt Library for Claude Code

> A reusable collection of prompt templates organized by data science lifecycle phase.
> Informed by Duke FinTech's "Data Science: From Insights to Impact" curriculum.
> Designed for use with Claude Code on real projects.

---

## How to Use This Library

Each prompt is a template. Replace `{{placeholders}}` with your project-specific details.
Many prompts include a **Context Block** — paste relevant details there so Claude Code
has the information it needs to act, not just advise.

Prompts are grouped by phase but the process is **iterative** — expect to revisit earlier
phases as you learn more.

---

## Phase 1: Problem Definition & Asking the Right Question

The analytical question shapes everything downstream — data choices, methods, metrics,
and how results are communicated. A vague question produces vague analysis.

### 1.1 — Decompose a Business Problem into Analytical Questions

```
I'm working on a data science project for {{domain/industry}}.

The business problem: {{describe the business problem in plain language}}

The stakeholders are: {{who will use the results — e.g., product managers, risk team, executives}}

Help me decompose this into specific analytical questions. For each question:
- Classify it: descriptive, exploratory, inferential, predictive, prescriptive, causal, or mechanistic
- State what data would be needed to answer it
- Identify what a successful answer looks like (metric, decision, or insight)
- Flag assumptions we're making

Start with the most impactful question and work outward.
```

### 1.2 — Sharpen a Vague Question

```
I have a general question I want to investigate:

"{{vague question — e.g., Why are customers leaving?}}"

Help me refine this into a precise, answerable analytical question by working through:
1. Who or what is the population of interest?
2. What outcome or behavior am I measuring?
3. Over what time period?
4. What comparison or baseline makes the answer meaningful?
5. What type of question is this (descriptive, predictive, causal, etc.)?
6. What would change in the business if we answered this well?

Produce 2-3 refined versions ranked by analytical tractability.
```

### 1.3 — Question Feasibility Check

```
Before I invest time in analysis, help me assess feasibility for this question:

Question: {{your refined analytical question}}
Available data: {{brief description of what data you have or expect to have}}
Timeline: {{how long you have}}
Constraints: {{regulatory, ethical, technical, or resource constraints}}

Evaluate:
- Can the available data plausibly answer this question?
- What's the minimum viable dataset?
- What are the biggest risks to getting a reliable answer?
- Is there a simpler proxy question that gets us 80% of the value?
```

---

## Phase 2: Data Collection

### 2.1 — Data Source Inventory and Acquisition Plan

```
I need to collect data to answer this question:

Question: {{your analytical question}}
Domain: {{industry/context}}

Help me create a data acquisition plan:
1. What data sources are likely needed (internal systems, APIs, public datasets, vendors)?
2. For each source, what fields/variables matter most?
3. What granularity do we need (row-level transactions vs. daily aggregates, etc.)?
4. What time range is appropriate?
5. What are the risks (access restrictions, cost, latency, quality concerns)?
6. Suggest a prioritized collection sequence — what do we get first?
```

### 2.2 — API Data Collection Script

```
I need to collect data from {{API name or type}}.

Endpoint(s): {{URL or description}}
Authentication: {{type — API key, OAuth, etc.}}
Rate limits: {{if known}}
Target fields: {{what I need from each response}}
Volume estimate: {{how many records/pages}}

Write a Python script that:
- Handles authentication and rate limiting
- Implements pagination
- Includes error handling and retry logic with exponential backoff
- Saves results incrementally (don't lose progress on failure)
- Logs progress and any anomalies
- Outputs to {{format — CSV, Parquet, JSON, SQLite}}
```

### 2.3 — Web Scraping with Structure

```
I need to collect data from {{website/source}}.

Target information: {{what you need}}
Pages/URLs: {{pattern or list}}
Expected volume: {{number of records}}

Write a scraping script that:
- Respects robots.txt and includes appropriate delays
- Handles pagination and dynamic content loading
- Extracts structured data into a clean DataFrame
- Handles missing fields gracefully
- Logs errors and skipped pages
- Saves checkpoints so we can resume
- Includes basic validation on collected data
```

---

## Phase 3: Clean, Prepare, and Store Data

### 3.1 — Initial Data Quality Assessment

```
I have a dataset loaded. Before cleaning, I need a thorough quality assessment.

The data is in: {{file path or variable name}}
It represents: {{what each row is — transactions, customers, daily prices, etc.}}
It should cover: {{expected time range, population, scope}}

Run a comprehensive data quality audit:
1. Shape, dtypes, memory usage
2. Missing values — counts, percentages, patterns (MCAR/MAR/MNAR hypotheses)
3. Duplicates — exact and near-duplicates
4. Per-column diagnostics:
   - Numeric: range, distribution shape, outlier detection (IQR and z-score), impossible values
   - Categorical: cardinality, unexpected categories, casing/whitespace inconsistencies
   - Datetime: format consistency, timezone issues, gaps, out-of-range dates
   - Text/ID fields: format validation, uniqueness
5. Cross-field consistency checks (e.g., end_date >= start_date, totals match components)
6. Provenance concerns — do values look like they came from different systems or eras?

Present findings as a structured report with severity ratings (critical / warning / info).
```

### 3.2 — Data Cleaning Pipeline

```
Based on the quality assessment, build a cleaning pipeline for {{dataset}}.

Known issues to address:
{{list the issues found in 3.1, or say "see the quality report above"}}

Requirements:
- Every transformation must be documented with rationale
- Preserve the original data — cleaning should produce a new DataFrame
- Handle missing values with explicit strategy per column (drop, impute, flag — justify each)
- Standardize categorical labels (casing, whitespace, known aliases)
- Parse dates into consistent timezone-aware format
- Flag but don't silently drop outliers — create an anomaly flag column
- Log a summary of what changed (rows dropped, values imputed, etc.)

Output the cleaned dataset and a cleaning log/report.
```

### 3.3 — Feature Engineering Foundation

```
I have a cleaned dataset for {{project purpose}}.

Target variable: {{if applicable}}
Key fields: {{list the important columns}}
Domain: {{industry context}}

Suggest and implement foundational feature engineering:
1. Datetime features (day of week, month, quarter, holiday flags, time since event)
2. Categorical encoding strategy (one-hot, ordinal, target encoding — justify each)
3. Numeric transformations if distributions are skewed (log, sqrt, Box-Cox)
4. Interaction features that make domain sense
5. Aggregation features (rolling windows, group-level statistics)
6. Lag features if temporal

For each feature, explain the analytical reasoning — why might this help answer our question?
Don't create features blindly; connect each to a hypothesis.
```

### 3.4 — Data Storage and Format

```
I need to store this prepared dataset for reproducible analysis.

Dataset size: {{approximate rows and columns}}
Update frequency: {{one-time vs. periodic refresh}}
Who will use it: {{just me, team, production pipeline}}
Tools downstream: {{pandas, SQL, Spark, etc.}}

Recommend and implement a storage approach:
- File format (CSV, Parquet, SQLite, etc.) with rationale
- Schema documentation (column names, types, descriptions, units)
- Metadata (creation date, source, cleaning version, row counts)
- Partitioning strategy if the data is large or time-series
- A simple data dictionary as a companion file
```

---

## Phase 4: Exploratory Data Analysis

### 4.1 — Clarifying the Analytical Goal Before Exploring

```
Before I start exploring, help me set up the EDA properly.

My analytical question: {{your question}}
Dataset: {{what it contains}}
What I already know or suspect: {{domain knowledge, prior findings, stakeholder hypotheses}}

Help me define:
1. What specifically am I looking for in this EDA? (Not "explore the data" — be precise)
2. What would confirm the data can support my question?
3. What would be a red flag that the data is unsuitable?
4. What assumptions am I carrying into this exploration?
5. What segments or subgroups should I examine separately?
6. Draft an EDA checklist tailored to this project — what must I examine before moving forward?
```

### 4.2 — Understanding and Validating Data (Provenance & Trust)

```
I need to validate this dataset before trusting it for analysis.

Dataset: {{description}}
Source(s): {{where it came from — system, API, vendor, manual entry}}
Expected properties: {{what you believe should be true}}

Walk through data validation:
1. Does the row count match expectations? Any unexplained gaps in coverage?
2. Do value ranges make domain sense? (e.g., ages between 0-120, prices > 0)
3. Are timestamps consistent? Check for timezone drift, mixed formats, or future dates.
4. Do categorical fields have stable, expected values across the full time range?
5. Are there signs of system artifacts? (e.g., midnight spikes, round-number clustering, default values)
6. Does the sampling appear representative, or is there selection bias?
7. Have definitions changed over time? (e.g., a "customer" field that changed meaning)
8. Cross-check aggregates against known external benchmarks if available.

For each check, show the code, the result, and your interpretation.
```

### 4.3 — Distributional Exploration

```
Perform distributional EDA on {{dataset}}.

Focus variables: {{list key variables, or say "all numeric and key categoricals"}}

For each variable:
1. Visualize the distribution (histogram, KDE, box plot as appropriate)
2. Compute summary statistics (mean, median, std, skewness, kurtosis, percentiles)
3. Identify outliers and anomalies — are they real or data errors?
4. Check for multimodality — could this variable represent mixed populations?
5. Assess missingness patterns — is missingness random or informative?
6. Note anything that violates common statistical assumptions (normality, homoscedasticity)

For categorical variables:
7. Frequency distributions and class imbalance
8. Rare categories that may need grouping
9. Inconsistent labels or encoding issues

Interpret each finding: what does it mean for our analysis? What questions does it raise?
```

### 4.4 — Relational Exploration

```
Explore relationships between variables in {{dataset}}.

Key relationships to investigate:
{{list specific pairs or groups, or say "all pairwise among these variables: ..."}}

Target variable (if applicable): {{target}}

Perform:
1. Correlation analysis (Pearson, Spearman, and/or Kendall — choose based on distributions)
2. Scatterplots with trend lines for key numeric pairs
3. Cross-tabulations for categorical-categorical relationships
4. Grouped statistics for categorical-numeric relationships (box plots, violin plots)
5. Check for nonlinear relationships that correlation misses
6. Look for confounders — does a relationship hold up when controlling for a third variable?
7. Identify potential multicollinearity among predictors

Critical: For each observed relationship, note that correlation is not causation.
Propose at least one alternative explanation for any strong association found.
```

### 4.5 — Comparative Exploration

```
Perform comparative analysis across segments in {{dataset}}.

Segments to compare: {{e.g., customer tiers, time periods, regions, channels, before/after an event}}
Metrics to compare: {{what outcomes or behaviors to examine across segments}}

For each comparison:
1. Visualize distributions side by side (overlapping histograms, grouped box plots)
2. Compute group-level summary statistics
3. Assess whether differences look meaningful or could be noise
4. Check if sample sizes across groups are sufficient for reliable comparison
5. Look for Simpson's Paradox — does the story change when you further segment?
6. Examine whether the comparison groups are actually comparable (confounders, selection effects)

Flag which comparisons are most promising for further investigation and which are likely artifacts.
```

### 4.6 — Structural & Temporal Exploration

```
Explore the structural and temporal patterns in {{dataset}}.

Time field: {{column name}}
Key metrics to examine over time: {{list them}}
Expected patterns: {{seasonality, growth, regime changes, etc.}}

Investigate:
1. Time-series plots of key metrics (raw and smoothed/rolling)
2. Seasonality analysis (day-of-week, monthly, quarterly patterns)
3. Trend decomposition (trend, seasonal, residual components)
4. Structural breaks — are there points where behavior clearly shifts?
5. Rolling statistics (mean, variance) to detect drift or volatility clustering
6. Gaps or irregular spacing in the time index
7. Autocorrelation and partial autocorrelation
8. Sequence patterns — do events follow predictable sequences?

For each finding, consider: Is this a real behavioral pattern or a data/system artifact?
(e.g., a spike at midnight might be batch processing, not real activity)
```

---

## Phase 5: Interpreting Findings and Refining Hypotheses

### 5.1 — Synthesize EDA Findings

```
I've completed exploratory analysis on {{dataset}} for the question: {{question}}.

Here are my key observations:
{{paste or summarize your top 5-10 findings from EDA}}

Help me synthesize:
1. Which findings are robust (appear across segments, time periods, multiple views)?
2. Which are fragile (disappear when you cut the data differently)?
3. For each strong pattern, generate at least two competing explanations:
   - A behavioral/real-world explanation
   - A data artifact or confounding explanation
4. What assumptions from Phase 1 have been confirmed, challenged, or invalidated?
5. What new questions has the EDA surfaced that we didn't anticipate?
6. What is the EDA telling us that we can trust vs. what requires further investigation?

Frame findings as hypotheses, not conclusions. EDA generates questions; it does not answer them.
```

### 5.2 — Document Hypotheses and Alternative Explanations

```
Based on the EDA, I've identified these patterns:

{{list 3-5 key patterns with brief descriptions}}

For each pattern, help me build a hypothesis document:
1. State the pattern precisely (what, where, when, how strong)
2. Hypothesis A: the most intuitive explanation
3. Hypothesis B: an alternative explanation (operational, data-quality, or confounding)
4. Hypothesis C: a null/noise explanation
5. What additional data or analysis would distinguish between these?
6. What's the cost of acting on Hypothesis A if it's actually B or C?

This discipline prevents us from treating exploratory patterns as confirmed findings.
```

---

## Phase 6: Deciding Next Steps

### 6.1 — EDA-to-Action Decision Framework

```
My EDA is complete. The core question was: {{question}}
Key findings: {{brief summary}}

Help me decide the best next step. The options are:
1. **Communicate now** — the EDA itself answers the question; build the narrative
2. **Build a model** — patterns suggest a predictive or classification task is warranted
3. **Collect more data** — critical gaps prevent confident analysis
4. **Redesign the process** — findings suggest the problem is upstream (data generation, business process)
5. **Refine and re-explore** — the question has evolved; go back to EDA with sharper focus

For the recommended path:
- Why this path over the others?
- What are the risks of this path?
- What would a minimum viable version look like?
- What's the timeline estimate?
```

---

## Phase 7: Building Models

### 7.1 — Model Selection and Baseline

```
I'm ready to model. Here's the setup:

Question: {{what the model should predict or classify}}
Target variable: {{name, type, distribution}}
Feature set: {{number of features, types, any known important ones}}
Dataset size: {{rows for training}}
Constraints: {{interpretability requirements, latency, fairness, regulatory}}
Evaluation priority: {{which metric matters most and why}}

Help me:
1. Recommend 2-3 candidate model families with rationale for each
2. Build a simple baseline first (e.g., majority class, mean prediction, logistic regression)
3. Set up a proper train/validation/test split (or cross-validation scheme)
   - If time-series: temporal split, no future leakage
   - If grouped: group-aware splitting
4. Establish the evaluation framework before fitting anything complex
5. Document what "good enough" looks like for this problem
```

### 7.2 — Model Training and Iteration

```
I have a baseline model established.

Baseline performance: {{metric = value}}
Candidate model: {{model type you want to try next}}
Feature set: {{description or count}}

Build and train this model:
1. Implement with sensible default hyperparameters first
2. Check for data leakage (features that encode the target, temporal leakage)
3. Examine learning curves — is this a bias or variance problem?
4. Perform hyperparameter tuning (specify strategy: grid, random, Bayesian)
5. Compare against baseline on the same validation set
6. Check performance across segments/subgroups — does it work equally well everywhere?
7. Log all experiments: parameters, metrics, timestamps
```

### 7.3 — Feature Importance and Selection

```
I have a trained model on {{dataset}}.

Model type: {{e.g., random forest, gradient boosting, logistic regression}}
Number of features: {{count}}

Analyze feature importance:
1. Model-native importance (coefficients, Gini, gain — whichever applies)
2. Permutation importance on the validation set
3. SHAP values for global and local interpretability
4. Check for redundant features (high correlation among top features)
5. Test a reduced feature set — how much performance do we lose?
6. Do the important features make domain sense? Flag any surprises.
7. Are there features that shouldn't be there (leakage, protected attributes)?
```

---

## Phase 8: Evaluation and Interpretation

### 8.1 — Comprehensive Model Evaluation

```
Evaluate the model thoroughly before any deployment discussion.

Model: {{type and version}}
Task: {{classification, regression, ranking, etc.}}
Test set: {{size, how it was held out}}

Run a full evaluation:
1. Primary metric on test set: {{metric}}
2. Secondary metrics (precision/recall/F1, MAE/RMSE, AUC, etc.)
3. Confusion matrix or residual analysis as appropriate
4. Calibration assessment — are predicted probabilities reliable?
5. Performance by subgroup (demographic, temporal, segment) — fairness check
6. Error analysis — what does the model get wrong, and is there a pattern?
7. Comparison table: baseline vs. candidate vs. best model
8. Stress test: how does the model perform on edge cases or distribution shifts?
9. Statistical significance: is the improvement over baseline real or noise?
```

### 8.2 — Interpretability and Explanation

```
The model needs to be explainable to {{audience — e.g., regulators, executives, end users}}.

Model: {{type}}
Top features: {{list them}}
A sample prediction to explain: {{describe a specific case}}

Produce:
1. Global explanations — what drives the model's behavior overall?
2. Local explanations — for the sample case, why did the model predict what it did?
3. Counterfactual analysis — what would need to change for a different prediction?
4. Plain-language summary suitable for the target audience
5. Limitations and caveats that must accompany any explanation
6. What the model cannot tell us (causal claims it can't make, populations it doesn't cover)
```

---

## Phase 9: Deployment and Monitoring

### 9.1 — Deployment Readiness Checklist

```
We want to deploy {{model description}} to {{production environment}}.

Help me build a deployment plan:
1. Code quality: Is the training/inference code modular, tested, and documented?
2. Dependencies: Pin all library versions; document the environment
3. Input validation: What checks should run on new data before prediction?
4. Output validation: What sanity checks should run on predictions?
5. Fallback strategy: What happens if the model fails or produces anomalous output?
6. Latency/throughput requirements and whether the model meets them
7. Data pipeline: How does new data flow in? What could break?
8. Rollback plan: How do we revert to the previous model version?
9. A/B testing or shadow mode plan before full rollout
```

### 9.2 — Monitoring Dashboard Design

```
Design a monitoring plan for {{deployed model}}.

Model type: {{type}}
Prediction frequency: {{real-time, daily batch, etc.}}
Key metric: {{what defines success in production}}

Monitor for:
1. Input drift — are feature distributions shifting? (PSI, KS test, summary stats)
2. Prediction drift — is the distribution of outputs changing?
3. Performance drift — once ground truth arrives, is accuracy degrading?
4. Latency and error rates
5. Data quality — missing values, schema changes, volume anomalies
6. Fairness metrics over time — are subgroup disparities emerging?
7. Alert thresholds — what triggers manual review vs. automatic rollback?

Produce a monitoring specification I can implement, with suggested tools and check frequencies.
```

---

## Phase 10: Communicate Insights and Tell the Story

### 10.1 — Findings Narrative for Stakeholders

```
I need to present findings to {{audience — e.g., executives, product team, risk committee, regulators}}.

Project: {{brief description}}
Key findings: {{list 3-5 findings}}
Recommendation: {{what action you're proposing}}

Help me structure a compelling narrative:
1. Start with the business question and why it matters (1-2 sentences)
2. What we did (methods, in plain language — no jargon)
3. What we found (lead with the most important finding)
4. What it means for the business (translate to dollars, risk, customer impact)
5. What we recommend and why
6. What we're uncertain about — limitations and caveats, stated honestly
7. What comes next

Tailor the language, detail level, and emphasis to the audience.
Do NOT overstate what the analysis can support. Frame EDA findings as patterns, not proof.
```

### 10.2 — Visualization for Communication

```
I need to create visualizations to communicate {{specific finding or story}} to {{audience}}.

The key message: {{one sentence — what should the viewer take away?}}
Data: {{what's being plotted}}
Context: {{where will this be shown — slide deck, report, dashboard, email}}

Create visualizations that:
1. Lead with the message, not the methodology
2. Use appropriate chart types (don't default to bar charts for everything)
3. Annotate key points directly on the chart
4. Use color purposefully (highlight, don't decorate)
5. Include proper titles, axis labels, and source notes
6. Handle uncertainty honestly (confidence intervals, ranges, caveats)
7. Are accessible (colorblind-safe palette, sufficient contrast)

Produce the visualization code and a one-paragraph caption I can use alongside it.
```

### 10.3 — Technical Report / Reproducible Analysis

```
I need to create a technical report documenting this analysis for {{audience — e.g., peer data scientists, future self, audit team}}.

Project: {{description}}
Code location: {{repo or directory}}

Structure the report as:
1. **Objective** — the analytical question and its business context
2. **Data** — sources, time range, grain, known limitations
3. **Methodology** — approach, tools, key decisions and their rationale
4. **Findings** — organized by sub-question, with supporting evidence
5. **Model details** (if applicable) — architecture, training, evaluation results
6. **Limitations and risks** — what could go wrong, what we don't know
7. **Recommendations** — prioritized, with confidence levels
8. **Reproducibility** — how to re-run the analysis, dependencies, data access

Include code snippets for critical steps. The goal is that someone could pick this up
in six months and understand what was done and why.
```

---

## Cross-Cutting Prompts (Use Anytime)

### CC.1 — Sanity Check My Analysis

```
I just completed {{describe the analysis step}}.

Here's what I found: {{key result or output}}
Here's what I expected: {{what you thought would happen}}

Help me sanity check:
1. Does this result make domain sense?
2. What are three things that could make this result misleading?
3. Is there a simple cross-check I can run to verify?
4. Am I seeing signal or noise?
5. What would a skeptical reviewer challenge about this?
```

### CC.2 — Explain This To Me Like I'm New

```
I'm looking at {{code output, statistical result, visualization, or concept}} and I'm not sure
I understand what it's telling me.

{{paste the output or describe what you're looking at}}

Explain:
1. What this means in plain language
2. Why it matters for my analysis
3. What I should do about it (if anything)
4. Common misinterpretations to avoid
```

### CC.3 — Code Review for Data Science

```
Review this code for correctness and best practices:

{{paste your code or reference the file}}

Check for:
1. Data leakage (target information leaking into features)
2. Off-by-one errors in time-based operations
3. Incorrect handling of missing values
4. Statistical mistakes (wrong test, violated assumptions)
5. Reproducibility issues (random seeds, non-deterministic operations)
6. Performance concerns for the data size
7. Readability and documentation
```

### CC.4 — What Am I Missing?

```
I'm at the {{phase name}} stage of my project.

What I've done so far: {{brief summary}}
What I'm about to do next: {{planned next step}}

Before I move forward:
1. What common mistakes happen at this stage?
2. What do experienced data scientists check that beginners skip?
3. Is there a simpler approach I should consider?
4. Am I over-engineering or under-investigating anything?
5. What would I wish I had done now when I'm three steps further along?
```

---

## Quick Reference: Question Type Taxonomy

| Type | Purpose | Example |
|------|---------|---------|
| **Descriptive** | Summarize what happened | What was our churn rate last quarter? |
| **Exploratory** | Discover patterns or relationships | Are there customer segments with unusually high churn? |
| **Inferential** | Generalize from sample to population | Is the observed churn difference between segments statistically significant? |
| **Predictive** | Forecast future outcomes | Which current customers are most likely to churn next quarter? |
| **Prescriptive** | Recommend actions | What intervention would most reduce churn among at-risk customers? |
| **Causal** | Establish cause and effect | Did the new pricing policy cause the increase in churn? |
| **Mechanistic** | Explain the underlying mechanism | What behavioral sequence leads a customer from engagement to churn? |

---

## Quick Reference: Four Dimensions of EDA

| Dimension | Focus | Key Questions |
|-----------|-------|---------------|
| **Distributional** | Individual variables | What's typical? What's unusual? Is it skewed? Are there outliers? |
| **Relational** | Connections between variables | What correlates? Are relationships linear? Do confounders exist? |
| **Comparative** | Differences across groups | Do segments differ? Is a before/after change real? Simpson's Paradox? |
| **Structural** | Temporal and sequential patterns | Are there trends, seasonality, regime changes? Artifacts vs. real patterns? |

---

## Quick Reference: EDA Principles Checklist

- [ ] Start with an open mind — let data challenge your assumptions
- [ ] Segment early — averages hide heterogeneity
- [ ] Surface assumptions — make implicit beliefs explicit and test them
- [ ] Expect drift — patterns valid last year may not hold today
- [ ] Distinguish artifacts from behavior — system quirks vs. real patterns
- [ ] Document hypotheses — write down multiple explanations for each pattern
- [ ] Correlation ≠ causation — always propose alternative explanations
- [ ] EDA generates hypotheses, it does not confirm them
- [ ] Visualize thoughtfully — bad charts create bad conclusions
- [ ] Know when to stop — EDA should inform decisions, not delay them

---

*Version 1.0 — April 2026*
*Source framework: Duke FinTech "Data Science: From Insights to Impact"*
