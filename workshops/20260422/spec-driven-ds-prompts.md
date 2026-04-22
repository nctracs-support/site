# Spec-Driven Data Science: A Prompt & Artifact Library for Claude Code

> Adapts specification-driven development (SDD) principles to the data science lifecycle.
> Each phase produces a **specification artifact** before execution begins.
> The spec is the source of truth — Claude Code validates its work against it.

---

## The Core Idea

Traditional data science with AI assistants is "vibe analysis" — you prompt, get code,
run it, prompt again. It works for small tasks but falls apart on real projects because
context drifts, decisions go undocumented, and nobody can reproduce or explain what happened.

Spec-driven data science fixes this:

```
 ┌──────────────────────────────────────────────────┐
 │  1. SPECIFY  →  Write what you intend to do      │
 │  2. PLAN     →  Break it into verifiable tasks    │
 │  3. EXECUTE  →  Claude Code implements            │
 │  4. VALIDATE →  Check output against spec         │
 │  5. UPDATE   →  Revise spec with what you learned │
 └──────────────────────────────────────────────────┘
        ↑                                    │
        └────────────────────────────────────┘
                    (iterate)
```

**Specs are not documentation written after the fact.**
They are blueprints written before execution that the agent reasons against.

---

## Project Structure

A spec-driven data science project has this file layout:

```
project/
├── AGENTS.md                  # Steering file — project-level agent instructions
├── specs/
│   ├── 01-problem.md          # Problem & question specification
│   ├── 02-data.md             # Data acquisition & quality specification
│   ├── 03-eda.md              # Exploratory analysis specification
│   ├── 04-modeling.md         # Modeling specification (if applicable)
│   ├── 05-deployment.md       # Deployment specification (if applicable)
│   └── 06-communication.md   # Findings & storytelling specification
├── progress.md                # Living log — decisions, findings, status
├── tasks.md                   # Current task breakdown with status
├── data/
│   ├── raw/                   # Untouched source data
│   ├── interim/               # Intermediate transformations
│   └── processed/             # Analysis-ready datasets
├── notebooks/                 # Exploratory notebooks (numbered)
├── src/                       # Production code (pipelines, models, utils)
├── tests/                     # Validation and regression tests
├── outputs/                   # Reports, visualizations, deliverables
└── docs/
    └── data-dictionary.md     # Schema, definitions, lineage
```

---

## AGENTS.md — Project Steering File

This is the first file you create. Claude Code reads it at session start.

### Prompt: Generate the Steering File

```
I'm starting a data science project. Help me create a AGENTS.md steering file.

Project name: {{name}}
Domain: {{industry/context}}
Goal: {{one-sentence objective}}
Tech stack: {{Python version, key libraries, database, etc.}}
Constraints: {{regulatory, ethical, timeline, team size}}

The AGENTS.md should include:

## Project Overview
- One-paragraph description of the project and its business context

## Architecture
- Project directory structure (use the spec-driven layout above)
- Data flow: source → raw → cleaned → features → model → output

## Coding Standards
- Language and version (e.g., Python 3.11+)
- Required libraries and pinned versions
- Style: PEP 8, type hints, docstrings (Google style)
- Notebook conventions: numbered, markdown headers, no dead code
- Logging: use Python logging, not print()

## Data Conventions
- Date format: ISO 8601, timezone-aware (UTC unless specified)
- Missing values: use pd.NA, never fill silently
- Column naming: snake_case, descriptive, include units where relevant
- Every transformation must be documented in progress.md

## Spec Workflow
- Before executing any phase, read the relevant spec in specs/
- After completing a phase, update progress.md with findings and decisions
- Do not proceed to the next phase until the current spec's acceptance criteria are met
- When EDA reveals something that changes the problem spec, update specs/01-problem.md

## Testing
- Data validation tests in tests/test_data.py
- Pipeline reproducibility: random seeds, pinned dependencies
- Model evaluation: never evaluate on training data

## Common Commands
- `python -m pytest tests/ -x -q` — run all tests
- `jupyter nbconvert --execute notebooks/*.ipynb` — execute notebooks
- `python src/pipeline.py` — run the full pipeline
```

---

## Phase 1: Problem Specification

### Prompt: Write the Problem Spec

```
Help me write specs/01-problem.md for this data science project.

Business context: {{describe the business situation}}
Stakeholders: {{who needs the results and what decisions they'll make}}
What we think the question is: {{initial question, even if vague}}

The spec should have these sections:

## Business Context
- What is the business problem? Who cares and why?
- What decision will this analysis inform?

## Analytical Questions
For each question:
- Statement: precise, unambiguous
- Type: descriptive / exploratory / inferential / predictive / prescriptive / causal / mechanistic
- Required data: what we need to answer it
- Success criteria: what does a good answer look like?
- Failure criteria: what would tell us the question is unanswerable?

## Scope
- In scope: explicitly list what we will address
- Out of scope: explicitly list what we will NOT address
- Time range: what period does the analysis cover?
- Population: who/what are the units of analysis?

## Assumptions
- List every assumption we're making (we'll validate these in EDA)

## Constraints
- Regulatory, ethical, timeline, data access, team

## Acceptance Criteria
- [ ] At least one question is classified and precise
- [ ] Success criteria are defined for each question
- [ ] Stakeholders have reviewed and agreed on scope
- [ ] Assumptions are documented and testable

## Phase Gate
This spec must be reviewed before proceeding to Phase 2 (Data).
```

---

## Phase 2: Data Specification

### Prompt: Write the Data Spec

```
Based on specs/01-problem.md, help me write specs/02-data.md.

Available data sources: {{list what you have access to}}
Known gaps: {{what you suspect is missing}}

The spec should include:

## Required Data
For each dataset needed:
- Source: where it comes from
- Granularity: what each row represents
- Key fields: columns we need and their expected types/ranges
- Time coverage: what period
- Access method: API, database, file, scrape
- Refresh frequency: one-time or recurring
- Known quality issues: anything we already suspect

## Data Collection Plan
- Prioritized order of acquisition
- Dependencies between datasets (what needs what)
- Estimated effort per source

## Schema Specification
- Target schema for the analysis-ready dataset
- Column name, type, description, units, valid range, nullable
- Primary keys and join keys across datasets

## Quality Requirements
- Maximum acceptable missing rate per critical field
- Deduplication rules
- Timestamp consistency requirements
- Categorical standardization rules

## Storage Plan
- Raw data: format and location
- Processed data: format (Parquet recommended), partitioning strategy
- Data dictionary location: docs/data-dictionary.md

## Acceptance Criteria
- [ ] All required datasets are identified with access confirmed
- [ ] Schema is defined with types, ranges, and descriptions
- [ ] Quality thresholds are set for critical fields
- [ ] Data dictionary is created
- [ ] At least one dataset is loaded and passes basic shape/type validation

## Phase Gate
Data spec must be validated with an initial data quality audit before proceeding to EDA.
```

### Prompt: Execute Data Collection Against Spec

```
Read specs/02-data.md and implement the data collection plan.

For each dataset in the spec:
1. Acquire the data using the specified access method
2. Save raw data to data/raw/ (never modify raw files after saving)
3. Validate against the schema specification:
   - Do columns match expected names and types?
   - Do value ranges fall within spec?
   - Is the row count within expected bounds?
4. Log results to progress.md:
   - Dataset name, source, rows, columns, acquisition timestamp
   - Any deviations from the spec (unexpected columns, different date range, etc.)
5. Run data quality checks defined in the spec:
   - Missing value rates vs. thresholds
   - Duplicate detection
   - Timestamp format consistency
6. Create/update tests/test_data.py with assertions for each quality requirement

If any acceptance criteria in the spec fail, stop and report what's wrong.
Do NOT proceed to cleaning until the raw data is validated.
```

### Prompt: Execute Data Cleaning Against Spec

```
Read specs/02-data.md (quality requirements and schema) and clean the raw data.

For each transformation:
1. Document what you're changing and why in progress.md
2. Never modify data/raw/ — write cleaned data to data/interim/ or data/processed/
3. Follow these rules from the spec:
   - Missing value strategy per column (drop / impute / flag — as specified)
   - Categorical standardization (as specified in the schema)
   - Timestamp normalization (as specified)
   - Outlier handling (flag, don't silently remove)
4. Produce a cleaning report:
   - Rows before/after
   - Columns added/removed/modified
   - Missing values before/after per column
   - Anomalies flagged
5. Update docs/data-dictionary.md with any new or changed columns
6. Run tests/test_data.py to verify the cleaned data meets spec

Check the acceptance criteria in specs/02-data.md. Report which are met and which are not.
```

---

## Phase 3: EDA Specification

### Prompt: Write the EDA Spec

```
Based on specs/01-problem.md (questions) and specs/02-data.md (data), write specs/03-eda.md.

The EDA spec should include:

## Analytical Goal
- Restate the question from the problem spec
- What must EDA establish before modeling is justified?
- What would cause us to revise the question?

## Data Validation Checks (Stage 1-2: Trust & Provenance)
For each critical field:
- [ ] Values fall within domain-valid ranges
- [ ] No unexpected format changes across time
- [ ] Distributions match domain expectations
- [ ] Cross-field consistency holds (e.g., totals = sum of parts)
- [ ] No signs of system artifacts (midnight spikes, round-number clustering, default values)
- [ ] Sampling appears representative

## Exploration Plan (Stage 3: Four Dimensions)

### Distributional
For each key variable:
- What distribution do we expect and why?
- What would be surprising?
- Acceptance: all key variables have documented distributions with outlier assessment

### Relational
For each hypothesis about variable relationships:
- Variables: X and Y
- Expected relationship: positive/negative/none
- Confounders to check
- Acceptance: correlation matrix computed, key relationships visualized with alternatives noted

### Comparative
For each segment comparison:
- Groups: A vs B (e.g., before/after, segment X vs Y)
- Metric: what to compare
- Acceptance: group distributions visualized, Simpson's Paradox checked

### Structural / Temporal
For each time-dependent pattern:
- Variable and time grain
- Expected patterns (trend, seasonality, regime changes)
- Acceptance: time-series decomposition completed, structural breaks identified

## Hypotheses to Test
- List initial hypotheses from domain knowledge
- Each must have: statement, what would confirm it, what would refute it

## Assumptions to Validate
- Copied from specs/01-problem.md, each with a specific test

## Acceptance Criteria
- [ ] All validation checks passed (or failures documented with impact assessment)
- [ ] All four EDA dimensions explored for key variables
- [ ] At least two competing explanations documented for each strong pattern
- [ ] Assumptions from problem spec validated or flagged
- [ ] EDA findings do NOT overstate confidence (hypotheses, not conclusions)
- [ ] progress.md updated with key findings and revised hypotheses

## Phase Gate
EDA spec must be reviewed. Decide: communicate findings / build model / collect more data / refine question.
```

### Prompt: Execute EDA Against Spec

```
Read specs/03-eda.md and execute the exploration plan in a jupyter notebook.

Work through the spec section by section:

1. **Data Validation (Stage 1-2)**
   Run every validation check listed. For each:
   - Show the code and result
   - Mark pass/fail against the spec
   - If fail: assess severity and whether it blocks downstream analysis

2. **Distributional Exploration**
   For each variable listed in the spec:
   - Create the visualization specified
   - Compare actual distribution to expected (from spec)
   - Document surprises in progress.md

3. **Relational Exploration**
   For each relationship in the spec:
   - Compute the specified correlation/visualization
   - Check for confounders as specified
   - Note: "this is a hypothesis, not a confirmed finding"

4. **Comparative Exploration**
   For each comparison in the spec:
   - Visualize group differences
   - Check for Simpson's Paradox
   - Assess whether group sizes support reliable comparison

5. **Structural/Temporal Exploration**
   For each time pattern in the spec:
   - Create time-series visualizations
   - Run decomposition
   - Identify structural breaks

6. **Hypothesis Refinement**
   For each finding, document in progress.md:
   - The pattern observed
   - Explanation A (behavioral/real)
   - Explanation B (artifact/confounding)
   - What would distinguish them

7. **Acceptance Criteria Check**
   Go through each acceptance criterion in the spec.
   Report:  met /  not met /  partially met with explanation.

Do NOT frame any EDA finding as a conclusion. Use language like
"the data suggests," "this pattern is consistent with," "further investigation needed."
```

---

## Phase 4: Modeling Specification

### Prompt: Write the Modeling Spec

```
Based on EDA findings in progress.md and the problem spec, write specs/04-modeling.md.

## Modeling Objective
- What does the model predict/classify/cluster?
- Target variable: name, type, distribution, class balance
- How will model output be used in the business decision?

## Approach Selection
For each candidate approach:
- Algorithm family and rationale (why this one?)
- Assumptions it makes and whether EDA supports them
- Interpretability vs. performance tradeoff
- Recommended: start with {{simplest reasonable model}} as baseline

## Feature Specification
- Feature list with source column, transformation, and rationale
- Features explicitly excluded and why (leakage risk, protected attributes, irrelevant)
- Feature engineering steps with connection to EDA findings

## Evaluation Framework
- Primary metric: {{name}} — why this metric?
- Secondary metrics: {{list}}
- Baseline performance: {{what naive/simple model achieves}}
- Target performance: {{what would be good enough for the business decision}}
- Validation strategy: {{holdout / k-fold / time-series split — justify}}
- Fairness checks: {{subgroups to evaluate separately}}

## Leakage Prevention
- [ ] No feature encodes the target
- [ ] No future information in training features
- [ ] Temporal ordering respected in all splits
- [ ] Aggregation features respect prediction-time constraints

## Experiment Tracking
- Parameters to log for each run
- Naming convention for experiments
- Comparison table format

## Acceptance Criteria
- [ ] Baseline model established and metrics logged
- [ ] At least one candidate model compared against baseline
- [ ] Feature importance analyzed and sanity-checked against domain knowledge
- [ ] Performance evaluated on held-out test set (touched only once)
- [ ] Subgroup performance checked — no unacceptable disparities
- [ ] All experiments logged with parameters and metrics

## Phase Gate
Model spec must be reviewed. Model proceeds to deployment spec only if acceptance criteria are met.
```

### Prompt: Execute Modeling Against Spec

```
Read specs/04-modeling.md and implement the modeling plan.

Work through the spec:

1. **Baseline First**
   Build the simplest model specified in the approach section.
   Log: model type, parameters, train/val metrics.
   This is the floor — everything else must beat it or be discarded.

2. **Feature Engineering**
   Implement only the features listed in the feature specification.
   Do NOT add features that aren't in the spec without updating the spec first.
   Run leakage prevention checks.

3. **Candidate Models**
   For each candidate in the spec:
   - Train with default hyperparameters first
   - Compare against baseline using the primary metric
   - If promising, tune hyperparameters (strategy specified in spec)
   - Log everything to the experiment tracking format

4. **Evaluation**
   Using the held-out test set (ONCE):
   - Compute all metrics specified
   - Run subgroup analysis
   - Generate confusion matrix / residual analysis
   - Produce feature importance analysis
   - Check: do important features make domain sense?

5. **Update progress.md**
   - Best model and its performance
   - Comparison table (all models, all metrics)
   - Key findings about what drives predictions
   - Honest assessment of limitations

6. **Acceptance Criteria Check**
   Report each criterion: met /  not met /  partially met with explanation.
```

---

## Phase 5: Deployment Specification

### Prompt: Write the Deployment Spec

```
Based on the validated model, write specs/05-deployment.md.

## Deployment Context
- Where will this model run? (batch pipeline, API, embedded in app)
- Prediction frequency and latency requirements
- Who/what consumes the predictions?

## Input/Output Contract
- Input schema: exact columns, types, valid ranges
- Output schema: prediction format, confidence scores, metadata
- Input validation: what checks run before prediction?
- Output validation: what sanity checks run on predictions?

## Monitoring Specification
- Input drift detection: method, threshold, check frequency
- Output drift detection: method, threshold
- Performance monitoring: metric, ground truth lag, degradation threshold
- Data quality checks: missing rates, schema violations
- Alert rules: what triggers human review vs. automatic rollback

## Operational Requirements
- Dependencies: pinned versions
- Resource requirements: memory, compute
- Fallback strategy: what happens when the model fails?
- Rollback procedure: how to revert to previous version

## Acceptance Criteria
- [ ] Model is packaged and reproducible from source
- [ ] Input/output contracts are tested
- [ ] Monitoring is implemented and alerting on test data
- [ ] Fallback strategy is tested
- [ ] Documentation is sufficient for someone else to operate it
```

---

## Phase 6: Communication Specification

### Prompt: Write the Communication Spec

```
Based on all project findings, write specs/06-communication.md.

## Audience
- Primary: {{who}} — what they care about, their technical level
- Secondary: {{who}} — different framing needs

## Key Messages
1. {{Most important finding — one sentence}}
2. {{Second finding — one sentence}}
3. {{Recommendation — one sentence}}

## Narrative Arc
1. The question and why it matters (1-2 sentences)
2. What we did (methods, plain language)
3. What we found (evidence, not jargon)
4. What it means (business impact — dollars, risk, customers)
5. What we recommend
6. What we don't know (limitations, honestly stated)
7. What comes next

## Deliverables
- [ ] Executive summary (1 page)
- [ ] Supporting visualizations (max {{N}})
- [ ] Technical appendix (for peer review)
- [ ] Reproducibility documentation

## Visualization Spec
For each key chart:
- Message: what the viewer should take away
- Data: what's plotted
- Type: chart type and why
- Annotations: what to highlight

## Guard Rails
- Do NOT overstate what EDA can support (hypotheses ≠ conclusions)
- Do NOT claim causation from observational data
- Do NOT present model performance without limitations
- DO frame uncertainty honestly

## Acceptance Criteria
- [ ] All visualizations pass the "so what?" test
- [ ] Limitations are stated before recommendations
- [ ] Non-technical stakeholder can understand the narrative
- [ ] Technical stakeholder can reproduce the analysis
```

---

## progress.md — Living Decision Log

### Prompt: Initialize the Progress Log

```
Create progress.md for this project with the following structure:

# Progress Log — {{Project Name}}

## Status
- Current phase: {{phase}}
- Last updated: {{date}}
- Next action: {{what's next}}

## Decision Log
| Date | Decision | Rationale | Spec Updated |
|------|----------|-----------|--------------|

## Key Findings
| Date | Finding | Confidence | Implications | Spec Reference |
|------|---------|------------|--------------|----------------|

## Open Questions
| Question | Raised By | Phase | Status |
|----------|-----------|-------|--------|

## Assumptions Tracker
| Assumption | Source | Validated? | Method | Result |
|------------|--------|------------|--------|--------|

## Data Lineage
| Dataset | Source | Rows | Cols | Acquired | Cleaned | Notes |
|---------|--------|------|------|----------|---------|-------|
```

---

## tasks.md — Task Breakdown with Verification

### Prompt: Generate Task Breakdown from Specs

```
Read all specs in specs/ and generate tasks.md.

For each phase, create tasks that:
1. Are small enough to complete in one Claude Code session
2. Have a clear "done" definition tied to a spec acceptance criterion
3. Include a verification step (test to run, check to perform)
4. Are ordered by dependency

Format:

# Tasks — {{Project Name}}

## Phase 1: Problem Definition
- [x] Write specs/01-problem.md
- [x] Review with stakeholder
- [ ] Finalize acceptance criteria

## Phase 2: Data
- [ ] Acquire dataset A (spec: 02-data.md §Required Data)
  - Verify: tests/test_data.py::test_dataset_a_schema passes
- [ ] Acquire dataset B
  - Verify: tests/test_data.py::test_dataset_b_schema passes
- [ ] Run data quality audit
  - Verify: all quality thresholds in spec met
- [ ] Clean and prepare data
  - Verify: tests/test_data.py::test_cleaned_data passes
- [ ] Update data dictionary

## Phase 3: EDA
- [ ] Run validation checks (spec: 03-eda.md §Validation)
  - Verify: all checks pass or failures documented
- [ ] Distributional exploration
  - Verify: all key variables documented with distribution type and outlier assessment
- [ ] Relational exploration
  - Verify: correlation matrix, key relationships visualized, alternatives noted
- [ ] Comparative exploration
  - Verify: segment comparisons with Simpson's Paradox check
- [ ] Structural/temporal exploration
  - Verify: time-series decomposition, structural breaks identified
- [ ] Synthesize findings and update progress.md
  - Verify: each finding has ≥2 competing explanations
- [ ] Phase gate review: decide next step

## Phase 4: Modeling (if warranted)
...
```

---

## Cross-Cutting: Session Management Prompts

### Start a New Session

```
Read AGENTS.md, then read progress.md to understand where we are.
Read the spec for the current phase (check progress.md for current phase).
Read tasks.md to see what's next.

Summarize:
1. Where we are in the project
2. What the current spec requires
3. What task is next
4. Any blockers or open questions from progress.md
```

### End a Session

```
Before we end this session:

1. Update progress.md with:
   - Any decisions made and their rationale
   - Any findings with confidence level
   - Any new open questions
   - Current status and next action

2. Update tasks.md:
   - Mark completed tasks
   - Note any blocked or changed tasks

3. If any finding changes our understanding of the problem,
   note which spec(s) need updating and what should change.

4. Commit all changes with a descriptive message.
```

### Sanity Check Against Spec

```
I want to verify our current work against the spec.

Read specs/{{current phase spec}}.
Read the relevant outputs (data, notebooks, results).

For each acceptance criterion in the spec:
- Met — show the evidence
- Not met — explain what's missing
- Partially met — explain the gap

Should we update the spec based on what we've learned,
or do more work to meet the existing criteria?
```

### Spec Revision Prompt

```
Our EDA has revealed something that changes our understanding.

Finding: {{what you learned}}
Impact: {{how it changes the question, data needs, or approach}}

Help me update the relevant specs:
1. What changes in specs/01-problem.md? (question, scope, assumptions)
2. What changes in specs/02-data.md? (new data needed, schema changes)
3. What changes in specs/03-eda.md? (new exploration needed)
4. What changes in tasks.md? (new or revised tasks)

Log the revision in progress.md with rationale.
This is the spec-driven cycle working as intended — specs evolve with understanding.
```

---

## Principles: Why Spec-Driven Data Science Works

**1. The spec prevents drift.**
Without a spec, an EDA session wanders. You explore a variable, find something interesting,
chase it, forget what you were originally looking for. The spec is the anchor —
it tells you what you intended to investigate and what "done" looks like.

**2. The spec enables verification.**
"Do good EDA" is not verifiable. "Check that all timestamp columns use consistent
UTC formatting and flag any rows with future dates" is. Specs turn vague goals into
testable acceptance criteria.

**3. The spec survives context loss.**
Claude Code sessions end. Context compacts. Notebooks get long. The spec persists
as a version-controlled artifact that the next session can read to pick up exactly
where you left off — without re-explaining the project.

**4. The spec separates thinking from doing.**
Writing the spec forces you to think through what you want before the agent starts
writing code. This is where experienced data scientists add the most value —
not in writing pandas code, but in deciding what questions to ask and what
would constitute a trustworthy answer.

**5. The spec makes iteration explicit.**
When EDA changes your question, you update the spec. That update is logged in
progress.md with rationale. Six months later, anyone can trace why the project
evolved from "predict churn" to "identify retention-sensitive segments."

---

## Quick Reference: Spec-Driven Workflow

| Step | Human Does | Claude Code Does |
|------|------------|-----------------|
| **Specify** | Writes/reviews the spec | Generates draft spec from prompt, refines based on feedback |
| **Plan** | Approves task breakdown | Generates tasks.md from specs, identifies dependencies |
| **Execute** | Reviews outputs periodically | Implements tasks, writes code, runs analyses |
| **Validate** | Confirms acceptance criteria | Checks outputs against spec, reports ✅/❌/⚠️ |
| **Update** | Decides what to change | Updates specs, progress.md, tasks.md with rationale |

---

## Quick Reference: Artifact Map

| Artifact | Purpose | Updated When |
|----------|---------|--------------|
| `AGENTS.md` | Agent steering (standards, structure, conventions) | Project setup; when conventions change |
| `specs/01-problem.md` | Analytical question & scope | Phase 1; when EDA changes the question |
| `specs/02-data.md` | Data requirements & quality standards | Phase 2; when new data is needed |
| `specs/03-eda.md` | Exploration plan with acceptance criteria | Phase 3; when hypotheses evolve |
| `specs/04-modeling.md` | Model approach & evaluation framework | Phase 4; when model strategy changes |
| `specs/05-deployment.md` | Production requirements & monitoring | Phase 5; when operational needs change |
| `specs/06-communication.md` | Audience, narrative, deliverables | Phase 6; when stakeholder needs shift |
| `progress.md` | Living log of decisions and findings | Every session |
| `tasks.md` | Current task breakdown with verification | Every session |
| `docs/data-dictionary.md` | Schema, definitions, lineage | When data or features change |

---
