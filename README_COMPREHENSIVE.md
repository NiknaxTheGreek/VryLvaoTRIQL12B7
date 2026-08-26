# Term Deposit Customer Ranking — Comprehensive Project Guide

This document is the extended companion to `README.md`. It records the complete analytical logic, modelling design, validated results, business interpretation, limitations and notebook-to-report traceability for the final project.

## 1. Project Purpose

The business problem is not simply to maximise a classification score. It is to answer a more practical question:

> **Can a bank rank already eligible customers before a call so that limited call-centre capacity reaches more likely subscribers first?**

The model is therefore designed to produce a **ranking signal**. It is not intended to determine product eligibility, approve or reject customers, set prices or credit limits, guarantee who will subscribe, replace consent/contact-frequency rules, or provide causal estimates of how much calling itself changes subscription behaviour.

The operational decision is:

**eligible customers → pre-call score → descending rank → call from the top until capacity is exhausted**.

This framing makes PR-AUC, ROC-AUC, ordered robustness and lift at realistic call depth more important than a single fixed classification threshold.

## 2. Data Audit and Prediction Timing

The source dataset contains **40,000 historical marketing calls**.

| Measure | Result |
|---|---:|
| Records | 40,000 |
| Subscribers | 2,896 |
| Non-subscribers | 37,104 |
| Subscription rate | 7.24% |
| Majority-class accuracy | 92.76% |
| Exact duplicate source rows | 0 |
| Conventional null values | 0 |
| Blank values | 0 |

The source includes 13 predictor fields plus the target `y`. The valid pre-call predictor set contains 12 features:

`age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`, `contact`, `day`, `month`, `campaign`.

### Explicit `unknown` categories

The dataset contains literal `unknown` strings rather than conventional missing values:

| Feature | `unknown` count |
|---|---:|
| `contact` | 12,765 |
| `education` | 1,531 |
| `job` | 235 |

These are retained as legitimate categories because replacing them with invented information would introduce assumptions not supported by the source data.

### Why `duration` is excluded

`duration` records how long the completed marketing call lasted. It is unavailable at the moment the bank must decide whom to call, so it cannot be used in a deployable pre-call ranking system.

Its Pearson correlation with the binary target is **0.461169**, far larger than any valid numeric pre-call association. This is useful diagnostic evidence of leakage, but not a reason to include the feature.

Selected pre-call Pearson correlations are small:

| Numeric feature | Pearson correlation with subscription |
|---|---:|
| `campaign` | -0.04035 |
| `balance` | +0.03023 |
| `age` | -0.02027 |
| `day` | -0.00642 |

Selected bias-corrected Cramér's V values for categorical variables are:

| Categorical feature | Cramér's V |
|---|---:|
| `month` | 0.19453 |
| `contact` | 0.08958 |
| `job` | 0.05890 |
| `marital` | 0.05786 |
| `housing` | 0.05411 |

Pearson correlation and Cramér's V are descriptive association measures. They are not merged with model-based importance into a single synthetic ranking.

## 3. Time and Ordering Limitation

The data contain month labels but no verified year, campaign identifier or trustworthy timestamp. The project therefore constructs **contiguous month-labelled source-order blocks** whenever the month label changes.

There are **13 contiguous blocks**. These are used as an ordered robustness proxy, but they must not be described as confirmed calendar months across known years.

This distinction matters because the main validation warning in the project is the large decline from shuffled historical evaluation to later-block ordered evaluation.

## 4. Model Families

### Logistic Regression

Logistic Regression is the interpretable linear baseline and final pilot candidate. It is attractive operationally because coefficients and preprocessing are transparent, the model is straightforward to monitor, score changes are easier to investigate, and it provides a strong reference against which additional model complexity can be justified.

### HistGradientBoosting

HistGradientBoosting is the nonlinear challenger. It can capture interactions and nonlinear relationships without extensive manual feature engineering and provides a meaningful test of whether additional model complexity improves ranking.

### Why only these two?

This is intentionally a **focused model-family comparison**, not an exhaustive benchmark of every classifier. The project tests whether a transparent linear model is sufficient or whether a materially more flexible nonlinear model provides enough extra value to justify added complexity.

## 5. Validation Design

All notebooks use `SEED=42`.

The 40,000 rows are split into:

- **32,000 development rows**;
- **8,000 untouched final-holdout rows**.

The development set is further split into:

- **24,000 tuning-train rows**;
- **8,000 tuning-validation rows**.

For HGB candidate search, a deterministic stratified **10,000-row tuning subsample** is used.

The hyperparameter search uses one fixed train/validation split. It is **not five-fold cross-validation tuning**. PR-AUC is the primary tuning objective and ROC-AUC is the deterministic tie-breaker.

## 6. Hyperparameters, Search Space, Baselines and Tuned Winners

This section deliberately separates four concepts: **what each hyperparameter means; what values are searched; the common baseline/default model specifications; and the four separately selected tuned configurations**.

### 6.1 Hyperparameter descriptions and tuning values

| Model | Hyperparameter | Description | Values searched |
|---|---|---|---|
| LR | `C` | Inverse regularisation strength; larger values apply weaker regularisation | 15 log-spaced values from 0.001 to 100: 0.001, 0.002276, 0.005179, 0.011788, 0.026827, 0.061054, 0.138950, 0.316228, 0.719686, 1.637894, 3.727594, 8.483429, 19.306977, 43.939706, 100 |
| LR | `penalty` | Form of coefficient regularisation | L1, L2 |
| LR | `class_weight` | Relative weighting of the minority and majority classes | None, balanced |
| HGB | `max_iter` | Maximum boosting iterations | 50, 100, 150, 200, 300, 400 |
| HGB | `max_depth` | Maximum depth of each tree | 3, 4, 5, 6, 8, None |
| HGB | `learning_rate` | Contribution made by each boosting iteration | 0.01, 0.03, 0.05, 0.10, 0.20 |
| HGB | `max_leaf_nodes` | Maximum terminal leaves per tree | 15, 31, 63, 127 |
| HGB | `l2_regularization` | L2 penalty applied to leaf values | 0, 0.1, 0.5, 1, 2 |
| HGB | `min_samples_leaf` | Minimum number of training observations in a leaf | 10, 20, 30, 50 |
| HGB | `class_weight` | Relative class weighting | None, balanced |

For LR tuning, `solver=liblinear`, `max_iter=3000` and `random_state=42` are fixed while `C`, `penalty` and `class_weight` are searched. That produces **60 exhaustive LR configurations per feature condition**.

For HGB tuning, `early_stopping=True` and `random_state=42` are fixed while the seven dimensions above define the search space. The full Cartesian space contains **28,800 combinations**. `ParameterSampler(..., n_iter=100, random_state=42)` selects the **same 100 candidates every rerun for each feature condition**.

Across LR pre-call, LR + duration, HGB pre-call and HGB + duration, exactly **320 candidate configurations** are evaluated.

### 6.2 Baseline/default model specifications

The baseline stage is a like-for-like feature-availability experiment. **Pre-call and + duration use exactly the same baseline hyperparameters within each model family. The only change is whether `duration` is present.**

There are therefore only **two baseline model specifications**:

| Baseline model | Hyperparameters used unchanged for both feature conditions |
|---|---|
| LR | `C=1.0`, L2 penalty, `solver=lbfgs`, `max_iter=1000`, `tol=0.0001`, `class_weight=None`, `random_state=42` |
| HGB | `loss=log_loss`, `learning_rate=0.1`, `max_iter=100`, `max_leaf_nodes=31`, `max_depth=None`, `min_samples_leaf=20`, `l2_regularization=0`, `max_bins=255`, `early_stopping=auto`, `class_weight=None`, `random_state=42` |

This means the baseline comparisons are:

- LR baseline + 12 pre-call predictors **versus the same LR baseline + `duration`**;
- HGB baseline + 12 pre-call predictors **versus the same HGB baseline + `duration`**.

That structure isolates the effect of post-call information without changing the baseline model settings at the same time.

### 6.3 Tuned model specifications

Tuning is performed **separately for each model × feature condition**, so the four tuned winners are intentionally allowed to differ.

| Tuned configuration | Selected hyperparameters |
|---|---|
| LR pre-call | L2, `C=8.483428982440726`, `class_weight=None`; fixed: `solver=liblinear`, `max_iter=3000`, `random_state=42` |
| LR + duration | L1, `C=0.0610540229658533`, `class_weight=None`; fixed: `solver=liblinear`, `max_iter=3000`, `random_state=42` |
| HGB pre-call | `max_iter=200`, `max_depth=8`, `learning_rate=0.05`, `max_leaf_nodes=15`, `l2_regularization=0`, `min_samples_leaf=20`, `class_weight=None`, `early_stopping=True`, `random_state=42` |
| HGB + duration | `max_iter=200`, `max_depth=3`, `learning_rate=0.05`, `max_leaf_nodes=127`, `l2_regularization=2`, `min_samples_leaf=10`, `class_weight=balanced`, `early_stopping=True`, `random_state=42` |

The selected LR pre-call candidate is run 46 of its deterministic search and the selected HGB pre-call candidate is run 79. Reproducibility depends on the dataset, split logic, random seeds and pinned package environment remaining unchanged.

## 7. Baseline, Tuned and Leakage Results

### Untouched 20% final holdout

| Stage | Model | Feature set | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Baseline/default | LR | Pre-call | 0.928250 | 0.575758 | 0.032815 | 0.062092 | 0.241676 | 0.711156 |
| Baseline/default | LR | + duration | 0.938125 | 0.655556 | 0.305699 | 0.416961 | 0.525049 | 0.932379 |
| Baseline/default | HGB | Pre-call | 0.928000 | 0.534884 | 0.039724 | 0.073955 | 0.268264 | 0.734102 |
| Baseline/default | HGB | + duration | 0.941125 | 0.639896 | 0.426598 | 0.511917 | 0.585190 | 0.951528 |
| Tuned | LR | Pre-call | 0.928125 | 0.558824 | 0.032815 | 0.061990 | 0.240265 | 0.709681 |
| Tuned | LR | + duration | 0.936500 | 0.646091 | 0.271157 | 0.381995 | 0.520693 | 0.933188 |
| Tuned | HGB | Pre-call | 0.928875 | 0.619048 | 0.044905 | 0.083736 | 0.272820 | 0.733930 |
| Tuned | HGB | + duration | 0.866000 | 0.338572 | 0.892919 | 0.490978 | 0.549825 | 0.943115 |

The key interpretation is not that a model with `duration` is better. It is that the post-call feature creates an enormous retrospective advantage that would be unavailable at decision time.

Tuning changes pre-call ranking only modestly:

- LR PR-AUC: **0.241676 → 0.240265**;
- HGB PR-AUC: **0.268264 → 0.272820**.

This is one reason the final decision relies on robustness and lift rather than treating the highest single holdout metric as sufficient.

## 8. Feature Importance and Reduction

Direct association, grouped LR coefficient magnitude and HGB permutation importance are treated as different evidence types. Notebook 03 generates separate horizontal bar plots for the LR grouped coefficient magnitude and HGB held-out permutation importance; these are the canonical source visuals for the Technical Report.

The reduction order is defined **only by HGB held-out permutation importance using decrease in PR-AUC**.

| Rank | Feature | Permutation importance |
|---:|---|---:|
| 1 | `month` | 0.168555 |
| 2 | `contact` | 0.119955 |
| 3 | `day` | 0.081196 |
| 4 | `housing` | 0.025086 |
| 5 | `age` | 0.011867 |
| 6 | `balance` | 0.005755 |
| 7 | `marital` | 0.003635 |
| 8 | `campaign` | 0.003570 |
| 9 | `loan` | 0.002746 |
| 10 | `education` | 0.002696 |
| 11 | `default` | -0.000307 |
| 12 | `job` | -0.000900 |

Negative permutation values should not be interpreted as proof that a variable is intrinsically harmful in every setting.

### Feature-reduction holdout comparison

| Features | Model | Accuracy | F1 | PR-AUC | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Top 3 | LR | 0.928125 | 0.046434 | 0.214884 | 0.691444 |
| Top 3 | HGB | 0.928750 | 0.086538 | 0.265716 | 0.713004 |
| Top 4 | LR | 0.928875 | 0.065681 | 0.235485 | 0.698631 |
| Top 4 | HGB | 0.929375 | 0.107425 | **0.274531** | 0.719196 |
| Top 5 | LR | 0.928625 | 0.068515 | 0.235275 | 0.699099 |
| Top 5 | HGB | 0.928000 | 0.058824 | 0.272162 | 0.720062 |
| All 12 | LR | 0.928125 | 0.061990 | **0.240265** | **0.709681** |
| All 12 | HGB | 0.928875 | 0.083736 | 0.272820 | **0.733930** |

HGB Top-4 has slightly higher PR-AUC than HGB All-12 on this exploratory holdout comparison, but All-12 has materially stronger ROC-AUC. No subset-specific retuning is performed. The evidence therefore does not justify replacing the full valid pre-call set in the final pilot specification.

## 9. Class-Imbalance Experiments

The project compares six strategies on the tuned pre-call model structures: no adjustment, balanced class weighting, random oversampling, random undersampling, SMOTENC and threshold tuning.

| Model | Strategy | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| LR | none | 0.928125 | 0.558824 | 0.032815 | 0.061990 | 0.240265 | 0.709681 |
| LR | class weight | 0.662875 | 0.126587 | 0.620035 | 0.210249 | 0.231243 | 0.710575 |
| LR | oversampling | 0.661000 | 0.126967 | 0.626943 | 0.211169 | 0.232024 | 0.710790 |
| LR | undersampling | 0.651875 | 0.123549 | 0.625216 | 0.206327 | 0.218951 | 0.708137 |
| LR | SMOTENC | 0.689875 | 0.119295 | 0.514680 | 0.193695 | 0.172603 | 0.656204 |
| LR | threshold tuning | 0.895125 | 0.282609 | 0.291883 | **0.287171** | 0.240265 | 0.709681 |
| HGB | none | 0.928875 | 0.619048 | 0.044905 | 0.083736 | 0.272820 | 0.733930 |
| HGB | class weight | 0.754750 | 0.161527 | 0.569948 | 0.251716 | 0.271896 | 0.732946 |
| HGB | oversampling | 0.753625 | 0.159491 | 0.563040 | 0.248570 | **0.274559** | 0.733041 |
| HGB | undersampling | 0.671750 | 0.134072 | 0.647668 | 0.222156 | 0.241083 | 0.723593 |
| HGB | SMOTENC | 0.775250 | 0.150716 | 0.454231 | 0.226334 | 0.210334 | 0.674565 |
| HGB | threshold tuning | 0.919375 | 0.421053 | 0.303972 | **0.353059** | 0.272820 | 0.733930 |

Thresholds are selected using five-fold stratified out-of-fold predictions inside development and then applied once to the untouched holdout:

- LR threshold: **0.141**, development OOF F1 **0.268323**, holdout F1 **0.287171**;
- HGB threshold: **0.156**, development OOF F1 **0.321025**, holdout F1 **0.353059**.

Threshold tuning changes the hard decision cutoff only. PR-AUC and ROC-AUC remain identical to the underlying unweighted model because the score ordering is unchanged.

The ranking recommendation therefore remains **unweighted**. Weighting and resampling alter operating trade-offs but do not provide a material general ranking advantage.

## 10. Exploratory Higher-Conversion Population

A descriptive rule identified during EDA is:

`contact = cellular AND (job = retired OR age >= 71 OR balance >= Q3)`

On the full dataset:

- balance Q3 = **1,319**;
- population size = **6,774**;
- subscribers = **778**;
- conversion rate = **11.485%**;
- descriptive relative uplift versus 7.24% overall = **58.63%**.

For predictive validation, the Q3 threshold is learned within each training fold rather than from the full dataset, and the full-population **baseline/default** model structures are used without subgroup-specific hyperparameter tuning.

| Model | Population PR-AUC | Population ROC-AUC |
|---|---:|---:|
| LR baseline/default | 0.286349 | 0.692504 |
| HGB baseline/default | 0.303476 | 0.694120 |

This population is useful business context, but it is **not a recommendation to replace model ranking with a demographic rule**.

## 11. Shuffled and Pseudo-Profile Robustness

### Shuffled five-fold validation

| Model | PR-AUC | ROC-AUC |
|---|---:|---:|
| LR | 0.215433 | 0.684767 |
| HGB | **0.245333** | **0.717461** |

HGB is clearly stronger under conventional shuffled validation.

### Pseudo-profile grouped sensitivity

The dataset has no verified customer identifier. Exact combinations of `age + job + marital + education + balance` are used as pseudo-groups in `StratifiedGroupKFold`.

| Model | PR-AUC | ROC-AUC |
|---|---:|---:|
| LR | 0.215690 | 0.685003 |
| HGB | **0.248093** | **0.716737** |

These results are similar to ordinary shuffled validation. That is reassuring, but it **does not prove true customer-level independence**, because the pseudo-group is not an actual customer ID.

## 12. Expanding-Window Ordered Validation

The ordered robustness design trains on all eligible earlier blocks and tests on a later contiguous month-labelled block. Eight blocks pass the eligibility gate, covering **30,526 test rows** and **2,457 subscribers**.

Weighted by test-block size:

| Model | PR-AUC | ROC-AUC | Brier |
|---|---:|---:|---:|
| LR | **0.112626** | **0.566290** | 0.074776 |
| HGB | 0.096333 | 0.551600 | **0.074181** |

This is the central model-selection result. HGB wins shuffled historical evaluation, but LR is stronger under the more demanding ordered test.

The decline from shuffled to ordered performance means the dataset is not temporally stable enough to support an unconditional claim that the best shuffled model will remain best in later campaign conditions.

## 13. Calibration Diagnostics

Calibration is assessed period by period using local quantile bins, Brier score and expected calibration error (ECE). No sigmoid calibration model is fitted and no isotonic calibration model is fitted.

| Model | Weighted mean score | Weighted Brier | Weighted ECE |
|---|---:|---:|---:|
| LR | 0.075410 | 0.074776 | 0.044442 |
| HGB | 0.048578 | 0.074181 | 0.035101 |

The scores are therefore used as **relative ranking signals**. Probability-dependent economics should wait for prospective recalibration on new campaign data.

## 14. Period-Local Lift and Call Depth

Lift compares the subscription rate among the highest-ranked records with the subscription rate expected from random calling at the same volume.

| Call depth | LR lift | HGB lift |
|---:|---:|---:|
| 5% | **1.853857×** | 1.284690× |
| 10% | **1.606389×** | 1.293245× |
| 20% | **1.401474×** | 1.248918× |
| 50% | **1.187549×** | 1.133829× |

For LR:

- at 5% depth: **1,528 calls**, **228 subscribers captured**, about **122.99 expected randomly**;
- at 20% depth: **6,108 calls**, **689 subscribers captured**, about **491.63 expected randomly**.

These are retrospective sensitivity points, not prescribed production quotas. Actual call depth should be determined by staff capacity, contact cost, expected contribution from a successful subscription and risk controls.

## 15. Final Model Decision

The final pilot recommendation is:

**Primary candidate: unweighted Logistic Regression using all 12 valid pre-call features.**

**Challenger: HistGradientBoosting using the tuned unweighted pre-call structure.**

LR is not chosen because it wins every historical metric. It does not. HGB has stronger shuffled PR-AUC and ROC-AUC. LR is chosen because the model-selection decision incorporates later-block ordered robustness, period-local historical lift, transparency, monitoring simplicity, limited benefit from additional imbalance complexity and the need for a conservative controlled pilot rather than immediate production rollout.

## 16. Controlled Pilot Design

The next business step is a prospective test, not further retrospective score polishing.

A suitable pilot compares **equal calling capacity** across a model-ranked LR arm and a business-as-usual or random comparator arm, with an optional HGB challenger arm if operational capacity allows.

The primary business outcome should be **subscriptions per 1,000 assigned calls**. Useful secondary measures include contactability, opt-outs, complaints, repeat contacts, customer mix and agent time where available.

The pilot should predefine stop/go criteria and avoid interpreting retrospective ranking lift as causal uplift from deployment.

## 17. Economics

The historical dataset does not contain the bank's actual call cost or net contribution per new term-deposit subscription. The project therefore does **not invent ROI**.

Once those values are supplied:

**net value = successful subscriptions × net contribution per subscription − completed calling effort × relevant call cost**.

If probability-based expected value is required at customer level, the model scores must first be prospectively recalibrated.

## 18. Key Limitations

1. **Temporal information is incomplete.** Ordered blocks preserve source order but are not verified campaign timestamps.
2. **Ordered performance is materially weaker.** This is the main reason for recommending a controlled pilot only.
3. **No true customer ID exists.** Pseudo-profile grouped validation is a sensitivity check, not proof of customer-level independence.
4. **Historical lift is not causal uplift.** It shows enrichment in retrospective ranking, not the incremental treatment effect of calling.
5. **Raw scores are not trusted probabilities.** Calibration diagnostics are descriptive and no recalibration model is fitted.
6. **The exploratory higher-conversion population is descriptive.** It should not replace model ranking or be treated as a standalone policy rule.
7. **Feature-reduction comparisons are exploratory.** A small PR-AUC difference on one holdout does not prove a reduced subset is universally superior.

## 19. Notebook Traceability

Every analytical result used in the final project should be traceable to an executed notebook. The notebooks are the numerical source of truth.

| Evidence / output | Source notebook | Purpose |
|---|---|---|
| 40,000 rows; 2,896 subscribers; 7.24% rate | `01_data_eda.ipynb` | Data audit and target definition |
| Missing/unknown audit and duplicate count | `01_data_eda.ipynb` | Data quality |
| Pearson and Cramér's V associations | `01_data_eda.ipynb` | Descriptive association |
| `duration` correlation diagnostic | `01_data_eda.ipynb` | Leakage evidence |
| Ordered month-labelled block construction and subscription-rate plot | `01_data_eda.ipynb` | Ordering / temporal context |
| Exploratory population size and 11.485% conversion | `01_data_eda.ipynb` | Descriptive business context |
| Common baseline/default LR/HGB specifications | `02_baseline_tuning.ipynb` | Like-for-like baseline comparison |
| Hyperparameter descriptions, search values and candidate counts | `02_baseline_tuning.ipynb` | Reproducible search design |
| Four separately selected tuned configurations | `02_baseline_tuning.ipynb` | Tuned model specification |
| Untouched 20% holdout table and six-metric panel | `02_baseline_tuning.ipynb` | Core historical evaluation |
| LR grouped-coefficient feature-importance bar plot | `03_features_imbalance.ipynb` | Linear model interpretation |
| HGB permutation-importance bar plot | `03_features_imbalance.ipynb` | Nonlinear feature interpretation / reduction order |
| Top-3 / Top-4 / Top-5 / All-12 comparison and six-metric panel | `03_features_imbalance.ipynb` | Feature reduction |
| Six imbalance strategies and six-metric panel | `03_features_imbalance.ipynb` | Imbalance trade-offs |
| OOF threshold selection and holdout F1 | `03_features_imbalance.ipynb` | Hard-classification diagnostic |
| Exploratory population predictive validation | `04_segments_robustness.ipynb` | Population robustness |
| Shuffled five-fold validation | `04_segments_robustness.ipynb` | Conventional robustness baseline |
| Pseudo-profile grouped sensitivity | `04_segments_robustness.ipynb` | Similar-record sensitivity |
| Expanding-window block results and weighted ordered metrics | `04_segments_robustness.ipynb` | Forward robustness |
| Shuffled vs Ordered Validation figure | `04_segments_robustness.ipynb` | Report-facing model-selection comparison |
| Period-local lift table and figure | `05_final_validation_reporting.ipynb` | Business call-depth evidence |
| Calibration table and reliability figure | `05_final_validation_reporting.ipynb` | Score reliability diagnostic |

All five notebooks contain explanatory Markdown between substantive code stages so the calculation sequence, rationale and interpretation remain visible alongside the executable analysis.

## 20. Repository Structure

The project repository is intentionally **flat**. It contains no project folders and no HTML notebook exports.

```text
01_data_eda.ipynb
02_baseline_tuning.ipynb
03_features_imbalance.ipynb
04_segments_robustness.ipynb
05_final_validation_reporting.ipynb
Business_Recommendations.pdf
README.md
README_COMPREHENSIVE.md
Technical_Report.pdf
requirements.txt
term-deposit-marketing-2020.csv
term-deposit-marketing-2020-labelled.csv
```

The original source CSV is retained alongside the labelled working copy. The labelled copy contains the derived `y_binary` helper, but each notebook drops and recreates that helper before analysis so stale helper values are never treated as source truth.

## 21. Reproducing the Analysis

Install the pinned environment:

```bash
pip install -r requirements.txt
```

Then execute the notebooks in order:

```text
01_data_eda.ipynb
02_baseline_tuning.ipynb
03_features_imbalance.ipynb
04_segments_robustness.ipynb
05_final_validation_reporting.ipynb
```

The notebooks read `term-deposit-marketing-2020-labelled.csv` directly from the repository root. They do not require `data/`, `results/`, `figures/`, `scripts/` or `html/` directories and do not require cached CSV, JSON or NPY intermediates.

The pinned environment is recorded in `requirements.txt` so the deterministic splits, seeded candidate sampling and fitted implementations can be reproduced as closely as possible.

## 22. Project Decision in One Sentence

**The historical evidence supports testing an all-12 unweighted Logistic Regression ranking system in a controlled live pilot, with HGB retained as a challenger, because LR is more robust under the stricter ordered validation and produces the strongest historical call-depth lift even though HGB wins conventional shuffled validation.**