# Term Deposit Customer Ranking

Machine-learning workflow for prioritising which already eligible customers should be called first about a term deposit.

## 1. Project Overview

This project develops a customer-prioritisation system for a capacity-constrained call centre using 40,000 completed marketing-call records.

The target variable is `y`, where `yes` means the customer subscribed to the term deposit and `no` means the customer did not subscribe. Only 2,896 of the 40,000 historical calls resulted in a subscription, giving a 7.24% conversion rate. Because 92.76% of outcomes are non-subscriptions, accuracy alone is not a useful measure of model quality.

The project therefore focuses on whether a model can place more likely subscribers near the top of a ranked call list. Logistic Regression and HistGradientBoosting are compared across exploratory analysis, leakage testing, baseline modelling, tuning, feature reduction, class-imbalance experiments, population analysis, ordered robustness, calibration and business lift.

The final recommendation is an unweighted Logistic Regression model using all 12 pre-call features, with HistGradientBoosting retained as a challenger. Under the corrected forward evaluation, calling the top-ranked 5% of customers across the eight eligible ordered test periods captured 228 subscribers versus about 123 expected under random selection at the same call volume, corresponding to a lift of approximately 1.85.

These results are retrospective and should be validated through a controlled live pilot before broader deployment.

## 2. Business Objective

The model is not intended to determine product eligibility, approve or reject customers, set prices or credit limits, guarantee that a customer will subscribe, or replace contact-frequency or operational rules.

Its intended role is to rank already eligible customers from highest to lowest priority:

**eligible customers → model score → ranked list → call highest-ranked customers first → stop when available call-centre capacity is reached**

The business question is therefore:

> Can the bank obtain more subscriptions from the same calling capacity?

Ranking is more useful than a rigid yes/no prediction because the number of customers contacted depends on available capacity, campaign economics and the value of moving further down the ranked list. The main evaluation measures are therefore ROC-AUC, PR-AUC and call-depth lift, supported by precision, recall and F1 where hard-classification behaviour is examined.

## 3. Dataset

| Measure | Value |
|---|---:|
| Records | 40,000 |
| Candidate predictors | 13 |
| Pre-call predictors | 12 |
| Subscribers | 2,896 |
| Non-subscribers | 37,104 |
| Subscription rate | 7.24% |
| Conventional null values | 0 |
| Exact duplicate rows | 0 |

The 12 pre-call predictors are `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`, `contact`, `day`, `month` and `campaign`.

`duration` is retained for diagnostic analysis only. It records completed call duration and is therefore unavailable when deciding whom to contact.

Explicit `unknown` categories are present in the data:

| Feature | Unknown records | Share |
|---|---:|---:|
| `contact` | 12,765 | 31.91% |
| `education` | 1,531 | 3.83% |
| `job` | 235 | 0.59% |

The dataset contains month labels but no verified year, campaign identifier or trustworthy timestamp. Ordered evaluation is therefore interpreted as a robustness test across changing campaign conditions rather than definitive future-calendar validation.

## 4. Quantitative Source of Truth

The five executed notebooks are the quantitative source of truth for this repository. Numerical claims in the README and the two PDF reports should be traceable to executed notebook outputs; a prose report must not override a conflicting notebook result.

The notebooks are reviewed in this order:

1. `01_data_eda.ipynb` — data inspection, class balance, missing/`unknown` handling, direct associations and ordered month-labelled exploration.
2. `02_baseline_tuning.ipynb` — untuned leakage baselines, pre-call baselines, four hyperparameter searches and selected configurations.
3. `03_features_imbalance.ipynb` — class-imbalance interventions, threshold analysis, feature importance and reduced-feature comparisons.
4. `04_segments_robustness.ipynb` — exploratory population analysis and expanding-window ordered validation.
5. `05_final_validation_reporting.ipynb` — final validation summaries, calibration, period-local lift and reporting outputs.

The published notebooks contain executed outputs. The current flat repository does not include the original CSV or the intermediate `results/` and `figures/` artifacts referenced by some notebook cells, so a fresh clone is reviewable but is not a completely self-contained clean rerun environment. That execution dependency is separate from the repository layout: the absence of folders, HTML exports or standalone `.py` scripts is not treated as a defect in the published submission.

## 5. Verified Modelling Sequence

The modelling sequence was audited against the executed notebooks and the computational code retained in the repository history.

### Step 1 — Untuned leakage baselines

Both model families are first evaluated using all 12 pre-call predictors plus `duration`. The purpose is diagnostic: completed-call duration is known to be unavailable at the pre-call decision point, so this comparison measures how strongly a post-call variable can inflate retrospective performance.

### Step 2 — Untuned deployable baselines

`duration` is removed and the same untuned model defaults are evaluated on the 12 deployable pre-call predictors. This makes the leakage comparison controlled: the predictor set changes, not the baseline model configuration.

### Step 3 — Tune four model/predictor configurations

Four searches are run separately:

- Logistic Regression, pre-call;
- Logistic Regression, with duration;
- HistGradientBoosting, pre-call;
- HistGradientBoosting, with duration.

Hyperparameter selection uses PR-AUC as the primary validation metric and ROC-AUC as the tie-break. The untouched final holdout is not used to choose hyperparameters.

### Step 4 — Class-imbalance and threshold experiments on tuned pre-call models

The selected tuned pre-call model structures are held fixed while six approaches are compared:

1. no adjustment / unweighted;
2. class weighting;
3. random oversampling;
4. random undersampling;
5. SMOTENC;
6. nested threshold optimisation.

The structural hyperparameters are not retuned inside these arms. Class weighting, oversampling, undersampling and SMOTENC alter fitting or the training distribution; threshold optimisation is different because it leaves the score ordering unchanged and selects a decision cutoff from inner out-of-fold predictions. It is therefore one of the six evaluated imbalance/operating-point approaches, but it is not a resampling strategy.

### Step 5 — Feature reduction

The reduced feature sets are selected directly from the HistGradientBoosting permutation-importance ranking measured by PR-AUC decrease on a stratified holdout:

- Top 3: `month`, `contact`, `day`;
- Top 4: `month`, `contact`, `day`, `housing`;
- Top 5: `month`, `contact`, `day`, `housing`, `age`;
- All 12: every valid pre-call predictor.

Pearson correlations, Cramér's V and Logistic Regression grouped coefficients are supporting interpretation only; they were not combined into a formal ranking to choose the tested subsets.

The already selected full-model hyperparameters are transferred unchanged to those reduced predictor sets; the reduced models are not retuned. The comparison therefore tests whether fewer predictors preserve ranking performance rather than giving reduced models an additional tuning advantage.

### Step 6 — Exploratory higher-conversion population

The exploratory rule `contact = cellular AND (job = retired OR age >= 71 OR balance >= Q3)` identifies 6,774 customers and 778 subscribers, giving an observed 11.49% conversion rate versus 7.24% overall.

For the canonical exploratory-population comparison, the **Baseline** rows are the relevant result: they use the same full-population untuned model settings inside the population and do not perform subgroup-specific hyperparameter tuning. The notebook also reports **Tuned transfer** rows, where the full-population tuned pre-call settings are transferred unchanged. Those rows are retained as a secondary sensitivity comparison, not as evidence of segment-specific retuning.

### Step 7 — Expanding-window ordered validation

The ordered robustness test uses the tuned, unweighted, full-12-feature pre-call Logistic Regression and HistGradientBoosting models. Hyperparameters are fixed before the expanding-window evaluation; they are not retuned within each later test block.

A test block is eligible only when at least 5,000 earlier training rows and at least 100 positive and 100 negative test outcomes are available. Eight blocks satisfy the gate. The primary ordered ROC-AUC is weighted by test-block size.

The executed ordered summary is:

| Metric | Logistic Regression | HistGradientBoosting |
|---|---:|---:|
| Eligible periods | 8 | 8 |
| Test rows | 30,526 | 30,526 |
| Test positives | 2,457 | 2,457 |
| Weighted ordered ROC-AUC | 0.5663 | 0.5530 |
| Weighted ordered PR-AUC | 0.1126 | 0.0968 |

### Step 8 — Calibration and sensitivity analyses

Calibration, period-local call-depth lift and pseudo-profile/group sensitivity are reported after the ordered ranking analysis. They answer different questions and are not used to retroactively select the model or ordered-validation hyperparameters.

## 6. Baseline Configurations and Hyperparameter Tuning

### Logistic Regression baseline

The untuned Logistic Regression baseline uses `C=1`, L2 regularisation, no class weighting and the standard unweighted fit. The same baseline configuration is used for the pre-call and with-duration comparisons.

The explicit constructor in the computational notebook is equivalent to:

```text
LogisticRegression(C=1, max_iter=100, tol=1e-4, random_state=42)
```

with the remaining scikit-learn defaults supplying L2 regularisation, `class_weight=None` and the default solver.

### Logistic Regression search

Each Logistic Regression predictor condition evaluates the exhaustive grid:

- `C`: 15 logarithmically spaced values from 0.001 to 100;
- penalty: L1 or L2;
- class weight: `None` or `balanced`;
- solver: `liblinear` for the tuned search;
- deterministic seed: 42.

This is `15 × 2 × 2 = 60` candidate configurations per predictor condition and 120 Logistic Regression candidates in total.

`C=1` is the baseline value but is **not** one of the 15 values produced by `np.logspace(-3, 2, 15)`. It should therefore not be described as a tuned-grid candidate.

The selected Logistic Regression configurations are:

| Predictor condition | Selected configuration |
|---|---|
| Pre-call | L2, `C=8.4834289824`, unweighted |
| With duration | L2, `C=0.1389495494`, unweighted |

### HistGradientBoosting baseline

The untuned HistGradientBoosting baseline uses 100 boosting iterations, learning rate 0.10, 31 maximum leaf nodes, L2 regularisation 0, minimum leaf size 20 and no class weighting, with seed 42. `max_depth` is left at its default `None`.

### HistGradientBoosting search

The full candidate space is:

- `max_iter`: 50, 100, 150, 200, 300, 400;
- `max_depth`: 3, 4, 5, 6, 8, `None`;
- `learning_rate`: 0.01, 0.03, 0.05, 0.10, 0.20;
- `max_leaf_nodes`: 15, 31, 63, 127;
- `l2_regularization`: 0, 0.1, 0.5, 1, 2;
- `min_samples_leaf`: 10, 20, 30, 50;
- `class_weight`: `None`, `balanced`.

This gives `6 × 6 × 5 × 4 × 5 × 4 × 2 = 28,800` possible combinations. For each predictor condition, 100 candidates are sampled reproducibly using `ParameterSampler(..., n_iter=100, random_state=42)`.

The same seeded candidate list is evaluated independently for the pre-call and with-duration conditions. Candidate 79 ranks first in both searches; the shared winning configuration is therefore an empirical result of the two searches, not a configuration copied from one condition to the other:

```text
max_iter=200
max_depth=8
learning_rate=0.05
max_leaf_nodes=15
l2_regularization=0
min_samples_leaf=20
class_weight=None
```

### Validation design used for tuning

The data are split into an 80% development set and a 20% untouched final holdout. Within development, the tuning-train and tuning-validation partitions contain 24,000 and 8,000 rows respectively. Logistic Regression candidates are fitted on the tuning-train split. HistGradientBoosting candidate screening uses a reproducible stratified 10,000-row subsample of the tuning-train data against the same 8,000-row validation set, after which the selected HGB candidate is refitted on the full 24,000-row tuning-train split.

Candidate ranking is by validation PR-AUC first and validation ROC-AUC second.

The four searches therefore contain **320 candidate configurations** in total:

- 60 LR pre-call;
- 60 LR with duration;
- 100 HGB pre-call;
- 100 HGB with duration.

The number 320 refers to candidate configurations evaluated by the four searches. It should not be described as 320 cross-validation fits.

## 7. Key Results

HistGradientBoosting performs better under shuffled historical evaluation, while Logistic Regression is more robust across later ordered campaign conditions and produces stronger historical lift at the tested call depths.

### Duration leakage and direct associations

`duration` has Pearson correlation `+0.4612` with the target, far larger than any valid numeric pre-call predictor. It is nevertheless diagnostic only because final call length does not exist when the bank chooses whom to contact.

The valid numeric pre-call Pearson correlations, ranked by absolute magnitude, are:

| Feature | Pearson r |
|---|---:|
| `campaign` | -0.0404 |
| `balance` | +0.0302 |
| `age` | -0.0203 |
| `day` | -0.0064 |

There is no fifth valid numeric pre-call predictor. `duration` must not be included in the deployable Pearson figure or feature-selection ranking.

The strongest categorical associations are:

| Feature | Cramér's V |
|---|---:|
| `month` | 0.1945 |
| `contact` | 0.0896 |
| `job` | 0.0589 |
| `marital` | 0.0579 |
| `housing` | 0.0541 |

These values describe direct univariate association; they are not themselves a feature-selection rule.

### Untouched-holdout baseline and tuned ranking results

The leakage comparison must distinguish baseline from tuned models and pre-call from duration-inclusive models. On the common untouched 20% holdout:

| Model | Stage | Predictors | ROC-AUC | PR-AUC |
|---|---|---|---:|---:|
| LR | Untuned | Pre-call | 0.7112 | 0.2417 |
| LR | Untuned | + duration | 0.9324 | 0.5250 |
| HGB | Untuned | Pre-call | 0.7341 | 0.2683 |
| HGB | Untuned | + duration | 0.9515 | 0.5852 |
| LR | Tuned | Pre-call | 0.7097 | 0.2403 |
| LR | Tuned | + duration | 0.9330 | 0.5232 |
| HGB | Tuned | Pre-call | 0.7299 | 0.2698 |
| HGB | Tuned | + duration | 0.9511 | 0.5815 |

Tuning changes these holdout ranking metrics only modestly. The major performance jump is caused by including post-call `duration`, not by hyperparameter optimisation. All deployable analyses therefore exclude `duration`.

### Feature importance and reduction

Logistic Regression grouped coefficient importance and HGB permutation importance answer different questions. The leading LR grouped coefficient magnitudes are `month` 0.9579, `contact` 0.4748, and a three-way displayed tie at 0.3444 for `loan`, `default` and `housing`. The HGB permutation ranking used to define the tested reduced subsets is `month` 0.1478, `contact` 0.1051, `day` 0.0559, `housing` 0.0348 and `age` 0.0144.

The transferred-parameter feature-reduction results are:

| Model | Metric | Top 3 | Top 4 | Top 5 | All 12 |
|---|---|---:|---:|---:|---:|
| LR | ROC-AUC | 0.6673 | 0.6728 | 0.6738 | **0.6848** |
| LR | PR-AUC | 0.1953 | 0.2122 | 0.2103 | **0.2154** |
| HGB | ROC-AUC | 0.7036 | 0.7062 | 0.7072 | **0.7153** |
| HGB | PR-AUC | 0.2439 | **0.2463** | 0.2446 | 0.2447 |

`month` and `contact` are the clearest consensus features. `day` has almost no direct Pearson relationship yet ranks third under HGB permutation importance, which is consistent with nonlinear or interaction value. `housing` contributes across direct association and model-based views, while `age` has weak direct Pearson correlation but measurable HGB permutation value.

All 12 predictors give the best ROC-AUC for both model families and the best LR PR-AUC. HGB Top 4 exceeds All 12 PR-AUC by only 0.0016 while its ROC-AUC is 0.0091 lower. That small PR-AUC difference is insufficient evidence to discard eight predictors, so the pilot configuration retains all 12 valid pre-call predictors.

### Class imbalance and threshold behaviour

The notebook comparison shows that weighting and resampling mainly move the precision/recall operating point rather than creating a large, consistent ranking gain. For HGB, random oversampling raises ROC-AUC from 0.7153 to 0.7191 and PR-AUC from 0.2447 to 0.2505, but accuracy falls from 0.9276 to 0.7507 and precision from 0.5086 to 0.1557. SMOTENC degrades ranking for both model families.

Nested threshold optimisation gives the strongest hard-classification F1 in this comparison: 0.2744 for LR and 0.3220 for HGB. Because threshold optimisation changes the binary cutoff rather than the score ordering, LR ROC-AUC/PR-AUC remain 0.6848/0.2154 and HGB remain 0.7153/0.2447.

For the ranking pilot, the unweighted model remains the operating form and call depth is selected from capacity and economics rather than a universal 0.5 probability cutoff.

### Shuffled versus ordered validation

The tuned, unweighted pre-call models show a clear validation-regime reversal:

| Model | Shuffled ROC-AUC | Shuffled PR-AUC | Weighted ordered ROC-AUC | Weighted ordered PR-AUC |
|---|---:|---:|---:|---:|
| LR | 0.6848 | 0.2154 | **0.5663** | **0.1126** |
| HGB | **0.7153** | **0.2447** | 0.5530 | 0.0968 |

HGB is stronger on row-stratified shuffled folds, but LR is stronger across the eight eligible later ordered blocks. The ordered comparison is therefore the more relevant robustness evidence for selecting the pilot ranker.

### Calibration and pseudo-profile sensitivity

Across the eight eligible ordered periods, the observed subscription rate is 8.05%. LR's mean score is 7.54%, with period-weighted Brier score 0.0748 and ECE 0.0444. HGB's mean score is 4.93%, with Brier score 0.0741 and ECE 0.0337. These diagnostics do not justify treating either score as a literal subscription probability; prospective recalibration is required.

A pseudo-profile grouped five-fold sensitivity using `age + job + marital + education + balance` leaves ranking results almost unchanged: LR ROC-AUC/PR-AUC are 0.6850/0.2157 and HGB are 0.7151/0.2444. Because the dataset has no true customer identifier, this is only a dependence sensitivity check and cannot prove customer-level independence.

### Exploratory population

The static full-data exploratory rule contains 6,774 customers and 778 subscribers, a historical subscription rate of 11.49%. In the fold-defined modelling comparison the eligible masks contain 6,772 observations because the balance-Q3 cutoff is learned within each training fold.

The canonical Baseline rows are LR ROC-AUC/PR-AUC 0.6925/0.2863 and HGB 0.6941/0.3035. Full-population tuned parameters transferred into the same population are retained only as a secondary sensitivity; no segment-specific hyperparameter tuning is claimed.

### Ordered robustness and lift

Across the eight eligible expanding-window periods, weighted ordered ROC-AUC is 0.5663 for Logistic Regression and 0.5530 for HistGradientBoosting. Period-local lift is stronger for Logistic Regression at the tested call depths; at 5% call depth it captures 228 subscribers compared with about 123 expected under random selection at the same call volume, a lift of approximately 1.85. At 20% depth it captures 689 subscribers versus about 492 expected randomly, corresponding to lift 1.4009.

The forward scores are useful for ranking but should not yet be treated as exact subscription probabilities. Prospective recalibration is required before probability-based expected-value or revenue calculations.

## 8. Final Recommendation

The recommended pilot model is **unweighted Logistic Regression using all 12 pre-call features**.

The model should be used to **rank already eligible customers from highest to lowest priority**, not as a rigid `yes/no` classifier. The number of customers contacted should be determined by available call-centre capacity, campaign economics and the observed value of moving further down the ranked list.

HistGradientBoosting should be retained as a **challenger model**. It performs better under shuffled historical evaluation, but Logistic Regression shows stronger robustness across later ordered campaign conditions and produces higher historical lift at the call depths tested.

The current model scores should be interpreted as **relative ranking scores rather than exact subscription probabilities**. A higher score supports calling one customer before another, but it should not yet be converted directly into a claimed probability of subscription or expected profit.

The next step is a **controlled live pilot testing whether Logistic-Regression-based customer prioritisation produces better business outcomes than the most appropriate available benchmark at the same calling effort**—either the bank's existing targeting process, if one is clearly defined, or random selection from the same eligible customer pool if no reliable process exists.

Broader deployment should occur only if the pilot shows that model-based prioritisation delivers more subscriptions for the same calling effort and produces positive business value without unacceptable operational or customer outcomes.

## 9. Repository Structure

The published repository is a flat submission bundle. No folder hierarchy, standalone `.py` scripts or HTML notebook exports are required for this version.

```text
VryLvaoTRIQL12B7/
├── README.md
├── 01_data_eda.ipynb
├── 02_baseline_tuning.ipynb
├── 03_features_imbalance.ipynb
├── 04_segments_robustness.ipynb
├── 05_final_validation_reporting.ipynb
├── Technical_Report.pdf
└── Business_Recommendations.pdf
```

The five `.ipynb` files are the analytical artifacts and contain code together with executed outputs. The two PDFs are presentation layers derived from the analysis. The README documents the project logic, verified modelling sequence, selected configurations and submission contents.

## 10. Reviewing the Analysis

For review, open the notebooks in numerical order from `01_data_eda.ipynb` through `05_final_validation_reporting.ipynb`. Their embedded executed outputs provide the quantitative evidence used by the project.

A completely clean rerun from a fresh clone requires the original source dataset and any intermediate result artifacts expected by the notebook cells; those dependencies are not included in the current flat repository. This README therefore does not claim that the published eight-file bundle is independently executable from scratch.

If the analysis is rerun in a full working environment, preserve the documented predictor definitions, random seed, tuning design, selection metrics and modelling sequence so that newly generated results remain comparable with the audited outputs.

## 11. Limitations and Next Steps

The main limitation is that the historical dataset does not contain verified years, campaign identifiers or trustworthy timestamps. The ordered evaluation therefore tests robustness across changing campaign conditions, but it cannot prove true future-calendar performance.

The available predictors are also limited. The dataset does not include richer CRM history such as previous contact outcomes, time since prior contact or broader customer relationship information. In addition, `contact=unknown` affects approximately 31.9% of records, making it an important data-quality issue for future campaigns.

The source data contain no true customer identifier. A pseudo-profile grouping based on `age + job + marital + education + balance` is therefore only a sensitivity check and cannot prove customer-level independence.

The model scores are useful for ranking, but they should not yet be treated as exact subscription probabilities. Prospective campaign data would be needed for stronger calibration and for reliable expected-value or profitability calculations.

The next step is a controlled live pilot of the selected Logistic Regression prioritisation approach, with HistGradientBoosting retained as a challenger. The pilot should compare outcomes at equal calling effort, record reliable campaign and operational data, and measure incremental subscriptions, cost per acquisition, realised lift and customer outcomes.

Future iterations should focus on better data before greater model complexity. Reliable campaign dates, campaign IDs, previous contact history, previous campaign outcomes, clearer contact-channel information and richer CRM variables are likely to provide more value than repeatedly tuning the same historical feature set.