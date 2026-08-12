# Term Deposit Customer Ranking

Machine-learning workflow for prioritising which already eligible customers should be called first about a term deposit.

## 1. Project Overview

This project develops a customer-prioritisation system for a capacity-constrained call centre using 40,000 completed marketing-call records.

The target variable is `y`, where `yes` means the customer subscribed to the term deposit and `no` means the customer did not subscribe. Only 2,896 of the 40,000 historical calls resulted in a subscription, giving a 7.24% conversion rate. Because 92.76% of outcomes are non-subscriptions, accuracy alone is not a useful measure of model quality.

The project therefore focuses on whether a model can place more likely subscribers near the top of a ranked call list. Logistic Regression and HistGradientBoosting are compared across exploratory analysis, leakage testing, baseline modelling, tuning, feature reduction, class-imbalance experiments, segment testing, ordered robustness, calibration and business lift.

The final recommendation is an unweighted Logistic Regression model using all 12 pre-call features, with HistGradientBoosting retained as a challenger. Under the corrected forward evaluation, calling the top-ranked 5% of customers across the eight eligible ordered test periods captured 228 subscribers versus about 123 expected under random selection at the same call volume, corresponding to a lift of approximately 1.85.

These results are retrospective and should be validated through a controlled live pilot before broader deployment.

## 2. Business Objective

The model is not intended to determine product eligibility, approve or reject customers, set prices or credit limits, guarantee that a customer will subscribe, or replace contact-frequency or operational rules.

Its intended role is to rank already eligible customers from highest to lowest priority:

**eligible customers → model score → ranked list → call highest-ranked customers first → stop when available call-centre capacity is reached**

The business question is therefore:

> Can the bank obtain more subscriptions from the same calling capacity?

Ranking is more useful than a rigid yes/no prediction because the number of customers contacted depends on available capacity, campaign economics and the value of moving further down the ranked list. The main evaluation measures are therefore ROC-AUC, PR-AUC and call-depth lift, supported by precision, recall, F1 and confusion matrices where relevant.

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

## 4. Analytical Workflow

The workflow begins with exploratory data analysis covering class balance, data quality, explicit `unknown` categories, numeric and categorical distributions, conversion patterns and ordered month-labelled blocks. Numeric feature-to-target relationships are measured using Pearson correlation, while categorical relationships are measured using bias-corrected Cramér's V. These are treated as complementary direct-association measures rather than interchangeable statistics.

Logistic Regression and HistGradientBoosting are then evaluated as untuned baselines under stratified five-fold validation. A controlled leakage experiment compares both models using the 12 pre-call predictors and the same predictors plus `duration`.

Hyperparameter tuning uses PR-AUC as the optimisation objective, but parameter selection is separated from final evaluation. The data are split into an 80% development set and an untouched 20% final holdout. Within development, Logistic Regression is selected using a fixed tuning-train/tuning-validation split, while each HistGradientBoosting search evaluates 100 reproducible random candidates using a stratified 10,000-row tuning subsample and the separate tuning-validation set. The untouched holdout is used only after the settings have been selected.

| Model | Baseline configuration | Hyperparameter search |
|---|---|---|
| Logistic Regression | L2, `C=1.0`, no class weighting | 15 logarithmically spaced `C` values from 0.001 to 100 × L1/L2 × none/balanced = 60 combinations |
| HistGradientBoosting | 100 iterations, depth=None, learning rate=0.10, 31 leaves, L2=0, minimum leaf size=20, unweighted | iterations 50–400, depth 3–8/None, learning rate 0.01–0.20, 15–127 leaves, L2 0–2, minimum leaf size 10–50, none/balanced; 100 candidates sampled from ~28,800 combinations |

With-duration and pre-call models are tuned separately.

After tuning, feature importance is examined using Pearson correlation and Cramér's V, grouped absolute standardised Logistic Regression coefficient magnitude, and held-out HistGradientBoosting permutation importance. The combined evidence is used to define Top-3, Top-4 and Top-5 feature subsets. The selected full-model hyperparameters are transferred unchanged to these reduced feature sets; the reduced models are not retuned.

Class-imbalance experiments compare no adjustment, class weighting, random oversampling, random undersampling, SMOTENC and threshold optimisation. For the standardised imbalance comparison, the selected tuned models are unweighted. Weighting and resampling are treated as intervention sensitivities with the structural hyperparameters held fixed rather than as separately retuned model contests. Threshold optimisation uses nested inner out-of-fold predictions within each outer fold.

An exploratory higher-conversion population is then examined. Baseline models are fitted within the population and the full-population selected hyperparameters are transferred unchanged; there is no segment-specific retuning.

Finally, tuned unweighted Logistic Regression and HistGradientBoosting are compared under expanding-window ordered validation. A test block is included only when at least 5,000 earlier training rows and at least 100 positive and 100 negative test outcomes are available. Eight blocks satisfy this gate, and the primary ordered ROC-AUC is weighted by test-block size. Calibration is assessed within each eligible forward period before aggregation, followed by period-local call-depth lift. A pseudo-group sensitivity analysis based on `age + job + marital + education + balance` is also reported because no true customer identifier is available.

## 5. Key Results

HistGradientBoosting performs better under shuffled historical validation, while Logistic Regression is more robust across later ordered campaign conditions and produces stronger historical lift at the tested call depths.

### Selected configurations

| Model | Feature set | Selected configuration |
|---|---|---|
| Logistic Regression | Pre-call | L2, `C=8.483`, unweighted |
| HistGradientBoosting | Pre-call | 200 iterations, depth 8, learning rate 0.05, 15 leaves, L2=0, minimum leaf size=20, unweighted |
| Logistic Regression | With duration | L2, `C=0.139`, unweighted |
| HistGradientBoosting | With duration | 200 iterations, depth 8, learning rate 0.05, 15 leaves, L2=0, minimum leaf size=20, unweighted |

### Baseline and tuning

On the untouched 20% final holdout:

- Logistic Regression baseline pre-call ROC-AUC: approximately 0.711; tuned: approximately 0.710.
- HistGradientBoosting baseline pre-call ROC-AUC: approximately 0.734; tuned: approximately 0.730.
- Tuning therefore produces little or no holdout ranking improvement; its role is model selection rather than evidence that additional complexity improves generalisation.
- The four searches contain 320 candidate runs in total: 60 LR pre-call, 60 LR with duration, 100 HGB pre-call and 100 HGB with duration.

### Duration leakage

On the untouched final holdout, including completed-call `duration` increases:

- Logistic Regression ROC-AUC from about 0.710 to 0.933;
- HistGradientBoosting ROC-AUC from about 0.730 to 0.951.

This confirms that `duration` contains strong retrospective information but cannot be used for pre-call targeting.

### Feature reduction

The selected full-model hyperparameters are transferred unchanged to the reduced feature sets. Reduced models are not retuned.

| Model | Top 3 ROC-AUC | Top 4 | Top 5 | All 12 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.667 | 0.673 | 0.674 | 0.685 |
| HistGradientBoosting | 0.704 | 0.706 | 0.707 | 0.715 |

All 12 features retain the best ROC-AUC for both model families. Some reduced HGB sets have very similar PR-AUC, but the combined evidence does not justify dropping variables from the pilot model.

### Class imbalance

Weighting and resampling mainly alter the precision/recall trade-off. For Logistic Regression they provide little ranking benefit. HGB oversampling gives only a small ROC-AUC/PR-AUC increase while sharply reducing precision and accuracy. SMOTENC degrades ranking performance for both model families.

Nested threshold optimisation gives the strongest hard-classification F1 while preserving the underlying score ordering:

- Logistic Regression F1: approximately 0.274;
- HistGradientBoosting F1: approximately 0.322.

The pilot therefore retains the unweighted ranking models and chooses call depth from capacity/economics rather than from the default 0.5 threshold.

### Exploratory population

The exploratory rule `contact = cellular AND (job = retired OR age >= 71 OR balance >= Q3)` identifies 6,774 customers and 778 subscribers, giving 11.49% conversion versus 7.24% overall. However, uncertainty analysis does not show a reliable advantage for a separate segment-specific model.

### Ordered robustness and lift

Eight forward blocks satisfy the corrected evidence gate of at least 5,000 earlier training rows plus at least 100 positive and 100 negative outcomes in the test block. The primary ordered result is weighted by test-block size.

| Metric | Logistic Regression | HistGradientBoosting |
|---|---:|---:|
| Shuffled ROC-AUC | ~0.685 | ~0.715 |
| Weighted ordered ROC-AUC | 0.566 | 0.553 |
| 5% call-depth lift | 1.85 | 1.31 |
| 10% call-depth lift | 1.61 | 1.27 |
| 20% call-depth lift | 1.40 | 1.27 |

At 5% call depth, Logistic Regression captured 228 subscribers compared with about 123 expected under random selection at the same call volume. At 20% depth, it captured 689 compared with about 492 expected under random selection.

Forward Logistic Regression scores remain useful for ranking but should not be treated as literal probabilities. Across the eight eligible periods the mean score is about 7.54% versus an observed 8.05% subscription rate, with period-weighted ECE about 0.044. Prospective recalibration is still required before probability-based revenue or expected-value calculations.

## 6. Final Recommendation

The recommended pilot model is **unweighted Logistic Regression using all 12 pre-call features**.

The model should be used to **rank already eligible customers from highest to lowest priority**, not as a rigid `yes/no` classifier. The number of customers contacted should be determined by available call-centre capacity, campaign economics and the observed value of moving further down the ranked list.

HistGradientBoosting should be retained as a **challenger model**. It performs better under shuffled historical validation, but Logistic Regression shows stronger robustness across later ordered campaign conditions and produces higher historical lift at the call depths tested.

The current model scores should be interpreted as **relative ranking scores rather than exact subscription probabilities**. A higher score supports calling one customer before another, but it should not yet be converted directly into a claimed probability of subscription or expected profit.

The next step is therefore a **controlled live pilot testing whether Logistic-Regression-based customer prioritisation produces better business outcomes than the most appropriate available benchmark at the same calling effort**—either the bank's existing targeting process, if one is clearly defined, or random selection from the same eligible customer pool if no reliable process exists.

Broader deployment should occur only if the pilot shows that model-based prioritisation delivers **more subscriptions for the same calling effort and produces positive business value without unacceptable operational or customer outcomes**.

## 7. Repository Structure

```text
xQgRpzDwwGEKdDsD/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── term-deposit-marketing-2020.csv
│
├── scripts/
│   ├── 01_data_eda.py
│   ├── 02_baseline_tuning.py
│   ├── 03_features_imbalance.py
│   ├── 04_segments_robustness.py
│   └── 05_final_validation_reporting.py
│
├── html/
│   ├── 01_data_eda.html
│   ├── 02_baseline_tuning.html
│   ├── 03_features_imbalance.html
│   ├── 04_segments_robustness.html
│   └── 05_final_validation_reporting.html
│
├── reports/
│   ├── Technical_Report.pdf
│   └── Business_Recommendations.pdf
│
├── figures/
│
└── results/
```

The final submission contains the **source dataset, five Python workflows, five executed HTML workflow outputs, supporting figures and numerical results, Technical Report, Business Recommendations, README, and environment requirements**.

## 8. How to Run

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Place the source dataset at:

```text
data/term-deposit-marketing-2020.csv
```

Run the workflows sequentially:

```bash
python scripts/01_data_eda.py
python scripts/02_baseline_tuning.py
python scripts/03_features_imbalance.py
python scripts/04_segments_robustness.py
python scripts/05_final_validation_reporting.py
```

The workflows save reusable numerical outputs under `results/` and report figures under `figures/`. The `.py` files provide the runnable source code, while the corresponding HTML files provide executed views of each analytical workflow and its outputs.

## 9. Outputs

The repository produces data-inspection and EDA summaries, class-balance and explicit-unknown analyses, Pearson and Cramér's V association outputs, baseline and leakage-test metrics, tuning search results and selected hyperparameters, feature-importance and feature-reduction results, class-imbalance comparisons, segment analysis, ordered robustness results, calibration outputs, call-depth and lift tables, supporting figures, the Technical Report PDF and the Business Recommendations PDF.

## 10. Limitations and Next Steps

The main limitation is that the historical dataset does not contain **verified years, campaign identifiers or trustworthy timestamps**. The ordered evaluation therefore tests robustness across changing campaign conditions, but it cannot prove true future-calendar performance.

The available predictors are also limited. The dataset does not include richer CRM history such as previous contact outcomes, time since prior contact or broader customer relationship information. In addition, `contact=unknown` affects approximately **31.9% of records**, making it an important data-quality issue for future campaigns.

The source data also contain no true customer identifier. A pseudo-profile grouping based on `age + job + marital + education + balance` produces very similar five-fold ROC-AUC to the row-stratified analysis, but this is only a sensitivity check and cannot prove customer-level independence.

The model scores are useful for **ranking**, but they should not yet be treated as exact subscription probabilities. Prospective campaign data would be needed for stronger calibration and for reliable expected-value or profitability calculations.

The next step is a **controlled live pilot** of the selected Logistic Regression prioritisation approach, with HistGradientBoosting retained as a challenger. The pilot should compare outcomes at equal calling effort, record reliable campaign and operational data, and measure incremental subscriptions, cost per acquisition, realised lift and customer outcomes.

Future iterations should focus on **better data before greater model complexity**. Reliable campaign dates, campaign IDs, previous contact history, previous campaign outcomes, clearer contact-channel information and richer CRM variables are likely to provide more value than repeatedly tuning the same historical feature set.
