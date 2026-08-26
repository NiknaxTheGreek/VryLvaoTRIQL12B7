# Term Deposit Customer Ranking

Machine-learning workflow for prioritising which already eligible customers should be called first about a term deposit.

## 1. Project overview

This project analyses 40,000 completed bank marketing calls and develops a ranking model for a capacity-constrained call centre. The target is term-deposit subscription (`y`). There are 2,896 subscribers and 37,104 non-subscribers, so the historical subscription rate is 7.24%.

Because an always-`no` classifier would already be 92.76% accurate, accuracy is not the main decision metric. The project focuses on ranking quality using ROC-AUC and PR-AUC, then evaluates business usefulness with period-local call-depth lift and subscriber capture.

The recommended pilot candidate is an **unweighted Logistic Regression model using all 12 valid pre-call predictors**, with HistGradientBoosting retained as a nonlinear challenger. Under the corrected ordered evaluation, Logistic Regression reaches weighted ROC-AUC 0.5663 and weighted PR-AUC 0.1126 across eight eligible later blocks. At 5% call depth it captures 228 subscribers versus about 123 expected under random selection, approximately 1.85× lift.

These are retrospective results. The model should be tested in a controlled live pilot before broader use.

## 2. Business objective

The model does not determine eligibility, approve or reject customers, set prices, guarantee subscription, or replace consent/contact-frequency rules. It ranks already eligible customers from highest to lowest priority:

**eligible customers → model score → ranked list → call highest-ranked customers first → stop at available capacity**

The operational question is therefore: **can the bank obtain more subscriptions from the same calling capacity?**

## 3. Dataset and prediction point

| Measure | Value |
|---|---:|
| Records | 40,000 |
| Candidate predictors | 13 |
| Valid pre-call predictors | 12 |
| Subscribers | 2,896 |
| Non-subscribers | 37,104 |
| Subscription rate | 7.24% |
| Conventional null values | 0 |
| Exact duplicate rows | 0 |

The 12 pre-call predictors are `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`, `contact`, `day`, `month` and `campaign`.

`duration` is the completed call length. It does not exist when the bank decides whom to call, so it is retained only as a diagnostic leakage benchmark and is excluded from every deployable/pre-call model, feature-selection ranking and pre-call association figure.

The file contains month labels but no verified year, campaign identifier or trustworthy timestamp. Ordered validation is therefore a robustness test across file-ordered campaign conditions, not proof of future-calendar performance.

## 4. Quantitative source of truth

The five executed notebooks are the quantitative source of truth. Numerical claims in the README and reports must trace to these notebook outputs; prose does not override a conflicting executed result.

1. `01_data_eda.ipynb` — data integrity, class balance, unknown categories, direct associations and ordered month-labelled exploration.
2. `02_baseline_tuning.ipynb` — controlled duration comparison, untuned baselines, four searches, selected hyperparameters and untouched-holdout results.
3. `03_features_imbalance.ipynb` — feature importance, transferred-parameter reduction, imbalance interventions and threshold analysis.
4. `04_segments_robustness.ipynb` — exploratory higher-conversion population and expanding-window ordered robustness.
5. `05_final_validation_reporting.ipynb` — calibration, period-local lift and final reporting summaries.

The repository is intentionally flat. No folder hierarchy, standalone `.py` scripts or HTML exports are required for the final published submission.

The notebooks contain stored executed outputs, but the current eight-file repository does not contain the original CSV or the intermediate `results/` and `figures/` files referenced by some reader cells. A fresh clone is therefore reviewable but is not claimed to be a fully self-contained clean-rerun environment.

## 5. Verified modelling sequence

1. **Untuned leakage baseline:** fit Logistic Regression and HistGradientBoosting using all 12 pre-call variables plus `duration`.
2. **Untuned pre-call baseline:** remove only `duration` and keep the same model defaults.
3. **Independent tuning:** tune LR pre-call, LR with duration, HGB pre-call and HGB with duration separately.
4. **Imbalance/operating-point study:** evaluate the selected tuned pre-call structures under no adjustment, class weighting, random oversampling, random undersampling, SMOTENC and nested threshold optimisation. Threshold optimisation changes the cutoff, not the ranking.
5. **Feature reduction:** transfer the selected full-model hyperparameters unchanged to Top-3, Top-4 and Top-5 predictor sets; do not retune reduced models.
6. **Exploratory higher-conversion population:** transfer the already-selected **tuned full-population pre-call model unchanged** into the subgroup. No subgroup-specific tuning is performed.
7. **Ordered robustness:** evaluate tuned, unweighted, all-12 pre-call LR and HGB across expanding later blocks without block-specific retuning.
8. **Downstream diagnostics:** evaluate calibration, call-depth lift and pseudo-profile sensitivity after the ranking analysis; these do not retroactively select the model.

### Why tuned transfer is canonical for the exploratory population

The subgroup analysis asks an operational question: **does the already-selected global model still rank usefully inside a historically higher-response population?** The canonical model rows are therefore the **Tuned transfer** rows, where the full-population tuned configuration is transferred unchanged.

Untuned full-population baselines are retained only as contextual comparators. They are not the canonical subgroup model because the intended pilot would not discard the globally selected configuration when entering the segment. Equally, the subgroup is **not retuned**: avoiding subgroup-specific hyperparameter selection reduces overfitting risk in a smaller, data-derived population and keeps the test aligned with how the global production candidate would actually be applied.

The exploratory rule is:

`contact = cellular AND (job = retired OR age >= 71 OR balance >= Q3)`

with `Q3(balance)=1319`. It contains 6,774 customers and 778 subscribers, an 11.49% historical response rate versus 7.24% overall.

Canonical transferred-model results from Notebook 04 are:

| Model | Configuration | ROC-AUC | PR-AUC | Brier |
|---|---|---:|---:|---:|
| Logistic Regression | Tuned transfer | 0.6928 | 0.2863 | 0.0928 |
| HistGradientBoosting | Tuned transfer | 0.7152 | 0.3090 | 0.0907 |

These results do **not** justify a separate subgroup model or automatic eligibility rule. The segment remains exploratory and suitable for pilot stratification/monitoring only.

## 6. Baseline configurations and tuning

### Logistic Regression

Untuned baseline:

`LogisticRegression(C=1, max_iter=100, tol=1e-4, random_state=42)`

with scikit-learn defaults supplying L2 regularisation, `class_weight=None` and the default solver. The identical baseline settings are used with and without `duration`.

Tuned search per predictor condition:

- `C = np.logspace(-3, 2, 15)`;
- penalty: L1 or L2;
- class weight: `None` or `balanced`;
- solver: `liblinear`;
- seed: 42.

This gives 60 candidates per LR condition and 120 LR candidates overall. `C=1` is the untuned baseline value but is **not** one of the 15 values in that logarithmic grid.

Selected LR configurations:

| Predictor condition | Selected configuration |
|---|---|
| Pre-call | L2, `C=8.4834289824`, unweighted |
| With duration | L2, `C=0.1389495494`, unweighted |

### HistGradientBoosting

Untuned baseline:

- `max_iter=100`
- `max_depth=None`
- `learning_rate=0.10`
- `max_leaf_nodes=31`
- `l2_regularization=0`
- `min_samples_leaf=20`
- `class_weight=None`
- `random_state=42`

Search space:

- `max_iter`: 50, 100, 150, 200, 300, 400
- `max_depth`: 3, 4, 5, 6, 8, `None`
- `learning_rate`: 0.01, 0.03, 0.05, 0.10, 0.20
- `max_leaf_nodes`: 15, 31, 63, 127
- `l2_regularization`: 0, 0.1, 0.5, 1, 2
- `min_samples_leaf`: 10, 20, 30, 50
- `class_weight`: `None`, `balanced`

The full HGB space has 28,800 combinations. One hundred candidates are sampled independently for each predictor condition with `ParameterSampler(..., n_iter=100, random_state=42)`.

The same winning structure appears independently in both HGB searches:

`max_iter=200, max_depth=8, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0, min_samples_leaf=20, class_weight=None`

### Tuning validation design

The data are split into 32,000 development rows and an untouched 8,000-row final holdout. Within development, LR tuning uses 24,000 training rows and 8,000 validation rows. HGB screening uses a reproducible stratified 10,000-row subsample of the tuning-train data against the same 8,000-row validation set, after which the selected HGB candidate is refit on the full development training data used by the workflow.

PR-AUC is the primary selection metric and ROC-AUC is the tie-break. Across the four independent searches there are **320 candidate configurations**: 60 + 60 LR and 100 + 100 HGB. This is a candidate count, not “320 CV fits”.

## 7. Phase 5 — reconciled performance results

The current executed Notebook 02 holdout table is canonical for the baseline/tuned comparison. Older stored audit rows that disagree with this current notebook generation are superseded rather than averaged or mixed.

| Model | Stage | Predictors | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| LR | Untuned | Pre-call | 0.9282 | 0.5758 | 0.0328 | 0.0621 | 0.7112 | 0.2417 |
| LR | Untuned | + duration | 0.9381 | 0.6556 | 0.3057 | 0.4170 | 0.9324 | 0.5250 |
| HGB | Untuned | Pre-call | 0.9280 | 0.5349 | 0.0397 | 0.0740 | 0.7341 | 0.2683 |
| HGB | Untuned | + duration | 0.9411 | 0.6399 | 0.4266 | 0.5119 | 0.9515 | 0.5852 |
| LR | Tuned | Pre-call | 0.9281 | 0.5588 | 0.0328 | 0.0620 | 0.7097 | 0.2403 |
| LR | Tuned | + duration | 0.9375 | 0.6537 | 0.2902 | 0.4019 | 0.9330 | 0.5232 |
| HGB | Tuned | Pre-call | 0.9275 | 0.4737 | 0.0155 | 0.0301 | 0.7299 | 0.2698 |
| HGB | Tuned | + duration | 0.9414 | 0.6580 | 0.3955 | 0.4941 | 0.9511 | 0.5815 |

The key result is not that tuning creates a large gain—it does not. The dominant retrospective jump comes from `duration`, which is unavailable at the pre-call decision point.

## 8. Phase 6 — duration policy and direct associations

### Pearson correlation

Valid pre-call numeric correlations with the target, ranked by absolute magnitude:

| Feature | Pearson r |
|---|---:|
| `campaign` | -0.0404 |
| `balance` | +0.0302 |
| `age` | -0.0203 |
| `day` | -0.0064 |

There is no fifth valid numeric pre-call predictor.

Diagnostic only: `duration = +0.4612`.

The large `duration` relationship is useful for demonstrating the size of outcome-proximate information, but timing—not merely the metric drop when it is removed—is why it is excluded from pre-call deployment.

### Cramér's V

| Feature | Cramér's V |
|---|---:|
| `month` | 0.1945 |
| `contact` | 0.0896 |
| `job` | 0.0589 |
| `marital` | 0.0579 |
| `housing` | 0.0541 |

These are descriptive univariate associations, not causal effects and not the rule used to create reduced feature sets.

For publication, the Pearson visual should contain only the four valid pre-call numeric predictors; `duration` belongs in a separately labelled diagnostic comparison. The Cramér's V visual should use the five values above. Confusion-matrix-style association graphics are not appropriate for these quantities.

## 9. Phase 7 — feature importance and reduction

Three forms of evidence are kept separate because they answer different questions.

### Logistic Regression grouped coefficient magnitude

| Feature | Importance |
|---|---:|
| `month` | 0.9579 |
| `contact` | 0.4748 |
| `loan` | 0.3444 |
| `default` | 0.3444 |
| `housing` | 0.3444 |

`loan`, `default` and `housing` are tied at the displayed precision.

### HGB permutation importance

Measured on a stratified 30% holdout using decrease in PR-AUC:

| Feature | Importance |
|---|---:|
| `month` | 0.1478 |
| `contact` | 0.1051 |
| `day` | 0.0559 |
| `housing` | 0.0348 |
| `age` | 0.0144 |

The **HGB permutation order directly defines the tested nested subsets**:

- Top 3: `month`, `contact`, `day`
- Top 4: `month`, `contact`, `day`, `housing`
- Top 5: `month`, `contact`, `day`, `housing`, `age`
- All 12: every valid pre-call predictor

Pearson, Cramér's V and LR coefficients support interpretation only. The project does **not** claim that a formal combined-evidence ranking generated the feature sets.

Selected full-model hyperparameters are transferred unchanged to each reduced set. Reduced models are not retuned, so the comparison isolates predictor removal rather than mixing removal with a new search.

| Model | Metric | Top 3 | Top 4 | Top 5 | All 12 |
|---|---|---:|---:|---:|---:|
| LR | ROC-AUC | 0.6673 | 0.6728 | 0.6738 | **0.6848** |
| LR | PR-AUC | 0.1953 | 0.2122 | 0.2103 | **0.2154** |
| HGB | ROC-AUC | 0.7036 | 0.7062 | 0.7072 | **0.7153** |
| HGB | PR-AUC | 0.2439 | **0.2463** | 0.2446 | 0.2447 |

All 12 predictors give the best ROC-AUC for both families and the best LR PR-AUC. HGB Top-4 has a very small PR-AUC edge over HGB All-12 (0.2463 vs 0.2447), but that isolated difference is not enough to justify discarding the broader feature set. The all-12 models remain the primary candidates for ordered robustness, where stability across later campaign conditions matters more than a small shuffled-holdout difference.

## 10. Ordered robustness

The expanding-window analysis uses tuned, unweighted, all-12 pre-call models without block-specific retuning. Eight later blocks meet the predeclared eligibility gate.

| Metric | Logistic Regression | HistGradientBoosting |
|---|---:|---:|
| Eligible periods | 8 | 8 |
| Test rows | 30,526 | 30,526 |
| Test positives | 2,457 | 2,457 |
| Weighted ROC-AUC | **0.5663** | 0.5530 |
| Weighted PR-AUC | **0.1126** | 0.0968 |

The reversal relative to shuffled historical evaluation is central to the final model choice: the nonlinear challenger looks stronger in mixed historical data, but Logistic Regression is more reliable in the later ordered blocks.

## 11. Class imbalance and threshold policy

Class weighting and resampling mainly change the precision/recall trade-off. They do not provide a compelling improvement in score ordering. SMOTENC degrades ranking performance for both model families. Nested threshold optimisation gives the strongest hard-classification F1 while leaving ROC-AUC/PR-AUC unchanged because it changes the cutoff rather than the ranking.

For a capacity-constrained call centre, the preferred operating policy is therefore to keep the ranking model unweighted and choose a call depth based on available capacity and pilot economics.

## 12. Final recommendation

Use the unweighted all-12-feature Logistic Regression as the pilot champion and HGB as a challenger. Apply normal eligibility/consent rules before scoring, rank eligible customers by model score, call from the top of the list to the available capacity, and measure realised subscription rate, lift, complaints, opt-outs and segment composition against a control arm.

The exploratory high-response population can be used for stratification and monitoring, but not as an automatic exclusion rule or separately tuned production model.

## 13. Repository structure

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

This flat structure is intentional for the final portfolio submission.