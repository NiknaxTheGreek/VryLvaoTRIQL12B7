# Term Deposit Customer Ranking

Machine-learning workflow for prioritising which already eligible customers should be called first about a term deposit.

## Executive Summary

This project asks a practical business question: **can a bank obtain more term-deposit subscriptions from the same calling capacity by ranking eligible customers before the call?**

The source data contain **40,000 historical calls**, of which **2,896 subscribed (7.24%)**. The output is therefore not primarily a yes/no classifier. It is a **pre-call ranking score** used to order eligible customers so that limited calling capacity is spent on the most promising records first.

Two deliberately different model families are compared: **Logistic Regression (LR)** as the transparent linear model and **HistGradientBoosting (HGB)** as the nonlinear challenger. `duration` is known only after a call finishes, so it is retained solely as a leakage diagnostic and excluded from every deployable model.

On the untouched 20% holdout, tuned HGB ranks customers better than tuned LR: **PR-AUC / ROC-AUC = 0.272820 / 0.733930** for HGB versus **0.240265 / 0.709681** for LR. Under expanding-window ordered validation, however, both models weaken substantially and LR is more robust: **0.112626 / 0.566290** versus **0.096333 / 0.551600** for HGB.

The business-facing result is period-local lift. Across the eight eligible forward blocks, LR captured **228 subscribers in the top-ranked 5%** of calls versus about **123 expected under random calling at the same volume**, corresponding to **1.853857× lift**. LR lift remains **1.606389× at 10%** and **1.401474× at 20%** call depth.

The final recommendation is therefore **unweighted Logistic Regression using all 12 valid pre-call features for a controlled pilot**, with HGB retained as a challenger. Raw model scores should be treated as **ranking signals, not literal subscription probabilities**, until prospective recalibration is performed on new campaign data.

## Business Objective

The intended operating flow is:

**eligible customers → pre-call score → ranked list → call highest-ranked customers first → stop when capacity is exhausted**

The model is not intended to determine product eligibility, approve or reject customers, set prices or credit limits, guarantee who will subscribe, or replace consent/contact-frequency rules.

Because the target rate is only 7.24%, the project evaluates **accuracy, precision, recall, F1, PR-AUC and ROC-AUC**, while the deployment decision focuses most strongly on ranking quality, ordered robustness and call-depth lift.

## Dataset and Prediction Timing

| Measure | Value |
|---|---:|
| Historical calls | 40,000 |
| Subscribers | 2,896 |
| Non-subscribers | 37,104 |
| Subscription rate | 7.24% |
| Original predictor fields | 13 |
| Valid pre-call predictors | 12 |
| Exact duplicate source rows | 0 |
| Conventional null values | 0 |

The 12 deployable predictors are `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`, `contact`, `day`, `month` and `campaign`.

`duration` is post-call. Its Pearson correlation with subscription is **0.461169**, and adding it produces a large retrospective performance jump, but that is precisely why it is treated as leakage evidence rather than a deployable feature.

The data contain month labels but no verified year, campaign identifier or trustworthy timestamp. Consecutive changes in month label are therefore treated as **ordered month-labelled blocks**, not proven calendar periods.

## Analytical Workflow

The first five executed notebooks form the retrospective analytical source of truth. A sixth executed notebook implements the prospective pilot preparation without inventing live outcomes:

1. `01_data_eda.ipynb` — data quality, class balance, associations, ordered blocks and descriptive higher-conversion population.
2. `02_baseline_tuning.ipynb` — baseline/default LR/HGB, leakage comparison, deterministic tuning and untouched holdout evaluation.
3. `03_features_imbalance.ipynb` — feature importance, feature reduction, imbalance strategies and threshold diagnostics.
4. `04_segments_robustness.ipynb` — exploratory population validation, shuffled validation, pseudo-profile grouped sensitivity and expanding-window ordered validation.
5. `05_final_validation_reporting.ipynb` — period-local lift, calibration diagnostics and final reporting summaries.
6. `06_controlled_pilot.ipynb` — freezes the final LR/HGB pilot specifications, validates live-input schema, enforces the `duration` leakage guard, provides deterministic randomisation/equal-capacity assignment, sample-size planning, outcome analysis and economics. Without a genuinely new eligible-customer file it executes to a **READY / WAITING FOR LIVE INPUT** state and reports no fabricated pilot outcome.

The five retrospective analytical notebooks use `SEED=42`; the pilot notebook uses the frozen model seed plus a separate fixed pilot-randomisation seed. All contain explanatory Markdown, use root-level inputs and do **not** depend on generated folders or cached result files.

## Is the selection of Logistic Regression and HistGradientBoosting justified?

Yes, for this project. This is **not an exhaustive benchmark of every classifier**. The pair creates a focused comparison between a simple, interpretable linear ranking model and a nonlinear tree-based challenger capable of learning interactions and nonlinear effects.

HGB is stronger in conventional shuffled historical evaluation. LR becomes the preferred pilot model only after ordered robustness, historical lift, interpretability and monitoring simplicity are considered together.

## Validation Design

The 40,000 rows are split deterministically with `SEED=42`:

- development set: **32,000** rows;
- untouched final holdout: **8,000** rows;
- tuning train inside development: **24,000** rows;
- tuning validation inside development: **8,000** rows;
- HGB search uses a reproducible **10,000-row stratified tuning subsample**.

Hyperparameter selection uses one fixed train/validation split, **not K-fold hyperparameter tuning**. PR-AUC is the primary selection objective and ROC-AUC is the deterministic tie-breaker.

## Hyperparameters and Tuning Specification

This section separates three things that should not be confused: **what the hyperparameters mean and what values were searched; the common baseline/default model settings; and the four separately selected tuned winners**.

### Hyperparameter descriptions and search values

| Model | Hyperparameter | What it controls | Values searched |
|---|---|---|---|
| LR | `C` | Inverse regularisation strength; larger values mean weaker regularisation | 15 log-spaced values from 0.001 to 100: 0.001, 0.002276, 0.005179, 0.011788, 0.026827, 0.061054, 0.138950, 0.316228, 0.719686, 1.637894, 3.727594, 8.483429, 19.306977, 43.939706, 100 |
| LR | `penalty` | Coefficient regularisation type | L1, L2 |
| LR | `class_weight` | Relative class weighting | None, balanced |
| HGB | `max_iter` | Maximum boosting iterations | 50, 100, 150, 200, 300, 400 |
| HGB | `max_depth` | Maximum tree depth | 3, 4, 5, 6, 8, None |
| HGB | `learning_rate` | Contribution of each boosting iteration | 0.01, 0.03, 0.05, 0.10, 0.20 |
| HGB | `max_leaf_nodes` | Maximum terminal leaves per tree | 15, 31, 63, 127 |
| HGB | `l2_regularization` | L2 penalty on leaf values | 0, 0.1, 0.5, 1, 2 |
| HGB | `min_samples_leaf` | Minimum observations required in a leaf | 10, 20, 30, 50 |
| HGB | `class_weight` | Relative class weighting | None, balanced |

For LR tuning, `solver=liblinear`, `max_iter=3000` and `random_state=42` are fixed while the three parameters above are searched. LR exhaustively evaluates **60 configurations per feature condition**. HGB has **28,800 possible combinations** and samples the same **100 candidates per feature condition** with `ParameterSampler(..., random_state=42)`; `early_stopping=True` and `random_state=42` are fixed during HGB tuning. Across both models and both feature conditions, exactly **320 candidate configurations** are evaluated.

### Baseline/default model specifications

The baseline comparison is deliberately like-for-like. **Within each model family, the pre-call baseline and the + duration baseline use exactly the same model hyperparameters. Only the feature set changes.** There are therefore two baseline model specifications, not four.

| Baseline model | Hyperparameters used for both Pre-call and + duration |
|---|---|
| LR | `C=1.0`, L2 penalty, `solver=lbfgs`, `max_iter=1000`, `tol=0.0001`, `class_weight=None`, `random_state=42` |
| HGB | `loss=log_loss`, `learning_rate=0.1`, `max_iter=100`, `max_leaf_nodes=31`, `max_depth=None`, `min_samples_leaf=20`, `l2_regularization=0`, `max_bins=255`, `early_stopping=auto`, `class_weight=None`, `random_state=42` |

Thus the baseline leakage test is:

- LR baseline settings + 12 pre-call features **versus the same LR settings** + `duration`;
- HGB baseline settings + 12 pre-call features **versus the same HGB settings** + `duration`.

### Tuned model specifications

Tuning is then run **separately for each model × feature condition**, so the four tuned winners are allowed to differ.

| Tuned configuration | Selected hyperparameters |
|---|---|
| LR pre-call | L2, `C=8.483428982440726`, `class_weight=None`; fixed tuning settings `solver=liblinear`, `max_iter=3000`, `random_state=42` |
| LR + duration | L1, `C=0.0610540229658533`, `class_weight=None`; fixed tuning settings `solver=liblinear`, `max_iter=3000`, `random_state=42` |
| HGB pre-call | `max_iter=200`, `max_depth=8`, `learning_rate=0.05`, `max_leaf_nodes=15`, `l2_regularization=0`, `min_samples_leaf=20`, `class_weight=None`, `early_stopping=True`, `random_state=42` |
| HGB + duration | `max_iter=200`, `max_depth=3`, `learning_rate=0.05`, `max_leaf_nodes=127`, `l2_regularization=2`, `min_samples_leaf=10`, `class_weight=balanced`, `early_stopping=True`, `random_state=42` |

## Untouched Final-Holdout Results

| Stage | Model | Features | Accuracy | F1 | PR-AUC | ROC-AUC |
|---|---|---|---:|---:|---:|---:|
| Baseline/default | LR | Pre-call | 0.928250 | 0.062092 | 0.241676 | 0.711156 |
| Baseline/default | LR | + duration | 0.938125 | 0.416961 | 0.525049 | 0.932379 |
| Baseline/default | HGB | Pre-call | 0.928000 | 0.073955 | 0.268264 | 0.734102 |
| Baseline/default | HGB | + duration | 0.941125 | 0.511917 | 0.585190 | 0.951528 |
| Tuned | LR | Pre-call | 0.928125 | 0.061990 | 0.240265 | 0.709681 |
| Tuned | LR | + duration | 0.936500 | 0.381995 | 0.520693 | 0.933188 |
| Tuned | HGB | Pre-call | 0.928875 | 0.083736 | 0.272820 | 0.733930 |
| Tuned | HGB | + duration | 0.866000 | 0.490978 | 0.549825 | 0.943115 |

The with-duration results are diagnostic only. They demonstrate the size of the leakage effect and are not candidates for deployment. Tuning changes pre-call ranking only modestly: LR PR-AUC moves from **0.241676 to 0.240265**, while HGB moves from **0.268264 to 0.272820**.

## Feature Reduction and Class Imbalance

Feature reduction is ordered **only by held-out HGB permutation importance measured as PR-AUC decrease**. Direct association, grouped LR coefficient magnitude and HGB permutation importance are kept separate because they answer different questions. The corresponding LR and HGB feature-importance bar plots are generated directly in Notebook 03.

The five highest HGB permutation features are `month`, `contact`, `day`, `housing` and `age`. HGB Top-4 reaches PR-AUC **0.274531**, slightly above All-12 at **0.272820**, but its ROC-AUC falls to **0.719196** from **0.733930**. That exploratory result is not sufficient to declare the four-feature subset universally superior, so the final specification retains all 12 valid pre-call predictors.

For imbalance handling, weighting and resampling mainly shift the precision/recall trade-off. HGB oversampling reaches PR-AUC **0.274559**, only **+0.001739** above unadjusted HGB while materially reducing accuracy and precision. SMOTENC degrades ranking for both model families.

Threshold tuning gives the strongest hard-classification F1 — **0.287171 for LR** and **0.353059 for HGB** — but it does not improve score ordering. The operational recommendation therefore remains an **unweighted ranking model**, with call depth chosen from capacity and economics rather than a fixed classification threshold.

## Robustness: Shuffled vs Ordered

| Validation | LR PR-AUC | LR ROC-AUC | HGB PR-AUC | HGB ROC-AUC |
|---|---:|---:|---:|---:|
| Shuffled 5-fold | 0.215433 | 0.684767 | **0.245333** | **0.717461** |
| Pseudo-profile grouped sensitivity | 0.215690 | 0.685003 | **0.248093** | **0.716737** |
| Ordered expanding-window | **0.112626** | **0.566290** | 0.096333 | 0.551600 |

The pseudo-profile analysis groups exact combinations of `age + job + marital + education + balance`. It is a sensitivity check only; because no true customer identifier exists, it **cannot prove customer-level independence**.

Eight ordered blocks meet the eligibility gate, covering **30,526 test rows and 2,457 subscribers**. The large shuffled-to-ordered drop is the central robustness warning and is the main reason the project does not recommend unrestricted production deployment from retrospective results alone.

## Calibration and Period-Local Lift

No sigmoid or isotonic recalibration model is fitted. Period-local quantile bins, Brier score and expected calibration error (ECE) are used only to diagnose score reliability.

| Model | Weighted Brier | Weighted ECE |
|---|---:|---:|
| LR | 0.074776 | 0.044442 |
| HGB | 0.074181 | 0.035101 |

### Historical lift by call depth

| Call depth | LR lift | HGB lift |
|---:|---:|---:|
| 5% | **1.853857×** | 1.284690× |
| 10% | **1.606389×** | 1.293245× |
| 20% | **1.401474×** | 1.248918× |
| 50% | **1.187549×** | 1.133829× |

At 5% depth, LR makes **1,528 calls** and captures **228 subscribers**, versus approximately **122.99 expected under random selection** at the same volume. These call depths are sensitivity points, not prescribed quotas.

## Controlled Prospective Pilot Implementation

The recommended next step has been implemented as an executable preparation layer rather than left as prose only. `06_controlled_pilot.ipynb` and `PILOT_PROTOCOL.md` freeze the selected all-12 LR model, retain HGB as an optional challenger, prohibit `duration` and outcome fields at assignment time, validate a genuinely new eligible-customer snapshot, randomise policy pools, enforce equal call capacity, define subscriptions per 1,000 assigned calls as the primary outcome, provide confidence-interval and sample-size functions, and calculate economics once the bank supplies real cost and contribution inputs.

The pilot-preparation notebook has been executed successfully in the pinned environment. Because no genuinely new bank customer snapshot or prospective outcomes are present in the repository, it correctly terminates in a **READY / WAITING FOR LIVE INPUT** state. This adds implementation readiness but **does not change or manufacture any retrospective model result**.

## Final Recommendation

Use **unweighted all-12 Logistic Regression** as the primary candidate in a controlled live pilot and keep HGB as a challenger.

The pilot should compare equal-capacity model-ranked calling against a business-as-usual or random comparator. The primary operational outcome should be **subscriptions per 1,000 assigned calls**, with contactability, opt-outs, complaints, repeat contacts, customer mix and agent time monitored where available.

Production deployment should wait until the bank has prospective evidence on new campaigns, real call-cost and contribution economics, and recalibration evidence if probability-based decisions are required.

## Repository Contents

The repository is intentionally **flat: no folders and no HTML exports**.

```text
01_data_eda.ipynb
02_baseline_tuning.ipynb
03_features_imbalance.ipynb
04_segments_robustness.ipynb
05_final_validation_reporting.ipynb
06_controlled_pilot.ipynb
Business_Recommendations.pdf
PILOT_PROTOCOL.md
README.md
README_COMPREHENSIVE.md
Technical_Report.pdf
requirements.txt
term-deposit-marketing-2020.csv
term-deposit-marketing-2020-labelled.csv
```

- `term-deposit-marketing-2020.csv` — original source dataset.
- `term-deposit-marketing-2020-labelled.csv` — working copy with the derived `y_binary` helper; each notebook drops/recreates that helper before analysis.
- `Technical_Report.pdf` — detailed technical methodology, results and limitations.
- `Business_Recommendations.pdf` — concise decision-oriented interpretation and controlled-pilot recommendation.
- `README_COMPREHENSIVE.md` — extended methodology, result interpretation and notebook traceability.
- `PILOT_PROTOCOL.md` — frozen prospective experimental protocol, operational data contract, guardrails and go/no-go logic.
- `06_controlled_pilot.ipynb` — executed pilot-preparation implementation; no live result is fabricated when fresh bank data are absent.

## Reproducibility

Install the pinned environment and execute the notebooks in order:

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace 01_data_eda.ipynb --ExecutePreprocessor.timeout=900
jupyter nbconvert --to notebook --execute --inplace 02_baseline_tuning.ipynb --ExecutePreprocessor.timeout=900
jupyter nbconvert --to notebook --execute --inplace 03_features_imbalance.ipynb --ExecutePreprocessor.timeout=900
jupyter nbconvert --to notebook --execute --inplace 04_segments_robustness.ipynb --ExecutePreprocessor.timeout=900
jupyter nbconvert --to notebook --execute --inplace 05_final_validation_reporting.ipynb --ExecutePreprocessor.timeout=900
jupyter nbconvert --to notebook --execute --inplace 06_controlled_pilot.ipynb --ExecutePreprocessor.timeout=900
```

The commands execute the five retrospective analytical notebooks and then the pilot-preparation notebook. In the absence of `pilot_eligible_customers.csv`, Notebook 06 deliberately reports readiness/waiting status rather than producing a fake prospective result. No generated folders or intermediate result files are required.

## Key Limitation

The strongest limitation is temporal robustness. The ordered blocks preserve source order but are **not verified campaign timestamps**, and performance is substantially weaker than under shuffled validation. The repository now includes the complete controlled-pilot preparation protocol and executable implementation, but actual prospective effectiveness, economics and recalibration remain unknowable until genuinely new bank campaign data and outcomes are supplied. The project therefore supports **controlled pilot execution**, not an unconditional claim of production readiness.