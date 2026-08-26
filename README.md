# Term Deposit Customer Ranking

Machine-learning workflow for prioritising which already eligible customers should be called first about a term deposit.

## Executive Summary

This project asks a practical business question: **can a bank obtain more term-deposit subscriptions from the same calling capacity by ranking eligible customers before the call?**

The dataset contains 40,000 historical marketing calls. Only 2,896 customers subscribed, giving a 7.24% subscription rate. Because 92.76% of records are non-subscriptions, accuracy alone is not sufficient. The useful output is a **ranked call list** that puts more promising customers near the top.

Two deliberately different model families are compared: **Logistic Regression (LR)** as the simple, interpretable linear model and **HistGradientBoosting (HGB)** as the nonlinear challenger.

The analysis first checks which variables are genuinely available before a call. `duration` is known only after the call has finished, so it is retained solely as a leakage diagnostic and excluded from every deployable downstream model.

On the untouched 20% holdout, tuned HGB ranks customers better than tuned LR: HGB PR-AUC / ROC-AUC = **0.272820 / 0.733930**, versus LR = **0.240265 / 0.709681**. However, both models weaken substantially when tested in expanding-window source order across later month-labelled blocks. LR is more robust there: weighted ordered PR-AUC / ROC-AUC = **0.112626 / 0.566290**, compared with HGB = **0.096333 / 0.551600**.

The business-facing result is period-local lift. Across the eight eligible forward blocks, LR captured **228 subscribers in the top-ranked 5%** of calls versus about **123 expected from random selection at the same call volume**, corresponding to **1.853857× lift**. At 10% and 20% call depth, LR lift is **1.606389×** and **1.401474×**.

The recommended controlled-pilot model is therefore **unweighted Logistic Regression using all 12 valid pre-call features**, with HGB retained as a challenger. Scores should be treated as **ranking signals rather than literal subscription probabilities** until prospective recalibration is performed on new campaign data.

---

## 1. Business Objective

The intended workflow is:

**eligible customers → pre-call model score → ranked list → call highest-ranked customers first → stop when capacity is exhausted**

The model is not intended to determine product eligibility, approve or reject customers, set prices or credit limits, guarantee who will subscribe, or replace consent/contact-frequency rules.

The central question is:

> Can the bank obtain more subscriptions from the same calling effort?

That makes ranking quality and call-depth lift more relevant than a single fixed yes/no threshold.

---

## 2. Dataset and Prediction Timing

| Measure | Value |
|---|---:|
| Historical call records | 40,000 |
| Subscribers | 2,896 |
| Non-subscribers | 37,104 |
| Subscription rate | 7.24% |
| Original predictors | 13 |
| Deployable pre-call predictors | 12 |
| Exact duplicate source rows | 0 |
| Conventional null values | 0 |

The 12 deployable pre-call predictors are `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`, `contact`, `day`, `month` and `campaign`.

`duration` is excluded from deployment because it records completed call duration and therefore does not exist at the moment the bank decides whom to call.

Explicit string categories of `unknown` are present and are not conventional nulls:

| Feature | `unknown` records |
|---|---:|
| `contact` | 12,765 |
| `education` | 1,531 |
| `job` | 235 |

The dataset contains month labels but no verified year, campaign identifier or trustworthy timestamp. Consecutive changes in month label are therefore treated as **ordered month-labelled blocks**, not proven calendar periods.

---

## 3. Analytical Workflow

The executed notebooks follow this sequence:

1. **Data audit and EDA** — class balance, data quality, direct associations and ordered blocks.
2. **Leakage experiment** — compare default LR and HGB with and without `duration`.
3. **Hyperparameter selection** — tune LR and HGB separately for pre-call and with-duration conditions.
4. **Class-imbalance experiments** — compare weighting, resampling, SMOTENC and threshold tuning on tuned pre-call model structures.
5. **Feature reduction** — compare Top-3, Top-4, Top-5 and all 12 predictors without subset-specific retuning.
6. **Exploratory higher-conversion population** — evaluate a predefined population using default model structures only, with no subgroup retuning.
7. **Robustness validation** — shuffled, pseudo-profile grouped sensitivity and expanding-window ordered validation.
8. **Calibration and lift** — assess score reliability and period-local call-depth lift separately.

All notebooks use `SEED=42`, contain explanatory Markdown cells, read the root-level labelled dataset directly, and do not depend on generated `data/`, `results/`, `figures/`, `scripts/` or `html/` folders.

---

## 4. Is the selection of Logistic Regression and HistGradientBoosting justified?

Yes, for the purpose of this project.

This is not an exhaustive benchmark of every classifier. The two selected model families create a focused comparison between a transparent linear ranking model and a nonlinear tree-based challenger capable of capturing interactions and nonlinear effects without extensive manual feature engineering.

HGB does improve conventional shuffled ranking, but LR becomes the preferred pilot model once ordered robustness, historical lift, interpretability and monitoring simplicity are considered together.

---

## 5. Baselines, Leakage and Hyperparameter Selection

### Validation design

The 40,000 rows are split deterministically using `SEED=42`:

- development set: **32,000 rows**;
- untouched final holdout: **8,000 rows**;
- tuning train within development: **24,000 rows**;
- tuning validation within development: **8,000 rows**;
- HGB search uses a reproducible **10,000-row stratified tuning subsample**.

This is a fixed train/validation selection design, **not K-fold hyperparameter tuning**.

### Search spaces

| Model | Search |
|---|---|
| Logistic Regression | 15 log-spaced `C` values from 0.001 to 100 × L1/L2 × none/balanced class weight = **60 configurations per feature condition** |
| HistGradientBoosting | iterations 50–400; depth 3/4/5/6/8/None; learning rate 0.01–0.20; leaves 15–127; L2 0–2; min leaf 10–50; none/balanced class weight = **28,800 possible combinations**, with **100 reproducibly sampled candidates per feature condition** |

Across LR pre-call, LR with duration, HGB pre-call and HGB with duration, exactly **320 candidate configurations** are evaluated. Selection uses **PR-AUC** as the primary objective and **ROC-AUC** as tie-breaker.

### Selected hyperparameters

| Configuration | Selected parameters |
|---|---|
| LR pre-call | L2, `C=8.4834289824`, unweighted |
| LR with duration | L1, `C=0.0610540230`, unweighted |
| HGB pre-call | 200 iterations, depth 8, learning rate 0.05, 15 leaves, L2=0, min leaf 20, unweighted, early stopping |
| HGB with duration | 200 iterations, depth 3, learning rate 0.05, 127 leaves, L2=2, min leaf 10, **balanced**, early stopping |

### Untouched final-holdout performance

| Stage | Model | Feature set | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Default | LR | Pre-call | 0.928250 | 0.575758 | 0.032815 | 0.062092 | 0.241676 | 0.711156 |
| Tuned | LR | Pre-call | 0.928125 | 0.558824 | 0.032815 | 0.061990 | 0.240265 | 0.709681 |
| Default | HGB | Pre-call | 0.928000 | 0.534884 | 0.039724 | 0.073955 | 0.268264 | 0.734102 |
| Tuned | HGB | Pre-call | 0.928875 | 0.619048 | 0.044905 | 0.083736 | 0.272820 | 0.733930 |
| Default | LR | With duration | 0.938125 | 0.655556 | 0.305699 | 0.416961 | 0.525049 | 0.932379 |
| Tuned | LR | With duration | 0.936500 | 0.646091 | 0.271157 | 0.381995 | 0.520693 | 0.933188 |
| Default | HGB | With duration | 0.941125 | 0.639896 | 0.426598 | 0.511917 | 0.585190 | 0.951528 |
| Tuned | HGB | With duration | 0.866000 | 0.338572 | 0.892919 | 0.490978 | 0.549825 | 0.943115 |

Including `duration` creates a very large retrospective performance jump. That does **not** make it deployable; it confirms that a post-call variable contains strong information unavailable at the decision point.

Tuning gives little ranking improvement on the pre-call holdout: LR changes from PR-AUC **0.241676 → 0.240265**, while HGB changes from **0.268264 → 0.272820**.

---

## 6. Feature Importance and Feature Reduction

Direct association, LR coefficient magnitude and HGB permutation importance answer different questions and are not merged into one “combined” ranking.

The feature-reduction order is defined **only by held-out HGB permutation importance measured using PR-AUC decrease**.

| Rank | Feature | HGB permutation importance |
|---:|---|---:|
| 1 | `month` | 0.168555 |
| 2 | `contact` | 0.119955 |
| 3 | `day` | 0.081196 |
| 4 | `housing` | 0.025086 |
| 5 | `age` | 0.011867 |

| Features | LR PR-AUC | LR ROC-AUC | HGB PR-AUC | HGB ROC-AUC |
|---|---:|---:|---:|---:|
| Top 3 | 0.214884 | 0.691444 | 0.265716 | 0.713004 |
| Top 4 | 0.235485 | 0.698631 | **0.274531** | 0.719196 |
| Top 5 | 0.235275 | 0.699099 | 0.272162 | 0.720062 |
| All 12 | **0.240265** | **0.709681** | 0.272820 | **0.733930** |

HGB Top-4 has a slightly higher PR-AUC than All-12 on this exploratory holdout comparison, but materially lower ROC-AUC. The evidence does not justify declaring the four-feature subset universally superior. The final pilot specification retains all 12 valid pre-call predictors.

---

## 7. Class Imbalance

Six strategies are compared: no adjustment, balanced class weighting, random oversampling, random undersampling, SMOTENC and threshold tuning.

Weighting and resampling mainly alter the precision/recall trade-off. HGB oversampling gives PR-AUC **0.274559**, only +0.001739 above the unadjusted HGB value of 0.272820, while accuracy and precision fall substantially. SMOTENC degrades ranking for both model families.

Threshold tuning gives the strongest hard-classification F1 while leaving score ordering unchanged:

| Model | Selected threshold | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|
| LR | 0.141 | 0.287171 | 0.240265 | 0.709681 |
| HGB | 0.156 | 0.353059 | 0.272820 | 0.733930 |

The operational recommendation remains an **unweighted ranking model**. Call depth should be selected from capacity and economics rather than from a fixed classification threshold.

---

## 8. Exploratory Higher-Conversion Population

The descriptive rule is:

`contact = cellular AND (job = retired OR age >= 71 OR balance >= Q3)`

Using the full dataset for description, `balance` Q3 = **1,319** and the population contains **6,774 customers**, **778 subscribers**, and **11.485% conversion** versus 7.24% overall — about **58.6% descriptive relative uplift**.

During validation, Q3 is learned inside each training fold so the threshold does not leak information from validation rows. No subgroup-specific hyperparameter tuning is performed.

| Model | PR-AUC | ROC-AUC |
|---|---:|---:|
| LR default | 0.286349 | 0.692504 |
| HGB default | 0.303476 | 0.694120 |

This is exploratory business context, not evidence that the bank should deploy a separate demographic targeting model.

---

## 9. Shuffled, Grouped and Ordered Robustness

### Shuffled five-fold validation

| Model | PR-AUC | ROC-AUC |
|---|---:|---:|
| LR | 0.215433 | 0.684767 |
| HGB | 0.245333 | 0.717461 |

### Pseudo-profile grouped sensitivity

Because there is no true customer identifier, exact combinations of `age + job + marital + education + balance` are used as pseudo-groups.

| Model | PR-AUC | ROC-AUC |
|---|---:|---:|
| LR | 0.215690 | 0.685003 |
| HGB | 0.248093 | 0.716737 |

The similarity to shuffled results is reassuring, but this **cannot prove true customer-level independence** because the grouping key is only a pseudo-profile.

### Expanding-window ordered validation

A later block is eligible only if it has at least 5,000 earlier training rows, 100 positive test outcomes and 100 negative test outcomes. Eight blocks pass the gate, covering **30,526 test rows and 2,457 subscribers**.

| Model | Weighted PR-AUC | Weighted ROC-AUC | Weighted Brier |
|---|---:|---:|---:|
| LR | **0.112626** | **0.566290** | 0.074776 |
| HGB | 0.096333 | 0.551600 | **0.074181** |

The large shuffled-to-ordered drop is the central robustness warning. HGB wins conventional shuffled validation, but LR performs better under the stricter ordered test.

---

## 10. Calibration and Call-Depth Lift

Calibration is assessed period by period using local score-quantile bins, Brier score and expected calibration error (ECE). **No sigmoid or isotonic recalibration model is fitted.**

| Model | Weighted mean score | Weighted Brier | Weighted ECE |
|---|---:|---:|---:|
| LR | 0.075410 | 0.074776 | 0.044442 |
| HGB | 0.048578 | 0.074181 | 0.035101 |

These raw outputs should therefore be interpreted as **ranking signals**, not exact subscription probabilities.

### Period-local lift

| Call depth | LR lift | HGB lift |
|---:|---:|---:|
| 5% | **1.853857** | 1.284690 |
| 10% | **1.606389** | 1.293245 |
| 20% | **1.401474** | 1.248918 |
| 50% | **1.187549** | 1.133829 |

At 5% depth, LR makes **1,528 calls** and captures **228 subscribers**, versus approximately **122.99 expected under random calling at the same volume**.

At 20% depth, LR makes **6,108 calls** and captures **689 subscribers**, versus approximately **491.63 expected randomly**.

These depths are sensitivity points, not mandatory quotas. The bank should choose operational call depth using capacity and real economics.

---

## 11. Final Recommendation

The recommended controlled-pilot model is **unweighted Logistic Regression using all 12 pre-call features**. HGB should remain as a **challenger**.

The reason is not that LR wins every retrospective metric. HGB is stronger under shuffled historical evaluation. LR is selected because it is more robust under ordered evaluation, produces stronger historical lift at the tested call depths, and is simpler to interpret and monitor.

The next step is a controlled live pilot at equal calling effort against the most appropriate benchmark: the bank’s existing process if it is clearly defined, otherwise a random eligible-customer comparator.

The primary business outcome should be **subscriptions per 1,000 assigned calls**. Secondary monitoring should include contactability, complaints/opt-outs, repeat contacts, customer mix and agent time where available.

Broader deployment should occur only if the pilot shows more subscriptions for the same calling effort **and** positive business value without unacceptable operational or customer outcomes.

---

## 12. Limitations

The main limitations are the lack of verified year/campaign identifiers, no true customer identifier, limited pre-call CRM history, `contact=unknown` in 12,765 records, the large shuffled-to-ordered performance drop, and absence of prospective probability calibration or bank-supplied cost/value inputs.

Future iterations should prioritise **better data before greater model complexity**: verified dates and campaign IDs, previous-contact history, richer CRM variables, contact outcomes, time since previous contact and actual business cost/value fields.

---

## 13. Repository Structure

The repository is intentionally **flat: no folders and no HTML exports**.

```text
VryLvaoTRIQL12B7/
├── README.md
├── requirements.txt
├── term-deposit-marketing-2020.csv
├── term-deposit-marketing-2020-labelled.csv
├── 01_data_eda.ipynb
├── 02_baseline_tuning.ipynb
├── 03_features_imbalance.ipynb
├── 04_segments_robustness.ipynb
├── 05_final_validation_reporting.ipynb
├── Technical_Report.pdf
└── Business_Recommendations.pdf
```

`README_COMPREHENSIVE.md` will be added as the deeper methodology/traceability companion after the report rewrite is finalised.

---

## 14. How to Run

Install the pinned environment:

```bash
pip install -r requirements.txt
```

Run the notebooks sequentially from the repository root:

```text
01_data_eda.ipynb
02_baseline_tuning.ipynb
03_features_imbalance.ipynb
04_segments_robustness.ipynb
05_final_validation_reporting.ipynb
```

The notebooks load `term-deposit-marketing-2020-labelled.csv` directly from the root. The original 14-column source dataset is preserved as `term-deposit-marketing-2020.csv`.

No `data/`, `results/`, `figures/`, `scripts/`, `reports/` or `html/` directory is required.

---

## 15. Deliverables

- `01_data_eda.ipynb` — data audit, EDA, associations and ordered blocks;
- `02_baseline_tuning.ipynb` — leakage experiment, baselines, reproducible tuning and untouched holdout evaluation;
- `03_features_imbalance.ipynb` — feature importance, feature reduction and imbalance experiments;
- `04_segments_robustness.ipynb` — exploratory population, shuffled/group sensitivity and ordered robustness;
- `05_final_validation_reporting.ipynb` — forward lift and calibration;
- `Technical_Report.pdf` — detailed technical methodology and evidence;
- `Business_Recommendations.pdf` — management-facing decision and pilot recommendation;
- `requirements.txt` — pinned Python environment;
- original and labelled datasets at repository root.

The PDFs are being rewritten against the final executed notebook evidence; stale figures or values from older report versions are not authoritative.