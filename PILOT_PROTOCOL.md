# Controlled Prospective Pilot Protocol

## Purpose

The retrospective modelling project is complete. The next phase is a prospective test of whether the frozen pre-call ranking policy improves real campaign performance on genuinely new eligible customers.

The primary policy is the final all-12 unweighted Logistic Regression. HistGradientBoosting remains an optional challenger. The control is the bank's business-as-usual targeting process; if no BAU ranking exists, the control may use random selection.

This protocol does not treat historical lift as causal evidence. The pilot is intended to generate the first prospective evidence of operational and economic value.

## Frozen primary model

Primary policy:

- Logistic Regression;
- pre-call predictors: `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`, `contact`, `day`, `month`, `campaign`;
- L2 penalty;
- `C=8.483428982440726`;
- `class_weight=None`;
- `solver=liblinear`;
- `max_iter=3000`;
- `random_state=42`.

Optional challenger:

- HistGradientBoosting;
- `max_iter=200`;
- `max_depth=8`;
- `learning_rate=0.05`;
- `max_leaf_nodes=15`;
- `l2_regularization=0`;
- `min_samples_leaf=20`;
- `class_weight=None`;
- `early_stopping=True`;
- `random_state=42`.

`duration` is prohibited from all scoring, ranking, randomisation, eligibility and call-selection logic.

## Training for live scoring

The model family, feature set and hyperparameters are already selected. For live scoring, the frozen model may be refit once on all 40,000 historical rows using the same preprocessing and the 12 valid pre-call predictors. This increases the training information available to the frozen model without reopening model selection.

No tuning or feature changes are permitted after pilot outcomes are visible.

## Required live customer file

Place a fresh, genuinely prospective eligible-customer snapshot at the repository root as:

`pilot_eligible_customers.csv`

Required columns:

`customer_id`, `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`, `contact`, `day`, `month`, `campaign`.

The file must already have the bank's normal product eligibility, consent, suppression and contact-frequency rules applied.

The assignment input must not contain `duration`, `y`, `y_binary` or `subscribed`.

`customer_id` must be unique within the pilot snapshot.

## Experimental design

The cleanest minimum design has two arms:

| Arm | Policy |
|---|---|
| LR-ranked | Rank customers by frozen LR score and call from the top |
| Control | Apply the bank's BAU priority; if none exists, select randomly |

An HGB-ranked challenger arm may be added if the call-centre capacity is sufficient.

Customers are randomised to policy pools before each policy selects customers to call. Broad pre-call LR score strata may be used only to improve balance between the randomised pools.

Each arm must receive the same call budget.

The historical 5%, 10% and 20% call depths are sensitivity results only. They are not automatically reused as pilot quotas.

## Inputs that must be fixed before launch

Before producing the live assignment, freeze:

- the fresh eligible-customer snapshot;
- the exact control policy;
- whether HGB is included;
- calls per arm;
- pilot start and end window;
- primary outcome definition;
- exclusion/missing-outcome rules;
- statistical confidence level;
- minimum worthwhile improvement;
- call-cost definition;
- net contribution per subscription;
- go/no-go guardrails.

These must be locked before outcomes are inspected.

## Assignment process

The operational sequence is:

**business eligibility and consent rules → fresh eligible pool → pre-call model scoring → randomised policy pools → equal-capacity policy selection → frozen assignment → calling → outcomes**

The frozen assignment should retain:

- `customer_id`;
- `pilot_assignment_id`;
- policy arm;
- LR score and rank;
- HGB score and rank if challenger is used;
- score stratum used for randomisation;
- whether the customer was assigned a call;
- the 12 feature values at scoring time;
- model/version identifier;
- scoring and assignment timestamp.

The assignment must be frozen before call outcomes are observed.

## Outcomes to capture

Primary outcome:

- `subscribed` as 0/1.

Operational measures where available:

- call attempted;
- successful contact;
- opt-out;
- complaint;
- repeat contact;
- agent minutes;
- contact outcome;
- campaign identifier/date;
- eventual term-deposit subscription date.

The pre-call feature snapshot must not be overwritten by later customer-data updates.

## Primary analysis

The primary comparison is between customers actually assigned a call under each randomised policy pool, using equal call capacity.

Headline measure:

**subscriptions per 1,000 assigned calls**

For each arm calculate:

- assigned calls;
- subscriptions;
- subscription rate;
- subscriptions per 1,000 assigned;
- absolute subscription-rate difference versus control;
- relative lift versus control;
- 95% confidence interval for the absolute difference.

A live result is superior only if it is operationally meaningful, statistically credible and economically valuable.

## Guardrail analysis

Also compare, where available:

- contactability;
- opt-out rate;
- complaint rate;
- repeat-contact rate;
- customer-mix concentration;
- agent time;
- unsuccessful call burden.

A targeting policy should not be promoted merely because it increases subscriptions if it creates unacceptable customer or operational harm.

## Sample size

Pilot size should be based on:

- expected control conversion rate;
- minimum absolute improvement worth detecting;
- chosen alpha;
- desired statistical power.

The historical 7.24% rate can be used only as a planning reference if the bank has no more recent control estimate. The minimum detectable improvement must come from business value, not from retrospective model lift.

`06_controlled_pilot.ipynb` contains a two-proportion sample-size helper.

## Economics

The historical dataset does not contain the bank's true call cost or net contribution per new term-deposit subscription.

When supplied, calculate:

`net value = subscriptions × net contribution per subscription − assigned calling effort × relevant call cost`

The key economic comparison is:

`incremental pilot value = net value of LR arm − net value of control arm`

If probability-based expected value is required at customer level, prospective calibration should be evaluated after the pilot rather than assuming the current raw scores are literal probabilities.

## Go/no-go decision

Production should require all four gates:

1. **Ranking value:** LR improves subscriptions per 1,000 assigned versus control.
2. **Reliability:** the improvement is statistically credible and reasonably stable.
3. **Economics:** incremental net value is positive at realistic operating costs.
4. **Risk:** customer-experience and operational guardrails remain acceptable.

If HGB is included, it should replace LR only if the prospective evidence justifies its additional complexity.

## What can be done now

The pilot implementation notebook is ready and can:

- refit the frozen LR/HGB models on all historical rows;
- validate a new eligible-customer file;
- reject post-call/outcome leakage;
- score new customers;
- randomise balanced policy pools;
- select equal call capacity;
- audit the frozen assignment;
- calculate sample-size scenarios;
- analyse prospective outcomes;
- calculate economics once cost/contribution inputs are supplied.

## What cannot be truthfully completed without the bank

A real prospective result cannot be produced until there are new customers, a call budget, a defined control process and observed live outcomes.

The project must not substitute historical rows or synthetic outcomes and label them as a prospective pilot.
