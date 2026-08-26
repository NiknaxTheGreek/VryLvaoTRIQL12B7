import json, textwrap, ast
from pathlib import Path

FILES = [
    "01_data_eda.ipynb",
    "02_baseline_tuning.ipynb",
    "03_features_imbalance.ipynb",
    "04_segments_robustness.ipynb",
    "05_final_validation_reporting.ipynb",
]

DESCRIPTIONS = {
"01_data_eda.ipynb": {
1:"Import dependencies and define reusable EDA helpers for confidence intervals, associations, conversion tables and ordered blocks.",
3:"Load the root-level CSV, recreate `y_binary`, identify feature types and calculate the basic dataset/target structure.",
5:"Audit conventional missingness, explicit `unknown` categories, exact duplicates and pseudo-profile repetition.",
7:"Plot the target imbalance and explicit `unknown` shares with enough axis margin for value labels.",
9:"Summarise numeric distributions and category-level subscription rates with Wilson intervals.",
11:"Reconstruct contiguous source-order month-labelled blocks and visualise their subscription rates.",
13:"Calculate deployable Pearson/Cramér associations, numeric inter-feature correlations and a separate post-call `duration` leakage diagnostic.",
15:"Describe the exploratory higher-conversion population and its descriptive conversion uplift.",
17:"Collect the principal EDA quantities into one canonical cross-check table.",
},
"02_baseline_tuning.ipynb": {
1:"Load the dataset, define pre-call and +duration feature sets, and create deterministic development/tuning/final-holdout splits.",
3:"Define preprocessing, metrics and explicit baseline/default model settings. Within each model family, pre-call and +duration baselines use identical model hyperparameters.",
5:"Define and display the hyperparameter meanings, exact search values/ranges, candidate counts and deterministic selection rule.",
7:"Run four independent tuning searches: LR pre-call, LR +duration, HGB pre-call and HGB +duration.",
9:"Audit candidate rankings and display the four tuned winners separately from the baseline/default settings.",
11:"Refit baseline and tuned configurations on the full development sample and score the untouched final holdout.",
13:"Separate the performance change caused by `duration` from the performance change caused by tuning.",
15:"Plot all six headline metrics from the common untouched-holdout table.",
17:"Record canonical search counts and split sizes for downstream verification.",
},
"03_features_imbalance.ipynb": {
1:"Load the data, recreate the canonical split and transfer only the freshly selected pre-call LR/HGB parameters from Notebook 02.",
3:"Define common preprocessing, model constructors, six metrics and bias-corrected Cramér's V.",
5:"Compute direct association, grouped LR coefficient magnitude and held-out HGB permutation importance as separate evidence types.",
7:"Build nested Top-3/Top-4/Top-5 feature sets from HGB permutation ranking and compare them with all 12 pre-call predictors.",
9:"Visualise all six metrics for the feature-reduction experiment.",
11:"Compare no adjustment, weighting, over/undersampling, SMOTENC and nested threshold tuning on the same pre-call model structures.",
13:"Visualise the six headline metrics for every imbalance strategy.",
15:"Print compact feature/imbalance cross-checks for downstream reporting.",
},
"04_segments_robustness.ipynb": {
1:"Load the data and transferred tuned pre-call settings used throughout the robustness analysis.",
3:"Define common preprocessing, default/tuned constructors and metric helpers so validation designs differ without changing implementation.",
5:"Recreate the exploratory higher-conversion population and its descriptive summary.",
7:"Validate that population with default models while learning the balance-Q3 threshold inside each training fold.",
9:"Generate shuffled five-fold and pseudo-profile grouped out-of-fold predictions.",
11:"Reconstruct ordered month-labelled blocks and apply the forward eligibility gate.",
13:"Fit each model on all earlier rows only and score each eligible later block.",
15:"Aggregate ordered metrics using block-size weights and visualise ROC-AUC across eligible blocks.",
17:"Compare shuffled and ordered validation directly in a report-facing notebook figure.",
19:"Display the exploratory, shuffled/grouped and ordered summaries together for final cross-checking.",
},
"05_final_validation_reporting.ipynb": {
1:"Reconstruct the eligible forward predictions directly from the source dataset without cached result files.",
3:"Compute period-local lift, subscribers captured and random expectation at 5%, 10%, 20% and 50% call depth.",
5:"Visualise period-local lift by call depth with bootstrap uncertainty.",
7:"Calculate period-wise calibration bins, Brier score and expected calibration error without fitting a recalibration model.",
9:"Visualise score-bin calibration against the perfect-calibration diagonal.",
11:"Combine ordered discrimination, calibration diagnostics and lift into the final ranking summary.",
},
}

def load(path):
    return json.loads(Path(path).read_text())

def save(path, nb):
    Path(path).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")

def annotate(nb, mapping):
    for idx, desc in mapping.items():
        c = nb["cells"][idx]
        assert c["cell_type"] == "markdown", (idx, c["cell_type"])
        s = "".join(c.get("source", [])).rstrip()
        if "**Coding step.**" not in s:
            s += "\n\n**Coding step.** " + desc
            s += "\n\n**Why this step matters.** The explanation is kept immediately beside the code so the analytical logic and the meaning of its output are auditable."
        c["source"] = (s + "\n").splitlines(True)

# Notebook 1
nb = load("01_data_eda.ipynb")
annotate(nb, DESCRIPTIONS["01_data_eda.ipynb"])
src = "".join(nb["cells"][8]["source"])
if "ax.margins(y=0.16)" not in src:
    src = src.replace(
        "ax.set(title='Term-Deposit Subscription Outcome',ylabel='Customers')\n",
        "ax.set(title='Term-Deposit Subscription Outcome',ylabel='Customers')\nax.margins(y=0.16)\n"
    )
    src = src.replace(
        "ax.set(title=\"Explicit 'unknown' Categories\",ylabel='Share of records (%)')\n",
        "ax.set(title=\"Explicit 'unknown' Categories\",ylabel='Share of records (%)')\n    ax.margins(y=0.16)\n"
    )
nb["cells"][8]["source"] = src.splitlines(True)
save("01_data_eda.ipynb", nb)

# Notebook 2
nb = load("02_baseline_tuning.ipynb")
annotate(nb, DESCRIPTIONS["02_baseline_tuning.ipynb"])
nb["cells"][3]["source"] = textwrap.dedent("""
## 2. Model construction and baseline/default hyperparameters

The baseline comparison is deliberately **like-for-like within each model family**. Logistic Regression uses one fixed default/baseline configuration for both the pre-call and `+ duration` feature sets; HistGradientBoosting does the same. Therefore the baseline pre-call versus `+ duration` comparison changes the **information available to the model**, not its hyperparameters.

**Coding step.** Define preprocessing and metrics, declare the explicit LR/HGB baseline settings, and display those same settings side-by-side for the two feature conditions.

**Why this step matters.** This prevents the `duration` leakage effect from being confused with a model-setting change. Tuning happens later and produces separate winners for each model × feature condition.
""").strip().splitlines(True)

cell4 = "".join(nb["cells"][4]["source"])
start = cell4.index("def baseline_model(name):")
cell4 = cell4[:start] + textwrap.dedent("""
BASELINE_PARAMS = {
    'LR': {
        'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs',
        'max_iter': 1000, 'tol': 1e-4, 'class_weight': None,
        'random_state': SEED
    },
    'HGB': {
        'loss': 'log_loss', 'learning_rate': 0.1, 'max_iter': 100,
        'max_leaf_nodes': 31, 'max_depth': None, 'min_samples_leaf': 20,
        'l2_regularization': 0.0, 'max_bins': 255,
        'early_stopping': 'auto', 'class_weight': None,
        'random_state': SEED
    }
}

def baseline_model(name):
    return (
        LogisticRegression(**BASELINE_PARAMS['LR'])
        if name == 'LR'
        else HistGradientBoostingClassifier(**BASELINE_PARAMS['HGB'])
    )

baseline_lr_table = pd.DataFrame([
    {'feature_condition': condition, **{
        k: BASELINE_PARAMS['LR'][k]
        for k in ['C','penalty','solver','max_iter','tol','class_weight']
    }}
    for condition in ['Pre-call','With duration']
])
baseline_hgb_table = pd.DataFrame([
    {'feature_condition': condition, **{
        k: BASELINE_PARAMS['HGB'][k]
        for k in ['loss','learning_rate','max_iter','max_leaf_nodes','max_depth',
                  'min_samples_leaf','l2_regularization','max_bins',
                  'early_stopping','class_weight']
    }}
    for condition in ['Pre-call','With duration']
])

print('Logistic Regression baseline/default settings — identical across feature conditions:')
display(baseline_lr_table.set_index('feature_condition').T)
print('HistGradientBoosting baseline/default settings — identical across feature conditions:')
display(baseline_hgb_table.set_index('feature_condition').T)
""").lstrip()
nb["cells"][4]["source"] = cell4.splitlines(True)

nb["cells"][5]["source"] = textwrap.dedent("""
## 3. Hyperparameters: descriptions, tuning ranges and search resolution

This section presents the **parameters and search design only** before the tuned winners are shown.

| Model | Hyperparameter | Description | Values searched |
|---|---|---|---|
| LR | `C` | Inverse regularisation strength | 15 log-spaced values from 0.001 to 100: `np.logspace(-3,2,15)` |
| LR | `penalty` | Coefficient regularisation type | `l1`, `l2` |
| LR | `class_weight` | Class reweighting | `None`, `balanced` |
| HGB | `max_iter` | Maximum boosting iterations | 50, 100, 150, 200, 300, 400 |
| HGB | `max_depth` | Maximum tree depth | 3, 4, 5, 6, 8, `None` |
| HGB | `learning_rate` | Contribution of each boosting iteration | 0.01, 0.03, 0.05, 0.10, 0.20 |
| HGB | `max_leaf_nodes` | Maximum leaves per tree | 15, 31, 63, 127 |
| HGB | `l2_regularization` | L2 regularisation of leaf values | 0, 0.1, 0.5, 1, 2 |
| HGB | `min_samples_leaf` | Minimum observations in a leaf | 10, 20, 30, 50 |
| HGB | `class_weight` | Class reweighting | `None`, `balanced` |

LR exhaustively evaluates 60 combinations per feature condition. HGB has 28,800 possible combinations and samples 100 reproducibly with `random_state=42` per feature condition. Across LR pre-call, LR + duration, HGB pre-call and HGB + duration, exactly **320 candidate configurations** are evaluated.

Selection is by validation **PR-AUC** with **ROC-AUC** as tie-breaker. The final holdout is not consulted.

**Coding step.** Construct the exact LR grid and seeded HGB candidate sample and display the machine-readable search specification and candidate counts.
""").strip().splitlines(True)

cell6 = "".join(nb["cells"][6]["source"])
if "search_space_table = pd.DataFrame" not in cell6:
    cell6 += textwrap.dedent("""

search_space_table = pd.DataFrame([
    {'model':'LR','parameter':'C','values_tested':'15 log-spaced values: ' + ', '.join(f'{v:.6g}' for v in np.logspace(-3,2,15))},
    {'model':'LR','parameter':'penalty','values_tested':'l1, l2'},
    {'model':'LR','parameter':'class_weight','values_tested':'None, balanced'},
    {'model':'HGB','parameter':'max_iter','values_tested':'50, 100, 150, 200, 300, 400'},
    {'model':'HGB','parameter':'max_depth','values_tested':'3, 4, 5, 6, 8, None'},
    {'model':'HGB','parameter':'learning_rate','values_tested':'0.01, 0.03, 0.05, 0.10, 0.20'},
    {'model':'HGB','parameter':'max_leaf_nodes','values_tested':'15, 31, 63, 127'},
    {'model':'HGB','parameter':'l2_regularization','values_tested':'0, 0.1, 0.5, 1, 2'},
    {'model':'HGB','parameter':'min_samples_leaf','values_tested':'10, 20, 30, 50'},
    {'model':'HGB','parameter':'class_weight','values_tested':'None, balanced'}
])
display(search_space_table)
print('LR exhaustive candidates per feature condition:', len(LR_GRID))
print('HGB possible combinations:', HGB_SEARCH_SPACE_SIZE)
print('HGB sampled candidates per feature condition:', len(HGB_CANDIDATES))
print('Total candidates across four tuned conditions:', 2*len(LR_GRID) + 2*len(HGB_CANDIDATES))
""")
nb["cells"][6]["source"] = cell6.splitlines(True)

nb["cells"][9]["source"] = textwrap.dedent("""
## 5. Tuned winners after the search

The four model × feature conditions are tuned **independently**. These selected configurations are not baseline settings.

**Coding step.** Show the highest-ranked validation candidates, then display LR and HGB tuned winners with pre-call and `+ duration` settings side-by-side.

**Interpretation.** The two baseline versions within each model family were identical in Section 2; the tuned pre-call and tuned `+ duration` configurations shown here are allowed to differ. Only the tuned **pre-call** settings are transferred to deployable downstream notebooks.
""").strip().splitlines(True)

cell10 = "".join(nb["cells"][10]["source"])
marker = "selected_table = pd.DataFrame(["
if marker in cell10:
    cell10 = cell10.split(marker)[0] + textwrap.dedent("""
selected_table = pd.DataFrame([
    {
        'stage':'Tuned winner',
        'model':'LR' if k.startswith('LR') else 'HGB',
        'feature_condition':'Pre-call' if k.endswith('pre_call') else 'With duration',
        'configuration':k,
        'selected_parameters':json.dumps(v, sort_keys=True)
    }
    for k,v in selected.items()
])

lr_tuned_table = pd.DataFrame([
    {
        'feature_condition':condition,
        'C':selected[key]['C'],
        'penalty':selected[key]['penalty'],
        'class_weight':selected[key]['class_weight']
    }
    for condition,key in [('Pre-call','LR_pre_call'),('With duration','LR_with_duration')]
])
hgb_tuned_table = pd.DataFrame([
    {
        'feature_condition':condition,
        **{p:selected[key][p] for p in [
            'max_iter','max_depth','learning_rate','max_leaf_nodes',
            'l2_regularization','min_samples_leaf','class_weight','early_stopping'
        ]}
    }
    for condition,key in [('Pre-call','HGB_pre_call'),('With duration','HGB_with_duration')]
])

print('Tuned Logistic Regression winners — independently selected:')
display(lr_tuned_table.set_index('feature_condition').T)
print('Tuned HistGradientBoosting winners — independently selected:')
display(hgb_tuned_table.set_index('feature_condition').T)
""").lstrip()
nb["cells"][10]["source"] = cell10.splitlines(True)
save("02_baseline_tuning.ipynb", nb)

# Notebook 3
nb = load("03_features_imbalance.ipynb")
annotate(nb, DESCRIPTIONS["03_features_imbalance.ipynb"])
if not any("Notebook-generated feature-importance bar plots" in "".join(c.get("source",[])) for c in nb["cells"]):
    insert_at = 7
    nb["cells"][insert_at:insert_at] = [
        {
            "cell_type":"markdown","metadata":{},
            "source":textwrap.dedent("""
### 3.1 Notebook-generated feature-importance bar plots

**Coding step.** Visualise grouped LR coefficient magnitude and HGB held-out permutation importance as separate horizontal bar charts.

**Why separate plots.** The two importance measures have different scales and meanings and should not be combined into a synthetic score.

**Report traceability.** These notebook outputs are the canonical source for LR/HGB feature-importance figures used in the Technical Report. `duration` cannot appear because only the 12 pre-call predictors are fitted.
""").strip().splitlines(True)
        },
        {
            "cell_type":"code","execution_count":None,"metadata":{},"outputs":[],
            "source":textwrap.dedent("""
lr_plot = importance[["feature","lr_importance"]].sort_values("lr_importance")
fig, ax = plt.subplots(figsize=(8.5,5.8))
ax.barh(lr_plot["feature"], lr_plot["lr_importance"])
ax.set(title="Logistic Regression Feature Importance",
       xlabel="Maximum absolute encoded coefficient", ylabel="")
ax.margins(x=0.05)
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.show()

hgb_plot = importance[["feature","hgb_permutation"]].sort_values("hgb_permutation")
fig, ax = plt.subplots(figsize=(8.5,5.8))
ax.barh(hgb_plot["feature"], hgb_plot["hgb_permutation"])
ax.axvline(0, linewidth=1)
ax.set(title="HistGradientBoosting Permutation Importance",
       xlabel="Mean decrease in holdout PR-AUC after permutation", ylabel="")
ax.margins(x=0.05)
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.show()
""").strip().splitlines(True)
        }
    ]
save("03_features_imbalance.ipynb", nb)

# Notebooks 4 and 5
for fn in ["04_segments_robustness.ipynb","05_final_validation_reporting.ipynb"]:
    nb = load(fn)
    annotate(nb, DESCRIPTIONS[fn])
    save(fn, nb)

# Clear stale outputs because the next step will execute these exact sources.
for fn in FILES:
    nb = load(fn)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None
            ast.parse("".join(c.get("source",[])))
    types = [c["cell_type"] for c in nb["cells"]]
    assert not any(types[i]=="code" and types[i+1]=="code" for i in range(len(types)-1)), fn
    for i,c in enumerate(nb["cells"]):
        if c["cell_type"]=="code":
            assert i>0 and nb["cells"][i-1]["cell_type"]=="markdown", (fn,i)
    src = "\n".join("".join(c.get("source",[])) for c in nb["cells"])
    assert "term-deposit-marketing-2020-labelled.csv" in src
    assert "/mnt/data/" not in src
    save(fn, nb)

assert "ax.margins(y=0.16)" in Path("01_data_eda.ipynb").read_text()
nb2 = load("02_baseline_tuning.ipynb")
s2 = "\n".join("".join(c.get("source",[])) for c in nb2["cells"])
assert "identical across feature conditions" in s2
assert "320 candidate configurations" in s2
nb3 = load("03_features_imbalance.ipynb")
s3 = "\n".join("".join(c.get("source",[])) for c in nb3["cells"])
assert "Logistic Regression Feature Importance" in s3
assert "HistGradientBoosting Permutation Importance" in s3
print("Step 1 notebook source fixes validated.")
