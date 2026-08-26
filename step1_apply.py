import ast
import json
import textwrap
from pathlib import Path

FILES = [
    "01_data_eda.ipynb",
    "02_baseline_tuning.ipynb",
    "03_features_imbalance.ipynb",
    "04_segments_robustness.ipynb",
    "05_final_validation_reporting.ipynb",
]

def load(path):
    return json.loads(Path(path).read_text())

def save(path, nb):
    Path(path).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")

def text(cell):
    return "".join(cell.get("source", []))

def set_text(cell, value):
    cell["source"] = value.strip().splitlines(True)

def find_md(nb, *needles):
    needles = [n.lower() for n in needles]
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown":
            s = text(cell).lower()
            if any(n in s for n in needles):
                return i
    raise KeyError(f"Markdown section not found: {needles}")

def next_code(nb, md_index):
    for j in range(md_index + 1, len(nb["cells"])):
        if nb["cells"][j]["cell_type"] == "code":
            return j
        if nb["cells"][j]["cell_type"] == "markdown":
            break
    raise KeyError(f"No code directly associated with markdown cell {md_index}")

def heading(cell):
    for line in text(cell).splitlines():
        if line.strip().startswith("#"):
            return line.lstrip("# ").strip()
    return "this analysis stage"

def stage_description(fn, h):
    h = h.lower()
    rules = {
        "01_data_eda.ipynb": [
            (("setup","reusable"), "Import dependencies and define reusable EDA helpers used by the later data-quality, association and ordered-block calculations."),
            (("load","dataset","structure"), "Load the root-level labelled dataset, recreate the binary target and establish the predictor/target structure."),
            (("quality","missing","duplicate"), "Audit conventional missingness, explicit `unknown` categories, exact duplicates and repeated pseudo-profiles before modelling."),
            (("target balance","unknown"), "Visualise class imbalance and explicit `unknown` category shares, with axis margins reserved for readable annotations."),
            (("numeric","categorical","conversion"), "Summarise numeric distributions and category-level subscription rates with uncertainty intervals."),
            (("ordered","month"), "Reconstruct contiguous source-order month-labelled blocks and visualise how observed subscription rate varies across them."),
            (("association","cram","pearson"), "Calculate pre-call Pearson/Cramér associations and keep `duration` separate as a post-call leakage diagnostic."),
            (("higher-conversion","exploratory","population"), "Describe the exploratory higher-conversion population without treating it as a separately tuned deployment model."),
            (("canonical","cross-check","summary"), "Collect the main EDA quantities into a compact cross-check table for downstream notebooks and reports."),
        ],
        "02_baseline_tuning.ipynb": [
            (("validation design","feature sets"), "Load the data, define pre-call and +duration feature sets, and create deterministic development, tuning and final-holdout splits."),
            (("baseline","preprocessing"), "Define preprocessing, metrics and explicit baseline/default model settings shared across the two feature conditions within each model family."),
            (("hyperparameter","search"), "Construct the exact LR grid and seeded HGB candidate sample, and report the tested ranges and candidate counts."),
            (("fresh tuning","selected"), "Run the four independent model × feature-condition searches and display the selected tuned winners separately from the baselines."),
            (("holdout","baseline","tuned"), "Refit baseline and tuned configurations on development data and evaluate them once on the untouched final holdout."),
            (("duration","delta","leakage"), "Separate the performance change caused by adding post-call `duration` from the change caused by hyperparameter tuning."),
            (("six-metric","metric panel"), "Visualise Accuracy, Precision, Recall, F1, PR-AUC and ROC-AUC from the common holdout results table."),
            (("canonical","cross-check","record"), "Record the selected settings, search counts and split sizes used for downstream verification."),
        ],
        "03_features_imbalance.ipynb": [
            (("setup","canonical split","upstream"), "Load the data, recreate the canonical split and transfer the freshly selected pre-call LR/HGB settings from Notebook 02 without retuning."),
            (("reusable","metric functions"), "Define common preprocessing, model constructors, six headline metrics and association helpers."),
            (("feature importance",), "Compute direct association, grouped LR coefficient magnitude and held-out HGB permutation importance as distinct evidence types."),
            (("feature reduction",), "Build nested Top-3/Top-4/Top-5 subsets from HGB permutation ranking and compare them with all 12 pre-call predictors."),
            (("feature-reduction","six-metric"), "Visualise all six headline metrics for the feature-reduction comparison from the notebook result table."),
            (("class imbalance","imbalance"), "Compare no adjustment, weighting, over/undersampling, SMOTENC and nested threshold optimisation using the same pre-call model structures."),
            (("imbalance","six-metric"), "Visualise the six headline metrics for each imbalance strategy and model family."),
            (("cross-check","conclusion"), "Print compact feature and imbalance cross-checks for report traceability."),
        ],
        "04_segments_robustness.ipynb": [
            (("setup","upstream"), "Load the data and transferred tuned pre-call settings used throughout the robustness analysis."),
            (("model","metric","reusable"), "Define common preprocessing, default/tuned constructors and metric helpers so validation designs can be compared consistently."),
            (("exploratory","population","segment"), "Recreate the exploratory higher-conversion population and its descriptive summary."),
            (("population model","segment validation"), "Validate the exploratory population with default models while learning the balance-Q3 threshold inside each training fold."),
            (("shuffled","pseudo","group"), "Generate shuffled five-fold and pseudo-profile grouped out-of-fold predictions as complementary sensitivity checks."),
            (("ordered","block","period"), "Reconstruct ordered month-labelled blocks and apply the forward-validation eligibility gate."),
            (("forward","expanding"), "Fit on all earlier rows only and score each eligible later block without using future rows."),
            (("aggregate","forward performance"), "Aggregate block-level forward metrics using test-block-size weights and visualise performance across eligible blocks."),
            (("shuffled versus ordered","shuffled vs ordered"), "Compare shuffled and ordered validation directly in a simple report-facing figure."),
            (("conclusion","cross-check"), "Display the exploratory, shuffled/grouped and ordered summaries together for final robustness cross-checking."),
        ],
        "05_final_validation_reporting.ipynb": [
            (("setup","forward","prediction"), "Reconstruct eligible forward predictions directly from the dataset and transferred tuned pre-call settings, without cached result files."),
            (("lift","call depth"), "Compute period-local lift, captured subscribers and random expectation at 5%, 10%, 20% and 50% call depth."),
            (("lift visual","visualisation"), "Visualise how historical ranking lift changes with call depth, including bootstrap uncertainty already calculated in the notebook."),
            (("calibration","brier","ece"), "Calculate period-wise calibration bins, Brier score and expected calibration error without fitting a recalibration model."),
            (("calibration visual","visualisation"), "Plot score-bin calibration against the perfect-calibration diagonal as a diagnostic rather than a deployed probability model."),
            (("summary","conclusion","final"), "Combine ordered discrimination, calibration diagnostics and lift into the final ranking summary used for reporting."),
        ],
    }
    for keys, desc in rules[fn]:
        if any(k in h for k in keys):
            return desc
    return f"Execute the calculations defined in **{h}** and display the resulting evidence before moving to the next analytical stage."

def annotate_all(fn, nb):
    for i, cell in enumerate(nb["cells"][:-1]):
        if cell["cell_type"] != "markdown" or nb["cells"][i+1]["cell_type"] != "code":
            continue
        s = text(cell).rstrip()
        if "**Coding step.**" not in s:
            s += "\n\n**Coding step.** " + stage_description(fn, heading(cell))
        if "**Why this step matters.**" not in s:
            s += "\n\n**Why this step matters.** Keeping the rationale next to the code makes the analytical sequence, validation logic and interpretation auditable."
        set_text(cell, s)

# Notebook 1: fix the first figure presentation.
nb = load("01_data_eda.ipynb")
mi = find_md(nb, "target balance", "explicit unknown")
ci = next_code(nb, mi)
src = text(nb["cells"][ci])
assert "Term-Deposit Subscription Outcome" in src
if "ax.margins(y=0.16)" not in src:
    src = src.replace("ax.set(title='Term-Deposit Subscription Outcome',ylabel='Customers')", "ax.set(title='Term-Deposit Subscription Outcome',ylabel='Customers')\nax.margins(y=0.16)", 1)
    src = src.replace('ax.set(title="Explicit \'unknown\' Categories",ylabel=\'Share of records (%)\')', 'ax.set(title="Explicit \'unknown\' Categories",ylabel=\'Share of records (%)\')\n    ax.margins(y=0.16)', 1)
nb["cells"][ci]["source"] = src.splitlines(True)
annotate_all("01_data_eda.ipynb", nb)
save("01_data_eda.ipynb", nb)

# Notebook 2: baseline/default settings versus independently tuned winners.
nb = load("02_baseline_tuning.ipynb")
mi = find_md(nb, "preprocessing, baseline", "baseline models")
set_text(nb["cells"][mi], textwrap.dedent("""
## 2. Preprocessing and baseline/default hyperparameters

The baseline comparison is deliberately **like-for-like within each model family**. Logistic Regression uses one fixed baseline/default configuration for both the **pre-call** and **+ duration** feature sets; HistGradientBoosting does the same. The baseline experiment therefore changes the information available to the model, **not its model hyperparameters**.

There are only two baseline model specifications: one LR baseline and one HGB baseline. Each is evaluated once with the 12 valid pre-call predictors and once with the same predictors plus post-call `duration`.

**Coding step.** Define preprocessing and metrics, declare the explicit LR/HGB baseline/default settings, and display those identical settings side-by-side for the two feature conditions.

**Why this step matters.** The initial `duration` comparison must be a genuine feature-availability experiment rather than a comparison confounded by different model settings.
"""))
ci = next_code(nb, mi)
src = text(nb["cells"][ci])
if "BASELINE_PARAMS =" not in src:
    src += textwrap.dedent("""

# Explicit baseline/default settings; reused unchanged across feature conditions.
BASELINE_PARAMS = {
    "LR": {"C":1.0,"penalty":"l2","solver":"lbfgs","max_iter":1000,"tol":1e-4,"class_weight":None,"random_state":SEED},
    "HGB": {"loss":"log_loss","learning_rate":0.1,"max_iter":100,"max_leaf_nodes":31,"max_depth":None,"min_samples_leaf":20,"l2_regularization":0.0,"max_bins":255,"early_stopping":"auto","class_weight":None,"random_state":SEED},
}

def baseline(kind):
    return LogisticRegression(**BASELINE_PARAMS["LR"]) if kind == "LR" else HistGradientBoostingClassifier(**BASELINE_PARAMS["HGB"])

baseline_lr_table = pd.DataFrame([{"feature_condition":condition, **{k:BASELINE_PARAMS["LR"][k] for k in ["C","penalty","solver","max_iter","tol","class_weight"]}} for condition in ["Pre-call","With duration"]])
baseline_hgb_table = pd.DataFrame([{"feature_condition":condition, **{k:BASELINE_PARAMS["HGB"][k] for k in ["loss","learning_rate","max_iter","max_leaf_nodes","max_depth","min_samples_leaf","l2_regularization","max_bins","early_stopping","class_weight"]}} for condition in ["Pre-call","With duration"]])
print("Logistic Regression baseline/default settings — identical across feature conditions:")
display(baseline_lr_table.set_index("feature_condition").T)
print("HistGradientBoosting baseline/default settings — identical across feature conditions:")
display(baseline_hgb_table.set_index("feature_condition").T)
""")
nb["cells"][ci]["source"] = src.splitlines(True)

mi = find_md(nb, "hyperparameter search spaces", "hyperparameters:")
set_text(nb["cells"][mi], textwrap.dedent("""
## 3. Hyperparameters: descriptions, tuning ranges and search resolution

This section presents the **parameters and search design only**. Baseline/default settings were fixed in Section 2; tuning searches new settings separately for each model × feature condition.

| Model | Hyperparameter | What it controls | Values searched |
|---|---|---|---|
| LR | `C` | Inverse regularisation strength | 15 log-spaced values from 0.001 to 100 (`np.logspace(-3,2,15)`) |
| LR | `penalty` | Coefficient regularisation type | `l1`, `l2` |
| LR | `class_weight` | Relative weighting of the two classes | `None`, `balanced` |
| HGB | `max_iter` | Maximum boosting iterations | 50, 100, 150, 200, 300, 400 |
| HGB | `max_depth` | Maximum tree depth | 3, 4, 5, 6, 8, `None` |
| HGB | `learning_rate` | Contribution of each boosting iteration | 0.01, 0.03, 0.05, 0.10, 0.20 |
| HGB | `max_leaf_nodes` | Maximum terminal leaves per tree | 15, 31, 63, 127 |
| HGB | `l2_regularization` | L2 penalty on leaf values | 0, 0.1, 0.5, 1, 2 |
| HGB | `min_samples_leaf` | Minimum observations in a leaf | 10, 20, 30, 50 |
| HGB | `class_weight` | Relative weighting of the two classes | `None`, `balanced` |

LR exhaustively evaluates **60** configurations per feature condition. HGB has **28,800** possible combinations and samples **100** reproducibly with `random_state=42` per feature condition. Across LR pre-call, LR + duration, HGB pre-call and HGB + duration, exactly **320 candidate configurations** are evaluated.

Selection uses validation **PR-AUC as the primary objective** and **ROC-AUC as the deterministic tie-breaker**. The final 20% holdout is not consulted during selection.

**Coding step.** Construct the exact LR grid and seeded HGB candidate sample and display the machine-readable search specification and candidate counts.

**Why this step matters.** Stating the tested values and sampling rule makes the tuning reproducible and separates search design from the selected winners shown next.
"""))
ci = next_code(nb, mi)
src = text(nb["cells"][ci])
if "search_space_table = pd.DataFrame" not in src:
    src += textwrap.dedent("""

search_space_table = pd.DataFrame([
    {"model":"LR","parameter":"C","values_tested":"15 log-spaced values from 0.001 to 100"},
    {"model":"LR","parameter":"penalty","values_tested":"l1, l2"},
    {"model":"LR","parameter":"class_weight","values_tested":"None, balanced"},
    {"model":"HGB","parameter":"max_iter","values_tested":"50, 100, 150, 200, 300, 400"},
    {"model":"HGB","parameter":"max_depth","values_tested":"3, 4, 5, 6, 8, None"},
    {"model":"HGB","parameter":"learning_rate","values_tested":"0.01, 0.03, 0.05, 0.10, 0.20"},
    {"model":"HGB","parameter":"max_leaf_nodes","values_tested":"15, 31, 63, 127"},
    {"model":"HGB","parameter":"l2_regularization","values_tested":"0, 0.1, 0.5, 1, 2"},
    {"model":"HGB","parameter":"min_samples_leaf","values_tested":"10, 20, 30, 50"},
    {"model":"HGB","parameter":"class_weight","values_tested":"None, balanced"},
])
display(search_space_table)
print("LR exhaustive candidates per feature condition:", len(LR_GRID))
print("HGB possible combinations:", HGB_SPACE_SIZE)
print("HGB sampled candidates per feature condition:", len(HGB_CANDIDATES))
print("Total candidates across four tuned conditions:", 2*len(LR_GRID) + 2*len(HGB_CANDIDATES))
""")
nb["cells"][ci]["source"] = src.splitlines(True)

mi = find_md(nb, "fresh tuning run", "selected configurations")
set_text(nb["cells"][mi], textwrap.dedent("""
## 4. Fresh tuning run and four independently selected winners

The four model × feature conditions are tuned **independently**: LR pre-call, LR + duration, HGB pre-call and HGB + duration. Unlike the baseline comparison, these four tuned configurations are allowed to differ because each search selects the best validation PR-AUC for its own model and feature condition.

**Coding step.** Rerun all four searches from scratch, audit their validation rankings, and display the tuned pre-call and tuned +duration winners side-by-side for each model family.

**Why this step matters.** The distinction is explicit: **same baseline settings across feature conditions; separate tuned winners after search**. Only the tuned pre-call settings are transferred to deployable downstream notebooks.
"""))
ci = next_code(nb, mi)
src = text(nb["cells"][ci])
if "Tuned Logistic Regression winners" not in src:
    src += textwrap.dedent("""

lr_tuned_table = pd.DataFrame([{"feature_condition":condition,"C":selected[key]["C"],"penalty":selected[key]["penalty"],"class_weight":selected[key]["class_weight"]} for condition,key in [("Pre-call","LR_pre_call"),("With duration","LR_with_duration")]])
hgb_tuned_table = pd.DataFrame([{"feature_condition":condition, **{p:selected[key][p] for p in ["max_iter","max_depth","learning_rate","max_leaf_nodes","l2_regularization","min_samples_leaf","class_weight","early_stopping"]}} for condition,key in [("Pre-call","HGB_pre_call"),("With duration","HGB_with_duration")]])
print("Tuned Logistic Regression winners — independently selected:")
display(lr_tuned_table.set_index("feature_condition").T)
print("Tuned HistGradientBoosting winners — independently selected:")
display(hgb_tuned_table.set_index("feature_condition").T)
""")
nb["cells"][ci]["source"] = src.splitlines(True)

try:
    hi = find_md(nb, "untouched-holdout baseline and tuned", "holdout baseline")
    s = text(nb["cells"][hi]).rstrip()
    if "baseline rows use the same" not in s.lower():
        s += "\n\nThe baseline rows use the same default LR/HGB settings across pre-call and +duration; the tuned rows use the four independently selected winners from Section 4."
    set_text(nb["cells"][hi], s)
except KeyError:
    pass
annotate_all("02_baseline_tuning.ipynb", nb)
save("02_baseline_tuning.ipynb", nb)

# Notebook 3: add notebook-generated feature-importance plots.
nb = load("03_features_imbalance.ipynb")
mi = find_md(nb, "## 3. feature importance", "feature importance")
ci = next_code(nb, mi)
if not any("Notebook-generated feature-importance bar plots" in text(c) for c in nb["cells"]):
    plot_md = {"cell_type":"markdown","metadata":{},"source":textwrap.dedent("""
### 3.1 Notebook-generated feature-importance bar plots

**Coding step.** Plot grouped LR coefficient magnitude and held-out HGB permutation importance as two separate horizontal bar charts using the `importance` table calculated above.

**Why this step matters.** The measures have different meanings and scales, so they should not be blended into a synthetic score. These notebook outputs are the canonical source for the corresponding Technical Report figures. `duration` cannot appear because only the 12 valid pre-call predictors are fitted.
""").strip().splitlines(True)}
    plot_code = {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":textwrap.dedent("""
lr_plot = importance[["feature","lr_importance"]].sort_values("lr_importance")
fig, ax = plt.subplots(figsize=(8.5,5.8))
ax.barh(lr_plot["feature"], lr_plot["lr_importance"])
ax.set(title="Logistic Regression Feature Importance", xlabel="Maximum absolute encoded coefficient", ylabel="")
ax.margins(x=0.05)
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.show()

hgb_plot = importance[["feature","hgb_permutation"]].sort_values("hgb_permutation")
fig, ax = plt.subplots(figsize=(8.5,5.8))
ax.barh(hgb_plot["feature"], hgb_plot["hgb_permutation"])
ax.axvline(0, linewidth=1)
ax.set(title="HistGradientBoosting Permutation Importance", xlabel="Mean decrease in holdout PR-AUC after permutation", ylabel="")
ax.margins(x=0.05)
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.show()
""").strip().splitlines(True)}
    nb["cells"][ci+1:ci+1] = [plot_md, plot_code]
annotate_all("03_features_imbalance.ipynb", nb)
save("03_features_imbalance.ipynb", nb)

# Notebooks 4 and 5: explicit narrative beside every substantive code stage.
for fn in ["04_segments_robustness.ipynb", "05_final_validation_reporting.ipynb"]:
    nb = load(fn)
    annotate_all(fn, nb)
    save(fn, nb)

# Validate the final source structure. Outputs are deliberately cleared; Step 2
# will execute these exact sources from 01 through 05 and repopulate outputs.
for fn in FILES:
    nb = load(fn)
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(text(cell))
            cell["outputs"] = []
            cell["execution_count"] = None
    types = [c["cell_type"] for c in nb["cells"]]
    assert not any(types[i]=="code" and types[i+1]=="code" for i in range(len(types)-1)), fn
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            assert i > 0 and nb["cells"][i-1]["cell_type"] == "markdown", (fn,i)
            assert "**Coding step.**" in text(nb["cells"][i-1]), (fn,i,"missing coding-step markdown")
    all_source = "\n".join(text(c) for c in nb["cells"])
    assert "term-deposit-marketing-2020-labelled.csv" in all_source, fn
    assert "/mnt/data/" not in all_source, fn
    save(fn, nb)

assert "ax.margins(y=0.16)" in Path("01_data_eda.ipynb").read_text()
nb2_text = Path("02_baseline_tuning.ipynb").read_text()
assert "identical across feature conditions" in nb2_text
assert "320 candidate configurations" in nb2_text
assert "Tuned Logistic Regression winners" in nb2_text
assert "Tuned HistGradientBoosting winners" in nb2_text
nb3_text = Path("03_features_imbalance.ipynb").read_text()
assert "Logistic Regression Feature Importance" in nb3_text
assert "HistGradientBoosting Permutation Importance" in nb3_text
print("Step 1 notebook source fixes validated successfully.")
for fn in FILES:
    nb = load(fn)
    print(fn, "cells=",len(nb["cells"]), "markdown=",sum(c["cell_type"]=="markdown" for c in nb["cells"]), "code=",sum(c["cell_type"]=="code" for c in nb["cells"]))
