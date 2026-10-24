# Belle II Continuum Suppression

Binary classification of simulated Belle II signal events and continuum background
using event-level and event-shape features.

## Status

The instructor's private CSV is the primary dataset. Ingestion, training-only
EDA, training-defined baselines, and provisional validation for Logistic
Regression, Decision Tree, Linear SVM, and histogram Gradient Boosting BDT are
implemented. Training-only correlation analysis, PCA, tree importance, RFE,
full/reduced MLP validation, and an optional physics-grouped CNN are implemented
and the frozen protocol has been evaluated once on held-out test data.

## Planned classifiers

- Logistic Regression
- Decision Tree
- Support Vector Machine
- Gradient Boosting BDT
- Feed-forward neural network (MLP)
- Experimental CNN, subject to a defensible feature representation

Raw data, generated SQLite databases, trained models, and internal planning documents
are deliberately excluded from Git.

Every model uses the same persisted, label-stratified 70/15/15 assignment: training
for fitting, validation for tuning and selection, and test for final evaluation only.

Install the core environment with:

```powershell
python -m pip install -r requirements.txt
```

The CNN requires the larger optional PyTorch dependency. Install it separately
only when running that model:

```powershell
python -m pip install -r requirements-cnn.txt
```

Without PyTorch, the notebook reports that the CNN was skipped and continues with
all core stages. The CNN uses convolutions only for ordered CLEO-cone and KSFW
moment families; it does not treat arbitrary table-column order as spatial data.

## Frozen final evaluation

After validation choices are frozen, the held-out test is evaluated through one
guarded command:

```powershell
python -m scripts.run_final_evaluation
```

The command refits the frozen trainable configurations on combined training and
validation rows, writes the final metrics and RFE-20 feature list under
`results/tables/`, and writes ROC/confusion-matrix figures under
`results/figures/`. It refuses to overwrite existing final artifacts. Ordinary
notebook execution only displays those saved files and therefore does not reopen
test repeatedly.

## Final held-out results

The frozen protocol was evaluated once on 10,591 test events after refitting on
60,015 combined training and validation events.

| Model | Features | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Histogram Gradient Boosting BDT | 59 | 0.8784 | 0.8815 | **0.9457** |
| MLP (full) | 59 | 0.8740 | 0.8783 | 0.9416 |
| MLP (RFE-20) | 20 | 0.8722 | 0.8772 | 0.9399 |
| Physics-grouped 1D CNN | 59 | 0.8540 | 0.8584 | 0.9273 |
| Logistic Regression | 59 | 0.8420 | 0.8471 | 0.9151 |
| Linear SVM | 59 | 0.8420 | 0.8480 | 0.9151 |
| Decision Tree | 59 | 0.8297 | 0.8366 | 0.9022 |

The BDT is the strongest tested model. RFE reduces the MLP input from 59 to 20
measured features while losing only 0.0017 ROC-AUC, supporting comparable—not
improved—reduced-feature performance. The CNN is a valid experimental sixth
family but does not outperform the BDT or either MLP. These conclusions apply to
one fixed split of the instructor-provided simulated dataset; they are not claims
of detector-data or cross-dataset generalization.

## Post-hoc extensions

The additional model is deliberately separated from the frozen held-out
comparison. The extension runner evaluates one binary Random Forest on the frozen
RFE-20 feature set using five-fold out-of-fold predictions on the original
train+validation rows only. Original test rows are excluded.

```powershell
python -m scripts.run_posthoc_extensions
```

The runner reports the five binary metrics plus an out-of-fold ROC curve and
confusion matrix. Outputs use `posthoc_*` filenames under `results/tables/` and
`results/figures/`; they must not be inserted into the frozen test ranking because
the evaluation protocol differs.

The completed development-only result uses 60,015 out-of-fold predictions and 20
features: accuracy 0.8595, precision 0.8513, recall 0.8756, F1 0.8633, and ROC-AUC
0.9332. It demonstrates useful separation with the reduced feature set, but it is
not evidence that Random Forest outranks a model measured on the frozen test set.

The strongest frozen-test family is also studied under a matched post-hoc
feature-reduction protocol. This runner evaluates the unchanged histogram BDT
on all 59 measurements and the frozen RFE-20 subset using identical five-fold
development-only out-of-fold predictions:

```powershell
python -m scripts.run_reduced_bdt_comparison
```

It writes `posthoc_bdt_feature_comparison_cv.csv` and
`posthoc_bdt_feature_comparison.png`. The original test rows remain excluded;
therefore these results measure the development-CV effect of reducing BDT inputs
and do not replace or extend the frozen test ranking.

Across 60,015 matched out-of-fold predictions, the full BDT reaches accuracy
0.8773, F1 0.8804, and ROC-AUC 0.9481; the RFE-20 BDT reaches accuracy 0.8690,
F1 0.8726, and ROC-AUC 0.9419. Removing 39 measurements therefore costs about
0.0083 accuracy and 0.0062 ROC-AUC while retaining most discrimination. This is
a measured simplicity/performance trade-off, not evidence that reduction improves
the BDT.

A final neural-capacity extension uses the frozen RFE-20 measurements in a deeper
PyTorch dense network: `20 → 128 → 64 → 1`, with LayerNorm, ReLU, dropout,
AdamW, and fold-internal early stopping. It produces five-fold development-only
OOF predictions and never reads original test rows:

```powershell
python -m scripts.run_deep_dense_extension
```

Install `requirements-cnn.txt` for the optional PyTorch dependency. The guarded
runner writes separate `posthoc_dense_*` artifacts; results remain exploratory
and outside the frozen test ranking.

The completed dense network reaches accuracy 0.8725, precision 0.8549, recall
0.9012, F1 0.8774, and ROC-AUC 0.9432 across 60,015 OOF predictions. Its high
recall comes with 4,650 false-positive backgrounds and 3,004 missed signals.
The selected fold epochs `(44, 58, 59, 60, 56)` average 55.4; because two folds
approach the fixed 60-epoch ceiling, incomplete convergence is a limitation, not
a reason to retune after inspecting the result.

## Seed robustness of the representative best models

The final development-only robustness study compares the full BDT and deep dense
RFE-20 network across ten complete five-fold OOF repetitions. The fixed seeds are
42, 123, 2026, 7, 19, 73, 314, 2718, 4096, and 10007. Both fold construction and
estimator/training randomness vary; original test rows remain excluded.

```powershell
python -m scripts.run_seed_robustness
```

The guarded runner performs 100 total fits and stores every seed-level result, an
across-seed mean/sample-standard-deviation/minimum/maximum table, and a diagnostic
figure. It requires optional PyTorch dependencies and is intentionally separate
from the frozen test ranking. Progress is checkpointed after each model-seed result
under ignored `data/interim/`, so rerunning the same command resumes missing work.

Across ten complete OOF repetitions, the full BDT reaches ROC-AUC
`0.948093 ± 0.000194` and F1 `0.879696 ± 0.000522`. The deep dense RFE-20 network
reaches ROC-AUC `0.943377 ± 0.000460` and F1 `0.876410 ± 0.000911`. The narrow
ranges describe sensitivity to these ten fold/training seeds on the same finite
dataset. They are not confidence intervals or estimates of future-data uncertainty.
The ordering is unchanged across these runs, but that alone does not establish a
population-level difference.

## Linear Discriminant Analysis study

Study D adds a transparent linear baseline beyond Logistic Regression by fitting
shrinkage Linear Discriminant Analysis on all 59 measurements and the frozen
RFE-20 subset. Scaling and covariance estimation occur inside each fold. Both
variants use the same ten-seed, five-fold development-only OOF protocol; original
test rows remain excluded.

```powershell
python -m scripts.run_lda_study
```

The guarded runner writes seed-level and across-seed metric tables, ROC/confusion
diagnostics, and signal/background distributions along the single discriminant
axis available in a binary LDA problem.

Across ten seeds, full LDA reaches ROC-AUC `0.899909 ± 0.000061`, while RFE-20
LDA reaches `0.894893 ± 0.000090`. Full LDA ranks events better overall. RFE-20
uses a more signal-efficient operating point—recall `0.874056` versus `0.829237`—
so its F1 is higher (`0.829851` versus `0.824490`) despite lower accuracy,
precision, and ROC-AUC. This threshold trade-off does not make reduced LDA the
stronger ranker.

## Uncertainty, operating points, and nested reduction

The first stage reconstructs the historical fixed-test models only to calculate
paired stratified-bootstrap ROC-AUC intervals, AUC-difference intervals, and
interpolated operating points at exactly 90% and 95% signal efficiency. It reports
background efficiency, rejected-background fraction, and inverse rejection factor
with distinct names and bootstrap intervals. It also writes generator-process
acceptance, process-versus-signal AUC, exact duplicate-feature columns, and feature
zero fractions. ROC interpolation represents randomized operation between adjacent
thresholds; the uninformative majority baseline has no headline rejection entry.
The audit does not select a new model or threshold using the test set.

```powershell
python -m scripts.run_methodology_correction --stage audit
```

The longer second stage excludes original test rows. It uses nested outer/inner
cross-validation, a limited four-candidate search per classical family, fold-local
BDT permutation selection, and fold-local PCA retaining 95% variance. It reports
mean and spread of fold-level AUC, avoiding pooled SVM decision-score scale effects,
plus selection frequencies and pairwise Jaccard overlap. The four-candidate search
does not equalize effort across untuned neural, Random Forest, CNN, or LDA models.

```powershell
python -m scripts.run_methodology_correction --stage nested
```

A third stage adds a quantile-binned calibration curve on the recorded test split
and a three-fold development learning curve for the representative BDT:

```powershell
python -m scripts.run_methodology_correction --stage diagnostics
```

Commit `20d0b84` is the first Git record containing the original final artifacts.
It documents when those files entered the repository; it is not proof that the
protocol was preregistered or that the test was opened only once. The grouped CNN
is retained as a negative result: CLEO cones have a physical angular order, while
the KSFW-moment order gives convolution only a weak inductive justification.

The deeper dense network used an inner 10% early-stopping split within every outer
training fold. Four of five selected epochs `(44, 58, 59, 60, 56)` are at or within
four epochs of the 60-epoch cap, so its comparison with BDT is convergence-limited.
The project uses scikit-learn histogram gradient boosting as a portable BDT-family
benchmark; it does not claim to reproduce Belle II FastBDT.

Accuracy, precision, recall, and F1 remain course deliverables and appendix metrics, but they reflect
the supplied nearly balanced sample. The source contains no validated luminosity
or event-weight column, so the project does not invent a physical class prior or
claim luminosity-weighted significance. The public notebook reads committed CSV
artifacts and generates its headline table, abstract result, and conclusion from them.

The instructor sample has an anomalous generator composition: only 462 charged-B
rows versus 35,303 neutral-B rows, and charm dominates the continuum subset. This
does not resemble physical production fractions and needs instructor confirmation;
the 80-row charged-B test acceptance is descriptive only. The file also lacks
`deltaE` and `Mbc`, so fit-variable decorrelation/sculpting cannot be checked.
Exact duplicate and zero-inflated features are disclosed rather than silently
transformed; log and zero-indicator transforms require a future fold-safe study.
The audit finds two exact duplicate pairs:
`KSFWVariables__bohoo1__cmFS1__bc` duplicates `KSFWVariables__bohoo1__bc`, and
`KSFWVariables__bohoo3__cmFS1__bc` duplicates `KSFWVariables__bohoo3__bc`.
Sixteen measurements are selected in all five outer folds; pairwise Jaccard
overlap for the complete 20-feature selections ranges from 0.739 to 0.905, so
the report describes a stable core rather than one universally stable subset.

## Data source

Place the instructor-provided `data_hep - data_hep.csv` in `data/raw/` without
modifying it. Its `type` flags 0 and 1 are B-pair signal (`target=1`); flags 2,
3, 4, and 5 are c/u/d/s continuum background (`target=0`). The public notebook
`notebooks/continuum_suppression.ipynb` reads that exact CSV, creates a local
SQLite database, and reuses one persisted train/validation/test split. Neither
the CSV nor the database is committed.

## Optional Belle II MDST preparation (separate from the primary notebook)

`scripts/generate_belle2_ntuple.py` requires a configured Belle II `basf2`
environment and an MDST file for each invocation. If the separately acquired
official example MDSTs are placed in `data/raw/`, run signal and continuum jobs
separately:

```bash
basf2 scripts/generate_belle2_ntuple.py -- --input data/raw/Bd2K0spi0_to_train.root --output data/interim/signal_example.root --label 1
basf2 scripts/generate_belle2_ntuple.py -- --input data/raw/ccbar_sample_to_train.root --output data/interim/background_example.root --label 0
```

The script processes **one input MDST per call** and writes continuum-suppression
measurements plus a constant binary `target` branch. Its feature schema includes
`R2`, `thrustBm`, `thrustOm`, KSFW moments, and CLEO cones; it does **not** emit
the professor CSV's `thrust` or `foxWolframR2` columns. It does not write
`isNotContinuumEvent`, combine the two inputs, or feed the primary notebook
automatically. Inspect the output schemas and candidate/event diversity before
any separate study; do not concatenate these example rows with the professor
CSV. Generated ROOT files are private and ignored by Git.
