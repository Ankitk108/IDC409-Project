# Belle II Continuum Suppression

Binary classification of simulated Belle II signal events and continuum background
using event-level and event-shape features.

## Status

The instructor's private CSV is the primary dataset. Ingestion, training-only
EDA, training-defined baselines, and provisional validation for Logistic
Regression, Decision Tree, Linear SVM, and histogram Gradient Boosting BDT are
implemented. Training-only correlation analysis, PCA, tree importance, RFE,
full/reduced MLP validation, and an optional physics-grouped CNN are implemented
for the next contribution. The protocol must be frozen before final test
evaluation.

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
| Physics-grouped 1D CNN | 59 | 0.8534 | 0.8577 | 0.9270 |
| Logistic Regression | 59 | 0.8424 | 0.8475 | 0.9152 |
| Linear SVM | 59 | 0.8420 | 0.8480 | 0.9151 |
| Decision Tree | 59 | 0.8297 | 0.8366 | 0.9022 |

The BDT is the strongest tested model. RFE reduces the MLP input from 59 to 20
measured features while losing only 0.0017 ROC-AUC, supporting comparable—not
improved—reduced-feature performance. The CNN is a valid experimental sixth
family but does not outperform the BDT or either MLP. These conclusions apply to
one fixed split of the instructor-provided simulated dataset; they are not claims
of detector-data or cross-dataset generalization.

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
