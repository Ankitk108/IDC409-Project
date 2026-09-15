# Belle II Continuum Suppression

Binary classification of simulated Belle II signal events and continuum background
using event-level and event-shape features.

## Status

The instructor's private CSV is the primary dataset. Ingestion, training-only
EDA, and provisional validation for Logistic Regression and Decision Tree are
implemented. Shared baselines, the other four agreed models, feature reduction,
and final test evaluation remain team work.

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
