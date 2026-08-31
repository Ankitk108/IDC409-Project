# Belle II Continuum Suppression

Binary classification of simulated Belle II signal events and continuum background
using event-level and event-shape features.

## Status

The repository structure and reproducibility contracts are initialized. Data
ingestion, EDA, and model implementation begin after the privately supplied dataset
is placed in `data/raw/`.

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

Prefer the analysis-level CSV supplied by the instructor. If it is unavailable, use
the official Belle II starterkit MC fallback from a configured `basf2` environment:

```bash
basf2 scripts/generate_belle2_ntuple.py --output belle2_continuum.root
```

The script combines the official `B02ks0pi0`, `uubar`, and `ccbar` examples and
writes standard continuum-suppression features plus `isNotContinuumEvent`. Generated
ROOT/CSV data is private and ignored by Git. Conversion to the canonical CSV is done
only after the generated branch schema has been inspected.
