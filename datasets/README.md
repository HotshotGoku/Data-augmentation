# datasets/ — data builders, onboarding, and per-dataset results
One folder per domain: `rattray/` `multiplexed/` `generalist/` `kl_2species/` `branching/` `wider/`
(each holds its `build_*_pairs`, pilot/eval shells, and per-dataset result tables).
- `preprocess/` — raw→256px canonicalization notebooks + experiment-seed prep.
- `harness/` — the generic `ds_*` driver (train→gen→downstream→analyze for any dataset/backbone).
- `adapt_curve/` — data-efficiency curves (quality vs #pairs) + subset builder.
Dataset definitions & exact composition: `docs/MODELS_AND_DATASETS.md`, `docs/DATASETS.md`.
