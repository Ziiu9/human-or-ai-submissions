# Group 15 — Human or AI?

**Team members:** Jingyu Zhang (jz2523), [Xinyu Zhang] ([x2225]), [name] ([netid])
**Kaggle team name:** `group15`

## Setup

Python version: 3.12.7

```bash
pip install -r requirements.txt
```

Place the Kaggle data files (`train.csv`, `unlabeled.csv`, `test.csv`, `sample_submission.csv`) in `./data/` (not committed).

Run every command below from this folder (`submissions/group15/`).

## Train / validate

All models use the same fixed validation set: 2,000 rows from `train.csv`, stratified by label, ids saved in `splits/val_ids.txt`. Models are trained on the other 8,000 rows and scored with `common.evaluate` (train AUROC, validation AUROC, fit time). Hyperparameters are picked on validation AUROC only.

```bash
# data checks
python eda.py --data data

# Model A: word TF-IDF + logistic regression
python experiments/model_a.py            # chosen settings
python experiments/model_a.py --tune     # 32-setting grid -> experiments/model_a_tuning.txt

# Model B: [teammate]
# Model C: [teammate]

# Q2 learning curve -> experiments/learning_curve.txt, figures/learning_curve.png
python experiments/learning_curve.py

# Q3 unlabeled data: [teammate]
# Q4 error analysis: [teammate]
```

## Predict (writes Kaggle submission)

```bash
# [final command once train.py / predict.py are written]
```

For now, Model A's submission can be made with:

```bash
python experiments/model_a.py --submit   # retrains on all 10,000 rows, writes submission_model_a.csv
```

## Seeds and notes

- Global seed: `20261023` (`common.SEED`), used for the validation split, learning-curve subsampling and every model's `random_state`
- Validation: stratified 2,000-row holdout from `train.csv`, fixed across all comparisons
- Saved predictions: `preds/<model>_val.npy` (validation rows, in `val_ids.txt` order) and `preds/<model>_test.npy` (in `test.csv` order)

## AI assistance

- Jingyu: used Claude to explain course concepts and to draft `eda.py`, `common.py`, `experiments/model_a.py` and `experiments/learning_curve.py`. Ran every script, checked the outputs, and made the design choices (keeping case and punctuation, the fixed validation protocol, choosing C=10).
- [teammate]:
- [teammate]:

No AI tool or external API was used to label or predict the competition passages.
