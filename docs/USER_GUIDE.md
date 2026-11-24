# User Guide

This guide walks through the full workflow: get the data, build the AI-app subset, label reviews, train the model, and read the results.

## 1. Get the data

| Dataset | Where | Used for |
|---|---|---|
| MHARD (~200k Google Play reviews, 73 apps) | [Sensify-Lab/MHARD](https://github.com/Sensify-Lab/MHARD) | Main review text |
| Labeled pilot sample (100 reviews) | `data/ai_subset_clean_sample_100_labeled_merged.csv` | Training and evaluation |
| AI-Mediated Mental Health Support on Reddit | [Zenodo 18751157](https://zenodo.org/records/18751157) | Reference for the theme taxonomy only (no raw text) |

The CSVs in this repo are stored with Git LFS. Run `git lfs install` before cloning and `git lfs pull` after, or the files will be small pointer files instead of data.

## 2. Build the AI-app subset

Open `data/data_exploration.ipynb` with `MHARD_dataset.csv` in the same folder and run all cells. It:

1. Reports missing values, duplicates, rating and date distributions, and review lengths
2. Measures how often each app's reviews mention *bot / chatbot / AI / robot*
3. Keeps Tier 1 (Wysa, Woebot, Youper) and Tier 2 (Amaha, Happify, Healthily, VOS, Voidpet, Heyy) apps
4. Drops empty reviews, duplicate app+review text, and very short reviews
5. Builds a labeling sample: up to 30 keyword candidates per theme plus 80 random reviews (seed 42)

Keyword hits are only used to choose which reviews to label. They are never labels.

## 3. Label reviews

Follow [CODEBOOK.md](../CODEBOOK.md). The short version:

- Label the raw `review` text, not `review_cleaned`, and ignore the star rating and developer response
- Mark each theme 1 or 0. A review can have several themes. Other/None = 1 only if all themes are 0
- **General praise is Other/None**, even when strong ("this app helps me so much")
- Write a short reason in `Notes` whenever you're unsure

Label columns the training notebook accepts (either format works):

| Theme | Column name | Values |
|---|---|---|
| Dependence | `dependence` or `label_dependence` | TRUE/FALSE, Yes/blank, 1/0 |
| Attachment | `attachment` or `label_attachment` | same |
| Isolation / Displacement | `isolation` or `label_isolation_displacement` | same |
| Crisis-Response Frustration | `crisis` or `label_crisis_response_frustration` | same |

The file also needs a `review` column. `UID`, `app_name` and `rating` are optional but kept in the outputs if present.

To check agreement between two labelers:
```python
from sklearn.metrics import cohen_kappa_score
cohen_kappa_score(df.labeler_A_primary, df.labeler_B_primary)
```
Below 0.4 means the codebook needs work. 0.6 or higher is solid.

## 4. Train and evaluate

```bash
export DATA_PATH=data/ai_subset_clean_sample_100_labeled_merged.csv
export OUT_DIR=results
export N_REPEATS=3
jupyter notebook notebooks/baseline_tfidf_logreg_updated.ipynb
```

In Colab, leave the defaults and put the CSV at `MyDrive/dataset/`. The notebook mounts Drive automatically.

The notebook runs in this order:
1. Load labels and count positives per theme. Themes with fewer than 5 positives are skipped. Themes with fewer than 15 are trained but flagged as unreliable.
2. Compare five TF-IDF settings with repeated cross-validation and pick the best
3. Train one logistic regression per theme, tuning `C` and the decision threshold inside each training fold
4. Report precision, recall, F1, AP and AUC per theme, macro F1, and confusion matrices
5. Fit final models on all data, plot the top words, and save everything to `OUT_DIR`

## 5. Read the results

- **Compare F1 with `dummy_F1`.** `dummy_F1` is what you'd get by flagging every review. A useful model beats it.
- **Compare AP with `base_rate`.** AP equal to the base rate means the model ranks no better than chance.
- **`F1_std`** shows how much results change between CV repeats. Large values mean too little data.
- **`oof_predictions.csv`** lists every review with its true and predicted labels. Filter for `true_x == 1` and `pred_x == 0` to read the misses and improve the codebook or features.

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `pd.read_csv` returns 3 rows starting with `version https://git-lfs` | Run `git lfs pull` |
| `no column found for dependence` | Rename your label columns to one of the names in section 3 |
| `unexpected values in ...` | Label values must be TRUE/FALSE, YES/NO, 1/0 or blank |
| `Same folds for every theme: False` | `pip install iterative-stratification` |
