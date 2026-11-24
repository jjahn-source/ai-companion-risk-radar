# API Reference

Reference for the settings, functions and saved files in `notebooks/baseline_tfidf_logreg_updated.ipynb`.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `/content/drive/MyDrive/dataset/ai_subset_clean_sample_100_labeled_merged.csv` | Labeled CSV to train on. If it starts with `/content/drive`, Google Drive is mounted. |
| `OUT_DIR` | `/content/drive/MyDrive/dataset/results` | Folder for all outputs. Created if missing. |
| `N_REPEATS` | `3` | How many times the full cross-validation is repeated |

## Settings (top of the notebook)

| Name | Value | Description |
|---|---|---|
| `SEED` | `42` | Random seed. Repeat `r` uses `SEED + r`. |
| `N_SPLITS` | `5` | Outer CV folds |
| `INNER_SPLITS` | `3` | Inner folds for choosing `C` and the threshold |
| `C_GRID` | `[0.1, 0.3, 1, 3, 10, 30]` | Regularization values tried. Smaller = simpler model. |
| `MIN_POS_TO_TRAIN` | `5` | Themes with fewer positives are skipped |
| `MIN_POS_RELIABLE` | `15` | Themes with fewer positives are trained but marked unreliable |
| `CANDIDATES` | 5 TF-IDF configs | Settings compared in the feature-selection step |
| `COLUMN_ALIASES` | dict | Accepted label column names per theme |

## Functions

### `make_pipe(vec_kwargs, C=1.0) -> sklearn.pipeline.Pipeline`
Builds `TfidfVectorizer(**vec_kwargs)` followed by `LogisticRegression(C=C, class_weight="balanced", max_iter=5000)`. TF-IDF sits inside the pipeline so it is re-fit on each training fold and never sees test data.

### `make_splits(seed) -> iterator of (train_idx, test_idx)`
Outer CV splits. Uses `MultilabelStratifiedKFold` so every fold contains every theme, and falls back to `StratifiedKFold` if `iterative-stratification` is not installed.

### `score_setting(vec_kwargs) -> (mean_f1, mean_ap_lift)`
Scores one TF-IDF setting across the reliable themes with repeated CV. AP lift = average precision / base rate.

### `best_f1_threshold(y_true, proba) -> float`
Returns the probability cutoff that maximizes F1 on the given predictions.

### `tune(X_tr, y_tr, seed) -> (best_C, threshold)`
Inner CV on training data only. Picks `C` from `C_GRID` by average precision, then the F1-maximizing threshold for that `C`.

### `avg_confusion(c) -> np.ndarray (2x2)`
Confusion matrix for class `c`, averaged over CV repeats.

## Saved files

### `baseline_logreg_models.joblib`
```python
{
    "models":        {theme: Pipeline},   # fitted on all labeled data
    "thresholds":    {theme: float},      # median tuned threshold across folds
    "tfidf_setting": str,                 # key from CANDIDATES
}
```

Example:
```python
import joblib
bundle = joblib.load("results/baseline_logreg_models.joblib")

def predict_themes(texts):
    out = []
    for text in texts:
        row = {}
        for theme, model in bundle["models"].items():
            p = model.predict_proba([text])[0, 1]
            row[theme] = {"prob": round(float(p), 3), "flag": bool(p >= bundle["thresholds"][theme])}
        row["other_none"] = not any(v["flag"] for v in row.values())
        out.append(row)
    return out

predict_themes(["It's the only one I can talk to, I don't need my friends anymore"])
```

### CSV outputs

| File | Columns |
|---|---|
| `results_table.csv` | `class, n_reviews, base_rate, precision, recall, F1, AP, AUC, F1_std, dummy_F1, status` |
| `confusion_counts.csv` | `class, true positive, false positive, false negative, true negative` |
| `tfidf_settings_comparison.csv` | `setting, macro_F1, AP_lift, n_features` |
| `oof_predictions.csv` | `UID, review, app_name, rating, true_<class>, pred_<class>` |
