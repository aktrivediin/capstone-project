# Part 2 — Supervised ML: Regression + Classification

**Run:** `python part2_ml.py` (reads `../part1_eda/cleaned_data.csv`)

## Label definitions

- **`y_reg`**: `W` (season wins), continuous.
- **`y_clf`**: `(W > W.median()).astype(int)` → 1 = above-median-wins ("winning") team, 0 = below.
  (Median win total ≈ 80; resulting split is 285 vs 296, i.e. already near-balanced.)
- **Features (`X`)**: all columns except `W`, `ID`, and `Win_Tier`. `Win_Tier` was dropped even
  though it's a valid categorical column, because it is a **direct bucketed transform of the
  target `W`** — including it would leak the label straight into the model.

## Task 2 — Encoding

`Run_Diff_Type` (nominal, 2 unordered categories: Run-Positive / Run-Negative) → one-hot encoded
with `drop_first=True` (produces `Run_Diff_Type_Run-Positive`). No ordinal categorical feature
remained in `X` after dropping the target-derived `Win_Tier`, so label encoding wasn't needed here.
**Why one-hot avoids the false-ordinal problem:** label encoding assigns integers (0, 1, 2, ...) to
categories, and a linear/logistic model reads that integer as a real numeric distance — for a
nominal feature with 3+ unordered categories this wrongly tells the model "category 2 is between
category 1 and category 3," a relationship that doesn't exist for e.g. arbitrary labels. One-hot
avoids this by giving each category an independent binary column with no implied ordering.

## Task 3 — Leakage-free split & scaling

`train_test_split(X, y, test_size=0.2, random_state=42)` → 464 train / 117 test rows.
`StandardScaler` was **fit only on `X_train`**, then used to `.transform()` both `X_train` and
`X_test`. Fitting the scaler on the combined dataset would bake test-set mean/variance into the
per-feature standardization used during training — a subtle leak that inflates evaluation metrics
because the model implicitly "sees" test-set statistics before ever being tested.

## Task 4 — Regression: Linear vs Ridge

| Model | MSE | R² |
|---|---|---|
| Linear Regression | 15.28 | 0.8895 |
| Ridge (α=1.0) | 15.34 | 0.8891 |

Top 3 |coefficient| features (Linear Regression): **OPS** (≈2.99M), **SLG** (≈−2.19M), **OBA**
(≈−0.97M). These numbers are extreme and **not directly interpretable** at face value — see the
multicollinearity note below.

**Coefficient interpretation (general rule):** for a scaled feature, a coefficient of +c means a
one-standard-deviation increase in that feature is associated with a c-unit increase in predicted
wins, holding other features constant; a negative coefficient means a one-std-dev increase is
associated with a c-unit *decrease* in predicted wins.

**Why the top-3 OLS coefficients are so large — and why Ridge differs:** `OPS = OBA + SLG` by
definition (on-base-plus-slugging), so `OPS`, `OBA`, and `SLG` are near-perfectly collinear
(pairwise correlations 0.77–0.98 in this dataset). Ordinary Least Squares has no way to split
credit stably among three columns that are almost linear combinations of one another, so it
assigns huge, sign-flipping coefficients to them that cancel out in combination (the net
prediction is fine — R²=0.89 — but each individual coefficient is unstable and not trustworthy in
isolation). **Ridge's L2 penalty shrinks coefficient magnitude** as a direct function of the
`alpha` parameter (`alpha` controls the strength of the penalty added to the loss:
`loss = SSE + alpha * sum(coef^2)`), which is exactly why Ridge produces a completely different,
far more sane coefficient profile for this trio — under Ridge, `OPS`, `OBA`, and `SLG` each land
around **+1.0**, a stable, mutually-consistent answer, while `RA` (runs allowed, ≈−9.8) and `G`/`R`
(≈+9.1/+7.5) become the dominant, trustworthy predictors. This is the practical justification for
preferring Ridge's coefficients for interpretation, even though both models have nearly identical
predictive accuracy (MSE/R²) on this dataset.

## Task 5 — Classification: Logistic Regression

Class balance check: minority class = 49.4% of training rows — already balanced, so
`class_weight='balanced'` was applied defensively (a no-cost safeguard) rather than because it was
strictly required by the ≤35% rule; SMOTE is documented as a commented-out alternative in the code
(`imblearn.over_sampling.SMOTE`) for datasets where imbalance is more severe.

**Confusion matrix:**
```
[[63  4]
 [ 8 42]]
```
**Classification report:** precision 0.89 (class 0) / 0.91 (class 1), recall 0.94 / 0.84,
F1 0.91 / 0.88, overall accuracy **90%**.

**ROC / AUC:** AUC = **0.963** (`roc_curve.png`). An AUC of 0.963 means that, for a randomly chosen
above-median-win team and a randomly chosen below-median-win team, the model ranks the winning
team's predicted probability higher **96.3% of the time** — very strong class separation.

**Formulas:**
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)

**Which matters more here:** for a "should we flag this team as a likely winning season" use case,
**precision and recall are roughly equally important** — there's no obviously asymmetric cost (a
false positive isn't clearly worse than a false negative in a descriptive sports-analytics
context, unlike e.g. medical screening). If this fed a betting/investment decision where acting on
a false "winning team" prediction is costly, **precision** would be prioritized; if it fed a scouting
alert system where missing a genuine winning team is costly, **recall** would be prioritized.

## Task 5b — Decision-threshold sensitivity

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.898 | 0.880 | 0.889 |
| 0.40 | 0.915 | 0.860 | 0.887 |
| 0.50 | 0.913 | 0.840 | 0.875 |
| 0.60 | 0.913 | 0.840 | 0.875 |
| 0.70 | 0.913 | 0.840 | 0.875 |

**F1-maximizing threshold: 0.30** (F1 = 0.889). Formulas: Precision = TP/(TP+FP),
Recall = TP/(TP+FN). Given the roughly-symmetric cost structure discussed above, **recall is
slightly favored** in this dataset if the goal is not to miss genuinely strong teams, which argues
for **lowering** the threshold from the default 0.5 toward 0.3 — the cost of doing so is accepting
more false positives (teams flagged as "winning" that end up below median), trading some precision
for higher recall.

## Task 6 — Regularization experiment (C=1.0 vs C=0.01)

| C | Precision | Recall | AUC |
|---|---|---|---|
| 1.0 (baseline) | 0.913 | 0.840 | 0.963 |
| 0.01 (strong L2) | 0.913 | 0.840 | 0.928 |

`C` is the **inverse** of regularization strength in scikit-learn's `LogisticRegression`
(`C = 1/λ`) — a small `C` means a large penalty on coefficient magnitude, forcing the model toward
simpler, flatter decision boundaries. Here, **reducing C to 0.01 measurably hurt AUC** (0.963 →
0.928) while leaving the default-threshold precision/recall unchanged; strong regularization
under-fit this particular feature set slightly, suggesting the baseline `C=1.0` model was not
overfitting badly to begin with (581 rows, 23 features).

## Task 6b — Bootstrap 95% CI for the AUC difference

500 bootstrap resamples of the test set (`np.random.choice(..., replace=True)`, seed 42), AUC
difference = AUC(C=1.0) − AUC(C=0.01) computed per resample:

- **Mean AUC difference:** 0.0351
- **95% CI:** [0.0071, 0.0662]
- **Excludes zero:** **Yes**

Since the interval does not contain 0, the `C=1.0` model's AUC advantage over `C=0.01` is
**statistically reliable across resamples of this test set**, not just an artifact of one
particular train/test split.

## Acceptance checklist
- [x] MSE, R², Ridge vs Linear table, coefficient interpretation
- [x] Confusion matrix, classification report, ROC curve (`roc_curve.png`), AUC
- [x] Class-imbalance check performed (documented balanced-by-default result + SMOTE alternative)
- [x] Threshold sensitivity table (0.30–0.70) + formulas + F1-max threshold + justification
- [x] C=1.0 vs C=0.01 comparison + explanation of C
- [x] 500-sample bootstrap AUC-difference CI, reported and interpreted
