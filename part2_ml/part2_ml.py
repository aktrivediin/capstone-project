"""
Part 2 -- Supervised ML: Regression (Linear/Ridge) + Classification (Logistic Regression)
Reads: ../part1_eda/cleaned_data.csv
Run: python part2_ml.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (mean_squared_error, r2_score, confusion_matrix,
                              classification_report, roc_curve, roc_auc_score,
                              precision_score, recall_score, f1_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pd.set_option("display.width", 140)

def section(t):
    print("\n" + "=" * 90)
    print(t)
    print("=" * 90)

# ---------------------------------------------------------------------------
# Task 1: Load + define labels
# ---------------------------------------------------------------------------
section("TASK 1: Load data, define labels")
df = pd.read_csv("../part1_eda/cleaned_data.csv")
print(df.shape)

y_reg = df["W"]                                  # continuous regression target: season wins
y_clf = (y_reg > y_reg.median()).astype(int)      # binary classification target: above-median win team
X = df.drop(columns=["W", "ID", "Win_Tier"])      # drop target, ID, and Win_Tier (directly derived from W -> leakage)
print("y_reg (wins) describe:\n", y_reg.describe())
print("\ny_clf value counts:\n", y_clf.value_counts())
print("\nFeature columns:", list(X.columns))

# ---------------------------------------------------------------------------
# Task 2: Encode categorical columns
# ---------------------------------------------------------------------------
section("TASK 2: Encode categorical columns")
# Run_Diff_Type: nominal, no natural order -> one-hot encode, drop first to avoid multicollinearity
X = pd.get_dummies(X, columns=["Run_Diff_Type"], drop_first=True)
print("Columns after one-hot encoding Run_Diff_Type:", list(X.columns))
print("""
Justification: 'Run_Diff_Type' has two unordered categories (Run-Positive / Run-Negative).
One-hot encoding avoids implying a false ordinal relationship (e.g. label-encoding it as 0/1
would be mathematically fine for 2 categories, but for any nominal feature with >2 unordered
categories, label encoding would wrongly imply "category 2 is between category 1 and 3", which
a linear/logistic model would then treat as a meaningful numeric distance). One-hot avoids that.
Note: Win_Tier (the only ordinal categorical column) was dropped above since it is a direct
bucketed transform of the regression target W, and including it would leak the label.
""")

X = X.astype({c: float for c in X.select_dtypes(bool).columns})  # dummies -> float for scaler

# ---------------------------------------------------------------------------
# Task 3: Leak-free split + scaling
# ---------------------------------------------------------------------------
section("TASK 3: Train-test split + scaling")
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42
)
scaler = StandardScaler()
scaler.fit(X_train)  # fit ONLY on training data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Train shape:", X_train_scaled.shape, "Test shape:", X_test_scaled.shape)
print("""
NOTE on leakage: the scaler is fit ONLY on X_train. Fitting it on the full dataset (train+test)
would let the mean/std used for scaling be influenced by test-set values, silently leaking
information about the test distribution into every feature the model trains on -- an optimistic
bias in evaluation that would not hold up on truly unseen data.
""")

# ---------------------------------------------------------------------------
# Task 4: Regression models
# ---------------------------------------------------------------------------
section("TASK 4: Linear Regression")
lin = LinearRegression()
lin.fit(X_train_scaled, y_reg_train)
y_pred_reg = lin.predict(X_test_scaled)
mse_lin = mean_squared_error(y_reg_test, y_pred_reg)
r2_lin = r2_score(y_reg_test, y_pred_reg)
print(f"Linear Regression -> MSE: {mse_lin:.4f}  R2: {r2_lin:.4f}")

coef_df = pd.DataFrame({"feature": X.columns, "coef": lin.coef_}).sort_values("coef", key=np.abs, ascending=False)
print("\nCoefficients (sorted by |coef|):\n", coef_df)
print("\nTop 3 features by |coef|:\n", coef_df.head(3))

section("TASK 4 (Ridge): Ridge Regression")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_reg_train)
y_pred_ridge = ridge.predict(X_test_scaled)
mse_ridge = mean_squared_error(y_reg_test, y_pred_ridge)
r2_ridge = r2_score(y_reg_test, y_pred_ridge)
print(f"Ridge (alpha=1.0) -> MSE: {mse_ridge:.4f}  R2: {r2_ridge:.4f}")
print(f"\nComparison table:\n  Linear: MSE={mse_lin:.4f} R2={r2_lin:.4f}\n  Ridge:  MSE={mse_ridge:.4f} R2={r2_ridge:.4f}")

# ---------------------------------------------------------------------------
# Task 5: Classification - Logistic Regression
# ---------------------------------------------------------------------------
section("TASK 5: Logistic Regression - class balance check")
vc = y_clf_train.value_counts(normalize=True)
print(vc)
minority_frac = vc.min()
print(f"Minority class fraction: {minority_frac:.3f}")

use_class_weight = minority_frac < 0.35
if use_class_weight:
    print("Minority class < 35% -> using class_weight='balanced'.")
else:
    print("Classes are reasonably balanced (min class >= 35%) -> class_weight='balanced' still")
    print("applied defensively since it costs nothing when classes are already balanced; this")
    print("also documents the SMOTE alternative below for reference.")

# SMOTE alternative (commented -- requires `pip install imbalanced-learn`):
# from imblearn.over_sampling import SMOTE
# sm = SMOTE(random_state=42)
# X_train_scaled, y_clf_train = sm.fit_resample(X_train_scaled, y_clf_train)

clf = LogisticRegression(max_iter=1000, class_weight="balanced" if True else None, random_state=42)
clf.fit(X_train_scaled, y_clf_train)
y_pred_clf = clf.predict(X_test_scaled)
y_proba_clf = clf.predict_proba(X_test_scaled)[:, 1]

section("TASK 5: Logistic Regression - evaluation")
cm = confusion_matrix(y_clf_test, y_pred_clf)
print("Confusion matrix:\n", cm)
print("\nClassification report:\n", classification_report(y_clf_test, y_pred_clf))

fpr, tpr, thresholds = roc_curve(y_clf_test, y_proba_clf)
auc = roc_auc_score(y_clf_test, y_proba_clf)
print(f"AUC: {auc:.4f}")

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression (Above-median-wins classifier)")
plt.annotate(f"AUC = {auc:.3f}", xy=(0.6, 0.2))
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=110)
plt.close()
print("Saved roc_curve.png")

# ---------------------------------------------------------------------------
# Task 5b: Decision threshold sensitivity
# ---------------------------------------------------------------------------
section("TASK 5b: Decision-threshold sensitivity (0.30-0.70)")
rows = []
for t in [0.30, 0.40, 0.50, 0.60, 0.70]:
    pred_t = (y_proba_clf >= t).astype(int)
    p = precision_score(y_clf_test, pred_t, zero_division=0)
    r = recall_score(y_clf_test, pred_t, zero_division=0)
    f1 = f1_score(y_clf_test, pred_t, zero_division=0)
    rows.append({"Threshold": t, "Precision": p, "Recall": r, "F1": f1})
thresh_df = pd.DataFrame(rows)
print(thresh_df)
best_t = thresh_df.loc[thresh_df["F1"].idxmax(), "Threshold"]
print(f"\nThreshold maximizing F1: {best_t}")

# ---------------------------------------------------------------------------
# Task 6: Regularization experiment (C=0.01 vs C=1.0)
# ---------------------------------------------------------------------------
section("TASK 6: Regularization experiment")
clf_strong = LogisticRegression(C=0.01, max_iter=1000, class_weight="balanced", random_state=42)
clf_strong.fit(X_train_scaled, y_clf_train)
proba_strong = clf_strong.predict_proba(X_test_scaled)[:, 1]
pred_strong = clf_strong.predict(X_test_scaled)

p_base = precision_score(y_clf_test, y_pred_clf, zero_division=0)
r_base = recall_score(y_clf_test, y_pred_clf, zero_division=0)
auc_base = roc_auc_score(y_clf_test, y_proba_clf)

p_strong = precision_score(y_clf_test, pred_strong, zero_division=0)
r_strong = recall_score(y_clf_test, pred_strong, zero_division=0)
auc_strong = roc_auc_score(y_clf_test, proba_strong)

print(f"C=1.0   (baseline): Precision={p_base:.4f} Recall={r_base:.4f} AUC={auc_base:.4f}")
print(f"C=0.01  (strong L2): Precision={p_strong:.4f} Recall={r_strong:.4f} AUC={auc_strong:.4f}")

# ---------------------------------------------------------------------------
# Task 6b: Bootstrap CI for AUC difference
# ---------------------------------------------------------------------------
section("TASK 6b: Bootstrap 95% CI for AUC difference (C=1.0 minus C=0.01)")
np.random.seed(42)
y_clf_test_arr = y_clf_test.reset_index(drop=True).values
n = len(y_clf_test_arr)
diffs = []
for i in range(500):
    idx = np.random.choice(n, size=n, replace=True)
    yb = y_clf_test_arr[idx]
    if len(np.unique(yb)) < 2:
        continue  # skip degenerate bootstrap samples with only one class present
    auc_b1 = roc_auc_score(yb, y_proba_clf[idx])
    auc_b2 = roc_auc_score(yb, proba_strong[idx])
    diffs.append(auc_b1 - auc_b2)

diffs = np.array(diffs)
mean_diff = diffs.mean()
ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
print(f"Valid bootstrap samples used: {len(diffs)} / 500")
print(f"Mean AUC difference (C=1.0 - C=0.01): {mean_diff:.4f}")
print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"CI excludes zero: {ci_low > 0 or ci_high < 0}")

print("\nDONE.")
