"""
Part 3 -- Ensembles, Tuning, Full ML Pipeline
Reads: ../part1_eda/cleaned_data.csv
Run: python part3_ensembles.py
Outputs: best_model.pkl
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

pd.set_option("display.width", 140)

def section(t):
    print("\n" + "=" * 90)
    print(t)
    print("=" * 90)

# ---------------------------------------------------------------------------
# Rebuild the same leak-free split/scale/encode pipeline as Part 2
# ---------------------------------------------------------------------------
df = pd.read_csv("../part1_eda/cleaned_data.csv")
y_reg = df["W"]
y_clf = (y_reg > y_reg.median()).astype(int)
X = df.drop(columns=["W", "ID", "Win_Tier"])
X = pd.get_dummies(X, columns=["Run_Diff_Type"], drop_first=True)
X = X.astype({c: float for c in X.select_dtypes(bool).columns})

X_train, X_test, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# Task 1: Decision Tree baseline (unconstrained)
# ---------------------------------------------------------------------------
section("TASK 1: Unconstrained Decision Tree")
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train_scaled, y_clf_train)
train_acc_full = accuracy_score(y_clf_train, dt_full.predict(X_train_scaled))
test_acc_full = accuracy_score(y_clf_test, dt_full.predict(X_test_scaled))
print(f"Unconstrained tree -> train acc: {train_acc_full:.4f}  test acc: {test_acc_full:.4f}  gap: {train_acc_full-test_acc_full:.4f}")

# ---------------------------------------------------------------------------
# Task 2: Controlled Decision Tree
# ---------------------------------------------------------------------------
section("TASK 2: Controlled Decision Tree (max_depth=5, min_samples_split=20)")
dt_ctrl = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
dt_ctrl.fit(X_train_scaled, y_clf_train)
train_acc_ctrl = accuracy_score(y_clf_train, dt_ctrl.predict(X_train_scaled))
test_acc_ctrl = accuracy_score(y_clf_test, dt_ctrl.predict(X_test_scaled))
print(f"Controlled tree -> train acc: {train_acc_ctrl:.4f}  test acc: {test_acc_ctrl:.4f}  gap: {train_acc_ctrl-test_acc_ctrl:.4f}")

# ---------------------------------------------------------------------------
# Task 3: Gini vs Entropy
# ---------------------------------------------------------------------------
section("TASK 3: Gini vs Entropy (max_depth=5)")
dt_gini = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=42).fit(X_train_scaled, y_clf_train)
dt_entropy = DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=42).fit(X_train_scaled, y_clf_train)
acc_gini = accuracy_score(y_clf_test, dt_gini.predict(X_test_scaled))
acc_entropy = accuracy_score(y_clf_test, dt_entropy.predict(X_test_scaled))
print(f"Gini test acc: {acc_gini:.4f}   Entropy test acc: {acc_entropy:.4f}")

# ---------------------------------------------------------------------------
# Task 4: Random Forest
# ---------------------------------------------------------------------------
section("TASK 4: Random Forest")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train_scaled, y_clf_train)
rf_train_acc = accuracy_score(y_clf_train, rf.predict(X_train_scaled))
rf_test_acc = accuracy_score(y_clf_test, rf.predict(X_test_scaled))
rf_auc = roc_auc_score(y_clf_test, rf.predict_proba(X_test_scaled)[:, 1])
print(f"RF -> train acc: {rf_train_acc:.4f}  test acc: {rf_test_acc:.4f}  AUC: {rf_auc:.4f}")

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 5 features by importance:\n", importances.head(5))

# ---------------------------------------------------------------------------
# Task 4a: Gradient Boosting
# ---------------------------------------------------------------------------
section("TASK 4a: Gradient Boosting")
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train_scaled, y_clf_train)
gb_train_acc = accuracy_score(y_clf_train, gb.predict(X_train_scaled))
gb_test_acc = accuracy_score(y_clf_test, gb.predict(X_test_scaled))
gb_auc = roc_auc_score(y_clf_test, gb.predict_proba(X_test_scaled)[:, 1])
print(f"GB -> train acc: {gb_train_acc:.4f}  test acc: {gb_test_acc:.4f}  AUC: {gb_auc:.4f}")

# ---------------------------------------------------------------------------
# Task 4b: Feature ablation study
# ---------------------------------------------------------------------------
section("TASK 4b: Feature ablation study (drop 5 lowest-importance features)")
lowest5 = importances.sort_values(ascending=True).head(5).index.tolist()
print("5 lowest-importance features:", lowest5)

keep_cols = [c for c in X.columns if c not in lowest5]
X_train_reduced = pd.DataFrame(X_train_scaled, columns=X.columns)[keep_cols].values
X_test_reduced = pd.DataFrame(X_test_scaled, columns=X.columns)[keep_cols].values

rf_reduced = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_reduced.fit(X_train_reduced, y_clf_train)
auc_full = rf_auc
auc_reduced = roc_auc_score(y_clf_test, rf_reduced.predict_proba(X_test_reduced)[:, 1])
print(f"Full-model AUC: {auc_full:.4f}   Reduced-model AUC (5 features dropped): {auc_reduced:.4f}")

# ---------------------------------------------------------------------------
# Task 5: Cross-validated comparison
# ---------------------------------------------------------------------------
section("TASK 5: 5-fold cross-validated AUC comparison")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "DecisionTree(depth=5)": DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
}
cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_clf_train, cv=skf, scoring="roc_auc")
    cv_results[name] = (scores.mean(), scores.std())
    print(f"{name}: mean AUC = {scores.mean():.4f}  std = {scores.std():.4f}")

# ---------------------------------------------------------------------------
# Task 6: GridSearchCV
# ---------------------------------------------------------------------------
section("TASK 6: GridSearchCV on Random Forest pipeline")
pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RandomForestClassifier(random_state=42))
param_grid = {
    "randomforestclassifier__n_estimators": [50, 100, 200],
    "randomforestclassifier__max_depth": [5, 10, None],
    "randomforestclassifier__min_samples_leaf": [1, 5],
}
grid = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                     scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_clf_train)  # unscaled -- pipeline handles imputation + scaling internally
print("Best params:", grid.best_params_)
print("Best CV AUC score:", grid.best_score_)
n_configs = 1
for v in param_grid.values():
    n_configs *= len(v)
print(f"Total configurations evaluated: {n_configs} param combos x 5 folds = {n_configs*5} fits")

best_pipeline = grid.best_estimator_
best_test_auc = roc_auc_score(y_clf_test, best_pipeline.predict_proba(X_test)[:, 1])
print(f"Best pipeline test-set AUC: {best_test_auc:.4f}")

# ---------------------------------------------------------------------------
# Task 7: Manual learning curve
# ---------------------------------------------------------------------------
section("TASK 7: Manual learning curve (best pipeline, 20%-100% of training data)")
lc_rows = []
for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
    n_rows = int(frac * len(X_train))
    X_sub = X_train.iloc[:n_rows]
    y_sub = y_clf_train.iloc[:n_rows]
    pipe_f = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            RandomForestClassifier(random_state=42, **{k.split("__")[1]: v for k, v in grid.best_params_.items()}))
    pipe_f.fit(X_sub, y_sub)
    train_auc = roc_auc_score(y_sub, pipe_f.predict_proba(X_sub)[:, 1])
    test_auc = roc_auc_score(y_clf_test, pipe_f.predict_proba(X_test)[:, 1])
    lc_rows.append({"Training fraction": frac, "Training AUC": train_auc, "Test AUC": test_auc})
lc_df = pd.DataFrame(lc_rows)
print(lc_df)

# ---------------------------------------------------------------------------
# Task 8: Serialize best model
# ---------------------------------------------------------------------------
section("TASK 8: Serialize best model")
joblib.dump(best_pipeline, "best_model.pkl")
print("Saved best_model.pkl")

# Reload-and-predict demonstration
loaded = joblib.load("best_model.pkl")
sample_rows = X_test.iloc[:2]
preds = loaded.predict(sample_rows)
print("Reloaded model predictions on 2 hand-crafted/sample test rows:", preds)

# ---------------------------------------------------------------------------
# Task 9: Summary comparison table
# ---------------------------------------------------------------------------
section("TASK 9: Summary comparison table")
summary_rows = []
for name, (mean_auc, std_auc) in cv_results.items():
    summary_rows.append({"Model": name, "CV mean AUC": mean_auc, "CV std AUC": std_auc})
summary_rows.append({"Model": "RandomForest (GridSearchCV best)", "CV mean AUC": grid.best_score_, "CV std AUC": np.nan})
summary_df = pd.DataFrame(summary_rows)
print(summary_df)
print(f"\nBest pipeline held-out test AUC: {best_test_auc:.4f}")

print("\nDONE.")
