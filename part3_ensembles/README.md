# Part 3 — Ensembles, Tuning, Full ML Pipeline

**Run:** `python part3_ensembles.py` (reads `../part1_eda/cleaned_data.csv`, writes `best_model.pkl`)

Uses the same leak-free split/encode/scale setup as Part 2 (same `random_state=42`), rebuilt at the
top of the script so this Part is independently runnable.

## Task 1 — Unconstrained Decision Tree

Train acc: **1.0000**, Test acc: **0.8718**, gap: **0.128**. The tree perfectly memorizes the
training set (100% train accuracy is a dead giveaway) and loses ~13 points of accuracy on unseen
data — classic overfitting. Decision trees are high-variance because each split is chosen greedily
to best separate the *current* node's training rows, with no mechanism to revisit or regularize
earlier splits, so with no depth limit the tree keeps splitting until it isolates individual
training points, memorizing noise along with signal.

## Task 2 — Controlled Decision Tree (`max_depth=5`, `min_samples_split=20`)

Train acc: **0.9461**, Test acc: **0.8803**, gap: **0.066** — roughly half the overfitting gap of
the unconstrained tree. `max_depth` caps how many sequential splits a path can have, directly
limiting how finely the tree can carve up the feature space (trading some bias for much lower
variance). `min_samples_split=20` blocks any split on a node with fewer than 20 samples, preventing
the tree from creating rules that respond to a handful of possibly-noisy training rows.

## Task 3 — Gini vs Entropy (`max_depth=5`)

Gini test accuracy: **0.8974**. Entropy test accuracy: **0.8974** — identical on this dataset (a
common outcome; the two criteria usually produce very similar trees).

- **Gini impurity:** `1 − Σ pᵢ²`
- **Entropy:** `−Σ pᵢ log₂(pᵢ)`

A node with **Gini = 0** is perfectly pure — every sample at that node belongs to a single class,
so there is no impurity left to reduce by splitting further.

## Task 4 — Random Forest (`n_estimators=100`, `max_depth=10`)

Train acc: **1.0000**, Test acc: **0.8974**, AUC: **0.9367**.

Top 5 features by importance:

| Feature | Importance |
|---|---|
| Run_Diff_Type_Run-Positive | 0.319 |
| RA (runs allowed) | 0.125 |
| R (runs scored) | 0.069 |
| H (hits) | 0.053 |
| OPS | 0.046 |

**How Random Forest computes feature importance:** for each split that uses a given feature across
every tree in the forest, the reduction in Gini impurity produced by that split is recorded and
weighted by how many samples pass through the node; these reductions are averaged across all
trees to produce the final importance score. This differs fundamentally from a linear regression
coefficient, which represents a *linear, additive, sign-and-magnitude* effect on the raw
prediction scale — feature importance is always non-negative and only reflects how *useful* a
feature was for splitting, with no information about the direction of its effect on the target.

**Bagging concept:** each of the 100 trees is trained on a *bootstrap sample* — a random sample of
rows drawn with replacement from the training set, the same size as the original but with some
rows repeated and others omitted. At every split, only a random subset of ≈√(n_features) columns is
even considered as a candidate. Both randomizations decorrelate the individual trees, so their
errors are less likely to point in the same direction; averaging (or majority-voting) across many
decorrelated trees cancels out much of the variance any single deep tree would otherwise show,
without needing to constrain any individual tree's depth as aggressively as in Task 2.

## Task 4a — Gradient Boosting

Train acc: **1.0000**, Test acc: **0.8803**, AUC: **0.9424** — slightly higher AUC than the
single Random Forest, consistent with boosting's sequential error-correction typically edging out
bagging on tabular data of this size.

## Task 4b — Feature ablation study

5 lowest-importance features (from the Task 4 Random Forest): **SF, SO, HBP, CS, 3B**.

| Model | Test AUC |
|---|---|
| Full model (all features) | 0.9367 |
| Reduced model (5 lowest-importance features removed) | 0.9439 |

AUC did not drop after removing the 5 lowest-importance features — **it actually improved
slightly**, suggesting these five columns were contributing little genuine signal and possibly a
small amount of noise that the full model had to "work around." **Production implication:** a
simpler model built on 5 fewer features has lower inference cost, a smaller feature-engineering/
monitoring surface, and (here) no accuracy penalty — this is exactly the situation where shipping
the reduced model is an easy call. In general this trade-off is only acceptable when the AUC drop
(if any) stays below whatever tolerance the business use case allows; here there was no drop at all.

## Task 5 — 5-fold cross-validated AUC comparison

| Model | CV mean AUC | CV std AUC |
|---|---|---|
| LogisticRegression | 0.9720 | 0.0098 |
| DecisionTree (depth=5) | 0.9217 | 0.0288 |
| RandomForest | 0.9705 | 0.0105 |
| GradientBoosting | 0.9648 | 0.0145 |

Cross-validation is more reliable than a single train/test split because it evaluates the model on
**5 different held-out folds** and averages the result — a single split's score can be optimistic
or pessimistic purely by chance (which particular rows happened to land in the test set), while the
5-fold mean/std gives both a more stable point estimate and a direct measure (the std) of how much
that estimate varies across different partitions of the same data.

## Task 6 — GridSearchCV (Random Forest pipeline)

Pipeline: `SimpleImputer(median) → StandardScaler → RandomForestClassifier`. Grid:
`n_estimators ∈ {50,100,200} × max_depth ∈ {5,10,None} × min_samples_leaf ∈ {1,5}` = **18
configurations × 5 folds = 90 total model fits**.

**Best params:** `max_depth=5, min_samples_leaf=5, n_estimators=200`
**Best CV AUC:** **0.9738**
**Held-out test-set AUC of the best pipeline:** **0.9400**

**Grid vs Randomized Search trade-off:** Grid Search exhaustively evaluates every combination in
the grid, guaranteeing the best combination *within that grid* is found, but its cost grows
multiplicatively with the number of hyperparameters and values (90 fits here, and this was a small
grid) — it becomes computationally infeasible for larger grids or more hyperparameters. Randomized
Search instead samples a fixed number of random combinations from the specified distributions,
trading a guarantee of exhaustiveness for the ability to cover a much larger hyperparameter space
in a fixed compute budget.

## Task 7 — Manual learning curve (best pipeline)

| Training fraction | Training AUC | Test AUC |
|---|---|---|
| 0.2 | 0.9990 | 0.8907 |
| 0.4 | 0.9960 | 0.9182 |
| 0.6 | 0.9955 | 0.9263 |
| 0.8 | 0.9957 | 0.9388 |
| 1.0 | 0.9922 | 0.9400 |

- **(i) Training AUC** stays essentially flat and extremely high (0.992–0.999) at every fraction —
  it does **not** show the classic "high-variance model overfitting a tiny dataset" decreasing
  pattern; the model fits training data almost perfectly regardless of how much data it sees.
- **(ii) Test AUC increases** steadily with more training data (0.891 → 0.940), and the gain from
  80%→100% (0.9388→0.9400) is much smaller than earlier increments (e.g. 20%→40% gained +0.028) —
  the curve is flattening but has **not fully plateaued**.
- **(iii) Conclusion:** the model sits closer to the **capacity-limited** end of the spectrum at
  this point — the near-perfect training AUC combined with a shrinking-but-still-positive test AUC
  gain suggests modest further gains are possible from more data, but the persistent
  train/test AUC gap (~0.05) is primarily explained by model variance/capacity rather than data
  scarcity; tuning regularization (e.g. lower `n_estimators`, shallower trees, higher
  `min_samples_leaf`) is likely to close the gap faster than simply collecting more rows.

## Task 8 — Serialization

`best_pipeline` (the GridSearchCV winner) saved to `best_model.pkl` via `joblib.dump`. Reload-and-
predict block:
```python
import joblib
loaded = joblib.load("best_model.pkl")
sample_rows = X_test.iloc[:2]      # 2 hand-crafted/held-out rows
preds = loaded.predict(sample_rows)
print(preds)   # -> [0 0]
```
Ran without errors; predictions are visible in the console output.

## Task 9 — Summary comparison table & recommendation

| Model | CV mean AUC | CV std AUC | Test AUC |
|---|---|---|---|
| Logistic Regression | 0.9720 | 0.0098 | 0.9630 *(Part 2)* |
| Decision Tree (depth=5) | 0.9217 | 0.0288 | — |
| Random Forest (default-tuned) | 0.9705 | 0.0105 | 0.9367 |
| Gradient Boosting | 0.9648 | 0.0145 | 0.9424 |
| **Random Forest (GridSearchCV-tuned)** | **0.9738** | — | **0.9400** |

**Recommendation: the GridSearchCV-tuned Random Forest (`max_depth=5, min_samples_leaf=5,
n_estimators=200`).** It posts the highest 5-fold CV mean AUC of any model tested (0.9738),
confirming it isn't a lucky single-split result. Logistic Regression is a very close second
(0.9720 CV AUC, and actually the highest single test-set AUC at 0.963) and is far cheaper to run
and easier to explain — it remains a strong, defensible fallback if interpretability or inference
latency matter more than the last fraction of a percentage point of AUC. But given the client's
explicit ask for "the most robust model," the tuned Random Forest is recommended: it combines
competitive accuracy with the variance-reduction benefits of bagging (Task 4) and was chosen
through a systematic, cross-validated search rather than a single default configuration.

## Acceptance checklist
- [x] Unconstrained vs constrained tree train/test gap comparison
- [x] Gini and Entropy formulas written out
- [x] Top-5 feature importances reported + explained
- [x] GridSearchCV best params + best score printed
- [x] `best_model.pkl` present
- [x] Reload-and-predict block runs without errors
- [x] Summary table + final recommendation
- [x] GradientBoostingClassifier trained and included in CV comparison
- [x] Feature ablation: 5 lowest-importance features dropped, AUC compared, interpreted
- [x] Manual learning curve (5 fractions), train/test AUC trend interpreted, data- vs capacity-limited conclusion
