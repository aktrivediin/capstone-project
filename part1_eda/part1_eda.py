"""
Part 1 -- Data Acquisition, Cleaning, and Exploratory Analysis
Dataset: MLB team-season batting/pitching-against stats (train.csv)

Run: python part1_eda.py
Outputs: cleaned_data.csv  (+ figures/*.png, printed console report)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RNG = np.random.RandomState(42)
os.makedirs("figures", exist_ok=True)
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

# ---------------------------------------------------------------------------
# Task 1: Load data
# ---------------------------------------------------------------------------
section("TASK 1: Load data")
df = pd.read_csv("train.csv")
print(df.head())
print("\nDtypes:\n", df.dtypes)
print("\nShape:", df.shape)

# ---------------------------------------------------------------------------
# Task 0 (disclosed deviation): engineer 2 categorical columns + inject nulls
# This block is documented in the README -- the raw file is 100% numeric with
# zero missing values, which conflicts with several required tasks below.
# ---------------------------------------------------------------------------
section("TASK 0 (disclosed): engineer categorical columns + synthetic nulls")

# Ordinal categorical: Win_Tier, binned from wins (W)
bins = [-1, 70, 85, 95, 200]
labels_ord = ["Low", "Medium", "High", "Elite"]
df["Win_Tier"] = pd.cut(df["W"], bins=bins, labels=labels_ord)
print("Win_Tier value counts:\n", df["Win_Tier"].value_counts())

# Nominal categorical: Run_Diff_Type, from run differential (R - RA)
df["Run_Diff_Type"] = np.where(df["R"] - df["RA"] >= 0, "Run-Positive", "Run-Negative")
print("\nRun_Diff_Type value counts:\n", df["Run_Diff_Type"].value_counts())

# Synthetic missingness in two numeric columns (documented, seeded, ~10%)
for col, frac in [("HBP", 0.10), ("SF", 0.08)]:
    mask = RNG.rand(len(df)) < frac
    df.loc[mask, col] = np.nan
print(f"\nInjected nulls -> HBP: {df['HBP'].isnull().sum()}, SF: {df['SF'].isnull().sum()}")

# ---------------------------------------------------------------------------
# Task 2: Null value analysis
# ---------------------------------------------------------------------------
section("TASK 2: Null value analysis")
null_counts = df.isnull().sum()
null_pct = (df.isnull().sum() / df.shape[0]) * 100
null_report = pd.DataFrame({"null_count": null_counts, "null_pct": null_pct})
print(null_report[null_report["null_count"] > 0])

over_20 = null_report[null_report["null_pct"] > 20]
print("\nColumns exceeding 20% null rate:", list(over_20.index) if len(over_20) else "None")

numeric_cols_for_fill = [c for c in df.select_dtypes(include=np.number).columns
                          if c not in ("ID",) and null_report.loc[c, "null_pct"] < 20 and null_report.loc[c, "null_pct"] > 0]
for col in numeric_cols_for_fill:
    df[col] = df[col].fillna(df[col].median())
print("\nFilled (median) columns with <20% nulls:", numeric_cols_for_fill)
print("Remaining nulls after fill:\n", df.isnull().sum()[df.isnull().sum() > 0])

# ---------------------------------------------------------------------------
# Task 3: Duplicate detection and removal
# ---------------------------------------------------------------------------
section("TASK 3: Duplicate detection and removal")
n_dup = df.duplicated().sum()
print("Duplicate rows found:", n_dup)
null_pct_before = (df.isnull().sum() / df.shape[0]) * 100
df = df.drop_duplicates()
null_pct_after = (df.isnull().sum() / df.shape[0]) * 100
print(f"Rows removed: {n_dup}. Shape after: {df.shape}")
print("Null% changed after dedup?:", not null_pct_before.equals(null_pct_after))

# ---------------------------------------------------------------------------
# Task 4: Data type correction
# ---------------------------------------------------------------------------
section("TASK 4: Data type correction")
mem_before = df.memory_usage(deep=True).sum()

# (a) Simulate + fix a numeric column incorrectly typed as object.
#     'SB' (stolen bases) is forced to string/object to mimic a messy raw
#     import (e.g., values arriving as "105", " 98", etc.), then corrected.
df["SB"] = df["SB"].astype(str)
print("SB dtype after forcing to object:", df["SB"].dtype)
df["SB"] = pd.to_numeric(df["SB"], errors="coerce")
print("SB dtype after pd.to_numeric correction:", df["SB"].dtype)

# (b) Convert repetitive string columns to category dtype
df["Win_Tier"] = df["Win_Tier"].astype("category")
df["Run_Diff_Type"] = df["Run_Diff_Type"].astype("category")

mem_after = df.memory_usage(deep=True).sum()
print(f"\nMemory usage before: {mem_before:,} bytes")
print(f"Memory usage after:  {mem_after:,} bytes")
print(f"Reduction: {mem_before - mem_after:,} bytes ({(1 - mem_after/mem_before)*100:.2f}%)")

# ---------------------------------------------------------------------------
# Task 5: Descriptive statistics and skewness
# ---------------------------------------------------------------------------
section("TASK 5: Descriptive statistics and skewness")
numeric_cols = df.select_dtypes(include=np.number).columns.drop("ID")
print(df[numeric_cols].describe())

skew_vals = df[numeric_cols].skew().sort_values(key=np.abs, ascending=False)
print("\nSkewness (sorted by |skew|):\n", skew_vals)
most_skewed_col = skew_vals.index[0]
print(f"\nMost skewed column: {most_skewed_col} (skew={skew_vals.iloc[0]:.3f})")

# ---------------------------------------------------------------------------
# Task 6: Outlier detection with IQR
# ---------------------------------------------------------------------------
section("TASK 6: Outlier detection (IQR)")
# Note: 'G' (games played) is excluded even though it is the most-skewed
# column -- it is a near-constant discrete value (almost every team plays
# 162 games), so Q1==Q3 and the IQR rule degenerates. We instead demonstrate
# IQR outlier detection on two columns with genuine continuous spread.
iqr_cols = ["HR", "SO"]
iqr_report = {}
for col in iqr_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    iqr_report[col] = dict(Q1=Q1, Q3=Q3, IQR=IQR, lower=lower, upper=upper, n_outliers=n_out)
    print(f"{col}: Q1={Q1:.3f} Q3={Q3:.3f} IQR={IQR:.3f} bounds=({lower:.3f}, {upper:.3f}) outliers={n_out}")

# ---------------------------------------------------------------------------
# Task 7: Visualizations
# ---------------------------------------------------------------------------
section("TASK 7: Visualizations")

# 7a. Line plot
plt.figure(figsize=(9, 4))
plt.plot(df.index, df["W"].sort_values().values)
plt.title("Team Wins (W), sorted ascending by row index")
plt.xlabel("Row index (sorted)")
plt.ylabel("Wins (W)")
plt.tight_layout()
plt.savefig("figures/01_line_wins.png", dpi=110)
plt.close()

# 7b. Bar chart: mean HR by Win_Tier
plt.figure(figsize=(7, 4))
df.groupby("Win_Tier", observed=True)["HR"].mean().plot.bar(color="steelblue")
plt.title("Mean Home Runs (HR) by Win Tier")
plt.xlabel("Win Tier")
plt.ylabel("Mean HR")
plt.tight_layout()
plt.savefig("figures/02_bar_hr_by_wintier.png", dpi=110)
plt.close()

# 7c. Histogram of most skewed column
plt.figure(figsize=(7, 4))
sns.histplot(df[most_skewed_col], bins=20, kde=True)
plt.title(f"Histogram of {most_skewed_col} (most skewed, skew={skew_vals.iloc[0]:.2f})")
plt.tight_layout()
plt.savefig("figures/03_hist_skewed.png", dpi=110)
plt.close()

# 7d. Scatter plot: AB vs H (expected correlation)
plt.figure(figsize=(7, 4))
sns.scatterplot(data=df, x="AB", y="H")
plt.title("At-Bats (AB) vs Hits (H)")
plt.tight_layout()
plt.savefig("figures/04_scatter_ab_h.png", dpi=110)
plt.close()

# 7e. Box plot: OPS by Win_Tier
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="Win_Tier", y="OPS")
plt.title("OPS distribution by Win Tier")
plt.tight_layout()
plt.savefig("figures/05_box_ops_by_wintier.png", dpi=110)
plt.close()

# 7f. Correlation heatmap
corr_pearson = df[numeric_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_pearson, annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 6})
plt.title("Pearson Correlation Heatmap (numeric columns)")
plt.tight_layout()
plt.savefig("figures/06_heatmap_pearson.png", dpi=110)
plt.close()

corr_unstack = corr_pearson.where(~np.eye(len(corr_pearson), dtype=bool)).abs().unstack().sort_values(ascending=False)
top_pair = corr_unstack.index[0]
print(f"Highest |correlation| pair: {top_pair} = {corr_pearson.loc[top_pair]:.4f}")

print("Saved 6 figures to ./figures/")

# ---------------------------------------------------------------------------
# Task 8a: Imputation strategy comparison (mean vs median) for top-2 skewed cols
# ---------------------------------------------------------------------------
section("TASK 8a: Imputation strategy comparison (mean vs median)")
top2_skew_cols = skew_vals.index[:2].tolist()
for col in top2_skew_cols:
    mean_v, median_v = df[col].mean(), df[col].median()
    print(f"{col}: mean={mean_v:.4f}  median={median_v:.4f}  skew={skew_vals[col]:.4f}")
    df[col] = df[col].fillna(df[col].median())
print("\nNulls remaining in those columns:\n", df[top2_skew_cols].isnull().sum())

# ---------------------------------------------------------------------------
# Task 8b: Spearman vs Pearson
# ---------------------------------------------------------------------------
section("TASK 8b: Spearman vs Pearson correlation")
corr_spearman = df[numeric_cols].corr(method="spearman")
diff = (corr_spearman - corr_pearson).abs()
diff_unstack = diff.where(~np.eye(len(diff), dtype=bool)).unstack().sort_values(ascending=False)
diff_unstack = diff_unstack[~diff_unstack.index.duplicated()]
# remove mirrored duplicate pairs (a,b) vs (b,a)
seen = set()
top3_pairs = []
for (a, b), v in diff_unstack.items():
    key = frozenset((a, b))
    if key in seen or a == b:
        continue
    seen.add(key)
    top3_pairs.append((a, b, v))
    if len(top3_pairs) == 3:
        break

print("Top 3 pairs by |Spearman - Pearson|:")
diff_table = []
for a, b, v in top3_pairs:
    p = corr_pearson.loc[a, b]
    s = corr_spearman.loc[a, b]
    print(f"  {a} vs {b}: Pearson={p:.4f}  Spearman={s:.4f}  |diff|={v:.4f}")
    diff_table.append({"pair": f"{a}-{b}", "pearson": p, "spearman": s, "abs_diff": v})
diff_df = pd.DataFrame(diff_table)
print(diff_df)

# ---------------------------------------------------------------------------
# Task 8c: Grouped aggregation
# ---------------------------------------------------------------------------
section("TASK 8c: Grouped aggregation")
group_col, num_col = "Win_Tier", "OPS"
agg = df.groupby(group_col, observed=True)[num_col].agg(["mean", "std", "count"])
print(agg)
highest_mean_group = agg["mean"].idxmax()
highest_std_group = agg["std"].idxmax()
mean_ratio = agg["mean"].max() / agg["mean"].min()
print(f"\nHighest mean group: {highest_mean_group}")
print(f"Highest std group: {highest_std_group}")
print(f"Ratio of highest to lowest group mean: {mean_ratio:.4f}")

# ---------------------------------------------------------------------------
# Save cleaned dataset
# ---------------------------------------------------------------------------
section("SAVE cleaned_data.csv")
df.to_csv("cleaned_data.csv", index=False)
print("Saved cleaned_data.csv with shape", df.shape)
print("\nDONE.")
