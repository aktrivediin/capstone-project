# Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis

**Run:** `python part1_eda.py` (reads `train.csv`, writes `cleaned_data.csv` and `figures/*.png`)

## Dataset choice

`train.csv` — MLB team-season batting/pitching-against statistics. 581 rows, 24 raw numeric
columns (wins `W`, games `G`, runs `R`, at-bats `AB`, hits `H`, hit types, walks `BB`, strikeouts
`SO`, stolen bases `SB`, caught stealing `CS`, hit-by-pitch `HBP`, sac flies `SF`, outs, runs
allowed `RA`, and rate stats `BA`/`OBA`/`SLG`/`OPS`). Target for regression: `W`. Target for
classification: `W` binarized at the median (`Win_Tier`/derived binary label used in Part 2).

**Declared deviation:** the raw file arrived 100% numeric with **zero missing values and zero
duplicates**, which does not exercise several required tasks (null analysis, categorical
encoding, categorical box plots/groupby). Task 0 below documents exactly what was engineered and
why, before any cleaning begins.

## Task 0 — Engineered categorical columns + synthetic nulls (disclosed)

- **`Win_Tier`** (ordinal): `W` binned into Low (≤70) / Medium (71–85) / High (86–95) / Elite (>95).
  Chosen because win totals have a natural, universally-understood ordering in baseball
  (a "High" win team is unambiguously better than a "Low" one) — this justifies label/ordinal
  encoding rather than one-hot in Part 2.
- **`Run_Diff_Type`** (nominal): `Run-Positive` if `R - RA >= 0` else `Run-Negative`. No inherent
  order between the two categories, so it is a genuine one-hot-encoding candidate.
- **Synthetic missingness**: ~10% of `HBP` and ~8% of `SF` set to `NaN` with a fixed seed
  (`RandomState(42)`), to make the null-handling tasks meaningful. This is the only place random
  noise was added to the data; every other number in this report comes from the file as supplied.

## Task 2 — Null value analysis

| Column | Null count | Null % |
|---|---|---|
| HBP | 70 | 12.05% |
| SF | 41 | 7.06% |

No column exceeds the 20% threshold, so both were **median-filled** rather than dropped.
**Why median, not mean:** both `HBP` and `SF` are right-skewed count-like columns with a large
mass of exact zeros (many teams recorded 0 sac flies / 0 HBP under old scoring rules) and a long
tail of high values. The mean is pulled upward by that tail, so it overstates the "typical" team;
the median sits inside the dense zero/low-value cluster and is a more representative fill value.

## Task 3 — Duplicates

`df.duplicated().sum()` → **0** duplicate rows. Nothing was removed, and consequently the null
percentages were identical before and after the (no-op) `drop_duplicates()` call.

## Task 4 — Data type correction

- `SB` (stolen bases) was deliberately cast to `object`/string to simulate a messy raw import,
  then corrected with `pd.to_numeric(df['SB'], errors='coerce')` back to `int64`.
- `Win_Tier` and `Run_Diff_Type` (repetitive string/category columns) were converted to
  `category` dtype.
- **Memory usage:** 148,092 bytes → 113,354 bytes (**23.46% reduction**), driven almost entirely
  by the two columns moving from Python object storage to compact category codes.

## Task 5 — Descriptive statistics and skewness

`df.describe()` is printed in full by the script. Skewness ranking (by |skew|), top of list:

| Column | Skew |
|---|---|
| G (games played) | −3.94 |
| Outs | −3.78 |
| AB | −3.72 |
| H | −1.75 |

**Most skewed column: `G`, skew = −3.94 (strong negative/left skew).** Interpretation: this
column is left-skewed because almost every team plays the full 162-game season (a hard ceiling),
so the bulk of values cluster tightly near 162 while a small number of shortened/strike-affected
seasons pull a long tail toward lower values. **Consequence for imputation:** if `G` had missing
values, filling with the mean would pull the fill value below the dominant 162 mode (since the
left tail drags the mean down), understating games played for a typical team — the median (162)
would be the far more representative choice, exactly the same reasoning applied to `HBP`/`SF` above.

## Task 6 — Outlier detection (IQR)

`G` was excluded from this task despite being the most-skewed column: it is a near-constant
discrete value (most teams play exactly 162 games), so `Q1 == Q3 == 162`, the IQR collapses to 0,
and the 1.5×IQR rule degenerates (it would flag 144 rows as "outliers" purely because they aren't
exactly 162 — not a meaningful outlier signal). Two columns with genuine continuous spread were
used instead:

| Column | Q1 | Q3 | IQR | Lower bound | Upper bound | Outlier rows |
|---|---|---|---|---|---|---|
| HR | 116 | 176 | 60 | 26 | 266 | 0 |
| SO | 842 | 1093 | 251 | 465.5 | 1469.5 | 5 |

No outliers found in `HR` — home run totals stay within a tight, well-behaved range across teams.
5 teams have strikeout totals (`SO`) outside the 1.5×IQR fence — these are retained, not dropped.
**Handling decision for Part 2:** these will be **retained**, not capped. Extreme strikeout totals
are real team performance signals (some franchises genuinely strike out far more than others), not
data-entry errors, and tree-based models used in Part 3 are robust to them; capping would discard
real information that could help predict wins.

## Task 7 — Visualizations

All six required plots are produced in `figures/`:

1. **Line plot** (`01_line_wins.png`) — `W` sorted ascending across row index; shows a smooth,
   roughly linear ramp from ~37 to 116 wins with no discontinuities, i.e., wins are well-spread
   across the full range rather than clustered.
2. **Bar chart** (`02_bar_hr_by_wintier.png`) — mean `HR` by `Win_Tier`; mean home runs increase
   monotonically from Low → Elite tiers, consistent with power hitting contributing to winning.
3. **Histogram** (`03_hist_skewed.png`) — distribution of `G`, the most-skewed column; a tall,
   narrow spike near 162 with a thin left tail trailing down toward ~103, visually confirming the
   left-skew computed in Task 5.
4. **Scatter plot** (`04_scatter_ab_h.png`) — `AB` vs `H`; a clear, tight positive linear
   relationship (more at-bats mechanically produce more hit opportunities), roughly strong
   (visually consistent with the Pearson correlation of ≈0.7 reported in Task 8b).
5. **Box plot** (`05_box_ops_by_wintier.png`) — `OPS` by `Win_Tier`; median OPS rises steadily from
   Low to Elite tiers, and the spread (box height) narrows somewhat for Elite teams — elite teams
   are both better and more consistently good offensively.
6. **Correlation heatmap** (`06_heatmap_pearson.png`) — full Pearson correlation matrix of all
   numeric columns.

**Highest-correlation pair: `G` and `Outs`, r = 0.9875.** This is very unlikely to be a novel
causal discovery — it is almost mechanical: teams that play more games (`G`) accumulate more total
outs (`Outs`) simply by playing more innings, so the "third variable" explaining the correlation is
**games played itself acting as a scale factor** on nearly every counting statistic in the table
(the same logic applies to `AB`, `H`, `BB`, etc. all correlating with `G`).

## Task 8a — Imputation strategy comparison (mean vs median)

| Column | Mean | Median | Skew | Chosen |
|---|---|---|---|---|
| G | 158.97 | 162.00 | −3.94 | Median |
| Outs | 4013.32 | 4085.00 | −3.78 | Median |

Both columns are strongly negatively skewed, so in both cases the mean sits noticeably below the
median (pulled down by the left tail of shortened seasons). The median was used for filling
(`isnull().sum()` confirmed 0 remaining nulls in both), consistent with the reasoning in Task 2/5:
median is more representative of a "typical" value when the mean is dragged off-center by skew.

## Task 8b — Spearman vs Pearson

Top 3 pairs by |Spearman − Pearson|:

| Pair | Pearson | Spearman | |diff| |
|---|---|---|---|
| H – Outs | 0.732 | −0.057 | 0.790 |
| 1B – Outs | 0.669 | −0.043 | 0.712 |
| Outs – Outsinplay | 0.780 | 0.176 | 0.604 |

For all three pairs, |Pearson| >> |Spearman| — the opposite of the "monotonic-but-nonlinear"
signature. This pattern (a real linear Pearson correlation that nearly vanishes under Spearman)
indicates the relationship is **driven by a small number of high-leverage points** rather than a
consistent rank-order relationship across the whole dataset — i.e., Pearson is picking up a linear
trend that is not monotonic when the extreme `Outs` values (the same near-constant/skewed column
flagged in Task 6) are rank-transformed. **Feature-selection guidance for Part 2:** Pearson will be
relied on here, since the relationships that matter for the linear regression model in Part 2 are
genuinely linear-scale ones (counting-stat correlations), and Spearman's rank compression on the
near-constant `Outs`/`G` columns would understate real linear signal.

## Task 8c — Grouped aggregation

`groupby('Win_Tier')['OPS'].agg(['mean','std','count'])`:

| Win_Tier | mean | std | count |
|---|---|---|---|
| Low | 0.7075 | 0.0370 | 137 |
| Medium | 0.7254 | 0.0429 | 246 |
| High | 0.7439 | 0.0401 | 138 |
| Elite | 0.7665 | 0.0364 | 60 |

- **Highest mean group:** Elite (0.7665). **Highest std group:** Medium (0.0429).
- Within-group standard deviation is fairly similar across all four tiers (0.036–0.043), so it is
  **not a major concern** — `Win_Tier` alone does not fully determine `OPS`, but the groups are not
  wildly heterogeneous either; other features will still be needed for precise prediction.
- **Ratio of highest to lowest group mean:** 0.7665 / 0.7075 = **1.083**. This is a fairly small
  ratio (~8% spread) — it suggests `Win_Tier` carries **real but modest** predictive signal for
  OPS; it is informative but far from sufficient on its own.

## Acceptance checklist
- [x] Runs top-to-bottom without errors
- [x] Null % table printed for every column with nulls
- [x] All 5 + heatmap visualizations produced (`figures/`)
- [x] `cleaned_data.csv` saved
- [x] Mean/median comparison + justification + nulls confirmed at 0
- [x] Spearman vs Pearson, top-3 pairs, interpretation + Part 2 guidance
- [x] Grouped aggregation, highest mean/std groups, ratio interpretation
