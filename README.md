# Applied AI & ML Capstone — MLB Team-Season Batting Dataset

Single repository, one folder per Part, as required by the submission guidelines.

## Repository layout
```
part1_eda/          Part 1 — Data Acquisition, Cleaning, EDA
part2_ml/            Part 2 — Supervised ML (Regression + Classification)
part3_ensembles/     Part 3 — Ensembles, Tuning, Pipeline
part4_llm/           Part 4 — LLM-Powered Feature (Track C: Model Prediction Explanation)
requirements.txt     All Python dependencies for the whole project
```

## Dataset

**Source / choice:** `train.csv` (client-provided) — an MLB team-season batting/pitching-against
statistics table. 581 rows, 24 raw columns (team `ID`, wins `W`, games `G`, runs `R`, at-bats `AB`,
hits `H`, singles/doubles/triples/home-runs, walks, strikeouts, stolen bases, caught-stealing, HBP,
sac-flies, outs, runs-allowed `RA`, and rate stats `BA`/`OBA`/`SLG`/`OPS`).

**Why this dataset:** it has 24 columns (>>5 numeric), 581 rows (>500), and a natural continuous
target (`W`, season wins) that binarizes cleanly at the median into a "winning season" vs.
"losing season" classification label.

**Declared deviations from the raw file (documented honestly, not hidden):**
1. The raw file is **100% numeric with zero missing values and zero duplicates**. The brief
   requires ≥2 categorical columns and genuine null-handling work. Two categorical columns were
   **engineered from existing numeric columns** (see Part 1 README, Task 0) rather than pulled from
   an unrelated source, so they remain semantically meaningful:
   - `Win_Tier` (ordinal: Low/Medium/High/Elite, binned from `W`)
   - `Run_Diff_Type` (nominal: Run-Positive/Run-Negative, from `R - RA`)
2. **Synthetic missingness (~8–12%)** was injected into two numeric columns (`HBP`, `SF`) using a
   fixed random seed, to make the null-analysis tasks non-vacuous. This is disclosed in Part 1's
   README and the injection code is visible in `part1_eda/part1_eda.py`.

Every other step (cleaning, modeling, ensembles, LLM explanation layer) operates on the data as a
normal client dataset would be treated — the above two adjustments are the only artificial changes,
and both are called out wherever they matter (imputation choice, encoding choice, categorical EDA).

## How to reproduce

Each Part folder contains **both** a `.py` script and an equivalent `.ipynb` notebook (same code,
split into cells with markdown headings) — use whichever your grader/environment prefers.

**Script version:**
```bash
pip install -r requirements.txt
cd part1_eda && python part1_eda.py      # produces cleaned_data.csv
cd ../part2_ml && python part2_ml.py     # reads ../part1_eda/cleaned_data.csv
cd ../part3_ensembles && python part3_ensembles.py   # produces best_model.pkl
cd ../part4_llm && export LLM_API_KEY=sk-...
python part4_llm.py
```

**Notebook version (Jupyter / VS Code):** open `partX_.../partX_....ipynb` and Run All. Each
notebook was generated directly from its corresponding `.py` file and verified to produce
identical output when its cells are executed top-to-bottom, in order.

**Notebook version (Google Colab):** Colab has its own separate filesystem, so relative paths like
`../part1_eda/cleaned_data.csv` won't resolve unless you either (a) upload the whole `repo` folder
to Colab preserving its structure (e.g. via Google Drive + `%cd`), or (b) upload just that Part's
required input files into the same Colab session folder. Each notebook's Task 1 cell reads its
input with a plain relative path exactly as the script does — no code changes needed once the
files are in place.

Each Part folder has its own README.md with full write-ups, results, and interpretation.
