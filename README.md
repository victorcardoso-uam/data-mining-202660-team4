# NASA project: prioritizing near-Earth objects for the PDCO

**Client:** NASA/JPL Planetary Defense Coordination Office.
**Goal:** rank near-Earth objects to prioritize telescope follow-up observations, under a "zero false negative" policy.

## Repository structure

```
data-mining-202660-team4/
├── nasa-project/            # capstone project (client: NASA PDCO)
│   ├── data/raw/            # Kaggle CSVs (not versioned)
│   ├── notebooks/           # one notebook per step
│   ├── reports/             # generated figures and tables (versioned)
│   └── src/                 # utility scripts (download)
├── activities/              # weekly course activities
├── pyproject.toml           # project dependencies
└── uv.lock                  # exact dependency versions
```

## Data

Two Kaggle CSVs (`darkmatternet/nasa-near-earth-asteroids-and-close-approaches`), stored in `nasa-project/data/raw/` (not versioned):

- `near_earth_asteroids_2025*.csv`: catalogue of 41,281 objects (orbital elements, H, diameter, MOID, `pha`);
- `asteroid_close_approaches_2015_2035*.csv`: 27,430 approaches closer than 0.1 AU.

## Installation with uv

The project uses [uv](https://docs.astral.sh/uv/) to manage Python (version 3.14, pinned in `.python-version`) and the dependencies (`pyproject.toml` and `uv.lock`). There is no need to install Python manually: uv downloads it if needed.

### 1. Install uv (once per machine)

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or with winget: `winget install --id=astral-sh.uv -e`

**macOS**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or with Homebrew: `brew install uv`

**Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen the terminal, then check the installation:

```bash
uv --version
```

### 2. Clone the repository and create the environment

The commands are the same on all three systems:

```bash
git clone https://github.com/victorcardoso-uam/data-mining-202660-team4.git
cd data-mining-202660-team4
uv sync
```

`uv sync` installs the right Python version, creates the `.venv/` folder and installs the exact versions from `uv.lock`, including development dependencies such as `ipykernel`.

> On Windows, if the repository is in a folder whose path contains spaces (OneDrive, for example), put the path in quotes: `cd "C:\Users\...\data-mining-202660-team4"`.

### 3. Download the data

From the repository root:

```bash
uv run python nasa-project/src/download_data.py
```

The CSVs are copied to `nasa-project/data/raw/`.

## Running the notebooks

One notebook per step, in `nasa-project/notebooks/`. Figures and tables are written to `nasa-project/reports/<step>/` (versioned, for the slides).

| Step | Notebook | Outputs |
|---|---|---|
| 2. Exploratory data analysis | `nasa-project/notebooks/02_eda.ipynb` | `nasa-project/reports/02_eda/` |
| 3. Leakage-free preprocessing | `nasa-project/notebooks/03_preprocessing.ipynb` | `nasa-project/reports/03_preprocessing/` and `nasa-project/data/processed/03_preprocessing/` |
| 4. Diameter regression | `nasa-project/notebooks/04_diameter_regression.ipynb` | `nasa-project/reports/04_diameter/` and `nasa-project/data/processed/04_diameter/` |
| 5. Classification and prioritization | `nasa-project/notebooks/05_classification.ipynb` | `nasa-project/reports/05_classification/` and `nasa-project/data/processed/05_classification/` |
| 6. Risk-profile clustering | `nasa-project/notebooks/06_clustering.ipynb` | `nasa-project/reports/06_clustering/` and `nasa-project/data/processed/06_clustering/` |
| 7. Prioritization policy | `nasa-project/notebooks/07_risk_policy.ipynb` | `nasa-project/reports/07_risk_policy/` and `nasa-project/data/processed/07_risk_policy/priority_queue.csv` |

The notebooks must be run **in order**: from step 3 onwards, each notebook saves to `nasa-project/data/processed/` (not versioned) the train/test sets and preprocessors used by the following steps. After a `git clone`, run the notebooks in order (03, then 04, then 05…).

### Option A: VS Code (recommended)

1. Open the repository folder in VS Code (with the **Python** and **Jupyter** extensions installed).
2. Open the notebook, click **Select Kernel** at the top right, then choose the project's `.venv` environment.
3. Click **Run All**.

### Option B: Jupyter Lab in the browser

Same commands on Windows, macOS and Linux:

```bash
uv run --with jupyter jupyter lab
```

`--with jupyter` uses Jupyter temporarily, without adding it to the project dependencies.

### Option C: run a notebook without opening it

Handy to regenerate all the figures of a step:

```bash
uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace nasa-project/notebooks/02_eda.ipynb
```

## Activating the environment (optional)

`uv run <command>` already uses the right environment, without activation. If you prefer to activate it:

| System | Command |
|---|---|
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (cmd) | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

To deactivate it: `deactivate`.

> On Windows, if PowerShell refuses to run the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## Adding a dependency

```bash
uv add package-name            # project dependency
uv add --dev package-name      # development tool (ipykernel, pytest…)
```

Then commit `pyproject.toml` and `uv.lock`. Teammates only need to run `uv sync` after a `git pull`.

Never commit the `.venv/` folder: it is machine-specific and already ignored by `.gitignore`.

## Troubleshooting

### Windows and OneDrive: `failed to hardlink file` (os error 396)

By default, uv creates hard links between its cache and the `.venv`. OneDrive does not accept them, which produces the error "The cloud operation cannot be performed on a file with incompatible hardlinks" (shown in your system language).

To fix it, tell uv to copy files instead of linking them. In PowerShell:

```powershell
[System.Environment]::SetEnvironmentVariable("UV_LINK_MODE", "copy", "User")
```

Close and reopen the terminal (and VS Code), clear the damaged cache, then run again:

```powershell
uv cache clean
uv sync
```

For the current session only: `$env:UV_LINK_MODE = "copy"`.

On macOS and Linux, the equivalent is `export UV_LINK_MODE=copy`, to be added to `~/.zshrc` or `~/.bashrc` to make it permanent.

## Key EDA findings

- `pha` matches the rule MOID ≤ 0.05 AU and H ≤ 22 for 99.9% of objects. MOID, H and the columns derived from H must be excluded if `pha` is the target.
- 97% of the diameters are computed from H with a fixed albedo of 0.14. Only 1,246 diameters are actually measured.
- The drop in approaches after March 2026 comes from the extraction date: objects not yet discovered have no predictions.
- 80% of the objects have a single approach in the 2015–2035 window.

## Key preprocessing findings

- Features kept for classification: orbital elements `a, e, i, q, ad` and the `class` family. MOID, H, diameter, `size_category` and albedo are excluded.
- `data_arc_years`, `condition_code` and `first_obs_year` are proxies for size (|ρ| with H up to 0.77). They form a variant tested separately.
- Cross-validated leakage audit (PR-AUC, random = 0.06): H + MOID = 1.00 (leakage), orbit only = 0.15, orbit + observation = 0.48.
- Diameter regression: 1,244 measured diameters, albedo excluded. Measured objects are much larger than the ones to predict (median H 19 vs 24).

## Key diameter regression findings

- Selected model: "physics-guided" Random Forest. It keeps the exact law log D = constant − H/5 and only learns the albedo term, from H and the orbit.
- On the test set, compared with the catalogue formula (albedo 0.14): RMSE from 0.222 to 0.203 dex, and errors larger than a factor of 2 drop from 20% to 12%. Accuracy equivalent to the linear model, but the forest does not extrapolate absurd trends to small objects.
- Objects whose orbit extends beyond 2.17 AU are darker (albedo ≈ 0.07): the catalogue formula underestimates their diameter.
- 90% interval: from ×0.52 to ×2.14 around the predicted diameter (88% coverage on the test set). The upper bound is used as the pessimistic diameter in step 7.
- 41% of the catalogue is smaller than 98% of the training objects (H > 24.7) and carries the `out_of_domain` flag.

## Key classification findings

- Target: `pha`, predicted without H or MOID. Selected model: Random Forest on the orbit and observation variables, with `class_weight="balanced_subsample"`. The notebook takes about 5 minutes to run (grid search).
- The decision threshold is chosen on the training set to reach 99% recall (threshold ≈ 0.055 instead of 0.5).
- On the test set: 99% of PHAs caught (5 missed out of 508) while monitoring 33% of the catalogue, vs 73% with the rule "the orbit crosses Earth's orbit" and 99% at random.
- With the orbit alone, 72% of the catalogue must be monitored for the same recall: most of the gain comes from the observation variables, partly because they reflect size.
- The missed PHAs are almost all objects discovered a few days or weeks earlier: they must be monitored systematically, regardless of their score.

## Key clustering findings

- K-Means on four standardized variables: log diameter (step 4), log MOID, inclination and eccentricity. `pha` is only used to describe the groups afterwards.
- Variable choice checked with correlations and PCA: `a`, `q` and `ad` are redundant (r up to 0.98), and none of the 4 selected variables is superfluous (3 components needed to exceed 90% of the information). PCA is used to visualize the clusters in the PC1/PC2 plane.
- K = 4 selected (silhouette ≈ 0.27). The groups overlap, because near-Earth objects form a continuum, but they are reproducible (ARI ≈ 0.99 between reruns).
- The clusters do not rediscover the orbital families (ARI ≈ 0.08): three out of four mix Apollo and Amor in equal parts.
- The cluster of large objects near the ecliptic holds 64% of the catalogue's PHAs: it justifies a priority follow-up campaign.
- The cluster of small close approachers (median MOID 0.005 AU) has 44% of objects with a past or expected approach closer than 5 lunar distances, but only 1% PHAs: rapid follow-up at discovery.

## Key prioritization policy findings

- Three tiers: tier 1 (observe this cycle), tier 2 (next cycle), tier 3 (passive surveillance). All thresholds are grouped in the first cell of the notebook.
- Tier 1 = probable PHA (MOID ≤ 0.05 AU and estimated diameter ≥ 140 m) **and** urgent (approach closer than 20 lunar distances by 2035, uncertain orbit with `condition_code` ≥ 7, or discovered less than a year ago): 730 objects (1.8% of the catalogue).
- Tier 2 = non-urgent probable PHAs, possible PHAs (only the diameter's upper bound exceeds 140 m) and recent discoveries flagged by the step 5 model: 4,880 objects (11.8%).
- 99.5% of the official PHAs are in tier 1 or 2. The other 12 have a measured diameter below 140 m.
- The policy adds 662 probable PHAs and 2,184 possible PHAs missing from the official list, because they are darker than the assumed 0.14 albedo.
- Telescope time (illustrative assumptions): 81% less than "monitor everything", 74% less than the expert rule, about 50% more than the official list, but allocated by urgency.
- `priority_queue.csv` ranks the whole catalogue with the reasons for each decision: it is the planned data source for the Streamlit app.
