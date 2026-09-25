# Team Collaboration & Git Workflow Guide

Welcome to the **Amazon ML Challenge 2026: Business Entity Resolution** project!

This repository is maintained with a strict **50/50 technical contribution** between two active developers:
- **Moksh**
- **Dhwaj (Team Leader)**

Neither member is restricted to documentation or a single isolated silo. Both members understand the complete ML pipeline, conduct experiments, and write core code.

---

## 1. 50/50 Technical Division & Leadership Areas

| Stage / Component | Lead: Moksh | Lead: Dhwaj [Team Leader] | Joint Responsibility (Both) |
| :--- | :--- | :--- | :--- |
| **Exploratory Data Analysis** | EDA Experiment A (`notebooks/eda_moksh.ipynb`): Name noise, legal suffixes, typos | EDA Experiment B (`notebooks/eda_dhwaj.ipynb`): Address noise, landmarks, countries, cardinality | Cross-review findings, avoid aggressive normalization |
| **Preprocessing** | Name normalization (`normalize_name` in `src/preprocessing.py`) | Address & country normalization (`normalize_address`, `normalize_country`) | Modular integration & chunk processing |
| **Blocking / Candidates** | Strategy A (`BlockingStrategyA` in `src/blocking.py`): Name token index, prefix keys | Strategy B (`BlockingStrategyB` in `src/blocking.py`): Country partition, address tokens | `MultiStrategyBlocker`, recall ceiling & reduction ratio |
| **Feature Engineering** | Name similarity features (Levenshtein, Jaro-Winkler, token overlap) | Address & context features (token sets, country match, source indicator) | Unified feature extraction & ablation |
| **Machine Learning Model** | ML Model Experiment A: Baseline Logistic Regression & thresholding | ML Model Experiment B: Tabular classifiers (Random Forest / GBDT) | Model comparison, F0.5 validation |
| **Validation & Evaluation** | Threshold sweep Experiment A | Threshold sweep Experiment B | Entity-level F0.5 evaluation, singleton analysis |
| **Error Analysis** | Error Analysis A: Dissecting name-driven errors | Error Analysis B: Dissecting address-driven errors | Taxonomical classification to guide improvements |
| **Pipeline & Submission** | Pipeline testing & integration | Submission validation & packaging | End-to-end inference verification |

---

## 2. Git Branch Strategy

We use a simple, robust branch model:

* **`main`**: Always points to the stable, runnable baseline.
  - **Never push untested or broken code directly to `main`.**
  - Code enters `main` only through reviewed Pull Requests.
* **`moksh-work`**: Dedicated development branch for Moksh.
* **`dhwaj-work`**: Dedicated development branch for Dhwaj.

---

## 3. Step-by-Step Workflow

### Step 1: Sync with `main` before starting work
```bash
git checkout main
git pull origin main
```

### Step 2: Switch to your branch and rebase/merge `main`
```bash
# For Moksh:
git checkout moksh-work
git merge main

# For Dhwaj:
git checkout dhwaj-work
git merge main
```

### Step 3: Write modular code and make meaningful commits
Write clean code with docstrings. Use clear, imperative commit messages:
* `"Add name normalization"`
* `"Implement candidate blocking"`
* `"Add similarity features"`
* `"Add baseline matcher"`
* `"Improve F0.5 validation"`

Example:
```bash
git status
git add src/preprocessing.py
git commit -m "Add name normalization"
```

### Step 4: Push your branch
```bash
# For Moksh:
git push origin moksh-work

# For Dhwaj:
git push origin dhwaj-work
```

### Step 5: Pull Request & Mutual Code Review
1. Open a PR: `base: main` ← `compare: moksh-work` (or `dhwaj-work`).
2. The other teammate reviews the code, checks validation results, and confirms no data files are tracked.
3. Merge into `main` after mutual agreement.

---

## 4. Critical Data Safety Rules

> [!CAUTION]
> **NEVER commit the 2.5 GB dataset or huge generated files to GitHub!**

* `.tsv`, `.csv`, `.parquet`, `.pkl`, and `dataset/` are blocked in `.gitignore`.
* Always run `git status` before committing.
* When working in `notebooks/`, **clear all cell outputs** before committing (`Kernel -> Restart & Clear Output`).
