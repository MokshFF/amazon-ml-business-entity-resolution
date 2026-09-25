# Contributing & Team Collaboration Guide

Welcome to the team! This repository is set up for collaborative development during the **Amazon ML Challenge 2026: Business Entity Resolution Challenge**.

To ensure seamless progress without merge conflicts or repository corruption, all members must adhere to the following Git workflow.

---

## 1. Branch Strategy

* **`main` Branch:**
  * The `main` branch must always remain stable, cleanly formatted, and runnable.
  * **Direct commits or pushes to `main` are strictly prohibited.**
  * All updates reach `main` exclusively through reviewed and approved Pull Requests (PRs).

* **Feature Branches:**
  * Every team member must create and work within an isolated feature branch branched off `main`.
  * **Designated Branch Names:**
    * `feature/eda` — Assigned to **Member 1** (Data profiling, EDA, text cleaning & normalization).
    * `feature/blocking` — Assigned to **Member 2** (Candidate generation, indexing, `candidate_pairs.tsv`).
    * `feature/ml-matching` — Assigned to **Member 3** (Similarity features, ML classifier, F0.5 threshold tuning).
    * `feature/integration` — Assigned to **Member 4** (Pipeline orchestration, test inference, packaging).
  * If working on a focused fix or enhancement, follow the pattern:
    * `fix/<short-description>`
    * `feature/<short-description>`

---

## 2. Step-by-Step Git Workflow

### Step 1: Update your local `main` branch
Before starting new work, always pull the latest updates from the remote repository:
```bash
git checkout main
git pull origin main
```

### Step 2: Create or switch to your feature branch
Create your assigned branch:
```bash
git checkout -b feature/<your-role>
```
*(If the branch already exists locally or remotely, run `git checkout feature/<your-role>` followed by `git pull origin feature/<your-role>`)*

### Step 3: Make modular commits
Keep commits atomic and provide descriptive messages explaining *why* changes were made:
```bash
git status
git add src/preprocessing.py
git commit -m "feat(preprocessing): add legal suffix normalization for private limited"
```

### Step 4: Push branch to GitHub
Push your local branch to the remote repository:
```bash
git push -u origin feature/<your-role>
```

### Step 5: Open a Pull Request (PR)
1. Go to the GitHub repository: `https://github.com/MokshFF/amazon-ml-business-entity-resolution`.
2. Click **New Pull Request**.
3. Set `base: main` ← `compare: feature/<your-role>`.
4. Provide a clear PR description detailing:
   - What module was updated
   - How changes were verified or tested
   - Any dependency updates needed
5. Request a review from at least one teammate.

### Step 6: Code Review & Merging
- Reviewers verify code quality, check that no dataset files are tracked, and confirm tests pass.
- Once approved, merge the PR into `main` using **Squash and Merge** or **Rebase and Merge** to maintain a clean git history.
- Delete the feature branch on GitHub after merging.

---

## 3. Critical Dataset & File Safety Rules

> [!CAUTION]
> **NEVER commit dataset files to GitHub!**
> The raw 2.5 GB challenge dataset must remain strictly local or in private shared storage.

* **Never run `git add .` blindly.** Always check `git status` first to inspect what files are staged.
* `.tsv`, `.csv`, `.parquet`, `.pkl`, and directories named `dataset/` or `student_resource/` are blocked in `.gitignore`.
* If you create exploratory Jupyter notebooks in `notebooks/`, **clear all cell outputs** before committing to keep file diffs clean and lightweight.
* Submission files (`output/*.tsv`) are strictly for local testing and Portal upload; do not commit large generated TSVs to git.
