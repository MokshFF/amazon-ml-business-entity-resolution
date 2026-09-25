# Notebooks

This directory is intended for interactive exploration, data profiling, error analysis, and experimental prototyping by team members.

## Guidelines for Team Collaboration

1. **Clear Notebook Outputs Before Committing:**
   - Notebook execution outputs can contain sample data or large figures that bloat the git repository.
   - Always run **Kernel → Restart & Clear Output** before pushing to your feature branch (or use pre-commit filters).

2. **Notebook Naming Convention:**
   - Prefix notebooks with the responsible member or workflow stage:
     - `01_eda_data_profiling.ipynb` (Member 1 - EDA & Data Profiling)
     - `02_preprocessing_experiments.ipynb` (Member 1 - Normalization testing)
     - `03_blocking_evaluation.ipynb` (Member 2 - Blocking & Candidate generation)
     - `04_feature_engineering.ipynb` (Member 3 - Similarity features)
     - `05_model_training_f05_tuning.ipynb` (Member 3 - ML Matching & F0.5 evaluation)
     - `06_error_analysis.ipynb` (Member 4 - Integration, False Positives/Negatives)

3. **Transition Prototyped Logic to `src/`:**
   - Once a preprocessing function, blocking index, or feature computation is validated in a notebook, refactor it into the appropriate module under `src/` so it can be reused across the team and executed by `predict.py`.
