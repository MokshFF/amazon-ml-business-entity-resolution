# Amazon ML Challenge 2026: Business Entity Resolution

## 1. Project Title
**Business Entity Resolution Challenge (Amazon ML Challenge 2026)**  
Repository: [amazon-ml-business-entity-resolution](https://github.com/MokshFF/amazon-ml-business-entity-resolution)

---

## 2. Challenge Objective
In large-scale commercial platforms, business identity data originates from multiple independent channels, resulting in partial, noisy, and unstructured fragments of information describing the same real-world businesses. These records share no common unique identifiers.

The objective of this challenge is to build a robust, scalable Machine Learning entity resolution pipeline that, given records from three independent data sources, resolves which records across sources refer to the exact same real-world business entity.

* **Source 1:** Deduplicated reference source.
* **Source 2:** Noisy business records.
* **Source 3:** Noisy business records.

For every Source 1 entity, our pipeline must identify matching records from Source 2 and Source 3. A Source 1 entity may match zero (singleton), one, or multiple records from Source 2 and Source 3.

---

## 3. Problem Explanation
Entity resolution across massive, noisy commercial datasets presents several technical hurdles:

1. **Quadratic Pairwise Complexity:** Comparing all Source 1 records against all Source 2 and Source 3 records scales as $\mathcal{O}(N_{S1} \times (N_{S2} + N_{S3}))$. Efficient candidate generation (blocking) is required to reduce the search space to a manageable candidate pool while maintaining a near-100% recall ceiling.
2. **Noise Patterns:**
   - **Business Name Variations:** Legal suffix inconsistencies (`Corp` vs. `Corporation`, `Pvt Ltd` vs. `Private Limited`), abbreviations, trade names (DBA), punctuation (`&` vs. `and`), typos, and word transpositions.
   - **Address Inconsistencies:** Road abbreviations (`Rd` vs. `Road`, `St` vs. `Street`), missing PIN/postal codes, landmark references, and regional transliteration differences.
3. **Precision-Dominant Metric ($F_{0.5}$):** The competition evaluates submissions using the $F_{0.5}$ score, which penalizes false positives twice as heavily as false negatives. Correctly detecting singletons (entities with zero matches) is critical.
4. **Open-Set Country Distribution:** The training set covers records from `US` and `India`, whereas the test set introduces `France` (and potentially other open-set labels). The pipeline must handle open-set countries without filtering them out.

---

## 4. Dataset Structure Expected by the Challenge

All dataset files are tab-separated values (`.tsv`).

### Source Record Schema (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`)
Each source file contains the following 4 columns:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `entity_id` | String | Unique record identifier with source prefix (`S1-`, `S2-`, or `S3-`) |
| `business_name` | String | Business entity name (may contain abbreviations, noise, typos) |
| `business_address` | String | Business address string (may contain missing tokens, landmarks) |
| `country` | String | Country identifier (`US`, `India`, `France`, etc.) |

### Ground Truth Schema (`train_ground_truth.tsv` - Training Set Only)
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `source1_entity_id` | String | Unique identifier of a Source 1 entity (`S1-...`) |
| `matched_entity_ids` | String | Comma-separated list of matching Source 2 and Source 3 IDs (empty if singleton) |

*Note: The test set does not provide ground-truth labels.*

---

## 5. Overall Pipeline Architecture

The end-to-end pipeline is designed in modular stages:

```
[Raw Data: S1, S2, S3]
         │
         ▼
[1. Preprocessing & Normalization]
   - Legal suffix standardizing
   - Address token cleaning
   - Whitespace & case folding
         │
         ▼
[2. Blocking / Candidate Generation] ──────► [candidate_pairs.tsv]
   - Token & n-gram inverted indexing
   - Country & prefix partitioning
   - Search space reduction
         │
         ▼
[3. Pairwise Feature Engineering]
   - String distances (Levenshtein, Jaro-Winkler)
   - Token sort / Token set ratios
   - TF-IDF address/name overlaps
         │
         ▼
[4. ML Matching Model]
   - Pairwise binary classifier
   - Match probability scoring
         │
         ▼
[5. F0.5 Threshold Decision Engine]
   - Precision-weighted thresholding
   - Singleton identification
         │
         ▼
[6. Final Submission Assembly] ─────────────► [matching_results.tsv]
```

### Stage Artifacts:
* **Candidate Generation:** Generates `output/candidate_pairs.tsv` containing all candidate pairs generated before ML scoring.
* **Final Matching:** Generates `output/matching_results.tsv` containing the final filtered matches for every Source 1 entity.

---

## 6. Team Roles

| Member | Primary Focus Areas | Key Deliverables |
| :--- | :--- | :--- |
| **Member 1** | **EDA, Profiling & Preprocessing** | • Exploratory data analysis & noise profiling<br>• Text normalization & legal suffix dictionaries<br>• Address standardization pipeline (`src/preprocessing.py`) |
| **Member 2** | **Blocking & Candidate Generation** | • Inverted index & blocking key strategies<br>• High-recall candidate pair filtering<br>• Candidate set export (`src/blocking.py` → `candidate_pairs.tsv`) |
| **Member 3** | **ML Matching & Feature Engineering** | • Pairwise similarity feature extraction (`src/features.py`)<br>• Classification model training (`src/model.py`)<br>• Precision-focused $F_{0.5}$ threshold optimization |
| **Member 4** | **Integration & Submission Packaging** | • End-to-end pipeline orchestration (`src/predict.py`)<br>• Test inference & output validation (`matching_results.tsv`)<br>• Submission verification & technical documentation |

---

## 7. Repository Structure

```
amazon-ml-business-entity-resolution/
│
├── src/                               # Modular source code
│   ├── __init__.py                    # Package initializer
│   ├── preprocessing.py               # Text cleaning & normalization functions
│   ├── blocking.py                    # Candidate generation & blocking indices
│   ├── features.py                    # Pairwise similarity feature extraction
│   ├── model.py                       # ML model training & F0.5 threshold tuning
│   ├── predict.py                     # End-to-end test inference orchestration
│   └── utils.py                       # Shared I/O, config, logging, and metrics
│
├── notebooks/                         # Exploratory data analysis & prototyping
│   └── README.md                      # Team notebook practices and conventions
│
├── configs/                           # Pipeline parameter configurations
│   └── config.yaml                    # Configurable paths, thresholds, and seeds
│
├── output/                            # Local submission output files
│   └── .gitkeep                       # Preserves directory in git tracking
│
├── tests/                             # Unit tests and format validators
│   └── README.md                      # Testing guidelines
│
├── README.md                          # Project documentation
├── requirements.txt                   # Pinned lightweight dependencies
├── .gitignore                         # Strict exclusion rules for datasets and artifacts
└── CONTRIBUTING.md                    # Git collaboration rules & PR process
```

---

## 8. How to Install Dependencies

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MokshFF/amazon-ml-business-entity-resolution.git
   cd amazon-ml-business-entity-resolution
   ```

2. **Create and activate a virtual environment:**
   - **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```

3. **Install required packages:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 9. How to Run the Pipeline

1. **Configure paths:**  
   Ensure your local dataset directory is accessible and check `configs/config.yaml`.
2. **Execute end-to-end inference:**
   ```bash
   python -m src.predict --config configs/config.yaml
   ```
3. **Validate output format:**  
   Confirm generated files meet the official submission format requirements using your local validation script.

---

## 10. Git/GitHub Collaboration Workflow

Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for full details.

1. **Branches:**
   - Work only on dedicated feature branches:
     - `feature/eda`
     - `feature/blocking`
     - `feature/ml-matching`
     - `feature/integration`
2. **Pull Requests:**
   - Push feature branches to GitHub and open a Pull Request against `main`.
   - Never push directly to `main`.
3. **Data Safety:**
   - Never commit raw datasets, large TSVs, or model checkpoints to GitHub.

---

## 11. Important Challenge Constraints

* **Dataset Isolation:** The 2.5 GB dataset must remain strictly local or in private shared storage. It must never be committed to GitHub.
* **Prohibited Augmentations:** External business lookup, external databases, geocoding APIs, and internet-based identity enrichment are **strictly prohibited** by challenge rules.
* **Country Generalization:** The pipeline must support unseen countries (`France` in test) without crashing or dropping rows.
* **Leaderboard Metric ($F_{0.5}$):**
  $$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{(0.5^2 \times \text{Precision}) + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  High precision is weighted higher than recall; false matches incur a heavy penalty.

---

## 12. Output Files

The pipeline produces two tab-separated files saved under `output/`:

1. **`output/candidate_pairs.tsv`**
   - Candidate entity pairs produced by blocking before ML scoring.
   - Schema: `source1_entity_id\tcandidate_entity_ids`
2. **`output/matching_results.tsv`**
   - Final predicted entity matches evaluated on the leaderboard.
   - Schema: `source1_entity_id\tmatched_entity_ids`
   - Singletons have an empty string under `matched_entity_ids`.
