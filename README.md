# Amazon ML Challenge 2026: Business Entity Resolution

## 1. Challenge Overview
**Challenge:** Amazon ML Challenge 2026 – Business Entity Resolution Challenge  
**Repository:** [amazon-ml-business-entity-resolution](https://github.com/MokshFF/amazon-ml-business-entity-resolution)  
**Team:** Moksh & Dhwaj (Team Leader) — *Strict 50/50 Technical Contribution*

In large-scale commercial platforms, business records arrive from multiple independent, disparate data sources. Each source provides partial, noisy, and unstructured descriptions without shared global identifiers. The objective of this challenge is to construct a scalable, high-precision Machine Learning entity resolution system that maps noisy business records across three independent sources to the same real-world business entity.

---

## 2. Problem Statement
Given three independent data sources:
* **Source 1:** Deduplicated reference source.
* **Source 2:** Noisy business records.
* **Source 3:** Noisy business records.

For each record in Source 1, the pipeline must identify all matching records from Source 2 and Source 3. A Source 1 entity may match:
* **Zero records** (singleton / no-match case)
* **One record**
* **Multiple records**

### Technical Complexity:
1. **Large Scale (~2.5 GB data):** Comparing every Source 1 entity against all records in Source 2 and Source 3 ($\mathcal{O}(N_{S1} \times (N_{S2} + N_{S3}))$) is computationally intractable and would exceed available RAM. Efficient blocking / candidate generation is required.
2. **Noise and Variation:** Real-world inconsistencies include typos, legal suffixes (`Corp`, `Pvt Ltd`, `LLC`), abbreviations (`Rd` vs. `Road`), landmark references, missing PIN/postal codes, and transliteration differences.
3. **Evaluation Metric ($F_{0.5}$):** The challenge evaluates submissions with the $F_{0.5}$ score, which penalizes false positives twice as heavily as false negatives ($\beta = 0.5$). Correctly detecting singletons is essential to avoid severe precision penalties.
4. **Open-Set Countries:** Training data covers records from `US` and `India`, whereas the test set introduces `France` (and potentially other open-set labels). The pipeline must dynamically handle unseen countries.

---

## 3. Dataset Structure

All data files are tab-separated values (`.tsv`). The dataset remains outside Git tracking to maintain repository health.

```
dataset/
├── train/
│   ├── train_source1.tsv           # Source 1 deduplicated reference records
│   ├── train_source2.tsv           # Source 2 noisy records
│   ├── train_source3.tsv           # Source 3 noisy records
│   └── train_ground_truth.tsv     # Ground-truth entity matches
└── test/
    ├── test_source1.tsv            # Source 1 evaluation records
    ├── test_source2.tsv            # Source 2 candidate records
    └── test_source3.tsv            # Source 3 candidate records
```

### Record Columns (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`)
* `entity_id`: Unique identifier (`S1-...`, `S2-...`, `S3-...`)
* `business_name`: Entity name string
* `business_address`: Entity address string
* `country`: Country label (`US`, `India`, `France`, etc.)

### Ground Truth Columns (`train_ground_truth.tsv`)
* `source1_entity_id`: Source 1 identifier
* `matched_entity_ids`: Comma-separated list of matching Source 2 / Source 3 IDs (empty string for singletons)

---

## 4. End-to-End Pipeline Architecture

We adhere to a validation-driven development philosophy:
```
Raw Data (S1, S2, S3)
       │
       ▼
1. Preprocessing & Normalization ──────► Data-driven cleaning (backed by EDA)
       │
       ▼
2. Multi-Strategy Blocking ────────────► candidate_pairs.tsv (Candidate pool)
       │
       ▼
3. Similarity Feature Engineering ─────► Pairwise similarity matrix
       │
       ▼
4. ML Matching Model ──────────────────► Pairwise match probability scoring
       │
       ▼
5. Threshold Decision Engine ──────────► Precision-heavy tuning for F0.5
       │
       ▼
6. Final Output Generation ────────────► matching_results.tsv
```

* **`candidate_pairs.tsv`:** The candidate set passed directly to the final matching model.
* **`matching_results.tsv`:** The final matched entities uploaded to the competition portal.

---

## 5. Team & 50/50 Technical Ownership

Both active members maintain a full end-to-end understanding of the system and write core code. Leadership areas divide exploratory experiments evenly:

| Stage | Lead: Moksh | Lead: Dhwaj (Team Leader) |
| :--- | :--- | :--- |
| **EDA & Profiling** | EDA Experiment A (`notebooks/eda_moksh.ipynb`): Name noise, suffixes, typos | EDA Experiment B (`notebooks/eda_dhwaj.ipynb`): Address noise, landmarks, countries, cardinality |
| **Preprocessing** | Name cleaning (`normalize_name` in `src/preprocessing.py`) | Address & country cleaning (`normalize_address`, `normalize_country`) |
| **Blocking / Candidates** | Strategy A (`BlockingStrategyA`): Name token index, prefix keys | Strategy B (`BlockingStrategyB`): Country partitioning, address token index |
| **Feature Engineering** | Name similarity metrics (Levenshtein, Jaro-Winkler, token overlap) | Address & context features (token sets, country match, source indicator) |
| **ML Model** | Model Experiment A: Baseline Logistic Regression & linear boundaries | Model Experiment B: Tabular classifiers (Random Forest / GBDT) |
| **Threshold Tuning** | Threshold Experiment A: Decision boundary sweep | Threshold Experiment B: Singleton vs multi-match precision sweep |
| **Error Analysis** | Error Analysis A: Dissecting name-driven errors | Error Analysis B: Dissecting address-driven errors |
| **Integration** | Pipeline orchestration & memory profiling | Output verification & submission packaging |

---

## 6. Installation & Environment Setup

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

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 7. Running the Pipeline

1. **Verify configuration:** Inspect `configs/config.yaml` to ensure dataset paths point to your local data folder.
2. **Execute end-to-end test inference:**
   ```bash
   python -m src.predict --config configs/config.yaml
   ```
3. **Outputs generated:**
   - `output/candidate_pairs.tsv`
   - `output/matching_results.tsv`

---

## 8. Validation Strategy ($F_{0.5}$)

* **Holdout Split:** We hold out an internal validation split from the training dataset. We never evaluate or train on the test set.
* **Metric:** Entity-level $F_{0.5}$ computed at the Source 1 entity level:
  $$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{(0.5^2 \times \text{Precision}) + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
* **Threshold Sweeping:** Systematic search across decision thresholds (`[0.50, 0.55, ..., 0.95]`) to maximize the precision-heavy objective.
* **Singleton Tracking:** Rigorous monitoring of false merges on true singletons.

---

## 9. Output Format

Both files are tab-separated (`sep='\t'`) with exact schemas:

1. **`output/candidate_pairs.tsv`:**
   ```tsv
   source1_entity_id	candidate_entity_ids
   S1-00001	S2-00047,S2-00193,S3-00812
   S1-00002	S3-00004
   S1-00003	
   ```

2. **`output/matching_results.tsv`:**
   ```tsv
   source1_entity_id	matched_entity_ids
   S1-00001	S2-00047,S3-00812
   S1-00002	S3-00004
   S1-00003	
   ```

---

## 10. Git Collaboration Workflow

Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for full instructions:
* **Branches:** `main` (stable), `moksh-work`, and `dhwaj-work`.
* **Commits:** Meaningful, descriptive commit messages (`"Add name normalization"`, `"Implement candidate blocking"`).
* **Code Review:** Mutual Pull Request review before merging into `main`.
* **Data Hygiene:** Never commit dataset files or oversized outputs.

---

## 11. Challenge Restrictions & Guidelines

* **No External Data:** External business lookups, Google Maps, web scraping, and government registry queries are **strictly prohibited**.
* **No Geocoding APIs:** All matching must be derived purely from the provided textual data.
* **No Dataset Commits:** The 2.5 GB dataset remains local/shared only.
* **Licensing & Model Size:** Prohibited from using oversized models or models that violate open-source licensing rules.
