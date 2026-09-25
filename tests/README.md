# Tests

This directory contains automated unit and integration tests to ensure code reliability across the pipeline.

## Testing Focus Areas

1. **Preprocessing Tests:**
   - Text normalization behavior (case folding, punctuation removal, whitespace collapsing).
   - Legal suffix standardization (e.g., `Corp` vs. `Corporation`, `Pvt Ltd` vs. `Private Limited`).
   - Handling of null, missing, or irregular address strings.
   - Validation that `country` strings are not altered or filtered out (especially `France` for the test set).

2. **Blocking Tests:**
   - Candidate generation returns valid candidate lists.
   - Every Source 1 entity appears exactly once in the candidates mapping.
   - Candidate IDs strictly belong to Source 2 (`S2-`) or Source 3 (`S3-`).
   - No duplicate candidate IDs within any entity's candidate list.

3. **Feature Engineering Tests:**
   - String similarity metrics (RapidFuzz Levenshtein, Jaro-Winkler, token ratios) produce valid float values in range `[0.0, 1.0]`.
   - Missing or empty strings produce non-NaN default values.

4. **Output Format Tests:**
   - Matching output file is strictly tab-separated (`sep='\t'`).
   - Output contains exact headers: `source1_entity_id\tmatched_entity_ids`.
   - Singletons have empty `matched_entity_ids`.
   - All matches are subsets of generated candidates.

## Running Tests

Run all unit tests using `pytest` or Python standard `unittest`:

```bash
python -m unittest discover tests
```
