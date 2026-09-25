# Business Entity Resolution

This project implements a reproducible baseline for the **ML Challenge 2026
Business Entity Resolution** task.

The goal is to identify which records in Source 2 and Source 3 represent the
same real-world business as each record in Source 1. Source 1 is the reference
source. A Source-1 record can match:

- no records;
- one record; or
- multiple records from Source 2 and/or Source 3.

The challenge is evaluated using macro F0.5. F0.5 weights precision more than
recall, so an incorrect merge is more harmful than a missed match. Correctly
identifying singleton entities with no matches is also important.

## Project location

The extracted challenge resources are located at:

```text
/home2/home/abhishek_chaudhari/Hackathon/student_resource/
```

The complete implementation is located at:

```text
/home2/home/abhishek_chaudhari/Hackathon/student_resource/code/business_entity_resolution/
```

The challenge archive is located at:

```text
/home2/home/abhishek_chaudhari/Hackathon/6ab10eb3b23ba_student_resource.zip
```

## Directory structure

```text
student_resource/
├── README.md
├── Documentation_template.md
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── utils/
│   └── validate_submission.py
└── code/
    └── business_entity_resolution/
        ├── README.md
        └── src/
            ├── __init__.py
            ├── entity_resolution.py
            └── evaluate.py
```

Generated databases, predictions, and other runtime files should be stored in a
separate `work/` or `output/` directory. They are not committed as part of the
source implementation.

## Dataset files

All files are UTF-8, tab-separated files. Always pass `sep="\t"` when using
Pandas or an equivalent explicit delimiter when using another parser.

### Training data

| File | Description |
|---|---|
| `dataset/train/train_source1.tsv` | Reference Source-1 business records |
| `dataset/train/train_source2.tsv` | Source-2 business records |
| `dataset/train/train_source3.tsv` | Source-3 business records |
| `dataset/train/train_ground_truth.tsv` | Source-1 to Source-2/3 matching labels |

The source files contain:

```text
entity_id
business_name
business_address
country
```

The ground-truth file contains:

```text
source1_entity_id
matched_entity_ids
```

`matched_entity_ids` is a comma-separated list. It is empty for a singleton
Source-1 entity.

### Test data

| File | Description |
|---|---|
| `dataset/test/test_source1.tsv` | Source-1 records requiring predictions |
| `dataset/test/test_source2.tsv` | Candidate Source-2 records |
| `dataset/test/test_source3.tsv` | Candidate Source-3 records |

The training data contains US and India. The test data also contains France.
The implementation treats country as an open string value and does not hard-code
the training countries.

## Dataset analysis completed

The supplied training data was profiled before implementing the baseline:

- Source 1: **2,206,821** records
- Source 2: **5,034,616** records
- Source 3: **5,285,603** records
- Ground-truth Source-1 rows: **2,206,821**
- Positive links: approximately **7.64 million**
- True singleton Source-1 entities: **123,247**
- Each Source-1 entity has between 0 and 11 labeled matches.
- Source-2 and Source-3 addresses are missing for a small minority of rows,
  so the matcher must not require an address.

A sampled analysis of 20,000 Source-1 records found that exact normalized
name/address blocks covered approximately **51.2% of positive links**. This
confirmed that exact blocking is a useful precision-oriented baseline but is
not sufficient for a final high-recall competition solution.

## Implemented pipeline

The current implementation is a disk-backed baseline with four stages.

### 1. Text normalization

The implementation:

- applies Unicode compatibility normalization;
- removes accents where possible;
- case-folds text;
- converts ampersands to `and`;
- removes punctuation;
- collapses whitespace;
- removes common legal suffixes from business names;
- sorts normalized name tokens;
- normalizes common address abbreviations such as `road`/`rd`,
  `street`/`st`, and `avenue`/`ave`.

The raw values remain available in the index. Normalized values are used for
blocking and similarity features.

### 2. Disk-backed target index

`entity_resolution.py build-index` creates a SQLite database containing Source 2
and Source 3 records. It stores:

- original entity ID and source;
- original business name and address;
- country;
- normalized name and address;
- compact normalized forms;
- selective token block keys.

SQLite is used so the full candidate sources do not have to be loaded into
Python memory at once.

### 3. Candidate generation

For each Source-1 record, candidates are collected using multiple country-aware
blocking rules:

- exact normalized business name;
- exact normalized address;
- compact normalized name;
- selective name-token keys;
- numeric address-token keys;
- a combined name/address block key when available.

The index also stores boundary-padded character 3- to 5-grams for the
normalized name and address. Corpus document frequencies provide TF-IDF
cosine similarity, and bottom-k MinHash signatures (16 hashes) are divided
into four LSH bands. Matching LSH buckets provide an approximate-nearest-neighbor
candidate set without loading the target corpus into memory. Exact blocks and
ANN results are unioned, remain country-aware, and are deduplicated before
matching. The resulting candidate set is written to `candidate_pairs.tsv` for
recall measurement and debugging.

To bound worst-case work from common ANN or token buckets, each ANN bucket and
selective block-key lookup contributes at most 500 rows. Exact name and address
lookups remain uncapped.

### 4. Conservative matching

Each candidate receives a simple score based on:

- exact normalized name agreement;
- exact normalized address agreement;
- normalized name token Jaccard similarity;
- normalized address token Jaccard similarity;
- a weighted combination of name and address similarity.

Candidates above the configured threshold are written to
`matching_results.tsv`. Matching is independent rather than one-to-one, so
valid one-to-many relationships are preserved.

The default threshold is `0.88`. It is intentionally exposed as a command-line
argument and should be tuned using held-out training data before final test
inference.

## Source files

### `src/entity_resolution.py`

Main pipeline implementation. It provides:

- `normalize_text`
- `normalize_name`
- `normalize_address`
- `block_keys`
- `build_index`
- `run_matching`

It has two command-line modes:

```text
build-index
match
```

### `src/evaluate.py`

Evaluates prediction files against the training ground truth using the
challenge's macro F0.5 definition. It handles:

- empty predictions;
- empty ground-truth lists;
- one-to-many matches;
- per-entity precision and recall;
- macro averaging across all Source-1 entities.

### `src/__init__.py`

Package marker for the implementation source directory.

## Environment

The implementation uses the Python standard library only. The tested workspace
interpreter is Python 3.10.

No external business databases, APIs, geocoding services, or data augmentation
were used. All processing is based only on the supplied challenge files.

## Running the pipeline on training data

Run these commands from the `student_resource/` directory:

```bash
cd /home2/home/abhishek_chaudhari/Hackathon/student_resource
mkdir -p work
```

Build an index over the training target sources:

```bash
python code/business_entity_resolution/src/entity_resolution.py build-index \
  --source dataset/train/train_source2.tsv \
  --source dataset/train/train_source3.tsv \
  --database work/train_targets.sqlite
```

Generate baseline predictions and candidates for the training Source-1 file:

```bash
python code/business_entity_resolution/src/entity_resolution.py match \
  --source1 dataset/train/train_source1.tsv \
  --database work/train_targets.sqlite \
  --output work/train_predictions.tsv \
  --candidate work/train_candidates.tsv \
  --threshold 0.88
```

Evaluate the predictions:

```bash
python code/business_entity_resolution/src/evaluate.py \
  --predictions work/train_predictions.tsv \
  --labels dataset/train/train_ground_truth.tsv
```

For meaningful threshold selection, use a deterministic held-out subset of the
training Source-1 entities. Do not tune a threshold on the same labels used to
report the final validation score.

The blocking recall diagnostic caches candidates once before sweeping multiple
thresholds. On Linux, pass `--workers 8` to score thresholds in parallel after
candidate retrieval; this avoids repeating the SQLite/TF-IDF retrieval work for
each threshold.

## Running test inference

Build an index from the test target sources:

```bash
cd /home2/home/abhishek_chaudhari/Hackathon/student_resource
mkdir -p output work

python code/business_entity_resolution/src/entity_resolution.py build-index \
  --source dataset/test/test_source2.tsv \
  --source dataset/test/test_source3.tsv \
  --database work/test_targets.sqlite
```

Generate the required output files:

```bash
python code/business_entity_resolution/src/entity_resolution.py match \
  --source1 dataset/test/test_source1.tsv \
  --database work/test_targets.sqlite \
  --output output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --threshold 0.88
```

The output files have the required headers:

```text
matching_results.tsv:
source1_entity_id    matched_entity_ids

candidate_pairs.tsv:
source1_entity_id    candidate_entity_ids
```

Every Source-1 test entity receives exactly one row. Empty lists are represented
by an empty second column.

## Validating submission files

Run the supplied validator from `student_resource/`:

```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

For the optional, memory-intensive ID existence check:

```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test \
  --check-ids
```

The validator checks:

- exact tab-separated headers;
- one row per test Source-1 entity;
- duplicate Source-1 rows;
- duplicate IDs inside lists;
- invalid Source-1 self-matches;
- invalid ID prefixes;
- IDs that do not exist in the test target files when `--check-ids` is used;
- whether final matches are contained in the candidate file.

## Current status

### Completed

- Extracted and verified the clean challenge archive.
- Profiled the full training schema, missingness, country distribution, and
  ground-truth match cardinalities.
- Added a macro F0.5 evaluator matching the challenge metric.
- Implemented Unicode, name, and address normalization.
- Implemented a disk-backed SQLite target index.
- Implemented exact and selective token blocking.
- Implemented conservative pair scoring.
- Implemented required matching and candidate output formats.
- Verified Python syntax and editor diagnostics.
- Verified the pipeline with a small end-to-end smoke test.
- Measured the limitations of exact normalized blocking on a training sample.

### Not yet completed

The current code is a reproducible baseline, not the final leaderboard model.
The following work remains before a competition-grade submission:

1. Build a deterministic held-out validation runner over the full training
   corpus.
2. Measure candidate recall and candidate-set size for every blocking rule.
3. Add character n-gram or approximate nearest-neighbor blocking.
4. Generate hard negatives from ambiguous blocks.
5. Train and calibrate a pair classifier using name, address, country, numeric
   token, and ambiguity features.
6. Tune separate thresholds for Source 2 and Source 3 and for missing-address
   cases.
7. Add entity-level ambiguity and singleton guards.
8. Run full test inference with the selected threshold policy.
9. Validate final outputs with `validate_submission.py`.
10. Fill in `Documentation_template.md` with measured validation results and
    package the reproducible code and outputs.

## Important limitations of the current baseline

- Exact and selective token blocking can still miss heavily corrupted names,
  transliterations, and records with incomplete addresses.
- The current scorer is a deterministic heuristic, not a trained classifier.
- The default threshold has not yet been selected using a full held-out
  validation sweep.
- The complete multi-million-row index build requires substantial time and disk
  I/O. Runtime should be benchmarked on the target machine before final
  submission.
- No final test predictions are claimed by this README. Predictions should be
  generated only after validation-based threshold selection.

## Final submission layout

The challenge expects a package with this structure:

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

The current implementation does not include a generated final output or a
filled methodology document yet. Those should be added after the model and
threshold policy have been validated.
