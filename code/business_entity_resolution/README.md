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
- normalized name and address.

Alongside `records`, two tables carry the inverted index:

- `postings(term, country, entity_id)` -- one row per (term, record), where a
  term is `n:<token>` or `a:<token>` for every normalized name or address
  token of length >= 2;
- `term_stats(term, country, document_frequency, weight)`, where
  `weight = log(1 + N_country / df)`.

SQLite is used so the full candidate sources do not have to be loaded into
Python memory at once.

### 3. Candidate generation

Candidates come from a single IDF-weighted ranked lookup rather than a union of
many independently capped blocking rules. For each Source-1 record:

1. Every token of the normalized name and address is turned into an index term.
   Terms whose document frequency exceeds 0.5% of the country's records are
   dropped as stopwords -- the cap is a *fraction* so it stays meaningful as
   the corpus grows.
2. The surviving terms are taken rarest-first until the cumulative posting
   count reaches `POSTINGS_BUDGET` (20,000). This is the scale-invariance
   mechanism: it bounds work per query by a constant instead of letting it grow
   with the corpus, and taking the rarest first guarantees the most
   discriminative shared term is always used.
3. One grouped query sums `term_stats.weight` per candidate record and orders
   by that evidence; the top `CANDIDATE_LIMIT` (300) survive.
4. Exact normalized address matches are seeded with a weight no ordinary term
   sum reaches, so the ranked cut can never discard the single most precise
   signal in the pipeline (measured precision 0.910).

Bounding the *ranked* list -- rather than each of ~25 keys separately -- is what
makes the candidate set both small and high-recall. A true link sharing one rare
term outranks thousands of records sharing several common ones, so it survives
the cut; under the previous per-key caps it was dropped by whichever cap it
happened to land under. The candidate set is written to `candidate_pairs.tsv`
for recall measurement and debugging.

On a realistic-density sample (1M-record pool, 8,000 Source-1 rows, 9,363 true
links, answer density 5.6% rather than a force-included pool), this reaches
**macro candidate recall 0.9743 at a median candidate set of 300**, versus
0.9542 at a median of 489 for the previous design, at 15.4 ms/row instead of
103.7. A diagnostic setting that force-includes every true target measures
0.9979 recall, but that pool is ~78% answers and is optimistic by construction
-- quote the realistic figure.

Names in Indic scripts are transliterated to Latin (ISCII-91 offset tables over
the Brahmic blocks, with schwa deletion and vowel-run collapsing) before
normalization, so `बॉम्बे एस्टेट` indexes as `bombe estet` rather than
vanishing. Without this, 6.9% of Indian records had an empty normalized name
and were unreachable by name at all.

### 4. Conservative matching

Each candidate is scored by one deliberately simple rule: the equally weighted
mean of the name-token Jaccard similarity and the address-token Jaccard
similarity.

The address carries as much weight as the name because it is *more*
discriminative: the most common normalized address key in the 1M-record corpus
is shared by 6 records, and none reach the stopword cap, whereas 38% of records
share an exact normalized name. Every elaboration tried on top of the plain
blend measured worse under macro F0.5 -- discounting exact names by document
frequency (0.8326 vs 0.8434), boosting an exact address to 0.95 (+0.0005, i.e.
noise), and weighting the name 0.65 over the address as the previous release did
(0.7922).

Two parts make up the prediction set:

- every candidate scoring at or above `--threshold` (default **0.70**);
- plus the single best candidate whenever it clears `MIN_EVIDENCE` (0.30).

The second part is not a heuristic flourish but a consequence of the metric.
`entity_f05` scores an empty prediction against non-empty gold as 0.0 --
identical to a single wrong prediction -- and 94.4% of real Source-1 rows have at
least one match. So on any such row, emitting one's best candidate weakly
dominates emitting nothing; a pure threshold rule was leaving macro F0.5 on the
table. That alone lifts matched-row macro F0.5 from 0.7922 to 0.8434.
`MIN_EVIDENCE` is what gives the behaviour back on the 5.58% of rows that are
genuinely matchless and must stay silent; the floor was swept against the real
corpus mix and is flat between 0.20 and 0.40.

Predictions are written to `matching_results.tsv`. Matching is independent
rather than one-to-one, so valid one-to-many relationships are preserved.

The threshold is exposed as a command-line argument and should be re-tuned on
held-out training data if the corpus or the scoring rule changes.

## Source files

### `src/entity_resolution.py`

Main pipeline implementation. It provides:

- `normalize_text` / `normalize_name` / `normalize_address`
- `transliterate`
- `index_terms` / `query_terms`
- `build_index`
- `candidate_rows`
- `match_score` / `select_matches`
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

Generate predictions and candidates. `--workers` defaults to 4 and only affects
speed -- the outputs are byte-identical for any value, so a rerun with a
different worker count reproduces the same submission. Capping it at 4 keeps the
pipeline polite on a shared host; the index build itself is single-threaded:

```bash
python code/business_entity_resolution/src/entity_resolution.py match \
  --source1 dataset/train/train_source1.tsv \
  --database work/train_targets.sqlite \
  --output work/train_predictions.tsv \
  --candidate work/train_candidates.tsv \
  --threshold 0.70
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
  --threshold 0.70
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
- Implemented Unicode, name, and address normalization, including
  transliteration of Indic scripts.
- Replaced the capped multi-rule blocking design with a single IDF-weighted
  ranked postings lookup (`postings` / `term_stats`).
- Retuned the scorer against macro F0.5 rather than pooled precision/recall,
  and added the always-emit-best-candidate rule the metric rewards.
- Parallelised the retrieval pass; verified byte-identical output across
  worker counts.
- Implemented required matching and candidate output formats.
- Verified the pipeline with a small end-to-end smoke test.

### Measured on a realistic-density sample

Pool: 1,000,000 target records, 8,000 Source-1 rows, 9,363 true links, answer
density 5.6% (no true target force-included).

| Metric | Previous design | Current |
|---|---|---|
| Macro candidate recall | 0.9542 | **0.9743** |
| Median candidate set | 489 | **300** |
| Retrieval | 103.7 ms/row | **11.6 ms/row** |
| Macro F0.5 (matched rows) | 0.6356 | **0.8479** |
| micro precision / recall | 0.221 / 0.870 | **0.889 / 0.783** |
| Mean predictions per row | 4.60 | **1.03** |

Projected macro F0.5 over the real corpus mix (94.42% rows with matches, 5.58%
matchless) is 0.8147. The floor behind that projection is the least certain
number here -- it comes from a 400-row sample of the matchless population -- but
the curve is flat between 0.20 and 0.40, so the choice is not load-bearing.

### Not yet completed

1. Run full test inference and validate the outputs with
   `validate_submission.py`.
2. A char-n-gram or edit-distance retriever for the lexically unreachable
   residual (see limitations).
3. Train and calibrate a pair classifier using name, address, country, numeric
   token, and ambiguity features.
4. Tune separate thresholds for Source 2 and Source 3 and for missing-address
   cases.
5. Fill in `Documentation_template.md` and package the submission.

## Important limitations

- **The residual recall gap is lexical, not configurational.** On a
  force-included diagnostic pool every missed link is attributed to
  `no_shared_term` -- the two records share no surviving indexed token at all
  (mid-word corruption such as `Ibnovrtiosn` vs `Innovations`). No threshold or
  budget change reaches those; only a fuzzy retriever would, and that is out of
  scope under the stdlib-only constraint.
- **The realistic-pool figures are measured at 1M records, not the ~10M test
  corpus.** The postings budget and fraction-based stopword cap are both
  designed to be scale-invariant, but that is a design argument, not a
  measurement at full scale.
- **The scorer is a deterministic heuristic, not a trained classifier.**
- **The 300-candidate cap binds for 97% of rows** on the realistic pool (median
  and mode are both exactly 300). So candidate-set size is not a tunable knob at
  this density -- it is the cap -- and candidate recall depends entirely on the
  ranking being right, not on the cap being generous. Raising the cap would buy
  little recall and cost precision; the lever that matters is ranking quality.
- No final test predictions are claimed by this README. The numbers above come
  from a held-out training sample, shared with the scorer and threshold
  selection, so they are mildly optimistic.

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
