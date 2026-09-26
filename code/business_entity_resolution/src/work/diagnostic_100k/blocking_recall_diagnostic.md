# Blocking Recall Diagnostic

```text
Sampling 100,000 Source-1 rows (seed=42)...
Streaming full ground truth to collect true matches for the sampled ids...
Sampled 100,000 Source-1 ids: 94,419 have >=1 true match (345,707 true target links total, 167,308 in Source 2 / 178,399 in Source 3), 5,581 are true singletons.
Building target sample: Source 2 = 167,308 forced + up to 50,000 random fill; Source 3 = 178,399 forced + up to 50,000 random fill.
Final target sample sizes: Source 2 = 217,308, Source 3 = 228,399.
[production] building index (217,308 + 228,399 rows)...
  (pruned 1,835,921 stopword postings)
  (statistics and indexes built in 36.3s)
[production] index built in 188.0s
[production] distinct `source` values actually stored: ['source2', 'source3']
[production] recall pass over 100,000 rows done in 74.2s (4 worker(s))
[production] diagnosed 6,692 individual misses across 5,302 entities in 15.2s

==============================================================================
CANDIDATE RECALL SUMMARY (entities with >=1 true match only)
==============================================================================

--- production (stored source tags: ['source2', 'source3']) ---
  macro candidate recall : 0.9805
  micro candidate recall : 0.9806
  entities w/ >=1 hit, combined            : 94,210 / 94,419
  entities w/ >=1 hit, via exact_address    : 26,924 / 94,419
  entities w/ >=1 hit, via exact_name       : 76,642 / 94,419
  entities w/ >=1 hit, via term_overlap     : 94,208 / 94,419
  candidate-set size (true-match entities)  : n=94419 mean=291.40 median=300.0 p95=300.0 max=300
  candidate-set size (true singletons)      : n=5581 mean=290.90 median=300.0 p95=300.0 max=300
  entities with FULL hit (all true matches found)    : 89,117
  entities with PARTIAL hit (some but not all found) : 5,093
  entities with ZERO hit (nothing found)              : 209
  missed target links: 6,692, by likely cause:
    beyond_rank_cut             : 4,582
    no_shared_term              : 2,110
  -> see production/missed_target_diagnostics.csv for the full per-miss breakdown, and production/per_entity_candidate_recall.csv (missed_target_ids column) for which specific ids were missed per entity.

Per-entity CSVs and sampled tsvs are under: /home2/home/abhishek_chaudhari/Hackathon/student_resource/code/business_entity_resolution/src/work/diagnostic_100k
```
