#!/usr/bin/env python3
"""
blocking_recall_diagnostic.py

Step-1 diagnostic for the Business Entity Resolution baseline: measures
candidate-blocking recall (broken down by *which* retrieval mechanism found
each true match) and the precision/recall/macro-F0.5 curve of the current
conservative scorer, on a bounded-size sample of the training data.

--------------------------------------------------------------------------
WHY THIS ISN'T A PLAIN "RANDOM 100k / RANDOM 100k" SAMPLE
--------------------------------------------------------------------------
Source 1 has ~2.2M records with ~7.64M positive links against a combined
Source 2 + Source 3 pool of ~10.3M records. The positive rate is low enough
that an *independent* random 100k Source-1 sample against an *independent*
random 100k Source-2/3 sample would, by chance, contain almost none of the
true matches for those specific Source-1 records -- recall would come out
near zero and would tell you nothing about the blocking rules.

Instead this script:
  1. Randomly samples `--source1-sample-size` (default 100,000) Source-1
     records.
  2. Reads the FULL ground truth once (streaming) and pulls out every true
     target id for exactly those sampled Source-1 records. These are
     force-included in the target sample, so recall is measured against the
     complete true positive set for the sampled entities -- not degraded by
     sampling loss on the target side.
  3. Fills the remaining target budget (`--target-fill-size`, default
     100,000 total, split across Source 2 / Source 3 by their relative
     sizes) with randomly sampled DISTRACTOR records, so the candidate
     generator still has to do real work (block collisions, ANN buckets
     with unrelated neighbours, etc.) rather than being handed only the
     answers.

--------------------------------------------------------------------------
SOURCE-TAG AND CANDIDATE-SIZE DIAGNOSTICS
--------------------------------------------------------------------------
The current entity_resolution.py build_index() tags every indexed record with:

    source = source_path.stem.split("_", 1)[1]

This makes the normal train_source2.tsv and train_source3.tsv filenames
compatible with the ANN/LSH and selective block-key source filters. Each
selective block and ANN bucket is also capped to bound candidate-set size.

This script builds an index over the sampled data and reports:
  - candidate recall overall and by retrieval mechanism;
  - candidate-set size distributions for matched and singleton entities.

It then reports candidate recall under both indexes side by side, plus a
per-mechanism breakdown (exact name / exact compact name / exact address /
ANN-LSH / selective block key) so you can see exactly how much each
mechanism is (or, currently, isn't) contributing.

--------------------------------------------------------------------------
WHAT IT MEASURES
--------------------------------------------------------------------------
1. Candidate recall (micro and macro-over-entities-with-matches), overall
   and split by mechanism, by source (S2 vs S3), and by whether the target
   record's address was empty.
2. Candidate-set size distribution (mean/median/p95), split by whether the
   Source-1 entity is a true singleton or has real matches -- this is the
   workload / precision-burden a future pair classifier would have to sort
   through.
3. Precision / recall / macro-F0.5 of the CURRENT conservative scorer
   (entity_resolution.select_matches), swept over `--thresholds`, computed
   with your actual evaluate.py (imported directly, not reimplemented) so
   the numbers are guaranteed consistent with how the challenge is scored.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python blocking_recall_diagnostic.py \\
        --student-resource-dir /home2/home/abhishek_chaudhari/Hackathon/student_resource \\
        --work-dir work/diagnostic_100k \\
        --source1-sample-size 100000 \\
        --target-fill-size 100000 \\
        --seed 42 \\
        --thresholds 0.5 0.6 0.7 0.75 0.8 0.85 0.88 0.9 0.95

For a fast sanity check before committing to the full run:

    python blocking_recall_diagnostic.py --student-resource-dir <dir> \\
        --source1-sample-size 2000 --target-fill-size 2000 --quick

Assumptions baked in (all verified against the uploaded source, not
guessed): SOURCE tsv header is
    entity_id  business_name  business_address  country
ground-truth tsv header is
    source1_entity_id  matched_entity_ids
matched_entity_ids / candidate ids are comma-separated; entity_id prefixes
S1-/S2-/S3- identify source per the challenge README. `entity_resolution.py`
and `evaluate.py` are imported directly (not subprocessed), so this always
uses the exact scoring/matching logic your pipeline uses.
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import multiprocessing as mp
import random
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

SOURCE_HEADER = ("entity_id", "business_name", "business_address", "country")
GT_HEADER = ("source1_entity_id", "matched_entity_ids")

_SWEEP_CONTEXT = None


def _score_threshold(threshold: float) -> Tuple[float, float]:
    er, ev, source1_rows, source1_ids, labels, candidate_cache = _SWEEP_CONTEXT
    predictions: Dict[str, Set[str]] = {}
    for row, sid, candidates in zip(source1_rows, source1_ids, candidate_cache):
        matches, _candidate_ids = er.select_matches(row, candidates, threshold)
        predictions[sid] = set(matches)
    return threshold, ev.macro_f05(predictions, labels)


def _init_sweep_worker(context) -> None:
    global _SWEEP_CONTEXT
    _SWEEP_CONTEXT = context


# --------------------------------------------------------------------------
# Recall-pass worker: parallelizes attributed_candidates() across rows.
# Each worker opens its OWN sqlite connection (read-only) -- sqlite
# connections must not be shared across processes -- and only returns
# small, aggregated per-row results (never the full candidate set) to keep
# inter-process pickling cheap even when candidate sets are large.
# --------------------------------------------------------------------------

_RECALL_CONTEXT = None


def _init_recall_worker(context) -> None:
    global _RECALL_CONTEXT
    _RECALL_CONTEXT = context


def chunk_list(items: List, n_chunks: int) -> List[List]:
    n_chunks = max(1, n_chunks)
    size = max(1, -(-len(items) // n_chunks))  # ceil division
    chunks = [items[i:i + size] for i in range(0, len(items), size)]
    return chunks or [[]]


def _recall_chunk(chunk: List[Tuple[dict, str]]) -> List[dict]:
    er, db_path, labels = _RECALL_CONTEXT
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        results = []
        for row, sid in chunk:
            true_set = labels.get(sid, set())
            mechanisms = attributed_candidates(er, connection, row)
            combined = set(mechanisms)
            hit = combined & true_set
            group_hit = {}
            for group, tags in MECHANISM_GROUPS.items():
                group_candidates = {eid for eid, m in mechanisms.items() if m & tags}
                group_hit[group] = bool(group_candidates & true_set)
            results.append({
                "sid": sid,
                "true_count": len(true_set),
                "candidate_count": len(combined),
                "hit_count": len(hit),
                # Only the DELTA is returned, not the full candidate set --
                # true_set is small (a handful of ids at most), so this stays
                # cheap to pickle even though `combined` can be large.
                "missed_ids": sorted(true_set - combined),
                "group_hit_combined": bool(hit) if true_set else False,
                "group_hit_exact_any": group_hit["exact_any"] if true_set else False,
                "group_hit_ann_lsh": group_hit["ann_lsh"] if true_set else False,
                "group_hit_selective_block_key": group_hit["selective_block_key"] if true_set else False,
            })
        return results
    finally:
        connection.close()


def run_recall_pass(
    er, db_path: Path, source1_rows: List[dict], source1_ids: List[str],
    labels: Dict[str, Set[str]], workers: int,
) -> List[dict]:
    """Runs attributed_candidates() over every sampled Source-1 row, in
    parallel across `workers` processes if requested. Returns one small dict
    per row (see _recall_chunk above) -- never the full candidate sets, to
    keep this cheap even at 100k+ rows with hundreds of candidates each."""
    pairs = list(zip(source1_rows, source1_ids))
    if workers <= 1:
        _init_recall_worker((er, str(db_path), labels))
        chunks = chunk_list(pairs, 1)
        return [item for chunk in chunks for item in _recall_chunk(chunk)]
    chunks = chunk_list(pairs, workers * 4)
    context_manager = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    with context_manager.Pool(processes=workers, initializer=_init_recall_worker,
                               initargs=((er, str(db_path), labels),)) as pool:
        chunk_results = pool.map(_recall_chunk, chunks)
    return [item for chunk in chunk_results for item in chunk]


# --------------------------------------------------------------------------
# Miss diagnosis: for every Source-1 record that missed at least one true
# target, figure out WHY that specific target wasn't retrieved. This only
# runs over the (typically small) set of actual misses, so it stays cheap
# even though it does a few extra targeted queries per miss.
# --------------------------------------------------------------------------

def diagnose_misses(
    er, db_path: Path, miss_rows: List[Tuple[dict, str, List[str]]],
) -> List[dict]:
    """miss_rows: list of (source1_row, source1_id, missed_target_ids)."""
    connection = sqlite3.connect(db_path)
    diagnoses = []
    for row, sid, missed_ids in miss_rows:
        name_key = er.normalize_name(row["business_name"])
        address_key = er.normalize_address(row["business_address"])
        query_search_key = er.search_text(name_key, address_key)
        query_ngrams = er.character_ngrams(query_search_key)
        query_buckets = set(er.ann_bucket_keys(er.minhash_signature(query_ngrams)))
        query_block_keys = set(er.block_keys(name_key, address_key))

        for target_id in missed_ids:
            target = connection.execute(
                "SELECT business_name, business_address, country, source "
                "FROM records WHERE entity_id = ?",
                (target_id,),
            ).fetchone()
            if target is None:
                diagnoses.append({
                    "source1_entity_id": sid, "missed_target_id": target_id,
                    "reason": "target_not_in_sampled_index",
                    "country_mismatch": "", "ann_bucket_truncated": "",
                    "block_key_truncated": "",
                })
                continue
            target_name, target_address, target_country, target_source = target

            country_mismatch = row["country"] != target_country

            # Was this target actually IN one of the query's LSH buckets,
            # just excluded by the ANN_BUCKET_LIMIT cap on the query side?
            ann_truncated = False
            if not country_mismatch and query_buckets:
                placeholders = ",".join("?" for _ in query_buckets)
                found = connection.execute(
                    f"SELECT 1 FROM ann_buckets WHERE entity_id = ? "
                    f"AND bucket_key IN ({placeholders}) LIMIT 1",
                    (target_id, *query_buckets),
                ).fetchone()
                ann_truncated = bool(found)

            # Same check for selective block keys.
            block_key_truncated = False
            if not country_mismatch and query_block_keys:
                placeholders = ",".join("?" for _ in query_block_keys)
                found = connection.execute(
                    f"SELECT 1 FROM block_keys WHERE entity_id = ? "
                    f"AND block_key IN ({placeholders}) LIMIT 1",
                    (target_id, *query_block_keys),
                ).fetchone()
                block_key_truncated = bool(found)

            if country_mismatch:
                reason = "country_mismatch"
            elif ann_truncated or block_key_truncated:
                reason = "truncated_by_limit"
            else:
                reason = "no_mechanism_overlap"

            diagnoses.append({
                "source1_entity_id": sid,
                "missed_target_id": target_id,
                "reason": reason,
                "country_mismatch": country_mismatch,
                "ann_bucket_truncated": ann_truncated,
                "block_key_truncated": block_key_truncated,
            })
    connection.close()
    return diagnoses


# --------------------------------------------------------------------------
# Streaming sampling helpers (single pass, O(sample size) memory -- safe for
# multi-million-row files)
# --------------------------------------------------------------------------

def _check_header(actual, expected, path):
    if tuple(actual or ()) != expected:
        raise ValueError(f"{path} has unexpected header {actual}, expected {expected}")


def reservoir_sample_tsv(path: Path, k: int, seed: int) -> Tuple[List[str], List[dict]]:
    """Uniform random sample of k rows from a large tsv, single streaming pass."""
    rng = random.Random(seed)
    sample: List[dict] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        _check_header(reader.fieldnames, SOURCE_HEADER, path)
        for index, row in enumerate(reader):
            if index < k:
                sample.append(row)
            else:
                j = rng.randint(0, index)
                if j < k:
                    sample[j] = row
    return list(SOURCE_HEADER), sample


def extract_and_fill(
    path: Path, forced_ids: Set[str], fill_budget: int, seed: int
) -> List[dict]:
    """One streaming pass: keep every row whose entity_id is in forced_ids
    (unconditionally), and reservoir-sample fill_budget more rows from the
    remainder as distractors."""
    rng = random.Random(seed)
    forced_rows: List[dict] = []
    fill_sample: List[dict] = []
    fill_seen = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        _check_header(reader.fieldnames, SOURCE_HEADER, path)
        for row in reader:
            if row["entity_id"] in forced_ids:
                forced_rows.append(row)
                continue
            if fill_budget <= 0:
                continue
            if fill_seen < fill_budget:
                fill_sample.append(row)
            else:
                j = rng.randint(0, fill_seen)
                if j < fill_budget:
                    fill_sample[j] = row
            fill_seen += 1
    found_ids = {row["entity_id"] for row in forced_rows}
    missing = forced_ids - found_ids
    if missing:
        print(
            f"  WARNING: {len(missing)} forced ids from ground truth were not "
            f"found in {path.name} (stale/mismatched ground truth?). "
            f"Example: {sorted(missing)[:3]}",
            file=sys.stderr,
        )
    return forced_rows + fill_sample


def collect_ground_truth(gt_path: Path, wanted_ids: Set[str]) -> Dict[str, Set[str]]:
    """Single streaming pass over the full ground truth, keeping only rows
    for the sampled Source-1 ids."""
    result: Dict[str, Set[str]] = {}
    with gt_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        _check_header(reader.fieldnames, GT_HEADER, gt_path)
        for row in reader:
            sid = row["source1_entity_id"]
            if sid in wanted_ids:
                result[sid] = set(filter(None, row["matched_entity_ids"].split(",")))
    missing = wanted_ids - set(result)
    if missing:
        print(
            f"  WARNING: {len(missing)} sampled Source-1 ids had no ground-truth "
            f"row at all (unexpected -- every Source-1 id should have one, even "
            f"if empty). Treating them as empty-match singletons.",
            file=sys.stderr,
        )
        for sid in missing:
            result[sid] = set()
    return result


def write_tsv(path: Path, header: Iterable[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_ground_truth(path: Path, labels: Dict[str, Set[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(list(GT_HEADER))
        for sid, matches in labels.items():
            writer.writerow([sid, ",".join(sorted(matches))])


# --------------------------------------------------------------------------
# Percentile helper (stdlib-only, no numpy dependency, matches the
# project's "Python standard library only" environment note)
# --------------------------------------------------------------------------

def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def summarize_sizes(values: List[int]) -> str:
    if not values:
        return "n=0"
    return (
        f"n={len(values)} mean={statistics.mean(values):.2f} "
        f"median={statistics.median(values):.1f} p95={percentile(values, 95):.1f} "
        f"max={max(values)}"
    )


# --------------------------------------------------------------------------
# Attributed candidate retrieval -- mirrors entity_resolution.candidate_rows
# exactly (same SQL, same source filter, including its current bug), but
# tags each candidate with WHICH mechanism found it, and can be pointed at
# the production database.
# --------------------------------------------------------------------------

def attributed_candidates(er, connection: sqlite3.Connection, row: dict) -> Dict[str, Set[str]]:
    name_key = er.normalize_name(row["business_name"])
    address_key = er.normalize_address(row["business_address"])
    country = row["country"]
    query_search_key = er.search_text(name_key, address_key)
    query_ngrams = er.character_ngrams(query_search_key)
    query_buckets = er.ann_bucket_keys(er.minhash_signature(query_ngrams))

    mechanisms: Dict[str, Set[str]] = defaultdict(set)

    if name_key:
        for (eid,) in connection.execute(
            "SELECT entity_id FROM records WHERE country = ? AND name_key = ?",
            (country, name_key),
        ):
            mechanisms[eid].add("exact_name")
        for (eid,) in connection.execute(
            "SELECT entity_id FROM records WHERE country = ? AND name_compact = ?",
            (country, er.compact_key(name_key)),
        ):
            mechanisms[eid].add("exact_name_compact")
    if address_key:
        for (eid,) in connection.execute(
            "SELECT entity_id FROM records WHERE country = ? AND address_key = ?",
            (country, address_key),
        ):
            mechanisms[eid].add("exact_address")
    ann_limit = er.adaptive_limit(
        er.ANN_BUCKET_LIMIT, len(query_buckets), er.ANN_BUCKET_MIN_LIMIT
    )
    for bucket in query_buckets:
        for (eid,) in connection.execute(
            "SELECT r.entity_id FROM ann_buckets b JOIN records r "
            "ON r.entity_id = b.entity_id WHERE b.bucket_key = ? "
            "AND b.source IN ('source2', 'source3') AND r.country = ? "
            "ORDER BY b.entity_id LIMIT ?",
            (bucket, country, ann_limit),
        ):
            mechanisms[eid].add("ann_lsh")
    query_block_keys = er.block_keys(name_key, address_key)
    block_limit = er.adaptive_limit(
        er.BLOCK_KEY_LIMIT, len(query_block_keys), er.BLOCK_KEY_MIN_LIMIT
    )
    for key in query_block_keys:
        for (eid,) in connection.execute(
            "SELECT r.entity_id FROM block_keys b JOIN records r "
            "ON r.entity_id = b.entity_id WHERE b.block_key = ? "
            "AND b.source IN ('source2', 'source3') AND r.country = "
            "? ORDER BY b.entity_id LIMIT ?",
            (key, country, block_limit),
        ):
            mechanisms[eid].add("selective_block_key")
    return mechanisms


MECHANISM_GROUPS = {
    "exact_any": {"exact_name", "exact_name_compact", "exact_address"},
    "ann_lsh": {"ann_lsh"},
    "selective_block_key": {"selective_block_key"},
}


# --------------------------------------------------------------------------
# Main diagnostic run
# --------------------------------------------------------------------------

def run_index_variant(
    er,
    variant_name: str,
    source1_rows: List[dict],
    source1_ids: List[str],
    labels: Dict[str, Set[str]],
    source2_rows: List[dict],
    source3_rows: List[dict],
    work_dir: Path,
    progress_every: int,
    workers: int = 1,
    skip_miss_diagnosis: bool = False,
) -> dict:
    """Build one production index and measure candidate
    recall + mechanism attribution against it."""
    variant_dir = work_dir / variant_name
    s2_path = variant_dir / "train_source2.tsv"
    s3_path = variant_dir / "train_source3.tsv"

    write_tsv(s2_path, SOURCE_HEADER, source2_rows)
    write_tsv(s3_path, SOURCE_HEADER, source3_rows)
    db_path = variant_dir / "index.sqlite"
    if db_path.exists():
        db_path.unlink()

    print(f"[{variant_name}] building index ({len(source2_rows):,} + {len(source3_rows):,} rows)...")
    t0 = time.time()
    er.build_index([s2_path, s3_path], db_path)
    print(f"[{variant_name}] index built in {time.time() - t0:.1f}s")

    connection = sqlite3.connect(db_path)
    stored_sources = sorted(
        r[0] for r in connection.execute("SELECT DISTINCT source FROM records")
    )
    connection.close()
    print(f"[{variant_name}] distinct `source` values actually stored: {stored_sources}")

    t0 = time.time()
    row_results = run_recall_pass(er, db_path, source1_rows, source1_ids, labels, workers)
    print(f"[{variant_name}] recall pass over {len(source1_rows):,} rows done in "
          f"{time.time() - t0:.1f}s ({workers} worker(s))")

    per_entity_recall: List[float] = []
    micro_true_positive = 0
    micro_true_total = 0
    candidate_sizes_matched: List[int] = []
    candidate_sizes_singleton: List[int] = []
    group_recall_hits = {group: 0 for group in MECHANISM_GROUPS}
    group_recall_hits["combined"] = 0
    n_zero_hit = n_partial_hit = n_full_hit = 0
    per_entity_rows_for_csv = []
    # (source1_row, source1_id, missed_target_ids) for rows with >=1 miss --
    # the input to the bounded miss-diagnosis pass below.
    miss_inputs: List[Tuple[dict, str, List[str]]] = []
    row_by_id = dict(zip(source1_ids, source1_rows))

    for result in row_results:
        sid = result["sid"]
        true_count = result["true_count"]
        candidate_count = result["candidate_count"]
        hit_count = result["hit_count"]
        missed_ids = result["missed_ids"]

        if true_count:
            candidate_sizes_matched.append(candidate_count)
        else:
            candidate_sizes_singleton.append(candidate_count)

        if true_count:
            recall = hit_count / true_count
            per_entity_recall.append(recall)
            micro_true_positive += hit_count
            micro_true_total += true_count
            if result["group_hit_combined"]:
                group_recall_hits["combined"] += 1
            if result["group_hit_exact_any"]:
                group_recall_hits["exact_any"] += 1
            if result["group_hit_ann_lsh"]:
                group_recall_hits["ann_lsh"] += 1
            if result["group_hit_selective_block_key"]:
                group_recall_hits["selective_block_key"] += 1

            if hit_count == 0:
                n_zero_hit += 1
            elif hit_count < true_count:
                n_partial_hit += 1
            else:
                n_full_hit += 1

        if missed_ids:
            miss_inputs.append((row_by_id[sid], sid, missed_ids))

        per_entity_rows_for_csv.append(
            (sid, true_count, candidate_count, hit_count, ",".join(missed_ids))
        )

    matched_entity_count = len(per_entity_recall)
    macro_recall = statistics.mean(per_entity_recall) if per_entity_recall else float("nan")
    micro_recall = (
        micro_true_positive / micro_true_total if micro_true_total else float("nan")
    )

    csv_path = variant_dir / "per_entity_candidate_recall.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source1_entity_id", "true_match_count", "candidate_count",
                          "hits", "missed_target_ids"])
        writer.writerows(per_entity_rows_for_csv)

    miss_reason_counts = {}
    if not skip_miss_diagnosis and miss_inputs:
        t0 = time.time()
        diagnoses = diagnose_misses(er, db_path, miss_inputs)
        print(f"[{variant_name}] diagnosed {len(diagnoses):,} individual misses "
              f"across {len(miss_inputs):,} entities in {time.time() - t0:.1f}s")
        diag_csv_path = variant_dir / "missed_target_diagnostics.csv"
        with diag_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "source1_entity_id", "missed_target_id", "reason",
                "country_mismatch", "ann_bucket_truncated", "block_key_truncated",
            ])
            writer.writeheader()
            writer.writerows(diagnoses)
        for d in diagnoses:
            miss_reason_counts[d["reason"]] = miss_reason_counts.get(d["reason"], 0) + 1

    return {
        "variant": variant_name,
        "stored_sources": stored_sources,
        "matched_entity_count": matched_entity_count,
        "macro_candidate_recall": macro_recall,
        "micro_candidate_recall": micro_recall,
        "entities_with_any_hit_combined": group_recall_hits["combined"],
        "entities_with_any_hit_exact_any": group_recall_hits["exact_any"],
        "entities_with_any_hit_ann_lsh": group_recall_hits["ann_lsh"],
        "entities_with_any_hit_selective_block_key": group_recall_hits["selective_block_key"],
        "candidate_size_matched": summarize_sizes(candidate_sizes_matched),
        "candidate_size_singleton": summarize_sizes(candidate_sizes_singleton),
        "n_zero_hit": n_zero_hit,
        "n_partial_hit": n_partial_hit,
        "n_full_hit": n_full_hit,
        "miss_reason_counts": miss_reason_counts,
        "total_missed_links": sum(len(m[2]) for m in miss_inputs),
        "s2_path": s2_path,
        "s3_path": s3_path,
        "db_path": db_path,
    }


def run_threshold_sweep(
    er,
    ev,
    variant_result: dict,
    source1_rows: List[dict],
    source1_ids: List[str],
    labels: Dict[str, Set[str]],
    thresholds: List[float],
    work_dir: Path,
    variant_name: str,
    workers: int = 1,
) -> List[Tuple[float, float]]:
    """Run the CURRENT conservative scorer at each threshold and compute
    macro F0.5 using the project's own evaluate.py (imported, not
    reimplemented)."""
    connection = sqlite3.connect(variant_result["db_path"])
    # Candidate retrieval is independent of the threshold. Cache it once so
    # threshold sweeps only repeat the cheap scorer.
    candidate_cache = [er.candidate_rows(connection, row) for row in source1_rows]
    connection.close()
    context = (er, ev, source1_rows, source1_ids, labels, candidate_cache)
    if workers > 1 and len(thresholds) > 1:
        try:
            context_manager = mp.get_context("fork")
            with context_manager.Pool(
                processes=workers,
                initializer=_init_sweep_worker,
                initargs=(context,),
            ) as pool:
                results = pool.map(_score_threshold, thresholds)
        except ValueError:
            # Platforms without fork support still get the cache benefit.
            results = []
            _init_sweep_worker(context)
            for threshold in thresholds:
                results.append(_score_threshold(threshold))
    else:
        results = []
        _init_sweep_worker(context)
        for threshold in thresholds:
            results.append(_score_threshold(threshold))
    for threshold, macro_f05 in results:
        print(f"[{variant_name}] threshold={threshold:.2f}  macro_f05={macro_f05:.6f}")
    return sorted(results)


def _run_diagnostic() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--student-resource-dir", type=Path, required=True,
                         help="Root containing dataset/train/*.tsv")
    parser.add_argument("--src-dir", type=Path, default=None,
                         help="Directory with entity_resolution.py / evaluate.py "
                              "(default: <student-resource-dir>/code/business_entity_resolution/src)")
    parser.add_argument("--work-dir", type=Path, default=Path("work/diagnostic_100k"))
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Markdown file for diagnostic output (default: <work-dir>/blocking_recall_diagnostic.md).",
    )
    parser.add_argument("--source1-sample-size", type=int, default=100_000)
    parser.add_argument("--target-fill-size", type=int, default=100_000,
                         help="Total random distractor budget across Source 2 + "
                              "Source 3, split proportionally to their file sizes. "
                              "True positive targets are always included on top "
                              "of this budget, never counted against it.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thresholds", type=float, nargs="+",
                         default=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.95])
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=1,
                         help="Processes used for BOTH the recall pass "
                              "(attributed_candidates per row) and the "
                              "threshold-sweep scoring after candidate "
                              "retrieval is cached (Linux supports fork).")
    parser.add_argument("--skip-miss-diagnosis", action="store_true",
                         help="Skip the per-miss country-mismatch / ANN-truncation "
                              "/ block-key-truncation breakdown (recall numbers are "
                              "unaffected either way -- this only skips the extra "
                              "root-cause queries for the misses).")
    parser.add_argument("--skip-threshold-sweep", action="store_true",
                         help="Only measure candidate recall; skip the "
                              "precision/recall/F0.5 threshold sweep (much faster).")
    parser.add_argument("--quick", action="store_true",
                         help="Shortcut for a fast smoke test: forces sample "
                              "sizes down to 2,000/2,000 and a single threshold, "
                              "unless explicitly overridden.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    if args.quick:
        if args.source1_sample_size == 100_000:
            args.source1_sample_size = 2_000
        if args.target_fill_size == 100_000:
            args.target_fill_size = 2_000
        if args.thresholds == [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.95]:
            args.thresholds = [0.88]

    src_dir = args.src_dir or (
        args.student_resource_dir / "code" / "business_entity_resolution" / "src"
    )
    sys.path.insert(0, str(src_dir))
    try:
        import entity_resolution as er  # type: ignore
        import evaluate as ev  # type: ignore
    except ImportError as exc:
        print(f"Could not import entity_resolution/evaluate from {src_dir}: {exc}", file=sys.stderr)
        print("Pass --src-dir explicitly if your source layout differs.", file=sys.stderr)
        sys.exit(1)

    train_dir = args.student_resource_dir / "dataset" / "train"
    source1_path = train_dir / "train_source1.tsv"
    source2_path = train_dir / "train_source2.tsv"
    source3_path = train_dir / "train_source3.tsv"
    gt_path = train_dir / "train_ground_truth.tsv"
    for p in (source1_path, source2_path, source3_path, gt_path):
        if not p.exists():
            print(f"Expected input file not found: {p}", file=sys.stderr)
            sys.exit(1)

    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sampling {args.source1_sample_size:,} Source-1 rows (seed={args.seed})...")
    _, source1_rows = reservoir_sample_tsv(source1_path, args.source1_sample_size, args.seed)
    source1_ids = [row["entity_id"] for row in source1_rows]
    source1_id_set = set(source1_ids)
    write_tsv(work_dir / "sampled_source1" / "train_source1.tsv", SOURCE_HEADER, source1_rows)

    print("Streaming full ground truth to collect true matches for the sampled ids...")
    labels = collect_ground_truth(gt_path, source1_id_set)
    write_ground_truth(work_dir / "sampled_source1" / "train_ground_truth.filtered.tsv", labels)

    all_true_ids = set()
    for matches in labels.values():
        all_true_ids |= matches
    forced_s2 = {eid for eid in all_true_ids if eid.startswith("S2-")}
    forced_s3 = {eid for eid in all_true_ids if eid.startswith("S3-")}
    other_prefix = all_true_ids - forced_s2 - forced_s3
    if other_prefix:
        print(f"  WARNING: {len(other_prefix)} true ids have neither S2- nor S3- "
              f"prefix, e.g. {sorted(other_prefix)[:3]} -- check id conventions.",
              file=sys.stderr)

    n_singletons = sum(1 for m in labels.values() if not m)
    n_with_matches = len(labels) - n_singletons
    print(f"Sampled {len(labels):,} Source-1 ids: {n_with_matches:,} have >=1 true "
          f"match ({len(all_true_ids):,} true target links total, "
          f"{len(forced_s2):,} in Source 2 / {len(forced_s3):,} in Source 3), "
          f"{n_singletons:,} are true singletons.")

    fill_s2 = max(0, args.target_fill_size // 2)
    fill_s3 = max(0, args.target_fill_size - fill_s2)
    print(f"Building target sample: Source 2 = {len(forced_s2):,} forced + up to "
          f"{fill_s2:,} random fill; Source 3 = {len(forced_s3):,} forced + up to "
          f"{fill_s3:,} random fill.")
    source2_rows = extract_and_fill(source2_path, forced_s2, fill_s2, args.seed + 1)
    source3_rows = extract_and_fill(source3_path, forced_s3, fill_s3, args.seed + 2)
    print(f"Final target sample sizes: Source 2 = {len(source2_rows):,}, "
          f"Source 3 = {len(source3_rows):,}.")

    variants_to_run = ["production"]
    variant_results = []
    for variant_name in variants_to_run:
        result = run_index_variant(
            er, variant_name, source1_rows, source1_ids, labels,
            source2_rows, source3_rows, work_dir, args.progress_every,
            workers=args.workers, skip_miss_diagnosis=args.skip_miss_diagnosis,
        )
        variant_results.append(result)

    print("\n" + "=" * 78)
    print("CANDIDATE RECALL SUMMARY (entities with >=1 true match only)")
    print("=" * 78)
    for result in variant_results:
        print(f"\n--- {result['variant']} (stored source tags: {result['stored_sources']}) ---")
        print(f"  macro candidate recall : {result['macro_candidate_recall']:.4f}")
        print(f"  micro candidate recall : {result['micro_candidate_recall']:.4f}")
        print(f"  entities w/ >=1 hit, combined            : "
              f"{result['entities_with_any_hit_combined']:,} / {result['matched_entity_count']:,}")
        print(f"  entities w/ >=1 hit, exact matching only  : "
              f"{result['entities_with_any_hit_exact_any']:,} / {result['matched_entity_count']:,}")
        print(f"  entities w/ >=1 hit, ANN/LSH only         : "
              f"{result['entities_with_any_hit_ann_lsh']:,} / {result['matched_entity_count']:,}")
        print(f"  entities w/ >=1 hit, selective block key  : "
              f"{result['entities_with_any_hit_selective_block_key']:,} / {result['matched_entity_count']:,}")
        print(f"  candidate-set size (true-match entities)  : {result['candidate_size_matched']}")
        print(f"  candidate-set size (true singletons)      : {result['candidate_size_singleton']}")
        print(f"  entities with FULL hit (all true matches found)    : {result['n_full_hit']:,}")
        print(f"  entities with PARTIAL hit (some but not all found) : {result['n_partial_hit']:,}")
        print(f"  entities with ZERO hit (nothing found)              : {result['n_zero_hit']:,}")
        if result["miss_reason_counts"]:
            print(f"  missed target links: {result['total_missed_links']:,}, by likely cause:")
            for reason, count in sorted(result["miss_reason_counts"].items(),
                                         key=lambda kv: -kv[1]):
                print(f"    {reason:<28s}: {count:,}")
            print(f"  -> see {result['variant']}/missed_target_diagnostics.csv for the "
                  f"full per-miss breakdown, and {result['variant']}/"
                  f"per_entity_candidate_recall.csv (missed_target_ids column) "
                  f"for which specific ids were missed per entity.")

    if not args.skip_threshold_sweep:
        print("\n" + "=" * 78)
        print("PRECISION / RECALL / MACRO-F0.5 THRESHOLD SWEEP (current scorer)")
        print("=" * 78)
        for result in variant_results:
            print(f"\n--- {result['variant']} ---")
            sweep = run_threshold_sweep(
                er, ev, result, source1_rows, source1_ids, labels,
                args.thresholds, work_dir, result["variant"],
                workers=args.workers,
            )
            best_threshold, best_f05 = max(sweep, key=lambda pair: pair[1])
            print(f"  best on this sample: threshold={best_threshold:.2f} "
                  f"macro_f05={best_f05:.6f}")

    print(f"\nPer-entity CSVs and sampled tsvs are under: {work_dir.resolve()}")


def main() -> None:
    work_dir = Path("work/diagnostic_100k")
    output_md = None
    arguments = sys.argv[1:]
    for index, argument in enumerate(arguments):
        if argument == "--work-dir" and index + 1 < len(arguments):
            work_dir = Path(arguments[index + 1])
        elif argument == "--output-md" and index + 1 < len(arguments):
            output_md = Path(arguments[index + 1])

    output_md = output_md or work_dir / "blocking_recall_diagnostic.md"
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with output_md.open("w", encoding="utf-8") as report:
        report.write("# Blocking Recall Diagnostic\n\n")
        report.write("```text\n")
        with contextlib.redirect_stdout(report):
            _run_diagnostic()
        report.write("```\n")
    print(f"Diagnostic report written to: {output_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
