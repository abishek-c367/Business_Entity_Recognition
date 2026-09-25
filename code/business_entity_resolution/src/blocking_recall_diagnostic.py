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
    for bucket in query_buckets:
        for (eid,) in connection.execute(
            "SELECT r.entity_id FROM ann_buckets b JOIN records r "
            "ON r.entity_id = b.entity_id WHERE b.bucket_key = ? "
            "AND b.source IN ('source2', 'source3') AND r.country = ? "
            "ORDER BY b.entity_id LIMIT ?",
            (bucket, country, er.ANN_BUCKET_LIMIT),
        ):
            mechanisms[eid].add("ann_lsh")
    for key in er.block_keys(name_key, address_key):
        for (eid,) in connection.execute(
            "SELECT r.entity_id FROM block_keys b JOIN records r "
            "ON r.entity_id = b.entity_id WHERE b.block_key = ? "
            "AND b.source IN ('source2', 'source3') AND r.country = ?",
            (key, country),
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
    print(f"[{variant_name}] distinct `source` values actually stored: {stored_sources}")

    per_entity_recall: List[float] = []
    micro_true_positive = 0
    micro_true_total = 0
    candidate_sizes_matched: List[int] = []
    candidate_sizes_singleton: List[int] = []
    group_recall_hits = {group: 0 for group in MECHANISM_GROUPS}
    group_recall_hits["combined"] = 0

    per_entity_rows_for_csv = []

    t0 = time.time()
    for i, (row, sid) in enumerate(zip(source1_rows, source1_ids)):
        true_set = labels.get(sid, set())
        mechanisms = attributed_candidates(er, connection, row)
        combined_candidates = set(mechanisms)

        if true_set:
            candidate_sizes_matched.append(len(combined_candidates))
        else:
            candidate_sizes_singleton.append(len(combined_candidates))

        if true_set:
            hit = combined_candidates & true_set
            recall = len(hit) / len(true_set)
            per_entity_recall.append(recall)
            micro_true_positive += len(hit)
            micro_true_total += len(true_set)
            if hit:
                group_recall_hits["combined"] += 1
            for group, tags in MECHANISM_GROUPS.items():
                group_candidates = {eid for eid, m in mechanisms.items() if m & tags}
                if group_candidates & true_set:
                    group_recall_hits[group] += 1

        per_entity_rows_for_csv.append(
            (sid, len(true_set), len(combined_candidates), len(true_set & combined_candidates))
        )

        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed else 0
            print(f"[{variant_name}] {i + 1:,}/{len(source1_rows):,} rows "
                  f"({rate:.0f} rows/s)")

    connection.close()

    matched_entity_count = len(per_entity_recall)
    macro_recall = statistics.mean(per_entity_recall) if per_entity_recall else float("nan")
    micro_recall = (
        micro_true_positive / micro_true_total if micro_true_total else float("nan")
    )

    csv_path = variant_dir / "per_entity_candidate_recall.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source1_entity_id", "true_match_count", "candidate_count", "hits"])
        writer.writerows(per_entity_rows_for_csv)

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--student-resource-dir", type=Path, required=True,
                         help="Root containing dataset/train/*.tsv")
    parser.add_argument("--src-dir", type=Path, default=None,
                         help="Directory with entity_resolution.py / evaluate.py "
                              "(default: <student-resource-dir>/code/business_entity_resolution/src)")
    parser.add_argument("--work-dir", type=Path, default=Path("work/diagnostic_100k"))
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
                         help="Processes for threshold scoring after candidate "
                              "retrieval is cached (Linux supports fork).")
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


if __name__ == "__main__":
    main()
