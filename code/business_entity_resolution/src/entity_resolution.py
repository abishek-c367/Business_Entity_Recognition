from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import math
import re
import sqlite3
import time
import unicodedata
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator


SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
LEGAL_SUFFIXES = {
    "co",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "limited",
    "llc",
    "ltd",
    "pvt",
    "private",
}
ABBREVIATIONS = {
    "avenue": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "highway": "hwy",
    "road": "rd",
    "street": "st",
    "suite": "ste",
}
CHAR_NGRAM_MIN = 3
CHAR_NGRAM_MAX = 5
MINHASH_SIZE = 16
LSH_BANDS = 4
ANN_BUCKET_LIMIT = 500
BLOCK_KEY_LIMIT = 250
ANN_BUCKET_MIN_LIMIT = 100
BLOCK_KEY_MIN_LIMIT = 25


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold()
    value = "".join(
        character for character in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(character)
    )
    value = value.replace("&", " and ")
    value = re.sub(r"[^0-9a-z]+", " ", value)
    return " ".join(value.split())


def normalize_name(value: str) -> str:
    tokens = normalize_text(value).split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(sorted(tokens))


def normalize_address(value: str) -> str:
    tokens = [
        ABBREVIATIONS.get(token, token)
        for token in normalize_text(value).split()
    ]
    return " ".join(tokens)


def compact_key(value: str) -> str:
    return value.replace(" ", "")


def search_text(name_key: str, address_key: str) -> str:
    return f"{name_key} {address_key}".strip()


def character_ngrams(value: str) -> set[str]:
    """Return boundary-padded character n-grams for fuzzy retrieval."""
    value = f"  {value}  "
    return {
        value[index:index + size]
        for size in range(CHAR_NGRAM_MIN, CHAR_NGRAM_MAX + 1)
        for index in range(len(value) - size + 1)
    }


def _hash_ngram(ngram: str, seed: int) -> int:
    # A non-cryptographic hash is sufficient here: this value only needs to
    # spread n-grams roughly uniformly for bottom-k MinHash bucketing, not
    # resist deliberate collision attacks. zlib.crc32 is a C-implemented
    # stdlib function and is ~5x faster than blake2b in practice, which
    # matters a great deal here because this function is called once per
    # character n-gram per record (commonly 100+ times per record).
    return zlib.crc32(f"{seed}:{ngram}".encode("utf-8"))


def minhash_signature(ngrams: set[str]) -> tuple[int, ...]:
    if not ngrams:
        return (0,) * MINHASH_SIZE
    # Bottom-k MinHash uses one stable hash per n-gram and avoids a
    # permutations-by-document cost during the disk-backed index build.
    # heapq.nsmallest is O(n log k) instead of a full O(n log n) sort --
    # meaningful here since k=16 is typically far smaller than the number
    # of n-grams in a record (often 100+).
    values = heapq.nsmallest(MINHASH_SIZE, (_hash_ngram(ngram, 0) for ngram in ngrams))
    return tuple(values + [0] * (MINHASH_SIZE - len(values)))


def ann_bucket_keys(signature: tuple[int, ...]) -> tuple[str, ...]:
    band_size = MINHASH_SIZE // LSH_BANDS
    return tuple(
        f"{band}:{hashlib.blake2b(repr(signature[start:start + band_size]).encode(), digest_size=8).hexdigest()}"
        for band, start in enumerate(range(0, MINHASH_SIZE, band_size))
    )


def _tfidf_similarity(
    left: str, right: str, document_frequencies: dict[str, int], document_count: int
) -> float:
    left_ngrams = character_ngrams(left)
    right_ngrams = character_ngrams(right)
    if not left_ngrams or not right_ngrams or not document_count:
        return 0.0
    def vector(ngrams: set[str]) -> dict[str, float]:
        return {
            ngram: math.log((document_count + 1) / (document_frequencies.get(ngram, 0) + 1)) + 1.0
            for ngram in ngrams
        }
    left_vector, right_vector = vector(left_ngrams), vector(right_ngrams)
    dot = sum(left_vector[term] * right_vector.get(term, 0.0) for term in left_vector)
    left_norm = math.sqrt(sum(value * value for value in left_vector.values()))
    right_norm = math.sqrt(sum(value * value for value in right_vector.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def iter_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
            raise ValueError(f"{path} has unexpected columns: {reader.fieldnames}")
        yield from reader


def connect_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    # Default page cache is ~2MB, far smaller than the working set of these
    # tables once the index holds hundreds of thousands to millions of rows
    # (records + block_keys + ann_buckets + char_ngram_df all being written
    # interleaved in the same batches). A larger cache and MEMORY temp store
    # reduce disk I/O during the bulk load; mmap lets SQLite read pages
    # directly instead of copying through its own page cache.
    connection.execute("PRAGMA cache_size=-262144")  # ~256MB page cache
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA mmap_size=268435456")  # 256MB
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS records (
            entity_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            business_name TEXT NOT NULL,
            business_address TEXT NOT NULL,
            country TEXT NOT NULL,
            name_key TEXT NOT NULL,
            address_key TEXT NOT NULL,
            name_compact TEXT NOT NULL,
            address_compact TEXT NOT NULL
        )
        """
    )
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(records)")
    }
    if "search_key" not in columns:
        connection.execute("ALTER TABLE records ADD COLUMN search_key TEXT NOT NULL DEFAULT ''")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS char_ngram_df (
            ngram TEXT PRIMARY KEY,
            document_frequency INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS ann_buckets (
            bucket_key TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            source TEXT NOT NULL,
            PRIMARY KEY (bucket_key, entity_id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS block_keys (
            block_key TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            source TEXT NOT NULL,
            PRIMARY KEY (block_key, entity_id)
        )
        """
    )
    # NOTE: the secondary (non-PRIMARY-KEY) lookup indexes are deliberately
    # NOT created here -- see create_indexes() below and its call site in
    # build_index(). Creating them up front forces every single insert
    # during the bulk load to incrementally update every index's B-tree,
    # which gets more expensive as each index grows (this is the classic
    # SQLite "index-before-bulk-load" anti-pattern, and the ratio of rows
    # to slowdown observed on real data -- much worse than linear -- is
    # consistent with this being the dominant cost at real scale). The
    # PRIMARY KEY constraints on entity_id / (block_key, entity_id) /
    # (bucket_key, entity_id) still exist and are still enforced during the
    # load (INSERT OR IGNORE / OR REPLACE rely on them) -- only the *extra*
    # lookup indexes used by query-time candidate retrieval are deferred.
    return connection


def create_indexes(connection: sqlite3.Connection) -> None:
    """Build the secondary lookup indexes once, after all rows are loaded.
    A single bulk index build (SQLite sorts and builds the B-tree in one
    pass) is dramatically cheaper than maintaining the same index
    incrementally across hundreds of thousands of individual inserts."""
    connection.execute(
        "CREATE INDEX IF NOT EXISTS ann_buckets_lookup ON ann_buckets(bucket_key, source, entity_id)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS block_keys_lookup "
        "ON block_keys(block_key, source, entity_id)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS records_name ON records(country, name_key)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS records_address ON records(country, address_key)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS records_name_compact "
        "ON records(country, name_compact)"
    )


def build_index(source_paths: Iterable[Path], database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = connect_database(database_path)
    connection.execute("DELETE FROM records")
    connection.execute("DELETE FROM block_keys")
    connection.execute("DELETE FROM ann_buckets")
    connection.execute("DELETE FROM char_ngram_df")
    # Explicitly drop the secondary lookup indexes even if this database
    # file was already built by an older version of this script (or an
    # earlier, interrupted run) that created them up front. Without this,
    # reusing an existing database file would silently fall back to
    # incremental (slow) index maintenance during the bulk load below.
    for index_name in (
        "ann_buckets_lookup", "block_keys_lookup",
        "records_name", "records_address", "records_name_compact",
    ):
        connection.execute(f"DROP INDEX IF EXISTS {index_name}")
    connection.commit()
    insert_sql = """
        INSERT OR REPLACE INTO records (
            entity_id, source, business_name, business_address, country,
            name_key, address_key, name_compact, address_compact, search_key
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    block_sql = "INSERT OR IGNORE INTO block_keys VALUES (?, ?, ?)"
    ann_sql = "INSERT OR IGNORE INTO ann_buckets VALUES (?, ?, ?)"
    try:
        for source_path in source_paths:
            rows = []
            ann_rows_pending = []  # (entity_id, bucket_key, source) precomputed once per row
            ngram_counts: defaultdict[str, int] = defaultdict(int)
            source = source_path.stem.split("_", 1)[1]
            for row in iter_rows(source_path):
                name_key = normalize_name(row["business_name"])
                address_key = normalize_address(row["business_address"])
                search_key = search_text(name_key, address_key)
                ngrams = character_ngrams(search_key)
                for ngram in ngrams:
                    ngram_counts[ngram] += 1
                # Computed once here and reused below -- this used to be
                # recomputed from scratch a second time per row when the
                # ANN batch was flushed, doubling n-gram + minhash cost.
                entity_id = row["entity_id"]
                for bucket in ann_bucket_keys(minhash_signature(ngrams)):
                    ann_rows_pending.append((bucket, entity_id, source))
                rows.append(
                    (
                        entity_id,
                        source,
                        row["business_name"],
                        row["business_address"],
                        row["country"],
                        name_key,
                        address_key,
                        compact_key(name_key),
                        compact_key(address_key),
                        search_key,
                    )
                )
                if len(rows) >= 10_000:
                    connection.executemany(insert_sql, rows)
                    connection.executemany(
                        block_sql,
                        [
                            (key, entity_id, source)
                            for entity_id, _, _, _, _, name_key, address_key, _, _, _ in rows
                            for key in block_keys(name_key, address_key)
                        ],
                    )
                    connection.executemany(ann_sql, ann_rows_pending)
                    connection.executemany(
                        "INSERT INTO char_ngram_df VALUES (?, ?) "
                        "ON CONFLICT(ngram) DO UPDATE SET document_frequency = "
                        "document_frequency + excluded.document_frequency",
                        ngram_counts.items(),
                    )
                    connection.commit()
                    rows.clear()
                    ann_rows_pending.clear()
                    ngram_counts.clear()
            if rows:
                connection.executemany(insert_sql, rows)
                connection.executemany(
                    block_sql,
                    [
                        (key, entity_id, source)
                        for entity_id, _, _, _, _, name_key, address_key, _, _, _ in rows
                        for key in block_keys(name_key, address_key)
                    ],
                )
                connection.executemany(ann_sql, ann_rows_pending)
                connection.executemany(
                    "INSERT INTO char_ngram_df VALUES (?, ?) "
                    "ON CONFLICT(ngram) DO UPDATE SET document_frequency = "
                    "document_frequency + excluded.document_frequency",
                    ngram_counts.items(),
                )
                connection.commit()
                ann_rows_pending.clear()
        # All rows are loaded now -- build the secondary lookup indexes in
        # one bulk pass instead of incrementally during every insert above.
        t_index = time.time()
        create_indexes(connection)
        connection.commit()
        print(f"  (secondary indexes built in {time.time() - t_index:.1f}s)")
        connection.execute("ANALYZE")
        connection.commit()
    finally:
        connection.close()


def _tokens(value: str) -> set[str]:
    return {token for token in value.split() if len(token) > 1}


def block_keys(name_key: str, address_key: str) -> set[str]:
    """Return selective keys that preserve recall without indexing every token."""
    name_tokens = sorted(_tokens(name_key), key=lambda token: (-len(token), token))
    address_tokens = _tokens(address_key)
    keys = {f"n:{token}" for token in name_tokens[:2] if len(token) >= 4}
    numeric = {token for token in address_tokens if token.isdigit() and len(token) >= 2}
    keys.update(f"a:{token}" for token in numeric)
    address_words = sorted(
        (token for token in address_tokens if len(token) >= 5 and not token.isdigit()),
        key=lambda token: (-len(token), token),
    )
    keys.update(f"a:{token}" for token in address_words[:6])
    for index, left in enumerate(address_words[:6]):
        for right in address_words[index + 1:6]:
            keys.add(f"aa:{left}:{right}")
    if name_tokens and address_tokens:
        keys.add(f"na:{name_tokens[0]}:{min(address_tokens)}")
    return keys


def adaptive_limit(total_limit: int, key_count: int, minimum: int) -> int:
    """Distribute a retrieval budget while protecting sparse-key recall."""
    if key_count <= 0:
        return minimum
    return max(minimum, (total_limit + key_count - 1) // key_count)


def _similarity(left: str, right: str) -> float:
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def candidate_rows(
    connection: sqlite3.Connection, row: dict[str, str]
) -> list[tuple[str, str, str, str, str, float]]:
    name_key = normalize_name(row["business_name"])
    address_key = normalize_address(row["business_address"])
    country = row["country"]
    query_search_key = search_text(name_key, address_key)
    query_ngrams = character_ngrams(query_search_key)
    query_buckets = ann_bucket_keys(minhash_signature(query_ngrams))
    document_count = connection.execute(
        "SELECT COUNT(*) FROM records"
    ).fetchone()[0]
    frequencies = {
        ngram: frequency
        for ngram, frequency in connection.execute(
            "SELECT ngram, document_frequency FROM char_ngram_df "
            f"WHERE ngram IN ({','.join('?' for _ in query_ngrams)})",
            tuple(query_ngrams),
        )
    } if query_ngrams else {}
    queries = []
    if name_key:
        queries.extend(
            [
                (
                    "SELECT entity_id, business_name, business_address, country, source "
                    "FROM records WHERE country = ? AND name_key = ?",
                    (country, name_key),
                ),
                (
                    "SELECT entity_id, business_name, business_address, country, source "
                    "FROM records WHERE country = ? AND name_compact = ?",
                    (country, compact_key(name_key)),
                ),
            ]
        )
    if address_key:
        queries.append(
            (
                "SELECT entity_id, business_name, business_address, country, source "
                "FROM records WHERE country = ? AND address_key = ?",
                (country, address_key),
            )
        )
    candidates: dict[str, tuple[str, str, str, str, str]] = {}
    for query, parameters in queries:
        for candidate in connection.execute(query, parameters):
            candidates[candidate[0]] = candidate
    ann_limit = adaptive_limit(
        ANN_BUCKET_LIMIT, len(query_buckets), ANN_BUCKET_MIN_LIMIT
    )
    for bucket in query_buckets:
        for candidate in connection.execute(
            "SELECT r.entity_id, r.business_name, r.business_address, "
            "r.country, r.source FROM ann_buckets b JOIN records r "
            "ON r.entity_id = b.entity_id WHERE b.bucket_key = ? "
            "AND b.source IN ('source2', 'source3') AND r.country = ? "
            "ORDER BY b.entity_id LIMIT ?",
            (bucket, country, ann_limit),
        ):
            candidates[candidate[0]] = candidate
    query_block_keys = block_keys(name_key, address_key)
    block_limit = adaptive_limit(
        BLOCK_KEY_LIMIT, len(query_block_keys), BLOCK_KEY_MIN_LIMIT
    )
    for key in query_block_keys:
        for candidate in connection.execute(
            "SELECT r.entity_id, r.business_name, r.business_address, "
            "r.country, r.source FROM block_keys b JOIN records r "
            "ON r.entity_id = b.entity_id WHERE b.block_key = ? "
            "AND b.source IN ('source2', 'source3') AND r.country = "
            "? ORDER BY b.entity_id LIMIT ?",
            (key, country, block_limit),
        ):
            candidates[candidate[0]] = candidate
    return [
        candidate + (
            _tfidf_similarity(
                query_search_key,
                search_text(normalize_name(candidate[1]), normalize_address(candidate[2])),
                frequencies,
                document_count,
            ),
        )
        for candidate in candidates.values()
    ]


def select_matches(
    row: dict[str, str],
    candidates: Iterable[tuple[str, str, str, str, str]],
    threshold: float,
) -> tuple[list[str], list[str]]:
    name_key = normalize_name(row["business_name"])
    address_key = normalize_address(row["business_address"])
    selected: list[str] = []
    candidate_ids: list[str] = []
    for candidate in candidates:
        entity_id, name, address, country, _source = candidate[:5]
        tfidf_score = candidate[5] if len(candidate) > 5 else 0.0
        candidate_ids.append(entity_id)
        name_score = _similarity(name_key, normalize_name(name))
        address_score = _similarity(address_key, normalize_address(address))
        exact_name = name_key and name_key == normalize_name(name)
        exact_address = address_key and address_key == normalize_address(address)
        score = max(
            1.0 if exact_name else 0.0,
            1.0 if exact_address else 0.0,
            0.65 * name_score + 0.35 * address_score,
            tfidf_score,
        )
        if score >= threshold:
            selected.append(entity_id)
    return sorted(selected), sorted(set(candidate_ids))


def run_matching(source1_path: Path, database_path: Path, output_path: Path,
                 candidate_path: Path, threshold: float) -> None:
    connection = sqlite3.connect(database_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as match_handle, \
            candidate_path.open("w", encoding="utf-8", newline="") as candidate_handle:
        match_writer = csv.writer(match_handle, delimiter="\t", lineterminator="\n")
        candidate_writer = csv.writer(
            candidate_handle, delimiter="\t", lineterminator="\n"
        )
        match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for row in iter_rows(source1_path):
            candidates = candidate_rows(connection, row)
            matches, candidate_ids = select_matches(row, candidates, threshold)
            match_writer.writerow([row["entity_id"], ",".join(matches)])
            candidate_writer.writerow([row["entity_id"], ",".join(candidate_ids)])
    connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-index")
    build.add_argument("--source", action="append", type=Path, required=True)
    build.add_argument("--database", type=Path, required=True)

    match = subparsers.add_parser("match")
    match.add_argument("--source1", type=Path, required=True)
    match.add_argument("--database", type=Path, required=True)
    match.add_argument("--output", type=Path, required=True)
    match.add_argument("--candidate", type=Path, required=True)
    match.add_argument("--threshold", type=float, default=0.88)

    args = parser.parse_args()
    if args.command == "build-index":
        build_index(args.source, args.database)
    else:
        run_matching(
            args.source1, args.database, args.output, args.candidate, args.threshold
        )


if __name__ == "__main__":
    main()
