from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import sqlite3
import unicodedata
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
ANN_BUCKET_LIMIT = 2_000


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
    digest = hashlib.blake2b(
        f"{seed}:{ngram}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def minhash_signature(ngrams: set[str]) -> tuple[int, ...]:
    if not ngrams:
        return (0,) * MINHASH_SIZE
    # Bottom-k MinHash uses one stable hash per n-gram and avoids a
    # permutations-by-document cost during the disk-backed index build.
    values = sorted(_hash_ngram(ngram, 0) for ngram in ngrams)
    return tuple(values[:MINHASH_SIZE] + [0] * (MINHASH_SIZE - len(values)))


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
        "CREATE INDEX IF NOT EXISTS ann_buckets_lookup ON ann_buckets(bucket_key, source, entity_id)"
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
    connection.execute(
        "CREATE INDEX IF NOT EXISTS block_keys_lookup "
        "ON block_keys(block_key, source, entity_id)"
    )
    return connection


def build_index(source_paths: Iterable[Path], database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = connect_database(database_path)
    connection.execute("DELETE FROM records")
    connection.execute("DELETE FROM block_keys")
    connection.execute("DELETE FROM ann_buckets")
    connection.execute("DELETE FROM char_ngram_df")
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
            ngram_counts: defaultdict[str, int] = defaultdict(int)
            source = source_path.name.split("_")[1]
            for row in iter_rows(source_path):
                name_key = normalize_name(row["business_name"])
                address_key = normalize_address(row["business_address"])
                search_key = search_text(name_key, address_key)
                ngrams = character_ngrams(search_key)
                for ngram in ngrams:
                    ngram_counts[ngram] += 1
                rows.append(
                    (
                        row["entity_id"],
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
                    connection.executemany(
                        ann_sql,
                        [
                            (key, entity_id, source)
                            for entity_id, _, _, _, _, name_key, address_key, _, _, _ in rows
                            for key in ann_bucket_keys(minhash_signature(
                                character_ngrams(search_text(name_key, address_key))
                            ))
                        ],
                    )
                    connection.executemany(
                        "INSERT INTO char_ngram_df VALUES (?, ?) "
                        "ON CONFLICT(ngram) DO UPDATE SET document_frequency = "
                        "document_frequency + excluded.document_frequency",
                        ngram_counts.items(),
                    )
                    connection.commit()
                    rows.clear()
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
                connection.executemany(
                    ann_sql,
                    [
                        (key, entity_id, source)
                        for entity_id, _, _, _, _, name_key, address_key, _, _, _ in rows
                        for key in ann_bucket_keys(minhash_signature(
                            character_ngrams(search_text(name_key, address_key))
                        ))
                    ],
                )
                connection.executemany(
                    "INSERT INTO char_ngram_df VALUES (?, ?) "
                    "ON CONFLICT(ngram) DO UPDATE SET document_frequency = "
                    "document_frequency + excluded.document_frequency",
                    ngram_counts.items(),
                )
                connection.commit()
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
    if name_tokens and address_tokens:
        keys.add(f"na:{name_tokens[0]}:{min(address_tokens)}")
    return keys


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
    for bucket in query_buckets:
        for candidate in connection.execute(
            "SELECT r.entity_id, r.business_name, r.business_address, "
            "r.country, r.source FROM ann_buckets b JOIN records r "
            "ON r.entity_id = b.entity_id WHERE b.bucket_key = ? "
            "AND b.source IN ('source2', 'source3') AND r.country = ? "
            "ORDER BY b.entity_id LIMIT ?",
            (bucket, country, ANN_BUCKET_LIMIT),
        ):
            candidates[candidate[0]] = candidate
    for key in block_keys(name_key, address_key):
        for candidate in connection.execute(
            "SELECT r.entity_id, r.business_name, r.business_address, "
            "r.country, r.source FROM block_keys b JOIN records r "
            "ON r.entity_id = b.entity_id WHERE b.block_key = ? "
            "AND b.source IN ('source2', 'source3') AND r.country = ?",
            (key, country),
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
