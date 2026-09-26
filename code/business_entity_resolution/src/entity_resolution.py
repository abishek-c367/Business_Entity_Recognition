from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
import re
import sqlite3
import time
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
# --- candidate retrieval -----------------------------------------------------
# A single IDF-weighted ranked retrieval replaces the previous ~25 separately
# capped per-key queries (two name lookups, one address lookup, 4 MinHash
# buckets, and ~17 blocking keys, each independently truncated). That design
# had three problems: every true link sharing only a dropped key was lost
# outright (799 links had no surviving shared key at all), each cap truncated
# independently so the caps did not bound the total, and the truncation fell on
# the *rarest* key -- which is the most informative one -- because the keys
# were ordered by entity_id rather than by selectivity.
#
# Cost model, which is the part that matters for scale: a query uses its rarest
# terms first, up to a fixed postings budget, so per-query work is bounded by a
# constant rather than growing with the corpus. Measured on the 445k diagnostic
# pool, indexing every token of length >= 2 (rather than truncating to the
# top-4 name / top-5 address tokens by length) raises structural reachability
# from 0.9961 to 0.99991 -- 30 unreachable links out of 345,707.
MIN_TERM_LENGTH = 2
CANDIDATE_LIMIT = 300
POSTINGS_BUDGET = 20_000
# Terms more common than this fraction of the corpus are stopwords. The cap is
# a *fraction* so it stays meaningful as the corpus grows. It is deliberately
# generous (0.5%): the failure mode this targets is a shared address locality
# word ("nagar", "delhi", "mumbai"), and a tight cap would drop exactly the
# evidence that the hard cases depend on.
DF_FRACTION = 0.005
DF_FLOOR = 64
# Exact address matches (P=0.910) are seeded with this much evidence so that
# the ranked cut can never discard one, however common its tokens are.
EXACT_ADDRESS_SEED = 1_000.0

# --- match scoring -----------------------------------------------------------
# These constants are not guesses. They were derived from the precision of each
# evidence type measured over 1,425,391 (query, candidate) pairs drawn from an
# index whose answer density (5.6%) and target:source ratio (5.95) match the
# real corpus -- 2,793 of those pairs were true links (0.20% base precision).
#
#   exact name, document frequency = 1     n=   892   P=0.888
#   exact name, document frequency 2-5     n= 1,226   P=0.276
#   exact name, document frequency 6-50    n= 4,619   P=0.037
#   exact name, document frequency 51+     n= 2,556   P=0.010
#   exact address                          n=   387   P=0.910
#   exact name AND exact address           n=   121   P=1.000
#   fuzzy: name_sim>=0.5 AND addr_sim>=0.4 n=   689   P=0.853
#
# Those precisions are real, but building a score out of them does not follow.
# What the challenge actually scores is MACRO F0.5 -- the mean of per-entity
# F0.5 over Source-1 entities -- and on this candidate pool macro F0.5 is
# decided almost entirely by precision at k=1, because the pool is ~99.6%
# noise: with ~1.14 true links among ~297 candidates, predicting all 300 scores
# 0.0052 while the oracle's single best pick scores 0.9506. Recall is nearly
# free (candidate recall is 0.9743); ranking is the binding constraint.
#
# Two measured consequences shape the code below:
#
#  1. A plain equal-weight blend of the two token-Jaccard similarities, scored
#     at threshold 0.70, beats every rarity- and exact-match-weighted variant
#     tested. Discounting exact names by document frequency scored 0.8326
#     against the blend's 0.8434, and adding a 0.95 exact-address bonus moved
#     it by +0.0005 -- noise. The reason is that the exact-match features are
#     already saturated on the pairs the ranker has to choose between; what
#     separates the top few candidates is graded token overlap, and the address
#     field, being more distinctive, deserves the same weight as the name
#     rather than the smaller one the old 0.65/0.35 split gave it.
#  2. entity_f05 scores an empty prediction against non-empty gold as 0.0 --
#     identical to a single wrong prediction. 94.4% of real Source-1 rows have
#     at least one match, so for those rows emitting the best candidate weakly
#     dominates emitting nothing. A pure threshold rule is therefore
#     structurally wrong: it buys precision the metric does not reward at the
#     cost of entities scored a flat zero.
DEFAULT_THRESHOLD = 0.70
NAME_WEIGHT = 0.50
ADDRESS_WEIGHT = 0.50
# Below this best-candidate score, treat the row as having no counterpart and
# predict nothing. 5.58% of real Source-1 rows are genuinely matchless and
# entity_f05 gives them 1.0 only if the prediction is *also* empty, so the
# floor trades matched-row coverage against that 5.58%. Swept against the real
# corpus mix (94.42% matched) the curve is: 0.00 -> 0.8012, 0.20 -> 0.8026,
# 0.30 -> 0.8147, 0.40 -> 0.7998, 0.50 -> 0.7602, 0.70 -> 0.5398. Flat enough
# between 0.20 and 0.40 that the exact value is not load-bearing.
MIN_EVIDENCE = 0.30


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold()
    value = "".join(
        character for character in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(character)
    )
    value = transliterate(value)
    value = value.replace("&", " and ")
    value = re.sub(r"[^0-9a-z]+", " ", value)
    return " ".join(value.split())


# --- Indic transliteration ---------------------------------------------------
# The line above this block drops everything outside [0-9a-z], which turned the
# 69,390 records in the realistic test pool whose business_name is written in a
# Brahmic script into the empty string -- no terms, no possible match, and the
# largest single block of unreachable links. Indic blocks follow the ISCII-91
# layout, so one shared offset->Latin table covers Devanagari, Bengali,
# Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada and Malayalam (base + 0x15
# is always KA).
#
# The scheme is phonetic-approximate on purpose: the strings being matched are
# English company names spelled in Indic script ("बेस्ट" = "best"), so what
# matters is that the Latin output shares character n-grams with the English
# original, not scholarly accuracy. Handles schwa deletion (a consonant carries
# an inherent 'a' unless a matra, virama, or word end follows).
_INDIC_BASES = (0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00)
_INDIC_CONSONANTS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch",
    0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th",
    0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d",
    0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b",
    0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l",
    0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s",
    0x39: "h",
}
_INDIC_VOWELS = {
    0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu",
    0x0B: "ri", 0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai",
    0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
}
_INDIC_MATRAS = {
    0x3E: "aa", 0x3F: "i", 0x40: "ii", 0x41: "u", 0x42: "uu", 0x43: "ri",
    0x44: "ri", 0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o",
    0x4A: "o", 0x4B: "o", 0x4C: "au",
}
_INDIC_NUKTA = {0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r",
                0x5D: "rh", 0x5E: "f", 0x5F: "y"}
_INDIC_VIRAMA = 0x4D
_VOWEL_RUN = re.compile(r"([aeiou])\1+")


def _indic_base(codepoint: int) -> int | None:
    for base in _INDIC_BASES:
        if base <= codepoint < base + 0x80:
            return base
    return None


def _indic_lookup(codepoint: int) -> tuple[str, str] | None:
    """Classify one Indic codepoint as ('c'onsonant|'m'atra|'v'owel|'x'virama
    |'s'ign|'d'igit, latin)."""
    base = _indic_base(codepoint)
    if base is None:
        return None
    offset = codepoint - base
    if offset in _INDIC_CONSONANTS:
        return ("c", _INDIC_CONSONANTS[offset])
    if offset in _INDIC_VOWELS:
        return ("v", _INDIC_VOWELS[offset])
    if offset in _INDIC_MATRAS:
        return ("m", _INDIC_MATRAS[offset])
    if offset == _INDIC_VIRAMA:
        return ("x", "")
    if offset in _INDIC_NUKTA:
        return ("c", _INDIC_NUKTA[offset])
    if 0x01 <= offset <= 0x03:      # candrabindu / anusvara / visarga
        return ("s", "n")
    if offset == 0x4E:              # prishthamatra e
        return ("m", "e")
    if 0x66 <= offset <= 0x6F:      # digits
        return ("d", str(offset - 0x66))
    return None


def transliterate(value: str) -> str:
    """Romanize any Brahmic-script runs in `value`; other characters pass
    through untouched. Returns `value` unchanged when it is pure ASCII, which
    is the common case and keeps the fast path cheap."""
    if value.isascii():
        return value
    out: list[str] = []
    length = len(value)
    for index, character in enumerate(value):
        if _indic_base(ord(character)) is None:
            out.append(character)
            continue
        kind = _indic_lookup(ord(character))
        if kind is None:
            continue
        tag, latin = kind
        if tag == "c":
            following = ord(value[index + 1]) if index + 1 < length else None
            next_kind = _indic_lookup(following) if following is not None else None
            if next_kind and next_kind[0] in ("x", "m"):
                out.append(latin)          # virama or matra supplies the vowel
            elif next_kind and next_kind[0] == "c":
                out.append(latin)          # consonant follows -> schwa deleted
            elif following is None or not chr(following).isalnum():
                out.append(latin)          # word-final -> schwa deleted
            else:
                out.append(latin + "a")
        else:
            out.append(latin)
    # Long/short vowel distinctions are not recoverable from the Latin spelling
    # being matched against, and Indic loans of English words over-apply them
    # ("maarketing"), so collapse runs of the same vowel.
    return _VOWEL_RUN.sub(lambda match: match.group(0)[0], "".join(out))


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
    # (records + postings + term_stats all being written interleaved in the
    # same batches). A larger cache and MEMORY temp store reduce disk I/O
    # during the bulk load; mmap lets SQLite read pages directly instead of
    # copying through its own page cache.
    connection.execute("PRAGMA cache_size=-262144")  # ~256MB page cache
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA mmap_size=268435456")  # 256MB
    # Only the fields retrieval and scoring actually read are stored. The old
    # schema also carried name_compact / address_compact / search_key, which
    # existed solely to feed the ANN-LSH retriever and the compact-name lookup;
    # both are gone, so those columns and the records_name_compact index over
    # them were a pure build-time cost on every record.
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS records (
            entity_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            business_name TEXT NOT NULL,
            business_address TEXT NOT NULL,
            country TEXT NOT NULL,
            name_key TEXT NOT NULL,
            address_key TEXT NOT NULL
        )
        """
    )
    # CREATE TABLE IF NOT EXISTS is a no-op on an existing file, so a database
    # built by the previous schema would keep its extra NOT NULL columns and
    # reject the 7-column insert below. build_index empties this table anyway,
    # so dropping it is free and the CREATE above then restores the new shape.
    existing_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(records)")
    }
    if existing_columns - {
        "entity_id", "source", "business_name", "business_address",
        "country", "name_key", "address_key",
    }:
        connection.execute("DROP TABLE records")
        connection.execute(
            """
            CREATE TABLE records (
                entity_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                business_name TEXT NOT NULL,
                business_address TEXT NOT NULL,
                country TEXT NOT NULL,
                name_key TEXT NOT NULL,
                address_key TEXT NOT NULL
            )
            """
        )
    # The inverted index behind candidate retrieval: one row per (term, record).
    # WITHOUT ROWID because every access is by term prefix and the primary key
    # *is* the access path -- a rowid table would store this twice.
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS postings (
            term TEXT NOT NULL,
            country TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            PRIMARY KEY (term, country, entity_id)
        ) WITHOUT ROWID
        """
    )
    # One row per (term, country): how many records contain it, and the IDF
    # weight log(1 + N_country / df) that retrieval sums. Carrying `weight`
    # here means the ranking never recomputes a log per posting.
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS term_stats (
            term TEXT NOT NULL,
            country TEXT NOT NULL,
            document_frequency INTEGER NOT NULL,
            weight REAL NOT NULL,
            PRIMARY KEY (term, country)
        ) WITHOUT ROWID
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
        "CREATE INDEX IF NOT EXISTS records_name ON records(country, name_key)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS records_address ON records(country, address_key)"
    )


def index_terms(name_key: str, address_key: str) -> set[str]:
    """The terms a record is indexed under, and a query retrieves against.

    Deliberately *every* token of length >= 2, tagged by which field it came
    from. The previous design selected a few tokens per record by string
    length and capped each key's postings independently; both choices threw
    away the specific token a true pair shared, which is why 799 links had no
    surviving shared key at all and a further 10,210 were lost to truncation.
    Selectivity is handled at query time by IDF ranking instead, which is the
    right place for it: the same token can be decisive for one query and noise
    for another.
    """
    terms = {f"n:{token}" for token in name_key.split() if len(token) >= MIN_TERM_LENGTH}
    terms |= {f"a:{token}" for token in address_key.split() if len(token) >= MIN_TERM_LENGTH}
    return terms


def build_index(source_paths: Iterable[Path], database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = connect_database(database_path)
    for table in ("records", "postings", "term_stats"):
        connection.execute(f"DELETE FROM {table}")
    # Explicitly drop the secondary lookup indexes even if this database
    # file was already built by an older version of this script (or an
    # earlier, interrupted run) that created them up front. Without this,
    # reusing an existing database file would silently fall back to
    # incremental (slow) index maintenance during the bulk load below.
    for index_name in ("records_name", "records_address", "records_name_compact"):
        connection.execute(f"DROP INDEX IF EXISTS {index_name}")
    connection.commit()
    insert_sql = """
        INSERT OR REPLACE INTO records (
            entity_id, source, business_name, business_address, country,
            name_key, address_key
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    posting_sql = "INSERT OR IGNORE INTO postings VALUES (?, ?, ?)"
    try:
        for source_path in source_paths:
            rows = []
            postings: list[tuple[str, str, str]] = []
            source = source_path.stem.split("_", 1)[1]
            for row in iter_rows(source_path):
                name_key = normalize_name(row["business_name"])
                address_key = normalize_address(row["business_address"])
                entity_id = row["entity_id"]
                country = row["country"]
                postings.extend(
                    (term, country, entity_id)
                    for term in index_terms(name_key, address_key)
                )
                rows.append(
                    (
                        entity_id,
                        source,
                        row["business_name"],
                        row["business_address"],
                        country,
                        name_key,
                        address_key,
                    )
                )
                if len(rows) >= 10_000:
                    connection.executemany(insert_sql, rows)
                    connection.executemany(posting_sql, postings)
                    connection.commit()
                    rows.clear()
                    postings.clear()
            if rows:
                connection.executemany(insert_sql, rows)
                connection.executemany(posting_sql, postings)
                connection.commit()
        # All rows are loaded now -- derive the statistics and build the
        # secondary lookup indexes in bulk, instead of maintaining them
        # incrementally across hundreds of thousands of individual inserts.
        t_index = time.time()
        connection.execute(
            "INSERT INTO term_stats (term, country, document_frequency, weight) "
            "SELECT term, country, COUNT(*), 0.0 FROM postings GROUP BY term, country"
        )
        connection.commit()
        _prune_stopwords(connection)
        _compute_weights(connection)
        create_indexes(connection)
        connection.commit()
        print(f"  (statistics and indexes built in {time.time() - t_index:.1f}s)")
        connection.execute("ANALYZE")
        connection.commit()
    finally:
        connection.close()


def _country_sizes(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        country: count
        for country, count in connection.execute(
            "SELECT country, COUNT(*) FROM records GROUP BY country"
        )
    }


def _prune_stopwords(connection: sqlite3.Connection) -> None:
    """Drop terms whose country-filtered document frequency exceeds the cap.

    Removing them from `postings` (rather than only excluding them at query
    time) is what keeps the index compact: the highest-frequency terms are the
    ones with millions of postings, and they are exactly the ones that carry no
    information. Country matters here -- "delhi" is common in India and rare in
    the US, so a single global cap would misjudge both.
    """
    pruned = 0
    for country, total in _country_sizes(connection).items():
        cap = max(DF_FLOOR, int(DF_FRACTION * total))
        pruned += connection.execute(
            "DELETE FROM postings WHERE country = ? AND term IN "
            "(SELECT term FROM term_stats WHERE country = ? AND document_frequency > ?)",
            (country, country, cap),
        ).rowcount
        connection.execute(
            "DELETE FROM term_stats WHERE country = ? AND document_frequency > ?",
            (country, cap),
        )
    connection.commit()
    print(f"  (pruned {pruned:,} stopword postings)")


def _compute_weights(connection: sqlite3.Connection) -> None:
    """Set weight = log(1 + N_country / df) for every surviving term.

    Grouping by (country, document_frequency) means one UPDATE per distinct
    frequency rather than one per term -- there are far fewer distinct
    frequencies than terms, and the weight only depends on the pair.
    """
    sizes = _country_sizes(connection)
    frequencies = connection.execute(
        "SELECT country, document_frequency FROM term_stats "
        "GROUP BY country, document_frequency"
    ).fetchall()
    for country, document_frequency in frequencies:
        weight = math.log(1.0 + sizes[country] / document_frequency)
        connection.execute(
            "UPDATE term_stats SET weight = ? "
            "WHERE country = ? AND document_frequency = ?",
            (weight, country, document_frequency),
        )
    connection.commit()


def _tokens(value: str) -> set[str]:
    return {token for token in value.split() if len(token) > 1}


def _similarity(left: str, right: str) -> float:
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def query_terms(
    connection: sqlite3.Connection, country: str, terms: set[str]
) -> list[str]:
    """Choose which of a query's terms to retrieve on, rarest first.

    Stops once the cumulative posting count reaches POSTINGS_BUDGET. That budget
    is the scale-invariance mechanism: it bounds the work per query by a
    constant instead of letting it grow with the corpus. Taking the rarest terms
    first also guarantees the most discriminative shared term is always used --
    the opposite of the previous design, which truncated keys in entity_id order
    and so dropped the rarest (most informative) key first. The `chosen` guard
    keeps at least one term even when a single term alone exceeds the budget, so
    a query whose only terms are common words still retrieves something.
    """
    if not terms:
        return []
    placeholders = ",".join("?" for _ in terms)
    statistics = connection.execute(
        f"SELECT term, document_frequency FROM term_stats "
        f"WHERE country = ? AND term IN ({placeholders})",
        (country, *terms),
    ).fetchall()
    statistics.sort(key=lambda statistic: (statistic[1], statistic[0]))
    chosen: list[str] = []
    used = 0
    for term, document_frequency in statistics:
        if chosen and used + document_frequency > POSTINGS_BUDGET:
            break
        chosen.append(term)
        used += document_frequency
    return chosen


def candidate_rows(
    connection: sqlite3.Connection, row: dict[str, str]
) -> list[tuple[str, str, str, str, str]]:
    """The records worth scoring for one Source-1 row, best evidence first.

    Returns at most CANDIDATE_LIMIT rows, ranked by summed IDF weight of the
    shared terms. Bounding the *ranked* list (rather than each of ~25 keys
    separately) is what makes the candidate set both small and high-recall: a
    true link that shares one rare term outranks thousands of records that
    share several common ones, so it survives the cut, whereas under the old
    per-key caps it was dropped by whichever cap it happened to land under.
    """
    name_key = normalize_name(row["business_name"])
    address_key = normalize_address(row["business_address"])
    country = row["country"]
    evidence: defaultdict[str, float] = defaultdict(float)
    terms = query_terms(connection, country, index_terms(name_key, address_key))
    if terms:
        placeholders = ",".join("?" for _ in terms)
        for entity_id, weight in connection.execute(
            "SELECT p.entity_id, SUM(s.weight) FROM postings p "
            "JOIN term_stats s ON s.term = p.term AND s.country = p.country "
            f"WHERE p.country = ? AND p.term IN ({placeholders}) "
            "GROUP BY p.entity_id",
            (country, *terms),
        ):
            evidence[entity_id] = weight
    # An exact address match is the most precise evidence in the pipeline
    # (P=0.910), so it is seeded with a weight no ordinary term sum reaches.
    # That guarantees it survives the ranked cut without needing a separate
    # uncapped lookup -- the previous design unioned it in unconditionally,
    # which let a single query return thousands of candidates.
    if address_key:
        for (entity_id,) in connection.execute(
            "SELECT entity_id FROM records WHERE country = ? AND address_key = ?",
            (country, address_key),
        ):
            evidence[entity_id] += EXACT_ADDRESS_SEED
    if not evidence:
        return []
    ranked = sorted(evidence.items(), key=lambda item: (-item[1], item[0]))
    ranked = ranked[:CANDIDATE_LIMIT]
    identifiers = [entity_id for entity_id, _weight in ranked]
    placeholders = ",".join("?" for _ in identifiers)
    details = {
        record[0]: record
        for record in connection.execute(
            "SELECT entity_id, business_name, business_address, country, source "
            f"FROM records WHERE entity_id IN ({placeholders})",
            tuple(identifiers),
        )
    }
    return [
        details[entity_id]
        for entity_id in identifiers
        if entity_id in details
    ]


def match_score(name_score: float, address_score: float) -> float:
    """Combine name and address overlap into the one ranking score.

    Deliberately simple, and deliberately equal-weight. Everything tried on
    top of it lost: discounting exact names by document frequency (0.8326 vs
    0.8434), boosting an exact address to 0.95 (+0.0005, i.e. noise), and
    weighting the name above the address the way the old 0.65/0.35 split did
    (0.7922). The address carries as much signal as the name precisely because
    it is *more* distinctive, not less: the most common normalized address key
    in the 1M-record corpus is shared by 6 records and none reach the stopword
    cap, whereas 38% of records share an exact normalized name.

    So an exact match is not a separate kind of evidence needing its own
    bonus -- it is the endpoint of the same graded token-overlap scale, and it
    is the middle of that scale that decides between the handful of candidates
    any single row actually has to choose between.
    """
    return NAME_WEIGHT * name_score + ADDRESS_WEIGHT * address_score


def select_matches(
    row: dict[str, str],
    candidates: Iterable[tuple],
    threshold: float,
) -> tuple[list[str], list[str]]:
    """Choose the matches for one row, and the candidate list to report.

    Candidates arrive from `candidate_rows` in retrieval-evidence order, but
    are re-scored by content similarity here. The two orders disagree on
    purpose: retrieval evidence measures how rare the shared terms are, which
    is right for *finding* candidates and wrong for choosing between them.

    Two parts, for the reason set out in the constants block:

    * every candidate at or above `threshold`;
    * plus the single best candidate whenever it clears MIN_EVIDENCE.

    The second part is what makes this dominate a plain threshold rule. An
    empty prediction against non-empty gold scores 0.0 -- the same as one
    wrong prediction -- so on any row that has a counterpart at all, choosing
    something beats choosing nothing. On matched rows that alone lifted macro
    F0.5 from 0.7922 to 0.8434; MIN_EVIDENCE is then what gives it back on the
    5.58% of rows that are genuinely matchless and must stay silent.
    """
    name_key = normalize_name(row["business_name"])
    address_key = normalize_address(row["business_address"])
    scored: list[tuple[float, str]] = []
    candidate_ids: list[str] = []
    for candidate in candidates:
        entity_id, name, address = candidate[0], candidate[1], candidate[2]
        candidate_ids.append(entity_id)
        scored.append((
            match_score(
                _similarity(name_key, normalize_name(name)),
                _similarity(address_key, normalize_address(address)),
            ),
            entity_id,
        ))
    if not scored:
        return [], []

    selected = {entity_id for score, entity_id in scored if score >= threshold}
    best_score, best_entity = max(scored, key=lambda item: (item[0], item[1]))
    if best_score >= MIN_EVIDENCE:
        selected.add(best_entity)
    return sorted(selected), sorted(set(candidate_ids))


_MATCH_CONTEXT = None


def _init_match_worker(context) -> None:
    global _MATCH_CONTEXT
    _MATCH_CONTEXT = context


def _match_chunk(chunk: list[dict[str, str]]) -> list[tuple[str, str, str]]:
    """Score one chunk of Source-1 rows. Runs in a worker process.

    Each process opens its own read-only connection -- sqlite connections
    cannot be shared across a fork -- and returns only the finished strings, so
    what crosses the process boundary is exactly what gets written.
    """
    database_path, threshold = _MATCH_CONTEXT
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    try:
        results = []
        for row in chunk:
            candidates = candidate_rows(connection, row)
            matches, candidate_ids = select_matches(row, candidates, threshold)
            results.append(
                (row["entity_id"], ",".join(matches), ",".join(candidate_ids))
            )
        return results
    finally:
        connection.close()


def _chunked(rows: Iterable[dict[str, str]], size: int) -> Iterator[list[dict[str, str]]]:
    chunk: list[dict[str, str]] = []
    for row in rows:
        chunk.append(row)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def run_matching(source1_path: Path, database_path: Path, output_path: Path,
                 candidate_path: Path, threshold: float,
                 workers: int = 1) -> None:
    """Write matches and candidates for every Source-1 row.

    Parallelised with a fork pool when asked, because retrieval is the dominant
    cost and the corpus is large: at ~12 ms/row, the 1.73M-row test file is
    hours of work that is embarrassingly parallel across rows. Chunks are
    mapped in order and written as they arrive, so the output is byte-identical
    to a serial run regardless of `workers` -- parallelising must not change
    the submission.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    # Rows are read in chunks sized so that one chunk's *candidate* lists stay
    # a few MB in flight; the full candidate sets are far too large to buffer.
    chunk_size = 2_000

    with output_path.open("w", encoding="utf-8", newline="") as match_handle, \
            candidate_path.open("w", encoding="utf-8", newline="") as candidate_handle:
        match_writer = csv.writer(match_handle, delimiter="\t", lineterminator="\n")
        candidate_writer = csv.writer(
            candidate_handle, delimiter="\t", lineterminator="\n"
        )
        match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        chunks = _chunked(iter_rows(source1_path), chunk_size)
        if workers == 1 or "fork" not in mp.get_all_start_methods():
            _init_match_worker((str(database_path), threshold))
            results = (_match_chunk(chunk) for chunk in chunks)
            serial = True
        else:
            pool = mp.get_context("fork").Pool(
                processes=workers,
                initializer=_init_match_worker,
                initargs=((str(database_path), threshold),),
            )
            # imap preserves chunk order, so the two writers see rows in file
            # order exactly as the serial path would.
            results = pool.imap(_match_chunk, chunks)
            serial = False
        try:
            for chunk_results in results:
                for entity_id, matches, candidate_ids in chunk_results:
                    match_writer.writerow([entity_id, matches])
                    candidate_writer.writerow([entity_id, candidate_ids])
        finally:
            if not serial:
                pool.close()
                pool.join()


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
    match.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    match.add_argument(
        "--workers", type=int, default=min(4, max(1, (os.cpu_count() or 1) - 1)),
        help="Processes for the retrieval pass (fork, Linux), capped at 4 to "
             "stay polite on a shared host. The output is byte-identical for "
             "any value, so this only trades time for load -- pass 1 for a "
             "serial run.",
    )

    args = parser.parse_args()
    if args.command == "build-index":
        build_index(args.source, args.database)
    else:
        run_matching(
            args.source1, args.database, args.output, args.candidate,
            args.threshold, args.workers,
        )


if __name__ == "__main__":
    main()
