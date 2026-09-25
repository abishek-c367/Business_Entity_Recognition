from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_mapping(path: Path, key_column: str, value_column: str) -> dict[str, set[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {
            row[key_column]: set(filter(None, row[value_column].split(",")))
            for row in reader
        }


def entity_f05(predicted: set[str], actual: set[str]) -> float:
    if not predicted and not actual:
        return 1.0
    if not predicted or not actual:
        return 0.0
    true_positive = len(predicted & actual)
    precision = true_positive / len(predicted)
    recall = true_positive / len(actual)
    if precision == 0.0 or recall == 0.0:
        return 0.0
    return 1.25 * precision * recall / (0.25 * precision + recall)


def macro_f05(predictions: dict[str, set[str]], labels: dict[str, set[str]]) -> float:
    if set(predictions) != set(labels):
        missing = set(labels) - set(predictions)
        extra = set(predictions) - set(labels)
        raise ValueError(f"Prediction IDs differ: missing={len(missing)}, extra={len(extra)}")
    return sum(entity_f05(predictions[key], labels[key]) for key in labels) / len(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()
    predictions = read_mapping(
        args.predictions, "source1_entity_id", "matched_entity_ids"
    )
    labels = read_mapping(args.labels, "source1_entity_id", "matched_entity_ids")
    print(f"macro_f05={macro_f05(predictions, labels):.8f}")


if __name__ == "__main__":
    main()
