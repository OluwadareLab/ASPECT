#!/usr/bin/env python3
"""
Build stratified, non-overlapping train/val/test datasets from processed SpliceSeq events.

Pipeline:
1. load all processed event CSVs
2. keep one example per exact sequence
3. drop ambiguous sequences that map to multiple splice types
4. create a global stratified train/val/test split
5. derive multi-class and class-combination datasets from those fixed splits
"""

from __future__ import annotations

import argparse
import csv
import itertools
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


SPLITS = ("train", "val", "test")


@dataclass
class SequenceRecord:
    splice_type: str
    sequence: str


def load_excluded_sequences(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    excluded: set[str] = set()
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return excluded
        if "sequence" not in reader.fieldnames:
            raise SystemExit(f"Exclude file must have a 'sequence' column: {path}")
        for row in reader:
            sequence = str(row["sequence"]).strip().upper()
            if sequence:
                excluded.add(sequence)
    return excluded


def load_processed_records(
    processed_dir: Path,
    excluded_sequences: set[str] | None = None,
) -> list[SequenceRecord]:
    excluded_sequences = excluded_sequences or set()
    records: list[SequenceRecord] = []
    for csv_path in sorted(processed_dir.glob("*_all_events_1024.csv")):
        with open(csv_path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                splice_type = row["splice_type"].strip().upper()
                sequence = row["sequence"].strip().upper()
                if splice_type and sequence and sequence not in excluded_sequences:
                    records.append(SequenceRecord(splice_type=splice_type, sequence=sequence))
    return records


def split_counts(n: int, fracs: tuple[float, float, float]) -> tuple[int, int, int]:
    train_frac, val_frac, test_frac = fracs
    raw = [n * train_frac, n * val_frac, n * test_frac]
    counts = [int(x) for x in raw]
    remainder = n - sum(counts)
    fractions = sorted(((raw[i] - counts[i], i) for i in range(3)), reverse=True)
    for _, idx in fractions[:remainder]:
        counts[idx] += 1

    if n >= 3:
        for idx in range(3):
            if counts[idx] == 0:
                donor = max(range(3), key=lambda i: counts[i])
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[idx] += 1

    return counts[0], counts[1], counts[2]


def write_label_sequence_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["splice_type", "sequence"])
        writer.writerows(rows)


def write_group_summary(
    path: Path,
    combo_name: str,
    split_rows: dict[str, list[tuple[str, str]]],
    labels: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            header = ["dataset", "split", "rows_out"] + [f"count_{label}" for label in labels]
            writer.writerow(header)
        for split_name, rows in split_rows.items():
            counts = Counter(label for label, _ in rows)
            writer.writerow([combo_name, split_name, len(rows)] + [counts.get(label, 0) for label in labels])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-dir",
        default=str(Path(__file__).resolve().parent.parent / "data" / "processed_1024"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent.parent / "datasets"),
    )
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-sequences-file",
        default=None,
        help="CSV with a 'sequence' column; matching rows are removed before deduplication/splitting.",
    )
    args = parser.parse_args()

    fracs = (args.train_frac, args.val_frac, args.test_frac)
    if abs(sum(fracs) - 1.0) > 1e-9:
        raise SystemExit("Split fractions must sum to 1.0")

    processed_dir = Path(args.processed_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    excluded_path = Path(args.exclude_sequences_file).resolve() if args.exclude_sequences_file else None
    excluded_sequences = load_excluded_sequences(excluded_path)

    records = load_processed_records(processed_dir, excluded_sequences=excluded_sequences)
    if not records:
        raise SystemExit(f"No processed event CSVs found in {processed_dir}")

    labels_by_sequence: dict[str, set[str]] = defaultdict(set)
    for record in records:
        labels_by_sequence[record.sequence].add(record.splice_type)

    ambiguous_sequences = {seq: labels for seq, labels in labels_by_sequence.items() if len(labels) > 1}
    clean_sequences_by_label: dict[str, list[str]] = defaultdict(list)
    for sequence, labels in labels_by_sequence.items():
        if len(labels) == 1:
            clean_sequences_by_label[next(iter(labels))].append(sequence)

    rng = random.Random(args.seed)
    split_to_rows: dict[str, list[tuple[str, str]]] = {split: [] for split in SPLITS}
    for label, sequences in sorted(clean_sequences_by_label.items()):
        rng.shuffle(sequences)
        n_train, n_val, n_test = split_counts(len(sequences), fracs)
        split_to_rows["train"].extend((label, seq) for seq in sequences[:n_train])
        split_to_rows["val"].extend((label, seq) for seq in sequences[n_train : n_train + n_val])
        split_to_rows["test"].extend((label, seq) for seq in sequences[n_train + n_val : n_train + n_val + n_test])

    for split in SPLITS:
        rng.shuffle(split_to_rows[split])

    labels = tuple(sorted(clean_sequences_by_label))
    if len(labels) < 2:
        raise SystemExit(f"Need at least 2 classes after cleaning, found: {labels}")

    multi_dir = output_dir / "multi_class_datasets"
    for split in SPLITS:
        write_label_sequence_csv(multi_dir / f"{split}.csv", split_to_rows[split])

    with open(multi_dir / "split_class_distribution.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "splice_type", "count", "percent_in_split"])
        for split in SPLITS:
            counts = Counter(label for label, _ in split_to_rows[split])
            total = sum(counts.values())
            for label in labels:
                count = counts.get(label, 0)
                pct = 0.0 if total == 0 else round(100.0 * count / total, 4)
                writer.writerow([split, label, count, pct])

    group_specs: list[tuple[str, int, str]] = [
        ("binary_datsets", 2, "binary_dataset_summary.csv"),
        ("three_class_datasets", 3, "three_class_dataset_summary.csv"),
        ("four_class_datasets", 4, "four_class_dataset_summary.csv"),
    ]

    for group_name, combo_size, summary_name in group_specs:
        if len(labels) < combo_size:
            continue
        summary_path = output_dir / group_name / summary_name
        if summary_path.exists():
            summary_path.unlink()
        for combo in itertools.combinations(labels, combo_size):
            combo_name = "_vs_".join(combo)
            combo_dir = output_dir / group_name / combo_name
            combo_split_rows: dict[str, list[tuple[str, str]]] = {}
            for split in SPLITS:
                rows = [row for row in split_to_rows[split] if row[0] in combo]
                combo_split_rows[split] = rows
                write_label_sequence_csv(combo_dir / f"{split}.csv", rows)
            write_group_summary(summary_path, combo_name, combo_split_rows, combo)

    audit_path = output_dir / "build_audit_summary.csv"
    with open(audit_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["input_rows", len(records)])
        writer.writerow(["excluded_holdout_sequences", len(excluded_sequences)])
        writer.writerow(["unique_sequences", len(labels_by_sequence)])
        writer.writerow(["ambiguous_sequences_dropped", len(ambiguous_sequences)])
        writer.writerow(["clean_sequences", sum(len(v) for v in clean_sequences_by_label.values())])
        for label in labels:
            writer.writerow([f"clean_count_{label}", len(clean_sequences_by_label[label])])
        for split in SPLITS:
            writer.writerow([f"{split}_rows", len(split_to_rows[split])])

    conflicts_path = output_dir / "ambiguous_sequence_examples.csv"
    with open(conflicts_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence_prefix", "labels"])
        for sequence, seq_labels in list(sorted(ambiguous_sequences.items()))[:500]:
            writer.writerow([sequence[:120], "|".join(sorted(seq_labels))])

    print(f"Built stratified datasets at: {output_dir}")
    print(f"Detected labels: {', '.join(labels)}")
    print(f"Ambiguous sequences dropped: {len(ambiguous_sequences)}")


if __name__ == "__main__":
    main()
