#!/usr/bin/env python3
"""
Time TM-align on rows from lddt_tm_scores.csv (id1, id2, seq1, seq2, ...).

Coordinates are loaded from local PDB files ({PDB_DIRECTORY}/{id}.pdb) or
downloaded from the Swiss-Model REST API when missing.

Usage:
    python benchmark_tm_align_timing_csv.py

Edit INPUT_CSV, START_ROW, and END_ROW in the configuration block below.
Row slice is [START_ROW, END_ROW) (0-based, end exclusive).
"""
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from tmtools import tm_align

from pdb_coord_fetch import fetch_coords_batch

# --- Configuration ---
INPUT_CSV = "lddt_tm_scores.csv"
PDB_DIRECTORY = "./pdbs"
START_ROW = 0       # 0-based inclusive
END_ROW = 10000      # 0-based exclusive; None = through end of file
FETCH_IF_MISSING = True
BUCKET_SIZE = 200
# ---------------------


def get_length_bucket(length: int, bucket_size: int = 200) -> str:
    bucket_num = (length - 1) // bucket_size
    lower = bucket_num * bucket_size
    upper = (bucket_num + 1) * bucket_size
    return f"({lower},{upper}]"


def compute_tm_score(reference_coords, model_coords, reference_seq, model_seq):
    try:
        start = time.time()
        result = tm_align(reference_coords, model_coords, reference_seq, model_seq)
        elapsed = time.time() - start
        return result.tm_norm_chain1, elapsed
    except Exception as e:
        print(f"Error computing TM-score: {e}")
        return None, None


def _count_csv_rows(path: str) -> int:
    """Fast row count (excluding header) without loading full CSV into memory."""
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1


def _read_csv_slice(path: str, start_row: int, end_row):
    """
    Read only rows [start_row, end_row) (0-based, end exclusive).
    Does not load the entire file into memory.
    """
    nrows = None if end_row is None else end_row - start_row
    skiprows = range(1, start_row + 1) if start_row > 0 else None
    return pd.read_csv(path, skiprows=skiprows, nrows=nrows)


def main():
    input_csv = INPUT_CSV
    start_row = START_ROW
    end_row = END_ROW

    if start_row < 0:
        raise ValueError(f"START_ROW must be >= 0, got {start_row}")
    if end_row is not None and end_row <= start_row:
        raise ValueError(f"END_ROW ({end_row}) must be > START_ROW ({start_row})")

    print(f"Input: {input_csv}")
    print(f"Row range: START_ROW={start_row}, END_ROW={end_row} (end exclusive)")

    total_in_file = _count_csv_rows(input_csv)
    if start_row >= total_in_file:
        raise ValueError(f"start_row {start_row} >= rows in file ({total_in_file})")

    end_slice = end_row if end_row is not None else total_in_file
    end_slice = min(end_slice, total_in_file)

    print(f"Rows in file: {total_in_file}; reading slice [{start_row}, {end_slice}) ...")
    df = _read_csv_slice(input_csv, start_row, end_slice).reset_index(drop=True)

    required = {"id1", "id2", "seq1", "seq2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns {missing}. Found: {list(df.columns)}")

    print(f"Loaded {len(df)} rows for processing")
    print(f"PDB directory: {PDB_DIRECTORY} (fetch_if_missing={FETCH_IF_MISSING})")

    unique_ids = sorted(set(df["id1"]).union(set(df["id2"])))
    print(f"Fetching coordinates for {len(unique_ids)} unique protein IDs...")
    t0 = time.time()
    pdb_info = fetch_coords_batch(unique_ids, PDB_DIRECTORY, fetch_if_missing=FETCH_IF_MISSING)
    print(f"Loaded {len(pdb_info)}/{len(unique_ids)} structures in {time.time() - t0:.2f}s")

    bucket_times = defaultdict(list)
    bucket_lengths = defaultdict(list)
    skipped = 0

    for idx, row in df.iterrows():
        id1, id2 = row["id1"], row["id2"]
        if id1 not in pdb_info or id2 not in pdb_info:
            print(f"Skipping {id1} vs {id2}: missing PDB coordinates")
            skipped += 1
            continue

        coords1, _seq_from_pdb1 = pdb_info[id1]
        coords2, _seq_from_pdb2 = pdb_info[id2]
        seq1 = str(row["seq1"])
        seq2 = str(row["seq2"])

        aa_length = len(coords1)
        bucket = get_length_bucket(aa_length)
        label = f"{id1}_{id2}"
        print(f"\nProcessing {label} (length: {aa_length}, bucket: {bucket}):")

        tm_score, tm_time = compute_tm_score(coords1, coords2, seq1, seq2)
        if tm_score is not None and tm_time is not None:
            stored_tm = row.get("tm_score")
            if stored_tm is not None and not (isinstance(stored_tm, float) and np.isnan(stored_tm)):
                print(f"  TM-score (computed): {tm_score:.4f}  (csv): {float(stored_tm):.4f}")
            else:
                print(f"  TM-score: {tm_score:.4f}")
            print(f"  TM-align time: {tm_time:.4f} seconds")
            bucket_times[bucket].append(tm_time)
            bucket_lengths[bucket].append(aa_length)
        else:
            print("  Failed to compute TM-score")
            skipped += 1

    if not bucket_times:
        print("\nNo successful TM-align calculations to time.")
        print(f"Skipped rows: {skipped}")
        return

    print(f"\n{'=' * 70}")
    print("TM-ALIGN TIMING STATISTICS BY AMINO ACID LENGTH BUCKET")
    print(f"{'=' * 70}")

    all_times = []
    for bucket in sorted(bucket_times.keys(), key=lambda x: int(x.split(",")[0][1:])):
        times = bucket_times[bucket]
        lengths = bucket_lengths[bucket]
        all_times.extend(times)
        print(f"\nBucket: {bucket} amino acids")
        print(f"  Number of calculations: {len(times)}")
        print(f"  Length range: {min(lengths)} - {max(lengths)}")
        print(f"  Average time: {np.mean(times):.4f} seconds")
        print(f"  Median time: {np.median(times):.4f} seconds")
        print(f"  Min time: {np.min(times):.4f} seconds")
        print(f"  Max time: {np.max(times):.4f} seconds")
        print(f"  Std deviation: {np.std(times):.4f} seconds")

    print(f"\n{'=' * 70}")
    print("OVERALL STATISTICS")
    print(f"{'=' * 70}")
    print(f"Total calculations: {len(all_times)}")
    print(f"Skipped rows: {skipped}")
    print(f"Average time: {np.mean(all_times):.4f} seconds")
    print(f"Median time: {np.median(all_times):.4f} seconds")
    print(f"Min time: {np.min(all_times):.4f} seconds")
    print(f"Max time: {np.max(all_times):.4f} seconds")
    print(f"Std deviation: {np.std(all_times):.4f} seconds")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
