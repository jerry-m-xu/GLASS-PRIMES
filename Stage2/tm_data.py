#!/usr/bin/env python3

import os
import sys

import csv
import pyarrow.parquet as pq
from tmtools import tm_align

from parse_pdb import parse_pdb

# global config 
PARQUET_PATH = "/jet/home/jxu23/OCEANDIR/swiss_under_1000_320M.parquet"
PDB_DIR = "/jet/home/jxu23/OCEANDIR/pdbs"
OUTPUT_CSV = "/jet/home/jxu23/OCEANDIR/tm_scores.csv"
PROGRESS_FILE = "/jet/home/jxu23/OCEANDIR/tm_data_progress.txt"

START_ROW = 32000
END_ROW = 100000

COL1 = "chain_1"
COL2 = "chain_2"

MAX_SEQUENCE_LENGTH = 1000
FLUSH_EVERY_ROWS = 100


def pdb_path(uniprot_id: str) -> str:
    return os.path.join(PDB_DIR, f"{uniprot_id}.pdb")


def run_tm_align(uniprot_id1: str, uniprot_id2: str):
    path1 = pdb_path(uniprot_id1)
    path2 = pdb_path(uniprot_id2)

    if not os.path.exists(path1):
        raise FileNotFoundError(path1)
    if not os.path.exists(path2):
        raise FileNotFoundError(path2)

    coords1, seq1 = parse_pdb(path1)
    coords2, seq2 = parse_pdb(path2)

    if len(seq1) == 0:
        raise ValueError(f"No CA atoms found in {path1}")
    if len(seq2) == 0:
        raise ValueError(f"No CA atoms found in {path2}")
    if MAX_SEQUENCE_LENGTH is not None and len(seq1) > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"{uniprot_id1} sequence length {len(seq1)} exceeds "
            f"MAX_SEQUENCE_LENGTH={MAX_SEQUENCE_LENGTH}"
        )
    if MAX_SEQUENCE_LENGTH is not None and len(seq2) > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"{uniprot_id2} sequence length {len(seq2)} exceeds "
            f"MAX_SEQUENCE_LENGTH={MAX_SEQUENCE_LENGTH}"
        )

    result = tm_align(coords1, coords2, seq1, seq2)

    return {
        "id1": uniprot_id1,
        "id2": uniprot_id2,
        "seq1": seq1,
        "seq2": seq2,
        "tm_score_norm1": float(result.tm_norm_chain1),
        "seqxA": result.seqxA,
        "seqM": result.seqM,
        "seqyA": result.seqyA,
    }


def iter_pairs(parquet_path: str, col1: str, col2: str, start_row: int, end_row):
    parquet_file = pq.ParquetFile(parquet_path)
    total_rows = parquet_file.metadata.num_rows

    if start_row is None:
        start_row = 0

    if end_row is None:
        end_row = total_rows

    current_global_row = 0
    for row_group_idx in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group_idx)
        df_chunk = table.to_pandas()

        for local_idx, (_, row) in enumerate(df_chunk.iterrows()):
            global_row_idx = current_global_row + local_idx

            if global_row_idx < start_row:
                continue
            if global_row_idx >= end_row:
                return

            id1 = str(row[col1]).strip()
            id2 = str(row[col2]).strip()
            yield global_row_idx, id1, id2

        current_global_row += len(df_chunk)


def output_fields():
    return [
        "id1",
        "id2",
        "seq1",
        "seq2",
        "tm_score_norm1",
        "seqxA",
        "seqM",
        "seqyA",
        "row",
    ]


def open_output_writer(csv_path: str):
    output_dir = os.path.dirname(csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    write_header = not os.path.exists(csv_path)
    output_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(output_file, fieldnames=output_fields())
    if write_header:
        writer.writeheader()

    return output_file, writer


def write_progress(row_idx):
    progress_dir = os.path.dirname(PROGRESS_FILE)
    if progress_dir:
        os.makedirs(progress_dir, exist_ok=True)

    with open(PROGRESS_FILE, "w") as progress_file:
        progress_file.write(str(row_idx))


def main():
    if not os.path.exists(PARQUET_PATH):
        print(f"Error: parquet file not found: {PARQUET_PATH}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(PDB_DIR):
        print(f"Error: PDB directory not found: {PDB_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Parquet: {PARQUET_PATH}")
    print(f"PDB dir: {PDB_DIR}")
    print(f"Rows: [{START_ROW}, {END_ROW if END_ROW is not None else 'EOF'})")
    print(f"Max sequence length: {MAX_SEQUENCE_LENGTH}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Progress file: {PROGRESS_FILE}")
    print(f"Flush every rows: {FLUSH_EVERY_ROWS}")

    success = 0
    failed = 0
    rows_since_flush = 0
    last_row_processed = None

    output_file, writer = open_output_writer(OUTPUT_CSV)
    with output_file:
        for row_idx, id1, id2 in iter_pairs(
            PARQUET_PATH, COL1, COL2, START_ROW, END_ROW
        ):
            try:
                result = run_tm_align(id1, id2)
                result["row"] = row_idx
                writer.writerow(result)
                success += 1
                print(
                    f"[row {row_idx}] {id1} vs {id2}: "
                    f"TM={result['tm_score_norm1']:.4f}"
                )
            except Exception as exc:
                failed += 1
                print(
                    f"[row {row_idx}] {id1} vs {id2}: skipped ({exc})",
                    file=sys.stderr,
                )
            finally:
                last_row_processed = row_idx
                rows_since_flush += 1
                if rows_since_flush >= FLUSH_EVERY_ROWS:
                    output_file.flush()
                    write_progress(row_idx)
                    rows_since_flush = 0

        output_file.flush()
        if last_row_processed is not None:
            write_progress(last_row_processed)

    print("-" * 60)
    print(f"Done. Successful: {success}, failed/skipped: {failed}")
    print(f"Last row processed: {last_row_processed}")
    print(f"Results saved to: {os.path.abspath(OUTPUT_CSV)}")


if __name__ == "__main__":
    main()