#!/usr/bin/env python3

import csv
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch

from embed_esm2 import get_esm_embeddings, load_esm
from parse_pdb import parse_pdb

# global config
PARQUET_PATH = "/jet/home/jxu23/OCEANDIR/swiss_under_1000_320M.parquet"
PDB_DIR = "/jet/home/jxu23/OCEANDIR/pdbs"
EMBEDDING_DIR = "/jet/home/jxu23/OCEANDIR/embeddings"
MANIFEST_CSV = "/jet/home/jxu23/OCEANDIR/esm_manifest.csv"

START_ROW = 0
END_ROW = 100

COL1 = "chain_1"
COL2 = "chain_2"

DEVICE = "cuda"
SAVE_DTYPE = np.float16
MAX_SEQUENCE_LENGTH = 1000


def pdb_path(uniprot_id: str) -> str:
    return os.path.join(PDB_DIR, f"{uniprot_id}.pdb")


def embedding_path(uniprot_id: str) -> str:
    return os.path.join(EMBEDDING_DIR, f"{uniprot_id}.npy")


def valid_id(value) -> bool:
    protein_id = str(value).strip()
    return protein_id != "" and protein_id.lower() != "nan"


def sequence_too_long(protein_id: str, sequence: str) -> bool:
    if MAX_SEQUENCE_LENGTH is None:
        return False

    if len(sequence) <= MAX_SEQUENCE_LENGTH:
        return False

    print(
        f"{protein_id}: sequence length {len(sequence)} exceeds "
        f"MAX_SEQUENCE_LENGTH={MAX_SEQUENCE_LENGTH}",
        file=sys.stderr,
    )
    return True


def iter_unique_protein_ids(parquet_path: str):
    parquet_file = pq.ParquetFile(parquet_path)
    total_rows = parquet_file.metadata.num_rows

    start_row = 0 if START_ROW is None else START_ROW
    end_row = total_rows if END_ROW is None else min(END_ROW, total_rows)

    seen = set()
    current_global_row = 0

    for row_group_idx in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group_idx, columns=[COL1, COL2])
        df_chunk = table.to_pandas()

        for local_idx, (_, row) in enumerate(df_chunk.iterrows()):
            global_row_idx = current_global_row + local_idx

            if global_row_idx < start_row:
                continue
            if global_row_idx >= end_row:
                return

            id1 = str(row[COL1]).strip()
            id2 = str(row[COL2]).strip()
            if not valid_id(id1) or not valid_id(id2):
                print(
                    f"[row {global_row_idx}] invalid protein ID; skipping pair",
                    file=sys.stderr,
                )
                continue

            try:
                _, seq1 = load_sequence(id1)
                _, seq2 = load_sequence(id2)
            except Exception as exc:
                print(
                    f"[row {global_row_idx}] {id1} vs {id2}: "
                    f"skipping pair ({exc})",
                    file=sys.stderr,
                )
                continue

            if sequence_too_long(id1, seq1) or sequence_too_long(id2, seq2):
                print(
                    f"[row {global_row_idx}] {id1} vs {id2}: "
                    "skipping both proteins",
                    file=sys.stderr,
                )
                continue

            for protein_id in (id1, id2):
                if protein_id in seen:
                    continue
                seen.add(protein_id)
                yield protein_id

        current_global_row += len(df_chunk)


def manifest_fields():
    return [
        "id",
        "sequence_length",
        "embedding_shape",
        "embedding_dtype",
        "embedding_path",
        "status",
        "error",
    ]


def write_manifest_row(writer, protein_id, sequence_length, shape, status, error=""):
    writer.writerow(
        {
            "id": protein_id,
            "sequence_length": sequence_length,
            "embedding_shape": "x".join(str(dim) for dim in shape),
            "embedding_dtype": str(np.dtype(SAVE_DTYPE)),
            "embedding_path": embedding_path(protein_id),
            "status": status,
            "error": error,
        }
    )


def save_embedding(protein_id: str, embedding: np.ndarray):
    final_path = embedding_path(protein_id)
    tmp_path = f"{final_path}.tmp"

    embedding = embedding.astype(SAVE_DTYPE)
    with open(tmp_path, "wb") as handle:
        np.save(handle, embedding)

    os.replace(tmp_path, final_path)


def load_sequence(protein_id: str):
    path = pdb_path(protein_id)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    coords, sequence = parse_pdb(path)
    if len(sequence) == 0:
        raise ValueError(f"No CA atoms found in {path}")

    return coords, sequence


def compute_and_save_embedding(protein_id: str, model, batch_converter):
    coords, sequence = load_sequence(protein_id)
    if MAX_SEQUENCE_LENGTH is not None and len(sequence) > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"Sequence length {len(sequence)} exceeds "
            f"MAX_SEQUENCE_LENGTH={MAX_SEQUENCE_LENGTH}"
        )

    embedding = get_esm_embeddings(
        sequence,
        model,
        batch_converter,
        device=DEVICE,
    )

    if len(coords) != len(embedding):
        raise ValueError(
            f"Coordinate/embedding length mismatch: "
            f"{len(coords)} coords vs {len(embedding)} embeddings"
        )

    save_embedding(protein_id, embedding)
    return sequence, embedding.shape


def main():
    if not os.path.exists(PARQUET_PATH):
        print(f"Error: parquet file not found: {PARQUET_PATH}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(PDB_DIR):
        print(f"Error: PDB directory not found: {PDB_DIR}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(EMBEDDING_DIR, exist_ok=True)
    manifest_dir = os.path.dirname(MANIFEST_CSV)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)

    print(f"Parquet: {PARQUET_PATH}")
    print(f"PDB dir: {PDB_DIR}")
    print(f"Embedding dir: {EMBEDDING_DIR}")
    print(f"Manifest: {MANIFEST_CSV}")
    print(f"Rows: [{START_ROW}, {END_ROW if END_ROW is not None else 'EOF'})")
    print(f"Max sequence length: {MAX_SEQUENCE_LENGTH}")
    print(f"Device: {DEVICE}")
    print(f"Save dtype: {np.dtype(SAVE_DTYPE)}")

    model, _, batch_converter = load_esm()

    write_header = not os.path.exists(MANIFEST_CSV)
    processed = 0
    skipped_existing = 0
    failed = 0

    with open(MANIFEST_CSV, "a", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=manifest_fields())
        if write_header:
            writer.writeheader()

        for protein_id in iter_unique_protein_ids(PARQUET_PATH):
            final_path = embedding_path(protein_id)

            if os.path.exists(final_path):
                skipped_existing += 1
                print(f"{protein_id}: exists, skipping")
                continue

            try:
                sequence, shape = compute_and_save_embedding(
                    protein_id,
                    model,
                    batch_converter,
                )
                processed += 1
                write_manifest_row(
                    writer,
                    protein_id,
                    len(sequence),
                    shape,
                    status="ok",
                )
                print(f"{protein_id}: saved {shape} -> {final_path}")
            except Exception as exc:
                if "out of memory" in str(exc).lower() and DEVICE == "cuda":
                    torch.cuda.empty_cache()

                failed += 1
                try:
                    _, failed_sequence = load_sequence(protein_id)
                    failed_sequence_length = len(failed_sequence)
                except Exception:
                    failed_sequence_length = 0

                write_manifest_row(
                    writer,
                    protein_id,
                    sequence_length=failed_sequence_length,
                    shape=(),
                    status="failed",
                    error=str(exc),
                )
                print(f"{protein_id}: failed ({exc})", file=sys.stderr)

            manifest_file.flush()

    print("-" * 60)
    print(
        f"Done. Saved: {processed}, "
        f"skipped existing: {skipped_existing}, failed: {failed}"
    )
    print(f"Embeddings saved to: {os.path.abspath(EMBEDDING_DIR)}")
    print(f"Manifest saved to: {os.path.abspath(MANIFEST_CSV)}")


if __name__ == "__main__":
    main()
