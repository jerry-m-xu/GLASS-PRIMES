#!/usr/bin/env python3

import csv
import os
import sys

import numpy as np
import pandas as pd

from fgw import compute_fgw_from_features
from parse_pdb import parse_pdb

# global config
TM_DATA_CSV = "/jet/home/jxu23/OCEANDIR/tm_scores.csv"
PDB_DIR = "/jet/home/jxu23/OCEANDIR/pdbs"
EMBEDDING_DIR = "/jet/home/jxu23/OCEANDIR/embeddings"
OUTPUT_CSV = "/jet/home/jxu23/OCEANDIR/fgw_scores.csv"

START_ROW = 0
END_ROW = 50
CSV_CHUNK_SIZE = 1000
FLUSH_EVERY_ROWS = 10
K_NEIGHBORS = 32
ALPHA = 0.7
EPS = 0.05
SINKHORN_ITER = 30
STRUCTURE_EXP_SCALE = 0.1

SEQM_VALUES = {":", ".", " "}


def pdb_path(uniprot_id: str) -> str:
    return os.path.join(PDB_DIR, f"{uniprot_id}.pdb")


def embedding_path(uniprot_id: str) -> str:
    return os.path.join(EMBEDDING_DIR, f"{uniprot_id}.npy")


def compute_local_fgw(
    coords1: np.ndarray,
    coords2: np.ndarray,
    features1: np.ndarray,
    features2: np.ndarray,
):
    return compute_fgw_from_features(
        coords1,
        coords2,
        features1,
        features2,
        alpha=ALPHA,
        eps=EPS,
        sinkhorn_iter=SINKHORN_ITER,
        return_components=True,
    )


def exponential_scaled_distance(value: float, scale: float) -> float:
    return 1 - np.exp(-value / scale)


def knn_indices(coords: np.ndarray, residue_idx: int) -> np.ndarray:
    distances = np.linalg.norm(coords - coords[residue_idx], axis=1)
    k = min(K_NEIGHBORS, len(coords))
    return np.argsort(distances)[:k]


def iter_aligned_residue_pairs(seqxA: str, seqM: str, seqyA: str):
    if not (len(seqxA) == len(seqM) == len(seqyA)):
        raise ValueError("seqxA, seqM, and seqyA must have the same length")

    idx1 = 0
    idx2 = 0

    for align_pos, (aa1, marker, aa2) in enumerate(zip(seqxA, seqM, seqyA)):
        residue_idx1 = idx1 if aa1 != "-" else None
        residue_idx2 = idx2 if aa2 != "-" else None

        if aa1 != "-":
            idx1 += 1
        if aa2 != "-":
            idx2 += 1

        if marker not in SEQM_VALUES:
            continue
        if residue_idx1 is None or residue_idx2 is None:
            continue

        yield align_pos, residue_idx1, residue_idx2, aa1, marker, aa2


def load_protein_data(uniprot_id: str):
    pdb_file = pdb_path(uniprot_id)
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(pdb_file)

    embedding_file = embedding_path(uniprot_id)
    if not os.path.exists(embedding_file):
        raise FileNotFoundError(embedding_file)

    coords, sequence = parse_pdb(pdb_file)
    if len(sequence) == 0:
        raise ValueError(f"No CA atoms found in {pdb_file}")

    embeddings = np.load(embedding_file).astype(np.float32)
    if len(coords) != len(embeddings):
        raise ValueError(
            f"Coordinate/embedding length mismatch for {uniprot_id}: "
            f"{len(coords)} coords vs {len(embeddings)} embeddings"
        )

    return coords, sequence, embeddings


def input_rows(csv_path: str):
    start_row = 0 if START_ROW is None else START_ROW
    end_row = END_ROW
    current_row = 0

    for df_chunk in pd.read_csv(
        csv_path,
        chunksize=CSV_CHUNK_SIZE,
        keep_default_na=False,
    ):
        chunk_start = current_row
        chunk_end = current_row + len(df_chunk)

        if chunk_end <= start_row:
            current_row = chunk_end
            continue
        if end_row is not None and chunk_start >= end_row:
            return

        local_start = max(start_row - chunk_start, 0)
        local_end = len(df_chunk)
        if end_row is not None:
            local_end = min(end_row - chunk_start, len(df_chunk))

        selected_rows = df_chunk.iloc[local_start:local_end]
        for offset, (_, row) in enumerate(selected_rows.iterrows()):
            yield chunk_start + local_start + offset, row

        current_row = chunk_end


def output_fields():
    return [
        "tm_data_row",
        "source_row",
        "id1",
        "id2",
        "tm_score_norm1",
        "align_pos",
        "residue_idx1",
        "residue_idx2",
        "aa1",
        "seqM",
        "aa2",
        "fgw_score",
        "fgw_structure_term",
        "fgw_structure_exp_scaled",
        "fgw_feature_term",
        "neighborhood_size1",
        "neighborhood_size2",
    ]


def write_result(
    writer,
    row,
    input_row_idx,
    residue_pair,
    fgw_score,
    structure_term,
    structure_exp_scaled,
    feature_term,
    n1,
    n2,
):
    align_pos, residue_idx1, residue_idx2, aa1, marker, aa2 = residue_pair
    source_row = row["row"] if "row" in row else input_row_idx

    writer.writerow(
        {
            "tm_data_row": input_row_idx,
            "source_row": source_row,
            "id1": row["id1"],
            "id2": row["id2"],
            "tm_score_norm1": row["tm_score_norm1"],
            "align_pos": align_pos,
            "residue_idx1": residue_idx1,
            "residue_idx2": residue_idx2,
            "aa1": aa1,
            "seqM": marker,
            "aa2": aa2,
            "fgw_score": fgw_score,
            "fgw_structure_term": structure_term,
            "fgw_structure_exp_scaled": structure_exp_scaled,
            "fgw_feature_term": feature_term,
            "neighborhood_size1": n1,
            "neighborhood_size2": n2,
        }
    )


def process_tm_row(row, input_row_idx, writer):
    id1 = str(row["id1"]).strip()
    id2 = str(row["id2"]).strip()

    coords1, _, embeddings1 = load_protein_data(id1)
    coords2, _, embeddings2 = load_protein_data(id2)

    seqxA = str(row["seqxA"])
    seqM = str(row["seqM"])
    seqyA = str(row["seqyA"])

    scored_pairs = 0
    for residue_pair in iter_aligned_residue_pairs(seqxA, seqM, seqyA):
        _, residue_idx1, residue_idx2, _, _, _ = residue_pair

        indices1 = knn_indices(coords1, residue_idx1)
        indices2 = knn_indices(coords2, residue_idx2)

        fgw_score, structure_term, feature_term = compute_local_fgw(
            coords1[indices1],
            coords2[indices2],
            embeddings1[indices1],
            embeddings2[indices2],
        )
        structure_exp_scaled = exponential_scaled_distance(
            structure_term,
            STRUCTURE_EXP_SCALE,
        )

        write_result(
            writer,
            row,
            input_row_idx,
            residue_pair,
            fgw_score,
            structure_term,
            structure_exp_scaled,
            feature_term,
            len(indices1),
            len(indices2),
        )
        scored_pairs += 1

    return scored_pairs


def main():
    if not os.path.exists(TM_DATA_CSV):
        print(f"Error: TM data CSV not found: {TM_DATA_CSV}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(PDB_DIR):
        print(f"Error: PDB directory not found: {PDB_DIR}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(EMBEDDING_DIR):
        print(
            f"Error: embedding directory not found: {EMBEDDING_DIR}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"TM data: {TM_DATA_CSV}")
    print(f"PDB dir: {PDB_DIR}")
    print(f"Embedding dir: {EMBEDDING_DIR}")
    print(f"Rows: [{START_ROW}, {END_ROW if END_ROW is not None else 'EOF'})")
    print(f"CSV chunk size: {CSV_CHUNK_SIZE}")
    print(f"Flush every rows: {FLUSH_EVERY_ROWS}")
    print(f"k-NN size: {K_NEIGHBORS}")
    print(f"Structure exp scale: {STRUCTURE_EXP_SCALE}")
    print(f"Output: {OUTPUT_CSV}")

    output_dir = os.path.dirname(OUTPUT_CSV)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    write_header = not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0
    total_pairs = 0
    failed_rows = 0
    rows_since_flush = 0

    with open(OUTPUT_CSV, "a", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=output_fields())
        if write_header:
            writer.writeheader()

        for input_row_idx, row in input_rows(TM_DATA_CSV):
            try:
                scored_pairs = process_tm_row(
                    row,
                    input_row_idx,
                    writer,
                )
                total_pairs += scored_pairs
                print(
                    f"[row {input_row_idx}] {row['id1']} vs {row['id2']}: "
                    f"{scored_pairs} local FGW scores"
                )
                rows_since_flush += 1
                if rows_since_flush >= FLUSH_EVERY_ROWS:
                    output_file.flush()
                    rows_since_flush = 0
            except Exception as exc:
                failed_rows += 1
                print(
                    f"[row {input_row_idx}] skipped ({exc})",
                    file=sys.stderr,
                )

        output_file.flush()

    print("-" * 60)
    print(f"Done. Local FGW scores: {total_pairs}, failed rows: {failed_rows}")
    print(f"Results saved to: {os.path.abspath(OUTPUT_CSV)}")


if __name__ == "__main__":
    main()
