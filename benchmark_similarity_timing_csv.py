#!/usr/bin/env python3
"""
Benchmark cosine-similarity timing on pairs from lddt_tm_scores.csv.

ProtT5 encoding and the Siamese transformer forward pass run outside the timer.
Only F.cosine_similarity (global + per-residue) is timed. Buckets by average
sequence length; skips pairs with avg length > 1000.

Usage:
    python benchmark_similarity_timing_csv.py

Edit the configuration block below.
"""
import time
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

from siamese_inference import SiameseInference
from siamese_transformer_train_csv_protbert_env import encode_sequences_prot_t5, freeze_module

# --- Configuration ---
INPUT_CSV = "lddt_tm_scores.csv"
MODEL_PATH = "06.07.pth"
START_ROW = 0       # 0-based inclusive
END_ROW = None      # 0-based exclusive; None = through end of file
TEST_SAMPLES = 1000
NUM_BATCHES = 300
PROT_T5_MODEL = "Rostlab/prot_t5_xl_uniref50"
SHUFFLE_SEED = 42
BUCKET_SIZE = 200
MAX_AVG_LENGTH = 1000
# ---------------------


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def get_length_bucket(length: int, bucket_size: int = 200) -> str:
    bucket_num = (length - 1) // bucket_size
    lower = bucket_num * bucket_size
    upper = (bucket_num + 1) * bucket_size
    return f"({lower},{upper}]"


def _count_csv_rows(path: str) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1


def _read_csv_slice(path: str, start_row: int, end_row: int | None):
    nrows = None if end_row is None else end_row - start_row
    skiprows = range(1, start_row + 1) if start_row > 0 else None
    return pd.read_csv(path, skiprows=skiprows, nrows=nrows)


class CsvPairSequenceDataset(Dataset):
    """seq1 / seq2 pairs from a DataFrame slice."""

    def __init__(self, df: pd.DataFrame):
        required = {"seq1", "seq2"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns {missing}. Found: {list(df.columns)}")
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        seq1 = str(row["seq1"]).strip()
        seq2 = str(row["seq2"]).strip()
        if not seq1 or not seq2 or seq1.lower() == "nan" or seq2.lower() == "nan":
            raise ValueError(f"Row {idx}: empty sequence")
        return {"seq1": seq1, "seq2": seq2}


def _collate_seq_pairs(batch: List[dict]):
    return {
        "seq1": [item["seq1"] for item in batch],
        "seq2": [item["seq2"] for item in batch],
    }


def _load_csv_pairs() -> pd.DataFrame:
    start_row = START_ROW
    end_row = END_ROW
    if start_row < 0:
        raise ValueError(f"START_ROW must be >= 0, got {start_row}")
    if end_row is not None and end_row <= start_row:
        raise ValueError(f"END_ROW ({end_row}) must be > START_ROW ({start_row})")

    total_in_file = _count_csv_rows(INPUT_CSV)
    if start_row >= total_in_file:
        raise ValueError(f"START_ROW {start_row} >= rows in file ({total_in_file})")

    end_slice = end_row if end_row is not None else total_in_file
    end_slice = min(end_slice, total_in_file)
    print(f"Input: {INPUT_CSV}")
    print(f"Row range: START_ROW={start_row}, END_ROW={end_row} (end exclusive)")
    print(f"Rows in file: {total_in_file}; reading [{start_row}, {end_slice})")
    return _read_csv_slice(INPUT_CSV, start_row, end_slice)


def test_model(model_path: str, test_samples: int, num_batches: int):
    inference = SiameseInference(model_path)
    config = inference.config
    max_len = int(config["max_seq_len"])
    batch_size = int(config.get("batch_size", 32))
    device = inference.device

    print("Loading ProtT5 encoder (frozen)...")
    tokenizer = T5Tokenizer.from_pretrained(PROT_T5_MODEL, do_lower_case=False)
    # safetensors avoids torch.load (requires torch>=2.6 with pickle checkpoints)
    prot_encoder = T5EncoderModel.from_pretrained(
        PROT_T5_MODEL, use_safetensors=True
    ).to(device)
    freeze_module(prot_encoder)

    df = _load_csv_pairs()
    print(f"Loaded {len(df)} rows from CSV slice")

    if len(df) > test_samples:
        perm = torch.randperm(len(df))[:test_samples].tolist()
        df = df.iloc[perm].reset_index(drop=True)
        print(f"Subsampled to {len(df)} rows (TEST_SAMPLES={test_samples})")

    test_dataset = CsvPairSequenceDataset(df)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_seq_pairs,
        num_workers=0,
    )

    print(f"\nBenchmarking cosine similarity on up to {test_samples} samples from CSV...")
    print("(ProtT5 + transformer forward pass are excluded from the timer)")
    print(f"Device: {device}")
    print(f"Dataset size: {len(test_dataset)}")
    print(f"Batch size: {batch_size}, max batches: {num_batches}")

    bucket_times = defaultdict(list)
    bucket_lengths = defaultdict(list)

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing batches")):
        if batch_idx >= num_batches:
            break

        emb1, mask1 = encode_sequences_prot_t5(
            batch["seq1"], tokenizer, prot_encoder, device, max_len
        )
        emb2, mask2 = encode_sequences_prot_t5(
            batch["seq2"], tokenizer, prot_encoder, device, max_len
        )

        n = emb1.size(0)
        for i in range(n):
            seq_len1 = int(mask1[i].sum().item())
            seq_len2 = int(mask2[i].sum().item())
            avg_length = (seq_len1 + seq_len2) // 2

            if avg_length > MAX_AVG_LENGTH:
                continue

            bucket = get_length_bucket(avg_length, BUCKET_SIZE)
            prot1_emb = emb1[i : i + 1]
            prot2_emb = emb2[i : i + 1]
            prot1_m = mask1[i : i + 1]
            prot2_m = mask2[i : i + 1]

            with torch.no_grad():
                new_emb1, new_emb2, global_emb1, global_emb2 = inference.model(
                    prot1_emb, prot2_emb, prot1_m, prot2_m
                )

            _sync_device(device)
            start_time = time.perf_counter()
            F.cosine_similarity(global_emb1, global_emb2, dim=1)
            F.cosine_similarity(new_emb1, new_emb2, dim=2)
            _sync_device(device)
            elapsed_time = time.perf_counter() - start_time

            bucket_times[bucket].append(elapsed_time)
            bucket_lengths[bucket].append(avg_length)

    if bucket_times:
        print(f"\n{'=' * 70}")
        print("COSINE SIMILARITY TIMING STATISTICS BY AMINO ACID LENGTH BUCKET")
        print(f"{'=' * 70}")

        sorted_buckets = sorted(bucket_times.keys(), key=lambda x: int(x.split(",")[0][1:]))
        all_times = []

        for bucket in sorted_buckets:
            times = bucket_times[bucket]
            lengths = bucket_lengths[bucket]
            all_times.extend(times)

            print(f"\nBucket: {bucket} amino acids")
            print(f"  Number of calculations: {len(times)}")
            print(f"  Length range: {min(lengths)} - {max(lengths)} amino acids")
            print(f"  Average time: {np.mean(times):.4f} seconds")
            print(f"  Median time: {np.median(times):.4f} seconds")
            print(f"  Min time: {np.min(times):.4f} seconds")
            print(f"  Max time: {np.max(times):.4f} seconds")
            print(f"  Std deviation: {np.std(times):.4f} seconds")

        print(f"\n{'=' * 70}")
        print("OVERALL STATISTICS (All Buckets Combined)")
        print(f"{'=' * 70}")
        print(f"Total calculations: {len(all_times)}")
        print(f"Average time: {np.mean(all_times):.4f} seconds")
        print(f"Median time: {np.median(all_times):.4f} seconds")
        print(f"Min time: {np.min(all_times):.4f} seconds")
        print(f"Max time: {np.max(all_times):.4f} seconds")
        print(f"Std deviation: {np.std(all_times):.4f} seconds")
        print(f"{'=' * 70}")
    else:
        print("\nNo successful cosine similarity calculations to time.")


def main():
    test_model(MODEL_PATH, TEST_SAMPLES, NUM_BATCHES)


if __name__ == "__main__":
    main()
