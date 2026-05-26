"""
MIT-BIH Arrhythmia Database Loader
------------------------------------
Downloads and parses the MIT-BIH dataset from PhysioNet using wfdb.
Extracts individual heartbeat windows (187 samples) centered on each
annotated R-peak, then maps MIT-BIH symbols → 5 AAMI classes.

Usage (swap into train.py):
    from mitbih_loader import load_mitbih_dataset
    X, y = load_mitbih_dataset(data_dir="./mitdb")

AAMI Class Mapping:
    0 = N  (Normal)
    1 = S  (Supraventricular)
    2 = V  (Ventricular)
    3 = F  (Fusion)
    4 = Q  (Unknown / Pacemaker)
"""

import os
import numpy as np
import wfdb

# ── All 48 MIT-BIH record IDs ────────────────────────────────────────────────
MIT_BIH_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119",
    "121", "122", "123", "124",
    "200", "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221", "222", "223",
    "228", "230", "231", "232", "233", "234",
]

# ── MIT-BIH annotation symbol → AAMI class ───────────────────────────────────
AAMI_MAP = {
    # N — Normal beats
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    # S — Supraventricular ectopic
    "A": 1, "a": 1, "J": 1, "S": 1,
    # V — Ventricular ectopic
    "V": 2, "E": 2,
    # F — Fusion
    "F": 3,
    # Q — Unknown / pacemaker
    "/": 4, "f": 4, "Q": 4,
}

# Beat window: 90 samples before R-peak + 97 after = 187 total (MIT-BIH standard)
BEFORE = 90
AFTER  = 97
SEQ_LEN = BEFORE + AFTER   # 187


def download_mitbih(data_dir: str = "./mitdb") -> None:
    """
    Download the full MIT-BIH Arrhythmia Database from PhysioNet.
    ~110 MB total. Only downloads if the folder doesn't already exist.
    """
    if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
        print(f"MIT-BIH data already present at '{data_dir}' — skipping download.")
        return

    print(f"Downloading MIT-BIH Arrhythmia Database → '{data_dir}' ...")
    print("(~110 MB, may take 1–3 minutes depending on connection)")
    os.makedirs(data_dir, exist_ok=True)
    wfdb.dl_database("mitdb", dl_dir=data_dir)
    print("Download complete.")


def _extract_beats(record_id: str, data_dir: str) -> tuple:
    """
    Extract all labeled heartbeat windows from a single MIT-BIH record.

    Returns:
        beats  : np.ndarray of shape (n_beats, 187)
        labels : np.ndarray of shape (n_beats,) with AAMI class integers
    """
    record_path = os.path.join(data_dir, record_id)
    record = wfdb.rdrecord(record_path)
    ann    = wfdb.rdann(record_path, "atr")

    # Use Lead II (channel 0) — standard for arrhythmia classification
    signal = record.p_signal[:, 0]

    beats, labels = [], []

    for sample_idx, symbol in zip(ann.sample, ann.symbol):
        # Skip symbols not in AAMI mapping (noise, artifact markers, etc.)
        if symbol not in AAMI_MAP:
            continue

        start = sample_idx - BEFORE
        end   = sample_idx + AFTER

        # Skip beats too close to start/end of recording
        if start < 0 or end > len(signal):
            continue

        beat = signal[start:end]

        # Normalize beat to zero-mean, unit-variance
        beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)

        beats.append(beat)
        labels.append(AAMI_MAP[symbol])

    return np.array(beats, dtype=np.float32), np.array(labels, dtype=np.int64)


def load_mitbih_dataset(
    data_dir: str = "./mitdb",
    auto_download: bool = True,
    max_records: int = None,
) -> tuple:
    """
    Load the full MIT-BIH dataset as (X, y) arrays ready for train.py.

    Args:
        data_dir      : folder containing MIT-BIH .dat/.hea/.atr files
        auto_download : if True, downloads the database if not present
        max_records   : limit to first N records (useful for quick testing)

    Returns:
        X : np.ndarray, shape (n_beats, 187) — one row per heartbeat
        y : np.ndarray, shape (n_beats,)     — AAMI class 0–4
    """
    if auto_download:
        download_mitbih(data_dir)

    records = MIT_BIH_RECORDS
    if max_records:
        records = records[:max_records]

    all_beats, all_labels = [], []

    for i, record_id in enumerate(records):
        print(f"  [{i+1:>2}/{len(records)}] Processing record {record_id} ...", end="\r")
        try:
            beats, labels = _extract_beats(record_id, data_dir)
            all_beats.append(beats)
            all_labels.append(labels)
        except Exception as exc:
            print(f"\n  Warning: skipped record {record_id} ({exc})")

    print()  # newline after \r progress

    X = np.concatenate(all_beats,  axis=0)
    y = np.concatenate(all_labels, axis=0)

    # Print class distribution
    class_names = ["N", "S", "V", "F", "Q"]
    counts = {class_names[i]: int(np.sum(y == i)) for i in range(5)}
    print(f"Loaded {len(X):,} beats from {len(records)} records.")
    print(f"Class distribution: {counts}")

    return X, y


# ── Quick standalone test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing MIT-BIH loader (first 2 records only) ...")
    X, y = load_mitbih_dataset(max_records=2)
    print(f"X shape: {X.shape}  |  y shape: {y.shape}")
    print(f"Sample beat (first 10 values): {X[0, :10]}")
