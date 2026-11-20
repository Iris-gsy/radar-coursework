"""Prepare the published KuRALS-PD LandCar split from downloaded RD frames."""

import argparse
from collections import Counter
import csv
from pathlib import Path

import numpy as np
from scipy.io import savemat
from scipy.ndimage import label


FIRST_SEQUENCE = "20220413_141133_Ku3km_LandCar"
SECOND_SEQUENCE = "20220413_144137_Ku3km_LandCar"
SEQUENCES = {"train": FIRST_SEQUENCE, "val": FIRST_SEQUENCE, "test": SECOND_SEQUENCE}
FRAME_COUNTS = {"train": 500, "val": 200, "test": 500}


def read_manifest(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [(row["split"], row["sequence"], row["frame_id"]) for row in csv.DictReader(handle)]
    if Counter(split for split, _, _ in rows) != FRAME_COUNTS:
        raise ValueError("The manifest must contain 500 train, 200 validation and 500 test frames.")
    if any(SEQUENCES[split] != sequence or len(frame) != 6 or not frame.isascii() or not frame.isdigit()
           for split, sequence, frame in rows):
        raise ValueError("Unexpected capture sequence or frame ID.")
    if len({(sequence, frame) for _, sequence, frame in rows}) != len(rows):
        raise ValueError("Repeated capture frame in the manifest.")
    return rows


def convert_frame(frame_path, label_path, split, sequence, frame, scene_id):
    rd = np.load(frame_path, allow_pickle=False)
    masks = np.load(label_path, allow_pickle=False)
    if rd.shape != (64, 800) or masks.shape != (5, 64, 800):
        raise ValueError(f"Unexpected RD or annotation dimensions: {frame_path}")

    components, count = label(masks[3].T > 0)
    range_bins, doppler_bins = [], []
    for component in range(1, count + 1):
        rows, columns = np.where(components == component)
        if len(rows) >= 3:
            range_bins.append(round(np.mean(rows)) + 1)
            doppler_bins.append(round(np.mean(columns)) + 1)
    range_bins = np.asarray(range_bins, dtype=float)
    doppler_bins = np.asarray(doppler_bins, dtype=float)
    return {
        "rdMagnitude": rd.T.astype(np.float32),
        "trueRangeBin1": range_bins,
        "trueDopplerBin1": doppler_bins,
        "trueRanges_m": (range_bins - 1) * 7.5,
        "trueVelocities_mps": (doppler_bins - 1) * 1.769 - 56.61,
        "rangeAxis_m": np.arange(800, dtype=np.float32) * 7.5,
        "velocityAxis_mps": np.arange(64, dtype=np.float32) * 1.769 - 56.61,
        "weather": "real",
        "split": split,
        "sceneID": scene_id,
        "sequenceID": sequence,
        "frameID": frame,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Folder containing the two LandCar captures")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for train/val/test .mat files")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("split_manifest.csv"))
    args = parser.parse_args()

    files = []
    for split, sequence, frame in read_manifest(args.manifest):
        capture = args.input / sequence
        rd = capture / "range_doppler_numpy" / f"{frame}.npy"
        ann = capture / "annotations" / "dense" / frame / "range_doppler.npy"
        for path in (rd, ann):
            if not path.is_file():
                raise FileNotFoundError(f"Missing KuRALS-PD frame or dense annotation: {path}")
        files.append((split, sequence, frame, rd, ann))

    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory.")

    processed = Counter()
    for split, sequence, frame, rd, ann in files:
        output = args.output / split
        output.mkdir(parents=True, exist_ok=True)
        processed[split] += 1
        data = convert_frame(rd, ann, split, sequence, frame, processed[split])
        savemat(output / f"{sequence}_{frame}.mat", {"data": data}, do_compression=True)

    for split in FRAME_COUNTS:
        print(f"{split}: {processed[split]} frames")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
