
"""Train a vehicle detector on KuRALS RD peaks and evaluate the held-out capture."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import maximum_filter, uniform_filter
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits


MATCH = np.array([3, 2])
VALID_RANGE = 594
ZERO_DOPPLER = 32
MAX_CANDIDATES = 250
ROUNDING_EPSILON = 1e-9


@dataclass
class Frame:
    sequence: str
    frame_id: str
    locations: np.ndarray
    features: np.ndarray
    matches: np.ndarray


def suppress(points, order, radius, limit, shape):
    """Retain ranked peaks with rectangular non-maximum suppression."""
    blocked = np.zeros(shape, dtype=bool)
    kept = []
    for index in order:
        r, d = points[index]
        if blocked[r, d]:
            continue
        kept.append(index)
        if len(kept) == limit:
            break
        left, right = radius
        blocked[max(0, r-left):r+left+1, max(0, d-right):d+right+1] = True
    return np.asarray(kept, dtype=int)


def feature_maps(amplitude):
    db = 20 * np.log10(np.maximum(amplitude, 1e-7))
    # Compare each peak to its own local background, to its range and
    # Doppler neighborhoods, and to competing peaks in the same frame.
    power = amplitude ** 2
    guard = uniform_filter(power, (5, 3), mode="nearest") * 15
    features = [db]
    for r, d in [(13, 7), (25, 13), (41, 9)]:
        total = uniform_filter(power, (r, d), mode="nearest") * r * d
        background = np.maximum((total - guard) / (r * d - 15), 1e-12)
        features.append(db - 10 * np.log10(background))
    for r, d in [(19, 3), (5, 11), (1, 9), (13, 1)]:
        mean_db = uniform_filter(db, (r, d), mode="nearest")
        features.extend([db - mean_db, mean_db])
    for radius in [(3, 9), (13, 3), (41, 9), (1, 64)]:
        features.append(db - maximum_filter(db, radius, mode="nearest"))
    features.append(db - db.max(axis=0))
    return db, features


def load_frames(root, split):
    frames = []
    for path in sorted((root / split).glob("*.mat")):
        data = loadmat(path, simplify_cells=True)["data"]
        if str(data["split"]) != split:
            raise ValueError(f"Unexpected split in {path}")
        amplitude = np.asarray(data["rdMagnitude"], dtype=np.float64)
        if amplitude.shape != (800, 64) or not np.isfinite(amplitude).all():
            raise ValueError(f"Invalid range–Doppler data in {path}")
        amplitude = np.maximum(amplitude[:VALID_RANGE + 1], 0)
        db, maps = feature_maps(amplitude)
        mask = (amplitude > 0) & (amplitude == maximum_filter(amplitude, 3))
        mask[[0, -1], :] = False
        mask[:, [0, -1, ZERO_DOPPLER]] = False
        d, r = np.nonzero(mask.T)
        points = np.column_stack((r, d))
        strengths = amplitude[tuple(points.T)]
        order = np.argsort(-strengths, kind="stable")
        points = points[suppress(points, order, (2, 1), MAX_CANDIDATES, amplitude.shape)]
        r, d = points.T
        features = [m[r, d] for m in maps]
        for dr, dd in [(-4, 0), (-2, 0), (-1, 0), (1, 0), (2, 0), (4, 0),
                       (0, -3), (0, -2), (0, -1), (0, 1), (0, 2), (0, 3)]:
            nearby = db[(r + dr).clip(0, len(db)-1), (d + dd).clip(0, db.shape[1]-1)]
            features.append(db[r, d] - nearby)
        features.extend([np.abs(d - ZERO_DOPPLER), r])
        truth = np.column_stack((np.atleast_1d(data["trueRangeBin1"]),
                                 np.atleast_1d(data["trueDopplerBin1"]))).astype(int) - 1
        matches = (np.abs(points[:, None, :] - truth) <= MATCH).all(axis=2)
        frames.append(Frame(str(data["sequenceID"]), str(data["frameID"]), points,
                            np.column_stack(features).astype(np.float32), matches))
    if not frames:
        raise ValueError(f"No prepared {split} frames under {root}")
    return frames


def fit_models(frames):
    x = np.concatenate([f.features for f in frames])
    y = np.concatenate([f.matches.any(axis=1) for f in frames])
    if not 0 < y.sum() < len(y):
        raise ValueError("Training requires positive and negative peaks")
    # Compensate for the class imbalance without duplicating correlated frames.
    weight = np.where(y, np.sqrt((~y).sum() / y.sum()), 1)
    for name, columns in [("contrast_shape", slice(None, -2)),
                          ("contrast_shape_speed", slice(None, -1)),
                          ("contrast_shape_speed_range", slice(None))]:
        model = HistGradientBoostingClassifier(
            max_iter=140, max_leaf_nodes=15, min_samples_leaf=32,
            learning_rate=0.08, l2_regularization=2, early_stopping=False,
            random_state=0,
        )
        model.fit(x[:, columns], y, sample_weight=weight)
        yield name, columns, model


def rank_detections(frames, model, columns):
    scored = model.predict_proba(np.concatenate([f.features[:, columns] for f in frames]))[:, 1]
    ranked = []
    offset = 0
    for frame in frames:
        count = len(frame.locations)
        scores = scored[offset:offset+count]
        offset += count
        ordered = np.argsort(-scores, kind="stable")
        kept = suppress(frame.locations, ordered, MATCH, count, (VALID_RANGE + 1, 64))
        ranked.append((kept, scores[kept]))
    return ranked


def evaluate(frames, ranked, threshold, keep_details=False):
    tp = fp = fn = covered = 0
    details = []
    for frame, (indices, scores) in zip(frames, ranked, strict=True):
        n_truth = frame.matches.shape[1]
        matched = np.zeros(n_truth, dtype=bool)
        covered += int(frame.matches.any(axis=0).sum())
        frame_tp = frame_fp = 0
        for index in indices[scores >= threshold]:
            available = np.flatnonzero(frame.matches[index] & ~matched)
            if len(available):
                matched[available[0]] = True
                frame_tp += 1
            else:
                frame_fp += 1
        frame_fn = n_truth - frame_tp
        tp += frame_tp
        fp += frame_fp
        fn += frame_fn
        if keep_details:
            details.append({"Sequence": frame.sequence, "FrameID": frame.frame_id,
                            "TrueTargets": n_truth, "TP": frame_tp, "FP": frame_fp,
                            "FN": frame_fn})
    metrics = {
        "Frames": len(frames), "TrueTargets": tp + fn, "TP": tp, "FP": fp, "FN": fn,
        "Pd": tp / (tp + fn) if tp + fn else 0,
        "Precision": tp / (tp + fp) if tp + fp else 0,
        "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        "FalseAlarmsPerFrame": fp / len(frames),
        "CandidateRecall": covered / (tp + fn) if tp + fn else 0,
        "DecisionThreshold": threshold,
    }
    return metrics, details


def choose_threshold(frames, ranked, max_fp_per_frame):
    # The number of false alarms is nonincreasing as the threshold rises.
    # Find the lowest threshold meeting the cap, making recall maximal.
    thresholds = np.unique(np.r_[np.nextafter(1.0, 2.0),
                                np.concatenate([score for _, score in ranked])])
    left, right = 0, len(thresholds) - 1
    while left < right:
        mid = (left + right) // 2
        metrics, _ = evaluate(frames, ranked, thresholds[mid])
        if metrics["FalseAlarmsPerFrame"] > max_fp_per_frame + ROUNDING_EPSILON:
            left = mid + 1
        else:
            right = mid
    selected, _ = evaluate(frames, ranked, thresholds[left])
    return selected


def strongest_only(ranked):
    return [(indices[:1], scores[:1]) for indices, scores in ranked]


def result_order(selection):
    metrics = selection[0]
    return metrics["Pd"], metrics["F1"]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared KuRALS train/val/test root")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-val-fp", type=float, default=0.498,
                        help="Validation false-alarm limit, in false alarms per frame")
    args = parser.parse_args()
    if not np.isfinite(args.max_val_fp) or args.max_val_fp < 0:
        parser.error("--max-val-fp must be finite and nonnegative")

    train, val = (load_frames(args.data, split) for split in ("train", "val"))
    if {(f.sequence, f.frame_id) for f in train} & {(f.sequence, f.frame_id) for f in val}:
        raise ValueError("Training and validation frames overlap")
    print(f"Loaded {len(train)} training frames and {len(val)} validation frames", flush=True)
    selections = []
    single_selections = []
    with threadpool_limits(limits=4):
        for name, columns, model in fit_models(train):
            candidates = rank_detections(val, model, columns)
            metrics = choose_threshold(val, candidates, args.max_val_fp)
            selections.append(({"Model": name, **metrics}, model, columns))
            print(f"Validation {name}: Pd={metrics['Pd']:.3f}, "
                  f"FP/frame={metrics['FalseAlarmsPerFrame']:.3f}", flush=True)
            single_metrics = choose_threshold(val, strongest_only(candidates), 0.25)
            single_selections.append(({"Model": name, **single_metrics}, model, columns))
            print(f"Validation {name} strongest only: Pd={single_metrics['Pd']:.3f}, "
                  f"FP/frame={single_metrics['FalseAlarmsPerFrame']:.3f}", flush=True)
        best, model, columns = max(selections, key=result_order)
        strict, strict_model, strict_columns = max(single_selections, key=result_order)
        print(f"Locked {best['Model']}, score threshold {best['DecisionThreshold']:.8f}", flush=True)
        print(f"Locked strongest-only {strict['Model']}, "
              f"score threshold {strict['DecisionThreshold']:.8f}", flush=True)
        test = load_frames(args.data, "test")
        if {f.sequence for f in test} & {f.sequence for f in train + val}:
            raise ValueError("Test capture overlaps training or validation")
        test_metrics, details = evaluate(test, rank_detections(test, model, columns),
                                         best["DecisionThreshold"], keep_details=True)
        strict_ranked = strongest_only(rank_detections(test, strict_model, strict_columns))
        strict_metrics, strict_details = evaluate(test, strict_ranked,
                                                  strict["DecisionThreshold"], keep_details=True)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "validation_metrics.csv", [item[0] for item in selections])
    write_csv(args.out / "independent_test_metrics.csv", [{"Model": best["Model"], **test_metrics}])
    write_csv(args.out / "independent_test_per_frame.csv", details)
    write_csv(args.out / "validation_strongest_only_metrics.csv", [item[0] for item in single_selections])
    write_csv(args.out / "independent_test_strongest_only_metrics.csv",
              [{"Model": strict["Model"], **strict_metrics}])
    write_csv(args.out / "independent_test_strongest_only_per_frame.csv", strict_details)
    print("Multi-candidate:", test_metrics, flush=True)
    print("Strongest only:", strict_metrics, flush=True)


if __name__ == "__main__":
    main()
