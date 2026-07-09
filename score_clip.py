"""
score_clip.py

End-to-end scorer: loads masks from segment.py (and flow fields from
flow.py if present), runs the volume and gravity checks, and writes one
JSON result per clip plus a summary CSV.

Gravity is scored from two sub-checks:
  - leading-edge quadratic fit (transient phase, always attempted)
  - flow-based v^2-vs-y linearity (steady phase, only if --flow_dir
    is given and flow files exist for the clip)

If neither sub-check is applicable for a clip (e.g. stream mask too
small throughout), gravity/composite come back as null in the JSON and
"NA" in the CSV -- deliberately distinct from 0.0, so "couldn't measure"
never gets confused with "measured and failed".

Usage:
  python score_clip.py --masks_dir ./masks --out_dir ./results
  python score_clip.py --masks_dir ./masks --flow_dir ./flow --out_dir ./results
"""

import argparse
import csv
import json
import os

import numpy as np

from checks import (
    volume_check,
    gravity_check_leading_edge,
    gravity_check_stream_flow,
    combine_gravity,
    composite_score,
)


def load_region_frames(region_dir):
    if not os.path.isdir(region_dir):
        return []
    files = sorted(f for f in os.listdir(region_dir) if f.endswith(".npy"))
    return [np.load(os.path.join(region_dir, f)) for f in files]


def load_flow_frames(flow_dir_for_clip):
    if not flow_dir_for_clip or not os.path.isdir(flow_dir_for_clip):
        return []
    files = sorted(f for f in os.listdir(flow_dir_for_clip) if f.endswith(".npy"))
    return [np.load(os.path.join(flow_dir_for_clip, f)) for f in files]


def pad_to_same_length(region_frames):
    """
    Regions may have different frame counts if SAM2 lost a mask early.
    Pad shorter ones with all-False frames so volume_check can sum
    across regions cleanly. Returns None if no region had any frames.
    """
    non_empty = [frames for frames in region_frames.values() if frames]
    if not non_empty:
        return None
    max_len = max(len(frames) for frames in non_empty)
    shape = non_empty[0][0].shape

    padded = {}
    for region, frames in region_frames.items():
        if not frames:
            padded[region] = [np.zeros(shape, dtype=bool) for _ in range(max_len)]
        else:
            pad = max_len - len(frames)
            padded[region] = frames + [np.zeros_like(frames[0]) for _ in range(pad)]
    return padded


def score_one_clip(clip_masks_dir, flow_dir_for_clip):
    regions = ["source", "stream", "pool"]
    region_frames = {r: load_region_frames(os.path.join(clip_masks_dir, r))
                     for r in regions}

    padded = pad_to_same_length(region_frames)
    if padded is None:
        return None

    vol_result = volume_check(padded)

    stream_frames = region_frames["stream"]
    edge_result = gravity_check_leading_edge(stream_frames)

    flow_frames = load_flow_frames(flow_dir_for_clip)
    if flow_frames and stream_frames:
        flow_result = gravity_check_stream_flow(flow_frames, stream_frames)
    else:
        flow_result = {"score": None,
                       "note": "no flow data (run flow.py and pass --flow_dir)"}

    gravity_combined = combine_gravity(edge_result, flow_result)
    combined = composite_score(vol_result, gravity_combined)

    return {
        "volume_check": vol_result,
        "gravity_leading_edge": edge_result,
        "gravity_stream_flow": flow_result,
        "scores": combined,
    }


def _fmt(x):
    return "NA" if x is None else round(x, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks_dir", required=True)
    parser.add_argument("--flow_dir", default=None,
                        help="output dir from flow.py; enables steady-stream gravity check")
    parser.add_argument("--out_dir", default="./results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    clip_names = sorted(
        d for d in os.listdir(args.masks_dir)
        if os.path.isdir(os.path.join(args.masks_dir, d))
    )

    summary_rows = []

    for clip_name in clip_names:
        clip_masks_dir = os.path.join(args.masks_dir, clip_name)
        flow_dir_for_clip = (os.path.join(args.flow_dir, clip_name)
                             if args.flow_dir else None)
        print(f"Scoring {clip_name} ...")
        result = score_one_clip(clip_masks_dir, flow_dir_for_clip)
        if result is None:
            print("  no masks found, skipping")
            continue

        with open(os.path.join(args.out_dir, f"{clip_name}.json"), "w") as f:
            json.dump(result, f, indent=2)

        s = result["scores"]
        summary_rows.append({
            "clip": clip_name,
            "volume_score": _fmt(s["volume_score"]),
            "gravity_score": _fmt(s["gravity_score"]),
            "composite": _fmt(s["composite"]),
        })
        print(f"  volume={_fmt(s['volume_score'])}  "
              f"gravity={_fmt(s['gravity_score'])}  "
              f"composite={_fmt(s['composite'])}")

    if summary_rows:
        summary_path = os.path.join(args.out_dir, "_summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["clip", "volume_score", "gravity_score", "composite"])
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSummary written to {summary_path}")

    print("Done.")


if __name__ == "__main__":
    main()
