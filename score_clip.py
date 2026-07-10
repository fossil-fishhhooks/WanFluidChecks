"""
score_clip.py  (v2)

Loads masks (segment.py) and flow (flow.py), runs v2 checks, writes one
JSON per clip and a summary CSV with per-component columns. Prints
diagnostics on the console so NA results explain themselves.

Usage:
  python score_clip.py --masks_dir ./masks --flow_dir ./flow --out_dir ./results
"""

import argparse
import csv
import json
import os

import numpy as np

import cv2

from checks import (
    volume_check,
    color_consistency,
    compensate_background_motion,
    gravity_taper,
    gravity_direction,
    gravity_speedup,
    gravity_leading_edge,
    combine_gravity,
    composite_score,
)

REGIONS = ["source", "stream", "pool"]


def load_clip_frames(clip_path):
    if not clip_path or not os.path.exists(clip_path):
        return []
    cap = cv2.VideoCapture(clip_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def load_region_frames(region_dir):
    if not os.path.isdir(region_dir):
        return []
    files = sorted(f for f in os.listdir(region_dir) if f.endswith(".npy"))
    return [np.load(os.path.join(region_dir, f)) for f in files]


def load_flow_frames(flow_dir_for_clip):
    """Returns (flow_frames, conf_frames); conf empty for old flow dirs."""
    if not flow_dir_for_clip or not os.path.isdir(flow_dir_for_clip):
        return [], []
    files = sorted(os.listdir(flow_dir_for_clip))
    flow = [np.load(os.path.join(flow_dir_for_clip, f))
            for f in files if f.startswith("frame_") and f.endswith(".npy")]
    conf = [np.load(os.path.join(flow_dir_for_clip, f))
            for f in files if f.startswith("conf_") and f.endswith(".npy")]
    return flow, conf


def pad_to_same_length(region_frames):
    non_empty = [f for f in region_frames.values() if f]
    if not non_empty:
        return None
    max_len = max(len(f) for f in non_empty)
    shape = non_empty[0][0].shape
    out = {}
    for region, frames in region_frames.items():
        if not frames:
            out[region] = [np.zeros(shape, dtype=bool)] * max_len
        else:
            out[region] = frames + [np.zeros_like(frames[0])] * (max_len - len(frames))
    return out


def stream_mask_sanity(stream_frames):
    """Geometry plausibility of the tracked stream: a pour stream should
    be tall and thin on most frames. A 'stream' mask that is squat or
    huge is almost certainly a segmentation failure (glass, container,
    merged regions) -- physics scores computed on it are meaningless.
    Returns (ok, note_or_None)."""
    aspects, areas = [], []
    for m in stream_frames:
        ys, xs = np.nonzero(m)
        if len(ys) < 30:
            continue
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        aspects.append(h / max(w, 1))
        areas.append(len(ys) / m.size)
    if not aspects:
        return False, "stream mask empty on all frames"
    med_aspect = float(np.median(aspects))
    med_area = float(np.median(areas))
    if med_aspect < 1.8 or med_area > 0.20:
        return False, (f"stream mask geometry implausible (median "
                       f"aspect={med_aspect:.1f}, area={100*med_area:.0f}% "
                       f"of frame) -- likely SEGMENTATION failure, not a "
                       f"physics failure; re-annotate this clip")
    return True, None


def score_one_clip(clip_masks_dir, flow_dir_for_clip, clip_path=None):
    region_frames = {r: load_region_frames(os.path.join(clip_masks_dir, r))
                     for r in REGIONS}
    padded = pad_to_same_length(region_frames)
    if padded is None:
        return None

    vol = volume_check(padded)

    stream = region_frames["stream"]
    flow, conf = load_flow_frames(flow_dir_for_clip)

    # liquid union = source | stream | pool — used for flow measurement
    liquid_union = [padded["source"][i] | padded["stream"][i] |
                    padded["pool"][i] for i in range(len(padded["stream"]))]

    drift_info = None
    if not stream:
        na = {"score": None, "notes": ["no stream masks saved"]}
        taper_r, dir_r, spd_r, edge_r = dict(na), dict(na), dict(na), dict(na)
    else:
        taper_r = gravity_taper(stream)
        edge_r = gravity_leading_edge(stream)
        if flow:
            # subtract camera/scene drift that RAFT paints into
            # textureless liquid interiors
            flow_c, drifts = compensate_background_motion(flow, liquid_union)
            drift_mags = [float(np.hypot(dx, dy)) for dx, dy in drifts]
            drift_info = {
                "per_frame_drift": drifts,
                "mean_drift_magnitude":
                    float(np.mean(drift_mags)) if drift_mags else 0.0,
            }
            dir_r = gravity_direction(flow_c, liquid_union, conf)
            spd_r = gravity_speedup(flow_c, liquid_union, conf)
        else:
            miss = {"score": None,
                    "notes": ["no flow files -- run flow.py and pass --flow_dir"]}
            dir_r, spd_r = dict(miss), dict(miss)

    grav = combine_gravity(taper_r, dir_r, spd_r, edge_r)

    mask_ok, mask_note = (stream_mask_sanity(stream) if stream
                          else (False, "no stream masks saved"))
    if mask_note:
        grav.setdefault("notes", []).append(mask_note)

    frames = load_clip_frames(clip_path)
    if frames and stream:
        n_align = min(len(frames), len(padded[REGIONS[0]]))
        color_r = color_consistency(
            frames[:n_align],
            {r: padded[r][:n_align] for r in REGIONS},
            stream[:n_align])
    else:
        color_r = {"score": None,
                   "notes": ["no clip video available for color analysis "
                             "(pass --clips_dir)" if not frames
                             else "no stream masks for color analysis"]}

    combined = composite_score(vol, grav, color_r)

    return {
        "volume_check": vol,
        "color_check": color_r,
        "background_drift": drift_info,
        "gravity_taper": taper_r,
        "gravity_direction": dir_r,
        "gravity_speedup": spd_r,
        "gravity_leading_edge": edge_r,
        "gravity_combined": grav,
        "stream_mask_ok": mask_ok,
        "scores": combined,
    }


def _fmt(x):
    return "NA" if x is None else round(x, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks_dir", required=True)
    parser.add_argument("--clips_dir", default="./clips",
                        help="original clips; enables the color check")
    parser.add_argument("--flow_dir", default=None)
    parser.add_argument("--out_dir", default="./results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    clip_names = sorted(d for d in os.listdir(args.masks_dir)
                        if os.path.isdir(os.path.join(args.masks_dir, d)))

    rows = []
    for clip_name in clip_names:
        flow_dir_for_clip = (os.path.join(args.flow_dir, clip_name)
                             if args.flow_dir else None)
        clip_path = (os.path.join(args.clips_dir, clip_name)
                     if args.clips_dir else None)
        print(f"\nScoring {clip_name}")
        result = score_one_clip(os.path.join(args.masks_dir, clip_name),
                                flow_dir_for_clip, clip_path)
        if result is None:
            print("  no masks found, skipping")
            continue

        with open(os.path.join(args.out_dir, f"{clip_name}.json"), "w") as f:
            json.dump(result, f, indent=2)

        s = result["scores"]
        t, d, sp, e = (result["gravity_taper"], result["gravity_direction"],
                       result["gravity_speedup"],
                       result["gravity_leading_edge"])
        rows.append({
            "clip": clip_name,
            "mask_ok": result["stream_mask_ok"],
            "volume": _fmt(s["volume_score"]),
            "pool_fill": _fmt(result["volume_check"].get("pool_fill_score")),
            "color": _fmt(s["color_score"]),
            "grav_taper": _fmt(t["score"]),
            "grav_direction": _fmt(d["score"]),
            "grav_speedup": _fmt(sp["score"]),
            "grav_leading_edge": _fmt(e["score"]),
            "gravity": _fmt(s["gravity_score"]),
            "composite": _fmt(s["composite"]),
        })

        print(f"  volume={_fmt(s['volume_score'])}  "
              f"color={_fmt(s['color_score'])}  "
              f"gravity={_fmt(s['gravity_score'])} "
              f"[taper={_fmt(t['score'])} dir={_fmt(d['score'])} "
              f"speedup={_fmt(sp['score'])} edge={_fmt(e['score'])}]  "
              f"composite={_fmt(s['composite'])}")
        all_notes = (result["volume_check"].get("notes", [])
                     + result["color_check"].get("notes", [])
                     + result["gravity_combined"].get("notes", []))
        drift = result.get("background_drift")
        if drift and drift["mean_drift_magnitude"] > 0.3:
            all_notes.append(f"compensated camera/scene drift of "
                             f"{drift['mean_drift_magnitude']:.2f} px/frame")
        seen = set()
        for note in all_notes:
            if note not in seen:
                seen.add(note)
                print(f"    note: {note}")

    if rows:
        path = os.path.join(args.out_dir, "_summary.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSummary written to {path}")
    print("Done.")


if __name__ == "__main__":
    main()
