"""
segment.py  (v2)

Loads point prompts from annotate.py (v1 or v2 format) and runs SAM2's
video predictor. v2 changes:

  - Each region's points are added at ITS OWN annotation frame (regions
    need not be visible at frame 0).
  - Propagation runs FORWARD and BACKWARD from the prompt frames, so the
    whole clip gets masks regardless of where annotation happened.
  - Writes an overlay preview video per clip (masks color-blended over
    frames) so you can visually verify tracking before trusting scores.
    THIS IS THE FIRST THING TO CHECK when scores look degenerate.

Usage:
  python segment.py --clips_dir ./clips --prompts prompts.json --out_dir ./masks

Output:
  ./masks/<clip>/<region>/frame_NNNN.npy      (bool HxW)
  ./masks/<clip>/_overlay.mp4                 (QC preview)
"""

import argparse
import json
import os

import cv2
import numpy as np

# --- CONFIG: adjust to your local SAM2 install ---
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_base_plus.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
# --------------------------------------------------

REGION_TO_OBJ_ID = {"source": 1, "stream": 2, "pool": 3}
OVERLAY_COLORS = {  # BGR
    "source": (0, 200, 255),
    "stream": (0, 255, 0),
    "pool": (255, 100, 0),
}


def load_sam2_video_predictor():
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as e:
        raise ImportError(
            "Could not import sam2. Install with: "
            "pip install git+https://github.com/facebookresearch/sam2.git"
        ) from e
    if not os.path.exists(SAM2_CHECKPOINT):
        raise FileNotFoundError(
            f"SAM2 checkpoint not found at {SAM2_CHECKPOINT} -- "
            "download it and/or update SAM2_CHECKPOINT in segment.py.")
    return build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT)


def normalize_regions(regions):
    """Accept v1 format (no frame_idx -> 0) and v2 format. Keys starting
    with '_' (e.g. auto_annotate's '_auto' metadata) are ignored."""
    out = {}
    for name, data in regions.items():
        if name.startswith("_"):
            continue
        out[name] = {
            "frame_idx": int(data.get("frame_idx", 0)),
            "points": data["points"],
            "labels": data["labels"],
        }
    return out


def video_to_frame_dir(video_path, frame_dir):
    os.makedirs(frame_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(frame_dir, f"{idx:05d}.jpg"), frame)
        frames.append(frame)
        idx += 1
    cap.release()
    return frames


def to_bool_mask(logits):
    m = (logits > 0.0)
    if hasattr(m, "cpu"):
        m = m.cpu().numpy()
    m = np.asarray(m)
    while m.ndim > 2:
        m = m[0]
    return m.astype(bool)


def write_overlay(frames, masks_per_region, out_path, fps=16):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i, frame in enumerate(frames):
        disp = frame.copy()
        for region, masks in masks_per_region.items():
            m = masks.get(i)
            if m is None or not m.any():
                continue
            color = np.array(OVERLAY_COLORS[region], dtype=np.float32)
            disp[m] = (0.55 * disp[m] + 0.45 * color).astype(np.uint8)
        cv2.putText(disp, f"f{i}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(disp)
    vw.release()


def run_clip(predictor, clip_path, regions, out_dir, tmp_root):
    import torch

    clip_name = os.path.basename(clip_path)
    frame_dir = os.path.join(tmp_root, clip_name.replace(".", "_") + "_frames")
    frames = video_to_frame_dir(clip_path, frame_dir)
    n_frames = len(frames)
    if n_frames == 0:
        print(f"  no frames extracted from {clip_name}, skipping")
        return

    regions = normalize_regions(regions)

    # SAM2's video predictor is designed to run under bfloat16 autocast
    # (the official notebooks enable it globally). Without it, prompting
    # at multiple/non-zero frames can crash with "mat1 and mat2 must have
    # the same dtype, but got BFloat16 and Float" inside memory attention.
    use_cuda = torch.cuda.is_available()
    autocast_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if use_cuda else torch.autocast("cpu", enabled=False))

    obj_id_to_region = {REGION_TO_OBJ_ID[r]: r for r in regions}
    # region -> {frame_idx: mask}
    collected = {r: {} for r in regions}

    with torch.inference_mode(), autocast_ctx:
        state = predictor.init_state(video_path=frame_dir)

        for region_name, data in regions.items():
            f_idx = min(max(data["frame_idx"], 0), n_frames - 1)
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=f_idx,
                obj_id=REGION_TO_OBJ_ID[region_name],
                points=np.array(data["points"], dtype=np.float32),
                labels=np.array(data["labels"], dtype=np.int32),
            )
            print(f"  prompt: {region_name} @ frame {f_idx} "
                  f"({sum(1 for l in data['labels'] if l == 1)} pos pts)")

        # forward from the earliest prompt frame to the end...
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            for obj_id, logits in zip(obj_ids, mask_logits):
                r = obj_id_to_region.get(int(obj_id))
                if r is not None:
                    collected[r][frame_idx] = to_bool_mask(logits)
        # ...and backward to cover frames before the prompt frames
        try:
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
                    state, reverse=True):
                for obj_id, logits in zip(obj_ids, mask_logits):
                    r = obj_id_to_region.get(int(obj_id))
                    if r is not None and frame_idx not in collected[r]:
                        collected[r][frame_idx] = to_bool_mask(logits)
        except TypeError:
            # older sam2 versions lack reverse=; frames before the earliest
            # prompt frame will simply have no masks
            print("  (this sam2 version lacks reverse propagation; frames "
                  "before the prompt frame are uncovered)")

    clip_out_dir = os.path.join(out_dir, clip_name)
    for region_name, per_frame in collected.items():
        rdir = os.path.join(clip_out_dir, region_name)
        os.makedirs(rdir, exist_ok=True)
        # save a mask for EVERY frame index 0..n-1; empty where untracked,
        # so downstream frame alignment with flow files is exact
        shape = frames[0].shape[:2]
        coverage = 0
        for i in range(n_frames):
            m = per_frame.get(i)
            if m is None:
                m = np.zeros(shape, dtype=bool)
            elif m.shape != shape:
                m = cv2.resize(m.astype(np.uint8),
                               (shape[1], shape[0])) > 0
            if m.any():
                coverage += 1
            np.save(os.path.join(rdir, f"frame_{i:04d}.npy"), m)
        print(f"  {region_name}: non-empty masks on {coverage}/{n_frames} frames")
        if coverage < n_frames // 4:
            print(f"    WARNING: {region_name} tracked on <25% of frames -- "
                  f"check the overlay video")

    overlay_path = os.path.join(clip_out_dir, "_overlay.mp4")
    write_overlay(frames, collected, overlay_path)
    print(f"  overlay preview -> {overlay_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_dir", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out_dir", default="./masks")
    parser.add_argument("--tmp_dir", default="./_frame_cache")
    parser.add_argument("--redo", action="store_true",
                        help="Re-segment clips that already have masks")
    args = parser.parse_args()

    with open(args.prompts, "r") as f:
        all_prompts = json.load(f)

    predictor = load_sam2_video_predictor()
    os.makedirs(args.out_dir, exist_ok=True)

    for clip_name, regions in all_prompts.items():
        if not regions:
            print(f"Skipping {clip_name}: no annotated regions")
            continue
        clip_path = os.path.join(args.clips_dir, clip_name)
        if not os.path.exists(clip_path):
            print(f"Skipping {clip_name}: file not found")
            continue
        overlay_done = os.path.exists(
            os.path.join(args.out_dir, clip_name, "_overlay.mp4"))
        if overlay_done and not args.redo:
            print(f"Skipping {clip_name}: already segmented "
                  f"(use --redo to overwrite)")
            continue
        print(f"Processing {clip_name} ...")
        try:
            run_clip(predictor, clip_path, regions, args.out_dir, args.tmp_dir)
        except Exception:
            import traceback
            print(f"  ERROR segmenting {clip_name} -- continuing with "
                  f"remaining clips:")
            traceback.print_exc()

    print("Done.")


if __name__ == "__main__":
    main()
