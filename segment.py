"""
segment.py

Loads the point prompts from annotate.py and runs them through SAM2's
video predictor to propagate masks across every frame of each clip.

Requires the `sam2` package (facebookresearch/sam2) and a downloaded
checkpoint. Adjust SAM2_CHECKPOINT / SAM2_CONFIG below to match your setup.

Usage:
  python segment.py --clips_dir ./clips --prompts prompts.json --out_dir ./masks

Output structure:
  ./masks/<clip_name>/<region>/frame_0000.npy   (bool array, HxW)
  ./masks/<clip_name>/<region>/frame_0001.npy
  ...

Each region (source / stream / pool) is tracked independently since
SAM2's video predictor supports multiple object IDs per session -- this
keeps the mask ID meaning stable, which downstream volume_check / gravity
checks rely on rather than re-deriving it from color or position.
"""

import argparse
import json
import os

import cv2
import numpy as np

# --- CONFIG: adjust these to your local SAM2 install ---
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
# ---------------------------------------------------------

REGION_TO_OBJ_ID = {"source": 1, "stream": 2, "pool": 3}


def load_sam2_video_predictor():
    """
    Lazy import + build so this file can be inspected/tested without
    sam2 installed. Raises a clear error if the package or checkpoint
    is missing.
    """
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as e:
        raise ImportError(
            "Could not import sam2. Install with: "
            "pip install git+https://github.com/facebookresearch/sam2.git"
        ) from e

    if not os.path.exists(SAM2_CHECKPOINT):
        raise FileNotFoundError(
            f"SAM2 checkpoint not found at {SAM2_CHECKPOINT}. "
            "Download from the sam2 repo's checkpoints/download_ckpts.sh "
            "and update SAM2_CHECKPOINT in segment.py."
        )

    predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT)
    return predictor


def video_to_frame_dir(video_path, frame_dir):
    """SAM2's video predictor expects a directory of JPEG frames."""
    os.makedirs(frame_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(frame_dir, f"{idx:05d}.jpg"), frame)
        idx += 1
    cap.release()
    return idx  # total frame count


def run_clip(predictor, clip_path, regions, out_dir, tmp_root):
    clip_name = os.path.basename(clip_path)
    frame_dir = os.path.join(tmp_root, clip_name.replace(".", "_") + "_frames")
    n_frames = video_to_frame_dir(clip_path, frame_dir)
    if n_frames == 0:
        print(f"  no frames extracted from {clip_name}, skipping")
        return

    state = predictor.init_state(video_path=frame_dir)

    for region_name, data in regions.items():
        obj_id = REGION_TO_OBJ_ID[region_name]
        points = np.array(data["points"], dtype=np.float32)
        labels = np.array(data["labels"], dtype=np.int32)
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )

    clip_out_dir = os.path.join(out_dir, clip_name)
    for region_name in regions:
        os.makedirs(os.path.join(clip_out_dir, region_name), exist_ok=True)

    obj_id_to_region = {v: k for k, v in REGION_TO_OBJ_ID.items() if k in regions}

    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        for obj_id, logits in zip(obj_ids, mask_logits):
            region_name = obj_id_to_region.get(obj_id)
            if region_name is None:
                continue
            mask = (logits > 0.0).cpu().numpy().squeeze().astype(bool)
            np.save(
                os.path.join(clip_out_dir, region_name, f"frame_{frame_idx:04d}.npy"),
                mask,
            )

    print(f"  saved masks for {clip_name} -> {clip_out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_dir", required=True)
    parser.add_argument("--prompts", required=True, help="prompts.json from annotate.py")
    parser.add_argument("--out_dir", default="./masks")
    parser.add_argument("--tmp_dir", default="./_frame_cache")
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
            print(f"Skipping {clip_name}: file not found at {clip_path}")
            continue
        print(f"Processing {clip_name} ...")
        run_clip(predictor, clip_path, regions, args.out_dir, args.tmp_dir)

    print("Done.")


if __name__ == "__main__":
    main()
