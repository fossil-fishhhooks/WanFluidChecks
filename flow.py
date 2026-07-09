"""
flow.py

Runs RAFT optical flow between consecutive frames of each clip and
saves the RAW (unmasked) flow field. Masking is applied at check time
in checks.py, so the same flow files can be reused for other region
checks later without recomputation.

This step is REQUIRED for the steady-state gravity check
(gravity_check_stream_flow); without it, only the transient
leading-edge check runs.

Uses torchvision's built-in RAFT (raft_large) -- no separate checkpoint
wrangling needed.

Usage:
  python flow.py --clips_dir ./clips --masks_dir ./masks --out_dir ./flow

Output:
  ./flow/<clip_name>/frame_0000.npy   (HxWx2 float32: dx, dy in pixels)
  ...
  (flow frame_i is the motion from frame i to frame i+1, so there is
   one fewer flow file than video frames)
"""

import argparse
import os

import cv2
import numpy as np
import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


def load_raft(device):
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights).to(device).eval()
    transforms = weights.transforms()
    return model, transforms


def load_clip_frames(clip_path):
    cap = cv2.VideoCapture(clip_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


@torch.no_grad()
def compute_flow_pair(model, transforms, frame_a, frame_b, device):
    t_a = torch.from_numpy(frame_a).permute(2, 0, 1).unsqueeze(0).float()
    t_b = torch.from_numpy(frame_b).permute(2, 0, 1).unsqueeze(0).float()
    t_a, t_b = transforms(t_a, t_b)
    flow_preds = model(t_a.to(device), t_b.to(device))
    # RAFT is iterative; last prediction is the refined one
    flow = flow_preds[-1][0].permute(1, 2, 0).cpu().numpy().astype(np.float32)

    # transforms may have resized the input; scale flow back to original
    # frame size so it aligns with the saved masks
    h, w = frame_a.shape[:2]
    fh, fw = flow.shape[:2]
    if (fh, fw) != (h, w):
        scale_x, scale_y = w / fw, h / fh
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        flow[..., 0] *= scale_x
        flow[..., 1] *= scale_y
    return flow


def process_clip(clip_path, out_dir_for_clip, model, transforms, device):
    frames = load_clip_frames(clip_path)
    if len(frames) < 2:
        print(f"  not enough frames in {clip_path}, skipping")
        return

    os.makedirs(out_dir_for_clip, exist_ok=True)

    for i in range(len(frames) - 1):
        out_path = os.path.join(out_dir_for_clip, f"frame_{i:04d}.npy")
        if os.path.exists(out_path):
            continue  # resume support
        flow = compute_flow_pair(model, transforms, frames[i], frames[i + 1], device)
        np.save(out_path, flow)

    print(f"  saved flow for {os.path.basename(clip_path)} -> {out_dir_for_clip}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_dir", required=True)
    parser.add_argument("--masks_dir", required=True,
                        help="output dir from segment.py (used to decide which clips to process)")
    parser.add_argument("--out_dir", default="./flow")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model, transforms = load_raft(device)

    clip_names = [
        d for d in os.listdir(args.masks_dir)
        if os.path.isdir(os.path.join(args.masks_dir, d))
    ]

    for clip_name in clip_names:
        clip_path = os.path.join(args.clips_dir, clip_name)
        if not os.path.exists(clip_path):
            print(f"Skipping {clip_name}: source video not found")
            continue
        print(f"Processing {clip_name} ...")
        process_clip(clip_path, os.path.join(args.out_dir, clip_name),
                     model, transforms, device)

    print("Done.")


if __name__ == "__main__":
    main()
