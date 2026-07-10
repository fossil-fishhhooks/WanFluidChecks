"""
flow.py  (v3)

Runs RAFT optical flow between consecutive frames and saves raw flow
plus a per-pixel CONFIDENCE map from forward-backward consistency.

v3 fixes and hardening:
  - ROOT-CAUSE FIX for garbage vectors: earlier versions fed float
    tensors in [0,255] to torchvision's RAFT transforms, which expect
    [0,1] floats or uint8 -- inputs were ~255x out of range. We now pass
    uint8 and let the transforms convert, which is immune to any
    double-scaling mistake.
  - VERSION STAMP per clip (_meta.json). Flow computed by older/broken
    versions is detected and recomputed automatically -- resume support
    previously kept stale garbage files alive even after the fix.
  - FB-CONSISTENCY CONFIDENCE: flow is computed both directions; pixels
    where forward and (warped) backward flow disagree (occlusions,
    hallucinated matches) get conf=False. Checks can then ignore
    unreliable vectors instead of scoring noise.
  - _flow_vis.mp4 per clip: hue = direction, brightness = magnitude,
    dimmed where low-confidence. Eyeball this like the mask overlays.

Usage:
  python flow.py --clips_dir ./clips --masks_dir ./masks --out_dir ./flow

Output per clip:
  frame_NNNN.npy  (HxWx2 float32 forward flow, px)
  conf_NNNN.npy   (HxW bool, True = reliable)
  _flow_vis.mp4, _meta.json
"""

import argparse
import json
import os
import shutil

import cv2
import numpy as np
import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

FLOW_VERSION = 3


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
def compute_flow_bidirectional(model, transforms, frame_a, frame_b, device):
    """Forward and backward flow in one batched call. Frames uint8 RGB."""
    t_a = torch.from_numpy(frame_a).permute(2, 0, 1)   # uint8, CHW
    t_b = torch.from_numpy(frame_b).permute(2, 0, 1)
    batch_1 = torch.stack([t_a, t_b])                  # [a, b]
    batch_2 = torch.stack([t_b, t_a])                  # [b, a]
    batch_1, batch_2 = transforms(batch_1, batch_2)    # uint8 -> [-1,1] floats
    preds = model(batch_1.to(device), batch_2.to(device))
    out = preds[-1].permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    fw, bw = out[0], out[1]                            # a->b, b->a

    h, w = frame_a.shape[:2]
    fh, fw_w = fw.shape[:2]
    if (fh, fw_w) != (h, w):
        sx, sy = w / fw_w, h / fh
        fw = cv2.resize(fw, (w, h), interpolation=cv2.INTER_LINEAR)
        bw = cv2.resize(bw, (w, h), interpolation=cv2.INTER_LINEAR)
        fw[..., 0] *= sx; fw[..., 1] *= sy
        bw[..., 0] *= sx; bw[..., 1] *= sy
    return fw, bw


def fb_confidence(fw, bw):
    """Classic forward-backward check: warp backward flow to source
    coords along the forward flow; consistent pixels satisfy
    |fw + bw(x+fw)|^2 < 0.01*(|fw|^2 + |bw_warp|^2) + 0.5"""
    h, w = fw.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    map_x = gx + fw[..., 0]
    map_y = gy + fw[..., 1]
    bw_warp = cv2.remap(bw, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    err2 = ((fw + bw_warp) ** 2).sum(-1)
    mag2 = (fw ** 2).sum(-1) + (bw_warp ** 2).sum(-1)
    return err2 < 0.01 * mag2 + 0.5


def flow_to_vis(flow, conf):
    mag = np.linalg.norm(flow, axis=-1)
    ang = (np.arctan2(flow[..., 1], flow[..., 0]) + np.pi) / (2 * np.pi)
    scale = max(np.percentile(mag, 98), 1.0)
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[..., 0] = (ang * 179).astype(np.uint8)
    hsv[..., 1] = 255
    val = np.clip(mag / scale, 0, 1)
    val = val * (0.4 + 0.6 * conf.astype(np.float32))   # dim low-confidence
    hsv[..., 2] = (val * 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def clip_meta_ok(out_dir_for_clip):
    meta_path = os.path.join(out_dir_for_clip, "_meta.json")
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path) as f:
            return json.load(f).get("flow_version") == FLOW_VERSION
    except Exception:
        return False


def process_clip(clip_path, out_dir_for_clip, model, transforms, device,
                 fps=16):
    if clip_meta_ok(out_dir_for_clip):
        print(f"  up-to-date (v{FLOW_VERSION}), skipping")
        return
    if os.path.isdir(out_dir_for_clip):
        # stale or partial (possibly garbage from the old normalization
        # bug) -- wipe and recompute
        print("  removing stale flow (old version) ...")
        shutil.rmtree(out_dir_for_clip)
    os.makedirs(out_dir_for_clip, exist_ok=True)

    frames = load_clip_frames(clip_path)
    if len(frames) < 2:
        print(f"  not enough frames in {clip_path}, skipping")
        return

    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(os.path.join(out_dir_for_clip, "_flow_vis.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    conf_fracs = []
    for i in range(len(frames) - 1):
        fw, bw = compute_flow_bidirectional(
            model, transforms, frames[i], frames[i + 1], device)
        conf = fb_confidence(fw, bw)
        np.save(os.path.join(out_dir_for_clip, f"frame_{i:04d}.npy"), fw)
        np.save(os.path.join(out_dir_for_clip, f"conf_{i:04d}.npy"), conf)
        vw.write(flow_to_vis(fw, conf))
        conf_fracs.append(float(conf.mean()))
    vw.release()

    with open(os.path.join(out_dir_for_clip, "_meta.json"), "w") as f:
        json.dump({"flow_version": FLOW_VERSION,
                   "mean_confidence": float(np.mean(conf_fracs))}, f)
    print(f"  saved flow+conf ({np.mean(conf_fracs)*100:.0f}% pixels "
          f"reliable) -> {out_dir_for_clip}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_dir", required=True)
    parser.add_argument("--masks_dir", required=True,
                        help="output dir from segment.py (selects clips)")
    parser.add_argument("--out_dir", default="./flow")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model, transforms = load_raft(device)

    clip_names = [d for d in os.listdir(args.masks_dir)
                  if os.path.isdir(os.path.join(args.masks_dir, d))]
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
