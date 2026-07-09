"""
annotate.py  (v2)

Manual click-annotation tool for SAM2 prompts -- now with frame scrubbing,
so each region can be annotated on WHICHEVER frame it first clearly
appears (the pour may not start at frame 1; the pool may form late).

Controls:
  a / d        -> step one frame back / forward
  A / D        -> jump 5 frames back / forward
  (trackbar also scrubs)
  1 / 2 / 3    -> switch label: source / stream / pool
  left click   -> add positive point for current label ON CURRENT FRAME
  right click  -> add negative (background) point for current label
  u            -> undo last point for current label
  s            -> save this clip's prompts, next clip
  q            -> quit without saving current clip

Each region records the frame it was annotated on; segment.py propagates
masks BOTH forward and backward from that frame, so annotating the
clearest frame mid-clip covers the whole video.

Constraint: all points for one region must be on ONE frame (SAM2 takes
the prompt at a single frame and tracks from there). Clicking the same
region on a new frame moves that region's points to the new frame after
a console warning. Different regions may use different frames.

Usage:
  python annotate.py --clips_dir ./clips --out prompts.json

Output format (v2 -- segment.py also accepts the old v1 format,
treating it as frame_idx 0):
{
  "clip_001.mp4": {
    "stream": {"frame_idx": 12, "points": [[200,150]], "labels": [1]},
    "pool":   {"frame_idx": 25, "points": [[150,420]], "labels": [1]}
  }
}
"""

import argparse
import json
import os

import cv2
import numpy as np

LABELS = ["source", "stream", "pool"]
LABEL_KEYS = {ord("1"): "source", ord("2"): "stream", ord("3"): "pool"}
LABEL_COLORS = {
    "source": (0, 200, 255),   # orange
    "stream": (0, 255, 0),     # green
    "pool": (255, 100, 0),     # blue-ish
}


def load_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"Could not read frames from {video_path}")
    return frames


class Annotator:
    def __init__(self, frames, window_name):
        self.frames = frames
        self.n = len(frames)
        self.idx = 0
        self.window = window_name
        self.current_label = "stream"
        # region -> {"frame_idx": int, "points": [[x,y],...], "labels": [...]}
        self.regions = {}

    # ---------- drawing ----------
    def redraw(self):
        disp = self.frames[self.idx].copy()
        for lbl, data in self.regions.items():
            color = LABEL_COLORS[lbl]
            on_this_frame = data["frame_idx"] == self.idx
            for (x, y), pl in zip(data["points"], data["labels"]):
                marker = cv2.MARKER_CROSS if pl == 1 else cv2.MARKER_TILTED_CROSS
                # dim markers that live on a different frame
                c = color if on_this_frame else tuple(int(v * 0.35) for v in color)
                cv2.drawMarker(disp, (x, y), c, markerType=marker,
                               markerSize=14, thickness=2)
        status = " | ".join(
            f"{lbl}@f{d['frame_idx']}({sum(1 for l in d['labels'] if l==1)}pt)"
            for lbl, d in self.regions.items()) or "no regions yet"
        cv2.putText(disp, f"frame {self.idx+1}/{self.n}  label:{self.current_label}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(disp, status,
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(disp, "a/d scrub  1/2/3 label  Lclick + Rclick -  u undo  s save  q quit",
                    (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        self.display = disp

    # ---------- events ----------
    def on_mouse(self, event, x, y, flags, param):
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        pt_label = 1 if event == cv2.EVENT_LBUTTONDOWN else 0
        lbl = self.current_label
        if lbl not in self.regions:
            self.regions[lbl] = {"frame_idx": self.idx, "points": [], "labels": []}
        elif self.regions[lbl]["frame_idx"] != self.idx:
            print(f"  [{lbl}] moving points from frame "
                  f"{self.regions[lbl]['frame_idx']} to frame {self.idx} "
                  f"(one frame per region)")
            self.regions[lbl] = {"frame_idx": self.idx, "points": [], "labels": []}
        self.regions[lbl]["points"].append([x, y])
        self.regions[lbl]["labels"].append(pt_label)
        self.redraw()

    def on_trackbar(self, pos):
        self.idx = int(np.clip(pos, 0, self.n - 1))
        self.redraw()

    def step(self, delta):
        self.idx = int(np.clip(self.idx + delta, 0, self.n - 1))
        cv2.setTrackbarPos("frame", self.window, self.idx)
        self.redraw()

    def undo(self):
        lbl = self.current_label
        if lbl in self.regions and self.regions[lbl]["points"]:
            self.regions[lbl]["points"].pop()
            self.regions[lbl]["labels"].pop()
            if not self.regions[lbl]["points"]:
                del self.regions[lbl]
            self.redraw()

    # ---------- main loop ----------
    def run(self):
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.on_mouse)
        cv2.createTrackbar("frame", self.window, 0, self.n - 1, self.on_trackbar)
        self.redraw()
        while True:
            cv2.imshow(self.window, self.display)
            key = cv2.waitKey(20) & 0xFF
            if key in LABEL_KEYS:
                self.current_label = LABEL_KEYS[key]
                self.redraw()
            elif key == ord("a"):
                self.step(-1)
            elif key == ord("d"):
                self.step(+1)
            elif key == ord("A"):
                self.step(-5)
            elif key == ord("D"):
                self.step(+5)
            elif key == ord("u"):
                self.undo()
            elif key == ord("s"):
                return "save"
            elif key == ord("q"):
                return "quit"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_dir", required=True)
    parser.add_argument("--out", default="prompts.json")
    parser.add_argument("--redo", action="store_true",
                        help="Re-annotate clips already present in the output JSON")
    args = parser.parse_args()

    clip_files = sorted(
        f for f in os.listdir(args.clips_dir)
        if f.lower().endswith((".mp4", ".mov", ".avi"))
    )
    if not clip_files:
        print(f"No video files found in {args.clips_dir}")
        return

    all_prompts = {}
    if os.path.exists(args.out):
        with open(args.out, "r") as f:
            all_prompts = json.load(f)

    for clip_name in clip_files:
        if clip_name in all_prompts and not args.redo:
            print(f"Skipping {clip_name} (already annotated; use --redo to overwrite)")
            continue

        clip_path = os.path.join(args.clips_dir, clip_name)
        print(f"\nAnnotating {clip_name} -- scrub to the frame where each "
              f"region is clearest before clicking")
        frames = load_all_frames(clip_path)
        annotator = Annotator(frames, window_name=clip_name)
        action = annotator.run()
        cv2.destroyWindow(clip_name)

        if action == "quit":
            print("Quitting without saving current clip.")
            break

        clean = {lbl: d for lbl, d in annotator.regions.items()
                 if any(l == 1 for l in d["labels"])}
        if not clean:
            print(f"  no positive points for {clip_name}, not saving")
            continue
        all_prompts[clip_name] = clean

        with open(args.out, "w") as f:
            json.dump(all_prompts, f, indent=2)
        print(f"Saved prompts for {clip_name} -> {args.out}")

    print("Done.")


if __name__ == "__main__":
    main()
