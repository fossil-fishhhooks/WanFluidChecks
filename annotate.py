"""
annotate.py

Manual click-annotation tool for SAM2 prompts.
Opens frame 1 of a clip, lets you click points to mark liquid regions.

Controls:
  1 -> switch to "source" label (liquid still in source container/cup)
  2 -> switch to "stream" label (liquid in free-fall / mid-air pour)
  3 -> switch to "pool" label (liquid that has landed / is pooling)
  left click -> add a positive point for the current label
  right click -> add a negative point for the current label (marks background)
  u -> undo last point
  s -> save and move to next clip
  q -> quit without saving current clip

Usage:
  python annotate.py --clips_dir ./clips --out prompts.json

Produces a JSON file like:
{
  "clip_001.mp4": {
    "source": {"points": [[120, 340]], "labels": [1]},
    "stream": {"points": [[200, 150], [205, 200]], "labels": [1, 1]},
    "pool":   {"points": [[150, 420]], "labels": [1]}
  },
  ...
}

Each region gets its own SAM2 prompt so segment.py can track them as
separate mask IDs (this matters for the volume-conservation check, which
needs source/pool as distinct regions, and the gravity check, which only
looks at "stream").
"""

import argparse
import json
import os

import cv2
import numpy as np

LABELS = ["source", "stream", "pool"]
LABEL_COLORS = {
    "source": (0, 200, 255),   # orange
    "stream": (0, 255, 0),     # green
    "pool": (255, 100, 0),     # blue-ish
}


def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame from {video_path}")
    return frame


class Annotator:
    def __init__(self, frame):
        self.frame = frame
        self.display = frame.copy()
        self.current_label = "source"
        # region -> {"points": [[x,y],...], "labels": [1 or 0, ...]}
        self.regions = {lbl: {"points": [], "labels": []} for lbl in LABELS}

    def redraw(self):
        self.display = self.frame.copy()
        for lbl in LABELS:
            color = LABEL_COLORS[lbl]
            for (x, y), pt_label in zip(
                self.regions[lbl]["points"], self.regions[lbl]["labels"]
            ):
                marker = cv2.MARKER_CROSS if pt_label == 1 else cv2.MARKER_TILTED_CROSS
                cv2.drawMarker(self.display, (x, y), color, markerType=marker,
                               markerSize=14, thickness=2)
        cv2.putText(
            self.display,
            f"label: {self.current_label}  (1=source 2=stream 3=pool, "
            f"L-click=+ R-click=- , u=undo, s=save, q=quit)",
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.regions[self.current_label]["points"].append([x, y])
            self.regions[self.current_label]["labels"].append(1)
            self.redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.regions[self.current_label]["points"].append([x, y])
            self.regions[self.current_label]["labels"].append(0)
            self.redraw()

    def undo(self):
        pts = self.regions[self.current_label]["points"]
        lbls = self.regions[self.current_label]["labels"]
        if pts:
            pts.pop()
            lbls.pop()
            self.redraw()

    def run(self, window_name):
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.on_mouse)
        self.redraw()
        while True:
            cv2.imshow(window_name, self.display)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("1"):
                self.current_label = "source"
                self.redraw()
            elif key == ord("2"):
                self.current_label = "stream"
                self.redraw()
            elif key == ord("3"):
                self.current_label = "pool"
                self.redraw()
            elif key == ord("u"):
                self.undo()
            elif key == ord("s"):
                return "save"
            elif key == ord("q"):
                return "quit"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_dir", required=True, help="Directory of .mp4 clips")
    parser.add_argument("--out", default="prompts.json", help="Output JSON path")
    args = parser.parse_args()

    clip_files = sorted(
        f for f in os.listdir(args.clips_dir)
        if f.lower().endswith((".mp4", ".mov", ".avi"))
    )
    if not clip_files:
        print(f"No video files found in {args.clips_dir}")
        return

    # resume support: load existing prompts if present
    all_prompts = {}
    if os.path.exists(args.out):
        with open(args.out, "r") as f:
            all_prompts = json.load(f)

    for clip_name in clip_files:
        if clip_name in all_prompts:
            print(f"Skipping {clip_name} (already annotated)")
            continue

        clip_path = os.path.join(args.clips_dir, clip_name)
        frame = get_first_frame(clip_path)
        annotator = Annotator(frame)
        action = annotator.run(window_name=clip_name)
        cv2.destroyWindow(clip_name)

        if action == "quit":
            print("Quitting without saving current clip.")
            break

        # only keep regions that actually got at least one positive point
        clean_regions = {
            lbl: data for lbl, data in annotator.regions.items()
            if any(l == 1 for l in data["labels"])
        }
        all_prompts[clip_name] = clean_regions

        with open(args.out, "w") as f:
            json.dump(all_prompts, f, indent=2)
        print(f"Saved prompts for {clip_name} -> {args.out}")

    print("Done.")


if __name__ == "__main__":
    main()
