"""
run_all.py

One-command orchestration of the full pipeline:

  clips -> auto_annotate -> segment -> flow -> score

Every stage already has resume/skip semantics, so rerunning after
adding new clips only processes what's new. Auto-annotation prints a
manual-review list at the end; re-annotate those with
`python annotate.py --clips_dir <dir> --out <prompts> --redo` and rerun
this script -- segment.py will pick up the changed prompts with --redo.

Usage:
  python run_all.py --clips_dir ./clips
  python run_all.py --clips_dir ./clips_real --workspace ./real  (separate real-footage run)
  python run_all.py --clips_dir ./clips --manual                 (click annotation instead)
"""

import argparse
import os
import subprocess
import sys


def run(step_name, cmd):
    print(f"\n{'='*60}\n{step_name}\n{'='*60}")
    result = subprocess.run([sys.executable] + cmd)
    if result.returncode != 0:
        print(f"\n{step_name} FAILED (exit {result.returncode}) -- stopping.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_dir", required=True)
    parser.add_argument("--workspace", default=".",
                        help="parent dir for prompts/masks/flow/results "
                             "(use separate workspaces for real vs generated)")
    parser.add_argument("--manual", action="store_true",
                        help="use click-based annotate.py instead of auto")
    parser.add_argument("--skip_flow", action="store_true")
    args = parser.parse_args()

    ws = args.workspace
    os.makedirs(ws, exist_ok=True)
    prompts = os.path.join(ws, "prompts.json")
    masks = os.path.join(ws, "masks")
    flow = os.path.join(ws, "flow")
    results = os.path.join(ws, "results")

    if args.manual:
        run("1/4 annotate (manual)",
            ["annotate.py", "--clips_dir", args.clips_dir, "--out", prompts])
    else:
        run("1/4 auto-annotate",
            ["auto_annotate.py", "--clips_dir", args.clips_dir,
             "--out", prompts])

    run("2/4 segment (SAM2)",
        ["segment.py", "--clips_dir", args.clips_dir,
         "--prompts", prompts, "--out_dir", masks])

    if not args.skip_flow:
        run("3/4 optical flow (RAFT)",
            ["flow.py", "--clips_dir", args.clips_dir,
             "--masks_dir", masks, "--out_dir", flow])

    score_cmd = ["score_clip.py", "--masks_dir", masks,
                 "--clips_dir", args.clips_dir, "--out_dir", results]
    if not args.skip_flow:
        score_cmd += ["--flow_dir", flow]
    run("4/4 score", score_cmd)

    print(f"\nAll done. Summary: {os.path.join(results, '_summary.csv')}")
    print(f"Inspect visually: python score_viewer.py "
          f"--clips_dir {args.clips_dir} --masks_dir {masks} "
          f"--flow_dir {flow} --results_dir {results} --prompts {prompts}")


if __name__ == "__main__":
    main()
