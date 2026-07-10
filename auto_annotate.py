"""
auto_annotate.py

Fully automatic region annotation: a drop-in replacement for the manual
annotate.py. Writes the same prompts.json (v2 format), so segment.py /
flow.py / score_clip.py run unchanged afterwards.

How it works (no new dependencies -- uses SAM2's own automatic mask
generator with the checkpoint you already have):

  1. KEYFRAME SEARCH: run SAM2 AMG on a few frames (25%, 50%, 75% of
     the clip) and keep the frame whose candidates best fit the roles.
  2. ROLE CLASSIFICATION by pour-scene geometry:
       stream -- tall, thin, roughly vertical, horizontally central
       pool   -- wide, low in the frame, below the stream's bottom
       source -- sits above the stream's top edge
  3. PROMPT SYNTHESIS: interior points of each chosen mask (distance-
     transform peaks) become SAM2 point prompts at that frame index.
  4. CONFIDENCE REPORT: per-clip role scores are printed and stored in
     the JSON (as "_auto" metadata, ignored by segment.py). Clips below
     --min_confidence are listed at the end for manual review with
     annotate.py -- automation you can audit, not blind trust.

Heuristics are pour/faucet oriented; fountains and exotic scenes will
classify poorly (by design -- review those manually).

Usage:
  python auto_annotate.py --clips_dir ./clips --out prompts.json
  python auto_annotate.py --clips_dir ./clips --out prompts.json --redo
"""

import argparse
import json
import os

import cv2
import numpy as np

# reuse the checkpoint/config configured in segment.py
from segment import SAM2_CHECKPOINT, SAM2_CONFIG

KEYFRAME_FRACS = (0.25, 0.5, 0.75)


# ------------------------- SAM2 AMG loading -------------------------

def load_amg(with_semantic=True):
    """Builds the model bundle: AMG proposals + SAM2 image predictor for
    box-seeded masks + optional OWLv2 semantic detector. (Name kept for
    backward compatibility with score_viewer.py.)"""
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as e:
        raise ImportError(
            "Could not import sam2 -- is it installed?") from e
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    bundle = {
        "amg": SAM2AutomaticMaskGenerator(
            model,
            points_per_side=24,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.8,
            min_mask_region_area=80,
        ),
        "img": SAM2ImagePredictor(model),
        "sem": load_semantic() if with_semantic else None,
    }
    return bundle


def box_seeded_candidates(img_predictor, frame_rgb, detections,
                          min_det_score=0.2):
    """Feed the best detection box per role to SAM2's image predictor.
    Recovers candidates AMG never proposed (transparent real-water
    streams are the common case). Seeded masks still pass through the
    same geometry/motion/affinity gates as everything else."""
    import torch
    best_per_role = {}
    for d in detections:
        if d["role"] == "fixture" or d["score"] < min_det_score:
            continue
        cur = best_per_role.get(d["role"])
        if cur is None or d["score"] > cur["score"]:
            best_per_role[d["role"]] = d
    if not best_per_role:
        return []
    img_predictor.set_image(frame_rgb)
    out = []
    with torch.inference_mode():
        for role, d in best_per_role.items():
            masks, scores, _ = img_predictor.predict(
                box=np.array(d["box"], dtype=np.float32),
                multimask_output=False)
            seg = np.asarray(masks[0]).astype(bool)
            if seg.sum() >= 50:
                out.append({"segmentation": seg})
    return out


def load_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


# ------------------------- semantic layer (OWLv2) -------------------------
# Optional zero-shot object detection. Heuristics cannot distinguish a
# chrome faucet from water (it reflects the moving water -> passes the
# dynamics gate; it is bright -> passes the darkness gate; chrome is
# achromatic like water -> passes color affinity). Telling "faucet" from
# "water" is semantics, so: detect fixtures explicitly and VETO them,
# detect liquid concepts and BOOST matching candidates, and seed SAM2
# with detection boxes so transparent real-water streams that AMG never
# proposes still become candidates.

ROLE_QUERIES = {
    "stream": ["a stream of water pouring", "falling water"],
    "pool": ["water in a glass", "water in a sink basin",
             "a puddle of liquid"],
    "source": ["a cup pouring liquid", "a bottle pouring liquid",
               "a tilted glass of water"],
}
FIXTURE_QUERIES = ["a metal faucet", "a tap fixture", "a sink drain",
                   "a shower head"]
_ALL_QUERIES = [q for qs in ROLE_QUERIES.values() for q in qs] + FIXTURE_QUERIES
_QUERY_ROLE = {q: role for role, qs in ROLE_QUERIES.items() for q in qs}


def load_semantic():
    """Returns a zero-shot detector callable, or None if unavailable."""
    try:
        import torch
        from transformers import pipeline
        det = pipeline("zero-shot-object-detection",
                       model="google/owlv2-base-patch16-ensemble",
                       device=0 if torch.cuda.is_available() else -1)
        return det
    except Exception as e:
        import traceback
        print(f"  (semantic layer unavailable -- heuristics only: {e})")
        traceback.print_exc()
        return None


def semantic_detect(detector, frame_rgb, threshold=0.15):
    """Run OWLv2 on one frame. Returns list of dicts:
    {box: (x0,y0,x1,y1), score, label, role or 'fixture'}"""
    from PIL import Image
    pil = Image.fromarray(frame_rgb)
    raw = detector(pil, candidate_labels=_ALL_QUERIES, threshold=threshold)
    out = []
    for d in raw:
        b = d["box"]
        label = d["label"]
        out.append({
            "box": (b["xmin"], b["ymin"], b["xmax"], b["ymax"]),
            "score": float(d["score"]),
            "label": label,
            "role": _QUERY_ROLE.get(label, "fixture"),
        })
    return out


def _mask_box_overlap(seg, box):
    """Fraction of the mask's pixels inside the box."""
    x0, y0, x1, y1 = [int(v) for v in box]
    total = seg.sum()
    if total == 0:
        return 0.0
    inside = seg[max(0, y0):max(0, y1), max(0, x0):max(0, x1)].sum()
    return float(inside) / float(total)


# ---------------------- role scoring heuristics ----------------------

def _clamp01(x):
    return float(max(0.0, min(1.0, x)))


def candidate_features(seg, H, W):
    ys, xs = np.nonzero(seg)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)
    return {
        "area_frac": len(ys) / (H * W),
        "cx": (x0 + x1) / 2 / W, "cy": (y0 + y1) / 2 / H,
        "top": y0 / H, "bottom": y1 / H,
        "w_frac": bw / W, "h_frac": bh / H,
        "aspect": bh / max(bw, 1),          # tall > 1
        "fill": len(ys) / (bw * bh),        # how solid the bbox is
    }


def score_stream(f):
    if f["area_frac"] > 0.25 or f["w_frac"] > 0.30:
        return 0.0
    tall = _clamp01((f["aspect"] - 1.5) / 4.0)        # want >= ~5x
    span = _clamp01((f["h_frac"] - 0.12) / 0.35)      # meaningful vertical run
    central = _clamp01(1.0 - abs(f["cx"] - 0.5) * 2.0)
    thin = _clamp01((0.30 - f["w_frac"]) / 0.30)
    return tall * span * (0.4 + 0.6 * central) * (0.3 + 0.7 * thin)


def temporal_change_map(frames, n_pairs=8):
    """Mean absolute frame-to-frame grayscale change per pixel. Fluid
    being poured (rippling pool, sloshing/draining source) changes;
    rigid fixtures (faucets, drains, containers' rims) do not."""
    n = len(frames)
    idxs = np.linspace(0, n - 2, min(n_pairs, n - 1)).astype(int)
    acc = None
    for i in idxs:
        g0 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        g1 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        d = np.abs(g1 - g0)
        acc = d if acc is None else acc + d
    return acc / max(len(idxs), 1)


def fluid_affinity(seg, change_map, lab_keyframe, stream_seg):
    """Multiplier in [~0.2, 1] distinguishing FLUID from FIXTURE:
      - dynamics: candidate should change over time noticeably more than
        the background (poured fluid moves; faucets/drains don't)
      - darkness: drain holes/shadows are much darker than the stream
      - color: same fluid as the stream -> similar chroma (lenient)
    """
    bg_change = float(np.median(change_map)) + 1e-6
    cand_change = float(np.median(change_map[seg]))
    ratio = cand_change / bg_change
    dynamics = _clamp01((ratio - 0.8) / 1.5)    # <=0.8x bg -> 0, >=2.3x -> 1

    med_c = np.median(lab_keyframe[seg], axis=0)
    med_s = np.median(lab_keyframe[stream_seg], axis=0)
    dark_ok = _clamp01((med_c[0] - 0.35 * med_s[0]) /
                       (0.65 * med_s[0] + 1e-6))
    d_ab = float(np.hypot(med_c[1] - med_s[1], med_c[2] - med_s[2]))
    color_sim = float(np.exp(-d_ab / 25.0))

    return (0.2 + 0.8 * dynamics) * (0.5 + 0.5 * dark_ok) \
        * (0.6 + 0.4 * color_sim)


def _stream_endpoint(seg, which, band_px=4):
    """(x, y) of the stream's top or bottom end (median x of the extreme
    rows -- robust to ragged mask edges)."""
    ys, xs = np.nonzero(seg)
    if which == "top":
        y = ys.min()
        sel = ys <= y + band_px
    else:
        y = ys.max()
        sel = ys >= y - band_px
    return float(np.median(xs[sel])), float(y)


def _min_dist_to_point(seg, px, py, stride=3):
    ys, xs = np.nonzero(seg)
    ys, xs = ys[::stride], xs[::stride]
    return float(np.min(np.hypot(xs - px, ys - py)))


def score_pool(f, seg, stream_seg, H, W):
    """Pool = the liquid body at the stream's BOTTOM endpoint: close to
    it, mostly at/below its level, wider than tall."""
    if f["area_frac"] < 0.003 or f["area_frac"] > 0.35:
        return 0.0
    if f["aspect"] > 1.2:            # taller than wide -> not a pool
        return 0.0
    wide = _clamp01((1.0 / max(f["aspect"], 1e-3) - 1.0) / 2.0)
    if stream_seg is None:
        low = _clamp01((f["cy"] - 0.45) / 0.35)
        return 0.4 * low * (0.3 + 0.7 * wide)      # weak fallback
    bx, by = _stream_endpoint(stream_seg, "bottom")
    diag = np.hypot(H, W)
    prox = _clamp01(1.0 - _min_dist_to_point(seg, bx, by) / (0.12 * diag))
    ys, _ = np.nonzero(seg)
    below_frac = float(np.mean(ys >= by - 0.04 * H))
    return prox * (0.3 + 0.7 * below_frac) * (0.4 + 0.6 * wide)


def score_source(f, seg, stream_seg, H, W):
    """Source = whatever the stream's TOP endpoint hangs from: close to
    it, mostly above it."""
    if f["area_frac"] < 0.002 or f["area_frac"] > 0.30:
        return 0.0
    if stream_seg is None:
        return 0.4 * _clamp01((0.65 - f["cy"]) / 0.5)   # weak fallback
    tx, ty = _stream_endpoint(stream_seg, "top")
    diag = np.hypot(H, W)
    prox = _clamp01(1.0 - _min_dist_to_point(seg, tx, ty) / (0.12 * diag))
    ys, _ = np.nonzero(seg)
    above_frac = float(np.mean(ys <= ty + 0.04 * H))
    x_align = _clamp01(1.0 - abs(f["cx"] - tx / W) / 0.30)
    return prox * (0.3 + 0.7 * above_frac) * (0.4 + 0.6 * x_align)


def stream_dynamics(seg, change_map, hard_floor=1.1):
    """Motion gate for stream candidates: a real stream shows temporal
    texture change (shimmer, scrolling highlights, droplets) in its
    INTERIOR. Static tall-thin objects (poles, frames, table legs) do
    not -- and under slight camera drift they only change at their
    high-contrast EDGES, so we erode the mask and measure interior
    change only, normalized against the background median.

    Returns 0.0 (ineligible) below hard_floor x background, else a
    continuous [0,1] bonus. Hard gate by design: a false-positive
    stream silently corrupts source/pool anchoring and every physics
    score; a false negative just routes the clip to manual review.
    """
    interior = cv2.erode(seg.astype(np.uint8),
                         np.ones((3, 3), np.uint8), 1).astype(bool)
    if interior.sum() < 30:
        interior = seg
    bg = float(np.median(change_map)) + 1e-6
    ratio = float(np.median(change_map[interior])) / bg
    if ratio < hard_floor:
        return 0.0
    return _clamp01((ratio - hard_floor) / 1.2)


def classify_candidates(cands, H, W, frame_bgr=None, change_map=None,
                        detections=None):
    """Greedy role assignment; returns {role: (seg, score)} and confidence.

    Stream is found first (most distinctive); source and pool are then
    scored primarily by proximity to the stream's top/bottom endpoints.
    Roles below ACCEPT_THRESH are simply omitted -- clips with no
    visible source or pool get none, rather than a forced bad match
    (the scoring pipeline handles missing regions fine).
    """
    ACCEPT_STREAM = 0.05
    ACCEPT_OTHER = 0.18

    feats = []
    for c in cands:
        seg = c["segmentation"]
        if seg.sum() < 50:
            continue
        feats.append((seg, candidate_features(seg, H, W)))

    fixture_boxes = [d for d in (detections or []) if d["role"] == "fixture"]
    role_boxes = {}
    for d in (detections or []):
        if d["role"] != "fixture":
            role_boxes.setdefault(d["role"], []).append(d)

    def fixture_overlap(seg):
        return max((_mask_box_overlap(seg, d["box"]) for d in fixture_boxes),
                   default=0.0)

    def role_boost(seg, role):
        """Multiplier > 1 when a candidate overlaps a matching detection."""
        best = 0.0
        for d in role_boxes.get(role, []):
            best = max(best, _mask_box_overlap(seg, d["box"]) * d["score"])
        return 1.0 + 1.5 * best

    out = {}
    stream_scored = []
    for seg, f in feats:
        g = score_stream(f)
        if g <= 0:
            continue
        if change_map is not None:
            dyn = stream_dynamics(seg, change_map)
            if dyn <= 0.0:
                continue           # static object -- not a stream, period
            g = g * (0.4 + 0.6 * dyn)
        g_raw = g
        g = min(1.0, g * role_boost(seg, "stream"))
        stream_scored.append((g, g_raw, seg, f))
    stream_scored.sort(reverse=True, key=lambda t: t[0])
    stream_seg = None
    if stream_scored and stream_scored[0][0] > ACCEPT_STREAM:
        s, s_raw, seg, f = stream_scored[0]
        out["stream"] = (seg, s, s_raw)
        stream_seg = seg

    used = [out[r][0] for r in out]

    def overlaps_used(seg):
        return any((seg & u).sum() > 0.4 * min(seg.sum(), u.sum())
                   for u in used)

    lab = None
    if frame_bgr is not None and stream_seg is not None:
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)

    def affinity(seg):
        """Fluid-vs-fixture multiplier; 1.0 when signals unavailable."""
        if lab is None or change_map is None:
            return 1.0
        return fluid_affinity(seg, change_map, lab, stream_seg)

    for role, scorer in (
            ("pool", lambda seg, f: score_pool(f, seg, stream_seg, H, W)),
            ("source", lambda seg, f: score_source(f, seg, stream_seg, H, W))):
        pool_src = []
        for seg, f in feats:
            if overlaps_used(seg):
                continue
            if fixture_overlap(seg) > 0.35:
                continue          # it's a faucet/tap/drain -- never fluid
            s_raw = scorer(seg, f) * affinity(seg)
            s = min(1.0, s_raw * role_boost(seg, role))
            pool_src.append((s, s_raw, seg))
        scored = sorted(pool_src, reverse=True, key=lambda t: t[0])
        if scored and scored[0][0] > ACCEPT_OTHER:
            out[role] = (scored[0][2], scored[0][0], scored[0][1])
            used.append(scored[0][2])

    # confidence: stream-weighted mean over FOUND roles only, so a clip
    # that legitimately has no pool/source is not penalized for it
    if not out:
        return out, 0.0
    # per-keyframe confidence from RAW scores: semantic boosts help pick
    # the right candidate, but only geometry/motion evidence attests to
    # mask quality -- boosted confidence was promoting blobs past the
    # strict-mode gate
    weights = {"stream": 0.6, "pool": 0.2, "source": 0.2}
    w_sum = sum(weights[r] for r in out)
    total = float(sum(weights[r] * v[2] for r, v in out.items()) / w_sum)
    return out, total


# ----------------------- prompt point sampling -----------------------

def mask_to_points(seg, n_points=3, min_sep_frac=0.25):
    dist = cv2.distanceTransform(seg.astype(np.uint8), cv2.DIST_L2, 5)
    pts = []
    d = dist.copy()
    h, w = seg.shape
    min_sep = max(4, int(min_sep_frac * np.sqrt(seg.sum())))
    for _ in range(n_points):
        y, x = np.unravel_index(np.argmax(d), d.shape)
        if d[y, x] <= 1.0:
            break
        pts.append([int(x), int(y)])
        y0, y1 = max(0, y - min_sep), min(h, y + min_sep)
        x0, x1 = max(0, x - min_sep), min(w, x + min_sep)
        d[y0:y1, x0:x1] = 0
    return pts


# ------------------------------ driver ------------------------------

def _mask_iou(a, b):
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / float(union) if union else 0.0


def auto_annotate_clip(models, frames):
    """Scan several keyframes; each role independently keeps its best
    (frame, mask, score).

    Confidence is built from RAW geometry/motion scores (semantic boosts
    affect selection only) and multiplied by CROSS-KEYFRAME STABILITY:
    an established stream is spatially static, so a trustworthy stream
    detection finds roughly the same region at every keyframe (high
    mutual IoU), while a flaky one jumps around. This is deliberately
    independent of the physics being scored -- selecting masks by e.g.
    how well they taper would rig the gravity metric.
    """
    if not isinstance(models, dict):
        models = {"amg": models, "img": None, "sem": None}
    amg, img_pred, sem = models["amg"], models.get("img"), models.get("sem")

    H, W = frames[0].shape[:2]
    change_map = temporal_change_map(frames)

    best = {}             # role -> (boosted, raw, seg, frame_idx)
    stream_per_kf = []    # stream seg found at each keyframe
    for frac in KEYFRAME_FRACS:
        idx = min(int(frac * len(frames)), len(frames) - 1)
        rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        cands = amg.generate(rgb)
        detections = None
        if sem is not None:
            detections = semantic_detect(sem, rgb)
            if img_pred is not None and detections:
                cands = cands + box_seeded_candidates(img_pred, rgb,
                                                      detections)
        assignment, _ = classify_candidates(
            cands, H, W, frame_bgr=frames[idx], change_map=change_map,
            detections=detections)
        if "stream" in assignment:
            stream_per_kf.append(assignment["stream"][0])
        for role, (seg, s, s_raw) in assignment.items():
            if role not in best or s > best[role][0]:
                best[role] = (s, s_raw, seg, idx)

    if "stream" not in best:
        return None, 0.0, 0
    stream_idx = best["stream"][3]

    # cross-keyframe stability of the stream detection
    if len(stream_per_kf) >= 2:
        ref = best["stream"][2]
        ious = [_mask_iou(ref, m) for m in stream_per_kf
                if m is not ref]
        stability = float(np.mean(ious)) if ious else 0.0
    else:
        stability = None   # only one keyframe found a stream

    regions = {}
    for role, (s, s_raw, seg, idx) in best.items():
        pts = mask_to_points(seg, n_points=4 if role == "stream" else 3)
        if not pts:
            continue
        regions[role] = {
            "frame_idx": int(idx),
            "points": pts,
            "labels": [1] * len(pts),
        }

    weights = {"stream": 0.6, "pool": 0.2, "source": 0.2}
    found = [r for r in regions if r in weights]
    w_sum = sum(weights[r] for r in found)
    conf = float(sum(weights[r] * best[r][1] for r in found) / w_sum) \
        if found else 0.0
    if stability is None:
        conf *= 0.75      # single-keyframe detection: less corroborated
    else:
        conf *= (0.4 + 0.6 * stability)

    if regions:
        regions["_auto"] = {
            "role_scores_raw": {r: round(float(best[r][1]), 3)
                                for r in found},
            "role_frames": {r: int(best[r][3]) for r in found},
            "stream_stability": (None if stability is None
                                 else round(stability, 3)),
            "confidence": round(conf, 3),
            "semantic": sem is not None,
        }
    return regions, conf, stream_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_dir", required=True)
    parser.add_argument("--out", default="prompts.json")
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--min_confidence", type=float, default=0.25,
                        help="clips below this are flagged for manual review")
    parser.add_argument("--no_semantic", action="store_true",
                        help="disable the OWLv2 semantic layer "
                             "(heuristics only)")
    parser.add_argument("--keep_low_confidence", action="store_true",
                        help="save prompts even below min_confidence "
                             "(default: low-confidence clips are NOT "
                             "written, so they stay available for manual "
                             "annotation instead of silently producing "
                             "bad masks)")
    args = parser.parse_args()

    clip_files = sorted(f for f in os.listdir(args.clips_dir)
                        if f.lower().endswith((".mp4", ".mov", ".avi")))
    all_prompts = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            all_prompts = json.load(f)

    amg = load_amg(with_semantic=not args.no_semantic)
    review = []

    for clip_name in clip_files:
        if clip_name in all_prompts and not args.redo:
            print(f"Skipping {clip_name} (already annotated)")
            continue
        print(f"Auto-annotating {clip_name} ...")
        frames = load_all_frames(os.path.join(args.clips_dir, clip_name))
        if not frames:
            print("  could not read frames, skipping")
            continue
        regions, conf, idx = auto_annotate_clip(amg, frames)
        if not regions:
            print("  NO STREAM CANDIDATE FOUND -- needs manual annotation")
            review.append((clip_name, 0.0))
            continue
        roles = [r for r in regions if r != "_auto"]
        print(f"  keyframe {idx}: {', '.join(roles)} "
              f"(confidence {conf:.2f})")
        if conf < args.min_confidence:
            review.append((clip_name, conf))
            if not args.keep_low_confidence:
                print("  LOW CONFIDENCE -- NOT saved (annotate manually, "
                      "or rerun with --keep_low_confidence)")
                continue
            print("  LOW CONFIDENCE -- saved anyway (--keep_low_confidence)")
        all_prompts[clip_name] = regions
        with open(args.out, "w") as f:
            json.dump(all_prompts, f, indent=2)

    if review:
        print("\nClips to review manually (annotate.py --redo):")
        for name, conf in review:
            print(f"  {name}  (confidence {conf:.2f})")
    print("Done.")


if __name__ == "__main__":
    main()
