"""
checks.py  (v2 -- continuous scoring, graceful degradation)

Design goals after v1 produced degenerate output (volume 1.0 everywhere,
gravity NA everywhere):

  1. CONTINUOUS scores, not thresholded pass/fail -- every clip lands
     somewhere in (0, 1), so clips are rankable even when none commits
     a flagrant violation.
  2. GRACEFUL DEGRADATION -- gravity is built from three components
     with different data requirements; the overall score uses whatever
     is computable and reports why the rest wasn't. All-NA should now
     only happen when the stream mask is essentially empty.
  3. DIAGNOSTICS FIRST-CLASS -- every result carries 'notes' explaining
     data quality (mask coverage, flow magnitude), surfaced by
     score_clip on the console.

Gravity components:
  A. direction  -- magnitude-weighted fraction of stream flow going
                   downward. Needs only: some stream mask + flow with
                   motion above the noise floor. Works for established
                   steady pours (no advancing phase needed).
  B. speedup    -- v^2 vs y linearity within the stream (free-fall
                   kinematics: v^2 = v0^2 + 2*a*dy). Needs a tall-enough
                   stream with measurable flow. Adaptive binning.
  C. leading edge -- quadratic fit to the advancing stream front.
                   Only applicable when the clip captures pour onset.

Volume: continuous score from the distribution of deviations off a
median-smoothed area trend, plus a dropout penalty for regions that
vanish mid-clip.
"""

import numpy as np


# ----------------------------- helpers -----------------------------

def _median_smooth(x, k=5):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return x
    half = k // 2
    return np.array([np.median(x[max(0, i - half):min(n, i + half + 1)])
                     for i in range(n)])


def _squash(x, tau):
    """exp decay: 0 -> 1.0, tau -> ~0.37, 3*tau -> ~0.05"""
    return float(np.exp(-max(0.0, x) / tau))


# --------------------------- volume check ---------------------------

def volume_check(region_masks_by_frame, smooth_k=5,
                 jitter_allowance=0.02, tau=0.12):
    """
    Continuous conservation score.

    Signal: per-frame total liquid area vs its median-smoothed trend.
    p90 of the relative deviation (minus a small jitter allowance for
    segmentation noise) is squashed through exp(-x/tau) -> (0, 1].

    Dropout penalty: frames where total area collapses below 10% of the
    running trend while the trend itself is substantial (liquid
    vanishing outright) multiply the score down hard.
    """
    region_names = list(region_masks_by_frame.keys())
    n_frames = len(region_masks_by_frame[region_names[0]])
    notes = []

    total_area = np.array([
        float(sum(region_masks_by_frame[r][i].sum() for r in region_names))
        for i in range(n_frames)
    ])

    if total_area.max() < 50:
        return {"total_area": total_area.tolist(), "score": None,
                "notes": ["total liquid mask is essentially empty -- "
                          "check segmentation overlay"]}

    smoothed = _median_smooth(total_area, k=smooth_k)
    denom = np.maximum(smoothed, 1.0)
    dev = np.abs(total_area - smoothed) / denom

    p90 = float(np.percentile(dev, 90))
    p_max = float(dev.max())
    base = _squash(p90 - jitter_allowance, tau)

    substantial = smoothed > 0.2 * smoothed.max()
    dropout = (total_area < 0.1 * smoothed) & substantial
    n_dropout = int(dropout.sum())
    if n_dropout:
        notes.append(f"{n_dropout} frame(s) where liquid nearly vanished")
    dropout_factor = float(max(0.0, 1.0 - 2.0 * n_dropout / n_frames))

    score = base * dropout_factor

    return {
        "total_area": total_area.tolist(),
        "smoothed_area": smoothed.tolist(),
        "deviation_p90": p90,
        "deviation_max": p_max,
        "dropout_frames": [int(i) for i in np.nonzero(dropout)[0]],
        "score": float(score),
        "notes": notes,
    }


# ------------------- background motion compensation -------------------

def compensate_background_motion(flow_by_frame, liquid_masks_by_frame,
                                 border=8, min_bg_pixels=100):
    """
    Subtract per-frame median background flow before analyzing stream
    motion.

    Why (found via real-footage calibration): RAFT produces dense flow
    by inpainting textureless regions with surrounding motion. A water
    stream's interior is textureless, so its reported flow is largely
    whatever the SCENE is doing -- camera drift in real footage, global
    "breathing" in generated clips. Uncompensated, that masquerades as
    coherent sideways/upward stream motion and poisons the direction
    and speedup checks identically on real and generated video.

    liquid_masks_by_frame: union of ALL liquid regions per frame
    (source+stream+pool) -- background = everything else, minus an
    image border where RAFT edge artifacts live.

    Returns (compensated_flows, drift_vectors).
    """
    out, drifts = [], []
    n = min(len(flow_by_frame), len(liquid_masks_by_frame))
    for i in range(n):
        flow = flow_by_frame[i]
        liquid = liquid_masks_by_frame[i]
        h, w = flow.shape[:2]
        bg = np.ones((h, w), dtype=bool)
        if liquid.shape == (h, w):
            bg &= ~liquid
        if border:
            bg[:border] = False
            bg[-border:] = False
            bg[:, :border] = False
            bg[:, -border:] = False
        if bg.sum() < min_bg_pixels:
            out.append(flow)
            drifts.append([0.0, 0.0])
            continue
        med = np.median(flow[bg], axis=0)
        out.append(flow - med[None, None, :])
        drifts.append([float(med[0]), float(med[1])])
    return out, drifts


# ---------------------- gravity component A: direction ----------------------

def gravity_direction(flow_by_frame, stream_masks_by_frame,
                      noise_floor=0.25, min_moving_frac=0.03,
                      min_coherence=0.3):
    """
    Coherence-gated flow direction check.

    KEY LIMITATION (learned from real-footage calibration): a steady
    laminar stream produces near-zero optical flow -- consecutive frames
    are nearly identical images even though the water moves fast. In
    that regime the only surviving flow is symmetric edge shimmer, which
    says nothing about gravity. So this check now returns NA (with a
    note) unless the flow inside the stream is BOTH substantial and
    directionally coherent; it only scores when there is real trackable
    motion (droplets, splash, turbulent texture, advancing front).

    Gates:
      - moving fraction: >= min_moving_frac of stream pixels above the
        noise floor, else NA ("stream static in image space")
      - coherence: |sum(v)| / sum(|v|) >= min_coherence, else NA
        ("flow present but incoherent / shimmer")

    When gated in: score = fraction of absolute stream motion that is
    vertical (up OR down), so a fountain arc with strong up-and-down
    motion scores well while purely horizontal drift scores 0.
    """
    sum_v = np.zeros(2, dtype=np.float64)
    abs_sum_v = np.zeros(2, dtype=np.float64)
    sum_mag = 0.0
    n_stream_px, n_moving_px = 0, 0
    mags_all = []
    n = min(len(flow_by_frame), len(stream_masks_by_frame))
    n_aligned = 0

    for i in range(n):
        flow, mask = flow_by_frame[i], stream_masks_by_frame[i]
        if mask.shape != flow.shape[:2] or not mask.any():
            continue
        n_aligned += 1
        v = flow[mask]
        mag = np.linalg.norm(v, axis=1)
        mags_all.append(mag)
        n_stream_px += len(mag)
        moving = mag > noise_floor
        n_moving_px += int(moving.sum())
        if moving.any():
            sum_v += v[moving].sum(axis=0)
            abs_sum_v += np.abs(v[moving]).sum(axis=0)
            sum_mag += float(mag[moving].sum())

    if n_aligned == 0:
        return {"score": None,
                "notes": ["no frames where stream mask aligned with flow "
                          "(empty masks or shape mismatch)"]}

    mean_mag = float(np.mean(np.concatenate(mags_all))) if mags_all else 0.0
    moving_frac = n_moving_px / max(n_stream_px, 1)

    if moving_frac < min_moving_frac or sum_mag <= 0:
        return {"mean_flow_magnitude": mean_mag,
                "moving_fraction": float(moving_frac), "score": None,
                "notes": [f"stream is static in image space (only "
                          f"{100*moving_frac:.1f}% of pixels moving, mean "
                          f"|v|={mean_mag:.3f} px/frame). Laminar streams "
                          f"and frozen streams are indistinguishable to "
                          f"optical flow -- taper check carries gravity "
                          f"scoring here"]}

    coherence = float(np.linalg.norm(sum_v) / sum_mag)
    if coherence < min_coherence:
        return {"mean_flow_magnitude": mean_mag,
                "moving_fraction": float(moving_frac),
                "coherence": coherence, "score": None,
                "notes": [f"stream flow is incoherent shimmer "
                          f"(coherence={coherence:.2f}) -- no usable "
                          f"direction signal"]}

    mean_dir = sum_v / np.linalg.norm(sum_v)
    total_abs = float(abs_sum_v[0] + abs_sum_v[1])
    if total_abs < 1e-10:
        vertical_frac = 0.0
    else:
        vertical_frac = float(abs_sum_v[1] / total_abs)
    score = vertical_frac

    return {
        "mean_flow_magnitude": mean_mag,
        "moving_fraction": float(moving_frac),
        "coherence": coherence,
        "mean_direction_xy": [float(mean_dir[0]), float(mean_dir[1])],
        "vertical_fraction": vertical_frac,
        "score": float(score),
        "notes": [] if vertical_frac > 0.4 else
                 ["flow is dominated by horizontal motion "
                  f"(vertical_fraction={vertical_frac:.2f}) -- no strong "
                  f"vertical component from gravity"],
    }


# ---------------------- gravity component B: speedup ----------------------

def gravity_speedup(flow_by_frame, stream_masks_by_frame,
                    noise_floor=0.25, min_bin_pixels=8, min_samples=5):
    """
    Free-fall kinematics under gravity: v^2 = v0^2 + 2*a*(y - y0).

    Unlike v1 (which assumed strictly downward flow), this version:
      - Accepts ANY significant vertical motion (up OR down)
      - Checks that the slope sign is consistent with gravity:
        * downward-moving parcels (mean vy > 0) should ACCELERATE
          (positive slope of v^2 vs y)
        * upward-moving parcels (mean vy < 0) should DECELERATE
          (negative slope of v^2 vs y)
      - A fountain's rising jet (decelerating against gravity) is
        therefore scored correctly instead of penalized.

    score = R^2 of the linear fit if slope sign is consistent with
    flow direction, else R^2 * 0.2.
    """
    samples_y, samples_v2 = [], []
    sum_vy, count_vy = 0.0, 0
    n = min(len(flow_by_frame), len(stream_masks_by_frame))

    for i in range(n):
        flow, mask = flow_by_frame[i], stream_masks_by_frame[i]
        if mask.shape != flow.shape[:2] or not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        vy = flow[ys, xs, 1]
        keep = np.abs(vy) > noise_floor
        if keep.sum() < min_bin_pixels * 2:
            continue
        ys_k, vy_k = ys[keep], vy[keep]
        sum_vy += float(vy_k.sum())
        count_vy += len(vy_k)

        height = ys_k.max() - ys_k.min()
        n_bins = int(np.clip(height // 12, 3, 16))
        if n_bins < 3:
            continue
        edges = np.linspace(ys_k.min(), ys_k.max() + 1, n_bins + 1)
        which = np.digitize(ys_k, edges) - 1
        for b in range(n_bins):
            sel = which == b
            if sel.sum() < min_bin_pixels:
                continue
            samples_y.append(0.5 * (edges[b] + edges[b + 1]))
            samples_v2.append(float(np.mean(vy_k[sel]) ** 2))

    if len(samples_y) < min_samples:
        return {"n_samples": len(samples_y), "score": None,
                "notes": [f"only {len(samples_y)} (y, v^2) samples "
                          f"(need >= {min_samples}) -- stream too "
                          f"short/thin or flow too weak"]}

    y_arr, v2_arr = np.array(samples_y), np.array(samples_v2)
    slope, intercept = np.polyfit(y_arr, v2_arr, 1)
    pred = slope * y_arr + intercept
    ss_res = float(np.sum((v2_arr - pred) ** 2))
    ss_tot = float(np.sum((v2_arr - v2_arr.mean()) ** 2))
    r2 = max(0.0, min(1.0, 1.0 - ss_res / ss_tot)) if ss_tot > 1e-9 else 0.0

    mean_vy = sum_vy / count_vy if count_vy > 0 else 0.0
    if mean_vy >= 0:
        slope_ok = bool(slope > 0)
    else:
        slope_ok = bool(slope < 0)
    score = r2 if slope_ok else r2 * 0.2

    notes = []
    if not slope_ok:
        notes.append(f"v^2 vs y slope has wrong sign (slope={slope:.2f}, "
                     f"mean_vy={mean_vy:.2f}) -- water "
                     f"{'accelerates upward' if mean_vy < 0 else 'slows while falling'}")

    return {
        "n_samples": len(samples_y),
        "slope_v2_vs_y": float(slope),
        "slope_sign_ok": slope_ok,
        "mean_vertical_velocity": float(mean_vy),
        "r_squared": float(r2),
        "score": float(score),
        "notes": notes,
    }


# ---------------------- gravity component T: taper ----------------------

def _spearman(x, y):
    """Rank correlation without scipy."""
    def rank(a):
        order = np.argsort(a)
        r = np.empty(len(a))
        r[order] = np.arange(len(a), dtype=np.float64)
        return r
    rx, ry = rank(np.asarray(x)), rank(np.asarray(y))
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 1e-12 else 0.0


def gravity_taper(stream_masks_by_frame, min_rows=10, trim_top=0.05,
                  trim_bottom=0.15, min_width=2, min_frames=3):
    """
    Geometric gravity check -- needs NO optical flow, so it works on
    laminar streams where motion is invisible to RAFT.

    Physics: mass continuity (A*v = const) + free fall
    (v^2 = v0^2 + 2*a*dy) force a falling stream to NARROW with fall
    distance; for a round jet, 1/w^4 is linear in y with positive slope.
    A frozen uniform-width column (a common video-model failure) or a
    stream that widens downward fails this.

    Method: per frame, width profile w(y) = pixel count per row inside
    the stream mask; aggregate across frames by median at each row;
    trim the top (spout attachment) and bottom (splash zone) of the
    extent. Score combines:
      - narrowing: Spearman rho of w vs y, mapped so rho=-1 -> 1,
        rho>=0 -> 0
      - power law: R^2 of the linear fit 1/w^4 vs y (slope must be +)

    score = 0.6 * narrowing + 0.4 * power_law
    """
    profiles = {}
    frames_used = 0
    for m in stream_masks_by_frame:
        if not m.any():
            continue
        ys = np.nonzero(m.any(axis=1))[0]
        if len(ys) < min_rows:
            continue
        frames_used += 1
        widths = m.sum(axis=1)
        for y in ys:
            if widths[y] >= min_width:
                profiles.setdefault(int(y), []).append(float(widths[y]))

    if frames_used < min_frames or len(profiles) < min_rows:
        return {"frames_used": frames_used, "score": None,
                "notes": [f"stream too small for taper analysis "
                          f"({frames_used} usable frames, "
                          f"{len(profiles)} rows)"]}

    # rows seen in enough frames -> median width profile
    min_count = max(1, frames_used // 4)
    rows = sorted(y for y, ws in profiles.items() if len(ws) >= min_count)
    if len(rows) < min_rows:
        return {"frames_used": frames_used, "score": None,
                "notes": ["too few consistently-tracked stream rows for "
                          "taper analysis"]}
    y_arr = np.array(rows, dtype=np.float64)
    w_arr = np.array([np.median(profiles[y]) for y in rows])

    # trim spout attachment (top) and splash zone (bottom)
    n_rows = len(rows)
    lo = int(n_rows * trim_top)
    hi = n_rows - int(n_rows * trim_bottom)
    if hi - lo < min_rows:
        lo, hi = 0, n_rows
    y_t, w_t = y_arr[lo:hi], w_arr[lo:hi]

    rho = _spearman(y_t, w_t)          # want strongly negative
    narrowing = max(0.0, -rho)

    inv_w4 = 1.0 / np.maximum(w_t, min_width) ** 4
    slope, intercept = np.polyfit(y_t, inv_w4, 1)
    pred = slope * y_t + intercept
    ss_res = float(np.sum((inv_w4 - pred) ** 2))
    ss_tot = float(np.sum((inv_w4 - inv_w4.mean()) ** 2))
    r2 = max(0.0, min(1.0, 1.0 - ss_res / ss_tot)) if ss_tot > 1e-12 else 0.0
    power_component = r2 if slope > 0 else 0.2 * r2

    score = 0.6 * narrowing + 0.4 * power_component
    notes = []
    if rho >= -0.1:
        notes.append(f"stream does not narrow while falling "
                     f"(width-vs-depth rho={rho:+.2f}) -- uniform or "
                     f"widening column, inconsistent with free fall + "
                     f"mass conservation")

    return {
        "frames_used": frames_used,
        "n_profile_rows": int(hi - lo),
        "width_top_median": float(w_t[0]),
        "width_bottom_median": float(w_t[-1]),
        "spearman_w_vs_y": float(rho),
        "invw4_slope": float(slope),
        "invw4_r2": float(r2),
        "score": float(score),
        "notes": notes,
    }


# -------------------- gravity component C: leading edge --------------------

def gravity_leading_edge(stream_masks_by_frame, min_frames=4,
                         min_pixels=20, stop_growth_tol=2.0):
    """
    Quadratic fit to the advancing stream front (pour onset only).
    score=None ("not applicable") when the clip starts mid-pour --
    common and fine; components A/B carry the gravity score then.
    """
    edge_y = []
    for m in stream_masks_by_frame:
        ys, _ = np.nonzero(m)
        edge_y.append(float(ys.max()) if len(ys) >= min_pixels else None)

    advancing, last_y = [], None
    for i, y in enumerate(edge_y):
        if y is None:
            if advancing:
                break
            continue
        if last_y is None or y > last_y + stop_growth_tol:
            advancing.append(i)
            last_y = y
        elif advancing:
            break

    if len(advancing) < min_frames:
        return {"advancing_frames": advancing, "score": None,
                "notes": [f"no advancing pour front captured "
                          f"({len(advancing)} frames) -- clip likely "
                          f"starts mid-pour (not an error)"]}

    t = np.array(advancing, dtype=np.float64)
    y = np.array([edge_y[i] for i in advancing])
    coeffs = np.polyfit(t, y, 2)
    fitted_a = 2.0 * coeffs[0]
    rms = float(np.sqrt(np.mean((y - np.polyval(coeffs, t)) ** 2)))
    norm_res = rms / max(float(y.max() - y.min()), 1.0)

    sign_ok = bool(fitted_a > 0)
    fit_q = max(0.0, 1.0 - norm_res)
    score = fit_q if sign_ok else fit_q * 0.2

    return {
        "advancing_frames": advancing,
        "fitted_acceleration": float(fitted_a),
        "normalized_residual": norm_res,
        "acceleration_sign_ok": sign_ok,
        "score": float(score),
        "notes": [],
    }


# ------------------------- color consistency -------------------------

def _lab_frames(frames_bgr):
    import cv2
    out = []
    for f in frames_bgr:
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float64)
        lab[..., 0] *= 100.0 / 255.0     # L to [0,100]
        lab[..., 1] -= 128.0             # a, b centered on 0
        lab[..., 2] -= 128.0
        out.append(lab)
    return out


def _splash_row_band(stream_masks, frac=0.12):
    """Rows around the impact point (stream bottom), where foam/aeration
    legitimately changes appearance -- excluded from color sampling."""
    bottoms, extents = [], []
    for m in stream_masks:
        ys = np.nonzero(m.any(axis=1))[0]
        if len(ys) >= 4:
            bottoms.append(ys.max())
            extents.append(ys.max() - ys.min())
    if not bottoms:
        return None
    impact = float(np.median(bottoms))
    half = frac * float(np.median(extents))
    return (impact - half, impact + half)


def color_consistency(frames_bgr, region_masks_by_frame, stream_masks,
                      gain_tau=0.15, min_pixels=30):
    """
    Fluid appearance check. Physical model: apparent fluid color =
    background transmitted through water + a stable intrinsic tint.
    Therefore:

      ALLOWED   -- tint magnitude decreasing (fading to clearer), over
                   time or where the stream is thin (less optical depth)
      PENALIZED -- tint HUE rotating over time (color-changing fluid),
                   and tint magnitude INCREASING (spontaneously gaining
                   color/murk away from the background)
      EXCLUDED  -- a row band around the splash/impact zone, where foam
                   legitimately whitens the fluid

    Components:
      temporal (0.7): chroma-weighted circular stability of tint hue
        across frames  x  exp(-positive-tint-gains / tau)
      spatial (0.3, stream only): hue stability along stream rows, with
        a bonus when tint magnitude tracks stream width (thin -> clearer)
      source_abrupt: multiplicative penalty on temporal when the SOURCE
        region undergoes a large-scale (>15% area) color change of
        >10 dE in a single frame step -- indicates generated artifacts
        like color bleeding or texture popping.

    frames_bgr must align 1:1 with mask frames.
    """
    lab = _lab_frames(frames_bgr)
    n = min(len(lab), *(len(v) for v in region_masks_by_frame.values()))
    band = _splash_row_band(stream_masks)

    def row_ok(y):
        return band is None or not (band[0] <= y <= band[1])

    # ---- per-frame tint of the liquid vs background ----
    tints, mags = [], []
    for i in range(n):
        liquid = np.logical_or.reduce(
            [region_masks_by_frame[r][i] for r in region_masks_by_frame])
        if liquid.sum() < min_pixels:
            continue
        h = liquid.shape[0]
        keep_rows = np.array([row_ok(y) for y in range(h)])
        sample = liquid & keep_rows[:, None]
        if sample.sum() < min_pixels:
            sample = liquid
        bg = ~liquid
        if bg.sum() < min_pixels:
            continue
        fluid_med = np.median(lab[i][sample], axis=0)
        bg_med = np.median(lab[i][bg], axis=0)
        t = fluid_med - bg_med
        tints.append(t)
        mags.append(float(np.linalg.norm(t)))

    if len(tints) < 4:
        return {"score": None,
                "notes": [f"only {len(tints)} frames with enough liquid+"
                          f"background pixels for color analysis"]}

    tints = np.array(tints)
    mags = np.array(mags)

    # hue stability: circular resultant of tint direction in (a, b),
    # weighted by chroma -- near-achromatic frames contribute little
    chroma = np.hypot(tints[:, 1], tints[:, 2])
    hue = np.arctan2(tints[:, 2], tints[:, 1])
    c_sum = float(chroma.sum())
    if c_sum < 2.0 * len(tints):     # essentially clear/achromatic fluid
        hue_stability = 1.0
        hue_note = None
    else:
        resultant = np.hypot((chroma * np.cos(hue)).sum(),
                             (chroma * np.sin(hue)).sum()) / c_sum
        hue_stability = float(resultant)
        hue_note = (f"fluid tint hue is unstable over time "
                    f"(stability={hue_stability:.2f}) -- color-changing "
                    f"fluid" if hue_stability < 0.8 else None)

    # asymmetric magnitude penalty: gains in tint are suspect,
    # fading toward background is free. Two terms: per-frame jump gains
    # (p90 of positive deltas) and cumulative net drift (last-quarter vs
    # first-quarter median), so slow steady murk gain is also caught.
    # Net drift is half-weighted: a filling pool legitimately deepens
    # in tint as optical depth grows -- tune against real footage.
    deltas = np.diff(mags)
    scale = max(float(np.median(mags)), 5.0)
    gains = deltas[deltas > 0] / scale
    gain_p = float(np.percentile(gains, 90)) if len(gains) else 0.0
    q = max(len(mags) // 4, 2)
    net_gain = max(0.0, float(np.median(mags[-q:]) - np.median(mags[:q]))) / scale
    gain_metric = max(gain_p, 0.5 * net_gain)
    gain_factor = float(np.exp(-gain_metric / gain_tau))

    temporal = hue_stability * gain_factor

    # ---- source abrupt-change penalty ----
    source_masks = region_masks_by_frame.get("source", [])
    source_penalty = 1.0
    dE_threshold = 10.0
    area_threshold = 0.15
    worst_frac = 0.0
    n_src = min(n, len(source_masks))
    for i in range(n_src - 1):
        s0, s1 = source_masks[i], source_masks[i + 1]
        if not s0.any() or not s1.any():
            continue
        overlap = s0 & s1
        if overlap.sum() < min_pixels:
            continue
        delta = lab[i + 1][overlap] - lab[i][overlap]
        dE = np.sqrt((delta ** 2).sum(axis=1))
        frac = float((dE > dE_threshold).sum() / overlap.sum())
        if frac > worst_frac:
            worst_frac = frac
    if worst_frac > area_threshold:
        source_penalty = max(0.0, 1.0 - 3.0 * (worst_frac - area_threshold))
        temporal *= source_penalty

    # ---- spatial: along-stream hue consistency + thin->clear bonus ----
    row_hue, row_chroma, row_mag, row_w = [], [], [], []
    for i in range(min(n, len(stream_masks))):
        m = stream_masks[i]
        if not m.any():
            continue
        liquid = np.logical_or.reduce(
            [region_masks_by_frame[r][i] for r in region_masks_by_frame])
        bg = ~liquid
        if bg.sum() < min_pixels:
            continue
        bg_med = np.median(lab[i][bg], axis=0)
        ys = np.nonzero(m.any(axis=1))[0]
        for y in ys:
            if not row_ok(y):
                continue
            xs = np.nonzero(m[y])[0]
            if len(xs) < 2:
                continue
            t = np.median(lab[i][y, xs], axis=0) - bg_med
            row_hue.append(np.arctan2(t[2], t[1]))
            row_chroma.append(np.hypot(t[1], t[2]))
            row_mag.append(np.linalg.norm(t))
            row_w.append(len(xs))

    spatial = None
    width_tint_rho = None
    if len(row_hue) >= 20:
        rc = np.array(row_chroma)
        rh = np.array(row_hue)
        c_sum = float(rc.sum())
        if c_sum < 2.0 * len(rc):
            row_hue_stab = 1.0
        else:
            row_hue_stab = float(np.hypot((rc * np.cos(rh)).sum(),
                                          (rc * np.sin(rh)).sum()) / c_sum)
        width_tint_rho = _spearman(row_w, row_mag)
        # thin rows clearer (rho > 0) earns up to a 1.0 multiplier;
        # no or inverse coupling floors at 0.75 -- absence of fading in
        # a uniform stream is not a violation
        spatial = row_hue_stab * (0.75 + 0.25 * max(0.0, width_tint_rho))

    if spatial is None:
        score = temporal
    else:
        score = 0.7 * temporal + 0.3 * spatial

    notes = [x for x in [hue_note] if x]
    if gain_metric > 0.15:
        notes.append(f"fluid gains tint over time (jump p90={gain_p:.2f}, "
                     f"net drift={net_gain:.2f}) -- spontaneous coloring/"
                     f"murk; fading to clear would not be penalized")
    if worst_frac > area_threshold:
        notes.append(f"large abrupt color change in source region: "
                     f"{100*worst_frac:.0f}% of area shifted by >{dE_threshold} "
                     f"dE between consecutive frames")

    return {
        "n_frames_used": len(tints),
        "tint_magnitudes": mags.tolist(),
        "hue_stability": float(hue_stability),
        "gain_p90": gain_p,
        "net_gain": float(net_gain),
        "temporal_score": float(temporal),
        "spatial_score": None if spatial is None else float(spatial),
        "width_tint_spearman": width_tint_rho,
        "splash_band_rows": band,
        "source_abrupt_change": {
            "worst_fraction": float(worst_frac),
            "dE_threshold": dE_threshold,
            "penalty": float(source_penalty),
        },
        "score": float(score),
        "notes": notes,
    }


# --------------------------- combination ---------------------------

GRAVITY_WEIGHTS = {"taper": 0.45, "direction": 0.10,
                   "speedup": 0.10, "leading_edge": 0.35}


def combine_gravity(taper_r, direction_r, speedup_r, edge_r):
    parts = {"taper": taper_r, "direction": direction_r,
             "speedup": speedup_r, "leading_edge": edge_r}
    avail = {k: r["score"] for k, r in parts.items() if r["score"] is not None}
    notes = [n for r in parts.values() for n in r.get("notes", [])]
    if not avail:
        return {"score": None, "components_used": [], "notes": notes}
    w_sum = sum(GRAVITY_WEIGHTS[k] for k in avail)
    score = sum(GRAVITY_WEIGHTS[k] * v for k, v in avail.items()) / w_sum
    return {"score": float(score), "components_used": sorted(avail),
            "notes": notes}


def composite_score(volume_result, gravity_combined, color_result=None,
                    w_volume=0.35, w_gravity=0.40, w_color=0.25):
    axes = {"volume": (volume_result["score"], w_volume),
            "gravity": (gravity_combined["score"], w_gravity),
            "color": (None if color_result is None else color_result["score"],
                      w_color)}
    avail = {k: (s, w) for k, (s, w) in axes.items() if s is not None}
    if not avail:
        comp = None
    else:
        w_sum = sum(w for _, w in avail.values())
        comp = sum(s * w for s, w in avail.values()) / w_sum
    return {"volume_score": axes["volume"][0],
            "gravity_score": axes["gravity"][0],
            "color_score": axes["color"][0],
            "composite": None if comp is None else float(comp)}
