"""
checks.py

Two independent, self-referential checks (no ground truth needed):

1. volume_check: total liquid area (source + stream + pool combined)
   should evolve smoothly -- liquid is redistributed between regions,
   not created or destroyed. Compares per-frame area against a
   median-smoothed trend and flags spikes, so legitimate gradual change
   (splash spread, stream width variation) isn't penalized but sudden
   appearance/disappearance is.

2. gravity_check: two complementary sub-checks, because a pour has two
   physically distinct regimes:

   a) TRANSIENT (leading edge): before the stream reaches the pool, the
      front of the falling liquid is in free fall, so its vertical
      position follows y(t) = y0 + v0*t + 0.5*a*t^2. We track max-y of
      the stream mask and fit a quadratic. (NOT the centroid -- a steady
      stream's centroid is stationary while material flows through it.)

   b) STEADY STATE (flow field): in an established stream, downward
      speed must increase with fall distance: v^2 = v0^2 + 2*a*dy.
      So v_y^2 vs y within the stream mask should be linear with a
      positive slope. Uses RAFT flow from flow.py.

Both return raw signals plus a normalized [0, 1] score (1 = good).
"""

import numpy as np


# ----------------------------- helpers -----------------------------

def _mask_area(mask):
    return float(np.sum(mask))


def _median_smooth(x, k=5):
    """Simple odd-window median filter with edge clamping."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return x
    half = k // 2
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = np.median(x[lo:hi])
    return out


# --------------------------- volume check ---------------------------

def volume_check(region_masks_by_frame, spike_threshold=0.25, smooth_k=5):
    """
    region_masks_by_frame: dict[region_name] -> list of bool arrays
        (one per frame, same length across regions; pad missing frames
        with all-False masks of the same shape before calling)

    spike_threshold: fractional deviation from the smoothed trend that
        counts as a violation (0.25 = 25% off-trend)

    Returns dict:
      total_area:        per-frame total liquid area
      smoothed_area:     median-smoothed trend
      spike_magnitudes:  per-frame |area - trend| / trend
      violation_frames:  frame indices exceeding spike_threshold
      score:             [0, 1]; 1.0 = fully consistent, penalized by
                         both violation rate and mean violation magnitude
    """
    region_names = list(region_masks_by_frame.keys())
    n_frames = len(region_masks_by_frame[region_names[0]])

    total_area = np.array([
        sum(_mask_area(region_masks_by_frame[r][i]) for r in region_names)
        for i in range(n_frames)
    ])

    smoothed = _median_smooth(total_area, k=smooth_k)
    denom = np.maximum(smoothed, 1.0)
    spikes = np.abs(total_area - smoothed) / denom

    violation_frames = [int(i) for i in np.nonzero(spikes > spike_threshold)[0]]

    if n_frames == 0:
        score = 1.0
    else:
        rate = len(violation_frames) / n_frames
        mean_excess = (
            float(np.mean(spikes[spikes > spike_threshold] - spike_threshold))
            if violation_frames else 0.0
        )
        score = max(0.0, 1.0 - rate - mean_excess)

    return {
        "total_area": total_area.tolist(),
        "smoothed_area": smoothed.tolist(),
        "spike_magnitudes": spikes.tolist(),
        "violation_frames": violation_frames,
        "score": float(score),
    }


# --------------------- gravity: transient phase ---------------------

def gravity_check_leading_edge(stream_masks_by_frame, min_frames=4,
                               min_pixels=20, stop_growth_tol=2.0):
    """
    Track the leading (lowest) edge of the stream while it is still
    advancing downward, and fit y_edge(t) to a quadratic. Only the
    advancing portion is used: once the edge stops moving (stream hit
    the pool / bottom), later frames are excluded from the fit.

    Note on image coords: y increases downward, so physically correct
    free fall gives a POSITIVE fitted acceleration.

    Returns dict with fit details and a [0,1] score, or a low-info
    result with score=None if there aren't enough advancing frames
    (score=None means "not applicable", not "failed" -- e.g. the clip
    may start with the stream already established).
    """
    edge_y = []
    for m in stream_masks_by_frame:
        ys, _ = np.nonzero(m)
        edge_y.append(float(np.max(ys)) if len(ys) >= min_pixels else None)

    # find the advancing prefix: frames where the edge is still moving down
    advancing_idx = []
    last_y = None
    for i, y in enumerate(edge_y):
        if y is None:
            if advancing_idx:
                break
            continue
        if last_y is None or y > last_y + stop_growth_tol:
            advancing_idx.append(i)
            last_y = y
        elif advancing_idx:
            break  # edge stalled -> steady state reached

    if len(advancing_idx) < min_frames:
        return {
            "edge_y": edge_y,
            "advancing_frames": advancing_idx,
            "fit_coeffs": None,
            "score": None,
            "note": f"only {len(advancing_idx)} advancing frames "
                    f"(need >= {min_frames}); stream may start established",
        }

    t = np.array(advancing_idx, dtype=np.float64)
    y = np.array([edge_y[i] for i in advancing_idx], dtype=np.float64)

    coeffs = np.polyfit(t, y, 2)          # y = c2*t^2 + c1*t + c0
    fitted_a = 2.0 * coeffs[0]

    y_pred = np.polyval(coeffs, t)
    rms = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    y_range = max(float(np.max(y) - np.min(y)), 1.0)
    norm_residual = rms / y_range

    sign_ok = bool(fitted_a > 0)
    fit_quality = max(0.0, 1.0 - norm_residual)
    score = fit_quality if sign_ok else fit_quality * 0.3

    return {
        "edge_y": edge_y,
        "advancing_frames": advancing_idx,
        "fit_coeffs": coeffs.tolist(),
        "fitted_acceleration": float(fitted_a),
        "normalized_residual": norm_residual,
        "acceleration_sign_ok": sign_ok,
        "score": float(score),
    }


# -------------------- gravity: steady-state phase --------------------

def gravity_check_stream_flow(flow_by_frame, stream_masks_by_frame,
                              n_bins=12, min_pixels_per_bin=15,
                              min_valid_bins=5):
    """
    Steady-stream check using the RAFT flow field.

    Physics: for liquid falling under gravity, v^2 = v0^2 + 2*a*(y - y0).
    So within the stream mask, mean downward flow speed squared, binned
    by vertical position y, should be LINEAR in y with POSITIVE slope.
    (In image coords, downward flow means positive v_y.)

    We pool (y_bin_center, mean v_y^2) samples across all frames and fit
    a single line. Scored on:
      - slope sign (must be positive: speeding up while falling)
      - R^2 of the linear fit (flow should be orderly, not chaotic)
      - direction consistency: fraction of stream pixels with downward
        flow (v_y > 0) -- a stream drifting sideways/up is a violation

    flow_by_frame: list of HxWx2 float arrays (dx, dy) -- UNMASKED flow;
        the stream mask is applied here.
    stream_masks_by_frame: list of bool arrays; must align with flow
        frames (flow[i] is between mask frame i and i+1).

    Returns dict with fit details and [0,1] score, or score=None if the
    stream mask never has enough pixels (not applicable).
    """
    samples_y, samples_v2 = [], []
    down_pixels, total_pixels = 0, 0

    n = min(len(flow_by_frame), len(stream_masks_by_frame))
    for i in range(n):
        flow = flow_by_frame[i]
        mask = stream_masks_by_frame[i]
        if mask.shape != flow.shape[:2]:
            continue
        ys, xs = np.nonzero(mask)
        if len(ys) < min_pixels_per_bin:
            continue

        v_y = flow[ys, xs, 1]  # vertical flow component, +ve = downward
        down_pixels += int(np.sum(v_y > 0))
        total_pixels += len(v_y)

        y_min, y_max = ys.min(), ys.max()
        if y_max - y_min < n_bins:
            continue
        bin_edges = np.linspace(y_min, y_max + 1, n_bins + 1)
        bin_idx = np.digitize(ys, bin_edges) - 1

        for b in range(n_bins):
            sel = bin_idx == b
            if np.sum(sel) < min_pixels_per_bin:
                continue
            # mean of positive (downward) component only; upward-moving
            # pixels contribute to the direction penalty instead
            vy_sel = v_y[sel]
            vy_down = vy_sel[vy_sel > 0]
            if len(vy_down) < min_pixels_per_bin // 2:
                continue
            samples_y.append(0.5 * (bin_edges[b] + bin_edges[b + 1]))
            samples_v2.append(float(np.mean(vy_down) ** 2))

    if len(samples_y) < min_valid_bins:
        return {
            "n_samples": len(samples_y),
            "score": None,
            "note": f"only {len(samples_y)} (y, v^2) samples "
                    f"(need >= {min_valid_bins}); stream too small/short "
                    "or flow frames missing",
        }

    y_arr = np.array(samples_y)
    v2_arr = np.array(samples_v2)

    slope, intercept = np.polyfit(y_arr, v2_arr, 1)
    pred = slope * y_arr + intercept
    ss_res = float(np.sum((v2_arr - pred) ** 2))
    ss_tot = float(np.sum((v2_arr - np.mean(v2_arr)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
    r2 = max(0.0, min(1.0, r2))

    slope_ok = bool(slope > 0)
    down_fraction = down_pixels / total_pixels if total_pixels else 0.0

    base = r2 * down_fraction
    score = base if slope_ok else base * 0.3

    return {
        "n_samples": len(samples_y),
        "slope_v2_vs_y": float(slope),
        "slope_sign_ok": slope_ok,
        "r_squared": r2,
        "downward_flow_fraction": float(down_fraction),
        "score": float(score),
    }


# --------------------------- combination ---------------------------

def combine_gravity(leading_edge_result, stream_flow_result):
    """
    Combine the two gravity sub-checks, handling not-applicable cases
    (score=None). If both applicable, average; if one, use it; if
    neither, overall gravity score is None (flag for manual review
    rather than silently scoring 0).
    """
    scores = [r["score"] for r in (leading_edge_result, stream_flow_result)
              if r["score"] is not None]
    if not scores:
        return {"score": None, "note": "neither gravity sub-check applicable"}
    return {"score": float(np.mean(scores))}


def composite_score(volume_result, gravity_combined, w_volume=0.5, w_gravity=0.5):
    g = gravity_combined["score"]
    if g is None:
        return {
            "volume_score": volume_result["score"],
            "gravity_score": None,
            "composite": None,
            "note": "gravity not scoreable for this clip; see sub-check notes",
        }
    return {
        "volume_score": volume_result["score"],
        "gravity_score": g,
        "composite": w_volume * volume_result["score"] + w_gravity * g,
    }
