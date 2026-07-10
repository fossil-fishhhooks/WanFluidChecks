"""
score_viewer.py

Gradio GUI for inspecting Wan2.1 video physics scores with per-frame
overlays (SAM masks, RAFT flow arrows, taper profile, penalty highlights).

Usage:
  python gradio/score_viewer.py --results_dir ./results
"""

import argparse
import json
import os

import cv2
import gradio as gr
import numpy as np

REGIONS = ["source", "stream", "pool"]
OVERLAY_COLORS = {
    "source": (0, 200, 255),
    "stream": (0, 255, 0),
    "pool": (255, 100, 0),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class ClipData:
    """Lazy-loaded data for one video clip."""

    def __init__(self, clip_name, clips_dir, masks_dir, flow_dir, results_dir):
        self.clip_name = clip_name
        self._clips_dir = clips_dir
        self._masks_dir = os.path.join(masks_dir, clip_name) if masks_dir else None
        self._flow_dir = (os.path.join(flow_dir, clip_name)
                          if flow_dir else None)
        self._results_dir = results_dir
        self._frames = None
        self._masks = None
        self._flows = None
        self._result = None

    @property
    def result(self):
        if self._result is None:
            p = os.path.join(self._results_dir, f"{self.clip_name}.json")
            with open(p) as f:
                self._result = json.load(f)
        return self._result

    @property
    def n_frames(self):
        return len(self.result.get("volume_check", {}).get("total_area", []))

    def frame(self, i):
        if self._frames is None:
            cap = cv2.VideoCapture(os.path.join(self._clips_dir,
                                                self.clip_name))
            self._frames = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                self._frames.append(f)
            cap.release()
        return self._frames[i].copy()

    def masks(self, region):
        if self._masks is None:
            self._masks = {}
        if region not in self._masks:
            d = os.path.join(self._masks_dir, region)
            if not os.path.isdir(d):
                self._masks[region] = []
            else:
                files = sorted(f for f in os.listdir(d)
                               if f.endswith(".npy"))
                self._masks[region] = [np.load(os.path.join(d, f))
                                       for f in files]
        return self._masks[region]

    def flow(self, i):
        if self._flows is None:
            if self._flow_dir is None or not os.path.isdir(self._flow_dir):
                self._flows = []
                return None
            files = sorted(f for f in os.listdir(self._flow_dir)
                           if f.startswith("frame_") and f.endswith(".npy"))
            self._flows = [np.load(os.path.join(self._flow_dir, f))
                           for f in files]
        if i < len(self._flows):
            return self._flows[i]
        return None


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------

def _overlay_masks(frame, masks_dict, idx):
    for region, masks in masks_dict.items():
        if idx >= len(masks):
            continue
        m = masks[idx]
        if not m.any():
            continue
        color = np.array(OVERLAY_COLORS[region], dtype=np.float32)
        ov = frame.copy()
        ov[m] = (0.55 * frame[m] + 0.45 * color).astype(np.uint8)
        frame = ov
    return frame


def _overlay_flow_arrows(frame, flow, mask=None, step=16, noise_floor=0.1):
    if flow is None:
        return frame
    h, w = frame.shape[:2]
    ys = np.arange(step // 2, h, step)
    xs = np.arange(step // 2, w, step)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    dx = flow[Y, X, 0]
    dy = flow[Y, X, 1]
    mag = np.sqrt(dx ** 2 + dy ** 2)
    out = frame.copy()
    for j in range(len(ys)):
        for i in range(len(xs)):
            if mag[j, i] <= noise_floor:
                continue
            if mask is not None and not mask[int(Y[j, i]), int(X[j, i])]:
                continue
            pt1 = (int(X[j, i]), int(Y[j, i]))
            # arrow length: min 8px visible, cap at 40px; direction from (dx,dy)
            vis_len = max(8.0, min(40.0, mag[j, i]))
            norm = max(mag[j, i], 1e-6)
            ax = int(X[j, i] + dx[j, i] / norm * vis_len)
            ay = int(Y[j, i] + dy[j, i] / norm * vis_len)
            pt2 = (ax, ay)
            # log-scale color: blue→green→red
            t = min(1.0, np.log2(1 + mag[j, i]) / 6.0)
            b = max(0, 1 - 2 * t)
            g = 1 - abs(2 * t - 1)
            r = max(0, 2 * t - 1)
            color = (int(b * 255), int(g * 255), int(r * 255))
            cv2.arrowedLine(out, pt1, pt2, color, 1, cv2.LINE_AA, tipLength=0.3)
    return out


def _overlay_taper_outline(frame, stream_mask):
    if stream_mask is None or not stream_mask.any():
        return frame
    m = stream_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out = frame.copy()
    cv2.drawContours(out, contours, -1, (0, 255, 255), 1)
    return out


def _overlay_penalty_highlights(frame, cd, idx):
    result = cd.result
    out = frame.copy()

    # 1) Volume dropout frames
    dropouts = result.get("volume_check", {}).get("dropout_frames", [])
    if idx in dropouts:
        for r in REGIONS:
            masks = cd.masks(r)
            if idx < len(masks) and masks[idx].any():
                out[masks[idx]] = (0, 0, 200)

    # 2) Source abrupt color change
    src = result.get("color_check", {}).get("source_abrupt_change", {})
    if src.get("penalty", 1.0) < 1.0:
        src_masks = cd.masks("source")
        if idx < len(src_masks) and src_masks[idx].any():
            overlay = out.copy()
            overlay[src_masks[idx]] = (0, 0, 255)
            out = cv2.addWeighted(out, 0.5, overlay, 0.5, 0)

    return out


def render_frame(cd, idx, show_masks, show_flow, show_taper, show_penalties):
    nf = cd.n_frames
    if nf == 0:
        return np.zeros((480, 832, 3), dtype=np.uint8)
    idx = int(np.clip(idx, 0, nf - 1))

    frame = cd.frame(idx)
    masks_dict = {r: cd.masks(r) for r in REGIONS}

    if show_masks:
        frame = _overlay_masks(frame, masks_dict, idx)
    if show_flow:
        # combine source | stream | pool mask for this frame
        flow_mask = np.zeros(frame.shape[:2], dtype=bool)
        for r in REGIONS:
            m = cd.masks(r)
            if idx < len(m):
                flow_mask |= m[idx]
        frame = _overlay_flow_arrows(frame, cd.flow(idx), mask=flow_mask)
    if show_taper:
        stream_masks = cd.masks("stream")
        sm = stream_masks[idx] if idx < len(stream_masks) else None
        frame = _overlay_taper_outline(frame, sm)
    if show_penalties:
        frame = _overlay_penalty_highlights(frame, cd, idx)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Score sidebar
# ---------------------------------------------------------------------------

def _fmt(val):
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _sc(val):
    if val is None:
        return "#888"
    if val >= 0.7:
        return "#2e7d32"
    if val >= 0.4:
        return "#e65100"
    return "#c62828"


def _icon(val):
    if val is None:
        return "⊙"
    if val >= 0.7:
        return "✓"
    if val >= 0.4:
        return "⚡"
    return "✗"


def build_score_html(result):
    s = result.get("scores", {})
    vol, col, grav, comp = (s.get(k) for k in
                            ("volume_score", "color_score",
                             "gravity_score", "composite"))
    g = result.get("gravity_combined", {})
    taper = result.get("gravity_taper", {}).get("score")
    direc = result.get("gravity_direction", {}).get("score")
    speed = result.get("gravity_speedup", {}).get("score")
    ledge = result.get("gravity_leading_edge", {}).get("score")

    comp_pct = min(100, int(100 * (comp or 0)))

    notes = []
    for k in ("volume_check", "color_check", "gravity_combined"):
        notes.extend(result.get(k, {}).get("notes", []))

    src = result.get("color_check", {}).get("source_abrupt_change", {})

    lines = f"""
    <div style="font-family:monospace;font-size:13px;">
    <div style="font-size:22px;font-weight:bold;color:{_sc(comp)};">
      {_icon(comp)} Composite: {_fmt(comp)}
    </div>
    <div style="height:10px;background:#333;border-radius:5px;margin:6px 0;">
      <div style="height:10px;width:{comp_pct}%;background:{_sc(comp)};border-radius:5px;"></div>
    </div>
    <table style="width:100%;border-collapse:collapse;">
      <tr>
        <td><span style="color:{_sc(vol)};">{_icon(vol)}</span> Volume</td>
        <td style="text-align:right;color:{_sc(vol)};">{_fmt(vol)}</td>
      </tr>
      <tr>
        <td><span style="color:{_sc(col)};">{_icon(col)}</span> Color</td>
        <td style="text-align:right;color:{_sc(col)};">{_fmt(col)}</td>
      </tr>
      <tr>
        <td><span style="color:{_sc(grav)};">{_icon(grav)}</span> Gravity</td>
        <td style="text-align:right;color:{_sc(grav)};">{_fmt(grav)}</td>
      </tr>
      <tr><td>&nbsp;&nbsp;├ Taper</td><td style="text-align:right;color:{_sc(taper)};">{_fmt(taper)}</td></tr>
      <tr><td>&nbsp;&nbsp;├ Direction</td><td style="text-align:right;color:{_sc(direc)};">{_fmt(direc)}</td></tr>
      <tr><td>&nbsp;&nbsp;├ Speedup</td><td style="text-align:right;color:{_sc(speed)};">{_fmt(speed)}</td></tr>
      <tr><td>&nbsp;&nbsp;└ Leading Edge</td><td style="text-align:right;color:{_sc(ledge)};">{_fmt(ledge)}</td></tr>
    </table>
    <hr style="border:none;border-top:1px solid #555;margin:10px 0;">
    <div style="font-size:12px;">
    """
    if src.get("penalty", 1.0) < 1.0:
        lines += (f'<div style="color:#c62828;margin:3px 0;">'
                  f'✗ Source abrupt change: '
                  f'{100*src["worst_fraction"]:.0f}% area, '
                  f'penalty={src["penalty"]:.2f}</div>')
    if notes:
        for n in notes:
            lines += f'<div style="color:#ccc;margin:2px 0;">• {n}</div>'
    else:
        lines += '<div style="color:#666;">No issues</div>'
    lines += "</div></div>"
    return lines


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Weight-tuning tab
# ---------------------------------------------------------------------------

GRAV_KEYS = ["gravity_taper", "gravity_direction", "gravity_speedup",
             "gravity_leading_edge"]
GRAV_LABELS = ["taper", "direction", "speedup", "leading_edge"]


def load_all_results(results_dir):
    """clip -> {volume, color, grav components, mask_ok} for recomputation."""
    rows = {}
    if not os.path.isdir(results_dir):
        return rows
    for f in sorted(os.listdir(results_dir)):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(results_dir, f)) as fh:
            r = json.load(fh)
        s = r.get("scores", {})
        rows[f[:-5]] = {
            "volume": s.get("volume_score"),
            "color": s.get("color_score"),
            "grav": {lbl: r.get(k, {}).get("score")
                     for k, lbl in zip(GRAV_KEYS, GRAV_LABELS)},
            "mask_ok": r.get("stream_mask_ok", True),
        }
    return rows


def recompute_table(results, w_vol, w_grav, w_col,
                    w_taper, w_dir, w_spd, w_edge):
    """Same NA-renormalization semantics as checks.py, custom weights."""
    gw = {"taper": w_taper, "direction": w_dir,
          "speedup": w_spd, "leading_edge": w_edge}
    rows = []
    for clip, d in results.items():
        avail_g = {k: v for k, v in d["grav"].items()
                   if v is not None and gw[k] > 0}
        g_wsum = sum(gw[k] for k in avail_g)
        gravity = (sum(gw[k] * v for k, v in avail_g.items()) / g_wsum
                   if g_wsum > 0 else None)

        axes = {"volume": (d["volume"], w_vol),
                "gravity": (gravity, w_grav),
                "color": (d["color"], w_col)}
        avail = {k: (s, w) for k, (s, w) in axes.items()
                 if s is not None and w > 0}
        wsum = sum(w for _, w in avail.values())
        comp = (sum(s * w for s, w in avail.values()) / wsum
                if wsum > 0 else None)

        fmt = lambda x: "NA" if x is None else round(x, 3)
        rows.append([clip[:40], str(d["mask_ok"]), fmt(d["volume"]),
                     fmt(d["color"]), fmt(gravity), fmt(comp)])
    rows.sort(key=lambda r: (r[5] == "NA", -(r[5] if r[5] != "NA" else 0)))
    return rows


def weights_snippet(w_vol, w_grav, w_col, w_taper, w_dir, w_spd, w_edge):
    t = w_vol + w_grav + w_col
    g = w_taper + w_dir + w_spd + w_edge
    if t <= 0 or g <= 0:
        return "all weights zero -- nothing to export"
    return (
        "# paste into checks.py\n"
        f"GRAVITY_WEIGHTS = {{'taper': {w_taper/g:.3f}, "
        f"'direction': {w_dir/g:.3f}, 'speedup': {w_spd/g:.3f}, "
        f"'leading_edge': {w_edge/g:.3f}}}\n"
        f"# composite_score defaults:\n"
        f"# w_volume={w_vol/t:.3f}, w_gravity={w_grav/t:.3f}, "
        f"w_color={w_col/t:.3f}")


# ---------------------------------------------------------------------------
# Auto-SAM test tab
# ---------------------------------------------------------------------------

_AMG = None  # lazy singleton -- loading SAM2 takes a few seconds


def _get_amg():
    global _AMG
    if _AMG is None:
        from auto_annotate import load_amg
        _AMG = load_amg()
    return _AMG


def _autosam_preview(frame_bgr, regions):
    """Render role masks are not stored -- draw prompt points + labels."""
    out = frame_bgr.copy()
    for role, data in regions.items():
        if role.startswith("_"):
            continue
        color = OVERLAY_COLORS.get(role, (255, 255, 255))
        for (x, y) in data["points"]:
            cv2.circle(out, (int(x), int(y)), 6, color, 2, cv2.LINE_AA)
        if data["points"]:
            x, y = data["points"][0]
            cv2.putText(out, role, (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def run_autosam(clips_dir, clip_name):
    """Run auto-annotation on one clip; returns (preview_rgb, json_str,
    status, regions_state)."""
    from auto_annotate import load_all_frames, auto_annotate_clip
    if not clip_name:
        return None, "", "Pick a clip first.", None
    path = os.path.join(clips_dir, clip_name)
    frames = load_all_frames(path)
    if not frames:
        return None, "", f"Could not read {clip_name}", None
    amg = _get_amg()
    regions, conf, idx = auto_annotate_clip(amg, frames)
    if not regions:
        return (cv2.cvtColor(frames[len(frames) // 2], cv2.COLOR_BGR2RGB),
                "", "No stream candidate found -- annotate manually.", None)
    roles = [r for r in regions if not r.startswith("_")]
    preview = _autosam_preview(frames[idx], regions)
    status = (f"keyframe {idx} | roles: {', '.join(roles)} | "
              f"confidence {conf:.2f}"
              + ("  (LOW -- review before trusting)" if conf < 0.25 else ""))
    return (cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
            json.dumps(regions, indent=2), status, regions)


def save_autosam(prompts_path, clip_name, regions):
    if not regions or not clip_name:
        return "Nothing to save -- run detection first."
    existing = {}
    if os.path.exists(prompts_path):
        with open(prompts_path) as f:
            existing = json.load(f)
    existing[clip_name] = regions
    with open(prompts_path, "w") as f:
        json.dump(existing, f, indent=2)
    return (f"Saved {clip_name} -> {prompts_path}. "
            f"Run segment.py (--redo if re-segmenting) to apply.")


def build_demo(clips_dir, masks_dir, flow_dir, results_dir,
               prompts_path="prompts.json"):
    clip_names = sorted(
        d for d in os.listdir(masks_dir)
        if os.path.isdir(os.path.join(masks_dir, d))
    )
    if not clip_names:
        raise ValueError(f"No clip directories found in {masks_dir}")

    # Pre-scan n_frames so we can set slider max statically
    nf_map = {}
    for name in clip_names:
        p = os.path.join(results_dir, f"{name}.json")
        if os.path.exists(p):
            with open(p) as f:
                r = json.load(f)
            nf_map[name] = len(r.get("volume_check", {}).get("total_area", []))

    data_cache = {}

    def get_cd(name):
        if name not in data_cache:
            data_cache[name] = ClipData(name, clips_dir, masks_dir,
                                         flow_dir, results_dir)
        return data_cache[name]

    demo = gr.Blocks(title="Wan2.1 Score Viewer")
    with demo:
        gr.Markdown("# Arin's Wan2.1 Video Fluid Physics Score Viewer")

        with gr.Tab("🤖 Auto-SAM"):
            all_clips = sorted(
                f for f in os.listdir(clips_dir)
                if f.lower().endswith((".mp4", ".mov", ".avi")))
            gr.Markdown(
                "Test automatic annotation on any clip: runs SAM2's "
                "automatic mask generator + pour-scene geometry to pick "
                "source/stream/pool and synthesize prompt points. Review "
                "the preview, then save into prompts.json.")
            with gr.Row():
                clip_dd = gr.Dropdown(all_clips, label="Clip",
                                      value=all_clips[0] if all_clips else None)
                run_btn = gr.Button("Run detection", variant="primary")
                save_btn = gr.Button("Save to prompts.json")
            status_tb = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                prev_img = gr.Image(type="numpy", height=432,
                                    label="Keyframe + prompt points")
                json_tb = gr.Textbox(label="Generated prompts (v2 format)",
                                     lines=18)
            regions_state = gr.State(None)

            run_btn.click(
                fn=lambda name: run_autosam(clips_dir, name),
                inputs=[clip_dd],
                outputs=[prev_img, json_tb, status_tb, regions_state])
            save_btn.click(
                fn=lambda name, regions: save_autosam(prompts_path, name,
                                                      regions),
                inputs=[clip_dd, regions_state],
                outputs=[status_tb])

        with gr.Tab("⚖️ Weights"):
            results_holder = {"data": load_all_results(results_dir)}
            gr.Markdown(
                "Drag to re-weight the scoring axes and gravity "
                "components; the leaderboard re-ranks live (NA handling "
                "matches checks.py: weights renormalize over whatever "
                "each clip could measure). Zero a slider to exclude that "
                "signal entirely. Copy the snippet into checks.py once "
                "you like the balance.")
            with gr.Row():
                s_vol = gr.Slider(0, 1, 0.55, step=0.05, label="Volume")
                s_grav = gr.Slider(0, 1, 0.50, step=0.05, label="Gravity")
                s_col = gr.Slider(0, 1, 0.30, step=0.05, label="Color")
            with gr.Row():
                s_taper = gr.Slider(0, 1, 0.35, step=0.05, label="grav: taper")
                s_dir = gr.Slider(0, 1, 0.25, step=0.05, label="grav: direction")
                s_spd = gr.Slider(0, 1, 0.85, step=0.05, label="grav: speedup")
                s_edge = gr.Slider(0, 1, 0.25, step=0.05, label="grav: leading edge")
            reload_btn = gr.Button("↻ Reload results (after re-scoring)")
            table = gr.Dataframe(
                headers=["clip", "mask_ok", "volume", "color",
                         "gravity", "composite"],
                value=recompute_table(results_holder["data"],
                                      0.55, 0.50, 0.30,
                                      0.35, 0.25, 0.85, 0.25),
                interactive=False)
            snippet = gr.Code(
                value=weights_snippet(0.55, 0.50, 0.30,
                                      0.35, 0.25, 0.85, 0.25),
                language="python", label="Export")

            w_inputs = [s_vol, s_grav, s_col, s_taper, s_dir, s_spd, s_edge]

            def on_weights(*ws):
                return (recompute_table(results_holder["data"], *ws),
                        weights_snippet(*ws))

            def on_reload(*ws):
                results_holder["data"] = load_all_results(results_dir)
                return (recompute_table(results_holder["data"], *ws),
                        weights_snippet(*ws))

            for s in w_inputs:
                s.release(fn=on_weights, inputs=w_inputs,
                          outputs=[table, snippet])
            reload_btn.click(fn=on_reload, inputs=w_inputs,
                             outputs=[table, snippet])

        for name in clip_names:
            short = name[:24] + ("…" if len(name) > 24 else "")
            nf = max(nf_map.get(name, 1), 1)

            with gr.Tab(short):
                cd = get_cd(name)

                with gr.Row():
                    with gr.Column(scale=3):
                        fd = gr.Image(type="numpy", height=432,
                                       show_label=False)
                        fs = gr.Slider(0, nf - 1, value=0, step=1,
                                       label="Frame", interactive=True)
                        with gr.Row():
                            m_ck = gr.Checkbox(value=True, label="Masks")
                            f_ck = gr.Checkbox(value=False, label="Flow")
                            t_ck = gr.Checkbox(value=False, label="Taper")
                            p_ck = gr.Checkbox(value=True,
                                               label="Penalties")
                    with gr.Column(scale=2):
                        sc = gr.HTML(build_score_html(cd.result))

                def update(idx, m, f, t, p, name=name):
                    cd_ = get_cd(name)
                    return render_frame(cd_, idx, m, f, t, p)

                for ev in (fs.change, m_ck.change, f_ck.change,
                           t_ck.change, p_ck.change):
                    ev(fn=update,
                       inputs=[fs, m_ck, f_ck, t_ck, p_ck],
                       outputs=fd)

                demo.load(fn=update,
                          inputs=[fs, m_ck, f_ck, t_ck, p_ck],
                          outputs=fd)

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_dir", default="./clips")
    parser.add_argument("--masks_dir", default="./masks")
    parser.add_argument("--flow_dir", default="./flow")
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--prompts", default="prompts.json")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_demo(args.clips_dir, args.masks_dir, args.flow_dir,
                      args.results_dir, prompts_path=args.prompts)
    demo.launch(server_port=args.port)


if __name__ == "__main__":
    main()
