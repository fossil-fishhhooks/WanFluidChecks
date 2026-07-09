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
                           if f.endswith(".npy"))
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


def _overlay_flow_arrows(frame, flow, mask=None, step=16, noise_floor=0.25):
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
            # skip arrows outside liquid mask
            if mask is not None and not mask[int(Y[j, i]), int(X[j, i])]:
                continue
            pt1 = (int(X[j, i]), int(Y[j, i]))
            # cap arrow length to 40px for display; color shows true magnitude
            scale = min(1.0, 40.0 / max(mag[j, i], 1e-6))
            ax = int(X[j, i] + dx[j, i] * scale)
            ay = int(Y[j, i] + dy[j, i] * scale)
            pt2 = (ax, ay)
            t = min(1.0, mag[j, i] / 80.0)
            # blue (slow) -> green -> red (fast)
            color = (int(255 * t), int(200 * (1 - abs(t - 0.5) * 2)), 0)
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

def build_demo(clips_dir, masks_dir, flow_dir, results_dir):
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
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_demo(args.clips_dir, args.masks_dir, args.flow_dir,
                      args.results_dir)
    demo.launch(server_port=args.port)


if __name__ == "__main__":
    main()
