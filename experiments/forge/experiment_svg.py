"""Deterministic pure-Python SVG target & protocol schematic generator for Forge tasks.

Generates lightweight (typically 2-6 KB), valid XML SVG diagrams directly from each
task's declared mathematical specification. Every SVG uses a neutral light card canvas
so it renders crisply in both GitHub Light/Dark and MkDocs Material Light/Dark themes,
and explicitly carries the badge: "Target & Protocol Schematic · not training output".
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

SVG_WIDTH = 640
SVG_HEIGHT = 320
BADGE_TEXT = "Target & Protocol Schematic · not training output"

# Neutral theme-safe palette
BG_FILL = "#f8fafc"
BG_STROKE = "#cbd5e1"
PANEL_FILL = "#ffffff"
PANEL_STROKE = "#e2e8f0"
GRID_STROKE = "#eef2f7"
AXIS_STROKE = "#94a3b8"
TEXT_DARK = "#0f172a"
TEXT_MUTED = "#475569"
TEXT_LIGHT = "#64748b"
BADGE_BG = "#fef3c7"
BADGE_BORDER = "#f59e0b"
BADGE_FG = "#92400e"

BLUE = "#2563eb"
TEAL = "#0d9488"
AMBER = "#d97706"
ROSE = "#e11d48"
VIOLET = "#7c3aed"
EMERALD = "#059669"
INDIGO = "#4f46e5"
SLATE = "#334155"
CLASS_COLORS = (BLUE, ROSE, EMERALD, AMBER, VIOLET, TEAL, INDIGO, SLATE)


def _fmt(val: float) -> str:
    return f"{val:.1f}"


def _el(tag: str, attrs: dict[str, Any], content: str | None = None) -> str:
    parts = [f'{k}="{escape(str(v))}"' for k, v in attrs.items() if v is not None]
    if content is None:
        return f"<{tag} {' '.join(parts)}/>"
    return f"<{tag} {' '.join(parts)}>{content}</{tag}>"


def _text(x: float, y: float, text: str, *, size: int = 11, fill: str = TEXT_DARK,
          weight: str = "normal", anchor: str = "start") -> str:
    attrs: dict[str, Any] = {"x": _fmt(x), "y": _fmt(y), "font-size": size, "fill": fill}
    if weight != "normal":
        attrs["font-weight"] = weight
    if anchor != "start":
        attrs["text-anchor"] = anchor
    return _el("text", attrs, escape(text))


def _dots_el(points: list[tuple[float, float]], r: float, fill: str, opacity: str = "0.85") -> str:
    """Compact round-cap dot batch: exact circles in ~10 bytes/dot (keeps 256-mode SVGs ~5 KB)."""
    d = "".join(f"M{_fmt(x)},{_fmt(y)}h0" for x, y in points)
    return _el("path", {"d": d, "fill": "none", "stroke": fill, "stroke-width": _fmt(2.0 * r),
                        "stroke-linecap": "round", "stroke-opacity": opacity})


def _wrap_svg(title: str, subtitle: str, body: list[str], callout_lines: list[tuple[str, str]] | None = None) -> str:
    """Wrap schematic elements into a complete, self-contained 640x320 SVG."""
    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" '
        f'viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" role="img" aria-label="{escape(title)}">',
        f"<title>{escape(title)} — {escape(BADGE_TEXT)}</title>",
        "<style>text{font-family:system-ui,-apple-system,sans-serif}</style>",
        "<defs>",
        f'<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 1.5L10 5 0 8.5z" fill="{SLATE}"/></marker>',
        f'<marker id="arrow-rose" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 1.5L10 5 0 8.5z" fill="{ROSE}"/></marker>',
        f'<marker id="arrow-blue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 1.5L10 5 0 8.5z" fill="{BLUE}"/></marker>',
        f'<marker id="arrow-teal" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 1.5L10 5 0 8.5z" fill="{TEAL}"/></marker>',
        "</defs>",
        # Outer card
        _el("rect", {"x": 1, "y": 1, "width": SVG_WIDTH - 2, "height": SVG_HEIGHT - 2, "rx": 10,
                     "fill": BG_FILL, "stroke": BG_STROKE, "stroke-width": 1.2}),
        # Header title & subtitle
        _text(18, 25, title[:52], size=14, fill=TEXT_DARK, weight="700"),
        _text(18, 41, subtitle[:78], size=10, fill=TEXT_MUTED),
        # Explicit schematic badge (top-right)
        _el("rect", {"x": 388, "y": 10, "width": 236, "height": 22, "rx": 11,
                     "fill": BADGE_BG, "stroke": BADGE_BORDER, "stroke-width": 1}),
        _text(506, 25, BADGE_TEXT, size=9, fill=BADGE_FG, weight="600", anchor="middle"),
    ]
    out.extend(body)
    if callout_lines:
        # Right-hand specification card
        cx, cy, cw, ch = 416, 54, 208, 248
        out.append(_el("rect", {"x": cx, "y": cy, "width": cw, "height": ch, "rx": 8,
                                "fill": PANEL_FILL, "stroke": PANEL_STROKE, "stroke-width": 1}))
        out.append(_text(cx + 12, cy + 20, "Protocol Spec", size=11, fill=TEXT_DARK, weight="700"))
        out.append(_el("line", {"x1": cx + 12, "y1": cy + 27, "x2": cx + cw - 12, "y2": cy + 27,
                                "stroke": PANEL_STROKE, "stroke-width": 1}))
        y = cy + 44
        for label, val in callout_lines[:9]:
            out.append(_text(cx + 12, y, label[:28], size=9, fill=TEXT_LIGHT, weight="600"))
            out.append(_text(cx + 12, y + 12, val[:32], size=10, fill=TEXT_DARK))
            y += 23
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _plot_frame(x0: float, y0: float, w: float, h: float, *, axes: bool = True, xlabel: str = "", ylabel: str = "") -> list[str]:
    elems = [
        _el("rect", {"x": _fmt(x0), "y": _fmt(y0), "width": _fmt(w), "height": _fmt(h), "rx": 6,
                     "fill": PANEL_FILL, "stroke": PANEL_STROKE, "stroke-width": 1}),
    ]
    if axes:
        for i in range(1, 4):
            gx = x0 + w * i / 4.0
            gy = y0 + h * i / 4.0
            elems.append(_el("line", {"x1": _fmt(gx), "y1": _fmt(y0), "x2": _fmt(gx), "y2": _fmt(y0 + h),
                                      "stroke": GRID_STROKE, "stroke-width": 1}))
            elems.append(_el("line", {"x1": _fmt(x0), "y1": _fmt(gy), "x2": _fmt(x0 + w), "y2": _fmt(gy),
                                      "stroke": GRID_STROKE, "stroke-width": 1}))
        # Crosshair axes
        elems.append(_el("line", {"x1": _fmt(x0), "y1": _fmt(y0 + h / 2), "x2": _fmt(x0 + w), "y2": _fmt(y0 + h / 2),
                                  "stroke": AXIS_STROKE, "stroke-width": 0.9, "stroke-dasharray": "3,3"}))
        elems.append(_el("line", {"x1": _fmt(x0 + w / 2), "y1": _fmt(y0), "x2": _fmt(x0 + w / 2), "y2": _fmt(y0 + h),
                                  "stroke": AXIS_STROKE, "stroke-width": 0.9, "stroke-dasharray": "3,3"}))
    if xlabel:
        elems.append(_text(x0 + w / 2, y0 + h - 4, xlabel, size=9, fill=TEXT_LIGHT, anchor="middle"))
    if ylabel:
        elems.append(_text(x0 + 6, y0 + 13, ylabel, size=9, fill=TEXT_LIGHT))
    return elems


def _map_2d(pts: list[tuple[float, float]], x0: float, y0: float, w: float, h: float, pad: float = 0.16):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    span = max(xmax - xmin, ymax - ymin, 1e-3)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * span * (1.0 + 2.0 * pad)
    scale = min(w, h) / (2.0 * half)
    mx = x0 + 0.5 * w
    my = y0 + 0.5 * h

    def to_px(x: float, y: float) -> tuple[float, float]:
        return mx + (x - cx) * scale, my - (y - cy) * scale

    return to_px, scale


def _eig2x2(a: float, b: float, d: float) -> tuple[float, float, float]:
    """Return (sigma_major, sigma_minor, angle_deg_svg) for 2x2 SPD covariance [[a, b], [b, d]]."""
    tr = a + d
    diff = a - d
    disc = math.sqrt(max(0.0, diff * diff + 4.0 * b * b))
    l1 = max(1e-6, 0.5 * (tr + disc))
    l2 = max(1e-6, 0.5 * (tr - disc))
    theta = 0.5 * math.atan2(2.0 * b, diff)
    # SVG y-axis is inverted
    return math.sqrt(l1), math.sqrt(l2), -math.degrees(theta)


def _strip_variant_prefix(task_id: str) -> str:
    base = task_id
    if base.startswith("recipe-prior-refactor-legacy-"):
        base = base[len("recipe-prior-refactor-legacy-"):]
    for suffix in (
        "_published",
        "_residual16",
        "_w64_optimizer_diagnostic",
        "_affine_square_named_v1_14k",
        "_affine_square_named_v1",
        "_affine_paired_laws_v1",
        "_release07_cloud_named_v1",
        "_14k",
    ):
        if base.endswith(suffix) and len(base) > len(suffix):
            base = base[:-len(suffix)]
            break
    return base


def _task_callouts(task: dict, doc: dict) -> list[tuple[str, str]]:
    ex = task.get("execution", {})
    ev = task.get("evaluation", {})
    prior = ex.get("prior", {})
    hd = ex.get("host_definition", {}) if isinstance(ex.get("host_definition"), dict) else {}
    res = ex.get("resources") or hd.get("resources") or {}
    prior_desc = f"{prior.get('kind', 'mog')} (sigma={prior.get('sigma', 'default')})"
    steps = ex.get("steps", hd.get("steps", "?"))
    batch = ex.get("batch_size") or res.get("batch_size") or hd.get("batch") or hd.get("batch_size") or "host"
    particles = res.get("num_particles") or hd.get("particles") or "host"
    z_dim = res.get("z_dim") or hd.get("z_dim") or "host"
    laws = ", ".join(doc.get("laws", [])[:3]) or "general"
    lines = [
        ("Task ID", task.get("id", "?")),
        ("Laws Targeted", laws),
        ("Prior", prior_desc),
        ("Budget / Batch", f"{steps} steps · B={batch}"),
        ("Latent", f"P={particles} · z_dim={z_dim}"),
        ("Eval Law", str(ev.get("eval_output_noise", "clean"))[:28]),
        ("Adapter", str(task.get("adapter", "?"))),
    ]
    if ex.get("continuation_of"):
        lines.append(("Continues", str(ex["continuation_of"])))
    elif ex.get("shift_step") or hd.get("shift_step"):
        lines.append(("Shift Step", str(ex.get("shift_step") or hd.get("shift_step"))))
    return lines


def _render_grid100(task: dict, doc: dict, mode: str) -> str:
    x0, y0, w, h = 18, 54, 386, 248
    body = _plot_frame(x0, y0, w, h, axes=True, xlabel="x1 (10x10 modes, 33.3-sigma valleys)", ylabel="x2")
    pts: list[tuple[float, float]] = []
    coords = [(-4.5 + i) for i in range(10)]
    if mode == "rotated":
        ang = math.radians(30.0)
        ca, sa = math.cos(ang), math.sin(ang)
        for ix in coords:
            for iy in coords:
                pts.append((ix * ca - iy * sa, ix * sa + iy * ca))
    elif mode == "staggered":
        for r, iy in enumerate(coords):
            shift = 0.5 if (r % 2 == 1) else 0.0
            for ix in coords:
                pts.append((ix + shift, iy))
    else:
        for ix in coords:
            for iy in coords:
                pts.append((ix, iy))
    to_px, _ = _map_2d(pts, x0, y0, w, h, pad=0.14)
    color = INDIGO if mode == "staggered" else (TEAL if mode == "rotated" else BLUE)
    body.append(_dots_el([to_px(px, py) for px, py in pts], 3.0, color))
    # Annotate special variants
    tid = task["id"]
    tag = None
    if "pacgan8" in tid:
        tag = "PacGAN m=8 packed critic D(x1..x8)"
    elif "14k" in tid:
        tag = "Continuation: 7k -> 14k endurance"
    elif "affine" in tid:
        tag = "Affine-init square cohort"
    elif "release07" in tid:
        tag = "Release 0.7 zero-sigma cloud cohort"
    if tag:
        body.append(_el("rect", {"x": 28, "y": 274, "width": 236, "height": 20, "rx": 4,
                                 "fill": "#eff6ff", "stroke": "#93c5fd"}))
        body.append(_text(36, 288, tag, size=9, fill="#1d4ed8", weight="600"))
    return _wrap_svg(doc.get("title", tid), f"100-mode 2D Gaussian lattice ({mode}) · sigma=0.03-0.05",
                     body, _task_callouts(task, doc))


def _render_gaussian_mixture_2d(task: dict, doc: dict) -> str:
    tid = task["id"]
    base = _strip_variant_prefix(tid)
    hd = task.get("execution", {}).get("host_definition", {})
    if not isinstance(hd, dict):
        hd = {}

    if base in ("grid100", "pacgan8_grid100"):
        return _render_grid100(task, doc, "square")
    if base == "rotated100":
        return _render_grid100(task, doc, "rotated")
    if base == "staggered100":
        return _render_grid100(task, doc, "staggered")

    x0, y0, w, h = 18, 54, 386, 248
    body = _plot_frame(x0, y0, w, h, axes=True, xlabel="x1", ylabel="x2")

    # 1. Spiral / ALICE spiral BiGAN
    if base in ("vector_spiral", "alice_spiral_bigan"):
        pts = []
        n_pts = 80
        if base == "alice_spiral_bigan":
            tmin, tmax, r0, slope = 0.5 * math.pi, 3.5 * math.pi, 0.5, 0.25
        else:
            tmin, tmax, r0, slope = 0.0, 3.0 * math.pi, 0.3, (2.0 - 0.3) / (3.0 * math.pi)
        for i in range(n_pts):
            t = tmin + (tmax - tmin) * i / (n_pts - 1)
            r = r0 + slope * (t - tmin) if base == "vector_spiral" else r0 + slope * t
            pts.append((r * math.cos(t), r * math.sin(t)))
        to_px, scale = _map_2d(pts, x0, y0, w, h, pad=0.18)
        poly = " ".join(f"{_fmt(to_px(px, py)[0])},{_fmt(to_px(px, py)[1])}" for px, py in pts)
        body.append(_el("polyline", {"points": poly, "fill": "none", "stroke": BLUE,
                                     "stroke-width": "12", "stroke-opacity": "0.18", "stroke-linecap": "round"}))
        body.append(_el("polyline", {"points": poly, "fill": "none", "stroke": BLUE,
                                     "stroke-width": "2.4", "stroke-linecap": "round"}))
        # Arc-bin markers
        for k in range(0, n_pts, 5):
            sx, sy = to_px(*pts[k])
            body.append(_el("circle", {"cx": _fmt(sx), "cy": _fmt(sy), "r": "2.6", "fill": TEAL}))
        if base == "alice_spiral_bigan":
            body.append(_el("rect", {"x": 28, "y": 274, "width": 250, "height": 20, "rx": 4,
                                     "fill": "#f0fdfa", "stroke": "#5eead4"}))
            body.append(_text(36, 288, "BiGAN + ALICE cycle: G(E(x)) ~ x, E(G(z)) ~ z", size=9, fill="#0f766e", weight="600"))
        return _wrap_svg(doc.get("title", tid), "Continuous 2D Archimedes spiral manifold + radial noise",
                         body, _task_callouts(task, doc))

    # 2. 270-degree circular arc + unseen k-NN chords
    if base == "interp_sparse_arc":
        n_arc = 72
        arc_pts = []
        for i in range(n_arc):
            th = -0.25 * math.pi + 1.5 * math.pi * i / (n_arc - 1)
            arc_pts.append((2.0 * math.cos(th), 2.0 * math.sin(th)))
        particles = []
        for k in range(12):
            th = -0.25 * math.pi + 1.5 * math.pi * k / 11.0
            particles.append((2.0 * math.cos(th), 2.0 * math.sin(th)))
        to_px, _ = _map_2d(arc_pts + [(-2.2, -2.2), (2.2, 2.2)], x0, y0, w, h, pad=0.14)
        poly = " ".join(f"{_fmt(to_px(px, py)[0])},{_fmt(to_px(px, py)[1])}" for px, py in arc_pts)
        body.append(_el("polyline", {"points": poly, "fill": "none", "stroke": BLUE,
                                     "stroke-width": "10", "stroke-opacity": "0.18", "stroke-linecap": "round"}))
        body.append(_el("polyline", {"points": poly, "fill": "none", "stroke": BLUE,
                                     "stroke-width": "2.2", "stroke-linecap": "round"}))
        # k-NN chords & unseen midpoint markers
        for k in range(len(particles) - 1):
            p1, p2 = to_px(*particles[k]), to_px(*particles[k + 1])
            body.append(_el("line", {"x1": _fmt(p1[0]), "y1": _fmt(p1[1]), "x2": _fmt(p2[0]), "y2": _fmt(p2[1]),
                                     "stroke": AMBER, "stroke-width": "1.5", "stroke-dasharray": "4,3"}))
            mx, my = 0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1])
            body.append(_el("circle", {"cx": _fmt(mx), "cy": _fmt(my), "r": "3.0", "fill": ROSE}))
        for px, py in particles:
            sx, sy = to_px(px, py)
            body.append(_el("circle", {"cx": _fmt(sx), "cy": _fmt(sy), "r": "4.2", "fill": BLUE,
                                       "stroke": "#ffffff", "stroke-width": "1.2"}))
        body.append(_text(28, 288, "Blue: 12 learned particles · Amber: unseen k-NN chords · Rose: midpoints", size=9, fill=TEXT_MUTED))
        return _wrap_svg(doc.get("title", tid), "270-degree circular arc (R=2.0, sigma=0.03) with 12 zero-sigma particles",
                         body, _task_callouts(task, doc))

    # 3. 64-atom noisy hexagonal Zipf codebook
    if base == "fl_law1_noisy_atomic_codebook":
        pts = []
        weights = []
        delta = 0.65
        for r in range(8):
            for c in range(8):
                x = (c - 3.5 + (0.5 if r % 2 else 0.0)) * delta
                y = (r - 3.5) * delta * math.sqrt(3.0) / 2.0
                pts.append((x, y))
                rank = ((r * 8 + c) * 19 + 7) % 64 + 1
                weights.append(rank ** (-0.5))
        wsum = sum(weights)
        weights = [wt / wsum for wt in weights]
        wmax = max(weights)
        to_px, _ = _map_2d(pts, x0, y0, w, h, pad=0.14)
        # Group into 3 Zipf radius tiers for ultra-compact round-cap paths
        tiers: dict[int, list[tuple[float, float]]] = {0: [], 1: [], 2: []}
        for (px, py), wt in zip(pts, weights):
            ratio = math.sqrt(wt / wmax)
            tier = 2 if ratio >= 0.72 else (1 if ratio >= 0.48 else 0)
            tiers[tier].append(to_px(px, py))
        for tier, rad in ((0, 2.4), (1, 3.8), (2, 5.4)):
            if tiers[tier]:
                body.append(_dots_el(tiers[tier], rad + 1.8, VIOLET, opacity="0.16"))
                body.append(_dots_el(tiers[tier], rad, VIOLET, opacity="0.86"))
        body.append(_text(28, 288, "64 hexagonal atoms · Zipf(s=0.5) 8:1 radii · jitter sigma/Delta=0.02 · P=96 != K=64",
                          size=9, fill=TEXT_MUTED))
        return _wrap_svg(doc.get("title", tid), "8x8 staggered hexagonal codebook (K=64 atoms, P=96 particles)",
                         body, _task_callouts(task, doc))

    # 4. Thin codimension-1 circle
    if base == "fl_law2_pure_adversarial_codim1":
        cx, cy, r_px = x0 + w * 0.5, y0 + h * 0.5, 84.0
        body.append(_el("circle", {"cx": _fmt(cx), "cy": _fmt(cy), "r": _fmt(r_px),
                                   "fill": "none", "stroke": BLUE, "stroke-width": "2.6"}))
        # Normal-axis oscillation arrows (Law 2 intersect Law 4.5)
        for k in range(8):
            th = k * math.pi / 4.0
            c, s = math.cos(th), math.sin(th)
            x1, y1 = cx + (r_px - 16) * c, cy - (r_px - 16) * s
            x2, y2 = cx + (r_px + 18) * c, cy - (r_px + 18) * s
            body.append(_el("line", {"x1": _fmt(x1), "y1": _fmt(y1), "x2": _fmt(x2), "y2": _fmt(y2),
                                     "stroke": ROSE, "stroke-width": "1.3", "marker-end": "url(#arrow-rose)",
                                     "marker-start": "url(#arrow-rose)"}))
        body.append(_text(cx, cy + 4, "R = 1.5, sigma_perp = 0", size=10, fill=TEXT_DARK, weight="600", anchor="middle"))
        body.append(_text(28, 288, "Rose arrows: normal-direction orbit oscillation damped only when mu_D > 0", size=9, fill=ROSE))
        return _wrap_svg(doc.get("title", tid), "Zero-thickness codim-1 circle in R^2 · normal orbit stability",
                         body, _task_callouts(task, doc))

    # 5. Shift protocols (target_shift_recovery & t3_iso_horizon_saddle_shift)
    if base in ("target_shift_recovery", "t3_iso_horizon_saddle_shift"):
        if base == "t3_iso_horizon_saddle_shift":
            pre = [(-1.5, 0.0), (1.5, 0.0)]
            post = [(0.0, -1.5), (0.0, 1.5)]
            to_px, _ = _map_2d(pre + post, x0, y0, w, h, pad=0.35)
            for px, py in pre:
                sx, sy = to_px(px, py)
                body.append(_el("circle", {"cx": _fmt(sx), "cy": _fmt(sy), "r": "14", "fill": BLUE, "fill-opacity": "0.24",
                                           "stroke": BLUE, "stroke-width": "2"}))
                body.append(_text(sx, sy + 26, f"({px:+.1f}, 0)", size=9, fill=BLUE, anchor="middle"))
            for px, py in post:
                sx, sy = to_px(px, py)
                body.append(_el("circle", {"cx": _fmt(sx), "cy": _fmt(sy), "r": "14", "fill": ROSE, "fill-opacity": "0.24",
                                           "stroke": ROSE, "stroke-width": "2", "stroke-dasharray": "4,2"}))
                body.append(_text(sx + 20, sy + 4, f"(0, {py:+.1f}) new @t=300", size=9, fill=ROSE))
            # Arrows from pre to orthogonal post
            p_l, p_u = to_px(-1.3, 0.2), to_px(-0.2, 1.3)
            p_r, p_d = to_px(1.3, -0.2), to_px(0.2, -1.3)
            body.append(_el("line", {"x1": _fmt(p_l[0]), "y1": _fmt(p_l[1]), "x2": _fmt(p_u[0]), "y2": _fmt(p_u[1]),
                                     "stroke": ROSE, "stroke-width": "1.6", "marker-end": "url(#arrow-rose)"}))
            body.append(_el("line", {"x1": _fmt(p_r[0]), "y1": _fmt(p_r[1]), "x2": _fmt(p_d[0]), "y2": _fmt(p_d[1]),
                                     "stroke": ROSE, "stroke-width": "1.6", "marker-end": "url(#arrow-rose)"}))
            body.append(_text(28, 288, "G = nn.Identity() · 42.4-sigma valley · paired step-300 frozen control", size=9, fill=TEXT_MUTED))
            return _wrap_svg(doc.get("title", tid), "R^4 orthogonal 2 -> 4 mode expansion at step 300 (e1-e2 plane)",
                             body, _task_callouts(task, doc))
        else:
            # target_shift_recovery: 8-mode ring rotation / shift with frozen control
            pre = [(2.0 * math.cos(2 * math.pi * k / 8), 2.0 * math.sin(2 * math.pi * k / 8)) for k in range(8)]
            post = [(2.0 * math.cos(2 * math.pi * (k + 0.5) / 8), 2.0 * math.sin(2 * math.pi * (k + 0.5) / 8)) for k in range(8)]
            to_px, _ = _map_2d(pre + post, x0, y0, w, h, pad=0.25)
            for (ax, ay), (bx, by) in zip(pre, post):
                sa, sb = to_px(ax, ay), to_px(bx, by)
                body.append(_el("circle", {"cx": _fmt(sa[0]), "cy": _fmt(sa[1]), "r": "7", "fill": BLUE, "fill-opacity": "0.35",
                                           "stroke": BLUE, "stroke-width": "1.5"}))
                body.append(_el("circle", {"cx": _fmt(sb[0]), "cy": _fmt(sb[1]), "r": "7", "fill": ROSE, "fill-opacity": "0.35",
                                           "stroke": ROSE, "stroke-width": "1.5", "stroke-dasharray": "3,2"}))
                body.append(_el("line", {"x1": _fmt(sa[0]), "y1": _fmt(sa[1]), "x2": _fmt(sb[0]), "y2": _fmt(sb[1]),
                                         "stroke": AMBER, "stroke-width": "1.2", "marker-end": "url(#arrow)"}))
            body.append(_text(28, 288, "Blue: pre-shift target · Rose: post-shift target · Paired frozen control", size=9, fill=TEXT_MUTED))
            return _wrap_svg(doc.get("title", tid), "Paired non-stationary target shift recovery + frozen control",
                             body, _task_callouts(task, doc))

    # 6. Explicit Gaussian mixtures (rings, anisotropic, unequal mass/width, overlapping, rare 1%)
    means: list[list[float]] = []
    covs: list[list[list[float]]] = []
    masses: list[float] = []
    subtitle = "2D Gaussian mixture target geometry"
    extra_note = ""

    if base in ("ring16_acquisition", "ring16_bigan_continuous"):
        n_m, rad, sig = 16, 3.0, 0.10
        means = [[rad * math.cos(2 * math.pi * k / n_m), rad * math.sin(2 * math.pi * k / n_m)] for k in range(n_m)]
        covs = [[[sig * sig, 0.0], [0.0, sig * sig]] for _ in range(n_m)]
        masses = [1.0 / n_m] * n_m
        subtitle = "16-mode 2D Gaussian ring (R=3.0, sigma=0.10)"
        if base == "ring16_bigan_continuous":
            extra_note = "Joint BiGAN (G(z), E(x), D(x,z)) · mode_consistency >= 0.98"
    elif base in ("ring_hold", "ring_extension", "mode_hold"):
        n_m, rad, sig = 8, 2.0, 0.05
        means = [[rad * math.cos(2 * math.pi * k / n_m), rad * math.sin(2 * math.pi * k / n_m)] for k in range(n_m)]
        covs = [[[sig * sig, 0.0], [0.0, sig * sig]] for _ in range(n_m)]
        masses = [1.0 / n_m] * n_m
        subtitle = f"8-mode 2D Gaussian ring · {'endurance extension' if 'extension' in base else 'sustained mode hold'}"
    elif base == "fl_law1_aniso_unequal_modes":
        n_m, rad = 8, 3.0
        masses = [0.25, 0.25, 0.125, 0.125, 0.0625, 0.0625, 0.0625, 0.0625]
        s_maj, s_min = 0.25, 0.025
        for k in range(n_m):
            th = 2.0 * math.pi * k / n_m
            means.append([rad * math.cos(th), rad * math.sin(th)])
            rot = th + 0.5 * math.pi
            c, s = math.cos(rot), math.sin(rot)
            v1, v2 = s_maj * s_maj, s_min * s_min
            covs.append([[v1 * c * c + v2 * s * s, (v1 - v2) * c * s],
                         [(v1 - v2) * c * s, v1 * s * s + v2 * c * c]])
        subtitle = "8 rotated 10:1 ellipses on R=3.0 · 4:1 mass skew · B_f=16 < K"
        extra_note = "Ellipse size scales with weight (25%, 12.5%, 6.25%); 10:1 aspect ratio"
    elif base == "infogan_anisotropic_modes":
        n_m, rad = 8, 3.0
        masses = [0.125] * n_m
        s_maj, s_min = 0.28, 0.06
        for k in range(n_m):
            th = 2.0 * math.pi * k / n_m
            means.append([rad * math.cos(th), rad * math.sin(th)])
            c, s = math.cos(th), math.sin(th)
            v1, v2 = s_maj * s_maj, s_min * s_min
            covs.append([[v1 * c * c + v2 * s * s, (v1 - v2) * c * s],
                         [(v1 - v2) * c * s, v1 * s * s + v2 * c * c]])
        subtitle = "InfoGAN: 8 anisotropic modes (c_disc=mode, c_cont=long-axis coord)"
        extra_note = "Variational Q(c|x) enforces discrete NMI >= 0.85 & continuous R^2 >= 0.65"
    elif base == "fl_law41_trust_rare1pct_tanh":
        means = [[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5], [1.5, 1.5], [0.0, 2.85]]
        covs = [[[0.08**2, 0.0], [0.0, 0.08**2]] for _ in range(5)]
        masses = [0.2475, 0.2475, 0.2475, 0.2475, 0.01]
        subtitle = "4 major modes (99%) + 1% minority mode near 3*tanh rail"
        extra_note = "Dashed rose line: saturating 3.0*tanh boundary at y = +3.0"
    elif base == "t3_iso_scale_mass_8x":
        means = [[-1.2, -1.2], [-1.2, 1.2], [1.2, -1.2], [1.2, 1.2]]
        widths = [0.04, 0.08, 0.16, 0.32]
        covs = [[[s * s, 0.0], [0.0, s * s]] for s in widths]
        masses = [0.48, 0.26, 0.18, 0.08]
        subtitle = "G=Identity · 8x width disparity [0.04..0.32] & 6x mass [48%..8%]"
        extra_note = "Theory-1-free & Theory-2-verified isolation of Law 3A"
    elif base == "pacgan_unequal_mass":
        n_m, rad, sig = 5, 2.0, 0.08
        means = [[rad * math.cos(2 * math.pi * k / n_m), rad * math.sin(2 * math.pi * k / n_m)] for k in range(n_m)]
        covs = [[[sig * sig, 0.0], [0.0, sig * sig]] for _ in range(n_m)]
        masses = [0.40, 0.30, 0.20, 0.08, 0.02]
        subtitle = "PacGAN m=4 packed critic · 5-mode ring with rare 2% mode"
    elif base == "cgan_overlapping_classes":
        means = [[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]]
        covs = [[[0.35**2, 0.0], [0.0, 0.35**2]] for _ in range(4)]
        masses = [0.25] * 4
        subtitle = "Projection cGAN · 4 overlapping classes (sigma=0.35)"
        extra_note = "Colors = class labels y in {0,1,2,3}; tests overlap density preservation"
    elif base == "cramer_energy_overlap":
        means = [[x, y] for x in (-1.5, 0.0, 1.5) for y in (-1.5, 0.0, 1.5)]
        covs = [[[0.16**2, 0.0], [0.0, 0.16**2]] for _ in range(9)]
        masses = [0.18, 0.08, 0.18, 0.08, 0.16, 0.08, 0.10, 0.06, 0.08]
        subtitle = "CramerGAN energy distance · 3x3 overlapping unequal-mass mixture"
    elif isinstance(hd.get("means"), list) and hd["means"] and len(hd["means"][0]) == 2:
        means = [[float(m[0]), float(m[1])] for m in hd["means"]]
        masses = [float(m) for m in hd.get("masses", [1.0 / len(means)] * len(means))]
        if isinstance(hd.get("covariances"), list):
            covs = hd["covariances"]
        else:
            sig = float(hd.get("sigma", 0.15))
            covs = [[[sig * sig, 0.0], [0.0, sig * sig]] for _ in means]
        subtitle = str(hd.get("target") or hd.get("family") or base)[:75]
    else:
        means = [[-1.0, 0.0], [1.0, 0.0]]
        covs = [[[0.25**2, 0.0], [0.0, 0.25**2]]] * 2
        masses = [0.5, 0.5]

    # Map points into plot box
    bounds_pts = [(m[0], m[1]) for m in means]
    if base == "fl_law41_trust_rare1pct_tanh":
        bounds_pts.extend([(-3.1, -3.1), (3.1, 3.1)])
    else:
        # Include 2-sigma extent
        for m, cov in zip(means, covs):
            sx = math.sqrt(max(cov[0][0], 1e-4))
            sy = math.sqrt(max(cov[1][1], 1e-4))
            bounds_pts.append((m[0] - 2.5 * sx, m[1] - 2.5 * sy))
            bounds_pts.append((m[0] + 2.5 * sx, m[1] + 2.5 * sy))
    to_px, scale = _map_2d(bounds_pts, x0, y0, w, h, pad=0.18)

    # Optional 3.0*tanh saturation rails
    if base == "fl_law41_trust_rare1pct_tanh":
        p_tl, p_br = to_px(-3.0, 3.0), to_px(3.0, -3.0)
        body.append(_el("rect", {"x": _fmt(p_tl[0]), "y": _fmt(p_tl[1]),
                                 "width": _fmt(p_br[0] - p_tl[0]), "height": _fmt(p_br[1] - p_tl[1]),
                                 "fill": "none", "stroke": ROSE, "stroke-width": "1.3", "stroke-dasharray": "5,3"}))

    max_m = max(masses) if masses else 1.0
    show_mass_labels = (max_m / max(min(masses), 1e-6)) > 1.3 and len(means) <= 9
    for idx, (m, cov, wt) in enumerate(zip(means, covs, masses)):
        cx, cy = to_px(m[0], m[1])
        s1, s2, deg = _eig2x2(cov[0][0], cov[0][1], cov[1][1])
        # Visual radius clamped cleanly so tiny sigma stays legible while ratios stay clear
        rx = max(4.5, min(65.0, s1 * scale * 2.2))
        ry = max(2.2, min(65.0, s2 * scale * 2.2))
        if show_mass_labels and abs(s1 - s2) < 1e-3 and base not in ("vector_unequal_width", "t3_iso_scale_mass_8x"):
            # Scale disk mildly with sqrt(mass) for unequal-mass clarity
            fac = 0.55 + 0.85 * math.sqrt(wt / max_m)
            rx = max(5.0, rx * fac)
            ry = max(5.0, ry * fac)
        col = CLASS_COLORS[idx % len(CLASS_COLORS)] if base in ("cgan_overlapping_classes", "infogan_anisotropic_modes") else BLUE
        if base == "fl_law41_trust_rare1pct_tanh" and idx == 4:
            col = ROSE
        # Outer 2-sigma halo + inner 1-sigma ellipse
        tf = f"rotate({_fmt(deg)} {_fmt(cx)} {_fmt(cy)})" if abs(deg) > 0.5 else None
        body.append(_el("ellipse", {"cx": _fmt(cx), "cy": _fmt(cy), "rx": _fmt(rx), "ry": _fmt(ry),
                                    "fill": col, "fill-opacity": "0.18", "stroke": col, "stroke-width": "1.5",
                                    "transform": tf}))
        body.append(_el("ellipse", {"cx": _fmt(cx), "cy": _fmt(cy), "rx": _fmt(max(2.0, rx * 0.45)),
                                    "ry": _fmt(max(1.5, ry * 0.45)), "fill": col, "fill-opacity": "0.65",
                                    "transform": tf}))
        if show_mass_labels:
            lbl = f"{wt * 100:.0f}%" if wt >= 0.015 else f"{wt * 100:.1f}%"
            body.append(_text(cx, cy - ry - 4, lbl, size=9, fill=col, weight="600", anchor="middle"))
        elif base == "vector_unequal_width":
            body.append(_text(cx, cy - ry - 4, f"s={s1:.2f}", size=9, fill=TEXT_DARK, weight="600", anchor="middle"))

    if extra_note:
        body.append(_text(28, 288, extra_note[:72], size=9, fill=TEXT_MUTED))
    return _wrap_svg(doc.get("title", tid), subtitle, body, _task_callouts(task, doc))


def _render_gaussian_1d_shift(task: dict, doc: dict) -> str:
    tid = task["id"]
    base = _strip_variant_prefix(tid)
    x0, y0, w, h = 18, 54, 386, 248
    body = _plot_frame(x0, y0, w, h, axes=False)

    if base in ("zero_tanh_smoke", "zero_tanh_rail_escape"):
        # 2D tanh box [-1, 1]^2 with saturation rails and origin target
        bx, by, bw, bh = x0 + 68, y0 + 24, 210, 196
        body.append(_el("rect", {"x": _fmt(bx), "y": _fmt(by), "width": _fmt(bw), "height": _fmt(bh),
                                 "fill": "#fff1f2", "fill-opacity": "0.35", "stroke": ROSE,
                                 "stroke-width": "2", "stroke-dasharray": "5,3"}))
        body.append(_el("line", {"x1": _fmt(bx), "y1": _fmt(by + bh / 2), "x2": _fmt(bx + bw), "y2": _fmt(by + bh / 2),
                                 "stroke": AXIS_STROKE, "stroke-width": "1"}))
        body.append(_el("line", {"x1": _fmt(bx + bw / 2), "y1": _fmt(by), "x2": _fmt(bx + bw / 2), "y2": _fmt(by + bh),
                                 "stroke": AXIS_STROKE, "stroke-width": "1"}))
        cx, cy = bx + bw / 2, by + bh / 2
        body.append(_el("circle", {"cx": _fmt(cx), "cy": _fmt(cy), "r": "9", "fill": EMERALD, "fill-opacity": "0.3",
                                   "stroke": EMERALD, "stroke-width": "2"}))
        body.append(_el("circle", {"cx": _fmt(cx), "cy": _fmt(cy), "r": "3", "fill": EMERALD}))
        body.append(_text(cx + 12, cy - 8, "Target (0, 0)", size=10, fill=EMERALD, weight="700"))
        body.append(_text(bx + bw + 6, by + 12, "tanh rail +1", size=9, fill=ROSE))
        body.append(_text(bx + bw + 6, by + bh - 4, "tanh rail -1", size=9, fill=ROSE))
        if base == "zero_tanh_rail_escape":
            rx, ry = bx + bw * 0.92, by + bh * 0.92
            body.append(_el("circle", {"cx": _fmt(rx), "cy": _fmt(ry), "r": "8", "fill": ROSE, "fill-opacity": "0.35",
                                       "stroke": ROSE, "stroke-width": "2"}))
            body.append(_text(rx - 14, ry + 16, "Phase 1 (0.9, -0.9)", size=9, fill=ROSE, anchor="end", weight="600"))
            body.append(_el("line", {"x1": _fmt(rx - 8), "y1": _fmt(ry - 8), "x2": _fmt(cx + 10), "y2": _fmt(cy + 10),
                                     "stroke": AMBER, "stroke-width": "2.2", "marker-end": "url(#arrow)"}))
            body.append(_text(cx + 55, cy + 44, "Step 1001 escape", size=9, fill=AMBER, weight="700"))
        return _wrap_svg(doc.get("title", tid), "Point-mass target under saturating tanh output rails",
                         body, _task_callouts(task, doc))

    # 1D Gaussian bell curve (+ optional shift or precision badge)
    base_y = y0 + h - 42
    body.append(_el("line", {"x1": _fmt(x0 + 22), "y1": _fmt(base_y), "x2": _fmt(x0 + w - 22), "y2": _fmt(base_y),
                             "stroke": AXIS_STROKE, "stroke-width": "1.2"}))

    def bell_points(mu: float, sig: float, xmin: float, xmax: float, peak_h: float) -> str:
        pts = []
        for i in range(65):
            xv = xmin + (xmax - xmin) * i / 64.0
            yv = math.exp(-0.5 * ((xv - mu) / sig) ** 2)
            px = x0 + 26 + (w - 52) * (xv - xmin) / (xmax - xmin)
            py = base_y - peak_h * yv
            pts.append(f"{_fmt(px)},{_fmt(py)}")
        return " ".join(pts)

    is_shift = "stability" in base
    xmin, xmax = (-3.5, 4.5) if is_shift else (0.0, 4.0)
    pts1 = bell_points(2.0, 0.5, xmin, xmax, 130.0)
    fill1 = f"{_fmt(x0 + 26)},{_fmt(base_y)} {pts1} {_fmt(x0 + w - 26)},{_fmt(base_y)}"
    body.append(_el("polygon", {"points": fill1, "fill": BLUE, "fill-opacity": "0.18"}))
    body.append(_el("polyline", {"points": pts1, "fill": "none", "stroke": BLUE, "stroke-width": "2.4"}))
    mu1_x = x0 + 26 + (w - 52) * (2.0 - xmin) / (xmax - xmin)
    body.append(_text(mu1_x, base_y - 136, "N(2.0, 0.5^2)", size=10, fill=BLUE, weight="700", anchor="middle"))

    if is_shift:
        pts2 = bell_points(-1.5, 0.5, xmin, xmax, 130.0)
        fill2 = f"{_fmt(x0 + 26)},{_fmt(base_y)} {pts2} {_fmt(x0 + w - 26)},{_fmt(base_y)}"
        body.append(_el("polygon", {"points": fill2, "fill": ROSE, "fill-opacity": "0.18"}))
        body.append(_el("polyline", {"points": pts2, "fill": "none", "stroke": ROSE, "stroke-width": "2.4",
                                     "stroke-dasharray": "5,3"}))
        mu2_x = x0 + 26 + (w - 52) * (-1.5 - xmin) / (xmax - xmin)
        body.append(_text(mu2_x, base_y - 136, "Shifted target", size=10, fill=ROSE, weight="700", anchor="middle"))
        body.append(_el("line", {"x1": _fmt(mu1_x - 24), "y1": _fmt(base_y - 80),
                                 "x2": _fmt(mu2_x + 24), "y2": _fmt(base_y - 80),
                                 "stroke": AMBER, "stroke-width": "2.2", "marker-end": "url(#arrow)"}))
        body.append(_text(0.5 * (mu1_x + mu2_x), base_y - 88, "Stationary hold -> Shift -> Reacquire",
                          size=9, fill=AMBER, weight="700", anchor="middle"))

    if "precision_gaussian_" in base:
        fmt_tag = "FP32 baseline" if "fp32" in base else ("MXFP8 (E4M3 + block scale)" if "mxfp8" in base else "NVFP4 (E2M1 + micro-scale)")
        col = TEAL if "fp32" in base else (VIOLET if "mxfp8" in base else ROSE)
        body.append(_el("rect", {"x": x0 + 24, "y": y0 + 14, "width": 230, "height": 24, "rx": 5,
                                 "fill": "#f5f3ff", "stroke": col, "stroke-width": "1.2"}))
        body.append(_text(x0 + 34, y0 + 30, f"Quantized FPROP/DGRAD/WGRAD: {fmt_tag}", size=9, fill=col, weight="700"))

    body.append(_text(x0 + w / 2, base_y + 24, "Scalar x · gates: mean_err<=0.2s, std in [0.8, 1.2], KS<=0.05",
                      size=9, fill=TEXT_MUTED, anchor="middle"))
    return _wrap_svg(doc.get("title", tid), "1D scalar Gaussian acquisition / stability protocol",
                     body, _task_callouts(task, doc))


def _render_saddle_3d(task: dict, doc: dict) -> str:
    x0, y0, w, h = 18, 54, 386, 248
    body = _plot_frame(x0, y0, w, h, axes=False)

    def surf_z(x: float, y: float) -> float:
        return 0.8 * (x * x - y * y) + 0.9 * math.cos(0.75 * math.pi * x) * math.cos(0.75 * math.pi * y)

    def proj(x: float, y: float, z: float) -> tuple[float, float]:
        # Clean isometric projection
        px = x0 + w * 0.5 + 95.0 * (x - 0.85 * y)
        py = y0 + h * 0.56 - 46.0 * z + 34.0 * (x + 0.85 * y)
        return px, py

    grid = [-1.0 + 2.0 * i / 10.0 for i in range(11)]
    for x in grid:
        pts = " ".join(f"{_fmt(proj(x, y, surf_z(x, y))[0])},{_fmt(proj(x, y, surf_z(x, y))[1])}" for y in grid)
        body.append(_el("polyline", {"points": pts, "fill": "none", "stroke": BLUE, "stroke-width": "1.0", "stroke-opacity": "0.45"}))
    for y in grid:
        pts = " ".join(f"{_fmt(proj(x, y, surf_z(x, y))[0])},{_fmt(proj(x, y, surf_z(x, y))[1])}" for x in grid)
        body.append(_el("polyline", {"points": pts, "fill": "none", "stroke": TEAL, "stroke-width": "1.0", "stroke-opacity": "0.45"}))

    # Highlight a 2D k-NN triangle & its barycenter on the surface
    tri = [(-0.4, -0.4), (0.4, -0.2), (-0.1, 0.5)]
    tpx = [proj(tx, ty, surf_z(tx, ty)) for tx, ty in tri]
    body.append(_el("polygon", {"points": " ".join(f"{_fmt(p[0])},{_fmt(p[1])}" for p in tpx),
                                "fill": AMBER, "fill-opacity": "0.35", "stroke": AMBER, "stroke-width": "2"}))
    for p in tpx:
        body.append(_el("circle", {"cx": _fmt(p[0]), "cy": _fmt(p[1]), "r": "4", "fill": BLUE}))
    bx = sum(t[0] for t in tri) / 3.0
    by = sum(t[1] for t in tri) / 3.0
    bp = proj(bx, by, surf_z(bx, by))
    body.append(_el("circle", {"cx": _fmt(bp[0]), "cy": _fmt(bp[1]), "r": "5", "fill": ROSE,
                               "stroke": "#ffffff", "stroke-width": "1.2"}))
    body.append(_text(28, 288, "Blue: 24 particles · Amber: 2D k-NN triangle · Rose: unseen barycenter", size=9, fill=TEXT_MUTED))
    return _wrap_svg(doc.get("title", task["id"]), "Non-harmonic 3D saddle sheet w(x,y) on [-1,1]^2 + barycenters",
                     body, _task_callouts(task, doc))


def _render_image_grid_8x8(task: dict, doc: dict) -> str:
    tid = task["id"]
    base = _strip_variant_prefix(tid)
    x0, y0, w, h = 18, 54, 386, 248
    body = _plot_frame(x0, y0, w, h, axes=False)

    # Build 8x8 templates directly from image_tasks.py geometry
    grids: list[tuple[str, list[list[float]]]] = []
    if base == "img_stripes2":
        g0 = [[1.0 if r in (3, 4) else 0.0 for c in range(8)] for r in range(8)]
        g1 = [[1.0 if c in (3, 4) else 0.0 for c in range(8)] for r in range(8)]
        grids = [("Horizontal", g0), ("Vertical", g1)]
    elif base == "img_bars4":
        grids = [
            ("Col 1-2", [[1.0 if c in (1, 2) else 0.0 for c in range(8)] for r in range(8)]),
            ("Col 5-6", [[1.0 if c in (5, 6) else 0.0 for c in range(8)] for r in range(8)]),
            ("Row 1-2", [[1.0 if r in (1, 2) else 0.0 for c in range(8)] for r in range(8)]),
            ("Row 5-6", [[1.0 if r in (5, 6) else 0.0 for c in range(8)] for r in range(8)]),
        ]
    elif base == "img_blobs4":
        for label, (rr, cc) in (("Top-Left", (1, 1)), ("Top-Right", (1, 5)), ("Bot-Left", (5, 1)), ("Bot-Right", (5, 5))):
            grids.append((label, [[1.0 if rr <= r < rr + 2 and cc <= c < cc + 2 else 0.0 for c in range(8)] for r in range(8)]))
    elif base == "img_intensity2":
        grids = [
            ("Dim (0.35)", [[0.35 if 2 <= r < 6 and 2 <= c < 6 else 0.0 for c in range(8)] for r in range(8)]),
            ("Bright (0.85)", [[0.85 if 2 <= r < 6 and 2 <= c < 6 else 0.0 for c in range(8)] for r in range(8)]),
        ]
    elif base == "interp_img_patch_shift":
        grids = [
            ("z_a (t=0)", [[1.0 if 1 <= r < 3 and 1 <= c < 3 else 0.0 for c in range(8)] for r in range(8)]),
            ("Chord t=0.5 (sharp!)", [[1.0 if 3 <= r < 5 and 3 <= c < 5 else 0.0 for c in range(8)] for r in range(8)]),
            ("z_b (t=1)", [[1.0 if 5 <= r < 7 and 5 <= c < 7 else 0.0 for c in range(8)] for r in range(8)]),
        ]

    n = len(grids)
    cell = 10.0 if n >= 4 else (12.0 if n == 3 else 14.0)
    box_w = 8 * cell
    gap = (w - n * box_w) / (n + 1)
    gy = y0 + (h - box_w) * 0.44
    for idx, (label, mat) in enumerate(grids):
        gx = x0 + gap + idx * (box_w + gap)
        body.append(_el("rect", {"x": _fmt(gx - 2), "y": _fmt(gy - 2), "width": _fmt(box_w + 4), "height": _fmt(box_w + 4),
                                 "fill": "#0f172a", "rx": 3}))
        by_fill: dict[str, list[str]] = {}
        cs = cell - 0.5
        for r in range(8):
            for c in range(8):
                val = mat[r][c]
                if val > 0.0:
                    shade = int(round(255 * val))
                    fill = f"rgb({shade},{shade},{shade})"
                    by_fill.setdefault(fill, []).append(
                        f"M{_fmt(gx + c * cell)},{_fmt(gy + r * cell)}h{_fmt(cs)}v{_fmt(cs)}h-{_fmt(cs)}z"
                    )
        for fill, segs in by_fill.items():
            body.append(_el("path", {"d": "".join(segs), "fill": fill}))
        body.append(_text(gx + box_w / 2, gy + box_w + 18, label, size=10, fill=TEXT_DARK, weight="600", anchor="middle"))
        if base == "interp_img_patch_shift" and idx < n - 1:
            ax1 = gx + box_w + 6
            ax2 = gx + box_w + gap - 6
            body.append(_el("line", {"x1": _fmt(ax1), "y1": _fmt(gy + box_w / 2),
                                     "x2": _fmt(ax2), "y2": _fmt(gy + box_w / 2),
                                     "stroke": AMBER, "stroke-width": "2", "marker-end": "url(#arrow)"}))

    note = "Unseen latent k-NN chord must shift 1 sharp peak (no double-exposure ghosting)" if base == "interp_img_patch_shift" else "8x8 grayscale templates · exact finite-prior enumeration"
    body.append(_text(x0 + w / 2, y0 + h - 14, note, size=9, fill=TEXT_MUTED, anchor="middle"))
    return _wrap_svg(doc.get("title", tid), f"Discrete 1x8x8 procedural image target ({n} modes/frames)",
                     body, _task_callouts(task, doc))


def _render_rgb_sprites(task: dict, doc: dict) -> str:
    x0, y0, w, h = 18, 54, 386, 248
    body = _plot_frame(x0, y0, w, h, axes=False)
    labels = ("Red Ring", "Green Cross", "Blue Frame", "Gold Diamond")
    sz = 68.0
    gap = (w - 4 * sz) / 5.0
    sy = y0 + 36
    for i, lbl in enumerate(labels):
        sx = x0 + gap + i * (sz + gap)
        body.append(_el("rect", {"x": _fmt(sx), "y": _fmt(sy), "width": _fmt(sz), "height": _fmt(sz),
                                 "rx": 4, "fill": "#0f172a"}))
        cx, cy = sx + sz / 2, sy + sz / 2
        if i == 0:
            body.append(_el("circle", {"cx": _fmt(cx), "cy": _fmt(cy), "r": "20", "fill": "none", "stroke": "#ef4444", "stroke-width": "7"}))
        elif i == 1:
            body.append(_el("rect", {"x": _fmt(cx - 5), "y": _fmt(sy + 12), "width": "10", "height": _fmt(sz - 24), "fill": "#22c55e"}))
            body.append(_el("rect", {"x": _fmt(sx + 12), "y": _fmt(cy - 5), "width": _fmt(sz - 24), "height": "10", "fill": "#22c55e"}))
        elif i == 2:
            body.append(_el("rect", {"x": _fmt(sx + 12), "y": _fmt(sy + 12), "width": _fmt(sz - 24), "height": _fmt(sz - 24),
                                     "fill": "none", "stroke": "#3b82f6", "stroke-width": "7"}))
        else:
            pts = f"{_fmt(cx)},{_fmt(sy+11)} {_fmt(sx+sz-11)},{_fmt(cy)} {_fmt(cx)},{_fmt(sy+sz-11)} {_fmt(sx+11)},{_fmt(cy)}"
            body.append(_el("polygon", {"points": pts, "fill": "#eab308"}))
        body.append(_text(cx, sy + sz + 16, lbl, size=9, fill=TEXT_DARK, weight="600", anchor="middle"))

    # ResNet+GN + ViT Critic flow bar
    by = y0 + 156
    boxes = [("32x32x3 RGB", BLUE), ("ResNet + GroupNorm", TEAL), ("Patch-4 ViT (SDPA)", VIOLET), ("Critic Score", ROSE)]
    bw = 80.0
    bgap = (w - 4 * bw) / 5.0
    for i, (txt, col) in enumerate(boxes):
        bx = x0 + bgap + i * (bw + bgap)
        body.append(_el("rect", {"x": _fmt(bx), "y": _fmt(by), "width": _fmt(bw), "height": "34", "rx": 6,
                                 "fill": "#f8fafc", "stroke": col, "stroke-width": "1.6"}))
        body.append(_text(bx + bw / 2, by + 21, txt, size=8, fill=col, weight="700", anchor="middle"))
        if i < len(boxes) - 1:
            body.append(_el("line", {"x1": _fmt(bx + bw), "y1": _fmt(by + 17), "x2": _fmt(bx + bw + bgap - 2), "y2": _fmt(by + 17),
                                     "stroke": SLATE, "stroke-width": "1.4", "marker-end": "url(#arrow)"}))
    body.append(_text(x0 + w / 2, y0 + h - 16, "Tests Law 2 fwAD vs RoR compatibility on GroupNorm + Attention critics",
                      size=9, fill=TEXT_MUTED, anchor="middle"))
    return _wrap_svg(doc.get("title", task["id"]), "32x32 4-class RGB sprites with ResNet+GroupNorm + ViT critic",
                     body, _task_callouts(task, doc))


def _render_dcgan_scaling(task: dict, doc: dict) -> str:
    perf = task.get("execution", {}).get("performance", {})
    sz = perf.get("image_size", 64)
    bs = perf.get("batch_size", 16)
    ch = perf.get("base_channels", 64)
    warm = perf.get("warmup_updates", 3)
    meas = perf.get("measured_updates", 10)
    x0, y0, w, h = 18, 54, 386, 248
    body = _plot_frame(x0, y0, w, h, axes=False)

    # Resolution ladder
    sizes = [32, 64, 128, 256]
    cx = x0 + 30
    for s in sizes:
        side = {32: 24, 64: 38, 128: 56, 256: 78}[s]
        active = (s == sz)
        col = BLUE if active else "#cbd5e1"
        fill = "#dbeafe" if active else "#f1f5f9"
        by = y0 + 132 - side
        body.append(_el("rect", {"x": _fmt(cx), "y": _fmt(by), "width": _fmt(side), "height": _fmt(side),
                                 "rx": 4, "fill": fill, "stroke": col, "stroke-width": "2.2" if active else "1.2"}))
        body.append(_text(cx + side / 2, by + side + 16, f"{s}x{s}", size=10,
                          fill=BLUE if active else TEXT_LIGHT, weight="700" if active else "normal", anchor="middle"))
        cx += side + 24

    # Workload bar
    body.append(_el("rect", {"x": x0 + 24, "y": y0 + 166, "width": w - 48, "height": 58, "rx": 6,
                             "fill": "#f8fafc", "stroke": PANEL_STROKE}))
    body.append(_text(x0 + 36, y0 + 186, f"Active Workload: {sz}x{sz} RGB · Batch B={bs} · Width W={ch}",
                      size=10, fill=TEXT_DARK, weight="700"))
    body.append(_text(x0 + 36, y0 + 202, f"Timing: {warm} warmup + {meas} measured updates (mock Uniform[-1,1] data)",
                      size=9, fill=TEXT_MUTED))
    body.append(_text(x0 + 36, y0 + 216, "Measures ms/step, images/s, peak VRAM & NVML Joules · No image-quality gate",
                      size=9, fill=AMBER, weight="600"))
    return _wrap_svg(doc.get("title", task["id"]), f"DCGAN throughput & memory scaling ({sz}x{sz}, B={bs}, W={ch})",
                     body, _task_callouts(task, doc))


def _render_ddgan_chain(task: dict, doc: dict) -> str:
    tid = task["id"]
    has_ae = "ae_ddgan" in tid
    bimodal = "bimodal" in tid
    x0, y0, w, h = 18, 54, 386, 248
    body = _plot_frame(x0, y0, w, h, axes=False)

    # 4-step reverse diffusion chain x4 -> x3 -> x2 -> x1 -> x0
    nodes = [("x4 ~ N(0,I)", SLATE), ("x3", INDIGO), ("x2", BLUE), ("x1", TEAL), ("x0 (data)", EMERALD)]
    ny = y0 + 68
    nw = 58.0
    ngap = (w - 28 - len(nodes) * nw) / (len(nodes) - 1)
    for i, (lbl, col) in enumerate(nodes):
        nx = x0 + 14 + i * (nw + ngap)
        body.append(_el("rect", {"x": _fmt(nx), "y": _fmt(ny), "width": _fmt(nw), "height": "34", "rx": 6,
                                 "fill": "#ffffff", "stroke": col, "stroke-width": "1.8"}))
        body.append(_text(nx + nw / 2, ny + 21, lbl, size=9, fill=col, weight="700", anchor="middle"))
        if i < len(nodes) - 1:
            body.append(_el("line", {"x1": _fmt(nx + nw), "y1": _fmt(ny + 17),
                                     "x2": _fmt(nx + nw + ngap - 2), "y2": _fmt(ny + 17),
                                     "stroke": SLATE, "stroke-width": "1.4", "marker-end": "url(#arrow)"}))
    body.append(_text(x0 + w / 2, ny - 12, "4-Step Conditional Reverse Diffusion Chain G(x_t, z, t)",
                      size=10, fill=TEXT_DARK, weight="700", anchor="middle"))

    # Conditional critic + optional AE loop
    cy = y0 + 138
    body.append(_el("rect", {"x": x0 + 24, "y": _fmt(cy), "width": 162, "height": 42, "rx": 6,
                             "fill": "#fff1f2", "stroke": ROSE, "stroke-width": "1.5"}))
    body.append(_text(x0 + 105, cy + 17, "Conditional Critic", size=9, fill=ROSE, weight="700", anchor="middle"))
    body.append(_text(x0 + 105, cy + 31, "D(x_{t-1}, x_t, t) + Fiber Bank", size=8, fill=TEXT_MUTED, anchor="middle"))

    body.append(_el("rect", {"x": x0 + 202, "y": _fmt(cy), "width": 162, "height": 42, "rx": 6,
                             "fill": "#eff6ff" if not has_ae else "#f5f3ff",
                             "stroke": BLUE if not has_ae else VIOLET, "stroke-width": "1.5"}))
    if has_ae:
        body.append(_text(x0 + 283, cy + 17, "Particle AE Bridge", size=9, fill=VIOLET, weight="700", anchor="middle"))
        body.append(_text(x0 + 283, cy + 31, "x -> E(x) -> latent DDGAN -> Dec", size=8, fill=TEXT_MUTED, anchor="middle"))
    else:
        tgt_txt = "Bimodal (+-0.75, 0)" if bimodal else "Correlated 2D Gaussian"
        body.append(_text(x0 + 283, cy + 17, "Target Distribution", size=9, fill=BLUE, weight="700", anchor="middle"))
        body.append(_text(x0 + 283, cy + 31, tgt_txt, size=8, fill=TEXT_MUTED, anchor="middle"))

    # Gate strip
    body.append(_text(x0 + w / 2, y0 + h - 20, "Gates: ancestral marginal CDF/cov + conditioned t=1,2,3 query slices (Law 3B)",
                      size=9, fill=TEXT_MUTED, anchor="middle"))
    return _wrap_svg(doc.get("title", tid), f"{'AE + ' if has_ae else ''}4-step DDGAN conditional chain ({'bimodal' if bimodal else 'Gaussian'})",
                     body, _task_callouts(task, doc))


def _render_latent_loop(task: dict, doc: dict) -> str:
    tid = task["id"]
    base = _strip_variant_prefix(tid)
    x0, y0, w, h = 18, 54, 386, 248
    body = _plot_frame(x0, y0, w, h, axes=False)

    is_words = "five_word" in base
    is_vae = "vae" in base
    title_mid = "5-Word Joint BiGAN (G, E, D)" if is_words else ("Particle VAE (Posterior + KL + Prior)" if is_vae else "Particle AE / AE-GAN Loop")
    body.append(_text(x0 + w / 2, y0 + 24, title_mid, size=11, fill=TEXT_DARK, weight="700", anchor="middle"))

    boxes = [
        (x0 + 20, y0 + 52, 82, 42, "Data x", BLUE),
        (x0 + 150, y0 + 52, 86, 42, "Encoder E(x)", VIOLET),
        (x0 + 282, y0 + 52, 86, 42, "Latent z / P", TEAL),
        (x0 + 282, y0 + 134, 86, 42, "Decoder G(z)", EMERALD),
        (x0 + 150, y0 + 134, 86, 42, "Critic D(x,z)" if is_words else "Critic D(x)", ROSE),
    ]
    for bx, by, bw, bh, lbl, col in boxes:
        body.append(_el("rect", {"x": _fmt(bx), "y": _fmt(by), "width": _fmt(bw), "height": _fmt(bh),
                                 "rx": 6, "fill": "#ffffff", "stroke": col, "stroke-width": "1.7"}))
        body.append(_text(bx + bw / 2, by + 25, lbl, size=9, fill=col, weight="700", anchor="middle"))

    # Arrows
    arrows = [
        (x0 + 102, y0 + 73, x0 + 148, y0 + 73),
        (x0 + 236, y0 + 73, x0 + 280, y0 + 73),
        (x0 + 325, y0 + 94, x0 + 325, y0 + 132),
        (x0 + 282, y0 + 155, x0 + 238, y0 + 155),
        (x0 + 61, y0 + 94, x0 + 148, y0 + 150),
    ]
    for x1, y1, x2, y2 in arrows:
        body.append(_el("line", {"x1": _fmt(x1), "y1": _fmt(y1), "x2": _fmt(x2), "y2": _fmt(y2),
                                 "stroke": SLATE, "stroke-width": "1.4", "marker-end": "url(#arrow)"}))

    if is_words:
        words = ["apple", "grape", "lemon", "melon", "berry"]
        wx = x0 + 26
        for wd in words:
            body.append(_el("rect", {"x": _fmt(wx), "y": _fmt(y0 + 196), "width": "60", "height": "22", "rx": 4,
                                     "fill": "#eff6ff", "stroke": "#93c5fd"}))
            body.append(_text(wx + 30, y0 + 211, wd, size=9, fill=BLUE, weight="600", anchor="middle"))
            wx += 68
    else:
        footer = "Tests recon MSE, shuffle ratio, posterior MI & KL identity" if is_vae else "Tests faithful reconstruction + independent generative coverage"
        body.append(_text(x0 + w / 2, y0 + h - 18, footer, size=9, fill=TEXT_MUTED, anchor="middle"))
    return _wrap_svg(doc.get("title", tid), title_mid, body, _task_callouts(task, doc))


def _render_misgan_masked(task: dict, doc: dict) -> str:
    tid = task["id"]
    is_block = "block" in tid or "four_mode" in tid
    is_3pair = "imputer" in tid or "p80" in tid or "four_mode" in tid
    p_miss = "p=0.80 (+10% anchors)" if "p80" in tid else ("p=0.50 MCAR" if "p50" in tid else "8D block sensor dropout")
    x0, y0, w, h = 18, 54, 386, 248
    body = _plot_frame(x0, y0, w, h, axes=False)

    # Observed/masked 6x8 matrix visual (batched into 2 paths)
    body.append(_text(x0 + 20, y0 + 22, f"Incomplete 8D Data Matrix ({p_miss})", size=10, fill=TEXT_DARK, weight="700"))
    mx0, my0, cw, ch = x0 + 20, y0 + 32, 18.0, 14.0
    obs_p: list[str] = []
    miss_p: list[str] = []
    for r in range(6):
        for c in range(8):
            if is_block:
                masked = (r % 2 == 0 and c >= 4) or (r % 2 == 1 and c < 4)
            elif "p80" in tid:
                masked = (r > 0) and ((r * 7 + c * 3) % 5 != 0)
            else:
                masked = ((r * 5 + c * 3) % 2 == 0)
            seg = f"M{_fmt(mx0 + c * (cw + 2))},{_fmt(my0 + r * (ch + 2))}h{_fmt(cw)}v{_fmt(ch)}h-{_fmt(cw)}z"
            (miss_p if masked else obs_p).append(seg)
    if obs_p:
        body.append(_el("path", {"d": "".join(obs_p), "fill": "#bbf7d0", "stroke": EMERALD, "stroke-width": "0.8"}))
    if miss_p:
        body.append(_el("path", {"d": "".join(miss_p), "fill": "#fecdd3", "stroke": ROSE, "stroke-width": "0.8"}))
    body.append(_text(mx0, my0 + 108, "Green = observed (m=1) · Rose = missing (m=0)", size=8, fill=TEXT_MUTED))

    # Architecture blocks on right side of plot panel
    rx = x0 + 204
    items = [
        ("Data Gen G_x + Critic D_x", BLUE, "Recovers complete manifold"),
        ("Mask Gen G_m + Critic D_m", AMBER, "Detached mask grad in L_x"),
    ]
    if is_3pair:
        items.append(("Neural Imputer G_i + D_i", VIOLET, "G_i([x*m, m, z_i]), no ppost"))
    ry = y0 + 34
    for lbl, col, sub in items:
        body.append(_el("rect", {"x": _fmt(rx), "y": _fmt(ry), "width": "164", "height": "46", "rx": 6,
                                 "fill": "#ffffff", "stroke": col, "stroke-width": "1.6"}))
        body.append(_text(rx + 82, ry + 19, lbl, size=9, fill=col, weight="700", anchor="middle"))
        body.append(_text(rx + 82, ry + 34, sub, size=8, fill=TEXT_MUTED, anchor="middle"))
        ry += 56
    body.append(_text(x0 + w / 2, y0 + h - 14, f"{'3-pair MisGAN (with neural imputer)' if is_3pair else '2-pair MisGAN'} on 8D embedded manifold",
                      size=9, fill=TEXT_MUTED, anchor="middle"))
    return _wrap_svg(doc.get("title", tid), f"MisGAN incomplete-data learning ({p_miss})",
                     body, _task_callouts(task, doc))


def _render_cyclegan_dual(task: dict, doc: dict) -> str:
    x0, y0, w, h = 18, 54, 386, 248
    body = _plot_frame(x0, y0, w, h, axes=False)
    # Left: 8-mode ring (Domain X)
    lx, ly, r = x0 + 82, y0 + 124, 52.0
    body.append(_el("rect", {"x": _fmt(x0 + 16), "y": _fmt(y0 + 36), "width": "132", "height": "176", "rx": 8,
                             "fill": "#eff6ff", "stroke": "#93c5fd"}))
    body.append(_text(lx, y0 + 54, "Domain X: 8-Ring", size=10, fill=BLUE, weight="700", anchor="middle"))
    for k in range(8):
        th = 2.0 * math.pi * k / 8.0
        body.append(_el("circle", {"cx": _fmt(lx + r * math.cos(th)), "cy": _fmt(ly + r * math.sin(th)),
                                   "r": "5", "fill": BLUE}))
    # Right: 8-mode square perimeter (Domain Y)
    rx, ry = x0 + 304, y0 + 124
    body.append(_el("rect", {"x": _fmt(x0 + 238), "y": _fmt(y0 + 36), "width": "132", "height": "176", "rx": 8,
                             "fill": "#f0fdfa", "stroke": "#5eead4"}))
    body.append(_text(rx, y0 + 54, "Domain Y: 8-Square", size=10, fill=TEAL, weight="700", anchor="middle"))
    sq = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
    for sx, sy in sq:
        body.append(_el("circle", {"cx": _fmt(rx + sx * 42), "cy": _fmt(ry + sy * 42), "r": "5", "fill": TEAL}))
    # Dual translation arrows
    body.append(_el("line", {"x1": _fmt(x0 + 152), "y1": _fmt(y0 + 102), "x2": _fmt(x0 + 234), "y2": _fmt(y0 + 102),
                             "stroke": BLUE, "stroke-width": "2.0", "marker-end": "url(#arrow-blue)"}))
    body.append(_text(x0 + 193, y0 + 95, "G_XY + D_Y", size=9, fill=BLUE, weight="700", anchor="middle"))
    body.append(_el("line", {"x1": _fmt(x0 + 234), "y1": _fmt(y0 + 146), "x2": _fmt(x0 + 152), "y2": _fmt(y0 + 146),
                             "stroke": TEAL, "stroke-width": "2.0", "marker-end": "url(#arrow-teal)"}))
    body.append(_text(x0 + 193, y0 + 162, "G_YX + D_X", size=9, fill=TEAL, weight="700", anchor="middle"))
    body.append(_text(x0 + w / 2, y0 + h - 14, "Unpaired mode bijection + cycle RMSE <= 0.12 (G_YX(G_XY(x)) ~ x)",
                      size=9, fill=TEXT_MUTED, anchor="middle"))
    return _wrap_svg(doc.get("title", task["id"]), "CycleGAN unpaired 8-ring <-> 8-square constellation translation",
                     body, _task_callouts(task, doc))


def _render_bytegan_hypercube(task: dict, doc: dict) -> str:
    is_bigan = "bigan" in task["id"]
    x0, y0, w, h = 18, 54, 386, 248
    body = _plot_frame(x0, y0, w, h, axes=False)
    # 16x16 nibble projection of 256 vertices of {-1, +1}^8 (batched into 4 paths for compact SVG size)
    gx0, gy0, step = x0 + 24, y0 + 28, 11.0
    buckets: dict[int, list[tuple[float, float]]] = {0: [], 1: [], 2: [], 3: []}
    for r in range(16):
        for c in range(16):
            buckets[(r ^ c) % 4].append((gx0 + c * step, gy0 + r * step))
    for parity, col in enumerate((BLUE, TEAL, INDIGO, VIOLET)):
        body.append(_dots_el(buckets[parity], 3.1, col))
    body.append(_text(gx0 + 82, gy0 + 186, "256 vertices of {-1, +1}^8 (16x16 nibble map)", size=9, fill=TEXT_MUTED, anchor="middle"))

    rx = x0 + 216
    body.append(_el("rect", {"x": _fmt(rx), "y": _fmt(y0 + 36), "width": "154", "height": "66", "rx": 6,
                             "fill": "#eff6ff", "stroke": BLUE, "stroke-width": "1.5"}))
    body.append(_text(rx + 77, y0 + 56, "256-Particle Bijection", size=9, fill=BLUE, weight="700", anchor="middle"))
    body.append(_text(rx + 77, y0 + 72, "Zero-sigma ParticlePrior", size=8, fill=TEXT_MUTED, anchor="middle"))
    body.append(_text(rx + 77, y0 + 86, "256/256 unique bytes, 0 collisions", size=8, fill=EMERALD, weight="600", anchor="middle"))

    if is_bigan:
        body.append(_el("rect", {"x": _fmt(rx), "y": _fmt(y0 + 118), "width": "154", "height": "66", "rx": 6,
                                 "fill": "#f5f3ff", "stroke": VIOLET, "stroke-width": "1.5"}))
        body.append(_text(rx + 77, y0 + 138, "Joint BiGAN Lookup", size=9, fill=VIOLET, weight="700", anchor="middle"))
        body.append(_text(rx + 77, y0 + 154, "E(x) -> z_k -> G(z_k) == x", size=8, fill=TEXT_MUTED, anchor="middle"))
        body.append(_text(rx + 77, y0 + 168, "256/256 exact byte reconstructions", size=8, fill=VIOLET, weight="600", anchor="middle"))
    else:
        body.append(_el("rect", {"x": _fmt(rx), "y": _fmt(y0 + 118), "width": "154", "height": "66", "rx": 6,
                                 "fill": "#fffbeb", "stroke": AMBER, "stroke-width": "1.5"}))
        body.append(_text(rx + 77, y0 + 138, "Law 1 Atomic Support", size=9, fill=AMBER, weight="700", anchor="middle"))
        body.append(_text(rx + 77, y0 + 154, "RMS bit error <= 0.12", size=8, fill=TEXT_MUTED, anchor="middle"))
        body.append(_text(rx + 77, y0 + 168, "Zero interior ghost bytes", size=8, fill=TEXT_MUTED, anchor="middle"))
    return _wrap_svg(doc.get("title", task["id"]), "8D binary hypercube {-1,+1}^8 · 256 exact uint8 byte modes",
                     body, _task_callouts(task, doc))


def _render_nash_imbalance(task: dict, doc: dict) -> str:
    tid = task["id"]
    g_big = "overpowered_generator" in tid
    unequal = "unequal_width" in tid
    x0, y0, w, h = 18, 54, 386, 248
    body = _plot_frame(x0, y0, w, h, axes=False)

    # Parameter imbalance visual bars
    body.append(_text(x0 + 20, y0 + 24, "Hostile >600x Capacity Asymmetry + Off-Manifold Init", size=10, fill=TEXT_DARK, weight="700"))
    g_len = 240.0 if g_big else 22.0
    d_len = 22.0 if g_big else 240.0
    body.append(_text(x0 + 20, y0 + 50, "Generator G", size=9, fill=BLUE, weight="700"))
    body.append(_el("rect", {"x": _fmt(x0 + 104), "y": _fmt(y0 + 38), "width": _fmt(g_len), "height": "16", "rx": 4, "fill": BLUE}))
    body.append(_text(x0 + 110 + g_len, y0 + 50, "512x4 (2.2x amp)" if g_big else "32x2 (0.08x comp)", size=8, fill=TEXT_MUTED))

    body.append(_text(x0 + 20, y0 + 78, "Critic D", size=9, fill=ROSE, weight="700"))
    body.append(_el("rect", {"x": _fmt(x0 + 104), "y": _fmt(y0 + 66), "width": _fmt(d_len), "height": "16", "rx": 4, "fill": ROSE}))
    body.append(_text(x0 + 110 + d_len, y0 + 78, "32x2 (tilted)" if g_big else "512x4 (F3 tilted)", size=8, fill=TEXT_MUTED))

    # Hostile init schematic box
    body.append(_el("rect", {"x": x0 + 20, "y": y0 + 102, "width": 346, "height": 124, "rx": 6,
                             "fill": "#f8fafc", "stroke": PANEL_STROKE}))
    # Target cluster(s) on left, off-manifold shifted G on right, hostile upside-down tilt
    tx, ty = x0 + 112, y0 + 164
    if unequal:
        for (dx, dy), rr in (((-26, -22), 5), ((-26, 22), 8), ((26, -22), 12), ((26, 22), 16)):
            body.append(_el("circle", {"cx": _fmt(tx + dx), "cy": _fmt(ty + dy), "r": _fmt(rr),
                                       "fill": BLUE, "fill-opacity": "0.35", "stroke": BLUE, "stroke-width": "1.3"}))
        body.append(_text(tx, y0 + 216, "4 unequal-width modes", size=8, fill=BLUE, anchor="middle"))
    else:
        for k in range(16):
            th = 2.0 * math.pi * k / 16.0
            body.append(_el("circle", {"cx": _fmt(tx + 36 * math.cos(th)), "cy": _fmt(ty + 36 * math.sin(th)),
                                       "r": "3.2", "fill": BLUE}))
        body.append(_text(tx, y0 + 216, "16-mode Ring (R=3.0)", size=8, fill=BLUE, anchor="middle"))

    ox, oy = x0 + 296, y0 + 144
    body.append(_el("circle", {"cx": _fmt(ox), "cy": _fmt(oy), "r": "14" if g_big else "7",
                               "fill": AMBER, "fill-opacity": "0.35", "stroke": AMBER, "stroke-width": "1.8"}))
    body.append(_text(ox, oy + 26, "Step-0 G(z) corner", size=8, fill=AMBER, weight="700", anchor="middle"))
    body.append(_el("line", {"x1": _fmt(ox - 18), "y1": _fmt(oy + 4), "x2": _fmt(tx + 46), "y2": _fmt(ty - 4),
                             "stroke": EMERALD, "stroke-width": "1.8", "marker-end": "url(#arrow-teal)"}))
    body.append(_text(x0 + 214, y0 + 132, "Must reach Nash", size=8, fill=EMERALD, weight="700", anchor="middle"))
    body.append(_text(x0 + 296, y0 + 194, "Hostile tilt: ~ -30 logit margin", size=8, fill=ROSE, weight="600", anchor="middle"))
    return _wrap_svg(doc.get("title", tid), f"Extreme Nash imbalance ({'G >> D' if g_big else 'D >> G'}, >600x params)",
                     body, _task_callouts(task, doc))


def _render_clockfree_audit(task: dict, doc: dict) -> str:
    x0, y0, w, h = 18, 54, 386, 248
    body = _plot_frame(x0, y0, w, h, axes=False)
    body.append(_text(x0 + w / 2, y0 + 24, "Four-Perturbation Clock & State Invariance Audit", size=11, fill=TEXT_DARK, weight="700", anchor="middle"))
    # Reference box
    body.append(_el("rect", {"x": x0 + 18, "y": y0 + 96, "width": "92", "height": "52", "rx": 6,
                             "fill": "#eff6ff", "stroke": BLUE, "stroke-width": "1.8"}))
    body.append(_text(x0 + 64, y0 + 118, "Reference Run", size=9, fill=BLUE, weight="700", anchor="middle"))
    body.append(_text(x0 + 64, y0 + 134, "SHA-256(state)", size=8, fill=TEXT_MUTED, anchor="middle"))

    conds = [
        ("1. Step label (+8192)", TEAL),
        ("2. Horizon (128 -> 4096)", VIOLET),
        ("3. Evaluation cadence", AMBER),
        ("4. Checkpoint restart", EMERALD),
    ]
    for i, (lbl, col) in enumerate(conds):
        cy = y0 + 42 + i * 48
        body.append(_el("line", {"x1": _fmt(x0 + 110), "y1": _fmt(y0 + 122), "x2": _fmt(x0 + 150), "y2": _fmt(cy + 16),
                                 "stroke": SLATE, "stroke-width": "1.2", "marker-end": "url(#arrow)"}))
        body.append(_el("rect", {"x": _fmt(x0 + 154), "y": _fmt(cy), "width": "148", "height": "32", "rx": 5,
                                 "fill": "#ffffff", "stroke": col, "stroke-width": "1.5"}))
        body.append(_text(x0 + 228, cy + 20, lbl, size=9, fill=col, weight="700", anchor="middle"))
        body.append(_el("line", {"x1": _fmt(x0 + 302), "y1": _fmt(cy + 16), "x2": _fmt(x0 + 324), "y2": _fmt(cy + 16),
                                 "stroke": SLATE, "stroke-width": "1.2"}))
        body.append(_text(x0 + 348, cy + 20, "== ref?", size=9, fill=TEXT_DARK, weight="700", anchor="middle"))

    body.append(_text(x0 + w / 2, y0 + h - 12, "Bit-exact state-hash parity + zero unexplained clock/schedule hooks",
                      size=9, fill=TEXT_MUTED, anchor="middle"))
    return _wrap_svg(doc.get("title", task["id"]), "Clock-free & restart invariance verification (4 perturbations)",
                     body, _task_callouts(task, doc))


def _render_deep_spectral_flow(task: dict, doc: dict) -> str:
    tid = task["id"]
    x0, y0, w, h = 18, 54, 386, 248
    body = _plot_frame(x0, y0, w, h, axes=False)

    if tid == "fl_law42_deep_resbn_and_conv1x1":
        body.append(_text(x0 + w / 2, y0 + 24, "Isomorphic 8-Block ResBN Twin: 2D Linear vs 4D Conv2d 1x1",
                          size=10, fill=TEXT_DARK, weight="700", anchor="middle"))
        for row_i, (lbl, col, sub) in enumerate((
            ("Primary: 2D Linear ResBN (D=128, d=8)", BLUE, "sw1 <= 0.10, var_ratio_err <= 0.25"),
            ("Isomorphic Twin: 4D Conv2d 1x1", VIOLET, "Identical init & batches · sw1 ratio <= 1.05"),
        )):
            ry = y0 + 50 + row_i * 86
            body.append(_text(x0 + 22, ry, lbl, size=9, fill=col, weight="700"))
            for blk in range(8):
                bx = x0 + 22 + blk * 43
                body.append(_el("rect", {"x": _fmt(bx), "y": _fmt(ry + 10), "width": "34", "height": "32", "rx": 4,
                                         "fill": "#ffffff", "stroke": col, "stroke-width": "1.4"}))
                body.append(_text(bx + 17, ry + 30, f"B{blk+1}", size=8, fill=col, weight="600", anchor="middle"))
                if blk < 7:
                    body.append(_el("line", {"x1": _fmt(bx + 34), "y1": _fmt(ry + 26), "x2": _fmt(bx + 42), "y2": _fmt(ry + 26),
                                             "stroke": SLATE, "stroke-width": "1.1"}))
            body.append(_text(x0 + 22, ry + 56, sub, size=8, fill=TEXT_MUTED))
        body.append(_text(x0 + w / 2, y0 + h - 14, "Target: 64-mode R^128 Gaussian mixture · exposes 4D Conv vs 2D Linear bugs",
                          size=9, fill=TEXT_MUTED, anchor="middle"))
        return _wrap_svg(doc.get("title", tid), "8-block ResBN + isomorphic 4D Conv2d 1x1 parity on R^128",
                         body, _task_callouts(task, doc))

    swap_d = "swap_deep_critic" in tid
    res_tanh = "res16" in tid
    scaled = "scaled_tanh" in tid
    heading = ("Shallow 3L G vs Deep 16L Skip-Free Critic (fourier=0)" if swap_d
               else ("Deep 16-Block ResNet + Scaled Tanh -> 100-Grid" if scaled
                     else ("Deep 16-Block ResNet + Tanh -> 100-Grid" if res_tanh
                           else "Deep 16-Layer Skip-Free MLP -> 100-Grid")))
    body.append(_text(x0 + w / 2, y0 + 24, heading, size=10, fill=TEXT_DARK, weight="700", anchor="middle"))

    # Draw 16 stacked thin layer slices showing spectral propagation
    col = ROSE if swap_d else (VIOLET if res_tanh else BLUE)
    lx0 = x0 + 24
    for i in range(16):
        lx = lx0 + i * 16.5
        body.append(_el("rect", {"x": _fmt(lx), "y": _fmt(y0 + 52), "width": "11", "height": "96", "rx": 2,
                                 "fill": col, "fill-opacity": f"{0.22 + 0.04 * i:.2f}", "stroke": col, "stroke-width": "0.8"}))
        if res_tanh and i % 2 == 0 and i < 15:
            body.append(_el("path", {"d": f"M {_fmt(lx + 5.5)} {_fmt(y0 + 50)} Q {_fmt(lx + 14)} {_fmt(y0 + 38)} {_fmt(lx + 22)} {_fmt(y0 + 50)}",
                                     "fill": "none", "stroke": TEAL, "stroke-width": "1.2"}))
    body.append(_text(lx0 + 130, y0 + 166, "16 layers · spectral conditioning preserves singular values", size=9, fill=col, weight="600", anchor="middle"))

    # Target inset on right of plot frame
    ix, iy = x0 + 332, y0 + 100
    body.append(_el("rect", {"x": _fmt(ix - 38), "y": _fmt(iy - 48), "width": "76", "height": "96", "rx": 6,
                             "fill": "#f8fafc", "stroke": PANEL_STROKE}))
    if swap_d:
        for k in range(8):
            th = 2.0 * math.pi * k / 8.0
            body.append(_el("circle", {"cx": _fmt(ix + 22 * math.cos(th)), "cy": _fmt(iy + 22 * math.sin(th)),
                                       "r": "3.5", "fill": BLUE}))
        body.append(_text(ix, iy + 40, "8-Ring", size=8, fill=TEXT_MUTED, anchor="middle"))
    else:
        for r in range(5):
            for c in range(5):
                body.append(_el("circle", {"cx": _fmt(ix - 20 + c * 10), "cy": _fmt(iy - 22 + r * 10),
                                           "r": "2.2", "fill": BLUE}))
        body.append(_text(ix, iy + 40, "100-Grid", size=8, fill=TEXT_MUTED, anchor="middle"))

    footer = "Law 4.2: deep skip-free critic gradient contraction (~1e-7 without conditioning)" if swap_d else "Law 4.2: pathology-driven spectral conditioning across depth 16"
    body.append(_text(x0 + w / 2, y0 + h - 18, footer, size=9, fill=TEXT_MUTED, anchor="middle"))
    return _wrap_svg(doc.get("title", tid), heading, body, _task_callouts(task, doc))


def _render_highd_manifold(task: dict, doc: dict) -> str:
    tid = task["id"]
    x0, y0, w, h = 18, 54, 386, 248
    body = _plot_frame(x0, y0, w, h, axes=False)

    if tid == "fl_law3_highd_orphan_d32":
        body.extend(_plot_frame(x0 + 8, y0 + 8, w - 16, h - 28, axes=True, xlabel="Haar-rotated R^32 (principal plane)", ylabel="u2"))
        pts = [(-2.2, -2.2), (-2.2, 2.2), (2.2, -2.2), (2.2, 2.2), (7.5, 0.0)]
        widths = [0.05, 0.09, 0.15, 0.22, 0.08]
        wts = [36, 26, 18, 12, 8]
        to_px, _ = _map_2d(pts + [(-3.0, -3.5), (8.5, 3.5)], x0 + 8, y0 + 8, w - 16, h - 28, pad=0.14)
        for i, ((px, py), sig, wt) in enumerate(zip(pts, widths, wts)):
            sx, sy = to_px(px, py)
            col = ROSE if i == 4 else BLUE
            rad = 6.0 + sig * 60.0
            body.append(_el("circle", {"cx": _fmt(sx), "cy": _fmt(sy), "r": _fmt(rad),
                                       "fill": col, "fill-opacity": "0.24", "stroke": col, "stroke-width": "1.7"}))
            body.append(_text(sx, sy - rad - 4, f"{wt}% (s={sig:.2f})", size=8, fill=col, weight="700", anchor="middle"))
        # Screening barrier dashed line
        bx, _ = to_px(4.8, 0.0)
        body.append(_el("line", {"x1": _fmt(bx), "y1": _fmt(y0 + 14), "x2": _fmt(bx), "y2": _fmt(y0 + h - 24),
                                 "stroke": AMBER, "stroke-width": "1.6", "stroke-dasharray": "4,3"}))
        body.append(_text(bx + 4, y0 + 30, "Screening horizon (dim=32 > 16)", size=8, fill=AMBER, weight="700"))
        return _wrap_svg(doc.get("title", tid), "Haar-rotated R^32 4-core + distant 8% orphan cluster",
                         body, _task_callouts(task, doc))

    # fl_law44_highd_sparse_prior_and_shock
    # Split left: R^512 10% masked manifold; right: 1e-5x -> 1e4x shock timeline
    body.append(_el("rect", {"x": x0 + 12, "y": y0 + 16, "width": "172", "height": "214", "rx": 6,
                             "fill": "#f8fafc", "stroke": PANEL_STROKE}))
    body.append(_text(x0 + 98, y0 + 34, "Part A: R^512 Sparse Mask", size=9, fill=BLUE, weight="700", anchor="middle"))
    body.append(_text(x0 + 98, y0 + 48, "64 modes · P=1600 · 10% obs", size=8, fill=TEXT_MUTED, anchor="middle"))
    obs_cells: list[str] = []
    unobs_cells: list[str] = []
    for r in range(8):
        for c in range(8):
            obs = ((r * 7 + c * 11) % 10 == 0)
            seg = f"M{_fmt(x0 + 30 + c * 17)},{_fmt(y0 + 62 + r * 17)}h13v13h-13z"
            (obs_cells if obs else unobs_cells).append(seg)
    if unobs_cells:
        body.append(_el("path", {"d": "".join(unobs_cells), "fill": "#e2e8f0"}))
    if obs_cells:
        body.append(_el("path", {"d": "".join(obs_cells), "fill": BLUE}))
    body.append(_text(x0 + 98, y0 + 216, "RMSNorm critic + masked prior", size=8, fill=TEXT_MUTED, anchor="middle"))

    body.append(_el("rect", {"x": x0 + 198, "y": y0 + 16, "width": "176", "height": "214", "rx": 6,
                             "fill": "#f8fafc", "stroke": PANEL_STROKE}))
    body.append(_text(x0 + 286, y0 + 34, "Part B: 1e-5x + 1e4x Shock", size=9, fill=ROSE, weight="700", anchor="middle"))
    body.append(_text(x0 + 286, y0 + 48, "Scale spike & mode shift @t=200", size=8, fill=TEXT_MUTED, anchor="middle"))
    # Pulse curve
    pxs = [(214, 186), (256, 186), (272, 82), (288, 186), (358, 186)]
    poly = " ".join(f"{_fmt(x0 + px - 18)},{_fmt(y0 + py)}" for px, py in pxs)
    body.append(_el("polyline", {"points": poly, "fill": "none", "stroke": ROSE, "stroke-width": "2.4"}))
    body.append(_text(x0 + 272 - 18, y0 + 74, "1e4x spike!", size=8, fill=ROSE, weight="700", anchor="middle"))
    body.append(_text(x0 + 286, y0 + 216, "Tests scale-free settling (Law 4.4)", size=8, fill=TEXT_MUTED, anchor="middle"))
    return _wrap_svg(doc.get("title", tid), "High-D 10%-masked R^512 manifold + extreme 1e-5x/1e4x shock",
                     body, _task_callouts(task, doc))


def _render_behavior_probe(task: dict, doc: dict) -> str:
    tid = task["id"]
    base = _strip_variant_prefix(tid)
    x0, y0, w, h = 18, 54, 386, 248
    body = _plot_frame(x0, y0, w, h, axes=False)

    specs = {
        "two_pole": (
            "Two-Pole Directional Separation",
            [("Negative Pole (-u)", BLUE), ("Neutral (0)", SLATE), ("Positive Pole (+u)", ROSE)],
            "mean_abs >= 0.30 · critic grad_med <= 1.0",
        ),
        "two_pole_800_schedule800_diagnostic_v1": (
            "Two-Pole 800-Step Horizon (schedule=800)",
            [("Negative Pole (-u)", BLUE), ("800-step horizon", AMBER), ("Positive Pole (+u)", ROSE)],
            "Diagnostic horizon check for scheduled baselines",
        ),
        "two_pole_800_schedule80_diagnostic_v1": (
            "Two-Pole 800-Step Horizon (schedule=80)",
            [("Negative Pole (-u)", BLUE), ("80-step schedule", AMBER), ("Positive Pole (+u)", ROSE)],
            "Diagnostic schedule-horizon mismatch check",
        ),
        "unipolar": (
            "Unipolar Concept Activation + Neutral Hold",
            [("Neutral (hold >= 0.85)", TEAL), ("Off-caption (<= 0.05)", AMBER), ("+Concept (cover >= 0.85)", BLUE)],
            "Activates +pole while holding neutral & avoiding off-caption bleed",
        ),
        "unused_token_hold": (
            "Unused Token Invariance vs Active Concept Move",
            [("Unused Slots (hold >= 0.85)", EMERALD), ("Shared Network", SLATE), ("Active Concept (move >= 0.85)", BLUE)],
            "Updates target concept without corrupting unused vocabulary slots",
        ),
        "cover_leftover": (
            "Dual-Cloud Cover + Residual Content Preservation",
            [("Cloud +u / -u (err <= 0.20)", BLUE), ("Leak Ratio (<= 0.20)", ROSE), ("Content Kept (>= 0.75)", EMERALD)],
            "Separates concept poles while preserving residual content subspace",
        ),
        "mid_scale_identity": (
            "Mid-Scale Concept Direction & Magnitude Identity",
            [("Cosine Align (>= 0.85)", BLUE), ("Mid-Scale Mapping", VIOLET), ("Mag in [0.75, 1.25]", TEAL)],
            "Preserves both +/- concept alignment and calibrated scale",
        ),
        "residual_student": (
            "Residual Student Identity + Protected Multi-Objective",
            [("Identity MSE (<= 0.02)", TEAL), ("Protected Pad (wrong=0)", EMERALD), ("Task Success (100%)", BLUE)],
            "Jointly satisfies identity preservation & adversarial task target",
        ),
        "trajectory": (
            "Trajectory Identity Consistency",
            [("Input State x", BLUE), ("Stepwise Trajectory", INDIGO), ("Identity MSE (<= 0.02)", EMERALD)],
            "Preserves identity across learned trajectory updates",
        ),
    }
    heading, nodes, footer = specs.get(
        base,
        (
            doc.get("title", tid),
            [("Prior / Input", BLUE), ("Generator + Critic", VIOLET), ("Sustained Gate", EMERALD)],
            "Behavioral stability & representation preservation probe",
        ),
    )
    body.append(_text(x0 + w / 2, y0 + 28, heading, size=11, fill=TEXT_DARK, weight="700", anchor="middle"))

    bw = 104.0
    gap = (w - 3 * bw) / 4.0
    by = y0 + 82
    for i, (lbl, col) in enumerate(nodes):
        bx = x0 + gap + i * (bw + gap)
        body.append(_el("rect", {"x": _fmt(bx), "y": _fmt(by), "width": _fmt(bw), "height": "68", "rx": 8,
                                 "fill": "#ffffff", "stroke": col, "stroke-width": "1.8"}))
        # Wrap label across 2 lines cleanly
        words = lbl.split(" ")
        mid = max(1, len(words) // 2)
        l1 = " ".join(words[:mid])
        l2 = " ".join(words[mid:])
        if l2:
            body.append(_text(bx + bw / 2, by + 30, l1, size=9, fill=col, weight="700", anchor="middle"))
            body.append(_text(bx + bw / 2, by + 45, l2, size=9, fill=TEXT_MUTED, anchor="middle"))
        else:
            body.append(_text(bx + bw / 2, by + 38, l1, size=9, fill=col, weight="700", anchor="middle"))
        if i < 2:
            body.append(_el("line", {"x1": _fmt(bx + bw), "y1": _fmt(by + 34),
                                     "x2": _fmt(bx + bw + gap - 2), "y2": _fmt(by + 34),
                                     "stroke": SLATE, "stroke-width": "1.5", "marker-end": "url(#arrow)"}))

    body.append(_el("rect", {"x": x0 + 24, "y": y0 + 180, "width": w - 48, "height": 42, "rx": 6,
                             "fill": "#f8fafc", "stroke": PANEL_STROKE}))
    body.append(_text(x0 + w / 2, y0 + 205, footer[:68], size=9, fill=TEXT_DARK, weight="600", anchor="middle"))
    return _wrap_svg(doc.get("title", tid), heading, body, _task_callouts(task, doc))


def infer_diagram_kind(task: dict, doc: dict | None = None) -> str:
    """Determine the deterministic SVG schematic renderer for a Forge task."""
    if isinstance(doc, dict) and doc.get("diagram_kind"):
        return str(doc["diagram_kind"])
    tid = task["id"]
    base = _strip_variant_prefix(tid)
    adapter = task.get("adapter", "")

    if adapter == "dcgan_performance_v1" or base.startswith("dcgan_performance"):
        return "dcgan_scaling"
    if adapter == "clockfree_audit" or base.startswith("clockfree_"):
        return "clockfree_audit"
    if base in ("bytegan_hypercube8", "bytegan_bigan_dictionary"):
        return "bytegan_hypercube"
    if base == "interp_saddle_barycentric":
        return "saddle_3d_barycentric"
    if base == "interp_img_patch_shift" or adapter == "transfer_image" or base.startswith("img_"):
        return "image_grid_8x8"
    if base == "fl_law2_groupnorm_and_vit_sprites":
        return "rgb_sprites_32x32"
    if "ddgan" in base:
        return "ddgan_chain"
    if "misgan" in base:
        return "misgan_masked"
    if base == "cyclegan_unpaired_constellation":
        return "cyclegan_dual"
    if base.startswith("nash_"):
        return "nash_imbalance"
    if base in ("grid100_mlp16", "grid100_res16_tanh", "grid100_res16_scaled_tanh",
                "fl_law42_swap_deep_critic_16l", "fl_law42_deep_resbn_and_conv1x1"):
        return "deep_spectral_flow"
    if base in ("fl_law3_highd_orphan_d32", "fl_law44_highd_sparse_prior_and_shock"):
        return "highd_manifold"
    if any(k in base for k in ("gaussian1d", "precision_gaussian", "zero_tanh")):
        return "gaussian_1d_shift"
    if base in ("ae_gan_hold", "model_ae_reconstruction", "model_ae_gan_multimodal",
                "model_vae_categorical", "model_vae_hard") or base.startswith("five_word_"):
        return "latent_loop"
    if base in ("two_pole", "two_pole_800_schedule800_diagnostic_v1", "two_pole_800_schedule80_diagnostic_v1",
                "unipolar", "unused_token_hold", "cover_leftover", "mid_scale_identity",
                "residual_student", "trajectory"):
        return "behavior_probe"
    return "gaussian_mixture_2d"


RENDERERS = {
    "gaussian_mixture_2d": _render_gaussian_mixture_2d,
    "gaussian_1d_shift": _render_gaussian_1d_shift,
    "saddle_3d_barycentric": _render_saddle_3d,
    "image_grid_8x8": _render_image_grid_8x8,
    "rgb_sprites_32x32": _render_rgb_sprites,
    "dcgan_scaling": _render_dcgan_scaling,
    "ddgan_chain": _render_ddgan_chain,
    "latent_loop": _render_latent_loop,
    "misgan_masked": _render_misgan_masked,
    "cyclegan_dual": _render_cyclegan_dual,
    "bytegan_hypercube": _render_bytegan_hypercube,
    "nash_imbalance": _render_nash_imbalance,
    "clockfree_audit": _render_clockfree_audit,
    "deep_spectral_flow": _render_deep_spectral_flow,
    "highd_manifold": _render_highd_manifold,
    "behavior_probe": _render_behavior_probe,
}


def render_task_svg(task: dict, doc: dict | None = None) -> str:
    """Render a deterministic XML SVG schematic string for a Forge task."""
    doc_obj = doc if isinstance(doc, dict) else {}
    kind = infer_diagram_kind(task, doc_obj)
    renderer = RENDERERS.get(kind, _render_gaussian_mixture_2d)
    return renderer(task, doc_obj)


def write_task_svg(task: dict, output_path: Path, doc: dict | None = None) -> Path:
    """Write the deterministic SVG schematic for a task to ``output_path``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    svg_text = render_task_svg(task, doc)
    output_path.write_text(svg_text, encoding="utf-8")
    return output_path
