"""Brand-colored SVG figures for the README. The palette is the web client's earlier dark one (.frontend/app/globals.css
has since moved to a light palette; these figures keep theirs)."""

from __future__ import annotations

import json
from pathlib import Path

# The earlier app chrome: #121212 body, lavender / gold / sky / pink radial gradient.
BG = "#121212"
PANEL = "#1A1A24"
TEXT = "#F4F4F8"
MUTED = "#A8B0C0"
GRID = "#2A2A38"
BLUE = "#8EB4E8"
LAVENDER = "#BABAE9"
GOLD = "#E8D6A0"
PINK = "#FBDAEF"
SKY = "#C2D5FF"
LOSS = "#6B7280"


def _bar_chart(
    path: Path,
    title: str,
    subtitle: str,
    series: list[tuple[str, float, str]],
    ymax: float = 1.0,
    ci: list[tuple[float, float] | None] | None = None,
    marker: list[float | None] | None = None,
    marker_label: str = "",
) -> None:
    """Horizontal bars. `ci` draws a 95% interval whisker per bar; `marker` draws a
    dashed tick per bar (used for the random-list baseline) labelled once."""
    w, h = 860, 420
    left, right, top, bottom = 210, 40, 72, 48
    plot_w = w - left - right
    plot_h = h - top - bottom
    n = len(series)
    gap = 14
    bar_h = (plot_h - gap * (n - 1)) / max(n, 1)
    bars = []
    for i, (label, value, color) in enumerate(series):
        y = top + i * (bar_h + gap)
        bw = max(2, plot_w * (value / ymax))
        value_txt = f"{value:.3f}"
        label_x = left + bw + 8
        bars.append(
            f'<rect x="{left}" y="{y}" width="{bw:.1f}" height="{bar_h:.1f}" rx="4" fill="{color}"/>'
            f'<text x="{left - 12}" y="{y + bar_h * 0.68:.1f}" text-anchor="end" fill="{TEXT}" '
            f'font-family="ui-sans-serif, system-ui, sans-serif" font-size="13">{_esc(label)}</text>'
        )
        interval = ci[i] if ci and i < len(ci) else None
        if interval is not None:
            lo, hi = interval
            x_lo = left + plot_w * (lo / ymax)
            x_hi = left + plot_w * (hi / ymax)
            ym = y + bar_h / 2
            bars.append(
                f'<line x1="{x_lo:.1f}" y1="{ym:.1f}" x2="{x_hi:.1f}" y2="{ym:.1f}" stroke="{TEXT}" stroke-width="2" opacity="0.9"/>'
                f'<line x1="{x_lo:.1f}" y1="{ym - 6:.1f}" x2="{x_lo:.1f}" y2="{ym + 6:.1f}" stroke="{TEXT}" stroke-width="2" opacity="0.9"/>'
                f'<line x1="{x_hi:.1f}" y1="{ym - 6:.1f}" x2="{x_hi:.1f}" y2="{ym + 6:.1f}" stroke="{TEXT}" stroke-width="2" opacity="0.9"/>'
            )
            value_txt = f"{value:.3f}  [{lo:.3f}, {hi:.3f}]"
            label_x = max(label_x, x_hi + 8)
        tick = marker[i] if marker and i < len(marker) else None
        if tick is not None:
            x_t = left + plot_w * (tick / ymax)
            bars.append(
                f'<line x1="{x_t:.1f}" y1="{y - 2:.1f}" x2="{x_t:.1f}" y2="{y + bar_h + 2:.1f}" stroke="{PINK}" '
                f'stroke-width="2" stroke-dasharray="4 3"/>'
            )
        bars.append(
            f'<text x="{label_x:.1f}" y="{y + bar_h * 0.68:.1f}" fill="{GOLD}" '
            f'font-family="ui-sans-serif, system-ui, sans-serif" font-size="13" font-weight="600">'
            f"{_esc(value_txt)}</text>"
        )
    if marker and marker_label:
        bars.append(
            f'<line x1="{w - right - 150}" y1="{h - 22}" x2="{w - right - 130}" y2="{h - 22}" stroke="{PINK}" stroke-width="2" stroke-dasharray="4 3"/>'
            f'<text x="{w - right - 124}" y="{h - 18}" fill="{MUTED}" font-family="ui-sans-serif, system-ui, sans-serif" font-size="12">{_esc(marker_label)}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{_esc(title)}">
  <rect width="{w}" height="{h}" fill="{BG}"/>
  <rect x="12" y="12" width="{w-24}" height="{h-24}" rx="16" fill="{PANEL}" stroke="{GRID}"/>
  <text x="32" y="44" fill="{TEXT}" font-family="ui-sans-serif, system-ui, sans-serif" font-size="18" font-weight="700">{_esc(title)}</text>
  <text x="32" y="64" fill="{MUTED}" font-family="ui-sans-serif, system-ui, sans-serif" font-size="12">{_esc(subtitle)}</text>
  {''.join(bars)}
</svg>
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg)


def _line_chart(path: Path, title: str, subtitle: str, xs: list[float], ys: list[float], y2: list[float] | None, xlabel: str, y1label: str, y2label: str | None) -> None:
    w, h = 860, 420
    left, right, top, bottom = 72, 72, 80, 56
    plot_w = w - left - right
    plot_h = h - top - bottom
    xmin, xmax = min(xs), max(xs) or 1
    y1max = max(ys) * 1.12 if ys else 1.0
    y2max = (max(y2) * 1.12 if y2 else 1.0) or 1.0

    def px(x):
        return left + (x - xmin) / (xmax - xmin or 1) * plot_w

    def py1(y):
        return top + plot_h - (y / y1max) * plot_h

    def py2(y):
        return top + plot_h - (y / y2max) * plot_h

    pts = " ".join(f"{px(x):.1f},{py1(y):.1f}" for x, y in zip(xs, ys))
    dots = "".join(
        f'<circle cx="{px(x):.1f}" cy="{py1(y):.1f}" r="4" fill="{BLUE}"/>'
        f'<text x="{px(x):.1f}" y="{py1(y) - 10:.1f}" text-anchor="middle" fill="{BLUE}" '
        f'font-size="11" font-family="ui-sans-serif, system-ui, sans-serif">{y:.2f}</text>'
        for x, y in zip(xs, ys)
    )
    grid = []
    for g in range(5):
        yy = top + plot_h * g / 4
        grid.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+plot_w}" y2="{yy:.1f}" stroke="{GRID}" />')
    extra = ""
    if y2:
        pts2 = " ".join(f"{px(x):.1f},{py2(y):.1f}" for x, y in zip(xs, y2))
        extra = f'<polyline fill="none" stroke="{GOLD}" stroke-width="2.5" points="{pts2}"/>'
        extra += "".join(
            f'<circle cx="{px(x):.1f}" cy="{py2(y):.1f}" r="3.5" fill="{GOLD}"/>'
            for x, y in zip(xs, y2)
        )
        extra += (
            f'<text x="{w-28}" y="48" text-anchor="end" fill="{GOLD}" font-size="12" '
            f'font-family="ui-sans-serif, system-ui, sans-serif">{_esc(y2label or "")}</text>'
        )
        extra += (
            f'<text x="{w-20}" y="{top+plot_h/2:.0f}" fill="{GOLD}" font-size="12" '
            f'font-family="ui-sans-serif, system-ui, sans-serif" '
            f'transform="rotate(90 {w-20} {top+plot_h/2:.0f})">{_esc(y2label or "")}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img">
  <rect width="{w}" height="{h}" fill="{BG}"/>
  <rect x="12" y="12" width="{w-24}" height="{h-24}" rx="16" fill="{PANEL}" stroke="{GRID}"/>
  <text x="32" y="44" fill="{TEXT}" font-family="ui-sans-serif, system-ui, sans-serif" font-size="18" font-weight="700">{_esc(title)}</text>
  <text x="32" y="64" fill="{MUTED}" font-family="ui-sans-serif, system-ui, sans-serif" font-size="12">{_esc(subtitle)}</text>
  {''.join(grid)}
  <polyline fill="none" stroke="{BLUE}" stroke-width="2.5" points="{pts}"/>
  {dots}
  {extra}
  <text x="{left+plot_w/2:.0f}" y="{h-22}" text-anchor="middle" fill="{MUTED}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">{_esc(xlabel)}</text>
  <text x="22" y="{top+plot_h/2:.0f}" fill="{BLUE}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif" transform="rotate(-90 22 {top+plot_h/2:.0f})">{_esc(y1label)}</text>
</svg>
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg)


def _esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def recall50_bm25_to_hybrid(blob: dict | None, arrow: str = "→") -> str:
    """"Recall@50  <bm25_only> → <hybrid>" from a results file (main.json), or "" when either row is missing."""
    rows = {row["config"]: row["retrieval"] for row in (blob or {}).get("results", [])}
    if "bm25_only" not in rows or "hybrid" not in rows:
        return ""
    return f"Recall@50  {rows['bm25_only']['recall@50']:.3f} {arrow} {rows['hybrid']['recall@50']:.3f}"


def write_static_story_svgs(out_dir: Path, blob: dict | None = None) -> None:
    """Pipeline / query-trace / staleness cards. Numbers live in ablation.svg; the one
    the hybrid card shows is read from `blob` (main.json), never typed here."""
    out_dir.mkdir(parents=True, exist_ok=True)
    hybrid_recall = _esc(recall50_bm25_to_hybrid(blob))
    # query trace
    rows = [
        ("route", "semantic  (not a path, not an aggregation verb)", BLUE),
        ("bm25 + dense", "RRF k=60 over fused top-50", LAVENDER),
        ("rerank", "overlap teacher in CI; bge-reranker-v2-m3 when loaded", GOLD),
        ("staleness", "drop configs/archive/run_047_v1.yaml  (superseded)", PINK),
        ("cite", "configs/run_047.yaml  bytes 0–312", SKY),
        ("answer", "3e-4", TEXT),
    ]
    blocks = []
    y = 88
    for name, detail, color in rows:
        blocks.append(
            f'<rect x="36" y="{y}" width="788" height="46" rx="8" fill="#121212" stroke="{color}"/>'
            f'<text x="52" y="{y+20}" fill="{color}" font-size="12" font-family="ui-monospace, monospace" font-weight="700">{_esc(name)}</text>'
            f'<text x="52" y="{y+38}" fill="{TEXT}" font-size="13" font-family="ui-sans-serif, system-ui, sans-serif">{_esc(detail)}</text>'
        )
        y += 54
    (out_dir / "query-trace.svg").write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="860" height="460" viewBox="0 0 860 460">
  <rect width="860" height="460" fill="{BG}"/>
  <rect x="12" y="12" width="836" height="436" rx="16" fill="{PANEL}" stroke="{GRID}"/>
  <text x="36" y="44" fill="{TEXT}" font-size="18" font-weight="700" font-family="ui-sans-serif, system-ui, sans-serif">Live retrieval  ·  current learning rate for DINOv2 run 47</text>
  <text x="36" y="66" fill="{MUTED}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">Gold answer is 3e-4. Archive still has 1e-5. The pipeline has to prefer the live config.</text>
  {''.join(blocks)}
</svg>
'''
    )
    # staleness
    (out_dir / "staleness.svg").write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="860" height="300" viewBox="0 0 860 300">
  <rect width="860" height="300" fill="{BG}"/>
  <rect x="12" y="12" width="836" height="276" rx="16" fill="{PANEL}" stroke="{GRID}"/>
  <text x="36" y="48" fill="{TEXT}" font-size="18" font-weight="700" font-family="ui-sans-serif, system-ui, sans-serif">Version cluster  ·  run_047.yaml</text>
  <rect x="36" y="72" width="380" height="180" rx="12" fill="#121212" stroke="{LOSS}"/>
  <text x="56" y="104" fill="{LOSS}" font-size="12" font-family="ui-monospace, monospace">SUPERSEDED</text>
  <text x="56" y="132" fill="{MUTED}" font-size="14" font-family="ui-sans-serif, system-ui, sans-serif">configs/archive/run_047_v1.yaml</text>
  <text x="56" y="160" fill="{MUTED}" font-size="14" font-family="ui-monospace, monospace">learning_rate: 1e-5</text>
  <text x="56" y="188" fill="{MUTED}" font-size="13" font-family="ui-sans-serif, system-ui, sans-serif">mtime older  ·  not the retrieval target</text>
  <rect x="444" y="72" width="380" height="180" rx="12" fill="#121212" stroke="{GOLD}"/>
  <text x="464" y="104" fill="{GOLD}" font-size="12" font-family="ui-monospace, monospace">CURRENT</text>
  <text x="464" y="132" fill="{TEXT}" font-size="14" font-family="ui-sans-serif, system-ui, sans-serif">configs/run_047.yaml</text>
  <text x="464" y="160" fill="{TEXT}" font-size="14" font-family="ui-monospace, monospace">learning_rate: 3e-4</text>
  <text x="464" y="188" fill="{SKY}" font-size="13" font-family="ui-sans-serif, system-ui, sans-serif">default retrieval target  ·  gold for staleness Qs</text>
</svg>
'''
    )
    # pipeline
    stages = [
        ("route", "regex\n+ verbs"),
        ("plan", "cap=3"),
        ("search", "BM25\n+ dense"),
        ("grade", "overlap"),
        ("rewrite", "or stop"),
        ("cite", "path +\nbytes"),
    ]
    boxes = []
    x = 28
    for i, (name, sub) in enumerate(stages):
        color = [BLUE, LAVENDER, GOLD, PINK, SKY, TEXT][i]
        boxes.append(
            f'<rect x="{x}" y="90" width="120" height="88" rx="10" fill="#121212" stroke="{color}"/>'
            f'<text x="{x+60}" y="128" text-anchor="middle" fill="{color}" font-size="14" font-weight="700" font-family="ui-sans-serif, system-ui, sans-serif">{name}</text>'
            f'<text x="{x+60}" y="152" text-anchor="middle" fill="{MUTED}" font-size="11" font-family="ui-sans-serif, system-ui, sans-serif">{sub.splitlines()[0]}</text>'
            f'<text x="{x+60}" y="168" text-anchor="middle" fill="{MUTED}" font-size="11" font-family="ui-sans-serif, system-ui, sans-serif">{sub.splitlines()[-1] if chr(10) in sub else ""}</text>'
        )
        if i < len(stages) - 1:
            boxes.append(f'<polygon points="{x+128},134 {x+140},128 {x+140},140" fill="{MUTED}"/>')
        x += 138
    (out_dir / "agent-loop.svg").write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="860" height="240" viewBox="0 0 860 240">
  <rect width="860" height="240" fill="{BG}"/>
  <rect x="12" y="12" width="836" height="216" rx="16" fill="{PANEL}" stroke="{GRID}"/>
  <text x="32" y="48" fill="{TEXT}" font-size="18" font-weight="700" font-family="ui-sans-serif, system-ui, sans-serif">Retrieval sits inside the loop</text>
  <text x="32" y="70" fill="{MUTED}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">Hard iteration cap. Empty retrieval fails loud. Moves never execute without approval.</text>
  {''.join(boxes)}
</svg>
'''
    )
    (out_dir / "router.svg").write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="860" height="280" viewBox="0 0 860 280">
  <rect width="860" height="280" fill="{BG}"/>
  <rect x="12" y="12" width="836" height="256" rx="16" fill="{PANEL}" stroke="{GRID}"/>
  <text x="32" y="44" fill="{TEXT}" font-size="18" font-weight="700" font-family="ui-sans-serif, system-ui, sans-serif">Query router  ·  40 lines, no extra LLM</text>
  <text x="32" y="66" fill="{MUTED}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">Gold set: 8 queries skipped embed, 41 skipped rerank. Exact-path nDCG@10 0.596 → 0.938.</text>
  <rect x="32" y="88" width="250" height="156" rx="12" fill="#121212" stroke="{GOLD}"/>
  <text x="48" y="116" fill="{GOLD}" font-size="13" font-family="ui-monospace, monospace">lexical_path</text>
  <text x="48" y="144" fill="{TEXT}" font-size="13" font-family="ui-sans-serif, system-ui, sans-serif">open configs/run_047.yaml</text>
  <text x="48" y="172" fill="{MUTED}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">skip embed + rerank</text>
  <text x="48" y="200" fill="{SKY}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">path lookup + BM25</text>
  <rect x="304" y="88" width="250" height="156" rx="12" fill="#121212" stroke="{LAVENDER}"/>
  <text x="320" y="116" fill="{LAVENDER}" font-size="13" font-family="ui-monospace, monospace">aggregation</text>
  <text x="320" y="144" fill="{TEXT}" font-size="13" font-family="ui-sans-serif, system-ui, sans-serif">lowest val RMSE?</text>
  <text x="320" y="172" fill="{MUTED}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">skip rerank</text>
  <text x="320" y="200" fill="{SKY}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">BM25 over configs + CSV</text>
  <rect x="576" y="88" width="250" height="156" rx="12" fill="#121212" stroke="{BLUE}"/>
  <text x="592" y="116" fill="{BLUE}" font-size="13" font-family="ui-monospace, monospace">semantic</text>
  <text x="592" y="144" fill="{TEXT}" font-size="13" font-family="ui-sans-serif, system-ui, sans-serif">what does fusion do</text>
  <text x="592" y="172" fill="{MUTED}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">full pipeline</text>
  <text x="592" y="200" fill="{SKY}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">BM25 + dense + RRF</text>
</svg>
'''
    )
    (out_dir / "mcp-hitl.svg").write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="860" height="260" viewBox="0 0 860 260">
  <rect width="860" height="260" fill="{BG}"/>
  <rect x="12" y="12" width="836" height="236" rx="16" fill="{PANEL}" stroke="{GRID}"/>
  <text x="32" y="44" fill="{TEXT}" font-size="18" font-weight="700" font-family="ui-sans-serif, system-ui, sans-serif">MCP filesystem  ·  human interrupt on mutate</text>
  <text x="32" y="66" fill="{MUTED}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">propose_move never executes. apply_plan is a no-op until approved=True.</text>
  <rect x="32" y="88" width="250" height="136" rx="12" fill="#121212" stroke="{BLUE}"/>
  <text x="48" y="116" fill="{BLUE}" font-size="12" font-family="ui-monospace, monospace">1  propose_move</text>
  <text x="48" y="144" fill="{TEXT}" font-size="13" font-family="ui-sans-serif, system-ui, sans-serif">logs/run_040.out</text>
  <text x="48" y="164" fill="{MUTED}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">→ archive/run_040.out</text>
  <text x="48" y="196" fill="{GOLD}" font-size="12" font-family="ui-monospace, monospace">pending_approval</text>
  <rect x="304" y="88" width="250" height="136" rx="12" fill="#121212" stroke="{PINK}"/>
  <text x="320" y="116" fill="{PINK}" font-size="12" font-family="ui-monospace, monospace">2  apply_plan</text>
  <text x="320" y="144" fill="{TEXT}" font-size="13" font-family="ui-sans-serif, system-ui, sans-serif">approved missing</text>
  <text x="320" y="196" fill="{PINK}" font-size="12" font-family="ui-monospace, monospace">ApprovalRequired</text>
  <rect x="576" y="88" width="250" height="136" rx="12" fill="#121212" stroke="{GOLD}"/>
  <text x="592" y="116" fill="{GOLD}" font-size="12" font-family="ui-monospace, monospace">3  apply_plan</text>
  <text x="592" y="144" fill="{TEXT}" font-size="13" font-family="ui-sans-serif, system-ui, sans-serif">approved=True</text>
  <text x="592" y="196" fill="{SKY}" font-size="12" font-family="ui-monospace, monospace">applied</text>
</svg>
'''
    )
    (out_dir / "graphrag.svg").write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="860" height="280" viewBox="0 0 860 280">
  <rect width="860" height="280" fill="{BG}"/>
  <rect x="12" y="12" width="836" height="256" rx="16" fill="{PANEL}" stroke="{GRID}"/>
  <text x="32" y="44" fill="{TEXT}" font-size="18" font-weight="700" font-family="ui-sans-serif, system-ui, sans-serif">GraphRAG-lite  ·  entities from the files, not invented</text>
  <text x="32" y="66" fill="{MUTED}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">run:47 —uses_encoder→ dinov2. Communities are connected components. Optional LLM adds triples.</text>
  <rect x="40" y="92" width="140" height="56" rx="28" fill="#121212" stroke="{GOLD}"/>
  <text x="110" y="126" text-anchor="middle" fill="{GOLD}" font-size="13" font-family="ui-sans-serif, system-ui, sans-serif">run:47</text>
  <rect x="40" y="176" width="140" height="56" rx="28" fill="#121212" stroke="{LAVENDER}"/>
  <text x="110" y="210" text-anchor="middle" fill="{LAVENDER}" font-size="13" font-family="ui-sans-serif, system-ui, sans-serif">run:46</text>
  <rect x="360" y="92" width="160" height="56" rx="28" fill="#121212" stroke="{BLUE}"/>
  <text x="440" y="126" text-anchor="middle" fill="{BLUE}" font-size="13" font-family="ui-sans-serif, system-ui, sans-serif">encoder:dinov2</text>
  <rect x="360" y="176" width="160" height="56" rx="28" fill="#121212" stroke="{PINK}"/>
  <text x="440" y="210" text-anchor="middle" fill="{PINK}" font-size="13" font-family="ui-sans-serif, system-ui, sans-serif">src/fusion.py</text>
  <rect x="640" y="134" width="180" height="56" rx="28" fill="#121212" stroke="{SKY}"/>
  <text x="730" y="168" text-anchor="middle" fill="{SKY}" font-size="13" font-family="ui-sans-serif, system-ui, sans-serif">lr: 3e-4</text>
  <line x1="180" y1="120" x2="360" y2="120" stroke="{MUTED}"/>
  <line x1="180" y1="204" x2="360" y2="204" stroke="{MUTED}"/>
  <line x1="520" y1="120" x2="640" y2="162" stroke="{MUTED}"/>
  <text x="210" y="112" fill="{MUTED}" font-size="11" font-family="ui-monospace, monospace">uses_encoder</text>
</svg>
'''
    )
    (out_dir / "colpali.svg").write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="860" height="280" viewBox="0 0 860 280">
  <rect width="860" height="280" fill="{BG}"/>
  <rect x="12" y="12" width="836" height="256" rx="16" fill="{PANEL}" stroke="{GRID}"/>
  <text x="32" y="44" fill="{TEXT}" font-size="18" font-weight="700" font-family="ui-sans-serif, system-ui, sans-serif">Page-image retrieval  ·  ColPali-style MaxSim</text>
  <text x="32" y="66" fill="{MUTED}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">Tables stay pixels. Patch grid, late interaction, fused with text and CLIP/SigLIP via RRF.</text>
  <rect x="36" y="88" width="200" height="160" rx="10" fill="#121212" stroke="{SKY}"/>
  <text x="136" y="112" text-anchor="middle" fill="{SKY}" font-size="12" font-family="ui-monospace, monospace">PDF page</text>
  <rect x="56" y="128" width="72" height="48" rx="4" fill="{PANEL}" stroke="{GOLD}"/>
  <rect x="140" y="128" width="72" height="48" rx="4" fill="{PANEL}" stroke="{BLUE}"/>
  <rect x="56" y="184" width="72" height="48" rx="4" fill="{PANEL}" stroke="{PINK}"/>
  <rect x="140" y="184" width="72" height="48" rx="4" fill="{PANEL}" stroke="{LAVENDER}"/>
  <text x="270" y="168" fill="{MUTED}" font-size="22" font-family="ui-sans-serif, system-ui, sans-serif">→</text>
  <rect x="320" y="108" width="220" height="120" rx="10" fill="#121212" stroke="{GOLD}"/>
  <text x="430" y="148" text-anchor="middle" fill="{GOLD}" font-size="14" font-family="ui-sans-serif, system-ui, sans-serif">MaxSim</text>
  <text x="430" y="176" text-anchor="middle" fill="{MUTED}" font-size="12" font-family="ui-monospace, monospace">Σ_i max_j q_i · d_j</text>
  <text x="560" y="168" fill="{MUTED}" font-size="22" font-family="ui-sans-serif, system-ui, sans-serif">→</text>
  <rect x="600" y="108" width="220" height="120" rx="10" fill="#121212" stroke="{BLUE}"/>
  <text x="710" y="148" text-anchor="middle" fill="{BLUE}" font-size="14" font-family="ui-sans-serif, system-ui, sans-serif">RRF fusion</text>
  <text x="710" y="176" text-anchor="middle" fill="{MUTED}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">text + page + image</text>
</svg>
'''
    )
    (out_dir / "corpus-tree.svg").write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="860" height="320" viewBox="0 0 860 320">
  <rect width="860" height="320" fill="{BG}"/>
  <rect x="12" y="12" width="836" height="296" rx="16" fill="{PANEL}" stroke="{GRID}"/>
  <text x="32" y="44" fill="{TEXT}" font-size="18" font-weight="700" font-family="ui-sans-serif, system-ui, sans-serif">The desk  ·  61 frozen files, one experiment tree</text>
  <text x="32" y="66" fill="{MUTED}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">YAML, Slurm, .out logs, ablation CSV, two paper drafts, source, checkpoint sidecars. Hash f1ba0386…</text>
  <rect x="32" y="88" width="150" height="88" rx="10" fill="#121212" stroke="{GOLD}"/>
  <text x="107" y="124" text-anchor="middle" fill="{GOLD}" font-size="13" font-family="ui-sans-serif, system-ui, sans-serif">configs/</text>
  <text x="107" y="148" text-anchor="middle" fill="{MUTED}" font-size="11" font-family="ui-monospace, monospace">16 live + 3 archive</text>
  <rect x="198" y="88" width="150" height="88" rx="10" fill="#121212" stroke="{LAVENDER}"/>
  <text x="273" y="124" text-anchor="middle" fill="{LAVENDER}" font-size="13" font-family="ui-sans-serif, system-ui, sans-serif">logs/ + slurm/</text>
  <text x="273" y="148" text-anchor="middle" fill="{MUTED}" font-size="11" font-family="ui-monospace, monospace">run_040.out … 055</text>
  <rect x="364" y="88" width="150" height="88" rx="10" fill="#121212" stroke="{BLUE}"/>
  <text x="439" y="124" text-anchor="middle" fill="{BLUE}" font-size="13" font-family="ui-sans-serif, system-ui, sans-serif">paper/</text>
  <text x="439" y="148" text-anchor="middle" fill="{MUTED}" font-size="11" font-family="ui-monospace, monospace">draft_v1  vs  v2</text>
  <rect x="530" y="88" width="150" height="88" rx="10" fill="#121212" stroke="{PINK}"/>
  <text x="605" y="124" text-anchor="middle" fill="{PINK}" font-size="13" font-family="ui-sans-serif, system-ui, sans-serif">src/</text>
  <text x="605" y="148" text-anchor="middle" fill="{MUTED}" font-size="11" font-family="ui-monospace, monospace">fusion.py</text>
  <rect x="696" y="88" width="132" height="88" rx="10" fill="#121212" stroke="{SKY}"/>
  <text x="762" y="124" text-anchor="middle" fill="{SKY}" font-size="13" font-family="ui-sans-serif, system-ui, sans-serif">results/</text>
  <text x="762" y="148" text-anchor="middle" fill="{MUTED}" font-size="11" font-family="ui-monospace, monospace">ablation.csv</text>
  <rect x="32" y="196" width="796" height="88" rx="10" fill="#121212" stroke="{GRID}"/>
  <text x="52" y="228" fill="{MUTED}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">A researcher types into localhost:8000. A content manager tags the same tree. An engineer asks for fusion.py. Same index.</text>
  <text x="52" y="256" fill="{GOLD}" font-size="13" font-family="ui-monospace, monospace">136 gold questions  ·  8 categories  ·  no LLM judge required</text>
</svg>
'''
    )
    (out_dir / "hybrid-split.svg").write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="860" height="300" viewBox="0 0 860 300">
  <rect width="860" height="300" fill="{BG}"/>
  <rect x="12" y="12" width="836" height="276" rx="16" fill="{PANEL}" stroke="{GRID}"/>
  <text x="32" y="44" fill="{TEXT}" font-size="18" font-weight="700" font-family="ui-sans-serif, system-ui, sans-serif">Same query  ·  two retrievers  ·  RRF</text>
  <text x="32" y="66" fill="{MUTED}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">“what does the fusion module do”  —  BM25 wants the filename. Dense wants the idea.</text>
  <rect x="32" y="84" width="380" height="180" rx="12" fill="#121212" stroke="{LAVENDER}"/>
  <text x="52" y="112" fill="{LAVENDER}" font-size="12" font-family="ui-monospace, monospace">BM25</text>
  <text x="52" y="140" fill="{TEXT}" font-size="13" font-family="ui-sans-serif, system-ui, sans-serif">1  src/fusion.py</text>
  <text x="52" y="164" fill="{MUTED}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">token “fusion” in the path</text>
  <text x="52" y="196" fill="{MUTED}" font-size="13" font-family="ui-sans-serif, system-ui, sans-serif">2  configs/run_047.yaml</text>
  <text x="52" y="220" fill="{MUTED}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">fusion: true</text>
  <rect x="448" y="84" width="380" height="180" rx="12" fill="#121212" stroke="{BLUE}"/>
  <text x="468" y="112" fill="{BLUE}" font-size="12" font-family="ui-monospace, monospace">dense</text>
  <text x="468" y="140" fill="{TEXT}" font-size="13" font-family="ui-sans-serif, system-ui, sans-serif">1  configs where fusion is off</text>
  <text x="468" y="164" fill="{MUTED}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">paraphrase of “fusion was off”</text>
  <text x="468" y="196" fill="{GOLD}" font-size="13" font-family="ui-sans-serif, system-ui, sans-serif">RRF keeps fusion.py at rank 1</text>
  <text x="468" y="220" fill="{MUTED}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">{hybrid_recall}</text>
</svg>
'''
    )
    (out_dir / "stack.svg").write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="860" height="280" viewBox="0 0 860 280">
  <rect width="860" height="280" fill="{BG}"/>
  <rect x="12" y="12" width="836" height="256" rx="16" fill="{PANEL}" stroke="{GRID}"/>
  <text x="32" y="44" fill="{TEXT}" font-size="18" font-weight="700" font-family="ui-sans-serif, system-ui, sans-serif">What you run at localhost:8000</text>
  <text x="32" y="66" fill="{MUTED}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">Same radial gradient. Next.js talks to FastAPI. FastAPI talks to Postgres: pgvector and full text.</text>
  <rect x="32" y="92" width="250" height="140" rx="12" fill="#121212" stroke="{SKY}"/>
  <text x="157" y="128" text-anchor="middle" fill="{SKY}" font-size="14" font-weight="700" font-family="ui-sans-serif, system-ui, sans-serif">Next.js</text>
  <text x="157" y="156" text-anchor="middle" fill="{MUTED}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">.frontend/</text>
  <text x="157" y="180" text-anchor="middle" fill="{TEXT}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">chat, files, approve</text>
  <rect x="304" y="92" width="250" height="140" rx="12" fill="#121212" stroke="{GOLD}"/>
  <text x="429" y="128" text-anchor="middle" fill="{GOLD}" font-size="14" font-weight="700" font-family="ui-sans-serif, system-ui, sans-serif">FastAPI</text>
  <text x="429" y="156" text-anchor="middle" fill="{MUTED}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">app/</text>
  <text x="429" y="180" text-anchor="middle" fill="{TEXT}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">route · retrieve · agents</text>
  <rect x="576" y="92" width="250" height="140" rx="12" fill="#121212" stroke="{LAVENDER}"/>
  <text x="701" y="128" text-anchor="middle" fill="{LAVENDER}" font-size="14" font-weight="700" font-family="ui-sans-serif, system-ui, sans-serif">Postgres</text>
  <text x="701" y="156" text-anchor="middle" fill="{MUTED}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">pgvector + full text</text>
  <text x="701" y="180" text-anchor="middle" fill="{TEXT}" font-size="12" font-family="ui-sans-serif, system-ui, sans-serif">dense + ts_rank_cd + mtime</text>
</svg>
'''
    )


def write_from_results(root: Path) -> None:
    out_dir = root / "doc" / "figures"
    # The committed baseline is the artifact the README table is rendered from, so the
    # figures read it too (falling back to latest.json before a baseline exists).
    results_dir = root / "bench" / "results"
    source = results_dir / "main.json"
    if not source.exists():
        source = results_dir / "latest.json"
    write_static_story_svgs(out_dir, json.loads(source.read_text()) if source.exists() else None)
    if source.exists():
        blob = json.loads(source.read_text())
        ndcg, rec10, rec50 = [], [], []
        ndcg_ci, rec10_ci, rec50_ci, rand50 = [], [], [], []
        colors = [GOLD, LAVENDER, BLUE, LOSS, SKY, PINK]
        for i, row in enumerate(blob.get("results", [])):
            name = row["config"]
            r = row["retrieval"]
            ci = row.get("ci") or {}
            color = colors[i % len(colors)]

            def interval(metric: str):
                block = ci.get(metric)
                return (block["lo"], block["hi"]) if block else None

            ndcg.append((name, r["ndcg@10"], color))
            ndcg_ci.append(interval("ndcg@10"))
            rec10.append((name, r.get("recall@10", 0.0), color))
            rec10_ci.append(interval("recall@10"))
            rec50.append((name, r["recall@50"], color))
            rec50_ci.append(interval("recall@50"))
            rand50.append(r.get("random_recall@50"))
        n_gold = blob.get("n_gold", "")
        n_files = blob.get("n_files", "")
        boot = blob.get("bootstrap") or {}
        ci_note = f"95% CI: paired percentile bootstrap over queries, {boot.get('n_boot', '')} resamples, seed {boot.get('seed', '')}." if boot else ""
        from app.eval.report import components_excluding_zero

        excl = components_excluding_zero(blob)
        _bar_chart(
            out_dir / "ndcg.svg",
            f"nDCG@10  ·  {n_gold} questions, {n_files} files, hash embeddings, overlap reranker",
            f"Added components whose gain over the previous row excludes zero: {', '.join(excl) if excl else 'none'}. {ci_note}",
            ndcg,
            ci=ndcg_ci,
        )
        _bar_chart(
            out_dir / "recall.svg",
            f"Recall@10  ·  {n_gold} questions, {n_files} files",
            f"The recall number that is not saturated by list length. {ci_note}",
            rec10,
            ci=rec10_ci,
        )
        _bar_chart(
            out_dir / "recall50.svg",
            f"Recall@50 next to a random list of the same length  ·  {n_files} files",
            "Dashed tick per bar: expected Recall@50 of a random list with that bar's number of unique files. A bar near its tick is measuring list length.",
            rec50,
            ci=rec50_ci,
            marker=rand50,
            marker_label="random list, same length",
        )
    sweeps = root / "bench" / "results" / "sweeps.json"
    if sweeps.exists():
        s = json.loads(sweeps.read_text())
        if s.get("hnsw"):
            xs = [p["ef_search"] for p in s["hnsw"]]
            ys = [p["recall@10"] for p in s["hnsw"]]
            y2 = [p["p50_ms"] for p in s["hnsw"]]
            _line_chart(
                out_dir / "hnsw.svg",
                "HNSW ef_search sweep  ·  recall vs latency",
                f"{s.get('n_chunks', 0)} chunk vectors, dim {s.get('embed_dim', 0)}. Hash-space NSW saturates; pgvector HNSW on bge-large is the production knob.",
                xs, ys, y2, "ef_search", "Recall@10", "p50 ms",
            )
        if s.get("storage"):
            series = []
            palette = [BLUE, GOLD, PINK]
            for i, p in enumerate(s["storage"]):
                series.append((f"{p['mode']}  ({p['bytes_per_vec']} B/vec)", p["recall@10"], palette[i % 3]))
            _bar_chart(
                out_dir / "storage.svg",
                "Storage vs recall  ·  float32 / halfvec / binary+rescore",
                "halfvec is 2x smaller. Binary is ~32x smaller, then a float rescoring pass.",
                series,
            )
        if s.get("graph_hops"):
            hops = []
            palette = [BLUE, GOLD, PINK]
            for i, p in enumerate(s["graph_hops"]):
                hops.append(
                    (
                        p["config"],
                        p["retrieval"]["recall@50"],
                        palette[i % 3],
                    )
                )
            by_hops = {p["config"]: p["retrieval"] for p in s["graph_hops"]}
            zero, one = by_hops.get("hybrid+hops=0"), by_hops.get("hybrid+hops=1")
            note = (f"One hop {zero['recall@50']:.3f} → {one['recall@50']:.3f} recall. nDCG@10 falls "
                    f"{zero['ndcg@10']:.3f} → {one['ndcg@10']:.3f}. Default stays hops=0." if zero and one else "Default stays hops=0.")
            _bar_chart(
                out_dir / "hops.svg",
                "Graph hops  ·  Recall@50",
                note,  # read from sweeps.json, never typed (it was, and matched the file)
                hops,
            )
    write_models_2026_figures(root, out_dir)  # M41: nothing is drawn while its results files are missing
    write_small_rerankers_figure(root, out_dir)  # M43: likewise
    write_small_rerankers_shapes_figure(root, out_dir)  # M51: likewise (from its frontier file)


# ---- M41: the current open models beside the app's default ones (make figures; app/eval/models_2026.py)

MODELS_2026_FIGURES = ("models-2026-embedders.svg", "models-2026-rerankers.svg")
# The app's default configuration, marked in both charts: the embedder its model-backed configurations set
# (EMBEDDING_MODEL in docker-compose.ml.yml, docker-compose.prod.yml and infra/modules/demo/ecs.tf; the code's own fallback
# in app/settings.py is bge-large-en-v1.5) and the reranker app/engine/retriever.py defaults to. A deployment can run
# another configuration, so the charts do not call it the live demo.
DEFAULT_LABEL = "app's default"
DEFAULT_NOTE = ("The app's default is bge-small with bge-reranker-v2-m3 (marked app's default); the larger models are measured "
                "here for the benchmark tables.")
VERDICTS = {"above zero": ("▲ above", GOLD), "below zero": ("▼ below", PINK), "covers zero": ("= tie", MUTED)}


def _panels_chart(path: Path, title: str, subtitle: list[str], panels: list[dict], footer: str) -> None:
    """Horizontal nDCG@10 bars on a 0-1 axis, one panel per dataset. Each row: label, bar, 95% interval whisker, the
    value with its interval, and (for a row that replaces a bge model) the verdict of the paired interval against that
    row; the row of the app's default configuration is outlined and labelled."""
    w, left, right = 1040, 330, 290
    plot_w = w - left - right
    row_h, gap, head = 20, 7, 48
    top = 70 + 16 * len(subtitle)
    h = top + sum(head + len(p["rows"]) * (row_h + gap) + 14 for p in panels) + 44
    font = 'font-family="ui-sans-serif, system-ui, sans-serif"'
    parts = []
    y = top
    for panel in panels:
        parts.append(f'<text x="32" y="{y + 18}" fill="{TEXT}" {font} font-size="14" font-weight="700">{_esc(panel["title"])}</text>')
        for g in range(6):  # the axis: 0, 0.2, ... 1.0
            x = left + plot_w * g / 5
            parts.append(f'<line x1="{x:.1f}" y1="{y + head - 6}" x2="{x:.1f}" y2="{y + head + len(panel["rows"]) * (row_h + gap) - gap}" '
                         f'stroke="{GRID}" stroke-width="1"/>')
            parts.append(f'<text x="{x:.1f}" y="{y + head - 9}" text-anchor="middle" fill="{MUTED}" {font} font-size="10">{g / 5:.1f}</text>')
        y += head
        for row in panel["rows"]:
            ym = y + row_h / 2
            bw = max(2.0, plot_w * row["value"])
            outline = f' stroke="{GOLD}" stroke-width="2"' if row.get("default") else ""
            parts.append(f'<text x="{left - 12}" y="{ym + 4:.1f}" text-anchor="end" fill="{TEXT}" {font} font-size="12">{_esc(row["label"])}</text>')
            parts.append(f'<rect x="{left}" y="{y}" width="{bw:.1f}" height="{row_h}" rx="3" fill="{row["color"]}"{outline}/>')
            text = f'{row["value"]:.3f}'
            if row.get("ci"):
                lo, hi = row["ci"]
                x_lo, x_hi = left + plot_w * lo, left + plot_w * hi
                whisker = f"M{x_lo:.1f},{ym:.1f}H{x_hi:.1f}M{x_lo:.1f},{ym - 5:.1f}V{ym + 5:.1f}M{x_hi:.1f},{ym - 5:.1f}V{ym + 5:.1f}"
                # a dark halo under a light stroke: the interval reads on a light bar and on the dark panel alike
                parts.append(f'<path d="{whisker}" stroke="{BG}" stroke-width="4" fill="none"/>'
                             f'<path d="{whisker}" stroke="{TEXT}" stroke-width="1.6" fill="none"/>')
                text += f"  [{lo:.3f}, {hi:.3f}]"
            parts.append(f'<text x="{left + plot_w + 12}" y="{ym + 4:.1f}" fill="{GOLD}" {font} font-size="12" font-weight="600">{_esc(text)}</text>')
            tag = VERDICTS.get(row.get("verdict") or "")
            if row.get("default"):
                tag = (DEFAULT_LABEL, GOLD)
            if tag:
                parts.append(f'<text x="{w - 32}" y="{ym + 4:.1f}" text-anchor="end" fill="{tag[1]}" {font} font-size="12" font-weight="700">{_esc(tag[0])}</text>')
            y += row_h + gap
        y += 14
    sub = "".join(f'<text x="32" y="{64 + 16 * i}" fill="{MUTED}" {font} font-size="12">{_esc(line)}</text>' for i, line in enumerate(subtitle))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{_esc(title)}">
  <rect width="{w}" height="{h}" fill="{BG}"/>
  <rect x="12" y="12" width="{w - 24}" height="{h - 24}" rx="16" fill="{PANEL}" stroke="{GRID}"/>
  <text x="32" y="44" fill="{TEXT}" {font} font-size="18" font-weight="700">{_esc(title)}</text>
  {sub}
  {''.join(parts)}
  <text x="32" y="{h - 26}" fill="{MUTED}" {font} font-size="11">{_esc(footer)}</text>
</svg>
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg)


def _m41_row(blob: dict | None, name: str, label: str, color: str, *, default: bool = False, judged: bool = True) -> dict | None:
    """A chart row from a results row: nDCG@10, its 95% interval and the verdict of its paired interval against its
    parent (the bge row it replaces; judged=False for a reference row)."""
    from app.eval.report import verdict

    row = next((r for r in (blob or {}).get("results", []) if r["config"] == name), None)
    if row is None:
        return None
    ci = (row.get("ci") or {}).get("ndcg@10") or {}
    parent = (row.get("delta_vs_parent") or row.get("delta_vs_previous") or {}).get("ndcg@10")
    return {"label": label, "value": row["retrieval"]["ndcg@10"], "ci": (ci["lo"], ci["hi"]) if ci else None, "color": color,
            "default": default, "verdict": verdict(parent) if judged and parent else None}


def _size(blob: dict) -> str:
    if "n_docs" in blob:
        return f"{blob['n_docs']} documents, {blob['n_queries']} test queries"
    return f"{blob['n_gold']} questions over {blob['n_files']} files"


def write_models_2026_figures(root: Path, out_dir: Path) -> list[Path]:
    """The two M41 charts, from the committed results files (nothing is drawn for a file that is missing): nDCG@10 per
    embedder, and per reranker over hybrid@bge-small, on fixture v1, BEIR SciFact and BEIR FiQA."""
    from app.eval.models_2026 import EMBEDDERS, HARRIER, RERANKERS

    results = root / "bench" / "results"
    load = lambda name: json.loads((results / name).read_text()) if (results / name).exists() else None  # noqa: E731
    datasets = [("fixture v1", load("models_2026.json"), load("models_2026_harrier.json"), "dense_only"),
                ("BEIR SciFact", load("beir_scifact_models_2026.json"), load("beir_scifact_models_2026_harrier.json"), "dense"),
                ("BEIR FiQA", load("beir_fiqa_models_2026.json"), load("beir_fiqa_models_2026_harrier.json"), "dense")]
    datasets = [d for d in datasets if d[1]]
    if not datasets:
        return []
    bm25 = {"fixture v1": "bm25_only"}
    written = []
    embed_panels, rerank_panels = [], []
    for title, blob, harrier, dense in datasets:
        rows = [_m41_row(blob, bm25.get(title, "bm25"), "bm25 (no embedder)", LOSS, judged=False),
                _m41_row(blob, f"{dense}@bge-small", "dense · bge-small", BLUE, judged=False),
                _m41_row(blob, "hybrid@bge-small", "hybrid · bge-small", LAVENDER, default=True, judged=False)]
        for tag in [t for t in EMBEDDERS if t != HARRIER]:
            rows += [_m41_row(blob, f"{dense}@{tag}", f"dense · {tag}", BLUE), _m41_row(blob, f"hybrid@{tag}", f"hybrid · {tag}", LAVENDER)]
        if harrier:
            rows += [_m41_row(harrier, f"{dense}@{HARRIER}", f"dense · {HARRIER} (own run)", SKY),
                     _m41_row(harrier, f"hybrid@{HARRIER}", f"hybrid · {HARRIER} (own run)", SKY)]
        embed_panels.append({"title": f"{title}  ·  {_size(blob)}", "rows": [r for r in rows if r]})
        beir = "n_docs" in blob
        suffix = "/fp32-512/top20" if beir else ""
        rows = [_m41_row(blob, "hybrid@bge-small", "hybrid · bge-small, no reranker", LOSS, judged=False),
                _m41_row(blob, f"hybrid+bge-rerank@bge-small{suffix}", "+ bge-reranker-v2-m3 (fp32)", GOLD, default=True, judged=False)]
        for tag, (model, precision) in RERANKERS.items():
            name = f"hybrid+{tag}@bge-small" + (f"/{precision}-512/top20" if beir else "")
            rows.append(_m41_row(blob, name, f"+ {tag} ({precision})", SKY if "mxbai" in tag else PINK))
        how = "the fused top 20 at max_length 512" if beir else "every fused candidate, the model's own token cap"
        rerank_panels.append({"title": f"{title}  ·  {_size(blob)}  ·  reranks {how}", "rows": [r for r in rows if r]})
    sources = ", ".join(f"{t}: results sha {b.get('git_sha')}" + (f" (harrier {h.get('git_sha')})" if h else "") for t, b, h, _ in datasets)
    rerank_sources = ", ".join(f"{t}: results sha {b.get('git_sha')}" for t, b, _, _ in datasets)  # no harrier rows in that chart
    common = ["nDCG@10 with 95% bootstrap intervals over the queries (whiskers). The tag on a new model's row is its paired interval",
              "against the bge row it replaces: above or below zero, or a tie when it covers zero; every row is kept, losers included.",
              DEFAULT_NOTE]
    _panels_chart(out_dir / MODELS_2026_FIGURES[0], "Current open embedders beside bge-small  ·  nDCG@10", common, embed_panels,
                  f"make figures, from {sources}.")
    written.append(out_dir / MODELS_2026_FIGURES[0])
    _panels_chart(out_dir / MODELS_2026_FIGURES[1], "Current open rerankers over hybrid@bge-small  ·  nDCG@10", common, rerank_panels,
                  f"make figures, from {rerank_sources}.")
    written.append(out_dir / MODELS_2026_FIGURES[1])
    return written


# ---- M43: small rerankers, quality against indicative CPU time (make figures; app/eval/small_rerankers.py)

SMALL_RERANKERS_FIGURE = "small-rerankers.svg"
# Three roles, three hues: slots 1 to 3 of a palette checked for this chart's dark panel (lightness band, chroma floor,
# colour-vision separation over all pairs, contrast against the panel). The verdict is the marker's shape, never its
# colour, and every model is named by a label: no reading depends on telling two hues apart.
ROLE_COLORS = {"candidate": "#3987E5", "student": "#D95926", "default": "#199E70"}
ROLE_LABELS = {"candidate": "small cross-encoder as published", "student": "distilled student", "default": "app's default reranker"}
SERVED_SHAPE = (20, 512)  # the depth and max_length whose CPU time is the x position: the single host's
DODGE = 16.0  # pixels between the points of models whose times nearly coincide


def _nice_step(span: float) -> float:
    """A tick step of 1, 2 or 5 times a power of ten that gives four to eight ticks over `span`."""
    import math

    raw = span / 6.0
    power = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * power:
            return m * power
    return 10 * power


def _log_ticks(lo: float, hi: float) -> list[float]:
    """1, 2, 5 times powers of ten inside [lo, hi]."""
    import math

    out = []
    for e in range(math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 1):
        for m in (1, 2, 5):
            v = m * 10.0 ** e
            if lo <= v <= hi:
                out.append(v)
    return out


def _marker(x: float, y: float, verdict: str, color: str) -> str:
    """A point: filled circle (interval above zero), hollow circle (covers zero), filled triangle pointing down (below
    zero); each with a ring in the panel colour so it stays legible over a line or another point."""
    if verdict == "below zero":
        pts = f"{x - 6:.1f},{y - 5:.1f} {x + 6:.1f},{y - 5:.1f} {x:.1f},{y + 6:.1f}"
        return f'<polygon points="{pts}" fill="{color}" stroke="{PANEL}" stroke-width="2"/>'
    if verdict == "above zero":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{color}" stroke="{PANEL}" stroke-width="2"/>'
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="{PANEL}"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{PANEL}" stroke="{color}" stroke-width="2.5"/>')


def write_small_rerankers_figure(root: Path, out_dir: Path) -> list[Path]:
    """M43's chart: for each small reranker with a CPU measurement, the paired change in nDCG@10 against the unreranked
    hybrid (with its 95% interval) on the three datasets, against the indicative CPU time of one rerank call at the served
    shape. Nothing is drawn while the results files or the CPU file are missing."""
    import math

    from app.eval import small_rerankers as sr
    from app.eval.report import verdict

    cpu_path = root / "bench" / "results" / "small_rerankers_cpu.json"
    table = sr.bar_table(root)
    if not cpu_path.exists() or not table["rows"]:
        return []
    cpu = json.loads(cpu_path.read_text())
    depth, max_length = SERVED_SHAPE
    seconds = {}
    for row in cpu["results"]:
        shape = next((s for s in row["shapes"] if s["depth"] == depth), None)
        if row["max_length"] == max_length and shape:
            seconds[row["tag"]] = shape["p50_ms"] / 1000.0
    datasets = [(d, label) for d, label in (("fixture", "fixture v1, held-out half"), ("scifact", "BEIR SciFact"), ("fiqa", "BEIR FiQA")) if d in table["datasets"]]
    rows = [r for r in table["rows"] if r["tag"] in seconds]
    if not rows or not datasets:
        return []
    untimed = [r for r in table["rows"] if r["tag"] not in seconds]
    role = lambda tag: "student" if tag in sr.STUDENTS else ("default" if tag == "bge-rerank" else "candidate")  # noqa: E731

    w, left, right = 1040, 96, 44
    plot_w = w - left - right
    panel_h, panel_gap = 170, 46
    font = 'font-family="ui-sans-serif, system-ui, sans-serif"'
    x_lo, x_hi = min(seconds[r["tag"]] for r in rows) / 1.7, max(seconds[r["tag"]] for r in rows) * 1.7
    sx = lambda v: left + plot_w * (math.log10(v) - math.log10(x_lo)) / (math.log10(x_hi) - math.log10(x_lo))  # noqa: E731

    # models whose times nearly coincide (a student and the model it was trained from, or two models of one size) are
    # drawn side by side, DODGE pixels apart around their common time, each on its own dotted guide
    ordered = sorted(rows, key=lambda r: (seconds[r["tag"]], r["tag"]))
    xpos: dict[str, float] = {}
    group: list[str] = []
    for r in [*ordered, None]:
        if group and (r is None or sx(seconds[r["tag"]]) - sx(seconds[group[-1]]) >= DODGE):
            centre = sum(sx(seconds[t]) for t in group) / len(group)
            xpos.update({t: centre + (k - (len(group) - 1) / 2) * DODGE for k, t in enumerate(group)})
            group = []
        if r is not None:
            group.append(r["tag"])
    # model names above the first panel, staggered so that neighbours on the time axis do not collide
    lanes: list[float] = []  # the right edge of the last label in each lane
    placed = []
    for r in ordered:
        x, width = xpos[r["tag"]], 6.4 * len(r["tag"]) + 10
        x0 = min(max(x - width / 2, 16), w - 16 - width)
        lane = next((i for i, edge in enumerate(lanes) if x0 >= edge + 6), len(lanes))
        if lane == len(lanes):
            lanes.append(0.0)
        lanes[lane] = x0 + width
        placed.append((r["tag"], x, x0 + width / 2, lane))
    label_h = 18 * len(lanes)
    subtitle = [
        "Paired change in nDCG@10 against hybrid@bge-small with no reranker (dots, with 95% bootstrap intervals), over the fused top 20 at max_length 512.",
        f"Time: one rerank call of 20 passages cut at 512 tokens in the API process, p50, {cpu['device'].get('threads')} threads on "
        f"{cpu['device'].get('cpu_model') or 'the recorded CPU'}:",
        "another machine, not the demo's host. Models whose times nearly coincide are drawn side by side, each on its own dotted guide.",
        "A model passes the quality bar when its interval is above zero in all three panels. Whether it fits the host is measured on the host.",
    ]
    top = 70 + 16 * len(subtitle) + 10
    first_panel = top + label_h + 16
    h = first_panel + len(datasets) * (panel_h + panel_gap) + 86
    parts = []
    for tag, x, cx, lane in placed:
        y = top + 12 + 18 * lane
        parts.append(f'<line x1="{x:.1f}" y1="{y + 5}" x2="{x:.1f}" y2="{first_panel + len(datasets) * (panel_h + panel_gap) - panel_gap}" stroke="{GRID}" stroke-width="1" stroke-dasharray="2 4"/>')
        parts.append(f'<circle cx="{cx - 3.2 * len(tag) - 9:.1f}" cy="{y - 4}" r="4" fill="{ROLE_COLORS[role(tag)]}"/>')
        parts.append(f'<text x="{cx + 3:.1f}" y="{y}" text-anchor="middle" fill="{TEXT}" {font} font-size="11.5">{_esc(tag)}</text>')
    for i, (d, label) in enumerate(datasets):
        y0 = first_panel + i * (panel_h + panel_gap)
        cells = [(r, r["datasets"][d]) for r in rows if d in r["datasets"]]
        refs = [(r, r["datasets"][d]) for r in untimed if d in r["datasets"]]
        values = [0.0] + [v for _, c in cells for v in (c["vs_parent"]["lo"], c["vs_parent"]["hi"])] + [c["vs_parent"]["delta"] for _, c in refs]
        step = _nice_step(max(values) - min(values))
        y_lo, y_hi = math.floor(min(values) / step) * step, math.ceil(max(values) / step) * step
        sy = lambda v, y0=y0, y_lo=y_lo, y_hi=y_hi: y0 + panel_h * (y_hi - v) / (y_hi - y_lo)  # noqa: E731
        n = table["datasets"][d]["n"]
        parts.append(f'<text x="{left}" y="{y0 - 12}" fill="{TEXT}" {font} font-size="13" font-weight="700">{_esc(label)}</text>'
                     f'<text x="{left + 7.4 * len(label) + 10:.1f}" y="{y0 - 12}" fill="{MUTED}" {font} font-size="11.5">{n} questions; '
                     f'no reranker: nDCG@10 {table["datasets"][d]["parent_ndcg@10"]["mean"]:.3f}</text>')
        ticks = round((y_hi - y_lo) / step)
        for k in range(ticks + 1):
            v = y_lo + k * step
            y = sy(v)
            zero = abs(v) < step / 1000
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="{MUTED if zero else GRID}" stroke-width="{1.4 if zero else 1}"/>')
            parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" fill="{MUTED}" {font} font-size="10.5">{v:+.2f}</text>')
        for v in _log_ticks(x_lo, x_hi):
            parts.append(f'<line x1="{sx(v):.1f}" y1="{y0 + panel_h}" x2="{sx(v):.1f}" y2="{y0 + panel_h + 4}" stroke="{MUTED}" stroke-width="1"/>')
            if i == len(datasets) - 1:
                parts.append(f'<text x="{sx(v):.1f}" y="{y0 + panel_h + 17}" text-anchor="middle" fill="{MUTED}" {font} font-size="10.5">{v:g} s</text>')
        parts.append(f'<text x="{left + plot_w - 4}" y="{sy(0.0) - 5:.1f}" text-anchor="end" fill="{MUTED}" {font} font-size="10.5">no reranker</text>')
        names = []  # a reference's name goes on top of the points, with a halo, so no whisker crosses it
        for r, c in refs:  # a reference with no CPU time: a dashed line at its change, named at the left
            y = sy(c["vs_parent"]["delta"])
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="{MUTED}" stroke-width="1.2" stroke-dasharray="6 5"/>')
            names.append(f'<text x="{left + 6}" y="{y - 5:.1f}" fill="{MUTED}" stroke="{PANEL}" stroke-width="4" paint-order="stroke" {font} '
                         f'font-size="10.5">{_esc(r["tag"])} {c["vs_parent"]["delta"]:+.3f} (not timed on a CPU)</text>')
        for r, c in cells:
            x, block = xpos[r["tag"]], c["vs_parent"]
            y_a, y_b = sy(block["hi"]), sy(block["lo"])
            whisker = f"M{x:.1f},{y_a:.1f}V{y_b:.1f}M{x - 4:.1f},{y_a:.1f}H{x + 4:.1f}M{x - 4:.1f},{y_b:.1f}H{x + 4:.1f}"
            parts.append(f'<path d="{whisker}" stroke="{PANEL}" stroke-width="4" fill="none"/><path d="{whisker}" stroke="{TEXT}" stroke-width="1.4" fill="none"/>')
            parts.append(_marker(x, sy(block["delta"]), verdict(block), ROLE_COLORS[role(r["tag"])]))
        parts.extend(names)
        parts.append(f'<text x="{left - 62}" y="{y0 + panel_h / 2:.0f}" fill="{MUTED}" {font} font-size="10.5" text-anchor="middle" '
                     f'transform="rotate(-90 {left - 62} {y0 + panel_h / 2:.0f})">Δ nDCG@10</text>')
    base = first_panel + len(datasets) * (panel_h + panel_gap) - panel_gap
    parts.append(f'<text x="{left + plot_w / 2:.0f}" y="{base + 36}" text-anchor="middle" fill="{MUTED}" {font} font-size="11.5">'
                 f'indicative CPU time of one rerank call, seconds (log scale)</text>')
    x = left
    for key in ("candidate", "student", "default"):
        if any(role(r["tag"]) == key for r in rows):
            parts.append(f'<circle cx="{x + 5}" cy="{base + 58}" r="5" fill="{ROLE_COLORS[key]}"/>'
                         f'<text x="{x + 16}" y="{base + 62}" fill="{TEXT}" {font} font-size="11.5">{_esc(ROLE_LABELS[key])}</text>')
            x += 26 + 6.6 * len(ROLE_LABELS[key])
    x += 14
    for shape, text in (("above zero", "interval above zero"), ("covers zero", "covers zero"), ("below zero", "below zero")):
        parts.append(_marker(x + 6, base + 58, shape, MUTED) + f'<text x="{x + 18}" y="{base + 62}" fill="{TEXT}" {font} font-size="11.5">{text}</text>')
        x += 30 + 6.6 * len(text)
    sub = "".join(f'<text x="32" y="{64 + 16 * i}" fill="{MUTED}" {font} font-size="12">{_esc(line)}</text>' for i, line in enumerate(subtitle))
    sources = ", ".join(f"{d}: results sha {table['datasets'][d]['git_sha']}" for d, _ in datasets) + f", CPU: results sha {cpu.get('git_sha')}"
    title = "Small rerankers  ·  quality against indicative CPU time"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{_esc(title)}">
  <rect width="{w}" height="{h}" fill="{BG}"/>
  <rect x="12" y="12" width="{w - 24}" height="{h - 24}" rx="16" fill="{PANEL}" stroke="{GRID}"/>
  <text x="32" y="44" fill="{TEXT}" {font} font-size="18" font-weight="700">{_esc(title)}</text>
  {sub}
  {"".join(parts)}
  <text x="32" y="{h - 26}" fill="{MUTED}" {font} font-size="11">{_esc("make figures, from " + sources + ".")}</text>
</svg>
'''
    out = out_dir / SMALL_RERANKERS_FIGURE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg)
    return [out]





# ---- M51: the small rerankers at cheaper shapes, quality against indicative CPU p95 (make figures; app/eval/small_rerankers_shapes.py)

SMALL_RERANKERS_SHAPES_FIGURE = "small-rerankers-shapes.svg"
# The same three role hues as M43's chart (slots 1 to 3 of the reference palette's dark steps; all-pairs validated on
# PANEL). Identity is never the hue alone: each model's six cells are joined by a line with its name at the end, and the
# second model of a role is dashed. The verdict is the marker's shape.
SHAPE_ROLE = {"minilm-l6": "candidate", "minilm-l12": "candidate", "student-minilm-l6": "student", "student-minilm-l12": "student",
              "gte-modernbert-base": "default", "bge-rerank": "default"}
SHAPE_DASHED = ("minilm-l12", "student-minilm-l12", "bge-rerank")
SHAPE_ROLE_LABELS = {"candidate": "published MiniLM cross-encoder", "student": "distilled student (attempt 1)", "default": "larger reference (over budget at every shape)"}
VERDICT_WORD = {"win": "above zero", "tie": "covers zero", "loss": "below zero"}


def write_small_rerankers_shapes_figure(root: Path, out_dir: Path) -> list[Path]:
    """M51's chart: every cell's paired change in nDCG@10 against no reranker (95% interval) on the three datasets,
    against its indicative CPU p95, with the 1,000 and 2,500 ms band edges. Nothing is drawn without the frontier file."""
    import math

    from app.eval import small_rerankers_shapes as shapes

    path = root / shapes.FRONTIER_FILE
    if not path.exists():
        return []
    front = json.loads(path.read_text())
    cells = front["cells"]
    datasets = (("fixture", "fixture v1, held-out half"), ("scifact", "BEIR SciFact"), ("fiqa", "BEIR FiQA"))
    w, left, right = 1040, 96, 190
    plot_w = w - left - right
    panel_h, panel_gap = 190, 50
    font = 'font-family="ui-sans-serif, system-ui, sans-serif"'
    secs = [c["indicative_cpu"]["p95_ms"] / 1000.0 for c in cells]
    x_lo, x_hi = min(secs) / 1.4, max(secs) * 1.4
    sx = lambda v: left + plot_w * (math.log10(v) - math.log10(x_lo)) / (math.log10(x_hi) - math.log10(x_lo))  # noqa: E731
    rules = shapes.BAND_RULES
    subtitle = [
        "Paired change in nDCG@10 against hybrid@bge-small with no reranker (markers, with 95% bootstrap intervals) for each model at depth 5, 10 and 20",
        f"and max_length 256 and 512; x is the cell's indicative p95 for one rerank call on one HPC CPU ({front['limit_mib']} MiB api limit), not the demo's host.",
        f"Dashed verticals: the {rules['p95_ms_under']:g} ms and {rules['p95_ms_borderline']:g} ms band edges; a cell left of an edge can still be in a higher band by its memory peak.",
        f"F1 answer (under-indicative-budget cells): {front['answers']['F1']['answer']}; F2 answer (with the borderline cells): {front['answers']['F2']['answer']}.",
    ]
    top = 70 + 16 * len(subtitle) + 18
    h = top + len(datasets) * (panel_h + panel_gap) + 70
    parts = []
    for i, (d, label) in enumerate(datasets):
        y0 = top + i * (panel_h + panel_gap)
        blocks = [(c, c["datasets"][d]["vs_parent"]) for c in cells if c["datasets"][d].get("vs_parent")]
        values = [0.0] + [v for _, b in blocks for v in (b["lo"], b["hi"])]
        step = _nice_step(max(values) - min(values))
        y_lo, y_hi = math.floor(min(values) / step) * step, math.ceil(max(values) / step) * step
        sy = lambda v, y0=y0, y_lo=y_lo, y_hi=y_hi: y0 + panel_h * (y_hi - v) / (y_hi - y_lo)  # noqa: E731
        parts.append(f'<text x="{left}" y="{y0 - 12}" fill="{TEXT}" {font} font-size="13" font-weight="700">{_esc(label)}</text>')
        for k in range(round((y_hi - y_lo) / step) + 1):
            v = y_lo + k * step
            zero = abs(v) < step / 1000
            parts.append(f'<line x1="{left}" y1="{sy(v):.1f}" x2="{left + plot_w}" y2="{sy(v):.1f}" stroke="{MUTED if zero else GRID}" stroke-width="{1.4 if zero else 1}"/>'
                         f'<text x="{left - 8}" y="{sy(v) + 4:.1f}" text-anchor="end" fill="{MUTED}" {font} font-size="10.5">{v:+.2f}</text>')
        for edge in (rules["p95_ms_under"] / 1000.0, rules["p95_ms_borderline"] / 1000.0):
            parts.append(f'<line x1="{sx(edge):.1f}" y1="{y0}" x2="{sx(edge):.1f}" y2="{y0 + panel_h}" stroke="{MUTED}" stroke-width="1" stroke-dasharray="5 4"/>')
            if i == 0:
                parts.append(f'<text x="{sx(edge) + 4:.1f}" y="{y0 + 12}" fill="{MUTED}" {font} font-size="10.5">{edge * 1000:,.0f} ms</text>')
        for v in _log_ticks(x_lo, x_hi):
            parts.append(f'<line x1="{sx(v):.1f}" y1="{y0 + panel_h}" x2="{sx(v):.1f}" y2="{y0 + panel_h + 4}" stroke="{MUTED}" stroke-width="1"/>')
            if i == len(datasets) - 1:
                parts.append(f'<text x="{sx(v):.1f}" y="{y0 + panel_h + 17}" text-anchor="middle" fill="{MUTED}" {font} font-size="10.5">{v:g} s</text>')
        ends = [[sy(0.0), left + plot_w - 6, "no reranker", MUTED]]  # the zero line is labelled in the margin, with the models
        for tag in shapes.MODELS:
            mine = sorted(((c, b) for c, b in blocks if c["tag"] == tag), key=lambda cb: cb[0]["indicative_cpu"]["p95_ms"])
            if not mine:
                continue
            color = ROLE_COLORS[SHAPE_ROLE[tag]]
            dash = ' stroke-dasharray="6 4"' if tag in SHAPE_DASHED else ""
            pts = " ".join(f"{sx(c['indicative_cpu']['p95_ms'] / 1000.0):.1f},{sy(b['delta']):.1f}" for c, b in mine)
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"{dash} opacity="0.85"/>')
            for c, b in mine:
                x = sx(c["indicative_cpu"]["p95_ms"] / 1000.0)
                whisker = f"M{x:.1f},{sy(b['hi']):.1f}V{sy(b['lo']):.1f}"
                parts.append(f'<path d="{whisker}" stroke="{PANEL}" stroke-width="4" fill="none"/><path d="{whisker}" stroke="{TEXT}" stroke-width="1.2" fill="none" opacity="0.7"/>')
                parts.append(_marker(x, sy(b["delta"]), VERDICT_WORD[shapes.verdict(b)], color))
            last_c, last_b = mine[-1]
            ends.append([sy(last_b["delta"]), sx(last_c["indicative_cpu"]["p95_ms"] / 1000.0), tag, color])
        ends.sort()
        previous = -1e9
        for e in ends:  # labels in the right margin, top to bottom, at least 13 px apart, each with a leader to its line's end
            previous = max(e[0], previous + 13)
            e.append(previous)
        for y, x, tag, color, ly in ends:
            lx = left + plot_w + 14
            parts.append(f'<line x1="{x + 6:.1f}" y1="{y:.1f}" x2="{lx - 3}" y2="{ly:.1f}" stroke="{GRID}" stroke-width="1"/>'
                         f'<text x="{lx}" y="{ly + 4:.1f}" fill="{MUTED if color == MUTED else TEXT}" {font} font-size="11">{_esc(tag)}</text>')
        parts.append(f'<text x="{left - 62}" y="{y0 + panel_h / 2:.0f}" fill="{MUTED}" {font} font-size="10.5" text-anchor="middle" '
                     f'transform="rotate(-90 {left - 62} {y0 + panel_h / 2:.0f})">Δ nDCG@10</text>')
    base = top + len(datasets) * (panel_h + panel_gap) - panel_gap
    parts.append(f'<text x="{left + plot_w / 2:.0f}" y="{base + 36}" text-anchor="middle" fill="{MUTED}" {font} font-size="11.5">'
                 f'indicative p95 of one rerank call on the HPC CPU, seconds (log scale)</text>')
    x = left - 60
    for key in ("candidate", "student", "default"):
        parts.append(f'<circle cx="{x + 5}" cy="{base + 58}" r="5" fill="{ROLE_COLORS[key]}"/>'
                     f'<text x="{x + 16}" y="{base + 62}" fill="{TEXT}" {font} font-size="11.5">{_esc(SHAPE_ROLE_LABELS[key])}</text>')
        x += 26 + 6.2 * len(SHAPE_ROLE_LABELS[key])
    x = left - 60  # the verdict's marker shapes on a row of their own, under the colours
    for shape, text in (("above zero", "95% interval above zero"), ("covers zero", "covers zero"), ("below zero", "below zero")):
        parts.append(_marker(x + 6, base + 80, shape, MUTED) + f'<text x="{x + 18}" y="{base + 84}" fill="{TEXT}" {font} font-size="11.5">{text}</text>')
        x += 30 + 6.2 * len(text)
    sub = "".join(f'<text x="32" y="{64 + 16 * i}" fill="{MUTED}" {font} font-size="12">{_esc(line)}</text>' for i, line in enumerate(subtitle))
    title = "Small rerankers at cheaper shapes  ·  quality against indicative CPU p95"
    h += 20
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{_esc(title)}">
  <rect width="{w}" height="{h}" fill="{BG}"/>
  <rect x="12" y="12" width="{w - 24}" height="{h - 24}" rx="16" fill="{PANEL}" stroke="{GRID}"/>
  <text x="32" y="44" fill="{TEXT}" {font} font-size="18" font-weight="700">{_esc(title)}</text>
  {sub}
  {"".join(parts)}
  <text x="32" y="{h - 26}" fill="{MUTED}" {font} font-size="11">{_esc("make figures, from " + shapes.FRONTIER_FILE + " at " + str(front.get("git_sha")) + "; dashed: the second model of a colour.")}</text>
</svg>
'''
    out = out_dir / SMALL_RERANKERS_SHAPES_FIGURE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg)
    return [out]


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    write_from_results(root)
    print("wrote", root / "doc" / "figures")


if __name__ == "__main__":
    main()
