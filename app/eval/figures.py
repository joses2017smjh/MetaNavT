"""Brand-colored SVG figures for the README. Palette matches .frontend/app/globals.css."""

from __future__ import annotations

import json
from pathlib import Path

# App chrome: #121212 body, lavender / gold / sky / pink radial gradient.
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


def _bar_chart(path: Path, title: str, subtitle: str, series: list[tuple[str, float, str]], ymax: float = 1.0) -> None:
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
        bars.append(
            f'<rect x="{left}" y="{y}" width="{bw:.1f}" height="{bar_h:.1f}" rx="4" fill="{color}"/>'
            f'<text x="{left - 12}" y="{y + bar_h * 0.68:.1f}" text-anchor="end" fill="{TEXT}" '
            f'font-family="ui-sans-serif, system-ui, sans-serif" font-size="13">{_esc(label)}</text>'
            f'<text x="{left + bw + 8:.1f}" y="{y + bar_h * 0.68:.1f}" fill="{GOLD}" '
            f'font-family="ui-sans-serif, system-ui, sans-serif" font-size="13" font-weight="600">'
            f"{value:.3f}</text>"
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


def write_static_story_svgs(out_dir: Path) -> None:
    """Pipeline / query-trace / staleness cards. Numbers live in ablation.svg."""
    out_dir.mkdir(parents=True, exist_ok=True)
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


def write_from_results(root: Path) -> None:
    out_dir = root / "doc" / "figures"
    write_static_story_svgs(out_dir)
    latest = root / "bench" / "results" / "latest.json"
    if latest.exists():
        blob = json.loads(latest.read_text())
        rec, ndcg = [], []
        colors = [GOLD, LAVENDER, BLUE, LOSS, SKY, PINK]
        for i, row in enumerate(blob.get("results", [])):
            name = row["config"]
            rec.append((name, row["retrieval"]["recall@50"], colors[i % len(colors)]))
            ndcg.append((name, row["retrieval"]["ndcg@10"], colors[i % len(colors)]))
        _bar_chart(
            out_dir / "recall.svg",
            "Recall@50  ·  frozen 136-question gold set",
            "Hash dense + in-memory BM25. Hybrid RRF is the jump. Losers stay in the table.",
            rec,
        )
        _bar_chart(
            out_dir / "ndcg.svg",
            "nDCG@10  ·  same gold set, same budget",
            "Staleness Tier 1 is the ranking win. Overlap reranker did not beat RRF.",
            ndcg,
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
            _bar_chart(
                out_dir / "hops.svg",
                "Graph hops  ·  Recall@50",
                "One hop 0.938 → 0.986 recall. nDCG@10 falls 0.495 → 0.446. Default stays hops=0.",
                hops,
            )


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    write_from_results(root)
    print("wrote", root / "doc" / "figures")


if __name__ == "__main__":
    main()
