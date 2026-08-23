"""Looping README GIFs that replay the live demo. Palette matches globals.css."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

BG = (18, 18, 18)
PANEL = (26, 26, 36)
TEXT = (244, 244, 248)
MUTED = (168, 176, 192)
BLUE = (142, 180, 232)
LAVENDER = (186, 186, 233)
GOLD = (232, 214, 160)
PINK = (251, 218, 239)
SKY = (194, 213, 255)
LINE = (42, 42, 56)

SANS = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"
SANS_B = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf"
MONO_B = "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono-Bold.ttf"


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _round_rect(draw: ImageDraw.ImageDraw, xy, r, fill=None, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=width)


def _canvas(w: int, h: int) -> Image.Image:
    img = Image.new("RGBA", (w, h), (*BG, 255))
    blobs = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(blobs)
    d.ellipse((-80, -90, 340, 280), fill=(186, 186, 233, 70))
    d.ellipse((w - 360, -120, w + 80, 260), fill=(232, 214, 160, 68))
    d.ellipse((w - 280, 80, w + 40, 420), fill=(194, 213, 255, 80))
    d.ellipse((-60, 140, 280, h + 40), fill=(251, 218, 239, 55))
    blobs = blobs.filter(ImageFilter.GaussianBlur(42))
    return Image.alpha_composite(img, blobs)


def _chrome(base: Image.Image, title: str) -> Image.Image:
    img = base.copy()
    d = ImageDraw.Draw(img)
    w, h = img.size
    _round_rect(d, (18, 18, w - 18, h - 18), 18, fill=(*PANEL, 235), outline=LINE, width=1)
    d.ellipse((36, 36, 52, 52), fill=(251, 218, 239))
    d.ellipse((62, 36, 78, 52), fill=(232, 214, 160))
    d.ellipse((88, 36, 104, 52), fill=(194, 213, 255))
    d.text((124, 36), title, fill=MUTED, font=_font(MONO, 14))
    return img


def _paste_mascot(img: Image.Image, xy: tuple[int, int], size: int = 44) -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / ".frontend" / "public" / "metanavit.jpeg"
    if not path.exists():
        return
    m = Image.open(path).convert("RGBA").resize((size, size), Image.Resampling.NEAREST)
    img.paste(m, xy, m)


def _chip(d, xy, label, active: bool):
    x0, y0, x1, y1 = xy
    fill = BLUE if active else BG
    outline = BLUE
    color = BG if active else TEXT
    _round_rect(d, xy, 16, fill=fill, outline=outline, width=1)
    d.text((x0 + 12, y0 + 6), label, fill=color, font=_font(SANS, 13))


def gif_ask(out: Path) -> None:
    w, h = 860, 480
    query = "what's the current learning rate for the DINOv2 run 47"
    stages = [
        ("route", "semantic", GOLD),
        ("search", "BM25 + dense  RRF k=60  top-50", LAVENDER),
        ("rerank", "overlap teacher / bge-reranker-v2-m3", BLUE),
        ("staleness", "drop  archive/run_047_v1.yaml  (1e-5)", PINK),
        ("cite", "configs/run_047.yaml  bytes 0-312", SKY),
        ("answer", "3e-4", TEXT),
    ]
    frames = []
    durations = []
    for typed in range(0, len(query) + 1, 3):
        frames.append(_ask_frame(w, h, query[:typed] + ("|" if typed < len(query) else ""), 0, show_hits=False))
        durations.append(40)
    durations[-1] = 220
    for n in range(1, len(stages) + 1):
        frames.append(_ask_frame(w, h, query, n, show_hits=n >= 4))
        durations.append(520 if n < len(stages) else 1600)
    _save(out, frames, durations)


def _ask_frame(w, h, typed: str, n_stages: int, show_hits: bool) -> Image.Image:
    img = _chrome(_canvas(w, h), "localhost:8000  /  live retrieval")
    d = ImageDraw.Draw(img)
    _paste_mascot(img, (40, 70), 48)
    d.text((100, 74), "MetaNaviT", fill=TEXT, font=_font(SANS_B, 22))
    d.text((100, 102), "Ask the frozen tree. It has to cite the live config.", fill=MUTED, font=_font(SANS, 13))
    _round_rect(d, (40, 140, w - 40, 184), 10, fill=BG, outline=GOLD, width=1)
    d.text((56, 152), typed, fill=TEXT, font=_font(MONO, 15))
    y = 204
    stage_names = ["route", "search", "rerank", "staleness", "cite", "answer"]
    x = 40
    for i, name in enumerate(stage_names):
        on = i < n_stages
        colors = [GOLD, LAVENDER, BLUE, PINK, SKY, TEXT]
        fill = colors[i] if on else BG
        fg = BG if on else MUTED
        tw = 14 * len(name) + 22
        _round_rect(d, (x, y, x + tw, y + 28), 8, fill=fill, outline=colors[i] if on else LINE, width=1)
        d.text((x + 10, y + 6), name, fill=fg if on else MUTED, font=_font(MONO, 12))
        x += tw + 8
    rows = [
        ("route", "semantic  (not a path, not an aggregation)"),
        ("search", "BM25 + dense  ->  RRF k=60  ->  top-50"),
        ("rerank", "overlap teacher in CI; bge-reranker when loaded"),
        ("staleness", "drop  configs/archive/run_047_v1.yaml   (1e-5)"),
        ("cite", "configs/run_047.yaml   bytes 0-312"),
        ("answer", "3e-4"),
    ]
    y = 248
    for i, (name, detail) in enumerate(rows):
        if i >= n_stages:
            break
        colors = [GOLD, LAVENDER, BLUE, PINK, SKY, TEXT]
        _round_rect(d, (40, y, 520, y + 32), 8, fill=BG, outline=colors[i], width=1)
        d.text((52, y + 8), f"{name:<10}  {detail}", fill=TEXT, font=_font(MONO, 12))
        y += 36
    if show_hits:
        _round_rect(d, (540, 248, w - 40, 312), 10, fill=BG, outline=GOLD, width=2)
        d.text((556, 258), "CURRENT", fill=GOLD, font=_font(MONO_B, 11))
        d.text((556, 278), "configs/run_047.yaml", fill=SKY, font=_font(MONO, 12))
        d.text((556, 296), "learning_rate: 3e-4", fill=TEXT, font=_font(MONO, 12))
        _round_rect(d, (540, 324, w - 40, 388), 10, fill=BG, outline=PINK, width=1)
        d.text((556, 334), "DROPPED", fill=PINK, font=_font(MONO_B, 11))
        d.text((556, 354), "archive/run_047_v1.yaml", fill=MUTED, font=_font(MONO, 12))
        d.text((556, 372), "learning_rate: 1e-5", fill=MUTED, font=_font(MONO, 12))
    return img.convert("RGB")


def gif_hybrid(out: Path) -> None:
    w, h = 860, 400
    frames, durs = [], []
    for step in range(1, 5):
        img = _chrome(_canvas(w, h), "localhost:8000  /  hybrid RRF")
        d = ImageDraw.Draw(img)
        d.text((40, 72), "what does the fusion module do", fill=TEXT, font=_font(SANS_B, 18))
        d.text((40, 100), "BM25 wants the filename. Dense wants the idea. RRF keeps both.", fill=MUTED, font=_font(SANS, 13))
        left_on = step >= 1
        right_on = step >= 2
        fuse_on = step >= 3
        _round_rect(d, (40, 132, 410, 340), 12, fill=BG, outline=LAVENDER if left_on else LINE, width=2)
        d.text((56, 148), "BM25", fill=LAVENDER if left_on else MUTED, font=_font(MONO_B, 13))
        if left_on:
            d.text((56, 184), "1  src/fusion.py", fill=TEXT, font=_font(SANS, 15))
            d.text((56, 208), "token  fusion  in the path", fill=MUTED, font=_font(SANS, 13))
            d.text((56, 248), "2  configs/run_047.yaml", fill=MUTED, font=_font(SANS, 15))
            d.text((56, 272), "fusion: true", fill=MUTED, font=_font(MONO, 13))
        _round_rect(d, (450, 132, 820, 340), 12, fill=BG, outline=BLUE if right_on else LINE, width=2)
        d.text((466, 148), "dense", fill=BLUE if right_on else MUTED, font=_font(MONO_B, 13))
        if right_on:
            d.text((466, 184), "1  configs where fusion is off", fill=TEXT, font=_font(SANS, 15))
            d.text((466, 208), "paraphrase of  fusion was off", fill=MUTED, font=_font(SANS, 13))
            d.text((466, 248), "2  src/fusion.py", fill=MUTED, font=_font(SANS, 15))
        if fuse_on:
            msg = "RRF  ->  src/fusion.py at rank 1     Recall@50  0.843 -> 0.938"
            _round_rect(d, (40, 352, 820, 382), 8, fill=BG, outline=GOLD, width=1)
            d.text((56, 358), msg, fill=GOLD, font=_font(MONO, 13))
        frames.append(img.convert("RGB"))
        durs.append(900 if step < 4 else 1600)
    _save(out, frames, durs)


def gif_router(out: Path) -> None:
    w, h = 860, 380
    routes = [
        ("lexical_path", "open configs/run_047.yaml", "skip embed + rerank", GOLD, "nDCG@10  0.596 -> 0.938"),
        ("aggregation", "which run had the lowest val RMSE", "skip rerank", LAVENDER, "answer  47"),
        ("semantic", "what does the fusion module do", "full pipeline", BLUE, "cite  src/fusion.py"),
    ]
    frames, durs = [], []
    for i, _ in enumerate(routes):
        img = _chrome(_canvas(w, h), "localhost:8000  /  query router")
        d = ImageDraw.Draw(img)
        d.text((40, 72), "Forty lines. No extra LLM.", fill=TEXT, font=_font(SANS_B, 18))
        d.text((40, 100), "8 queries skipped embed. 41 skipped rerank. Semantic recall did not move.", fill=MUTED, font=_font(SANS, 13))
        box_w = 240
        for j, (name, q, skip, color, result) in enumerate(routes):
            x = 40 + j * (box_w + 20)
            on = j == i
            _round_rect(d, (x, 140, x + box_w, 320), 12, fill=BG, outline=color if on else LINE, width=2)
            d.text((x + 16, 158), name, fill=color if on else MUTED, font=_font(MONO_B, 13))
            d.text((x + 16, 190), q[:28], fill=TEXT if on else MUTED, font=_font(SANS, 13))
            d.text((x + 16, 214), q[28:] or " ", fill=TEXT if on else MUTED, font=_font(SANS, 13))
            d.text((x + 16, 250), skip, fill=SKY if on else MUTED, font=_font(SANS, 13))
            if on:
                d.text((x + 16, 286), result, fill=GOLD, font=_font(MONO, 12))
        frames.append(img.convert("RGB"))
        durs.append(1400)
    _save(out, frames, durs)


def gif_mcp(out: Path) -> None:
    w, h = 860, 360
    steps = [
        (0, "1  propose_move", "logs/run_040.out\n-> archive/", "pending_approval", GOLD, False),
        (1, "2  apply_plan", "approved missing", "ApprovalRequired", PINK, False),
        (2, "3  apply_plan", "approved=True", "applied", SKY, True),
    ]
    frames, durs = [], []
    for active, *_ in steps:
        img = _chrome(_canvas(w, h), "localhost:8000  /  MCP human interrupt")
        d = ImageDraw.Draw(img)
        d.text((40, 72), "A move that will not execute", fill=TEXT, font=_font(SANS_B, 18))
        d.text((40, 100), "propose_move never writes. apply_plan is a no-op until approved=true.", fill=MUTED, font=_font(SANS, 13))
        for j, (_, title, body, status, color, _) in enumerate(steps):
            x = 40 + j * 270
            on = j <= active
            outline = color if on else LINE
            _round_rect(d, (x, 140, x + 250, 300), 12, fill=BG, outline=outline, width=2)
            d.text((x + 16, 160), title, fill=color if on else MUTED, font=_font(MONO_B, 13))
            lines = body.split("\n")
            yy = 196
            for line in lines:
                d.text((x + 16, yy), line, fill=TEXT if on else MUTED, font=_font(SANS, 13))
                yy += 20
            if on:
                d.text((x + 16, 250), status, fill=color, font=_font(MONO_B, 16))
        frames.append(img.convert("RGB"))
        durs.append(1100 if active < 2 else 1700)
    _save(out, frames, durs)


def gif_graph(out: Path) -> None:
    w, h = 860, 360
    lines = [
        "[community 0]  runs 40-55",
        "encoders  clip, dinov2, resnet50",
        "run:47  --uses_encoder-->  dinov2",
        "run:47  --has_learning_rate-->  3e-4",
        "run:47  --documented_in-->  configs/run_047.yaml",
    ]
    frames, durs = [], []
    for n in range(1, len(lines) + 1):
        img = _chrome(_canvas(w, h), "localhost:8000  /  GraphRAG")
        d = ImageDraw.Draw(img)
        d.text((40, 72), "what is in this corpus", fill=TEXT, font=_font(SANS_B, 18))
        d.text((40, 100), "Entities from the files. No invented edges.", fill=MUTED, font=_font(SANS, 13))
        _round_rect(d, (40, 140, 820, 320), 12, fill=BG, outline=SKY, width=1)
        y = 160
        for line in lines[:n]:
            d.text((60, y), line, fill=GOLD if "3e-4" in line else TEXT, font=_font(MONO, 15))
            y += 28
        frames.append(img.convert("RGB"))
        durs.append(500 if n < len(lines) else 1600)
    _save(out, frames, durs)


def _demo_case(payload: dict, case_id: str) -> dict:
    return next(case for case in payload["cases"] if case["id"] == case_id)


def _rank_panel(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    result: dict,
    color,
) -> None:
    x0, y0, x1, y1 = xy
    _round_rect(draw, xy, 12, fill=BG, outline=color, width=2)
    draw.text((x0 + 14, y0 + 12), title, fill=color, font=_font(MONO_B, 12))
    gold_rank = result.get("gold_rank")
    rank_text = f"best gold rank  {gold_rank}" if gold_rank is not None else "gold not in top-50"
    draw.text((x0 + 14, y0 + 36), rank_text, fill=TEXT, font=_font(MONO, 11))
    y = y0 + 64
    for hit in (result.get("hits") or [])[:5]:
        marker = "*" if hit.get("status") == "gold" else "x" if hit.get("status") == "superseded" else " "
        fill = GOLD if marker == "*" else PINK if marker == "x" else MUTED
        label = f"{marker} {hit['rank']:>2}  {hit['path']}"
        draw.text((x0 + 14, y), label[:43], fill=fill, font=_font(MONO, 10))
        y += 21


def gif_hard_graph(out: Path, payload: dict) -> None:
    """Generated q107/q108/q114 control-vs-PPR traces."""
    w, h = 860, 450
    frames, durs = [], []
    for case_id in ("q107", "q108", "q114"):
        case = _demo_case(payload, case_id)
        img = _chrome(_canvas(w, h), f"hard benchmark  /  {case_id}")
        d = ImageDraw.Draw(img)
        d.text((40, 70), case["title"], fill=TEXT, font=_font(SANS_B, 18))
        d.text((40, 98), case["question"][:92], fill=MUTED, font=_font(SANS, 12))
        _rank_panel(d, (40, 130, 410, 360), "CONTROL  single-query hybrid", case["control"], LAVENDER)
        _rank_panel(d, (450, 130, 820, 360), "METHOD  typed PPR rank fusion", case["method_result"], SKY)
        passed = sum(1 for row in case["checks"] if row["pass"])
        footer = (
            f"gold: {case['gold_answer']}    checks: {passed}/{len(case['checks'])} pass"
        )
        _round_rect(d, (40, 378, 820, 414), 8, fill=BG, outline=GOLD, width=1)
        d.text((54, 388), footer[:95], fill=GOLD, font=_font(MONO, 11))
        frames.append(img.convert("RGB"))
        durs.append(1900)
    _save(out, frames, durs)


def gif_hard_staleness(out: Path, payload: dict) -> None:
    """Generated q115/q117 current-vs-superseded traces."""
    w, h = 860, 430
    frames, durs = [], []
    for case_id in ("q115", "q117"):
        case = _demo_case(payload, case_id)
        img = _chrome(_canvas(w, h), f"hard benchmark  /  {case_id}")
        d = ImageDraw.Draw(img)
        d.text((40, 70), case["title"], fill=TEXT, font=_font(SANS_B, 18))
        d.text((40, 98), case["question"][:92], fill=MUTED, font=_font(SANS, 12))
        _rank_panel(d, (40, 130, 410, 350), "BEFORE  hybrid candidates", case["control"], PINK)
        _rank_panel(d, (450, 130, 820, 350), "AFTER  current-version filter", case["method_result"], SKY)
        dropped = case["method_result"].get("dropped") or []
        conflicts = case.get("conflicts") or []
        footer = f"dropped {len(dropped)} superseded paths  |  surfaced {len(conflicts)} conflict(s)"
        _round_rect(d, (40, 366, 820, 402), 8, fill=BG, outline=GOLD, width=1)
        d.text((54, 376), footer, fill=GOLD, font=_font(MONO, 11))
        frames.append(img.convert("RGB"))
        durs.append(2200)
    _save(out, frames, durs)


def gif_artifacts(out: Path, payload: dict) -> None:
    """Generated Paper2Code + sandbox + HITL trace."""
    artifact = _demo_case(payload, "artifact-run47")
    safety = _demo_case(payload, "hitl-sandbox")
    w, h = 860, 430
    frames, durs = [], []
    steps = [
        ("1  PLAN", "research-repro", "current config + current paper + source", LAVENDER),
        ("2  GENERATE", "claim-support audit", "zero unsupported claims", GOLD),
        ("3  EXECUTE", "restricted sandbox", artifact["execution"]["stdout"].strip(), SKY),
        ("4  APPLY", "without approval -> blocked", "approved temp write -> applied", PINK),
    ]
    for active in range(len(steps)):
        img = _chrome(_canvas(w, h), "coding artifact  /  Paper2Code + HITL")
        d = ImageDraw.Draw(img)
        d.text((40, 70), artifact["question"], fill=TEXT, font=_font(SANS_B, 18))
        citations = [row["path"] for row in artifact["spec"]["citations"]]
        d.text((40, 100), ("cites  " + "  |  ".join(citations))[:105], fill=MUTED, font=_font(MONO, 10))
        for i, (title, middle, result, color) in enumerate(steps):
            x = 40 + (i % 2) * 400
            y = 140 + (i // 2) * 120
            on = i <= active
            _round_rect(d, (x, y, x + 370, y + 100), 12, fill=BG, outline=color if on else LINE, width=2)
            d.text((x + 14, y + 14), title, fill=color if on else MUTED, font=_font(MONO_B, 12))
            d.text((x + 14, y + 42), middle[:46], fill=TEXT if on else MUTED, font=_font(SANS, 12))
            if on:
                d.text((x + 14, y + 68), result[:48], fill=color, font=_font(MONO, 10))
        passed = sum(1 for row in artifact["checks"] + safety["checks"] if row["pass"])
        d.text((40, 392), f"{passed}/7 deterministic checks pass  |  no corpus write during demo export", fill=GOLD, font=_font(MONO, 11))
        frames.append(img.convert("RGB"))
        durs.append(1100 if active < 3 else 2200)
    _save(out, frames, durs)


def gif_visualization(out: Path, payload: dict, root: Path) -> None:
    """Spreadsheet schema → aggregation → approval → MATLAB chart."""
    case = _demo_case(payload, "matlab-visualization")
    chart_path = root / "doc" / case["image_path"]
    chart = Image.open(chart_path).convert("RGB")
    chart.thumbnail((360, 230), Image.Resampling.LANCZOS)
    w, h = 860, 460
    stages = [
        ("inspect", "8 rows / 5 columns", LAVENDER),
        ("aggregate", "mean(val_rmse) by encoder", GOLD),
        ("recommend", "dot plot; alternatives bar / line", SKY),
        ("user input", "ApprovalRequired -> approve dot", PINK),
        ("MATLAB", "write .m -> render PNG", BLUE),
    ]
    frames, durs = [], []
    for active in range(len(stages)):
        img = _chrome(_canvas(w, h), "local spreadsheet  /  approval-gated MATLAB")
        d = ImageDraw.Draw(img)
        d.text((40, 70), case["question"], fill=TEXT, font=_font(SANS_B, 17))
        d.text((40, 98), "Auditable decision trace, not hidden chain-of-thought.", fill=MUTED, font=_font(SANS, 12))
        _round_rect(d, (40, 130, 430, 405), 12, fill=BG, outline=LAVENDER, width=1)
        y = 148
        for i, (name, detail, color) in enumerate(stages):
            on = i <= active
            d.text((58, y), f"{i + 1}  {name}", fill=color if on else MUTED, font=_font(MONO_B, 11))
            d.text((178, y), detail, fill=TEXT if on else MUTED, font=_font(MONO, 10))
            y += 40
        rows = case["aggregation"]["rows"]
        if active >= 1:
            y = 350
            values = "  ".join(
                f"{row['encoder']}={row['mean_val_rmse']:.6f}" for row in rows
            )
            d.text((58, y), values[:56], fill=GOLD, font=_font(MONO, 9))
        _round_rect(d, (450, 130, 820, 405), 12, fill=(250, 250, 250), outline=SKY if active >= 4 else LINE, width=2)
        if active >= 4:
            x = 455 + (360 - chart.width) // 2
            y_img = 145 + (230 - chart.height) // 2
            img.paste(chart, (x, y_img))
            d.text((468, 382), "MATLAB local render  |  baseline 0.055", fill=BG, font=_font(MONO_B, 10))
        else:
            d.text((530, 245), "chart waits for user approval", fill=(80, 80, 90), font=_font(MONO, 11))
        frames.append(img.convert("RGB"))
        durs.append(1100 if active < 4 else 2300)
    _save(out, frames, durs)


def _save(path: Path, frames: list[Image.Image], durations: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    first, rest = frames[0], frames[1:]
    first.save(
        path,
        save_all=True,
        append_images=rest,
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )


def write_all(root: Path | None = None) -> list[Path]:
    root = root or Path(__file__).resolve().parents[2]
    out = root / "doc" / "gifs"
    traces = root / "doc" / "demo" / "traces.json"
    if not traces.exists():
        from app.eval.demo_export import write_demo

        write_demo(root)
    payload = json.loads(traces.read_text())
    jobs = [
        ("ask.gif", gif_ask),
        ("hybrid.gif", gif_hybrid),
        ("router.gif", gif_router),
        ("mcp.gif", gif_mcp),
        ("graph.gif", gif_graph),
        ("hard-graph.gif", lambda path: gif_hard_graph(path, payload)),
        ("hard-staleness.gif", lambda path: gif_hard_staleness(path, payload)),
        ("artifacts.gif", lambda path: gif_artifacts(path, payload)),
        ("visualization.gif", lambda path: gif_visualization(path, payload, root)),
    ]
    written = []
    for name, fn in jobs:
        p = out / name
        fn(p)
        written.append(p)
        print("wrote", p, p.stat().st_size)
    return written


def main() -> None:
    write_all()


if __name__ == "__main__":
    main()
