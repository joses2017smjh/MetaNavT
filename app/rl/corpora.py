"""Training corpora for the search policy (M44): seeded variants of the fixture's corpus, and the leakage rules.

Fixture v1 is the test set, so the policy never trains on it. The corpus builder (bench/corpus/build.py) writes a
corpus and its gold questions from a Spec; with the default Spec that is fixture v1, byte for byte. variant_spec(seed)
draws another Spec: sixteen runs whose ids lie outside fixture v1's 40-55, encoder and bark names from pools that
hold none of fixture v1's, a different baseline, and every value (encoder, fusion, pairs, bark, learning rate, RMSE)
drawn again, so neither a question's text nor a count ("how many runs used ...") carries over from one corpus to
another. The draw is then patched until the Spec's contract holds (bench.corpus.build.Spec), so every question
template has an answer in every variant.

training_set() applies the two drop rules declared before any training run:

  (a) a question whose normalised text equals the text of any fixture v1 question (tune or held-out half);
  (b) a question whose answer is a constant of the builder (the same string for every Spec: the sentences about
      src/fusion.py, "baseline val RMSE", a fixed path), because under strict exact match such a question would
      teach the policy a held-out gold string.

What is left still shares its TEMPLATES with the test questions ("what learning rate did run N use"); that is stated
with the result, not removed. Variant corpora are written at run time, outside the repository (`make
search-policy-corpora`, or `python -m app.rl.corpora --seeds 1-32 --out DIR`), and never into bench/.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from app.eval.corpus import load_manifest, verify_manifest
from app.eval.gold import GoldQuestion, load_gold
from app.eval.provenance import sha256_file
from bench.corpus import build as corpus_build
from bench.corpus.build import V1, Encoder, Spec, ablation_names, build_gold
from bench.corpus.registry import AblationRow, Run

ROOT = Path(__file__).resolve().parents[2]
N_RUNS = 16
# None of fixture v1's encoders (dinov2, resnet50, clip) or bark types (oak, pine, maple, birch).
ENCODER_POOL = (
    Encoder("convnext", "ConvNeXt", "convnext", "convnext_base", "ConvNeXt-B"),
    Encoder("swinv2", "SwinV2", "swin", "swinv2_base_window12", "Swin-B/4"),
    Encoder("siglip", "SigLIP", "siglip", "siglip_base_patch16", "ViT-B/16"),
    Encoder("effnetb3", "EfficientNet-B3", "effnet", "efficientnet_b3", "B3"),
    Encoder("mae", "MAE", "mae", "mae_vitl16", "ViT-L/16"),
    Encoder("beit", "BEiT", "beit", "beit_base_patch16", "ViT-B/16"),
    Encoder("regnety16", "RegNetY-16", "regnet", "regnety_160", "16GF"),
    Encoder("deit3", "DeiT-3", "deit", "deit3_small_patch16", "ViT-S/16"),
    Encoder("eva02", "EVA-02", "eva", "eva02_base_patch14", "ViT-B/14"),
    Encoder("sam", "SAM", "sam", "sam_vit_b", "ViT-B/16"),
)
BARK_POOL = ("cedar", "ash", "elm", "fir", "spruce", "walnut", "alder", "beech", "larch", "hemlock", "cherry", "poplar")
LEARNING_RATES = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3)
BASELINES = (0.0480, 0.0500, 0.0520, 0.0530, 0.0570, 0.0590, 0.0600, 0.0620, 0.0640)  # never fixture v1's 0.0550
RUN_SEEDS = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)
FIRST_RUN_IDS = range(60, 980, 10)  # a block of sixteen ids starts here: 60..75 up to 970..985, never 40..55


def _short_lr(lr: float) -> str:
    return corpus_build._short_lr(lr)


def variant_spec(seed: int) -> Spec:
    """The Spec of training corpus `seed` (a positive integer): a pure function of the seed."""
    if seed < 1:
        raise ValueError("variant seeds start at 1 (fixture v1 is bench.corpus.build.V1, not a variant)")
    rng = random.Random(f"metanavit-search-policy-corpus-{seed}")
    enc_a, enc_b, enc_c = rng.sample(ENCODER_POOL, 3)
    bark_a, bark_b, bark_c, bark_d = rng.sample(BARK_POOL, 4)
    barks = (bark_a, bark_b, bark_c, bark_d)
    first = rng.choice(FIRST_RUN_IDS)
    baseline = rng.choice(BASELINES)

    rows = [{"run_id": first + i,
             "encoder": rng.choices((enc_a, enc_b, enc_c), weights=(5, 3, 2))[0].key,
             "fusion": rng.random() < 0.55,
             "num_pairs": rng.choices((1, 2, 3), weights=(2, 5, 6))[0],
             "bark": rng.choice(barks),
             "lr": rng.choice(LEARNING_RATES),
             "seed": rng.choice(RUN_SEEDS),
             "notes": ""} for i in range(N_RUNS)]

    # Eight runs carry the patterns the ablation rows (and through them the questions) name.
    best, off_b, on_b, off_a, on_a, c_fusion, b_three, a_one = rng.sample(range(N_RUNS), 8)
    rows[best].update(encoder=enc_a.key, fusion=True, num_pairs=3, bark=bark_a, notes="best current")
    rows[off_b].update(encoder=enc_b.key, fusion=False)
    rows[on_b].update(encoder=enc_b.key, fusion=True)
    rows[off_a].update(encoder=enc_a.key, fusion=False)
    rows[on_a].update(encoder=enc_a.key, fusion=True, num_pairs=2, notes="beats baseline")
    rows[c_fusion].update(encoder=enc_c.key, fusion=True)
    rows[b_three].update(encoder=enc_b.key, fusion=True, num_pairs=3, bark=bark_b)
    rows[a_one].update(encoder=enc_a.key, fusion=True, num_pairs=1)
    free = [i for i in range(N_RUNS) if i not in (best, off_b, on_b, off_a, on_a, c_fusion, b_three, a_one)]
    rows[rng.choice(free)]["bark"] = bark_c  # the third bark the questions name is in the corpus
    for i, row in enumerate(rows):  # the headline run is the only one with its encoder, pair count and bark
        if i != best and (row["encoder"], row["num_pairs"], row["bark"]) == (enc_a.key, 3, bark_a):
            row["bark"] = rng.choice((bark_b, bark_c, bark_d))

    # RMSE in units of 1e-4, all different: a few runs under the baseline, the headline run lowest.
    unit = round(baseline * 10000)
    n_low = rng.randint(3, 7)
    low = sorted(rng.sample(range(unit - 45, unit), n_low))
    high = rng.sample(range(unit + 1, unit + 180), N_RUNS - n_low)
    rows[best]["val_rmse"] = low[0]
    second = rng.choice(low[1:])
    rows[on_a]["val_rmse"] = second
    rest = [v for v in low[1:] if v != second] + high
    rng.shuffle(rest)
    for i in range(N_RUNS):
        if i not in (best, on_a):
            rows[i]["val_rmse"] = rest.pop()
    used = {row["val_rmse"] for row in rows}
    older = rng.sample([v for v in range(unit + 20, unit + 160) if v not in used], 3)

    def run(row: dict, **changes) -> Run:
        values = {**row, **changes}
        return Run(values["run_id"], values["encoder"], values["fusion"], values["num_pairs"], values["bark"], values["lr"],
                   round(values["val_rmse"] / 10000, 4), seed=values["seed"], current=values.get("current", True),
                   version=values.get("version", 2), notes=values["notes"])

    runs = tuple(run(row) for row in rows)
    third = rng.choice([i for i in range(N_RUNS) if i not in (best, on_a)])  # the third run with trellis wires
    lr_best = rng.choice([lr for lr in LEARNING_RATES if lr != rows[best]["lr"]])
    lr_on_a = rng.choice([lr for lr in LEARNING_RATES if lr != rows[on_a]["lr"]])
    archived = (
        run(rows[best], lr=lr_best, val_rmse=older[0], current=False, version=1, notes=f"superseded learning rate {_short_lr(lr_best)}"),
        run(rows[on_a], lr=lr_on_a, val_rmse=older[1], current=False, version=1, notes="superseded lr"),
        run(rows[third], fusion=not rows[third]["fusion"], val_rmse=older[2], current=False, version=1,
            notes=f"superseded fusion {'off' if rows[third]['fusion'] else 'on'}"),
    )
    spec_runs = {name: runs[i] for name, i in zip(
        ablation_names(Spec(runs, archived, (), baseline, (enc_a, enc_b, enc_c), (bark_a, bark_b, bark_c), ())),
        (off_b, on_b, off_a, on_a, best, c_fusion, b_three, a_one))}
    ablations = tuple(AblationRow(name, r.encoder, r.fusion, r.val_rmse, r.run_id) for name, r in spec_runs.items())
    spec = Spec(runs=runs, archived=archived, ablations=ablations, baseline_rmse=baseline, encoders=(enc_a, enc_b, enc_c),
                barks=(bark_a, bark_b, bark_c), trellis_runs=tuple(sorted(runs[i].run_id for i in (best, on_a, third))))
    problems = spec_problems(spec)
    if problems:
        raise AssertionError(f"variant {seed} breaks the Spec contract: {problems}")
    return spec


def spec_problems(spec: Spec) -> list[str]:
    """What stops build_gold from answering every template for this Spec (empty: nothing). Fixture v1 passes."""
    out: list[str] = []
    runs = {r.run_id: r for r in spec.runs}
    if len(runs) != len(spec.runs):
        out.append("two runs share an id")
    if len({r.val_rmse for r in spec.runs}) != len(spec.runs):
        out.append("two runs share a val RMSE (lowest and highest must be unique)")
    names = [row.name for row in spec.ablations]
    if tuple(names) != ablation_names(spec):
        out.append(f"ablation rows are {names}, not {list(ablation_names(spec))}")
    for row in spec.ablations:
        r = runs.get(row.run_id)
        if r is None or (r.encoder, r.fusion, r.val_rmse) != (row.encoder, row.fusion, row.val_rmse):
            out.append(f"ablation row {row.name} does not describe run {row.run_id}")
    if out:
        return out
    enc_a = spec.encoders[0]
    best = min(spec.runs, key=lambda r: r.val_rmse)
    if not (best.encoder == enc_a.key and best.fusion and best.num_pairs == 3 and best.bark == spec.barks[0]):
        out.append("the lowest-RMSE run is not the headline encoder with fusion on, three pairs and the first bark")
    if not best.val_rmse < spec.baseline_rmse:
        out.append("the lowest-RMSE run does not beat the baseline")
    if best.run_id not in spec.trellis_runs:
        out.append("the lowest-RMSE run has no checkpoint metadata (not in trellis_runs)")
    if sum(1 for r in spec.runs if (r.encoder, r.num_pairs, r.bark) == (enc_a.key, 3, spec.barks[0])) != 1:
        out.append("more than one run has the headline encoder, three pairs and the first bark")
    if not any(r.num_pairs == 2 and r.val_rmse < spec.baseline_rmse for r in spec.runs):
        out.append("no two-pair run beats the baseline")
    by_name = {row.name: runs[row.run_id] for row in spec.ablations}
    if by_name[f"{enc_a.short}_3pair_fusion"].run_id != best.run_id:
        out.append("the three-pair fusion row is not the lowest-RMSE run")
    if len(spec.archived) != 3 or any(a.run_id not in runs or a.current for a in spec.archived):
        out.append("archived must hold older versions of three current runs")
    elif spec.archived[0].run_id != best.run_id:
        out.append("the first archived run is not an older version of the lowest-RMSE run")
    if any(i not in runs for i in spec.trellis_runs):
        out.append("trellis_runs names a run that does not exist")
    if out:
        return out
    for q in build_gold(spec):
        if not q.answer.strip() or not q.relevant_paths:
            out.append(f"{q.id} ({q.question!r}) has an empty answer or no relevant path")
    return out


def normalise_question(text: str) -> str:
    """Lower case, whitespace collapsed: the form in which a training question is compared with a fixture question."""
    return " ".join(text.lower().split())


@dataclass(frozen=True)
class TrainQuestion:
    corpus: int  # the variant seed
    id: str  # its id in that corpus's gold file
    question: str
    category: str
    answer: str
    relevant_paths: tuple[str, ...]
    template: str

    @property
    def key(self) -> str:
        return f"v{self.corpus:04d}/{self.id}"


def training_set(seeds: Sequence[int], fixture_gold: Sequence[GoldQuestion] | None = None) -> tuple[list[TrainQuestion], dict]:
    """(the training questions of the variant corpora `seeds`, the leakage report). A pure function of the seeds and
    of fixture v1's gold file: nothing is read from a variant's files, so the report is the same wherever it runs."""
    fixture = list(fixture_gold) if fixture_gold is not None else load_gold(ROOT / "bench" / "gold" / "questions.jsonl")
    fixture_text = {normalise_question(q.question): q.id for q in fixture}
    fixture_answers = {q.answer.strip().lower() for q in fixture}
    fixture_run_ids = {r.run_id for r in V1.runs}
    kept: list[TrainQuestion] = []
    dropped_text: Counter = Counter()
    dropped_constant: Counter = Counter()
    n_built = n_constant = 0
    overlap_ids: set[int] = set()
    for seed in seeds:
        spec = variant_spec(seed)
        overlap_ids |= {r.run_id for r in spec.runs} & fixture_run_ids
        meta: list[dict] = []
        for q, m in zip(build_gold(spec, meta), meta):
            n_built += 1
            n_constant += bool(m["constant_answer"])
            if normalise_question(q.question) in fixture_text:  # rule (a)
                dropped_text[m["template"]] += 1
            elif m["constant_answer"]:  # rule (b)
                dropped_constant[m["template"]] += 1
            else:
                kept.append(TrainQuestion(seed, q.id, q.question, q.category, q.answer, tuple(q.relevant_paths), m["template"]))
    v1_meta: list[dict] = []
    build_gold(V1, v1_meta)
    template_of = {m["id"]: m["template"] for m in v1_meta}
    kept_templates = {q.template for q in kept}
    same_answer = Counter(q.answer.strip().lower() for q in kept if q.answer.strip().lower() in fixture_answers)
    report = {
        "seeds": list(seeds),
        "n_corpora": len(seeds),
        "n_questions_built": n_built,
        "rule_a": "dropped: normalised text (lower case, whitespace collapsed) equals a fixture v1 question's, tune or held-out half",
        "rule_b": ("dropped: the answer is a constant of the builder, the same string for every corpus (it would teach a "
                   "held-out gold string under strict exact match); counted after rule (a)"),
        "n_dropped_same_text_as_a_fixture_question": sum(dropped_text.values()),
        "dropped_same_text_by_template": dict(sorted(dropped_text.items())),
        "n_constant_answer_questions_built": n_constant,
        "n_dropped_constant_answer": sum(dropped_constant.values()),
        "dropped_constant_answer_by_template": dict(sorted(dropped_constant.items())),
        "n_kept": len(kept),
        "kept_by_category": dict(sorted(Counter(q.category for q in kept).items())),
        "kept_by_template": dict(sorted(Counter(q.template for q in kept).items())),
        # checked on what was kept, not assumed from the rules
        "n_kept_with_a_fixture_question_text": sum(1 for q in kept if normalise_question(q.question) in fixture_text),
        "variant_run_ids_also_in_fixture_v1": sorted(overlap_ids),
        "kept_answers_equal_to_a_fixture_answer": {
            "note": ("a kept training answer that is also some fixture answer: short values a corpus must be read for "
                     "(a pair count, a learning rate, yes / no); none is a phrase"),
            "n_questions": sum(same_answer.values()),
            "answers": dict(sorted(same_answer.items())),
            "longest_in_words": max((len(a.split()) for a in same_answer), default=0),
        },
        "fixture_templates_without_a_training_question": sorted(set(template_of.values()) - kept_templates),
        "fixture_questions_whose_template_is_not_trained": sorted(i for i, t in template_of.items() if t not in kept_templates),
        "templates_shared_with_fixture_v1": ("every kept training question is built from a template that also builds fixture "
                                             "v1 questions: same wording, other run ids, names and values"),
    }
    return kept, report


def variant_root(out: Path, seed: int) -> Path:
    return Path(out) / f"variant_{seed:04d}"


def write_variants(seeds: Sequence[int], out: Path) -> list[dict]:
    """Write each variant corpus under `out` (never under this repository) and return what identifies it."""
    out = Path(out).resolve()
    if out == ROOT.resolve() or ROOT.resolve() in out.parents:
        raise ValueError("variant corpora are written outside the repository (fixture v1 is the only corpus under bench/)")
    rows = []
    for seed in seeds:
        root = variant_root(out, seed)
        summary = corpus_build.main(root, variant_spec(seed))
        files_root = root / "bench" / "corpus" / "files"
        problems = verify_manifest(files_root, load_manifest(root / "bench" / "corpus" / "MANIFEST.json"))
        if problems:
            raise RuntimeError(f"variant {seed} does not verify against its own manifest: {problems[:3]}")
        rows.append({"seed": seed, "n_files": summary["n_files"], "n_gold": summary["n_gold"], "corpus_sha256": summary["sha256"],
                     "gold_sha256": sha256_file(root / "bench" / "gold" / "questions.jsonl")})
    return rows


def corpus_identities(seeds: Sequence[int]) -> list[dict]:
    """What write_variants would record for `seeds`, computed in a temporary directory (the corpora are not kept)."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="search-policy-corpora-") as tmp:
        return write_variants(seeds, Path(tmp))


def parse_seeds(text: str) -> list[int]:
    """"1-32" or "1,2,5-8" -> the seeds, in order."""
    seeds: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        seeds += list(range(int(lo), int(hi or lo) + 1))
    return seeds


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default="1-32", help="variant seeds: 1-32, or 1,2,5-8")
    p.add_argument("--out", type=Path, required=True, help="a directory outside the repository")
    args = p.parse_args(argv)
    seeds = parse_seeds(args.seeds)
    rows = write_variants(seeds, args.out)
    kept, report = training_set(seeds)
    blob = {"corpora": rows, "leakage": report, "questions": [asdict(q) for q in kept]}
    (Path(args.out) / "training_set.json").write_text(json.dumps(blob, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k.startswith("n_") or k in ("kept_by_category",)}, indent=1))
    print(f"wrote {len(rows)} corpora and training_set.json under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
