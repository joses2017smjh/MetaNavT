"""Generate the approval-gated spreadsheet → MATLAB demo artifact."""

from __future__ import annotations

import json
from pathlib import Path

from app.artifacts.visualization import propose_visualization
from app.eval.harness import project_root
from app.eval.demo_operator import operator_fixture
from app.mcp.filesystem import ApprovalRequired, FilesystemTools


QUESTION = "Compare average val RMSE by encoder against the 0.055 baseline"
SOURCE = "bench/corpus/files/results/ablation.csv"
SCRIPT = "doc/matlab/mean_val_rmse_by_encoder.m"
CHART = "doc/figures/mean-val-rmse-by-encoder.png"


def build(root: Path | None = None, *, backend: str = "matlab") -> dict:
    root = root or project_root()
    # Never overwrite real files through an agent approval bypass. Execution
    # runs in an explicitly authorized disposable operator fixture; publishing
    # its generated demo output below is an ordinary host demo export.
    with operator_fixture() as (tools, issue):
        copied = tools.root / SOURCE
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes((root / SOURCE).read_bytes())
        plan = propose_visualization(tools.root, SOURCE, QUESTION)
        plan.script_path = SCRIPT
        plan.chart_path = CHART
        tools.visualizations[plan.plan_id] = plan
        tools._register_review(plan.plan_id)
        blocked = False
        try:
            tools.apply_visualization(plan.plan_id)
        except ApprovalRequired:
            blocked = True
        options = {"chart_type": plan.recommended_chart, "execute": True, "backend": backend}
        result = tools.apply_visualization(plan.plan_id, approval_token=issue(plan.plan_id, **options), **options)
        for path in (SCRIPT, CHART):
            generated = tools.root / path
            if generated.is_file():
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(generated.read_bytes())
    checks = [
        {
            "label": "schema inferred before aggregation",
            "pass": plan.group_by == "encoder" and plan.value == "val_rmse",
            "value": f"{plan.operation}({plan.value}) by {plan.group_by}",
        },
        {
            "label": "user checkpoint blocked the first apply",
            "pass": blocked,
            "value": "ApprovalRequired" if blocked else "not blocked",
        },
        {
            "label": "MATLAB code written",
            "pass": (root / SCRIPT).is_file(),
            "value": SCRIPT,
        },
        {
            "label": "chart rendered locally",
            "pass": result["chart_exists"] and bool(result["execution"]["ok"]),
            "value": f"{result['execution']['backend']}: {CHART}",
        },
    ]
    return {
        "id": "matlab-visualization",
        "gold_id": None,
        "title": "Spreadsheet aggregation to MATLAB chart",
        "feature": "visualization",
        "method": "inspect → aggregate → recommend → disposable operator fixture → MATLAB",
        "question": QUESTION,
        "category": "synthetic disposable operator fixture (not human approval, retrieval gold or OS sandbox)",
        "gold_answer": "mean val RMSE by encoder with a 0.055 baseline",
        "gold_paths": [SOURCE],
        "aggregation": {
            "group_by": plan.group_by,
            "value": plan.value,
            "operation": plan.operation,
            "rows": plan.aggregated_rows,
        },
        "recommended_chart": plan.recommended_chart,
        "alternatives": plan.alternatives,
        "decision_trace": plan.decision_trace,
        "user_questions": plan.user_questions,
        "selected_chart": result["selected_chart"],
        "code_preview": "\n".join(plan.matlab_code.splitlines()[:36]),
        "script_path": SCRIPT,
        "image_path": "figures/mean-val-rmse-by-encoder.png",
        "execution": result["execution"],
        "checks": checks,
        "status": "pass" if all(check["pass"] for check in checks) else "mixed",
        "answer_source": "local CSV + generated MATLAB code",
    }


def write(root: Path | None = None, *, backend: str = "matlab") -> Path:
    root = root or project_root()
    payload = build(root, backend=backend)
    out = root / "doc" / "demo" / "visualization.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def main() -> None:
    path = write()
    print("wrote", path, path.stat().st_size)


if __name__ == "__main__":
    main()
