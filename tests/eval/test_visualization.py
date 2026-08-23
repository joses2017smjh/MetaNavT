"""Approval-gated spreadsheet aggregation and MATLAB visualization."""

import pytest

from app.agent.schemas import VisualizationSpec
from app.artifacts.visualization import (
    available_backend,
    inspect_spreadsheet,
    propose_visualization,
)
from app.mcp.filesystem import ApprovalRequired, FilesystemTools


CSV = """ablation,encoder,fusion,val_rmse,run_id
fusion_off_resnet,resnet50,False,0.0612,40
fusion_on_resnet,resnet50,True,0.0588,41
fusion_off_dino,dinov2,False,0.0568,45
fusion_on_dino,dinov2,True,0.0542,46
dino_3pair_fusion,dinov2,True,0.0521,47
clip_fusion,clip,True,0.0595,44
resnet_3pair,resnet50,True,0.0571,42
dino_1pair,dinov2,True,0.0548,52
"""


def _sheet(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    path = results / "ablation.csv"
    path.write_text(CSV)
    return path


def test_inspect_and_propose_mean_rmse_threshold_dot_plot(tmp_path):
    path = _sheet(tmp_path)
    report = inspect_spreadsheet(path)
    kinds = {column["name"]: column["kind"] for column in report["columns"]}
    assert report["n_rows"] == 8
    assert kinds["encoder"] == "categorical"
    assert kinds["val_rmse"] == "numeric"

    plan = propose_visualization(
        tmp_path,
        "results/ablation.csv",
        "Compare average val RMSE by encoder against the 0.055 baseline",
    )
    assert plan.group_by == "encoder"
    assert plan.value == "val_rmse"
    assert plan.operation == "mean"
    assert plan.recommended_chart == "dot"
    assert plan.baseline == 0.055
    values = {row["encoder"]: row["mean_val_rmse"] for row in plan.aggregated_rows}
    assert values == {"clip": 0.0595, "dinov2": 0.054475, "resnet50": 0.059033}
    assert plan.user_questions
    assert plan.decision_trace[-1]["stage"] == "checkpoint"
    assert "textscan" in plan.matlab_code
    assert "scatter(" in plan.matlab_code
    assert "0.055" in plan.matlab_code


def test_visualization_requires_input_then_accepts_chart_override(tmp_path):
    _sheet(tmp_path)
    tools = FilesystemTools(root=tmp_path)
    pending = tools.propose_visualization(
        "results/ablation.csv",
        "Compare average val RMSE by encoder",
    )
    assert pending["status"] == "pending_approval"
    assert pending["requires_user_input"] is True
    with pytest.raises(ApprovalRequired):
        tools.apply_visualization(pending["plan_id"], approved=False)

    result = tools.apply_visualization(
        pending["plan_id"],
        approved=True,
        chart_type="bar",
        execute=False,
    )
    assert result["status"] == "applied"
    assert result["selected_chart"] == "bar"
    script = tmp_path / result["script_path"]
    assert script.is_file()
    assert "bar(values" in script.read_text()
    assert result["chart_exists"] is False


def test_octave_renders_generated_matlab_chart_when_available(tmp_path):
    _sheet(tmp_path)
    backend, _ = available_backend("octave")
    if backend is None:
        pytest.skip("Octave is not installed")
    tools = FilesystemTools(root=tmp_path)
    pending = tools.propose_visualization(
        "results/ablation.csv",
        "Compare average val RMSE by encoder against the 0.055 baseline",
    )
    result = tools.apply_visualization(
        pending["plan_id"],
        approved=True,
        execute=True,
        backend="octave",
    )
    if not result["execution"]["ok"] and "shared libraries" in result["execution"]["stderr"]:
        pytest.skip("installed Octave has unresolved shared-library dependencies")
    assert result["status"] == "applied", result["execution"]["stderr"]
    assert result["execution"]["backend"] == "octave"
    chart = tmp_path / result["chart_path"]
    assert chart.is_file()
    assert chart.stat().st_size > 1_000


def test_visualization_tools_and_schema(tmp_path):
    _sheet(tmp_path)
    tools = FilesystemTools(root=tmp_path)
    names = {tool["name"] for tool in tools.tool_specs()}
    assert {
        "inspect_spreadsheet",
        "propose_visualization",
        "apply_visualization",
    } <= names
    spec = VisualizationSpec(
        plan_id="v1",
        source_path="results/ablation.csv",
        question="average RMSE by encoder",
        group_by="encoder",
        value="val_rmse",
        operation="mean",
        recommended_chart="bar",
        alternatives=["dot", "line"],
        script_path="artifacts/visualizations/rmse.m",
        chart_path="artifacts/visualizations/rmse.png",
    )
    assert spec.requires_user_input is True
    assert spec.requires_approval is True
    assert spec.writes_tree is False
