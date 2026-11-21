"""doc/deploy.md, the AWS runbook, and the two links to it (the README and the reference's Run it section).

doc/deploy.md is outside `make doc-numbers-check`, so its own rule is checked here: every dollar figure names where it
comes from and when it was read (an AWS pricing page and its date, the table's prices, or the file in this repository
that sets it), and the monthly subtotal is the sum of its rows. Nothing here reaches the network.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNBOOK = (ROOT / "doc" / "deploy.md").read_text()
READ_ON = "2026-11-20"
# A line with a dollar figure must carry one of these: a source with its date, the table's own prices, or a file.
SOURCED = re.compile(rf"aws\.amazon\.com/[\w/-]+.*{READ_ON}|{READ_ON}.*aws\.amazon\.com|at the prices above|at the same prices"
                     r"|rows? above|app/[\w/]+\.py")
MONEY = re.compile(r"\$\d[\d,]*(?:\.\d+)?")


def _money(text: str) -> float:
    return float(text.replace("$", "").replace(",", ""))


def test_every_dollar_figure_names_its_source_and_when_it_was_read():
    unsourced = [line for line in RUNBOOK.splitlines() if MONEY.search(line) and not SOURCED.search(line)
                 and not line.lstrip().startswith(("#", "terraform", "aws ", "gh "))]
    assert unsourced == []
    assert f"read {READ_ON}" in RUNBOOK and "730 hours" in RUNBOOK


def test_the_monthly_subtotal_is_the_sum_of_its_rows_and_each_row_is_price_times_quantity():
    table = RUNBOOK.split("| Resource (production, always on)", 1)[1].split("\n\n", 1)[0]
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in table.splitlines()[2:]]
    *items, subtotal = rows
    assert abs(sum(_money(row[3]) for row in items) - _money(subtotal[3].strip("*"))) < 0.01
    checked = 0
    for name, unit, formula, monthly, _source in items:
        prices = [_money(m) for m in MONEY.findall(unit)]
        count = re.match(r"(\d+(?:\.\d+)?) ", formula)
        quantity = float(count.group(1)) if count and not formula.startswith("730 h") else 1.0
        hours = 730 if "730 h" in formula else 1
        if len(prices) == 1:  # the WAF row has two prices: $5 + 3 x $1 is checked below
            assert abs(prices[0] * quantity * hours - _money(monthly)) < 0.006, name
            checked += 1
    assert checked == len(items) - 1
    (waf,) = [row for row in items if row[0] == "WAF"]
    assert waf[2] == "1 web ACL + 3 rules" and _money(waf[3]) == 5 + 3 * 1
    assert "$0.04048 per vCPU-hour" in table and "$0.004445 per GB-hour" in table


def test_the_runbook_names_no_account_id_secret_or_real_address():
    assert not re.search(r"(?<![\w.:-])\d{12}(?![\w])", RUNBOOK)
    assert set(re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", RUNBOOK)) <= {"you@example.com"}
    assert "sk-ant-" not in RUNBOOK and "AKIA" not in RUNBOOK


def test_the_runbook_covers_the_owners_steps_in_order():
    order = ["Rotate the leaked secrets first", "The AWS account", "Apply the bootstrap", "GitHub settings",
             "Apply production", "The Anthropic key", "## The first deploy", "## Rollback", "## Monthly costs",
             "## Rotating keys", "## Raising or cutting the budget", "## Destroy staging or everything",
             "## What has not been run"]
    positions = [RUNBOOK.index(heading) for heading in order]
    assert positions == sorted(positions)
    assert "Required reviewers" in RUNBOOK and "`main` only" in RUNBOOK  # the approval and the branch rule are set by hand
    assert "**Status: code only.**" in RUNBOOK


def test_the_readme_and_the_reference_link_the_runbook():
    readme = " ".join((ROOT / "README.md").read_text().split())
    assert "[Deploy to AWS](doc/deploy.md)" in readme
    assert "**Public demo mode (not deployed yet).**" in readme  # still true: nothing is applied
    reference = (ROOT / "doc" / "engineering-reference.md").read_text()
    run_it = reference.split("## Run it", 1)[1].split("## What we are not claiming", 1)[0]
    assert "](deploy.md)" in run_it
