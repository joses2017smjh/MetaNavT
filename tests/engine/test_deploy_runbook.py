"""doc/deploy.md, the AWS runbook, and the two links to it (the README and the reference's Run it section).

doc/deploy.md is outside `make doc-numbers-check`, so its own rule is checked here: every dollar figure names where it
comes from and when it was read (an AWS pricing page and its date, the table's prices, or the file in this repository
that sets it: a module under app/, or a Terraform file under infra/ for a setting such as production's daily answer
budget), and the monthly subtotal is the sum of its rows. Nothing here reaches the network. The figures of Claude's
cost are also computed from the app's price table in tests/engine/test_infra_terraform.py.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNBOOK = (ROOT / "doc" / "deploy.md").read_text()
READ_ON = "2026-11-20"
# A line with a dollar figure must carry one of these: a source with its date, the table's own prices, or a file.
# The live demo's section cites the AWS Price List's offer files by their publication dates, the Price List query API
# and AWS documentation pages read on 2026-11-21.
LIVE_SOURCED = (r"offer file, published 2026-11-(?:19|21)|Price List query API, read 2026-11-21"
                r"|\]\(https://docs\.aws\.amazon\.com/[\w./-]+\), read 2026-11-21")
SOURCED = re.compile(rf"aws\.amazon\.com/[\w/-]+.*{READ_ON}|{READ_ON}.*aws\.amazon\.com|at the prices above|at the same prices"
                     r"|rows? above|app/[\w/]+\.py|infra/[\w/.-]+\.tf|" + LIVE_SOURCED)
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
    assert "**Status: code only, but for the bootstrap.**" in RUNBOOK  # the owner applied only the bootstrap


def test_the_readme_and_the_reference_link_the_runbook():
    readme = " ".join((ROOT / "README.md").read_text().split())
    assert "[Deploy to AWS](doc/deploy.md)" in readme
    assert "**Public demo mode (not deployed yet).**" in readme  # still true: nothing is applied
    reference = (ROOT / "doc" / "engineering-reference.md").read_text()
    run_it = reference.split("## Run it", 1)[1].split("## What we are not claiming", 1)[0]
    assert "](deploy.md)" in run_it


# ---- the live demo: one host on the AWS Free plan (the runbook's first section) ----------------------------------

LIVE_HEADING = "## The live demo: one host on the AWS Free plan"
LIVE = RUNBOOK.split(LIVE_HEADING, 1)[1].split("\n## ", 1)[0]
HOURS = 730


def test_the_live_demo_comes_first_and_every_dollar_figure_in_it_names_its_source_and_date():
    assert RUNBOOK.index(LIVE_HEADING) == RUNBOOK.index("\n## ") + 1  # the first section
    unsourced = [line for line in LIVE.splitlines() if MONEY.search(line)
                 and not re.search(LIVE_SOURCED + r"|\(the rows above\)", line)]
    assert unsourced == []
    for service, published in (("AmazonEC2", "2026-11-21T00:48:16Z"), ("AmazonVPC", "2026-11-19T20:55:45Z")):
        assert (f"(https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/{service}/current/us-west-2/index.csv),\n"
                f"publication date {published}") in LIVE, service
    assert f"A\nmonth is {HOURS} hours" in LIVE or f"month is {HOURS} hours" in " ".join(LIVE.split())


def _live_prices() -> dict[str, float]:
    table = LIVE.split("| Item | Price (us-west-2) | Source |", 1)[1].split("\n\n", 1)[0]
    prices = {}
    for line in table.splitlines()[2:]:
        item, price, _source = [cell.strip() for cell in line.strip().strip("|").split("|")]
        prices[item.split(" (")[0].split(",")[0]] = _money(MONEY.findall(price)[0])
    return prices


def test_the_monthly_costs_are_the_price_lists_figures_times_730_hours():
    prices = _live_prices()
    assert prices["t3.small"] == 0.0208 and prices["c7i-flex.large"] == 0.08479 and prices["t8i.small"] == 0.02496
    storage, address = 20 * prices["EBS gp3"], HOURS * prices["Public IPv4 address"]
    table = LIVE.split("| Host | Instance (× 730 h)", 1)[1].split("\n\n", 1)[0]
    monthly = {}
    for line in table.splitlines()[2:]:
        host, instance, gp3, ipv4, month, three = [cell.strip() for cell in line.strip().strip("|").split("|")]
        name = host.split(",")[0]
        assert abs(_money(instance) - HOURS * prices[name]) < 0.006, host
        assert abs(_money(gp3) - storage) < 0.006 and abs(_money(ipv4) - address) < 0.006, host
        total = HOURS * prices[name] + storage + address
        assert abs(_money(month) - total) < 0.006, host
        assert three.startswith(f"about ${round(3 * total)} "), host
        monthly[name] = total
    assert set(monthly) == {"t3.small", "c7i-flex.large"}
    assert 3 * monthly["c7i-flex.large"] > 200  # more than all the credits
    flat = " ".join(LIVE.split())
    assert f"$100 ÷ ${monthly['t3.small']:.2f} is {100 / monthly['t3.small']:.1f} months" in flat
    assert f"$200 ÷ ${monthly['t3.small']:.2f} would be {200 / monthly['t3.small']:.1f} months, so the six-month cap" in flat
    assert monthly["t3.small"] == min(HOURS * prices[t] + storage + address for t in ("t3.small", "t8i.small", "c7i-flex.large"))


def test_the_live_demos_memory_figures_are_the_committed_record():
    import json

    record = json.loads((ROOT / "bench" / "results" / "api_memory_cpu.json").read_text())
    flat = " ".join(LIVE.split())
    assert f"peaks at {record['memory_mib']['after_embedder']['VmHWM']:,} MiB with the embedder" in flat
    assert f"at {record['memory_mib']['after_reranker']['VmHWM']:,} MiB once the reranker has scored one query's" in flat
    assert f"{record['rerank_pairs']} pairs, which took {record['rerank_seconds']:.2f} s" in flat
    assert "bench/results/api_memory_cpu.json" in flat and "scripts/measure_api_memory.py" in flat
    assert "the CUDA build of torch" in flat and "an upper bound for the image's CPU wheel" in flat
    # the ECS stack's note: its one vCPU inherits the latency, so its production profile is not the live demo's
    assert "Its one vCPU inherits the reranker's measured latency" in " ".join(RUNBOOK.split())


def test_the_owner_works_as_an_iam_user_with_aws_login_never_iam_identity_center():
    flat = " ".join(RUNBOOK.split())
    assert "create an administrator identity (IAM Identity Center)" not in flat  # the old step 2
    assert "as an IAM user with MFA signed in with `aws login`" in flat and "not IAM Identity Center" in flat
    assert "aws login --profile metanavit --remote" in LIVE and "SignInLocalDevelopmentAccess" in LIVE
    assert "AWS CLI 2.32 or later" in flat
    # the credential_process profile only as a fallback for a long apply
    fallback = LIVE.split("credential_process = ", 1)[0].rsplit("\n\n", 2)[-2]
    assert fallback.startswith("A fallback, only if a long apply")
    assert "credential_process = aws configure export-credentials --profile metanavit --format process" in LIVE


def test_the_bootstraps_state_stays_outside_every_checkout_with_a_copy_in_the_bucket():
    flat = " ".join(LIVE.split())
    assert 'export BOOTSTRAP_DIR="$HOME/' in LIVE and 'export BOOTSTRAP_STATE="$BOOTSTRAP_DIR/bootstrap.tfstate"' in LIVE
    assert ('terraform -chdir=infra/bootstrap plan -state="$BOOTSTRAP_STATE" -out="$BOOTSTRAP_DIR/bootstrap.tfplan" '
            "\\ -var 'budget_email=you@example.com' -var 'monthly_budget_usd=30'") in flat
    assert ('terraform -chdir=infra/bootstrap apply -state="$BOOTSTRAP_STATE" -state-out="$BOOTSTRAP_STATE" \\ '
            '"$BOOTSTRAP_DIR/bootstrap.tfplan"') in flat
    assert 'aws s3 cp "$BOOTSTRAP_STATE" \\ "s3://$(terraform -chdir=infra/bootstrap output -state="$BOOTSTRAP_STATE" -raw state_bucket)/bootstrap/terraform.tfstate"' in flat
    assert "Plan: 0 to add, 1 to change, 0 to destroy" in flat and "`aws_iam_role_policy.deploy` updated in place" in flat
    # every command that reads the bootstrap's state names it, and none keeps it in the checkout
    for match in re.finditer(r"terraform -chdir=[\w./]*bootstrap (\w+)([^\n]*)", RUNBOOK):
        assert '-state="$BOOTSTRAP_STATE"' in match.group(2) or match.group(1) == "init", match.group(0)
    assert "Keep `terraform.tfstate` here safe" not in RUNBOOK


def test_the_readme_and_the_reference_name_the_live_demos_target():
    readme = " ".join((ROOT / "README.md").read_text().split())
    reference = " ".join((ROOT / "doc" / "engineering-reference.md").read_text().split())
    for text in (readme, reference):
        assert "deploy.md#the-live-demo-one-host-on-the-aws-free-plan" in text
        assert "Free-plan EC2 host behind CloudFront" in text
    assert "the project has no AWS account yet" not in readme + reference  # the owner's account exists
