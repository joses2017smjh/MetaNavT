# Deploy the public demo to AWS

**Status: code only, but for the bootstrap.** The owner's account exists (the Free plan) and the owner applied
`infra/bootstrap` on 2026-11-21; no environment is applied and no deploy has run. The Terraform is checked offline
(`terraform validate`, `terraform fmt`, tflint with the AWS rules) and by tests that read it; the deploy workflows are
checked by tests that run their scripts against a fake AWS CLI. [What has not been run](#what-has-not-been-run) and
[what only an apply can prove](#what-only-an-apply-can-prove) list what only the account can prove. Every step below
is for the owner, with the owner's credentials.

The demo is the API in public demo mode (`PUBLIC_DEMO=1`; [the reference](engineering-reference.md#public-demo-mode)):
search over the committed fixture and cited answers from Claude (production is configured for Claude Haiku 5.5 once
the owner stores a key: [Claude's cost](#claudes-cost)), under per-caller quotas and a daily budget. The same
image and settings run on one machine with `docker-compose.prod.yml` (the reference's Production rehearsal), which
is the cheaper place to try them first.

## The live demo: one host on the AWS Free plan

The live demo's target, the owner's decision of 2026-11-21: one on-demand EC2 instance sized to the AWS Free plan,
behind CloudFront, paid only from the plan's credits. Terraform: `infra/envs/single-host`, calling
`infra/modules/single-host`; the host's files: `deploy/single-host`; the deploy:
`.github/workflows/deploy-single-host.yml` with `scripts/deploy_single_host.py`. The repository variable
`DEPLOY_TARGET` picks what deploys: `single-host` this host, `ecs` the ECS stack, and unset nothing (each deploy
workflow then skips with a notice and stays green; so does the pull request plan). The ECS stack after this section
stays in the repository as the second, production-grade target; it is not applied for the live demo.

```text
viewer --HTTPS--> CloudFront (default certificate, PriceClass_100, AWS's security headers, no WAF)
                    |  /api/* and /health: never cached, every viewer header but Host
                    |  /_next/static/*: cached; every other path: not cached
                    '--HTTP, port 80, X-Origin-Verify--> Elastic IP --> one EC2 instance in one public subnet (no NAT)
the instance: Amazon Linux 2023; Docker Compose runs the api (the CPU model image, by digest), Postgres with
pgvector and Redis
```

- **The instance.** t3.small (2 GiB), on demand (no spot: the Free plan's offer is per instance type), with standard
  CPU credits (no surplus-credit charges), IMDSv2 only with a hop limit of 1 (a container gets no instance
  credentials), an encrypted 20 GB gp3 root volume, no key pair and no port 22: a shell is
  `aws ssm start-session --target <instance id>`. Its security group admits port 80 from CloudFront's origin-facing
  addresses only. cloud-init installs Docker (Amazon Linux's package) and the Compose plugin (pinned by version and
  sha256), makes a 2 GiB swap file, writes `/opt/metanavit` and `metanavit.service`, and runs the first deploy. A
  change to those files replaces the instance (cloud-init runs once per instance); the Elastic IP, and with it
  CloudFront's origin, stays. The api serves the web client's export at `/` itself, so the demo page, the API and
  `/docs` share one origin.
- **The secrets.** CloudFront adds `X-Origin-Verify` to every request, a random value Terraform keeps in Parameter
  Store; the app refuses a request without it (`ORIGIN_VERIFY_SECRET` in the reference's deploy contract), but
  `GET /health` from inside the container, its own health check. The Terraform state holds that value (CloudFront's
  origin carries it). An Anthropic key is optional: a SecureString the owner stores, never in Terraform; without it
  answers are off and search works. The host reads both at every start and deploy into a root-only file in memory;
  the Postgres password is made on the host and never leaves it.
- **No reranker on the live site: a labelled fallback.** `serve_reranker = false` sets `RERANKER_MODEL=none`, the
  app's own "not configured": the live site serves hybrid retrieval (BM25 plus bge-small, fused) without the
  bge-reranker-v2-m3 cross-encoder, and every response says `reranker_loaded: false`. The README's benchmark tables
  show what the reranker adds. Measured with the API's own loaders (`scripts/measure_api_memory.py`, recorded in
  `bench/results/api_memory_cpu.json`), on the HPC with the CUDA build of torch, CUDA hidden and 2 threads (an upper
  bound for the image's CPU wheel): the API process peaks at 842 MiB with the embedder, and at 2,719 MiB once the
  reranker has scored one query's 20 pairs, which took 89.92 s against a retrieval budget of 5 s
  (`RETRIEVE_TIMEOUT_SECONDS`). So the reranker fits neither 2 GiB beside Postgres nor the deadline; every type
  `instance_type` allows has 2 vCPUs, and none has been measured. The ECS stack's 1-vCPU task inherits the same
  latency: its production profile is not the live demo's.
- **A deploy has downtime.** On a 2 GiB host the index step and the API run one after the other, not side by side:
  each loads the embedder. `deploy.sh` pulls the new image while the API serves, stops the API, runs the migrations
  and the index step, then starts the API on the new image and waits for its health check; CloudFront answers with
  an error meanwhile. If a step fails, the previous image is started again. How long the API is down has not been
  measured.

### What it costs, and how long the credits last

Prices from the AWS Price List (us-west-2, On-Demand, Linux, shared tenancy): EC2 and EBS from
[the EC2 offer file](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-west-2/index.csv),
publication date 2026-11-21T00:48:16Z; public IPv4 from
[the VPC offer file](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonVPC/current/us-west-2/index.csv),
publication date 2026-11-19T20:55:45Z; t8i.small from the Price List query API (GetProducts), read 2026-11-21. A
month is 730 hours. These are list prices, which the credits pay.

| Item | Price (us-west-2) | Source |
|---|---|---|
| t3.small (x86_64, 2 GiB), the live demo's | $0.0208 per hour | EC2 offer file, published 2026-11-21 |
| t3.micro (x86_64, 1 GiB) | $0.0104 per hour | EC2 offer file, published 2026-11-21 |
| t8i.small (x86_64, 2 GiB) | $0.02496 per hour | Price List query API, read 2026-11-21 |
| c7i-flex.large (x86_64, 4 GiB) | $0.08479 per hour | EC2 offer file, published 2026-11-21 |
| m7i-flex.large (x86_64, 8 GiB) | $0.09576 per hour | EC2 offer file, published 2026-11-21 |
| t4g.small (ARM, 2 GiB: not usable, the image is x86_64 only) | $0.0168 per hour | EC2 offer file, published 2026-11-21 |
| t3.medium, t3.large (not Free Tier eligible) | $0.0416 and $0.0832 per hour | EC2 offer file, published 2026-11-21 |
| EBS gp3 | $0.08 per GB-month | EC2 offer file, published 2026-11-21 |
| Public IPv4 address, in use or idle | $0.005 per hour | VPC offer file, published 2026-11-19 |

| Host | Instance (× 730 h) | 20 GB gp3 | One public IPv4 (× 730 h) | Per month | Three months |
|---|---|---|---|---|---|
| t3.small, the live demo's | $15.18 | $1.60 | $3.65 | $20.43 | about $61 (the rows above) |
| c7i-flex.large | $61.90 | $1.60 | $3.65 | $67.15 | about $201 (the rows above) |

Three months of the c7i-flex.large would cost more than even the full $200 of credits (the rows above).

**Runway: remaining credits ÷ monthly cost, capped at six months** (the Free plan's length).
With the $100 sign-up credit alone, $100 ÷ $20.43 is 4.9 months of the t3.small (the rows above).
With all $200 of credits, $200 ÷ $20.43 would be 9.8 months, so the six-month cap (the rows above).
Usage charges come out of the same credits: CloudFront's always-free tier covers 1 TB out and 10 million requests a
month ([CloudFront pricing](https://aws.amazon.com/cloudfront/pricing/), read 2026-11-20); Parameter Store's standard
parameters, and Run Command and Session Manager on EC2 instances, are at no additional charge
([Systems Manager pricing](https://aws.amazon.com/systems-manager/pricing/), read 2026-11-21); the status-check alarm
is one of CloudWatch's ten free alarm metrics ([CloudWatch pricing](https://aws.amazon.com/cloudwatch/pricing/), read
2026-11-21). Claude's answers are paid apart, from Anthropic credit: the live demo sets the same model and daily
budget as the ECS production profile ([Claude's cost](#claudes-cost)).

### Free plan facts, with sources

Read 2026-11-21.

- The Free plan ends after six months or when the credits are used up, whichever comes first, and incurs no
  charges. When it ends the account closes: AWS keeps its content for 90 days, within which an upgrade to the Paid
  plan restores access ([Choosing a plan](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/free-tier-plans.html)).
- Credits: $100 at sign-up, then $20 for each completed activity, up to $100 more ([Earning additional credits](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/free-tier-plans-activities.html), read 2026-11-21).
  The six activities: launch an EC2 instance, use a model in the Bedrock playground, set up a cost budget, create a
  Lambda web app, create an RDS or Aurora database, create a DynamoDB table (the same page; `aws freetier
  list-account-activities` lists them for an account).
- Joining or creating an AWS Organization upgrades a Free plan account to the Paid plan, irreversibly, and its
  credits expire at once ([Enable IAM Identity Center](https://docs.aws.amazon.com/singlesignon/latest/userguide/enable-identity-center.html);
  [Choosing a plan](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/free-tier-plans.html) lists the other
  steps that upgrade it). Enabling IAM Identity Center on a standalone account offers to create an organization. So
  this runbook works with an IAM user with MFA and `aws login`, never IAM Identity Center.
- Free Tier eligible instance types, for accounts created on or after 2025-07-15: t3.micro, t3.small, t4g.micro,
  t4g.small, c7i-flex.large and m7i-flex.large ([EC2 Free Tier usage](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-free-tier-usage.html)).
  In us-west-2 the owner's account also lists t8i.micro, t8i.small, t9g.micro and t9g.small
  (`aws ec2 describe-instance-types --filters Name=free-tier-eligible,Values=true`, 2026-11-21). `instance_type`
  accepts the documented list's x86_64 types and t8i.small.
- A T3 instance in standard mode gets no launch credits
  ([standard mode concepts](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/burstable-performance-instances-standard-mode-concepts.html)):
  the first boot runs at the instance's baseline until it has earned CPU credits, so it may be slow.

### The steps (the owner, in order)

Done on 2026-11-21: the account (Free plan, no organization), the bootstrap (step 3) and the `production` environment
(step 4). The repository variables (step 4) are set after this code merges.

**1. An administrator that is not IAM Identity Center.** Turn on MFA for the root user and stop using it. In IAM,
create a user with Console access and MFA, attach `AdministratorAccess` (the applies create roles and policies) and
`SignInLocalDevelopmentAccess` (what `aws login` needs), and work as that user. Rotate the leaked secrets before
anything is deployed (the ECS stack's step 1, below).

**2. The CLI and Terraform, signed in with `aws login`.** AWS CLI 2.32 or later
([the CLI's sign-in guide](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sign-in.html)): with
`--remote` it prints a URL to open in a browser and asks for the code the browser shows. The CLI refreshes the
credentials every 15 minutes, for up to 12 hours. Terraform's AWS provider (v6.73.0, the version the lock files pin)
reads that profile directly: the bootstrap's plan and apply ran that way.

```bash
aws login --profile metanavit --remote
export AWS_PROFILE=metanavit
aws sts get-caller-identity          # the IAM user, not root
aws ec2 describe-instance-types --region us-west-2 --filters Name=free-tier-eligible,Values=true \
    --query "InstanceTypes[*].[InstanceType]" --output text | sort   # t3.small must be listed
```

A fallback, only if a long apply (a CloudFront create can take many minutes) fails on expired credentials: a profile
that asks the CLI for fresh ones each time (the guide's own example), then `export AWS_PROFILE=metanavit-process`.

```ini
# ~/.aws/config
[profile metanavit-process]
region = us-west-2
credential_process = aws configure export-credentials --profile metanavit --format process
```

**3. The bootstrap: its state outside every checkout, applied from a plan file, with a copy in the bucket.** Its
state is local and holds no secret, but losing it means importing every resource again, so it lives in a directory
that outlives any checkout or worktree; the plan file holds the budget's email, so it goes there too and is removed
after the apply:

```bash
export BOOTSTRAP_DIR="$HOME/.local/state/metanavit"      # durable, never inside a checkout or a worktree
export BOOTSTRAP_STATE="$BOOTSTRAP_DIR/bootstrap.tfstate"
mkdir -p "$BOOTSTRAP_DIR" && chmod 700 "$BOOTSTRAP_DIR"
terraform -chdir=infra/bootstrap init
terraform -chdir=infra/bootstrap plan -state="$BOOTSTRAP_STATE" -out="$BOOTSTRAP_DIR/bootstrap.tfplan" \
    -var 'budget_email=you@example.com' -var 'monthly_budget_usd=30'
terraform -chdir=infra/bootstrap apply -state="$BOOTSTRAP_STATE" -state-out="$BOOTSTRAP_STATE" \
    "$BOOTSTRAP_DIR/bootstrap.tfplan"
command rm -f "$BOOTSTRAP_DIR/bootstrap.tfplan"
aws s3 cp "$BOOTSTRAP_STATE" \
    "s3://$(terraform -chdir=infra/bootstrap output -state="$BOOTSTRAP_STATE" -raw state_bucket)/bootstrap/terraform.tfstate"
```

The last line keeps a copy in the state bucket, which is versioned: every later apply uploads again. Every command
that reads the bootstrap's outputs passes the same `-state="$BOOTSTRAP_STATE"`, as the last line does. The budget
counts gross cost, so its alerts fire while credits pay the bill. After this code merges, the same plan and apply
add the single host's permissions to the deploy role: the plan must say `Plan: 0 to add, 1 to change, 0 to destroy`,
the one change being `aws_iam_role_policy.deploy` updated in place (its policy document is updatable; only its name
or role would replace it, and neither changes).

**4. GitHub.** The `production` environment, created by hand before any deploy runs: required reviewers yourself
(with Prevent self-review off), deployment branches `main` only (the ECS stack's step 5, below, says why). Then the
repository variables, from the bootstrap's outputs, and the target:

```bash
terraform -chdir=infra/bootstrap output -state="$BOOTSTRAP_STATE" -json github_repository_variables \
  | python3 -c 'import json, sys; [print(k, v) for k, v in json.load(sys.stdin).items()]' \
  | while read -r name value; do gh variable set "$name" --body "$value"; done
gh variable set DEPLOY_TARGET --body single-host
```

**5. Apply the single host.** The image by digest: the one `:main-ml` points at now (the ECS stack's step 6, below,
has the command). The apply creates CloudFront first (it can take many minutes), then the instance, whose first boot
installs Docker and deploys that image: allow some minutes more before the URL answers.

```bash
cd infra/envs/single-host
terraform init -backend-config="bucket=$(terraform -chdir=../../bootstrap output -state="$BOOTSTRAP_STATE" -raw state_bucket)"
terraform apply -var 'image_digest=sha256:<the digest>' -var 'alert_email=you@example.com'
terraform output -json github_environment_variables \
  | python3 -c 'import json, sys; [print(k, v) for k, v in json.load(sys.stdin).items()]' \
  | while read -r name value; do gh variable set "$name" --env production --body "$value"; done
```

Confirm the alarm's SNS subscription from the email AWS sends. A later apply passes the last apply's digest back, so
it shows only the code's changes and keeps the instance: `terraform apply -var "image_digest=$(terraform output -raw
image_digest)"`. A new digest there, or a change to the host's files, replaces the instance, whose first boot rebuilds
the index from the fixture in the image.

**6. An Anthropic key, optional.** Without one the site searches and answers nothing. Create the key in a Console
workspace of its own with a monthly spend limit ([Claude's cost](#claudes-cost)), and store it without it reaching
the shell history:

```bash
umask 077
read -rs ANTHROPIC_KEY && printf '%s' "$ANTHROPIC_KEY" > anthropic.key && unset ANTHROPIC_KEY
aws ssm put-parameter --name "$(terraform output -raw anthropic_key_parameter)" --type SecureString \
    --value file://anthropic.key
command rm -f anthropic.key
```

The host reads it at its next start or deploy: run a deploy (step 7), or `sudo systemctl restart metanavit` in a
Session Manager shell. A new key later: the same command with `--overwrite`, then a deploy or a restart.

**7. Deploy, and every later deploy.** After a push to `main`, once CI has passed, `Deploy (single host)` waits for
your approval of its `production` job (by hand: `gh workflow run deploy-single-host.yml`; a rollback:
`gh workflow run deploy-single-host.yml -f image_digest=sha256:<previous digest>`). It finds the instance by its
tags, runs `/opt/metanavit/deploy.sh <digest>` there through Run Command and prints its output, then smokes the
public URL: the page, `GET /health`, `POST /api/retrieve/` with hits, and `POST /api/answer/` with an answer or a
reason. A rollback does not undo a schema migration: migrations only move forward.

### What only an apply can prove

Checked here: `terraform fmt`, `validate` and tflint with the AWS rules; the tests that read the Terraform, the Compose
file, cloud-init and the workflows; `deploy.sh` and `scripts/deploy_single_host.py` against fake `docker` and `aws`
commands; `docker compose config` of the host's file with the pinned Compose release (an opt-in test). Not seen
against AWS:

- the Free plan launching t3.small with these settings (standard CPU credits, IMDSv2 with one hop, the encrypted
  root volume), the newest Amazon Linux 2023 AMI, and the instance type offered in the zone the module picks;
- cloud-init on that AMI: the swap file, Amazon Linux's `docker` package with Compose v5.6.1, the sha256 check, the
  first deploy within the first boot (at the instance's baseline: no launch credits), and the gzipped user data
  under EC2's 16 KB limit (a test keeps it under 14 KiB, compressed by Python's gzip rather than Terraform's);
- the Elastic IP on the network interface before the instance exists, the interface kept when the instance is
  replaced, and CloudFront reaching the address's public DNS name over HTTP;
- the managed CloudFront policies by name, the security headers on the web client, CloudFront's 60-second origin
  timeout against a real answer, and the origin keep-alive against uvicorn's;
- the instance role reading both SecureStrings through the aws/ssm key, a container getting no instance credentials,
  the SSM agent registering, and Run Command from the deploy role under the `ssm:resourceTag` conditions;
- the API's memory on the CPU wheel under its 1,024 MiB limit, Postgres under 320 MiB, the swap file's use, and how
  long a deploy keeps the API down;
- the alarm's metric and the SNS email.

## What gets built

```text
viewer --HTTPS--> CloudFront (+ WAF, us-east-1)
                    |-- /api/*, /health --> ALB (HTTP; CloudFront's addresses and header only) --> ECS Fargate task
                    |                                                                (public subnet, public IP)
                    '-- everything else --> S3 bucket: the web client (private; origin access control)

ECS task --> RDS PostgreSQL 16 + pgvector, ElastiCache (Valkey, TLS)   private subnets, no route out
ECS task --> HTTPS out through its public IP: the image registry, the model downloads, the Anthropic API
```

| Part | Where | Notes |
|---|---|---|
| State bucket, GitHub OIDC provider, plan role, deploy role, monthly budget | `infra/bootstrap` | applied with administrator credentials (by the owner, 2026-11-21); its state stays local, outside every checkout, with a copy in the state bucket |
| One environment (network, database, cache, ECS, ALB, CloudFront, WAF, alarms) | `infra/modules/demo` | every choice is a variable (`infra/modules/demo/variables.tf`) |
| Production (on) | `infra/envs/production` | ElastiCache, WAF, RDS deletion protection |
| Staging (opt-in) | `infra/envs/staging` | Redis as a sidecar, no WAF, no deletion protection; applied only for a deploy test |
| Offline checks of the Terraform | `.github/workflows/infra.yml` | fmt, validate, tflint; a read-only plan of the `DEPLOY_TARGET` root once the repository variables are set |
| The deploy | `.github/workflows/deploy.yml`, `scripts/deploy_ecs.py` | when `DEPLOY_TARGET` is `ecs`: after CI passes on `main`, or by hand; production waits for the owner's approval |

Choices, each a variable: two availability zones; the ALB and the tasks in public subnets, the tasks with public
addresses for their outbound calls (no NAT gateway); one Fargate task, x86_64, 1 vCPU and 4 GB, running the CPU model
image `ghcr.io/joses2017smjh/metanavt` by digest; RDS `db.t4g.micro` with 20 GB of encrypted gp3, one day of
backups; ElastiCache `cache.t4g.micro` (Valkey, Redis-compatible, TLS in transit, so the app's `REDIS_URL` is
`rediss://`), or `redis_mode = "sidecar"`; CloudFront on its own `*.cloudfront.net` certificate (a domain and an ACM
certificate are optional, with a Route 53 zone); a WAF with AWS's common rules and two per-IP rate limits (100
requests per 5 minutes on `/api/answer/`, 1000 on everything else) and an optional CAPTCHA on `/api/answer/`.
The task's environment is the deploy contract of the reference (`TRUSTED_PROXY_HOPS=2`: CloudFront, then the ALB),
plus the database and the CPU models of `docker-compose.ml.yml`; the database password and the Anthropic key come
from Secrets Manager. Its one vCPU inherits the reranker's measured latency (the live demo's section above), so this
profile is not the live demo's.

The database password is generated during the apply and written to RDS and to its secret through write-only
arguments: it is in neither the Terraform state nor a plan. The state does hold the header value CloudFront sends to
the ALB (both resources carry it).

The web client in S3 is the image's Next.js export. Its home page reads `public_demo` from `/health` in the browser
and, from this API, shows the demo page: one question box, search, and cited answers with their reasons in plain
words ([the reference](engineering-reference.md#public-demo-mode)), not the chat interface, whose routes the public
API does not serve. One limit to know before the first deploy: CloudFront sends only `/api/*` and `/health` to the
API, so the OpenAPI pages (`/docs`, `/openapi.json`) are not reachable through it.

## Two ways to run it

- **Production only (the default).** One environment. The deploy workflow goes from CI straight to the production
  job, which waits for the owner's approval.
- **Staging, then production.** For testing a deploy: apply `infra/envs/staging`, add `"staging"` to the bootstrap's
  `deploy_environments` and apply the bootstrap, create the `staging` GitHub environment and its variables, and set
  the repository variable `DEPLOY_STAGING` to `true`. The workflow then deploys to staging and smokes it before
  production. A second always-on environment costs about as much again ([costs](#monthly-costs)), so destroy it after
  the test ([below](#destroy-staging-or-everything)) and unset `DEPLOY_STAGING`.

```bash
# staging mode, after the one-time steps below (bootstrap applied with "staging" in deploy_environments)
cd infra/envs/staging
terraform init -backend-config="bucket=$(terraform -chdir=../../bootstrap output -state="$BOOTSTRAP_STATE" -raw state_bucket)"
terraform apply -var 'image_digest=sha256:<a digest of the -ml image>'
terraform output -json github_environment_variables \
  | python3 -c 'import json, sys; [print(k, v) for k, v in json.load(sys.stdin).items()]' \
  | while read -r name value; do gh variable set "$name" --env staging --body "$value"; done
gh variable set DEPLOY_STAGING --body true
```

## One-time steps (the owner)

**1. Rotate the leaked secrets first.** Two `.env` files with secrets are in this repository's public history
(GitHub's secret scanning alerts #1 to #5). Revoke and replace every key they hold, at each provider, and close the
alerts, before anything is deployed. Do not reuse any of them below.

**2. The AWS account.** Create the account, turn on MFA for the root user and stop using it, and do the applies below
as an IAM user with MFA signed in with `aws login`, as the live demo's steps 1 and 2 above say: not IAM Identity
Center, whose setup on a standalone account creates an AWS Organization, which moves a Free plan account to the Paid
plan and ends its credits ([Free plan facts](#free-plan-facts-with-sources)).
New accounts get up to $200 in Free Tier credits ([AWS Free Tier FAQs](https://aws.amazon.com/free/free-tier-faqs/), read 2026-11-20),
and the free plan lasts six months or until they run out; the budget below counts gross cost, so its alerts still
fire while credits pay the bill. Choose the address the budget alerts go to.

**3. Tools.** Terraform 1.11 or later (CI uses 1.17.2), the AWS CLI 2.32 or later signed in with `aws login` (the
live demo's step 2), and `gh`.

**4. Apply the bootstrap.** It creates the state bucket, GitHub's OIDC provider, the two roles and the budget
(`monthly_budget_usd`, 150 USD by default in `infra/bootstrap/variables.tf`, alerts at 50%, 80% and 100% of actual
spend; the live demo's step 3 passes 30). Pass the email on the command line: it is never committed. Its state
stays outside the checkout, applied from a plan file with a copy in the state bucket: the commands are the live
demo's step 3 above (staging mode adds `-var 'deploy_environments=["production","staging"]'` to the plan).

If the account already has GitHub's OIDC provider, import it first:
`terraform -chdir=infra/bootstrap import -state="$BOOTSTRAP_STATE" aws_iam_openid_connect_provider.github arn:aws:iam::<account id>:oidc-provider/token.actions.githubusercontent.com`.

**5. GitHub settings.** The repository variables the workflows read, from the bootstrap's outputs:

```bash
terraform -chdir=infra/bootstrap output -state="$BOOTSTRAP_STATE" -json github_repository_variables \
  | python3 -c 'import json, sys; [print(k, v) for k, v in json.load(sys.stdin).items()]' \
  | while read -r name value; do gh variable set "$name" --body "$value"; done
gh variable set DEPLOY_TARGET --body ecs      # this stack; the live demo's is single-host
```

Then create the `production` environment yourself (Settings, Environments, New environment), before any deploy runs:
**Required reviewers**: yourself, with **Prevent self-review** left off (otherwise nobody can approve your own
deploys); **Deployment branches and tags**: selected branches, `main` only. The deploy role trusts any job that names
this environment, and GitHub creates an environment it has not seen without any protection, so this step is what
makes the approval and the branch rule real. In staging mode, create `staging` the same way (reviewers optional,
`main` only). Optionally add a repository secret `ALERT_EMAIL` for the plan job.

**6. Apply production.** The image is taken by digest. Until the first deploy, use the digest `:main-ml` points at
now (CI's image-ml job publishes it from `main`):

```bash
token=$(curl -s "https://ghcr.io/token?scope=repository:joses2017smjh/metanavt:pull" \
        | python3 -c 'import json, sys; print(json.load(sys.stdin)["token"])')
curl -sI -H "Authorization: Bearer $token" \
     -H 'Accept: application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json' \
     https://ghcr.io/v2/joses2017smjh/metanavt/manifests/main-ml | grep -i docker-content-digest

cd infra/envs/production
terraform init -backend-config="bucket=$(terraform -chdir=../../bootstrap output -state="$BOOTSTRAP_STATE" -raw state_bucket)"
terraform apply -var 'image_digest=sha256:<the digest above>' -var 'alert_email=you@example.com'
```

The apply starts the service; its first task waits (the API answers 503 and retries) until the first deploy has run
the migrations and the index step, so run the first deploy ([below](#the-first-deploy-and-every-later-one)) right
after steps 7 and 8. Confirm the alarm topic's subscription from the email AWS sends.

**7. The environment's variables** for the deploy workflow, from the root's outputs:

```bash
terraform output -json github_environment_variables \
  | python3 -c 'import json, sys; [print(k, v) for k, v in json.load(sys.stdin).items()]' \
  | while read -r name value; do gh variable set "$name" --env production --body "$value"; done
```

**8. The Anthropic key.** In the Anthropic Console, create a workspace for this demo only (not the default
workspace, whose spend limit may not stop all of its usage), set its monthly spend limit
($24 for the production profile, as `infra/envs/production/main.tf` notes: [Claude's cost](#claudes-cost)) and create
the demo's key in that workspace: the limit is the last cap if everything else fails. Console spend limits are set
for the organization or for a workspace; the docs describe none for a single key
([Rate limits](https://platform.claude.com/docs/en/api/rate-limits#setting-lower-limits-for-workspaces), read
2026-11-21). Store the key in the secret the apply created, without it reaching the shell history:

```bash
umask 077
read -rs ANTHROPIC_KEY && printf '%s' "$ANTHROPIC_KEY" > anthropic.key && unset ANTHROPIC_KEY
aws secretsmanager put-secret-value --secret-id "$(terraform output -raw anthropic_secret_name)" \
    --secret-string file://anthropic.key
command rm -f anthropic.key
```

Then set `anthropic_key_stored = true` in `infra/envs/production/main.tf` in a pull request, merge it, and apply
again. Until then the task runs without the key: answers off, retrieval on. (A task that references a secret with no
stored value does not start, which is why the reference waits for this flag.)

## The first deploy, and every later one

With the repository variable `DEPLOY_TARGET` set to `ecs`: after a push to `main`, when CI has passed (and its
image-ml job has published `:sha-<commit>-ml`), the Deploy workflow starts by itself. To start it by hand: Actions, Deploy, Run workflow (`gh workflow run deploy.yml`). It:

1. resolves the digest of `ghcr.io/joses2017smjh/metanavt:sha-<commit>-ml` and checks the image's revision label;
2. takes the web client's export out of that image;
3. (staging mode) deploys to staging and smokes it;
4. waits for your approval of the `production` job, then assumes the deploy role through OIDC and runs
   `scripts/deploy_ecs.py`: a new revision of the migrate/index task with the digest, run once until it exits 0 (the
   migrations, then the index step); a new revision of the API with the digest; the service moved to it and waited
   on until stable (the circuit breaker rolls back a failing one, and the deploy fails); the web client uploaded to
   S3 and CloudFront's cache invalidated; then the smoke against the public URL: the web page, `GET /health`,
   `POST /api/retrieve/` with hits, and `POST /api/answer/` with either an answer or a reason (no assertion that
   Claude answered: no key, no budget left and no Redis are reasons too).

A change to the task definition in Terraform (a variable, the key flag) takes effect at the next deploy: Terraform
registers the new revision, and the service stays on the deployed one until a deploy starts from the newest. A later
`terraform apply` takes an image digest too. Pass the last apply's, which the state keeps, so the plan shows only your
change; a newer one also works, and only registers new revisions of both task families (the service stays where the
last deploy put it). The pull request plan job does the same.

```bash
terraform apply -var "image_digest=$(terraform output -raw image_digest)"
```

## Rollback

Run the workflow by hand with the previous digest (its run summary says "Deploying ...@sha256:..."; the API's
task definition revisions list every digest deployed). The digest the service runs now:

```bash
aws ecs describe-task-definition --task-definition "$(aws ecs describe-services --cluster metanavit-production \
    --services metanavit-production-api --query 'services[0].taskDefinition' --output text)" \
    --query "taskDefinition.containerDefinitions[?name=='api'].image | [0]" --output text
```

Then:

```bash
gh workflow run deploy.yml -f image_digest=sha256:<previous digest>
```

The same steps run with that image. A rollback does not undo a schema migration: migrations only move forward.

## Monthly costs

Unit prices: AWS's us-west-2 prices from the AWS Price List API offer files, read 2026-11-20 (offer files published
2026-11-11 to 2026-11-20); each row links the service's public pricing page. A month is 730 hours. These are list
prices before Free Tier credits; check the pages again before relying on them.

| Resource (production, always on) | Unit price (us-west-2) | Formula | Per month | Source, read |
|---|---|---|---|---|
| Fargate task, vCPU (x86) | $0.04048 per vCPU-hour | 1 vCPU × 730 h | $29.55 | [Fargate pricing](https://aws.amazon.com/fargate/pricing/), 2026-11-20 |
| Fargate task, memory (x86) | $0.004445 per GB-hour | 4 GB × 730 h | $12.98 | [Fargate pricing](https://aws.amazon.com/fargate/pricing/), 2026-11-20 |
| Application Load Balancer | $0.0225 per hour | 730 h | $16.43 | [Elastic Load Balancing pricing](https://aws.amazon.com/elasticloadbalancing/pricing/), 2026-11-20 |
| Public IPv4 addresses | $0.005 per in-use address-hour | 3 addresses (the ALB's two, the task's one) × 730 h | $10.95 | [VPC pricing](https://aws.amazon.com/vpc/pricing/), 2026-11-20 |
| RDS PostgreSQL `db.t4g.micro`, Single-AZ | $0.016 per hour | 730 h | $11.68 | [RDS for PostgreSQL pricing](https://aws.amazon.com/rds/postgresql/pricing/), 2026-11-20 |
| RDS gp3 storage | $0.115 per GB-month | 20 GB | $2.30 | [RDS for PostgreSQL pricing](https://aws.amazon.com/rds/postgresql/pricing/), 2026-11-20 |
| ElastiCache `cache.t4g.micro`, Valkey | $0.0128 per hour | 730 h | $9.34 | [ElastiCache pricing](https://aws.amazon.com/elasticache/pricing/), 2026-11-20 |
| WAF | $5 per web ACL-month, $1 per rule-month | 1 web ACL + 3 rules | $8.00 | [WAF pricing](https://aws.amazon.com/waf/pricing/), 2026-11-20 |
| Secrets Manager | $0.40 per secret-month | 2 secrets | $0.80 | [Secrets Manager pricing](https://aws.amazon.com/secrets-manager/pricing/), 2026-11-20 |
| **Subtotal** | | | **$102.03** | the rows above |

Usage-based charges come on top, and this runbook puts no number on them:

| Charge | Formula | Source |
|---|---|---|
| ALB capacity units | $0.008 per LCU-hour × the LCUs used (one LCU on average all month would add $5.84) | [Elastic Load Balancing pricing](https://aws.amazon.com/elasticloadbalancing/pricing/), read 2026-11-20 |
| WAF requests | $0.60 per million requests | [WAF pricing](https://aws.amazon.com/waf/pricing/), read 2026-11-20 |
| CloudFront | free up to 1 TB of data out and 10 million requests a month (the always-free tier), then per GB and per request | [CloudFront pricing](https://aws.amazon.com/cloudfront/pricing/), read 2026-11-20 |
| CloudWatch (alarms, logs ingested and stored), S3 (the web client, the state), data transfer out | per alarm, per GB and per request | check [CloudWatch pricing](https://aws.amazon.com/cloudwatch/pricing/), [S3 pricing](https://aws.amazon.com/s3/pricing/) |
| Route 53 hosted zone and queries (only with a custom domain) | per zone-month and per query | check [Route 53 pricing](https://aws.amazon.com/route53/pricing/) |
| WAF CAPTCHA (only with `waf_answer_captcha`) | per challenge attempt | check [WAF pricing](https://aws.amazon.com/waf/pricing/) |
| Claude (Anthropic API) | at most `ANSWER_DAILY_BUDGET_USD` a day as the app estimates it: production sets $0.75 (the module's default and the contract's is $2.00), each call's cost estimated from the price table dated in `app/answer/claude.py`; the Console spend limit caps it too ([Claude's cost](#claudes-cost), below) | `infra/envs/production/main.tf`; the default in `app/api/limits.py` and the deploy contract; the prices in `app/answer/claude.py` (`PRICES_AS_OF`) |

Staging, while it exists, costs the production rows without ElastiCache and WAF: $84.68 a month at the same prices.
That is about $0.12 an hour at the same prices, so a deploy test of a few hours costs under a dollar before usage charges.

What each variable saves (production, at the prices above):

| Variable | Saving per month | Cost of the saving |
|---|---|---|
| `cpu_architecture = "ARM64"` | $8.50 (Fargate ARM: $0.03238 per vCPU-hour and $0.00356 per GB-hour, [Fargate pricing](https://aws.amazon.com/fargate/pricing/), read 2026-11-20) | needs an arm64 build of the image; CI publishes amd64 only |
| `redis_mode = "sidecar"` | $9.34 (the ElastiCache row above) | one task only, and Redis starts empty at every task start: the day's budget counter and the quotas reset at each deploy or restart |
| `enable_waf = false` | $8.00 (the WAF row above, plus its per-request charge) | no managed rules and no per-IP rate limit in front; the app's own quotas and daily budget remain |
| `desired_count = 0` (API paused, database kept) | $46.18 (the two Fargate rows above, $42.53, and one address, $3.65) | the demo is down; the ALB, the database, the cache and the WAF still cost |
| `task_memory = 2048` with `environment_overrides = { RERANKER_MODEL = "none" }` | $6.49 (2 GB less × $0.004445 × 730 h, the Fargate memory row above) | no cross-encoder: the fused order is served (and the smoke's reranker check must go) |
| `terraform destroy` | everything but the bootstrap | see below |

A NAT gateway, which the public subnets avoid, would add its own hourly and per-GB charges
([VPC pricing](https://aws.amazon.com/vpc/pricing/)); this runbook has no read of that price.

### Claude's cost

Production is configured to answer with Claude Haiku 5.5: `answer_model = "claude-haiku-5-5"` and
`answer_daily_budget_usd = 0.75` in `infra/envs/production/main.tf` (the owner's choice, 2026-11-21), and
`deploy/prod.env.example` sets the same two values for the Compose rehearsal. Claude is called only once the owner
has stored a key (step 8); until then answers are off (the route's `answers off`) and search works, so nothing below
is spent. The module's defaults stay the deploy contract's: `ANSWER_MODEL=claude-opus-5-5` and
`ANSWER_DAILY_BUDGET_USD=2.00`.

| Claude Haiku 5.5 (`claude-haiku-5-5`) | Price | Source |
|---|---|---|
| Input tokens, prompts up to 100K tokens | $0.10 per million input tokens | Anthropic's list price in the claude-api skill's price table, cached 2026-11-03: `PRICES` in `app/answer/claude.py` (`PRICES_AS_OF`) |
| Output tokens (the model's thinking included), prompts up to 100K tokens | $0.50 per million output tokens | the same table, `PRICES` in `app/answer/claude.py` |
| Input and output tokens, prompts over 100K tokens | $0.50 per million input tokens and $2.50 per million output tokens | the same table, `HAIKU_LONG_PROMPT` in `app/answer/claude.py` |

`app/answer/claude.py` estimates every call's cost with this table. An answer's prompt is the system prompt, the
question (at most 500 characters) and at most ten retrieved excerpts of at most 800 characters each (the demo page
asks for six): far below 100K tokens.

```text
cost of one answer (USD) = input tokens × 0.10 / 1,000,000 + output tokens × 0.50 / 1,000,000
answers in a day         ≈ ANSWER_DAILY_BUDGET_USD / cost of one answer
```

For example, an answer of 3,000 input and 600 output tokens (assumed counts: no answer has been measured in this
repository) costs 3,000 × $0.10 / 1,000,000 + 600 × $0.50 / 1,000,000 = $0.0003 + $0.0003 = $0.0006 at the prices above,
so production's daily budget (`infra/envs/production/main.tf`) pays for about $0.75 / $0.0006 = 1,250 such answers a
day. Longer replies mean fewer; a question repeated while its answer is cached (`ANSWER_CACHE_TTL_SECONDS`) costs
nothing.

The caps, from the first to the last:

1. Per caller: `ANSWER_LIMIT_PER_MINUTE` and `ANSWER_LIMIT_PER_DAY` (5 and 50 by default), and in front of them the
   WAF's rate limit on `/api/answer/` (100 requests per IP in 5 minutes).
2. The daily budget, per UTC day (`app/api/limits.py`). Before each call the app reserves that call's worst case (all
   of `ANSWER_MAX_TOKENS` as output, the input at a conservative tokens-per-character ratio) and refuses the call when
   the day's total would pass the budget; afterwards it settles the reservation to the reply's estimated cost. So a
   day's estimated total stays within $0.75 (`infra/envs/production/main.tf`), which is at most $23.25 in a 31-day
   month and $69.00 over 92 days (three months; the owner allows about $70, `infra/envs/production/main.tf`). It is
   an estimate from the price table, not Anthropic's bill.
3. The last cap: the $24 monthly spend limit (the owner's figure, noted in `infra/envs/production/main.tf`) on the
   demo's workspace in the Anthropic Console (step 8). A month of the daily budget, a 31-day one too, stays under it,
   so it is a backstop for whatever the app's estimate gets wrong. Once a limit you set is reached, the Messages API
   answers each call HTTP 400 `invalid_request_error` ("You have reached your specified workspace API usage limits")
   until access resumes ([Rate limits](https://platform.claude.com/docs/en/api/rate-limits#setting-your-own-spend-limit),
   read 2026-11-21). `app/answer/claude.py` reports that 400, and the one that says the credit balance is too low, as
   the reason `answer spend limit reached`: `/api/answer/` returns the hits with it and gives the call's budget
   reservation back (nothing was billed), and search keeps working. Only the usage tier's own monthly cap answers 429
   (`enforced_spend_limit_reached`), which the route reports as `upstream rate limited`.

## Rotating keys

- **The Anthropic key.** Create a new key in the demo's Console workspace (its spend limit stays), store it as in
  step 8 (it becomes the secret's current value), and deploy again
  (`gh workflow run deploy.yml -f image_digest=<the digest running now>`): a task reads its secrets when it starts.
  Then revoke the old key in the Console.
- **The database password.** Set `db_password_version = 2` (then 3, and so on) in the environment's module block and
  apply: a new random password is written to RDS and to its secret in the same apply. Deploy again straight after:
  running tasks keep their open connections, but a new connection needs the new password.
- **CloudFront's header to the ALB.** `terraform apply -replace=module.demo.random_password.origin_header`. CloudFront
  and the listener rule change in the same apply; requests can get a 403 while CloudFront's change propagates.
- **AWS credentials for GitHub.** There are none to rotate: the workflows get short-lived credentials through OIDC.

## Raising or cutting the budget

- The account's monthly budget: `monthly_budget_usd` (and `budget_alert_percentages`) in `infra/bootstrap`, then apply
  the bootstrap.
- The app's daily answer budget: `answer_daily_budget_usd` in the environment's module block, apply, then deploy. Also
  `answer_limit_per_minute`, `answer_limit_per_day` and `answer_model`: production sets `claude-haiku-5-5`, the
  cheapest of the three per token; `claude-sonnet-5-5` costs 20 times as much per token and `claude-opus-5-5` 40 times
  (the price table in `app/answer/claude.py`), so raise the budget with either.
- The Anthropic Console's spend limit on the demo's workspace: in the Console.

## Destroy staging or everything

Staging (the image digest is required by the root, even to destroy):

```bash
terraform -chdir=infra/envs/staging destroy -var 'image_digest=sha256:<any digest>'
gh variable delete DEPLOY_STAGING
```

Then drop `"staging"` from the bootstrap's `deploy_environments` and apply the bootstrap.

Production: set `deletion_protection = false` in `infra/envs/production/main.tf`, apply, then destroy. RDS keeps a
final snapshot, `metanavit-production-final`, which is billed as snapshot storage until you delete it
(`aws rds delete-db-snapshot --db-snapshot-identifier metanavit-production-final`); a second destroy needs that name
free. The secrets wait seven days before AWS deletes them; to re-create production sooner, restore them
(`aws secretsmanager restore-secret`) or delete them for good (`--force-delete-without-recovery`).

The bootstrap stays: the state bucket refuses to be destroyed (`prevent_destroy`). To remove it too, empty the bucket
of every version, remove `prevent_destroy`, and destroy the bootstrap; then delete the repository variables and the
GitHub environments.

## What has not been run

Of the ECS stack (the single host's list is [above](#what-only-an-apply-can-prove)): only the bootstrap has been
applied, by the owner on 2026-11-21, from the code before the single host's deploy-role permissions. None of this has
been seen working against AWS:

- `terraform plan` and `apply` of every root but the bootstrap: the data sources (availability zones, CloudFront's prefix list and
  managed policies), service quotas, the IAM policies' acceptance (the ECS condition keys, the deploy role's
  resource patterns), the WAF on CloudFront, the ACM path with a domain, Valkey's default engine version;
- the GitHub OIDC trust (the token's subject for an environment job), the deploy workflow end to end, the plan job;
- the app on RDS with `PGSSLMODE=require`, on ElastiCache over `rediss://` (redis-py checks the certificate against
  the image's CA bundle), the model downloads at every task start within the 900-second grace period, the fp32
  reranker within 4 GB, and real answer times against CloudFront's 60-second origin timeout (the task's settings
  bound the worst case below it: a 5-second retrieval budget plus a 25-second Claude timeout with one retry, which
  tests/engine/test_infra_terraform.py checks; a slower answer ends as the route's "upstream timeout", not a 504);
- the alarms' metric names and dimensions, and the EventBridge pattern that tells an unasked task stop from a
  deployment's;
- Claude Haiku 5.5 on the answer route: its token counts and cost per answer (the example under
  [Claude's cost](#claudes-cost) is assumed, not measured), and the Console's workspace spend limit stopping calls;
- the demo page against the deployed API: its `/health` check through CloudFront, real answers and refusals, and a
  browser's network panel showing no request to another host (the web tests mock the API and check that the page's
  own requests stay on its origin; nothing in this repository runs the export in a browser).

## Check the code locally (no AWS)

```bash
terraform fmt -check -recursive infra
for root in infra/bootstrap infra/envs/production infra/envs/single-host infra/envs/staging; do
  terraform -chdir=$root init -backend=false && terraform -chdir=$root validate
done
cd infra && tflint --init --config "$PWD/.tflint.hcl" && tflint --recursive --config "$PWD/.tflint.hcl"
python -m pytest tests/engine/test_infra_terraform.py tests/engine/test_infra_workflow.py tests/engine/test_deploy_workflow.py \
    tests/engine/test_single_host.py tests/engine/test_deploy_single_host.py tests/api/test_origin_verify.py
COMPOSE_BIN="docker compose" python -m pytest tests/engine/test_single_host.py   # also docker compose config
TERRAFORM_BIN=$(command -v terraform) python -m pytest tests/engine/test_infra_terraform.py   # also init + validate
```

`tflint --recursive` needs `--config` with an absolute path: without it the modules below `infra/` are linted without
the AWS rules.
