# Deploy the public demo to AWS

**Status: code only.** Nothing in this runbook has been applied: the project has no AWS account yet, and no
`terraform plan` or `apply` has run against AWS. The Terraform is checked offline (`terraform validate`, `terraform
fmt`, tflint with the AWS rules) and by tests that read it; the deploy workflow is checked by tests that run its
script against a fake AWS CLI. [What has not been run](#what-has-not-been-run) lists what only a real account can
prove. Every step below is for the owner, with the owner's credentials.

The demo is the API in public demo mode (`PUBLIC_DEMO=1`; [the reference](engineering-reference.md#public-demo-mode)):
search over the committed fixture and cited answers from Claude (production is configured for Claude Haiku 5.5 once
the owner stores a key: [Claude's cost](#claudes-cost)), under per-caller quotas and a daily budget. The same
image and settings run on one machine with `docker-compose.prod.yml` (the reference's Production rehearsal), which
is the cheaper place to try them first.

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
| State bucket, GitHub OIDC provider, plan role, deploy role, monthly budget | `infra/bootstrap` | applied once, with administrator credentials; its own state stays local with the owner |
| One environment (network, database, cache, ECS, ALB, CloudFront, WAF, alarms) | `infra/modules/demo` | every choice is a variable (`infra/modules/demo/variables.tf`) |
| Production (on) | `infra/envs/production` | ElastiCache, WAF, RDS deletion protection |
| Staging (opt-in) | `infra/envs/staging` | Redis as a sidecar, no WAF, no deletion protection; applied only for a deploy test |
| Offline checks of the Terraform | `.github/workflows/infra.yml` | fmt, validate, tflint; a read-only plan once the account exists |
| The deploy | `.github/workflows/deploy.yml`, `scripts/deploy_ecs.py` | after CI passes on `main`, or by hand; production waits for the owner's approval |

Choices, each a variable: two availability zones; the ALB and the tasks in public subnets, the tasks with public
addresses for their outbound calls (no NAT gateway); one Fargate task, x86_64, 1 vCPU and 4 GB, running the CPU model
image `ghcr.io/joses2017smjh/metanavt` by digest; RDS `db.t4g.micro` with 20 GB of encrypted gp3, one day of
backups; ElastiCache `cache.t4g.micro` (Valkey, Redis-compatible, TLS in transit, so the app's `REDIS_URL` is
`rediss://`), or `redis_mode = "sidecar"`; CloudFront on its own `*.cloudfront.net` certificate (a domain and an ACM
certificate are optional, with a Route 53 zone); a WAF with AWS's common rules and two per-IP rate limits (100
requests per 5 minutes on `/api/answer/`, 1000 on everything else) and an optional CAPTCHA on `/api/answer/`.
The task's environment is the deploy contract of the reference (`TRUSTED_PROXY_HOPS=2`: CloudFront, then the ALB),
plus the database and the CPU models of `docker-compose.ml.yml`; the database password and the Anthropic key come
from Secrets Manager.

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
terraform init -backend-config="bucket=$(terraform -chdir=../../bootstrap output -raw state_bucket)"
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

**2. The AWS account.** Create the account, turn on MFA for the root user and stop using it; create an administrator
identity (IAM Identity Center) for the applies below.
New accounts get up to $200 in Free Tier credits ([AWS Free Tier FAQs](https://aws.amazon.com/free/free-tier-faqs/), read 2026-11-20),
and the free plan lasts six months or until they run out; the budget below counts gross cost, so its alerts still
fire while credits pay the bill. Choose the address the budget alerts go to.

**3. Tools.** Terraform 1.11 or later (CI uses 1.17.2), the AWS CLI v2 signed in as the administrator, and `gh`.

**4. Apply the bootstrap.** It creates the state bucket, GitHub's OIDC provider, the two roles and the budget
(`monthly_budget_usd`, 150 USD by default in `infra/bootstrap/variables.tf`, alerts at 50%, 80% and 100% of actual
spend). Pass the email on the command line: it is never committed.

```bash
cd infra/bootstrap
terraform init
terraform apply -var 'budget_email=you@example.com'
# staging mode: -var 'deploy_environments=["production","staging"]'
```

If the account already has GitHub's OIDC provider, import it first:
`terraform import aws_iam_openid_connect_provider.github arn:aws:iam::<account id>:oidc-provider/token.actions.githubusercontent.com`.
Keep `terraform.tfstate` here safe (it is ignored by git): it holds no secret, but losing it means importing again.

**5. GitHub settings.** The repository variables the workflows read, from the bootstrap's outputs:

```bash
terraform -chdir=infra/bootstrap output -json github_repository_variables \
  | python3 -c 'import json, sys; [print(k, v) for k, v in json.load(sys.stdin).items()]' \
  | while read -r name value; do gh variable set "$name" --body "$value"; done
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
terraform init -backend-config="bucket=$(terraform -chdir=../../bootstrap output -raw state_bucket)"
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

After a push to `main`, when CI has passed (and its image-ml job has published `:sha-<commit>-ml`), the Deploy
workflow starts by itself. To start it by hand: Actions, Deploy, Run workflow (`gh workflow run deploy.yml`). It:

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
   read 2026-11-21). `app/answer/claude.py` treats a 400 as a fault, not as a reason, so `/api/answer/` then answers
   500, without the hits, while search keeps working. Only the usage tier's own monthly cap answers 429
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

No AWS account exists for this project, so none of this has been seen working against AWS:

- `terraform plan` and `apply` of every root: the data sources (availability zones, CloudFront's prefix list and
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
  [Claude's cost](#claudes-cost) is assumed, not measured), and the Console's workspace spend limit stopping calls
  (and the answer route's 500 then);
- the demo page against the deployed API: its `/health` check through CloudFront, real answers and refusals, and a
  browser's network panel showing no request to another host (the web tests mock the API and check that the page's
  own requests stay on its origin; nothing in this repository runs the export in a browser).

## Check the code locally (no AWS)

```bash
terraform fmt -check -recursive infra
for root in infra/bootstrap infra/envs/production infra/envs/staging; do
  terraform -chdir=$root init -backend=false && terraform -chdir=$root validate
done
cd infra && tflint --init --config "$PWD/.tflint.hcl" && tflint --recursive --config "$PWD/.tflint.hcl"
python -m pytest tests/engine/test_infra_terraform.py tests/engine/test_infra_workflow.py tests/engine/test_deploy_workflow.py
TERRAFORM_BIN=$(command -v terraform) python -m pytest tests/engine/test_infra_terraform.py   # also init + validate
```

`tflint --recursive` needs `--config` with an absolute path: without it the modules below `infra/` are linted without
the AWS rules.
