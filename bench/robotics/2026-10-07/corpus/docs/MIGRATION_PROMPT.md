# Reusable prompt: get a compute project off a cluster

Paste this into Claude Code, running in the project directory, when you are
about to lose access to a machine. Written after doing it once for real; the
parts that look fussy are the parts that went wrong.

---

I may lose access to this machine soon. Migrate everything that matters to
somewhere durable, and do it in this order.

**1. Measure before you move anything.** Report the size of every top-level
directory and, separately, the count and total size of: training checkpoints,
TensorBoard event files (or whatever holds the metrics), job logs, and the git
repo. Do not start copying until I can see that table. `du` on a network
filesystem is slow — background it and work on other steps meanwhile.

**2. Tier by what retraining cannot recreate, not by size.**

- *Reproducible* — virtualenvs, container images, package caches, compiled
  artifacts. Leave them. Name the exact command that rebuilds each, and check
  that command actually exists in the repo before claiming it.
- *Irreplaceable and small* — the numbers, and the configs that produced them.
  This goes in git, uncompressed and readable, and it is the tier that matters.
  If the machine vanished tonight, this is what lets me still write up the work.
- *Irreplaceable and large* — final model weights, raw metric files, job logs.
  Archive to release assets or object storage.
- *Expensive but redundant* — intermediate checkpoints, per-epoch dumps.
  Usually leave. Say how much you are dropping and what it would cost to
  recreate, so it is my decision and not a silent one.

**3. Extract the numbers to a format that outlives the tools.** Do not just
tar the event files. Flatten every scalar from every run into CSV — a manifest
with one row per run, and a long-form file with every (run, tag, step, value).
Pull *every* tag, not the ones current plots use: the point is to answer
questions nobody has asked yet. Copy each run's config files alongside. Commit
all of it.

**4. Check what auth I already have before designing around auth I don't.**
Look for `gh`, `rclone`, `aws`, `gsutil`, `az` and check which are *configured*,
not just installed. A tool that needs a browser OAuth flow cannot be used
non-interactively — if that is the only path, prepare the exact commands and
hand them to me rather than stalling. GitHub Releases take 2 GB per asset and
usually need no new setup if `gh auth status` is green.

**5. Verify every archive before uploading and after.** List the entries and
confirm the count matches what you intended to include. Use repo-relative paths
so a tarball unpacks back where the tooling expects. A corrupt or
wrongly-rooted archive is worse than no archive, because it will be trusted.

**6. Write down what you did not save.** A migration doc naming the tiers, the
restore commands, and the things deliberately left behind — including the
environment itself, which is never portable. Put the restore commands in a form
I can paste.

**Rules throughout.** Report sizes and counts, not adjectives. If something
fails, say so rather than moving on quietly — a half-finished migration that
reads as complete is the failure mode this whole exercise exists to prevent.
Prefer many small verifiable steps to one big copy. And do not delete anything
from the source machine; migration is copying, and I will do the deleting.
