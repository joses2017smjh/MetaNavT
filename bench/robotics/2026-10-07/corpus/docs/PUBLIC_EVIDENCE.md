# Public evidence and local execution records

The September 20 publication preserves evaluation metrics, seeds, task
definitions, pass/fail flags, incomplete-run flags, negative controls, training
steps, garment splits and selected recordings. It is a publication copy of
the completed audit; it does not modify the original experiment directories,
shared environments, checkpoints or scheduler jobs.

## What is omitted or rewritten

- The new raw `submissions.jsonl` receipts are not published. The existing
  historical job ledger remains as previously published; the new campaign's
  raw additions are retained locally.
- The public campaign summary omits scheduler job identifiers, node names,
  account information and submission counts. Narrative results describe
  experiments by purpose, policy, class, stage and seed instead.
- Absolute personal workspace locations in newly published JSON become
  logical `artifact://<project>/<relative-location>` identifiers. These are
  provenance labels, not downloadable URLs or working filesystem paths.
  Checkpoint/model weights and full garment datasets remain external.
- Capture manifests retain garment identities, episode IDs, success labels,
  frame counts, sizes and train/validation membership. Filesystem modification
  times are omitted. `original_manifest_sha256` identifies the original local
  training manifest before these publication-only edits, not the public JSON.
- GIF sidecars' `evidence_sha256` values identify the actual published JSON.
  `original_evidence_sha256` retains the hash of the unmodified local evidence.
  Video/GIF bytes, their hashes and measured episode outcomes are unchanged.

Per-step traces are retained. Missing or incomplete evaluations have not been
converted into zero-success completed runs, and failed policy demonstrations
have not been relabelled as successes. The known-route navigation result and
the separate frozen-gait MuJoCo demonstrations retain their protocol limits.

## Inspecting and regenerating the public summary

```bash
python scripts/summarize_weekend.py
```

This reads published evaluation JSON without requiring local receipts or
Slurm access. It refreshes `SUMMARY.md` and `summary.json` in the selected
directory. `--slurm` is only for an actual cluster checkout that has its local
`submissions.jsonl`; it fails clearly when those receipts are unavailable.

New batch wrappers derive their repository location from `REPO`, the submit
directory, or the current directory. Supply your account and partition using
`sbatch` flags, and set `BHL_PYTHON` for CPU jobs if `python` is not the desired
environment. The existing container/environment bootstrap still requires local
configuration and external installations; this change is not a clean-machine
installation test. Documentation uses relative sibling-project examples that
must match your own data layout.

The repository still has no root license file. Existing upstream code,
models and assets retain their own terms; publication does not grant a new
license for them.
