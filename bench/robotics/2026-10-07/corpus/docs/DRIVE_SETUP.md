# Sending the rest to Google Drive

With 2 TB, the tiering from [MIGRATION.md](MIGRATION.md) stops mattering. The
things that were dropped for size — 31 GB of intermediate checkpoints, and the
36 GB `lehome-data` project — fit with room to spare.

| | size | send? |
|---|---|---|
| `bhl-robustness-ladder/external/.../logs` — all 222 runs, all 10,584 checkpoints | 36 GB | **yes** |
| `lehome-data` (the other project) | 36 GB | **yes** |
| `Humanoid_Lite/logs` — Slurm console output | 2.2 GB | **yes** |
| venvs ×4 | 84 GB | **no** |
| apptainer container | 26 GB | **no** |

**~75 GB total, 3.7% of 2 TB.** The venvs and container are excluded not just
because they are reproducible but because they are *pinned to this cluster* —
Isaac Sim's plugins are matched to the driver here, and that pairing is the
subject of two findings in this repo. Restoring them elsewhere would give you
something that does not run.

## Step 1 — authorise rclone (needs a browser, so this part is yours)

`rclone` is already installed on the cluster at `~/bin/rclone` but has no config,
and its OAuth flow needs a browser the login node does not have. The headless
flow splits it in two.

**On your laptop** (install rclone from rclone.org first):

```bash
rclone authorize "drive"
```

A browser opens, you approve, and the terminal prints a token — a long JSON blob
starting `{"access_token":`. Copy the whole thing including the braces.

**On the cluster:**

```bash
~/bin/rclone config
```

Answer: `n` (new remote) → name it **`gdrive`** → storage `drive` →
client_id/secret blank → scope `1` (full access) → root_folder_id blank →
service_account_file blank → `n` to advanced config → **`n` to "Use auto config"**
— this is the step that matters, saying `n` is what makes it ask for the token —
→ paste the token → `n` to shared drive → `y` to confirm → `q` to quit.

Verify:

```bash
~/bin/rclone lsd gdrive:
```

## Step 2 — run the transfer

```bash
cd /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder
bash scripts/to_drive.sh
```

It is resumable — rerun it after a disconnect and it skips what already
transferred. Expect a few hours for 75 GB; Drive's cap is 750 GB/day, so the
size is not the constraint, the per-file API rate is.

Run it under `nohup` or `tmux` so a dropped SSH session does not kill it:

```bash
nohup bash scripts/to_drive.sh > /tmp/to_drive.log 2>&1 &
tail -f /tmp/to_drive.log
```

## Step 3 — verify before you trust it

The script runs `rclone check` at the end and prints a summary. Do not skip it.
A transfer that reports success while having silently skipped files is the
failure this whole exercise exists to prevent — and it has already happened once
in this project, when a job wrote two deploy configs that were byte-identical
copies of the wrong run.

## What is already safe without any of this

The [GitHub release](../../releases/tag/hpc-archive-2026-09-05) holds the final
checkpoint of every run, all raw event files, and every job log — 1.2 GB, done.
Drive is for the bulk that release assets are the wrong shape for.
