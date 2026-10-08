# Solutions investigation evidence (2026-09-30 to 2026-10-01)

An archived copy of `/nfs/hpc/share/sanchej7/Humanoid_Lite/solutions-20260930/` (outside the repo), the
evidence behind [`docs/SOLUTIONS_2026-10-01.md`](../../docs/SOLUTIONS_2026-10-01.md) and the 2026-10-01
corrections in [`SLURM_JOBS.md`](../../SLURM_JOBS.md). A citation of the form `solutions-20260930/<path>`, or
`verify-*/<path>`, resolves to this folder.

Diagnostics only: no Slurm job ran and no scored seed set was run. The ledger entry of 2026-10-01 lists the
seeds that were touched.

| Path | Contents |
|---|---|
| `REPORTS_completed.md` | investigation reports A–D (Mission 7, cooperative lift, turning, NavGym) |
| `mission7/`, `coop/`, `turning/`, `navgym/` | the investigations' scripts, logs and JSON |
| `verify-coop/`, `verify-turning/`, `verify-mission7/`, `verify-navgym/` | the independent read-only verification of A–D |
| `sysid/` | biped gait system identification and the NavGym gym re-evaluation (investigator report, not independently verified) |
| `imu-kit/` | synthetic recovery runs behind `scripts/sensors/imu_allan.py` (`results.md`) and the Monte Carlo runs that set the test tolerances |
| `humanoid_check/` | check of the humanoid-with-arms sensor render |
| `ledger_corrections.md` | source text of the 2026-10-01 ledger and roadmap corrections |

Not copied, so only on the NFS original:
- parsed-episode caches (`*.pkl`, about 0.2 GB);
- raw traces (`*.npz`, gitignored repo-wide);
- any file over 1 MB (for example a 5 MB re-run `episodes.json`);
- the test-harness copies (`test_*.py`), so that pytest cannot collect them;
- the superseded first IMU-kit draft (`imu_kit/`), a symlink to it, and the critic and fix scratch folders.

Scripts here may hard-code the original location's paths.
