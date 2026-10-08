# Independent publication check — September 20, 2026

## Prioritized findings

1. **The experiments are not all complete or successful.** All 36 maze stages
   completed and passed their gates. Only five of twelve folding evaluation
   cells completed; two failed and five timed out. Neither adaptation has
   demonstrated improved folding performance.
2. **A completed renderer is not a successful fold.** The replacement gate
   completed 600 actions with 601 validated renders. Both ever-triggered and
   terminal success are false. Historical success footage uses an earlier
   checkpoint. The original dependent media array remains blocked.
3. **Other training remains unresolved.** The older TaskV2 array has two
   completed tasks, six failed tasks and one running task at this check.
   Held jobs remain held. No job was canceled, released, requeued or submitted
   during this publication review.
4. **The GitHub entry point needed a faster read and clearer setup.** The
   revised README now leads with contribution, demo, measured outcomes and
   boundaries, followed by CPU setup, architecture and remaining work.

## Checks performed

- Independently recomputed maze first-episode successes from all 36 JSON gates:
  Approach **380/384**, Corridor **375/384**, Full **379/384**. All 36 referenced
  checkpoints exist in the research workspace, and accounting reports successful
  completion. Weights are not included in the public repository.
- **146 CPU tests passed in 17.84 seconds** on the exact final publication
  source using the existing Python 3.11
  environment and the two required pinned upstream robot fixtures. The full
  suite includes the concurrent update's receipt-free reporting regression.
- Verified all three mission-video sidecars against published source clips,
  result JSON and GIF hashes. Verified all five folding GIF/poster pairs;
  their source/result hashes were also checked against the sibling project.
- Checked local documentation links and public portfolio, résumé and four
  project-page endpoints. The README's hero and folding preview frames were
  visually inspected. No new simulation or training was needed for this check.
- Preserved the concurrent publication's reporting fix, privacy conventions,
  profile copy and media. The final follow-up changes only documentation and
  test setup; it does not republish raw scheduler receipts or private paths.

## Environment and practical limits

The tested environment has Python 3.11.16, NumPy 1.26.0, PyTorch 2.7.0+cu128
(executed on CPU), MuJoCo 3.3.5, OpenCV headless 4.11.0.86 and pytest 9.0.2.
The [CPU dependency file](../requirements-test.txt) and README describe a
fresh setup with a CPU Torch wheel. That clean installation has not been run;
this validation uses the existing environment.

Nested assets commit `fc90fedd008b1e56a22e3c5221548d6b24f49707` supplies the
URDF and actuator fixtures; the BHL submodule remains pinned at
`984741a3623c93b0583ccfdc479f1f8b1c4d900e`. Tests do not certify GPU integration,
hardware transfer, unseen layouts or terminal folding performance.

## Recruiter-facing handoff

[Audit and four recommended pins](PORTFOLIO.md) ·
[Exact profile Markdown](github-profile-README.md) ·
[Reusable project README](PROJECT_README_TEMPLATE.md) ·
[Demo sequences](PORTFOLIO.md#demo-shot-lists--at-most-30-seconds) ·
[Folding success/failure/wrist gallery](FOLDING_MEDIA.md).

Native profile pins still require a manual GitHub profile action. A root BHL
license requires the owner's selection. External weights/assets, clean-install
validation and unfinished experiments remain visible limitations.
