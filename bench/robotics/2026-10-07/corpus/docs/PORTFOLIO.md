# ML / AI and robotics portfolio audit — September 20, 2026

Target: **machine learning / AI engineer and robotics engineer**.
[Portfolio](https://jose-sanchez-portfolio-com.vercel.app) ·
[GitHub](https://github.com/joses2017smjh)

## Prioritized audit

1. **Lead with four inspectable projects.** SPUR shows model-to-service work;
   BHL shows learned control and evaluation; pruning connects perception to
   motion; MetaNaviT adds retrieval engineering. Feature folding as research.
2. **Correct stale or overstated results.** New maze training is complete.
   Folding adaptation scored 3/24 against 8/24 baseline, with only 5/12 cells
   complete. The new wrist recordings show one failure; successful historical
   footage uses an older checkpoint. Separate Isaac scores from MuJoCo demos.
3. **Align the existing profile, website and LinkedIn.** State both target
   areas, use one contact address and distinguish personal contributions from
   upstream methods and team work. The profile repository already exists.
4. **Put a compact demo beside the evidence.** GIFs suit repository READMEs;
   the site uses videos with static posters and reduced-motion support.
   Preserve success and failure examples with their actual controller labels.
5. **Make setup and limitations explicit.** External weights, assets and HPC
   configuration remain prerequisites. BHL has no root license; do not add an
   MIT badge or invent licensing. Public copies omit new raw scheduler records.

## Recommended pins

| Order | Repository | Recruiting signal | Main limit |
|---|---|---|---|
| 1 | [SPUR depth service](https://github.com/joses2017smjh/spur-depth-service) | PyTorch → FastAPI → split ONNX, parity and latency | Synthetic evaluation; no validated field accuracy |
| 2 | [BHL robustness ladder](https://github.com/joses2017smjh/bhl-robustness-ladder) | RL, simulator debugging, seeded evaluation, HPC | Known route; new Isaac checkpoints not verified on hardware |
| 3 | [Isaac pruning](https://github.com/joses2017smjh/isaac-sim-pruning-workflow) | RGB-D control, sensor gates, independent grader | One known target; classical tracking and rigid-piece release |
| 4 | [MetaNaviT](https://github.com/joses2017smjh/MetaNavT) | Retrieval, data layer and APIs in a six-person team | Small CPU fixture; BM25 beats hybrid overall nDCG@10 |

Use GitHub profile → Customize your pins. README features do not change the
actual pins. Keep repository names: existing résumé and demo links use them.

## Exact profile and LinkedIn copy

- [GitHub profile README](github-profile-README.md): the complete Markdown.
- [LinkedIn fields](LINKEDIN_2026-09-20.md): headline, About, project bullets,
  Featured links and a short publication checklist. Applying text in LinkedIn
  requires the account editor; this file does not represent a live profile edit.

Website hero: **I build ML systems and perception for robots.**

Role line: **Seeking machine learning / AI and robotics engineering roles.**

Intro: Explore depth inference, learned robot control, vision-guided pruning,
and retrieval software. Four case studies connect working demos to source code
and measured results.

Contact: **Hiring for ML, AI, or robotics?**

## Featured README changes

Lead with what the project does, a demo and one measured result. Then give
contribution, architecture, important tradeoffs, stack, actual setup commands,
and limits. Link the portfolio as a demo/case study, not as a hosted inference
service. Keep deeper evidence below the first screen.

- **SPUR:** label timing per six-view model call, precision and hardware;
  exclude HTTP/preprocessing from that claim. Preserve synthetic-data limits.
- **BHL:** show current 379/384 Isaac results, separate frozen-gait missions,
  negative controls, reproducibility requirements and folding media.
- **Pruning:** replace editorial instructions with a reader-facing walkthrough;
  identify classical tracking, selected episode and surrogate release.
- **MetaNaviT:** identify the team contribution and provide CPU fixture setup;
  show the BM25 comparison instead of implying an overall hybrid win.

Keep badges only for existing, meaningful checks or a verified license. Do
not invent CI status, deployment availability, experience or performance.

## Demo shot lists — at most 30 seconds

| Project | Shot list |
|---|---|
| SPUR, 25 s | 0–5 s synthetic RGB/calibration; 5–13 s depth request and point cloud; 13–20 s Torch/ONNX parity; 20–25 s named-GPU model latency and synthetic-only label. |
| BHL, 25 s | 0–5 s known map/stations; 5–15 s inspection route; 15–21 s wrong-branch control; 21–25 s linked scores, oracle routing and older frozen-gait label. |
| Pruning, 20 s | Use the existing complete capture: approach, gated release at 7.8 s, falling rigid spur, return home. Pair with tracking-loss stop; caption one target and 17/17 checks. |
| MetaNaviT, 25 s | 0–5 s conflicting configs; 5–13 s retrieve current source; 13–20 s inspect citations and proposed file operation; 20–25 s approval gate. Label illustrated fixture workflow. |
| Folding, 29 s | About 14 s earlier policy success, then 14 s failure; or compare the new failure with its wrist views. Label checkpoint, checker-first-hit outcome and adjusted playback. |

[Folding GIFs and provenance](FOLDING_MEDIA.md) include the actual left and
right arm cameras. A latched historical success does not certify the final
frame. Demonstration replay is separate from learned-policy action.

## Publication and checks

The public BHL copy retains metrics, traces and failure cases while removing
new private workspace paths and scheduler receipts; see
[PUBLIC_EVIDENCE.md](PUBLIC_EVIDENCE.md). The original workspace and HPC queue
remain separate from publication.

Check rendered pages, local assets, source/demo/contact links, mobile layout,
keyboard focus and reduced-motion behavior. Record actual commands and
outcomes in the local session handoff. No additional training is required for
these documentation and media changes.
