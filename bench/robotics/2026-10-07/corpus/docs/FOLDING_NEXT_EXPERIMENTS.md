# Folding: two next experiments

Research checked **20 September 2026**. These are proposed experiments, not
implemented algorithms, submitted jobs or claimed improvements.

The current strict short-pants evaluations measured **8/24** successes for the
historical raster-adapted baseline and **3/24** for adaptation seed 1; both had
0/4 on Unseen garments. The sample is small and the newer adaptation has not
demonstrated an improvement. Training data, schedule and checkpoint selection
also differ, so the decline does not isolate one cause. See the completed and
incomplete results separately in [the measured audit](CLOTH_FOLDING_WEEKEND.md).

Some bad historical numbers came from frozen CPU cloth or stale camera images.
Current incomplete class evaluations additionally expose a garment-switch
stall and a scorer/mesh index mismatch. Those infrastructure errors are not
policy failures. Diagnose the exact particle indexing and isolate garment
process lifetimes before extending the four-class benchmark; do not silently
drop a scoring condition or treat missing episodes as failed folds.

## 1. Shorter action execution, with the checkpoint held fixed

**Hypothesis:** executing 50 predicted actions before observing again permits
avoidable drift around a grasp. The LeHome winner predicts 30-action chunks
but executes only 3 actions for pants and 5 for tops. It also uses overlap
anchoring, so its settings are inspiration rather than a guaranteed optimum
for SmolVLA. [Learning to Fold, 25 June 2026, section 7 and table 4](https://arxiv.org/html/2606.27163v1).

Use the existing successful raster-adapted checkpoint and compare execution
lengths **50, 10 and 5**, leaving its predicted 50-action chunk, weights,
normalizers and denoising settings unchanged. Reset the action deque after
changing execution length. First screen eight fixed development poses across
multiple Seen garments: 24 episodes total at a fixed 600-action budget. Use
fresh single-garment processes to avoid the known switching stall.

Freeze the selected setting before comparing it with the default on the full
24-episode short-pants protocol with two evaluation seeds. Save exact starting
poses and sampling seeds; report Seen/Unseen separately. Measure checker
attainment, a separately rechecked terminal verdict, time to first success,
action discontinuities and inference cost. More frequent replanning can also
introduce jitter. This tests an inference setting, not a newly trained policy.

## 2. Target grasp supervision without changing the data and budget together

**Hypothesis:** sparse supervision near grasp commitment limits useful
adaptation. FoldNet++ reports targeted pre-grasp fine-tuning and a large wrist
camera ablation. Its main fine-tuning comparison changes learning rate and
update count; a shared-configuration appendix comparison still uses only ten
trials. These motivate a controlled local test, not a transferable success-rate
claim. [FoldNet++, 11 September 2026, section 4.5 and appendix G](https://arxiv.org/html/2609.12433v1).

First extend the clean raster captures for sparse classes: the current
stratified training split contains only two successful long-pants episodes.
Retain the existing held-out garment identities and audit whole episodes,
three camera views and genuine future action chunks. A replay's eventual
success does not prove every intermediate state matches its original recording.

Then compare two samplers on the **same frozen dataset and initial checkpoint**:
uniform class/episode sampling versus a 50:50 mixture of uniform samples and
annotated pre-grasp windows. Identify those windows from wrist footage and
executed gripper commands. Hold optimizer, update count, validation protocol
and two training seeds fixed; choose checkpoints without using evaluation
outcomes. Evaluate with the same complete protocol as the baseline. Keep the
original sampler if closed-loop outcomes do not improve.

This remains behavior cloning. Real on-policy correction would require valid
expert actions at the states the policy actually visits. XR-2's corrective-data
study motivates collecting those recoveries later; saving failed actions alone
is not DAgger. [XR-2, 3 September 2026, sections III-C and IV-C](https://arxiv.org/html/2609.03591v1).

## Research and reporting boundaries

- [FolDeX, 9 September 2026](https://arxiv.org/abs/2609.10243) supports controlled
  initializations and held-out physical objects. These simulation results do
  not establish real-robot transfer.
- [RotateIt!, 17 September 2026](https://arxiv.org/abs/2609.19817) is a newer
  single-arm dynamic unfolding approach. Adopting its rotation primitive would
  change the present bimanual folding problem and needs a separate study.
- FoldNet++'s [official project page](https://pku-epic.github.io/FoldNetXX/)
  still marks code and data “Coming Soon” at review time. Neither proposed
  experiment depends on an unavailable release or a replacement simulator.
- The [published-media documentation](FOLDING_MEDIA.md) correctly labels the
  older checkpoint's observed success and the new seed-0 failure. The older
  successful clip has no recorded terminal verdict; the new failure has an
  explicit terminal check. Curated GIFs are qualitative evidence, not benchmark
  denominators.
