# Trained-policy folding: successes, failures and wrist cameras

These clips show trained SmolVLA policies controlling two SO-ARM101 arms in
Isaac Sim 5.1. The policy consumes three Storm-rendered camera views and joint
state. Labels come from LeHome's geometric folding checker. The motions are
policy actions, not demonstration replay.

| Clip | What it establishes | Evidence |
|---|---|---|
| [Short-pants success](gifs/folding-policy-success.gif) | An earlier raster-adapted policy triggered the official checker during a 400-action rollout. | [Sidecar](gifs/folding-policy-success.json) |
| [Short-pants failure](gifs/folding-policy-failure.gif) | The same earlier policy failed from a different recorded starting pose. | [Sidecar](gifs/folding-policy-failure.json) |
| [New adapted-policy failure](gifs/folding-adapted-failure.gif) | The seed-0 adaptation completed 600 actions on a short-sleeve top; the checker never passed and the terminal state failed. | [Sidecar](gifs/folding-adapted-failure.json) |
| [Left wrist](gifs/folding-adapted-left-wrist.gif) | The actual left camera during that same new failure. | [Sidecar](gifs/folding-adapted-left-wrist.json) |
| [Right wrist](gifs/folding-adapted-right-wrist.gif) | The actual right camera during that same new failure. | [Sidecar](gifs/folding-adapted-right-wrist.json) |

The historical success belongs to an **earlier checkpoint**, not either newly
adapted policy. Its record latches success once the checker passes; it does
not identify the first passing step or establish that the final frame still
passes. These selected development episodes do not establish a success rate
or a controlled comparison between policy generations.

The new failure has 601 validated camera renders, 9,774 cloth particles and
0.215 m maximum particle displacement. It confirms a completed, visible
closed-loop rollout with cloth motion; it does not demonstrate a successful
fold. The three new camera clips depict the same episode.

The historical GIFs had missing or zero frame delays. Their full captured
sequences are presented at an explicitly adjusted 10 frames per second, with
a one-second end hold. Their playback is not a physical-time measurement.
The new episode spans 6.67 seconds of simulation and is shown at half speed,
with a one-second end hold. Captions state these playback choices.

All source frames are processed without generated motion or interpolation.
The historical top views are the middle third of the documented
left-wrist/top/right-wrist recordings. New wrist clips preserve each entire
native camera frame. Derivatives use spatial resizing, caption footers and
a fixed 96-color palette. Matching PNG files are static middle-frame previews.

Each JSON sidecar records the original clip and result hashes, portable paths
relative to [IsaacSimFolding](https://github.com/joses2017smjh/IsaacSimFolding),
camera identity, checker outcome, playback changes and output hashes. Original
recordings remain in that source project and are not bundled here. No new
simulation or training run was performed to produce these derivatives.
