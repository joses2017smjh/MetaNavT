# Calibrated stereo input

`scripts/perception/stereo_depth.py` runs actual CPU image matching with
[OpenCV StereoSGBM](https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html).
It reads a left/right pair, computes both disparity directions, rejects
inconsistent/low-texture/out-of-range matches, and writes left-camera
optical-axis depth, a boolean valid mask, disparity, and provenance metadata.
Invalid pixels are NaN; they are not filled or treated as free space.

This is a classical perception baseline, not a trained stereo network or a
demonstration of an integrated humanoid navigation policy. Existing maze PPO
still consumes ray-cast depths. Physical-camera accuracy remains unmeasured.
The [physical IMU contract](IMU_INPUT.md) documents the probable sensor model,
SI units, body/world frames and the remaining hardware acceptance checks.

Both input images must already be horizontally rectified at the calibrated
resolution with a common focal length. Calibration JSON is mandatory and has
schema `bhl-rectified-stereo-v1`, plus all of these fields:

| Field | Meaning |
|---|---|
| `image_width`, `image_height` | rectified dimensions in pixels, integers |
| `fx_px` | common rectified horizontal focal length, pixels |
| `baseline_m` | positive physical baseline, meters |
| `cx_left_px`, `cx_right_px` | rectified horizontal principal points, pixels |
| `rectified` | must be JSON `true` |

Use measured calibration or exported simulator camera calibration. No values
are supplied for the user's hardware because its exact calibration is unknown.
Raw/distorted images require calibration and rectification first; this program
does not silently treat them as rectified. Optional `--left-valid-mask` and
`--right-valid-mask` boolean `.npy` files exclude invalid rectification borders.

For disparity `d=u_left-u_right`, depth is
`z=fx*baseline/(d-(cx_left-cx_right))`. The principal-point correction matters
when rectification does not use identical horizontal principal points. Output
is image-plane depth, not Euclidean/slant range. OpenCV fixed-point disparity
is divided by 16 before conversion, following its
[calibration and reconstruction documentation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).

```bash
python scripts/perception/stereo_depth.py \
  --left left_rectified.png --right right_rectified.png \
  --calibration calibrated_rectified_camera.json \
  --left-time 1750000000.000 --right-time 1750000000.004 \
  --out results/stereo_capture_001

python scripts/perception/ssd_input.py \
  --image left_rectified.png --capture-time 1750000000.000 --replay \
  --weights /path/to/actual_ssdlite_checkpoint.pt --num-classes 5 \
  --depth-metadata results/stereo_capture_001/metadata.json \
  --out results/ssd_capture_001.json
```

Timestamps in the example are illustrative. Supply original capture times in
the same clock domain. The stereo program rejects pairs more than 35 ms apart
by default; its compute time never replaces capture time. The SSD adapter
verifies the left image path/hash and capture timestamp, loads the valid mask,
and retains depth provenance. `--replay` explicitly disables live freshness
claims; live inputs are checked at consumption time and may correctly be stale
after slow inference. Custom station classes require actual trained detector
weights; COCO classes do not recognize arbitrary maze buttons.

Tests use synthetic cameras only as numeric fixtures: an eight-pixel shift
at focal length160 pixels and baseline0.10 m must return depth2 m. They also
check principal-point offsets, left/right disagreement, invalid pixels,
textureless images, calibration rejection and the SSD metadata connection.
