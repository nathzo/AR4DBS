#!/usr/bin/env python3
"""Convert Depth Anything V2 Metric Indoor Small to a CoreML .mlpackage.

Usage:
    python convert_depth_metric.py <output.mlpackage>

Input expected by the model  : [1, 3, 518, 518] float32, ImageNet-normalised.
    CoreMLDepthEstimator applies ImageNet normalisation automatically for
    MultiArray inputs, so no normalisation should be embedded here.
Output produced by the model : [1, 518, 518] float32, metric depth in metres
    (larger value = farther from camera).
"""

import subprocess
import sys
import os


def pip(*args: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", *args]
    )


pip(
    "torch", "torchvision",
    "--index-url", "https://download.pytorch.org/whl/cpu",
)
pip("coremltools>=7.0", "timm", "opencv-python-headless")

import urllib.request  # noqa: E402
import torch  # noqa: E402  (installed above)
import coremltools as ct  # noqa: E402

# ── Clone Depth Anything V2 source for the DPT architecture ───────────────────
SRC = "/tmp/depth-anything-v2"
if not os.path.exists(SRC):
    subprocess.check_call([
        "git", "clone", "--depth", "1",
        "https://github.com/DepthAnything/Depth-Anything-V2", SRC,
    ])
sys.path.insert(0, os.path.join(SRC, "metric_depth"))

# Patch the int() casts in dpt.py that coremltools cannot lower to MIL ops.
# The tracer already treats these values as constants, so removing int() is safe.
_dpt_path = os.path.join(SRC, "metric_depth", "depth_anything_v2", "dpt.py")
with open(_dpt_path) as _f:
    _code = _f.read()
_code = _code.replace(
    "(int(patch_h * 14), int(patch_w * 14))",
    "(patch_h * 14, patch_w * 14)",
)
with open(_dpt_path, "w") as _f:
    _f.write(_code)

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

# ── Download metric weights (public, no authentication required) ───────────────
weights_path = "/tmp/depth_anything_v2_metric_hypersim_vits.pth"
urllib.request.urlretrieve(
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small"
    "/resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true",
    weights_path,
)

# ── Load model ─────────────────────────────────────────────────────────────────
# max_depth=20 matches the Hypersim checkpoint training configuration.
model = DepthAnythingV2(
    encoder="vits",
    features=64,
    out_channels=[48, 96, 192, 384],
    max_depth=20,
)
model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
model.eval()

# ── Trace ──────────────────────────────────────────────────────────────────────
# Fixed 518x518 input — no dynamic shapes, so tracing is safe.
# Values are ImageNet-normalised floats; CoreMLDepthEstimator handles that.
H, W = 518, 518
dummy = torch.randn(1, 3, H, W)
with torch.no_grad():
    traced = torch.jit.trace(model, dummy)

# ── Convert to CoreML ──────────────────────────────────────────────────────────
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="image", shape=(1, 3, H, W))],
    outputs=[ct.TensorType(name="depth")],
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.iOS17,
)

out = sys.argv[1] if len(sys.argv) > 1 else "DepthAnythingV2MetricIndoorSmallF16.mlpackage"
mlmodel.save(out)
print(f"Saved {out}")
