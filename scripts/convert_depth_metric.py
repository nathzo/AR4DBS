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
pip("coremltools>=7.0", "huggingface_hub", "timm")

import torch  # noqa: E402  (installed above)
import coremltools as ct  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

# ── Clone Depth Anything V2 source for the DPT architecture ───────────────────
SRC = "/tmp/depth-anything-v2"
if not os.path.exists(SRC):
    subprocess.check_call([
        "git", "clone", "--depth", "1",
        "https://github.com/DepthAnything/Depth-Anything-V2", SRC,
    ])
sys.path.insert(0, SRC)
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

# ── Download metric weights from HuggingFace ──────────────────────────────────
weights_path = hf_hub_download(
    repo_id="depth-anything/Depth-Anything-V2-Metric-Indoor-Small",
    filename="depth_anything_v2_metric_indoor_vits.pth",
    token=None,
)

# ── Load model ─────────────────────────────────────────────────────────────────
# max_depth=20 matches the published checkpoint training configuration.
model = DepthAnythingV2(
    encoder="vits",
    features=64,
    out_channels=[48, 96, 192, 384],
    max_depth=20,
)
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
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
