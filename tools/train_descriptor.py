#!/usr/bin/env python3
"""Train the descriptor CNN on self-supervised synthetic warp pairs.

No external dataset: random textured base images are generated (smoothed
noise + shapes + gradients), an anchor 32x32 patch is cropped, and its
positive is a random affine warp (rotation/scale/shear/translation) plus
brightness/contrast/noise jitter. In-batch InfoNCE contrastive loss teaches
the network to give warped copies of the same patch similar descriptors and
distinct patches dissimilar ones. That is exactly the property the pipeline
needs: stable 64-bit codes while a tracked keypoint moves.

The saved checkpoint uses the plain conv{i}.weight/conv{i}.bias + gemm.weight
/gemm.bias naming that export_descriptor_onnx.py --weights accepts, and
contains ONLY the slim architecture (no BatchNorm — the compiled graph has
none; brightness invariance comes from training-time intensity jitter).

Usage:
    python3 tools/train_descriptor.py [--steps 4000] [--batch 128] \
        [--out depthai_models/descriptor_weights_slim32.pth] \
        [--calibration_out depthai_models/descriptor_calib.npy]

Also dumps a (n, 512) float32 calibration array of raw descriptors for the
optional ITQ projection head (make_projection.py --strategy itq).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_ROOT / "tools"))
sys.path.insert(0, str(EXAMPLE_ROOT))
from export_descriptor_onnx import SLIM_CONV_STACK, DESC_DIM  # noqa: E402
from core.color import O1_SCALE, O2_SCALE  # noqa: E402

PATCH = 32


class SlimDescriptorNet(nn.Module):
    """Torch mirror of the slim ONNX graph (Conv/ReLU stack + GAP + Gemm).

    ``in_channels=3`` is the Opponent-LATCH-style color variant: the input is
    the (o1, o2, o3) opponent planes in [-1, 1] (see core/color.py).
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.relus = []
        for i, (cout, cin, stride, relu) in enumerate(SLIM_CONV_STACK):
            self.convs.append(
                nn.Conv2d(in_channels if i == 0 else cin, cout, 3,
                          stride=stride, padding=1)
            )
            self.relus.append(relu)
        self.gemm = nn.Linear(SLIM_CONV_STACK[-1][0], DESC_DIM)

    def forward(self, x):  # x: (B,1,32,32) in [0,1]
        for conv, relu in zip(self.convs, self.relus):
            x = conv(x)
            if relu:
                x = F.relu(x)
        x = x.mean(dim=(2, 3))  # GlobalAveragePool, matches the graph
        return self.gemm(x)

    def state_dict_for_export(self):
        """Map to the conv{i}/gemm naming the export tool understands."""
        out = {}
        for i, conv in enumerate(self.convs):
            out[f"conv{i}.weight"] = conv.weight.detach().cpu()
            out[f"conv{i}.bias"] = conv.bias.detach().cpu()
        out["gemm.weight"] = self.gemm.weight.detach().cpu()
        out["gemm.bias"] = self.gemm.bias.detach().cpu()
        return out


SRC = 64  # source crop size; positives are affine-sampled into 32x32


def make_batch(rng: np.random.Generator, batch: int, strong: bool):
    """Return (anchors, positives) float tensors (B,1,32,32) in [0,1].

    Fully vectorized: base textures are generated with torch ops and the
    positive warp uses F.affine_grid/grid_sample on the whole batch at once
    (~100x faster than per-patch numpy warps).
    """
    # Base textures: noise -> 3x3 box blur (twice) -> random rects -> gradient.
    img = torch.rand(batch, 1, SRC, SRC) * 255.0
    k = torch.ones(1, 1, 3, 3) / 9.0
    for _ in range(2):
        img = F.conv2d(img, k, padding=1)
    n_rect = torch.randint(3, 9, (batch,))
    for i in range(batch):
        for _ in range(int(n_rect[i])):
            x0, y0 = rng.integers(0, SRC - 12, 2)
            w, h = rng.integers(6, 24, 2)
            img[i, 0, y0 : y0 + h, x0 : x0 + w] = float(rng.integers(0, 255))
    gx = torch.linspace(-1, 1, SRC).view(1, 1, 1, SRC).expand(batch, 1, SRC, SRC)
    gy = torch.linspace(-1, 1, SRC).view(1, 1, SRC, 1).expand(batch, 1, SRC, SRC)
    ax = (torch.rand(batch, 1, 1, 1) * 80 - 40)
    ay = (torch.rand(batch, 1, 1, 1) * 80 - 40)
    img = (img + ax * gx + ay * gy).clamp(0, 255) / 255.0

    # Anchor: center-ish crop (small translation jitter), no warp.
    jx = torch.randint(-4, 5, (batch,)) 
    jy = torch.randint(-4, 5, (batch,))
    anchors = torch.empty(batch, 1, PATCH, PATCH)
    c = (SRC - PATCH) // 2
    for i in range(batch):
        anchors[i, 0] = img[i, 0, c + jy[i] : c + jy[i] + PATCH, c + jx[i] : c + jx[i] + PATCH]

    # Positive: random affine warp of the source crop sampled to 32x32.
    lim = 1.0 if strong else 0.45
    # Strong warps simulate viewpoint change for re-detection. Keep them mild:
    # the pipeline's core metric is per-frame temporal stability (tiny warps),
    # and aggressive rotation invariance destroys discrimination (measured:
    # +/-60 deg drops 256-way retrieval to ~0.2, +/-180 collapses it).
    ang = (torch.rand(batch) * 2 - 1) * (0.5 if strong else 0.3)
    scale = 1.0 + (torch.rand(batch) * 2 - 1) * (0.2 if strong else 0.15)
    shear = (torch.rand(batch) * 2 - 1) * (0.15 if strong else 0.1)
    tx = (torch.rand(batch) * 2 - 1) * lim
    ty = (torch.rand(batch) * 2 - 1) * lim
    ca, sa = torch.cos(ang), torch.sin(ang)
    # theta maps output (normalized) -> input (normalized); the 32x32 output
    # covers the central 32/64 = half of the source crop.
    r = PATCH / SRC
    theta = torch.zeros(batch, 2, 3)
    theta[:, 0, 0] = ca * scale * r + shear * sa * r
    theta[:, 0, 1] = -sa * scale * r
    theta[:, 1, 0] = sa * scale * r
    theta[:, 1, 1] = ca * scale * r
    theta[:, 0, 2] = tx * 0.5
    theta[:, 1, 2] = ty * 0.5
    grid = F.affine_grid(theta, (batch, 1, PATCH, PATCH), align_corners=False)
    positives = F.grid_sample(img, grid, mode="bilinear", padding_mode="reflection",
                              align_corners=False)

    # Intensity jitter (BN-free invariance): contrast, brightness, noise.
    def jitter(p):
        p = p * (torch.rand(batch, 1, 1, 1) * 0.8 + 0.6)
        p = p + (torch.rand(batch, 1, 1, 1) * 0.32 - 0.16)
        p = p + torch.randn_like(p) * (torch.rand(batch, 1, 1, 1) * 0.05)
        return p.clamp(0, 1)

    return jitter(anchors), jitter(positives)


def _blur3(img: torch.Tensor) -> torch.Tensor:  # (B,3,H,W) per-channel box blur
    k = torch.ones(3, 1, 3, 3) / 9.0
    for _ in range(2):
        img = F.conv2d(img, k, padding=1, groups=3)
    return img


def _opponent_jitter(p: torch.Tensor, batch: int) -> torch.Tensor:
    """Photometric jitter in normalized opponent space (B,3,32,32) [-1,1].

    Hue rotation on the (o1, o2) plane (axis rescaled to true opponent axes),
    saturation scale, intensity contrast/brightness, noise, then quantization
    onto the uint8 storage grid the device sees (zero at 127.5).
    """
    phi = (torch.rand(batch, 1, 1, 1) * 2 - 1) * 0.26  # +/-15 deg
    o1, o2, o3 = p[:, 0:1], p[:, 1:2], p[:, 2:3]
    o1r = torch.cos(phi) * o1 - torch.sin(phi) * o2 * (O2_SCALE / O1_SCALE)
    o2r = torch.sin(phi) * o1 * (O1_SCALE / O2_SCALE) + torch.cos(phi) * o2
    sat = torch.rand(batch, 1, 1, 1) * 0.7 + 0.7  # 0.7-1.4
    o1r, o2r = o1r * sat, o2r * sat
    o3 = o3 * (torch.rand(batch, 1, 1, 1) * 0.8 + 0.6) \
        + (torch.rand(batch, 1, 1, 1) * 0.32 - 0.16)
    q = torch.cat([o1r, o2r, o3], dim=1)
    q = q + torch.randn_like(q) * (torch.rand(batch, 1, 1, 1) * 0.05)
    q = q.clamp(-1.0, 1.0)
    return torch.round(q * 127.5) / 127.5  # uint8 grid, as on device


def make_color_batch(rng: np.random.Generator, batch: int, strong: bool,
                     isoluminant_only: bool = False):
    """(anchors, positives) (B,3,32,32) in [-1,1], quantized to the u8 grid.

    Textures are generated directly in normalized opponent space (rather than
    RGB -> transform) so chromatic edges are guaranteed: the network cannot
    solve training with the intensity channel alone. ~30% of samples are
    isoluminant (flat o3, all structure in o1/o2) and ~15% achromatic
    (o1=o2=0, structure only in o3, so plain grayscale scenes still work).
    """
    opp = _blur3(torch.rand(batch, 3, SRC, SRC) * 2 - 1)
    for i in range(batch):
        for _ in range(int(rng.integers(3, 9))):
            x0, y0 = rng.integers(0, SRC - 12, 2)
            w, h = rng.integers(6, 24, 2)
            val = torch.tensor(rng.uniform(-1, 1, 3), dtype=opp.dtype).view(3, 1, 1)
            opp[i, :, y0 : y0 + h, x0 : x0 + w] = val
    gx = torch.linspace(-1, 1, SRC).view(1, 1, 1, SRC)
    gy = torch.linspace(-1, 1, SRC).view(1, 1, SRC, 1)
    opp = opp + (torch.rand(batch, 3, 1, 1) * 0.6 - 0.3) * gx \
              + (torch.rand(batch, 3, 1, 1) * 0.6 - 0.3) * gy

    mode = rng.random(batch)
    iso = mode < (1.0 if isoluminant_only else 0.3)
    if iso.any():
        # Flat intensity (smooth gradient + offset); chroma keeps its edges.
        o3_flat = (torch.rand(batch, 1, 1, 1) * 0.6 - 0.3) * gx \
                + (torch.rand(batch, 1, 1, 1) * 0.6 - 0.3) * gy \
                + (torch.rand(batch, 1, 1, 1) * 0.4 - 0.2)
        opp[iso, 2] = o3_flat[iso, 0].clamp(-1, 1)
    if not isoluminant_only:
        gray = (mode >= 0.3) & (mode < 0.45)
        opp[gray, 0:2] = 0.0
    opp = opp.clamp(-1.0, 1.0)

    # Anchor: center-ish crop with translation jitter (shared across channels).
    jx = torch.randint(-4, 5, (batch,))
    jy = torch.randint(-4, 5, (batch,))
    anchors = torch.empty(batch, 3, PATCH, PATCH)
    c = (SRC - PATCH) // 2
    for i in range(batch):
        anchors[i] = opp[i, :, c + jy[i] : c + jy[i] + PATCH, c + jx[i] : c + jx[i] + PATCH]

    # Positive: random affine warp, same geometry for all three channels.
    lim = 1.0 if strong else 0.45
    ang = (torch.rand(batch) * 2 - 1) * (0.5 if strong else 0.3)
    scale = 1.0 + (torch.rand(batch) * 2 - 1) * (0.2 if strong else 0.15)
    shear = (torch.rand(batch) * 2 - 1) * (0.15 if strong else 0.1)
    tx = (torch.rand(batch) * 2 - 1) * lim
    ty = (torch.rand(batch) * 2 - 1) * lim
    ca, sa = torch.cos(ang), torch.sin(ang)
    r = PATCH / SRC
    theta = torch.zeros(batch, 2, 3)
    theta[:, 0, 0] = ca * scale * r + shear * sa * r
    theta[:, 0, 1] = -sa * scale * r
    theta[:, 1, 0] = sa * scale * r
    theta[:, 1, 1] = ca * scale * r
    theta[:, 0, 2] = tx * 0.5
    theta[:, 1, 2] = ty * 0.5
    grid = F.affine_grid(theta, (batch, 3, PATCH, PATCH), align_corners=False)
    positives = F.grid_sample(opp, grid, mode="bilinear", padding_mode="reflection",
                              align_corners=False)

    return _opponent_jitter(anchors, batch), _opponent_jitter(positives, batch)


def evaluate(model: nn.Module, rng: np.random.Generator, batch: int = 512,
             color: bool = False, isoluminant_only: bool = False) -> float:
    """Fraction of anchors whose nearest in-batch descriptor is its positive."""
    model.eval()
    with torch.no_grad():
        if color:
            a, p = make_color_batch(rng, batch, strong=True,
                                    isoluminant_only=isoluminant_only)
        else:
            a, p = make_batch(rng, batch, strong=True)
        da = F.normalize(model(a), dim=1)
        dp = F.normalize(model(p), dim=1)
        sim = da @ dp.T
        correct = (sim.argmax(dim=1) == torch.arange(batch)).float().mean().item()
    model.train()
    return correct


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--color", action="store_true",
                        help="Train the 3-channel opponent-color variant "
                        "(Opponent-LATCH style; input is (o1,o2,o3) in [-1,1]).")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--calibration_out", type=Path, default=None)
    args = parser.parse_args()
    if args.out is None:
        name = "descriptor_weights_slim32_color.pth" if args.color else "descriptor_weights_slim32.pth"
        args.out = EXAMPLE_ROOT / "depthai_models" / name
    if args.calibration_out is None:
        name = "descriptor_calib_color.npy" if args.color else "descriptor_calib.npy"
        args.calibration_out = EXAMPLE_ROOT / "depthai_models" / name

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = SlimDescriptorNet(in_channels=3 if args.color else 1)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    model.train()

    t0 = __import__("time").time()
    for step in range(args.steps):
        strong = step > args.steps // 4  # curriculum: easy warps first
        if args.color:
            a, p = make_color_batch(rng, args.batch, strong)
        else:
            a, p = make_batch(rng, args.batch, strong)
        da = F.normalize(model(a), dim=1)
        dp = F.normalize(model(p), dim=1)
        logits = da @ dp.T / 0.07  # InfoNCE temperature
        labels = torch.arange(args.batch)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps - 1:
            acc = (logits.argmax(dim=1) == labels).float().mean().item()
            print(f"step {step:5d} loss {loss.item():.3f} train-acc {acc:.3f} "
                  f"({__import__('time').time() - t0:.0f}s)", flush=True)

    acc = evaluate(model, np.random.default_rng(123), color=args.color)
    print(f"held-out warp retrieval accuracy (512-way): {acc:.3f}", flush=True)
    if args.color:
        iso = evaluate(model, np.random.default_rng(124), color=True,
                       isoluminant_only=True)
        print(f"held-out ISOLUMINANT retrieval accuracy (512-way): {iso:.3f} "
              "(intensity channel flat; color must carry the match)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict_for_export(), args.out)
    print(f"saved {args.out}", flush=True)

    # Calibration dump for the ITQ head: raw descriptors of random patches.
    model.eval()
    with torch.no_grad():
        calib = []
        gen = np.random.default_rng(99)
        for _ in range(40):
            if args.color:
                a, _ = make_color_batch(gen, 128, strong=True)
            else:
                a, _ = make_batch(gen, 128, strong=True)
            calib.append(model(a).numpy())
        calib = np.concatenate(calib, 0).astype(np.float32)
    np.save(args.calibration_out, calib)
    print(f"saved {args.calibration_out} {calib.shape}", flush=True)


if __name__ == "__main__":
    main()
