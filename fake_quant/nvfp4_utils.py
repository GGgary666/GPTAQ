"""NVFP4 (E2M1 + fp8-e4m3 group scale + tensor global scale).

Faithful port of compressed_tensors NVFP4 / llm-compressor NVFP4ReQuantGrid:
    scale_eff = scale_fp8 / global_scale
    z = E2M1_round(w / scale_eff)
    w_q = z * scale_eff

Weights: static TENSOR_GROUP, group_size=16, symmetric, zp=0.
Activations: same grid; local scales are dynamic; global_scale is optional
(static if set, else computed from the current tensor).
"""

import math

import torch
import torch.nn as nn

PRECISION = torch.float32

NVFP4_GROUP_SIZE = 16
FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)  # 448.0
FP8_E4M3_EPS = 0.125

NVFP4_GRID_VALUES = (
    -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5,
    0.0,
    0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
)


_GRID_CACHE = {}


def nvfp4_grid(device, dtype=PRECISION):
    key = ('grid', device, dtype)
    t = _GRID_CACHE.get(key)
    if t is None:
        t = torch.tensor(NVFP4_GRID_VALUES, device=device, dtype=dtype)
        _GRID_CACHE[key] = t
    return t


# Magnitudes of the E2M1 grid, and the midpoints between consecutive ones.
_FP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP4_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# compressed_tensors breaks ties to even, so a value sitting exactly on a
# midpoint snaps down at these indices and up at the others.
_FP4_TIE_DOWN = (0, 2, 4, 6)


def _fp4_tables(device, dtype):
    key = ('fp4', device, dtype)
    tables = _GRID_CACHE.get(key)
    if tables is None:
        bounds = torch.tensor(_FP4_MIDPOINTS, device=device, dtype=dtype)
        # bucketize(right=True) sends a value equal to a boundary to the upper
        # bucket, so nudge the tie-down boundaries above the midpoint.
        up = torch.full_like(bounds, float('inf'))
        for i in _FP4_TIE_DOWN:
            bounds[i] = torch.nextafter(bounds[i], up[i])
        mags = torch.tensor(_FP4_MAGNITUDES, device=device, dtype=dtype)
        tables = (bounds, mags)
        _GRID_CACHE[key] = tables
    return tables


def cast_to_fp4(x):
    """Nearest E2M1 value. Thresholds match compressed_tensors fp4_utils."""
    a = x.abs()
    bounds, mags = _fp4_tables(a.device, a.dtype)
    return torch.sign(x) * mags[torch.bucketize(a, bounds, right=True)]


def quantize_fp8_e4m3(x):
    """Round to fp8-e4m3 and upcast back to fp32 (exact representable values)."""
    x32 = x.to(PRECISION)
    return x32.to(torch.float8_e4m3fn).to(PRECISION)


def generate_global_scale(max_abs):
    """global_scale = FP8_max * FP4_max / max(|x|)."""
    max_abs = torch.as_tensor(max_abs, dtype=PRECISION)
    tiny = torch.finfo(PRECISION).tiny
    max_abs = max_abs.clamp(min=tiny)
    gs = (FP8_E4M3_MAX * FP4_E2M1_MAX) / max_abs
    gs = torch.nan_to_num(gs, nan=1.0, posinf=1.0, neginf=1.0)
    return gs.reshape(1).to(PRECISION)


def _require_group_aligned(last_dim, group_size=NVFP4_GROUP_SIZE):
    if last_dim % group_size != 0:
        raise ValueError(
            f'NVFP4 requires last dim divisible by {group_size}, got {last_dim}'
        )


def group_max(x, group_size=NVFP4_GROUP_SIZE):
    """Per-group abs-max on the last dimension. Returns [..., num_groups]."""
    _require_group_aligned(x.shape[-1], group_size)
    grouped = x.unflatten(-1, (x.shape[-1] // group_size, group_size))
    return grouped.abs().amax(dim=-1)


def compute_stored_scales(max_g, global_scale):
    """fp8-e4m3 stored scale: round(global_scale * max_g / 6)."""
    scale_local = max_g.to(PRECISION) / FP4_E2M1_MAX
    stored = quantize_fp8_e4m3(global_scale.to(PRECISION) * scale_local)
    return torch.where(
        stored == 0,
        torch.full_like(stored, FP8_E4M3_EPS),
        stored,
    )


def effective_scale(stored_scale, global_scale):
    return stored_scale.to(PRECISION) / global_scale.to(PRECISION)


def expand_group_scale(scale_g, group_size=NVFP4_GROUP_SIZE):
    """[..., num_groups] -> [..., num_groups * group_size]."""
    return scale_g.unsqueeze(-1).expand(*scale_g.shape, group_size).flatten(-2)


def quantize_nvfp4(x, stored_scale, global_scale, group_size=NVFP4_GROUP_SIZE):
    """Return E2M1 float codes with the same shape as x."""
    scale_eff = expand_group_scale(effective_scale(stored_scale, global_scale), group_size)
    scale_eff = scale_eff.to(device=x.device, dtype=PRECISION)
    return cast_to_fp4(x.to(PRECISION) / scale_eff.clamp(min=torch.finfo(PRECISION).tiny))


def dequantize_nvfp4(z, stored_scale, global_scale, group_size=NVFP4_GROUP_SIZE):
    scale_eff = expand_group_scale(effective_scale(stored_scale, global_scale), group_size)
    scale_eff = scale_eff.to(device=z.device, dtype=PRECISION)
    return z.to(PRECISION) * scale_eff


def fake_quantize_nvfp4(x, stored_scale, global_scale, group_size=NVFP4_GROUP_SIZE):
    z = quantize_nvfp4(x, stored_scale, global_scale, group_size)
    return dequantize_nvfp4(z, stored_scale, global_scale, group_size), z


def snap_to_grid(z):
    grid = nvfp4_grid(z.device, z.dtype)
    idx = torch.searchsorted(grid, z.contiguous())
    idx = (idx - 1).clamp(0, grid.numel() - 1)
    snap_low = grid[idx]
    snap_high = grid[(idx + 1).clamp(max=grid.numel() - 1)]
    nearer_high = (z - snap_low).abs() > (snap_high - z).abs()
    idx = torch.where(nearer_high, (idx + 1).clamp(max=grid.numel() - 1), idx)
    return grid[idx], idx


class NVFP4WeightQuantizer(nn.Module):
    """Static NVFP4 RTN for a Linear weight [out, in]."""

    format = 'nvfp4'

    def __init__(self, group_size=NVFP4_GROUP_SIZE):
        super().__init__()
        self.bits = 4
        self.group_size = group_size
        self.sym = True
        self.register_buffer('scale', torch.zeros(1))
        self.register_buffer('global_scale', torch.ones(1))
        self.register_buffer('zero', torch.zeros(1))

    def find_params(self, w):
        _require_group_aligned(w.shape[-1], self.group_size)
        w32 = w.detach().to(PRECISION)
        self.global_scale = generate_global_scale(w32.abs().max()).to(device=w.device)
        max_g = group_max(w32, self.group_size)
        self.scale = compute_stored_scales(max_g, self.global_scale).to(device=w.device)

    def quantize(self, w):
        q, _ = fake_quantize_nvfp4(w, self.scale, self.global_scale, self.group_size)
        return q.to(dtype=w.dtype)

    def ready(self):
        return self.scale.numel() > 1 or bool(self.scale.item() != 0)

    def cpu(self):
        self.scale = self.scale.cpu()
        self.global_scale = self.global_scale.cpu()
        self.zero = self.zero.cpu()
        return self


def apply_nvfp4_activation(x, global_scale=None, group_size=NVFP4_GROUP_SIZE, clip_ratio=1.0):
    """Dynamic-local NVFP4 fake-quant on the last dimension.

    ``clip_ratio`` only shrinks the observed max used for scales; the
    input itself is not hard-clipped (same spirit as INT ActQuantizer).
    """
    _require_group_aligned(x.shape[-1], group_size)
    x32 = x.to(PRECISION)
    if global_scale is None:
        gs = generate_global_scale(x32.abs().max() * clip_ratio).to(device=x.device)
    else:
        gs = global_scale.to(device=x.device, dtype=PRECISION).reshape(1)
    max_g = group_max(x32, group_size) * clip_ratio
    stored = compute_stored_scales(max_g, gs)
    q, _ = fake_quantize_nvfp4(x32, stored, gs, group_size)
    return q.to(dtype=x.dtype)
