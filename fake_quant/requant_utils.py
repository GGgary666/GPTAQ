"""ReQuant: fixed-grid discrete refinement (arXiv:2608.07019).

Faithful implementation of Algorithm 1 and Eq. 5–10. The quantization
grid (scale, zero-point, bit-width) is frozen; only codes move.
INT uses a uniform affine grid; NVFP4 uses the 15-value E2M1 grid.
"""

import math
import logging

import torch
import torch.nn as nn

import quant_utils
import model_utils
import nvfp4_utils
import utils

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

PRECISION = torch.float32
INF = float('inf')
# (total, quadratic, cross, const), matching _objective
_ZERO_LOSS = (0.0, 0.0, 0.0, 0.0)

SEQUENTIAL = [
    ['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'],
    ['self_attn.o_proj.module'],
    ['mlp.up_proj.module', 'mlp.gate_proj.module'],
    ['mlp.down_proj.module'],
]


def snapshot_linear_weights(model):
    weights = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weights[name] = module.weight.data.detach().cpu().clone()
    return weights


def recover_codes(w_fq, quantizer):
    """Invert the fake-quant weight to integer codes on the frozen grid."""
    scale = quantizer.scale.to(device=w_fq.device, dtype=PRECISION)
    zero = quantizer.zero.to(device=w_fq.device, dtype=PRECISION)
    maxq = quantizer.maxq.to(device=w_fq.device, dtype=PRECISION)
    w = w_fq.to(PRECISION)
    if quantizer.sym:
        z, _ = quant_utils.sym_quant(w, scale, maxq)
        zmin, zmax = -(maxq + 1), maxq
    else:
        z, _, _ = quant_utils.asym_quant(w, scale, zero, maxq)
        zmin, zmax = torch.zeros_like(maxq), maxq
    recon = dequant_codes(z, quantizer)
    rel = (recon - w).abs() / scale.clamp(min=1e-12)
    if rel.max().item() > 0.51:
        logging.warning(
            'recover_codes: max relative reconstruction %.4f (possible off-grid weight)',
            rel.max().item(),
        )
    return z.to(torch.int64), int(zmin.item()), int(zmax.item())


def dequant_codes(z, quantizer):
    scale = quantizer.scale.to(device=z.device, dtype=PRECISION)
    zero = quantizer.zero.to(device=z.device, dtype=PRECISION)
    if quantizer.sym:
        return scale * z.to(PRECISION)
    return scale * (z.to(PRECISION) - zero)


def _column_scale(quantizer, drow, device):
    scale = quantizer.scale.to(device=device, dtype=PRECISION)
    return scale.reshape(drow)


def is_nvfp4_quantizer(quantizer):
    return getattr(quantizer, 'format', 'int') == 'nvfp4'


class AffineReQuantGrid:
    """Uniform affine grid: q = (z - zero) * scale. Codes are the levels z."""

    def __init__(self, quantizer, neighborhood, device, drow):
        self.quantizer = quantizer
        self.ks = _offsets(neighborhood, device)
        # Δq = k * s is the same for every column, so build it once.
        s = _column_scale(quantizer, drow, device).unsqueeze(1)
        self.delta_q = self.ks.to(PRECISION).unsqueeze(0) * s
        self.cmin = None
        self.cmax = None

    def codes(self, w_fq):
        z, cmin, cmax = recover_codes(w_fq, self.quantizer)
        self.cmin, self.cmax = cmin, cmax
        return z

    def values(self, codes):
        return dequant_codes(codes, self.quantizer)

    def candidates(self, codes_col):
        new_codes = codes_col.unsqueeze(1) + self.ks
        valid = (new_codes >= self.cmin) & (new_codes <= self.cmax)
        return self.delta_q, valid, new_codes


class NVFP4ReQuantGrid:
    """Non-uniform E2M1 grid. Codes index NVFP4_GRID_VALUES; q = grid[c] * s."""

    def __init__(self, quantizer, neighborhood, device, drow):
        self.quantizer = quantizer
        self.group_size = getattr(quantizer, 'group_size', nvfp4_utils.NVFP4_GROUP_SIZE)
        self.ks = _offsets(neighborhood, device)
        self.lut = nvfp4_utils.nvfp4_grid(device)
        self.cmax = self.lut.numel() - 1
        self.scale_eff = nvfp4_utils.effective_scale(
            quantizer.scale.to(device=device, dtype=PRECISION),
            quantizer.global_scale.to(device=device, dtype=PRECISION),
        )

    def codes(self, w_fq):
        z = nvfp4_utils.quantize_nvfp4(
            w_fq, self.quantizer.scale, self.quantizer.global_scale, self.group_size
        )
        _, idx = nvfp4_utils.snap_to_grid(z.to(PRECISION))
        return idx.to(torch.int64)

    def values(self, codes):
        scale = nvfp4_utils.expand_group_scale(self.scale_eff, self.group_size)
        return self.lut[codes] * scale.to(device=codes.device)

    def set_column(self, col):
        self._s_col = self.scale_eff[:, col // self.group_size].unsqueeze(1)

    def candidates(self, codes_col):
        new_codes = codes_col.unsqueeze(1) + self.ks
        valid = (new_codes >= 0) & (new_codes <= self.cmax)
        delta_level = self.lut[new_codes.clamp(0, self.cmax)] - self.lut[codes_col].unsqueeze(1)
        return delta_level * self._s_col, valid, new_codes


def _offsets(neighborhood, device):
    ks = torch.arange(-neighborhood, neighborhood + 1, device=device)
    return ks[ks != 0]


def build_grid(quantizer, neighborhood, device, drow):
    cls = NVFP4ReQuantGrid if is_nvfp4_quantizer(quantizer) else AffineReQuantGrid
    return cls(quantizer, neighborhood, device, drow)


def _objective(e, w, H, B, C=None):
    """Eq. 5: ||e X̃ - w ΔX||^2, split into its three terms.

    Returns (total, quadratic, cross, const). The total is a sum of squares and
    must not be negative; C is what makes that check possible.
    """
    quad = (e * (e @ H)).sum().item()
    cross = 0.0 if B is None else -2.0 * (e * (w @ B)).sum().item()
    const = 0.0 if C is None else (w * (w @ C)).sum().item()
    return quad + cross + const, quad, cross, const


def refine_weight(
    w_fp,
    w_fq,
    hessian,
    cross_B,
    quantizer,
    sweeps,
    neighborhood,
    coord_order='forward',
    seed=0,
    drift_C=None,
):
    """Algorithm 1, vectorized over output rows.

    Args:
        w_fp: full-precision row-stacked weight [drow, dcol]
        w_fq: current fake-quantized weight (initializer output)
        hessian: H̃ = X̃ X̃^T  [dcol, dcol]
        cross_B: B = ΔX X̃^T with ΔX = X̃ - X; None drops the cross term
        quantizer: frozen WeightQuantizer or NVFP4WeightQuantizer
        sweeps: T
        neighborhood: K
        coord_order: 'forward' (paper), 'reverse', or 'random' (fixed perm)
        drift_C: C = ΔX ΔX^T, reporting only; None leaves the loss up to a
            constant and therefore possibly negative
    """
    if getattr(quantizer, 'bits', 16) >= 16:
        return w_fq, _ZERO_LOSS, _ZERO_LOSS

    device = w_fq.device
    final_dtype = w_fq.dtype
    w = w_fp.to(device=device, dtype=PRECISION)
    H = hessian.to(device=device, dtype=PRECISION)
    B = None if cross_B is None else cross_B.to(device=device, dtype=PRECISION)
    C = None if drift_C is None else drift_C.to(device=device, dtype=PRECISION)

    grid = build_grid(quantizer, neighborhood, device, w.shape[0])
    codes = grid.codes(w_fq.to(device=device, dtype=PRECISION))
    codes_init = codes.clone()
    q = grid.values(codes)
    e = w - q

    # Eq. 6: g = 2(e H̃ - w B)
    G = 2.0 * (e @ H)
    if B is not None:
        G = G - 2.0 * (w @ B)

    loss_before = _objective(e, w, H, B, C)

    dcol = w.shape[1]
    if coord_order == 'forward':
        columns = range(dcol)
    elif coord_order == 'reverse':
        columns = range(dcol - 1, -1, -1)
    elif coord_order == 'random':
        g = torch.Generator(device='cpu')
        g.manual_seed(int(seed))
        columns = torch.randperm(dcol, generator=g).tolist()
    else:
        raise ValueError(f'Unknown coord_order {coord_order}')

    set_column = getattr(grid, 'set_column', None)
    h_diag = H.diagonal()
    zero = torch.zeros((), device=device, dtype=PRECISION)
    for _ in range(sweeps):
        for j in columns:
            if set_column is not None:
                set_column(j)
            codes_col = codes[:, j]
            delta_q, valid, new_codes = grid.candidates(codes_col)

            # Eq. 7: ΔL = -Δq g_j + (Δq)^2 H̃_jj
            delta_L = -delta_q * G[:, j].unsqueeze(1) + delta_q.square() * h_diag[j]
            delta_L = torch.where(valid, delta_L, INF)

            best = delta_L.argmin(dim=1, keepdim=True)
            accept = delta_L.gather(1, best) < 0
            codes[:, j] = torch.where(
                accept.squeeze(1), new_codes.gather(1, best).squeeze(1), codes_col
            )
            # Eq. 10: g ← g - 2 Δq* H̃_j,:. Rejected rows contribute Δq* = 0, so
            # this needs no host sync on whether anything was accepted.
            dq = torch.where(accept, delta_q.gather(1, best), zero)
            G.addr_(dq.squeeze(1), H[j], alpha=-2.0)

    q_final = grid.values(codes)
    e_final = w - q_final
    loss_after = _objective(e_final, w, H, B, C)
    if not math.isfinite(loss_after[0]):
        raise ArithmeticError('Non-finite ReQuant loss')
    if loss_after[0] > loss_before[0] + 1e-5 * max(abs(loss_before[0]), 1.0):
        logging.warning(
            'ReQuant objective increased: %.6f -> %.6f', loss_before[0], loss_after[0]
        )
    if C is not None and loss_after[0] < -1e-3 * max(abs(loss_before[0]), 1.0):
        logging.warning(
            'ReQuant loss %.6f < 0 but Eq.5 is a sum of squares; H/B/C disagree',
            loss_after[0],
        )
    # Healthy refinement nudges a minority of codes by one step; a large moved
    # fraction or drift means the objective is being overfit.
    drift = (codes - codes_init).to(PRECISION)
    logging.debug(
        'ReQuant moved %.1f%% of codes, mean |dz| = %.3f, max |dz| = %.0f',
        100.0 * (drift != 0).float().mean().item(),
        drift.abs().mean().item(),
        drift.abs().max().item(),
    )
    return q_final.to(dtype=final_dtype), loss_before, loss_after


class ReQuant:
    """Accumulate H̃ and B with the same running-average convention as GPTAQ."""

    def __init__(self, layer):
        self.layer = layer
        self.dev = layer.weight.device
        columns = layer.weight.shape[1]
        self.H = torch.zeros((columns, columns), device=self.dev, dtype=PRECISION)
        self.B = torch.zeros((columns, columns), device=self.dev, dtype=PRECISION)
        # ΔX ΔX^T only enters Eq.5 through the e-independent term tr[w C w^T],
        # so the descent ignores it, but without it the reported loss is not the
        # true ||wX - q X̃||^2 and can come out negative.
        self.C = torch.zeros((columns, columns), device=self.dev, dtype=PRECISION)
        self.nsamples = 0
        # ||ΔX||^2 / ||X̃||^2 says how far the quantized branch has drifted from
        # FP, i.e. how much of Eq.5 the cross term is being asked to undo.
        self.delta_sq = 0.0
        self.input_sq = 0.0

    def add_batch(self, inp, fp_inp):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        x_fp = fp_inp
        if x_fp.shape != inp.shape:
            if x_fp.t().shape == inp.shape:
                x_fp = x_fp.t()
            elif x_fp.numel() == inp.numel():
                x_fp = x_fp.reshape(inp.shape)
            else:
                raise ValueError(f'FP/quant activation shape mismatch {x_fp.shape} vs {inp.shape}')

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.B *= self.nsamples / (self.nsamples + tmp)
        self.C *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        scale = math.sqrt(2 / self.nsamples)
        x_q = scale * inp.to(PRECISION)
        x_fp = scale * x_fp.to(PRECISION)
        self.H += x_q.matmul(x_q.t())
        # Paper: ΔX = X̃ - X, B = ΔX X̃^T
        dX = x_q - x_fp
        self.B += dX.matmul(x_q.t())
        self.C += dX.matmul(dX.t())
        self.delta_sq += dX.pow(2).sum().item()
        self.input_sq += x_q.pow(2).sum().item()

    def relative_drift(self):
        if self.input_sq <= 0:
            return 0.0
        return math.sqrt(self.delta_sq / self.input_sq)

    def free(self):
        self.H = None
        self.B = None
        self.C = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)


def _maybe_to(t, device):
    return t if t.device == device else t.to(device)


def _apply_weight_dict(modules, weight_dict, layer_idx=None, prefix=None):
    for name, module in modules.items():
        if prefix is None:
            key = f'model.layers.{layer_idx}.{name}'
        else:
            key = prefix + name
        if key not in weight_dict:
            raise KeyError(f'Missing snapshot for {key}')
        src = weight_dict[key]
        module.weight.data.copy_(src.to(device=module.weight.device, dtype=module.weight.dtype))


def _cache_fp_input(cache, name):
    def hook(_module, inp, _out):
        x = inp[0].detach()
        if len(x.shape) == 3:
            x = x.reshape((-1, x.shape[-1]))
        cache[name].append(x.t())
    return hook


def _flatten_quant_input(x):
    if len(x.shape) == 2:
        x = x.unsqueeze(0)
    if len(x.shape) == 3:
        x = x.reshape((-1, x.shape[-1]))
    return x


@torch.no_grad()
def requant_fwrd(model, dataloader, dev, args, quantizers, fp_weights):
    """One calibration pass collecting H̃, B, then T sweeps of Algorithm 1."""
    logging.info('-----ReQuant refinement (T=%d, K=%d, branch=%s, w_format=%s)-----',
                 args.requant_sweeps, args.requant_neighborhood, args.requant_fp_branch,
                 getattr(args, 'w_format', 'int'))
    w_format = getattr(args, 'w_format', 'int')
    if w_format == 'int' and args.w_groupsize != -1:
        raise ValueError('ReQuant INT path only supports per-channel weights (w_groupsize=-1)')
    if w_format == 'nvfp4' and args.w_bits != 4:
        raise ValueError('NVFP4 ReQuant requires --w_bits 4')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    model_utils.maybe_move_rotary(model, dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    nsamples = args.nsamples
    act_device = 'cpu' if args.requant_offload_activations else dev
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=act_device
    )
    cache = {'i': 0, 'layer_kwargs': {}}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.to(act_device)
            cache['i'] += 1
            cache['layer_kwargs'] = kwargs
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    # H̃ is [dcol, dcol] but is estimated from nsamples * seqlen tokens. Below
    # that, it is rank deficient and coordinate descent drives e to zero inside
    # the row space while it grows unchecked in the null space.
    cal_tokens = nsamples * model.seqlen
    max_dcol = max(model.config.hidden_size, model.config.intermediate_size)
    if cal_tokens < max_dcol:
        logging.warning(
            'ReQuant: %d calibration tokens < %d columns; H is rank deficient and '
            'the refinement will overfit. Paper uses 512 x 2048 = 1048576 tokens.',
            cal_tokens, max_dcol,
        )

    layer_kwargs = cache['layer_kwargs']
    fp_inps = inps.clone()
    outs = torch.zeros_like(inps)
    chunk = max(1, int(args.requant_chunk))
    fp_branch = args.requant_fp_branch

    for i in range(len(layers)):
        logging.info('ReQuant layer %d / %d', i, len(layers))
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        names = [n for n in full if 'lm_head' not in n]
        q_weights = {n: full[n].weight.data.detach().clone() for n in names}

        rq = {}
        for name in names:
            rq[name] = ReQuant(full[name])

        # Modules within a SEQUENTIAL group share an input, so only the first of
        # each group needs statistics; the rest copy them below.
        collectors = [g[0] for g in SEQUENTIAL if any(n in rq for n in g)]

        for start in range(0, nsamples, chunk):
            end = min(start + chunk, nsamples)
            fp_cache = {n: [] for n in collectors}

            bits_config = quant_utils.disable_act_quant(layer)
            if fp_branch == 'true_fp':
                _apply_weight_dict({n: full[n] for n in names}, fp_weights, layer_idx=i)
            handles = [
                full[name].register_forward_hook(_cache_fp_input(fp_cache, name))
                for name in collectors
            ]
            for j in range(start, end):
                x = _maybe_to(fp_inps[j], dev).unsqueeze(0)
                y = model_utils.forward_decoder_layer(layer, x, layer_kwargs, model)
                fp_inps[j] = _maybe_to(y.squeeze(0), act_device)
            for h in handles:
                h.remove()

            for n in names:
                full[n].weight.data.copy_(q_weights[n])
            quant_utils.enable_act_quant(layer, bits_config)

            # Weights are frozen until every chunk has been consumed, so all
            # groups can be hooked during a single pass over the chunk.
            handles = []
            for name in collectors:
                consumed = {'k': 0}

                def add_batch(_module, inp, _out, first_name=name, bucket=consumed):
                    x_q = _flatten_quant_input(inp[0].data)
                    idx = bucket['k']
                    bucket['k'] = idx + 1
                    rq[first_name].add_batch(x_q, fp_cache[first_name][idx])

                handles.append(full[name].register_forward_hook(add_batch))
            for j in range(start, end):
                x = _maybe_to(inps[j], dev).unsqueeze(0)
                y = model_utils.forward_decoder_layer(layer, x, layer_kwargs, model)
                outs[j] = _maybe_to(y.squeeze(0), act_device)
            for h in handles:
                h.remove()

        for group in SEQUENTIAL:
            subset = [n for n in group if n in rq]
            if len(subset) < 2:
                continue
            first = subset[0]
            for name in subset[1:]:
                rq[name].H = rq[first].H
                rq[name].B = rq[first].B
                rq[name].C = rq[first].C
                rq[name].nsamples = rq[first].nsamples
                rq[name].delta_sq = rq[first].delta_sq
                rq[name].input_sq = rq[first].input_sq

        for name in names:
            qkey = f'model.layers.{i}.{name}'
            if qkey not in quantizers:
                logging.warning('No quantizer for %s; skip', qkey)
                continue
            cross_B = rq[name].B
            if getattr(args, 'requant_objective', 'full') == 'simplified':
                cross_B = None
            elif getattr(args, 'requant_cross_alpha', 1.0) != 1.0:
                cross_B = cross_B * float(args.requant_cross_alpha)
            w_ref, loss0, loss1 = refine_weight(
                fp_weights[qkey],
                full[name].weight.data,
                rq[name].H,
                cross_B,
                quantizers[qkey],
                sweeps=args.requant_sweeps,
                neighborhood=args.requant_neighborhood,
                coord_order=args.requant_coord_order,
                seed=args.seed,
                drift_C=rq[name].C,
            )
            full[name].weight.data.copy_(w_ref.to(dtype=full[name].weight.dtype))
            # quad is the weight-only error ReQuant can actually remove; const is
            # the drift inherited from earlier layers, which it cannot.
            logging.info(
                '  %s: L %.4g -> %.4g  [quad %.4g -> %.4g, cross %.4g -> %.4g, '
                'const %.4g]  |dX|/|X~| = %.3f',
                name, loss0[0], loss1[0], loss0[1], loss1[1], loss0[2], loss1[2],
                loss0[3], rq[name].relative_drift(),
            )
        for name in names:
            rq[name].free()

        for j in range(nsamples):
            x = _maybe_to(inps[j], dev).unsqueeze(0)
            y = model_utils.forward_decoder_layer(layer, x, layer_kwargs, model)
            outs[j] = _maybe_to(y.squeeze(0), act_device)

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info('-----ReQuant refinement done-----\n')
