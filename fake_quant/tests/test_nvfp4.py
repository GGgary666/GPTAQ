"""CPU tests for NVFP4 Q/DQ and ReQuant on the E2M1 grid."""

import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nvfp4_utils
import requant_utils


GRID = torch.tensor(nvfp4_utils.NVFP4_GRID_VALUES)


def _make_nvfp4_quantizer(w):
    q = nvfp4_utils.NVFP4WeightQuantizer()
    q.find_params(w)
    return q


def _row_loss(w, q, X, X_tilde):
    return torch.norm(w @ X - q @ X_tilde).pow(2).item()


class NVFP4Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_cast_to_fp4_thresholds(self):
        x = torch.tensor([
            0.0, 0.25, 0.26, 0.74, 0.75, 1.25, 1.26, 1.74, 1.75, 2.5,
            2.51, 3.49, 3.5, 5.0, 5.01, 8.0, -0.26, -1.26, -5.01,
        ])
        got = nvfp4_utils.cast_to_fp4(x)
        want = torch.tensor([
            0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0,
            3.0, 3.0, 4.0, 4.0, 6.0, 6.0, -0.5, -1.5, -6.0,
        ])
        self.assertTrue(torch.equal(got, want), f'{got} != {want}')

    def test_codes_are_on_e2m1_grid(self):
        w = torch.randn(8, 32)
        qtz = _make_nvfp4_quantizer(w)
        w_fq = qtz.quantize(w)
        z = nvfp4_utils.quantize_nvfp4(w_fq, qtz.scale, qtz.global_scale)
        hit = torch.isclose(z.unsqueeze(-1), GRID, atol=1e-6).any(-1)
        self.assertTrue(hit.all())

    def test_roundtrip_fakequant(self):
        w = torch.randn(6, 32)
        qtz = _make_nvfp4_quantizer(w)
        w_fq = qtz.quantize(w)
        z = nvfp4_utils.quantize_nvfp4(w, qtz.scale, qtz.global_scale)
        recon = nvfp4_utils.dequantize_nvfp4(z, qtz.scale, qtz.global_scale)
        self.assertTrue(torch.allclose(recon, w_fq.float(), atol=1e-5))

    def test_effective_scale_formula(self):
        w = torch.randn(4, 16)
        qtz = _make_nvfp4_quantizer(w)
        scale_eff = nvfp4_utils.effective_scale(qtz.scale, qtz.global_scale)
        self.assertEqual(scale_eff.shape, (4, 1))
        self.assertTrue((scale_eff > 0).all())

    def test_global_scale_formula(self):
        gs = nvfp4_utils.generate_global_scale(torch.tensor(12.0))
        self.assertTrue(math.isclose(
            gs.item(), nvfp4_utils.FP8_E4M3_MAX * nvfp4_utils.FP4_E2M1_MAX / 12.0,
            rel_tol=1e-6,
        ))

    def test_requant_stays_on_grid_and_decreases_loss(self):
        drow, dcol, m = 6, 32, 24
        w = torch.randn(drow, dcol)
        X = torch.randn(dcol, m)
        X_tilde = X + 0.15 * torch.randn(dcol, m)
        qtz = _make_nvfp4_quantizer(w)
        w_fq = qtz.quantize(w)
        dX = X_tilde - X
        H = X_tilde @ X_tilde.t()
        B = dX @ X_tilde.t()
        refined, loss0, loss1 = requant_utils.refine_weight(
            w, w_fq, H, B, qtz, sweeps=3, neighborhood=2, drift_C=dX @ dX.t()
        )
        self.assertLess(loss1[0], loss0[0])
        self.assertGreaterEqual(loss1[0], 0.0)
        z = nvfp4_utils.quantize_nvfp4(refined, qtz.scale, qtz.global_scale)
        hit = torch.isclose(z.unsqueeze(-1), GRID, atol=1e-6).any(-1)
        self.assertTrue(hit.all())
        scale0 = qtz.scale.clone()
        gs0 = qtz.global_scale.clone()
        self.assertTrue(torch.equal(qtz.scale, scale0))
        self.assertTrue(torch.equal(qtz.global_scale, gs0))

    def test_matches_naive_algorithm1_nvfp4(self):
        """The index-space inner loop must reproduce Algorithm 1 exactly."""
        drow, dcol, m = 5, 48, 64
        w = torch.randn(drow, dcol)
        X = torch.randn(dcol, m)
        X_tilde = X + 0.1 * torch.randn(dcol, m)
        qtz = _make_nvfp4_quantizer(w)
        w_fq = qtz.quantize(w)
        H = (X_tilde @ X_tilde.t()).float()
        B = ((X_tilde - X) @ X_tilde.t()).float()
        sweeps, K = 3, 2

        scale = nvfp4_utils.effective_scale(qtz.scale, qtz.global_scale)
        grid_vals = GRID.tolist()
        z = nvfp4_utils.quantize_nvfp4(w_fq, qtz.scale, qtz.global_scale)
        idx = [[grid_vals.index(min(grid_vals, key=lambda v: abs(v - z[i, j].item())))
                for j in range(dcol)] for i in range(drow)]
        wf = w.float()

        for _ in range(sweeps):
            for i in range(drow):
                for j in range(dcol):
                    q_row = torch.tensor(
                        [grid_vals[idx[i][c]] * scale[i, c // 16].item() for c in range(dcol)]
                    )
                    g = 2.0 * ((wf[i] - q_row) @ H - wf[i] @ B)
                    s = scale[i, j // 16].item()
                    best, best_dl = None, 0.0
                    for k in list(range(-K, 0)) + list(range(1, K + 1)):
                        c = idx[i][j] + k
                        if c < 0 or c >= len(grid_vals):
                            continue
                        dq = (grid_vals[c] - grid_vals[idx[i][j]]) * s
                        dl = (-dq * g[j] + dq * dq * H[j, j]).item()
                        if dl < best_dl:
                            best, best_dl = c, dl
                    if best is not None:
                        idx[i][j] = best

        refined, _, _ = requant_utils.refine_weight(
            w, w_fq, H, B, qtz, sweeps=sweeps, neighborhood=K
        )
        grid = requant_utils.build_grid(qtz, K, w.device, drow)
        self.assertTrue(torch.equal(grid.codes(refined), torch.tensor(idx)))

    def test_eq7_matches_brute_force_nvfp4(self):
        drow, dcol, m = 4, 32, 16
        w = torch.randn(drow, dcol)
        X = torch.randn(dcol, m)
        X_tilde = X + 0.1 * torch.randn(dcol, m)
        qtz = _make_nvfp4_quantizer(w)
        w_fq = qtz.quantize(w)
        H = X_tilde @ X_tilde.t()
        B = (X_tilde - X) @ X_tilde.t()
        grid = requant_utils.build_grid(qtz, 2, w.device, drow)
        codes = grid.codes(w_fq)
        q = grid.values(codes)
        e = w.float() - q
        G = 2.0 * (e @ H.float() - w.float() @ B.float())
        j = 5
        grid.set_column(j)
        delta_q, valid, _ = grid.candidates(codes[:, j])
        for i in range(drow):
            for c in range(delta_q.shape[1]):
                if not valid[i, c]:
                    continue
                dq = delta_q[i, c]
                dl_closed = (-dq * G[i, j] + (dq ** 2) * H[j, j]).item()
                q2 = q.clone()
                q2[i, j] = q2[i, j] + dq
                dl_brute = _row_loss(w[i].float(), q2[i], X.float(), X_tilde.float()) - \
                    _row_loss(w[i].float(), q[i], X.float(), X_tilde.float())
                self.assertTrue(
                    math.isclose(dl_closed, dl_brute, rel_tol=1e-3, abs_tol=1e-3),
                    f'Eq.7 NVFP4 mismatch row {i} cand {c}: {dl_closed} vs {dl_brute}',
                )

    def test_act_nvfp4_last_dim_aligned(self):
        x = torch.randn(3, 7, 32)
        y = nvfp4_utils.apply_nvfp4_activation(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())
        self.assertFalse(torch.equal(y, x))
        gs = nvfp4_utils.generate_global_scale(x.abs().max())
        stored = nvfp4_utils.compute_stored_scales(nvfp4_utils.group_max(x.float()), gs)
        recon, z = nvfp4_utils.fake_quantize_nvfp4(x, stored, gs)
        self.assertTrue(torch.allclose(y.float(), recon, atol=1e-5))
        hit = torch.isclose(z.unsqueeze(-1), GRID, atol=1e-6).any(-1)
        self.assertTrue(hit.all())


if __name__ == '__main__':
    unittest.main()
