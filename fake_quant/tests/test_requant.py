"""CPU unit tests for Algorithm 1 / Eq. 5–10."""

import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import quant_utils
import requant_utils


def _make_quantizer(w, bits=4, sym=False):
    q = quant_utils.WeightQuantizer()
    q.configure(bits, perchannel=True, sym=sym, mse=False)
    q.find_params(w)
    return q


def _row_loss(w, q, X, X_tilde):
    # Eq. 5: ||w X - q X̃||^2
    return torch.norm(w @ X - q @ X_tilde).pow(2).item()


def _naive_refine(w_fp, w_fq, H, B, quantizer, sweeps, neighborhood):
    """Row-by-row, column-by-column copy of Algorithm 1."""
    z, zmin, zmax = requant_utils.recover_codes(w_fq, quantizer)
    q = requant_utils.dequant_codes(z, quantizer)
    w = w_fp.to(requant_utils.PRECISION)
    H = H.to(requant_utils.PRECISION)
    B = None if B is None else B.to(requant_utils.PRECISION)
    drow, dcol = w.shape
    s = requant_utils._column_scale(quantizer, drow, w.device)
    ks = list(range(-neighborhood, 0)) + list(range(1, neighborhood + 1))

    for _ in range(sweeps):
        for i in range(drow):
            e = w[i] - requant_utils.dequant_codes(z, quantizer)[i]
            g = 2.0 * (e @ H)
            if B is not None:
                g = g - 2.0 * (w[i] @ B)
            for j in range(dcol):
                best_dl = None
                best_k = None
                for k in ks:
                    z_new = int(z[i, j].item()) + k
                    if z_new < zmin or z_new > zmax:
                        continue
                    dq = k * s[i]
                    dl = (-dq * g[j] + (dq ** 2) * H[j, j]).item()
                    if best_dl is None or dl < best_dl:
                        best_dl = dl
                        best_k = k
                if best_dl is not None and best_dl < 0:
                    z[i, j] = z[i, j] + best_k
                    g = g - 2.0 * (best_k * s[i]) * H[j, :]
    return z


class ReQuantTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _random_problem(self, drow=6, dcol=8, m=16, bits=4, sym=False):
        w = torch.randn(drow, dcol)
        X = torch.randn(dcol, m)
        X_tilde = X + 0.1 * torch.randn(dcol, m)
        quantizer = _make_quantizer(w, bits=bits, sym=sym)
        w_fq = quantizer.quantize(w)
        H = X_tilde @ X_tilde.t()
        B = (X_tilde - X) @ X_tilde.t()
        return w, w_fq, H, B, X, X_tilde, quantizer

    def test_matches_naive_algorithm1(self):
        w, w_fq, H, B, _, _, quantizer = self._random_problem()
        refined, _, _ = requant_utils.refine_weight(
            w, w_fq, H, B, quantizer, sweeps=3, neighborhood=2
        )
        z_vec, _, _ = requant_utils.recover_codes(refined, quantizer)
        z_naive = _naive_refine(w, w_fq, H, B, quantizer, sweeps=3, neighborhood=2)
        self.assertTrue(torch.equal(z_vec, z_naive), 'vectorized codes != naive Algorithm 1')

    def test_eq7_matches_brute_force(self):
        w, w_fq, H, B, X, X_tilde, quantizer = self._random_problem()
        z, zmin, zmax = requant_utils.recover_codes(w_fq, quantizer)
        q = requant_utils.dequant_codes(z, quantizer)
        e = w - q
        G = 2.0 * (e @ H - w @ B)
        s = requant_utils._column_scale(quantizer, w.shape[0], w.device)
        j = 3
        k = 1
        for i in range(w.shape[0]):
            z_new = int(z[i, j].item()) + k
            if z_new < zmin or z_new > zmax:
                continue
            dq = k * s[i]
            dl_closed = (-dq * G[i, j] + (dq ** 2) * H[j, j]).item()
            q2 = q.clone()
            q2[i, j] = q2[i, j] + dq
            dl_brute = _row_loss(w[i], q2[i], X, X_tilde) - _row_loss(w[i], q[i], X, X_tilde)
            self.assertTrue(
                math.isclose(dl_closed, dl_brute, rel_tol=1e-4, abs_tol=1e-4),
                f'Eq.7 mismatch row {i}: closed={dl_closed} brute={dl_brute}',
            )

    def test_loss_strictly_decreases(self):
        w, w_fq, H, B, _, _, quantizer = self._random_problem(drow=8, dcol=12, m=24)
        _, loss0, loss1 = requant_utils.refine_weight(
            w, w_fq, H, B, quantizer, sweeps=4, neighborhood=2
        )
        self.assertLess(loss1[0], loss0[0])

    def test_objective_equals_true_layer_error(self):
        """_objective with C must reproduce ||wX - q X̃||^2 exactly."""
        w, w_fq, H, B, X, X_tilde, quantizer = self._random_problem(
            drow=8, dcol=12, m=40
        )
        dX = X_tilde - X
        C = dX @ dX.t()
        e = w - w_fq
        total, quad, cross, const = requant_utils._objective(e, w, H, B, C)
        brute = _row_loss(w, w_fq, X, X_tilde)
        self.assertAlmostEqual(total, brute, delta=1e-3 * brute)
        self.assertAlmostEqual(total, quad + cross + const, delta=1e-3 * brute)
        self.assertGreater(total, 0.0)

    def test_reported_loss_stays_nonnegative(self):
        """Eq. 5 is a sum of squares, so refinement can never report L < 0."""
        w, w_fq, H, B, X, X_tilde, quantizer = self._random_problem(
            drow=8, dcol=12, m=40
        )
        dX = X_tilde - X
        _, loss0, loss1 = requant_utils.refine_weight(
            w, w_fq, H, B, quantizer, sweeps=4, neighborhood=2, drift_C=dX @ dX.t()
        )
        self.assertGreaterEqual(loss0[0], 0.0)
        self.assertGreaterEqual(loss1[0], 0.0)
        self.assertLess(loss1[0], loss0[0])

    def test_codes_stay_in_range_and_grid_frozen(self):
        w, w_fq, H, B, _, _, quantizer = self._random_problem(sym=True)
        scale0 = quantizer.scale.clone()
        zero0 = quantizer.zero.clone()
        refined, _, _ = requant_utils.refine_weight(
            w, w_fq, H, B, quantizer, sweeps=2, neighborhood=2
        )
        z, zmin, zmax = requant_utils.recover_codes(refined, quantizer)
        self.assertTrue(((z >= zmin) & (z <= zmax)).all())
        self.assertTrue(torch.equal(quantizer.scale, scale0))
        self.assertTrue(torch.equal(quantizer.zero, zero0))

    def test_hessian_accumulation_matches_paper_B(self):
        dcol, tokens = 5, 7
        layer = torch.nn.Linear(dcol, 3, bias=False)
        rq = requant_utils.ReQuant(layer)
        x_fp = torch.randn(tokens, dcol)
        x_q = x_fp + 0.2 * torch.randn(tokens, dcol)
        rq.add_batch(x_q.unsqueeze(0), x_fp.t())
        scale = math.sqrt(2 / 1)
        xq = scale * x_q.t()
        xf = scale * x_fp.t()
        H = xq @ xq.t()
        B = (xq - xf) @ xq.t()
        C = (xq - xf) @ (xq - xf).t()
        self.assertTrue(torch.allclose(rq.H, H, atol=1e-5))
        self.assertTrue(torch.allclose(rq.B, B, atol=1e-5))
        self.assertTrue(torch.allclose(rq.C, C, atol=1e-5))

    def test_hook_and_add_batch_agree_when_square(self):
        """dcol == tokens must not let the fp/quant layout be guessed wrongly.

        Qwen3-1.7B at seqlen 2048 hits this exactly, and X̃^T would pass any
        shape check while producing a completely different B.
        """
        dcol = tokens = 6
        layer = torch.nn.Linear(dcol, 4, bias=False)
        cache = {'m': []}
        hook = requant_utils._cache_fp_input(cache, 'm')
        x_fp = torch.randn(1, tokens, dcol)
        hook(layer, (x_fp,), None)
        x_q = x_fp + 0.2 * torch.randn(1, tokens, dcol)

        rq = requant_utils.ReQuant(layer)
        rq.add_batch(x_q, cache['m'][0])

        s = math.sqrt(2 / 1)
        xq = s * x_q.reshape(-1, dcol).t()
        xf = s * x_fp.reshape(-1, dcol).t()
        self.assertTrue(torch.allclose(rq.B, (xq - xf) @ xq.t(), atol=1e-5))
        # The transposed input is a different matrix, so the test has teeth.
        self.assertFalse(torch.allclose(rq.B, (xq - xf.t()) @ xq.t(), atol=1e-5))

    def test_add_batch_rejects_untransposed_fp_input(self):
        dcol, tokens = 5, 7
        layer = torch.nn.Linear(dcol, 3, bias=False)
        rq = requant_utils.ReQuant(layer)
        with self.assertRaises(ValueError):
            rq.add_batch(torch.randn(1, tokens, dcol), torch.randn(tokens, dcol))

    def test_generalizes_only_with_enough_calibration_tokens(self):
        """Refinement must beat RTN on held-out data once H̃ is well conditioned.

        X̃ is a deterministic function of X here, as it is in a real network.
        Below ~32x dcol tokens the cross term is fit to calibration noise and
        the refined weight is worse than RTN off calibration.
        """
        drow, dcol = 32, 256
        w = torch.randn(drow, dcol)
        quantizer = _make_quantizer(w)
        w_fq = quantizer.quantize(w)

        def sample(tokens, seed):
            g = torch.Generator().manual_seed(seed)
            x = torch.randn(dcol, tokens, generator=g)
            s = x.abs().max() / 7.0
            return x, (x / s).round().clamp(-8, 7) * s

        x_eval, x_eval_q = sample(8192, 999)

        def held_out(weight):
            return _row_loss(w, weight.float(), x_eval, x_eval_q)

        baseline = held_out(w_fq)

        x, x_q = sample(64 * dcol, 1)
        H = x_q @ x_q.t()
        B = (x_q - x) @ x_q.t()
        refined, _, _ = requant_utils.refine_weight(
            w, w_fq, H, B, quantizer, sweeps=2, neighborhood=2
        )
        self.assertLess(held_out(refined), baseline)

        x, x_q = sample(dcol // 2, 2)
        H = x_q @ x_q.t()
        B = (x_q - x) @ x_q.t()
        overfit, _, _ = requant_utils.refine_weight(
            w, w_fq, H, B, quantizer, sweeps=2, neighborhood=2
        )
        self.assertGreater(held_out(overfit), baseline)


if __name__ == '__main__':
    unittest.main()
