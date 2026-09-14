"""CPU tests for the Hadamard factorizations used by QuaRot rotations."""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hadamard_utils


class TestPaley(unittest.TestCase):
    def test_order68_is_hadamard(self):
        H = hadamard_utils.get_hadPaley(68)
        self.assertEqual(H.shape, (68, 68))
        self.assertTrue((H.abs() == 1).all())
        self.assertTrue(torch.equal(H @ H.t(), 68 * torch.eye(68)))

    def test_rejects_unsupported_orders(self):
        with self.assertRaises(AssertionError):
            hadamard_utils.get_hadPaley(70)  # 69 is not prime
        with self.assertRaises(AssertionError):
            hadamard_utils.get_hadPaley(14)  # 13 = 1 (mod 4)


class TestGetHadK(unittest.TestCase):
    # Adding the Paley branch must not steal any dimension the tabulated
    # matrices already handled.
    EXPECTED = {
        2048: 1,      # qwen3-1.7b hidden
        4096: 1,      # qwen3-8b / llama-3 hidden
        5120: 40,     # qwen3-14b hidden
        6144: 12,     # qwen3-1.7b intermediate
        11008: 172,   # llama-2-7b intermediate
        12288: 12,    # qwen3-8b intermediate
        14336: 28,    # llama-3-8b intermediate
        17408: 68,    # qwen3-14b intermediate
    }

    def test_dispatch(self):
        for n, k in self.EXPECTED.items():
            with self.subTest(n=n):
                self.assertEqual(hadamard_utils.get_hadK(n)[1], k)

    def test_preserves_norm(self):
        for n in self.EXPECTED:
            with self.subTest(n=n):
                x = torch.randn(2, n, dtype=torch.float64)
                y = hadamard_utils.matmul_hadU(x)
                self.assertTrue(torch.allclose(y.norm(), x.norm(), rtol=1e-6))


if __name__ == '__main__':
    unittest.main()
