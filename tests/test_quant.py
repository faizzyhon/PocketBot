"""python -m unittest discover tests     (from the repo root)"""
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
from quant import fq_torch, pack4, unpack4


class Quant(unittest.TestCase):
    def setUp(self): self.w = np.random.default_rng(0).normal(size=(24, 96)).astype(np.float32)

    def test_roundtrip_error_is_within_half_a_step(self):
        p, s = pack4(self.w, 32); back = unpack4(p, s, 32)
        step = np.repeat(np.abs(s), 32, axis=1).reshape(self.w.shape)
        self.assertTrue(np.all(np.abs(back - self.w) <= step * 1.0001))   # the clipped +extreme can be off by up to one step, everything else by half
        self.assertLess(np.sqrt(((back - self.w) ** 2).mean()) / self.w.std(), 0.25)

    def test_packed_shape_and_range(self):
        p, s = pack4(self.w, 32); self.assertEqual(p.shape, (24, 3, 16)); self.assertEqual(s.shape, (24, 3)); self.assertEqual(p.dtype, np.uint8)

    def test_extreme_of_each_block_is_exact(self):
        p, s = pack4(self.w, 32); back = unpack4(p, s, 32)
        b = self.w.reshape(24, 3, 32); i = np.abs(b).argmax(-1)
        got = np.take_along_axis(back.reshape(24, 3, 32), i[..., None], -1)[..., 0]; want = np.take_along_axis(b, i[..., None], -1)[..., 0]
        np.testing.assert_allclose(got, want, rtol=1e-5)

    def test_training_and_export_use_the_same_grid(self):
        try: import torch
        except ImportError: self.skipTest('torch not installed')
        p, s = pack4(self.w, 32)
        np.testing.assert_allclose(fq_torch(torch.from_numpy(self.w), 32).numpy(), unpack4(p, s, 32), atol=1e-6)

    def test_rejects_widths_that_do_not_divide(self):
        with self.assertRaises(AssertionError): pack4(np.zeros((2, 50), np.float32), 32)


if __name__ == '__main__': unittest.main()
