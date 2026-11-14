"""The 4-bit rounding rule, in one place so training (02_train.py --stage qat) and export (03_export.py) can never disagree.

Weights are split into blocks of `blk` values along the input dimension. Each block has one scale. The block's signed extreme (largest
absolute value, keeping its sign) maps to the integer -8, so scale = extreme / -8, and every weight becomes round(w / scale) clipped to
[-8, 7]. Stored as unsigned nibbles (value + 8), two per byte, first value in the low nibble: this is the layout of ONNX Runtime's MatMulNBits
with its default zero point of 8."""
import numpy as np


def _scales(blocks):
    ext = np.take_along_axis(blocks, np.abs(blocks).argmax(axis=-1)[..., None], axis=-1)
    sc = ext / -8.0
    return np.where(np.abs(sc) < 1e-8, 1e-8, sc).astype(np.float32)


def pack4(M, blk):
    """[rows, cols] float -> (uint8 [rows, cols/blk, blk/2], float32 scales [rows, cols/blk])."""
    r, c = M.shape
    assert c % blk == 0, f'width {c} is not a multiple of the block size {blk}'
    b = M.reshape(r, c // blk, blk).astype(np.float32); sc = _scales(b)
    q = (np.clip(np.round(b / sc), -8, 7) + 8).astype(np.uint8)
    return (q[:, :, 0::2] | (q[:, :, 1::2] << 4)).astype(np.uint8), sc.reshape(r, c // blk)


def unpack4(packed, scales, blk):
    """Inverse of pack4: back to float values on the 4-bit grid."""
    lo, hi = (packed & 15).astype(np.float32), (packed >> 4).astype(np.float32)
    q = np.stack([lo, hi], axis=-1).reshape(packed.shape[0], packed.shape[1], blk) - 8
    return (q * scales[..., None]).reshape(packed.shape[0], -1)


def fq_torch(w, blk):
    """Training-time version: the same rounding applied to a torch tensor, with a straight-through gradient."""
    import torch
    sh = w.shape; b = w.reshape(-1, sh[-1] // blk, blk)
    ext = b.gather(-1, b.abs().argmax(-1, keepdim=True)); sc = ext / -8.0
    sc = torch.where(sc.abs() < 1e-8, torch.full_like(sc, 1e-8), sc)
    q = torch.clamp(torch.round(b / sc), -8, 7) * sc
    return w + (q - b).reshape(sh).detach()
