"""
Through-plane reconstruction (the "Through-plane reconstruction" display
algorithm, from the zero_shot_tumor version of mpr_core.py).

Only the slice axis is refined; the in-plane axes are never resampled. No
learned model — every value comes from the series' own slices:

1. cubic B-spline interpolation between the acquired slices (Unser 1999);
2. slice-profile consistency (iterative back-projection, Irani & Peleg
   1991; Greenspan et al. 2002 for MRI): each acquired slice is the average
   of the tissue over its slice thickness, so the fine slices are corrected
   until their average over each slice's thickness reproduces that slice.
   The limit of the iteration is computed directly (a small banded system
   per column). Only for contiguous slices (thickness >= SR_MIN_COVERAGE x
   spacing): with gaps part of the tissue was never measured, and the step
   gained nothing while it raised banding in a MIP;
3. an overshoot limit: every value stays within the range of the nearest
   acquired slices (+10 % of that range) and within the series' min..max.

Against each patient's own 0.5 mm sagittal acquisition this rebuilt the
1 mm axial / coronal reformats of data_AIIMS 0.1-1.2 dB more accurately than
B-spline (all 8 series tested). Registration-based slice interpolation was
also tried and dropped (nothing gained on 1 mm slices, noise streaks on
5.5 mm FLAIR).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

SR_OVERSHOOT_FRAC = 0.1       # allowed overshoot beyond the nearest acquired slices, as a fraction of their range
SR_MIN_COVERAGE = 0.95        # slice-profile step only if slice thickness >= this fraction of the slice spacing
SR_WORKERS = max(1, min(16, os.cpu_count() or 1))


def _mirror(j: int, n: int) -> int:
    """Index j reflected into 0..n-1 without repeating the edge slice (scipy's 'mirror')."""
    period = 2 * (n - 1)
    j = abs(j) % period
    return j if j < n else period - j


def _bspline_weights(t: float):
    """Cubic B-spline basis for the 4 coefficients around a point a fraction t past slice i."""
    return ((1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t ** 2 + 4) / 6,
            (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6, t ** 3 / 6)


def _bspline_upsample(stack: np.ndarray, factor: int, workers: int = SR_WORKERS) -> np.ndarray:
    """Cubic B-spline interpolation along axis 0 only onto ``factor``-times
    finer slices (fine slice i*factor is acquired slice i); the same as
    sitk.sitkBSpline along that axis, without touching the other two."""
    from scipy.ndimage import spline_filter1d

    n = stack.shape[0]
    coef = spline_filter1d(stack, order=3, axis=0, mode="mirror", output=np.float32)
    out = np.empty(((n - 1) * factor + 1,) + stack.shape[1:], np.float32)

    def one(f):
        i = min(f // factor, n - 1)
        w = _bspline_weights((f - i * factor) / factor)
        acc = w[0] * coef[_mirror(i - 1, n)]
        for m in (1, 2, 3):
            acc += w[m] * coef[_mirror(i - 1 + m, n)]
        out[f] = acc

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, range(out.shape[0])))
    return out


def _profile_taps(width: float) -> np.ndarray:
    """Box slice profile ``width`` fine slices wide, centred, fractional end taps, sums to 1."""
    half = width / 2
    m = int(np.ceil(half - 0.5))
    taps = np.ones(2 * m + 1)
    if m > 0:
        taps[0] = taps[-1] = half - (m - 0.5)
    return taps / taps.sum()


def _match_slice_profile(out: np.ndarray, stack: np.ndarray, factor: int, width: float,
                         workers: int = SR_WORKERS) -> None:
    """Correct ``out`` so its average over each acquired slice's thickness
    (a box ``width`` fine slices wide) reproduces that slice exactly.

    Iterative back-projection converges to out + U (D H U)^-1 (stack - D H out)
    (H slice profile, D sampling at the acquired positions, U B-spline
    interpolation, so the correction leaves no stripes); D H U is a small
    n x n matrix, so the limit is computed directly. Its smallest gain is
    0.5-0.7, so noise along the slice axis grows at most 2x.
    """
    n, nf = stack.shape[0], out.shape[0]
    taps = _profile_taps(width)
    half = len(taps) // 2
    rows = [np.clip(np.arange(n) * factor + o, 0, nf - 1) for o in range(-half, half + 1)]

    def profile_at_slices(fine):
        acc = np.zeros((n,) + fine.shape[1:], np.float32)
        for r, wgt in zip(rows, taps):
            acc += np.float32(wgt) * fine[r]
        return acc

    dhu = profile_at_slices(_bspline_upsample(np.eye(n, dtype=np.float32), factor, workers=1))
    resid = stack - profile_at_slices(out)
    corr = np.linalg.solve(dhu, resid.reshape(n, -1)).reshape(resid.shape).astype(np.float32)
    del resid
    out += _bspline_upsample(corr, factor, workers)


def _limit_overshoot(out: np.ndarray, stack: np.ndarray, factor: int, workers: int = SR_WORKERS) -> None:
    """Keep every fine slice within the range of its nearest acquired slices
    (2 either side, 1 at an acquired position) widened by SR_OVERSHOOT_FRAC,
    and within the series' [min, max]. A hard limit cost 0.9 dB because it
    also cuts real detail (a small vessel peaking between two slices)."""
    n = stack.shape[0]
    g_lo, g_hi = float(stack.min()), float(stack.max())

    def one(f):
        i = min(f // factor, n - 1)
        last = i + 1 if f == i * factor else i + 2
        near = stack[max(i - 1, 0): min(last, n - 1) + 1]
        lo, hi = near.min(axis=0), near.max(axis=0)
        margin = np.float32(SR_OVERSHOOT_FRAC) * (hi - lo)
        np.clip(out[f], np.maximum(lo - margin, g_lo), np.minimum(hi + margin, g_hi), out=out[f])

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, range(out.shape[0])))


def reconstruct_through_plane(array: np.ndarray, axis: int, factor: int,
                              profile_width: Optional[float]) -> np.ndarray:
    """``array`` with ``factor``-times finer slices along ``axis`` (the
    acquisition axis), the other two axes untouched: B-spline, then the
    slice-profile step if ``profile_width`` (slice thickness in fine slices)
    is > 1, then the overshoot limit. Float32, (n - 1) * factor + 1 slices."""
    stack = np.ascontiguousarray(np.moveaxis(array, axis, 0), dtype=np.float32)
    out = _bspline_upsample(stack, factor)
    if profile_width is not None and profile_width > 1.0:
        _match_slice_profile(out, stack, factor, profile_width)
    _limit_overshoot(out, stack, factor)
    return np.moveaxis(out, 0, axis)
