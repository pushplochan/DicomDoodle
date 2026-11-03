"""
volume_processing.py
---------------------
GPU-aware volume processing pipeline for the MRI MPR viewer:

  1. Skull stripping   (classical morphological method, always available;
                         optional HD-BET deep-learning backend if installed)
  2. Z-score normalization (with intensity clipping for robustness)
  3. 3D connected component analysis (CCA)

Each step is independently toggle-able and they compose in a fixed,
clinically-sensible order -- skull strip -> z-score -> CCA -- so every
combination the user asked for is just "which of these three are turned
on": raw, z-score alone, z-score -> CCA, skull-strip -> z-score -> CCA,
skull-strip -> CCA (no z-score), and so on. Steps that are off are simply
skipped; nothing about the module assumes all three run together.

GPU usage is automatic and optional throughout: every function accepts
use_gpu=True (the default) and transparently falls back to a NumPy/SciPy
CPU implementation if CuPy/cuCIM aren't importable, or if the GPU path
raises for any reason (out of memory, driver issue, etc). Nothing here
*requires* a GPU to run correctly -- it's purely an accelerator.

This module has no PyQt/VTK imports -- it is pure data processing, meant
to be imported as a plain module by the GUI (see mpr_gui.py's Processing
panel), so it can also be unit-tested or reused headlessly/in notebooks.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Dict

import numpy as np
from scipy import ndimage

# --------------------------------------------------------------------------- #
# GPU availability detection -- checked once at import time.
# --------------------------------------------------------------------------- #

try:
    import cupy as cp
    HAVE_CUPY = True
except Exception:
    cp = None
    HAVE_CUPY = False

try:
    from cucim.skimage.measure import label as cucim_label
    HAVE_CUCIM = True
except Exception:
    cucim_label = None
    HAVE_CUCIM = False

HAVE_GPU = HAVE_CUPY  # cuCIM is a nice-to-have on top of CuPy, not required

# HD-BET is an external CLI (a whole deep-learning skull-stripper); we only
# ever *shell out* to it if the user has it installed, never require it.
HAVE_HDBET = shutil.which("hd-bet") is not None


def gpu_status_text() -> str:
    """One-line, human-readable summary for a GUI status label."""
    if HAVE_CUPY and HAVE_CUCIM:
        return "GPU: available (CuPy + cuCIM)"
    if HAVE_CUPY:
        return "GPU: available (CuPy; cuCIM not found, using a slower custom GPU CCA kernel)"
    return "GPU: not available -- running on CPU (NumPy/SciPy)"


# --------------------------------------------------------------------------- #
# Colormaps for the 3D volume-rendering view. Each entry is a list of
# (position in [0,1], (r, g, b) in [0,1]) control points, ready to feed
# straight into a vtkColorTransferFunction.
# --------------------------------------------------------------------------- #

COLORMAPS: Dict[str, List[Tuple[float, Tuple[float, float, float]]]] = {
    "Greyscale":    [(0, (0, 0, 0)), (1, (1, 1, 1))],
    "Inverse Grey": [(0, (1, 1, 1)), (1, (0, 0, 0))],
    "Bone":         [(0, (0, 0, 0)), (.5, (.53, .53, .53)), (1, (.999, .999, .870))],
    "Hot":          [(0, (0, 0, 0)), (.33, (1, 0, 0)), (.66, (1, 1, 0)), (1, (1, 1, 1))],
    "Cool":         [(0, (0, 1, 1)), (1, (1, 0, 1))],
    "Warm":         [(0, (.2, 0, 0)), (.5, (1, .4, 0)), (1, (1, 1, .6))],
    "Viridis":      [(0, (.267, .004, .329)), (.25, (.282, .388, .604)),
                      (.5, (.129, .604, .529)), (.75, (.482, .824, .318)), (1, (.992, .906, .145))],
    "Plasma":       [(0, (.050, .030, .527)), (.25, (.459, .027, .741)),
                      (.5, (.796, .023, .549)), (.75, (.973, .463, .173)), (1, (.940, .975, .131))],
    "Inferno":      [(0, (0, 0, .016)), (.25, (.279, .014, .416)),
                      (.5, (.588, .109, .140)), (.75, (.900, .384, .106)), (1, (.988, .998, .644))],
    "Magma":        [(0, (0, 0, .016)), (.25, (.270, .012, .427)),
                      (.5, (.558, .108, .170)), (.75, (.872, .395, .365)), (1, (.987, .991, .750))],
    "Cyan Glow":    [(0, (0, 0, 0)), (.5, (0, .4, .5)), (1, (0, 1, 1))],
    "Rainbow":      [(0, (0, 0, 1)), (.25, (0, 1, 1)), (.5, (0, 1, 0)), (.75, (1, 1, 0)), (1, (1, 0, 0))],
    "Diverge B-R":  [(0, (0, .2, .8)), (.5, (1, 1, 1)), (1, (.8, 0, 0))],
    "Ice Fire":     [(0, (0, 0, .3)), (.25, (0, .5, 1)), (.5, (1, 1, 1)), (.75, (1, .4, 0)), (1, (.4, 0, 0))],
    "Copper":       [(0, (0, 0, 0)), (.5, (.5, .3, .1)), (1, (1, .7, .4))],
}

DEFAULT_COLORMAP = "Bone"


# --------------------------------------------------------------------------- #
# Step parameter dataclasses
# --------------------------------------------------------------------------- #

class SkullStripMethod(Enum):
    CLASSICAL = "Classical (threshold + morphology, no extra install)"
    HD_BET = "HD-BET (deep learning, most accurate, requires separate install)"


@dataclass
class SkullStripParams:
    enabled: bool = False
    method: SkullStripMethod = SkullStripMethod.CLASSICAL
    # Classical method only:
    use_otsu: bool = True                # robust auto threshold (recommended); see _otsu_threshold
    threshold_percentile: float = 5.0    # only used if use_otsu is False (manual override)
    peel_mm: float = 10.0                # erode this many mm inward to peel scalp+skull+CSF


@dataclass
class ZScoreParams:
    enabled: bool = False
    clip_low_pct: float = 1.0    # intensities below this percentile are clipped up to it
    clip_high_pct: float = 99.0  # intensities above this percentile are clipped down to it
    background_pct: float = 1.0  # percentile used to separate background/foreground for the stats


@dataclass
class CCAParams:
    enabled: bool = False
    threshold: float = 0.5
    min_size_voxels: int = 100
    connectivity: int = 1   # 1 = 6-connected, 2 = 18-connected, 3 = 26-connected (matches scipy)


@dataclass
class PipelineConfig:
    """Which steps are on, and their parameters. Steps always apply in this
    fixed order when enabled: skull_strip -> zscore -> cca."""
    skull_strip: SkullStripParams = field(default_factory=SkullStripParams)
    zscore: ZScoreParams = field(default_factory=ZScoreParams)
    cca: CCAParams = field(default_factory=CCAParams)
    use_gpu: bool = True

    def active_steps(self) -> List[str]:
        steps = []
        if self.skull_strip.enabled:
            steps.append("Skull Strip")
        if self.zscore.enabled:
            steps.append("Z-Score")
        if self.cca.enabled:
            steps.append("CCA")
        return steps


@dataclass
class StepReport:
    name: str
    used_gpu: bool
    seconds: float
    detail: str


@dataclass
class PipelineResult:
    data: np.ndarray
    reports: List[StepReport] = field(default_factory=list)

    def summary_text(self) -> str:
        if not self.reports:
            return "No processing applied (showing resampled data as-is)."
        parts = []
        for r in self.reports:
            gpu_tag = "GPU" if r.used_gpu else "CPU"
            parts.append(f"{r.name} [{gpu_tag}, {r.seconds:.2f}s]: {r.detail}")
        return "  ->  ".join(parts)


# --------------------------------------------------------------------------- #
# Z-score normalization
# --------------------------------------------------------------------------- #

def zscore_normalize(volume: np.ndarray, params: ZScoreParams,
                      use_gpu: bool = True) -> Tuple[np.ndarray, StepReport]:
    """
    Adaptive Z-score normalization: Z = (X - mean_foreground) / std_foreground,
    computed only over foreground voxels (background/air is excluded from the
    statistics so a large air region doesn't drag the mean/std around).

    Intensity clipping: before computing statistics, the volume is clipped to
    the [clip_low_pct, clip_high_pct] percentile range. This is the
    "intensity clipping for normalization" step -- it keeps a handful of
    extreme outlier voxels (scanner artifacts, hot pixels) from skewing the
    mean/std and therefore the whole normalized range.
    """
    t0 = time.time()
    xp = cp if (use_gpu and HAVE_CUPY) else np
    used_gpu = xp is cp

    data = xp.asarray(volume, dtype=xp.float32)

    lo = float(xp.percentile(data, params.clip_low_pct))
    hi = float(xp.percentile(data, params.clip_high_pct))
    if hi <= lo:
        hi = lo + 1e-3
    data = xp.clip(data, lo, hi)

    bg_cutoff = xp.percentile(data, params.background_pct)
    foreground_mask = data > bg_cutoff
    if not bool(xp.any(foreground_mask)):
        foreground = data.ravel()
    else:
        foreground = data[foreground_mask]

    mean_fg = float(xp.mean(foreground))
    std_fg = float(xp.std(foreground))

    if std_fg <= 1e-6:
        z = data - mean_fg
    else:
        z = (data - mean_fg) / std_fg
        z = xp.where(foreground_mask, z, xp.min(z))

    result = xp.asnumpy(z) if used_gpu else np.asarray(z)
    detail = (f"clipped to [{lo:.1f},{hi:.1f}], foreground mean={mean_fg:.2f} "
              f"std={std_fg:.2f}, output range=[{result.min():.2f},{result.max():.2f}]")
    return result.astype(np.float32), StepReport("Z-Score", used_gpu, time.time() - t0, detail)


# --------------------------------------------------------------------------- #
# 3D Connected Component Analysis
# --------------------------------------------------------------------------- #

def _gpu_connected_components_3d(binary_mask_gpu, connectivity: int):
    """Custom iterative GPU label-propagation CCA, used only when CuPy is
    available but cuCIM isn't. Slower than cuCIM but needs nothing beyond
    CuPy itself. Not needed at all when cuCIM is installed (see
    connected_component_analysis_3d)."""
    labeled = cp.zeros(binary_mask_gpu.shape, dtype=cp.int32)
    labeled[binary_mask_gpu] = cp.arange(1, int(cp.sum(binary_mask_gpu)) + 1, dtype=cp.int32)

    if connectivity == 1:
        offsets = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    elif connectivity == 2:
        offsets = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
                   if (i, j, k) != (0, 0, 0) and abs(i) + abs(j) + abs(k) <= 2]
    else:
        offsets = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
                   if (i, j, k) != (0, 0, 0)]

    for _ in range(200):
        changed = False
        for dz, dy, dx in offsets:
            shifted = cp.roll(labeled, shift=(dz, dy, dx), axis=(0, 1, 2))
            if dz > 0: shifted[:dz, :, :] = 0
            elif dz < 0: shifted[dz:, :, :] = 0
            if dy > 0: shifted[:, :dy, :] = 0
            elif dy < 0: shifted[:, dy:, :] = 0
            if dx > 0: shifted[:, :, :dx] = 0
            elif dx < 0: shifted[:, :, dx:] = 0

            mask = (labeled > 0) & (shifted > 0) & (shifted < labeled)
            if bool(cp.any(mask)):
                labeled[mask] = shifted[mask]
                changed = True
        if not changed:
            break

    unique_labels = cp.unique(labeled[labeled > 0])
    num_features = int(len(unique_labels))
    if num_features > 0:
        remap = cp.zeros(int(unique_labels.max()) + 1, dtype=cp.int32)
        remap[unique_labels] = cp.arange(1, num_features + 1, dtype=cp.int32)
        labeled = remap[labeled]
    return labeled, num_features


def connected_component_analysis_3d(volume: np.ndarray, params: CCAParams,
                                     use_gpu: bool = True
                                     ) -> Tuple[np.ndarray, np.ndarray, Dict, StepReport]:
    """
    Binarize `volume` at `params.threshold`, find connected components in
    3D, and zero out every component smaller than `params.min_size_voxels`.

    Returns (filtered_array, labeled_array, component_sizes, report).
    filtered_array keeps the ORIGINAL intensity values wherever a
    large-enough component was found, and is 0 elsewhere -- it is not just
    a binary mask, so whatever ran before this step (e.g. Z-score output)
    is what you still see in the surviving regions.

    GPU path: tries cuCIM's `label` first (fast); if cuCIM isn't
    installed but CuPy is, falls back to a slower custom GPU label-
    propagation kernel; if neither is available, uses scipy.ndimage.label
    on the CPU. All three give the same result, just at different speeds.
    """
    t0 = time.time()
    binary_mask = volume > params.threshold
    used_gpu = False
    backend = "CPU (scipy.ndimage)"

    labeled_array = None
    num_features = 0

    if use_gpu and HAVE_CUPY:
        try:
            mask_gpu = cp.asarray(binary_mask)
            if HAVE_CUCIM:
                labeled_gpu, num_features = cucim_label(mask_gpu, connectivity=params.connectivity,
                                                          return_num=True)
                backend = "GPU (cuCIM)"
            else:
                labeled_gpu, num_features = _gpu_connected_components_3d(mask_gpu, params.connectivity)
                backend = "GPU (custom CuPy kernel)"
            labeled_array = cp.asnumpy(labeled_gpu)
            used_gpu = True
        except Exception as exc:
            backend = f"CPU fallback (GPU path raised: {exc})"
            labeled_array = None

    if labeled_array is None:
        structure = ndimage.generate_binary_structure(3, params.connectivity)
        labeled_array, num_features = ndimage.label(binary_mask, structure=structure)

    component_sizes: Dict[int, int] = {}
    if num_features > 0:
        sizes = np.bincount(labeled_array.ravel())
        for comp_id in range(1, len(sizes)):
            if sizes[comp_id] > 0:
                component_sizes[comp_id] = int(sizes[comp_id])

    keep_ids = [cid for cid, sz in component_sizes.items() if sz >= params.min_size_voxels]
    keep_mask = np.isin(labeled_array, keep_ids) if keep_ids else np.zeros_like(binary_mask)
    filtered = np.where(keep_mask, volume, 0).astype(np.float32)

    detail = (f"{num_features} components found, kept {len(keep_ids)} with "
              f">= {params.min_size_voxels} voxels ({backend})")
    report = StepReport("CCA", used_gpu, time.time() - t0, detail)
    return filtered, labeled_array, component_sizes, report


# --------------------------------------------------------------------------- #
# Skull stripping
# --------------------------------------------------------------------------- #

def _otsu_threshold(data: np.ndarray, nbins: int = 256) -> float:
    """
    Classic Otsu's method: find the intensity threshold that best splits a
    histogram into two classes (background / foreground) by maximizing
    between-class variance. Used instead of a fixed percentile for the
    head-vs-background split, because a fixed percentile can accidentally
    land inside a real tissue intensity band rather than in the gap
    between background and tissue (verified against a synthetic
    concentric-shell head phantom during development -- a percentile-based
    threshold silently misclassified the skull as background because skull
    voxels made up a large, contiguous slice of the intensity histogram).
    Otsu instead looks for the histogram *valley*, which is robust to how
    much of the volume each tissue type occupies.
    """
    finite = data[np.isfinite(data)]
    hist, bin_edges = np.histogram(finite, bins=nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return float(np.min(data))

    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    cum_intensity = np.cumsum(hist * bin_centers)
    total_intensity = cum_intensity[-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_bg = cum_intensity / weight_bg
        mean_fg = (total_intensity - cum_intensity) / weight_fg
        between_class_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

    between_class_var = np.nan_to_num(between_class_var, nan=-1.0)
    best_idx = int(np.argmax(between_class_var))
    return float(bin_centers[best_idx])


def skull_strip_classical(volume: np.ndarray, spacing_zyx: Tuple[float, float, float],
                           params: SkullStripParams, use_gpu: bool = True
                           ) -> Tuple[np.ndarray, np.ndarray, StepReport]:
    """
    Classical morphological skull stripping -- no deep-learning model or
    extra install required, but noticeably less precise than HD-BET/BET.
    Good default so the feature works everywhere; use HD-BET (if you have
    it installed) when you need production-grade accuracy.

    Steps:
      1. Threshold at `threshold_percentile` of the nonzero voxels to
         separate "head" from background air.
      2. Fill internal holes and keep only the largest connected
         component, so we start from one clean, solid head-shaped blob.
      3. Erode that blob inward by `peel_mm` millimetres (converted to
         voxels using the actual voxel spacing on each axis, so this is a
         physical distance, not a fixed voxel count) -- this peels off the
         scalp + skull + CSF rim, which can't be separated from brain
         tissue by intensity alone but does sit at the outer few
         millimetres of the head.
      4. Keep only the largest remaining component (drops fragments the
         erosion may have disconnected).

    Returns (stripped_array, brain_mask, report). stripped_array is the
    original intensities inside the mask and 0 outside it.
    """
    t0 = time.time()
    data = volume.astype(np.float32)

    nonzero = data[data > 0]
    if nonzero.size == 0:
        mask = np.zeros_like(data, dtype=bool)
        return data * 0, mask, StepReport("Skull Strip", False, time.time() - t0,
                                           "volume is all-zero, nothing to strip")

    if params.use_otsu:
        thresh = _otsu_threshold(data)
        thresh_label = "Otsu"
    else:
        thresh = float(np.percentile(nonzero, params.threshold_percentile))
        thresh_label = f"p{params.threshold_percentile:.0f}"
    head_mask = data > thresh

    # NOTE: deliberately no fill_holes here yet. If the threshold correctly
    # excludes a dim inner shell (skull, which sits topologically between
    # excluded low-signal bone and the brighter scalp outside it),
    # fill_holes would treat that excluded shell as an enclosed "hole" in
    # the mask and fill it back in -- silently undoing the very exclusion
    # we wanted. Isolate the main blob first; only close small internal
    # gaps once we're safely working within an already-isolated region.
    head_mask = _largest_component(head_mask)

    voxels_per_mm = [1.0 / max(s, 1e-6) for s in spacing_zyx]
    erosion_iters = max(1, int(round(params.peel_mm * min(voxels_per_mm))))
    structure = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(head_mask, structure=structure, iterations=erosion_iters)

    brain_mask = _largest_component(eroded)
    # Safe to fill holes now: any gaps left are small internal dropouts
    # (e.g. a low-signal ventricle) inside the already-isolated brain blob,
    # not an outer structure we deliberately excluded.
    brain_mask = ndimage.binary_fill_holes(brain_mask)
    stripped = np.where(brain_mask, data, 0).astype(np.float32)

    detail = (f"threshold={thresh:.1f} ({thresh_label}), "
              f"peeled {params.peel_mm:.1f}mm ({erosion_iters} erosion iters), "
              f"kept {int(brain_mask.sum()):,}/{data.size:,} voxels")
    return stripped, brain_mask, StepReport("Skull Strip (classical)", False, time.time() - t0, detail)


def _largest_component(binary_mask: np.ndarray) -> np.ndarray:
    structure = ndimage.generate_binary_structure(3, 1)
    labeled, num = ndimage.label(binary_mask, structure=structure)
    if num == 0:
        return binary_mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    return labeled == largest


def skull_strip_hdbet(sitk_image, use_gpu: bool = True) -> Tuple[np.ndarray, StepReport]:
    """
    Skull-strip via the HD-BET deep-learning tool (https://github.com/
    MIC-DKFZ/HD-BET), if it's installed as a CLI on this machine. This is
    the same tool used in the reference vessel-explorer script, called the
    same way: write the volume to a temp NIfTI, invoke `hd-bet`, read the
    brain-only result back.

    Raises RuntimeError with a clear message if HD-BET isn't installed or
    the subprocess call fails -- callers should catch this and fall back
    to skull_strip_classical(), or surface it to the user.
    """
    import SimpleITK as sitk

    if not HAVE_HDBET:
        raise RuntimeError(
            "HD-BET is not installed (the 'hd-bet' command was not found on PATH). "
            "Install it from https://github.com/MIC-DKFZ/HD-BET, or use the "
            "Classical skull-stripping method instead.")

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "in.nii.gz")
        out_path = os.path.join(tmpdir, "out.nii.gz")
        sitk.WriteImage(sitk_image, in_path)

        device = "cuda" if (use_gpu and HAVE_CUPY) else "cpu"
        cmd = ["hd-bet", "-i", in_path, "-o", out_path, "-device", device]
        if device == "cpu":
            cmd.append("--disable_tta")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"HD-BET failed:\n{result.stderr}")

        stripped = sitk.GetArrayFromImage(sitk.ReadImage(out_path)).astype(np.float32)

    detail = f"HD-BET ({device}) skull strip complete"
    return stripped, StepReport("Skull Strip (HD-BET)", device == "cuda", time.time() - t0, detail)


# --------------------------------------------------------------------------- #
# Pipeline: compose whichever steps are enabled, in a fixed sensible order
# --------------------------------------------------------------------------- #

def run_pipeline(volume: np.ndarray, spacing_zyx: Tuple[float, float, float],
                  config: PipelineConfig, sitk_image_for_hdbet=None) -> PipelineResult:
    """
    Apply skull_strip -> zscore -> cca, skipping any step whose `.enabled`
    is False, in that fixed order. Passing a config with everything
    disabled returns the input unchanged (with an empty report list) --
    this is how "no processing" and "some subset of the three steps" are
    both represented by the same function.
    """
    data = np.asarray(volume, dtype=np.float32)
    reports: List[StepReport] = []

    if config.skull_strip.enabled:
        if config.skull_strip.method is SkullStripMethod.HD_BET:
            try:
                if sitk_image_for_hdbet is None:
                    raise RuntimeError("HD-BET needs the original SimpleITK image; none was provided.")
                data, report = skull_strip_hdbet(sitk_image_for_hdbet, use_gpu=config.use_gpu)
            except Exception as exc:
                # Fail soft to the classical method rather than aborting the
                # whole pipeline -- HD-BET has real-world install/runtime
                # fragility (CUDA driver mismatches, missing weights, etc.)
                data, _mask, report = skull_strip_classical(
                    data, spacing_zyx, config.skull_strip, use_gpu=config.use_gpu)
                report.detail = f"HD-BET unavailable/failed ({exc}); used classical instead. " + report.detail
        else:
            data, _mask, report = skull_strip_classical(
                data, spacing_zyx, config.skull_strip, use_gpu=config.use_gpu)
        reports.append(report)

    if config.zscore.enabled:
        data, report = zscore_normalize(data, config.zscore, use_gpu=config.use_gpu)
        reports.append(report)

    if config.cca.enabled:
        data, _labeled, _sizes, report = connected_component_analysis_3d(
            data, config.cca, use_gpu=config.use_gpu)
        reports.append(report)

    return PipelineResult(data=data, reports=reports)
