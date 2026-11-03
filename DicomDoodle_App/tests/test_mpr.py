"""
Tests for the Multiplanar view's core (dicomdoodle.core.mpr) on small
synthetic series: which files make a volume, filling traced outlines,
through-plane reconstruction, aspect-aware outlines, the Detect pipeline,
the main window's "Find tumour range" and the Multiplanar & 3D Viewer's 3D
model and skull stripping (with stand-in models and a stand-in for HD-BET,
so no AI weights are needed). The last tests need a display (run under
xvfb-run) and are skipped without one.

    python -m pytest tests          (or simply: python tests/test_mpr.py)
"""
import os
import shutil
import sys
import tempfile
import time
import types

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dicomdoodle.core import mpr  # noqa: E402
from dicomdoodle.core.mpr import segmentation as mpr_seg  # noqa: E402
from dicomdoodle.core.mpr import tumour_detect as td  # noqa: E402
from dicomdoodle.core.mpr import volume as mpr_volume  # noqa: E402


def write_slice(path, series, k, *, z=None, echo=None, sop=None, iop=(1, 0, 0, 0, 1, 0), size=32,
                pixels=None, frame=None, thickness=1.0):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = sop or generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = "1.2.3.4", series
    ds.FrameOfReferenceUID = frame or "1.2.3.5"
    ds.Modality, ds.PatientName, ds.PatientID, ds.SeriesDescription = "MR", "Test^Patient", "T1", "T1 test"
    ds.InstanceNumber = k + 1
    ds.ImagePositionPatient = [0.0, 0.0, float(k if z is None else z)]
    ds.ImageOrientationPatient = list(iop)
    ds.PixelSpacing, ds.SliceThickness = [1.0, 1.0], thickness
    if echo is not None:
        ds.EchoNumbers = echo
    ds.Rows = ds.Columns = size
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    if pixels is None:
        pixels = np.full((size, size), 100 + 10 * k, np.uint16)
    ds.PixelData = np.asarray(pixels, np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)


def sphere_volume(shape=(40, 64, 64), centre=(20, 30, 34), radius=7):
    zz, yy, xx = np.indices(shape)
    return ((zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2) <= radius ** 2


def write_sphere_series(folder, shape=(40, 64, 64)):
    """Axial slices of a bright sphere on a dark background (1 mm voxels)."""
    sphere = sphere_volume(shape)
    series = generate_uid()
    for k in range(shape[0]):
        write_slice(os.path.join(folder, f"img_{k:03d}.dcm"), series, k, size=shape[1],
                    pixels=np.where(sphere[k], 1000, 100))
    return sphere


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def test_find_series_splits_one_series_into_volumes():
    folder = tempfile.mkdtemp()
    try:
        echoes, gappy = generate_uid(), generate_uid()
        for k in range(5):                                   # one series, two echoes at the same positions
            for echo in (1, 2):
                write_slice(os.path.join(folder, f"e{echo}_{k}.dcm"), echoes, 2 * k + echo - 1, z=k, echo=echo)
        write_slice(os.path.join(folder, "localiser.dcm"), echoes, 20, iop=(0, 1, 0, 0, 0, -1))
        shutil.copy(os.path.join(folder, "e1_0.dcm"), os.path.join(folder, "copy_of_e1_0.dcm"))
        for k in (0, 1, 2, 4, 5):                            # a series with one slice missing
            write_slice(os.path.join(folder, f"gap_{k}" + ("" if k == 4 else ".dcm")), gappy, k, z=2.0 * k)

        volumes = {v.key: v for v in mpr.find_series(folder)}
        assert len(volumes) == 3, [v.label for v in volumes.values()]
        echo_volumes = [v for v in volumes.values() if v.key.startswith(echoes)]
        assert sorted(len(v.files) for v in echo_volumes) == [5, 5]
        assert {"echo 1", "echo 2"} <= {n for v in echo_volumes for n in v.notes}
        notes = " ".join(n for v in echo_volumes for n in v.notes)
        assert "left out" in notes and "duplicate" in notes

        gap = volumes[gappy]
        assert len(gap.files) == 5 and gap.dims[2] == 6           # the file without .dcm is found by its marker
        assert gap.slots == [0, 1, 2, 4, 5]
        image = mpr.load_dicom_series(folder, gappy)
        assert image.GetSize()[2] == 6 and abs(image.GetSpacing()[2] - 2.0) < 1e-6
        assert "missing slice" in mpr.get_series_notes(image)
        array = mpr.build_volume(image, mpr.InterpMethod.NONE_NATIVE).array
        assert array[3].max() == array.min()                      # the blank slice keeps the others in place
        assert array[4].max() == 140                              # slice k=4 at its true position
        assert mpr.series_files(folder, gappy)[3].endswith("gap_4")
    finally:
        shutil.rmtree(folder, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Annotations
# --------------------------------------------------------------------------- #

def test_outline_traced_in_open_pieces_is_filled_as_a_whole():
    """The AIIMS case: one tumour outline saved as two open pieces (two segments)."""
    from dicomdoodle.core.seg_io import HAND_DRAWN, write_segmentation
    folder = tempfile.mkdtemp()
    try:
        series = generate_uid()
        for k in range(3):
            write_slice(os.path.join(folder, f"img_{k}.dcm"), series, k, size=48)
        yy, xx = np.indices((48, 48))
        ring = np.abs(np.hypot(yy - 24, xx - 24) - 12) < 0.75     # a thin traced circle
        pieces = np.zeros((48, 48), np.uint8)
        pieces[ring & (xx < 24)] = 1                              # left half: "Tumor & Brain Boundary"
        pieces[ring & (xx >= 24)] = 2                             # right half: "Tumor & Dura Boundary"
        write_segmentation(pieces, os.path.join(folder, "img_1.dcm"), os.path.join(folder, "img_1_seg.dcm"), 1,
                           labels={1: "Tumor & Brain Boundary", 2: "Tumor & Dura Boundary"}, method=HAND_DRAWN)

        image = mpr.load_dicom_series(folder)
        filled = mpr.load_seg_as_labelmap(mpr.find_series_seg_files(image, folder), image)
        region = filled.array[1]
        assert abs(int((region > 0).sum()) - np.pi * 12.5 ** 2) < 60, int((region > 0).sum())
        assert set(np.unique(region)) == {0, 1}                   # one label: the lowest of its pieces
        assert [s.label for s in filled.segments.values()] == ["Tumor & Brain Boundary"]

        raw = mpr.load_seg_as_labelmap(mpr.find_series_seg_files(image, folder), image, fill_contours=False)
        assert (raw.array[1] > 0).sum() == ring.sum()             # unfilled: just the traced line
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_gaps_in_a_traced_outline_are_bridged():
    yy, xx = np.indices((64, 64))
    ring = np.abs(np.hypot(yy - 32, xx - 32) - 15) < 0.75
    ring[28:37, 40:] = False                                       # an opening of ~9 px on the right
    assert ndimage.binary_fill_holes(ring).sum() == ring.sum()      # a plain hole fill fills nothing
    filled = mpr_seg._fill_traced_outline(ring)
    assert abs(int(filled.sum()) - np.pi * 15.5 ** 2) < 120, int(filled.sum())
    scribble = np.zeros((64, 64), bool)
    scribble[10, 10:20] = True                                     # a short line encloses nothing
    assert mpr_seg._fill_traced_outline(scribble).sum() == scribble.sum()


def test_boundary_outline_follows_display_aspect():
    label = np.zeros((40, 40), np.uint8)
    label[10:30, 10:30] = 1
    segments = {1: mpr.SegmentInfo(1, "x", (255, 0, 0))}
    gray = np.zeros((40, 40, 3), np.uint8)
    square = mpr.blend_label_boundary(gray, label, segments, thickness=2)
    tall_rows = mpr.blend_label_boundary(gray, label, segments, thickness=2, aspect=4.0)
    red = lambda rgb: rgb[..., 0] == 255  # noqa: E731
    assert red(square)[9:12, 20].sum() == 3                         # 3x3 growth all round, as before
    assert red(tall_rows)[9:12, 20].sum() == 1                      # rows drawn 4x taller: not widened
    assert red(tall_rows)[20, 9:12].sum() == 3                      # columns still widened


# --------------------------------------------------------------------------- #
# Display algorithms
# --------------------------------------------------------------------------- #

def _image(array, spacing_xyz, thickness=None):
    import SimpleITK as sitk
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing_xyz)
    if thickness is not None:
        image.SetMetaData("slice_thickness_mm", str(thickness))
    return image


def test_through_plane_reconstruction():
    rng = np.random.default_rng(1)
    stack = (ndimage.gaussian_filter(rng.random((6, 24, 24)), 2) * 1000).astype(np.float32)

    gapped = mpr.build_volume(_image(stack, (0.5, 0.5, 2.0), thickness=1.0), mpr.InterpMethod.THROUGH_PLANE)
    assert gapped.shape == (21, 24, 24) and np.allclose(gapped.spacing_zyx, (0.5, 0.5, 0.5))
    assert np.allclose(gapped.array[::4], stack, atol=1e-2)        # thin slices with gaps: B-spline through them
    assert stack.min() - 1e-3 <= gapped.array.min() and gapped.array.max() <= stack.max() + 1e-3

    contiguous = mpr.build_volume(_image(stack, (0.5, 0.5, 2.0), thickness=2.0), mpr.InterpMethod.THROUGH_PLANE)
    assert contiguous.shape == (21, 24, 24)
    assert not np.allclose(contiguous.array, gapped.array)         # the slice-profile step ran

    iso = mpr.build_volume(_image(stack, (1.0, 1.0, 1.0)), mpr.InterpMethod.THROUGH_PLANE)
    assert np.array_equal(iso.array, stack) and iso.interp_method is mpr.InterpMethod.THROUGH_PLANE


def test_resampled_grid_stays_within_the_voxel_limit():
    saved = mpr_volume.MAX_RESAMPLED_VOXELS
    mpr_volume.MAX_RESAMPLED_VOXELS = 1.0e4
    try:
        volume = mpr.build_volume(_image(np.zeros((6, 32, 32), np.int16), (1.0, 1.0, 4.0)), mpr.InterpMethod.LINEAR)
        assert volume.array.size <= 1.15e4 and volume.spacing_zyx[0] > 1.0
    finally:
        mpr_volume.MAX_RESAMPLED_VOXELS = saved


# --------------------------------------------------------------------------- #
# Detect, with stand-in models
# --------------------------------------------------------------------------- #

class _Tensor:
    """Just enough of a torch tensor: .cpu().numpy(), len() and indexing."""

    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return _Tensor(self.a[i])


class _Result:
    def __init__(self, boxes=None, masks=None):
        self.boxes, self.masks = boxes, masks


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy, self.conf = _Tensor(xyxy), _Tensor(conf)

    def __len__(self):
        return len(self.xyxy)


class _Masks:
    def __init__(self, data):
        self.data = _Tensor(data)


class StubModels:
    """Stands in for ModelManager: the detector boxes every bright blob of
    the image it is shown, SAM returns the bright pixels inside its box (or
    the blob under its point) — so a wrong transpose or scale shows."""

    card = types.SimpleNamespace(label="Stub model card 1.0", identity="stub")
    device, ready, loaded, sam_loaded, all_loaded = "cpu", True, False, False, False

    def can(self, feature):
        return True

    def has(self, kind):
        return True

    def info(self):
        return []

    def gpu_failed(self):
        return False

    def device_name(self):
        return "CPU"

    def runs_on_cpu(self):
        return False

    def predict(self, kind, image, **kwargs):
        bright = image[..., 0] > 128
        if kind == "yolo":
            boxes = [(s[1].start, s[0].start, s[1].stop, s[0].stop) for s in ndimage.find_objects(ndimage.label(bright)[0])]
            area = np.array([(x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes], float)
            conf = 0.5 + 0.4 * area / area.max() if boxes else area     # the largest blob is the most confident
            return [_Result(boxes=_Boxes(np.array(boxes, float).reshape(-1, 4), conf))]
        mask = np.zeros_like(bright)
        if "bboxes" in kwargs:
            x0, y0, x1, y1 = (int(round(v)) for v in kwargs["bboxes"][0])
            mask[y0:y1, x0:x1] = bright[y0:y1, x0:x1]
        else:
            labels = ndimage.label(bright)[0]
            x, y = (int(v) for v in kwargs["points"][0])
            mask = (labels == labels[y, x]) & (labels[y, x] > 0)
        return [_Result(masks=_Masks(mask[None].astype(np.float32)))]


def test_detect_pipeline_finds_the_sphere():
    sphere = sphere_volume()
    array = np.where(sphere, 1000, 100).astype(np.int16)
    volume = mpr.build_volume(_image(array, (1.0, 1.0, 1.0)), mpr.InterpMethod.NONE_NATIVE)
    detector = td.TumourDetector(StubModels())
    for plane, index, pick in ((mpr.Plane.AXIAL, 20, None), (mpr.Plane.CORONAL, 30, (20, 34)),
                               (mpr.Plane.SAGITTAL, 34, (20, 30))):
        result = detector.detect(volume, plane, index, "Axial", pick=pick)
        assert mpr.dice_coefficient(result.mask, sphere) > 0.95, plane
        (z0, z1), (y0, y1), (x0, x1) = result.cuboid
        assert z0 <= 13 and z1 >= 28 and y0 <= 23 and y1 >= 38 and x0 <= 27 and x1 >= 42
        assert result.display_box(mpr.Plane.AXIAL, 20, volume.shape) is not None
        assert result.display_box(mpr.Plane.AXIAL, 0, volume.shape) is None
        assert ("under the crosshair" in result.seeds[0].how) == (pick is not None)
    seg = result.to_segmentation(volume.sitk_image)
    assert seg.segments[1].label == td.DETECT_NAME and seg.array.sum() == result.mask.sum()


def test_detect_reports_when_nothing_is_found():
    volume = mpr.build_volume(_image(np.full((8, 32, 32), 100, np.int16), (1.0, 1.0, 1.0)),
                              mpr.InterpMethod.NONE_NATIVE)
    try:
        td.TumourDetector(StubModels()).detect(volume, mpr.Plane.AXIAL, 4, "Axial")
        raise AssertionError("no NoTumourFound")
    except td.NoTumourFound as exc:
        assert "axial slice 5" in str(exc)


# --------------------------------------------------------------------------- #
# Find tumour range (main window), with stand-in models
# --------------------------------------------------------------------------- #

class _NoSam(StubModels):
    """Find tumour range uses the detector only."""

    def predict(self, kind, image, **kwargs):
        assert kind != "sam", "SAM was called"
        return super().predict(kind, image, **kwargs)


def two_spheres(shape=(40, 64, 64)):
    """The sphere (z 13-27) and a smaller one beside it on the same slices (z 16-24)."""
    return sphere_volume(shape) | sphere_volume(shape, centre=(20, 12, 12), radius=4)


def test_find_range_with_detector_boxes_only():
    array = np.where(two_spheres(), 1000, 100).astype(np.int16)
    volume = mpr.build_volume(_image(array, (1.0, 1.0, 1.0)), mpr.InterpMethod.NONE_NATIVE)
    detector = td.TumourDetector(_NoSam())
    found = detector.find_range(volume, mpr.Plane.AXIAL, 20, "Axial")
    assert found.slices == (13, 28) and found.acquisition_plane is mpr.Plane.AXIAL and not found.estimated
    assert [s.plane for s in found.seeds] == [mpr.Plane.AXIAL, mpr.Plane.CORONAL, mpr.Plane.SAGITTAL]
    assert found.extent(found.seeds[0]) is None and found.extent(found.seeds[1]) == (13, 28)
    small = detector.find_range(volume, mpr.Plane.AXIAL, 20, "Axial", hint=(6, 6, 19, 19))   # drawn around it
    assert small.slices == (16, 25) and "inside the drawn box" in small.seeds[0].how
    empty = detector.find_range(volume, mpr.Plane.AXIAL, 20, "Axial", hint=(44, 2, 60, 14))
    assert empty.seeds[0].how.startswith("the drawn box") and empty.estimated   # nothing there in any view
    try:
        detector.find_range(volume, mpr.Plane.AXIAL, 2, "Axial")
        raise AssertionError("no NoTumourFound")
    except td.NoTumourFound as exc:
        assert "axial slice 3" in str(exc)


class _BlindTo(_NoSam):
    """A detector that proposes nothing on images of the given shapes (= on chosen views)."""

    def __init__(self, *shapes):
        self.shapes = shapes

    def predict(self, kind, image, **kwargs):
        if kind == "yolo" and image.shape[:2] in self.shapes:
            return [_Result(boxes=_Boxes(np.zeros((0, 4)), np.zeros(0)))]
        return super().predict(kind, image, **kwargs)


def test_detected_box_beats_size_guess():
    zz, yy, xx = np.indices((40, 96, 96))
    flat = ((zz - 20) / 5.0) ** 2 + ((yy - 48) / 10.0) ** 2 + ((xx - 48) / 10.0) ** 2 <= 1   # z 15-25, 21 wide
    volume = mpr.build_volume(_image(np.where(flat, 1000, 100).astype(np.int16), (1.0, 1.0, 1.0)),
                              mpr.InterpMethod.NONE_NATIVE)
    sagittal_view, coronal_view = (40, 96), (96, 40)      # what the detector sees (coronal transposed)
    found = td.TumourDetector(_BlindTo(sagittal_view)).find_range(volume, mpr.Plane.AXIAL, 20, "Axial")
    coronal, sagittal = found.seeds[1:]
    assert sagittal.estimated and not coronal.estimated and not found.estimated
    assert found.extent(sagittal) == (10, 31)             # guessed from the 21 mm wide seed box: too long
    assert found.slices == found.extent(coronal) == (15, 26)
    guess = td.TumourDetector(_BlindTo(sagittal_view, coronal_view)).find_range(volume, mpr.Plane.AXIAL, 20, "Axial")
    assert guess.estimated and guess.slices == (10, 31)


def write_reversed_series(folder, spheres, n=40, size=64):
    """Slice k+1 lies at z = n-1-k (numbered from the top down) and its columns run
    towards the patient's right: the volume's order differs from the file order."""
    series = generate_uid()
    for k in range(n):
        write_slice(os.path.join(folder, f"img_{k:03d}.dcm"), series, k, z=n - 1 - k, iop=(-1, 0, 0, 0, 1, 0),
                    size=size, pixels=np.where(spheres[n - 1 - k], 1000, 100))


def test_tumour_range_in_main_window_slices():
    from dicomdoodle.core.dicom_series import DicomSeries
    from dicomdoodle.core.tumour_range import TumourRangeFinder
    folder = tempfile.mkdtemp()
    try:
        write_reversed_series(folder, two_spheres())
        finder = TumourRangeFinder(_NoSam(), DicomSeries.load(folder))
        found = finder.find(19)                               # z 20, the middle of both spheres
        assert (found.first, found.last, found.count) == (12, 26, 15) and not found.estimated   # z 27 ... 13
        assert np.allclose(found.seed_box, (27, 23, 42, 38), atol=0.6), found.seed_box
        assert sorted(v.plane for v in found.views) == ["coronal", "sagittal"]
        assert all(v.slices == (12, 26) and v.along_rows for v in found.views)
        for v in found.views:                                 # the band lies on the box, in the preview image
            assert v.image.dtype == np.uint8 and abs(v.band[0] - v.box[1]) < 1 and abs(v.band[1] - v.box[3]) < 1
        assert found.summary_text().startswith("Tumour range: slices 13–27 (15 slices) from slice 20")
        small = finder.find(19, hint=(6, 6, 19, 19))          # a box drawn on the slice, in its own pixels
        assert (small.first, small.last) == (15, 23)          # z 24 ... 16
        assert np.allclose(small.seed_box, (8, 8, 17, 17), atol=0.6), small.seed_box
        try:
            finder.find(2)
            raise AssertionError("no NoTumourFound")
        except td.NoTumourFound as exc:
            assert "slice 3" in str(exc)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Skull stripping (HD-BET itself is too slow for tests; checked by hand)
# --------------------------------------------------------------------------- #

def test_skull_strip_helpers():
    import SimpleITK as sitk
    from dicomdoodle.core import skull_strip
    saved = skull_strip.HAVE_HDBET
    try:
        skull_strip.HAVE_HDBET = False
        try:
            skull_strip.brain_mask(sitk.Image(4, 4, 4, sitk.sitkInt16))
            raise AssertionError("no SkullStripError")
        except skull_strip.SkullStripError as exc:
            assert skull_strip.INSTALL_HINT in str(exc)
    finally:
        skull_strip.HAVE_HDBET = saved
    image = np.arange(8, dtype=np.int16).reshape(2, 2, 2)
    stripped = skull_strip.apply_mask(image, image % 2)
    assert stripped.dtype == np.int16 and stripped.tolist() == [[[0, 1], [0, 3]], [[0, 5], [0, 7]]]
    folder = tempfile.mkdtemp()
    try:                                       # HD-BET's progress, read from its output
        log = os.path.join(folder, "hdbet.log")
        with open(log, "w", encoding="utf-8") as fh:
            fh.write("HD-BET on cpu\n  0%|          | 0/6 [00:00<?, ?it/s] 33%|███▎      | 2/6 [00:08<00:17,  4.3s/it]")
        assert skull_strip._tiles(log) == (2, 6)
        assert skull_strip._tiles(os.path.join(folder, "missing.log")) == (0, 0)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_hdbet_runner_in_windows_build():
    import subprocess
    from dicomdoodle.core import hdbet_runner, skull_strip
    assert skull_strip.runner_command("a.nii.gz", "m.nii.gz") == [sys.executable, skull_strip.RUNNER, "a.nii.gz", "m.nii.gz"]
    sys.frozen = True                          # as in DicomDoodle.exe (PyInstaller): no .py file to run
    try:
        assert skull_strip.runner_command("a.nii.gz", "m.nii.gz") == [sys.executable, hdbet_runner.FLAG,
                                                                      "a.nii.gz", "m.nii.gz"]
    finally:
        del sys.frozen
    main = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py")
    run = subprocess.run([sys.executable, main, hdbet_runner.FLAG], capture_output=True, text=True, timeout=60)
    assert run.returncode == 2 and "IMAGE MASK" in run.stderr     # main.py hands the option to the runner


# --------------------------------------------------------------------------- #
# Qt (needs a display: run under xvfb-run)
# --------------------------------------------------------------------------- #

def _qt_app():
    if not os.environ.get("DISPLAY"):
        return None
    from PyQt5.QtWidgets import QApplication
    from dicomdoodle.ui import theme
    app = QApplication.instance() or QApplication([])
    if theme._MANAGER is None:
        theme.install_theme_manager(app, "dark")
    return app


def _dispose(app, window):
    """Close and delete a test window now, not at interpreter exit (where Qt
    may already be gone and the teardown can crash)."""
    from PyQt5.QtCore import QCoreApplication, QEvent, QThreadPool
    window.close()
    QThreadPool.globalInstance().waitForDone()
    app.processEvents()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_guide_diagram_text_fits_at_any_dpi():
    if _qt_app() is None:
        print("skipped: no display")
        return
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QImage, QPainter
    from dicomdoodle.ui.guide import diagrams, drawing
    from dicomdoodle.ui.theme import DARK

    misfits, real_text = [], drawing.text

    def checked(p, rect, value, color, size=9.5, bold=False, align=Qt.AlignCenter):
        real_text(p, rect, value, color, size, bold, align)
        needed = p.boundingRect(rect, int(align) | Qt.TextWordWrap, value)
        if needed.width() > rect.width() + 0.5 or needed.height() > rect.height() + 0.5:
            misfits.append((p.device().logicalDpiY(), value))

    drawing.text = diagrams.text = checked
    try:
        for dpi in (96, 120, 144):
            for draw, (width, height) in diagrams.DIAGRAMS.values():
                image = QImage(width, height, QImage.Format_ARGB32)
                image.setDotsPerMeterX(round(dpi / 0.0254))
                image.setDotsPerMeterY(round(dpi / 0.0254))
                painter = QPainter(image)
                draw(painter, DARK)
                painter.end()
    finally:
        drawing.text = diagrams.text = real_text
    assert not misfits, misfits


def test_mpr_window_detect_button():
    app = _qt_app()
    if app is None:
        print("skipped: no display")
        return
    from PyQt5.QtWidgets import QFileDialog
    from dicomdoodle.ui.tools.mpr.window import MPRWindow
    folder = tempfile.mkdtemp()
    original = QFileDialog.getExistingDirectory
    try:
        sphere = write_sphere_series(folder)
        window = MPRWindow(None, folder, models=StubModels())
        QFileDialog.getExistingDirectory = staticmethod(lambda *args, **kwargs: folder)
        window.on_load_primary()
        assert window.volume is not None and window.axial_panel.detect_btn.isEnabled()
        z, y, x = window.volume.shape
        window.on_slice_changed(mpr.Plane.AXIAL, 20)
        window.axial_panel.positionPicked.emit(mpr.Plane.AXIAL, (y - 1) - 30, 34)   # click on the sphere
        window.on_detect_requested(mpr.Plane.AXIAL)
        assert window.annotation_combo.currentData() == "detected"
        assert mpr.dice_coefficient(window.detection.mask, sphere) > 0.95
        assert "under the crosshair" in window.detection.seeds[0].how
        assert window.statusBar().currentMessage().startswith("Detected tumour")
        assert window._active_label_array is not None and window._active_label_array.any()
        assert window.coronal_panel._det_rect is not None and window.coronal_panel._det_rect.get_visible()
        window.algo_combo.setCurrentIndex(window.algo_combo.findData(mpr.InterpMethod.THROUGH_PLANE))
        assert window.annotation_combo.currentData() == "detected"             # survives a display change
        _dispose(app, window)
    finally:
        QFileDialog.getExistingDirectory = original
        shutil.rmtree(folder, ignore_errors=True)


def write_sphere_annotation(folder, sphere):
    """The sphere's outline as DicomDoodle saves it, one _seg.dcm per slice it crosses."""
    from dicomdoodle.core import mask_ops
    from dicomdoodle.core.dicom_series import DicomSeries
    from dicomdoodle.core.seg_io import write_segmentation
    series = DicomSeries.load(folder)
    for i in range(len(series)):
        if sphere[i].any():
            write_segmentation(mask_ops.boundary_map(sphere[i].astype(np.uint8)), series.file(i), series.seg_path(i), i)


def test_mpr_window_3d_model_and_skull_strip():
    """The Multiplanar & 3D Viewer: the right-hand panel builds the 3D model, "Show" switches it, and one
    skull stripping serves the 3D view and the slice views (HD-BET replaced by a stand-in)."""
    app = _qt_app()
    if app is None:
        print("skipped: no display")
        return
    import SimpleITK as sitk
    from PyQt5.QtWidgets import QFileDialog
    from dicomdoodle.core import skull_strip
    from dicomdoodle.ui.tools.mpr.window import MPRWindow
    folder, runs = tempfile.mkdtemp(), []

    def brain_mask(image, progress=None):     # a ball around the centre stands in for HD-BET's brain
        runs.append(image.GetSize())
        z, y, x = sitk.GetArrayViewFromImage(image).shape
        zz, yy, xx = np.indices((z, y, x))
        mask = sitk.GetImageFromArray(((zz - z / 2) ** 2 + (yy - y / 2) ** 2 + (xx - x / 2) ** 2 <= 15 ** 2)
                                      .astype(np.uint8))
        mask.CopyInformation(image)
        return mask

    saved = QFileDialog.getExistingDirectory, skull_strip.brain_mask, skull_strip.HAVE_HDBET
    try:
        sphere = write_sphere_series(folder)
        write_sphere_annotation(folder, sphere)
        skull_strip.brain_mask, skull_strip.HAVE_HDBET = brain_mask, True
        window = MPRWindow(None, folder, tab="3d")
        assert [window.control_tabs.tabText(i) for i in range(2)] == ["Multiplanar", "3D view"]
        assert window.control_tabs.currentIndex() == 1 and window.side.series.paths()["Axial"] == folder
        window.build_model()
        volume = window.side.volume.results[0]      # the outline is saved just outside the region: a bit more
        assert window.model is not None and volume.annotated_slices == 15
        assert sphere.sum() <= volume.voxels < 1.25 * sphere.sum()
        assert window.side.metrics.metrics is not None
        show, view = window.view_combo, window.three_d_panel.view
        assert view.has_data and view.data_range[1] == 1000                    # Original MRI
        assert show.model().item(show.findData("annotated")).isEnabled()
        show.setCurrentIndex(show.findData("annotated"))
        assert view.data_range[1] > 1900                                       # the lesion brightened (x 1.99)
        show.setCurrentIndex(show.findData("skull_stripped"))
        assert window._view_mode == "skull_stripped" and view.data_range[0] == 0 and len(runs) == 1
        assert window.three_d_panel.info_label.text().strip().startswith("Skull stripped")

        QFileDialog.getExistingDirectory = staticmethod(lambda *args, **kwargs: folder)
        window.on_load_primary()
        image = window.image_combo
        image.setCurrentIndex(image.findData("skull_stripped"))
        assert len(runs) == 1                  # the same scanner space as the 3D model: no second run
        assert (window.volume.array == 0).mean() > 0.5 and window.volume.array.max() == 1000
        assert "Image: skull stripped" in window.info_banner.text()
        image.setCurrentIndex(image.findData("original"))
        assert (window.volume.array == 0).sum() == 0
        _dispose(app, window)
    finally:
        QFileDialog.getExistingDirectory, skull_strip.brain_mask, skull_strip.HAVE_HDBET = saved
        shutil.rmtree(folder, ignore_errors=True)


def test_main_window_find_tumour_range():
    app = _qt_app()
    if app is None:
        print("skipped: no display")
        return
    from dicomdoodle.settings import Settings
    from dicomdoodle.ui.label_manager import LabelManager
    from dicomdoodle.ui.main import MainWindow
    from dicomdoodle.ui.widgets.log_panel import QtLogHandler
    folder, config = tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        write_sphere_series(folder)                           # z 13-27, slices in z order
        settings = Settings(os.path.join(config, "settings.json"))
        window = MainWindow(settings, LabelManager(settings), _NoSam(), QtLogHandler())
        window.open_folder(folder, confirm=False)
        window.go_to_slice(20)
        window.actions.find_range.trigger()
        deadline = time.time() + 60
        while (window.busy or window.batch.found is None) and time.time() < deadline:
            app.processEvents()
            time.sleep(0.01)
        rng, panel = window.session.auto_range, window.batch_panel
        assert (rng.start, rng.end) == (13, 27), (rng.start, rng.end)
        assert not panel.preview.isHidden() and panel.found_label.text() == "From slice 21: slices 14–28"
        assert panel.run_auto_btn.text().endswith("(15)") and panel.run_auto_btn.isEnabled()
        assert window.navigator.timeline._bands, "no range band on the timeline"
        window.batch.range_action("clear")
        assert panel.preview.isHidden() and panel.found_label.isHidden()
        _dispose(app, window)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
        shutil.rmtree(config, ignore_errors=True)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"ok  {name}")
