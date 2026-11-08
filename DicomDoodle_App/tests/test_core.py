"""
Tests for the logic in dicomdoodle.core (and the save jobs built on it), on
a small synthetic series.

    python -m pytest tests          (or simply: python tests/test_core.py)
"""
import json
import os
import sys
import tempfile

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dicomdoodle.core import mask_ops  # noqa: E402
from dicomdoodle.core.annotation import AnnotationSession, SliceAnnotation, SliceRange  # noqa: E402
from dicomdoodle.core.dicom_series import DicomSeries, SegFileFilter, seg_path_for  # noqa: E402
from dicomdoodle.core.labels import LabelError, LabelRegistry  # noqa: E402
from dicomdoodle.core.seg_io import is_blank_segmentation, read_segmentation, write_segmentation  # noqa: E402
from dicomdoodle.core.volume_calculator import average_volume_cm3, compute_orientation_volume, compute_volumes  # noqa: E402,E501

SIZE, SLICES = 64, 4


def make_series(folder: str) -> None:
    """Write SLICES synthetic axial MR slices (out of order on disk)."""
    study, series, frame = generate_uid(), generate_uid(), generate_uid()
    for k in reversed(range(SLICES)):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame
        ds.Modality, ds.PatientName, ds.PatientID = "MR", "Test^Patient", "T1"
        ds.Manufacturer, ds.ManufacturerModelName = "Test", "Synthetic"
        ds.InstanceNumber = k + 1
        ds.ImagePositionPatient = [0.0, 0.0, float(k)]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing, ds.SliceThickness = [0.5, 0.5], 2.0
        ds.Rows = ds.Columns = SIZE
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        pixels = (np.arange(SIZE * SIZE, dtype=np.uint16).reshape(SIZE, SIZE) + k * 10)
        ds.PixelData = pixels.tobytes()
        ds.save_as(os.path.join(folder, f"slice_{k + 1}.dcm"), enforce_file_format=True)


def square_outline(shape, lo=20, hi=40):
    mask = np.zeros(shape, bool)
    mask[lo:hi, lo:hi] = True
    return mask


def test_label_rules():
    reg = LabelRegistry()
    assert [lab.value for lab in reg.labels()] == [1, 2, 3, 4, 5, 6]
    lab = reg.add("Enhancing Core", "#FF8800", "Core")
    assert lab.value == 7 and reg.find_by_name("enhancing core") is lab
    for name, color in [("", "#123456"), ("Tumor & Brain Boundary", "#123456"), ("X", "#000000"),
                        ("Y", "#FF0000"), ("No Segmentation", "#121212")]:
        try:
            reg.add(name, color)
        except LabelError:
            continue
        raise AssertionError(f"accepted invalid label {name!r} {color}")
    restored = LabelRegistry(reg.to_dicts())
    assert restored.get(7).name == "Enhancing Core"
    assert reg.value_for_segment("Magenta Boundary") == 6
    assert reg.value_for_segment("No Segmentation") == 0


def test_series_load_and_seg_roundtrip():
    with tempfile.TemporaryDirectory() as folder:
        make_series(folder)
        series = DicomSeries.load(folder)
        assert len(series) == SLICES and series.instance_numbers == [1, 2, 3, 4]
        assert series.info.plane == "Axial"

        reg = LabelRegistry()
        custom = reg.add("Cyst Boundary", "#88CCFF")
        labels = np.zeros((SIZE, SIZE), np.uint8)
        labels[square_outline((SIZE, SIZE))] = custom.value
        labels[5:10, 5:10] = 1
        out = series.seg_path(1)
        write_segmentation(mask_ops.boundary_map(labels), series.file(1), out, 1, registry=reg)
        assert os.path.isfile(out) and out == seg_path_for(series.file(1))

        data = read_segmentation(out, reg)
        assert {s.label for s in data.segments} == {"Tumor & Brain Boundary", "Cyst Boundary"}
        assert set(np.unique(data.label_map)) == {0, 1, custom.value}

        # a fresh registry still shows the unknown label (session-only)
        fresh = LabelRegistry()
        seen = read_segmentation(out, fresh)
        assert any(lab.transient and lab.name == "Cyst Boundary" for lab in fresh.labels())
        assert seen.label_map.any()

        # blank mask -> "No Segmentation"
        blank = series.seg_path(2)
        write_segmentation(np.zeros((SIZE, SIZE), np.uint8), series.file(2), blank, 2)
        blank_data = read_segmentation(blank, reg)
        assert blank_data.is_empty and blank_data.segments[0].label == "No Segmentation"

        # annotation files are not slices
        assert len(DicomSeries.load(folder)) == SLICES


def test_volume_from_filled_outline():
    with tempfile.TemporaryDirectory() as folder:
        make_series(folder)
        series = DicomSeries.load(folder)
        outline = mask_ops.boundary_map(square_outline((SIZE, SIZE)).astype(np.uint8))
        for i in (0, 1):
            write_segmentation(outline, series.file(i), series.seg_path(i), i)
        result = compute_orientation_volume(folder, "Axial", SegFileFilter.NATIVE)
        filled = int(np.count_nonzero(mask_ops.fill_closed_boundaries(outline)))
        assert result.annotated_slices == 2
        assert result.voxels == 2 * filled
        assert abs(result.volume_cm3 - 2 * filled * 0.5 * 0.5 * 2.0 / 1000) < 1e-9


def test_volume_average_counts_annotated_series_only():
    with tempfile.TemporaryDirectory() as root:
        annotated, plain = os.path.join(root, "ax"), os.path.join(root, "cor")
        for folder in (annotated, plain):
            os.makedirs(folder)
            make_series(folder)
        series = DicomSeries.load(annotated)
        outline = mask_ops.boundary_map(square_outline((SIZE, SIZE)).astype(np.uint8))
        write_segmentation(outline, series.file(0), series.seg_path(0), 0)
        results = compute_volumes({"Axial": annotated, "Coronal": plain, "Sagittal": os.path.join(root, "none")})
        assert [r.orientation for r in results] == ["Axial", "Coronal", "Sagittal"]
        assert results[0].counted and not results[1].counted and results[2].error
        assert abs(average_volume_cm3(results) - results[0].volume_cm3) < 1e-12


def test_annotation_history_and_range():
    ann = SliceAnnotation((8, 8))
    mask = np.zeros((8, 8), bool)
    mask[2:4, 2:4] = True
    ann.paint(mask, 3)
    assert ann.dirty and ann.can_undo and ann.label_map.sum() == 12
    ann.undo()
    assert ann.is_empty and ann.can_redo
    ann.redo()
    assert not ann.is_empty

    rng = SliceRange(5, 2)
    assert rng.indices() == [2, 3, 4, 5] and rng.describe() == "6 – 3  (4 slices)"


def test_hand_drawn_layer():
    shape = (SIZE, SIZE)
    ann = SliceAnnotation(shape)
    ann.begin_edit()                              # a pen stroke: a closed square outline, label 2
    ann.strokes[10, 10:30] = ann.strokes[29, 10:30] = ann.strokes[10:30, 10] = ann.strokes[10:30, 29] = 2
    assert not ann.is_empty and ann.dirty and ann.label_values() == [2]
    region = square_outline(shape, 40, 56)
    ann.paint(region, 1)                          # an AI region next to it
    assert ann.label_values() == [1, 2]
    saved = ann.outline_map()                     # region as its outline, strokes exactly as drawn
    assert np.array_equal(saved[ann.strokes > 0], ann.strokes[ann.strokes > 0])
    assert np.array_equal(saved == 1, mask_ops.boundary_map(region.astype(np.uint8)) > 0)
    filled = ann.filled_map()
    assert (filled[11:29, 11:29] == 2).all() and (filled[region] == 1).all()

    ann.begin_edit()                              # the eraser's first touch: regions become outlines
    assert ann.regions_to_outlines() and not ann.label_map.any()
    assert np.array_equal(ann.outline_map(), saved), "turning regions into outlines must not change the file"
    ann.undo()
    assert ann.label_map.any() and np.array_equal(ann.outline_map(), saved)
    ann.undo()
    assert ann.label_values() == [2] and not ann.label_map.any()
    ann.redo()
    assert ann.label_values() == [1, 2]
    ann.replace(np.zeros(shape, np.uint8), mark_dirty=False, strokes=saved)   # Edit saved outline
    assert not ann.dirty and not ann.label_map.any() and np.array_equal(ann.strokes, saved)
    ann.clear()
    assert ann.is_empty and ann.can_undo


def test_save_jobs_write_hand_drawn_outlines():
    from dicomdoodle.core.progress import Progress
    from dicomdoodle.workers.jobs import BatchItem, save_many_job, save_slice_job

    with tempfile.TemporaryDirectory() as folder:
        make_series(folder)
        series = DicomSeries.load(folder)
        names = {1: "Tumor & Brain Boundary", 2: "Tumor & Dura Boundary"}
        regions = np.zeros((SIZE, SIZE), np.uint8)
        regions[square_outline((SIZE, SIZE), 40, 56)] = 1
        strokes = np.zeros((SIZE, SIZE), np.uint8)
        strokes[10, 10:30] = strokes[29, 10:30] = strokes[10:30, 10] = strokes[10:30, 29] = 2
        result = save_slice_job(Progress(), regions, series.file(0), series.seg_path(0), 0, names, strokes=strokes)
        data = read_segmentation(result.path, LabelRegistry())
        assert not result.blank and set(np.unique(data.label_map)) == {0, 1, 2}
        assert (data.label_map[strokes > 0] == 2).all()
        report = save_many_job(Progress(), [BatchItem(1, series.file(1), series.seg_path(1),
                                                      label_map=np.zeros_like(regions), strokes=strokes)], names)
        assert [o.status for o in report.saved] == ["annotated"]
        assert (read_segmentation(series.seg_path(1), LabelRegistry()).label_map[strokes > 0] == 2).all()


def test_automatic_blank_mask_targets():
    with tempfile.TemporaryDirectory() as folder:
        make_series(folder)
        series = DicomSeries.load(folder)
        outline = mask_ops.boundary_map(square_outline((SIZE, SIZE)).astype(np.uint8))
        write_segmentation(outline, series.file(0), series.seg_path(0), 0)
        write_segmentation(np.zeros((SIZE, SIZE), np.uint8), series.file(1), series.seg_path(1), 1)
        assert is_blank_segmentation(series.seg_path(1)) and not is_blank_segmentation(series.seg_path(0))

        session = AnnotationSession(series)
        session.scan_files()
        assert session.files == {0, 1} and session.blank_files == {1}
        # slice 2 has an unsaved outline, so only slice 3 needs a blank mask
        session.annotation(2).paint(square_outline((SIZE, SIZE)), 1)
        assert session.blank_mask_targets() == [3] and session.blank_mask_targets(exclude=[3]) == []
        # batches skip slices saved this session, ask about outlines from earlier
        # sessions, and replace blank masks freely
        session.mark_saved(3, blank=True)
        assert session.split_existing([0, 1, 2, 3]) == ([3], [0], [1, 2])
        session.mark_auto_blank(2)
        assert session.is_blank(2) and not session.is_saved(2)


def test_save_writes_automatic_blank_masks():
    from dicomdoodle.core.progress import Progress
    from dicomdoodle.workers.jobs import BatchItem, save_slice_job

    with tempfile.TemporaryDirectory() as folder:
        make_series(folder)
        series = DicomSeries.load(folder)
        outline = mask_ops.boundary_map(square_outline((SIZE, SIZE)).astype(np.uint8))
        write_segmentation(outline, series.file(2), series.seg_path(2), 2)   # saved meanwhile: must survive
        blanks = [BatchItem(i, series.file(i), series.seg_path(i), shape=(SIZE, SIZE)) for i in (1, 2, 3)]
        labels = np.zeros((SIZE, SIZE), np.uint8)
        labels[square_outline((SIZE, SIZE))] = 1
        result = save_slice_job(Progress(), labels, series.file(0), series.seg_path(0), 0,
                                {1: "Tumor & Brain Boundary"}, blanks=blanks)
        assert not result.blank and not result.blanks.errors
        assert [o.index for o in result.blanks.auto_blanks] == [1, 3]
        assert [is_blank_segmentation(series.seg_path(i)) for i in range(SLICES)] == [False, True, False, True]


def test_seg_follows_the_dicom_standard():
    """Patient data copied unchanged, DicomDoodle as equipment, coded meaning, color and method per segment."""
    import pydicom
    from dicomdoodle.core.seg_io import AUTO_DETECTED, HAND_DRAWN

    with tempfile.TemporaryDirectory() as folder:
        make_series(folder)
        series = DicomSeries.load(folder)
        source = pydicom.dcmread(series.file(1))
        source.PatientName, source.PatientAge = "Anon", "58"              # no caret, age without unit
        source.save_as(series.file(1))
        reg = LabelRegistry()
        custom = reg.add("Cyst Boundary", "#88CCFF")
        labels = np.zeros((SIZE, SIZE), np.uint8)
        labels[square_outline((SIZE, SIZE))] = custom.value
        labels[5:10, 5:10] = 3
        out = write_segmentation(mask_ops.boundary_map(labels), series.file(1), series.seg_path(1), 1,
                                 registry=reg, manual_values=[custom.value])
        seg = pydicom.dcmread(out)
        assert str(seg.PatientName) == "Anon" and seg.PatientID == "T1" and seg.PatientAge == "058Y"
        assert seg.StudyInstanceUID == source.StudyInstanceUID and seg.FrameOfReferenceUID == source.FrameOfReferenceUID
        assert "PatientBirthDate" in seg and seg.PatientBirthDate == ""     # Type 2: present, not invented
        assert seg.Manufacturer == "DicomDoodle" and seg.ManufacturerModelName == "DicomDoodle"
        assert seg.SeriesDescription == "DicomDoodle annotation, image 2" and seg.SeriesNumber == 2
        segments = {str(s.SegmentLabel): s for s in seg.SegmentSequence}
        necrosis, cyst = segments["Tumor & Necrosis Boundary"], segments["Cyst Boundary"]
        assert necrosis.SegmentedPropertyCategoryCodeSequence[0].CodeValue == "49755003"   # CID 7150
        assert necrosis.SegmentedPropertyTypeCodeSequence[0].CodeValue == "6574001"        # Necrosis
        assert necrosis.SegmentAlgorithmType == "SEMIAUTOMATIC"
        assert "SegmentationAlgorithmIdentificationSequence" in necrosis
        assert cyst.SegmentAlgorithmType == "MANUAL" and "SegmentationAlgorithmIdentificationSequence" not in cyst
        assert cyst.SegmentedPropertyTypeCodeSequence[0].CodeMeaning == "Lesion"
        assert len(cyst.RecommendedDisplayCIELabValue) == 3

        # the same label on another slice is the same tracked finding; blank masks are Anatomical Structure / Brain
        other = write_segmentation(mask_ops.boundary_map(labels), series.file(2), series.seg_path(2), 2,
                                   registry=reg, method=AUTO_DETECTED)
        other_cyst = next(s for s in pydicom.dcmread(other).SegmentSequence if s.SegmentLabel == "Cyst Boundary")
        assert other_cyst.TrackingUID == cyst.TrackingUID and other_cyst.SegmentAlgorithmType == "AUTOMATIC"
        blank = pydicom.dcmread(write_segmentation(np.zeros((SIZE, SIZE), np.uint8), series.file(3),
                                                   series.seg_path(3), 3, method=HAND_DRAWN))
        item = blank.SegmentSequence[0]
        assert (item.SegmentedPropertyCategoryCodeSequence[0].CodeMeaning,
                item.SegmentedPropertyTypeCodeSequence[0].CodeMeaning) == ("Anatomical Structure", "Brain")

        # labels that do not fit the source's character set switch the file to UTF-8
        named = LabelRegistry([{"value": 7, "name": "ट्यूमर", "color": "#FF8800"}])
        utf8 = write_segmentation(labels == custom.value, series.file(0), series.seg_path(0), 0,
                                  labels={1: named.get(7)})
        assert pydicom.dcmread(utf8).SpecificCharacterSet == "ISO_IR 192"
        assert read_segmentation(utf8, named).segments[0].label == "ट्यूमर"


def test_dicom_code_dictionary():
    from dicomdoodle.core import coding

    first = coding.search_dicom("tumour")[0]                     # British spelling, everyday word
    assert (first.scheme, first.value, first.meaning) == ("SCT", "108369006", "Neoplasm")
    assert coding.search_dicom("oedema")[0].meaning == "Edema"
    assert coding.search_dicom("108369006")[0].meaning == "Neoplasm"
    assert coding.search_dicom("zzzz-no-such-term") == []
    assert len(coding.categories()) == 8 and all(t.is_category for t in coding.categories())
    assert coding.dicom_scheme("SNOMEDCT") == "SCT" and coding.dicom_scheme("NCIT") == "NCIt"
    local = coding.CodedTerm("X1", "OMIM", "Something", source="BioPortal", ontology="OMIM")
    assert not local.is_standard_scheme and coding.scheme_identification(local).CodingSchemeDesignator == "OMIM"
    assert coding.scheme_identification(first) is None


def test_apply_concepts_to_one_label():
    import pydicom
    from dicomdoodle.core import coding
    from dicomdoodle.core.seg_io import apply_concepts

    with tempfile.TemporaryDirectory() as folder:
        make_series(folder)
        series = DicomSeries.load(folder)
        labels = np.zeros((SIZE, SIZE), np.uint8)
        labels[square_outline((SIZE, SIZE))] = 1
        labels[5:10, 5:10] = 4
        write_segmentation(mask_ops.boundary_map(labels), series.file(0), series.seg_path(0), 0)
        write_segmentation(np.zeros((SIZE, SIZE), np.uint8), series.file(1), series.seg_path(1), 1)
        before = pydicom.dcmread(series.seg_path(0)).SOPInstanceUID
        glioma = coding.CodedTerm("X1", "OMIM", "Glioma", source="BioPortal", ontology="OMIM")
        category = coding.categories()[0]
        files = [series.seg_path(0), series.seg_path(1)]
        updated, errors = apply_concepts(files, category, glioma, "Tumor & Brain Boundary")
        assert (updated, errors) == (1, [])                       # the blank mask already had its terms
        ds = pydicom.dcmread(series.seg_path(0))
        types = {str(s.SegmentLabel): s.SegmentedPropertyTypeCodeSequence[0].CodeMeaning for s in ds.SegmentSequence}
        assert types == {"Tumor & Brain Boundary": "Glioma", "Edema & Brain Boundary": "Edema"}
        assert ds.SOPInstanceUID != before and ds.file_meta.MediaStorageSOPInstanceUID == ds.SOPInstanceUID
        assert [i.CodingSchemeDesignator for i in ds.CodingSchemeIdentificationSequence] == ["OMIM"]
        assert apply_concepts(files, category, glioma, "Tumor & Brain Boundary") == (0, [])   # nothing new


def make_enhanced_mr(path: str, frames: int = 3) -> np.ndarray:
    """A small Enhanced MR multi-frame file with shared and per-frame functional groups."""
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence
    from pydicom.uid import EnhancedMRImageStorage

    def item(**values):
        ds = Dataset()
        for key, value in values.items():
            setattr(ds, key, value)
        return Sequence([ds])

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID = EnhancedMRImageStorage, generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = EnhancedMRImageStorage, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = generate_uid(), generate_uid(), generate_uid()
    ds.Modality, ds.PatientName, ds.PatientID, ds.PatientAge = "MR", "Anon", "P1", "58"
    ds.ImageType = ["ORIGINAL", "PRIMARY", "M", "NONE"]
    ds.EchoPulseSequence, ds.SegmentedKSpaceTraversal, ds.SpectrallySelectedSuppression = "SPIN", "PARTIAL", "FAT"
    ds.MRAcquisitionType, ds.PulseSequenceName, ds.ResonantNucleus = "2D", "tse", "1H"
    ds.PixelPresentation, ds.NumberOfFrames = "MONOCHROME", frames
    ds.Rows = ds.Columns = SIZE
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    shared = Dataset()
    shared.PlaneOrientationSequence = item(ImageOrientationPatient=[1, 0, 0, 0, 1, 0])
    shared.PixelMeasuresSequence = item(PixelSpacing=[0.5, 0.5], SliceThickness=2.0)
    shared.MRTimingAndRelatedParametersSequence = item(RepetitionTime=723, FlipAngle=150, EchoTrainLength=3)
    shared.MRModifierSequence = item(InversionRecovery="NO")
    ds.SharedFunctionalGroupsSequence = Sequence([shared])
    ds.PerFrameFunctionalGroupsSequence = Sequence()
    for k in range(frames):
        group = Dataset()
        group.PlanePositionSequence = item(ImagePositionPatient=[0.0, 0.0, 4.0 * k])
        group.FrameVOILUTSequence = item(WindowCenter=300 + k, WindowWidth=600)
        group.MREchoSequence = item(EffectiveEchoTime=8.8)
        group.MRImageFrameTypeSequence = item(FrameType=["ORIGINAL", "PRIMARY", "M", "NONE"])
        group.FrameContentSequence = item(InStackPositionNumber=k + 1, FrameAcquisitionDateTime="20250111124047.5")
        ds.PerFrameFunctionalGroupsSequence.append(group)
    pixels = (np.arange(frames * SIZE * SIZE, dtype=np.uint16) % 4000).reshape(frames, SIZE, SIZE)
    ds.PixelData = pixels.tobytes()
    ds.save_as(path, enforce_file_format=True)
    return pixels


def test_multiframe_conversion_to_standard_slices():
    import pydicom
    from dicomdoodle.core.multiframe import MultiFrameDicom

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "enhanced.dcm")
        pixels = make_enhanced_mr(path)
        dicom = MultiFrameDicom(path)
        out = os.path.join(folder, "slices")
        assert dicom.split(out) == 3
        series = DicomSeries.load(out)
        assert len(series) == 3
        for k in range(3):
            ds = pydicom.dcmread(series.file(k))
            assert ds.SOPClassUID == MRImageStorage and ds.file_meta.TransferSyntaxUID == ExplicitVRLittleEndian
            assert np.array_equal(ds.pixel_array, pixels[k])
            assert [float(v) for v in ds.ImagePositionPatient] == [0.0, 0.0, 4.0 * k]
            assert [float(v) for v in ds.PixelSpacing] == [0.5, 0.5] and float(ds.WindowCenter) == 300 + k
            assert ds.ScanningSequence == "SE" and ds.SequenceVariant == "SK" and ds.ScanOptions == "FS"
            assert float(ds.EchoTime) == 8.8 and float(ds.RepetitionTime) == 723 and ds.SequenceName == "tse"
            assert list(ds.ImageType) == ["ORIGINAL", "PRIMARY", "M", "NONE"] and ds.InstanceNumber == k + 1
            assert ds.PatientAge == "058Y" and ds.AccessionNumber == "" and ds.AcquisitionDate == "20250111"
            for gone in ("NumberOfFrames", "SharedFunctionalGroupsSequence", "PerFrameFunctionalGroupsSequence",
                         "PixelPresentation", "EchoPulseSequence"):
                assert gone not in ds, gone
        first = pydicom.dcmread(series.file(0))
        source = pydicom.dcmread(path, stop_before_pixels=True)
        assert first.SeriesInstanceUID != source.SeriesInstanceUID           # a new series ...
        assert first.StudyInstanceUID == source.StudyInstanceUID
        assert first.FrameOfReferenceUID == source.FrameOfReferenceUID
        assert MultiFrameDicom(path).single_frame(0).SOPInstanceUID == first.SOPInstanceUID   # ... with stable UIDs
        # an annotation of a converted slice references it
        outline = mask_ops.boundary_map(square_outline((SIZE, SIZE)).astype(np.uint8))
        write_segmentation(outline, series.file(1), series.seg_path(1), 1)
        assert read_segmentation(series.seg_path(1), LabelRegistry()).label_map.any()


SEARCH_PAGE = {"collection": [
    {"prefLabel": "Glioma", "@id": "http://purl.bioontology.org/ontology/SNOMEDCT/393564001",
     "links": {"self": "https://data.bioontology.org/ontologies/SNOMEDCT/classes/glioma",
               "ontology": "https://data.bioontology.org/ontologies/SNOMEDCT"}},
    {"prefLabel": "Glioma", "notation": "C3058", "@id": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C3058",
     "links": {"ontology": "https://data.bioontology.org/ontologies/NCIT"}},
    {"prefLabel": "x" * 70, "@id": "http://example.org/1", "links": {"ontology": "https://example.org/X"}},
    {"prefLabel": "Code too long", "@id": "http://example.org/12345678901234567",
     "links": {"ontology": "https://example.org/X"}},
]}


def _fake_ontoportal(takes_timeout: bool, calls: list, error: Exception = None):
    """A stand-in for the ontoportal_client package: search() yields single results
    (as since 0.0.5); get_response takes a timeout from 0.0.6 on."""
    import types

    def get_response_0_0_5(self, path, params=None, raise_for_status=True, **kwargs):
        raise AssertionError("not used")

    def get_response_0_0_6(self, path, params=None, raise_for_status=True, timeout=None, **kwargs):
        raise AssertionError("not used")

    class Client:
        get_response = get_response_0_0_6 if takes_timeout else get_response_0_0_5

        def __init__(self, api_key=None):
            self.api_key = api_key

        def get_json(self, path, params=None, **kwargs):
            calls.append((path, params, kwargs))
            if error is not None:
                raise error
            return SEARCH_PAGE

        def search(self, text):
            yield from SEARCH_PAGE["collection"]

    return types.SimpleNamespace(BioPortalClient=Client)


def test_ontology_search_with_any_ontoportal_client():
    import types
    import urllib.error
    from dicomdoodle.core import ontology

    saved_module, saved_urlopen = sys.modules.get("ontoportal_client"), ontology.urllib.request.urlopen
    try:
        for takes_timeout in (True, False):           # 0.0.6+ and 0.0.5: one page through get_json
            calls = []
            sys.modules["ontoportal_client"] = _fake_ontoportal(takes_timeout, calls)
            terms = ontology.BioPortalClient("key").search("glioma")
            assert [(t.label, t.code_value, t.scheme) for t in terms] == [("Glioma", "393564001", "SCT"),
                                                                          ("Glioma", "C3058", "NCIt")]
            assert [t.ontology for t in terms] == ["SNOMEDCT", "NCIT"]   # DICOM designators, BioPortal source
            assert calls == [("/search", {"q": "glioma"}, {"timeout": 25.0} if takes_timeout else {})]

        unauthorized = OSError("401 Client Error: Unauthorized for url: …?apikey=key")
        unauthorized.response = types.SimpleNamespace(status_code=401)
        for error, expected in ((unauthorized, "API key"), (ValueError("bad JSON"), "could not be read")):
            sys.modules["ontoportal_client"] = _fake_ontoportal(True, [], error)
            try:
                ontology.BioPortalClient("key").search("glioma")
                raise AssertionError("no OntologyError")
            except ontology.OntologyError as exc:
                assert expected in str(exc) and "apikey" not in str(exc)
        sys.modules["ontoportal_client"] = _fake_ontoportal(True, [], AttributeError("a bug"))
        try:                                          # programming errors are not disguised as network trouble
            ontology.BioPortalClient("key").search("glioma")
            raise AssertionError("no AttributeError")
        except AttributeError:
            pass

        sys.modules["ontoportal_client"] = None       # not installed: the REST API directly
        urls = []

        class Answer:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return json.dumps(SEARCH_PAGE).encode("utf-8")

        ontology.urllib.request.urlopen = lambda request, timeout: urls.append(request.full_url) or Answer()
        assert len(ontology.BioPortalClient("key").search("glioma")) == 2
        assert urls == ["https://data.bioontology.org/search?q=glioma&apikey=key"]

        def offline(request, timeout):
            raise urllib.error.URLError("timed out")
        ontology.urllib.request.urlopen = offline
        try:
            ontology.BioPortalClient("key").search("glioma")
            raise AssertionError("no OntologyError")
        except ontology.OntologyError as exc:
            assert "did not answer in time" in str(exc)
    finally:
        ontology.urllib.request.urlopen = saved_urlopen
        if saved_module is None:
            sys.modules.pop("ontoportal_client", None)
        else:
            sys.modules["ontoportal_client"] = saved_module


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"ok  {name}")
