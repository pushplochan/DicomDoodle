"""
Reading and writing DICOM Segmentation (SEG) files — DicomDoodle's
``<slice>_seg.dcm`` annotation format.

``write_segmentation`` writes one Segmentation instance per slice, as the
original ``utils.create_dicom_segmentation`` did (one binary segment per
label value present, a "No Segmentation" segment for blank masks), and
follows the Segmentation IOD of PS3.3 (A.51, C.8.20) so that PACS and web
viewers such as OHIF read the files:

* Patient and study attributes are copied unchanged from the (anonymized)
  source slice, so the annotation stays in the source study. Missing
  mandatory values are added empty (Type 2) or as UIDs derived from the
  series (Type 1), never as invented data; a malformed Patient's Age
  ("64Y", "58") is written in the AS form ("064Y").
* The equipment is DicomDoodle — the software that made the file, not the
  scanner of the source image (Enhanced General Equipment Module).
* Each segment carries its label's coded category and type (core/coding.py),
  its display color (Recommended Display CIELab Value, so OHIF shows the
  label colors), how it was made (Segment Algorithm Type: SAM from a box is
  SEMIAUTOMATIC, Auto Detect batches AUTOMATIC, the pen MANUAL) and a
  tracking ID/UID shared by all slices of the same label in a series.
* Every file is its own series ("DicomDoodle annotation, image N") whose
  series number is the slice's instance number.

Readers map each segment back to a label value by its ``SegmentLabel`` and
place every frame on the slice it references (ReferencedSOPInstanceUID), so
multi-label files are handled correctly.
"""
from __future__ import annotations

import hashlib
import logging
import os
import platform
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pydicom
from pydicom.uid import generate_uid

from .. import APP_NAME, APP_VERSION
from .coding import ANATOMICAL_STRUCTURE, BRAIN, Code, CodedTerm, coded_concept, scheme_identification
from .gap_fill import fill_discontinuities
from .labels import EMPTY_SEGMENT_NAMES, LabelRegistry, SegLabel
from .progress import Progress, ensure_progress

log = logging.getLogger(__name__)

# The segment of a blank mask: the slice was looked at and holds no lesion.
BLANK_SEGMENT_LABEL = "No Segmentation"
BLANK_CATEGORY: Code = ANATOMICAL_STRUCTURE
BLANK_TYPE: Code = BRAIN


@dataclass(frozen=True)
class SegmentMethod:
    """How segments were made: the DICOM Segment Algorithm Type and, unless
    MANUAL, the algorithm (Segmentation Algorithm Identification)."""

    algorithm_type: str                     # AUTOMATIC | SEMIAUTOMATIC | MANUAL
    algorithm_name: str = ""
    family: Code = ("123110", "DCM", "Artificial Intelligence")   # CID 7162 Surface Processing Algorithm Families


AI_ASSISTED = SegmentMethod("SEMIAUTOMATIC", "SAM 2.1 (box prompt)")         # SAM on a box, reviewed by the user
AUTO_DETECTED = SegmentMethod("AUTOMATIC", "YOLO detection + SAM 2.1")       # Auto Detect batch, unattended
HAND_DRAWN = SegmentMethod("MANUAL")                                         # pen, and blank masks the user implies
TRANSFERRED = SegmentMethod("AUTOMATIC", "Label transfer by image registration",
                            ("125024", "DCM", "Image Content-based Alignment"))

LabelInfo = Union[SegLabel, str]


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #

def write_segmentation(label_map: np.ndarray, dicom_path: str, output_path: str,
                       slice_index: int = 0, registry: Optional[LabelRegistry] = None,
                       labels: Optional[Mapping[int, LabelInfo]] = None,
                       method: SegmentMethod = AI_ASSISTED, manual_values: Sequence[int] = ()) -> str:
    """Save ``label_map`` (0 = background) as a DICOM SEG of the slice ``dicom_path``.

    Each label value becomes a segment described by ``labels`` ({value:
    SegLabel or name}), else by ``registry`` (falling back to the built-in
    labels). ``method`` says how the segments were made; values in
    ``manual_values`` were drawn by hand. An empty map is saved as a blank
    mask. Returns ``output_path``.
    """
    import highdicom
    import highdicom.seg

    registry = registry or LabelRegistry()
    try:
        source_image = pydicom.dcmread(dicom_path)
    except Exception as exc:
        raise ValueError(f"Could not read source DICOM file: {exc}") from exc
    prepare_source(source_image, slice_index)

    gt = fill_discontinuities(np.asarray(label_map))
    values = [int(v) for v in np.unique(gt) if v != 0]
    descriptions = []
    pixel_array = np.zeros(gt.shape, dtype=np.uint8)
    for number, value in enumerate(values, start=1):
        label = _segment_label(value, labels, registry)
        segment_method = HAND_DRAWN if value in manual_values else method
        descriptions.append(_description(number, label.name, label.category, label.property_type,
                                         segment_method, source_image, label.color))
        pixel_array[gt == value] = number
    if not values:
        log.debug("No annotation on slice index %s; writing a blank segmentation.", slice_index)
        descriptions.append(_description(1, BLANK_SEGMENT_LABEL, BLANK_CATEGORY, BLANK_TYPE,
                                         method, source_image))
    choose_character_set(source_image, [str(d.SegmentLabel) for d in descriptions])

    instance = int(source_image.InstanceNumber)
    with warnings.catch_warnings():
        # anonymized names often have one part ("DummyPatName!"); they are copied as they are
        warnings.filterwarnings("ignore", message=".*unlikely to represent the intended person name")
        seg = highdicom.seg.Segmentation(
            source_images=[source_image],
            pixel_array=pixel_array,
            segmentation_type=highdicom.seg.SegmentationTypeValues.BINARY,
            segment_descriptions=descriptions,
            series_instance_uid=generate_uid(),
            series_number=instance,
            sop_instance_uid=generate_uid(),
            instance_number=instance,
            series_description=f"{APP_NAME} annotation, image {instance}",
            content_label="ANNOTATION",
            content_description=f"Lesion annotation made with {APP_NAME}",
            **equipment(),
            # a blank mask is one empty frame; saying so up front stops highdicom warning about it
            omit_empty_frames=bool(values),
        )

    # Write to a temporary file first so a crash never leaves a half-written SEG.
    tmp_path = output_path + ".tmp"
    seg.save_as(tmp_path)
    os.replace(tmp_path, output_path)
    if values:
        log.info("Saved DICOM segmentation: %s", output_path)
    else:   # blank masks are often written by the dozen; callers log a summary
        log.debug("Saved blank DICOM segmentation: %s", output_path)
    return output_path


def equipment() -> Dict[str, str]:
    """Enhanced General Equipment of the files DicomDoodle makes (the software, not the scanner)."""
    # a stable number per computer, without revealing its name
    serial = hashlib.sha1(platform.node().encode("utf-8")).hexdigest()[:12].upper()
    return {"manufacturer": APP_NAME, "manufacturer_model_name": APP_NAME,
            "software_versions": APP_VERSION, "device_serial_number": serial}


def _segment_label(value: int, labels: Optional[Mapping[int, LabelInfo]], registry: LabelRegistry) -> SegLabel:
    item = labels.get(value) if labels else None
    if isinstance(item, SegLabel):
        return item
    known = registry.get(value)
    if item is None:
        return known or SegLabel(value, registry.name_for(value), registry.color_for(value))
    if known is not None and known.name == item:
        return known
    # a name without a label (e.g. labels transferred in the Multiplanar view)
    return registry.find_by_name(str(item)) or SegLabel(value, str(item), registry.color_for(value))


def _description(number: int, name: str, category: Code, property_type: Code, method: SegmentMethod,
                 source_image, color: Optional[Tuple[int, int, int]] = None):
    import highdicom
    import highdicom.seg
    from highdicom.color import CIELabColor

    algorithm = None
    if method.algorithm_type != HAND_DRAWN.algorithm_type:
        algorithm = highdicom.AlgorithmIdentificationSequence(
            name=method.algorithm_name, family=coded_concept(method.family), version=APP_VERSION, source=APP_NAME)
    # the same label in the same series is the same finding on every slice
    tracking_uid = generate_uid(entropy_srcs=[str(source_image.StudyInstanceUID),
                                              str(source_image.SeriesInstanceUID), name])
    return highdicom.seg.SegmentDescription(
        segment_number=number,
        segment_label=name,
        segmented_property_category=coded_concept(category),
        segmented_property_type=coded_concept(property_type),
        algorithm_type=method.algorithm_type,
        algorithm_identification=algorithm,
        tracking_uid=tracking_uid,
        tracking_id=name,
        display_color=CIELabColor.from_rgb(*color) if color is not None else None,
    )


# Patient / study attributes the SEG copies from its source; Type 2 — present, but may be empty.
_TYPE2_SOURCE_ATTRIBUTES = ("PatientName", "PatientID", "PatientBirthDate", "PatientSex", "StudyDate",
                            "StudyTime", "ReferringPhysicianName", "StudyID", "AccessionNumber")


def prepare_source(source_image, slice_index: int = 0) -> None:
    """Make a (possibly incompletely anonymized) source slice usable as the SEG's source, in memory.

    Nothing that exists is changed except a Patient's Age written outside the
    AS form; nothing is invented: Type 2 attributes are added empty, missing
    Type 1 UIDs are derived from the series so that every slice agrees.
    """
    for keyword in ("PixelSpacing", "ImageOrientationPatient", "ImagePositionPatient", "SOPInstanceUID"):
        if keyword not in source_image:
            raise ValueError(f"Missing {keyword} in the source image.")
    source_image.decode()           # text as str, so the SEG may use another character set
    for keyword in _TYPE2_SOURCE_ATTRIBUTES:
        if keyword not in source_image:
            setattr(source_image, keyword, "")
    series_seed = str(source_image.get("SeriesInstanceUID") or os.path.dirname(source_image.filename or ""))
    if not source_image.get("SeriesInstanceUID"):
        source_image.SeriesInstanceUID = generate_uid(entropy_srcs=["series", series_seed])
    if not source_image.get("StudyInstanceUID"):
        source_image.StudyInstanceUID = generate_uid(entropy_srcs=["study", series_seed])
    if not source_image.get("FrameOfReferenceUID"):
        source_image.FrameOfReferenceUID = generate_uid(entropy_srcs=["frame of reference", series_seed])
    if source_image.get("InstanceNumber") in (None, ""):
        source_image.InstanceNumber = int(slice_index) + 1
    if "PatientAge" in source_image:
        age = normalized_age(str(source_image.PatientAge))
        if age:
            source_image.PatientAge = age
        else:
            del source_image.PatientAge             # Type 3: better absent than invalid


def choose_character_set(source_image, texts: Sequence[str]) -> None:
    """Keep the source's character set — the one viewers decode best (dcmjs, used by OHIF,
    reads text inside sequences as Latin-1) — unless a label needs another: then UTF-8."""
    from pydicom.charset import convert_encodings

    charset = source_image.get("SpecificCharacterSet") or ""
    names = [charset] if isinstance(charset, str) else list(charset)
    try:
        encoding = "ascii" if names[0] in ("", "ISO_IR 6") else convert_encodings(names)[0]
        for text in texts:
            text.encode(encoding)
    except (LookupError, UnicodeError, ValueError):
        source_image.SpecificCharacterSet = "ISO_IR 192"


def normalized_age(value: str) -> str:
    """'064Y' for '64Y', '64 Y' or '58' (years when no unit is given); '' if unreadable."""
    match = re.fullmatch(r"\s*0*(\d{1,3})\s*([DWMYdwmy]?)\s*", value or "")
    if not match:
        return ""
    return f"{int(match.group(1)):03d}{(match.group(2) or 'Y').upper()}"


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #

@dataclass
class SegmentRecord:
    """One entry of a SEG file's SegmentSequence."""

    number: int
    label: str
    category: str
    type: str


@dataclass
class SegFrame:
    """One 2D frame of a SEG file."""

    mask: np.ndarray                 # bool (rows, cols)
    segment_number: int
    referenced_sop_uid: Optional[str]


@dataclass
class SegmentationData:
    """A SEG file decoded into a label map (values from the LabelRegistry)."""

    path: str
    label_map: np.ndarray
    segments: List[SegmentRecord] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not np.any(self.label_map)

    @property
    def category(self) -> str:
        return self.segments[0].category if self.segments else "None"

    @property
    def type(self) -> str:
        return self.segments[0].type if self.segments else "None"


def _code_meaning(item, attribute: str) -> str:
    seq = item.get(attribute)
    if seq:
        return str(seq[0].get("CodeMeaning", "No Meaning"))
    return "No Meaning"


def read_segment_records(ds) -> List[SegmentRecord]:
    records = []
    for k, item in enumerate(getattr(ds, "SegmentSequence", []) or []):
        number = int(item.get("SegmentNumber", k + 1))
        records.append(SegmentRecord(number, str(item.get("SegmentLabel", "No Label")),
                                     _code_meaning(item, "SegmentedPropertyCategoryCodeSequence"),
                                     _code_meaning(item, "SegmentedPropertyTypeCodeSequence")))
    return records


def is_blank_segmentation(path: str) -> bool:
    """True if the file is a blank mask ("No Segmentation"). Reads only the header, so it is fast."""
    records = read_segment_records(pydicom.dcmread(path, stop_before_pixels=True))
    return bool(records) and all(r.label.strip().lower() in EMPTY_SEGMENT_NAMES for r in records)


def iter_frames(ds, records: Optional[List[SegmentRecord]] = None) -> Iterator[SegFrame]:
    """Yield every frame with the segment it belongs to and the slice it references."""
    records = records if records is not None else read_segment_records(ds)
    pixels = ds.pixel_array
    frames = pixels if pixels.ndim == 3 else pixels[np.newaxis, ...]
    per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", None) or []
    for i, frame in enumerate(frames):
        number, ref_uid = None, None
        if i < len(per_frame):
            fg = per_frame[i]
            seg_id = fg.get("SegmentIdentificationSequence")
            if seg_id:
                number = int(seg_id[0].ReferencedSegmentNumber)
            derivation = fg.get("DerivationImageSequence")
            if derivation and derivation[0].get("SourceImageSequence"):
                ref_uid = str(derivation[0].SourceImageSequence[0].ReferencedSOPInstanceUID)
        if number is None:   # very old files: frame i <-> segment i (original behaviour)
            number = records[min(i, len(records) - 1)].number if records else 1
        yield SegFrame(frame > 0, number, ref_uid)


def read_segmentation(path: str, registry: LabelRegistry,
                      shape: Optional[Tuple[int, int]] = None) -> SegmentationData:
    """Decode one SEG file into a label map using the registry's label values.

    Segments with unknown names get a session-only label so they still show.
    If ``shape`` is given and differs, the map is resized (nearest neighbour).
    """
    ds = pydicom.dcmread(path)
    records = read_segment_records(ds)
    names = {r.number: r.label for r in records}
    label_map = None
    for frame in iter_frames(ds, records):
        if label_map is None:
            label_map = np.zeros(frame.mask.shape, dtype=np.uint8)
        value = registry.value_for_segment(names.get(frame.segment_number))
        if value:
            label_map[frame.mask & (label_map == 0)] = value
    if label_map is None:
        label_map = np.zeros(shape or (0, 0), dtype=np.uint8)
    if shape is not None and label_map.shape != tuple(shape):
        import cv2
        label_map = cv2.resize(label_map, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return SegmentationData(path, label_map, records)


def collect_series_masks(slice_uids: Sequence[str], slice_instances: Sequence[Optional[int]],
                         seg_paths: Sequence[str], shape: Tuple[int, int],
                         progress: Optional[Progress] = None) -> Dict[int, np.ndarray]:
    """Union of all annotated pixels per slice index, across many SEG files.

    Frames are matched to slices by ReferencedSOPInstanceUID; files without
    that reference fall back to their InstanceNumber (as the original did).
    """
    progress = ensure_progress(progress)
    uid_to_index = {uid: i for i, uid in enumerate(slice_uids) if uid}
    instance_to_index = {inst: i for i, inst in enumerate(slice_instances) if inst is not None}
    masks: Dict[int, np.ndarray] = {}
    for k, path in enumerate(seg_paths):
        progress(k, len(seg_paths), "Reading annotations…")
        try:
            ds = pydicom.dcmread(path)
            frames = list(iter_frames(ds))
        except Exception as exc:
            log.warning("Skipping unreadable annotation %s: %s", path, exc)
            continue
        fallback = instance_to_index.get(int(ds.InstanceNumber)) if ds.get("InstanceNumber") else None
        for frame in frames:
            index = uid_to_index.get(frame.referenced_sop_uid, fallback)
            if index is None or not frame.mask.any():
                continue
            mask = frame.mask
            if mask.shape != tuple(shape):
                log.warning("Annotation %s does not match the slice size; skipped.", path)
                continue
            masks[index] = masks[index] | mask if index in masks else mask.copy()
    return masks


# --------------------------------------------------------------------------- #
# Ontology terms
# --------------------------------------------------------------------------- #

def apply_concepts(paths: Sequence[str], category: CodedTerm, type_: CodedTerm,
                   segment_label: Optional[str] = None,
                   progress: Optional[Progress] = None) -> Tuple[int, List[str]]:
    """Write a Segmented Property Category and Type into annotation files.

    The segments named ``segment_label`` (every segment if None) get the
    chosen concepts; blank-mask segments get Anatomical Structure / Brain.
    Concepts of a local coding scheme (not in PS3.16) are described in the
    Coding Scheme Identification Sequence, and every file whose content
    changes gets a new SOP Instance UID, as the standard requires.
    Returns (files updated, error messages).
    """
    from pydicom.sequence import Sequence as DicomSequence

    progress = ensure_progress(progress)
    wanted = segment_label.strip().casefold() if segment_label else None
    updated, errors = 0, []
    for k, path in enumerate(paths):
        name = os.path.basename(path)
        progress(k, len(paths), f"Updating {name}")
        try:
            ds = pydicom.dcmread(path)
            if not getattr(ds, "SegmentSequence", None):
                errors.append(f"{name}: no SegmentSequence")
                continue
            used: List[CodedTerm] = []
            changed = False
            for segment in ds.SegmentSequence:
                label = str(segment.get("SegmentLabel", "")).strip().casefold()
                if label in EMPTY_SEGMENT_NAMES:
                    concepts = (BLANK_CATEGORY, BLANK_TYPE)
                elif wanted is None or label == wanted:
                    concepts = (category.code, type_.code)
                    used += [category, type_]
                else:
                    continue
                for attribute, concept in zip(("SegmentedPropertyCategoryCodeSequence",
                                               "SegmentedPropertyTypeCodeSequence"), concepts):
                    if _code_of(segment, attribute) != concept:
                        setattr(segment, attribute, DicomSequence([coded_concept(concept)]))
                        changed = True
            if not changed:
                continue
            _add_scheme_identification(ds, used)
            ds.SOPInstanceUID = generate_uid()
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            tmp_path = path + ".tmp"
            ds.save_as(tmp_path)
            os.replace(tmp_path, path)
            updated += 1
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    progress(len(paths), len(paths), "Done")
    return updated, errors


def _code_of(item, attribute: str) -> Optional[Code]:
    seq = item.get(attribute)
    if not seq:
        return None
    code = seq[0]
    return (str(code.get("CodeValue", "")), str(code.get("CodingSchemeDesignator", "")),
            str(code.get("CodeMeaning", "")))


def _add_scheme_identification(ds, terms: Sequence[CodedTerm]) -> None:
    """Describe the local coding schemes of ``terms`` (PS3.3 C.12.1), once per designator."""
    from pydicom.sequence import Sequence as DicomSequence

    known = {str(item.get("CodingSchemeDesignator")) for item in ds.get("CodingSchemeIdentificationSequence", [])}
    for term in terms:
        item = scheme_identification(term)
        if item is not None and term.scheme not in known:
            if "CodingSchemeIdentificationSequence" not in ds:
                ds.CodingSchemeIdentificationSequence = DicomSequence()
            ds.CodingSchemeIdentificationSequence.append(item)
            known.add(term.scheme)
