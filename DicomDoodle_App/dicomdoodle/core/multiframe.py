"""
Multi-frame DICOM -> one standard single-frame DICOM file per frame.

The files are classic images that PACS, OHIF and DicomDoodle read like any
scanner series, and that DicomDoodle's annotation files can reference:

* SOP class: Enhanced (or Legacy Converted Enhanced) MR -> MR Image Storage
  (PS3.3 A.4), Enhanced CT -> CT Image Storage (A.3), anything else ->
  Secondary Capture Image Storage (A.8.1).
* The frame's functional groups — the shared ones, then the frame's own —
  become ordinary attributes: position, orientation, pixel spacing, slice
  thickness, window, rescale, frame type (Image Type), MR timing (TR, TE,
  flip angle, echo train), coils, CT exposure …
* MR Image attributes that Enhanced MR expresses differently are derived
  from it: Scanning Sequence (SE/GR/IR/EP), Sequence Variant, Scan Options,
  Sequence Name, Imaged Nucleus, Imaging Frequency, Acquisition Matrix.
* Attributes that only exist in multi-frame objects are left out. Patient,
  study and top-level private data are copied unchanged; missing Type 2
  attributes are added empty and a malformed Patient's Age is written in
  the AS form, as in the annotation files.
* The slices form a new series (the multi-frame original stays a separate
  object). Series and instance UIDs are derived from the original's UIDs,
  so converting again gives the same UIDs and annotation files of an
  earlier conversion still match.
* Pixel data is written uncompressed (Explicit VR Little Endian).
"""
from __future__ import annotations

import copy
import datetime
import os
from typing import Dict, Iterable, List, Optional

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import (CTImageStorage, ExplicitVRLittleEndian, MRImageStorage, SecondaryCaptureImageStorage,
                         generate_uid)

from .progress import Progress, ensure_progress
from .seg_io import normalized_age

_MR_CLASSES = {"1.2.840.10008.5.1.4.1.1.4", "1.2.840.10008.5.1.4.1.1.4.1", "1.2.840.10008.5.1.4.1.1.4.4"}
_CT_CLASSES = {"1.2.840.10008.5.1.4.1.1.2", "1.2.840.10008.5.1.4.1.1.2.1", "1.2.840.10008.5.1.4.1.1.2.2"}

# Attributes of multi-frame (Enhanced) objects that single-frame images do not have.
_MULTIFRAME_ONLY = (
    "NumberOfFrames", "SharedFunctionalGroupsSequence", "PerFrameFunctionalGroupsSequence",
    "DimensionOrganizationSequence", "DimensionIndexSequence", "DimensionOrganizationType",
    "ReferencedImageEvidenceSequence", "SourceImageEvidenceSequence", "ConcatenationUID",
    "ConcatenationFrameOffsetNumber", "InConcatenationNumber", "InConcatenationTotalNumber",
    "RepresentativeFrameNumber", "FrameIncrementPointer", "StereoPairsPresent", "PresentationLUTShape",
    "PixelPresentation", "VolumetricProperties", "VolumeBasedCalculationTechnique", "ComplexImageComponent",
    "AcquisitionContrast", "ContentQualification", "ImageType",
    # MR pulse sequence and Enhanced MR image modules (their content is mapped to MR Image attributes)
    "PulseSequenceName", "EchoPulseSequence", "MultipleSpinEcho", "MultiPlanarExcitation", "PhaseContrast",
    "TimeOfFlightContrast", "ArterialSpinLabelingContrast", "SteadyStatePulseSequence", "EchoPlanarPulseSequence",
    "SaturationRecovery", "SpectrallySelectedSuppression", "OversamplingPhase", "GeometryOfKSpaceTraversal",
    "SegmentedKSpaceTraversal", "RectilinearPhaseEncodeReordering", "NumberOfKSpaceTrajectories",
    "CoverageOfKSpace", "ResonantNucleus", "KSpaceFiltering", "ApplicableSafetyStandardAgency",
    "ApplicableSafetyStandardDescription", "AcquisitionDuration", "VelocityEncodingAcquisitionSequence",
    "DiffusionBValue", "IsotopeNumber",
    # whole-object values that would be wrong for one frame (recomputed where useful)
    "SmallestImagePixelValue", "LargestImagePixelValue",
)

# Functional group macro -> attributes of its item that classic images have under the same name.
_COPIED_FROM_GROUPS: Dict[str, tuple] = {
    "PixelMeasuresSequence": ("PixelSpacing", "SliceThickness", "SpacingBetweenSlices"),
    "PlanePositionSequence": ("ImagePositionPatient",),
    "PlaneOrientationSequence": ("ImageOrientationPatient",),
    "FrameVOILUTSequence": ("WindowCenter", "WindowWidth", "WindowCenterWidthExplanation", "VOILUTFunction"),
    "PixelValueTransformationSequence": ("RescaleIntercept", "RescaleSlope", "RescaleType"),
    "FrameAnatomySequence": ("AnatomicRegionSequence",),
    "MRTimingAndRelatedParametersSequence": ("RepetitionTime", "FlipAngle", "EchoTrainLength"),
    "MRFOVGeometrySequence": ("PercentSampling", "PercentPhaseFieldOfView", "InPlanePhaseEncodingDirection"),
    "MRAveragesSequence": ("NumberOfAverages",),
    "MREchoSequence": ("EchoNumbers",),
    "MRImagingModifierSequence": ("PixelBandwidth",),
    "MRReceiveCoilSequence": ("ReceiveCoilName",),
    "MRTransmitCoilSequence": ("TransmitCoilName",),
    "CTXRayDetailsSequence": ("KVP", "FilterType", "FocalSpots"),
    "CTReconstructionSequence": ("ConvolutionKernel", "ReconstructionDiameter"),
    "CTGeometrySequence": ("DistanceSourceToDetector", "DistanceSourceToPatient"),
}

_TYPE2 = ("PatientName", "PatientID", "PatientBirthDate", "PatientSex", "StudyDate", "StudyTime",
          "ReferringPhysicianName", "StudyID", "AccessionNumber", "SeriesNumber", "InstanceNumber")


class MultiFrameDicom:
    def __init__(self, path: str):
        self.path = path
        self.ds = pydicom.dcmread(path)
        if "PixelData" not in self.ds:
            raise ValueError("This file has no image data.")
        self.pixels = self.ds.pixel_array
        frames = int(self.ds.get("NumberOfFrames", 1) or 1)
        if frames < 2:
            raise ValueError("This is not a multi-frame 3D DICOM file.")
        if self.pixels.ndim == 2 or (self.pixels.ndim == 3 and self.ds.get("SamplesPerPixel", 1) > 1):
            self.pixels = self.pixels[np.newaxis, ...]

    def __len__(self) -> int:
        return int(self.pixels.shape[0])

    def _frame_group(self, index: int):
        groups = getattr(self.ds, "PerFrameFunctionalGroupsSequence", None)
        return groups[index] if groups is not None and len(groups) > index else None

    def _groups(self, index: int) -> List[Dataset]:
        """The functional groups that apply to a frame: shared first, then the frame's own."""
        out = []
        shared = self.ds.get("SharedFunctionalGroupsSequence")
        if shared:
            out.append(shared[0])
        own = self._frame_group(index)
        if own is not None:
            out.append(own)
        return out

    # --------------------------------------------------------------- display
    def frame_for_display(self, index: int) -> np.ndarray:
        """8-bit frame using the window center/width when available (original logic)."""
        frame = self.pixels[index]
        wc = ww = None
        if "WindowCenter" in self.ds and "WindowWidth" in self.ds:
            wc, ww = self.ds.WindowCenter, self.ds.WindowWidth
        else:
            for group in self._groups(index):
                voi = group.get("FrameVOILUTSequence")
                if voi and "WindowCenter" in voi[0] and "WindowWidth" in voi[0]:
                    wc, ww = voi[0].WindowCenter, voi[0].WindowWidth
        if wc is not None and ww is not None:
            wc = float(wc[0] if isinstance(wc, pydicom.multival.MultiValue) else wc)
            ww = float(ww[0] if isinstance(ww, pydicom.multival.MultiValue) else ww)
            lo, hi = wc - ww / 2, wc + ww / 2
        else:
            lo, hi = float(frame.min()), float(frame.max())
        if hi == lo:
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        frame = np.clip(frame, lo, hi)
        out = ((frame - lo) / (hi - lo)) * 255.0
        return out.astype(np.uint8)

    def frame_info(self, index: int) -> Dict[str, str]:
        position = index + 1
        group = self._frame_group(index)
        content = getattr(group, "FrameContentSequence", None) if group is not None else None
        if content:
            position = content[0].get("InStackPositionNumber", position)
        return {
            "Patient": str(self.ds.get("PatientName", "N/A")),
            "Study date": str(self.ds.get("StudyDate", "N/A")),
            "Dimensions": f"{self.ds.get('Columns', 'N/A')} × {self.ds.get('Rows', 'N/A')}",
            "Stack position": str(position),
            "Frame": f"{index + 1} / {len(self)}",
        }

    # ----------------------------------------------------------------- split
    def default_output_folder(self) -> str:
        directory = os.path.dirname(self.path)
        return os.path.join(directory, os.path.splitext(os.path.basename(self.path))[0])

    @property
    def target_sop_class(self) -> str:
        sop_class = str(self.ds.get("SOPClassUID", ""))
        color = int(self.ds.get("SamplesPerPixel", 1)) > 1
        if sop_class in _MR_CLASSES and not color:
            return MRImageStorage
        if sop_class in _CT_CLASSES and not color:
            return CTImageStorage
        return SecondaryCaptureImageStorage

    def split(self, output_folder: Optional[str] = None, progress: Optional[Progress] = None) -> int:
        """Write one DICOM file per frame; returns the number of files written."""
        progress = ensure_progress(progress)
        output_folder = output_folder or self.default_output_folder()
        os.makedirs(output_folder, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.path))[0]
        total = len(self)
        for index in range(total):
            progress(index, total, f"Saving slice {index + 1} of {total}")
            single = self.single_frame(index)
            single.save_as(os.path.join(output_folder, f"{base_name}_{index + 1}.dcm"), enforce_file_format=True)
        progress(total, total, "Done")
        return total

    def single_frame(self, index: int) -> Dataset:
        """The frame ``index`` as a complete single-frame dataset (see the module notes)."""
        src = self.ds
        sop_class = self.target_sop_class
        new = Dataset()
        for element in src:
            if element.tag.group in (0x0002, 0x7FE0) or element.keyword in _MULTIFRAME_ONLY:
                continue
            new.add(copy.deepcopy(element))

        groups = self._groups(index)
        for group in groups:
            for macro, keywords in _COPIED_FROM_GROUPS.items():
                item = _item(group, macro)
                for keyword in keywords:
                    if item is not None and keyword in item:
                        setattr(new, keyword, copy.deepcopy(item[keyword].value))
            if "ReferencedImageSequence" in group:            # e.g. the localizers
                new.ReferencedImageSequence = copy.deepcopy(group.ReferencedImageSequence)
        self._frame_values(new, groups, index)
        if sop_class == MRImageStorage:
            _mr_values(new, src, groups)
        elif sop_class == CTImageStorage:
            _ct_values(new, groups)
        else:
            new.ConversionType = "WSD"                       # SC Equipment: made on a workstation
            new.Modality = src.get("Modality", "OT")
        if sop_class == MRImageStorage and _is_identity_rescale(new):
            for keyword in ("RescaleIntercept", "RescaleSlope", "RescaleType"):
                if keyword in new:
                    del new[keyword]
        self._identity(new, sop_class, index)
        self._pixels(new, index)
        _complete(new)

        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = new.SOPClassUID
        meta.MediaStorageSOPInstanceUID = new.SOPInstanceUID
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        new.file_meta = meta
        new.preamble = b"\0" * 128
        return new

    def _frame_values(self, new: Dataset, groups: List[Dataset], index: int) -> None:
        for group in groups:
            frame_type = _first_value(group, ("MRImageFrameTypeSequence", "CTImageFrameTypeSequence"), "FrameType")
            if frame_type is not None:
                new.ImageType = list(frame_type)
            anatomy = _item(group, "FrameAnatomySequence")
            if anatomy is not None and anatomy.get("FrameLaterality") in ("R", "L", "U", "B"):
                new.ImageLaterality = anatomy.FrameLaterality
            content = _item(group, "FrameContentSequence")
            stamp = content and (content.get("FrameAcquisitionDateTime") or content.get("FrameReferenceDateTime"))
            if stamp:
                new.AcquisitionDateTime = stamp
                new.AcquisitionDate, new.AcquisitionTime = str(stamp)[:8], str(stamp)[8:].split("+")[0].split("-")[0]
        if "ImageType" not in new:
            new.ImageType = list(self.ds.get("ImageType", ["DERIVED", "SECONDARY"]))[:4]
        if "ContentDate" not in new:
            new.ContentDate = new.get("AcquisitionDate") or self.ds.get("SeriesDate") or self.ds.get("StudyDate", "")
        if "ContentTime" not in new:
            new.ContentTime = new.get("AcquisitionTime") or self.ds.get("SeriesTime") or self.ds.get("StudyTime", "")
        if "ImageOrientationPatient" in new and "ImagePositionPatient" in new:
            cosines = np.array(new.ImageOrientationPatient, dtype=float).reshape(2, 3)
            normal = np.cross(cosines[0], cosines[1])
            new.SliceLocation = f"{float(np.dot(normal, np.array(new.ImagePositionPatient, dtype=float))):.6g}"

    def _identity(self, new: Dataset, sop_class: str, index: int) -> None:
        src = self.ds
        source_uid = str(src.SOPInstanceUID)
        series_seed = str(src.get("SeriesInstanceUID") or source_uid)
        new.SOPClassUID = sop_class
        new.SOPInstanceUID = generate_uid(entropy_srcs=[source_uid, "frame", str(index + 1)])
        new.SeriesInstanceUID = generate_uid(entropy_srcs=[series_seed, source_uid, "single-frame series"])
        if not src.get("FrameOfReferenceUID") and "ImagePositionPatient" in new:
            new.FrameOfReferenceUID = generate_uid(entropy_srcs=[series_seed, "frame of reference"])
        new.InstanceNumber = index + 1
        now = datetime.datetime.now()
        new.InstanceCreationDate, new.InstanceCreationTime = now.strftime("%Y%m%d"), now.strftime("%H%M%S")

    def _pixels(self, new: Dataset, index: int) -> None:
        frame = np.asarray(self.pixels[index])
        if int(self.ds.get("BitsAllocated", 16)) == 1:
            raise ValueError("Converting 1-bit (binary) multi-frame images is not supported.")
        if frame.ndim == 3:                                   # color: decoded to RGB, interleaved
            new.PhotometricInterpretation, new.PlanarConfiguration = "RGB", 0
        new.Rows, new.Columns = int(frame.shape[0]), int(frame.shape[1])
        new.PixelData = np.ascontiguousarray(frame.astype(frame.dtype.newbyteorder("<"))).tobytes()
        if frame.ndim == 2:
            new.SmallestImagePixelValue, new.LargestImagePixelValue = int(frame.min()), int(frame.max())
            vr = "SS" if int(new.get("PixelRepresentation", 0)) else "US"
            new["SmallestImagePixelValue"].VR = new["LargestImagePixelValue"].VR = vr


def _item(group: Dataset, keyword: str) -> Optional[Dataset]:
    seq = group.get(keyword)
    return seq[0] if seq else None


def _first_value(group: Dataset, macros: Iterable[str], keyword: str):
    for macro in macros:
        item = _item(group, macro)
        if item is not None and keyword in item:
            return item[keyword].value
    return None


def _group_value(groups: List[Dataset], macro: str, keyword: str, default=None):
    """The value from the frame's own group if present, else the shared one."""
    value = default
    for group in groups:
        item = _item(group, macro)
        if item is not None and item.get(keyword) not in (None, ""):
            value = item[keyword].value
    return value


def _mr_values(new: Dataset, src: Dataset, groups: List[Dataset]) -> None:
    """MR Image module attributes (PS3.3 C.8.3.1) from their Enhanced MR counterparts."""
    def top(keyword: str, default: str = "") -> str:
        return str(src.get(keyword, default) or default).upper()

    pulse = top("EchoPulseSequence")
    inversion = _group_value(groups, "MRModifierSequence", "InversionRecovery", "NO")
    scanning = []
    if pulse in ("SPIN", "BOTH"):
        scanning.append("SE")
    if inversion == "YES":
        scanning.append("IR")
    if pulse in ("GRADIENT", "BOTH"):
        scanning.append("GR")
    if top("EchoPlanarPulseSequence") == "YES":
        scanning.append("EP")
    if "ScanningSequence" not in new:
        new.ScanningSequence = scanning or ["RM"]            # RM: research mode, when nothing is known

    variant = []
    if top("SegmentedKSpaceTraversal", "SINGLE") != "SINGLE":
        variant.append("SK")
    if _group_value(groups, "MRImagingModifierSequence", "MagnetizationTransfer", "NONE") != "NONE":
        variant.append("MTC")
    steady = top("SteadyStatePulseSequence", "NONE")
    if steady != "NONE":
        variant.append("TRSS" if steady == "TIME_REVERSED" else "SS")
    spoiling = src.get("Spoiling") or _group_value(groups, "MRTimingAndRelatedParametersSequence", "Spoiling", "NONE")
    if str(spoiling).upper() != "NONE":
        variant.append("SP")
    if top("OversamplingPhase", "NONE") != "NONE":
        variant.append("OSP")
    if "SequenceVariant" not in new:
        new.SequenceVariant = variant or ["NONE"]

    options = []
    if _group_value(groups, "CardiacSynchronizationSequence", "CardiacSynchronizationTechnique", "NONE") != "NONE":
        options.append("CG")
    if _group_value(groups, "RespiratorySynchronizationSequence", "RespiratoryMotionCompensationTechnique",
                    "NONE") != "NONE":
        options.append("RG")
    if _group_value(groups, "MRModifierSequence", "PartialFourier", "NO") == "YES":
        direction = _group_value(groups, "MRModifierSequence", "PartialFourierDirection", "PHASE")
        options.append("PFF" if direction == "FREQUENCY" else "PFP")
    if _group_value(groups, "MRModifierSequence", "SpatialPresaturation", "NONE") != "NONE":
        options.append("SP")
    if _group_value(groups, "MRModifierSequence", "FlowCompensation", "NONE") != "NONE":
        options.append("FC")
    if top("SpectrallySelectedSuppression", "NONE") in ("FAT", "FAT_AND_WATER"):
        options.append("FS")
    if "ScanOptions" not in new:
        new.ScanOptions = options

    effective_te = _group_value(groups, "MREchoSequence", "EffectiveEchoTime")
    if effective_te is not None:
        new.EchoTime = f"{float(effective_te):.6g}"
    if "IR" in scanning:
        times = _group_value(groups, "MRModifierSequence", "InversionTimes")
        if times is not None:
            first = times[0] if isinstance(times, (list, tuple, pydicom.multival.MultiValue)) else times
            new.InversionTime = f"{float(first):.6g}"
        elif "InversionTime" not in new:
            new.InversionTime = ""
    if src.get("PulseSequenceName"):
        new.SequenceName = src.PulseSequenceName
    if src.get("ResonantNucleus"):
        new.ImagedNucleus = src.ResonantNucleus
    frequency = _group_value(groups, "MRImagingModifierSequence", "TransmitterFrequency")
    if frequency is not None:
        new.ImagingFrequency = f"{float(frequency):.8g}"
    steps_f = _group_value(groups, "MRFOVGeometrySequence", "MRAcquisitionFrequencyEncodingSteps")
    steps_p = _group_value(groups, "MRFOVGeometrySequence", "MRAcquisitionPhaseEncodingStepsInPlane")
    if steps_f is not None and steps_p is not None:
        rows = new.get("InPlanePhaseEncodingDirection") == "ROW"
        new.AcquisitionMatrix = [0, int(steps_f), int(steps_p), 0] if rows else [int(steps_f), 0, 0, int(steps_p)]
    for keyword in ("MRAcquisitionType", "RepetitionTime", "EchoTime", "EchoTrainLength"):
        if keyword not in new:
            setattr(new, keyword, "")                        # Type 2: present, value unknown


def _ct_values(new: Dataset, groups: List[Dataset]) -> None:
    """CT Image module attributes (PS3.3 C.8.2.1) from their Enhanced CT counterparts."""
    exposure = {"ExposureTimeInms": "ExposureTime", "XRayTubeCurrentInmA": "XRayTubeCurrent",
                "ExposureInmAs": "Exposure"}
    for enhanced, classic in exposure.items():
        value = _group_value(groups, "CTExposureSequence", enhanced)
        if value is not None:
            setattr(new, classic, int(round(float(value))))
    new.RescaleIntercept = new.get("RescaleIntercept", 0)
    new.RescaleSlope = new.get("RescaleSlope", 1)
    for keyword in ("KVP", "AcquisitionNumber"):
        if keyword not in new:
            setattr(new, keyword, "")                        # Type 2


def _is_identity_rescale(ds: Dataset) -> bool:
    try:
        return float(ds.get("RescaleIntercept", 0)) == 0 and float(ds.get("RescaleSlope", 1)) == 1
    except (TypeError, ValueError):
        return False


def _complete(new: Dataset) -> None:
    """Type 2 attributes present (empty if unknown) and a valid Patient's Age."""
    for keyword in _TYPE2:
        if keyword not in new:
            setattr(new, keyword, "")
    if "PatientAge" in new:
        age = normalized_age(str(new.PatientAge))
        if age:
            new.PatientAge = age
        else:
            del new.PatientAge
