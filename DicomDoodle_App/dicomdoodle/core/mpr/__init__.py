"""
Multiplanar reconstruction (MPR) core — the original ``mpr_core.py``, split
by topic, with the updates of its zero_shot_tumor version. Everything is
re-exported here, so ``from dicomdoodle.core import mpr`` offers the same
functions the original module did.

    volume.py         resampling methods, Volume, slice extraction, windowing
    through_plane.py  through-plane reconstruction (a display algorithm)
    loading.py        find the volumes in a folder and load one (reoriented to LPS)
    registration.py   identity / rigid / deformable / ANTs registration
    segmentation.py   SEG -> label volume, label transfer, slice-wise saving
    overlay.py        draw labels on slices, Dice / IoU
    tumour_detect.py  the "Detect" button: tumour detector + SAM 2.1 in 3D
                      (imported on its own, it needs the AI models)
"""
from .loading import (SeriesInfo, detect_acquisition_plane, find_series, find_series_ids,  # noqa: F401
                      get_acquisition_plane, get_series_notes, load_dicom_series, series_files)
from .overlay import (blend_label_boundary, blend_label_overlay, dice_coefficient, gray_to_rgb,  # noqa: F401
                      iou_coefficient)
from .registration import (HAVE_ANTSPY, RegistrationBackend, RegistrationResult, register_deformable_sitk,  # noqa: F401
                           register_identity, register_rigid, register_rigid_antspy, register_rigid_mi)
from .segmentation import (Segmentation, SegmentInfo, find_seg_files, find_series_seg_files,  # noqa: F401
                           load_seg_as_labelmap, resample_label_array_like, resample_label_like,
                           save_transferred_seg_slice_wise, transfer_labelmap)
from .through_plane import reconstruct_through_plane  # noqa: F401
from .volume import (MAX_RESAMPLED_VOXELS, InterpMethod, Plane, Volume, build_volume, display_aspect,  # noqa: F401
                     extract_label_slice, extract_slice, intensity_range, plane_extent, window_to_uint8)
