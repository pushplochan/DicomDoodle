"""
Pure application logic — no Qt widgets live in this package.

Every module here can be used from a plain Python script or notebook (for
example to batch-convert or validate annotations), which also keeps the GUI
code in ``dicomdoodle.ui`` thin and easy to follow.

    progress.py           shared progress / cancel vocabulary
    labels.py             label names, colors and pixel values (+ user labels)
    dicom_series.py       load a folder of DICOM slices
    seg_io.py             read / write DICOM SEG annotation files
    gap_fill.py           boundary gap-closing used when saving (original algorithm)
    mask_ops.py           label-map helpers: boundaries, contours, overlays
    annotation.py         per-slice annotation (AI regions + hand-drawn outlines) with undo / redo,
                          which slices need blank masks
    ai_models.py          the AI model manager (lesion detector, fallback detector, SAM)
    ai/                   the model card, CPU or graphics card, ONNX Runtime and PyTorch models
    ontology.py           BioPortal search and DICOM coded concepts
    volume_calculator.py  lesion volume from annotated series (Multiplanar & 3D Viewer, right panel)
    multiframe.py         split a multi-frame DICOM into single-frame files
    reconstruction3d.py   build the 3D model of the Multiplanar & 3D Viewer
    tumor_metrics.py      volume / surface / sphericity / texture metrics
    skull_strip.py        HD-BET skull stripping (hdbet_runner.py runs it in its own process)
    tumour_range.py       "Find tumour range" of the main window's batch panel
    mpr/                  multiplanar reconstruction, registration, SEG transfer, tumour detection
"""
