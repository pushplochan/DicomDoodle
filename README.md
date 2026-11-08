# DicomDoodle

DicomDoodle annotates brain lesions on MRI DICOM series, with AI help (YOLO + SAM 2.1),
and saves them as standard DICOM Segmentation files (`<slice>_seg.dcm`).

## Codebase structure

```
DicomDoodle/
├── .github/workflows/windows-installer.yml   builds the Windows installer when a v* tag is pushed
├── LICENSE                                   GNU GPL v3
└── DicomDoodle_App/
    ├── main.py                    start the app: python main.py
    ├── requirements.txt           Python packages
    ├── scripts/                   make_guide_images.py (User Guide screenshots), make_model_card.py (model card)
    ├── packaging/                 Windows installer: PyInstaller spec, Inno Setup script, model card upload
    ├── tests/                     automated tests (pytest)
    └── dicomdoodle/
        ├── app.py                 builds the application, theme, splash and main window
        ├── resources.py           finds the logo images and the guide's screenshots
        ├── settings.py            preferences (JSON)
        ├── selftest.py            main.py --self-test
        ├── assets/                logo, splash, window icon; guide/ = User Guide screenshots
        ├── help/                  User Guide text, keyboard shortcuts, references
        ├── core/                  logic without any GUI
        │   ├── dicom_series.py    loading a folder of slices
        │   ├── seg_io.py          writing / reading DICOM SEG files
        │   ├── coding.py          DICOM codes for labels
        │   ├── ontology.py        BioPortal search
        │   ├── labels.py          built-in and custom labels
        │   ├── annotation.py      per-slice outlines, undo/redo
        │   ├── mask_ops.py        boundaries, contours, overlays
        │   ├── ai_models.py       AI model manager (detector, SAM)
        │   ├── ai/                model card, ONNX / PyTorch runners, CPU or GPU
        │   ├── tumour_range.py    Find tumour range
        │   ├── skull_strip.py     HD-BET skull stripping (hdbet_runner.py runs it in its own process)
        │   ├── volume_calculator.py, reconstruction3d.py, tumor_metrics.py, multiframe.py
        │   └── mpr/               multiplanar volumes, loading, registration, Detect
        ├── workers/               background jobs: detect, segment, save, batches
        └── ui/
            ├── main/              main window
            ├── widgets/           image canvas and drawing tools, panels
            ├── dialogs/           labels, preferences, progress, about …
            ├── guide/             User Guide window
            └── tools/             Multiplanar & 3D Viewer, overlay viewer, ontology search, multi-frame converter
```

## Get it from GitHub

**Windows installer:** download `DicomDoodle-<version>-Setup.exe` from
[Releases](https://github.com/pushplochan/DicomDoodle/releases/latest) (or from
[dicomdoodle.github.io](https://dicomdoodle.github.io/)).

**Source code:**

```bash
git clone https://github.com/pushplochan/DicomDoodle.git
cd DicomDoodle/DicomDoodle_App
```

**AI models (both ways):** download `DicomDoodle-ModelCard-<version>.ddcard` from the release
[model-card-1.0](https://github.com/pushplochan/DicomDoodle/releases/tag/model-card-1.0).
Without it, DicomDoodle works for manual annotation only.

## Install

**Windows installer:** run `DicomDoodle-<version>-Setup.exe`. If Windows SmartScreen
appears, click *More info ▸ Run anyway*.

**Source code** (Python 3.10 or newer):

```bash
pip install -r requirements.txt
pip install hd-bet          # optional: skull stripping in the Multiplanar & 3D Viewer
```

**Model card:** leave the `.ddcard` file in Downloads, on the Desktop or in Documents.
DicomDoodle finds and installs it at the next start. You can also use
**File ▸ Load Model Card…** or drag the file onto the window.

## Use

1. Start DicomDoodle: the Start menu entry, or `python main.py` in `DicomDoodle_App`.
2. **File ▸ Open DICOM Folder…** (Ctrl+O): choose a folder with one DICOM series.
3. Choose a label (keys 1–9).
4. Outline the lesion:
   - **Auto Detect** (Ctrl+D) finds and outlines it.
   - Or draw a box around it (B) and press Enter to outline it with SAM.
   - Or draw it by hand with the pen (P) and eraser (E).
5. **Save** (Ctrl+S): a `<slice>_seg.dcm` file appears next to each slice. Slices without
   a lesion get blank masks automatically.
6. **Tools ▸ Multiplanar & 3D Viewer**: axial, coronal, sagittal and 3D views, the 3D model
   and the lesion volume.

Press **F1** for the illustrated User Guide.

## Cite

If DicomDoodle helps your work, please cite the article that describes it:

Kumar, P. L., Shahin, M., Kumar, R., Suri, V., Gaikwad, S. B., Deb, B., Salunke, P., Bajaj, A.,
Srinivas, D., Sahu, R. N., Jaiswal, A. K., Vilanilam, G. C., Kedia, S., Vyas, T., & Pal, D. (2026).
DicomDoodle: Human-in-the-loop annotation of 3D MRI DICOM images with AI support and DICOM-SEG
compatibility. *Journal of Imaging Informatics in Medicine*. Advance online publication.
https://doi.org/10.1007/s10278-026-02485-y

```bibtex
@article{kumar2026dicomdoodle,
  author = {Kumar, P. L. and Shahin, M. and Kumar, R. and Suri, V. and Gaikwad, S. B. and Deb, B. and Salunke, P. and Bajaj, A. and Srinivas, D. and Sahu, R. N. and Jaiswal, A. K. and Vilanilam, G. C. and Kedia, S. and Vyas, T. and Pal, D.},
  title = {{DicomDoodle: Human-in-the-loop annotation of 3D MRI DICOM images with AI support and DICOM-SEG compatibility}},
  year = {2026},
  journal = {Journal of Imaging Informatics in Medicine},
  note = {Advance online publication},
  doi = {10.1007/s10278-026-02485-y}
}
```

Please also name the version you used (for example "DicomDoodle 2.1.0"). The same citation,
with BibTeX export, is in the app under **Help ▸ References & Citations**, and GitHub's
*Cite this repository* button reads it from `CITATION.cff`.

## Acknowledgements

DicomDoodle is built on these open-source tools. We thank their authors and communities.

| Tool | Used for | Citation |
| --- | --- | --- |
| [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) | lesion outline from a box | [Kirillov et al., ICCV 2023](https://doi.org/10.1109/ICCV51070.2023.00371) |
| [SAM 2](https://github.com/facebookresearch/sam2) | SAM 2.1 segmentation model | [Ravi et al., 2024](https://arxiv.org/abs/2408.00714) |
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | lesion detection | [Jocher et al., 2023](https://github.com/ultralytics/ultralytics) |
| [YOLOE](https://github.com/THU-MIG/yoloe) | fallback detector | [Wang et al., 2025](https://arxiv.org/abs/2503.07465) |
| [PyTorch](https://pytorch.org) | neural networks | [Paszke et al., NeurIPS 2019](https://arxiv.org/abs/1912.01703) |
| [ONNX Runtime](https://onnxruntime.ai) | running the model card on CPU or GPU | [ONNX Runtime developers, 2021](https://onnxruntime.ai) |
| [pydicom](https://github.com/pydicom/pydicom) | reading and writing DICOM files | [Mason, Medical Physics 2011](https://doi.org/10.1118/1.3611983) |
| [highdicom](https://github.com/ImagingDataCommons/highdicom) | DICOM Segmentation files | [Bridge et al., J Digit Imaging 2022](https://doi.org/10.1007/s10278-022-00683-y) |
| [GDCM](https://github.com/malaterre/GDCM) | decoding compressed DICOM | [Malaterre et al.](https://github.com/malaterre/GDCM) |
| [SimpleITK](https://simpleitk.org) | 3D volumes, resampling, registration | [Lowekamp et al., Front Neuroinform 2013](https://doi.org/10.3389/fninf.2013.00045) |
| [NumPy](https://numpy.org) | image and mask arithmetic | [Harris et al., Nature 2020](https://doi.org/10.1038/s41586-020-2649-2) |
| [SciPy](https://scipy.org) | gap closing, morphology | [Virtanen et al., Nature Methods 2020](https://doi.org/10.1038/s41592-019-0686-2) |
| [scikit-image](https://scikit-image.org) | lesion texture metrics | [van der Walt et al., PeerJ 2014](https://doi.org/10.7717/peerj.453) |
| [OpenCV](https://opencv.org) | edges and contours | [Bradski, Dr. Dobb's Journal 2000](https://opencv.org) |
| [Pillow](https://python-pillow.org) | image files | [Clark, 2015](https://pillow.readthedocs.io) |
| [VTK](https://vtk.org) | 3D model and volume rendering | [Schroeder et al., Kitware 2006](https://vtk.org) |
| [Matplotlib](https://matplotlib.org) | multiplanar slice views | [Hunter, Comput Sci Eng 2007](https://doi.org/10.1109/MCSE.2007.55) |
| [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) | user interface | [Riverbank Computing](https://www.riverbankcomputing.com/software/pyqt/) |
| [HD-BET](https://github.com/MIC-DKFZ/HD-BET) | skull stripping | [Isensee et al., Hum Brain Mapp 2019](https://doi.org/10.1002/hbm.24750) |
| [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) | network framework of HD-BET | [Isensee et al., Nature Methods 2021](https://doi.org/10.1038/s41592-020-01008-z) |
| [ANTsPy](https://github.com/ANTsX/ANTsPy) | optional registration engine | [Avants et al., NeuroImage 2011](https://doi.org/10.1016/j.neuroimage.2010.09.025) |
| [BioPortal](https://bioportal.bioontology.org) | ontology term search | [Noy et al., NAR 2009](https://doi.org/10.1093/nar/gkp440); [Vendetti et al., NAR 2025](https://doi.org/10.1093/nar/gkaf402) |
| [OHIF Viewer](https://ohif.org) | checking the Segmentation files | [Ziegler et al., JCO CCI 2020](https://doi.org/10.1200/CCI.19.00131) |
| [PyInstaller](https://pyinstaller.org) | building the Windows program | [PyInstaller](https://pyinstaller.org) |
| [Inno Setup](https://jrsoftware.org/isinfo.php) | the Windows installer | [Jordan Russell](https://jrsoftware.org/isinfo.php) |

The full reference list with BibTeX is in the app: **Help ▸ References & Citations**.
