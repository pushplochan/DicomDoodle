# DicomDoodle 2 — modular application

A redesigned DicomDoodle with a modern interface, light and dark themes,
unlimited custom labels, and a code base split into small, focused modules.
Every feature of the original tool is here, and annotation files stay
compatible with the original: same `<slice>_seg.dcm` DICOM Segmentation
files, same segment names, same pixel content. They now also follow the
DICOM standard's Segmentation rules (PS3.3 A.51, checked against the 2026d
edition), so PACS and web viewers such as OHIF read them.

## Run it

```bash
cd DicomDoodle/DicomDoodle_App
pip install -r requirements.txt      # once
python main.py                       # or: python main.py "path/to/dicom folder"
```

The logo images are found automatically in the parent `DicomDoodle/` folder.
The AI models come from the **model card** (`DicomDoodle-ModelCard-<version>.ddcard`),
a separate download: DicomDoodle finds it in Downloads, on the Desktop, in
Documents, next to the program or in the parent `DicomDoodle/` folder,
installs it once and loads its models at every start (or use **File ▸ Load
Model Card…**). Without a card the AI is off: manual annotation only. The models
run with ONNX Runtime (`pip install onnxruntime`, or `onnxruntime-gpu` /
`onnxruntime-directml` for a graphics card), on the processor by default; switch
to a graphics card with the AI button of the status bar. How a card is made:
[`scripts/make_model_card.py`](scripts/make_model_card.py).

**Windows, without Python:** download `DicomDoodle-<version>-Setup.exe` from the
[Releases](https://github.com/pushplochan/DicomDoodle/releases) page and run it,
and the model card from the release `model-card-<version>`. How they are built
and released: [packaging/README.md](packaging/README.md).

## New to DicomDoodle?

* **User Guide (F1)** — a built-in, illustrated and searchable guide written
  for every background: a five-minute Quick Start, one chapter per feature with
  annotated screenshots and diagrams, troubleshooting and a glossary. Links in
  the guide open the tools directly, and the (?) buttons in panels and tool
  windows open the matching chapter.
  The screenshots show real, anonymized scans (dates and IDs removed);
  `scripts/make_guide_images.py` makes them from a series you choose.
* **Help ▸ References & Citations** — how to cite DicomDoodle, and the papers
  and software it builds on, with DOIs, copy buttons and BibTeX export.

Preferences (theme, your labels, recent folders, window layout) are stored in
`~/.config/DicomDoodle/settings.json` (Linux), `%APPDATA%\DicomDoodle\settings.json`
(Windows) or `~/Library/Application Support/DicomDoodle/settings.json` (macOS).

## What's new for users

| Area | Original | DicomDoodle 2 |
| --- | --- | --- |
| Layout | three rows of 20 buttons | toolbar + menus, side panels for labels, display, batch processing and series info |
| Drawing by hand | separate Manual Draw window | **pen and eraser right on the image** in the main window (tools above the image); hand-drawn outlines share undo, saving and blank masks with AI outlines; *Edit saved outline* corrects a saved file |
| Lesion volume | separate Lesion Volume Calculator | in the right-hand panel of the **Multiplanar & 3D Viewer**, next to the 3D model and the 3D lesion metrics; unannotated series no longer lower the average |
| 3D model and multiplanar view | two windows: *3D Model Viewer* (two 3D views side by side) and *Multiplanar MRI View* | **one window, the Multiplanar & 3D Viewer**: axial / coronal / sagittal / 3D views; left panel with a *Multiplanar* tab (series, registration, display, annotation) and a *3D view* tab (what the 3D view shows — original MRI, MRI with annotation, skull stripped — and the 3D Model Viewer's settings); right panel as in the 3D Model Viewer (series folders, Build 3D model, lesion volume, 3D lesion metrics) |
| Volume processing | classical skull strip, z-score, connected components; HD-BET was asked for the GPU whenever CuPy was installed, which fails where PyTorch cannot use the GPU (as on this server) and the classical method was used instead | only **HD-BET skull stripping**, for the 2D views and the 3D view: run once per scan in its own process (Cancel works, about 40 s on a CPU) and shared by every series and grid of that session |
| Theme | fixed light gradient | **dark and light theme** (Ctrl+T), image views always dark |
| Labels | six fixed color buttons | six built-in labels **plus your own labels in any color** (full color palette), keys 1–9, import/export |
| Responsiveness | AI ran on the GUI thread (window froze) | AI, saving and batch jobs run in the background with progress, time left and **Cancel** |
| Unsaved work | lost when changing slice | kept per slice, multi-level undo/redo, “unsaved” markers, prompt before closing |
| Blank masks | set two slice ranges, then “Save Blank Masks” | **automatic**: annotate only the lesion slices — each save gives every slice without a `_seg.dcm` a blank mask (can be switched off) |
| Overview | — | slice timeline showing saved slices, outlines and blank masks on disk, unsaved edits and the auto-detect range |
| Safety | silently overwrote earlier `_seg.dcm` | asks before replacing outlines from earlier sessions (can be turned off); blank masks are replaced freely |
| Feedback | console prints | toasts, activity log, status bar (cursor value, zoom, AI status) |
| Saved annotations | — | shown as a faint layer on each slice; all viewers zoom and pan |
| Standard files | scanner listed as the SEG's maker, patient name altered (`Name^`), every label coded *Tissue / Connective tissue* | patient and study copied unchanged, DicomDoodle as maker, each label with standard codes (e.g. *Neoplasm*, *Edema*), its color and how it was made (manual / semi-automatic / automatic); readable by PACS and OHIF |
| Ontology | BioPortal only, internet required | **DICOM's own code list first** (pydicom, offline, about 13,000 codes), BioPortal when it has no match; codes written with DICOM's scheme names (SCT, NCIt …); terms for all labels or one label |
| Multi-frame files | split with scanner details partly lost (no echo time, wrong image type, always MR, compressed files broken) | converted to **standard single-frame MR / CT images**: every frame's geometry, window, rescale and MR parameters kept, new series with stable UIDs, pixel data identical |
| Auto-detect range | Start and End set by hand only | **Find tumour range**: from one slice, the tumour detector's boxes on that slice and on the two other views through it (coronal and sagittal for an axial series) give the first and last slice in a few seconds; the panel previews both views with the boxes and the range (uses the Multiplanar view's loader and detector) |
| Multiplanar view: finding the tumour | — | **Detect** on each view: the tumour detector + SAM 2.1 find the tumour in 3D from the slice shown (a click chooses it) and outline it on every slice of the series; shown as *Detected tumour* with its 3D box (from the `zero_shot_tumor` work; Dice 0.95 against the expert outline on the guide's example series) |
| Multiplanar view: loading | one series per folder, GDCM only | any DICOM image volume: files without `.dcm`, multi-frame (Enhanced) files, series split into echoes / stacks / blocks, duplicates read once, missing slices shown blank in place, pydicom when GDCM fails; unusual series noted in the status bar |
| Multiplanar view: annotations | outlines traced in open pieces (e.g. *Tumor & Brain Boundary* + *Tumor & Dura Boundary*) stayed bare lines, so reformats showed stripes (data_AIIMS 00021: 17.9 mL loaded instead of 83.9 mL) | pieces filled together and small gaps bridged: a patient's axial, coronal and sagittal annotations now agree within 14 %; outlines keep an even width on thick-slice reformats |
| Multiplanar view: display | — | **Through-plane reconstruction**: refines only the slice direction (B-spline + slice profile, no AI) — sharper reformats than B-spline against each patient's own thin-slice scan |

Also fixed along the way: the original could crash when a source DICOM lacked
FrameOfReferenceUID or SoftwareVersions; the overlay viewer showed “Magenta”
annotations as nothing; multi-label annotation files were skipped by the
volume calculator and spread over neighbouring slices in the 3D viewer; the
app failed to start without `ontoportal_client`, and its ontology search failed
with `ontoportal_client` 0.0.5 or newer (whose `search()` changed); “Definition and synonyms”
always showed the last search result; opening a second tool could close the first.

## Keyboard & mouse (main window)

| Keys | Action |
| --- | --- |
| Ctrl+O / Ctrl+S / Ctrl+Shift+S | open folder / save slice / save all modified slices |
| Mouse wheel, ← → ↑ ↓, PgUp/PgDn, Home/End | change slice |
| Left-drag on the image | Box tool: draw a SAM box (drag to move, corners to resize) |
| B / P / E / H, [ / ] | box / pen / eraser / pan tool, smaller / bigger pen or eraser |
| Enter or right-click ▸ Detect | segment the box with SAM |
| Ctrl+D | Auto Detect (YOLO → SAM) |
| 1 … 9 | choose the active label |
| Ctrl+Z / Ctrl+Y / Ctrl+R | undo / redo / reset slice |
| Ctrl + wheel, middle-drag or Space + drag | zoom, pan |
| Ctrl+M / Ctrl+L | pen tool / edit the saved outline of the slice |
| Ctrl+P / Ctrl+T / F1 | plot results / theme / User Guide |

**Help ▸ Keyboard Shortcuts** lists everything.

## How the code is organised

```
DicomDoodle_App/
├── main.py                    entry point (python main.py; also of DicomDoodle.exe)
├── scripts/
│   └── make_guide_images.py   re-creates the User Guide screenshots (real anonymized series, real AI)
├── packaging/                 Windows installer: PyInstaller recipe, Inno Setup script, model upload
│                              (built by .github/workflows/windows-installer.yml, see its README.md)
└── dicomdoodle/
    ├── app.py                 builds QApplication, theme, splash, logging, main window
    ├── resources.py           finds model weights, images and the guide's screenshots
    ├── settings.py            JSON preferences
    ├── selftest.py            main.py --self-test: modules, SEG files, AI models, HD-BET all load
    ├── assets/guide/          the User Guide screenshots (PNG)
    │
    ├── help/                  User Guide text — plain Python strings, no Qt
    │   ├── contents.py        table of contents (parts → chapters)
    │   ├── part_*.py          the chapters, in a small markup (see ui/guide/markup.py)
    │   ├── shortcuts.py       keyboard shortcuts (guide + shortcuts dialog)
    │   └── references.py      how to cite DicomDoodle + bibliography (APA text, BibTeX)
    │
    ├── core/                  pure logic — no Qt widgets, usable from scripts
    │   ├── labels.py          SegLabel + LabelRegistry (built-in + user labels)
    │   ├── dicom_series.py    load a folder of slices; annotation file naming
    │   ├── seg_io.py          write / read DICOM SEG files (standard Segmentation IOD)
    │   ├── coding.py          coded concepts: DICOM code dictionary search, scheme names
    │   ├── gap_fill.py        original boundary gap-closing algorithm
    │   ├── mask_ops.py        boundaries, contours, overlays
    │   ├── annotation.py      per-slice AI regions + hand-drawn outlines with undo/redo, file state,
    │   │                      blank-mask targets
    │   ├── ai_models.py       the AI model manager: lesion detector, fallback detector, SAM
    │   ├── ai/                devices.py (CPU or graphics card), model_files.py (what a model file
    │   │                      is, from its content), model_card.py (find / install the model card,
    │   │                      best model + format per purpose), onnx_models.py (ONNX Runtime),
    │   │                      torch_models.py (PyTorch / Ultralytics)
    │   ├── tumour_range.py    "Find tumour range": the series' 3D volume + the MPR detector's boxes
    │   ├── ontology.py        BioPortal search (second source of coded concepts)
    │   ├── volume_calculator.py  lesion volume per series (right panel of the Multiplanar & 3D Viewer)
    │   ├── multiframe.py, reconstruction3d.py, tumor_metrics.py
    │   ├── skull_strip.py     HD-BET skull stripping (+ hdbet_runner.py, run in its own process)
    │   └── mpr/               MPR volumes (+ through-plane reconstruction), loading any image volume,
    │                          registration, SEG transfer, tumour_detect.py (the Detect button)
    │
    ├── workers/
    │   ├── tasks.py           run_task(): any function in a thread, with progress/cancel
    │   └── jobs.py            annotator jobs: detect, segment, save (+ automatic blank masks), batches
    │
    └── ui/
        ├── theme.py           light/dark tokens → palette + style sheet
        ├── icons.py           vector icons tinted by the theme
        ├── label_manager.py   shared label list with change signals
        ├── widgets/           canvas with box/pen/eraser/pan tools, tool strip, slice timeline, panels, toasts …
        ├── dialogs/           label editor, progress, results, preferences, about, references …
        ├── guide/             the User Guide window: markup → HTML, diagrams, screenshots, search
        ├── main/              main window: window.py, actions.py, annotation.py, drawing.py, batch.py, tools.py,
        │                      ai.py (model card, CPU / graphics card, the AI button of the status bar)
        └── tools/             overlay_viewer, ontology_search, multiframe_converter,
                               mpr/ (the Multiplanar & 3D Viewer window), viewer3d/ (its 3D view,
                               3D settings and lesion volume panel)
```

### Guidelines for contributors

* **Logic goes in `core/`, widgets in `ui/`.** A core function never touches Qt,
  so it can be tested from a script.
* **Long work runs in the background:** `run_task(fn, *args, on_result=…, on_progress=…)`.
  `fn` receives a progress context as its first argument; pass it to core functions
  as their `progress` parameter so Cancel works.
* **Connect long-lived signals (`themeChanged`, `labelsChanged`) to bound methods**,
  not lambdas, so Qt disconnects them when a window closes.
* **Colors come from the theme** (`theme_color("accent")`, QSS object names such as
  `Card`, `Muted`, `SectionTitle`, and `set_role(button, "primary")`).
* **Labels:** pixel values 1–6 are the original labels and must not change;
  user labels start at 7. `LabelRegistry.name_for(value)` gives the DICOM segment name.
* A new tool window: add a module under `ui/tools/`, then one method in
  `ui/main/tools.py` and one action in `ui/main/actions.py`.
* **Documentation lives in `help/`.** To change the guide, edit the chapter text
  in `help/part_*.py`; new chapters go into `help/contents.py`. Diagrams are
  drawn in `ui/guide/diagrams.py`. After UI changes, re-create the screenshots:
  `python scripts/make_guide_images.py` (on a Linux server:
  `xvfb-run -a -s "-screen 0 1600x1000x24" python scripts/make_guide_images.py`).
