"""User Guide — Getting started."""
from .chapter import Chapter

WELCOME = Chapter(
    "welcome", "Welcome to DicomDoodle", "book",
    "A friendly tool for outlining lesions on MRI scans and saving them as standard medical files — "
    "no programming needed.",
    """
DicomDoodle helps you **mark (annotate) regions on MRI slices** — for example the edge of a brain tumour — and saves every annotation as a standard medical-imaging file that other software can read. Artificial intelligence (AI) does most of the drawing; you check it and correct it where needed.

![The whole journey in six steps](diagram:workflow)

## Who is it for?
- **Radiologists and clinicians** who want to mark findings quickly and consistently.
- **Researchers and students** preparing images for a study or for training AI models.
- **Anyone new to medical images** — every technical word is explained the first time it appears, and the [Glossary](chapter:glossary) collects them all.

## What you need
- A folder of **DICOM** files from one MRI series. DICOM is the standard file format of medical scanners; usually there is one file per slice.
- Nothing else: the AI models come with DicomDoodle. A computer with an NVIDIA graphics card (GPU) makes the AI faster, but everything also works without one.

> NOTE: The pictures in this guide show a real, anonymized brain MRI: an axial contrast-enhanced T1 series of 166 slices with a large tumour on slices 114 to 151. Open one of your own series and follow along.

## How this guide is organised
| Part | What you will learn |
|---|---|
| Getting started | The main window, opening scans, moving through slices |
| Annotating | Labels and colors, AI segmentation, drawing by hand with the pen and eraser |
| Saving & batch work | What is saved, automatic blank masks, and processing many slices at once |
| Reviewing & measuring | Checking saved work, standard medical terms, lesion volume (in the Multiplanar & 3D Viewer) |
| Advanced tools | 3D models, multiplanar views, file conversion, preferences |
| Reference | Keyboard shortcuts, troubleshooting, glossary |

> TIP: Press [[F1]] at any time to open this guide. Panels and tool windows have a {icon:help} button that opens the matching chapter directly.
""", "introduction overview start what is")

QUICK_START = Chapter(
    "quick-start", "Quick start: your first annotation in 5 minutes", "zap",
    "Follow these steps once with one of your own series and you will know the core workflow.",
    """
Use any MRI series that shows a lesion. Each step names the button to press; the slice numbers in brackets are those of the example series in the pictures (166 axial slices, a tumour on slices 114 to 151).

> TIP: Practise on a copy of the folder. DicomDoodle never changes your images, but it does write annotation files next to them.

## 1. Open your series
1. On the start page, click **Open DICOM folder…** — or [open one now](action:open) — and choose the folder that holds the slices of one series.
2. The slices load in a few seconds and the first slice appears in the middle of the window.

## 2. Find the lesion
1. Move the mouse over the image and **roll the mouse wheel** (or press [[→]]) to move through the slices.
2. Stop on a slice where the lesion is clearly visible (in the example, **slice 133**: a bright mass on the left of the picture). That is the lesion you will outline.

## 3. Let the AI outline it
1. In the **Labels** panel on the left, make sure **Tumor | Brain** (red) is selected.
2. Click {icon:sparkles} **Auto Detect** in the toolbar, or press [[Ctrl+D]].
3. After a few seconds a red outline hugs the lesion, and a green dashed box shows where the AI found it. The badge at the top right of the image now reads **Unsaved changes**.

## 4. Save it
1. Click {icon:save} **Save**, or press [[Ctrl+S]].
2. A message confirms the file name — the slice's own name plus `_seg.dcm` — and the badge turns green: **Saved**.
3. On this first save the message also says that **blank masks** were saved for all the other slices: files that say “no lesion here”. From now on you only annotate slices that show the lesion — saving one replaces its blank mask.

## 5. Try a box yourself
1. Go to a neighbouring slice with the mouse wheel (in the example, slice 128).
2. **Press and drag** with the left mouse button to draw a box around the lesion.
3. Press [[Enter]]: SAM, the segmentation AI, outlines what is inside your box.
4. Press [[Ctrl+S]] to save.

> TIP: Made a mistake? [[Ctrl+Z]] undoes the last change on the slice, and {icon:reset} **Reset** clears the slice completely. Prefer to draw yourself? Press [[P]] for the pen — see [Drawing by hand](chapter:manual-draw).

## 6. Look at your work
1. Click {icon:image} **Results** to see the slice with the region filled in, and with its outline as saved.
2. Click {icon:layers} **Annotated DICOM Viewer** on the right of the toolbar to review every saved slice of the folder.

That is the everyday workflow. The next chapters explain each part in detail, starting with [the main window](chapter:main-window).
""", "tutorial first steps begin practise", short="Quick start (5 minutes)")

MAIN_WINDOW = Chapter(
    "main-window", "The main window", "columns",
    "A tour of everything you see after opening a series.",
    """
![The main window with a contrast-enhanced T1 series open](shot:overview)

| | Area | What it does |
|---|---|---|
| {n:1} | Menu bar | Every command, grouped into File, Edit, View, Annotate, Batch, Tools and Help. |
| {n:2} | Toolbar | The most used buttons: Open, Save, Auto Detect, Detect Box, Results, Undo, Redo, Reset, zoom — and on the right the tools and the light / dark switch. |
| {n:3} | Tools | What the left mouse button does: **Box** (for SAM), **Pen**, **Eraser** or **Pan** — see [Drawing by hand](chapter:manual-draw). |
| {n:4} | Labels | The labels (names and colors). The selected label is used for the next outline. |
| {n:5} | Display | How annotations are drawn, and the brightness and contrast of the picture. |
| {n:6} | Image view | The current slice. The top-left corner shows the slice number and file name. |
| {n:7} | Status badge | **Unsaved changes** (orange), **Saved** (green), **Annotation file on disk** (blue) or **Blank mask** (grey). |
| {n:8} | Slice timeline | Moves through the slices; colored marks show saved slices, blank masks and the auto-detect range. |
| {n:9} | Side panels | **Batch Processing** (auto-detect range, blank masks) and **Series Info** (patient, spacing, current slice), as tabs. |
| {n:10} | Status bar | The pixel value under the mouse, the zoom level, and whether the AI is ready. |

## Panels can be moved
Drag a panel by its title to another edge of the window, or onto another panel to stack them as tabs. Closed panels come back from **View ▸ Panels**, and **View ▸ Reset Panel Layout** restores the original arrangement. DicomDoodle remembers your layout.

## Light or dark
Click {icon:sun} on the right of the toolbar (or press [[Ctrl+T]]) to switch between the dark and the light theme. Images always stay on a dark background, as is usual for medical images.

![The same window in the light theme](shot:light)
""", "interface layout toolbar menu panels tour")

OPENING = Chapter(
    "opening", "Opening scans", "folder-open",
    "Load a folder of DICOM slices — by button, from the recent list, or by dropping the folder onto the window.",
    """
An MRI **series** is one set of pictures taken in one go. It consists of many **slices** — thin cross-sections through the body — usually stored as one DICOM file each. DicomDoodle opens the whole folder and puts the slices in the right order (by their *instance number*).

![A series is a folder of slices; each slice gets its own annotation file](diagram:series)

## Three ways to open a folder
1. Click {icon:folder-open} **Open** in the toolbar (or **File ▸ Open DICOM Folder…**, [[Ctrl+O]]) and choose the folder that contains the `.dcm` files.
2. Click a folder under **Recent folders** on the start page, or use **File ▸ Open Recent**.
3. Drag the folder from your file manager and drop it anywhere on the DicomDoodle window.

A progress window shows the loading; large series take a few seconds and can be cancelled.

![The start page](shot:welcome)

| | On the start page |
|---|---|
| {n:1} | **Open DICOM folder…** — choose a folder of slices |
| {n:2} | **Read the Quick Start** — the five-minute introduction |
| {n:3} | **Recent folders** — one click opens them again |
| {n:4} | **Tools** — open any tool without loading a series first |

## What DicomDoodle reads and what it skips
- Annotation files (names ending in `_seg.dcm` or `_segmentation.dcm`) are recognised and are **not** shown as slices.
- A file that holds many slices at once (a *multi-frame* file) cannot be annotated directly — convert it first with the [Multi-frame Converter](chapter:converter) into standard single-slice files.
- Files that cannot be read are listed in the **Activity Log** (**View ▸ Panels ▸ Activity Log**).

> NOTE: Your original image files are never changed. Annotations are always written to separate files next to them.

## Series details
The **Series Info** tab on the right shows the patient name and ID, study date, scanner, plane (axial, coronal or sagittal), image size, pixel spacing and slice thickness — and, under *Current slice*, the file name, instance number and the labels present.
""", "load folder import dicom series recent drag drop")

VIEWING = Chapter(
    "viewing", "Moving through slices and viewing", "eye",
    "Scroll, zoom, pan and adjust the picture — and read the colored marks on the timeline.",
    """
## Changing slices
- **Mouse wheel** over the image: up = previous slice, down = next slice.
- Keyboard: [[←]] [[→]] (or [[↑]] [[↓]]) one slice, [[Page Up]] [[Page Down]] ten slices, [[Home]] [[End]] first and last slice.
- The **slice timeline** at the bottom: click or drag on it, or type a number in the box on its right.
- {icon:play} plays through the slices like a film (*cine*); click it again to stop.

## Reading the timeline
![Ticks, dots and the auto-detect band on the slice timeline](diagram:timeline)
- **Short ticks** under the line mark slices with an outline saved in an earlier session (grey), saved during this session (green) or with unsaved changes (orange).
- **Small dots** mark slices with a blank mask — no lesion (see [Saving annotations](chapter:saving)).
- The **purple band** is the auto-detect range (see [Batch processing](chapter:batch)).
- Hover over the timeline to see the slice number and what is marked there.

## Zoom and pan
- [[Ctrl]] + mouse wheel zooms around the mouse pointer. {icon:zoom-in} {icon:zoom-out} and [[Ctrl+=]] [[Ctrl+-]] do the same.
- {icon:fit} **Fit** ([[Ctrl+0]]) shows the whole slice again.
- To move a zoomed image, drag with the **middle mouse button**, hold [[Space]] and drag with the left button, or pick the {icon:move} **Pan** tool above the image ([[H]]).

## Brightness and contrast
Use the **Brightness** and **Contrast** sliders in the **Display** panel to make faint structures easier to see; **Reset image** restores the original. This changes only what you see — the AI and the saved files always use the original image.

## How annotations are drawn
In the **Display** panel:
- **Style** — *Boundary outline* (a colored line) or *Filled regions* (a see-through fill plus the line).
- **Outline** — line thickness on screen; **Fill** — how see-through the fill is.
- **Show saved annotation files** — a faint layer with what is already saved on disk for the slice, so you can compare it with your current work.

> TIP: The status bar shows the original pixel value under the mouse pointer — useful to check how bright a region really is.
""", "scroll navigate zoom pan brightness contrast cine timeline display", short="Moving through slices")

CHAPTERS = [WELCOME, QUICK_START, MAIN_WINDOW, OPENING, VIEWING]
