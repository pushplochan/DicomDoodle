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
- For the AI (Auto Detect, Detect in Box, Find tumour range): the **DicomDoodle model card**, a separate download — one file whose name starts with *DicomDoodle-ModelCard*. Leave it in your Downloads folder and start DicomDoodle: it finds the card, installs it once and shows the progress at the bottom right ([The AI models](chapter:ai-models)). Without it everything else works, and you draw outlines by hand.
- No graphics card is needed: the AI runs on the processor (CPU). When DicomDoodle finds a graphics card (GPU) it can use, you can switch to it for faster AI.

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
This step and the next box need the AI models, the model card ([The AI models](chapter:ai-models)); without it, draw the outline with the pen instead ([Drawing by hand](chapter:manual-draw)).
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

AI_MODELS = Chapter(
    "ai-models", "The AI models: model card, processor or graphics card", "cpu",
    "The AI comes as a separate file, the model card. Without it you annotate by hand.",
    """
DicomDoodle's AI — **Auto Detect**, **Detect in Box**, **Find tumour range** and **Detect** in the Multiplanar & 3D Viewer — needs the **model card**: one file with the AI models, downloaded separately from the program. Its name starts with *DicomDoodle-ModelCard* and ends in `.ddcard` (for example `DicomDoodle-ModelCard-1.0.ddcard`, about 860 MB).

## Getting the AI
1. Download the model card from DicomDoodle's download page.
2. Leave it in your **Downloads** folder (the Desktop, Documents or the program's folder work too).
3. Start DicomDoodle. It finds the card and installs it — the first time only, usually under a minute. A notification at the bottom right shows the progress and disappears when the AI is ready.

You can also choose the file with **File ▸ Load Model Card…**, or drag it onto the window. Once installed, the downloaded file may be deleted: DicomDoodle keeps its own copy (on Windows in `%LOCALAPPDATA%\\DicomDoodle\\ModelCard`). At every later start the notification shows the models loading for a few seconds, then goes away by itself.

## Without a model card
Everything except the AI works: open series, draw outlines with the pen and eraser, save, review and measure. The status bar says **Manual annotation only** and the AI buttons are grey.

## Processor or graphics card
The AI runs on the **processor (CPU)** unless you choose otherwise. When the computer has a graphics card (GPU) that can run it, DicomDoodle finds it by itself: switch with the AI button at the right of the status bar (click **AI ready · CPU**) or in **File ▸ Preferences ▸ AI Models**. On Windows any recent graphics card works — NVIDIA, AMD or Intel, through DirectML. The graphics card makes the AI several times faster; the outlines are the same.

| Status bar | Meaning |
|---|---|
| grey **Manual annotation only** | no model card: annotate by hand |
| orange **Installing the model card…**, **Loading the AI models…** | a moment, please |
| green **AI ready · CPU** or **AI ready · GPU** | the AI is ready, on the processor or on the graphics card |
| grey **AI models load on first use** | the models load at the first AI click (set in **Preferences ▸ AI Models**) |

## What is in the model card
Three models; DicomDoodle tells from each file what it is for, whatever its name:
- the **lesion detector** (YOLO26x-seg, trained on brain tumour MRI) finds the lesion and puts a box around it,
- the **fallback detector** (YOLOE) is tried when the lesion detector finds nothing,
- **SAM 2.1** traces the exact outline inside a box.

They are stored in the ONNX format, which runs faster than the original PyTorch files and gives the same results. A newer model card found later replaces the installed one. To delete the models, click **Remove** in **File ▸ Preferences ▸ AI Models**; DicomDoodle then works without AI until you load a model card again.
""", "ai model card models download install ddcard onnx gpu cpu graphics card processor directml cuda manual "
    "no ai offline", short="The AI models")

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

CHAPTERS = [WELCOME, QUICK_START, AI_MODELS, MAIN_WINDOW, OPENING, VIEWING]
