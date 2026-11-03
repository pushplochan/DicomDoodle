"""User Guide — Advanced tools."""
from .chapter import Chapter

MPR = Chapter(
    "mpr", "Multiplanar & 3D Viewer", "grid",
    "See a series from all three directions and in 3D, measure the lesion, remove the skull, and compare two scans.",
    """
An MRI series is usually taken in one direction, but the slices together form a 3D block. The **Multiplanar & 3D Viewer** cuts that block in all three standard directions — *multiplanar reconstruction*, MPR — and shows them side by side, next to a 3D view of the whole series. (Earlier versions had a separate *Multiplanar MRI View* and *3D Model Viewer*; both are now in this one window.)

![The three standard planes](diagram:planes)

The window has two side panels:
- **Controls** on the left, on two tabs: **Multiplanar** for the slice views, and **3D view** for the 3D view (see [The 3D view](chapter:viewer3d)).
- **Series & Measurements** on the right: the series of the 3D model, the lesion volume and the shape of the lesion.

{icon:sliders} **Controls** and {icon:bar-chart} **Series & Measurements** in the window's toolbar hide and show the panels, to give the views more room.

## Opening a series
1. Click {icon:grid} **Multiplanar & 3D Viewer** on the right of the main window's toolbar.
2. On the **Multiplanar** tab, click **Load primary series…** and choose a folder. If it holds more than one image volume — several series, two echoes, a localiser, stacks in other directions — pick one from the list. Multi-frame (*Enhanced*) files and files without the `.dcm` ending are read too.
3. The axial, coronal and sagittal views appear. Annotation files are loaded automatically: those beside the slices or, for a multi-frame file, those of its single-frame copies in a subfolder (as the [Multi-frame Converter](chapter:converter) makes them).
4. For the 3D view and the lesion volume, click **Build 3D model** on the right ([details](chapter:viewer3d)). The series open in the main window — or the one you loaded — is already filled in there.

If something about the series is unusual, the status bar says so — for example *missing slices*, which are shown blank so that every other slice keeps its true position.

![The Multiplanar & 3D Viewer with an axial contrast-enhanced T1 series](shot:mpr)

| | |
|---|---|
| {n:1} | **Load primary series…** |
| {n:2} | **Registration engine** and **Register second series…** |
| {n:3} | **Show** — switch between the primary and the registered second series |
| {n:4} | **Resampling algorithm** |
| {n:5} | **Image** — the original series, or the brain alone (*Skull stripped*, see below) |
| {n:6} | **Annotation** overlay and its options |
| {n:7} | The three planes — the green crosshair marks the same point in each |
| {n:8} | {icon:maximize} Enlarge one view; [[Esc]] or the button again restores the 2 × 2 layout |
| {n:9} | The 3D view of the model built on the right |
| {n:10} | {icon:sparkles} **Detect** — find the tumour in 3D, starting from this view's slice (see below) |
| {n:11} | The two tabs of the controls: **Multiplanar** and **3D view** |
| {n:12} | **Series & Measurements**: the 3D model's series, the lesion volume and its shape |

## Moving around
- **Click** in any view to move the crosshair there — the other two views jump to that point.
- The **mouse wheel**, the slider or the number box under each view changes its slice.

## Display options
- **Resampling algorithm** — how in-between points are calculated when the planes are rebuilt. *None (native voxels)*, the starting choice, shows the data exactly as stored; *Cubic B-Spline* gives smoother pictures with high accuracy.
- **Through-plane reconstruction** — the sharpest choice for the two views that cut across the slices. Only the slice direction is refined, and every value comes from the series' own slices (no AI): it takes into account that each slice is an average over its thickness. Tested against each patient's own thin-slice scan, it rebuilt those views more accurately than B-Spline. The values of the original slices are slightly re-estimated, so use *None* when you need them exactly.
- **Image** — *Original MRI*, or *Skull stripped (HD-BET)*: the brain alone (see the next section).
- **Annotation ▸ Overlay** — show the annotation of the primary series, of the second series, the one transferred from the primary, or the *Detected tumour*. **Show filled mask** fills the regions; **Fill SEG contours** turns saved outlines into solid regions — also outlines traced in several open pieces on one slice (for example *Tumor & Brain Boundary* plus *Tumor & Dura Boundary*), which are filled together as one region.

## Removing the skull (HD-BET)
*Skull stripping* removes skull and scalp so that only the brain remains — to look at the brain without the bright scalp around it, or at the surface of the brain in 3D.
1. Choose **Image ▸ Skull stripped (HD-BET)** on the **Multiplanar** tab — or, for the 3D view, **Show ▸ Skull stripped (HD-BET)** on the **3D view** tab.
2. The first time, HD-BET finds the brain. A window shows the progress; **Cancel** stops it.
3. The views show the brain alone. **Original MRI** brings the full image back at once.

![The brain alone in every view, skull and scalp removed by HD-BET](shot:skull)

| | |
|---|---|
| {n:1} | **Image ▸ Skull stripped (HD-BET)** |
| {n:2} | The slice views show the brain alone (here with the *Detected tumour*) |
| {n:3} | The 3D view with **Show ▸ Skull stripped (HD-BET)**, seen from the top |

HD-BET runs once per scan: the slice views, the registered second series and the 3D model share its result when they show series of the same session. In the example series it took about 40 seconds without a graphics card and found a brain of 1288 mL.

> NOTE: HD-BET is an optional package (`pip install hd-bet`; the Windows installer contains it); it downloads its network the first time it runs. Without it, only the original image can be shown. Its brain mask is used for display only — Detect and the lesion volume always work on the original slices.

## Finding the tumour in 3D (Detect)
Each of the three views has a {icon:sparkles} **Detect** button. It finds the tumour in the whole series, starting from the slice that view shows:
1. Scroll the view to a slice where the tumour is clearly visible.
2. **Click on the tumour.** This tells Detect which structure you mean (without a click it takes the most likely one).
3. Press **Detect** in the header of that view, and wait — a window shows the progress.

Detect first finds the tumour on your slice and on the matching slices of the other two views — the tumour detector (*YOLO*) proposes boxes, *SAM* outlines the tumour in them — which together give a 3D box around it. Then it outlines the tumour on every slice of the series inside that box, following it from slice to slice.

![Detect: the tumour found in 3D, its box dashed in every view](shot:mpr_detect)

| | |
|---|---|
| {n:1} | **Detect** in the header of the view you started from |
| {n:2} | The 3D box around the tumour (dashed), shown in every view whose slice passes through it |
| {n:3} | The result: **Detected tumour** in the overlay list |

The crosshair jumps to the centre of the tumour, and the status bar gives its volume, its slice range and — if the series has its own annotation — the *Dice* score against it. In the example: 41.7 mL on axial slices 113 to 152, Dice 0.96 against the annotation saved in the earlier chapters (and 0.95 against the expert's original outline of 42.9 mL).

> NOTE: Detect always works on the series' own slices, whatever resampling algorithm or image (original or skull stripped) is chosen. Without a graphics card it takes a few minutes (3 to 4 minutes for the 40 tumour slices of the example). The detected tumour is shown for review and comparison; it is not saved as annotation files.

> IMPORTANT: The tumour detector was trained on FLAIR scans with the skull removed. Always check the result slice by slice: small tumours, tumours growing into the eye socket and unusual sequences can be missed or outlined too large. If Detect picks the wrong structure, click on the tumour and press Detect again.

## Comparing two scans (registration)
*Registration* lines up a second scan — for example a scan from another day, or another MRI sequence — with the first one, so the same point means the same place in the body.
1. Load the primary series.
2. Choose a **Registration engine** (see the list below).
3. Click **Register second series…** and choose its folder. Use **Show** to switch between the two.

The engines, from simplest to most flexible:
- **Scanner Coordinates (Identity)** — for scans from the same session: the scanner positions already match, nothing is moved.
- **SimpleITK (ITK Mattes MI)** — for another session: the second scan is shifted and turned (*rigid*) until it fits. Fast.
- **SimpleITK (Deformable / BSpline)** — also allows gentle bending; more accurate, but slower.
- **ANTsPy (ANTs toolkit)** — an optional, very robust engine; it must be installed separately.

## Carrying an annotation to the second scan
After registration, the primary's annotation is **transferred** onto the second scan (*Transferred from primary* in the overlay list).
- **Compare annotations…** — if the second scan has its own annotation, see how well the two agree: *Dice* for the whole volume and *IoU* slice by slice.
- **Save transferred annotation…** — writes the transferred annotation as `_seg.dcm` files for the second scan's slices, into a folder you choose.

![Dice and IoU measure how well two outlines agree](diagram:overlap)

> NOTE: This tool needs *SimpleITK* and *matplotlib*; the 3D view also needs *vtk*.
""", "mpr multiplanar axial coronal sagittal crosshair registration transfer dice iou skull strip skull stripping "
   "hd-bet brain extraction detect tumour tumor yolo sam through-plane reconstruction echo localiser enhanced "
   "multi-frame 3d viewer panels tabs", short="Multiplanar & 3D")

VIEWER_3D = Chapter(
    "viewer3d", "The 3D view", "cube",
    "See the head, the brain or the annotated lesion in 3D, turn it around, cut into it — with the lesion volume "
    "beside it.",
    """
The 3D view is the bottom-right view of the [Multiplanar & 3D Viewer](chapter:mpr). It stacks all slices of a series into a three-dimensional model; when the folders contain annotation files, the lesion can be highlighted, and the panel on the right shows its volume and shape. (In earlier versions this was the separate *3D Model Viewer*.)

## Building the 3D model
1. Open the {icon:grid} **Multiplanar & 3D Viewer**. In the **Series** card on the right, the series open in the main window is already filled in, in the row of its plane.
2. Add the coronal and/or sagittal series of the **same patient** if you have them: {icon:folder} chooses a folder, {icon:x} removes it. All series are combined into one model.
3. Choose which **Annotation files** to use, then click **Build 3D model**. Measuring the lesion and building the model take a few seconds.

![The 3D view, enlarged, with its settings on the left and the measurements on the right](shot:viewer3d)

| | |
|---|---|
| {n:1} | **Show** — what the 3D view shows (see below) |
| {n:2} | The settings of the 3D view |
| {n:3} | The 3D view, here enlarged with {icon:maximize} |
| {n:4} | {icon:refresh} resets the camera; {icon:camera} saves a picture of the view (PNG) |
| {n:5} | **Series**: the folders and **Build 3D model** |
| {n:6} | **Lesion volume**, see [Measuring lesion volume](chapter:volume) |
| {n:7} | **3D lesion metrics**: the shape of the lesion |
| {n:8} | **Show lesion surface & metrics** — the lesion as a solid red surface, with its measurements in the view (the same as *Show lesion surface in 3D* at the bottom of the right-hand panel) |

If the 3D model cannot be built (for example from an unusual series), the lesion volume is still shown and a message explains the problem.

## What the 3D view shows
On the **3D view** tab, choose under **Show**:
- **Original MRI** — the series as it is.
- **MRI with annotation** — the annotated lesion made brighter so it stands out (offered when the series have annotation files).
- **Skull stripped (HD-BET)** — the brain alone, without skull and scalp (see [Removing the skull](chapter:mpr)). The first time, HD-BET needs about a minute without a graphics card.

Switching keeps the camera where it is; only the picture changes.

## Moving the model
- **Left-drag** turns the model, **right-drag** or the **mouse wheel** zooms, **middle-drag** moves it.
- In the header of the 3D view, {icon:refresh} resets the camera and {icon:camera} saves a picture of the view; {icon:maximize} enlarges it, as for the other views.

> TIP: The lesion sits *inside* the head, so its red surface is hidden at first. Tick **Z axis** under **Clipping planes** to cut the model open, then turn it until you look at the cut — as in the picture above.

## Settings (3D view tab)
- **Appearance** — the **Colormap** (13 color schemes, from *Grayscale* to *BrainTumorSpecific*), **Contrast**, **Ambience** (overall light), **Opacity threshold** (hides faint tissue), **Gradient opacity** (makes edges stand out), **Specular power** (shine), and **Scalar min / max** (the brightness range shown).
- **Surgery planning** — **Edge enhancement**, **Tissue density**, **Advanced shading**, and **Show lesion surface & metrics**.
- **Clipping planes** — tick **X**, **Y** or **Z axis** and move the slider to cut the model open and look inside.
- **Lighting** — the direction (**Azimuth**, **Elevation**) and **Intensity** of the light.
- **Reset to defaults** restores them.

## 3D lesion metrics
| Metric | Meaning |
|---|---|
| **Volume (3D model)** | Lesion volume measured on the 3D model; 1 cc = 1 cm³ = 1 millilitre |
| **Surface area** | The area of the lesion's outer surface |
| **Sphericity** | How round the lesion is: 1 = a perfect ball, lower = more irregular |
| **Texture contrast** | How much the brightness varies inside the lesion |
| **Texture homogeneity** | How even the brightness is inside the lesion |

> NOTE: The 3D view needs the optional packages *vtk* and *SimpleITK*. Without *vtk*, the rest of the window works and the 3D view says so.
""", "3d volume render model rotate surface metrics sphericity clipping colormap vtk lesion volume series build "
   "show original annotation skull stripped brain", short="3D view")

CONVERTER = Chapter(
    "converter", "Converting multi-frame files", "scissors",
    "Turn a file that holds a whole series into standard single-slice DICOM files, ready to annotate.",
    """
Some scanners and archives save a complete series as **one multi-frame file** (for example *Enhanced MR*). DicomDoodle annotates slice by slice, so such a file is converted first into ordinary single-slice files — the kind most MRI scanners write.

## Step by step
1. Click {icon:scissors} **Multi-frame Converter** on the right of the toolbar (or **Tools ▸ Multi-frame to Single-frame Converter**).
2. Click {icon:folder-open} **Open DICOM File…** and choose the multi-frame file.
3. Look through the frames with the mouse wheel or the timeline to check it is the right file.
4. Click {icon:save} **Save Slices…** and choose a folder. A new folder with one file per frame is created.
5. Open that folder in DicomDoodle and annotate as usual.

![The Multi-frame Converter](shot:converter)

| | |
|---|---|
| {n:1} | **Open DICOM File…** |
| {n:2} | **Save Slices…** — one file per frame |
| {n:3} | The current frame and its stack position |
| {n:4} | Frame timeline |
| {n:5} | Details of the file and of the current frame |

Each new file is named after the original file plus the frame number, for example `scan_1.dcm`, `scan_2.dcm` …

## What the converter writes
The new files follow the DICOM standard for single MR or CT images, so other software reads them as a normal series:
- **The same picture.** The pixel values are copied unchanged (also from compressed files); the files themselves are stored uncompressed.
- **The details of each frame.** Position, orientation, pixel spacing, slice thickness, brightness/contrast window and the scan settings (repetition time, echo time, flip angle, coils …) are taken from the multi-frame file and written in the form single-slice images use.
- **Patient and study unchanged.** They are copied as they are — an anonymized file stays anonymized. Values the standard requires but the original lacks are left empty, never invented.
- **A new series.** The slices form their own series, so the multi-frame original and the slices are never mixed up in an archive. Converting the same file again gives the very same identifiers, so annotation files made earlier still fit.

MR files become *MR Image* files and CT files *CT Image* files; any other kind of multi-frame file is written as *Secondary Capture* images.
""", "multi-frame multiframe enhanced split convert single frame slices standard mr ct", short="Multi-frame files")

PREFERENCES = Chapter(
    "preferences", "Preferences", "settings",
    "Theme, annotation display, saving, AI models and the ontology key — all in one place.",
    """
Open **File ▸ Preferences…** ([[Ctrl+,]]). Changes take effect when you click **Save**.

![Preferences](shot:preferences)

| Tab | What you can set |
|---|---|
| **Appearance** | Dark or light theme (also [[Ctrl+T]] at any time). |
| **Display** | Outline or filled annotations, outline width, fill opacity, and whether saved annotation files are shown as a faint layer. |
| **Files** | Whether DicomDoodle asks before replacing annotation files from earlier sessions, and whether blank masks are saved automatically. |
| **AI Models** | The model card: load or remove it, and which model does each step ([The AI models](chapter:ai-models)). Whether the AI runs on the processor or the graphics card, and whether the models are loaded when DicomDoodle starts. **Load model card…** and **Remove** act at once. |
| **Ontology** | Your BioPortal API key, used by [Add Ontology](chapter:ontology) when DICOM's own code list has no match. Leave it empty to use the built-in key. |

## Where DicomDoodle keeps its settings
Preferences, your own labels, recent folders and the window layout are stored in `settings.json` in your user settings folder:
- Windows: `%APPDATA%\\DicomDoodle`
- macOS: `~/Library/Application Support/DicomDoodle`
- Linux: `~/.config/DicomDoodle`

The app installed with the Windows installer also writes its messages to `dicomdoodle.log` in that folder; attach it when you report a problem.

> TIP: To give colleagues the same labels, use **Export my labels…** in the Labels panel instead of copying the settings file.
""", "preferences settings options theme dark light gpu cpu device model card graphics card api key config")

CHAPTERS = [MPR, VIEWER_3D, CONVERTER, PREFERENCES]
