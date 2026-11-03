"""User Guide — Saving & batch work."""
from .chapter import Chapter

SAVING = Chapter(
    "saving", "Saving annotations", "save",
    "What is written to disk, where it goes, and how DicomDoodle protects your work.",
    """
## One annotation file per slice
When you save, DicomDoodle writes a new file **next to the slice**, with the same name plus `_seg.dcm`. The slice `IMG_0133.dcm` gets `IMG_0133_seg.dcm` in the same folder. The original image file is never changed.

The file is a **DICOM Segmentation** (*DICOM SEG*) — the official DICOM format for marked regions — so it stays linked to its slice and can be read by other medical-imaging software. Inside it:
- every label used on the slice becomes one **segment**, named with the label's full name (for example *Tumor & Brain Boundary*);
- each region is stored as its **outline** — a one-pixel boundary line — exactly like the original DicomDoodle. Outlines you draw with the pen are saved exactly as drawn. Small gaps of up to 20 pixels are closed automatically. DicomDoodle's viewer and the 3D Model Viewer (with the lesion volume) fill the outlines again when needed.

![Filled region, saved outline and blank mask](diagram:saved)

## Saving
- **This slice:** {icon:save} **Save** in the toolbar, or [[Ctrl+S]].
- **Every slice with unsaved changes:** **File ▸ Save All Modified Slices**, or [[Ctrl+Shift+S]]. A progress window shows each file being written.

## Blank masks are automatic
You only annotate the slices that show a lesion. **Each time you save, every slice that has no annotation file yet gets a blank mask**: a valid annotation file with the segment *No Segmentation*, meaning “no lesion on this slice”. Once the lesion slices are saved, every slice of the series has its annotation file.

- The first save in a series writes all the blank masks at once. This takes a few seconds for a large series, and the message at the bottom right says how many were saved.
- A blank mask is only a placeholder. Annotate the slice and save, and the outline replaces the blank mask without any question. An [auto-detect range](chapter:batch) replaces blank masks too.
- Slices with unsaved outlines never get a blank mask.
- To record one slice as checked yourself, save it while it is empty: DicomDoodle asks *Save blank mask?*.
- The **Blank masks** card in the **Batch Processing** panel counts the slices that are annotated, blank or still without a file. Its switch *Save blank masks automatically* (also in the **Batch** menu and in **File ▸ Preferences ▸ Files**) turns the automatic blank masks off — for example when you only correct one slice in someone else's folder.

> IMPORTANT: A blank mask means “no lesion here”. Make sure every slice that shows the lesion is annotated — a slice you skip keeps its blank mask.

## Is this slice saved?
The badge at the top right of the image always tells you:

| Badge | Meaning |
|---|---|
| **Unsaved changes** (orange) | The slice has work that is not saved yet. |
| **Saved** (green) | Saved during this session — *Saved · blank mask* if you saved it empty. |
| **Annotation file on disk** (blue) | An outline from an earlier session exists. Turn on *Show saved annotation files* to see it. |
| **Blank mask** (grey) | The slice's file is a blank mask: no lesion. |

The ticks and dots under the [slice timeline](chapter:viewing) show the same information for the whole series.

## Your work is protected
- Before you open another folder, close the series or quit with unsaved annotations, DicomDoodle asks: **Save all**, **Discard** or **Cancel**.
- Before replacing an outline saved in an earlier session, it asks first (blank masks are replaced without asking). Tick *Don't ask again* if you prefer; the question can be turned back on in **File ▸ Preferences ▸ Files**.
- Files are written to a temporary file first and renamed when complete, so an interrupted save never leaves a damaged file.

> NOTE: The Activity Log (**View ▸ Panels ▸ Activity Log**) lists every file saved in this session, and how many blank masks each save added.

## Made to the DICOM standard
Annotation files follow the Segmentation rules of the DICOM standard (dicomstandard.org), so hospital archives (*PACS*) and web viewers such as **OHIF** read them like any other DICOM object:
- **Same patient, same study.** The patient and study details are copied unchanged from the slice, so the annotation is filed next to its image. An anonymized slice gives an anonymized annotation file — nothing is added. Values the standard requires but the slice lacks are left empty rather than made up.
- **DicomDoodle is named as the maker.** The file says which software and version created it, not the scanner that took the image.
- **A standard meaning for every segment.** Besides its name, each segment carries two codes: a *category* and a *type*, taken from DICOM's own list of medical codes. You can change them later with [Add Ontology](chapter:ontology).

| Label | Category | Type |
|---|---|---|
| Tumor & Brain, Tumor & Dura, Tumor & Edema Boundary | Morphologically Abnormal Structure | Neoplasm |
| Tumor & Necrosis Boundary | Morphologically Abnormal Structure | Necrosis |
| Edema & Brain Boundary | Morphologically Abnormal Structure | Edema |
| Magenta Boundary and your own labels | Morphologically Abnormal Structure | Lesion |
| Blank mask (*No Segmentation*) | Anatomical Structure | Brain |

- **Its color.** Each segment stores its label color, so OHIF draws it in the same color as DicomDoodle.
- **How it was made.** *Semi-automatic* — the AI outlined it from your box or your Auto Detect, and you checked it; *automatic* — an auto-detect range, without review; *manual* — drawn with the pen. Blank masks count as manual: you decided the slice shows no lesion.
- **One series per file.** In an archive each annotation file appears as its own series, called *DicomDoodle annotation, image N* — N is the number of its slice.

## Viewing the files in OHIF or a PACS
1. Send the slices **and** their `_seg.dcm` files to the archive (for example with your PACS upload tool, or by dragging the folder into a local OHIF viewer).
2. Open the study. Each annotation file shows as a **SEG** series next to the image series.
3. Open a SEG series: the outline appears on its slice, in the label's color, with the label's name.

> NOTE: Some web viewers cannot show letters outside the Latin alphabet inside annotation files: a label named in, say, Devanagari is stored correctly but may look garbled in OHIF. Latin letters, including accents (é, ö), are shown correctly.
""", "save ctrl+s seg dcm file dicom seg segmentation blank mask automatic empty no lesion placeholder outline "
    "boundary unsaved overwrite replace standard ohif pacs archive web viewer color code category type")

BATCH = Chapter(
    "batch", "Batch processing: many slices at once", "layers",
    "Auto-detect and save a whole run of lesion slices in one go — blank masks cover the rest.",
    """
In most series only some slices show the lesion. The **Batch Processing** panel on the right has two cards:
- **Auto-detect range** — every slice in the range is processed with [Auto Detect](chapter:auto-detect), and each result is saved straight away. Where nothing is found, a blank mask is saved. **Find tumour range** sets the range for you, from one slice of the tumour.
- **Blank masks** — every slice you do not annotate gets a blank mask automatically when you save (see [Saving annotations](chapter:saving)). The card counts the slices that are annotated, blank or still without a file.

![The Batch Processing panel](shot:batch)

| | |
|---|---|
| {n:1} | **Start** and **End** of the auto-detect range — the buttons show the chosen slices |
| {n:2} | {icon:x} clears the range |
| {n:3} | **Find tumour range** — sets Start and End from the slice on screen |
| {n:4} | What it found: the two other views through the tumour, with the detector's boxes (green) and the range (purple) |
| {n:5} | The label the results will get (the active label) |
| {n:6} | **Run auto-detect & save** — the number in brackets is how many slices the range covers |
| {n:7} | **Save blank masks automatically** — on unless you switch it off |
| {n:8} | How many slices are annotated, have a blank mask, or have no file yet |

## Finding the range automatically
1. Go to a slice near the middle of the tumour, where it is large and clear.
2. Click {icon:search} **Find tumour range** (also in the **Batch** menu). After a few seconds **Start** and **End** show the first and last slice of the tumour, and the purple band appears on the timeline.
3. Check the range in the two small views {n:4}: each green box should sit on the tumour, and the purple band should cover it from end to end.
4. Go to the first and last slice of the range and one slice beyond each. Move **Start** or **End** if the tumour starts or ends elsewhere.
5. Click **Run auto-detect & save** (below).

How it finds the range: the tumour detector draws a box around the tumour on the slice on screen. Through the middle of that box DicomDoodle cuts the two other views from the series — coronal and sagittal for an axial series, as in the [Multiplanar view](chapter:mpr) — and the detector boxes the tumour there too. Where those two boxes begin and end along the slice direction is the range; the range covers both. A box you drew stays on the image, and the detector's box on this slice is shown dashed in green. If the detector finds no box on one of the other views, that view shows a dotted grey guess and only the other view counts; if it finds none on either, the range is *estimated* from the tumour's size on this slice, and the panel says so — check such a range with extra care.

> TIP: If the detector picks something else, draw a box around the tumour first ({icon:box} **Box** tool, [[B]]), then click **Find tumour range**: only detector boxes centred inside your box are used — or your box itself, if there is none.

> NOTE: Only the detector runs, not SAM, so it takes a few seconds; the first time a little longer, while the series is read as a 3D volume. It needs a stack of parallel slices with position information, as every MRI series has. In the example series, Find tumour range on slice 133 gives slices 112 to 151; the tumour is outlined on slices 114 to 151. On FLAIR series the detector also boxes the swelling around a tumour, so the range is often wider than the tumour itself.

## Setting the range by hand
1. Go to the first slice that shows the lesion (mouse wheel, arrow keys or timeline).
2. Click **Start**. The button now shows the slice number, e.g. **Start · 114**.
3. Go to the last slice with the lesion and click **End**. The card shows how many slices the range covers, and a purple band appears on the slice timeline.

The **Batch** menu has the same commands. Start and End can be set in any order and changed at any time.

## Auto-detect the range
1. Choose the label in the **Labels** panel.
2. Set the range over the slices that show the lesion — with **Find tumour range** or by hand.
3. Click **Run auto-detect & save (N)**.
4. A progress window shows the current slice, the elapsed time and the time left. **Cancel** stops after the current slice — everything finished so far stays saved.
5. Finally, every other slice that has no annotation file yet gets its blank mask.

![The summary at the end of a batch](shot:progress)

## Slices that already have annotation files
- Slices **saved during this session** are always skipped, so a batch never overwrites work you have just checked.
- **Blank masks** are replaced without asking.
- If some slices have **outlines from an earlier session**, or **unsaved work** in the window, DicomDoodle asks what to do: **Replace them**, **Skip those slices** or **Cancel**.

> TIP: A fast, reliable routine: 1) an auto-detect range over the lesion, 2) step through the lesion slices, correct where needed and save again. The blank masks for all other slices are written for you.

> IMPORTANT: Check batch results like any other AI result before you use them. In the example series of this guide the detector outlined the tumour well on slices 115 to 141, but on slice 114 and on slices 142 to 151 it outlined the wrong region — those slices were corrected with the pen. Unchecked, they would have nearly doubled the measured volume.
""", "batch range auto detect range many slices bulk start end band timeline blank masks automatic "
    "find tumour tumor range automatic first last slice coronal sagittal detector box",
    short="Batch processing")

CHAPTERS = [SAVING, BATCH]
