"""User Guide — Reference: shortcuts, troubleshooting, glossary."""
import re

from .chapter import Chapter
from .shortcuts import SHORTCUTS


def _keys(text: str) -> str:
    """'Ctrl+Z  /  Ctrl+Y' -> '[[Ctrl+Z]] / [[Ctrl+Y]]' (markup key chips)."""
    def chip(key: str) -> str:
        return "[[" + re.sub(r"([\[\]|])", r"\\\1", key.strip()) + "]]"
    parts = re.split(r"(\s+/\s+|\s+or\s+)", text)
    return "".join(f" {p.strip()} " if i % 2 else chip(p) for i, p in enumerate(parts))


def _shortcuts_body() -> str:
    lines = ["Everything you can do with the keyboard and mouse. The same list is in **Help ▸ Keyboard Shortcuts**.", ""]
    for group, rows in SHORTCUTS:
        lines += [f"## {group}", "| Keys | What it does |", "|---|---|"]
        lines += [f"| {_keys(keys)} | {action} |" for keys, action in rows]
        lines.append("")
    lines.append("> TIP: Number keys and arrows act on the image, not while you type in a text or number box.")
    return "\n".join(lines)


SHORTCUTS_CHAPTER = Chapter(
    "shortcuts", "Keyboard and mouse shortcuts", "keyboard",
    "Work faster: every shortcut in one list.",
    _shortcuts_body(), "keys keyboard shortcut hotkey mouse wheel", short="Shortcuts")

TROUBLESHOOTING = Chapter(
    "troubleshooting", "Troubleshooting and questions", "help",
    "Answers to the questions people ask most.",
    """
### The AI takes a long time the first time
The models are loaded into memory the first time you use them in a session — a few seconds, up to a minute on slow computers. After that each slice takes about a second with a graphics card (GPU), longer on the processor (CPU). To load SAM while you are still opening files, tick *Load SAM in the background* in **File ▸ Preferences ▸ AI Models**.

### “AI model error” or “weights not found”
DicomDoodle needs three model files: `best_flair.pt`, `yoloe-11l-seg-pf.pt` and `sam2.1_l.pt`. **File ▸ Preferences ▸ AI Models** shows which ones were found; point **Models folder** at the folder that contains them.

### “No detection found for this slice”
The detector did not recognise a lesion — common on slices where it is very small. Draw a box around it ([Segment with a box](chapter:box)) or draw it by hand with the pen ([Drawing by hand](chapter:manual-draw)).

### The outline covers too much, or the wrong structure
Undo with [[Ctrl+Z]], draw a tighter box around just the lesion and press [[Enter]] again. For fine corrections use the pen and the eraser ([Drawing by hand](chapter:manual-draw)).

### Find tumour range gives the wrong slices, or finds nothing
Go to a slice near the middle of the tumour, where it is large and clear, and try again. The two small views under the button show what the detector boxed: if a green box is not on the tumour, draw a box around the tumour with the **Box** tool ([[B]]) and click **Find tumour range** again — only detector boxes centred inside your box are used. You can always correct **Start** and **End** by hand ([Batch processing](chapter:batch)).

### Detect in the Multiplanar view outlines the wrong structure, or says “No tumour found”
Scroll to a slice where the tumour is large and clear, **click on it**, then press **Detect** on that same view — the click tells Detect which structure you mean. Starting from another view (coronal instead of axial, for example) can also help. Detect works best on clearly enhancing tumours; check every result slice by slice ([Multiplanar view](chapter:mpr)).

### My folder opens, but some files are missing
Annotation files are never shown as slices, and files that are not images (reports, for example) are skipped. Open **View ▸ Panels ▸ Activity Log** to see which files were skipped and why. A series stored as a single file must be split first with the [Multi-frame Converter](chapter:converter).

### A tool says “Tool unavailable”
That tool needs an optional Python package (for example *vtk*, *SimpleITK* or *matplotlib*). The message names it; install everything at once with `pip install -r requirements.txt`. (The Windows installer already contains all of them.)

### The ontology search finds nothing, or BioPortal does not answer
DICOM's own code list is searched first and works without internet; try a simpler word (*neoplasm* rather than *brain neoplasm, primary*) or the code itself. BioPortal is asked when the list has no match, and that needs an internet connection and a valid BioPortal key (**File ▸ Preferences ▸ Ontology**; leave the field empty to use the built-in key). Firewalls in some hospitals block the BioPortal website. The line under the results says what went wrong. It works with any version of the optional *ontoportal-client* package, and without it.

### OHIF or our PACS does not show the annotation
Make sure the `_seg.dcm` files were sent **together with** their image slices — an annotation file refers to its slice and is shown on it. Annotation files from DicomDoodle 2.0 or later follow the DICOM standard (see [Saving annotations](chapter:saving)); files from older versions may lack details some archives check, so save the slices again to renew them.

### Where are my annotation files?
In the same folder as the slices, named `<slice name>_seg.dcm` — see [Saving annotations](chapter:saving). The **Series Info** tab shows the folder.

### Why do slices I never annotated have annotation files?
Those are **blank masks**. Each time you save, every slice without an annotation file gets one, so each slice records either an outline or “no lesion”. Annotate such a slice and save to replace its blank mask. To stop this, untick *Save blank masks automatically* in the **Batch Processing** panel — see [Saving annotations](chapter:saving).

### Can I change an annotation after saving it?
Yes. Annotate the slice again and save, or correct the saved outline: **Edit saved outline** ([[Ctrl+L]]) loads it for the pen and the eraser. The file is replaced. DicomDoodle asks first if the file is from an earlier session, unless you loaded it with *Edit saved outline*.

### Where are Manual Draw, the Lesion Volume Calculator and the 3D Model Viewer?
They are now part of other windows, so you no longer switch between windows. You draw by hand right on the image in the main window, with the **Pen** and **Eraser** above the image ([Drawing by hand](chapter:manual-draw)). The 3D model is the 3D view of the **Multiplanar & 3D Viewer** ([The 3D view](chapter:viewer3d)), and the lesion volume is in its right-hand panel ([Measuring lesion volume](chapter:volume)).

### Where did Volume processing go?
The Multiplanar view's *Volume processing* (classical skull strip, z-score, connected components) was replaced by one choice: **Image ▸ Skull stripped (HD-BET)**, the most accurate of the skull-stripping methods ([Removing the skull](chapter:mpr)).

### Skull stripping says HD-BET is not installed, or takes long
HD-BET is an optional package: install it with `pip install hd-bet` (the Windows installer already contains it). Its network is downloaded the first time it runs, so that run needs an internet connection. Without a graphics card it takes about a minute per scan; the result is kept, so switching back and forth is instant afterwards. **Cancel** in the progress window stops it.

### Does DicomDoodle send my scans anywhere?
No. Everything — including the AI — runs on your own computer. The only thing ever sent over the internet is the search words you type in the ontology search, and only when BioPortal is asked.

### Is DicomDoodle a medical device?
No. It is a research and annotation tool. Its AI suggestions must always be checked by a qualified person and must not be used on their own to make clinical decisions.

### Text or buttons look too small
Use [[Ctrl]] + mouse wheel to zoom the image. In this guide, **A+** and **A−** in the toolbar change the text size.
""", "problem error help faq question slow missing files privacy model weights blank masks manual draw find tumour range "
    "3d model viewer volume processing skull stripping hd-bet "
    "calculator where moved",
    short="Troubleshooting & FAQ")

GLOSSARY = Chapter(
    "glossary", "Glossary", "list",
    "Plain-language explanations of the technical words used in DicomDoodle.",
    """
| Word | Meaning |
|---|---|
| **Annotation** | Marking a region on an image — here, outlining a lesion on an MRI slice. |
| **Annotation file** | The file DicomDoodle saves for a slice, named `<slice>_seg.dcm`. |
| **Axial, coronal, sagittal** | The three standard directions of slices: from the top, from the front, and from the side. |
| **Batch processing** | Doing the same job for many slices at once. |
| **BioPortal** | A free online library of medical vocabularies (ontologies). DicomDoodle asks it when DICOM's own code list has no matching term. |
| **Blank mask** | An annotation file that says “no lesion on this slice”. DicomDoodle saves one automatically for every slice you do not annotate. |
| **Boundary / outline** | The line around the edge of a region. DicomDoodle saves regions this way. |
| **Brightness and contrast** | How light the picture is and how strongly light and dark differ. Changing them only affects what you see. |
| **Category and type** | The two codes that say what a segment shows: the broad kind (category, e.g. *Morphologically Abnormal Structure*) and the specific thing (type, e.g. *Neoplasm*). |
| **Cine** | Playing through the slices one after another, like a film. |
| **Code** | A short identifier for a medical term in an ontology, e.g. SNOMED CT *108369006* for *Neoplasm*. |
| **Crosshair** | The green cross that marks the same point in every view of the multiplanar view. |
| **DICOM** | The standard file format of medical scanners. Usually one file per slice. |
| **DICOM SEG** | The DICOM format for segmentations — the format of DicomDoodle's annotation files. |
| **DICOM standard** | The rule book of medical image files, published at dicomstandard.org. It says what a file must contain so that every program reads it the same way. |
| **Dice score** | How well two annotations of the same lesion agree, from 0 (not at all) to 1 (exactly). |
| **GPU** | Graphics card. Makes the AI much faster, but is not required. |
| **Instance number** | The number a scanner gives each slice; it sets the order of the slices. |
| **IoU** | *Intersection over union* — like the Dice score, a measure of agreement from 0 to 1. |
| **Label** | The name and color of what you outline, such as *Tumor \\| Brain*. |
| **Lesion** | An area of abnormal tissue, such as a tumour. |
| **MPR** | *Multiplanar reconstruction* — showing a series in all three directions at once. |
| **MRI** | *Magnetic resonance imaging* — a scanner that takes detailed pictures of the inside of the body. |
| **Multi-frame file** | One DICOM file that holds a whole series instead of a single slice. |
| **OHIF** | A free web-based viewer for medical images, used by many hospitals and research groups. It shows DicomDoodle's annotation files. |
| **Ontology** | An agreed list of terms with codes, such as SNOMED CT, so computers can understand them. |
| **PACS** | *Picture archiving and communication system* — the hospital archive where scans are stored and viewed. |
| **Pixel spacing** | The real width and height of one pixel, in millimetres. |
| **Registration** | Lining up two scans so the same point means the same place in the body. |
| **SAM** | *Segment Anything Model* — the AI that traces the exact edge of an object inside a box. |
| **Segment** | One labelled region inside an annotation file. |
| **Segmentation** | Dividing an image into regions — the regions you have outlined. |
| **Series** | One set of slices taken in one go, usually stored in one folder. |
| **Skull stripping** | Removing skull and scalp from a head scan so only the brain remains. |
| **Slice** | One cross-sectional picture of the body. |
| **Slice thickness** | How thick the piece of tissue shown in one slice is, in millimetres. |
| **SNOMED CT** | A large international medical ontology; most of DICOM's codes come from it (written as *SCT*). |
| **Voxel** | A 3D pixel: pixel area × slice thickness. |
| **YOLO / YOLOE** | AI models that find *where* something is in an image and mark it with a box. |
""", "glossary words terms definitions meaning dictionary")

CHAPTERS = [SHORTCUTS_CHAPTER, TROUBLESHOOTING, GLOSSARY]
