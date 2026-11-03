"""User Guide — Reviewing & measuring."""
from .chapter import Chapter

VIEWER = Chapter(
    "viewer", "Reviewing saved annotations", "layers",
    "The Annotated DICOM Viewer shows a whole series together with all its saved annotation files.",
    """
Use the viewer to check a finished series, to show your annotations to a colleague, or to look at annotation files made by someone else.

## Opening it
Click {icon:layers} **Annotated DICOM Viewer** on the right of the toolbar (or **Tools ▸ Annotated DICOM Viewer**). If a series is open in the main window, the viewer opens it straight away; otherwise click **Load DICOM Folder**.

![The Annotated DICOM Viewer](shot:viewer)

| | |
|---|---|
| {n:1} | **Load DICOM Folder** — any folder with slices and their annotation files |
| {n:2} | **Zoom** mode — when on, the mouse wheel zooms instead of changing slices |
| {n:3} | **Show Mask** — fills each closed outline with a color of its own |
| {n:4} | **Opacity** of the annotations |
| {n:5} | **Annotation files** — which kind of files to show (see below) |
| {n:6} | **Add Ontology** — give the files standard medical terms ([see next chapter](chapter:ontology)) |
| {n:7} | The annotation file(s) of this slice |
| {n:8} | **Segments on this slice**: label, category and type |
| {n:9} | How many slices are annotated, and how many have blank masks |

## Moving around
The controls are the same as in the main window: the mouse wheel or arrow keys change the slice, [[Ctrl]] + wheel zooms, the middle mouse button pans, and {icon:play} on the timeline plays the series as a film. On the timeline, annotated slices are marked with a green tick and blank masks with a small dot.

## Which annotation files are shown
Some folders contain annotation files from two different tools — DicomDoodle's `_seg.dcm` and, for example, `_segmentation.dcm` from another program. Choose with **Annotation files**:
- **\\*_seg.dcm (DicomDoodle)** — only DicomDoodle's files,
- **\\*_segmentation.dcm** — only the other kind,
- **All annotation files (merged)** — both together.
""", "review viewer overlay saved annotations show mask opacity segments seg files", short="Reviewing annotations")

ONTOLOGY = Chapter(
    "ontology", "Adding standard medical terms (ontology)", "share",
    "Tag your annotation files with official medical codes — from DICOM's own code list, or from BioPortal — "
    "so other software knows exactly what each region is.",
    """
A label name like *Tumor & Brain Boundary* is clear to people, but software needs a precise code. **Ontologies** are agreed lists of medical terms with codes — for example SNOMED CT or the NCI Thesaurus. DICOM picks its codes from them and publishes the ones used in medical images as one list, which DicomDoodle carries with it.

Each segment of an annotation file has two terms:
- **Segment category** — the broad kind of thing. DICOM gives eight choices: *Anatomical Structure*, *Body Substance*, *Function*, *Morphologically Abnormal Structure*, *Physical object*, *Spatial and Relational Concept*, *Substance* and *Tissue*.
- **Segment type** — the specific thing, for example *Neoplasm*, *Edema* or *Glioma*.

New annotations get sensible terms automatically (see [Saving annotations](chapter:saving)). Use **Add Ontology** to make them more precise — for example *Glioma* instead of *Neoplasm*.

## Where the terms come from
1. **DICOM's own code list first.** It holds about 13,000 medical codes, works without internet and answers instantly. British spellings and everyday words are understood: *tumour* finds *Neoplasm*, *oedema* finds *Edema*.
2. **BioPortal when needed.** If the list has no match — rarer diagnoses such as *glioma* or *meningioma* — DicomDoodle asks BioPortal, a free online library of medical ontologies, automatically. **Also search BioPortal** adds its results even when the list did find something.

Codes are always written as DICOM expects them (SNOMED CT as *SCT*, the NCI Thesaurus as *NCIt* …). A term from an ontology that DICOM does not list is written together with a short description of that ontology, as the standard requires.

## Step by step
1. Open the series in the [Annotated DICOM Viewer](chapter:viewer).
2. Click {icon:share} **Add Ontology** at the right of its toolbar. The **Ontology Search** window opens.
3. Type a term or a code (for example *glioma*) and press [[Enter]] or click **Search**. The first 10 results appear; **Show all results** lists the rest. Each result shows where it comes from: *[DICOM]* or *[BioPortal …]*.
4. Click a result to see its details. **Definition & synonyms** explains what it means (this part needs internet).
5. Choose the **Segment category** from the list, or click **Use as category** on a result. Click **Use as type** for the type (or right-click a result).
6. Under **Segments**, choose *All labels*, or one label to give only its segments these terms.
7. Click **Apply to files** and confirm. Blank masks keep *Anatomical Structure / Brain*.

![The Ontology Search window](shot:ontology)

| | |
|---|---|
| {n:1} | The search box — a word or a code |
| {n:2} | **Search** — DICOM's code list first, BioPortal if it has no match |
| {n:3} | Results: term, code and source |
| {n:4} | Details of the selected term |
| {n:5} | **Use as category** / **Use as type** |
| {n:6} | Which segments, the category and the type, and **Apply to files** |

The viewer's **Segments on this slice** table shows the new category and type immediately. Each changed file also gets a new identifier (*SOP Instance UID*), as the DICOM standard asks when a file's content changes.

> NOTE: Searching DICOM's code list never uses the internet. When BioPortal is asked, only the words you type are sent — never images or patient data. DicomDoodle includes a BioPortal key; to use your own, create a free account at bioportal.bioontology.org and enter its key in **File ▸ Preferences ▸ Ontology**.
""", "ontology bioportal dicom code list snomed sct ncit term code category type standard terminology cid 7150 "
    "glioma neoplasm offline", short="Standard terms (ontology)")

VOLUME = Chapter(
    "volume", "Measuring lesion volume", "calculator",
    "Turn your annotations into a volume in cm³, shown in the right-hand panel of the Multiplanar & 3D Viewer.",
    """
The lesion volume is shown in the **Multiplanar & 3D Viewer**, in the **Lesion volume** card of its right-hand panel, next to the 3D model and the shape of the lesion. DicomDoodle adds up the annotated area on every slice and multiplies it by the size of each pixel and the slice thickness. All sizes come from the DICOM files themselves.

![How the volume is calculated](diagram:volume)

## Step by step
1. Save the annotations of the series: the volume is measured from the annotation files.
2. Click {icon:grid} **Multiplanar & 3D Viewer** on the right of the toolbar. In the panel on the right, the series open in the main window is already filled in, in the row of its plane.
3. Optionally add the coronal and/or sagittal series of the same lesion with {icon:folder}. Each series is measured on its own.
4. Choose which **Annotation files** to count (normally *All annotation files*), then click **Build 3D model**.
5. Read the volume in the **Lesion volume** card.

![The right-hand panel of the Multiplanar & 3D Viewer, with the axial, coronal and sagittal series of one tumour](shot:volume)

| | |
|---|---|
| {n:1} | One folder per orientation; at least one is needed |
| {n:2} | Which annotation files to count |
| {n:3} | **Build 3D model**: measures the volume and builds the model |
| {n:4} | The lesion volume. With several annotated series, it is their average |
| {n:5} | Each series: its volume, how many slices are annotated, and the size of one voxel |
| {n:6} | **Copy** the table (to paste into a spreadsheet) or **Export CSV…** |
| {n:7} | The shape of the lesion in 3D, see [The 3D view](chapter:viewer3d) |

## Reading the result
- Only annotated series count. A series you load just for the 3D picture is listed, but it does not lower the average.
- **Voxel** is the area of one pixel times the slice thickness: one tiny block of tissue.
- Notes below the series point out problems, for example a file without a slice thickness (1 mm is then assumed).
- The *Volume (3D model)* among the 3D lesion metrics is measured on the 3D model, where all loaded series are combined on one grid. It can differ a little from the lesion volume.

> TIP: If the same lesion was scanned in two or three directions, annotate each series. Their volumes should agree closely; a large difference usually means that some slices of one series were skipped or outlined differently.

> NOTE: Only closed outlines can be filled. Annotations from DicomDoodle are always closed (gaps up to 20 px are joined when saving); files from other tools may not be.
""", "volume cm3 cc measure lesion size calculator csv export spacing thickness 3d side panel average",
    short="Lesion volume")

CHAPTERS = [VIEWER, ONTOLOGY, VOLUME]
