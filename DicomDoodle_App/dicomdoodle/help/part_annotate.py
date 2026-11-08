"""User Guide — Annotating."""
from .chapter import Chapter

LABELS = Chapter(
    "labels", "Labels and colors", "tag",
    "A label says *what* you outline. Pick one before you segment — each has its own color and name.",
    """
Every outline you make belongs to a **label**. The label decides the color on screen and the name stored in the annotation file, so anyone who opens the file later knows what the region means.

![A label in the panel becomes a named segment in the file](diagram:labels)

## The built-in labels
DicomDoodle comes with six labels — the same as the original DicomDoodle, so older annotation files keep their meaning. They cannot be changed or removed.

| | Shown as | Name saved in the file | Key |
|---|---|---|---|
| {color:FF0000} | Tumor \\| Brain | Tumor & Brain Boundary | [[1]] |
| {color:00FF00} | Tumor \\| Dura | Tumor & Dura Boundary | [[2]] |
| {color:0000FF} | Tumor \\| Necrosis | Tumor & Necrosis Boundary | [[3]] |
| {color:FFFF00} | Edema \\| Brain | Edema & Brain Boundary | [[4]] |
| {color:00FFFF} | Tumor \\| Edema | Tumor & Edema Boundary | [[5]] |
| {color:FF00FF} | Magenta | Magenta Boundary | [[6]] |

## Choosing the active label
The **active label** is the one used for the next Auto Detect, box or batch run. It is highlighted in the **Labels** panel and shown at the bottom left of the image (*Label: …*).
- Click a label in the **Labels** panel, or
- press its number key [[1]] … [[9]] (the number is shown at the right of each label), or
- use **Annotate ▸ Active Label**.

> TIP: Change the label first, then segment. A region keeps the label it was made with.

## Adding your own label
You can add as many labels as you need — for example *Enhancing tumour*, *Cyst* or *Resection cavity* — with any color.
1. Click {icon:plus} **Add label** under the label list (or **Annotate ▸ Add Label…**). The **New label** window opens.
2. Type the **Name** — this exact text is saved in the annotation files, so make it clear and complete.
3. Optionally type a **Short name** for the label list (for example *Enh. Tumor*).
4. Pick a color: click one of the **Suggested colors** (colors no other label uses yet) or choose any color in the full **Color palette** on the right — by clicking, by typing RGB / HSV values or an HTML code like `#FF8C00`, or with **Pick Screen Color**.
5. Check the **Preview** and the message below it, then click **Create label**. The new label is selected right away.

![The New label window](shot:label_editor)

| | In the New label window |
|---|---|
| {n:1} | **Name** — saved in every annotation file that uses the label (up to 64 characters) |
| {n:2} | **Short name** — optional, shown in the label list |
| {n:3} | **Preview** and check: a red message explains what to fix; orange means the color looks like another label's |
| {n:4} | **Suggested colors** — one click picks a color nobody uses yet |
| {n:5} | **Colors already in use** — hover to see which label has which color |
| {n:6} | **Color palette** — every possible color, with a screen color picker |
| {n:7} | **Create label** — becomes available when the name and color are valid |

> NOTE: Names must be unique and colors must differ from the other labels, so regions can always be told apart.

## Editing, removing and sharing labels
- **Double-click** one of your labels to change its name or color, or use the {icon:sliders} menu next to **Add label** ▸ **Edit selected label…**.
- {icon:sliders} ▸ **Remove selected label** deletes one of your labels. Files already saved keep their segment names.
- {icon:sliders} ▸ **Export my labels…** saves your labels to a small `.json` file; a colleague can load it with **Import labels…** so the whole team uses identical names and colors.
- Right-click any label for the same commands.

## Labels found in files
When you open a file annotated with a label you do not have (for example from a colleague), DicomDoodle shows it in *italics* with “(from file)”. Right-click it ▸ **Add to my labels** to keep it.
""", "colour color palette label name segment add custom active key number import export")

AUTO_DETECT = Chapter(
    "auto-detect", "Auto Detect: let the AI find and outline the lesion", "sparkles",
    "One click: the AI finds the lesion on the current slice and draws its outline.",
    """
**Auto Detect** is the fastest way to annotate. Two AI models work one after the other:

![Auto Detect in three steps](diagram:autodetect)

- **YOLO**, an object-detection model from DicomDoodle's model card ([The AI models](chapter:ai-models)), looks at the whole slice and marks *where* the lesion is with a box. If it finds nothing, a second, general model (**YOLOE**) has a go and the brightest region it finds is used.
- **SAM** (*Segment Anything Model*) then traces the exact edge of the lesion inside that box.

## How to use it
1. Go to a slice that shows the lesion.
2. Choose the label in the **Labels** panel (for example **Tumor | Brain**).
3. Click {icon:sparkles} **Auto Detect** in the toolbar, or press [[Ctrl+D]].
4. Wait a moment — a spinning message shows what the AI is doing.
5. Check the result: the outline appears in the label's color, and a green dashed box shows what the AI detected.
6. If it looks right, press [[Ctrl+S]] to save. If not, press [[Ctrl+Z]] to undo and [segment with a box](chapter:box) instead.

![After Auto Detect: the outline of the tumour and the box the detector found](shot:autodetect)

| | On the image |
|---|---|
| {n:1} | Green dashed box: where the AI found the lesion (*YOLO* or *YOLOE*) |
| {n:2} | The outline drawn by SAM, in the active label's color |
| {n:3} | **Unsaved changes** — the slice has work that is not saved yet |
| {n:4} | The active label, used for this result |

> IMPORTANT: AI results are suggestions, not diagnoses. Always look at the outline before you save, especially on the first and last slices of a lesion, where it is small and faint.

## When the AI is not ready
The AI needs the model card — see [The AI models](chapter:ai-models). DicomDoodle loads the models in the background when it starts; the bottom-right corner of the status bar shows the state:
- grey **Manual annotation only** — no model card: the AI buttons are off,
- orange **Loading the AI models…** — a few seconds,
- green **AI ready · CPU** or **AI ready · GPU** — everything after this is quick.

> TIP: When DicomDoodle found a graphics card, click the AI status to run the AI on it — several times faster.

## When nothing is found
If the message *No detection found for this slice* appears, the AI did not recognise a lesion. Draw a box around it yourself — see [Segment with a box](chapter:box). Brightness and contrast do not change what the AI sees; it always works on the original image.

## Many slices at once
To run Auto Detect on a whole range of slices and save every result, use the [auto-detect range](chapter:batch).
""", "ai yolo yoloe sam automatic detect segment ctrl+d model gpu", short="Auto Detect (AI)")

BOX = Chapter(
    "box", "Segment with a box", "box-detect",
    "Draw a rectangle around the lesion and SAM traces its exact edge — you decide where to look.",
    """
Use a box when Auto Detect misses the lesion or picks the wrong area, or when a slice shows more than one region to outline.

![From box to outline](diagram:sam)

## Step by step
1. Choose the label in the **Labels** panel.
2. Check that {icon:box} **Box** is selected in the row of tools above the image. It is, unless you switched to the pen — press [[B]] to switch back.
3. **Left-drag** on the image from one corner of the lesion to the opposite corner. A red dashed box with four corner handles appears.
4. Adjust it if needed: drag *inside* the box to move it, drag a *corner handle* to resize it.
5. Press [[Enter]], click {icon:box-detect} **Detect Box** in the toolbar ([[Ctrl+Enter]]), or right-click the box and choose **Detect**.
6. SAM outlines the object inside the box. The box disappears and the outline is drawn in the label's color.

![A box drawn around the lesion](shot:box)

| | |
|---|---|
| {n:1} | The box — drag inside it to move it |
| {n:2} | Corner handles — drag one to resize the box |
| {n:3} | The label the result will get |

![The result: the outline, the status badge, and the orange tick on the timeline](shot:segmented)

| | |
|---|---|
| {n:1} | The outline made by SAM |
| {n:2} | Slice number, instance number and file name |
| {n:3} | **Unsaved changes** until you press [[Ctrl+S]] |
| {n:4} | An orange tick on the timeline marks every slice with unsaved work |

## Tips for a good box
- Leave a **small margin** of a few pixels around the lesion — not much more.
- Put **one lesion** in each box. A lesion in two separate parts needs two boxes, one after the other.
- If SAM outlines the wrong structure, make the box **tighter** and try again.
- [[Delete]] or [[Esc]] removes the box without segmenting.

## Several regions or labels on one slice
Every Detect *adds* a region to the slice. You can outline, for example, the tumour with **Tumor | Brain** (red) and then the swelling around it with **Edema | Brain** (yellow). Where two regions overlap, the newest one wins. In the saved file each label becomes its own named segment.
""", "box rectangle sam detect draw enter right click roi handles")

MANUAL_DRAW = Chapter(
    "manual-draw", "Drawing by hand", "pen",
    "For full control: draw or correct outlines yourself with the pen and the eraser, right on the image.",
    """
The **pen** and the **eraser** work directly on the image in the main window. Use them for small or unusual lesions, to fix an AI outline, or when you prefer to draw everything yourself. What you draw is part of the slice's annotation, so undo, saving, *Save All* and the blank masks treat it exactly like an AI outline.

## The tools above the image
The row of tools above the image decides what the left mouse button does.

![Drawing with the pen in the main window](shot:manual_draw)

| | |
|---|---|
| {n:1} | **Box** ([[B]]): drag a box for SAM, see [Segment with a box](chapter:box). This is the tool you start with. |
| {n:2} | **Pen** ([[P]] or [[Ctrl+M]]): draw in the active label's color. The pointer becomes a solid dot in that color. |
| {n:3} | **Eraser** ([[E]]): rub out outlines. The pointer is a circle as big as the eraser. |
| {n:4} | **Pan** ([[H]]): drag to move the picture |
| {n:5} | **Size** of the pen or eraser in pixels; keys [[\\[]] and [[\\]]] |
| {n:6} | **Edit saved outline** ([[Ctrl+L]]): bring in the slice's saved file to correct it |
| {n:7} | Your outline in the label's color: exactly the pixels that are saved |
| {n:8} | The tool, its size and the label the pen draws with |

## Drawing an outline
1. Choose the label in the **Labels** panel (or press its number key).
2. Click {icon:pen} **Pen**, or press [[P]].
3. Zoom in with [[Ctrl]] + mouse wheel so the edge of the lesion is large on screen.
4. Hold the left mouse button and **trace the edge** of the lesion all the way round.
5. Made a mistake? Press [[Ctrl+Z]], or switch to the {icon:eraser} **Eraser** ([[E]]) and rub it out.
6. Press [[Ctrl+S]] to save. Click {icon:box} **Box** (or press [[B]]) to go back to drawing boxes for SAM.

> TIP: Draw the **outline**, not a filled area: that is how DicomDoodle stores regions. Gaps of up to 20 pixels in your line are closed automatically when you save. With the *Filled regions* style, the inside fills in as soon as your outline is closed.

## Correcting an AI outline
Add to an outline with the pen and remove parts with the eraser. The first time you use the eraser on a slice, its AI regions turn into outlines you can rub out: exactly the lines that would be saved. Erase the wrong part, then draw it again with the pen. One [[Ctrl+Z]] brings the AI region back.

## Correcting a saved annotation
1. Go to the slice. Its badge says **Annotation file on disk**.
2. Click {icon:download} **Edit saved outline** ([[Ctrl+L]]). The outline from the file appears in full color, and the pen is selected.
3. Erase or redraw the parts that are wrong.
4. Save. The file is replaced with your corrected version without a question, because you loaded it yourself.

## Good to know
- The mouse wheel still changes slices; [[Ctrl]] + wheel zooms. Middle-drag or [[Space]] + drag pans whatever the tool.
- Drawings stay on each slice while you move to other slices, until you save them. An orange tick on the timeline marks them.
- Choosing a label while the eraser is on switches back to the pen.
- Right-click the image to switch tools.
""", "manual draw pen eraser freehand pencil correct edit outline brush size pan tool box load saved annotation "
    "ctrl+m ctrl+l")

CORRECTING = Chapter(
    "correcting", "Checking and correcting your work", "check-circle",
    "Undo, start the slice again, look at the result closely, and compare with what is saved.",
    """
## Undo and redo
- [[Ctrl+Z]] undoes the last change on the current slice, [[Ctrl+Y]] redoes it. Each slice keeps its own history of up to 30 steps.
- {icon:reset} **Reset** ([[Ctrl+R]]) clears everything on the current slice, AI outlines and drawing alike, and reloads the image. Reset can be undone too.

## Look at the result closely
{icon:image} **Results** ([[Ctrl+P]]) opens the slice in a larger window: on the left the region filled in, on the right the outline exactly as it will be saved.

![The Results window](shot:results)

| | |
|---|---|
| {n:1} | The image with the segmented region filled in |
| {n:2} | The image with the outline as saved in the file |
| {n:3} | **Fill opacity** and **Boundary width** for these two pictures |
| {n:4} | Which color is which label |
| {n:5} | **Save image…** — export either picture as PNG, for a report or a slide |

## Compare with the saved file
Turn on **Show saved annotation files** in the **Display** panel. A faint layer shows what is already saved on disk for this slice, so you can see whether your current work differs from it.

## Doing a slice again
Annotate the slice again and save. If the slice has an outline from an *earlier* session, DicomDoodle asks before replacing it (you can switch this question off — and on again in **File ▸ Preferences ▸ Files**). Blank masks are replaced without asking. To change only part of a saved outline, use [Edit saved outline](chapter:manual-draw) and the pen and eraser.
""", "undo redo reset mistake fix correct results plot compare saved overlay", short="Checking & correcting")

CHAPTERS = [LABELS, AUTO_DETECT, BOX, MANUAL_DRAW, CORRECTING]
