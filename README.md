DicomDoodle: Comprehensive DICOM Annotation & Segmentation Tool ✨
DicomDoodle is a powerful, Python-based graphical user interface (GUI) tool designed for the annotation, segmentation, and visualization of DICOM medical images, with a primary focus on MRI data. It integrates state-of-the-art deep learning models, YOLO for object detection and SAM (Segment Anything Model) for precise segmentation, to automate and enhance the identification and annotation of regions of interest (e.g., tumors).

Key Features ⚙️
Deep Learning Automation: Uses YOLO for tumor detection and SAM for precise segmentation on single or multiple slices.

Manual Segmentation: Offers a versatile Manual Drawing Tool for freehand annotations and a manual bounding box option for SAM.

Comprehensive Viewers: Includes a dedicated Annotated DICOM Viewer for reviewing masks and a powerful 3D Volume Viewer for reconstructing and visualizing anatomical and tumor data.

DICOM Standard Integration: Saves all annotations as DICOM Segmentation files (_seg.dcm) with rich metadata, ensuring interoperability.

Semantic Labeling: Allows users to associate segmentations with specific anatomical labels (e.g., "Tumor & Brain Boundary") and integrate with BioPortal Ontology for standardized clinical context.

Utility Tools: Comes with a Lesion Volume Calculator and a Multi-Frame DICOM Converter to streamline common tasks.

Getting Started 🚀
Launching the Tool
Run the main Python script or the executable (DicomDoodle.exe).

The main window, "DICOM Viewer and Annotator," will appear.

Understanding the Interface
The interface is divided into several intuitive sections:

Menu Bar (☰): Provides access to "About," "Citations," and advanced viewers/calculators.

Horizontal Scrollbar: Located at the top for quick navigation through slices.

Graphics View: The central area where DICOM slices and their annotations are displayed.

Button Rows: Organized below the graphics view into rows for General Actions (e.g., Load DICOM Folder, Auto Detect), Batch Processing & Manual Drawing, and Color Selection.

Window Title: Dynamically shows the current slice and total slice count (e.g., "DICOM Viewer - Slice 1/100").

Core Functionalities
Loading DICOM Files
Click the Load DICOM Folder button (red). A file dialog will open, allowing you to select a folder containing valid .dcm files. The tool will automatically sort them by InstanceNumber and display the first slice.

Navigating Slices
Mouse Wheel: Scroll up to go to the previous slice and down to go to the next.

Scrollbar: Drag the horizontal scrollbar at the top to jump to any slice.

Automatic Segmentation (YOLO & SAM)
Single Slice Auto-Detection
Navigate to the desired slice.

Click the Auto Detect button (blue).

YOLO will detect a bounding box, and SAM will perform a precise segmentation within it. The result is overlaid on the slice.

Batch Auto-Detection
Set the start slice by clicking Set Auto Detect Range Start (purple).

Set the end slice by clicking Set Auto Detect Range End (magenta).

Click Save Auto Detected Masks (bright magenta) to process all slices in the range. The tool will automatically save a segmentation file (_seg.dcm) for each slice.

Manual Segmentation
Manual Bounding Box for SAM
Navigate to a slice.

Left-click and drag to draw a red bounding box around a region of interest.

Right-click inside the box and select Detect. SAM will then segment the area.

Freehand Drawing Tool
Click the Manual Draw button (orange-red) to open a new window.

Select a color from the bottom color buttons.

Left-click and drag to draw freehand annotations.

Use the Eraser button to remove parts of your drawing.

Click Save Mask (pink) to save your annotations as a .dcm file.

Saving Segmentations
Save Single Slice: After any segmentation, click Save Results (pink) to save the current slice's mask as a _seg.dcm file.

Save Blank Masks: Define a range of slices using Set Blank Mask Range Start and Set Blank Mask Range End buttons, then click Save Blank Masks to quickly create all-zero segmentation files. This is useful for slices with no tumors.

Changing Segmentation Colors & Labels
Use the color buttons in the third row to set the active drawing color. Each color is associated with a specific label (e.g., 🎨 Red (255, 0, 0):Tumor | Brain), which is stored in the DICOM metadata for semantic clarity.

Visualization & Advanced Tools
2D Results Plotter
Click the Plot Results button (lime green) to view two side-by-side plots of the current slice: one with a blended overlay and another with only the segmentation boundary highlighted.

Annotated DICOM Viewer
Access this from the Menu Bar (☰). This viewer provides a clean interface to review all your saved _seg.dcm files. Its key feature is Ontology Integration:

Click Add Ontology and use the "BioPortal Ontology Search" to find and assign standardized clinical labels to your segmentations.

3D Volume Viewer
Access this from the Menu Bar (☰). It allows you to reconstruct and visualize 3D models of the anatomy and segmented tumors.

Click Load and select folders for your Axial, Coronal, and Sagittal DICOM series.

The tool will process the data and render a translucent anatomical volume with the segmented tumor(s) rendered in red.

Use the mouse to rotate, zoom, and pan.

Advanced controls (under Anatomy Visualization and Volume Visualization in the menu) allow you to adjust colormaps, opacity, and even use clipping planes to reveal internal structures.

Lesion Volume Calculator
Find this in the Menu Bar (☰). Select the folders for your DICOM series, and the tool will calculate the volume of all segmented regions and provide an average volume in cm 
3
 .

Multi-Frame to Single-Frame Converter
Located in the Menu Bar (☰). This utility allows you to select a multi-frame DICOM file and save each individual frame as a separate, single-frame DICOM file, preserving all relevant metadata.

Example Workflow 📋
Load Data: Click Load DICOM Folder and select an MRI series.

Single-Slice Auto-Segmentation: Navigate to slice 10. Click Auto Detect.

Batch Auto-Segmentation: Set a range from slice 20 to 30. Click Save Auto Detected Masks.

Manual Annotation: Navigate to slice 15. Click Manual Draw, draw a custom shape around a region, and click Save Mask.

Review 2D Results: Navigate back to slice 10. Click Plot Results to visualize the segmentation.

Review 3D Model: Go to the Menu Bar and select View 3D Model. Load the same DICOM folder and Process 3D Plot to see the reconstructed tumor(s).

Calculate Volume: From the Menu Bar, open the Lesion Volume Calculator to get the total volume.

Citations: DicomDoodle is built upon a number of powerful open-source libraries, including PyQt5, NumPy, OpenCV, PyDICOM, Ultralytics, HighDICOM, and Pillow. You can find a complete list of citations in the Menu Bar (☰).
