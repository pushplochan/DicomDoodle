"""About DicomDoodle: description, features and contact (text from the original)."""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QLabel, QTextBrowser, QVBoxLayout

from ... import APP_NAME, APP_TAGLINE, APP_VERSION, AUTHOR, AUTHOR_EMAIL, resources
from ..qt_utils import label, make_button

DESCRIPTION = f"""
<p>{APP_NAME} is a Python-based, user-friendly graphical tool designed to streamline the annotation,
segmentation, and visualization of DICOM medical images, with a particular focus on MRI data. Built to
assist researchers, radiologists, and medical professionals, it integrates deep learning models — YOLO for
object detection and SAM (Segment Anything Model) for precise segmentation — to automate and enhance the
identification and annotation of regions of interest, such as tumors, in MRI images.</p>
<ul>
<li><b>Automated Segmentation</b>: YOLO and SAM detect and segment tumors with high accuracy, reducing manual effort.</li>
<li><b>Manual Annotation</b>: draw and erase outlines by hand right on the image, or define bounding boxes
for precise control.</li>
<li><b>Batch Processing</b>: auto-detect and save a whole range of slices at once.</li>
<li><b>Automatic Blank Masks</b>: annotate only the slices with a lesion; every other slice gets a blank mask
when you save.</li>
<li><b>Color-Coded Labels</b>: six built-in boundary labels plus your own labels in any color; each maps to a
segment label in the saved DICOM segmentation.</li>
<li><b>2D and 3D Visualization</b>: overlay viewer, multiplanar MRI views and VTK-based 3D models with the
lesion volume and shape metrics beside them.</li>
<li><b>DICOM Compatibility</b>: annotations are standard DICOM Segmentation files (<code>_seg.dcm</code>) with
coded segment meanings, readable by PACS and web viewers such as OHIF; multi-frame files convert to standard
single-frame images.</li>
</ul>
<p>{APP_NAME} builds on PyQt5, pydicom, highdicom, Ultralytics, SimpleITK and VTK. Whether you annotate MRI
scans for diagnosis, study tumor morphology, or prepare datasets for AI training, {APP_NAME} helps you work
efficiently and accurately.</p>
<h4>Contact</h4>
<p>For feedback, suggestions, or support:<br><b>Author</b>: {AUTHOR}<br><b>Email</b>:
<a href="mailto:{AUTHOR_EMAIL}">{AUTHOR_EMAIL}</a></p>
<p>Your input is invaluable in helping us improve this tool.</p>
"""


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"About {APP_NAME}")
        self.resize(720, 620)

        logo = QLabel()
        logo_path = resources.find_resource(resources.LOGO_IMAGE) or resources.find_resource(resources.SPLASH_IMAGE)
        if logo_path:
            logo.setPixmap(QPixmap(logo_path).scaled(96, 96, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        title = label(APP_NAME, "Title")
        subtitle = label(f"{APP_TAGLINE} · version {APP_VERSION}", "Subtitle")

        header = QHBoxLayout()
        header.setSpacing(16)
        header.addWidget(logo)
        text = QVBoxLayout()
        text.addStretch(1)
        text.addWidget(title)
        text.addWidget(subtitle)
        text.addStretch(1)
        header.addLayout(text, 1)

        body = QTextBrowser()
        body.setOpenExternalLinks(True)
        body.setHtml(DESCRIPTION)

        close_btn = make_button("Close", role="primary")
        close_btn.clicked.connect(self.accept)
        footer = QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(close_btn)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(14)
        root.addLayout(header)
        root.addWidget(body, 1)
        root.addLayout(footer)
