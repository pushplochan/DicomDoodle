from PyQt5.QtCore import Qt, QTime, QRectF, QPointF, QThread, pyqtSignal
from PyQt5.QtGui import QCursor,QBrush, QPen, QPixmap, QKeySequence, QColor, QImage, QPainter
from PyQt5.QtWidgets import (QMessageBox, QDialog, QSlider, QScrollBar,QProgressBar, QMenuBar, QMenu, QTextEdit, QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QGraphicsScene, QGraphicsView, QShortcut, QFileDialog, QSizePolicy, QLabel, QGraphicsRectItem, QGraphicsEllipseItem, QGridLayout, QProgressDialog, QSlider, QComboBox, QGroupBox, QCheckBox, QDoubleSpinBox, QAction, QToolBar, QLineEdit)

class AboutViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About DicomDoodle")
        self.setGeometry(300, 300, 700, 500)  # Adjusted size for better readability
        self.setStyleSheet("background-color: #F0F8FF;")  # Light blue background
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("About DicomDoodle Tool")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #4682B4; margin-bottom: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Description Section
        description_label = QLabel("Tool Description")
        description_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #4682B4; margin-top: 10px;")
        layout.addWidget(description_label)
        
        description = QTextEdit()
        description.setReadOnly(True)
        description.setStyleSheet("""
            QTextEdit {
                border: 1px solid #4682B4;
                border-radius: 5px;
                padding: 8px;
                background-color: white;
                font-size: 14px;
                line-height: 1.5;
            }
        """)
        description.setText(
            """
<h3>DicomDoodle: A Comprehensive Tool for DICOM Image Annotation and Visualization</h3>

<p>DicomDoodle is a python based, user-friendly graphical user interface (GUI) tool designed to streamline the annotation, segmentation, and visualization of DICOM medical images, with a particular focus on MRI data. Built to assist researchers, radiologists, and medical professionals, DicomDoodle integrates deep learning models—such as YOLO for object detection and SAM (Segment Anything Model) for precise segmentation—to automate and enhance the process of identifying and annotating regions of interest, such as tumors, in MRI images.</p>

<p>The tool offers a robust suite of features, including:</p>
<ul>
    <li><b>Automated Segmentation</b>: Leverage YOLO and SAM models to detect and segment tumors with high accuracy, reducing manual effort.</li>
    <li><b>Manual Annotation</b>: Draw freehand annotations or define custom bounding boxes for precise control over segmentation tasks.</li>
    <li><b>Batch Processing</b>: Annotate multiple slices at once, including the ability to save blank masks for slices without annotations.</li>
    <li><b>Color-Coded Annotations</b>: Use a variety of colors to differentiate between boundary types (e.g., tumor-brain, tumor-dura), with each color mapping to a specific label in the output DICOM segmentation files.</li>
    <li><b>2D and 3D Visualization</b>: Visualize segmentation results in 2D with overlaid boundaries, or explore 3D models of tumors and anatomical structures using VTK-based rendering.</li>
    <li><b>DICOM Compatibility</b>: Save annotations as standard DICOM segmentation files (<code>_seg.dcm</code>), ensuring compatibility with other medical imaging software.</li>
</ul>

<p>DicomDoodle is built on a foundation of powerful open-source libraries, including PyQt5 for the GUI, PyDICOM and HighDICOM for DICOM file handling, Ultralytics for deep learning, and VTK for 3D visualization. Its intuitive interface, combined with advanced automation, makes it an ideal tool for medical image analysis tasks, whether you're conducting research, preparing data for machine learning models, or performing clinical evaluations.</p>

<p>Whether you're a radiologist annotating MRI scans for diagnostic purposes, a researcher studying tumor morphology, or a data scientist preparing annotated datasets for AI training, DicomDoodle will assist you to work efficiently and accurately.</p>
            """
        )
        description.setMinimumHeight(300)  # Ensure enough space for content
        layout.addWidget(description)
        
        # Contact Information Section
        contact_label = QLabel("Contact Information")
        contact_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #4682B4; margin-top: 10px;")
        layout.addWidget(contact_label)
        
        contact_info = QTextEdit()
        contact_info.setReadOnly(True)
        contact_info.setStyleSheet("""
            QTextEdit {
                border: 1px solid #4682B4;
                border-radius: 5px;
                padding: 8px;
                background-color: white;
                font-size: 14px;
                line-height: 1.5;
            }
        """)
        contact_info.setText(
            """
<p>For feedback, suggestions, or support, please contact:</p>
<ul>
    <li><b>Author</b>: Pushp Lochan Kumar</li>
    <li><b>Email</b>: pushpl@iisc.ac.in</li>
</ul>
<p>Your input is invaluable in helping us improve this tool</p>
            """
        )
        contact_info.setMinimumHeight(100)  # Smaller height for contact section
        layout.addWidget(contact_info)
        
        # Add Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF6347; /* Tomato */
                color: white;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
                margin-top: 10px;
            }
            QPushButton:hover { background-color: #FF4500; } /* OrangeRed */
            QPushButton:pressed { background-color: #CD5C5C; } /* IndianRed */
        """)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn, alignment=Qt.AlignCenter)
        
        self.setLayout(layout)

# Optional: test/debug when run directly
if __name__ == "__main__":

    app = QApplication([])
    window = AboutViewer()
    window.show()
    app.exec_()

